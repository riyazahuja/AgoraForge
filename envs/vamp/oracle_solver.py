"""Reduced oracle solver for expected public-resolution trajectories in VAMP.

This solver uses a tractable full-information reduction aimed at the analysis
plot requested by the user:
- all formulas are treated as already concrete, so conjecturing is removed
- only true formulas are considered as proof targets
- the planner tracks publicly resolved formulas and active proof jobs
- successful proofs are treated as publicly available immediately on completion
- query, conjecture, market, and private-library branching are ignored

The result is an idealized upper-bound scheduler for "public resolved nodes vs
time" that is practical to compute on the current `F_size=8` runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Dict, List, Optional, Tuple

from envs.vamp.config import VampConfig
from envs.vamp.formula_graph import FormulaGraph


OracleJob = Optional[Tuple[int, int, int]]
OracleState = Tuple[int, int, Tuple[OracleJob, ...]]


@dataclass(frozen=True)
class OracleAction:
    kind: str
    formula: Optional[int] = None
    budget: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "formula": None if self.formula is None else int(self.formula),
            "budget": None if self.budget is None else int(self.budget),
        }


class PublicResolutionOracle:
    """Dynamic-programming oracle over public resolved formulas and proof jobs."""

    def __init__(self, cfg: VampConfig, initial_snapshot: dict):
        self.cfg = cfg
        self.graph = FormulaGraph.from_config(cfg)
        self.n_agents = cfg.n_agents
        self.initial_snapshot = initial_snapshot

        self.true_formulas = tuple(
            int(phi) for phi in range(cfg.F_size) if self.graph.is_true(phi)
        )
        self.formula_to_bit = {phi: idx for idx, phi in enumerate(self.true_formulas)}
        self.bit_to_formula = {idx: phi for phi, idx in self.formula_to_bit.items()}
        self.true_count = len(self.true_formulas)
        self.dep_mask_by_bit = self._build_dep_masks()

        # Pre-compute scalar config values to avoid repeated attribute lookups
        self._lambda_diff = float(cfg.lambda_diff)
        self._alpha_util = float(cfg.alpha_util)
        self._kappa = float(cfg.kappa)
        self._rho_0 = [float(r) for r in cfg.rho_0]
        self._rho_1 = [float(r) for r in cfg.rho_1]
        self._budget_levels = tuple(int(b) for b in cfg.budget_levels)
        self._max_timestep = int(cfg.max_timestep)

        # Pre-compute per-bit difficulty factor: exp(-lambda_diff * difficulty)
        self._r_diff = {}
        for bit, phi in self.bit_to_formula.items():
            self._r_diff[bit] = math.exp(-self._lambda_diff * self.graph.get_difficulty(phi))

        # Pre-compute edge weights between all true formula pairs
        self._weights: Dict[tuple, float] = {}
        for src_bit, src_phi in self.bit_to_formula.items():
            for tgt_bit, tgt_phi in self.bit_to_formula.items():
                w = self.graph.get_weight(src_phi, tgt_phi)
                if w > 0.0:
                    self._weights[(src_bit, tgt_bit)] = w ** self._alpha_util

        self.initial_state = self._state_from_snapshot(initial_snapshot)

    def _build_dep_masks(self) -> Dict[int, int]:
        dep_masks: Dict[int, int] = {}
        for phi in self.true_formulas:
            bit = self.formula_to_bit[phi]
            mask = 0
            for dep in self.graph.get_deps(phi):
                dep_bit = self.formula_to_bit.get(int(dep))
                if dep_bit is not None:
                    mask |= 1 << dep_bit
            dep_masks[bit] = mask
        return dep_masks

    def _state_from_snapshot(self, snapshot: dict) -> OracleState:
        public_mask = 0
        for item in snapshot["public_library"]["resolved"]:
            phi = int(item["formula"])
            bit = self.formula_to_bit.get(phi)
            if bit is not None:
                public_mask |= 1 << bit

        jobs: List[OracleJob] = []
        for job in snapshot["jobs"]:
            if job is None:
                jobs.append(None)
                continue
            if job["type"] != "prove":
                raise ValueError(
                    "Reduced oracle only supports states without active conjecture jobs."
                )
            bit = self.formula_to_bit.get(int(job["target"]))
            if bit is None:
                jobs.append(None)
                continue
            jobs.append((bit, int(job["tau_rem"]), int(job["tau_eff"])))

        return int(snapshot["timestep"]), public_mask, tuple(jobs)

    def _count_public(self, public_mask: int) -> int:
        return int(public_mask.bit_count())

    @lru_cache(maxsize=None)
    def _proof_rate(self, agent_id: int, public_mask: int, target_bit: int) -> float:
        r_diff = self._r_diff[target_bit]
        s_mass = 0.0
        for source_bit in range(self.true_count):
            if public_mask & (1 << source_bit):
                w = self._weights.get((source_bit, target_bit))
                if w is not None:
                    s_mass += w
        support = math.log1p(s_mass)
        return r_diff * (self._rho_0[agent_id] + self._rho_1[agent_id] * support)

    def _available_actions(
        self,
        agent_id: int,
        public_mask: int,
        jobs: Tuple[OracleJob, ...],
    ) -> List[OracleAction]:
        if jobs[agent_id] is not None:
            return [OracleAction("noop")]

        actions = [OracleAction("noop")]
        for bit, phi in self.bit_to_formula.items():
            if public_mask & (1 << bit):
                continue
            deps_mask = self.dep_mask_by_bit[bit]
            if deps_mask & ~public_mask:
                continue
            for budget_idx in range(len(self._budget_levels)):
                actions.append(
                    OracleAction("prove", formula=phi, budget=budget_idx)
                )
        return actions

    def _apply_actions(
        self,
        public_mask: int,
        jobs: Tuple[OracleJob, ...],
        joint_action: Tuple[OracleAction, ...],
    ) -> Tuple[int, Tuple[OracleJob, ...]]:
        next_jobs = list(jobs)
        for agent_id, action in enumerate(joint_action):
            if action.kind != "prove" or action.formula is None or action.budget is None:
                continue
            if next_jobs[agent_id] is not None:
                continue
            bit = self.formula_to_bit[action.formula]
            tau = self._budget_levels[action.budget]
            next_jobs[agent_id] = (bit, tau, tau)
        return public_mask, tuple(next_jobs)

    @lru_cache(maxsize=None)
    def _resolve_step(
        self,
        timestep: int,
        public_mask: int,
        jobs: Tuple[OracleJob, ...],
    ) -> Tuple[Tuple[float, OracleState], ...]:
        completion_events = []
        next_jobs: List[OracleJob] = list(jobs)

        for agent_id, job in enumerate(jobs):
            if job is None:
                continue
            target_bit, tau_rem, tau_eff = job
            tau_after = tau_rem - 1
            if tau_after <= 0:
                rate = self._proof_rate(agent_id, public_mask, target_bit)
                exponent = rate * (tau_eff ** self._kappa)
                p_success = max(0.0, min(1.0, 1.0 - math.exp(-exponent)))
                completion_events.append((target_bit, p_success))
                next_jobs[agent_id] = None
            else:
                next_jobs[agent_id] = (target_bit, tau_after, tau_eff)

        next_jobs_t = tuple(next_jobs)

        if not completion_events:
            return ((1.0, (timestep + 1, public_mask, next_jobs_t)),)

        branches: List[Tuple[float, OracleState]] = []
        for outcome_bits in product((0, 1), repeat=len(completion_events)):
            prob = 1.0
            next_public_mask = public_mask
            for bit_outcome, (target_bit, p_success) in zip(outcome_bits, completion_events):
                if bit_outcome:
                    prob *= p_success
                    next_public_mask |= 1 << target_bit
                else:
                    prob *= 1.0 - p_success
            if prob <= 0.0:
                continue
            branches.append((prob, (timestep + 1, next_public_mask, next_jobs_t)))
        return tuple(branches)

    @lru_cache(maxsize=None)
    def _solve_from_state(self, state: OracleState) -> Tuple[float, Tuple[dict, ...]]:
        timestep, public_mask, jobs = state
        if timestep >= self._max_timestep:
            return float(self._count_public(public_mask)), ()

        per_agent_actions = [
            self._available_actions(agent_id, public_mask, jobs)
            for agent_id in range(self.n_agents)
        ]

        # Prune: if an agent has prove actions available, drop noop
        # (noop is dominated — wasting a timestep is never better than trying)
        for i in range(len(per_agent_actions)):
            if len(per_agent_actions[i]) > 1:
                per_agent_actions[i] = [a for a in per_agent_actions[i] if a.kind != "noop"]

        best_value: Optional[float] = None
        best_joint_action: Optional[Tuple[dict, ...]] = None

        for joint_action in product(*per_agent_actions):
            next_public_mask, next_jobs = self._apply_actions(public_mask, jobs, joint_action)
            branches = self._resolve_step(timestep, next_public_mask, next_jobs)

            candidate_value = 0.0
            for prob, next_state in branches:
                branch_value, _ = self._solve_from_state(next_state)
                candidate_value += prob * branch_value

            if best_value is None or candidate_value > best_value + 1e-12:
                best_value = candidate_value
                best_joint_action = tuple(action.to_dict() for action in joint_action)

        assert best_value is not None
        assert best_joint_action is not None
        return best_value, best_joint_action

    def _curve_moments_from_policy(
        self,
        initial_state: OracleState,
    ) -> Tuple[List[float], List[float]]:
        distribution: Dict[OracleState, float] = {initial_state: 1.0}
        mean_curve = [float(self._count_public(initial_state[1]))]
        std_curve = [0.0]
        horizon = max(self.cfg.max_timestep - initial_state[0], 0)

        for _ in range(horizon):
            next_distribution: Dict[OracleState, float] = {}
            for state, state_prob in distribution.items():
                _, best_joint_action_dicts = self._solve_from_state(state)
                joint_action = tuple(OracleAction(**action) for action in best_joint_action_dicts)
                timestep, public_mask, jobs = state
                next_public_mask, next_jobs = self._apply_actions(public_mask, jobs, joint_action)
                branches = self._resolve_step(timestep, next_public_mask, next_jobs)
                for branch_prob, next_state in branches:
                    next_distribution[next_state] = (
                        next_distribution.get(next_state, 0.0) + state_prob * branch_prob
                    )

            expected_public = 0.0
            expected_public_sq = 0.0
            for state, state_prob in next_distribution.items():
                public_count = float(self._count_public(state[1]))
                expected_public += state_prob * public_count
                expected_public_sq += state_prob * (public_count ** 2)
            variance = max(expected_public_sq - expected_public ** 2, 0.0)
            mean_curve.append(float(expected_public))
            std_curve.append(math.sqrt(variance))
            distribution = next_distribution

        return mean_curve, std_curve

    def solve(self) -> dict:
        _, best_joint_action = self._solve_from_state(self.initial_state)
        mean_curve, std_curve = self._curve_moments_from_policy(self.initial_state)
        return {
            "objective": "reduced_expected_final_public_resolved_count",
            "horizon": int(max(self.cfg.max_timestep - self.initial_state[0], 0)),
            "initial_public_resolved": float(self._count_public(self.initial_state[1])),
            "expected_public_resolved_by_time": mean_curve,
            "public_resolved_std_by_time": std_curve,
            "solver_plan": [
                {
                    "timestep": int(self.initial_state[0]),
                    "joint_action": list(best_joint_action),
                }
            ],
        }


def solve_public_resolution_oracle(cfg: VampConfig, initial_snapshot: dict) -> dict:
    """Solve the reduced public-resolution scheduling problem."""
    return PublicResolutionOracle(cfg, initial_snapshot).solve()
