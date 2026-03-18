"""Deterministic oracle solver for public-resolution trajectories in VAMP.

This solver replaces the stochastic proof kernel with its closed-form expected
completion time, yielding a fully deterministic environment:

- The proof kernel gives success probability
      p = 1 - exp(-rate * tau^kappa)
  with geometric retries each costing tau timesteps, the expected time to first
  success is tau / p.  We round up (ceil) and treat the proof as succeeding
  deterministically after that many timesteps.

- Because all outcomes are deterministic, there is no branching: each state has
  exactly one successor per joint action.  This makes the DP dramatically
  faster than the stochastic version.

- When all agents are busy, the solver fast-forwards to the next proof
  completion rather than stepping one timestep at a time.

- For each provable formula, only the budget that minimises expected duration
  is offered as an action (shorter is always better in the deterministic
  setting).

- All other reductions from the original oracle are kept: only true formulas
  are considered, conjecturing is removed, proofs are publicly available
  immediately on completion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from envs.vamp.config import VampConfig
from envs.vamp.formula_graph import FormulaGraph


# Job is (target_bit, steps_remaining) or None.
OracleJob = Optional[Tuple[int, int]]
# State is (timestep, public_mask, jobs_tuple).
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
    """Deterministic DP oracle over public resolved formulas and proof jobs."""

    def __init__(self, cfg: VampConfig, initial_snapshot: dict):
        self.cfg = cfg
        self.graph = FormulaGraph.from_config(cfg)
        self.n_agents = cfg.n_agents

        self.true_formulas = tuple(
            int(phi) for phi in range(cfg.F_size) if self.graph.is_true(phi)
        )
        self.formula_to_bit = {phi: idx for idx, phi in enumerate(self.true_formulas)}
        self.bit_to_formula = {idx: phi for phi, idx in self.formula_to_bit.items()}
        self.true_count = len(self.true_formulas)
        self.dep_mask_by_bit = self._build_dep_masks()

        # Pre-compute scalar config values.
        self._kappa = float(cfg.kappa)
        self._lambda_diff = float(cfg.lambda_diff)
        self._alpha_util = float(cfg.alpha_util)
        self._rho_0 = [float(r) for r in cfg.rho_0]
        self._rho_1 = [float(r) for r in cfg.rho_1]
        self._budget_levels = tuple(int(b) for b in cfg.budget_levels)
        self._max_timestep = int(cfg.max_timestep)

        # Pre-compute per-bit difficulty factor: exp(-lambda_diff * difficulty).
        self._r_diff: Dict[int, float] = {}
        for bit, phi in self.bit_to_formula.items():
            self._r_diff[bit] = math.exp(
                -self._lambda_diff * self.graph.get_difficulty(phi)
            )

        # Pre-compute edge weights^alpha between all true formula pairs.
        self._weights: Dict[Tuple[int, int], float] = {}
        for src_bit, src_phi in self.bit_to_formula.items():
            for tgt_bit, tgt_phi in self.bit_to_formula.items():
                w = self.graph.get_weight(src_phi, tgt_phi)
                if w > 0.0:
                    self._weights[(src_bit, tgt_bit)] = w ** self._alpha_util

        self.initial_state = self._state_from_snapshot(initial_snapshot)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

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
        for agent_id, job in enumerate(snapshot["jobs"]):
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
            # Convert existing stochastic job to deterministic expected
            # remaining duration.
            tau_rem = int(job["tau_rem"])
            tau_eff = int(job["tau_eff"])
            expected_full = self._expected_duration(
                agent_id, public_mask, bit, tau_eff
            )
            steps_remaining = max(1, math.ceil(expected_full * tau_rem / tau_eff))
            jobs.append((bit, steps_remaining))

        return int(snapshot["timestep"]), public_mask, tuple(jobs)

    # ------------------------------------------------------------------
    # Proof kernel — deterministic expected duration
    # ------------------------------------------------------------------

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

    def _expected_duration(
        self, agent_id: int, public_mask: int, target_bit: int, tau: int
    ) -> int:
        """Expected timesteps for a proof to succeed (geometric retries).

        Each attempt costs *tau* timesteps and succeeds with probability
        ``p = 1 - exp(-rate * tau^kappa)``.  Expected attempts = 1/p, so
        expected total time = tau / p.  We return ceil of that.
        """
        rate = self._proof_rate(agent_id, public_mask, target_bit)
        p = 1.0 - math.exp(-rate * (tau ** self._kappa))
        if p < 1e-12:
            return self._max_timestep
        return math.ceil(tau / p)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _available_actions(
        self,
        agent_id: int,
        public_mask: int,
        jobs: Tuple[OracleJob, ...],
    ) -> List[OracleAction]:
        if jobs[agent_id] is not None:
            return [OracleAction("noop")]

        actions: List[OracleAction] = [OracleAction("noop")]
        for bit, phi in self.bit_to_formula.items():
            if public_mask & (1 << bit):
                continue
            deps_mask = self.dep_mask_by_bit[bit]
            if deps_mask & ~public_mask:
                continue
            # Pick the budget that minimises expected duration.
            best_budget_idx = 0
            best_dur = self._expected_duration(
                agent_id, public_mask, bit, self._budget_levels[0]
            )
            for budget_idx in range(1, len(self._budget_levels)):
                dur = self._expected_duration(
                    agent_id, public_mask, bit, self._budget_levels[budget_idx]
                )
                if dur < best_dur:
                    best_dur = dur
                    best_budget_idx = budget_idx
            actions.append(
                OracleAction("prove", formula=phi, budget=best_budget_idx)
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
            dur = self._expected_duration(agent_id, public_mask, bit, tau)
            next_jobs[agent_id] = (bit, dur)
        return public_mask, tuple(next_jobs)

    # ------------------------------------------------------------------
    # Deterministic step with fast-forward
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def _resolve_step(
        self,
        timestep: int,
        public_mask: int,
        jobs: Tuple[OracleJob, ...],
    ) -> OracleState:
        """Advance time.  When all agents are busy, fast-forward to the next
        proof completion instead of stepping one timestep at a time."""
        # Determine advance amount.
        min_remaining: Optional[int] = None
        all_busy = True
        for job in jobs:
            if job is None:
                all_busy = False
            else:
                _, steps_rem = job
                if min_remaining is None or steps_rem < min_remaining:
                    min_remaining = steps_rem

        advance = min_remaining if (all_busy and min_remaining is not None) else 1

        # Don't overshoot the horizon.
        advance = min(advance, self._max_timestep - timestep)
        if advance <= 0:
            return (self._max_timestep, public_mask, jobs)

        next_public = public_mask
        next_jobs: List[OracleJob] = list(jobs)
        for agent_id, job in enumerate(jobs):
            if job is None:
                continue
            target_bit, steps_rem = job
            new_rem = steps_rem - advance
            if new_rem <= 0:
                next_public |= 1 << target_bit
                next_jobs[agent_id] = None
            else:
                next_jobs[agent_id] = (target_bit, new_rem)

        return (timestep + advance, next_public, tuple(next_jobs))

    # ------------------------------------------------------------------
    # DP solver
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def _solve_from_state(self, state: OracleState) -> Tuple[float, Tuple[dict, ...]]:
        """Maximise the *area under the curve* (sum of resolved counts across
        all remaining timesteps).  This naturally prefers resolving formulas
        as early as possible, breaking the tie that the old final-count
        objective could not."""
        timestep, public_mask, jobs = state

        if timestep >= self._max_timestep:
            return 0.0, ()

        count = float(bin(public_mask).count("1"))
        remaining = self._max_timestep - timestep

        # Fast path: all agents busy — no decisions, just fast-forward.
        if all(job is not None for job in jobs):
            next_state = self._resolve_step(timestep, public_mask, jobs)
            advance = next_state[0] - timestep
            future, _ = self._solve_from_state(next_state)
            noop_dict = OracleAction("noop").to_dict()
            return count * advance + future, tuple(
                noop_dict for _ in range(self.n_agents)
            )

        per_agent_actions = [
            self._available_actions(agent_id, public_mask, jobs)
            for agent_id in range(self.n_agents)
        ]

        # Fast path: every agent can only noop — skip to the horizon.
        if all(len(acts) == 1 for acts in per_agent_actions):
            # All agents either busy (handled above) or have nothing to prove.
            # If any agent is busy, step normally; otherwise jump to horizon.
            if all(job is None for job in jobs):
                return count * remaining, ()
            # Some busy, some idle with no options — step to next completion.
            next_state = self._resolve_step(timestep, public_mask, jobs)
            advance = next_state[0] - timestep
            future, _ = self._solve_from_state(next_state)
            noop_dict = OracleAction("noop").to_dict()
            return count * advance + future, tuple(
                noop_dict for _ in range(self.n_agents)
            )

        best_value: Optional[float] = None
        best_joint_action: Optional[Tuple[dict, ...]] = None

        for joint_action in product(*per_agent_actions):
            next_public_mask, next_jobs = self._apply_actions(
                public_mask, jobs, joint_action
            )
            next_state = self._resolve_step(timestep, next_public_mask, next_jobs)
            advance = next_state[0] - timestep
            future, _ = self._solve_from_state(next_state)
            candidate_value = count * advance + future

            if best_value is None or candidate_value > best_value + 1e-12:
                best_value = candidate_value
                best_joint_action = tuple(a.to_dict() for a in joint_action)

        assert best_value is not None
        assert best_joint_action is not None
        return best_value, best_joint_action

    # ------------------------------------------------------------------
    # Trajectory curve (deterministic — single trajectory, std=0)
    # ------------------------------------------------------------------

    def _build_curve(self, initial_state: OracleState) -> List[float]:
        t0 = initial_state[0]
        horizon = max(self._max_timestep - t0, 0)
        # Pre-fill with initial resolved count; overwrite as proofs complete.
        initial_count = float(bin(initial_state[1]).count("1"))
        curve = [initial_count] * (horizon + 1)

        state = initial_state
        while state[0] < self._max_timestep:
            _, best_action_dicts = self._solve_from_state(state)
            if not best_action_dicts:
                break
            joint_action = tuple(OracleAction(**a) for a in best_action_dicts)
            timestep, public_mask, jobs = state
            _, next_jobs = self._apply_actions(public_mask, jobs, joint_action)
            state = self._resolve_step(timestep, public_mask, next_jobs)

            # Fill from this timestep onward with the new count.
            new_count = float(bin(state[1]).count("1"))
            idx_start = state[0] - t0
            for i in range(idx_start, horizon + 1):
                curve[i] = new_count

        return curve

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> dict:
        _, best_joint_action = self._solve_from_state(self.initial_state)
        curve = self._build_curve(self.initial_state)
        return {
            "objective": "deterministic_expected_public_resolved_count",
            "horizon": int(max(self._max_timestep - self.initial_state[0], 0)),
            "initial_public_resolved": float(bin(self.initial_state[1]).count("1")),
            "expected_public_resolved_by_time": curve,
            "public_resolved_std_by_time": [0.0] * len(curve),
            "solver_plan": [
                {
                    "timestep": int(self.initial_state[0]),
                    "joint_action": list(best_joint_action),
                }
            ],
        }


def _cache_key(cfg: VampConfig, initial_snapshot: dict) -> str:
    """Deterministic hash of the oracle inputs for disk caching."""
    import hashlib
    import json as _json

    key_data = _json.dumps(
        {
            "F_size": cfg.F_size,
            "n_agents": cfg.n_agents,
            "max_timestep": cfg.max_timestep,
            "budget_levels": list(cfg.budget_levels),
            "lambda_diff": cfg.lambda_diff,
            "alpha_util": cfg.alpha_util,
            "kappa": cfg.kappa,
            "rho_0": cfg.rho_0.tolist(),
            "rho_1": cfg.rho_1.tolist(),
            "num_theorems": cfg.num_theorems,
            "initial_snapshot": initial_snapshot,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()


_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".oracle_cache"


def solve_public_resolution_oracle(
    cfg: VampConfig,
    initial_snapshot: dict,
    *,
    cache_dir: Optional[Path] = None,
) -> dict:
    """Solve the deterministic public-resolution scheduling problem.

    Results are cached to disk under *cache_dir* (defaults to
    ``<repo>/.oracle_cache/``).  On a cache hit the solver is skipped
    entirely.
    """
    import json as _json

    cache = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    key = _cache_key(cfg, initial_snapshot)
    cache_file = cache / f"{key}.json"

    if cache_file.exists():
        return _json.loads(cache_file.read_text(encoding="utf-8"))

    result = PublicResolutionOracle(cfg, initial_snapshot).solve()

    cache.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(_json.dumps(result) + "\n", encoding="utf-8")
    return result
