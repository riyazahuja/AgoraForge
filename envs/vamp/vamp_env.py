"""VampEnv: multi-agent environment implementing the VAMP framework.

Orchestrates FormulaGraph, Library, ProofKernel, ConjectureKernel,
QueryModel, Market, and Encoder into a complete environment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from gymnasium import spaces

from envs.multiagentenv import MultiAgentEnv
from envs.vamp.config import VampConfig
from envs.vamp.formula_graph import FormulaGraph
from envs.vamp.library import Library
from envs.vamp.proof_kernel import ProofKernel
from envs.vamp.conjecture_kernel import ConjectureKernel
from envs.vamp.query_model import QueryModel
from envs.vamp.market import BilateralContractMarket, Position
from envs.vamp.encoding import VampEncoder


class VampEnv(MultiAgentEnv):
    """VAMP multi-agent environment."""

    def __init__(self, cfg: VampConfig, seed: int = 0):
        self.cfg = cfg
        self.n_agents = cfg.n_agents
        self.bounty_agent_id = cfg.n_agents
        self.episode_limit = cfg.max_timestep
        self.rng = np.random.default_rng(seed)

        # Build formula graph from config or generate random
        if cfg.truth_map is not None:
            self.graph = FormulaGraph.from_config(cfg)
        else:
            self.graph = FormulaGraph.random(cfg.num_theorems, rng=self.rng)
            # Populate config with generated data for consistency
            cfg.truth_map = self.graph.theorem_truth_map
            cfg.difficulty_map = self.graph.theorem_difficulty_map
            cfg.dependency_adj = self.graph.theorem_dependency_adj
            cfg.utility_weights = self.graph.theorem_utility_weights

        # Kernels
        self.proof_kernel = ProofKernel.from_config(cfg)
        self.conj_kernel = ConjectureKernel.from_config(cfg)

        # Encoder
        self.encoder = VampEncoder.from_config(cfg)

        # Spaces
        self.observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.encoder.local_obs_dim,),
                dtype=np.float32,
            )
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.encoder.global_obs_dim,),
                dtype=np.float32,
            )
            for _ in range(self.n_agents)
        ]
        self.action_space = [
            spaces.Discrete(self.encoder.action_dim) for _ in range(self.n_agents)
        ]

        # State (initialized in reset)
        self.libraries: Dict[int, Library] = {}
        self.public_library: Library = Library(cfg.F_size)
        self.market: BilateralContractMarket = BilateralContractMarket(
            cfg.max_offers, cfg.max_own_offers
        )
        self.query_models: Dict[int, QueryModel] = {}
        self.jobs: Dict[int, Optional[dict]] = {}
        self.cumulative_proof: Dict[int, np.ndarray] = {}
        self.cumulative_conj: Dict[int, np.ndarray] = {}
        self.query_responses: Dict[int, Optional[Tuple]] = {}
        self.timestep: int = 0

    def _get_env_state(self) -> dict:
        """Package current state for encoder."""
        return {
            "config": self.cfg,
            "graph": self.graph,
            "libraries": self.libraries,
            "public_library": self.public_library,
            "market": self.market,
            "query_models": self.query_models,
            "jobs": self.jobs,
            "cumulative_proof": self.cumulative_proof,
            "cumulative_conj": self.cumulative_conj,
            "query_responses": self.query_responses,
            "timestep": self.timestep,
        }

    def _seed_initial_public_library(self) -> None:
        """Populate the public library with initially concrete unresolved formulas."""
        cfg = self.cfg
        for base_phi in range(cfg.half_F):
            if self.rng.random() > cfg.initial_public_concrete_prob:
                continue
            true_phi = base_phi if self.graph.is_true(base_phi) else self.graph.neg(base_phi)
            self.public_library.add_concrete(true_phi)

    def _seed_initial_target_offers(self) -> None:
        """Seed deterministic bounty offers on every initially concrete formula.

        For each concrete formula (including negations), posts a short position
        (offering long to acceptors) with price=cfg.seed_bounty_price and
        quantity=cfg.bounty_quantity.  The bounty agent is given exactly the
        collateral it needs.
        """
        cfg = self.cfg
        if cfg.bounty_quantity <= 0:
            return

        price = cfg.seed_bounty_price
        deadline = cfg.max_timestep
        quantity = cfg.bounty_quantity

        # Every initially concrete formula and its negation.
        candidates = set()
        for phi in sorted(self.public_library.concrete):
            if self.public_library.is_resolved(phi):
                continue
            candidates.add(phi)
            neg_phi = self.graph.neg(phi)
            if not self.public_library.is_concrete(neg_phi):
                self.public_library.add_concrete(neg_phi)
            if not self.public_library.is_resolved(neg_phi):
                candidates.add(neg_phi)

        # Give the bounty agent exactly enough collateral: each contract ties
        # up 1.0 * quantity in worst-case liability (price + (1-price) = 1.0).
        needed_collateral = float(len(candidates)) * quantity
        self.market.cash[self.bounty_agent_id] = needed_collateral

        for phi in sorted(candidates):
            self.market.create_and_post(
                self.bounty_agent_id,
                phi,
                deadline,
                float(price),
                "short",          # poster keeps short, offers long to acceptors
                quantity=quantity,
                ignore_own_offer_limit=True,
            )

    def reset(self):
        """Initialize/reset the environment.

        Returns (obs, share_obs, avail_actions) each with shape (n_agents, dim).
        """
        cfg = self.cfg
        self.libraries = {}
        self.query_models = {}
        self.jobs = {}
        self.cumulative_proof = {}
        self.cumulative_conj = {}
        self.query_responses = {}

        # Public library state shared by all real agents.
        self.public_library = Library(cfg.F_size)
        self._seed_initial_public_library()

        # Libraries
        for i in range(self.n_agents):
            self.libraries[i] = Library(cfg.F_size)
            if self.public_library.concrete or self.public_library.resolved:
                self.libraries[i].merge_from(
                    self.public_library,
                    set(self.public_library.concrete),
                    self.public_library.resolved_formulas(),
                )
            if cfg.initial_concrete:
                for phi in cfg.initial_concrete:
                    self.libraries[i].add_concrete(phi)
            if cfg.initial_resolved:
                for phi, (deps, t, s) in cfg.initial_resolved.items():
                    self.libraries[i].add_resolved(phi, deps, t, s)

        # Market
        self.market.reset(
            self.n_agents + 1,
            cfg.initial_cash,
            initial_cash_overrides={self.bounty_agent_id: 0.0},
        )
        self._seed_initial_target_offers()

        # Query models
        for i in range(self.n_agents):
            self.query_models[i] = QueryModel.from_config(cfg, i, self.rng)

        # Jobs
        for i in range(self.n_agents):
            self.jobs[i] = None

        # Cumulative budgets
        for i in range(self.n_agents):
            self.cumulative_proof[i] = np.zeros(cfg.F_size, dtype=np.float64)
            self.cumulative_conj[i] = np.zeros(cfg.F_size, dtype=np.float64)

        # Query responses
        for i in range(self.n_agents):
            self.query_responses[i] = None

        self.timestep = 0

        return self._get_observations()

    def _get_observations(self):
        """Return (obs, share_obs, avail_actions) each shaped (n_agents, dim)."""
        env_state = self._get_env_state()

        obs = np.zeros((self.n_agents, self.encoder.local_obs_dim), dtype=np.float32)
        share_obs = np.zeros(
            (self.n_agents, self.encoder.global_obs_dim), dtype=np.float32
        )
        avail_actions = np.zeros(
            (self.n_agents, self.encoder.action_dim), dtype=np.float32
        )

        global_enc = self.encoder.encode_global_obs(env_state)
        for i in range(self.n_agents):
            obs[i] = self.encoder.encode_local_obs(i, env_state)
            share_obs[i] = global_enc
            avail_actions[i] = self.encoder.get_available_actions(i, env_state)

        return obs, share_obs, avail_actions

    def _serialize_library(self, lib: Library) -> dict:
        return {
            "concrete": sorted(int(phi) for phi in lib.concrete),
            "resolved": [
                {
                    "formula": int(phi),
                    "deps": sorted(int(dep) for dep in info.deps),
                    "solve_time": int(info.solve_time),
                    "solver": int(info.solver),
                }
                for phi, info in sorted(lib.resolved.items())
            ],
        }

    def _serialize_position(self, pos) -> dict:
        return {
            "target": int(pos.contract.target),
            "deadline": int(pos.contract.deadline),
            "price": float(pos.contract.price),
            "side": pos.side,
            "quantity": int(pos.quantity),
            "settled": bool(pos.settled),
            "pnl": float(pos.pnl),
        }

    def _serialize_offer(self, offer_id: int, offer) -> dict:
        return {
            "offer_id": int(offer_id),
            "target": int(offer.contract.target),
            "deadline": int(offer.contract.deadline),
            "contract_price": float(offer.contract.price),
            "side": offer.side,
            "quantity": int(offer.quantity),
            "poster": int(offer.poster),
        }

    def _query_diagnostics_for_agent(self, agent_id: int) -> dict:
        """Compare the learned query model against the proof kernel."""
        cfg = self.cfg
        lib = self.libraries[agent_id]
        qm = self.query_models[agent_id]
        resolved = lib.resolved_formulas()

        abs_all = []
        sq_all = []
        abs_feasible = []
        sq_feasible = []
        pred_all = []
        true_all = []

        for phi in range(cfg.F_size):
            if lib.is_resolved(phi):
                continue

            deps = self.graph.get_deps(phi)
            feasible = self.graph.is_true(phi) and deps.issubset(resolved)

            for tau in cfg.budget_levels:
                pred = qm.success_probability(self.graph, lib, phi, tau)
                truth = self.proof_kernel.success_probability(
                    agent_id, self.graph, lib, phi, tau
                )
                err = pred - truth
                pred_all.append(pred)
                true_all.append(truth)
                abs_all.append(abs(err))
                sq_all.append(err * err)
                if feasible:
                    abs_feasible.append(abs(err))
                    sq_feasible.append(err * err)

        def _mean(values):
            return float(np.mean(values)) if values else 0.0

        return {
            "agent_id": int(agent_id),
            "num_points_all": int(len(abs_all)),
            "num_points_feasible": int(len(abs_feasible)),
            "mean_pred_all": _mean(pred_all),
            "mean_true_all": _mean(true_all),
            "mae_all": _mean(abs_all),
            "rmse_all": float(np.sqrt(_mean(sq_all))) if sq_all else 0.0,
            "mae_feasible": _mean(abs_feasible),
            "rmse_feasible": float(np.sqrt(_mean(sq_feasible))) if sq_feasible else 0.0,
        }

    def snapshot(self) -> dict:
        return {
            "timestep": int(self.timestep),
            "jobs": [
                (
                    None
                    if self.jobs[i] is None
                    else {
                        "type": self.jobs[i]["type"],
                        "target": int(self.jobs[i]["target"]),
                        "tau_rem": int(self.jobs[i]["tau_rem"]),
                        "tau_eff": int(self.jobs[i]["tau_eff"]),
                    }
                )
                for i in range(self.n_agents)
            ],
            "query_responses": [
                (
                    None
                    if self.query_responses[i] is None
                    else {
                        "formula": int(self.query_responses[i][0]),
                        "p_hat": float(self.query_responses[i][1]),
                        "tau_hat": float(self.query_responses[i][2]),
                    }
                )
                for i in range(self.n_agents)
            ],
            "query_model_quality": [
                self._query_diagnostics_for_agent(i) for i in range(self.n_agents)
            ],
            "agents": [
                {
                    "agent_id": int(i),
                    "cash": float(self.market.get_cash(i)),
                    "worst_case_balance": float(self.market.worst_case_balance(i)),
                    "positions": [
                        self._serialize_position(pos)
                        for pos in self.market.positions[i]
                    ],
                    "library": self._serialize_library(self.libraries[i]),
                    "cumulative_proof": [
                        float(v) for v in self.cumulative_proof[i].tolist()
                    ],
                    "cumulative_conj": [
                        float(v) for v in self.cumulative_conj[i].tolist()
                    ],
                }
                for i in range(self.n_agents)
            ],
            "public_library": self._serialize_library(self.public_library),
            "market_summary": {
                "tracked_agent_cash_total": float(
                    sum(self.market.get_cash(i) for i in range(self.n_agents))
                ),
                "system_cash_total": float(sum(self.market.cash.values())),
                "bounty_agent_cash": float(self.market.get_cash(self.bounty_agent_id)),
            },
            "offers": [
                self._serialize_offer(offer_id, self.market.offers[offer_id])
                for offer_id in self.market.get_offer_ids_sorted()
            ],
        }

    def describe_action(self, action_idx: int) -> dict:
        action = self.encoder.decode_action(int(action_idx))
        return {
            "type": action.type,
            "formula": None if action.formula is None else int(action.formula),
            "budget": None if action.budget is None else int(action.budget),
            "deadline": None if action.deadline is None else int(action.deadline),
            "side": action.side,
            "price": None if action.price is None else int(action.price),
            "offer_slot": None if action.offer_slot is None else int(action.offer_slot),
            "accept_quantity": None if action.accept_quantity is None else int(action.accept_quantity),
        }

    def describe_actions(self, actions) -> List[dict]:
        return [self.describe_action(action_idx) for action_idx in actions]

    def step(self, actions):
        """Execute one environment step.

        Args:
            actions: array of shape (n_agents,) with discrete action indices

        Returns:
            (obs, share_obs, rewards, dones, infos, avail_actions)
        """
        cfg = self.cfg
        cash_before = {i: self.market.get_cash(i) for i in range(self.n_agents)}
        shaping_rewards = np.zeros(self.n_agents, dtype=np.float32)

        # Decode actions
        decoded = {}
        for i in range(self.n_agents):
            action_idx = (
                int(actions[i]) if hasattr(actions, "__getitem__") else int(actions)
            )
            decoded[i] = self.encoder.decode_action(action_idx)

        # ── Stage I: Non-market actions ──

        # 1. Pre-update: process non-market actions
        for i in range(self.n_agents):
            action = decoded[i]
            newly_public = self._process_nonmarket_action(i, action)
            if newly_public > 0 and cfg.publish_resolution_bonus > 0.0:
                bonus = float(cfg.publish_resolution_bonus) * newly_public
                shaping_rewards[i] += bonus

        # 2. Decrement timers and check completions
        for i in range(self.n_agents):
            if self.jobs[i] is not None:
                self.jobs[i]["tau_rem"] -= 1
                if self.jobs[i]["tau_rem"] <= 0:
                    if self._resolve_job(i):
                        shaping_rewards[i] += cfg.proof_success_bonus

        # 3. Update timestep
        self.timestep += 1

        # ── Stage II: Market actions ──

        # 1. Settle expired contracts
        public_resolved = self.public_library.resolved_formulas()
        self.market.settle(self.timestep, public_resolved, neg_fn=self.cfg.neg)

        # 2. Process market actions
        for i in range(self.n_agents):
            action = decoded[i]
            pos_before = len(self.market.positions[i])
            self._process_market_action(i, action)
            if action.type == "accept" and len(self.market.positions[i]) > pos_before:
                shaping_rewards[i] += cfg.bounty_accept_bonus
            shaping_rewards[i] += self._action_shaping_reward(action)

        # Compute rewards
        rewards = np.zeros((self.n_agents, 1), dtype=np.float32)
        economic_rewards = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            economic_rewards[i] = self.market.get_cash(i) - cash_before[i]
            rewards[i, 0] = economic_rewards[i] + shaping_rewards[i]

        # Check done
        done = self.timestep >= cfg.max_timestep
        dones = np.full((self.n_agents, 1), done, dtype=np.float32)

        # Info: list of dicts, one per agent; 'won' key for RolloutWorker compat
        infos = [
            {
                "won": False,
                "economic_reward": float(economic_rewards[i]),
                "shaping_reward": float(shaping_rewards[i]),
            }
            for i in range(self.n_agents)
        ]

        # Observations
        obs, share_obs, avail_actions = self._get_observations()

        return obs, share_obs, rewards, dones, infos, avail_actions

    def _process_nonmarket_action(self, agent_id: int, action) -> int:
        """Process non-market portion of an agent's action.

        Returns the number of newly public resolutions created by a pub action.
        """
        cfg = self.cfg
        lib = self.libraries[agent_id]

        if action.type == "prove" and self.jobs[agent_id] is None:
            phi = action.formula
            tau = cfg.budget_levels[action.budget]
            self.jobs[agent_id] = {
                "type": "prove",
                "target": phi,
                "tau_rem": tau,
                "tau_eff": tau,
            }
            self.cumulative_proof[agent_id][phi] += tau**cfg.kappa
            return 0

        elif action.type == "conj" and self.jobs[agent_id] is None:
            phi = action.formula
            if not lib.is_concrete(phi):
                return 0
            tau = cfg.budget_levels[action.budget]
            self.jobs[agent_id] = {
                "type": "conj",
                "target": phi,
                "tau_rem": tau,
                "tau_eff": tau,
            }
            self.cumulative_conj[agent_id][phi] += tau**cfg.kappa
            return 0

        elif action.type == "pub":
            if self.jobs[agent_id] is not None:
                return 0
            phi = action.formula
            if lib.is_resolved(phi):
                closed_c, closed_r = lib.dependency_closure({phi})
                newly_public = sorted(
                    closed_r - self.public_library.resolved_formulas()
                )
                self.public_library.merge_from(lib, closed_c, closed_r)
                for j in range(self.n_agents):
                    if j != agent_id:
                        self.libraries[j].merge_from(lib, closed_c, closed_r)
                for resolved_phi in newly_public:
                    for qm in self.query_models.values():
                        qm.observe_public_resolution(self.graph, resolved_phi)
                return len(newly_public)
            return 0

        elif action.type == "qry":
            if self.jobs[agent_id] is not None:
                return 0
            phi = action.formula
            qm = self.query_models[agent_id]
            p_hat, tau_hat = qm.query(self.graph, lib, phi)
            self.query_responses[agent_id] = (phi, p_hat, tau_hat)
            return 0

        return 0

    def _resolve_job(self, agent_id: int) -> bool:
        """Resolve a completed job (timer reached 0).

        Returns True if a proof job succeeded, False otherwise.
        """
        job = self.jobs[agent_id]
        if job is None:
            return False

        lib = self.libraries[agent_id]
        phi = job["target"]
        tau_eff = job["tau_eff"]
        qm = self.query_models[agent_id]

        if job["type"] == "prove":
            success = self.proof_kernel.sample(
                agent_id, self.graph, lib, phi, tau_eff, self.rng
            )
            qm.observe_proof_result(self.graph, lib, phi, tau_eff, success)
            if success:
                deps = self.graph.get_deps(phi)
                lib.add_resolved(phi, deps, self.timestep, agent_id)
                qm.observe_private_resolution(self.graph, phi)
                self.jobs[agent_id] = None
                return True

        elif job["type"] == "conj":
            success, proposal = self.conj_kernel.sample(
                agent_id, self.graph, lib, phi, tau_eff, self.rng
            )
            if success and proposal is not None:
                lib.add_concrete(proposal)

        self.jobs[agent_id] = None
        return False

    def _process_market_action(self, agent_id: int, action) -> None:
        """Process market portion of an agent's action."""
        if self.jobs[agent_id] is not None:
            return
        cfg = self.cfg

        if action.type == "create_post":
            phi = action.formula
            if not self.public_library.is_concrete(phi):
                return
            deadline = cfg.deadline_levels[action.deadline]
            price = cfg.price_levels[action.price]
            side = action.side
            self.market.create_and_post(
                agent_id, phi, self.timestep + deadline, price, side
            )

        elif action.type == "accept":
            offer_ids = self.market.get_offer_ids_sorted()
            slot = action.offer_slot
            if slot < len(offer_ids):
                offer_id = offer_ids[slot]
                offer = self.market.offers.get(offer_id)
                if offer is not None:
                    qty_level = action.accept_quantity if action.accept_quantity is not None else 0
                    quantity = self.encoder.resolve_accept_quantity(qty_level, offer.quantity)
                    # Clamp to max affordable quantity (no cash transfer, liability only)
                    if quantity > 1:
                        unit_liability = offer.contract.price if offer.side == 'long' else (1.0 - offer.contract.price)
                        wcb = self.market.worst_case_balance(agent_id)
                        if unit_liability > 0:
                            affordable = max(0, int(wcb / unit_liability))
                            quantity = min(quantity, affordable)
                    if quantity > 0:
                        self.market.accept_offer(agent_id, offer_id, quantity=quantity)

        elif action.type == "cancel":
            offer_ids = self.market.get_offer_ids_sorted()
            own_offers = [
                oid for oid in offer_ids if self.market.offers[oid].poster == agent_id
            ]
            slot = action.offer_slot
            if slot < len(own_offers):
                self.market.cancel_offer(agent_id, own_offers[slot])

    def _action_shaping_reward(self, action) -> float:
        fee = float(self.cfg.operation_gas_fee)
        if fee <= 0.0:
            return 0.0
        if action.type in {"create_post", "cancel"}:
            return -fee
        return 0.0

    # ── MultiAgentEnv interface ──

    def get_obs(self):
        env_state = self._get_env_state()
        return [
            self.encoder.encode_local_obs(i, env_state) for i in range(self.n_agents)
        ]

    def get_obs_agent(self, agent_id):
        return self.encoder.encode_local_obs(agent_id, self._get_env_state())

    def get_obs_size(self):
        return self.encoder.local_obs_dim

    def get_state(self):
        return self.encoder.encode_global_obs(self._get_env_state())

    def get_state_size(self):
        return self.encoder.global_obs_dim

    def get_avail_actions(self):
        env_state = self._get_env_state()
        return [
            self.encoder.get_available_actions(i, env_state)
            for i in range(self.n_agents)
        ]

    def get_avail_agent_actions(self, agent_id):
        return self.encoder.get_available_actions(agent_id, self._get_env_state())

    def get_total_actions(self):
        return self.encoder.action_dim

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
