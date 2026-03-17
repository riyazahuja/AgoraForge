"""VampEnv: multi-agent environment implementing the VAMP framework.

Orchestrates FormulaGraph, Library, ProofKernel, ConjectureKernel,
QueryModel, Market, and Encoder into a complete environment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from gym import spaces

from envs.multiagentenv import MultiAgentEnv
from envs.vamp.config import VampConfig
from envs.vamp.formula_graph import FormulaGraph
from envs.vamp.library import Library
from envs.vamp.proof_kernel import ProofKernel
from envs.vamp.conjecture_kernel import ConjectureKernel
from envs.vamp.query_model import QueryModel
from envs.vamp.market import BilateralContractMarket
from envs.vamp.encoding import VampEncoder


class VampEnv(MultiAgentEnv):
    """VAMP multi-agent environment."""

    def __init__(self, cfg: VampConfig, seed: int = 0):
        self.cfg = cfg
        self.n_agents = cfg.n_agents
        self.episode_limit = cfg.max_timestep
        self.rng = np.random.default_rng(seed)

        # Build formula graph from config or generate random
        if cfg.truth_map is not None:
            self.graph = FormulaGraph.from_config(cfg)
        else:
            self.graph = FormulaGraph.random(cfg.F_size, rng=self.rng)
            # Populate config with generated data for consistency
            cfg.truth_map = self.graph.truth_map
            cfg.difficulty_map = self.graph.difficulty_map
            cfg.dependency_adj = self.graph.dependency_adj
            cfg.utility_weights = self.graph.utility_weights

        # Kernels
        self.proof_kernel = ProofKernel.from_config(cfg)
        self.conj_kernel = ConjectureKernel.from_config(cfg)

        # Encoder
        self.encoder = VampEncoder.from_config(cfg)

        # Spaces
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf,
                       shape=(self.encoder.local_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf,
                       shape=(self.encoder.global_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.action_space = [
            spaces.Discrete(self.encoder.action_dim)
            for _ in range(self.n_agents)
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
            'config': self.cfg,
            'graph': self.graph,
            'libraries': self.libraries,
            'public_library': self.public_library,
            'market': self.market,
            'query_models': self.query_models,
            'jobs': self.jobs,
            'cumulative_proof': self.cumulative_proof,
            'cumulative_conj': self.cumulative_conj,
            'query_responses': self.query_responses,
            'timestep': self.timestep,
        }

    def reset(self):
        """Initialize/reset the environment.

        Returns (obs, share_obs, avail_actions) each with shape (n_agents, dim).
        """
        cfg = self.cfg

        # Libraries
        for i in range(self.n_agents):
            self.libraries[i] = Library(cfg.F_size)
            if cfg.initial_concrete:
                for phi in cfg.initial_concrete:
                    self.libraries[i].add_concrete(phi)
            if cfg.initial_resolved:
                for phi, (deps, t, s) in cfg.initial_resolved.items():
                    self.libraries[i].add_resolved(phi, deps, t, s)
        self.public_library = Library(cfg.F_size)

        # Market
        self.market.reset(self.n_agents, cfg.initial_cash)

        # Query models
        for i in range(self.n_agents):
            self.query_models[i] = QueryModel.from_config(cfg)

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
        share_obs = np.zeros((self.n_agents, self.encoder.global_obs_dim), dtype=np.float32)
        avail_actions = np.zeros((self.n_agents, self.encoder.action_dim), dtype=np.float32)

        global_enc = self.encoder.encode_global_obs(env_state)
        for i in range(self.n_agents):
            obs[i] = self.encoder.encode_local_obs(i, env_state)
            share_obs[i] = global_enc
            avail_actions[i] = self.encoder.get_available_actions(i, env_state)

        return obs, share_obs, avail_actions

    def step(self, actions):
        """Execute one environment step.

        Args:
            actions: array of shape (n_agents,) with discrete action indices

        Returns:
            (obs, share_obs, rewards, dones, infos, avail_actions)
        """
        cfg = self.cfg
        cash_before = {i: self.market.get_cash(i) for i in range(self.n_agents)}

        # Decode actions
        decoded = {}
        for i in range(self.n_agents):
            action_idx = int(actions[i]) if hasattr(actions, '__getitem__') else int(actions)
            decoded[i] = self.encoder.decode_action(action_idx)

        # ── Stage I: Non-market actions ──

        # 1. Pre-update: process non-market actions
        for i in range(self.n_agents):
            action = decoded[i]
            self._process_nonmarket_action(i, action)

        # 2. Decrement timers and check completions
        for i in range(self.n_agents):
            if self.jobs[i] is not None:
                self.jobs[i]['tau_rem'] -= 1
                if self.jobs[i]['tau_rem'] <= 0:
                    self._resolve_job(i)

        # 3. Update timestep
        self.timestep += 1

        # ── Stage II: Market actions ──

        # 1. Settle expired contracts
        public_resolved = self.public_library.resolved_formulas()
        self.market.settle(self.timestep, public_resolved)

        # 2. Process market actions
        for i in range(self.n_agents):
            action = decoded[i]
            self._process_market_action(i, action)

        # Compute rewards
        rewards = np.zeros((self.n_agents, 1), dtype=np.float32)
        for i in range(self.n_agents):
            rewards[i, 0] = self.market.get_cash(i) - cash_before[i]

        # Check done
        done = self.timestep >= cfg.max_timestep
        dones = np.full((self.n_agents, 1), done, dtype=np.float32)

        # Info: list of dicts, one per agent; 'won' key for RolloutWorker compat
        infos = [{'won': False} for _ in range(self.n_agents)]

        # Observations
        obs, share_obs, avail_actions = self._get_observations()

        return obs, share_obs, rewards, dones, infos, avail_actions

    def _process_nonmarket_action(self, agent_id: int, action) -> None:
        """Process non-market portion of an agent's action."""
        cfg = self.cfg
        lib = self.libraries[agent_id]

        if action.type == 'prove' and self.jobs[agent_id] is None:
            phi = action.formula
            tau = cfg.budget_levels[action.budget]
            self.jobs[agent_id] = {
                'type': 'prove',
                'target': phi,
                'tau_rem': tau,
                'tau_eff': tau,
            }
            self.cumulative_proof[agent_id][phi] += tau ** cfg.kappa

        elif action.type == 'conj' and self.jobs[agent_id] is None:
            phi = action.formula
            tau = cfg.budget_levels[action.budget]
            self.jobs[agent_id] = {
                'type': 'conj',
                'target': phi,
                'tau_rem': tau,
                'tau_eff': tau,
            }
            self.cumulative_conj[agent_id][phi] += tau ** cfg.kappa

        elif action.type == 'pub':
            phi = action.formula
            if lib.is_resolved(phi):
                closed_c, closed_r = lib.dependency_closure({phi})
                self.public_library.merge_from(lib, closed_c, closed_r)

        elif action.type == 'qry':
            phi = action.formula
            qm = self.query_models[agent_id]
            p_hat, tau_hat = qm.query(self.graph, lib, phi)
            self.query_responses[agent_id] = (phi, p_hat, tau_hat)

    def _resolve_job(self, agent_id: int) -> None:
        """Resolve a completed job (timer reached 0)."""
        job = self.jobs[agent_id]
        if job is None:
            return

        lib = self.libraries[agent_id]
        phi = job['target']
        tau_eff = job['tau_eff']
        qm = self.query_models[agent_id]
        bucket = qm.bucket_map(self.graph, lib, phi)

        if job['type'] == 'prove':
            success = self.proof_kernel.sample(
                agent_id, self.graph, lib, phi, tau_eff, self.rng
            )
            if success:
                deps = self.graph.get_deps(phi)
                lib.add_resolved(phi, deps, self.timestep, agent_id)
            qm.update(bucket, int(tau_eff), success)

        elif job['type'] == 'conj':
            success, proposal = self.conj_kernel.sample(
                agent_id, self.graph, lib, phi, tau_eff, self.rng
            )
            if success and proposal is not None:
                lib.add_concrete(proposal)
            qm.update(bucket, int(tau_eff), success)

        self.jobs[agent_id] = None

    def _process_market_action(self, agent_id: int, action) -> None:
        """Process market portion of an agent's action."""
        cfg = self.cfg

        if action.type == 'create_post':
            phi = action.formula
            deadline = cfg.deadline_levels[action.deadline]
            loss = cfg.loss_levels[action.loss]
            side = action.side
            # Use middle price as default (price not in create action encoding)
            price = cfg.price_levels[len(cfg.price_levels) // 2]
            self.market.create_and_post(
                agent_id, phi, self.timestep + deadline, loss, side, price
            )

        elif action.type == 'accept':
            offer_ids = self.market.get_offer_ids_sorted()
            slot = action.offer_slot
            if slot < len(offer_ids):
                self.market.accept_offer(agent_id, offer_ids[slot])

        elif action.type == 'cancel':
            offer_ids = self.market.get_offer_ids_sorted()
            own_offers = [
                oid for oid in offer_ids
                if self.market.offers[oid].poster == agent_id
            ]
            slot = action.offer_slot
            if slot < len(own_offers):
                self.market.cancel_offer(agent_id, own_offers[slot])

    # ── MultiAgentEnv interface ──

    def get_obs(self):
        env_state = self._get_env_state()
        return [self.encoder.encode_local_obs(i, env_state) for i in range(self.n_agents)]

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
        return [self.encoder.get_available_actions(i, env_state) for i in range(self.n_agents)]

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
