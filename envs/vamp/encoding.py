"""VampEncoder: observation and action encoding for VAMP environments.

Action space layout (discrete indices):
    [0]                                  NoOp
    [1 .. F*B]                           Prove(phi, tau)
    [F*B+1 .. 2*F*B]                     Conj(phi, tau)
    [2*F*B+1 .. 2*F*B+F]                Pub(phi)
    [2*F*B+F+1 .. 2*F*B+2F]             Qry(phi)
    [2*F*B+2F+1]                         MarketNoOp
    [2*F*B+2F+2 .. +F*D*L*P*2]          CreatePost(phi, deadline, loss, price, side)
    [next .. +MAX_OFFERS]                AcceptOffer(slot)
    [next .. +MAX_OWN_OFFERS]            CancelOffer(slot)

Local obs: 14*F + N + 7 + 6*max_offers dims
Global obs: (4*F + 5 + F) * N + 4*F + 6*max_offers + 1 dims
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import numpy as np


VampAction = namedtuple('VampAction', [
    'type',        # 'noop', 'prove', 'conj', 'pub', 'qry', 'market_noop',
                   # 'create_post', 'accept', 'cancel'
    'formula',     # target formula index (or None)
    'budget',      # budget level index (or None)
    'deadline',    # deadline level index (or None)
    'loss',        # loss level index (or None)
    'side',        # 'long' or 'short' (or None)
    'price',       # price level index (or None)
    'offer_slot',  # offer slot index (or None)
])


class VampEncoder:
    """Encodes observations and actions for VampEnv."""

    def __init__(self, F_size: int, n_agents: int, n_budget_levels: int,
                 n_deadline_levels: int, n_loss_levels: int, n_price_levels: int,
                 max_offers: int, max_own_offers: int):
        self.F = F_size
        self.N = n_agents
        self.B = n_budget_levels
        self.D = n_deadline_levels
        self.L = n_loss_levels
        self.P = n_price_levels
        self.max_offers = max_offers
        self.max_own_offers = max_own_offers

        # Compute action space offsets
        self._noop = 0
        self._prove_start = 1
        self._prove_end = 1 + self.F * self.B
        self._conj_start = self._prove_end
        self._conj_end = self._conj_start + self.F * self.B
        self._pub_start = self._conj_end
        self._pub_end = self._pub_start + self.F
        self._qry_start = self._pub_end
        self._qry_end = self._qry_start + self.F
        self._market_noop = self._qry_end
        self._create_start = self._market_noop + 1
        self._create_end = self._create_start + self.F * self.D * self.L * self.P * 2
        self._accept_start = self._create_end
        self._accept_end = self._accept_start + self.max_offers
        self._cancel_start = self._accept_end
        self._cancel_end = self._cancel_start + self.max_own_offers

    @classmethod
    def from_config(cls, cfg) -> VampEncoder:
        return cls(
            F_size=cfg.F_size,
            n_agents=cfg.n_agents,
            n_budget_levels=cfg.n_budget_levels,
            n_deadline_levels=cfg.n_deadline_levels,
            n_loss_levels=cfg.n_loss_levels,
            n_price_levels=cfg.n_price_levels,
            max_offers=cfg.max_offers,
            max_own_offers=cfg.max_own_offers,
        )

    @property
    def action_dim(self) -> int:
        return self._cancel_end

    @property
    def local_obs_dim(self) -> int:
        # Per formula (14 features): hidden_truth_pad(1), concrete(1), resolved(1), difficulty(1),
        #   dep_count(1), in_job(1), job_type(2), job_tau_rem(1), cum_proof(1), cum_conj(1),
        #   query_p(1), query_tau(1), weight_sum(1), pub_resolved(1)
        # Per agent: cash(1), worst_case(1), n_positions(1), job_active(1),
        #   query_response_valid(1), own_offer_count(1)
        # Scalar: timestep(1), agent_id one-hot(N)
        # + market slots: max_offers * 6 (target, deadline, loss, price, side, quantity)
        return 14 * self.F + 6 + 1 + self.N + self.max_offers * 6

    @property
    def global_obs_dim(self) -> int:
        # Per agent per formula: concrete(1), resolved(1), cum_proof(1), cum_conj(1) = 4*F
        # Per agent scalars: cash(1), worst_case(1), job_active(1), job_formula(1), job_type(1) = 5
        # Per agent: offer_vector(F) = F per agent
        # Global: public_concrete(F), public_resolved(F), truth_map(F), difficulty_map(F) = 4*F
        # Market: offer_slots(max_offers * 6), timestep(1)
        return (4 * self.F + 5 + self.F) * self.N + 4 * self.F + self.max_offers * 6 + 1

    def decode_action(self, action_idx: int) -> VampAction:
        """Decode a discrete action index into a VampAction."""
        if action_idx == self._noop:
            return VampAction('noop', None, None, None, None, None, None, None)

        if self._prove_start <= action_idx < self._prove_end:
            idx = action_idx - self._prove_start
            phi = idx // self.B
            tau = idx % self.B
            return VampAction('prove', phi, tau, None, None, None, None, None)

        if self._conj_start <= action_idx < self._conj_end:
            idx = action_idx - self._conj_start
            phi = idx // self.B
            tau = idx % self.B
            return VampAction('conj', phi, tau, None, None, None, None, None)

        if self._pub_start <= action_idx < self._pub_end:
            phi = action_idx - self._pub_start
            return VampAction('pub', phi, None, None, None, None, None, None)

        if self._qry_start <= action_idx < self._qry_end:
            phi = action_idx - self._qry_start
            return VampAction('qry', phi, None, None, None, None, None, None)

        if action_idx == self._market_noop:
            return VampAction('market_noop', None, None, None, None, None, None, None)

        if self._create_start <= action_idx < self._create_end:
            idx = action_idx - self._create_start
            side_idx = idx % 2
            idx //= 2
            price_idx = idx % self.P
            idx //= self.P
            loss_idx = idx % self.L
            idx //= self.L
            deadline_idx = idx % self.D
            phi = idx // self.D
            side = 'long' if side_idx == 0 else 'short'
            return VampAction('create_post', phi, None, deadline_idx, loss_idx, side, price_idx, None)

        if self._accept_start <= action_idx < self._accept_end:
            slot = action_idx - self._accept_start
            return VampAction('accept', None, None, None, None, None, None, slot)

        if self._cancel_start <= action_idx < self._cancel_end:
            slot = action_idx - self._cancel_start
            return VampAction('cancel', None, None, None, None, None, None, slot)

        return VampAction('noop', None, None, None, None, None, None, None)

    def encode_action(self, action: VampAction) -> int:
        """Encode a VampAction into a discrete index."""
        if action.type == 'noop':
            return self._noop
        if action.type == 'prove':
            return self._prove_start + action.formula * self.B + action.budget
        if action.type == 'conj':
            return self._conj_start + action.formula * self.B + action.budget
        if action.type == 'pub':
            return self._pub_start + action.formula
        if action.type == 'qry':
            return self._qry_start + action.formula
        if action.type == 'market_noop':
            return self._market_noop
        if action.type == 'create_post':
            side_idx = 0 if action.side == 'long' else 1
            idx = (
                action.formula * self.D * self.L * self.P * 2
                + action.deadline * self.L * self.P * 2
                + action.loss * self.P * 2
                + action.price * 2
                + side_idx
            )
            return self._create_start + idx
        if action.type == 'accept':
            return self._accept_start + action.offer_slot
        if action.type == 'cancel':
            return self._cancel_start + action.offer_slot
        return self._noop

    def encode_local_obs(self, agent_id: int, env_state: dict) -> np.ndarray:
        """Encode local observation for an agent.

        env_state keys: graph, libraries, public_library, market, query_models,
                        jobs, cumulative_proof, cumulative_conj, query_responses,
                        timestep, config
        """
        cfg = env_state['config']
        graph = env_state['graph']
        lib = env_state['libraries'][agent_id]
        pub_lib = env_state['public_library']
        market = env_state['market']
        qm = env_state['query_models'][agent_id]
        job = env_state['jobs'][agent_id]
        cum_proof = env_state['cumulative_proof'][agent_id]
        cum_conj = env_state['cumulative_conj'][agent_id]
        qr = env_state['query_responses'][agent_id]

        obs = np.zeros(self.local_obs_dim, dtype=np.float32)
        offset = 0

        # Per-formula features (14 * F)
        for phi in range(self.F):
            # Do not leak latent truth into the policy observation.
            obs[offset] = 0.0
            obs[offset + 1] = float(lib.is_concrete(phi))
            obs[offset + 2] = float(lib.is_resolved(phi))
            obs[offset + 3] = graph.difficulty_map[phi]
            obs[offset + 4] = len(graph.get_deps(phi)) / max(self.F, 1)
            # Job info for this formula
            if job is not None and job['target'] == phi:
                obs[offset + 5] = 1.0
                obs[offset + 6] = 1.0 if job['type'] == 'prove' else 0.0
                obs[offset + 7] = 1.0 if job['type'] == 'conj' else 0.0
                obs[offset + 8] = job['tau_rem'] / max(cfg.max_timestep, 1)
            obs[offset + 9] = cum_proof[phi]
            obs[offset + 10] = cum_conj[phi]
            # Query response for this formula
            if qr is not None and qr[0] == phi:
                obs[offset + 11] = qr[1]   # p_hat
                obs[offset + 12] = qr[2] / max(cfg.horizon_H, 1)  # tau_hat normalized
            obs[offset + 13] = float(pub_lib.is_resolved(phi))
            offset += 14

        # Agent scalars (6)
        obs[offset] = market.get_cash(agent_id) / max(cfg.initial_cash, 1)
        obs[offset + 1] = market.worst_case_balance(agent_id) / max(cfg.initial_cash, 1)
        obs[offset + 2] = len([p for p in market.positions[agent_id] if not p.settled]) / max(self.max_offers, 1)
        obs[offset + 3] = 1.0 if job is not None else 0.0
        obs[offset + 4] = 1.0 if qr is not None else 0.0
        obs[offset + 5] = market._agent_offer_count(agent_id) / max(self.max_own_offers, 1)
        offset += 6

        # Timestep (1)
        obs[offset] = env_state['timestep'] / max(cfg.max_timestep, 1)
        offset += 1

        # Agent ID one-hot (N)
        obs[offset + agent_id] = 1.0
        offset += self.N

        # Market offer slots (max_offers * 6)
        offer_ids = market.get_offer_ids_sorted()
        for slot_idx in range(self.max_offers):
            base = offset + slot_idx * 6
            if slot_idx < len(offer_ids):
                offer = market.offers[offer_ids[slot_idx]]
                obs[base] = offer.contract.target / max(self.F, 1)
                obs[base + 1] = offer.contract.deadline / max(cfg.max_timestep, 1)
                obs[base + 2] = offer.contract.loss
                obs[base + 3] = offer.price
                obs[base + 4] = 1.0 if offer.side == 'long' else 0.0
                obs[base + 5] = offer.quantity / max(cfg.target_init_max_quantity, 1)
        offset += self.max_offers * 6

        return obs

    def encode_global_obs(self, env_state: dict) -> np.ndarray:
        """Encode global (shared) observation."""
        cfg = env_state['config']
        graph = env_state['graph']
        market = env_state['market']
        pub_lib = env_state['public_library']

        obs = np.zeros(self.global_obs_dim, dtype=np.float32)
        offset = 0

        # Per agent (4*F + 5 + F each)
        for agent_id in range(self.N):
            lib = env_state['libraries'][agent_id]
            job = env_state['jobs'][agent_id]
            cum_proof = env_state['cumulative_proof'][agent_id]
            cum_conj = env_state['cumulative_conj'][agent_id]

            for phi in range(self.F):
                obs[offset] = float(lib.is_concrete(phi))
                obs[offset + 1] = float(lib.is_resolved(phi))
                obs[offset + 2] = cum_proof[phi]
                obs[offset + 3] = cum_conj[phi]
                offset += 4

            obs[offset] = market.get_cash(agent_id) / max(cfg.initial_cash, 1)
            obs[offset + 1] = market.worst_case_balance(agent_id) / max(cfg.initial_cash, 1)
            obs[offset + 2] = 1.0 if job is not None else 0.0
            if job is not None:
                obs[offset + 3] = job['target'] / max(self.F, 1)
                obs[offset + 4] = 1.0 if job['type'] == 'prove' else 0.0
            offset += 5

            # Per-agent offer presence vector
            for phi in range(self.F):
                for offer in market.offers.values():
                    if offer.poster == agent_id and offer.contract.target == phi:
                        obs[offset] = 1.0
                        break
                offset += 1

        # Global formula features (4*F)
        for phi in range(self.F):
            obs[offset] = float(pub_lib.is_concrete(phi))
            obs[offset + 1] = float(pub_lib.is_resolved(phi))
            # Keep the global observation aligned in size, but hide latent truth so
            # the critic cannot solve the game as a static oracle lookup.
            obs[offset + 2] = 0.0
            obs[offset + 3] = graph.difficulty_map[phi]
            offset += 4

        # Market offer slots (max_offers * 6)
        offer_ids = market.get_offer_ids_sorted()
        for slot_idx in range(self.max_offers):
            base = offset + slot_idx * 6
            if slot_idx < len(offer_ids):
                offer = market.offers[offer_ids[slot_idx]]
                obs[base] = offer.contract.target / max(self.F, 1)
                obs[base + 1] = offer.contract.deadline / max(cfg.max_timestep, 1)
                obs[base + 2] = offer.contract.loss
                obs[base + 3] = offer.price
                obs[base + 4] = 1.0 if offer.side == 'long' else 0.0
                obs[base + 5] = offer.quantity / max(cfg.target_init_max_quantity, 1)
        offset += self.max_offers * 6

        # Timestep
        obs[offset] = env_state['timestep'] / max(cfg.max_timestep, 1)
        offset += 1

        return obs

    def get_available_actions(self, agent_id: int, env_state: dict) -> np.ndarray:
        """Return binary mask of available actions for an agent."""
        avail = np.zeros(self.action_dim, dtype=np.float32)
        cfg = env_state['config']
        graph = env_state['graph']
        lib = env_state['libraries'][agent_id]
        market = env_state['market']
        job = env_state['jobs'][agent_id]

        # NoOp always available
        avail[self._noop] = 1.0

        # Prove/Conj only if no active job
        if job is None:
            for phi in range(self.F):
                for b in range(self.B):
                    # Prove: allow any unresolved target whose dependencies are currently met.
                    # The proof kernel itself decides whether the attempt can succeed.
                    if not lib.is_resolved(phi) and lib.is_concrete(phi):
                        deps = graph.get_deps(phi)
                        if deps.issubset(lib.resolved_formulas()):
                            avail[self._prove_start + phi * self.B + b] = 1.0
                    # Conj: must have ghost formulas
                    if not lib.is_concrete(phi):
                        avail[self._conj_start + phi * self.B + b] = 1.0

        # Pub: can publish any privately resolved formula
        for phi in range(self.F):
            if lib.is_resolved(phi) and not env_state['public_library'].is_resolved(phi):
                avail[self._pub_start + phi] = 1.0

        # Qry: can query any concrete formula
        for phi in range(self.F):
            if lib.is_concrete(phi):
                avail[self._qry_start + phi] = 1.0

        # Market NoOp always available
        avail[self._market_noop] = 1.0

        # CreatePost: check collateral and offer limits
        if market._agent_offer_count(agent_id) < self.max_own_offers and len(market.offers) < self.max_offers:
            for phi in range(self.F):
                for d in range(self.D):
                    for l in range(self.L):
                        for p in range(self.P):
                            for side_idx in range(2):
                                idx = (
                                    self._create_start
                                    + phi * self.D * self.L * self.P * 2
                                    + d * self.L * self.P * 2
                                    + l * self.P * 2
                                    + p * 2
                                    + side_idx
                                )
                                avail[idx] = 1.0

        # Accept: available offer slots
        offer_ids = market.get_offer_ids_sorted()
        for slot_idx in range(min(len(offer_ids), self.max_offers)):
            offer = market.offers[offer_ids[slot_idx]]
            if offer.poster != agent_id:
                avail[self._accept_start + slot_idx] = 1.0

        # Cancel: own offers mapped to slots
        own_offers = [oid for oid in offer_ids if market.offers[oid].poster == agent_id]
        for slot_idx in range(min(len(own_offers), self.max_own_offers)):
            avail[self._cancel_start + slot_idx] = 1.0

        return avail
