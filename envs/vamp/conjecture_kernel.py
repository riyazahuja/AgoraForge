"""ConjectureKernel: stochastic conjecture proposal model for VAMP.

Implements:
    p_conj^alpha(L, s, tau) = 1 - exp(-eta^alpha(L,s) * tau^kappa)

Where:
    q(psi | L, [Gamma |- phi]) = 1[phi in RS] * w(phi, psi) + 1[phi not in RS] * w(psi, phi)
    m^alpha(L, s) = sum_{psi in G(L)} q(psi | L, s)^beta_conj
    eta^alpha(L, s) = eta_0 + eta_1 * log(1 + m^alpha)

Proposal distribution: softmax over ghosts with scores exp(kappa_conj(tau) * phi_transform(q)).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from envs.vamp.formula_graph import FormulaGraph
from envs.vamp.library import Library


class ConjectureKernel:
    """Conjecture proposal kernel parameterized by VampConfig."""

    def __init__(
        self,
        kappa: float,
        beta_conj: float,
        eta_0: np.ndarray,
        eta_1: np.ndarray,
        kappa_conj_0: np.ndarray,
        kappa_conj_1: np.ndarray,
        phi_transform: str = 'identity',
    ):
        self.kappa = kappa
        self.beta_conj = beta_conj
        self.eta_0 = eta_0.copy()
        self.eta_1 = eta_1.copy()
        self.kappa_conj_0 = kappa_conj_0.copy()
        self.kappa_conj_1 = kappa_conj_1.copy()
        self.phi_transform = phi_transform

    @classmethod
    def from_config(cls, cfg) -> ConjectureKernel:
        return cls(
            kappa=cfg.kappa,
            beta_conj=cfg.beta_conj,
            eta_0=cfg.eta_0,
            eta_1=cfg.eta_1,
            kappa_conj_0=cfg.kappa_conj_0,
            kappa_conj_1=cfg.kappa_conj_1,
            phi_transform=cfg.phi_transform,
        )

    def _anchor_utility(
        self,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        psi: int,
    ) -> float:
        """q(psi | L, [Gamma |- phi]) = 1[phi in RS]*w(phi,psi) + 1[phi not in RS]*w(psi,phi)."""
        if library.is_resolved(phi):
            return graph.get_weight(phi, psi)
        else:
            return graph.get_weight(psi, phi)

    def _opportunity_mass(
        self,
        agent_id: int,
        graph: FormulaGraph,
        library: Library,
        phi: int,
    ) -> float:
        """m^alpha(L, s) = sum_{psi in G(L)} q(psi | L, s)^beta_conj."""
        ghosts = graph.ghost_formulas(library.concrete)
        mass = 0.0
        for psi in ghosts:
            q = self._anchor_utility(graph, library, phi, psi)
            if q > 0:
                mass += q ** self.beta_conj
        return mass

    def _rate(
        self,
        agent_id: int,
        graph: FormulaGraph,
        library: Library,
        phi: int,
    ) -> float:
        """eta^alpha(L, s) = eta_0 + eta_1 * log(1 + m^alpha)."""
        m = self._opportunity_mass(agent_id, graph, library, phi)
        return self.eta_0[agent_id] + self.eta_1[agent_id] * np.log1p(m)

    def success_probability(
        self,
        agent_id: int,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        tau_eff: float,
    ) -> float:
        """p_conj^alpha(L, s, tau) = 1 - exp(-eta * tau^kappa)."""
        eta = self._rate(agent_id, graph, library, phi)
        h_tau = tau_eff ** self.kappa
        return 1.0 - np.exp(-eta * h_tau)

    def _apply_transform(self, x: float) -> float:
        """Apply phi_transform to a score."""
        if self.phi_transform == 'log1p':
            return np.log1p(x)
        return x  # identity

    def sample_proposal(
        self,
        agent_id: int,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        tau_eff: float,
        rng: np.random.Generator,
    ) -> Optional[int]:
        """Sample a ghost formula from the proposal distribution.

        Softmax over ghosts with scores exp(kappa_conj(tau) * phi_transform(q)).
        """
        ghosts = sorted(graph.ghost_formulas(library.concrete))
        if not ghosts:
            return None

        # kappa_conj(tau) = kappa_0 + kappa_1 * tau^kappa
        h_tau = tau_eff ** self.kappa
        selectivity = self.kappa_conj_0[agent_id] + self.kappa_conj_1[agent_id] * h_tau

        scores = np.zeros(len(ghosts))
        for idx, psi in enumerate(ghosts):
            q = self._anchor_utility(graph, library, phi, psi)
            scores[idx] = selectivity * self._apply_transform(q)

        # Softmax with numerical stability
        scores -= scores.max()
        probs = np.exp(scores)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(ghosts)) / len(ghosts)
        else:
            probs /= total

        chosen_idx = rng.choice(len(ghosts), p=probs)
        return ghosts[chosen_idx]

    def sample(
        self,
        agent_id: int,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        tau_eff: float,
        rng: np.random.Generator,
    ) -> Tuple[bool, Optional[int]]:
        """Sample conjecture outcome and proposed ghost formula.

        Returns (success, proposed_ghost_formula).
        """
        p = self.success_probability(agent_id, graph, library, phi, tau_eff)
        success = bool(rng.random() < p)
        if success:
            proposal = self.sample_proposal(agent_id, graph, library, phi, tau_eff, rng)
            return True, proposal
        return False, None
