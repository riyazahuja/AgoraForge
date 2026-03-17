"""ProofKernel: stochastic proof completion model for VAMP.

Implements:
    p_proof^alpha([Gamma |- phi], tau) = g([Gamma |- phi]) * (1 - exp(-rho^alpha * tau^kappa))

Where:
    g([Gamma |- phi]) = 1[T(phi)=1] * 1[delta*(phi) subset Gamma]    (feasibility gate)
    r_diff(phi) = exp(-lambda_diff * d(phi))
    s_mass([Gamma |- phi]) = sum_{psi in Gamma} w(psi, phi)^alpha_util
    s([Gamma |- phi]) = log(1 + s_mass)
    rho^alpha = r_diff(phi) * (rho_0^alpha + rho_1^alpha * s)
"""

from __future__ import annotations

import numpy as np

from envs.vamp.formula_graph import FormulaGraph
from envs.vamp.library import Library


class ProofKernel:
    """Proof completion kernel parameterized by VampConfig."""

    def __init__(self, kappa: float, lambda_diff: float, alpha_util: float,
                 rho_0: np.ndarray, rho_1: np.ndarray):
        self.kappa = kappa
        self.lambda_diff = lambda_diff
        self.alpha_util = alpha_util
        self.rho_0 = rho_0.copy()
        self.rho_1 = rho_1.copy()

    @classmethod
    def from_config(cls, cfg) -> ProofKernel:
        return cls(
            kappa=cfg.kappa,
            lambda_diff=cfg.lambda_diff,
            alpha_util=cfg.alpha_util,
            rho_0=cfg.rho_0,
            rho_1=cfg.rho_1,
        )

    def _feasibility_gate(self, graph: FormulaGraph, library: Library, phi: int) -> bool:
        """g([Gamma |- phi]) = 1[T(phi)=1] * 1[delta*(phi) subset resolved]."""
        if not graph.is_true(phi):
            return False
        deps = graph.get_deps(phi)
        resolved = library.resolved_formulas()
        return deps.issubset(resolved)

    def _rate(self, agent_id: int, graph: FormulaGraph, library: Library, phi: int) -> float:
        """Compute rho^alpha for agent on sequent [Gamma |- phi]."""
        resolved = library.resolved_formulas()

        # r_diff(phi) = exp(-lambda_diff * d(phi))
        r_diff = np.exp(-self.lambda_diff * graph.get_difficulty(phi))

        # s_mass = sum_{psi in Gamma} w(psi, phi)^alpha_util
        s_mass = 0.0
        for psi in resolved:
            w = graph.get_weight(psi, phi)
            if w > 0:
                s_mass += w ** self.alpha_util

        # s = log(1 + s_mass)
        s = np.log1p(s_mass)

        # rho^alpha = r_diff * (rho_0 + rho_1 * s)
        rho = r_diff * (self.rho_0[agent_id] + self.rho_1[agent_id] * s)
        return rho

    def success_probability(
        self,
        agent_id: int,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        tau_eff: float,
    ) -> float:
        """p_proof^alpha([Gamma |- phi], tau) = g * (1 - exp(-rho * tau^kappa))."""
        if not self._feasibility_gate(graph, library, phi):
            return 0.0
        rho = self._rate(agent_id, graph, library, phi)
        h_tau = tau_eff ** self.kappa
        return 1.0 - np.exp(-rho * h_tau)

    def sample(
        self,
        agent_id: int,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        tau_eff: float,
        rng: np.random.Generator,
    ) -> bool:
        """Sample proof outcome. Dependencies are deterministic delta*(phi)."""
        p = self.success_probability(agent_id, graph, library, phi, tau_eff)
        return bool(rng.random() < p)
