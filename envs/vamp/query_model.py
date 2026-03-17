"""QueryModel: hierarchical feature-based surrogate of the proof kernel.

The model has three decoupled parts:
1. Per-formula truth belief updated only from resolution events.
2. A global feature model over observable graph/library features.
3. A per-formula residual that absorbs local misfit.

This is intentionally not a digital twin of the proof kernel. It uses
observable features and learns an empirical mapping to proof success.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from envs.vamp.formula_graph import FormulaGraph
from envs.vamp.library import Library


class QueryModel:
    """Agent-specific noisy feature model for proof success."""

    def __init__(
        self,
        F_size: int,
        horizon_H: int,
        kappa: float,
        init_weight_std: float,
        global_lr: float,
        local_lr: float,
        private_truth_boost: float,
        public_truth_boost: float,
        rng: np.random.Generator,
    ):
        self.F_size = F_size
        self.horizon_H = horizon_H
        self.kappa = kappa
        self.global_lr = global_lr
        self.local_lr = local_lr
        self.private_truth_boost = private_truth_boost
        self.public_truth_boost = public_truth_boost

        self.truth_logits = np.zeros(F_size, dtype=np.float64)
        self.w = rng.normal(0.0, init_weight_std, size=self.feature_dim).astype(np.float64)
        self.delta = np.zeros(F_size, dtype=np.float64)
        self.n_attempts = np.zeros(F_size, dtype=np.int32)
        self.n_successes = np.zeros(F_size, dtype=np.int32)

    @classmethod
    def from_config(cls, cfg, agent_id: int, rng: np.random.Generator) -> QueryModel:
        del agent_id
        return cls(
            F_size=cfg.F_size,
            horizon_H=cfg.horizon_H,
            kappa=cfg.kappa,
            init_weight_std=cfg.query_init_weight_std,
            global_lr=cfg.query_global_lr,
            local_lr=cfg.query_local_lr,
            private_truth_boost=cfg.query_private_truth_boost,
            public_truth_boost=cfg.query_public_truth_boost,
            rng=rng,
        )

    @property
    def feature_dim(self) -> int:
        return 7

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def _truth_prob(self, phi: int) -> float:
        return float(self._sigmoid(self.truth_logits[phi]))

    def _deps_feasible(self, graph: FormulaGraph, library: Library, phi: int) -> bool:
        return graph.get_deps(phi).issubset(library.resolved_formulas())

    def _features(self, graph: FormulaGraph, library: Library, phi: int) -> np.ndarray:
        deps = graph.get_deps(phi)
        resolved = library.resolved_formulas()
        deps_met = float(deps.issubset(resolved))
        deps_frac = len(deps & resolved) / max(len(deps), 1)
        difficulty = graph.get_difficulty(phi)

        support = 0.0
        for psi in resolved:
            weight = graph.get_weight(psi, phi)
            if weight > 0:
                support += weight

        in_deg = graph.in_degree(phi)
        out_deg = graph.out_degree(phi)

        return np.array(
            [
                1.0,
                deps_met,
                deps_frac,
                difficulty,
                np.log1p(support),
                np.log1p(in_deg),
                np.log1p(out_deg),
            ],
            dtype=np.float64,
        )

    def _rate(self, graph: FormulaGraph, library: Library, phi: int) -> Tuple[float, np.ndarray, float]:
        features = self._features(graph, library, phi)
        log_rate = float(self.w @ features + self.delta[phi])
        rate = float(self._softplus(log_rate))
        return rate, features, log_rate

    def success_probability(
        self,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        tau: float,
    ) -> float:
        if not self._deps_feasible(graph, library, phi):
            return 0.0
        truth_prob = self._truth_prob(phi)
        rate, _, _ = self._rate(graph, library, phi)
        h_tau = max(float(tau), 0.0) ** self.kappa
        return float(np.clip(truth_prob * (1.0 - np.exp(-rate * h_tau)), 0.0, 1.0))

    def query(self, graph: FormulaGraph, library: Library, phi: int) -> Tuple[float, float]:
        """Return (p_hat, tau_hat) for formula phi."""
        p_hat = 0.0
        expected_time = 0.0
        prev = 0.0
        for t in range(1, self.horizon_H + 1):
            curr = self.success_probability(graph, library, phi, float(t))
            p_at_t = max(curr - prev, 0.0)
            expected_time += p_at_t * t
            prev = curr
        p_hat = prev
        tau_hat = expected_time / p_hat if p_hat > 1e-12 else float(self.horizon_H)
        return float(p_hat), float(tau_hat)

    def confidence(self, phi: int) -> int:
        return int(self.n_attempts[phi])

    def observe_proof_result(
        self,
        graph: FormulaGraph,
        library: Library,
        phi: int,
        tau: float,
        success: bool,
    ) -> None:
        """Online update from an observed proof outcome."""
        self.n_attempts[phi] += 1
        if success:
            self.n_successes[phi] += 1

        pred = np.clip(self.success_probability(graph, library, phi, tau), 1e-6, 1.0 - 1e-6)
        y = 1.0 if success else 0.0
        dloss_dpred = (pred - y) / (pred * (1.0 - pred))

        truth_prob = self._truth_prob(phi)
        rate, features, log_rate = self._rate(graph, library, phi)
        h_tau = max(float(tau), 0.0) ** self.kappa
        exp_neg = np.exp(-rate * h_tau)

        dp_drate = truth_prob * h_tau * exp_neg
        drate_dlograte = float(self._sigmoid(log_rate))
        grad_common = dloss_dpred * dp_drate * drate_dlograte

        self.w -= self.global_lr * grad_common * features
        self.delta[phi] -= self.local_lr * grad_common

    def observe_resolution(self, graph: FormulaGraph, phi: int, boost: float) -> None:
        """Hard truth update from a resolution event."""
        neg_phi = graph.neg(phi)
        self.truth_logits[phi] = np.clip(self.truth_logits[phi] + boost, -8.0, 8.0)
        self.truth_logits[neg_phi] = np.clip(self.truth_logits[neg_phi] - boost, -8.0, 8.0)

    def observe_private_resolution(self, graph: FormulaGraph, phi: int) -> None:
        self.observe_resolution(graph, phi, self.private_truth_boost)

    def observe_public_resolution(self, graph: FormulaGraph, phi: int) -> None:
        self.observe_resolution(graph, phi, self.public_truth_boost)

    def reset(self) -> None:
        """Compatibility shim for older callers."""
        return None

    def copy(self) -> QueryModel:
        qm = QueryModel.__new__(QueryModel)
        qm.F_size = self.F_size
        qm.horizon_H = self.horizon_H
        qm.kappa = self.kappa
        qm.global_lr = self.global_lr
        qm.local_lr = self.local_lr
        qm.private_truth_boost = self.private_truth_boost
        qm.public_truth_boost = self.public_truth_boost
        qm.truth_logits = self.truth_logits.copy()
        qm.w = self.w.copy()
        qm.delta = self.delta.copy()
        qm.n_attempts = self.n_attempts.copy()
        qm.n_successes = self.n_successes.copy()
        return qm
