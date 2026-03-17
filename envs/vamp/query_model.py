"""QueryModel: bucketed discrete-time survival model for VAMP.

Per-agent model tracking proof/conjecture success statistics in
discretized buckets. Provides posterior estimates of success probability
and expected completion time.

State: N[b,k], D[b,k] for b in B, k in {1,...,H}
    Hazard: lambda_b(k) = (N[b,k] + a) / (D[b,k] + a + c)
    Readout: p_hat = 1 - prod(1 - lambda_b(k)), tau_hat = E[T|T<=H] / p_hat
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from envs.vamp.formula_graph import FormulaGraph
from envs.vamp.library import Library


class QueryModel:
    """Bucketed survival model for estimating task completion."""

    def __init__(self, n_buckets: int, horizon_H: int, prior_a: float, prior_c: float):
        self.n_buckets = n_buckets
        self.horizon_H = horizon_H
        self.prior_a = prior_a
        self.prior_c = prior_c

        # Sufficient statistics: N[b,k] = successes, D[b,k] = at-risk counts
        self.N = np.zeros((n_buckets, horizon_H), dtype=np.float64)
        self.D = np.zeros((n_buckets, horizon_H), dtype=np.float64)

    @classmethod
    def from_config(cls, cfg) -> QueryModel:
        return cls(
            n_buckets=cfg.n_buckets,
            horizon_H=cfg.horizon_H,
            prior_a=cfg.prior_a,
            prior_c=cfg.prior_c,
        )

    def bucket_map(self, graph: FormulaGraph, library: Library, phi: int) -> int:
        """Map a formula to a bucket index via discretized difficulty."""
        d = graph.get_difficulty(phi)
        bucket = int(d * self.n_buckets)
        return min(bucket, self.n_buckets - 1)

    def query(self, graph: FormulaGraph, library: Library, phi: int) -> Tuple[float, float]:
        """Compute posterior estimates (p_hat, tau_hat) for formula phi.

        p_hat = 1 - prod_{k=1}^{H} (1 - lambda_b(k))
        tau_hat = E[T | T <= H] / p_hat
        """
        b = self.bucket_map(graph, library, phi)

        # Compute hazard rates and survival
        survival = 1.0
        expected_time = 0.0

        for k in range(self.horizon_H):
            hazard = (self.N[b, k] + self.prior_a) / (self.D[b, k] + self.prior_a + self.prior_c)
            hazard = np.clip(hazard, 0.0, 1.0)

            # P(T = k+1) = survival_to_k * hazard_k
            p_at_k = survival * hazard
            expected_time += p_at_k * (k + 1)

            survival *= (1.0 - hazard)

        p_hat = 1.0 - survival
        tau_hat = expected_time / p_hat if p_hat > 1e-12 else float(self.horizon_H)

        return p_hat, tau_hat

    def update(self, bucket: int, tau: int, success: bool) -> None:
        """Update from observation (bucket, tau, success).

        D[b,k] += 1 for k <= min(tau, H)
        If success and tau <= H: N[b, tau-1] += 1
        """
        effective_tau = min(tau, self.horizon_H)
        for k in range(effective_tau):
            self.D[bucket, k] += 1.0
        if success and tau <= self.horizon_H:
            self.N[bucket, tau - 1] += 1.0

    def reset(self) -> None:
        """Reset sufficient statistics."""
        self.N[:] = 0.0
        self.D[:] = 0.0

    def copy(self) -> QueryModel:
        """Deep copy."""
        qm = QueryModel(self.n_buckets, self.horizon_H, self.prior_a, self.prior_c)
        qm.N = self.N.copy()
        qm.D = self.D.copy()
        return qm
