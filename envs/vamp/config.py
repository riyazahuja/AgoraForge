"""VampConfig: single dataclass holding all VAMP hyperparameters.

Captures the complete generator tuple:
    (N, P, L, beta, {A^alpha_mkt}, {Adm^alpha_mkt}, T_mkt, gamma, b_0)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class VampConfig:
    # ── 1a. Agent & formula universes ──
    F_size: int = 16                    # |F|, must be even (negation pairing: i <-> i + F_size//2)
    n_agents: int = 2                   # |N|

    # ── 1b. Latent environment structure (instance data) ──
    truth_map: Optional[np.ndarray] = None          # shape (F_size,), values {0,1}, T(i)+T(neg(i))=1
    difficulty_map: Optional[np.ndarray] = None      # shape (F_size,), values (0,1), defined where T=1
    dependency_adj: Optional[Dict[int, Set[int]]] = None  # delta*(phi) for each phi in F_true, must be acyclic
    utility_weights: Optional[Dict[Tuple[int, int], float]] = None  # w(psi,phi) in (0,1], w=1 iff psi in delta*(phi)

    # ── 1c. Initial world state (b_0 specification) ──
    initial_concrete: Optional[Set[int]] = None      # initial C_0 for all libraries (default: empty)
    initial_resolved: Optional[Dict[int, Tuple[Set[int], int, int]]] = None  # initial RS_0 (default: empty)
    initial_public_concrete_prob: float = 0.25       # probability of seeding a formula pair as public concrete at t=0
    initial_cash: float = 10.0                       # per-agent starting cash
    gamma: float = 0.99                              # discount factor
    max_timestep: int = 100                          # episode length / finite horizon

    # ── 1d. Proof kernel hyperparameters ──
    kappa: float = 0.5                  # budget concavity, h(tau) = tau^kappa, kappa in (0,1]
    lambda_diff: float = 1.0            # difficulty-to-rate transform, r_diff(phi) = exp(-lambda_diff * d(phi))
    alpha_util: float = 1.0             # utility sensitivity exponent in s_mass = sum w(psi,phi)^alpha
    rho_0: Optional[np.ndarray] = None  # baseline proof rate per agent, shape (n_agents,)
    rho_1: Optional[np.ndarray] = None  # utility-amplified proof rate per agent, shape (n_agents,)

    # ── 1e. Conjecture kernel hyperparameters ──
    beta_conj: float = 1.0              # opportunity mass exponent >= 1
    eta_0: Optional[np.ndarray] = None  # baseline conjecture rate per agent, shape (n_agents,)
    eta_1: Optional[np.ndarray] = None  # utility-amplified conjecture rate per agent, shape (n_agents,)
    kappa_conj_0: Optional[np.ndarray] = None  # selectivity baseline per agent, shape (n_agents,)
    kappa_conj_1: Optional[np.ndarray] = None  # selectivity budget sensitivity per agent, shape (n_agents,)
    phi_transform: str = 'identity'     # monotone transform for proposal scores ('identity' or 'log1p')

    # ── 1f. Query model hyperparameters ──
    n_buckets: int = 4                  # legacy field; no longer used by the parametric surrogate
    horizon_H: int = 20                 # finite horizon H used for query readout
    prior_a: float = 0.5               # legacy field; no longer used by the parametric surrogate
    prior_c: float = 1.0               # legacy field; no longer used by the parametric surrogate
    query_init_weight_std: float = 2.0
    query_global_lr: float = 0.03
    query_local_lr: float = 0.15
    query_private_truth_boost: float = 2.0
    query_public_truth_boost: float = 2.5

    # ── 1g. Market mechanism (Mechanism I: Collateralized Bilateral Contracts) ──
    budget_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    deadline_levels: List[int] = field(default_factory=lambda: [10, 25, 50])
    loss_levels: List[float] = field(default_factory=lambda: [0.25, 0.5])
    price_levels: List[float] = field(default_factory=lambda: [0.1 * i for i in range(1, 11)])
    max_offers: int = 8                 # max open offers in ledger
    max_own_offers: int = 4             # max offers per agent

    # Optional training-side shaping. These default to zero so the base game
    # remains unchanged unless explicitly enabled.
    operation_gas_fee: float = 0.0
    publish_resolution_bonus: float = 0.0
    target_init_prob: float = 0.5
    target_init_min_price: float = 0.1
    target_init_max_price: float = 0.3
    target_init_max_quantity: int = 4
    target_init_cash: float = 100.0

    # ── 1h. Implementation-side action simplification ──
    h_max: int = 0                      # query antecedent-width bound

    def __post_init__(self):
        # F_size even
        assert self.F_size % 2 == 0, "F_size must be even for negation pairing"

        # Default agent capability arrays
        if self.rho_0 is None:
            self.rho_0 = np.ones(self.n_agents) * 0.3
        if self.rho_1 is None:
            self.rho_1 = np.ones(self.n_agents) * 0.2
        if self.eta_0 is None:
            self.eta_0 = np.ones(self.n_agents) * 0.2
        if self.eta_1 is None:
            self.eta_1 = np.ones(self.n_agents) * 0.1
        if self.kappa_conj_0 is None:
            self.kappa_conj_0 = np.ones(self.n_agents) * 0.1
        if self.kappa_conj_1 is None:
            self.kappa_conj_1 = np.ones(self.n_agents) * 0.1

        # Validate hyperparameter ranges
        assert 0.0 < self.kappa <= 1.0, f"kappa must be in (0,1], got {self.kappa}"
        assert self.lambda_diff > 0, f"lambda_diff must be > 0, got {self.lambda_diff}"
        assert self.alpha_util >= 1.0, f"alpha_util must be >= 1, got {self.alpha_util}"
        assert self.beta_conj >= 1.0, f"beta_conj must be >= 1, got {self.beta_conj}"
        assert 0.0 <= self.initial_public_concrete_prob <= 1.0, \
            f"initial_public_concrete_prob must be in [0,1], got {self.initial_public_concrete_prob}"
        assert self.operation_gas_fee >= 0.0, \
            f"operation_gas_fee must be >= 0, got {self.operation_gas_fee}"
        assert self.publish_resolution_bonus >= 0.0, \
            f"publish_resolution_bonus must be >= 0, got {self.publish_resolution_bonus}"
        assert 0.0 <= self.target_init_prob <= 1.0, \
            f"target_init_prob must be in [0,1], got {self.target_init_prob}"
        assert 0.0 <= self.target_init_min_price <= 1.0, \
            f"target_init_min_price must be in [0,1], got {self.target_init_min_price}"
        assert 0.0 <= self.target_init_max_price <= 1.0, \
            f"target_init_max_price must be in [0,1], got {self.target_init_max_price}"
        assert self.target_init_min_price <= self.target_init_max_price, \
            "target_init_min_price must be <= target_init_max_price"
        assert self.target_init_max_quantity >= 0, \
            f"target_init_max_quantity must be >= 0, got {self.target_init_max_quantity}"
        assert self.target_init_cash >= 0.0, \
            f"target_init_cash must be >= 0, got {self.target_init_cash}"
        assert self.phi_transform in ('identity', 'log1p'), \
            f"phi_transform must be 'identity' or 'log1p', got {self.phi_transform}"

        # Validate agent array shapes
        for name in ('rho_0', 'rho_1', 'eta_0', 'eta_1', 'kappa_conj_0', 'kappa_conj_1'):
            arr = getattr(self, name)
            assert arr.shape == (self.n_agents,), \
                f"{name} must have shape ({self.n_agents},), got {arr.shape}"

        # Validate instance data if provided
        if self.truth_map is not None:
            self._validate_truth_map()
        if self.dependency_adj is not None:
            self._validate_dependency_adj()
        if self.utility_weights is not None and self.dependency_adj is not None:
            self._validate_utility_weights()

    def _validate_truth_map(self):
        assert self.truth_map.shape == (self.F_size,), \
            f"truth_map must have shape ({self.F_size},), got {self.truth_map.shape}"
        half = self.half_F
        for i in range(half):
            assert self.truth_map[i] + self.truth_map[i + half] == 1, \
                f"T({i}) + T({i + half}) must equal 1"

    def _validate_dependency_adj(self):
        # Check acyclicity via topological sort (Kahn's algorithm)
        true_formulas = set(self.dependency_adj.keys())
        in_degree = {phi: 0 for phi in true_formulas}
        for phi, deps in self.dependency_adj.items():
            for d in deps:
                if d in in_degree:
                    in_degree[d] = in_degree.get(d, 0)  # already initialized

        # Build forward edges: dep -> dependents
        forward = {phi: set() for phi in true_formulas}
        for phi, deps in self.dependency_adj.items():
            for d in deps:
                if d in forward:
                    forward[d].add(phi)
                    in_degree[phi] += 1

        queue = [phi for phi, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for neighbor in forward.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        assert visited == len(true_formulas), \
            "dependency_adj must induce an acyclic digraph"

    def _validate_utility_weights(self):
        for phi, deps in self.dependency_adj.items():
            for psi in deps:
                key = (psi, phi)
                assert key in self.utility_weights and self.utility_weights[key] == 1.0, \
                    f"w({psi},{phi}) must be 1.0 since {psi} in delta*({phi})"

    @property
    def half_F(self):
        return self.F_size // 2

    @property
    def n_budget_levels(self):
        return len(self.budget_levels)

    @property
    def n_deadline_levels(self):
        return len(self.deadline_levels)

    @property
    def n_loss_levels(self):
        return len(self.loss_levels)

    @property
    def n_price_levels(self):
        return len(self.price_levels)

    def neg(self, i: int) -> int:
        """Return the negation index of formula i."""
        half = self.half_F
        return i + half if i < half else i - half
