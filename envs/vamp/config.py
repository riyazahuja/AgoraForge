"""VampConfig: single dataclass holding all VAMP hyperparameters."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class VampConfig:
    # ── 1a. Agent & formula universes ──
    num_theorems: int = 4               # base theorem count n; runtime formula universe has size 2n
    F_size: Optional[int] = None        # deprecated alias / derived runtime formula count
    n_agents: int = 2                   # |N|

    # ── 1b. Latent environment structure (instance data) ──
    truth_map: Optional[np.ndarray] = None          # shape (num_theorems,), values {0,1}; picks true pair member
    difficulty_map: Optional[np.ndarray] = None     # shape (num_theorems,), values (0,1)
    dependency_adj: Optional[Dict[int, Set[int]]] = None  # theorem-id DAG over {0,...,num_theorems-1}
    utility_weights: Optional[Dict[Tuple[int, int], float]] = None  # theorem-id weights, w(dep,target)

    # ── 1c. Initial world state (b_0 specification) ──
    initial_concrete: Optional[Set[int]] = None
    initial_resolved: Optional[Dict[int, Tuple[Set[int], int, int]]] = None
    initial_public_concrete_prob: float = 0.25
    initial_cash: float = 10.0
    gamma: float = 0.99
    max_timestep: int = 100

    # ── 1d. Proof kernel hyperparameters ──
    kappa: float = 0.5
    lambda_diff: float = 1.0
    alpha_util: float = 1.0
    rho_0: Optional[np.ndarray] = None
    rho_1: Optional[np.ndarray] = None

    # ── 1e. Conjecture kernel hyperparameters ──
    beta_conj: float = 1.0
    eta_0: Optional[np.ndarray] = None
    eta_1: Optional[np.ndarray] = None
    kappa_conj_0: Optional[np.ndarray] = None
    kappa_conj_1: Optional[np.ndarray] = None
    phi_transform: str = 'identity'

    # ── 1f. Query model hyperparameters ──
    n_buckets: int = 4
    horizon_H: int = 20
    prior_a: float = 0.5
    prior_c: float = 1.0
    query_init_weight_std: float = 2.0
    query_global_lr: float = 0.03
    query_local_lr: float = 0.15
    query_private_truth_boost: float = 2.0
    query_public_truth_boost: float = 2.5

    # ── 1g. Market mechanism ──
    budget_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    deadline_levels: List[int] = field(default_factory=lambda: [10, 25, 50])
    loss_levels: List[float] = field(default_factory=lambda: [0.25, 0.5])
    price_levels: List[float] = field(default_factory=lambda: [0.1 * i for i in range(1, 11)])
    max_offers: int = 8
    max_own_offers: int = 4

    operation_gas_fee: float = 0.0
    publish_resolution_bonus: float = 0.0
    target_init_prob: float = 0.5
    target_init_min_price: float = 0.1
    target_init_max_price: float = 0.3
    target_init_max_quantity: int = 4
    target_init_cash: float = 100.0

    # ── 1h. Implementation-side action simplification ──
    h_max: int = 0

    def __post_init__(self):
        if self.F_size is None:
            self.F_size = 2 * int(self.num_theorems)
        else:
            self.F_size = int(self.F_size)
            assert self.F_size % 2 == 0, "F_size must be even for negation pairing"
            inferred_num_theorems = self.F_size // 2
            if self.num_theorems != 4 and self.num_theorems != inferred_num_theorems:
                raise AssertionError(
                    f"num_theorems={self.num_theorems} is inconsistent with F_size={self.F_size}"
                )
            self.num_theorems = inferred_num_theorems

        self.num_theorems = int(self.num_theorems)
        assert self.num_theorems > 0, "num_theorems must be positive"
        assert self.F_size == 2 * self.num_theorems, "runtime F_size must equal 2 * num_theorems"

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

        assert 0.0 < self.kappa <= 1.0, f"kappa must be in (0,1], got {self.kappa}"
        assert self.lambda_diff > 0, f"lambda_diff must be > 0, got {self.lambda_diff}"
        assert self.alpha_util >= 1.0, f"alpha_util must be >= 1, got {self.alpha_util}"
        assert self.beta_conj >= 1.0, f"beta_conj must be >= 1, got {self.beta_conj}"
        assert 0.0 <= self.initial_public_concrete_prob <= 1.0
        assert self.operation_gas_fee >= 0.0
        assert self.publish_resolution_bonus >= 0.0
        assert 0.0 <= self.target_init_prob <= 1.0
        assert 0.0 <= self.target_init_min_price <= 1.0
        assert 0.0 <= self.target_init_max_price <= 1.0
        assert self.target_init_min_price <= self.target_init_max_price
        assert self.target_init_max_quantity >= 0
        assert self.target_init_cash >= 0.0
        assert self.phi_transform in ('identity', 'log1p')

        for name in ('rho_0', 'rho_1', 'eta_0', 'eta_1', 'kappa_conj_0', 'kappa_conj_1'):
            arr = getattr(self, name)
            assert arr.shape == (self.n_agents,), \
                f"{name} must have shape ({self.n_agents},), got {arr.shape}"

        if self.truth_map is not None:
            self._validate_truth_map()
        if self.difficulty_map is not None:
            self._validate_difficulty_map()
        if self.dependency_adj is not None:
            self._validate_dependency_adj()
        if self.utility_weights is not None and self.dependency_adj is not None:
            self._validate_utility_weights()

    def _validate_truth_map(self):
        assert self.truth_map.shape == (self.num_theorems,), \
            f"truth_map must have shape ({self.num_theorems},), got {self.truth_map.shape}"
        assert np.all(np.isin(self.truth_map, [0, 1])), "truth_map must contain only 0/1 pair-member indicators"

    def _validate_difficulty_map(self):
        assert self.difficulty_map.shape == (self.num_theorems,), \
            f"difficulty_map must have shape ({self.num_theorems},), got {self.difficulty_map.shape}"

    def _validate_dependency_adj(self):
        nodes = set(range(self.num_theorems))
        for theorem_id, deps in self.dependency_adj.items():
            assert theorem_id in nodes, f"invalid theorem id {theorem_id} in dependency_adj"
            for dep in deps:
                assert dep in nodes, f"invalid dependency id {dep} in dependency_adj[{theorem_id}]"

        in_degree = {theorem_id: 0 for theorem_id in nodes}
        forward = {theorem_id: set() for theorem_id in nodes}
        for theorem_id, deps in self.dependency_adj.items():
            for dep in deps:
                forward[dep].add(theorem_id)
                in_degree[theorem_id] += 1

        queue = [theorem_id for theorem_id, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for neighbor in forward[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        assert visited == len(nodes), "dependency_adj must induce an acyclic digraph"

    def _validate_utility_weights(self):
        for theorem_id, deps in self.dependency_adj.items():
            for dep in deps:
                key = (dep, theorem_id)
                assert key in self.utility_weights and self.utility_weights[key] == 1.0, \
                    f"w({dep},{theorem_id}) must be 1.0 since {dep} is a dependency of {theorem_id}"

    @property
    def half_F(self):
        return self.num_theorems

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

    def formula_from_pair(self, sign: int, theorem_id: int) -> int:
        assert sign in (0, 1), f"sign must be 0/1, got {sign}"
        assert 0 <= theorem_id < self.num_theorems, f"invalid theorem_id {theorem_id}"
        return theorem_id + sign * self.num_theorems

    def pair_sign(self, phi: int) -> int:
        return 0 if phi < self.num_theorems else 1

    def theorem_id(self, phi: int) -> int:
        return phi % self.num_theorems

    def neg(self, i: int) -> int:
        theorem_id = self.theorem_id(i)
        return self.formula_from_pair(1 - self.pair_sign(i), theorem_id)
