"""FormulaGraph: static latent structure holder for VAMP environments.

Constructed from VampConfig instance data fields. Holds truth map,
difficulty map, dependency adjacency, and utility weights.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple
import numpy as np


class FormulaGraph:
    """Immutable latent structure of a VAMP instance."""

    def __init__(
        self,
        F_size: int,
        truth_map: np.ndarray,
        difficulty_map: np.ndarray,
        dependency_adj: Dict[int, Set[int]],
        utility_weights: Dict[Tuple[int, int], float],
    ):
        self.F_size = F_size
        self.half_F = F_size // 2
        self.truth_map = truth_map.copy()
        self.difficulty_map = difficulty_map.copy()
        self.dependency_adj = {k: set(v) for k, v in dependency_adj.items()}
        self.utility_weights = dict(utility_weights)

        # Derived: set of true formulas
        self.true_formulas: Set[int] = {i for i in range(F_size) if truth_map[i] == 1}

        # Validate
        self._validate_dag()
        self._validate_weights()

    def neg(self, i: int) -> int:
        """Negation: i <-> i + half_F."""
        return i + self.half_F if i < self.half_F else i - self.half_F

    def is_true(self, phi: int) -> bool:
        return self.truth_map[phi] == 1

    def get_difficulty(self, phi: int) -> float:
        return float(self.difficulty_map[phi])

    def get_deps(self, phi: int) -> Set[int]:
        """Return delta*(phi), the dependency set of phi."""
        return self.dependency_adj.get(phi, set())

    def get_weight(self, psi: int, phi: int) -> float:
        """Return w(psi, phi)."""
        return self.utility_weights.get((psi, phi), 0.0)

    def in_degree(self, phi: int) -> int:
        """Number of direct dependencies of phi."""
        return len(self.dependency_adj.get(phi, set()))

    def out_degree(self, phi: int) -> int:
        """Number of formulas that directly depend on phi."""
        count = 0
        for deps in self.dependency_adj.values():
            if phi in deps:
                count += 1
        return count

    def ghost_formulas(self, concrete: Set[int]) -> Set[int]:
        """G(L) = F \\ C -- formulas not yet concrete."""
        return set(range(self.F_size)) - concrete

    def _validate_dag(self):
        """Validate that dependency_adj induces an acyclic digraph via topological sort."""
        nodes = set(self.dependency_adj.keys())
        in_degree = {phi: 0 for phi in nodes}
        forward = {phi: set() for phi in nodes}
        for phi, deps in self.dependency_adj.items():
            for d in deps:
                if d in nodes:
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
        assert visited == len(nodes), "dependency_adj must induce an acyclic digraph"

    def _validate_weights(self):
        """Validate w(psi,phi)=1 iff psi in delta*(phi)."""
        for phi, deps in self.dependency_adj.items():
            for psi in deps:
                w = self.utility_weights.get((psi, phi), 0.0)
                assert w == 1.0, f"w({psi},{phi}) must be 1.0 since {psi} in delta*({phi})"

    @classmethod
    def from_config(cls, cfg) -> FormulaGraph:
        """Construct from a VampConfig with populated instance data."""
        return cls(
            F_size=cfg.F_size,
            truth_map=cfg.truth_map,
            difficulty_map=cfg.difficulty_map,
            dependency_adj=cfg.dependency_adj,
            utility_weights=cfg.utility_weights,
        )

    @classmethod
    def random(
        cls,
        F_size: int = 16,
        density: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> FormulaGraph:
        """Generate a valid random FormulaGraph instance.

        Args:
            F_size: formula universe size (must be even)
            density: probability of an edge in the dependency DAG
            rng: numpy random generator
        """
        assert F_size % 2 == 0
        if rng is None:
            rng = np.random.default_rng()

        half = F_size // 2

        # Truth map: for each pair (i, i+half), assign one as true
        truth_map = np.zeros(F_size, dtype=np.int32)
        for i in range(half):
            if rng.random() < 0.5:
                truth_map[i] = 1
            else:
                truth_map[i + half] = 1

        # Difficulty map: uniform in (0,1) for true formulas
        difficulty_map = np.zeros(F_size, dtype=np.float64)
        for i in range(F_size):
            if truth_map[i] == 1:
                difficulty_map[i] = rng.uniform(0.05, 0.95)

        # Dependency adjacency: random DAG among true formulas
        true_formulas = sorted([i for i in range(F_size) if truth_map[i] == 1])
        dependency_adj: Dict[int, Set[int]] = {phi: set() for phi in true_formulas}

        # Use topological ordering = sorted order to ensure acyclicity
        # Edge from true_formulas[j] -> true_formulas[i] means j is a dep of i (j < i)
        for idx_i, phi in enumerate(true_formulas):
            for idx_j in range(idx_i):
                psi = true_formulas[idx_j]
                if rng.random() < density:
                    dependency_adj[phi].add(psi)

        # Utility weights: w(psi,phi) = 1.0 for deps, random in (0,1) for other pairs
        utility_weights: Dict[Tuple[int, int], float] = {}
        for phi in true_formulas:
            for psi in true_formulas:
                if psi == phi:
                    continue
                if psi in dependency_adj[phi]:
                    utility_weights[(psi, phi)] = 1.0
                else:
                    w = rng.uniform(0.01, 0.99)
                    utility_weights[(psi, phi)] = w

        return cls(
            F_size=F_size,
            truth_map=truth_map,
            difficulty_map=difficulty_map,
            dependency_adj=dependency_adj,
            utility_weights=utility_weights,
        )
