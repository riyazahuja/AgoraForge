"""FormulaGraph: lifted paired-formula view of a theorem-level VAMP instance."""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np


class FormulaGraph:
    """Immutable latent structure of a VAMP instance."""

    def __init__(
        self,
        num_theorems: int,
        truth_map: np.ndarray,
        difficulty_map: np.ndarray,
        dependency_adj: Dict[int, Set[int]],
        utility_weights: Dict[Tuple[int, int], float],
    ):
        self.num_theorems = int(num_theorems)
        self.F_size = 2 * self.num_theorems
        self.half_F = self.num_theorems

        self.theorem_truth_map = truth_map.copy()
        self.theorem_difficulty_map = difficulty_map.copy()
        self.theorem_dependency_adj = {k: set(v) for k, v in dependency_adj.items()}
        self.theorem_utility_weights = dict(utility_weights)

        self.truth_map = np.zeros(self.F_size, dtype=np.int32)
        self.difficulty_map = np.zeros(self.F_size, dtype=np.float64)
        self.dependency_adj: Dict[int, Set[int]] = {}
        self.utility_weights: Dict[Tuple[int, int], float] = {}

        for theorem_id in range(self.num_theorems):
            true_phi = self.true_formula(theorem_id)
            false_phi = self.neg(true_phi)
            self.truth_map[true_phi] = 1
            self.truth_map[false_phi] = 0
            difficulty = float(self.theorem_difficulty_map[theorem_id])
            self.difficulty_map[true_phi] = difficulty
            self.difficulty_map[false_phi] = difficulty

        for theorem_id in range(self.num_theorems):
            true_phi = self.true_formula(theorem_id)
            lifted_deps = {self.true_formula(dep) for dep in self.theorem_dependency_adj.get(theorem_id, set())}
            self.dependency_adj[true_phi] = lifted_deps

        for src_theorem in range(self.num_theorems):
            src_phi = self.true_formula(src_theorem)
            for dst_theorem in range(self.num_theorems):
                if src_theorem == dst_theorem:
                    continue
                dst_phi = self.true_formula(dst_theorem)
                weight = float(self.theorem_utility_weights.get((src_theorem, dst_theorem), 0.0))
                if weight > 0.0:
                    self.utility_weights[(src_phi, dst_phi)] = weight

        self.true_formulas: Set[int] = {
            self.true_formula(theorem_id) for theorem_id in range(self.num_theorems)
        }

        self._validate_dag()
        self._validate_weights()

    def formula_from_pair(self, sign: int, theorem_id: int) -> int:
        return theorem_id + sign * self.num_theorems

    def pair_sign(self, phi: int) -> int:
        return 0 if phi < self.num_theorems else 1

    def theorem_id(self, phi: int) -> int:
        return phi % self.num_theorems

    def neg(self, i: int) -> int:
        theorem_id = self.theorem_id(i)
        return self.formula_from_pair(1 - self.pair_sign(i), theorem_id)

    def true_formula(self, theorem_id: int) -> int:
        sign = int(self.theorem_truth_map[theorem_id])
        return self.formula_from_pair(sign, theorem_id)

    def false_formula(self, theorem_id: int) -> int:
        return self.neg(self.true_formula(theorem_id))

    def is_true(self, phi: int) -> bool:
        return self.truth_map[phi] == 1

    def get_difficulty(self, phi: int) -> float:
        return float(self.theorem_difficulty_map[self.theorem_id(phi)])

    def get_deps(self, phi: int) -> Set[int]:
        return self.dependency_adj.get(phi, set())

    def get_weight(self, psi: int, phi: int) -> float:
        return self.utility_weights.get((psi, phi), 0.0)

    def in_degree(self, phi: int) -> int:
        return len(self.dependency_adj.get(phi, set()))

    def out_degree(self, phi: int) -> int:
        count = 0
        for deps in self.dependency_adj.values():
            if phi in deps:
                count += 1
        return count

    def ghost_formulas(self, concrete: Set[int]) -> Set[int]:
        return set(range(self.F_size)) - concrete

    def _validate_dag(self):
        nodes = set(self.true_formulas)
        in_degree = {phi: 0 for phi in nodes}
        forward = {phi: set() for phi in nodes}
        for phi, deps in self.dependency_adj.items():
            for dep in deps:
                if dep in nodes:
                    forward[dep].add(phi)
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
        for phi, deps in self.dependency_adj.items():
            for psi in deps:
                w = self.utility_weights.get((psi, phi), 0.0)
                assert w == 1.0, f"w({psi},{phi}) must be 1.0 since {psi} is a dependency of {phi}"

    @classmethod
    def from_config(cls, cfg) -> FormulaGraph:
        return cls(
            num_theorems=cfg.num_theorems,
            truth_map=cfg.truth_map,
            difficulty_map=cfg.difficulty_map,
            dependency_adj=cfg.dependency_adj,
            utility_weights=cfg.utility_weights,
        )

    @classmethod
    def random(
        cls,
        num_theorems: int = 4,
        density: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> FormulaGraph:
        if rng is None:
            rng = np.random.default_rng()

        truth_map = rng.integers(0, 2, size=num_theorems, dtype=np.int32)
        difficulty_map = rng.uniform(0.05, 0.95, size=num_theorems).astype(np.float64)

        dependency_adj: Dict[int, Set[int]] = {theorem_id: set() for theorem_id in range(num_theorems)}
        for theorem_id in range(num_theorems):
            for dep in range(theorem_id):
                if rng.random() < density:
                    dependency_adj[theorem_id].add(dep)

        utility_weights: Dict[Tuple[int, int], float] = {}
        for target in range(num_theorems):
            for source in range(num_theorems):
                if source == target:
                    continue
                if source in dependency_adj[target]:
                    utility_weights[(source, target)] = 1.0
                else:
                    utility_weights[(source, target)] = float(rng.uniform(0.01, 0.99))

        return cls(
            num_theorems=num_theorems,
            truth_map=truth_map,
            difficulty_map=difficulty_map,
            dependency_adj=dependency_adj,
            utility_weights=utility_weights,
        )
