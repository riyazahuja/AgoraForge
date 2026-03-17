"""Library: per-agent knowledge state in VAMP.

Each agent maintains a private library with:
- concrete set C: formulas whose truth value is known
- resolved set RS: formulas that have been proved, with metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple


@dataclass
class ResolvedInfo:
    """Metadata for a resolved formula."""
    deps: Set[int]      # dependency set used in the proof
    solve_time: int     # timestep when resolved
    solver: int         # agent_id who resolved it


class Library:
    """Agent knowledge state: concrete set C and resolved set RS."""

    def __init__(self, F_size: int):
        self.F_size = F_size
        self.half_F = F_size // 2
        self.concrete: Set[int] = set()
        self.resolved: Dict[int, ResolvedInfo] = {}

    def neg(self, i: int) -> int:
        return i + self.half_F if i < self.half_F else i - self.half_F

    def add_concrete(self, phi: int) -> None:
        """Add phi and its negation to the concrete set (negation pairing)."""
        self.concrete.add(phi)
        self.concrete.add(self.neg(phi))

    def add_resolved(self, phi: int, deps: Set[int], time: int, solver: int) -> None:
        """Add phi to resolved set with metadata.

        Enforces: phi and neg(phi) cannot both be resolved.
        """
        neg_phi = self.neg(phi)
        assert neg_phi not in self.resolved, \
            f"Cannot resolve {phi}: negation {neg_phi} already resolved"
        self.resolved[phi] = ResolvedInfo(deps=set(deps), solve_time=time, solver=solver)
        # Resolving phi also makes it concrete
        self.add_concrete(phi)

    def resolved_formulas(self) -> Set[int]:
        """Return the set of resolved formula indices."""
        return set(self.resolved.keys())

    def is_concrete(self, phi: int) -> bool:
        return phi in self.concrete

    def is_resolved(self, phi: int) -> bool:
        return phi in self.resolved

    def dependency_closure(self, targets: Set[int]) -> Tuple[Set[int], Set[int]]:
        """Compute transitive closure for publication.

        Returns (closed_concrete, closed_resolved): the minimal sets needed
        to support the target resolved formulas.
        """
        closed_resolved: Set[int] = set()
        stack = list(targets)
        while stack:
            phi = stack.pop()
            if phi in closed_resolved:
                continue
            if phi not in self.resolved:
                continue
            closed_resolved.add(phi)
            for dep in self.resolved[phi].deps:
                if dep not in closed_resolved and dep in self.resolved:
                    stack.append(dep)

        # Concrete closure: all concrete formulas for resolved + their negations
        closed_concrete: Set[int] = set()
        for phi in closed_resolved:
            closed_concrete.add(phi)
            closed_concrete.add(self.neg(phi))

        return closed_concrete, closed_resolved

    def merge_from(
        self,
        source: Library,
        new_concrete: Set[int],
        new_resolved: Set[int],
    ) -> None:
        """Merge knowledge from source library (for propagation/publication).

        Adds specified concrete and resolved formulas from source into this library.
        Monotonic: never removes existing knowledge.
        """
        for phi in new_concrete:
            self.concrete.add(phi)
        for phi in new_resolved:
            if phi in source.resolved and phi not in self.resolved:
                info = source.resolved[phi]
                neg_phi = self.neg(phi)
                if neg_phi not in self.resolved:
                    self.resolved[phi] = ResolvedInfo(
                        deps=set(info.deps),
                        solve_time=info.solve_time,
                        solver=info.solver,
                    )
                    self.add_concrete(phi)

    def copy(self) -> Library:
        """Create a deep copy of this library."""
        lib = Library(self.F_size)
        lib.concrete = set(self.concrete)
        lib.resolved = {
            phi: ResolvedInfo(
                deps=set(info.deps),
                solve_time=info.solve_time,
                solver=info.solver,
            )
            for phi, info in self.resolved.items()
        }
        return lib
