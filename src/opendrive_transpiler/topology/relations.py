"""Successor and adjacency inference.

lanelet2 stores no relations at all -- `RoutingGraph` derives them from geometry
every time. Since lanelet2 is not available to us, we derive them here, using the
same rules upstream uses:

* **Successor**: both bounds of A end exactly where the corresponding bounds of B
  begin. Endpoint-only, not full-bound matching -- that is what makes a run of
  lanelets built from shared points into a chain.
* **Lateral adjacency**: A's right bound *is* B's left bound. Boundary sharing is
  the primary signal, and the node-signature comparison catches both the shared
  object and the merely-coincident duplicate. A reversed match means B runs
  against A.

Conflict (interior overlap) is deliberately not computed: it needs polygon
intersection and only feeds junction heuristics, which this release does not ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir.model import LaneletIR
from .index import NodeIndex


@dataclass
class Relations:
    successors: dict[int, list[int]] = field(default_factory=dict)
    """Lanelet index -> indices of lanelets that follow it."""
    predecessors: dict[int, list[int]] = field(default_factory=dict)
    right_of: dict[int, int] = field(default_factory=dict)
    """Lanelet index -> the lanelet immediately to its right, if any."""
    left_of: dict[int, int] = field(default_factory=dict)
    antiparallel: set[tuple[int, int]] = field(default_factory=set)
    """Adjacent pairs whose travel directions oppose."""

    def successor_of(self, index: int) -> list[int]:
        return self.successors.get(index, [])

    def predecessor_of(self, index: int) -> list[int]:
        return self.predecessors.get(index, [])

    def is_branch(self, index: int) -> bool:
        return len(self.successor_of(index)) > 1 or len(self.predecessor_of(index)) > 1


def infer(lanelets: list[LaneletIR], index: NodeIndex) -> Relations:
    relations = Relations()
    count = len(lanelets)

    left_signature = [index.signature(ll.left) for ll in lanelets]
    right_signature = [index.signature(ll.right) for ll in lanelets]
    left_start = [sig[0] for sig in left_signature]
    left_end = [sig[-1] for sig in left_signature]
    right_start = [sig[0] for sig in right_signature]
    right_end = [sig[-1] for sig in right_signature]

    # -- longitudinal --------------------------------------------------------
    # Bucket by the (leftStart, rightStart) node pair so this stays linear in the
    # number of lanelets rather than quadratic.
    by_start: dict[tuple[int, int], list[int]] = {}
    for i in range(count):
        by_start.setdefault((left_start[i], right_start[i]), []).append(i)

    for i in range(count):
        for j in by_start.get((left_end[i], right_end[i]), ()):
            if j == i:
                continue
            relations.successors.setdefault(i, []).append(j)
            relations.predecessors.setdefault(j, []).append(i)

    # -- lateral -------------------------------------------------------------
    by_left: dict[tuple[int, ...], list[int]] = {}
    for i in range(count):
        by_left.setdefault(left_signature[i], []).append(i)

    for i in range(count):
        signature = right_signature[i]
        for j in by_left.get(signature, ()):
            if j != i:
                _link_lateral(relations, i, j, reversed_pair=False)
        # A neighbour running the other way shares the boundary in reverse.
        for j in by_left.get(tuple(reversed(signature)), ()):
            if j != i:
                _link_lateral(relations, i, j, reversed_pair=True)

    return relations


def _link_lateral(relations: Relations, left: int, right: int, *, reversed_pair: bool) -> None:
    """Record that `right` sits immediately to the right of `left`."""
    relations.right_of.setdefault(left, right)
    relations.left_of.setdefault(right, left)
    if reversed_pair:
        relations.antiparallel.add((min(left, right), max(left, right)))
