"""Successor and adjacency inference.

lanelet2 stores no relations at all -- `RoutingGraph` derives them from geometry
every time. Since lanelet2 is not available to us, we derive them here:

* **Successor**: both bounds of A end exactly where the corresponding bounds of B
  begin. Endpoint-only, not full-bound matching -- that is what makes a run of
  lanelets built from shared points into a chain.
* **Lateral adjacency**: A and B share a boundary. *Which* bounds match does not
  matter, and neither does their traversal order, because a lanelet names its
  bounds relative to its own direction of travel.

That last point is the subtle one, and it is what two-way roads turn on.

On a two-way road the shared centre line is the **left** bound of *both*
lanelets -- each driver has the centre on their left -- stored in opposite order.
A rule that only compares "A's right to B's left" therefore never sees a two-way
pair at all, which is why all four pairings are checked here.

Direction is then measured from geometry rather than from which bound matched:
both bounds of a well-formed lanelet run along the direction of travel, so their
average gives a stable answer that a short or unevenly sampled bound cannot skew.
Two neighbours oppose when those averages point away from each other.

Conflict (interior overlap) is deliberately not computed: it needs polygon
intersection and only feeds junction heuristics beyond what this ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..geometry.vec import Vec2, dot2, normalize2
from ..ir.model import LaneletIR
from .index import NodeIndex


@dataclass
class Relations:
    successors: dict[int, list[int]] = field(default_factory=dict)
    """Lanelet index -> indices of lanelets that follow it."""
    predecessors: dict[int, list[int]] = field(default_factory=dict)
    right_of: dict[int, int] = field(default_factory=dict)
    """Lanelet index -> the same-direction lanelet immediately to its right."""
    left_of: dict[int, int] = field(default_factory=dict)
    opposing_of: dict[int, int] = field(default_factory=dict)
    """Lanelet index -> an adjacent lanelet travelling the other way."""
    opposing: set[tuple[int, int]] = field(default_factory=set)
    """Adjacent pairs whose travel directions genuinely oppose."""

    def successor_of(self, index: int) -> list[int]:
        return self.successors.get(index, [])

    def predecessor_of(self, index: int) -> list[int]:
        return self.predecessors.get(index, [])

    def is_branch(self, index: int) -> bool:
        return len(self.successor_of(index)) > 1 or len(self.predecessor_of(index)) > 1

    def opposes(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.opposing


def travel_direction(lanelet: LaneletIR) -> Vec2:
    """Which way traffic runs, as a unit vector.

    lanelet2 stores both bounds ordered along the direction of travel, so the two
    agree on a well-formed lanelet and averaging them is more stable than trusting
    either alone -- a short or unevenly sampled bound cannot skew the answer.

    When the two *disagree* by more than a right angle the lanelet is malformed:
    its own edges claim opposite directions. There is no correct answer then, so
    the left bound wins, deterministically, and `bounds_disagree` lets the caller
    report it.
    """
    left = _span(lanelet.left.points)
    right = _span(lanelet.right.points)
    combined = (left[0] + right[0], left[1] + right[1])
    if combined != (0.0, 0.0) and dot2(left, right) > 0.0:
        return normalize2(combined)
    return left if left != (0.0, 0.0) else right


def bounds_disagree(lanelet: LaneletIR) -> bool:
    """Whether a lanelet's two bounds claim opposite directions of travel."""
    left = _span(lanelet.left.points)
    right = _span(lanelet.right.points)
    if left == (0.0, 0.0) or right == (0.0, 0.0):
        return False
    return dot2(left, right) < 0.0


def _span(points) -> Vec2:
    """Unit vector from a boundary's first point to its last."""
    if len(points) < 2:
        return (0.0, 0.0)
    return normalize2((points[-1].x - points[0].x, points[-1].y - points[0].y))


def infer(lanelets: list[LaneletIR], index: NodeIndex) -> Relations:
    relations = Relations()
    count = len(lanelets)

    left_signature = [index.signature(ll.left) for ll in lanelets]
    right_signature = [index.signature(ll.right) for ll in lanelets]
    directions = [travel_direction(ll) for ll in lanelets]

    # -- longitudinal --------------------------------------------------------
    # Bucket by the (leftStart, rightStart) node pair so this stays linear in the
    # number of lanelets rather than quadratic.
    by_start: dict[tuple[int, int], list[int]] = {}
    for i in range(count):
        by_start.setdefault((left_signature[i][0], right_signature[i][0]), []).append(i)

    for i in range(count):
        ends = (left_signature[i][-1], right_signature[i][-1])
        for j in by_start.get(ends, ()):
            if j == i:
                continue
            relations.successors.setdefault(i, []).append(j)
            relations.predecessors.setdefault(j, []).append(i)

    # -- lateral -------------------------------------------------------------
    # Any of a lanelet's bounds may be shared with any of a neighbour's, so all
    # four pairings are checked. Bucketing on the *unordered* boundary keeps this
    # linear: a line string and its reverse hash to the same key.
    def key(signature: tuple[int, ...]) -> tuple[int, ...]:
        return min(signature, tuple(reversed(signature)))

    holders: dict[tuple[int, ...], list[tuple[int, str]]] = {}
    for i in range(count):
        holders.setdefault(key(left_signature[i]), []).append((i, "left"))
        holders.setdefault(key(right_signature[i]), []).append((i, "right"))

    seen: set[tuple[int, int]] = set()
    for owners in holders.values():
        for position, (i, _side_i) in enumerate(owners):
            for j, _side_j in owners[position + 1 :]:
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                seen.add(pair)
                _classify(relations, lanelets, directions, i, j)

    return relations


def _classify(
    relations: Relations,
    lanelets: list[LaneletIR],
    directions: list[Vec2],
    i: int,
    j: int,
) -> None:
    """Record how two boundary-sharing lanelets sit relative to each other."""
    if dot2(directions[i], directions[j]) < 0.0:
        # Opposing traffic: neither is "left of" the other in the driving sense,
        # they are mirror images about the boundary they share.
        relations.opposing.add((min(i, j), max(i, j)))
        relations.opposing_of.setdefault(i, j)
        relations.opposing_of.setdefault(j, i)
        return

    # Same direction: decide which is on the right from geometry, not from which
    # bound happened to match.
    if _is_right_of(lanelets, directions, i, j):
        relations.right_of.setdefault(i, j)
        relations.left_of.setdefault(j, i)
    else:
        relations.right_of.setdefault(j, i)
        relations.left_of.setdefault(i, j)


def _is_right_of(lanelets: list[LaneletIR], directions: list[Vec2], i: int, j: int) -> bool:
    """Whether `j` lies to the right of `i`, along `i`'s direction of travel."""
    direction = directions[i]
    normal: Vec2 = (-direction[1], direction[0])  # left-hand normal

    def centre(index: int) -> Vec2:
        points = [*lanelets[index].left.points, *lanelets[index].right.points]
        return (
            sum(p.x for p in points) / len(points),
            sum(p.y for p in points) / len(points),
        )

    own = centre(i)
    other = centre(j)
    return dot2((other[0] - own[0], other[1] - own[1]), normal) < 0.0
