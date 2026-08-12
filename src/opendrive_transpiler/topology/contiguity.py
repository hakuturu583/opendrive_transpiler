"""Can the reference lines be made to meet, and at what cost?

A road's reference line is the leftmost boundary of its cross-section. Where the
cross-section changes across a join, the leftmost boundary is a different
physical line on either side, so the two roads are linked but their reference
lines are up to 12.5 m apart -- `W512`, and 34 of the 127 joins on the Lanelet2
Karlsruhe example.

What can continue across a join is not a *boundary*. A succession in lanelet2 is
a shared pair of **corner nodes**: the two lanelets carry different line strings
that meet at a corner. So the question is whether some boundary of one
cross-section ends where some boundary of the other begins -- and on that map the
answer is yes at 90 of 91 joins, usually with two or three corners to choose
between.

Choosing is the hard part, and it is a graph problem rather than a local one.
Picking the corner that satisfies one join fixes the boundary a road uses at
*both* its ends, which constrains its next neighbour in turn, and so on down the
chain. So this assigns an index to every cross-section of a component at once.

The components a real map produces are large -- 84, 66 and 58 cross-sections on
the Karlsruhe example -- but sparse, near enough to trees that one pass of
dynamic programming solves them. 25 of that map's 223 handovers do not meet
today; the assignment closes 17.

Nothing here changes what is emitted. It computes the assignment and reports what
it could and could not satisfy, which is the number that says whether following
it through is worth the cost -- and the cost is real: at a join that does not
already meet, at least one side has to give up its leftmost boundary, because if
both kept it and their corners agreed the join would meet already. A reference
line inside the cross-section puts the lanes left of it on positive ids, which
under right-hand traffic reads as travelling against `s`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..geometry.vec import Vec3

Corner = tuple[int, int]
"""A node position, quantised, so two corners that coincide compare equal."""


@dataclass(frozen=True)
class Edge:
    """Two cross-sections that have to hand the reference line over to each other."""

    left: int
    """Index into `Assignment.stacks` of the side the reference line arrives from."""
    right: int
    within_road: bool
    """True for consecutive groups of one road, false for a road-to-road join.

    A step inside a road does not show up as a gap -- the pieces are concatenated
    into one plan view, so it comes out as a segment jogging sideways across the
    carriageway instead. Worth separating in the report, not in the solving.
    """


@dataclass
class Assignment:
    """Which boundary of each cross-section carries the reference line."""

    chosen: list[int] = field(default_factory=list)
    """Index into that cross-section's left-to-right boundary stack."""
    stacks: list[int] = field(default_factory=list)
    """How many boundaries each cross-section has."""
    edges: list[Edge] = field(default_factory=list)
    satisfied: list[bool] = field(default_factory=list)
    """Per edge, whether the two chosen boundaries share a corner."""

    @property
    def met(self) -> int:
        return sum(self.satisfied)

    @property
    def interior(self) -> list[int]:
        """Cross-sections whose reference is not their leftmost boundary.

        These are what the assignment costs: every lane left of the chosen
        boundary takes a positive id.
        """
        return [i for i, index in enumerate(self.chosen) if index != 0]


def quantise(point: Vec3, tolerance: float) -> Corner:
    """A corner as a lattice cell, so coincident ends hash alike.

    Rounding rather than clustering: `NodeIndex` has already merged everything
    within tolerance, so anything still distinct here is distinct by more than a
    cell and cannot collide.
    """
    return (round(point[0] / tolerance), round(point[1] / tolerance))


def corners_of(stack: list[list[Vec3]], which: str, tolerance: float) -> list[Corner | None]:
    """Where each boundary of a cross-section begins, or ends."""
    out: list[Corner | None] = []
    for boundary in stack:
        if len(boundary) < 2:
            out.append(None)
            continue
        out.append(quantise(boundary[-1] if which == "end" else boundary[0], tolerance))
    return out


def solve(
    ends: list[list[Corner | None]],
    starts: list[list[Corner | None]],
    edges: list[Edge],
    *,
    stacks: list[int] | None = None,
) -> Assignment:
    """Assign each cross-section a boundary index, satisfying as many edges as possible.

    Per connected component, by dynamic programming over a spanning tree -- see
    `_best_for` for why the obvious exhaustive walk is not an option. Ties go to
    the leftmost boundary, so a component with nothing to fix comes out exactly as
    it does today.
    """
    count = len(ends)
    sizes = stacks if stacks is not None else [len(e) for e in ends]

    incident: dict[int, list[Edge]] = defaultdict(list)
    for edge in edges:
        incident[edge.left].append(edge)
        incident[edge.right].append(edge)

    chosen = [0] * count
    components = _components(count, edges, incident)

    for component in components:
        members = sorted(component)
        inside = [e for e in edges if e.left in component]
        best = _best_for(members, inside, ends, starts, sizes)
        for node, index in zip(members, best, strict=True):
            chosen[node] = index

    satisfied = [
        _agree(ends[edge.left][chosen[edge.left]], starts[edge.right][chosen[edge.right]])
        for edge in edges
    ]
    return Assignment(chosen=chosen, stacks=list(sizes), edges=list(edges), satisfied=satisfied)


UNSATISFIED = 1_000_000
"""Weight of one unmet handover, against 1 for one cross-section moving.

Lexicographic in one integer: satisfy as many handovers as possible, and among
the assignments that do, move as few reference lines off the leftmost boundary as
possible. No component comes close to a million cross-sections, so the two never
trade against each other.
"""


def _best_for(
    members: list[int],
    inside: list[Edge],
    ends: list[list[Corner | None]],
    starts: list[list[Corner | None]],
    sizes: list[int],
) -> list[int]:
    """Dynamic programming over a spanning tree of the component.

    An exhaustive walk was the obvious thing and was useless: the components a
    real map produces are 60 to 85 cross-sections with two to five boundaries
    each, so the search space runs to 1e31 and any budget explores none of it.
    What saved it is the shape of the graph -- 84 cross-sections joined by 84
    handovers is a tree plus one cycle, and a tree is solvable exactly in one pass.

    So the tree part is solved exactly, cheapest-first over
    `UNSATISFIED * unmet + moved`, and the handful of edges outside the spanning
    tree are scored but not optimised over. That makes the result optimal on a
    forest and a good lower bound anywhere else, which is the honest thing to
    report against.
    """
    neighbours: dict[int, list[tuple[int, Edge]]] = {node: [] for node in members}
    for edge in inside:
        neighbours[edge.left].append((edge.right, edge))
        neighbours[edge.right].append((edge.left, edge))

    # A BFS spanning tree; whatever it leaves out is scored afterwards.
    root = members[0]
    parent: dict[int, tuple[int, Edge] | None] = {root: None}
    order = [root]
    queue = [root]
    while queue:
        node = queue.pop(0)
        for other, edge in neighbours[node]:
            if other in parent:
                continue
            parent[other] = (node, edge)
            order.append(other)
            queue.append(other)

    def penalty(edge: Edge, at_left: int, at_right: int) -> int:
        return 0 if _agree(ends[edge.left][at_left], starts[edge.right][at_right]) else UNSATISFIED

    # Leaves upward: the best a subtree can do for each index its root might take.
    cost: dict[int, list[int]] = {}
    pick: dict[int, list[dict[int, int]]] = {}
    for node in reversed(order):
        options = max(1, sizes[node])
        # One point for moving off the leftmost boundary, so ties go to today's layout.
        cost[node] = [1 if index else 0 for index in range(options)]
        pick[node] = [{} for _ in range(options)]
        for other, edge in neighbours[node]:
            if parent.get(other) is None or parent[other][0] != node:
                continue
            for index in range(options):
                best, chosen_child = None, 0
                for child_index in range(max(1, sizes[other])):
                    at_left, at_right = (
                        (index, child_index) if edge.left == node else (child_index, index)
                    )
                    total = cost[other][child_index] + penalty(edge, at_left, at_right)
                    if best is None or total < best:
                        best, chosen_child = total, child_index
                cost[node][index] += best or 0
                pick[node][index][other] = chosen_child

    result = {root: min(range(len(cost[root])), key=lambda i: cost[root][i])}
    for node in order:
        for other, _edge in neighbours[node]:
            if parent.get(other) is not None and parent[other][0] == node:
                result[other] = pick[node][result[node]][other]

    return [result[node] for node in members]


def _agree(end: Corner | None, start: Corner | None) -> bool:
    return end is not None and end == start


def _components(count: int, edges: list[Edge], incident: dict[int, list[Edge]]) -> list[set[int]]:
    seen: set[int] = set()
    out: list[set[int]] = []
    for node in range(count):
        if node in seen or not incident[node]:
            continue
        stack, component = [node], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            for edge in incident[current]:
                other = edge.right if edge.left == current else edge.left
                if other not in component:
                    stack.append(other)
        seen |= component
        out.append(component)
    return out
