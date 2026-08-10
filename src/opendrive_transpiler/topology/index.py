"""Node identity: deciding when two points are the same physical node.

lanelet2 answers this two different ways at once, and both are real:

* **Storage identity.** The aliasing constructors and shared endpoints mean two
  handles genuinely refer to one point. This is exact.
* **Coordinate coincidence.** Hand-written maps and maps round-tripped through
  OSM repeat the same coordinate in distinct objects. This needs a tolerance.

Merging both into one union-find gives a single canonical node id per physical
node, which is what successor and adjacency inference are actually asking about.

Cell lookup checks the 26 neighbouring cells as well as its own, so two points a
hair apart but on opposite sides of a cell boundary still merge -- the classic
failure of naive coordinate quantisation.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..ir.model import BoundIR, LaneletIR, PointIR


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def add(self, item: int) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: int) -> int:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


class NodeIndex:
    """Maps every point in the map to a canonical integer node id."""

    def __init__(self, lanelets: Iterable[LaneletIR], tolerance: float = 1e-3) -> None:
        self.tolerance = tolerance
        self._union = UnionFind()
        self._coords: dict[int, tuple[float, float, float]] = {}
        self._canonical: dict[int, int] = {}

        points: list[PointIR] = []
        for lanelet in lanelets:
            for bound in (lanelet.left, lanelet.right):
                points.extend(bound.points)

        # Storage identity first: same key, same node, no tolerance involved.
        for point in points:
            self._union.add(point.key)
            self._coords.setdefault(point.key, point.xyz)

        self._merge_coincident(points)
        self._assign_ids()

    def _merge_coincident(self, points: list[PointIR]) -> None:
        tol = self.tolerance
        cells: dict[tuple[int, int, int], list[int]] = {}

        def cell_of(p: tuple[float, float, float]) -> tuple[int, int, int]:
            return (int(p[0] // tol), int(p[1] // tol), int(p[2] // tol))

        for point in points:
            cells.setdefault(cell_of(point.xyz), []).append(point.key)

        tol_sq = tol * tol
        for point in points:
            cx, cy, cz = cell_of(point.xyz)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for other in cells.get((cx + dx, cy + dy, cz + dz), ()):
                            if other == point.key:
                                continue
                            ox, oy, oz = self._coords[other]
                            if (
                                (ox - point.x) ** 2 + (oy - point.y) ** 2 + (oz - point.z) ** 2
                            ) <= tol_sq:
                                self._union.union(point.key, other)

    def _assign_ids(self) -> None:
        next_id = 0
        roots: dict[int, int] = {}
        for key in sorted(self._coords):
            root = self._union.find(key)
            if root not in roots:
                roots[root] = next_id
                next_id += 1
            self._canonical[key] = roots[root]

    # -- queries -----------------------------------------------------------
    def node(self, point: PointIR) -> int:
        return self._canonical.get(point.key, -1)

    def nodes(self, bound: BoundIR) -> tuple[int, ...]:
        return tuple(self.node(p) for p in bound.points)

    def signature(self, bound: BoundIR) -> tuple[int, ...]:
        """A boundary's identity as a node sequence.

        Two boundaries with the same signature are the same physical line,
        whether they are literally the same object or merely coincide.
        """
        return self.nodes(bound)

    def start(self, bound: BoundIR) -> int:
        return self.node(bound.points[0])

    def end(self, bound: BoundIR) -> int:
        return self.node(bound.points[-1])

    @property
    def node_count(self) -> int:
        return len(set(self._canonical.values()))
