"""Centerline computation, ported from lanelet2.

A pure-Python port of `crates/ll2-core/src/centerline.rs` in simple_lanelet2,
which is itself a port of `lanelet2_core/src/Lanelet.cpp`. It is *not* naive
midpoint pairing: the two bounds usually have different point counts and
different spacing, so the algorithm advances one side at a time, each step
picking the nearest point on that side whose resulting centerline segment crosses
neither bound.

Porting it rather than approximating it matters because the output is observable:
a script may read `lanelet.centerline`, and `--reference-line=centerline` puts it
straight into the OpenDRIVE planView.

One consequence worth knowing when reading the output: because the algorithm
advances one side at a time, evenly spaced parallel bounds of N points yield
2N-1 centerline points, not N. The geometry stage's collinearity merge exists
partly to undo that.

Candidate ordering follows the Rust port's `(distance, index)` rule rather than
upstream's unspecified r-tree tie-breaking, so results are deterministic on the
rectilinear maps where ties are common.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
Segment = tuple[Point2, Point2]

LEFT = 0
RIGHT = 1


def _sub(a: Point2, b: Point2) -> Point2:
    return (a[0] - b[0], a[1] - b[1])


def _cross(a: Point2, b: Point2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_is_left_of(p1: Point2, p2: Point2, q: Point2) -> bool:
    return _cross(_sub(p2, p1), _sub(q, p1)) > 0.0


def _orientation(a: Point2, b: Point2, c: Point2) -> int:
    value = _cross(_sub(b, a), _sub(c, a))
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _on_segment(a: Point2, b: Point2, q: Point2) -> bool:
    return min(a[0], b[0]) <= q[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= q[1] <= max(a[1], b[1])


def _segments_intersect(s: Segment, t: Segment) -> bool:
    """Boost's `intersects(Segment, Segment)`: touching endpoints count."""
    p1, p2 = s
    q1, q2 = t
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment(p1, p2, q1))
        or (o2 == 0 and _on_segment(p1, p2, q2))
        or (o3 == 0 and _on_segment(q1, q2, p1))
        or (o4 == 0 and _on_segment(q1, q2, p2))
    )


def _same_point(a: Point2, b: Point2) -> bool:
    return a[0] == b[0] and a[1] == b[1]


def _segments_of(points: Sequence[Point2]) -> list[Segment]:
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


class _BoundChecker:
    """Holds both bounds and answers the geometric questions the search asks."""

    def __init__(self, left: list[Point2], right: list[Point2]) -> None:
        self.left = left
        self.right = right
        self.left_segments = _segments_of(left)
        self.right_segments = _segments_of(right)
        self.entry: Segment = (right[0], left[0])
        self.exit: Segment = (left[-1], right[-1])

    def points(self, side: int) -> list[Point2]:
        return self.left if side == LEFT else self.right

    def segments(self, side: int) -> list[Segment]:
        return self.left_segments if side == LEFT else self.right_segments

    def intersects(self, seg: Segment) -> bool:
        return (
            self._intersects_bound(seg, LEFT)
            or self._intersects_bound(seg, RIGHT)
            or self._crosses_entry(seg)
            or self._crosses_exit(seg)
        )

    def _intersects_bound(self, seg: Segment, side: int) -> bool:
        # A bound segment starting where `seg` starts does not block it -- that is
        # what lets the candidate segment leave the bound it is anchored to.
        return any(
            _segments_intersect(bound, seg) and not _same_point(bound[0], seg[0])
            for bound in self.segments(side)
        )

    def second_crosses_bounds(self, seg: Segment, side: int) -> bool:
        """Whether the far endpoint of `seg` is blocked by a bound that does not
        simply terminate there."""
        return any(
            _segments_intersect(bound, seg)
            and not _same_point(bound[0], seg[1])
            and not _same_point(bound[1], seg[1])
            for bound in self.segments(side)
        )

    def _crosses_entry(self, seg: Segment) -> bool:
        return _segments_intersect(seg, self.entry) and _point_is_left_of(
            self.entry[0], self.entry[1], seg[1]
        )

    def _crosses_exit(self, seg: Segment) -> bool:
        return _segments_intersect(seg, self.exit) and _point_is_left_of(
            self.exit[0], self.exit[1], seg[1]
        )

    def by_distance_from(self, side: int, target: Point2) -> list[int]:
        points = self.points(side)
        return sorted(range(len(points)), key=lambda i: (_distance(points[i], target), i))


def _closest_nonintersecting_point(
    bounds: _BoundChecker,
    side: int,
    from_index: int,
    other: Point2,
    last_centerline_point: Point2,
) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    d_last_other = _distance(other, last_centerline_point)
    points = bounds.points(side)
    other_side = RIGHT if side == LEFT else LEFT

    for index in bounds.by_distance_from(side, other):
        if index < from_index:
            continue

        candidate = points[index]
        candidate_distance = _distance(candidate, other) / 2.0

        if best is not None:
            # The triangle inequality bounds how much closer a later candidate
            # could be; past that bound nothing better remains.
            if candidate_distance - d_last_other > best[1]:
                break
            if best[1] <= candidate_distance:
                continue

        bound_connection: Segment = (other, candidate)
        inverse_connection: Segment = (candidate, other)
        midpoint = ((other[0] + candidate[0]) / 2.0, (other[1] + candidate[1]) / 2.0)
        centerline_candidate: Segment = (last_centerline_point, midpoint)

        if (
            not bounds.intersects(centerline_candidate)
            and not bounds.second_crosses_bounds(bound_connection, side)
            and not bounds.second_crosses_bounds(inverse_connection, other_side)
        ):
            best = (index, candidate_distance)

    return best


def _midpoint3(a: Point3, b: Point3) -> Point3:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def centerline_coords(
    left: Sequence[Point3],
    right: Sequence[Point3],
    *,
    shares_first_point: bool = False,
) -> list[Point3]:
    """The centerline of two bounds, as (x, y, z) triples.

    `shares_first_point` reports whether the two bounds literally begin with the
    same point *object* -- a lanelet closed at its start. lanelet2 tests identity
    here, not coordinate equality, so the caller must answer it.
    """
    if not left or not right:
        return []

    left3 = [tuple(p) for p in left]
    right3 = [tuple(p) for p in right]
    bounds = _BoundChecker([(p[0], p[1]) for p in left3], [(p[0], p[1]) for p in right3])

    centerline: list[Point3] = [_midpoint3(left3[0], right3[0])]
    left_current = 1 if shares_first_point else 0
    right_current = 0

    while True:
        last = centerline[-1]
        last2d: Point2 = (last[0], last[1])

        left_candidate = (
            _closest_nonintersecting_point(
                bounds, LEFT, left_current + 1, bounds.right[right_current], last2d
            )
            if left_current < len(left3)
            else None
        )
        right_candidate = (
            _closest_nonintersecting_point(
                bounds, RIGHT, right_current + 1, bounds.left[left_current], last2d
            )
            if right_current < len(right3)
            else None
        )

        # Ties go to the left, matching upstream's `<=`.
        if left_candidate is not None and (
            right_candidate is None or left_candidate[1] <= right_candidate[1]
        ):
            centerline.append(_midpoint3(left3[left_candidate[0]], right3[right_current]))
            left_current = left_candidate[0]
        elif right_candidate is not None and (
            left_candidate is None or left_candidate[1] > right_candidate[1]
        ):
            centerline.append(_midpoint3(left3[left_current], right3[right_candidate[0]]))
            right_current = right_candidate[0]
        else:
            break

    # The midpoint of the two end points belongs in the result in any case.
    if not (left_current == len(left3) - 1 and right_current == len(right3) - 1):
        centerline.append(_midpoint3(left3[-1], right3[-1]))

    return centerline


def compute_centerline(left: Any, right: Any) -> Any:
    """Shadow-object wrapper: `ShadowLineString` in, `ShadowLineString` out.

    Imported lazily so that `ir` stays a leaf of the dependency graph and this
    module remains usable as plain geometry.
    """
    from ..frontend.shadow import (
        INVAL_ID,
        LineStringStorage,
        PointStorage,
        ShadowLineString,
        ShadowPoint,
    )

    if left is None or right is None:
        return ShadowLineString(LineStringStorage())

    left_points = left.points
    right_points = right.points
    shares_first = bool(
        left_points and right_points and left_points[0].storage is right_points[0].storage
    )
    coords = centerline_coords(
        [p.xyz for p in left_points],
        [p.xyz for p in right_points],
        shares_first_point=shares_first,
    )
    # A computed centerline carries no id -- that is how lanelet2 distinguishes
    # it from one the user assigned.
    points = [ShadowPoint(PointStorage(x=x, y=y, z=z, id=INVAL_ID)) for x, y, z in coords]
    return ShadowLineString(LineStringStorage(points=points, id=INVAL_ID))
