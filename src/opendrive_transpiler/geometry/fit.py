"""Turning a polyline into OpenDRIVE `<planView>` geometry records.

The shipped strategy is one `<line>` per polyline segment, placed with an
absolute start pose. The trade-off is worth stating plainly: every vertex lands
on its input coordinate, so positional error is exactly zero, but the heading is
discontinuous at each vertex (C0, not C1). That is the honest representation of
polyline input, and it is what most polyline-sourced OpenDRIVE looks like.
Curvature-continuous output is a *fitting* problem with unavoidable positional
error, so it belongs behind an opt-in flag rather than in the default path.

Before emitting, collinear runs are merged. This matters more than it sounds:
the lanelet2 centerline algorithm advances one bound at a time and so returns
2N-1 points for evenly spaced parallel bounds, most of them redundant.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..odr.model import GeometryRecord
from .polyline import dedupe, headings, segment_lengths
from .vec import (
    Vec2,
    Vec3,
    angle_difference,
    cross2,
    dot2,
    heading,
    left_normal,
    normalize2,
    point_segment_distance,
    sub2,
)


def merge_collinear(
    points: Sequence[Vec3],
    *,
    heading_tolerance: float = 1e-6,
    chord_tolerance: float = 1e-4,
) -> list[Vec3]:
    """Collapse runs of near-collinear vertices into single segments.

    A run may absorb the next vertex only if *every* intermediate vertex stays
    within `chord_tolerance` of the run's chord and every segment heading stays
    within `heading_tolerance` of the chord heading. Checking the whole run each
    time (rather than adjacent pairs) is what stops a long shallow curve from
    being swallowed one imperceptible step at a time.
    """
    pts = dedupe(points)
    if len(pts) <= 2:
        return pts

    out: list[Vec3] = [pts[0]]
    anchor = 0
    end = 1

    while end < len(pts) - 1:
        candidate = end + 1
        if _run_is_straight(pts, anchor, candidate, heading_tolerance, chord_tolerance):
            end = candidate
            continue
        out.append(pts[end])
        anchor = end
        end = end + 1

    out.append(pts[-1])
    return out


def _run_is_straight(
    pts: Sequence[Vec3],
    start: int,
    stop: int,
    heading_tolerance: float,
    chord_tolerance: float,
) -> bool:
    a: Vec2 = (pts[start][0], pts[start][1])
    b: Vec2 = (pts[stop][0], pts[stop][1])
    chord_heading = heading(a, b)

    for i in range(start, stop):
        p: Vec2 = (pts[i][0], pts[i][1])
        q: Vec2 = (pts[i + 1][0], pts[i + 1][1])
        if abs(angle_difference(heading(p, q), chord_heading)) > heading_tolerance:
            return False

    for i in range(start + 1, stop):
        p = (pts[i][0], pts[i][1])
        distance, _closest, _t = point_segment_distance(p, a, b)
        if distance > chord_tolerance:
            return False
    return True


def line_geometries(points: Sequence[Vec3]) -> list[GeometryRecord]:
    """One `<line>` per segment, each with its own absolute start pose."""
    pts = dedupe(points)
    if len(pts) < 2:
        return []

    records: list[GeometryRecord] = []
    s = 0.0
    for length, hdg, start in zip(segment_lengths(pts), headings(pts), pts[:-1], strict=False):
        if length <= 0.0:
            continue
        records.append(
            GeometryRecord(s=s, x=start[0], y=start[1], hdg=hdg, length=length, kind="line")
        )
        s += length
    return records


def build_plan_view(
    points: Sequence[Vec3],
    *,
    heading_tolerance: float = 1e-6,
    chord_tolerance: float = 1e-4,
) -> tuple[list[GeometryRecord], list[Vec3]]:
    """Merge, then emit. Returns the records and the simplified polyline.

    The simplified polyline is returned too because the width and elevation
    profiles must be sampled against the *same* reference line the planView
    describes, or the two disagree about where `s` is.
    """
    simplified = merge_collinear(
        points, heading_tolerance=heading_tolerance, chord_tolerance=chord_tolerance
    )
    return line_geometries(simplified), simplified


def signed_side(reference: Sequence[Vec3], other: Sequence[Vec3]) -> float:
    """Which side of `reference` the polyline `other` lies on.

    Positive means left (OpenDRIVE's +t and positive lane ids), negative right.
    Used to detect bounds that are geometrically swapped relative to their names,
    which is common enough in hand-written maps to be worth reporting rather than
    silently mirroring the road.
    """
    if len(reference) < 2 or not other:
        return 0.0

    total = 0.0
    for i in range(len(reference) - 1):
        a: Vec2 = (reference[i][0], reference[i][1])
        b: Vec2 = (reference[i + 1][0], reference[i + 1][1])
        direction = normalize2(sub2(b, a))
        if direction == (0.0, 0.0):
            continue
        midpoint: Vec2 = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        nearest = min(other, key=lambda p: (p[0] - midpoint[0]) ** 2 + (p[1] - midpoint[1]) ** 2)
        total += cross2(direction, sub2((nearest[0], nearest[1]), midpoint))
    return total


def lateral_offset(point: Vec3, hdg: float, target: Vec3) -> float:
    """Signed offset of `target` from `point` along the left-normal of `hdg`."""
    normal = left_normal(hdg)
    return dot2(sub2((target[0], target[1]), (point[0], point[1])), normal)
