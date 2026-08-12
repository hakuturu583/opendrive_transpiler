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

import math
from collections.abc import Sequence
from itertools import pairwise

from ..odr.model import GeometryRecord
from .polyline import dedupe, headings, segment_lengths
from .vec import (
    Vec2,
    Vec3,
    angle_difference,
    cross2,
    distance2,
    dot2,
    heading,
    left_normal,
    normalize2,
    normalize_angle,
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


# --------------------------------------------------------------------------
# Arc fitting
# --------------------------------------------------------------------------


def _circumcircle(a: Vec2, b: Vec2, c: Vec2) -> tuple[float, float, float] | None:
    """Centre and radius of the circle through three points, or None if collinear."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-15:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


ARC_SAGITTA = 0.1
"""Metres an `<arc>` may bulge off the polyline *between* two input vertices.

Arc fitting only makes sense under the reading that the vertices *sample* a
curve, so recovering that curve is the point and the chord between two samples
is the discretisation error rather than the truth. Held to `chord_tolerance`
instead -- sub-millimetre, the scale at which a vertex may be moved -- an arc
could essentially never fire: a quarter circle of radius 38 m sampled every 7.5
degrees, which is the `curved_road` fixture, already leaves its own chords by 81
mm. One arc survived on the whole Lanelet2 Karlsruhe example.

But the reading needs a bound, or it licenses anything. Unbounded, `_fit_arc`
recovered a circle of radius 21.8 m swept 171 degrees from three points -- 65 m
of arc across a 43 m gap, bulging 19 m off the road -- because three points lie
exactly on their own circumcircle whatever happens between them, so the
vertex-by-vertex check could not see it. 85 of 251 records strayed over 100 mm.

100 mm is below the survey noise of the maps this reads and narrower than a lane
marking, and it admits 46 arcs on that map while holding the worst stray to 125
mm. A caller who raises `--chord-tolerance` above it has asked for more slack and
gets it.
"""


def _fit_arc(
    pts: Sequence[Vec3], start: int, stop: int, chord_tolerance: float, sagitta: float
) -> tuple[float, float, float] | None:
    """Fit a circular arc through `pts[start..stop]`.

    Returns `(curvature, length, start_heading)`, or None when the run is not
    arc-like. Curvature is signed: positive turns left, matching OpenDRIVE.
    """
    a: Vec2 = (pts[start][0], pts[start][1])
    b: Vec2 = (pts[stop][0], pts[stop][1])
    middle = pts[(start + stop) // 2]
    circle = _circumcircle(a, (middle[0], middle[1]), b)
    if circle is None:
        return None
    cx, cy, radius = circle
    if radius <= 0.0 or not math.isfinite(radius):
        return None

    # Direction of travel around the centre.
    turn = cross2(sub2((middle[0], middle[1]), a), sub2(b, (middle[0], middle[1])))
    if turn == 0.0:
        return None
    sign = 1.0 if turn > 0.0 else -1.0

    def angle_at(point: Vec3 | Vec2) -> float:
        return math.atan2(point[1] - cy, point[0] - cx)

    theta_a = angle_at(a)

    def swept(point: Vec3 | Vec2) -> float:
        """How far round the circle `point` is from the start, in travel order."""
        delta = (angle_at(point) - theta_a) * sign
        return delta % (2.0 * math.pi)

    # Every vertex must lie on the circle and progress monotonically along it,
    # or this run is not a single arc. Vertices alone are not enough, though:
    # three points lie exactly on their own circumcircle whatever the arc does
    # between them, so the run has to be checked *between* the samples too --
    # `sagitta` is how far it may bulge there. See ARC_SAGITTA for why that is a
    # separate, looser bound than the one the vertices themselves are held to.
    previous = 0.0
    for index in range(start + 1, stop + 1):
        point = pts[index]
        if abs(math.hypot(point[0] - cx, point[1] - cy) - radius) > chord_tolerance:
            return None
        advance = swept(point)
        if advance <= previous:
            return None
        if radius * (1.0 - math.cos((advance - previous) / 2.0)) > sagitta:
            return None
        previous = advance

    total = previous
    if total <= 0.0 or total >= 2.0 * math.pi:
        return None

    curvature = sign / radius
    length = radius * total
    start_heading = normalize_angle(theta_a + sign * math.pi / 2.0)
    return curvature, length, start_heading


def arc_geometries(
    points: Sequence[Vec3],
    *,
    chord_tolerance: float = 1e-4,
    min_curvature: float = 1e-8,
    sagitta: float | None = None,
) -> list[GeometryRecord]:
    """Greedy arc fitting: longest arc-or-line run at each step.

    Unlike `line_geometries`, this trades exactness for curvature continuity
    within each run -- vertices may sit up to `chord_tolerance` off the emitted
    curve, and the curve *between* two vertices up to `sagitta` off the segment
    joining them (`ARC_SAGITTA`, or `chord_tolerance` where that is looser).
    Straight runs still come out as `<line>`, and a run that is not arc-like
    falls back to one line per segment -- so the *fitting* never does worse than
    the default, though the emitted curve is of course further from the input
    polyline than the default's zero.
    """
    limit = max(ARC_SAGITTA, chord_tolerance) if sagitta is None else sagitta
    pts = dedupe(points)
    if len(pts) < 2:
        return []

    records: list[GeometryRecord] = []
    s = 0.0
    index = 0

    while index < len(pts) - 1:
        best_stop = index + 1
        best: tuple[str, float, float, float] | None = None

        for stop in range(index + 2, len(pts)):
            if _run_is_straight(pts, index, stop, math.inf, chord_tolerance):
                best_stop, best = stop, ("line", 0.0, 0.0, 0.0)
                continue
            fitted = _fit_arc(pts, index, stop, chord_tolerance, limit)
            if fitted is None:
                break
            curvature, length, heading_ = fitted
            if abs(curvature) < min_curvature:
                best_stop, best = stop, ("line", 0.0, 0.0, 0.0)
            else:
                best_stop, best = stop, ("arc", curvature, length, heading_)

        start = pts[index]
        if best is None or best[0] == "line":
            end = pts[best_stop]
            length = distance2((start[0], start[1]), (end[0], end[1]))
            hdg = heading((start[0], start[1]), (end[0], end[1]))
            records.append(
                GeometryRecord(s=s, x=start[0], y=start[1], hdg=hdg, length=length, kind="line")
            )
        else:
            _kind, curvature, length, hdg = best
            records.append(
                GeometryRecord(
                    s=s,
                    x=start[0],
                    y=start[1],
                    hdg=hdg,
                    length=length,
                    kind="arc",
                    params={"curvature": curvature},
                )
            )
        s += length
        index = best_stop

    return records


# --------------------------------------------------------------------------
# Cubic (paramPoly3) fitting
# --------------------------------------------------------------------------


def vertex_headings(pts: Sequence[Vec3]) -> list[float]:
    """A heading per vertex, averaged across the two adjoining segments.

    Sharing one heading between the segment that ends at a vertex and the one
    that starts there is what makes the emitted curve C1-continuous.
    """
    segment = headings(pts)
    if not segment:
        return [0.0]
    out = [segment[0]]
    for before, after in pairwise(segment):
        # Average as unit vectors so the wrap at +/-pi behaves.
        x = math.cos(before) + math.cos(after)
        y = math.sin(before) + math.sin(after)
        out.append(math.atan2(y, x) if (x or y) else before)
    out.append(segment[-1])
    return out


def _hermite_length(
    au: float,
    bu: float,
    cu: float,
    du: float,
    av: float,
    bv: float,
    cv: float,
    dv: float,
    samples: int = 256,
) -> float:
    """Arc length of the cubic by Simpson's rule.

    Computed here as well as by the backend so the `s` we write and the length
    it derives agree; at this sample count they match to well under a micrometre.
    """
    del au, av

    def speed(p: float) -> float:
        return math.hypot(bu + 2 * cu * p + 3 * du * p * p, bv + 2 * cv * p + 3 * dv * p * p)

    n = samples if samples % 2 == 0 else samples + 1
    h = 1.0 / n
    total = speed(0.0) + speed(1.0)
    for i in range(1, n):
        total += speed(i * h) * (4.0 if i % 2 else 2.0)
    return total * h / 3.0


def param_poly3_geometries(points: Sequence[Vec3]) -> list[GeometryRecord]:
    """One cubic Hermite `<paramPoly3>` per segment, C1-continuous throughout.

    Endpoints are still exact -- only the path between them is a curve rather
    than a straight line, which is the point: the heading no longer jumps at
    vertices.
    """
    pts = dedupe(points)
    if len(pts) < 2:
        return []

    vertex = vertex_headings(pts)
    records: list[GeometryRecord] = []
    s = 0.0

    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        h0, h1 = vertex[i], vertex[i + 1]
        chord = distance2((a[0], a[1]), (b[0], b[1]))
        if chord <= 0.0:
            continue

        # Work in the local frame: origin at `a`, u-axis along h0.
        cos0, sin0 = math.cos(-h0), math.sin(-h0)
        dx, dy = b[0] - a[0], b[1] - a[1]
        bl_x = dx * cos0 - dy * sin0
        bl_y = dx * sin0 + dy * cos0

        relative = angle_difference(h0, h1)
        t0x, t0y = chord, 0.0
        t1x, t1y = chord * math.cos(relative), chord * math.sin(relative)

        au, bu = 0.0, t0x
        cu = 3.0 * bl_x - 2.0 * t0x - t1x
        du = -2.0 * bl_x + t0x + t1x
        av, bv = 0.0, t0y
        cv = 3.0 * bl_y - 2.0 * t0y - t1y
        dv = -2.0 * bl_y + t0y + t1y

        length = _hermite_length(au, bu, cu, du, av, bv, cv, dv)
        records.append(
            GeometryRecord(
                s=s,
                x=a[0],
                y=a[1],
                hdg=h0,
                length=length,
                kind="paramPoly3",
                params={
                    "au": au,
                    "bu": bu,
                    "cu": cu,
                    "du": du,
                    "av": av,
                    "bv": bv,
                    "cv": cv,
                    "dv": dv,
                },
            )
        )
        s += length

    return records


def build_plan_view(
    points: Sequence[Vec3],
    *,
    fit: str = "line",
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
    if fit == "arc":
        return arc_geometries(simplified, chord_tolerance=chord_tolerance), simplified
    if fit == "parampoly3":
        return param_poly3_geometries(simplified), simplified
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
