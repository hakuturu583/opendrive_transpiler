"""Polyline operations on lists of (x, y, z) triples.

Stations (`s`) are measured in the xy-plane, because that is what OpenDRIVE's
`s` axis means: elevation is a separate profile over the same `s`, not extra
arclength.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .vec import Vec2, Vec3, distance2, heading, point_segment_distance


def dedupe(points: Sequence[Vec3], tolerance: float = 1e-9) -> list[Vec3]:
    """Drop consecutive points that coincide in the xy-plane.

    Zero-length segments have no heading, so they must go before anything tries
    to compute one.
    """
    out: list[Vec3] = []
    for point in points:
        if not out or distance2((out[-1][0], out[-1][1]), (point[0], point[1])) > tolerance:
            out.append(tuple(point))  # type: ignore[arg-type]
    return out


def segment_lengths(points: Sequence[Vec3]) -> list[float]:
    return [
        distance2((points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]))
        for i in range(len(points) - 1)
    ]


def stations(points: Sequence[Vec3]) -> list[float]:
    """Cumulative arclength at each vertex, starting at 0."""
    result = [0.0]
    for length in segment_lengths(points):
        result.append(result[-1] + length)
    return result


def total_length(points: Sequence[Vec3]) -> float:
    # fsum, not sum: CPython 3.12 changed sum() on floats to compensated
    # summation, so a plain sum makes the emitted geometry depend on the
    # interpreter version. fsum is exactly rounded, and the same everywhere.
    return math.fsum(segment_lengths(points))


def headings(points: Sequence[Vec3]) -> list[float]:
    """Heading of each segment (one shorter than the point list)."""
    return [
        heading((points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]))
        for i in range(len(points) - 1)
    ]


def point_at_station(points: Sequence[Vec3], s: float) -> tuple[Vec3, float]:
    """Interpolate the polyline at arclength `s`, returning the point and heading."""
    sts = stations(points)
    if s <= 0.0:
        return (tuple(points[0]), headings(points)[0] if len(points) > 1 else 0.0)  # type: ignore[return-value]
    if s >= sts[-1]:
        return (tuple(points[-1]), headings(points)[-1] if len(points) > 1 else 0.0)  # type: ignore[return-value]

    for i in range(len(sts) - 1):
        if sts[i] <= s <= sts[i + 1]:
            span = sts[i + 1] - sts[i]
            t = 0.0 if span == 0.0 else (s - sts[i]) / span
            a, b = points[i], points[i + 1]
            interpolated = (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            )
            return (interpolated, heading((a[0], a[1]), (b[0], b[1])))
    return (tuple(points[-1]), 0.0)  # type: ignore[return-value]


def closest_point(points: Sequence[Vec3], query: Vec2) -> tuple[Vec3, float]:
    """Nearest point on the polyline to `query`, with its distance.

    Returns a full 3D point: z is interpolated along the winning segment, which
    is what the elevation profile needs.
    """
    if not points:
        return ((query[0], query[1], 0.0), math.inf)
    if len(points) == 1:
        p = points[0]
        return (tuple(p), distance2(query, (p[0], p[1])))  # type: ignore[return-value]

    best_distance = math.inf
    best_point: Vec3 = tuple(points[0])  # type: ignore[assignment]
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        distance, _closest, t = point_segment_distance(query, (a[0], a[1]), (b[0], b[1]))
        if distance < best_distance:
            best_distance = distance
            best_point = (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            )
    return (best_point, best_distance)


def station_of_point(points: Sequence[Vec3], query: Vec2) -> float:
    """Arclength of the point on the polyline nearest to `query`.

    Needed because merging collinear runs deletes vertices, so a group boundary
    that used to be a vertex has to be located by projection rather than by index.
    """
    if len(points) < 2:
        return 0.0

    sts = stations(points)
    best_distance = math.inf
    best_station = 0.0
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        distance, _closest, t = point_segment_distance(query, (a[0], a[1]), (b[0], b[1]))
        if distance < best_distance:
            best_distance = distance
            best_station = sts[i] + (sts[i + 1] - sts[i]) * t
    return best_station


def sample_stations(points: Sequence[Vec3], max_step: float) -> list[float]:
    """Every vertex station, subdivided so no gap exceeds `max_step`.

    Vertices are always included: they are where the geometry actually bends, so
    a sample there is worth more than a sample anywhere else.
    """
    sts = stations(points)
    if len(sts) < 2:
        return list(sts)

    out: list[float] = [sts[0]]
    for i in range(len(sts) - 1):
        start, end = sts[i], sts[i + 1]
        span = end - start
        if span > max_step > 0.0:
            steps = math.ceil(span / max_step)
            for k in range(1, steps):
                out.append(start + span * k / steps)
        out.append(end)
    return out


def is_ccw(points: Sequence[Vec3]) -> bool:
    """Signed-area test, used to tell a left bound from a right one."""
    area = 0.0
    for i in range(len(points)):
        a = points[i]
        b = points[(i + 1) % len(points)]
        area += a[0] * b[1] - b[0] * a[1]
    return area > 0.0
