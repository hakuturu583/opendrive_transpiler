"""Small 2D/3D vector helpers.

Pure `math`, no numpy. The whole geometry stage is dot products, `hypot` and
`atan2`; pulling in numpy for that would cost the package its zero-dependency
core and buy nothing at these array sizes.
"""

from __future__ import annotations

import math

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


def sub2(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def add2(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def scale2(a: Vec2, k: float) -> Vec2:
    return (a[0] * k, a[1] * k)


def dot2(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross2(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm2(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def distance2(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize2(a: Vec2) -> Vec2:
    length = norm2(a)
    if length == 0.0:
        return (0.0, 0.0)
    return (a[0] / length, a[1] / length)


def heading(a: Vec2, b: Vec2) -> float:
    """Direction of a -> b, in radians, matching OpenDRIVE's `hdg`."""
    return math.atan2(b[1] - a[1], b[0] - a[0])


def left_normal(hdg: float) -> Vec2:
    """Unit vector 90 degrees counter-clockwise from `hdg`.

    OpenDRIVE's `t` axis points this way, so a positive `t` offset is left of the
    reference line -- the same side as positive lane ids.
    """
    return (-math.sin(hdg), math.cos(hdg))


def normalize_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def angle_difference(a: float, b: float) -> float:
    """Smallest signed rotation carrying `a` onto `b`."""
    return normalize_angle(b - a)


def xy(point: Vec3 | Vec2) -> Vec2:
    return (point[0], point[1])


def point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> tuple[float, Vec2, float]:
    """Distance from `p` to segment `a-b`.

    Returns `(distance, closest_point, t)` where `t` in [0, 1] locates the closest
    point along the segment -- the caller needs it to interpolate z.
    """
    ab = sub2(b, a)
    length_squared = dot2(ab, ab)
    if length_squared == 0.0:
        return (distance2(p, a), a, 0.0)
    t = dot2(sub2(p, a), ab) / length_squared
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    closest = add2(a, scale2(ab, t))
    return (distance2(p, closest), closest, t)
