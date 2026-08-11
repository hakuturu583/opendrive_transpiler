"""Lane width and road elevation profiles.

Both are the same shape -- a value sampled along `s`, emitted as a run of cubics
-- so they share one builder. The cubics we emit are actually linear (`c = d = 0`):
piecewise-linear is exact at every sample, C0-continuous, and it is the honest
representation of piecewise-linear input. Fitting genuine cubics would smooth
data that was never smooth.

Widths are measured perpendicular to the reference line: at station `s`, take the
reference point `C` and heading `h`, build the left-normal `n = (-sin h, cos h)`,
and project each boundary's nearest point onto it. The width of the lane between
two boundaries is the absolute difference of their offsets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..odr.model import PolyRecord
from .fit import lateral_offset
from .polyline import closest_point, point_at_station, sample_stations, total_length
from .vec import Vec3


def offsets_along(
    reference: Sequence[Vec3], stations_: Sequence[float], boundary: Sequence[Vec3]
) -> list[float]:
    """Signed lateral offset of `boundary` from `reference` at each station."""
    result: list[float] = []
    for s in stations_:
        point, hdg = point_at_station(reference, s)
        nearest, _distance = closest_point(boundary, (point[0], point[1]))
        result.append(lateral_offset(point, hdg, nearest))
    return result


def elevations_along(reference: Sequence[Vec3], stations_: Sequence[float]) -> list[float]:
    return [point_at_station(reference, s)[0][2] for s in stations_]


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting, for the 4x4 normal equations."""
    n = len(rhs)
    augmented = [[*row, rhs[i]] for i, row in enumerate(matrix)]

    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, n):
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, n + 1):
                augmented[row][k] -= factor * augmented[column][k]

    solution = [0.0] * n
    for row in reversed(range(n)):
        total = augmented[row][n] - math.fsum(
            augmented[row][k] * solution[k] for k in range(row + 1, n)
        )
        solution[row] = total / augmented[row][row]
    return solution


def fit_cubic(
    stations_: Sequence[float], values: Sequence[float], *, origin: float = 0.0
) -> tuple[float, float, float, float] | None:
    """Least-squares cubic `a + b*ds + c*ds^2 + d*ds^3` over the samples.

    Pure Python: the normal equations are 4x4, so an explicit solve is cheaper
    and clearer than pulling in numpy for it.
    """
    if len(values) < 4:
        return None

    powers = [[(s - origin) ** k for k in range(4)] for s in stations_]
    matrix = [[math.fsum(p[i] * p[j] for p in powers) for j in range(4)] for i in range(4)]
    rhs = [math.fsum(p[i] * v for p, v in zip(powers, values, strict=False)) for i in range(4)]
    solved = _solve(matrix, rhs)
    if solved is None:
        return None
    return solved[0], solved[1], solved[2], solved[3]


def build_profile(
    stations_: Sequence[float],
    values: Sequence[float],
    *,
    tolerance: float = 1e-4,
    origin: float = 0.0,
    cubic: bool = False,
) -> list[PolyRecord]:
    """Emit `values(s)` as a run of records anchored at `stations_ - origin`.

    A profile that never varies by more than `tolerance` collapses to a single
    constant record -- which is what every constant-width lane produces, and what
    keeps the generated file readable.
    """
    if not values:
        return [PolyRecord(0.0, 0.0)]

    first = values[0]
    if all(abs(v - first) <= tolerance for v in values):
        return [PolyRecord(0.0, first)]

    if cubic:
        # One cubic across the whole run, but only if it genuinely fits every
        # sample. Falling back to piecewise linear keeps the profile exact
        # rather than smoothing data that was never smooth.
        fitted = fit_cubic(stations_, values, origin=origin)
        if fitted is not None:
            a, b, c, d = fitted
            residual = max(
                abs(a + b * ds + c * ds**2 + d * ds**3 - v)
                for ds, v in ((s - origin, v) for s, v in zip(stations_, values, strict=False))
            )
            if residual <= tolerance:
                return [PolyRecord(0.0, a, b, c, d)]

    records: list[PolyRecord] = []
    for i in range(len(values) - 1):
        start = stations_[i] - origin
        span = stations_[i + 1] - stations_[i]
        slope = 0.0 if span <= 0.0 else (values[i + 1] - values[i]) / span
        if records:
            # Extend the previous record when it already predicts this sample and
            # keeps the same slope: merging collinear runs keeps the list short.
            predicted = records[-1].a + records[-1].b * (start - records[-1].s)
            if abs(predicted - values[i]) <= tolerance and abs(records[-1].b - slope) <= tolerance:
                continue
        records.append(PolyRecord(s=max(start, 0.0), a=values[i], b=slope))

    if not records:
        records.append(PolyRecord(0.0, first))
    # OpenDRIVE requires the first record to start at the section origin.
    if records[0].s != 0.0:
        records[0] = PolyRecord(0.0, records[0].a, records[0].b, records[0].c, records[0].d)
    return records


def lane_widths(
    reference: Sequence[Vec3],
    inner: Sequence[Vec3],
    outer: Sequence[Vec3],
    *,
    max_step: float = 5.0,
    tolerance: float = 1e-4,
    stations_: Sequence[float] | None = None,
    cubic: bool = False,
) -> tuple[list[PolyRecord], float]:
    """Width records for the lane between `inner` and `outer`.

    Returns the records and the minimum sampled width, so the caller can report a
    lane whose bounds cross over (negative width is invalid OpenDRIVE).
    """
    sts = list(stations_) if stations_ is not None else sample_stations(reference, max_step)
    inner_offsets = offsets_along(reference, sts, inner)
    outer_offsets = offsets_along(reference, sts, outer)
    widths = [abs(o - i) for i, o in zip(inner_offsets, outer_offsets, strict=False)]
    minimum = min(widths) if widths else 0.0
    return build_profile(sts, widths, tolerance=tolerance, cubic=cubic), minimum


def road_elevation(
    reference: Sequence[Vec3],
    *,
    max_step: float = 5.0,
    tolerance: float = 1e-6,
    stations_: Sequence[float] | None = None,
    cubic: bool = False,
) -> list[PolyRecord]:
    sts = list(stations_) if stations_ is not None else sample_stations(reference, max_step)
    values = elevations_along(reference, sts)
    return build_profile(sts, values, tolerance=tolerance, cubic=cubic)


def road_superelevation(
    reference: Sequence[Vec3],
    left: Sequence[Vec3],
    right: Sequence[Vec3],
    *,
    max_step: float = 5.0,
    tolerance: float = 1e-6,
    stations_: Sequence[float] | None = None,
    cubic: bool = False,
) -> list[PolyRecord]:
    """Road roll angle, from the height difference across the cross-section.

    OpenDRIVE models superelevation as a roll about the s-axis, so a point at
    lateral offset `t` sits `t * sin(phi)` above the reference elevation.
    Inverting that across the outermost bounds gives

        phi = asin((z_left - z_right) / span)

    The two bounds are paired by *fraction of their own length*, not by nearest
    point. On a curve the outer bound is longer, and projecting one onto the
    other slips by a chord's worth of sagitta -- enough to invent a fraction of a
    degree of banking on a road that is perfectly flat. Fraction pairing is exact
    for concentric arcs and for parallel straights, which is what road bounds are.
    """
    sts = list(stations_) if stations_ is not None else sample_stations(reference, max_step)
    if len(left) < 2 or len(right) < 2 or not sts:
        return [PolyRecord(0.0, 0.0)]

    span_s = sts[-1] - sts[0]
    left_length = total_length(left)
    right_length = total_length(right)
    angles: list[float] = []

    for s in sts:
        fraction = 0.0 if span_s <= 0.0 else (s - sts[0]) / span_s
        left_point, _ = point_at_station(left, fraction * left_length)
        right_point, _ = point_at_station(right, fraction * right_length)
        span = math.dist((left_point[0], left_point[1]), (right_point[0], right_point[1]))
        if span < 1e-9:
            angles.append(0.0)
            continue
        ratio = (left_point[2] - right_point[2]) / span
        angles.append(math.asin(max(-1.0, min(1.0, ratio))))

    return build_profile(sts, angles, tolerance=tolerance, cubic=cubic)
