"""Geometry: exactness of the planView, and the width/elevation profiles.

The headline property is the one the whole `<line>`-per-segment strategy is
chosen for: every emitted geometry record starts exactly on an input vertex, and
the lengths sum to the input's own arclength. If that ever stops holding, the
"positionally exact" claim in the README is no longer true.
"""

from __future__ import annotations

import math
from itertools import pairwise

from opendrive_transpiler.geometry.fit import (
    build_plan_view,
    lateral_offset,
    line_geometries,
    merge_collinear,
    signed_side,
)
from opendrive_transpiler.geometry.polyline import (
    closest_point,
    dedupe,
    point_at_station,
    sample_stations,
    station_of_point,
    stations,
    total_length,
)
from opendrive_transpiler.geometry.profile import build_profile, lane_widths, road_elevation
from opendrive_transpiler.geometry.vec import angle_difference, left_normal, normalize_angle


def zigzag(count: int = 12) -> list[tuple[float, float, float]]:
    """A polyline that bends at every vertex, so nothing may be merged away."""
    return [(i * 3.0, (i % 2) * 2.0, i * 0.25) for i in range(count)]


# --------------------------------------------------------------------------
# planView exactness
# --------------------------------------------------------------------------


def test_every_geometry_record_starts_on_an_input_vertex():
    points = zigzag()
    records = line_geometries(points)
    assert len(records) == len(points) - 1
    for record, vertex in zip(records, points[:-1], strict=True):
        assert record.x == vertex[0]
        assert record.y == vertex[1]


def test_geometry_lengths_sum_to_the_polyline_length():
    points = zigzag()
    records = line_geometries(points)
    assert math.isclose(sum(r.length for r in records), total_length(points), rel_tol=1e-12)


def test_geometry_stations_are_cumulative_and_start_at_zero():
    records = line_geometries(zigzag())
    assert records[0].s == 0.0
    for previous, following in pairwise(records):
        assert math.isclose(following.s, previous.s + previous.length, rel_tol=1e-12)


def test_each_record_heading_points_at_the_next_vertex():
    points = zigzag()
    for record, (a, b) in zip(line_geometries(points), pairwise(points), strict=True):
        expected = math.atan2(b[1] - a[1], b[0] - a[0])
        assert abs(angle_difference(record.hdg, expected)) < 1e-12


# --------------------------------------------------------------------------
# Collinearity merging
# --------------------------------------------------------------------------


def test_a_straight_run_collapses_to_one_segment():
    straight = [(float(i), 0.0, 0.0) for i in range(10)]
    assert merge_collinear(straight) == [(0.0, 0.0, 0.0), (9.0, 0.0, 0.0)]


def test_merging_preserves_the_end_points_and_real_bends():
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 2.0, 0.0)]
    merged = merge_collinear(points)
    assert merged[0] == points[0]
    assert merged[-1] == points[-1]
    assert (2.0, 0.0, 0.0) in merged  # the bend survives


def test_merging_does_not_swallow_a_shallow_curve_one_step_at_a_time():
    """A long arc is many tiny turns; checking only adjacent pairs would fuse it."""
    arc = [
        (50.0 * math.cos(math.radians(a)), 50.0 * math.sin(math.radians(a)), 0.0)
        for a in range(0, 91, 3)
    ]
    merged = merge_collinear(arc, chord_tolerance=1e-4, heading_tolerance=1e-6)
    assert len(merged) == len(arc)


def test_duplicate_points_are_dropped_before_headings_are_taken():
    points = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert len(dedupe(points)) == 2
    assert len(line_geometries(points)) == 1


def test_build_plan_view_returns_the_polyline_it_described():
    records, simplified = build_plan_view(zigzag())
    assert len(records) == len(simplified) - 1
    assert math.isclose(sum(r.length for r in records), total_length(simplified), rel_tol=1e-12)


# --------------------------------------------------------------------------
# Stations and projection
# --------------------------------------------------------------------------


def test_station_of_point_recovers_a_vertex_station():
    points = zigzag()
    expected = stations(points)
    for index, point in enumerate(points):
        assert math.isclose(
            station_of_point(points, (point[0], point[1])), expected[index], abs_tol=1e-9
        )


def test_point_at_station_round_trips():
    points = zigzag()
    for s in (0.0, 3.5, 10.0, total_length(points)):
        point, _hdg = point_at_station(points, s)
        assert math.isclose(station_of_point(points, (point[0], point[1])), s, abs_tol=1e-9)


def test_closest_point_interpolates_z():
    points = [(0.0, 0.0, 0.0), (10.0, 0.0, 5.0)]
    nearest, distance = closest_point(points, (5.0, 3.0))
    assert math.isclose(nearest[2], 2.5, abs_tol=1e-12)
    assert math.isclose(distance, 3.0, abs_tol=1e-12)


def test_sample_stations_includes_every_vertex():
    points = zigzag()
    sampled = sample_stations(points, max_step=1.0)
    for station in stations(points):
        assert any(math.isclose(station, s, abs_tol=1e-9) for s in sampled)


def test_sample_stations_respects_the_maximum_step():
    sampled = sample_stations([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)], max_step=5.0)
    assert max(b - a for a, b in pairwise(sampled)) <= 5.0 + 1e-9


# --------------------------------------------------------------------------
# Sides and normals
# --------------------------------------------------------------------------


def test_left_normal_is_ninety_degrees_counter_clockwise():
    assert left_normal(0.0) == (0.0, 1.0)
    nx, ny = left_normal(math.pi / 2)
    assert math.isclose(nx, -1.0, abs_tol=1e-12)
    assert math.isclose(ny, 0.0, abs_tol=1e-12)


def test_signed_side_detects_which_bound_is_left():
    reference = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    assert signed_side(reference, [(0.0, 3.0, 0.0), (10.0, 3.0, 0.0)]) > 0
    assert signed_side(reference, [(0.0, -3.0, 0.0), (10.0, -3.0, 0.0)]) < 0


def test_lateral_offset_sign_matches_the_left_normal():
    assert lateral_offset((0.0, 0.0, 0.0), 0.0, (0.0, 2.0, 0.0)) > 0
    assert lateral_offset((0.0, 0.0, 0.0), 0.0, (0.0, -2.0, 0.0)) < 0


def test_normalize_angle_wraps_into_the_half_open_turn():
    assert math.isclose(normalize_angle(3 * math.pi), math.pi, abs_tol=1e-12)
    assert math.isclose(normalize_angle(-3 * math.pi), math.pi, abs_tol=1e-12)


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def test_a_constant_profile_collapses_to_one_record():
    records = build_profile([0.0, 5.0, 10.0], [3.5, 3.5, 3.5])
    assert len(records) == 1
    assert records[0].s == 0.0
    assert records[0].a == 3.5


def test_a_linear_profile_is_exact_at_every_sample():
    sts = [0.0, 5.0, 10.0, 20.0]
    values = [3.0, 3.5, 4.0, 5.0]
    records = build_profile(sts, values, tolerance=1e-9)
    for station, expected in zip(sts[:-1], values[:-1], strict=False):
        record = max((r for r in records if r.s <= station + 1e-12), key=lambda r: r.s)
        assert math.isclose(record.a + record.b * (station - record.s), expected, abs_tol=1e-9)


def test_the_first_profile_record_always_starts_at_zero():
    """OpenDRIVE requires width record 0 to sit at the section origin."""
    records = build_profile([2.0, 7.0, 12.0], [1.0, 2.0, 4.0], tolerance=1e-9)
    assert records[0].s == 0.0


def test_constant_lane_width_is_recovered_exactly():
    reference = [(0.0, 0.0, 0.0), (50.0, 0.0, 0.0)]
    outer = [(0.0, -3.25, 0.0), (50.0, -3.25, 0.0)]
    records, minimum = lane_widths(reference, reference, outer)
    assert len(records) == 1
    assert math.isclose(records[0].a, 3.25, abs_tol=1e-9)
    assert math.isclose(minimum, 3.25, abs_tol=1e-9)


def test_a_tapering_lane_produces_a_varying_profile():
    reference = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
    outer = [(0.0, -3.0, 0.0), (100.0, -5.0, 0.0)]
    records, minimum = lane_widths(reference, reference, outer, max_step=10.0)
    assert len(records) > 1
    assert math.isclose(minimum, 3.0, abs_tol=1e-6)
    assert math.isclose(records[0].a, 3.0, abs_tol=1e-6)


def test_flat_elevation_collapses_to_a_single_record():
    records = road_elevation([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
    assert len(records) == 1
    assert records[0].a == 0.0


def test_rising_elevation_is_tracked():
    records = road_elevation([(0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (20.0, 0.0, 3.0)])
    assert len(records) >= 2
    assert math.isclose(records[0].a, 0.0, abs_tol=1e-12)
