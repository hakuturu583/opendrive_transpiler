"""The centerline port, checked against its own upstream oracle.

Every expectation here is copied from the `#[cfg(test)]` block of
`crates/ll2-core/src/centerline.rs` in simple_lanelet2 -- not from intuition.
That distinction matters: the algorithm advances one bound at a time, so evenly
spaced parallel bounds produce *twice* as many centerline points as either bound,
and the unequal-count case lands on stations no one would guess.
"""

from __future__ import annotations

from opendrive_transpiler.ir.centerline import centerline_coords


def line(points: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    return [(x, y, 0.0) for x, y in points]


def xy(result: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    return [(p[0], p[1]) for p in result]


def test_parallel_bounds_interleave_both_sides():
    left = line([(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)])
    right = line([(0.0, -1.0), (1.0, -1.0), (2.0, -1.0)])
    assert xy(centerline_coords(left, right)) == [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.5, 0.0),
        (2.0, 0.0),
    ]


def test_two_point_bounds():
    left = line([(0.0, 1.0), (1.0, 1.0)])
    right = line([(0.0, -1.0), (1.0, -1.0)])
    assert xy(centerline_coords(left, right)) == [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]


def test_an_empty_bound_gives_an_empty_centerline():
    assert centerline_coords([], line([(0.0, 0.0)])) == []
    assert centerline_coords(line([(0.0, 0.0)]), []) == []


def test_unequal_point_counts_still_reach_both_end_points():
    left = line([(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0)])
    right = line([(0.0, -1.0), (2.0, -1.0)])
    assert xy(centerline_coords(left, right)) == [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (1.5, 0.0),
        (2.0, 0.0),
    ]


def test_z_is_averaged_even_though_the_search_is_planar():
    left = [(0.0, 1.0, 4.0), (1.0, 1.0, 4.0)]
    right = [(0.0, -1.0, 0.0), (1.0, -1.0, 0.0)]
    assert all(point[2] == 2.0 for point in centerline_coords(left, right))


def test_shared_first_point_skips_the_left_bounds_start():
    """A lanelet closed at its start shares the first point object between bounds.

    lanelet2 tests identity here, not coordinate equality, so the caller answers
    it -- and the answer changes where the search begins.
    """
    left = line([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
    right = line([(0.0, 0.0), (1.0, -1.0), (2.0, 0.0)])
    shared = centerline_coords(left, right, shares_first_point=True)
    unshared = centerline_coords(left, right, shares_first_point=False)
    assert shared != unshared
    assert shared[0] == (0.0, 0.0, 0.0)
