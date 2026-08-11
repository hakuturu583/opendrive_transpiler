"""Cutting a laneSection where a lane boundary starts and where it ends.

A lane in OpenDRIVE spans its whole laneSection, so a boundary that covers only
part of the road still has to be given a width over the rest of it. By default
that width comes from the boundary's nearest point: the profile stays defined,
but the lane edge is placed by guesswork, and on the Lanelet2 Karlsruhe example
30% of stations have no boundary under the normal to measure.

`--split-at-bound-extent` cuts the section at the stations where the boundary
really begins and ends, and tapers the lane to zero width over the stretch where
it is absent. That is a stronger claim than the input makes -- lanelet2 gives a
lanelet one polygon, not a width per station -- so it is opt-in, and every lane
it zeroes is named in a diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opendrive_transpiler import TranspileOptions, transpile_source

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def convert(source: str, **kwargs):
    return transpile_source(
        source, "t.py", options=TranspileOptions(strict=False, name="t", **kwargs)
    )


def codes(result) -> set[str]:
    return {d.code for d in result.diagnostics}


def sections(result):
    return [section for road in result.model.roads for section in road.lane_sections]


def lanelet(left: str, right: str) -> str:
    return f"""
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
left = LineString3d(getId(), [{left}])
right = LineString3d(getId(), [{right}])
ll = Lanelet(getId(), left, right); ll.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([ll])
"""


def point(x: float, y: float) -> str:
    return f"Point3d(getId(), {x}, {y}, 0)"


# 60 m of left bound, but the right bound does not start until 20 m in. Nothing
# in lanelet2 forbids that, and the Karlsruhe map has 99 lanelets like it.
LATE_START = lanelet(
    f"{point(0, 0)}, {point(60, 0)}",
    f"{point(20, -3.5)}, {point(60, -3.5)}",
)
EARLY_END = lanelet(
    f"{point(0, 0)}, {point(60, 0)}",
    f"{point(0, -3.5)}, {point(40, -3.5)}",
)
# The same shape, but the missing stretch is under the default 2 m threshold.
SHORT_GAP = lanelet(
    f"{point(0, 0)}, {point(60, 0)}",
    f"{point(1.2, -3.5)}, {point(60, -3.5)}",
)


# --------------------------------------------------------------------------
# The default is unchanged
# --------------------------------------------------------------------------


def test_the_default_gives_the_lane_a_width_it_cannot_measure():
    """What the option exists to fix, stated as the current behaviour.

    There is no right bound at all over the first 20 m, and the file says the
    lane is 3.5 m wide there.
    """
    result = convert(LATE_START)
    assert len(sections(result)) == 1
    assert sections(result)[0].lanes[0].constant_width == pytest.approx(3.5)
    assert "LL2ODR-W511" not in codes(result)


@pytest.mark.parametrize(
    "fixture", sorted(p.name for p in FIXTURES.glob("*.py") if not p.name.startswith("_"))
)
def test_no_fixture_splits_under_the_default(fixture: str):
    source = (FIXTURES / fixture).read_text(encoding="utf-8")
    plain = convert(source)
    assert "LL2ODR-W511" not in codes(plain)


@pytest.mark.parametrize(
    "fixture", sorted(p.name for p in FIXTURES.glob("*.py") if not p.name.startswith("_"))
)
def test_the_option_leaves_a_well_formed_fixture_alone(fixture: str):
    """Turning it on may not move a map whose bounds already span their road."""
    source = (FIXTURES / fixture).read_text(encoding="utf-8")
    plain = convert(source)
    split = convert(source, split_at_bound_extent=True)
    if "LL2ODR-W511" in codes(split):
        pytest.skip(f"{fixture} genuinely has a boundary that stops short")
    assert [s.s for s in sections(split)] == [s.s for s in sections(plain)]
    assert [lane.widths for s in sections(split) for lane in s.lanes] == [
        lane.widths for s in sections(plain) for lane in s.lanes
    ]


# --------------------------------------------------------------------------
# What the split does
# --------------------------------------------------------------------------


def test_a_bound_that_starts_late_opens_the_lane_from_zero():
    result = convert(LATE_START, split_at_bound_extent=True)
    first, second = sections(result)
    assert first.s == pytest.approx(0.0)
    assert second.s == pytest.approx(20.0, abs=0.01)

    opening = first.lanes[0].widths
    assert len(opening) == 1, "a straight taper is one linear record"
    assert opening[0].a == pytest.approx(0.0, abs=1e-6), "no lane where there is no bound"
    assert opening[0].b == pytest.approx(3.5 / 20.0, rel=1e-3), "reaching full width at the cut"
    assert second.lanes[0].constant_width == pytest.approx(3.5)


def test_a_bound_that_ends_early_closes_the_lane_to_zero():
    result = convert(EARLY_END, split_at_bound_extent=True)
    first, second = sections(result)
    assert second.s == pytest.approx(40.0, abs=0.01)
    assert first.lanes[0].constant_width == pytest.approx(3.5)

    closing = second.lanes[0].widths
    assert closing[0].a == pytest.approx(3.5, abs=1e-6)
    assert closing[0].b == pytest.approx(-3.5 / 20.0, rel=1e-3)


def test_the_two_halves_of_a_split_lanelet_are_linked():
    """A lanelet is not its own successor, so nothing else would join these."""
    first, second = sections(convert(LATE_START, split_at_bound_extent=True))
    assert first.lanes[0].successor == second.lanes[0].lane_id
    assert second.lanes[0].predecessor == first.lanes[0].lane_id


def test_every_zeroed_lane_is_named_with_the_stretch_it_is_missing_from():
    result = convert(LATE_START, split_at_bound_extent=True)
    message = next(d.message for d in result.diagnostics if d.code == "LL2ODR-W511")
    assert "0-20 m" in message
    assert "tapering from zero width" in message
    assert "carries no traffic" in message, "the routing consequence is the point"


def test_collapsing_an_inner_lane_does_not_move_the_lane_outside_it():
    """OpenDRIVE widths are cumulative, so a zeroed lane would shift its neighbours.

    Here the *shared* bound is the one that starts late. Lane -1 has to open from
    nothing, and lane -2 has to keep its own outer edge on the boundary that is
    there all along -- which means widening to cover the gap, not shifting inward.
    """
    source = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
l0 = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 60, 0, 0)])
l1 = LineString3d(getId(), [Point3d(getId(), 20, -3.5, 0), Point3d(getId(), 60, -3.5, 0)])
l2 = LineString3d(getId(), [Point3d(getId(), 0, -7, 0), Point3d(getId(), 60, -7, 0)])
a = Lanelet(getId(), l0, l1); a.attributes["subtype"] = "road"
b = Lanelet(getId(), l1, l2); b.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([a, b])
"""
    opening, full = sections(convert(source, split_at_bound_extent=True))
    inner, outer = opening.lanes
    assert (inner.lane_id, outer.lane_id) == (-1, -2)
    assert inner.widths[0].a == pytest.approx(0.0, abs=1e-6)
    assert outer.widths[0].a == pytest.approx(7.0, abs=1e-6), "the outer edge has not moved"
    # And the two together still add up to 7 m at every station of the taper.
    assert inner.widths[0].b == pytest.approx(-outer.widths[0].b, rel=1e-9)
    assert [lane.constant_width for lane in full.lanes] == [
        pytest.approx(3.5),
        pytest.approx(3.5),
    ]


def test_the_lanelet_id_is_carried_through_the_split():
    """Both halves have to claim the lanelet, or provenance loses one of them."""
    first, second = sections(convert(LATE_START, split_at_bound_extent=True))
    assert first.lanes[0].lanelet2_id == second.lanes[0].lanelet2_id


# --------------------------------------------------------------------------
# The threshold
# --------------------------------------------------------------------------


def test_a_gap_under_the_threshold_is_not_worth_a_section():
    """1.2 m of missing bound against a whole extra laneSection: not a trade."""
    result = convert(SHORT_GAP, split_at_bound_extent=True)
    assert len(sections(result)) == 1
    assert "LL2ODR-W511" not in codes(result)


def test_the_threshold_can_be_lowered_to_take_it():
    result = convert(SHORT_GAP, split_at_bound_extent=True, bound_extent_gap=1.0)
    assert len(sections(result)) == 2
    assert sections(result)[1].s == pytest.approx(1.2, abs=0.01)


def test_a_cut_that_would_leave_a_sliver_is_refused():
    """The threshold applies to *both* sides of a cut, not only the absent one.

    A boundary that stops 0.5 m before the road ends leaves plenty of road
    behind it, but the section in front would be too short to sample.
    """
    source = lanelet(
        f"{point(0, 0)}, {point(60, 0)}",
        f"{point(0, -3.5)}, {point(59.5, -3.5)}",
    )
    assert len(sections(convert(source, split_at_bound_extent=True))) == 1


def test_a_gap_of_zero_is_refused():
    with pytest.raises(ValueError, match="bound_extent_gap must be positive"):
        TranspileOptions(bound_extent_gap=0.0).validate()
