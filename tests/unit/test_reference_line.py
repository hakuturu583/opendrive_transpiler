"""Which curve the planView follows, and what each choice costs.

A lane is only as long as the reference line. lanelet2 does not require a
lanelet's two bounds to span the same stretch of it, so a road that follows the
left bound can end before the lanelet does and drop the rest -- 330 m of the
Lanelet2 Karlsruhe example map, a third of one lanelet in the worst case.

`--reference-line=auto` follows lanelet2's centerline on exactly those roads. It
is not the default because the recovery is not free: OpenDRIVE places a lane edge
at an offset measured *perpendicular* to the reference line, so a bound running
at an angle to it cannot be followed by one scalar per station. Extent and
boundary placement cannot both be had, and which matters is the caller's to say.
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


# A lanelet that outruns its own left bound: 24 m of left bound against a right
# bound that carries on for 120, so the lanelet opens into a wedge. This is the
# shape of lanelet 9037740909199276460 on the Karlsruhe map, the worst case there.
WEDGE = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
left = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 24, 0, 0)])
right = LineString3d(getId(), [Point3d(getId(), 0, -3.8, 0), Point3d(getId(), 60, -20, 0),
                               Point3d(getId(), 120, -45, 0)])
ll = Lanelet(getId(), left, right); ll.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([ll])
"""


def road_length(result) -> float:
    return sum(road.length for road in result.model.roads)


# --------------------------------------------------------------------------
# The default is unchanged
# --------------------------------------------------------------------------


def test_the_default_follows_the_bound_and_says_what_that_costs():
    """Exact geometry by default; the shortfall is reported rather than hidden."""
    result = convert(WEDGE)
    assert road_length(result) == pytest.approx(24.0, abs=0.01)
    assert not result.model.roads[0].lane_offsets, "the reference line is the bound itself"
    assert "LL2ODR-W510" not in codes(result), "nothing was switched"
    assert "LL2ODR-W503" in codes(result)


def test_the_shortfall_is_reported_in_metres():
    """ "its two bounds cover different stretches" is not actionable on its own."""
    result = convert(WEDGE)
    message = next(d.message for d in result.diagnostics if d.code == "LL2ODR-W503")
    assert "lanelet2 makes it" in message
    assert "is not represented" in message
    assert "%" in message, "the fraction covered is the number worth quoting"


@pytest.mark.parametrize(
    "fixture", sorted(p.name for p in FIXTURES.glob("*.py") if not p.name.startswith("_"))
)
def test_no_fixture_changes_under_the_default(fixture: str):
    """The choice is opt-in, so nothing already converting may move."""
    result = convert((FIXTURES / fixture).read_text(encoding="utf-8"))
    assert "LL2ODR-W510" not in codes(result)
    assert all(not road.lane_offsets for road in result.model.roads)


# --------------------------------------------------------------------------
# What `auto` buys
# --------------------------------------------------------------------------


def test_auto_follows_the_centerline_where_the_bound_would_cut_a_lanelet_short():
    result = convert(WEDGE, reference_line="auto")
    assert road_length(result) > 40.0, "the lanelet is far longer than its left bound"
    assert result.model.roads[0].lane_offsets, "lane 0 is off the reference line now"
    assert "LL2ODR-W510" in codes(result)


def test_auto_keeps_every_lane_on_the_right_of_lane_zero():
    """`+` means left, and left means travelling against `s`.

    Placing lane 0 on the boundary *nearest* the centerline instead would let it
    land on either side depending on geometry, and half the lanes of a real map
    would come out claiming a direction they do not have.
    """
    result = convert(WEDGE, reference_line="auto")
    lanes = [lane for road in result.model.roads for s in road.lane_sections for lane in s.lanes]
    assert lanes
    assert all(lane.lane_id < 0 for lane in lanes)


def test_auto_says_which_roads_it_moved_off_their_bound():
    result = convert(WEDGE, reference_line="auto")
    message = next(d.message for d in result.diagnostics if d.code == "LL2ODR-W510")
    assert "computed curve rather than input coordinates" in message


# --------------------------------------------------------------------------
# What `auto` must leave alone
# --------------------------------------------------------------------------


def test_a_curve_is_not_mistaken_for_a_truncation():
    """The trap a length comparison falls into.

    On an arc the centerline is longer than the inner bound purely because of the
    radius -- 5% on this fixture -- while nothing whatever is lost. Only an
    along-track overhang test tells the two apart.
    """
    source = (FIXTURES / "curved_road.py").read_text(encoding="utf-8")
    plain = convert(source)
    auto = convert(source, reference_line="auto")
    assert "LL2ODR-W510" not in codes(auto)
    assert road_length(auto) == pytest.approx(road_length(plain))


def test_bounds_that_do_not_overlap_are_left_alone():
    """A big overhang with nothing to gain: switching would cost geometry for free.

    The two bounds here are 24 m apart along the road and never overlap, so the
    centerline is no longer than either of them.
    """
    source = (FIXTURES / "with_projector.py").read_text(encoding="utf-8")
    auto = convert(source, reference_line="auto")
    assert "LL2ODR-W510" not in codes(auto)
    assert all(not road.lane_offsets for road in auto.model.roads)


def test_a_well_formed_road_is_untouched_by_auto():
    source = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
a = LineString3d(getId(), [Point3d(getId(), 0, 3.5, 0), Point3d(getId(), 40, 3.5, 0)])
b = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 40, 0, 0)])
ll = Lanelet(getId(), a, b); ll.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([ll])
"""
    auto = convert(source, reference_line="auto")
    assert "LL2ODR-W510" not in codes(auto)
    assert road_length(auto) == pytest.approx(40.0)


# --------------------------------------------------------------------------
# The explicit option keeps its own meaning
# --------------------------------------------------------------------------


def test_asking_for_a_centerline_outright_still_centres_the_cross_section():
    """`auto` is a fallback, not a redefinition of `centerline`.

    Asking for a centerline asks for a cross-section laid out around its middle,
    with lanes on both sides. That is a different thing from the fallback, which
    only wants a reference long enough to carry the lanelet.
    """
    source = (FIXTURES / "parallel_lanes.py").read_text(encoding="utf-8")
    result = convert(source, reference_line="centerline")
    lanes = [lane for road in result.model.roads for s in road.lane_sections for lane in s.lanes]
    assert any(lane.lane_id > 0 for lane in lanes)
    assert "LL2ODR-W510" not in codes(result), "asked for globally, so not news"


def test_an_invalid_reference_line_is_refused():
    with pytest.raises(ValueError, match="invalid reference_line"):
        TranspileOptions(reference_line="middle").validate()
