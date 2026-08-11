"""A cross-section whose bounds cross over, and the width sign that reveals it.

OpenDRIVE has no negative `<width>`, so a lanelet whose outer bound crosses
inside its inner one cannot be reproduced. What the transpiler did was take the
magnitude, which is the worst of the three available answers: the lane's outer
edge lands as far *past* its inner edge as it should have fallen short, so twice
the error, and `W703` -- the diagnostic that exists to report exactly this --
could never fire, because the value it tested had already been made positive.

Widths are now signed by the direction the lane runs and clamped at zero. Three
lanelets on the Lanelet2 Karlsruhe example say so, and the worst of them moves
1.31 m closer to its own bound.
"""

from __future__ import annotations

import pytest

from opendrive_transpiler import TranspileOptions, transpile_source


def convert(source: str, **kwargs):
    return transpile_source(
        source, "t.py", options=TranspileOptions(strict=False, name="t", **kwargs)
    )


def codes(result) -> set[str]:
    return {d.code for d in result.diagnostics}


def only_lane(result):
    lanes = [lane for road in result.model.roads for s in road.lane_sections for lane in s.lanes]
    assert len(lanes) == 1
    return lanes[0]


def width_at(lane, s: float) -> float:
    record = max((r for r in lane.widths if r.s <= s + 1e-9), key=lambda r: r.s)
    ds = s - record.s
    return record.a + record.b * ds + record.c * ds**2 + record.d * ds**3


# The right bound runs 3.5 m out for 50 m and then swings across the left bound,
# ending 0.5 m on its far side. Unambiguously the right bound for most of the
# road, so nothing is swapped -- it simply crosses at the end.
CROSSES_AT_THE_END = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
left = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 60, 0, 0)])
right = LineString3d(getId(), [Point3d(getId(), 0, -3.5, 0), Point3d(getId(), 50, -3.5, 0),
                               Point3d(getId(), 60, 0.5, 0)])
ll = Lanelet(getId(), left, right); ll.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([ll])
"""


def test_a_crossed_cross_section_is_reported():
    result = convert(CROSSES_AT_THE_END)
    assert "LL2ODR-W703" in codes(result)


def test_the_report_gives_the_depth_of_the_crossing():
    result = convert(CROSSES_AT_THE_END)
    message = next(d.message for d in result.diagnostics if d.code == "LL2ODR-W703")
    assert "0.5 m" in message, "how far past its inner edge the outer bound went"
    assert "zero width" in message, "and what was emitted instead"


def test_the_lane_closes_to_zero_rather_than_reopening_on_the_wrong_side():
    """The magnitude would send the outer edge back out at +0.5 m: twice the error."""
    lane = only_lane(convert(CROSSES_AT_THE_END))
    assert width_at(lane, 0.0) == pytest.approx(3.5)
    assert width_at(lane, 50.0) == pytest.approx(3.5)
    assert width_at(lane, 60.0) == pytest.approx(0.0, abs=1e-6)
    assert min(width_at(lane, s) for s in range(0, 61)) >= -1e-9, "never negative"


def test_a_bound_that_is_wholly_on_the_wrong_side_is_turned_round_first():
    """Crossing is not the same as being mislabelled, and the two are reported apart.

    Here the bound named "right" ends up left of centre overall, so `W501` turns
    the cross-section round; only what still crosses afterwards is `W703`.
    """
    result = convert("""
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
left = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 60, 0, 0)])
right = LineString3d(getId(), [Point3d(getId(), 0, -3.5, 0), Point3d(getId(), 60, 2.0, 0)])
ll = Lanelet(getId(), left, right); ll.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([ll])
""")
    assert {"LL2ODR-W501", "LL2ODR-W703"} <= codes(result)
    lane = only_lane(result)
    assert min(width_at(lane, s) for s in range(0, 60)) >= -1e-9


def test_a_well_formed_lanelet_says_nothing_about_crossing():
    result = convert("""
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
left = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 60, 0, 0)])
right = LineString3d(getId(), [Point3d(getId(), 0, -3.5, 0), Point3d(getId(), 60, -3.5, 0)])
ll = Lanelet(getId(), left, right); ll.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([ll])
""")
    assert "LL2ODR-W703" not in codes(result)
    assert only_lane(result).constant_width == pytest.approx(3.5)


def test_a_lane_on_the_left_of_the_reference_takes_the_other_sign():
    """`+1` runs outward as *rising* `t`, so the sign cannot simply be negative.

    Under `--reference-line=centerline` there are lanes on both sides, and a
    single sign convention would report every lane on one of them as crossed.
    """
    result = convert(
        """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
a = LineString3d(getId(), [Point3d(getId(), 0, 3.5, 0), Point3d(getId(), 60, 3.5, 0)])
b = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 60, 0, 0)])
c = LineString3d(getId(), [Point3d(getId(), 0, -3.5, 0), Point3d(getId(), 60, -3.5, 0)])
one = Lanelet(getId(), a, b); one.attributes["subtype"] = "road"
two = Lanelet(getId(), b, c); two.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([one, two])
""",
        reference_line="centerline",
    )
    lanes = [lane for road in result.model.roads for s in road.lane_sections for lane in s.lanes]
    assert {lane.lane_id for lane in lanes} == {1, -1}, "one lane each side"
    assert "LL2ODR-W703" not in codes(result)
    assert all(lane.constant_width == pytest.approx(3.5) for lane in lanes)
