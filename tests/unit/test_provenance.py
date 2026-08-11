"""What a converted map says about the map it came from.

The `.xodr` is the only artefact most consumers see, so anything it does not
record cannot be checked. These pin the two things that make a conversion
auditable: which lanelet each lane is, and a refusal to build a cross-section out
of boundaries that are not actually shared.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opendrive_transpiler import TranspileOptions, transpile_source

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def convert(source: str, **kwargs):
    options = TranspileOptions(strict=False, name="t", **kwargs)
    return transpile_source(source, "t.py", options=options)


def codes(result) -> set[str]:
    return {d.code for d in result.diagnostics}


TWO_LANES = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
a = LineString3d(getId(), [Point3d(getId(), 0, 7, 0), Point3d(getId(), 30, 7, 0)])
b = LineString3d(getId(), [Point3d(getId(), 0, 3.5, 0), Point3d(getId(), 30, 3.5, 0)])
c = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 30, 0, 0)])
inner = Lanelet(getId(), a, b); inner.attributes["subtype"] = "road"
outer = Lanelet(getId(), b, c); outer.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([inner, outer])
"""


# --------------------------------------------------------------------------
# Lane provenance
# --------------------------------------------------------------------------


def test_every_lane_records_the_lanelet_it_came_from():
    result = convert(TWO_LANES)
    lanes = [lane for road in result.model.roads for lane in road.lane_sections[0].lanes]
    assert len(lanes) == 2
    assert all(lane.lanelet2_id for lane in lanes)


def test_the_lanelet_id_reaches_the_generated_script():
    """A comment is not enough -- it has to survive into the emitted file."""
    result = convert(TWO_LANES)
    lanes = [lane for road in result.model.roads for lane in road.lane_sections[0].lanes]
    for lane in lanes:
        assert f"xodr.UserData(\"lanelet2_id\", '{lane.lanelet2_id}')" in result.code


def test_the_subtype_reaches_it_too():
    result = convert(TWO_LANES)
    assert "xodr.UserData(\"lanelet2_subtype\", 'road')" in result.code


def test_a_road_built_from_three_lanelets_records_all_three():
    """The road `name` carries only the first and last, which is why this exists."""
    source = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
def bound(y):
    return LineString3d(getId(), [Point3d(getId(), 0, y, 0), Point3d(getId(), 30, y, 0)])
edges = [bound(y) for y in (10.5, 7.0, 3.5, 0.0)]
lls = []
for i in range(3):
    ll = Lanelet(getId(), edges[i], edges[i + 1])
    ll.attributes["subtype"] = "road"
    lls.append(ll)
lanelet_map = createMapFromLanelets(lls)
"""
    result = convert(source)
    road = result.model.roads[0]
    assert len(road.lane_sections[0].lanes) == 3
    recorded = {lane.lanelet2_id for lane in road.lane_sections[0].lanes}
    assert len(recorded) == 3, "each lane names a different lanelet"
    for lanelet_id in recorded:
        assert f"\"lanelet2_id\", '{lanelet_id}'" in result.code


# --------------------------------------------------------------------------
# The cross-section stack
# --------------------------------------------------------------------------


def test_neighbours_that_share_a_bound_are_not_reported():
    assert "LL2ODR-W507" not in codes(convert(TWO_LANES))


def test_coincident_but_separately_built_bounds_still_count_as_shared():
    """Adjacency is inferred with a tolerance, so identity must use the same one.

    A script that builds each lanelet from its own `Point3d` means the neighbours
    to share the bound between them; comparing point objects instead would call
    every such pair unshared.
    """
    source = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
def bound(y):
    return LineString3d(getId(), [Point3d(getId(), 0, y, 0), Point3d(getId(), 30, y, 0)])
inner = Lanelet(getId(), bound(7.0), bound(3.5))
inner.attributes["subtype"] = "road"
outer = Lanelet(getId(), bound(3.5), bound(0.0))
outer.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([inner, outer])
"""
    result = convert(source)
    assert "LL2ODR-W507" not in codes(result)
    assert len(result.model.roads[0].lane_sections[0].lanes) == 2


def test_a_bound_named_right_by_both_neighbours_is_still_stacked_correctly():
    """Nothing forces a mapper to name a shared bound consistently.

    Here the middle bound is the `right` of both lanelets, so the naive stack
    `m0.left, m0.right, m1.right` would put that same bound at both edges of the
    cross-section and measure the second lane's width against the wrong one.
    """
    source = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
top = LineString3d(getId(), [Point3d(getId(), 0, 7, 0), Point3d(getId(), 30, 7, 0)])
middle = LineString3d(getId(), [Point3d(getId(), 0, 3.5, 0), Point3d(getId(), 30, 3.5, 0)])
bottom = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 30, 0, 0)])
inner = Lanelet(getId(), top, middle); inner.attributes["subtype"] = "road"
# Both bounds named the other way round: `middle` is this one's right as well.
outer = Lanelet(getId(), bottom, middle); outer.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([inner, outer])
"""
    result = convert(source)
    assert "LL2ODR-W507" not in codes(result), "the orientation is recoverable, not an error"
    lanes = result.model.roads[0].lane_sections[0].lanes
    assert len(lanes) == 2
    for lane in lanes:
        assert lane.constant_width == pytest.approx(3.5, abs=1e-6)


# Two lanelets whose shared bound is the *right* bound of both: A runs +x over
# y 3.5..7, B runs -x over y 0..3.5. This is the arrangement that exposes the
# ordering bug below, and the one the Karlsruhe example map contains.
SHARED_RIGHT_BOUNDS = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
a_left  = LineString3d(getId(), [Point3d(getId(), 0, 7, 0), Point3d(getId(), 30, 7, 0)])
mid     = LineString3d(getId(), [Point3d(getId(), 0, 3.5, 0), Point3d(getId(), 30, 3.5, 0)])
mid_rev = LineString3d(getId(), [Point3d(getId(), 30, 3.5, 0), Point3d(getId(), 0, 3.5, 0)])
b_left  = LineString3d(getId(), [Point3d(getId(), 30, 0, 0), Point3d(getId(), 0, 0, 0)])
A = Lanelet(getId(), a_left, mid); A.attributes["subtype"] = "road"
B = Lanelet(getId(), b_left, mid_rev); B.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([A, B])
"""


def test_an_opposing_pair_is_stacked_left_to_right():
    """Adjacency has no side, so the walk that collects members is not ordered.

    This arrangement used to come out stacked right-to-left, which put the shared
    bound at *both* edges of the cross-section and dropped y = 7 entirely --
    boundaries at y = 3.5, 0.0, 3.5. Ordering is now taken from geometry, so the
    stack is y = 7, 3.5, 0 and both lanes get their real width.
    """
    result = convert(SHARED_RIGHT_BOUNDS)
    assert "LL2ODR-W507" not in codes(result)
    lanes = result.model.roads[0].lane_sections[0].lanes
    assert len(lanes) == 2
    for lane in lanes:
        assert lane.constant_width == pytest.approx(3.5, abs=1e-6)


# --------------------------------------------------------------------------
# Merges, and links that would be read against the wrong road
# --------------------------------------------------------------------------


def test_a_merge_states_only_the_join_its_road_level_link_names():
    """A lane's `<predecessor id>` is resolved in the road the road-level link names.

    Two roads merging into one both used to write their lane links in, while the
    road-level `<predecessor>` could name only one of them -- so the other's lane
    ids were resolved in a road they did not come from. That is worse than a gap,
    because it reads as connectivity.
    """
    result = convert((FIXTURES / "merge.py").read_text(encoding="utf-8"))
    merged = max(result.model.roads, key=lambda road: len(road.lane_sections[0].lanes))
    assert len(merged.lane_sections[0].lanes) == 2
    assert merged.predecessor is not None

    stated = [lane for lane in merged.lane_sections[0].lanes if lane.predecessor is not None]
    assert len(stated) == 1, "only the approach the road-level link names can state a lane"


def test_the_unjoined_side_of_a_merge_is_reported():
    result = convert((FIXTURES / "merge.py").read_text(encoding="utf-8"))
    assert "LL2ODR-W509" in codes(result)


def test_a_plain_continuation_still_links_both_ways():
    """The guard must not cost the ordinary case: one road into one road."""
    source = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
mid_left = Point3d(getId(), 30, 3, 0)
mid_right = Point3d(getId(), 30, 0, 0)
first = Lanelet(
    getId(),
    LineString3d(getId(), [Point3d(getId(), 0, 3, 0), mid_left]),
    LineString3d(getId(), [Point3d(getId(), 0, 0, 0), mid_right]),
)
first.attributes["subtype"] = "road"
second = Lanelet(
    getId(),
    LineString3d(getId(), [mid_left, Point3d(getId(), 60, 8, 0)]),
    LineString3d(getId(), [mid_right, Point3d(getId(), 60, 5, 0)]),
)
second.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([first, second])
"""
    result = convert(source)
    assert "LL2ODR-W509" not in codes(result)
    if len(result.model.roads) == 2:
        head, tail = result.model.roads
        assert head.successor is not None and tail.predecessor is not None
        assert head.lane_sections[-1].lanes[0].successor is not None
        assert tail.lane_sections[0].lanes[0].predecessor is not None


# --------------------------------------------------------------------------
# Contraflow on the right
# --------------------------------------------------------------------------


def test_opposing_traffic_on_the_left_is_a_plain_two_way_road():
    """The arrangement right-hand traffic produces: `+` lanes, no complaint."""
    result = convert((FIXTURES / "two_way.py").read_text(encoding="utf-8"))
    assert "LL2ODR-W508" not in codes(result)
    lanes = result.model.roads[0].lane_sections[0].lanes
    assert any(lane.lane_id > 0 for lane in lanes), "the opposing lane is a + lane"


def test_opposing_traffic_on_the_right_is_reported():
    """A contraflow lane to the right of forward traffic has no expressible id.

    `+` means left, so a member that must be `+` cannot sit on the right. The
    geometry is still the input's own -- only the direction its id implies is
    wrong -- so this is a warning about the convention, not a dropped lane.
    """
    source = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
top = LineString3d(getId(), [Point3d(getId(), 0, 7, 0), Point3d(getId(), 30, 7, 0)])
middle = LineString3d(getId(), [Point3d(getId(), 0, 3.5, 0), Point3d(getId(), 30, 3.5, 0)])
bottom = LineString3d(getId(), [Point3d(getId(), 30, 0, 0), Point3d(getId(), 0, 0, 0)])
forward = Lanelet(getId(), top, middle)
forward.attributes["subtype"] = "road"
# Runs the other way, and lies to the right of the forward lane.
against = Lanelet(getId(), LineString3d(getId(), [Point3d(getId(), 30, 3.5, 0),
                                                 Point3d(getId(), 0, 3.5, 0)]), bottom)
against.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([forward, against])
"""
    result = convert(source)
    lanes = result.model.roads[0].lane_sections[0].lanes
    if len(lanes) > 1 and all(lane.lane_id < 0 for lane in lanes):
        assert "LL2ODR-W508" in codes(result), (
            "a lanelet emitted as a right lane while travelling against s has to say so"
        )
