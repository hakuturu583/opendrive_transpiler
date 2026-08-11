"""Barriers and crosswalks: the map furniture that is not a lane.

Both were previously lost. A `guard_rail` boundary correctly mapped to roadMark
`none` -- it is not a painted line -- and then vanished, and a crosswalk lanelet
became a road crossing the street at right angles with no junction between them.
These tests pin what each becomes instead, and pin that nothing is dropped
without saying so.
"""

from __future__ import annotations

from opendrive_transpiler import TranspileOptions, transpile_source
from opendrive_transpiler.mapping.tables import barrier_for

PRELUDE = "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"

STREET = """
gl = LineString3d(getId(), [Point3d(getId(), 0.0, 2.0, 0.0),
                            Point3d(getId(), 40.0, 2.0, 0.0)])
gr = LineString3d(getId(), [Point3d(getId(), 0.0, -2.0, 0.0),
                            Point3d(getId(), 20.0, -2.0, 0.0),
                            Point3d(getId(), 40.0, -2.0, 0.0)])
street = Lanelet(getId(), gl, gr)
street.attributes['subtype'] = 'road'
"""

CROSSING = """
crossing = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 8.0, 5.0, 0.0),
                           Point3d(getId(), 8.0, -5.0, 0.0)]),
    LineString3d(getId(), [Point3d(getId(), 12.0, 5.0, 0.0),
                           Point3d(getId(), 12.0, -5.0, 0.0)]))
crossing.attributes['subtype'] = 'crosswalk'
"""


def convert(body: str, **kwargs):
    options = TranspileOptions(strict=False, name="furniture", **kwargs)
    return transpile_source(PRELUDE + body, "furniture.py", options=options)


def objects(result, source: str) -> list:
    return [o for road in result.model.roads for o in road.objects if o.source == source]


def codes(result) -> set[str]:
    return {d.code for d in result.diagnostics}


# --------------------------------------------------------------------------
# Barriers
# --------------------------------------------------------------------------


def test_the_barrier_table_covers_the_physical_boundary_types():
    assert barrier_for({"type": "guard_rail"}) == "barrier"
    assert barrier_for({"type": "fence"}) == "railing"
    assert barrier_for({"type": "wall"}) == "barrier"


def test_a_painted_line_is_not_a_barrier():
    assert barrier_for({"type": "line_thin", "subtype": "solid"}) is None
    assert barrier_for({"type": "curbstone", "subtype": "high"}) is None
    assert barrier_for({}) is None


def test_a_guard_rail_boundary_becomes_a_barrier_object():
    result = convert(STREET + "gr.attributes['type'] = 'guard_rail'\n")
    barriers = objects(result, "barrier")
    assert len(barriers) == 1
    barrier = barriers[0]
    assert barrier.type == "barrier"
    # One corner per boundary point, and the outline is *not* closed: a rail is a
    # polyline, and closing it would draw a return leg that is not in the input.
    assert len(barrier.corners) == 3
    assert barrier.closed is False


def test_a_barrier_keeps_its_roadmark_of_none():
    """It is a physical thing, not a marking; both facts are true at once."""
    result = convert(STREET + "gr.attributes['type'] = 'guard_rail'\n")
    marks = [
        lane.road_mark
        for road in result.model.roads
        for section in road.lane_sections
        for lane in section.lanes
    ]
    assert any(mark.type == "none" and mark.source == "guard_rail" for mark in marks)


def test_the_barrier_height_is_a_stated_convention():
    """lanelet2 carries no height, so the number has to come from options."""
    result = convert(STREET + "gr.attributes['type'] = 'guard_rail'\n", barrier_height=1.25)
    assert objects(result, "barrier")[0].height == 1.25


def test_a_fence_is_a_railing_and_a_wall_is_a_barrier():
    for tag, expected in (("fence", "railing"), ("wall", "barrier")):
        result = convert(STREET + f"gr.attributes['type'] = {tag!r}\n")
        assert objects(result, "barrier")[0].type == expected


def test_a_shared_barrier_boundary_is_emitted_once():
    """Two lanes of one road share their middle boundary; the rail is still one."""
    result = convert("""
mid = LineString3d(getId(), [Point3d(getId(), 0.0, 0.0, 0.0),
                             Point3d(getId(), 40.0, 0.0, 0.0)])
mid.attributes['type'] = 'wall'
a = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 3.0, 0.0),
                           Point3d(getId(), 40.0, 3.0, 0.0)]), mid)
a.attributes['subtype'] = 'road'
b = Lanelet(getId(), mid,
    LineString3d(getId(), [Point3d(getId(), 0.0, -3.0, 0.0),
                           Point3d(getId(), 40.0, -3.0, 0.0)]))
b.attributes['subtype'] = 'road'
""")
    assert len(result.model.roads) == 1, "the two lanes should be one road"
    assert len(objects(result, "barrier")) == 1


def test_disabling_objects_disables_barriers():
    result = convert(STREET + "gr.attributes['type'] = 'guard_rail'\n", objects=False)
    assert objects(result, "barrier") == []


# --------------------------------------------------------------------------
# Crosswalks
# --------------------------------------------------------------------------


def test_a_crosswalk_is_an_object_and_not_a_road():
    result = convert(STREET + CROSSING)
    assert len(result.model.roads) == 1, "the crossing must not become a road of its own"

    crossings = objects(result, "Crosswalk")
    assert len(crossings) == 1
    assert crossings[0].type == "crosswalk"
    # Four bound points make a closed ring: down one side and back up the other.
    assert len(crossings[0].corners) == 4
    assert crossings[0].closed is True


def test_the_crosswalk_lands_where_it_crosses():
    """s along the street, not at its start."""
    result = convert(STREET + CROSSING)
    crossing = objects(result, "Crosswalk")[0]
    assert 7.0 < crossing.s < 9.0


def test_a_converted_crosswalk_counts_as_converted():
    """It became an object, so calling it skipped would be a lie."""
    result = convert(STREET + CROSSING)
    assert result.stats.lanelets_in == 2
    assert result.stats.lanelets_converted == 2
    assert result.stats.lanelets_skipped == 0


def test_the_change_of_shape_is_reported():
    result = convert(STREET + CROSSING)
    notes = [d for d in result.diagnostics if d.code == "LL2ODR-I910"]
    assert len(notes) == 1
    assert "routable" in notes[0].message


def test_a_walkway_is_still_a_road():
    """It runs alongside a road rather than across one, so it is a path."""
    result = convert(
        STREET
        + """
path = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 6.0, 0.0),
                           Point3d(getId(), 40.0, 6.0, 0.0)]),
    LineString3d(getId(), [Point3d(getId(), 0.0, 4.0, 0.0),
                           Point3d(getId(), 40.0, 4.0, 0.0)]))
path.attributes['subtype'] = 'walkway'
"""
    )
    assert len(result.model.roads) == 2
    assert "LL2ODR-I910" not in codes(result)


def test_a_map_of_only_crosswalks_says_there_is_no_road_to_put_them_on():
    result = convert(CROSSING)
    assert result.model.roads == []
    assert "LL2ODR-I910" in codes(result)


def test_disabling_objects_reports_the_crosswalk_as_unconverted():
    """Turning objects off must not turn a crosswalk into silence."""
    result = convert(STREET + CROSSING, objects=False)
    assert objects(result, "Crosswalk") == []
    assert "LL2ODR-I910" in codes(result)
    assert result.stats.lanelets_skipped == 1


def test_a_malformed_crosswalk_is_still_checked():
    """Partitioning crosswalks out of road building must not skip their geometry.

    The outline is built from these same bounds, so a crosswalk whose bounds
    disagree about where it ends becomes a malformed `<object>` -- which has to be
    reported, even though the lanelet is no longer a road.
    """
    result = convert(
        STREET
        + """
bad = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 8.0, 5.0, 0.0),
                           Point3d(getId(), 8.0, -5.0, 0.0)]),
    LineString3d(getId(), [Point3d(getId(), 12.0, 5.0, 0.0),
                           Point3d(getId(), 40.0, -60.0, 0.0)]))
bad.attributes['subtype'] = 'crosswalk'
"""
    )
    assert "LL2ODR-W503" in codes(result)
