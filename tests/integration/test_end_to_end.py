"""Running the generated scripts and checking the OpenDRIVE that comes out.

These need the `[emit]` extra. Everything above this file works without it --
that separation is the point of the zero-dependency core, and CI proves it by
running the rest of the suite with nothing installed.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from opendrive_transpiler import TranspileOptions, transpile, transpile_to_xodr

from ..conftest import needs_emit

pytestmark = needs_emit


def convert(fixture_path: Path, tmp_path: Path) -> ET.Element:
    out = tmp_path / f"{fixture_path.stem}.xodr"
    transpile_to_xodr(
        fixture_path, out, options=TranspileOptions(strict=False, name=fixture_path.stem)
    )
    return ET.parse(out).getroot()


CONVERTIBLE = [
    "minimal",
    "two_way",
    "chain",
    "parallel_lanes",
    "curved_road",
    "with_projector",
    "branch",
    "merge",
]


@pytest.fixture(params=CONVERTIBLE)
def converted(request, tmp_path: Path) -> tuple[str, ET.Element]:
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    return request.param, convert(fixtures / f"{request.param}.py", tmp_path)


def test_output_is_well_formed_opendrive(converted):
    _name, root = converted
    assert root.tag == "OpenDRIVE"
    assert root.find("header") is not None
    assert root.findall("road")


def test_every_road_has_geometry_and_lanes(converted):
    _name, root = converted
    for road in root.findall("road"):
        assert road.findall("planView/geometry")
        assert road.findall("lanes/laneSection")


def test_geometry_lengths_sum_to_the_road_length(converted):
    """A road whose records do not tile its length is malformed."""
    _name, root = converted
    for road in root.findall("road"):
        total = sum(float(g.get("length")) for g in road.findall("planView/geometry"))
        assert math.isclose(total, float(road.get("length")), rel_tol=1e-9, abs_tol=1e-9)


def test_geometry_records_are_contiguous_in_s(converted):
    _name, root = converted
    for road in root.findall("road"):
        s = 0.0
        for geometry in road.findall("planView/geometry"):
            assert math.isclose(float(geometry.get("s")), s, abs_tol=1e-9)
            s += float(geometry.get("length"))


def test_lane_ids_descend_from_the_reference_line(converted):
    _name, root = converted
    for section in root.findall("road/lanes/laneSection"):
        ids = [int(lane.get("id")) for lane in section.findall("right/lane")]
        assert ids == list(range(-1, -len(ids) - 1, -1))


def test_every_lane_width_is_positive(converted):
    _name, root = converted
    for lane in root.findall("road/lanes/laneSection/right/lane"):
        for width in lane.findall("width"):
            assert float(width.get("a")) > 0.0


def test_the_first_width_record_starts_at_the_section_origin(converted):
    _name, root = converted
    for lane in root.findall("road/lanes/laneSection/right/lane"):
        widths = lane.findall("width")
        assert widths
        assert math.isclose(float(widths[0].get("sOffset")), 0.0, abs_tol=1e-12)


def test_lane_sections_ascend_in_s(converted):
    _name, root = converted
    for road in root.findall("road"):
        offsets = [float(s.get("s")) for s in road.findall("lanes/laneSection")]
        assert offsets == sorted(offsets)
        assert offsets[0] == 0.0


# --------------------------------------------------------------------------
# Specific expectations per fixture
# --------------------------------------------------------------------------


def test_a_chain_becomes_one_road_with_four_sections(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "chain.py", tmp_path)
    roads = root.findall("road")
    assert len(roads) == 1
    assert len(roads[0].findall("lanes/laneSection")) == 4
    assert math.isclose(float(roads[0].get("length")), 40.0, abs_tol=1e-9)


def test_parallel_lanes_become_one_road_with_two_lanes_per_section(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "parallel_lanes.py", tmp_path)
    roads = root.findall("road")
    assert len(roads) == 1
    sections = roads[0].findall("lanes/laneSection")
    assert len(sections) == 2
    for section in sections:
        assert len(section.findall("right/lane")) == 2


def test_speed_and_road_type_reach_the_output(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "parallel_lanes.py", tmp_path)
    road_type = root.find("road/type")
    assert road_type.get("type") == "motorway"
    speed = road_type.find("speed")
    assert speed.get("unit") == "km/h"
    assert math.isclose(float(speed.get("max")), 100.0)


def test_road_marks_reflect_the_boundary_tags(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "parallel_lanes.py", tmp_path)
    section = root.find("road/lanes/laneSection")
    lanes = section.findall("right/lane")
    # The shared middle boundary is dashed; the outer one is solid.
    assert lanes[0].find("roadMark").get("type") == "broken"
    assert lanes[1].find("roadMark").get("type") == "solid"


def test_the_projector_origin_reaches_the_header(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "with_projector.py", tmp_path)
    geo = root.find("header/geoReference")
    assert geo is not None
    # Zone 32's central meridian, spelled out: `+proj=utm` would ignore the
    # origin shift this projector exists to carry.
    assert "+lon_0=9.0" in geo.text
    assert "+proj=utm" not in geo.text


def test_elevation_is_carried_through(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "curved_road.py", tmp_path)
    elevations = root.findall("road/elevationProfile/elevation")
    assert len(elevations) > 1
    assert max(float(e.get("a")) for e in elevations) > 5.0


def test_a_two_way_road_puts_the_directions_on_opposite_sides(tmp_path: Path):
    """One road, one lane each way, straddling a reference line on the centre."""
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "two_way.py", tmp_path)

    roads = root.findall("road")
    assert len(roads) == 1, "a two-way road is one road, not two"

    section = roads[0].find("lanes/laneSection")
    left = [int(lane.get("id")) for lane in section.findall("left/lane")]
    right = [int(lane.get("id")) for lane in section.findall("right/lane")]
    assert left == [1]
    assert right == [-1]

    # Both carriageways are the same width, and the reference line sits between
    # them -- on the shared centre, at y = 0.
    widths = [
        float(width.get("a"))
        for lane in section.findall("left/lane") + section.findall("right/lane")
        for width in lane.findall("width")
    ]
    assert all(math.isclose(w, 3.5, abs_tol=1e-9) for w in widths)

    geometry = roads[0].find("planView/geometry")
    assert math.isclose(float(geometry.get("y")), 0.0, abs_tol=1e-9)


def test_a_bidirectional_lanelet_gets_the_bidirectional_lane_type(tmp_path: Path):
    """`one_way=no` is a lane type in OpenDRIVE, not a modifier on driving."""
    from opendrive_transpiler import transpile_source

    source = (
        "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
        "a = LineString3d(getId(), [Point3d(getId(), 0.0, 2.0, 0.0), "
        "Point3d(getId(), 40.0, 2.0, 0.0)])\n"
        "b = LineString3d(getId(), [Point3d(getId(), 0.0, -2.0, 0.0), "
        "Point3d(getId(), 40.0, -2.0, 0.0)])\n"
        "ll = Lanelet(getId(), a, b)\n"
        "ll.attributes['subtype'] = 'road'\n"
        "ll.attributes['one_way'] = 'no'\n"
    )
    result = transpile_source(source, "bidi.py", options=TranspileOptions(strict=False))
    assert "xodr.LaneType.bidirectional" in result.code
    assert any(d.code == "LL2ODR-I907" for d in result.diagnostics)


def test_a_script_that_loads_from_disk_is_refused(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    result = transpile(fixtures / "loads_from_file.py", options=TranspileOptions(strict=False))
    assert not result.ok
    assert any(d.code == "LL2ODR-E402" for d in result.errors)


def test_a_right_of_way_becomes_a_junction_priority(tmp_path: Path):
    """`<priority>` names connecting roads, and follows the connections in order."""
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "right_of_way.py", tmp_path)

    junction = root.find("junction")
    assert junction is not None

    priorities = junction.findall("priority")
    assert len(priorities) == 1
    high = priorities[0].get("high")
    low = priorities[0].get("low")

    connecting = {c.get("connectingRoad") for c in junction.findall("connection")}
    assert {high, low} <= connecting, "priority must name connecting roads"
    assert high != low

    # The schema orders <junction> as connection*, then priority*.
    tags = [child.tag for child in junction]
    assert tags == sorted(tags, key=["connection", "priority"].index)


def test_a_geocentric_map_is_georeferenced_to_its_tangent_plane(tmp_path: Path):
    """Earth-centred input still produces a usable, correctly labelled plan view."""
    from opendrive_transpiler.mapping.proj import geodetic_to_ecef

    left = [geodetic_to_ecef(35.68 + i * 1e-4, 139.7, 40.0) for i in range(3)]
    right = [geodetic_to_ecef(35.68 + i * 1e-4, 139.70004, 40.0) for i in range(3)]

    def points(name, coords):
        body = ", ".join(f"Point3d(getId(), {x!r}, {y!r}, {z!r})" for x, y, z in coords)
        return f"{name} = LineString3d(getId(), [{body}])\n"

    source = (
        "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
        "from lanelet2.projection import GeocentricProjector\n"
        "proj = GeocentricProjector()\n"
        + points("a", left)
        + points("b", right)
        + "ll = Lanelet(getId(), a, b)\n"
        "ll.attributes['subtype'] = 'road'\n"
    )

    script = tmp_path / "geocentric.py"
    script.write_text(source)
    root = convert(script, tmp_path)

    geo = root.find("header/geoReference")
    assert geo is not None
    assert "+proj=topocentric" in geo.text

    # The plan view is local metres, not the millions the script wrote.
    geometry = root.find("road/planView/geometry")
    assert abs(float(geometry.get("x"))) < 1000.0
    assert abs(float(geometry.get("y"))) < 1000.0


def test_street_furniture_reaches_the_xodr(tmp_path: Path):
    """A guard rail and a crosswalk, as the two shapes of `<object>`."""
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "street_furniture.py", tmp_path)

    # The crossing must not have become a road of its own.
    assert len(root.findall("road")) == 1

    objects = {obj.get("type"): obj for obj in root.findall("road/objects/object")}
    assert set(objects) == {"barrier", "crosswalk"}

    # A rail is a polyline: its outline stays open and has one corner per point.
    rail = objects["barrier"]
    rail_outline = rail.find("outlines/outline")
    assert rail_outline.get("closed") == "false"
    assert len(rail_outline.findall("cornerRoad")) == 3
    assert float(rail.get("height")) > 0.0

    # A crosswalk is a footprint: closed, and sitting along the road it crosses.
    crossing = objects["crosswalk"]
    crossing_outline = crossing.find("outlines/outline")
    assert crossing_outline.get("closed") == "true"
    assert len(crossing_outline.findall("cornerRoad")) == 4
    assert 0.0 < float(crossing.get("s")) < float(root.find("road").get("length"))


def test_a_merge_links_lanes_across_roads_of_different_widths(tmp_path: Path):
    """The case the backend's own lane linking refuses outright.

    `create_lane_links` raises `NotSameAmountOfLanesError` when two connected roads
    carry different lane counts, so these links are written down from the lanelet
    topology instead. The junction's `<laneLink>` records disambiguate which
    incoming lane feeds which outgoing one.
    """
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "merge.py", tmp_path)

    roads = {road.get("id"): road for road in root.findall("road")}
    assert len(roads) == 3

    # The two approaches are one lane each; what they merge into carries two.
    widths = {rid: len(r.findall("lanes/laneSection/right/lane")) for rid, r in roads.items()}
    assert sorted(widths.values()) == [1, 1, 2]

    # A road carries exactly one <predecessor>, so road-level links cannot state a
    # merge at all -- one approach would be left out. The junction carries it, and
    # the merged road is its connecting road.
    merged = next(r for rid, r in roads.items() if widths[rid] == 2)
    junction = root.find("junction")
    assert junction is not None
    assert merged.get("junction") == junction.get("id")
    assert merged.find("link/predecessor") is None, (
        "a connecting road takes its incoming ends from the junction, not a road link"
    )

    # Both approaches point at the junction rather than at each other.
    approaches = {rid: r for rid, r in roads.items() if widths[rid] == 1}
    for road in approaches.values():
        successor = road.find("link/successor")
        assert successor is not None
        assert successor.get("elementType") == "junction"
        assert successor.get("elementId") == junction.get("id")

    # The junction is what says which approach feeds which lane, unambiguously,
    # and it carries *both* correspondences.
    links = {
        (c.get("incomingRoad"), link.get("from"), link.get("to"))
        for c in root.findall("junction/connection")
        for link in c.findall("laneLink")
    }
    outgoing = {to for _incoming, _frm, to in links}
    assert len(outgoing) == 2, "each approach must feed a different outgoing lane"
