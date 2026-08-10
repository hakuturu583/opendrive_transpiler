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
    "chain",
    "parallel_lanes",
    "curved_road",
    "with_projector",
    "branch",
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
    assert "+zone=32" in geo.text


def test_elevation_is_carried_through(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    root = convert(fixtures / "curved_road.py", tmp_path)
    elevations = root.findall("road/elevationProfile/elevation")
    assert len(elevations) > 1
    assert max(float(e.get("a")) for e in elevations) > 5.0


def test_a_script_that_loads_from_disk_is_refused(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    result = transpile(fixtures / "loads_from_file.py", options=TranspileOptions(strict=False))
    assert not result.ok
    assert any(d.code == "LL2ODR-E402" for d in result.errors)
