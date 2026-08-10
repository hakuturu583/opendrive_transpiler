"""The exactness invariant, checked by reading the .xodr back.

This is the strongest claim the project makes: with `--fit=line`, the emitted
planView reproduces the input polyline *exactly*, not approximately. Because
each `<line>` record carries an absolute start pose, reconstructing the reference
line from the file and comparing it against the lanelet2 boundary the script
built should agree to machine epsilon -- not to some tolerance.

If this test ever needs a loosened tolerance, the README's "positionally exact"
wording has to change with it.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import pytest

from opendrive_transpiler import TranspileOptions, transpile
from opendrive_transpiler.config import TranspileOptions as Options
from opendrive_transpiler.diagnostics import DiagnosticBag
from opendrive_transpiler.frontend.interp import execute
from opendrive_transpiler.frontend.loader import parse_source, read_source
from opendrive_transpiler.ir.model import build_ir
from opendrive_transpiler.runner import run_generated

from ..conftest import needs_emit

pytestmark = needs_emit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

EXACT = ["minimal", "chain", "parallel_lanes", "curved_road", "branch", "two_way"]


def reconstruct(road: ET.Element) -> list[tuple[float, float]]:
    """Rebuild a road's reference line from its `<line>` records."""
    points: list[tuple[float, float]] = []
    for geometry in road.findall("planView/geometry"):
        x = float(geometry.get("x"))
        y = float(geometry.get("y"))
        hdg = float(geometry.get("hdg"))
        length = float(geometry.get("length"))
        assert geometry.find("line") is not None, "only <line> geometry is emitted"
        points.append((x, y))
        points.append((x + length * math.cos(hdg), y + length * math.sin(hdg)))
    # Drop the duplicated joints.
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or math.dist(deduped[-1], point) > 1e-12:
            deduped.append(point)
    return deduped


def input_bounds(path: Path) -> list[list[tuple[float, float]]]:
    """Every boundary polyline the input script built, in xy."""
    options = Options(strict=False)
    bag = DiagnosticBag(strict=False)
    text = read_source(path, bag)
    module = parse_source(text, str(path), bag)
    registry = execute(module, str(path), bag, options)
    ir = build_ir(registry, bag, options)
    out: list[list[tuple[float, float]]] = []
    for lanelet in ir.lanelets:
        for bound in (lanelet.left, lanelet.right):
            out.append([(p.x, p.y) for p in bound.points])
    return out


def distance_to_polyline(point: tuple[float, float], polyline: list[tuple[float, float]]) -> float:
    best = math.inf
    for a, b in pairwise(polyline):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        if span == 0.0:
            best = min(best, math.dist(point, a))
            continue
        t = ((point[0] - ax) * dx + (point[1] - ay) * dy) / span
        t = max(0.0, min(1.0, t))
        best = min(best, math.dist(point, (ax + dx * t, ay + dy * t)))
    return best


@pytest.mark.parametrize("name", EXACT)
def test_reference_lines_reproduce_an_input_boundary_exactly(name: str, tmp_path: Path):
    path = FIXTURES / f"{name}.py"
    result = transpile(path, options=TranspileOptions(strict=False, name=name))
    assert result.ok

    out = run_generated(result.code, tmp_path / f"{name}.xodr")
    root = ET.parse(out).getroot()
    bounds = input_bounds(path)

    for road in root.findall("road"):
        reconstructed = reconstruct(road)
        assert len(reconstructed) >= 2
        for point in reconstructed:
            # Every reconstructed vertex must lie on *some* input boundary.
            deviation = min(distance_to_polyline(point, b) for b in bounds if len(b) >= 2)
            assert deviation < 1e-9, (
                f"{name}: road {road.get('id')} vertex {point} is {deviation:.3g} m "
                "off every input boundary"
            )


@pytest.mark.parametrize("name", EXACT)
def test_road_length_matches_the_source_polyline(name: str, tmp_path: Path):
    path = FIXTURES / f"{name}.py"
    result = transpile(path, options=TranspileOptions(strict=False, name=name))
    out = run_generated(result.code, tmp_path / f"{name}.xodr")
    root = ET.parse(out).getroot()

    for road in root.findall("road"):
        reconstructed = reconstruct(road)
        walked = sum(math.dist(a, b) for a, b in pairwise(reconstructed))
        assert math.isclose(walked, float(road.get("length")), rel_tol=1e-9, abs_tol=1e-9)


def test_pyxodr_can_read_what_we_write(tmp_path: Path):
    """pyxodr is a reader, which is exactly the role it is useful in here."""
    pyxodr = pytest.importorskip("pyxodr.road_objects.network")

    path = FIXTURES / "chain.py"
    result = transpile(path, options=TranspileOptions(strict=False, name="chain"))
    out = run_generated(result.code, tmp_path / "chain.xodr")

    network = pyxodr.RoadNetwork(str(out))
    roads = network.get_roads()
    assert len(roads) == 1
    assert len(roads[0].reference_line) >= 2
