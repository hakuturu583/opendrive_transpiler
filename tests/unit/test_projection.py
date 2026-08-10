"""Earth-centred maps, and the tangent plane they are rotated onto.

Every other lanelet2 projector hands over a planar metre frame that passes
straight through. `GeocentricProjector` does not: its coordinates are
earth-centred XYZ, so the mapping stage has to move the geometry before anything
measures it. These tests pin that move, and pin that it is *rigid* -- lengths,
adjacency and node identity all have to survive it.
"""

from __future__ import annotations

import math
from itertools import pairwise

from opendrive_transpiler import TranspileOptions, transpile_source
from opendrive_transpiler.config import TranspileOptions as Options
from opendrive_transpiler.diagnostics import DiagnosticBag
from opendrive_transpiler.frontend.interp import execute
from opendrive_transpiler.frontend.loader import parse_source
from opendrive_transpiler.ir.model import build_ir
from opendrive_transpiler.mapping import localise
from opendrive_transpiler.mapping.proj import geodetic_to_ecef

# A flat, roughly 40 m road near Tokyo at a constant altitude, expressed in
# earth-centred metres. The degrees-per-metre figures below are the usual
# approximation, so the road is only about 40 m long -- deliberately, since no
# test here should depend on that number. What matters is the *frame*: coordinates
# in the millions, and a road that is straight and level in reality.
ANCHOR = (35.68, 139.7, 40.0)


def _ecef_road(length: float = 40.0, half_width: float = 2.0) -> tuple[list, list]:
    """Two boundaries a nominal `length` apart along a meridian, in ECEF."""
    latitude, longitude, altitude = ANCHOR
    per_metre_lat = 1.0 / 111320.0
    per_metre_lon = 1.0 / (111320.0 * math.cos(math.radians(latitude)))

    def at(along: float, across: float):
        return geodetic_to_ecef(
            latitude + along * per_metre_lat, longitude + across * per_metre_lon, altitude
        )

    left = [at(s, half_width) for s in (0.0, length / 2, length)]
    right = [at(s, -half_width) for s in (0.0, length / 2, length)]
    return left, right


def _source(*, projector: str = "GeocentricProjector()") -> str:
    left, right = _ecef_road()

    def points(name: str, coords) -> str:
        body = ", ".join(f"Point3d(getId(), {x!r}, {y!r}, {z!r})" for x, y, z in coords)
        return f"{name} = LineString3d(getId(), [{body}])\n"

    return (
        "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
        "from lanelet2.projection import GeocentricProjector, UtmProjector\n"
        "from lanelet2.io import Origin\n"
        f"proj = {projector}\n"
        + points("a", left)
        + points("b", right)
        + "ll = Lanelet(getId(), a, b)\n"
        "ll.attributes['subtype'] = 'road'\n"
    )


def _ir(source: str):
    options = Options(strict=False)
    bag = DiagnosticBag(strict=False)
    module = parse_source(source, "<test>", bag)
    registry = execute(module, "<test>", bag, options)
    return build_ir(registry, bag, options)


def convert(source: str):
    return transpile_source(source, "geo.py", options=TranspileOptions(strict=False, name="geo"))


# --------------------------------------------------------------------------
# The rebasing itself
# --------------------------------------------------------------------------


def test_rebasing_puts_the_map_around_the_origin():
    """Earth-centred coordinates are millions of metres from anywhere useful."""
    ir = _ir(_source())
    before = ir.lanelets[0].left.points[0]
    assert abs(before.x) > 1e6  # earth-centred, as the script wrote it

    moved, _anchor, _geodetic = localise.rebase(ir)
    after = moved.lanelets[0].left.points[0]
    assert max(abs(after.x), abs(after.y), abs(after.z)) < 100.0


def test_rebasing_keeps_node_identity():
    """Topology matches nodes by key, so a coordinate rewrite must not touch it."""
    ir = _ir(_source())
    moved, _anchor, _geodetic = localise.rebase(ir)

    for original, shifted in zip(ir.lanelets, moved.lanelets, strict=True):
        for side in ("left", "right"):
            was = getattr(original, side)
            now = getattr(shifted, side)
            assert now.key == was.key
            assert now.node_keys == was.node_keys
            assert now.reversed_view == was.reversed_view


def test_rebasing_preserves_distances():
    ir = _ir(_source())
    moved, _anchor, _geodetic = localise.rebase(ir)

    def span(lanelet) -> float:
        points = lanelet.left.points
        return math.dist(points[0].xyz, points[-1].xyz)

    assert math.isclose(span(ir.lanelets[0]), span(moved.lanelets[0]), rel_tol=1e-12)


def test_the_anchor_is_the_centroid_of_the_boundary_points():
    ir = _ir(_source())
    anchor = localise.anchor_of(ir)
    points = [p for ll in ir.lanelets for b in (ll.left, ll.right) for p in b.points]
    for axis, value in enumerate(anchor):
        assert math.isclose(value, sum(p.xyz[axis] for p in points) / len(points), rel_tol=1e-12)


def test_a_map_with_no_lanelets_has_no_anchor():
    ir = _ir("from lanelet2.core import Lanelet\n")
    assert localise.anchor_of(ir) is None
    assert localise.rebase(ir) is None


# --------------------------------------------------------------------------
# What comes out the other end
# --------------------------------------------------------------------------


def test_a_geocentric_map_keeps_the_length_its_own_coordinates_state():
    """The reference line must be as long after the rotation as before it.

    Asserting a round number instead would test the fixture's degrees-per-metre
    arithmetic rather than the transform. The input's own chord length is the
    invariant a rigid motion has to preserve.
    """
    source = _source()
    reference = _ir(source).lanelets[0].left.points
    expected = sum(math.dist(a.xyz, b.xyz) for a, b in pairwise(reference))

    result = convert(source)
    assert result.model.roads, "an earth-centred map should still produce a road"
    assert math.isclose(result.model.roads[0].length, expected, rel_tol=1e-9)


def test_a_flat_geocentric_road_stays_flat():
    """Constant altitude in, no elevation profile to speak of out.

    Not exactly zero: a tangent plane departs from the ellipsoid by the sagitta,
    about 30 microns over a road this short, and that is real rather than error.
    """
    road = convert(_source()).model.roads[0]
    elevations = [e.a for e in road.elevations]
    assert max(abs(value) for value in elevations) < 0.001


def test_the_rotation_is_reported_rather_than_done_silently():
    result = convert(_source())
    localised = [d for d in result.diagnostics if d.code == "LL2ODR-I909"]
    assert len(localised) == 1
    assert "tangent plane" in localised[0].message


def test_the_geo_reference_describes_the_plane_the_geometry_now_lives_in():
    result = convert(_source())
    assert "+proj=topocentric" in result.model.geo_reference
    assert "+X_0=" in result.model.geo_reference


def test_the_topocentric_origin_is_the_maps_own_centroid():
    """The projector carries no origin, so the anchor has to come from the data."""
    source = _source()
    anchor = localise.anchor_of(_ir(source))

    reference = convert(source).model.geo_reference
    emitted = tuple(float(reference.split(f"+{axis}_0=")[1].split()[0]) for axis in ("X", "Y", "Z"))
    for got, want in zip(emitted, anchor, strict=True):
        # The round trip through geodetic and back is exact to well under a micron.
        assert math.isclose(got, want, abs_tol=1e-6)


def test_a_planar_projector_is_left_alone():
    """Only geocentric maps move; UTM coordinates are already a plan view."""
    source = _source(projector="UtmProjector(Origin(35.68, 139.7))")
    result = convert(source)
    assert not [d for d in result.diagnostics if d.code == "LL2ODR-I909"]
    assert "+proj=utm" in result.model.geo_reference
