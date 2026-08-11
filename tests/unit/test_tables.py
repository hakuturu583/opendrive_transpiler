"""Attribute mapping tables.

Pure lookups, so exhaustive coverage is cheap -- and worth having, because these
tables are the whole of the semantic conversion. Every subtype listed in
simple_lanelet2's own traffic-rules test corpus appears here.
"""

from __future__ import annotations

import math

import pytest

from opendrive_transpiler.config import TranspileOptions
from opendrive_transpiler.ir.model import ProjectionIR
from opendrive_transpiler.mapping.proj import (
    central_meridian,
    ecef_to_enu,
    ecef_to_geodetic,
    enu_basis,
    geodetic_to_ecef,
    utm_forward,
)
from opendrive_transpiler.mapping.tables import (
    geo_reference_for,
    lane_type_for,
    road_mark_for,
    road_type_for,
    speed_for,
    utm_zone,
)

OPTIONS = TranspileOptions()

# The full subtype vocabulary from simple_lanelet2 tests/cases/0500_traffic_rules.py.
CORPUS_SUBTYPES = [
    "",
    "road",
    "highway",
    "play_street",
    "emergency_lane",
    "exit",
    "walkway",
    "crosswalk",
    "stairs",
    "shared_walkway",
    "bicycle_lane",
    "bus_lane",
    "parking",
    "freespace",
    "traffic_island",
    "pedestrian_lane",
    "lane",
    "rail",
]

# The full boundary vocabulary from the same file.
CORPUS_BOUNDARIES = [
    ("line_thin", "dashed"),
    ("line_thin", "solid"),
    ("line_thin", "dashed_solid"),
    ("line_thin", "solid_dashed"),
    ("line_thin", "solid_solid"),
    ("line_thick", "dashed"),
    ("line_thick", "solid"),
    ("curbstone", "low"),
    ("curbstone", "high"),
    ("virtual", ""),
    ("road_border", ""),
    ("guard_rail", ""),
    ("pedestrian_marking", ""),
    ("zebra_marking", ""),
    ("bike_marking", ""),
    ("keepout", ""),
    ("zig-zag", ""),
    ("symbol", ""),
    ("rail", ""),
    ("stop_line", ""),
]


@pytest.mark.parametrize("subtype", CORPUS_SUBTYPES)
def test_every_corpus_subtype_is_recognised(subtype: str):
    lane_type, recognised = lane_type_for(subtype)
    assert recognised, f"{subtype!r} is in the lanelet2 vocabulary but has no mapping"
    assert lane_type


@pytest.mark.parametrize(
    "subtype,expected",
    [
        ("road", "driving"),
        ("highway", "driving"),
        ("exit", "exit"),
        ("walkway", "sidewalk"),
        ("crosswalk", "sidewalk"),
        ("bicycle_lane", "biking"),
        ("bus_lane", "bus"),
        ("parking", "parking"),
        ("freespace", "restricted"),
        ("traffic_island", "median"),
        ("stairs", "none"),
        ("rail", "rail"),
    ],
)
def test_lane_types(subtype: str, expected: str):
    assert lane_type_for(subtype)[0] == expected


def test_an_unknown_subtype_falls_back_and_says_so():
    lane_type, recognised = lane_type_for("teleporter")
    assert lane_type == "driving"
    assert not recognised


def test_subtype_matching_ignores_case_and_padding():
    assert lane_type_for("  RoAd ")[0] == "driving"


@pytest.mark.parametrize(
    "subtype,location,expected",
    [
        ("highway", "", "motorway"),
        ("road", "urban", "town"),
        ("road", "nonurban", "rural"),
        ("play_street", "urban", "townPlayStreet"),
        ("walkway", "urban", "pedestrian"),
        ("bicycle_lane", "urban", "bicycle"),
        ("road", "", "unknown"),
    ],
)
def test_road_types(subtype: str, location: str, expected: str):
    assert road_type_for(subtype, location) == expected


@pytest.mark.parametrize("kind,subtype", CORPUS_BOUNDARIES)
def test_every_corpus_boundary_is_recognised(kind: str, subtype: str):
    mark, recognised = road_mark_for({"type": kind, "subtype": subtype}, OPTIONS)
    assert recognised, f"({kind!r}, {subtype!r}) is in the vocabulary but has no mapping"
    assert mark.type


@pytest.mark.parametrize(
    "kind,subtype,expected",
    [
        ("line_thin", "solid", "solid"),
        ("line_thin", "dashed", "broken"),
        ("line_thin", "solid_solid", "solid_solid"),
        ("line_thin", "dashed_solid", "broken_solid"),
        ("line_thin", "solid_dashed", "solid_broken"),
        ("line_thick", "solid", "solid"),
        ("curbstone", "high", "curb"),
        ("road_border", "", "edge"),
        ("virtual", "", "none"),
        ("guard_rail", "", "none"),
        ("pedestrian_marking", "", "none"),
        ("zebra_marking", "", "none"),
        ("keepout", "", "none"),
        ("zig-zag", "", "none"),
        ("rail", "", "none"),
    ],
)
def test_road_mark_types(kind: str, subtype: str, expected: str):
    assert road_mark_for({"type": kind, "subtype": subtype}, OPTIONS)[0].type == expected


def test_thick_lines_are_wider_and_bold():
    thin, _ = road_mark_for({"type": "line_thin", "subtype": "solid"}, OPTIONS)
    thick, _ = road_mark_for({"type": "line_thick", "subtype": "solid"}, OPTIONS)
    assert thick.width > thin.width
    assert thick.weight == "bold"
    assert thin.weight == "standard"


def test_dashed_marks_carry_a_dash_pattern():
    mark, _ = road_mark_for({"type": "line_thin", "subtype": "dashed"}, OPTIONS)
    assert mark.length == OPTIONS.dash_length
    assert mark.space == OPTIONS.dash_space


def test_solid_marks_carry_no_dash_pattern():
    mark, _ = road_mark_for({"type": "line_thin", "subtype": "solid"}, OPTIONS)
    assert mark.length is None and mark.space is None


def test_mark_colour_is_carried_when_known():
    mark, _ = road_mark_for({"type": "line_thin", "subtype": "solid", "color": "yellow"}, OPTIONS)
    assert mark.color == "yellow"


def test_an_unknown_colour_falls_back_to_standard():
    mark, _ = road_mark_for(
        {"type": "line_thin", "subtype": "solid", "color": "chartreuse"}, OPTIONS
    )
    assert mark.color == "standard"


def test_an_untagged_boundary_is_reported_as_unrecognised():
    mark, recognised = road_mark_for({}, OPTIONS)
    assert mark.type == "none"
    assert recognised  # an absent "type" maps to no marking, which is a real answer


def test_an_unknown_thin_subtype_defaults_to_solid_and_says_so():
    mark, recognised = road_mark_for({"type": "line_thin", "subtype": "zigzag"}, OPTIONS)
    assert mark.type == "solid"
    assert not recognised


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30 km/h", (30.0, "km/h")),
        ("30km/h", (30.0, "km/h")),
        ("30kmh", (30.0, "km/h")),
        ("30 kph", (30.0, "km/h")),
        ("50", (50.0, "km/h")),
        ("25 mph", (25.0, "mph")),
        ("13.9 m/s", (13.9, "m/s")),
        ("13.9ms", (13.9, "m/s")),
    ],
)
def test_speed_parsing(raw: str, expected: tuple[float, str]):
    assert speed_for(raw) == expected


def test_speed_units_are_restricted_to_what_opendrive_accepts():
    for raw in ("30 km/h", "50", "25 mph", "13.9 m/s"):
        assert speed_for(raw)[1] in {"m/s", "mph", "km/h"}


@pytest.mark.parametrize("raw", ["", "fast", "30 furlongs/fortnight", "-10", "km/h"])
def test_unparseable_speeds_return_none(raw: str):
    assert speed_for(raw) is None


@pytest.mark.parametrize(
    "longitude,zone",
    [(-180.0, 1), (-177.0, 1), (8.4, 32), (0.0, 31), (139.7, 54), (179.9, 60)],
)
def test_utm_zones(longitude: float, zone: int):
    assert utm_zone(longitude) == zone


def test_utm_geo_reference_names_the_zone_and_hemisphere():
    proj, caveat = geo_reference_for(
        ProjectionIR("utm", lat=49.0, lon=8.4, alt=0.0, use_offset=True)
    )
    # Written as the tmerc that +proj=utm expands to -- see
    # test_a_shifted_zone_is_never_written_as_proj_utm for why it has to be.
    assert "+proj=tmerc" in proj
    assert f"+lon_0={central_meridian(32)!r}" in proj
    assert caveat is None


def test_a_shifted_zone_is_never_written_as_proj_utm():
    """`+proj=utm` hardcodes its false easting and ignores any override.

    A string combining the two parses, reads correctly, and silently places the
    map on the equator -- `(0, 0)` in `+proj=utm +zone=32 +x_0=43885
    +y_0=-5427629` comes back as 0.000000N 4.511256E rather than the origin it
    names. Every shifted frame must therefore spell out the tmerc expansion.
    """
    shifted = [
        ProjectionIR("utm", lat=49.0, lon=8.4, alt=0.0, use_offset=True),
        ProjectionIR("utm", lat=-33.9, lon=151.2, alt=0.0, use_offset=True),
        ProjectionIR("mgrs", lat=49.0, lon=8.4, alt=0.0, use_offset=False),
        ProjectionIR("mgrs", lat=0.0, lon=0.0, alt=0.0, use_offset=False, mgrs_code="54SUE"),
    ]
    for projection in shifted:
        proj, _caveat = geo_reference_for(projection)
        assert "+x_0=" in proj, "these frames are all shifted"
        assert "+proj=utm" not in proj, f"+proj=utm would ignore the shift: {proj}"


def test_utm_with_offset_carries_the_origin_shift():
    """lanelet2 subtracts the origin's easting/northing; PROJ must say so."""
    proj, _ = geo_reference_for(ProjectionIR("utm", lat=49.0, lon=8.4, alt=0.0, use_offset=True))
    assert "+x_0=" in proj and "+y_0=" in proj
    x0 = float(proj.split("+x_0=")[1].split()[0])
    y0 = float(proj.split("+y_0=")[1].split()[0])
    easting, northing = utm_forward(49.0, 8.4)
    assert math.isclose(x0, 500000.0 - easting, abs_tol=1e-6)
    assert math.isclose(y0, -northing, abs_tol=1e-6)


def test_utm_without_offset_is_plain_utm():
    proj, caveat = geo_reference_for(
        ProjectionIR("utm", lat=49.0, lon=8.4, alt=0.0, use_offset=False)
    )
    assert caveat is None
    assert "+x_0=" not in proj


@pytest.mark.parametrize(
    "lat,lon,easting,northing",
    [
        # Reference values cross-checked against pyproj (EPSG:326xx / 327xx).
        (49.0, 8.4, 456114.596, 5427629.204),
        (35.68, 139.76, 387789.174, 3949165.002),
        (-33.9, 151.2, 333568.941, 6247473.337),
    ],
)
def test_utm_forward_matches_a_reference_projection(
    lat: float, lon: float, easting: float, northing: float
):
    got_e, got_n = utm_forward(lat, lon)
    assert math.isclose(got_e, easting, abs_tol=1e-3)
    assert math.isclose(got_n, northing, abs_tol=1e-3)


def test_mgrs_is_reproduced_as_an_offset_utm_zone():
    proj, caveat = geo_reference_for(
        ProjectionIR("mgrs", lat=49.0, lon=8.4, alt=0.0, use_offset=False)
    )
    assert "+proj=tmerc" in proj and "+x_0=" in proj
    # The offsets land on a 100 km square corner.
    assert float(proj.split("+y_0=")[1].split()[0]) % 100000.0 == 0.0
    assert caveat is not None  # the grid letters are not carried


def test_geocentric_names_the_tangent_plane_it_was_rotated_onto():
    """By this stage the map is in east/north/up, so that is what the header says.

    PROJ's topocentric takes its origin in earth-centred metres, which describes
    the tangent plane exactly rather than standing in for it with a conformal
    projection.
    """
    proj, caveat = geo_reference_for(
        ProjectionIR("geocentric", lat=35.68, lon=139.7, alt=40.0, use_offset=False)
    )
    assert caveat is None
    assert "+proj=topocentric" in proj
    x0 = float(proj.split("+X_0=")[1].split()[0])
    y0 = float(proj.split("+Y_0=")[1].split()[0])
    z0 = float(proj.split("+Z_0=")[1].split()[0])
    assert (x0, y0, z0) == geodetic_to_ecef(35.68, 139.7, 40.0)


def test_southern_hemisphere_utm():
    proj, _ = geo_reference_for(
        ProjectionIR("utm", lat=-33.9, lon=151.2, alt=0.0, use_offset=False)
    )
    assert "+south" in proj


def test_no_projector_means_no_geo_reference():
    assert geo_reference_for(None) == (None, None)


# --------------------------------------------------------------------------
# Earth-centred coordinates
# --------------------------------------------------------------------------
# Reference values cross-checked against pyproj (+proj=geocent +datum=WGS84),
# embedded rather than imported so the suite stays dependency-free -- the same
# arrangement the UTM cases above use.

ECEF_CASES = [
    ((35.68, 139.7, 40.0), (-3955823.9633827023, 3354782.6878813645, 3699431.973876118)),
    ((0.0, 0.0, 0.0), (6378137.0, 0.0, 0.0)),
    ((-33.87, 151.2, 5.0), (-4645575.419946929, 2553926.6992282085, -3534485.692082713)),
    ((90.0, 0.0, 0.0), (0.0, 0.0, 6356752.314245179)),
]


@pytest.mark.parametrize("geodetic,ecef", ECEF_CASES)
def test_geodetic_to_ecef_matches_proj(geodetic, ecef):
    ours = geodetic_to_ecef(*geodetic)
    for got, want in zip(ours, ecef, strict=True):
        assert math.isclose(got, want, abs_tol=1e-6)


@pytest.mark.parametrize("geodetic,ecef", ECEF_CASES)
def test_ecef_to_geodetic_inverts_it(geodetic, ecef):
    latitude, longitude, altitude = ecef_to_geodetic(*ecef)
    want_lat, want_lon, want_alt = geodetic
    assert math.isclose(latitude, want_lat, abs_tol=1e-9)
    assert math.isclose(altitude, want_alt, abs_tol=1e-6)
    if abs(want_lat) != 90.0:  # longitude is undefined at a pole
        assert math.isclose(longitude, want_lon, abs_tol=1e-9)


def test_the_enu_basis_is_orthonormal_and_right_handed():
    east, north, up = enu_basis(35.68, 139.7)
    for vector in (east, north, up):
        assert math.isclose(math.sqrt(sum(c * c for c in vector)), 1.0, abs_tol=1e-12)
    for a, b in ((east, north), (north, up), (up, east)):
        assert math.isclose(sum(x * y for x, y in zip(a, b, strict=True)), 0.0, abs_tol=1e-12)
    # east x north == up, which is what makes the frame right-handed.
    cross = (
        east[1] * north[2] - east[2] * north[1],
        east[2] * north[0] - east[0] * north[2],
        east[0] * north[1] - east[1] * north[0],
    )
    for got, want in zip(cross, up, strict=True):
        assert math.isclose(got, want, abs_tol=1e-12)


def test_east_and_north_point_the_way_their_names_say():
    """A step of known bearing must land on the expected axis."""
    latitude, longitude, altitude = 35.68, 139.7, 40.0
    anchor = geodetic_to_ecef(latitude, longitude, altitude)
    basis = enu_basis(latitude, longitude)

    # 0.001 degrees of longitude is eastward; of latitude, northward.
    east_step = ecef_to_enu(geodetic_to_ecef(latitude, longitude + 0.001, altitude), anchor, basis)
    north_step = ecef_to_enu(geodetic_to_ecef(latitude + 0.001, longitude, altitude), anchor, basis)

    assert east_step[0] > 80.0 and abs(east_step[1]) < 1e-3
    assert north_step[1] > 100.0 and abs(north_step[0]) < 1e-3
    # Both stay on the tangent plane's surface, bar the curvature drop.
    assert abs(east_step[2]) < 0.01 and abs(north_step[2]) < 0.01


def test_the_anchor_is_its_own_origin():
    latitude, longitude, altitude = -33.87, 151.2, 5.0
    anchor = geodetic_to_ecef(latitude, longitude, altitude)
    at_origin = ecef_to_enu(anchor, anchor, enu_basis(latitude, longitude))
    assert at_origin == (0.0, 0.0, 0.0)


def test_the_tangent_transform_preserves_distance():
    """It is a rotation and a translation, so lengths must survive exactly."""
    anchor = geodetic_to_ecef(35.68, 139.7, 40.0)
    basis = enu_basis(35.68, 139.7)
    a = geodetic_to_ecef(35.681, 139.701, 45.0)
    b = geodetic_to_ecef(35.682, 139.703, 60.0)

    before = math.dist(a, b)
    after = math.dist(ecef_to_enu(a, anchor, basis), ecef_to_enu(b, anchor, basis))
    assert math.isclose(before, after, rel_tol=1e-12)
