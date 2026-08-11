"""Translating lanelet2's string tags into OpenDRIVE's typed vocabulary.

lanelet2 carries every semantic as a `str -> str` tag, so these tables are where
a map stops being geometry and starts being a road network: which lanelets are
drivable, what the boundary markings look like, how fast traffic may go.

Where lanelet2 says nothing, this module picks a documented convention rather
than inventing precision it does not have -- marking widths and dash patterns are
the obvious cases, and both are configurable.
"""

from __future__ import annotations

import re

from ..config import TranspileOptions
from ..ir.model import ProjectionIR
from ..odr.model import RoadMarkSpec
from .proj import (
    central_meridian,
    geodetic_to_ecef,
    mgrs_code_offsets,
    mgrs_square_offsets,
    utm_offsets,
    utm_zone,
)

# --------------------------------------------------------------------------
# Lanelet subtype -> OpenDRIVE lane type
# --------------------------------------------------------------------------

LANE_TYPE_BY_SUBTYPE: dict[str, str] = {
    "": "driving",
    "road": "driving",
    "highway": "driving",
    "play_street": "driving",
    "emergency_lane": "driving",
    "lane": "driving",
    "exit": "exit",
    "walkway": "sidewalk",
    "pedestrian_lane": "sidewalk",
    "shared_walkway": "sidewalk",
    "crosswalk": "sidewalk",
    "stairs": "none",
    "bicycle_lane": "biking",
    "bus_lane": "bus",
    "parking": "parking",
    "freespace": "restricted",
    "traffic_island": "median",
    # A rail lanelet carries rail vehicles, not cars. OpenDRIVE distinguishes
    # `rail` from `tram`; lanelet2 does not, so the literal match is the honest one.
    "rail": "rail",
}

DEFAULT_LANE_TYPE = "driving"

# Subtypes that are not lanes in the driving sense; they still convert, but a
# consumer should not route over them.
NON_DRIVABLE_SUBTYPES = frozenset(
    {"walkway", "pedestrian_lane", "shared_walkway", "crosswalk", "stairs", "parking"}
)


def lane_type_for(subtype: str, *, one_way: bool = True) -> tuple[str, bool]:
    """Returns the OpenDRIVE lane type and whether the subtype was recognised.

    A lanelet tagged `one_way=no` is drivable in both directions, which OpenDRIVE
    spells `bidirectional` -- a distinct lane type rather than a modifier, so it
    replaces the driving type instead of qualifying it.
    """
    key = (subtype or "").strip().lower()
    recognised = key in LANE_TYPE_BY_SUBTYPE
    lane_type = LANE_TYPE_BY_SUBTYPE[key] if recognised else DEFAULT_LANE_TYPE
    if not one_way and lane_type == "driving":
        return "bidirectional", recognised
    return lane_type, recognised


# --------------------------------------------------------------------------
# Lanelet subtype + location -> OpenDRIVE road type
# --------------------------------------------------------------------------


def road_type_for(subtype: str, location: str) -> str:
    key = (subtype or "").strip().lower()
    where = (location or "").strip().lower()

    if key == "highway":
        return "motorway"
    if key == "play_street":
        return "townPlayStreet"
    if key in {"walkway", "crosswalk", "pedestrian_lane", "shared_walkway", "stairs"}:
        return "pedestrian"
    if key == "bicycle_lane":
        return "bicycle"
    if where == "urban":
        return "town"
    if where == "nonurban":
        return "rural"
    return "unknown"


# --------------------------------------------------------------------------
# Boundary (type, subtype) -> roadMark
# --------------------------------------------------------------------------

_MARK_BY_PAIR: dict[tuple[str, str], tuple[str, str]] = {
    # (linestring type, subtype) -> (roadMark type, weight)
    ("line_thin", "solid"): ("solid", "standard"),
    ("line_thin", "dashed"): ("broken", "standard"),
    ("line_thin", "solid_solid"): ("solid_solid", "standard"),
    ("line_thin", "dashed_solid"): ("broken_solid", "standard"),
    ("line_thin", "solid_dashed"): ("solid_broken", "standard"),
    ("line_thick", "solid"): ("solid", "bold"),
    ("line_thick", "dashed"): ("broken", "bold"),
    ("line_thick", "solid_solid"): ("solid_solid", "bold"),
    ("line_thick", "dashed_solid"): ("broken_solid", "bold"),
    ("line_thick", "solid_dashed"): ("solid_broken", "bold"),
}

# Boundary types that are not lane dividers. Mapping them to `none` is a real
# answer rather than a fallback: each is either a physical feature or a surface
# marking, and OpenDRIVE's roadMark vocabulary describes neither. Where the thing
# itself can be carried, it is carried elsewhere -- `guard_rail`, `fence` and
# `wall` also become a barrier `<object>`, and a crossing becomes an
# `<object type="crosswalk">`.
_MARK_BY_TYPE: dict[str, str] = {
    "curbstone": "curb",
    "road_border": "edge",
    "virtual": "none",
    "guard_rail": "none",
    "fence": "none",
    "wall": "none",
    "stop_line": "none",
    # Markings painted across or beside the carriageway rather than along a lane
    # edge: crossing stripes, hatched keep-out boxes, no-parking zig-zags.
    "pedestrian_marking": "none",
    "zebra_marking": "none",
    "bike_marking": "none",
    "keepout": "none",
    "zig-zag": "none",
    "symbol": "none",
    # A tram rail embedded in the road surface.
    "rail": "none",
    "": "none",
}

_KNOWN_COLORS = frozenset({"standard", "blue", "green", "red", "white", "yellow", "orange"})


def road_mark_for(
    attributes: dict[str, str], options: TranspileOptions
) -> tuple[RoadMarkSpec, bool]:
    """Map a boundary linestring's tags to a `<roadMark>`.

    Returns the spec and whether the tags were recognised, so the caller can
    report an unmapped marking instead of silently drawing nothing.
    """
    kind = (attributes.get("type", "") or "").strip().lower()
    subtype = (attributes.get("subtype", "") or "").strip().lower()
    source = f"{kind}/{subtype}" if subtype else kind or "<untagged>"

    pair = _MARK_BY_PAIR.get((kind, subtype))
    if pair is not None:
        mark_type, weight = pair
        width = options.thick_mark_width if kind == "line_thick" else options.thin_mark_width
        spec = RoadMarkSpec(
            type=mark_type,
            width=width,
            weight=weight,
            color=_color_of(attributes),
            length=options.dash_length if "broken" in mark_type else None,
            space=options.dash_space if "broken" in mark_type else None,
            source=source,
        )
        return spec, True

    if kind in _MARK_BY_TYPE:
        return RoadMarkSpec(type=_MARK_BY_TYPE[kind], source=source), True

    if kind in {"line_thin", "line_thick"}:
        # Right type, unrecognised subtype: a solid line is the safer default.
        width = options.thick_mark_width if kind == "line_thick" else options.thin_mark_width
        return (
            RoadMarkSpec(
                type="solid",
                width=width,
                weight="bold" if kind == "line_thick" else "standard",
                color=_color_of(attributes),
                source=source,
            ),
            False,
        )

    return RoadMarkSpec(type="none", source=source), False


# Boundary types that are a physical barrier rather than a painted line. Their
# roadMark is `none` -- correctly, since none of them is a marking -- so without
# this they would reach the output as nothing at all.
_BARRIER_BY_TYPE: dict[str, str] = {
    "guard_rail": "barrier",
    "fence": "railing",
    "wall": "barrier",
}


def barrier_for(attributes: dict[str, str]) -> str | None:
    """The OpenDRIVE object type for a barrier boundary, or None if it is not one."""
    kind = (attributes.get("type", "") or "").strip().lower()
    return _BARRIER_BY_TYPE.get(kind)


def _color_of(attributes: dict[str, str]) -> str:
    color = (attributes.get("color", "") or "").strip().lower()
    return color if color in _KNOWN_COLORS else "standard"


# --------------------------------------------------------------------------
# Speed limits
# --------------------------------------------------------------------------

_SPEED_PATTERN = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/?h|kph|kmh|mph|m/?s|ms)?\s*$",
    re.IGNORECASE,
)

_UNIT_ALIASES = {
    "kmh": "km/h",
    "km/h": "km/h",
    "kph": "km/h",
    "mph": "mph",
    "ms": "m/s",
    "m/s": "m/s",
}


def speed_for(raw: str) -> tuple[float, str] | None:
    """Parse a lanelet2 `speed_limit` tag.

    A bare number means km/h, which is lanelet2's convention. Anything
    unparseable returns None so the caller can report it rather than guess.
    """
    match = _SPEED_PATTERN.match(raw or "")
    if not match:
        return None
    value = float(match.group("value"))
    unit = (match.group("unit") or "kmh").lower().replace(" ", "")
    return value, _UNIT_ALIASES.get(unit, "km/h")


# --------------------------------------------------------------------------
# Projector -> PROJ string for <header><geoReference>
# --------------------------------------------------------------------------


def _shifted_utm(zone: int, north: bool, x0: float, y0: float) -> str:
    """A UTM zone whose origin has been moved, written so PROJ obeys the move.

    Spelled out as the `tmerc` that `+proj=utm` is shorthand for, because `utm`
    hardcodes its own false easting and northing and **silently ignores** any
    `+x_0` / `+y_0` given alongside it. A string built the obvious way parses,
    looks right, and puts the map on the equator: `(0, 0)` in
    `+proj=utm +zone=32 +x_0=43885 +y_0=-5427629` comes back as 0.000000N
    4.511256E instead of the origin it names.
    """
    return (
        f"+proj=tmerc +lat_0=0 +lon_0={central_meridian(zone)!r} +k_0=0.9996 "
        f"+x_0={x0!r} +y_0={y0!r} +datum=WGS84 +units=m +no_defs" + ("" if north else " +south")
    )


def geo_reference_for(projection: ProjectionIR | None) -> tuple[str | None, str | None]:
    """Returns `(proj_string, caveat)`.

    `caveat` is set when the string describes the projection family faithfully
    but something about the frame could not be reproduced. Emitting a subtly
    wrong transform in silence would be worse than saying so.
    """
    if projection is None:
        return None, None

    kind = projection.kind
    lat, lon = projection.lat, projection.lon

    if kind == "utm":
        if projection.use_offset:
            # lanelet2 subtracts the origin's easting/northing, so the PROJ
            # string has to carry that shift or it points at the wrong continent.
            x0, y0 = utm_offsets(lat, lon)
            return _shifted_utm(utm_zone(lon), lat >= 0, x0, y0), None
        hemisphere = "+north" if lat >= 0 else "+south"
        return (
            f"+proj=utm +zone={utm_zone(lon)} {hemisphere} +datum=WGS84 +units=m +no_defs",
            None,
        )

    if kind == "mercator":
        return (
            f"+proj=merc +lat_ts={lat!r} +lon_0={lon!r} +datum=WGS84 +units=m +no_defs",
            None,
        )

    if kind == "local_cartesian":
        return (
            f"+proj=tmerc +lat_0={lat!r} +lon_0={lon!r} +k=1 +x_0=0 +y_0=0 "
            "+datum=WGS84 +units=m +no_defs",
            None,
        )

    if kind == "transverse_mercator":
        return (
            f"+proj=tmerc +lat_0={lat!r} +lon_0={lon!r} +k=0.9996 +x_0=0 +y_0=0 "
            "+datum=WGS84 +units=m +no_defs",
            None,
        )

    if kind == "mgrs":
        # MGRS coordinates are metres inside a 100 km grid square, which is UTM
        # shifted to that square's south-west corner.
        if projection.mgrs_code:
            # `setMGRSCode` names the square outright, so no origin is needed --
            # and this is how Autoware maps are georeferenced.
            decoded = mgrs_code_offsets(projection.mgrs_code)
            if decoded is None:
                return None, (
                    f"MGRS code {projection.mgrs_code!r} is not a 100 km grid square "
                    "designator (zone, band, two letters); <geoReference> omitted"
                )
            zone, x0, y0, square_lat = decoded
            return _shifted_utm(zone, square_lat >= 0, x0, y0), None

        x0, y0 = mgrs_square_offsets(lat, lon)
        return (
            _shifted_utm(utm_zone(lon), lat >= 0, x0, y0),
            "MGRS coordinates were reproduced as UTM offset to the origin's 100 km "
            "square; the grid-square letters themselves are not carried in PROJ",
        )

    if kind == "geocentric":
        # By the time this runs the map has been rotated onto the tangent plane
        # at (lat, lon, alt), so what the header must describe is that plane --
        # not the earth-centred frame the script wrote. PROJ calls it topocentric
        # and wants the origin in earth-centred metres, which is exact: no
        # conformal projection stands in for the tangent plane here.
        x0, y0, z0 = geodetic_to_ecef(lat, lon, projection.alt)
        return (
            f"+proj=topocentric +ellps=WGS84 +X_0={x0!r} +Y_0={y0!r} +Z_0={z0!r} +units=m +no_defs",
            None,
        )

    return None, f"{kind} projection has no PROJ equivalent"
