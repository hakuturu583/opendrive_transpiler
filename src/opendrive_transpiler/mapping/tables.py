"""Translating lanelet2's string tags into OpenDRIVE's typed vocabulary.

lanelet2 carries every semantic as a `str -> str` tag, so these tables are where
a map stops being geometry and starts being a road network: which lanelets are
drivable, what the boundary markings look like, how fast traffic may go.

Where lanelet2 says nothing, this module picks a documented convention rather
than inventing precision it does not have -- marking widths and dash patterns are
the obvious cases, and both are configurable.
"""

from __future__ import annotations

import math
import re

from ..config import TranspileOptions
from ..ir.model import ProjectionIR
from ..odr.model import RoadMarkSpec

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
}

DEFAULT_LANE_TYPE = "driving"

# Subtypes that are not lanes in the driving sense; they still convert, but a
# consumer should not route over them.
NON_DRIVABLE_SUBTYPES = frozenset(
    {"walkway", "pedestrian_lane", "shared_walkway", "crosswalk", "stairs", "parking"}
)


def lane_type_for(subtype: str) -> tuple[str, bool]:
    """Returns the OpenDRIVE lane type and whether the subtype was recognised."""
    key = (subtype or "").strip().lower()
    if key in LANE_TYPE_BY_SUBTYPE:
        return LANE_TYPE_BY_SUBTYPE[key], True
    return DEFAULT_LANE_TYPE, False


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

# Boundary types that are physical features rather than painted markings.
_MARK_BY_TYPE: dict[str, str] = {
    "curbstone": "curb",
    "road_border": "edge",
    "virtual": "none",
    "guard_rail": "none",
    "fence": "none",
    "wall": "none",
    "stop_line": "none",
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


def utm_zone(longitude: float) -> int:
    return math.floor((longitude + 180.0) / 6.0) + 1


def geo_reference_for(projection: ProjectionIR | None) -> tuple[str | None, str | None]:
    """Returns `(proj_string, caveat)`.

    `caveat` is set when the string is a faithful description of the projection
    family but not of the exact origin offset -- notably UTM with `useOffset`,
    where lanelet2 subtracts the origin's easting/northing and reproducing that
    needs a real forward projection. Emitting the family and saying so beats
    emitting a subtly wrong transform in silence.
    """
    if projection is None:
        return None, None

    kind = projection.kind
    lat, lon = projection.lat, projection.lon

    if kind == "utm":
        hemisphere = "+north" if lat >= 0 else "+south"
        proj = f"+proj=utm +zone={utm_zone(lon)} {hemisphere} +datum=WGS84 +units=m +no_defs"
        caveat = None
        if projection.use_offset:
            caveat = (
                "UtmProjector was built with useOffset=True, so map coordinates are "
                "relative to the origin's easting/northing; the emitted geoReference "
                "describes the UTM zone but not that offset"
            )
        return proj, caveat

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

    # MGRS and geocentric have no single PROJ string that matches what lanelet2
    # produced; the frontend has already reported this.
    return None, f"{kind} projection has no PROJ equivalent"
