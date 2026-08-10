"""Placing signals and objects onto roads.

lanelet2 locates a regulatory element by *reference*: a traffic light is a set of
linestrings, and the lanelets that obey it hold a pointer to it. OpenDRIVE
locates a `<signal>` by *position*: an `(s, t)` along one road. Converting one to
the other means projecting the element's own geometry onto the reference line of
the road built from the lanelet that refers to it.

The same projection places `<object>` outlines, which is how an `Area` or a
standalone `Polygon` -- neither of which is a road -- reaches the output at all.

Signal type codes are the one place here that is convention rather than data.
lanelet2 does not carry OpenDRIVE codes, so the generic `country="OpenDRIVE"`
catalogue is used and the choice is recorded on every emitted signal.
"""

from __future__ import annotations

from ..config import TranspileOptions
from ..geometry.fit import lateral_offset
from ..geometry.polyline import point_at_station, station_of_point
from ..geometry.vec import Vec3
from ..ir.model import AreaIR, LaneletIR, PolygonIR, RegElemIR
from ..odr.model import ObjectSpec, OutlineCorner, RoadSpec, SignalSpec
from . import tables

# OpenDRIVE's generic signal catalogue. lanelet2 carries no country codes, so
# these are conventions, stated once here rather than scattered through the code.
_GENERIC_TRAFFIC_LIGHT = "1000001"
_GENERIC_SPEED_LIMIT = "274"
_GENERIC_SIGN = "1000013"

_SIGNAL_KINDS = {
    "TrafficLight": (_GENERIC_TRAFFIC_LIGHT, True),
    "AutowareTrafficLight": (_GENERIC_TRAFFIC_LIGHT, True),
    "SpeedLimit": (_GENERIC_SPEED_LIMIT, False),
    "TrafficSign": (_GENERIC_SIGN, False),
}


def _anchor(regelem: RegElemIR) -> tuple[float, float, float] | None:
    """A single point that stands for where the element is.

    The stop line if there is one -- that is what a driver actually stops at --
    otherwise the mean of whatever geometry the element refers to.
    """
    for role in ("ref_line", "refers"):
        lines = regelem.geometry.get(role)
        if not lines:
            continue
        points = [point for line in lines for point in line]
        if not points:
            continue
        count = len(points)
        return (
            sum(p[0] for p in points) / count,
            sum(p[1] for p in points) / count,
            sum(p[2] for p in points) / count,
        )
    return None


def _project(reference: list[Vec3], point: tuple[float, float, float]) -> tuple[float, float]:
    """`(s, t)` of a world point against a road's reference line."""
    s = station_of_point(reference, (point[0], point[1]))
    on_line, heading = point_at_station(reference, s)
    return s, lateral_offset(on_line, heading, point)


def signals_for(
    road: RoadSpec,
    reference: list[Vec3],
    lanelets: list[LaneletIR],
    options: TranspileOptions,
) -> list[SignalSpec]:
    """Every signal the lanelets of this road refer to, placed on it."""
    if not options.signals:
        return []

    out: list[SignalSpec] = []
    seen: set[int] = set()

    for lanelet in lanelets:
        for regelem in lanelet.regelems:
            entry = _SIGNAL_KINDS.get(regelem.kind)
            if entry is None:
                continue
            # One element may govern several lanelets of the same road; it is
            # still one signal.
            if regelem.lanelet2_id and regelem.lanelet2_id in seen:
                continue

            point = _anchor(regelem)
            if point is None:
                continue
            s, t = _project(reference, point)
            if not 0.0 <= s <= road.length:
                continue

            seen.add(regelem.lanelet2_id)
            type_code, dynamic = entry
            value = unit = None
            if regelem.kind == "SpeedLimit":
                parsed = tables.speed_for(lanelet.attributes.get("speed_limit", ""))
                if parsed is not None:
                    value, unit = parsed

            lane_ids = [
                lane.lane_id
                for section in road.lane_sections
                for lane in section.lanes
                if lane.lanelet2_id == lanelet.lanelet2_id
            ]
            validity = (min(lane_ids), max(lane_ids)) if lane_ids else None

            out.append(
                SignalSpec(
                    s=s,
                    t=t,
                    type=type_code,
                    subtype=regelem.attributes.get("subtype", "-1") or "-1",
                    name=f"{regelem.kind}_{regelem.lanelet2_id}",
                    dynamic=dynamic,
                    value=value,
                    unit=unit,
                    validity=validity,
                    lanelet2_id=regelem.lanelet2_id,
                    source=regelem.kind,
                )
            )

    return out


def _outline(
    reference: list[Vec3], rings: list[list[tuple[float, float, float]]]
) -> tuple[float, float, tuple[OutlineCorner, ...]]:
    """Project a ring of world points into road-relative outline corners."""
    corners: list[OutlineCorner] = []
    for ring in rings:
        for point in ring:
            s, t = _project(reference, point)
            on_line, _heading = point_at_station(reference, s)
            corners.append(OutlineCorner(s=s, t=t, dz=point[2] - on_line[2]))
    if not corners:
        return 0.0, 0.0, ()
    anchor_s = min(corner.s for corner in corners)
    anchor_t = sum(corner.t for corner in corners) / len(corners)
    # Corners are relative to the object's own anchor.
    relative = tuple(
        OutlineCorner(s=c.s - anchor_s, t=c.t - anchor_t, dz=c.dz, height=c.height) for c in corners
    )
    return anchor_s, anchor_t, relative


def _ring_points(bounds) -> list[tuple[float, float, float]]:
    """Chain a ring's line strings into one point list.

    lanelet2 rings are stored as separate line strings that share end points, so
    the joints duplicate. The closing point is dropped too: the outline is
    emitted with `closed=True`, which already implies it.
    """
    points: list[tuple[float, float, float]] = []
    for bound in bounds:
        for point in bound.coords:
            if not points or points[-1] != point:
                points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def objects_for(
    road: RoadSpec,
    reference: list[Vec3],
    areas: list[AreaIR],
    polygons: list[PolygonIR],
    options: TranspileOptions,
) -> list[ObjectSpec]:
    """Areas and polygons that fall along this road, as `<object>` outlines.

    Each is attached to whichever road it lies nearest, which the caller decides;
    this only does the projection.
    """
    if not options.objects:
        return []

    out: list[ObjectSpec] = []

    for area in areas:
        rings = [_ring_points(area.outer)]
        rings += [_ring_points(inner) for inner in area.inners]
        s, t, corners = _outline(reference, [r for r in rings if len(r) >= 3])
        if not corners:
            continue
        out.append(
            ObjectSpec(
                s=s,
                t=t,
                type=_object_type(area.attributes),
                name=f"area_{area.lanelet2_id}",
                corners=corners,
                lanelet2_id=area.lanelet2_id,
                source="Area",
            )
        )

    for polygon in polygons:
        if polygon.bound is None or len(polygon.bound) < 3:
            continue
        s, t, corners = _outline(reference, [polygon.bound.coords])
        if not corners:
            continue
        out.append(
            ObjectSpec(
                s=s,
                t=t,
                type=_object_type(polygon.attributes),
                name=f"polygon_{polygon.lanelet2_id}",
                corners=corners,
                lanelet2_id=polygon.lanelet2_id,
                source="Polygon",
            )
        )

    del road
    return out


_OBJECT_TYPES = {
    "parking": "parkingSpace",
    "crosswalk": "crosswalk",
    "traffic_island": "obstacle",
    "building": "building",
    "vegetation": "vegetation",
}


def _object_type(attributes: dict[str, str]) -> str:
    for key in ("subtype", "type"):
        value = (attributes.get(key, "") or "").strip().lower()
        if value in _OBJECT_TYPES:
            return _OBJECT_TYPES[value]
    return "none"
