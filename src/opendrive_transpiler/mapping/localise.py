"""Rotating an earth-centred map onto a local tangent plane.

Every other lanelet2 projector hands us a planar metre frame, so the geometry
passes through untouched and the projection only decides what `<geoReference>`
says those metres mean. `GeocentricProjector` is the exception: it emits
earth-centred XYZ, in which no two axes span a horizontal plane. Near Tokyo a
road sits at roughly (-3.96e6, 3.35e6, 3.70e6) metres and its "up" direction is a
mixture of all three, so reading x and y as a plan view would foreshorten every
road and tilt every flat one.

The fix is to rotate the map into east/north/up about an anchor before anything
downstream measures a length or fits an arc. The anchor is the map's own
centroid, so the result is centred on the data and the numbers stay small.

Node identity survives: a `PointIR` keeps its `key`, which is what topology uses
to decide whether two boundaries share a node. Only coordinates change, and the
transform is rigid, so lengths, angles and adjacency are all preserved.
"""

from __future__ import annotations

from dataclasses import replace

from ..geometry.vec import Vec3
from ..ir.model import AreaIR, BoundIR, LaneletIR, MapIR, PointIR, PolygonIR, RegElemIR
from .proj import ecef_to_enu, ecef_to_geodetic, enu_basis


def anchor_of(ir: MapIR) -> Vec3 | None:
    """The centroid of every boundary point, in ECEF.

    Every point counts once per appearance rather than once per node. Weighting
    by node would need a dedupe pass to answer a question -- where is the middle
    of this map -- that does not need that much precision.
    """
    total = [0.0, 0.0, 0.0]
    count = 0
    for lanelet in ir.lanelets:
        for bound in (lanelet.left, lanelet.right):
            for point in bound.points:
                total[0] += point.x
                total[1] += point.y
                total[2] += point.z
                count += 1
    if not count:
        return None
    return (total[0] / count, total[1] / count, total[2] / count)


def rebase(ir: MapIR) -> tuple[MapIR, Vec3, Vec3] | None:
    """Rewrite `ir` into the tangent plane at its own centroid.

    Returns `(map, anchor_ecef, anchor_geodetic)`, or None when there is nothing
    to anchor to.
    """
    anchor = anchor_of(ir)
    if anchor is None:
        return None

    latitude, longitude, altitude = ecef_to_geodetic(*anchor)
    basis = enu_basis(latitude, longitude)

    def move(x: float, y: float, z: float) -> Vec3:
        return ecef_to_enu((x, y, z), anchor, basis)

    moved = MapIR(
        lanelets=[_lanelet(ll, move) for ll in ir.lanelets],
        areas=[_area(area, move) for area in ir.areas],
        polygons=[_polygon(polygon, move) for polygon in ir.polygons],
        regelems=[_regelem(regelem, move) for regelem in ir.regelems],
        projection=ir.projection,
        source_name=ir.source_name,
    )
    return moved, anchor, (latitude, longitude, altitude)


# --------------------------------------------------------------------------
# Rewriting, one node type at a time
# --------------------------------------------------------------------------
# Each of these rebuilds a frozen record with new coordinates and everything
# else -- ids, keys, attributes, view orientation -- carried across unchanged.


def _point(point: PointIR, move) -> PointIR:
    x, y, z = move(point.x, point.y, point.z)
    return replace(point, x=x, y=y, z=z)


def _bound(bound: BoundIR, move) -> BoundIR:
    return replace(bound, points=tuple(_point(p, move) for p in bound.points))


def _lanelet(lanelet: LaneletIR, move) -> LaneletIR:
    centerline = lanelet.centerline
    return replace(
        lanelet,
        left=_bound(lanelet.left, move),
        right=_bound(lanelet.right, move),
        regelems=tuple(_regelem(r, move) for r in lanelet.regelems),
        centerline=None if centerline is None else tuple(move(*xyz) for xyz in centerline),
    )


def _area(area: AreaIR, move) -> AreaIR:
    return replace(
        area,
        outer=tuple(_bound(b, move) for b in area.outer),
        inners=tuple(tuple(_bound(b, move) for b in ring) for ring in area.inners),
    )


def _polygon(polygon: PolygonIR, move) -> PolygonIR:
    if polygon.bound is None:
        return polygon
    return replace(polygon, bound=_bound(polygon.bound, move))


def _regelem(regelem: RegElemIR, move) -> RegElemIR:
    if not regelem.geometry:
        return regelem
    return replace(
        regelem,
        geometry={
            role: tuple(tuple(move(*xyz) for xyz in line) for line in lines)
            for role, lines in regelem.geometry.items()
        },
    )
