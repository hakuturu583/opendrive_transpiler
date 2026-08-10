"""The lanelet2 API registry: dotted name -> shadow constructor.

Everything the input script can reach from `lanelet2` resolves through here. A
name that is not in this table never resolves to a real import -- it becomes
`Unknown`. That is what keeps the frontend from needing lanelet2 installed, and
it is also what keeps it from executing anything.

Constructor overloads are hand-parsed in the same order Boost.Python's overload
chain would try them, because that ordering is observable: `Point3d(p2d)` aliases
storage while `Point3d(id, x, y)` builds a new point, and only argument *shape*
distinguishes them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..diagnostics import (
    E_LOAD_FROM_FILE,
    E_NOT_INSTANTIABLE,
    I_PROJECTION_SKIPPED,
    I_QUERY_IGNORED,
    DiagnosticBag,
    SourceSpan,
)
from .shadow import (
    UNKNOWN,
    AttributeMap,
    BasicPoint,
    GPSPoint,
    LineStringStorage,
    OpaqueValue,
    Origin,
    PointStorage,
    ProjectionInfo,
    ShadowArea,
    ShadowLanelet,
    ShadowLaneletWithStopLine,
    ShadowLineString,
    ShadowMap,
    ShadowPoint,
    ShadowRegulatoryElement,
    is_unknown,
)


class ModuleRef:
    """A bound module name, e.g. what `import lanelet2` puts in scope."""

    __slots__ = ("dotted",)

    def __init__(self, dotted: str) -> None:
        self.dotted = dotted

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<module {self.dotted}>"


@dataclass
class Args:
    """Positional/keyword arguments, resolved by position-or-name like PyO3 does."""

    positional: list[Any] = field(default_factory=list)
    keyword: dict[str, Any] = field(default_factory=dict)

    def get(self, index: int, *names: str, default: Any = None) -> Any:
        if index < len(self.positional):
            return self.positional[index]
        for name in names:
            if name in self.keyword:
                return self.keyword[name]
        return default

    def has(self, index: int, *names: str) -> bool:
        if index < len(self.positional):
            return True
        return any(name in self.keyword for name in names)

    @property
    def empty(self) -> bool:
        return not self.positional and not self.keyword

    def only_positional(self, count: int) -> bool:
        return len(self.positional) == count and not self.keyword


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None or is_unknown(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_attributes(value: Any) -> AttributeMap:
    if isinstance(value, AttributeMap):
        return value
    if isinstance(value, dict):
        return AttributeMap(value)
    return AttributeMap()


class Registry:
    """Owns the id counter and every lanelet2 constructor.

    Also records the by-products the later stages need: the maps that were built,
    every lanelet that was constructed (so a script that never builds a map is
    still convertible), and the projector, whose origin becomes `<geoReference>`.
    """

    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag
        self._next_id = 1
        self.maps: list[ShadowMap] = []
        self.lanelets: list[ShadowLanelet] = []
        self.areas: list[ShadowArea] = []
        self.regelems: list[ShadowRegulatoryElement] = []
        self.projection: ProjectionInfo | None = None
        self._table: dict[str, Callable[[Args, SourceSpan], Any]] = {}
        self._build_table()

    # -- ids ---------------------------------------------------------------
    def get_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def register_id(self, ident: int) -> None:
        self._next_id = max(self._next_id, int(ident) + 1)

    # -- resolution --------------------------------------------------------
    def resolve(self, dotted: str) -> Callable[[Args, SourceSpan], Any] | None:
        return self._table.get(dotted)

    def is_module(self, dotted: str) -> bool:
        return dotted in KNOWN_MODULES

    def is_known_prefix(self, dotted: str) -> bool:
        return any(dotted.startswith(prefix) for prefix in KNOWN_ROOTS)

    # -- table -------------------------------------------------------------
    def _build_table(self) -> None:
        t = self._table
        core = "lanelet2.core."

        # Points -----------------------------------------------------------
        for name, dim, const in (
            ("Point3d", 3, False),
            ("Point2d", 2, False),
            ("ConstPoint3d", 3, True),
            ("ConstPoint2d", 2, True),
        ):
            t[core + name] = self._point_ctor(name, dim, const)

        t[core + "BasicPoint3d"] = lambda a, s: BasicPoint(
            _as_float(a.get(0, "x")), _as_float(a.get(1, "y")), _as_float(a.get(2, "z")), 3
        )
        t[core + "BasicPoint2d"] = lambda a, s: BasicPoint(
            _as_float(a.get(0, "x")), _as_float(a.get(1, "y")), 0.0, 2
        )
        t[core + "GPSPoint"] = lambda a, s: GPSPoint(
            _as_float(a.get(0, "lat")), _as_float(a.get(1, "lon")), _as_float(a.get(2, "ele"))
        )
        t[core + "AttributeMap"] = lambda a, s: _as_attributes(a.get(0, "attributes"))

        # LineStrings and polygons ----------------------------------------
        for name, dim, const, hybrid, polygon in (
            ("LineString3d", 3, False, False, False),
            ("LineString2d", 2, False, False, False),
            ("ConstLineString3d", 3, True, False, False),
            ("ConstLineString2d", 2, True, False, False),
            ("ConstHybridLineString3d", 3, True, True, False),
            ("ConstHybridLineString2d", 2, True, True, False),
            ("Polygon3d", 3, False, False, True),
            ("Polygon2d", 2, False, False, True),
            ("ConstPolygon3d", 3, True, False, True),
            ("ConstPolygon2d", 2, True, False, True),
            ("ConstHybridPolygon3d", 3, True, True, True),
            ("ConstHybridPolygon2d", 2, True, True, True),
        ):
            t[core + name] = self._linestring_ctor(name, dim, const, hybrid, polygon)

        # Lanelet / Area ---------------------------------------------------
        t[core + "Lanelet"] = self._lanelet_ctor("Lanelet", const=False)
        t[core + "ConstLanelet"] = self._lanelet_ctor("ConstLanelet", const=True)
        t[core + "Area"] = self._area_ctor("Area")
        t[core + "ConstArea"] = self._area_ctor("ConstArea")

        # Maps -------------------------------------------------------------
        t[core + "LaneletMap"] = lambda a, s: self._new_map(submap=False)
        t[core + "LaneletSubmap"] = lambda a, s: self._new_map(submap=True)

        for kind, submap in (("Map", False), ("Submap", True)):
            for what in ("Points", "LineStrings", "Polygons", "Lanelets", "Areas"):
                t[f"{core}create{kind}From{what}"] = self._create_map_ctor(submap)

        t[core + "getId"] = lambda a, s: self.get_id()
        t[core + "registerId"] = lambda a, s: self.register_id(_as_int(a.get(0, "id")))

        # Regulatory elements ---------------------------------------------
        t[core + "TrafficLight"] = self._regelem_ctor(
            "TrafficLight", [("refers", 2, ("trafficLights",)), ("ref_line", 3, ("stopLine",))]
        )
        t[core + "RightOfWay"] = self._regelem_ctor(
            "RightOfWay",
            [
                ("right_of_way", 2, ("rightOfWayLanelets",)),
                ("yield", 3, ("yieldLanelets",)),
                ("ref_line", 4, ("stopLine",)),
            ],
        )
        for name in ("TrafficSign", "SpeedLimit"):
            t[core + name] = self._regelem_ctor(
                name,
                [
                    ("refers", 2, ("trafficSigns",)),
                    ("cancels", 3, ("cancellingTrafficSigns",)),
                    ("ref_line", 4, ("refLines",)),
                    ("cancel_line", 5, ("cancelLines",)),
                ],
            )
        t[core + "AllWayStop"] = self._regelem_ctor(
            "AllWayStop", [("refers", 2, ("lltsWithStop",)), ("signs", 3, ("signs",))]
        )
        t[core + "RegulatoryElement"] = self._not_instantiable("RegulatoryElement")
        t[core + "LaneletWithStopLine"] = lambda a, s: ShadowLaneletWithStopLine(
            a.get(0, "lanelet"), a.get(1, "stopLine")
        )
        t[core + "ConstLaneletWithStopLine"] = t[core + "LaneletWithStopLine"]
        t[core + "TrafficSignsWithType"] = lambda a, s: OpaqueValue("TrafficSignsWithType")

        # Autoware extension regulatory elements -- modelled generically.
        awext = "autoware_lanelet2_extension_python.regulatory_elements."
        for name, roles in (
            ("Crosswalk", [("refers", 2, ("crosswalk_lanelet",))]),
            ("DetectionArea", [("refers", 2, ("detectionAreas",)), ("ref_line", 3, ("stopLine",))]),
            ("NoParkingArea", [("refers", 2, ("no_parking_areas",))]),
            ("NoStoppingArea", [("refers", 2, ("no_stopping_areas",))]),
            ("RoadMarking", [("refers", 2, ("road_marking",))]),
            ("SpeedBump", [("refers", 2, ("speed_bump",))]),
            ("AutowareTrafficLight", [("refers", 2, ("trafficLights",))]),
        ):
            t[awext + name] = self._regelem_ctor(name, roles)
        t[awext + "VirtualTrafficLight"] = self._not_instantiable("VirtualTrafficLight")

        # io ----------------------------------------------------------------
        t["lanelet2.io.Origin"] = self._origin_ctor
        t["lanelet2.io.load"] = self._load_unsupported
        t["lanelet2.io.loadRobust"] = self._load_unsupported
        t["lanelet2.io.write"] = lambda a, s: None
        t["lanelet2.io.writeRobust"] = lambda a, s: []

        # projection ---------------------------------------------------------
        for name, kind, has_origin in (
            ("UtmProjector", "utm", True),
            ("MercatorProjector", "mercator", True),
            ("LocalCartesianProjector", "local_cartesian", True),
            ("GeocentricProjector", "geocentric", False),
        ):
            t["lanelet2.projection." + name] = self._projector_ctor(kind, has_origin)
        awproj = "autoware_lanelet2_extension_python.projection."
        t[awproj + "MGRSProjector"] = self._projector_ctor("mgrs", True)
        t[awproj + "TransverseMercatorProjector"] = self._projector_ctor(
            "transverse_mercator", True
        )

    # -- constructor builders ---------------------------------------------
    def _new_map(self, *, submap: bool) -> ShadowMap:
        shadow = ShadowMap(submap=submap)
        shadow._id_source = self.get_id
        self.maps.append(shadow)
        return shadow

    def _point_ctor(self, class_name: str, dim: int, const: bool):
        def ctor(args: Args, span: SourceSpan) -> Any:
            if const:
                self.bag.error(
                    E_NOT_INSTANTIABLE, f"{class_name} cannot be constructed directly", span
                )
                return UNKNOWN

            if args.empty:
                return ShadowPoint(PointStorage(), dim=dim, const=const)

            # `Point3d(p2d)` -- alias the *other* dimension's storage.
            if args.only_positional(1) and isinstance(args.positional[0], ShadowPoint):
                return args.positional[0].alias(dim, const)

            ident = _as_int(args.get(0, "id"))
            second = args.get(1, "x", "point")

            if isinstance(second, BasicPoint):
                storage = PointStorage(
                    x=second.x,
                    y=second.y,
                    z=second.z if second.dim == 3 else 0.0,
                    id=ident,
                    attributes=_as_attributes(args.get(2, "attributes")),
                )
            else:
                storage = PointStorage(
                    x=_as_float(second),
                    y=_as_float(args.get(2, "y")),
                    z=_as_float(args.get(3, "z")),
                    id=ident,
                    attributes=_as_attributes(args.get(4, "attributes")),
                )
            return ShadowPoint(storage, dim=dim, const=const)

        return ctor

    def _linestring_ctor(self, class_name: str, dim: int, const: bool, hybrid: bool, polygon: bool):
        def ctor(args: Args, span: SourceSpan) -> Any:
            if args.empty:
                return ShadowLineString(
                    LineStringStorage(), dim=dim, const=const, hybrid=hybrid, polygon=polygon
                )

            # `LineString3d(ls2d)` / `ConstHybrid...(const_ls)` -- alias.
            if args.only_positional(1) and isinstance(args.positional[0], ShadowLineString):
                return args.positional[0].alias(dim, const, hybrid)

            points = [p for p in _as_list(args.get(1, "points")) if isinstance(p, ShadowPoint)]
            storage = LineStringStorage(
                points=points,
                id=_as_int(args.get(0, "id")),
                attributes=_as_attributes(args.get(2, "attributes")),
            )
            return ShadowLineString(storage, dim=dim, const=const, hybrid=hybrid, polygon=polygon)

        return ctor

    def _lanelet_ctor(self, class_name: str, *, const: bool):
        def ctor(args: Args, span: SourceSpan) -> Any:
            # `Lanelet(other)` shares storage with the original.
            if args.only_positional(1) and isinstance(args.positional[0], ShadowLanelet):
                other = args.positional[0]
                clone = ShadowLanelet(
                    id=other.id,
                    left=other.left,
                    right=other.right,
                    attributes=other.attributes,
                    regelems=other.regelems,
                    centerline_override=other.centerline_override,
                )
                self.lanelets.append(clone)
                return clone

            lanelet = ShadowLanelet(
                id=_as_int(args.get(0, "id")),
                left=args.get(1, "leftBound"),
                right=args.get(2, "rightBound"),
                attributes=_as_attributes(args.get(3, "attributes")),
                regelems=_as_list(args.get(4, "regelems")),
            )
            if not isinstance(lanelet.left, ShadowLineString):
                lanelet.left = None
            if not isinstance(lanelet.right, ShadowLineString):
                lanelet.right = None
            self.lanelets.append(lanelet)
            return lanelet

        return ctor

    def _area_ctor(self, class_name: str):
        def ctor(args: Args, span: SourceSpan) -> Any:
            if args.only_positional(1) and isinstance(args.positional[0], ShadowArea):
                return args.positional[0]
            inner_raw = _as_list(args.get(2, "innerBounds"))
            inners = [_as_list(ring) for ring in inner_raw]
            area = ShadowArea(
                id=_as_int(args.get(0, "id")),
                outer=_as_list(args.get(1, "outerBound")),
                inners=inners,
                attributes=_as_attributes(args.get(3, "attributes")),
            )
            self.areas.append(area)
            return area

        return ctor

    def _create_map_ctor(self, submap: bool):
        def ctor(args: Args, span: SourceSpan) -> Any:
            shadow = self._new_map(submap=submap)
            for value in _as_list(args.get(0, "values")):
                shadow.add(value)
            return shadow

        return ctor

    def _regelem_ctor(self, kind: str, roles: list[tuple[str, int, tuple[str, ...]]]):
        def ctor(args: Args, span: SourceSpan) -> Any:
            parameters: dict[str, list[Any]] = {}
            for role, index, names in roles:
                value = args.get(index, *names)
                members = _as_list(value)
                if members:
                    parameters[role] = members
            regelem = ShadowRegulatoryElement(
                kind=kind,
                id=_as_int(args.get(0, "id")),
                attributes=_as_attributes(args.get(1, "attributes")),
                parameters=parameters,
            )
            self.regelems.append(regelem)
            return regelem

        return ctor

    def _not_instantiable(self, class_name: str):
        def ctor(args: Args, span: SourceSpan) -> Any:
            self.bag.error(E_NOT_INSTANTIABLE, f"{class_name} cannot be constructed directly", span)
            return UNKNOWN

        return ctor

    def _origin_ctor(self, args: Args, span: SourceSpan) -> Any:
        if args.only_positional(1) and isinstance(args.positional[0], GPSPoint):
            return Origin(args.positional[0])
        return Origin(
            GPSPoint(
                _as_float(args.get(0, "lat")),
                _as_float(args.get(1, "lon")),
                _as_float(args.get(2, "alt", "ele")),
            )
        )

    def _projector_ctor(self, kind: str, has_origin: bool):
        def ctor(args: Args, span: SourceSpan) -> Any:
            lat = lon = alt = 0.0
            if has_origin:
                origin = args.get(0, "origin")
                if isinstance(origin, Origin):
                    lat, lon, alt = origin.position.lat, origin.position.lon, origin.position.ele
                elif isinstance(origin, GPSPoint):
                    lat, lon, alt = origin.lat, origin.lon, origin.ele
            info = ProjectionInfo(
                kind=kind,
                lat=lat,
                lon=lon,
                alt=alt,
                use_offset=bool(args.get(1, "useOffset", default=True)),
            )
            # Last projector wins: scripts that build several are choosing the
            # one they pass to write(), and we cannot tell which from here.
            self.projection = info
            if kind in {"mgrs", "geocentric"}:
                self.bag.info(
                    I_PROJECTION_SKIPPED,
                    f"{kind} projection has no PROJ equivalent; <geoReference> omitted",
                    span,
                )
            return info

        return ctor

    def _load_unsupported(self, args: Args, span: SourceSpan) -> Any:
        self.bag.error(
            E_LOAD_FROM_FILE,
            "lanelet2.io.load() reads a map at runtime, so it cannot be resolved "
            "statically; this transpiler converts maps that the script *builds*",
            span,
        )
        return UNKNOWN

    # -- inert query APIs ---------------------------------------------------
    def opaque_call(self, dotted: str, span: SourceSpan) -> Any:
        """Handle a call into a module we deliberately do not model.

        Routing, traffic rules, geometry queries and matching all *read* a map
        that has already been built. A script may run dozens of such queries
        after constructing a perfectly convertible map, so these must be inert
        rather than fatal.
        """
        self.bag.info(
            I_QUERY_IGNORED,
            f"{dotted}(...) is a query API; ignored (it does not affect the map)",
            span,
        )
        return OpaqueValue(dotted)


KNOWN_ROOTS = ("lanelet2", "autoware_lanelet2_extension_python")

KNOWN_MODULES = {
    "lanelet2",
    "lanelet2.core",
    "lanelet2.geometry",
    "lanelet2.io",
    "lanelet2.matching",
    "lanelet2.projection",
    "lanelet2.routing",
    "lanelet2.traffic_rules",
    "autoware_lanelet2_extension_python",
    "autoware_lanelet2_extension_python.regulatory_elements",
    "autoware_lanelet2_extension_python.projection",
    "autoware_lanelet2_extension_python.utility",
    "autoware_lanelet2_extension_python.utility.query",
    "autoware_lanelet2_extension_python.utility.utilities",
}

# Modules whose calls are inert: they query a finished map rather than build one.
QUERY_MODULES = (
    "lanelet2.routing",
    "lanelet2.traffic_rules",
    "lanelet2.geometry",
    "lanelet2.matching",
    "autoware_lanelet2_extension_python.utility",
)

# Module-level constants the scripts read (enum members and the like). Values are
# irrelevant to conversion; they only have to compare and format sensibly.
MODULE_CONSTANTS: dict[str, Any] = {
    "lanelet2.core.ManeuverType": OpaqueValue("ManeuverType"),
    "lanelet2.routing.RelationType": OpaqueValue("RelationType"),
    "lanelet2.traffic_rules.Locations": OpaqueValue("Locations"),
    "lanelet2.traffic_rules.Participants": OpaqueValue("Participants"),
    "lanelet2.BUG_COMPAT": False,
    "lanelet2.core.math": math,
}
