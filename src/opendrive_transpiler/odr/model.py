"""A backend-neutral description of the OpenDRIVE road network to emit.

This layer exists so that "what OpenDRIVE should say" is decided once, in
`mapping/`, and "how to spell it" is decided separately, in `codegen/`. That
keeps the code generator a dumb, exhaustively testable serializer, and leaves
room for a direct-XML backend without touching the conversion logic.

Names and units follow the OpenDRIVE spec, not scenariogeneration's Python API.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeometryRecord:
    """One `<geometry>` in the road's `<planView>`, with an absolute start pose."""

    s: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str = "line"
    """"line", "arc" or "paramPoly3"."""
    params: dict[str, float] = field(default_factory=dict)
    """Curvature for an arc; au..dv for a paramPoly3. Empty for a line."""


@dataclass(frozen=True)
class PolyRecord:
    """A cubic `a + b*ds + c*ds^2 + d*ds^3` anchored at `s`.

    Serves both `<width>` (where `s` is the lane-section-relative `sOffset`) and
    `<elevation>` (where it is the road-relative `s`).
    """

    s: float
    a: float
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0


@dataclass(frozen=True)
class RoadMarkSpec:
    type: str = "none"
    """OpenDRIVE roadMark type: none, solid, broken, solid_solid, curb, edge, ..."""
    width: float | None = None
    color: str = "standard"
    weight: str = "standard"
    length: float | None = None
    space: float | None = None
    rule: str | None = None
    source: str = ""
    """Provenance: the lanelet2 (type, subtype) this came from."""


@dataclass
class LaneSpec:
    lane_id: int
    """OpenDRIVE lane id: negative right of the reference line, positive left."""
    lane_type: str = "driving"
    widths: list[PolyRecord] = field(default_factory=list)
    road_mark: RoadMarkSpec = field(default_factory=RoadMarkSpec)
    predecessor: int | None = None
    successor: int | None = None
    lanelet2_id: int = 0
    subtype: str = ""

    @property
    def constant_width(self) -> float | None:
        if len(self.widths) == 1 and not any(
            (self.widths[0].b, self.widths[0].c, self.widths[0].d)
        ):
            return self.widths[0].a
        return None


@dataclass
class LaneSectionSpec:
    s: float
    left: list[LaneSpec] = field(default_factory=list)
    """Innermost first (+1, +2, ...)."""
    right: list[LaneSpec] = field(default_factory=list)
    """Innermost first (-1, -2, ...)."""
    center_road_mark: RoadMarkSpec = field(default_factory=RoadMarkSpec)

    @property
    def lanes(self) -> list[LaneSpec]:
        return [*self.left, *self.right]


@dataclass(frozen=True)
class LinkSpec:
    element_type: str
    """"road" or "junction"."""
    element_id: int
    contact_point: str | None = None
    """"start" or "end"; omitted when linking to a junction."""


@dataclass
class RoadSpec:
    road_id: int
    name: str = ""
    geometries: list[GeometryRecord] = field(default_factory=list)
    lane_sections: list[LaneSectionSpec] = field(default_factory=list)
    elevations: list[PolyRecord] = field(default_factory=list)
    superelevations: list[PolyRecord] = field(default_factory=list)
    lane_offsets: list[PolyRecord] = field(default_factory=list)
    """`<laneOffset>`: where lane 0 sits relative to the reference line. Zero by
    construction when the reference line *is* a boundary, non-zero when it is a
    computed centerline that no boundary lies exactly on."""
    road_type: str = "unknown"
    speed: tuple[float, str] | None = None
    """(value, unit) for the `<type><speed>` record."""
    junction: int = -1
    """Junction id, or -1 when the road is not part of one."""
    predecessor: LinkSpec | None = None
    successor: LinkSpec | None = None
    signals: list[SignalSpec] = field(default_factory=list)
    objects: list[ObjectSpec] = field(default_factory=list)
    lanelet2_ids: tuple[int, ...] = ()
    """Provenance: which lanelets this road was built from."""
    rule: str = "RHT"

    @property
    def length(self) -> float:
        if not self.geometries:
            return 0.0
        last = self.geometries[-1]
        return last.s + last.length


@dataclass(frozen=True)
class SignalSpec:
    """A `<signal>`: a traffic light, sign or speed restriction on a road."""

    s: float
    t: float
    country: str = "OpenDRIVE"
    type: str = "1000001"
    subtype: str = "-1"
    name: str = ""
    dynamic: bool = False
    value: float | None = None
    unit: str | None = None
    z_offset: float = 1.5
    validity: tuple[int, int] | None = None
    """(fromLane, toLane) the signal applies to."""
    lanelet2_id: int = 0
    source: str = ""


@dataclass(frozen=True)
class OutlineCorner:
    s: float
    t: float
    dz: float = 0.0
    height: float = 0.0


@dataclass(frozen=True)
class ObjectSpec:
    """An `<object>`: map furniture with an outline, such as an area or polygon."""

    s: float
    t: float
    type: str = "none"
    name: str = ""
    corners: tuple[OutlineCorner, ...] = ()
    lanelet2_id: int = 0
    source: str = ""


@dataclass
class ConnectionSpec:
    incoming_road: int
    connecting_road: int
    contact_point: str = "start"
    lane_links: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class JunctionSpec:
    junction_id: int
    name: str = ""
    connections: list[ConnectionSpec] = field(default_factory=list)


@dataclass
class OdrModel:
    name: str = "map"
    rev_major: str = "1"
    rev_minor: str = "5"
    geo_reference: str | None = None
    roads: list[RoadSpec] = field(default_factory=list)
    junctions: list[JunctionSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Free-text provenance lines for the generated file's header comment."""

    @property
    def lane_count(self) -> int:
        return sum(len(section.lanes) for road in self.roads for section in road.lane_sections)

    @property
    def total_length(self) -> float:
        return sum(road.length for road in self.roads)


@dataclass
class TranspileStats:
    """What the run actually produced, for the CLI summary and the file header."""

    lanelets_in: int = 0
    lanelets_converted: int = 0
    lanelets_skipped: int = 0
    roads: int = 0
    lane_sections: int = 0
    lanes: int = 0
    junctions: int = 0
    signals: int = 0
    objects: int = 0
    areas_skipped: int = 0
    polygons_skipped: int = 0
    regelems_skipped: int = 0

    def describe(self) -> str:
        parts = [
            f"{self.lanelets_converted}/{self.lanelets_in} lanelets converted",
            f"{self.roads} roads",
            f"{self.lanes} lanes",
        ]
        if self.signals:
            parts.append(f"{self.signals} signals")
        if self.objects:
            parts.append(f"{self.objects} objects")
        if self.lanelets_skipped:
            parts.append(f"{self.lanelets_skipped} lanelets skipped")
        if self.areas_skipped:
            parts.append(f"{self.areas_skipped} areas skipped")
        if self.polygons_skipped:
            parts.append(f"{self.polygons_skipped} polygons skipped")
        if self.regelems_skipped:
            parts.append(f"{self.regelems_skipped} regulatory elements skipped")
        return ", ".join(parts)
