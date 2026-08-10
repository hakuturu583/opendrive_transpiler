"""The intermediate representation: a frozen snapshot of what the script built.

The shadow objects the frontend produces are live, aliased and mutable -- exactly
what is needed while interpreting, and exactly what the later stages should not
have to reason about. This module takes one snapshot at the boundary and turns
storage aliasing (an interpreter concept) into integer `node`/`bound` keys (a
graph concept), which is all topology inference actually needs.

Everything downstream of here is plain data.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import TranspileOptions
from ..diagnostics import (
    I_AREA_SKIPPED,
    I_REGELEM_SKIPPED,
    W_DEGENERATE_LANELET,
    W_NO_LANELETS,
    DiagnosticBag,
    SourceSpan,
)


@dataclass(frozen=True)
class PointIR:
    x: float
    y: float
    z: float
    lanelet2_id: int
    key: int
    """Identity of the underlying storage: equal keys are literally the same node."""

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class BoundIR:
    """One boundary linestring, in view order."""

    lanelet2_id: int
    key: int
    """Identity of the linestring storage; shared keys mean a shared boundary."""
    reversed_view: bool
    points: tuple[PointIR, ...]
    attributes: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def coords(self) -> list[tuple[float, float, float]]:
        return [p.xyz for p in self.points]

    @property
    def node_keys(self) -> tuple[int, ...]:
        return tuple(p.key for p in self.points)


@dataclass(frozen=True)
class RegElemIR:
    kind: str
    lanelet2_id: int
    attributes: dict[str, str] = field(default_factory=dict)
    roles: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """Role name -> lanelet2 ids of the members, for reporting."""


@dataclass(frozen=True)
class LaneletIR:
    lanelet2_id: int
    left: BoundIR
    right: BoundIR
    attributes: dict[str, str] = field(default_factory=dict)
    regelems: tuple[RegElemIR, ...] = ()
    centerline: tuple[tuple[float, float, float], ...] | None = None
    """Set only when the script assigned one explicitly."""

    @property
    def subtype(self) -> str:
        return self.attributes.get("subtype", "")

    @property
    def one_way(self) -> bool:
        # lanelet2's default is one-way; only an explicit "no" makes it two-way.
        return self.attributes.get("one_way", "yes").lower() not in {"no", "false", "0"}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"LaneletIR(#{self.lanelet2_id}, subtype={self.subtype!r})"


@dataclass(frozen=True)
class AreaIR:
    lanelet2_id: int
    attributes: dict[str, str] = field(default_factory=dict)
    outer: tuple[BoundIR, ...] = ()


@dataclass(frozen=True)
class ProjectionIR:
    kind: str
    lat: float
    lon: float
    alt: float
    use_offset: bool


@dataclass
class MapIR:
    lanelets: list[LaneletIR] = field(default_factory=list)
    areas: list[AreaIR] = field(default_factory=list)
    regelems: list[RegElemIR] = field(default_factory=list)
    projection: ProjectionIR | None = None
    source_name: str = "map"

    @property
    def empty(self) -> bool:
        return not self.lanelets


# --------------------------------------------------------------------------
# Snapshotting
# --------------------------------------------------------------------------


def _snapshot_attributes(attributes: Any) -> dict[str, str]:
    try:
        return {str(k): str(v) for k, v in attributes.items()}
    except AttributeError:
        return {}


def _snapshot_bound(bound: Any) -> BoundIR:
    points = tuple(
        PointIR(
            x=float(p.storage.x),
            y=float(p.storage.y),
            z=float(p.storage.z),
            lanelet2_id=int(p.storage.id),
            key=id(p.storage),
        )
        for p in bound.points
    )
    return BoundIR(
        lanelet2_id=int(bound.storage.id),
        key=id(bound.storage),
        reversed_view=bool(bound.inverted_view),
        points=points,
        attributes=_snapshot_attributes(bound.storage.attributes),
    )


def _snapshot_regelem(regelem: Any) -> RegElemIR:
    roles = {
        role: tuple(int(getattr(m, "id", 0) or 0) for m in members)
        for role, members in getattr(regelem, "parameters", {}).items()
    }
    return RegElemIR(
        kind=getattr(regelem, "kind", "RegulatoryElement"),
        lanelet2_id=int(getattr(regelem, "id", 0) or 0),
        attributes=_snapshot_attributes(getattr(regelem, "attributes", {})),
        roles=roles,
    )


def _selected_lanelets(registry: Any) -> list[Any]:
    """Which lanelets are "the map".

    A script that builds a map is stating which lanelets it means, so those win.
    A script that only constructs lanelets (very common in small examples and in
    the test corpus) still gets converted -- from everything it constructed.
    """
    from_maps: list[Any] = []
    seen: set[int] = set()
    for shadow_map in registry.maps:
        for lanelet in shadow_map.lanelets:
            if id(lanelet) not in seen:
                seen.add(id(lanelet))
                from_maps.append(lanelet)
    if from_maps:
        return from_maps
    return list(registry.lanelets)


def build_ir(
    registry: Any,
    bag: DiagnosticBag,
    options: TranspileOptions,
    *,
    source_name: str = "map",
) -> MapIR:
    """Freeze a finished `Registry` into a `MapIR`."""
    del options  # reserved: no option currently changes the snapshot

    ir = MapIR(source_name=source_name)

    if registry.projection is not None:
        proj = registry.projection
        ir.projection = ProjectionIR(
            kind=proj.kind,
            lat=proj.lat,
            lon=proj.lon,
            alt=proj.alt,
            use_offset=proj.use_offset,
        )

    seen_bounds: set[tuple[int, int]] = set()
    for lanelet in _selected_lanelets(registry):
        left, right = lanelet.left, lanelet.right
        if left is None or right is None:
            bag.warn(
                W_DEGENERATE_LANELET,
                f"lanelet #{lanelet.id} has no {'left' if left is None else 'right'} "
                "bound; skipped",
            )
            continue
        if len(left) < 2 or len(right) < 2:
            bag.warn(
                W_DEGENERATE_LANELET,
                f"lanelet #{lanelet.id} has a bound with fewer than 2 points "
                f"(left={len(left)}, right={len(right)}); skipped",
            )
            continue

        # `Lanelet(other)` clones share both bounds; emitting each would produce
        # duplicate overlapping roads.
        signature = (id(left.storage), id(right.storage))
        if signature in seen_bounds:
            continue
        seen_bounds.add(signature)

        regelems = tuple(_snapshot_regelem(r) for r in lanelet.regelems)
        centerline = None
        if lanelet.centerline_override is not None:
            centerline = tuple(p.xyz for p in lanelet.centerline_override.points)

        ir.lanelets.append(
            LaneletIR(
                lanelet2_id=int(lanelet.id),
                left=_snapshot_bound(left),
                right=_snapshot_bound(right),
                attributes=_snapshot_attributes(lanelet.attributes),
                regelems=regelems,
                centerline=centerline,
            )
        )

    ir.areas = [
        AreaIR(
            lanelet2_id=int(area.id),
            attributes=_snapshot_attributes(area.attributes),
            outer=tuple(_snapshot_bound(b) for b in area.outer),
        )
        for area in registry.areas
    ]
    ir.regelems = [_snapshot_regelem(r) for r in registry.regelems]

    _report_deferred(ir, bag)
    return ir


def _report_deferred(ir: MapIR, bag: DiagnosticBag) -> None:
    """Name what was recognised but is not converted in this release.

    These codes are deliberately paired with unchecked rows in the README support
    matrix; if one fires, the matrix says why.
    """
    if not ir.lanelets:
        bag.warn(
            W_NO_LANELETS,
            "the script produced no convertible lanelets; nothing to emit",
        )

    if ir.areas:
        bag.info(
            I_AREA_SKIPPED,
            f"{len(ir.areas)} Area(s) recognised; areas have no OpenDRIVE road "
            "equivalent and are not converted yet (planned as road <object> outlines)",
        )

    if ir.regelems:
        kinds = sorted({r.kind for r in ir.regelems})
        bag.info(
            I_REGELEM_SKIPPED,
            f"{len(ir.regelems)} regulatory element(s) recognised "
            f"({', '.join(kinds)}); signals and priorities are not converted yet",
        )


def iter_bounds(lanelets: Iterable[LaneletIR]) -> Iterable[BoundIR]:
    for lanelet in lanelets:
        yield lanelet.left
        yield lanelet.right


__all__ = [
    "AreaIR",
    "BoundIR",
    "LaneletIR",
    "MapIR",
    "PointIR",
    "ProjectionIR",
    "RegElemIR",
    "SourceSpan",
    "build_ir",
    "iter_bounds",
]
