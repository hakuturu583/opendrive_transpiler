"""Emitting a `scenariogeneration.xodr` script from an `OdrModel`.

Three things about the generated script are deliberate.

**It uses `add_fixed_geometry`, not `add_geometry`.** We already know every
coordinate exactly, so there is nothing to solve for. `add_fixed_geometry` stores
an absolute `(x, y, hdg)` per record and marks the planView adjusted, which is
what keeps the emitted geometry bit-for-bit faithful to the input. (The two
methods are mutually exclusive -- mixing them raises `MixOfGeometryAddition` --
so this choice is all-or-nothing.)

**It does not call `adjust_roads_and_lanes()`.** That call is
`adjust_startpoints()` plus `create_lane_links()` over every pair of roads. The
first is a no-op here because the geometry is already fixed, so the only reason to
call it was lane linking -- and lane linking is inferred there from geometry, which
fails on real maps: it refuses outright when two connected roads carry different
lane counts, which a lane widening precisely is. The links are written down from
the lanelet topology instead, where the correspondence is known exactly.

**It carries provenance comments.** Each road names the lanelets it came from and
each lane its subtype and width. The generated file is meant to be read and
edited, which is why this module writes source text directly rather than going
through `ast.unparse` (which would discard every comment).

The result is parsed with `ast.parse` before it is returned, so an emitter bug
fails here rather than downstream.
"""

from __future__ import annotations

import ast
import hashlib

from ..config import TranspileOptions
from ..diagnostics import DiagnosticBag
from ..odr.model import (
    LaneSectionSpec,
    LaneSpec,
    OdrModel,
    PolyRecord,
    RoadMarkSpec,
    RoadSpec,
    TranspileStats,
)
from .writer import SourceWriter, literal

GENERATOR = "opendrive_transpiler"


def _mark_expr(mark: RoadMarkSpec) -> str:
    parts = [f"xodr.RoadMarkType.{mark.type}"]
    if mark.width is not None:
        parts.append(f"width={literal(mark.width)}")
    if mark.length is not None:
        parts.append(f"length={literal(mark.length)}")
    if mark.space is not None:
        parts.append(f"space={literal(mark.space)}")
    if mark.color != "standard":
        parts.append(f"color=xodr.RoadMarkColor.{mark.color}")
    if mark.weight != "standard":
        parts.append(f"marking_weight=xodr.RoadMarkWeight.{mark.weight}")
    return f"xodr.RoadMark({', '.join(parts)})"


def _geometry_expr(kind: str, length: float, params: dict[str, float]) -> str:
    if kind == "line":
        return f"xodr.Line({literal(length)})"
    if kind == "arc":
        return f"xodr.Arc({literal(params['curvature'])}, length={literal(length)})"
    if kind == "paramPoly3":
        args = ", ".join(
            literal(params[name]) for name in ("au", "bu", "cu", "du", "av", "bv", "cv", "dv")
        )
        return f"xodr.ParamPoly3({args})"
    raise ValueError(f"unsupported geometry kind: {kind}")


def _width_args(record: PolyRecord) -> str:
    parts = [f"a={literal(record.a)}"]
    for name in ("b", "c", "d"):
        value = getattr(record, name)
        if value:
            parts.append(f"{name}={literal(value)}")
    parts.append(f"soffset={literal(record.s)}")
    return ", ".join(parts)


class ScenarioGenerationEmitter:
    def __init__(self, options: TranspileOptions) -> None:
        self.options = options

    # ------------------------------------------------------------------
    def emit(
        self,
        model: OdrModel,
        stats: TranspileStats,
        bag: DiagnosticBag,
        *,
        source_name: str = "<string>",
        source_text: str = "",
    ) -> str:
        writer = SourceWriter()
        self._header(writer, model, stats, bag, source_name, source_text)
        writer.blank()
        needs_priority = any(junction.priorities for junction in model.junctions)
        if needs_priority:
            writer.line("import xml.etree.ElementTree as ET")
            writer.blank()
        writer.line("from scenariogeneration import xodr")
        if needs_priority:
            writer.blank(2)
            self._priority_shim(writer)
        writer.blank(2)
        writer.line("def build() -> xodr.OpenDrive:")
        with writer.block():
            writer.line(f'"""Build the OpenDRIVE model transpiled from {source_name}."""')
            self._open_drive(writer, model)
            for road in model.roads:
                writer.blank()
                self._road(writer, road)
            for junction in model.junctions:
                writer.blank()
                self._junction(writer, junction)
            writer.blank()
            if model.roads:
                writer.comment(
                    "No adjust_roads_and_lanes() call: geometry is already fixed to the "
                    "input coordinates, and every road and lane link was resolved from "
                    "the lanelet topology rather than re-derived from geometry."
                )
            writer.line("return odr")
        writer.blank(2)
        writer.line('if __name__ == "__main__":')
        with writer.block():
            writer.line(f"build().write_xml({literal(model.name + '.xodr')})")

        source = writer.render()
        # A generated file that does not parse is an emitter bug; surface it here
        # rather than letting it reach the user as a confusing ImportError later.
        ast.parse(source, filename="<generated>")
        return source

    def _priority_shim(self, writer: SourceWriter) -> None:
        """Emit a Junction that can carry `<priority>`.

        `scenariogeneration` does not model the element -- its `Junction` writes
        only attributes and connections -- so the alternative to these eight lines
        would be dropping right-of-way rules the input map states outright.
        Subclassing keeps everything else about the junction the library's job.
        """
        writer.line("class _JunctionWithPriority(xodr.Junction):")
        with writer.block():
            writer.line('"""xodr.Junction plus <priority>, which the library omits."""')
            writer.blank()
            writer.line("def __init__(self, *args, **kwargs):")
            with writer.block():
                writer.line("super().__init__(*args, **kwargs)")
                writer.line("self.priorities = []")
            writer.blank()
            writer.line("def get_element(self):")
            with writer.block():
                writer.line("element = super().get_element()")
                writer.comment("OpenDRIVE orders <junction> as connection*, priority*.")
                writer.line("for high, low in self.priorities:")
                with writer.block():
                    writer.line(
                        'ET.SubElement(element, "priority", {"high": str(high), "low": str(low)})'
                    )
                writer.line("return element")

    # ------------------------------------------------------------------
    def _header(
        self,
        writer: SourceWriter,
        model: OdrModel,
        stats: TranspileStats,
        bag: DiagnosticBag,
        source_name: str,
        source_text: str,
    ) -> None:
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest() if source_text else ""
        lines = [
            f"Generated by {GENERATOR}. Do not edit if you intend to regenerate.",
            "",
            f"source:  {source_name}",
        ]
        if digest:
            lines.append(f"sha256:  {digest}")
        lines.append(f"summary: {stats.describe()}")

        counts = bag.counts()
        if any(counts.values()):
            parts = [f"{n} {name}(s)" for name, n in counts.items() if n]
            lines.append(f"notices: {', '.join(parts)}")
            # Every notice is repeated here, informational ones included: the
            # generated file has to say what the conversion left out, or a reader
            # has no way to tell a complete road network from a partial one.
            for diagnostic in bag:
                lines.append(f"    {diagnostic.code}: {diagnostic.message}")
        else:
            lines.append("notices: none")

        for note in model.notes:
            lines.append(f"note:    {note}")

        lines += [
            "",
            f"Run this file to write {model.name}.xodr. Requires scenariogeneration:",
            '    pip install "opendrive-transpiler[emit]"',
        ]
        writer.docstring("\n".join(lines))

    def _open_drive(self, writer: SourceWriter, model: OdrModel) -> None:
        args = [
            literal(model.name),
            f"revMajor={literal(model.rev_major)}",
            f"revMinor={literal(model.rev_minor)}",
        ]
        if model.geo_reference:
            args.append(f"geo_reference={literal(model.geo_reference)}")
        writer.line(f"odr = xodr.OpenDrive({', '.join(args)})")

    # ------------------------------------------------------------------
    def _road(self, writer: SourceWriter, road: RoadSpec) -> None:
        rid = road.road_id
        origins = ", ".join(f"#{i}" for i in road.lanelet2_ids) or "none"
        writer.rule(f"road {rid}  <-  lanelet {origins}")
        writer.comment(
            f"reference line: {len(road.geometries)} geometry record(s), length {road.length:.6g} m"
        )

        writer.line(f"pv_{rid} = xodr.PlanView()")
        for geometry in road.geometries:
            expr = _geometry_expr(geometry.kind, geometry.length, geometry.params)
            writer.line(
                f"pv_{rid}.add_fixed_geometry("
                f"{expr}, {literal(geometry.x)}, {literal(geometry.y)}, "
                f"{literal(geometry.hdg)}, s={literal(geometry.s)})"
            )

        writer.blank()
        writer.line(f"lanes_{rid} = xodr.Lanes()")
        for offset in road.lane_offsets:
            writer.line(
                f"lanes_{rid}.add_laneoffset(xodr.LaneOffset("
                f"s={literal(offset.s)}, a={literal(offset.a)}, b={literal(offset.b)}, "
                f"c={literal(offset.c)}, d={literal(offset.d)}))"
            )
        for section_index, section in enumerate(road.lane_sections):
            self._lane_section(writer, rid, section_index, section)

        writer.blank()
        road_args = [
            literal(rid),
            f"pv_{rid}",
            f"lanes_{rid}",
        ]
        if road.junction != -1:
            road_args.append(f"road_type={literal(road.junction)}")
        if road.name:
            road_args.append(f"name={literal(road.name)}")
        writer.line(f"road_{rid} = xodr.Road({', '.join(road_args)})")

        type_args = [f"xodr.RoadType.{road.road_type}", literal(0.0)]
        if road.speed is not None:
            value, unit = road.speed
            type_args.append(f"speed={literal(value)}")
            type_args.append(f"speed_unit={literal(unit)}")
        writer.line(f"road_{rid}.add_type({', '.join(type_args)})")

        for elevation in road.elevations:
            writer.line(
                f"road_{rid}.add_elevation({literal(elevation.s)}, {literal(elevation.a)}, "
                f"{literal(elevation.b)}, {literal(elevation.c)}, {literal(elevation.d)})"
            )

        # A flat road produces a single zero record, which says nothing; only a
        # genuinely banked one is worth writing out.
        if any(any((r.a, r.b, r.c, r.d)) for r in road.superelevations):
            for banking in road.superelevations:
                writer.line(
                    f"road_{rid}.add_superelevation({literal(banking.s)}, "
                    f"{literal(banking.a)}, {literal(banking.b)}, "
                    f"{literal(banking.c)}, {literal(banking.d)})"
                )

        links = ((road.predecessor, "add_predecessor"), (road.successor, "add_successor"))
        for link, method in links:
            if link is None:
                continue
            args = [f"xodr.ElementType.{link.element_type}", literal(link.element_id)]
            if link.contact_point:
                args.append(f"contact_point=xodr.ContactPoint.{link.contact_point}")
            writer.line(f"road_{rid}.{method}({', '.join(args)})")

        for index, signal in enumerate(road.signals):
            self._signal(writer, rid, index, signal)
        for index, obj in enumerate(road.objects):
            self._object(writer, rid, index, obj)

        writer.line(f"odr.add_road(road_{rid})")

    def _signal(self, writer: SourceWriter, rid: int, index: int, signal) -> None:
        name = f"sig_{rid}_{index}"
        writer.comment(f"{signal.source} #{signal.lanelet2_id} at s = {signal.s:.6g} m")
        args = [
            literal(signal.s),
            literal(signal.t),
            literal(signal.country),
            literal(signal.type),
            f"subtype={literal(signal.subtype)}",
            f"name={literal(signal.name)}",
        ]
        if signal.dynamic:
            args.append("dynamic=xodr.Dynamic.yes")
        if signal.value is not None:
            args.append(f"value={literal(signal.value)}")
            args.append(f"unit={literal(signal.unit)}")
        args.append(f"zOffset={literal(signal.z_offset)}")
        writer.line(f"{name} = xodr.Signal({', '.join(args)})")
        if signal.validity is not None:
            low, high = signal.validity
            writer.line(f"{name}.add_validity({literal(low)}, {literal(high)})")
        writer.line(f"road_{rid}.add_signal({name})")

    def _object(self, writer: SourceWriter, rid: int, index: int, obj) -> None:
        name = f"obj_{rid}_{index}"
        note = f"{obj.source} #{obj.lanelet2_id}: {len(obj.corners)} outline corner(s)"
        if not obj.closed:
            note += ", open"
        if obj.height:
            note += f", height {obj.height:.6g} m assumed"
        writer.comment(note)
        args = [
            literal(obj.s),
            literal(obj.t),
            f"Type={literal(obj.type)}",
            f"name={literal(obj.name)}",
        ]
        if obj.height:
            args.append(f"height={literal(obj.height)}")
        writer.line(f"{name} = xodr.Object({', '.join(args)})")
        # A barrier is a polyline; closing it would draw a return leg that is not
        # in the input.
        writer.line(f"{name}_outline = xodr.Outline(closed={literal(obj.closed)})")
        for corner in obj.corners:
            writer.line(
                f"{name}_outline.add_corner(xodr.CornerRoad("
                f"{literal(corner.s)}, {literal(corner.t)}, {literal(corner.dz)}, "
                f"{literal(corner.height)}))"
            )
        writer.line(f"{name}.add_outline({name}_outline)")
        writer.line(f"road_{rid}.add_object({name})")

    def _junction(self, writer: SourceWriter, junction) -> None:
        jid = junction.junction_id
        counts = f"{len(junction.connections)} connection(s)"
        if junction.priorities:
            counts += f", {len(junction.priorities)} priority record(s)"
        writer.rule(f"junction {jid}  ({counts})")
        constructor = "_JunctionWithPriority" if junction.priorities else "xodr.Junction"
        writer.line(f"junction_{jid} = {constructor}({literal(junction.name)}, {literal(jid)})")
        for priority in junction.priorities:
            writer.comment(f"RightOfWay #{priority.regelem2_id}")
            writer.line(
                f"junction_{jid}.priorities.append("
                f"({literal(priority.high)}, {literal(priority.low)}))"
            )
        for index, connection in enumerate(junction.connections):
            name = f"conn_{jid}_{index}"
            writer.line(
                f"{name} = xodr.Connection({literal(connection.incoming_road)}, "
                f"{literal(connection.connecting_road)}, "
                f"xodr.ContactPoint.{connection.contact_point})"
            )
            for incoming, outgoing in connection.lane_links:
                writer.line(f"{name}.add_lanelink({literal(incoming)}, {literal(outgoing)})")
            writer.line(f"junction_{jid}.add_connection({name})")
        writer.line(f"odr.add_junction(junction_{jid})")

    def _lane_section(
        self, writer: SourceWriter, rid: int, section_index: int, section: LaneSectionSpec
    ) -> None:
        tag = f"{rid}_{section_index}"
        writer.blank()
        writer.comment(f"lane section {section_index} at s = {section.s:.6g} m")

        writer.line(f"center_{tag} = xodr.Lane(a={literal(0.0)})")
        writer.line(f"center_{tag}.add_roadmark({_mark_expr(section.center_road_mark)})")
        writer.line(f"ls_{tag} = xodr.LaneSection({literal(section.s)}, center_{tag})")

        for lane in section.left:
            self._lane(writer, tag, lane, side="left")
        for lane in section.right:
            self._lane(writer, tag, lane, side="right")

        writer.line(f"lanes_{rid}.add_lanesection(ls_{tag})")

    def _lane(self, writer: SourceWriter, tag: str, lane: LaneSpec, *, side: str) -> None:
        name = f"lane_{tag}_{'p' if lane.lane_id > 0 else 'm'}{abs(lane.lane_id)}"
        constant = lane.constant_width
        description = (
            f"width {constant:.6g} m (constant)"
            if constant is not None
            else f"{len(lane.widths)} width record(s)"
        )
        writer.comment(
            f"lane {lane.lane_id}  <-  lanelet #{lane.lanelet2_id}"
            f"{f' (subtype={lane.subtype!r})' if lane.subtype else ''}, {description}"
        )

        first = lane.widths[0] if lane.widths else PolyRecord(0.0, 0.0)
        args = [f"lane_type=xodr.LaneType.{lane.lane_type}", f"a={literal(first.a)}"]
        for coefficient in ("b", "c", "d"):
            value = getattr(first, coefficient)
            if value:
                args.append(f"{coefficient}={literal(value)}")
        # The Lane constructor *is* width record 0, and OpenDRIVE requires that
        # record to start at the section origin.
        args.append(f"soffset={literal(0.0)}")
        writer.line(f"{name} = xodr.Lane({', '.join(args)})")

        for record in lane.widths[1:]:
            writer.line(f"{name}.add_lane_width({_width_args(record)})")

        writer.line(f"{name}.add_roadmark({_mark_expr(lane.road_mark)})")

        if lane.predecessor is not None:
            writer.line(f'{name}.add_link("predecessor", {literal(lane.predecessor)})')
        if lane.successor is not None:
            writer.line(f'{name}.add_link("successor", {literal(lane.successor)})')

        writer.line(f"ls_{tag}.add_{side}_lane({name})")


def emit_source(
    model: OdrModel,
    stats: TranspileStats,
    bag: DiagnosticBag,
    options: TranspileOptions,
    *,
    source_name: str = "<string>",
    source_text: str = "",
) -> str:
    return ScenarioGenerationEmitter(options).emit(
        model, stats, bag, source_name=source_name, source_text=source_text
    )
