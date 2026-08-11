"""Assembling the OpenDRIVE model from the IR and the inferred topology.

The central decision here is the choice of reference line: **the outer-left
boundary of the leftmost lanelet in the cross-section**, with every forward lane
emitted to its right as ids -1, -2, ...

That is not the obvious choice -- the centerline is -- but it is the right one:

* It is *real input geometry*, so the planView reproduces the source coordinates
  exactly instead of approximating a computed curve.
* A centerline reference would force a single-lanelet road into a +1/-1 pair of
  half-width lanes, which breaks lane linking and reads nothing like the input.
* "Which lanelet is the middle one" has no answer for an even lane count.
* It generalises unchanged from one lane to N.

OpenDRIVE permits an off-centre reference line, and consumers handle it.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from typing import NamedTuple

from ..config import TranspileOptions
from ..diagnostics import (
    I_AREA_SKIPPED,
    I_CROSSWALK_OBJECT,
    I_GEO_REFERENCE,
    I_JUNCTION_SKIPPED,
    I_POLYGON_SKIPPED,
    I_PROJECTION_LOCALISED,
    I_REGELEM_SKIPPED,
    I_TWO_WAY,
    W_BAD_SPEED_LIMIT,
    W_BOUNDS_DISAGREE,
    W_BOUNDS_SWAPPED,
    W_CONTRAFLOW_RIGHT,
    W_DEGENERATE_LANELET,
    W_EMPTY_SUBTYPE,
    W_NEGATIVE_WIDTH,
    W_PIVOT_REFERENCE,
    W_SHORT_ROAD,
    W_STACK_NOT_SHARED,
    W_UNEQUAL_BOUND_ENDS,
    W_UNKNOWN_ROADMARK,
    W_UNKNOWN_SUBTYPE,
    DiagnosticBag,
)
from ..geometry.fit import build_plan_view, merge_collinear, signed_side
from ..geometry.polyline import (
    dedupe,
    point_at_station,
    sample_stations,
    station_of_point,
    total_length,
)
from ..geometry.profile import (
    build_profile,
    lane_widths,
    offsets_along,
    road_elevation,
    road_superelevation,
)
from ..geometry.vec import Vec3
from ..ir.centerline import centerline_coords
from ..ir.model import LaneletIR, MapIR
from ..odr.model import (
    LaneSectionSpec,
    LaneSpec,
    LinkSpec,
    OdrModel,
    RoadSpec,
    TranspileStats,
)
from ..topology import grouping, relations
from ..topology.index import NodeIndex
from . import furniture, junctions, localise, tables

BoundaryKey = tuple[int, ...]


class OrientedBounds(NamedTuple):
    """A lanelet's bounds with the geometric left first.

    The keys travel with the coordinates so a cross-section can tell whether two
    members really share the boundary it is about to stack them on. Comparing
    coordinates instead would answer a different question -- two distinct line
    strings can run through the same place -- and node identity is what lanelet2
    means by a shared bound.
    """

    left: list[Vec3]
    right: list[Vec3]
    left_attributes: dict[str, str]
    right_attributes: dict[str, str]
    left_key: BoundaryKey
    right_key: BoundaryKey


def boundary_key(boundary, index: NodeIndex) -> BoundaryKey:
    """Identity of a boundary, independent of the direction it is stored in.

    Two lanelets sharing a bound traverse it opposite ways, so a line string and
    its reverse have to hash alike.

    Identity comes from the `NodeIndex` rather than from the point objects,
    because that is the notion adjacency was inferred with: a script that builds
    two neighbours from separately constructed `Point3d` at the same coordinate
    still means them to share the bound, and storage identity alone would call
    every such pair unshared.
    """
    keys = index.signature(boundary)
    return min(keys, tuple(reversed(keys)))


def build_model(
    ir: MapIR, bag: DiagnosticBag, options: TranspileOptions
) -> tuple[OdrModel, TranspileStats]:
    stats = TranspileStats(lanelets_in=len(ir.lanelets))
    model = OdrModel(
        name=options.name or ir.source_name,
        rev_major=options.revision[0],
        rev_minor=options.revision[1],
    )

    if not ir.lanelets:
        return model, stats

    # Earth-centred coordinates are not a plan view, so they are rotated onto a
    # tangent plane before anything below measures a length or fits an arc.
    ir = _localise(ir, bag)

    _apply_geo_reference(ir, model, bag, options)

    # A crosswalk is a marking across a carriageway, not a carriageway of its own,
    # so it is held back from road building and emitted as an <object> instead.
    ir, crosswalks = _partition_crosswalks(ir, bag)

    if not ir.lanelets:
        _report_crosswalks_without_a_road(crosswalks, bag)
        return model, stats

    index = NodeIndex(ir.lanelets, options.point_tolerance)
    rels = relations.infer(ir.lanelets, index)
    network = grouping.build(ir.lanelets, rels)

    _report_topology(network, ir, crosswalks, bag, options)

    builder = _RoadBuilder(ir, options, bag, rels, index)
    chain_of_group: dict[int, int] = {}
    for chain_index, chain in enumerate(network.chains):
        for group in chain.groups:
            chain_of_group[id(group)] = chain_index

    roads: list[RoadSpec | None] = []
    for chain_index, chain in enumerate(network.chains):
        roads.append(builder.build_road(chain, road_id=chain_index + 1))

    _link_roads(network, rels, roads, ir, bag)

    _attach_furniture(builder, roads, ir, crosswalks, options)

    if options.junctions:
        model.junctions = junctions.build(network, rels, roads, ir, bag)
        stats.junctions = len(model.junctions)

    for road in roads:
        if road is None:
            continue
        model.roads.append(road)
        stats.roads += 1
        stats.lane_sections += len(road.lane_sections)
        stats.lanes += sum(len(section.lanes) for section in road.lane_sections)
        stats.lanelets_converted += len(road.lanelet2_ids)
        stats.signals += len(road.signals)
        stats.objects += len(road.objects)

    # A crosswalk that became an object did convert; counting only road members
    # would report it as dropped.
    converted = [o for road in model.roads for o in road.objects if o.source == "Crosswalk"]
    stats.lanelets_converted += len(converted)
    stats.lanelets_skipped = stats.lanelets_in - stats.lanelets_converted
    _report_crosswalks(crosswalks, converted, bag)
    _report_furniture(model, ir, stats, bag, options)
    return model, stats


def _partition_crosswalks(ir: MapIR, bag: DiagnosticBag) -> tuple[MapIR, list[LaneletIR]]:
    """Split crosswalk lanelets out of the carriageway.

    Only `crosswalk`. A `walkway` or `shared_walkway` runs *alongside* a road and
    is a path in its own right, so it stays a road; a crosswalk runs *across* one,
    and building it as a road produces a carriageway overlapping the street at
    right angles with no junction between them.
    """
    del bag  # reported later, once it is known whether each one found a road
    crosswalks = [ll for ll in ir.lanelets if ll.subtype.strip().lower() == "crosswalk"]
    if not crosswalks:
        return ir, []
    keep = [ll for ll in ir.lanelets if ll.subtype.strip().lower() != "crosswalk"]
    return replace(ir, lanelets=keep), crosswalks


def _report_crosswalks(crosswalks: list[LaneletIR], converted: list, bag: DiagnosticBag) -> None:
    """Say that crosswalks changed shape, and name any that found no road."""
    if not crosswalks:
        return

    placed = {obj.lanelet2_id for obj in converted}
    if placed:
        bag.info(
            I_CROSSWALK_OBJECT,
            f'{len(placed)} crosswalk lanelet(s) became <object type="crosswalk"> on the '
            "road they cross rather than roads of their own, which is how OpenDRIVE "
            "models a crossing; they are no longer routable paths",
        )

    orphaned = [ll for ll in crosswalks if ll.lanelet2_id not in placed]
    if orphaned:
        names = ", ".join(f"#{ll.lanelet2_id}" for ll in orphaned)
        bag.info(
            I_CROSSWALK_OBJECT,
            f"{len(orphaned)} crosswalk lanelet(s) ({names}) were not converted: an "
            "object has to sit on a road, and none was near enough or object output "
            "is disabled",
        )


def _report_crosswalks_without_a_road(crosswalks: list[LaneletIR], bag: DiagnosticBag) -> None:
    """A map of nothing but crosswalks has no carriageway to put them on."""
    if crosswalks:
        bag.info(
            I_CROSSWALK_OBJECT,
            f"{len(crosswalks)} crosswalk lanelet(s) were the only lanelets in the map; "
            "a crosswalk is emitted as an object on the road it crosses, and there is "
            "no such road here",
        )


def _report_furniture(
    model: OdrModel,
    ir: MapIR,
    stats: TranspileStats,
    bag: DiagnosticBag,
    options: TranspileOptions,
) -> None:
    """Say what reached the output and what did not, now that it is known."""
    emitted = [obj for road in model.roads for obj in road.objects]
    areas_out = sum(1 for obj in emitted if obj.source == "Area")
    polygons_out = sum(1 for obj in emitted if obj.source == "Polygon")
    stats.areas_skipped = len(ir.areas) - areas_out
    stats.polygons_skipped = len(ir.polygons) - polygons_out

    if stats.areas_skipped:
        bag.info(
            I_AREA_SKIPPED,
            f"{stats.areas_skipped} of {len(ir.areas)} Area(s) were not converted"
            + ("; object output is disabled" if not options.objects else ""),
        )
    if stats.polygons_skipped:
        bag.info(
            I_POLYGON_SKIPPED,
            f"{stats.polygons_skipped} of {len(ir.polygons)} Polygon(s) were not converted"
            + ("; object output is disabled" if not options.objects else ""),
        )

    # A regulatory element reaches the output as a <signal> or, for right-of-way
    # rules, as a junction <priority>. Counting only signals would report a rule
    # that did convert as dropped.
    converted = {signal.lanelet2_id for road in model.roads for signal in road.signals}
    converted |= {p.regelem2_id for junction in model.junctions for p in junction.priorities}
    unconverted = [r for r in ir.regelems if r.lanelet2_id not in converted]
    stats.regelems_skipped = len(unconverted)
    if unconverted:
        kinds = sorted({r.kind for r in unconverted})
        reason = (
            "; signal output is disabled"
            if not options.signals
            else "; these kinds have no <signal> equivalent"
        )
        bag.info(
            I_REGELEM_SKIPPED,
            f"{len(unconverted)} regulatory element(s) not converted ({', '.join(kinds)}){reason}",
        )


def _attach_furniture(
    builder,
    roads,
    ir: MapIR,
    crosswalks: list[LaneletIR],
    options: TranspileOptions,
) -> None:
    """Give every area, polygon and crosswalk to the road it lies nearest.

    Map furniture belongs to no road in lanelet2, so one has to be chosen. The
    nearest reference line is the only defensible answer, and attaching each to
    exactly one road avoids the same parking bay appearing three times.
    """
    if not options.objects or not (ir.areas or ir.polygons or crosswalks):
        return

    live = [road for road in roads if road is not None and road.road_id in builder._references]
    if not live:
        return

    def nearest(points) -> RoadSpec:
        def distance(road: RoadSpec) -> float:
            reference = builder._references[road.road_id]
            anchor = points[0]
            station = station_of_point(reference, (anchor[0], anchor[1]))
            on_line, _heading = point_at_station(reference, station)
            return (on_line[0] - anchor[0]) ** 2 + (on_line[1] - anchor[1]) ** 2

        return min(live, key=distance)

    for area in ir.areas:
        points = [p for bound in area.outer for p in bound.coords]
        if not points:
            continue
        road = nearest(points)
        road.objects.extend(
            furniture.objects_for(road, builder._references[road.road_id], [area], [], options)
        )

    for crosswalk in crosswalks:
        ring = furniture.crosswalk_ring(crosswalk)
        if len(ring) < 3:
            continue
        road = nearest(ring)
        road.objects.extend(
            furniture.crosswalks_for(builder._references[road.road_id], [crosswalk], options)
        )

    for polygon in ir.polygons:
        if polygon.bound is None or not polygon.bound.coords:
            continue
        road = nearest(polygon.bound.coords)
        road.objects.extend(
            furniture.objects_for(road, builder._references[road.road_id], [], [polygon], options)
        )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------


def _localise(ir: MapIR, bag: DiagnosticBag) -> MapIR:
    """Move an earth-centred map into east/north/up about its own centroid.

    Any other projection is already a planar metre frame and is left alone.
    The projector carries no origin -- geocentric coordinates are absolute -- so
    the anchor is recovered from the data and written back onto the projection,
    which is what lets `<geoReference>` name the frame the geometry now lives in.
    """
    if ir.projection is None or ir.projection.kind != "geocentric":
        return ir

    rebased = localise.rebase(ir)
    if rebased is None:  # pragma: no cover - guarded by the caller's emptiness check
        return ir

    moved, _anchor, (latitude, longitude, altitude) = rebased
    moved.projection = replace(
        ir.projection, lat=latitude, lon=longitude, alt=altitude, use_offset=False
    )
    bag.info(
        I_PROJECTION_LOCALISED,
        "geocentric coordinates are earth-centred XYZ, which is not a plan view; "
        f"the map was rotated onto the tangent plane at lat={latitude:.9f}, "
        f"lon={longitude:.9f}, h={altitude:.3f} m -- a rigid transform, so lengths "
        "and adjacency are unchanged",
    )
    return moved


def _apply_geo_reference(
    ir: MapIR, model: OdrModel, bag: DiagnosticBag, options: TranspileOptions
) -> None:
    if not options.emit_geo_reference:
        return
    if options.geo_reference is not None:
        model.geo_reference = options.geo_reference
        return

    proj, caveat = tables.geo_reference_for(ir.projection)
    model.geo_reference = proj
    if caveat:
        # Informational: the map is already in metres, so the geometry we emit is
        # unaffected either way -- only the georeferencing metadata is.
        bag.info(I_GEO_REFERENCE, caveat)
    if ir.projection is not None and proj:
        model.notes.append(
            f"geoReference derived from {ir.projection.kind} projector at "
            f"lat={ir.projection.lat}, lon={ir.projection.lon}"
        )


def _report_topology(
    network: grouping.Network,
    ir: MapIR,
    crosswalks: list[LaneletIR],
    bag: DiagnosticBag,
    options: TranspileOptions,
) -> None:
    if network.branch_lanelets and not options.junctions:
        names = ", ".join(f"#{ir.lanelets[i].lanelet2_id}" for i in network.branch_lanelets)
        bag.info(
            I_JUNCTION_SKIPPED,
            f"branch/merge at lanelet(s) {names}; junction generation is disabled, "
            "so these roads are emitted unconnected",
        )
    for left, right in network.two_way_pairs:
        bag.info(
            I_TWO_WAY,
            f"lanelets #{ir.lanelets[left].lanelet2_id} and "
            f"#{ir.lanelets[right].lanelet2_id} travel in opposite directions across a "
            "shared boundary; they become one road with +/- lanes, and its reference "
            "line follows that boundary regardless of --reference-line",
        )

    # Geometry checks cover the crosswalks too. They are not roads, but their
    # outline is still built from these bounds, so malformed ones would otherwise
    # become a silently malformed `<object>`.
    for lanelet in (*ir.lanelets, *crosswalks):
        if relations.bounds_misaligned(lanelet):
            bag.warn(
                W_UNEQUAL_BOUND_ENDS,
                f"lanelet #{lanelet.lanelet2_id}: its two bounds cover different "
                "stretches, so they disagree about where the lanelet ends; anything "
                "projected onto this road is clamped to its reference line and will "
                "be silently truncated",
            )

    for lanelet in (*ir.lanelets, *crosswalks):
        if relations.bounds_disagree(lanelet):
            bag.warn(
                W_BOUNDS_DISAGREE,
                f"lanelet #{lanelet.lanelet2_id}: its two bounds run in opposite "
                "directions, so the lanelet does not say which way traffic goes; "
                "the left bound was taken as authoritative",
            )

    # `one_way` is a lane type, so it stays on the carriageway: a crosswalk that
    # became an object has no direction of travel to describe.
    for lanelet in ir.lanelets:
        if not lanelet.one_way:
            bag.info(
                I_TWO_WAY,
                f"lanelet #{lanelet.lanelet2_id} is tagged one_way=no; emitted as "
                "LaneType.bidirectional, whose consumer support varies",
            )


# --------------------------------------------------------------------------
# Roads
# --------------------------------------------------------------------------


class _RoadBuilder:
    def __init__(
        self,
        ir: MapIR,
        options: TranspileOptions,
        bag: DiagnosticBag,
        rels: relations.Relations,
        index: NodeIndex,
    ) -> None:
        self.ir = ir
        self.options = options
        self.bag = bag
        self.rels = rels
        self.index = index
        self._swap_reported: set[int] = set()
        self._unshared_reported: set[tuple[int, int]] = set()
        self._references: dict[int, list[Vec3]] = {}

    # -- orientation -------------------------------------------------------
    def oriented_bounds(self, lanelet: LaneletIR) -> OrientedBounds:
        """Bounds ordered so the first really is on the geometric left.

        Hand-written maps get this wrong often enough to be worth checking --
        the canonical five-line lanelet2 example does. Silently accepting the
        names would mirror the road.
        """
        left, right = lanelet.left, lanelet.right
        left_coords = dedupe(left.coords)
        right_coords = dedupe(right.coords)

        if len(left_coords) >= 2 and signed_side(left_coords, right_coords) > 0.0:
            if lanelet.lanelet2_id not in self._swap_reported:
                self._swap_reported.add(lanelet.lanelet2_id)
                self.bag.warn(
                    W_BOUNDS_SWAPPED,
                    f"lanelet #{lanelet.lanelet2_id}: the bound named 'right' lies to the "
                    "left of the bound named 'left'; using the geometric left as the "
                    "reference line",
                )
            return OrientedBounds(
                right_coords,
                left_coords,
                right.attributes,
                left.attributes,
                boundary_key(right, self.index),
                boundary_key(left, self.index),
            )
        return OrientedBounds(
            left_coords,
            right_coords,
            left.attributes,
            right.attributes,
            boundary_key(left, self.index),
            boundary_key(right, self.index),
        )

    # -- assembly ----------------------------------------------------------
    def build_road(self, chain: grouping.RoadChain, road_id: int) -> RoadSpec | None:
        lanelets = self.ir.lanelets
        if not chain.groups:
            return None

        group_refs: list[list[Vec3]] = [
            merge_collinear(
                self._group_reference(group),
                heading_tolerance=self.options.heading_tolerance,
                chord_tolerance=self.options.chord_tolerance,
            )
            for group in chain.groups
        ]

        self._report_pivot(chain, road_id)
        self._report_contraflow(chain, road_id)

        concatenated: list[Vec3] = []
        for piece in group_refs:
            concatenated.extend(piece if not concatenated else piece[1:])
        concatenated = dedupe(concatenated)

        if len(concatenated) < 2:
            names = ", ".join(f"#{lanelets[i].lanelet2_id}" for i in chain.lanelet_indices)
            self.bag.warn(
                W_DEGENERATE_LANELET,
                f"road {road_id} ({names}) has no boundary with any extent, so there is "
                "no reference line to follow; dropped",
            )
            return None

        geometries, reference = build_plan_view(
            concatenated,
            fit=self.options.fit,
            heading_tolerance=self.options.heading_tolerance,
            chord_tolerance=self.options.chord_tolerance,
        )
        length = total_length(reference)
        if length < self.options.min_road_length or not geometries:
            self.bag.warn(
                W_SHORT_ROAD,
                f"road {road_id} has length {length:.6g} m; dropped",
            )
            return None

        leader = lanelets[chain.groups[0].members[0]]
        road = RoadSpec(
            road_id=road_id,
            name=self._road_name(chain, lanelets),
            geometries=geometries,
            elevations=road_elevation(
                reference,
                max_step=self.options.width_sample_step,
                tolerance=1e-9,
                cubic=self.options.cubic_profiles,
            ),
            superelevations=self._superelevation(chain, reference),
            road_type=tables.road_type_for(leader.subtype, leader.attributes.get("location", "")),
            speed=self._speed_for(leader),
            lanelet2_ids=tuple(lanelets[i].lanelet2_id for i in chain.lanelet_indices),
        )

        for group_index, group in enumerate(chain.groups):
            s = (
                0.0
                if group_index == 0
                else station_of_point(
                    reference, (group_refs[group_index][0][0], group_refs[group_index][0][1])
                )
            )
            road.lane_sections.append(
                self._build_section(group, group_refs[group_index], s, road_id)
            )

        members = [lanelets[i] for i in chain.lanelet_indices]
        road.signals = furniture.signals_for(road, reference, members, self.options)
        road.objects.extend(furniture.barriers_for(road, reference, members, self.options))
        road.lane_offsets = self._lane_offsets(chain.groups[0], reference)
        self._references[road.road_id] = reference
        self._link_lane_sections(road, chain)
        return road

    def _lane_offsets(self, group: grouping.LaneGroup, reference: list[Vec3]):
        """Where lane 0 sits relative to the reference line.

        Zero whenever the reference line *is* a boundary, so the default layout
        emits nothing. With a computed centreline no boundary lies exactly on it,
        and this records the difference instead of quietly mislocating the lanes.
        """
        if self.options.reference_line != "centerline":
            return []
        boundaries, _attributes, _owners = self.cross_section(group)
        center = self._center_index(boundaries, reference, group)
        stations_ = sample_stations(reference, self.options.width_sample_step)
        values = offsets_along(reference, stations_, boundaries[center])
        records = build_profile(stations_, values, tolerance=self.options.width_tolerance)
        if len(records) == 1 and abs(records[0].a) <= self.options.width_tolerance:
            return []
        return records

    # -- cross-section ------------------------------------------------------
    def cross_section(
        self, group: grouping.LaneGroup
    ) -> tuple[list[list[Vec3]], list[dict[str, str]], list[LaneletIR]]:
        """A lane group as an ordered left-to-right stack of boundaries.

        For members `m0..mn` the boundaries are `m0.left, m0.right, m1.right, …`,
        so lane `k` lies between boundary `k` and `k+1` and belongs to `m_k`.
        Expressing a cross-section this way is what lets the reference line sit
        anywhere in the stack rather than only at its left edge.

        That layout holds only while each member's left edge really is the
        previous member's right edge, and on a surveyed map it sometimes is not:
        a mapper may name both neighbours' shared bound "right", or name it left
        on one and right on the other, with no direction change to reveal it. So
        each join is checked by node identity, and a member whose edges turn out
        to be mirrored is turned round rather than stacked from the wrong sides.
        """
        lanelets = self.ir.lanelets
        boundaries: list[list[Vec3]] = []
        attributes: list[dict[str, str]] = []
        owners: list[LaneletIR] = []
        previous_key: BoundaryKey | None = None

        for position, member in enumerate(group.members):
            lanelet = lanelets[member]
            bounds = self.oriented_bounds(lanelet)
            left, right = bounds.left, bounds.right
            left_attrs, right_attrs = bounds.left_attributes, bounds.right_attributes
            left_key, right_key = bounds.left_key, bounds.right_key
            if group.reversed_[position]:
                # This lanelet faces the other way, so its own right edge is the
                # road's left edge, and its polylines have to be turned round to
                # run with the road's s-axis.
                left, right = list(reversed(right)), list(reversed(left))
                left_attrs, right_attrs = right_attrs, left_attrs
                left_key, right_key = right_key, left_key

            if previous_key is not None and left_key != previous_key:
                if right_key == previous_key:
                    # Mirrored relative to the group: the edge it calls its outer
                    # one is the edge it shares with its neighbour.
                    left, right = list(reversed(right)), list(reversed(left))
                    left_attrs, right_attrs = right_attrs, left_attrs
                    left_key, right_key = right_key, left_key
                else:
                    self._report_unshared(group.members[position - 1], member)

            if position == 0:
                boundaries.append(left)
                attributes.append(left_attrs)
            boundaries.append(right)
            attributes.append(right_attrs)
            owners.append(lanelet)
            previous_key = right_key

        return boundaries, attributes, owners

    def _report_unshared(self, previous: int, member: int) -> None:
        """Say when two lanelets placed side by side share no boundary at all.

        Adjacency was inferred from a shared bound, so reaching here means the
        pair shares one that neither of them has left over -- the cross-section
        is then built from unrelated edges and its widths mean nothing. Stating
        it is the least that can be done; the alternative of silently stacking
        them is what made this class of error invisible.
        """
        lanelets = self.ir.lanelets
        pair = (lanelets[previous].lanelet2_id, lanelets[member].lanelet2_id)
        if pair in self._unshared_reported:
            return
        self._unshared_reported.add(pair)
        self.bag.warn(
            W_STACK_NOT_SHARED,
            f"lanelets #{pair[0]} and #{pair[1]} are placed side by side in one "
            "cross-section but share no boundary between them, so the lane widths "
            "across that join are measured between unrelated edges",
        )

    def centre_boundary(self, group: grouping.LaneGroup) -> int | None:
        """For a two-way group, the boundary dividing the two carriageways.

        That boundary is where lane 0 has to sit: opposing lanes then fall on the
        `+` side and forward lanes on the `-` side, which is precisely what an
        OpenDRIVE left lane means. Any other choice puts traffic on the wrong
        side of the reference line, so this overrides `--reference-line`.

        Only the `reversed, ..., forward, ...` order can be divided this way, and
        that is the order right-hand traffic produces. The mirror order --
        forward lanes with something coming the other way to their *right*, a
        contraflow cycle track being the usual case -- has no dividing boundary
        at all: `+` means left, so a member that must be `+` cannot be on the
        right. Flipping the road's direction does not help, because it reverses
        the stack and inverts every flag at once and so preserves the order.
        `contraflow_on_the_right` names that case for reporting; here it just
        means there is nothing to return.
        """
        if not group.two_way:
            return None
        for position in range(len(group.members) - 1):
            if group.reversed_[position] and not group.reversed_[position + 1]:
                # Boundary `position + 1` separates member `position` from the
                # next one, and the direction changes across it.
                return position + 1
        return None

    @staticmethod
    def contraflow_on_the_right(group: grouping.LaneGroup) -> list[int]:
        """Members that travel against the road while lying right of a forward one.

        These are the ones the sign convention cannot describe. Returned rather
        than reported here so the caller can name the road they ended up in.
        """
        if not group.two_way:
            return []
        seen_forward = False
        out: list[int] = []
        for position, member in enumerate(group.members):
            if group.reversed_[position]:
                if seen_forward:
                    out.append(member)
            else:
                seen_forward = True
        return out

    def _report_contraflow(self, chain: grouping.RoadChain, road_id: int) -> None:
        """Say when a lane's id claims a direction the lanelet does not travel.

        Emitting it as `-1` and saying nothing is the harmful outcome: a consumer
        reads a right lane as running with `s`, and here it runs against it. The
        geometry is still the input's own, so the lane is in the right place --
        only the direction the sign implies is wrong.
        """
        for group in chain.groups:
            members = self.contraflow_on_the_right(group)
            if not members:
                continue
            names = ", ".join(f"#{self.ir.lanelets[i].lanelet2_id}" for i in members)
            self.bag.warn(
                W_CONTRAFLOW_RIGHT,
                f"road {road_id}: lanelet(s) {names} travel against the road but lie to "
                "the right of lanes that travel with it, which no lane id can express "
                "(`+` means left); they are emitted as right lanes, so their id reads as "
                "travelling with s when they do not",
            )

    def _report_pivot(self, chain: grouping.RoadChain, road_id: int) -> None:
        """Say when the reference line had to move off the leftmost boundary.

        It only moves for a corner pivot, and the move has a consequence worth
        stating: the reference then runs along the lanelet's *right* edge, so the
        lane sits to its left and is emitted as `+1`. Consumers read a left lane
        as carrying traffic against `s`, which here it does not -- the geometry is
        exact but that one convention cannot be honoured, because honouring it
        would need a reference line along an edge that is a single point.
        """
        boundaries, _attributes, _owners = self.cross_section(chain.groups[0])
        if self._first_extended(boundaries) == 0:
            return
        names = ", ".join(f"#{self.ir.lanelets[i].lanelet2_id}" for i in chain.lanelet_indices)
        self.bag.warn(
            W_PIVOT_REFERENCE,
            f"road {road_id} ({names}) pivots on a corner, so its inner edge is a single "
            "point and cannot be the reference line; the outer edge is used instead and "
            "the lane is emitted to its left as +1, which reads as travelling against s",
        )

    @staticmethod
    def _first_extended(boundaries: list[list[Vec3]]) -> int:
        """The first boundary that actually goes somewhere.

        A turn that pivots on a shared corner has a *single-point* inner bound --
        stored as a `[corner, corner]` line string, which lanelet2 accepts and
        which the tightest near-side turn of every intersection arm produces.
        Taking that as the reference line gives a road of zero length, so the
        reference moves along to a bound with a direction and the pivot becomes
        the lane's far edge instead. The lane is then the pie slice between them,
        which is the shape the input describes.
        """
        for index, bound in enumerate(boundaries):
            if len(dedupe(bound)) >= 2:
                return index
        return 0

    def _group_reference(self, group: grouping.LaneGroup) -> list[Vec3]:
        """The polyline the planView follows for this cross-section."""
        boundaries, _attributes, _owners = self.cross_section(group)
        divider = self.centre_boundary(group)
        if divider is not None:
            # Two-way: follow the boundary between the carriageways, so the two
            # directions land on opposite sides of the reference line.
            return boundaries[divider]
        if self.options.reference_line == "centerline" and len(boundaries) >= 2:
            # The centre of the whole cross-section, using lanelet2's own
            # algorithm so a script that reads `lanelet.centerline` and the
            # emitted reference line agree.
            return centerline_coords(boundaries[0], boundaries[-1]) or boundaries[0]
        return boundaries[self._first_extended(boundaries)]

    def _center_index(
        self,
        boundaries: list[list[Vec3]],
        reference: list[Vec3],
        group: grouping.LaneGroup | None = None,
    ) -> int:
        """Which boundary lane 0 sits on: the one nearest the reference line."""
        if group is not None:
            forced = self.centre_boundary(group)
            if forced is not None:
                return forced
        if self.options.reference_line != "centerline":
            # Lane 0 sits on whichever boundary the reference line follows, which
            # is not boundary 0 when that one is a corner pivot.
            return self._first_extended(boundaries)
        stations_ = sample_stations(reference, self.options.width_sample_step)
        offsets = [
            math.fsum(abs(t) for t in offsets_along(reference, stations_, bound))
            / max(len(stations_), 1)
            for bound in boundaries
        ]
        return min(range(len(offsets)), key=lambda i: offsets[i])

    def _superelevation(self, chain: grouping.RoadChain, reference: list[Vec3]):
        """Roll angle from the height difference across the widest cross-section."""
        boundaries, _attributes, _owners = self.cross_section(chain.groups[0])
        if len(boundaries) < 2:
            return []
        return road_superelevation(
            reference,
            boundaries[0],
            boundaries[-1],
            max_step=self.options.width_sample_step,
            cubic=self.options.cubic_profiles,
        )

    def _build_section(
        self,
        group: grouping.LaneGroup,
        local_reference: list[Vec3],
        s: float,
        road_id: int,
    ) -> LaneSectionSpec:
        boundaries, attributes, owners = self.cross_section(group)
        center = self._center_index(boundaries, local_reference, group)

        center_mark = self._road_mark(attributes[center], f"road {road_id} centre")
        section = LaneSectionSpec(s=s, center_road_mark=center_mark)

        stations_ = sample_stations(local_reference, self.options.width_sample_step)

        # Lanes above the reference line run outward as +1, +2, …; lanes below it
        # as -1, -2, …. With a boundary reference (`center == 0`) there is
        # nothing above, which is the single-sided layout the default produces.
        for position in range(len(owners)):
            inner_index, outer_index = position, position + 1
            if position < center:
                lane_id = center - position
                # Going left, the *inner* edge is the one nearer the reference.
                inner_index, outer_index = position + 1, position
            else:
                lane_id = -(position - center + 1)

            lane = self._build_lane(
                lane_id=lane_id,
                lanelet=owners[position],
                inner=boundaries[inner_index],
                outer=boundaries[outer_index],
                outer_attrs=attributes[outer_index],
                local_reference=local_reference,
                stations_=stations_,
            )
            (section.left if lane_id > 0 else section.right).append(lane)

        section.left.sort(key=lambda lane: lane.lane_id)
        section.right.sort(key=lambda lane: -lane.lane_id)
        return section

    def _road_mark(self, attributes: dict[str, str], where: str):
        mark, known = tables.road_mark_for(attributes, self.options)
        if not known:
            self.bag.warn(
                W_UNKNOWN_ROADMARK,
                f"{where}: boundary tags {mark.source!r} have no roadMark mapping; "
                "emitted as the nearest equivalent",
            )
        return mark

    def _build_lane(
        self,
        *,
        lane_id: int,
        lanelet: LaneletIR,
        inner: list[Vec3],
        outer: list[Vec3],
        outer_attrs: dict[str, str],
        local_reference: list[Vec3],
        stations_: list[float],
    ) -> LaneSpec:
        widths, minimum = lane_widths(
            local_reference,
            inner,
            outer,
            tolerance=self.options.width_tolerance,
            stations_=stations_,
            cubic=self.options.cubic_profiles,
        )
        if minimum < 0.0:
            self.bag.warn(
                W_NEGATIVE_WIDTH,
                f"lanelet #{lanelet.lanelet2_id}: bounds cross over "
                f"(minimum width {minimum:.4g} m)",
            )

        lane_type, recognised = tables.lane_type_for(lanelet.subtype, one_way=lanelet.one_way)
        if not lanelet.subtype:
            self.bag.warn(
                W_EMPTY_SUBTYPE,
                f"lanelet #{lanelet.lanelet2_id} has no 'subtype' tag; "
                f"assuming lane type {lane_type!r}",
            )
        elif not recognised:
            self.bag.warn(
                W_UNKNOWN_SUBTYPE,
                f"lanelet #{lanelet.lanelet2_id}: unknown subtype "
                f"{lanelet.subtype!r}; assuming lane type {lane_type!r}",
            )

        return LaneSpec(
            lane_id=lane_id,
            lane_type=lane_type,
            widths=widths,
            road_mark=self._road_mark(outer_attrs, f"lanelet #{lanelet.lanelet2_id}"),
            lanelet2_id=lanelet.lanelet2_id,
            subtype=lanelet.subtype,
        )

    def _link_lane_sections(self, road: RoadSpec, chain: grouping.RoadChain) -> None:
        """Link lanes across consecutive sections by *lanelet* succession.

        A road may change lane count between sections, so matching lane ids
        would link the wrong lanes -- or miss a lane that shifted from -2 to -1
        when its neighbour ended. The lanelet successor relation is the ground
        truth, and it is what carried the sections into one road in the first
        place.
        """
        ids = {lanelet.lanelet2_id: index for index, lanelet in enumerate(self.ir.lanelets)}

        for previous, following in pairwise(road.lane_sections):
            by_lanelet = {lane.lanelet2_id: lane for lane in following.lanes}
            for lane in previous.lanes:
                index = ids.get(lane.lanelet2_id)
                if index is None:
                    continue
                for successor in self.rels.successor_of(index):
                    counterpart = by_lanelet.get(self.ir.lanelets[successor].lanelet2_id)
                    if counterpart is not None:
                        lane.successor = counterpart.lane_id
                        counterpart.predecessor = lane.lane_id
                        break
        del chain

    def _speed_for(self, lanelet: LaneletIR) -> tuple[float, str] | None:
        raw = lanelet.attributes.get("speed_limit", "")
        if not raw:
            return None
        parsed = tables.speed_for(raw)
        if parsed is None:
            self.bag.warn(
                W_BAD_SPEED_LIMIT,
                f"lanelet #{lanelet.lanelet2_id}: cannot parse speed_limit {raw!r}; "
                "no <speed> record emitted",
            )
        return parsed

    @staticmethod
    def _road_name(chain: grouping.RoadChain, lanelets: list[LaneletIR]) -> str:
        ids = [lanelets[i].lanelet2_id for i in chain.lanelet_indices]
        if len(ids) == 1:
            return f"lanelet_{ids[0]}"
        return f"lanelets_{ids[0]}_{ids[-1]}"


def _link_roads(
    network: grouping.Network,
    rels: relations.Relations,
    roads: list[RoadSpec | None],
    ir: MapIR,
    bag: DiagnosticBag,
) -> None:
    """Connect roads whose ends correspond one-for-one.

    Chains break at a change in lane count as well as at a branch. A widening
    like 1 -> 2 lanes is still an unambiguous road link even though it is not a
    single road, so it is worth emitting; a genuine branch is not, and is left
    for junction support.

    Individual *lanes* are linked across the boundary too. The backend can infer
    those itself, but only by comparing geometry, and it refuses outright when the
    two roads carry different lane counts -- which a widening precisely is. Here
    the correspondence is already known exactly, lanelet by lanelet, so it is
    written down rather than re-derived.
    """
    del bag  # branch points are already reported once, in _report_topology

    group_to_chain: dict[int, int] = {}
    for chain_index, chain in enumerate(network.chains):
        for group in chain.groups:
            group_to_chain[id(group)] = chain_index

    head_group = {id(chain.groups[0]): i for i, chain in enumerate(network.chains) if chain.groups}
    tail_group = {id(chain.groups[-1]): i for i, chain in enumerate(network.chains) if chain.groups}

    for tail_id, chain_index in tail_group.items():
        chain = network.chains[chain_index]
        last = chain.groups[-1]
        if id(last) != tail_id:
            continue

        successor_groups: set[int] = set()
        ok = True
        for member in last.members:
            following = rels.successor_of(member)
            if len(following) != 1:
                ok = False
                break
            group_index = network.group_of.get(following[0])
            if group_index is None:
                ok = False
                break
            successor_groups.add(group_index)
        if not ok or len(successor_groups) != 1:
            continue

        target_group = network.groups[successor_groups.pop()]
        target_chain = head_group.get(id(target_group))
        if target_chain is None or target_chain == chain_index:
            continue
        # Only link when nothing else merges into the target.
        if any(len(rels.predecessor_of(m)) != 1 for m in target_group.members):
            continue

        source_road = roads[chain_index]
        target_road = roads[target_chain]
        if source_road is None or target_road is None:
            continue
        source_road.successor = LinkSpec("road", target_road.road_id, "start")
        target_road.predecessor = LinkSpec("road", source_road.road_id, "end")
        _link_lanes_across(source_road, target_road, last, rels, ir)


def _link_lanes_across(
    source: RoadSpec,
    target: RoadSpec,
    last: grouping.LaneGroup,
    rels: relations.Relations,
    ir: MapIR,
) -> None:
    """Point each lane at the lane it continues into on the next road."""
    outgoing = {lane.lanelet2_id: lane for lane in source.lane_sections[-1].lanes}
    incoming = {lane.lanelet2_id: lane for lane in target.lane_sections[0].lanes}

    for member in last.members:
        following = rels.successor_of(member)
        if len(following) != 1:
            continue
        here = outgoing.get(ir.lanelets[member].lanelet2_id)
        there = incoming.get(ir.lanelets[following[0]].lanelet2_id)
        if here is None or there is None:
            continue
        here.successor = there.lane_id
        there.predecessor = here.lane_id
