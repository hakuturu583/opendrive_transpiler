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

from itertools import pairwise

from ..config import TranspileOptions
from ..diagnostics import (
    I_AREA_SKIPPED,
    I_GEO_REFERENCE,
    I_JUNCTION_SKIPPED,
    I_POLYGON_SKIPPED,
    I_REGELEM_SKIPPED,
    I_TWO_WAY,
    W_BAD_SPEED_LIMIT,
    W_BOUNDS_SWAPPED,
    W_EMPTY_SUBTYPE,
    W_NEGATIVE_WIDTH,
    W_SHORT_ROAD,
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
from . import furniture, junctions, tables


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

    _apply_geo_reference(ir, model, bag, options)

    index = NodeIndex(ir.lanelets, options.point_tolerance)
    rels = relations.infer(ir.lanelets, index)
    network = grouping.build(ir.lanelets, rels)

    _report_topology(network, ir, bag, options)

    builder = _RoadBuilder(ir, options, bag, rels)
    chain_of_group: dict[int, int] = {}
    for chain_index, chain in enumerate(network.chains):
        for group in chain.groups:
            chain_of_group[id(group)] = chain_index

    roads: list[RoadSpec | None] = []
    for chain_index, chain in enumerate(network.chains):
        roads.append(builder.build_road(chain, road_id=chain_index + 1))

    _link_roads(network, rels, roads, bag)

    _attach_furniture(builder, roads, ir, options)

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

    stats.lanelets_skipped = stats.lanelets_in - stats.lanelets_converted
    _report_furniture(model, ir, stats, bag, options)
    return model, stats


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

    signalled = {signal.lanelet2_id for road in model.roads for signal in road.signals}
    unconverted = [r for r in ir.regelems if r.lanelet2_id not in signalled]
    stats.regelems_skipped = len(unconverted)
    if unconverted:
        kinds = sorted({r.kind for r in unconverted})
        reason = (
            "; signal output is disabled"
            if not options.signals
            else "; these kinds have no <signal> equivalent (priority rules need "
            "junction <priority>, which the backend does not model)"
        )
        bag.info(
            I_REGELEM_SKIPPED,
            f"{len(unconverted)} regulatory element(s) not converted ({', '.join(kinds)}){reason}",
        )


def _attach_furniture(builder, roads, ir: MapIR, options: TranspileOptions) -> None:
    """Give every area and polygon to the road it lies nearest.

    Map furniture belongs to no road in lanelet2, so one has to be chosen. The
    nearest reference line is the only defensible answer, and attaching each to
    exactly one road avoids the same parking bay appearing three times.
    """
    if not options.objects or not (ir.areas or ir.polygons):
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
            f"#{ir.lanelets[right].lanelet2_id} share a boundary in opposite directions; "
            "opposing lanes are emitted as separate roads rather than +/- lanes",
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
    ) -> None:
        self.ir = ir
        self.options = options
        self.bag = bag
        self.rels = rels
        self._swap_reported: set[int] = set()
        self._references: dict[int, list[Vec3]] = {}

    # -- orientation -------------------------------------------------------
    def oriented_bounds(self, lanelet: LaneletIR) -> tuple[list[Vec3], list[Vec3], dict, dict]:
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
            return right_coords, left_coords, right.attributes, left.attributes
        return left_coords, right_coords, left.attributes, right.attributes

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

        concatenated: list[Vec3] = []
        for piece in group_refs:
            concatenated.extend(piece if not concatenated else piece[1:])
        concatenated = dedupe(concatenated)

        if len(concatenated) < 2:
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

        road.signals = furniture.signals_for(
            road, reference, [lanelets[i] for i in chain.lanelet_indices], self.options
        )
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
        center = self._center_index(boundaries, reference)
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
        """
        lanelets = self.ir.lanelets
        boundaries: list[list[Vec3]] = []
        attributes: list[dict[str, str]] = []
        owners: list[LaneletIR] = []

        for position, member in enumerate(group.members):
            lanelet = lanelets[member]
            left, right, left_attrs, right_attrs = self.oriented_bounds(lanelet)
            if position == 0:
                boundaries.append(left)
                attributes.append(left_attrs)
            boundaries.append(right)
            attributes.append(right_attrs)
            owners.append(lanelet)

        return boundaries, attributes, owners

    def _group_reference(self, group: grouping.LaneGroup) -> list[Vec3]:
        """The polyline the planView follows for this cross-section."""
        boundaries, _attributes, _owners = self.cross_section(group)
        if self.options.reference_line == "centerline" and len(boundaries) >= 2:
            # The centre of the whole cross-section, using lanelet2's own
            # algorithm so a script that reads `lanelet.centerline` and the
            # emitted reference line agree.
            return centerline_coords(boundaries[0], boundaries[-1]) or boundaries[0]
        return boundaries[0]

    def _center_index(self, boundaries: list[list[Vec3]], reference: list[Vec3]) -> int:
        """Which boundary lane 0 sits on: the one nearest the reference line."""
        if self.options.reference_line != "centerline":
            return 0
        stations_ = sample_stations(reference, self.options.width_sample_step)
        offsets = [
            sum(abs(t) for t in offsets_along(reference, stations_, bound)) / max(len(stations_), 1)
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
        center = self._center_index(boundaries, local_reference)

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

        lane_type, recognised = tables.lane_type_for(lanelet.subtype)
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
    bag: DiagnosticBag,
) -> None:
    """Connect roads whose ends correspond one-for-one.

    Chains break at a change in lane count as well as at a branch. A widening
    like 1 -> 2 lanes is still an unambiguous road link even though it is not a
    single road, so it is worth emitting; a genuine branch is not, and is left
    for junction support.
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
