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

from ..config import TranspileOptions
from ..diagnostics import (
    I_GEO_REFERENCE,
    I_JUNCTION_SKIPPED,
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
from ..geometry.polyline import dedupe, sample_stations, station_of_point, total_length
from ..geometry.profile import lane_widths, road_elevation
from ..geometry.vec import Vec3
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
from . import tables


def build_model(
    ir: MapIR, bag: DiagnosticBag, options: TranspileOptions
) -> tuple[OdrModel, TranspileStats]:
    stats = TranspileStats(
        lanelets_in=len(ir.lanelets),
        areas_skipped=len(ir.areas),
        polygons_skipped=len(ir.polygons),
        regelems_skipped=len(ir.regelems),
    )
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

    _report_topology(network, ir, bag)

    builder = _RoadBuilder(ir, options, bag)
    chain_of_group: dict[int, int] = {}
    for chain_index, chain in enumerate(network.chains):
        for group in chain.groups:
            chain_of_group[id(group)] = chain_index

    roads: list[RoadSpec | None] = []
    for chain_index, chain in enumerate(network.chains):
        roads.append(builder.build_road(chain, road_id=chain_index + 1))

    _link_roads(network, rels, roads, bag)

    for road in roads:
        if road is None:
            continue
        model.roads.append(road)
        stats.roads += 1
        stats.lane_sections += len(road.lane_sections)
        stats.lanes += sum(len(section.lanes) for section in road.lane_sections)
        stats.lanelets_converted += len(road.lanelet2_ids)

    stats.lanelets_skipped = stats.lanelets_in - stats.lanelets_converted
    return model, stats


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


def _report_topology(network: grouping.Network, ir: MapIR, bag: DiagnosticBag) -> None:
    if network.branch_lanelets:
        names = ", ".join(f"#{ir.lanelets[i].lanelet2_id}" for i in network.branch_lanelets)
        bag.info(
            I_JUNCTION_SKIPPED,
            f"branch/merge at lanelet(s) {names}; each branch ends a road and junction "
            "generation is not enabled in this release, so these roads are emitted "
            "unconnected",
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
    def __init__(self, ir: MapIR, options: TranspileOptions, bag: DiagnosticBag) -> None:
        self.ir = ir
        self.options = options
        self.bag = bag
        self._swap_reported: set[int] = set()

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

        group_refs: list[list[Vec3]] = []
        for group in chain.groups:
            leader = lanelets[group.members[0]]
            reference, _right, _la, _ra = self.oriented_bounds(leader)
            group_refs.append(
                merge_collinear(
                    reference,
                    heading_tolerance=self.options.heading_tolerance,
                    chord_tolerance=self.options.chord_tolerance,
                )
            )

        concatenated: list[Vec3] = []
        for piece in group_refs:
            concatenated.extend(piece if not concatenated else piece[1:])
        concatenated = dedupe(concatenated)

        if len(concatenated) < 2:
            return None

        geometries, reference = build_plan_view(
            concatenated,
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
            ),
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

        _link_lane_sections(road)
        return road

    def _build_section(
        self,
        group: grouping.LaneGroup,
        local_reference: list[Vec3],
        s: float,
        road_id: int,
    ) -> LaneSectionSpec:
        lanelets = self.ir.lanelets
        leader = lanelets[group.members[0]]
        _left, _right, left_attrs, _right_attrs = self.oriented_bounds(leader)

        center_mark, known = tables.road_mark_for(left_attrs, self.options)
        if not known:
            self.bag.warn(
                W_UNKNOWN_ROADMARK,
                f"road {road_id}: boundary tags {center_mark.source!r} have no roadMark "
                "mapping; emitted as the nearest equivalent",
            )
        section = LaneSectionSpec(s=s, center_road_mark=center_mark)

        stations_ = sample_stations(local_reference, self.options.width_sample_step)

        for offset, member in enumerate(group.members, start=1):
            lanelet = lanelets[member]
            inner, outer, _inner_attrs, outer_attrs = self.oriented_bounds(lanelet)
            widths, minimum = lane_widths(
                local_reference,
                inner,
                outer,
                tolerance=self.options.width_tolerance,
                stations_=stations_,
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

            mark, known_mark = tables.road_mark_for(outer_attrs, self.options)
            if not known_mark:
                self.bag.warn(
                    W_UNKNOWN_ROADMARK,
                    f"lanelet #{lanelet.lanelet2_id}: boundary tags {mark.source!r} have "
                    "no roadMark mapping; emitted as the nearest equivalent",
                )

            section.right.append(
                LaneSpec(
                    lane_id=-offset,
                    lane_type=lane_type,
                    widths=widths,
                    road_mark=mark,
                    lanelet2_id=lanelet.lanelet2_id,
                    subtype=lanelet.subtype,
                )
            )

        return section

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


def _link_lane_sections(road: RoadSpec) -> None:
    """Link matching lane ids across consecutive lane sections of one road.

    Chains are built with a constant lane count, so lane -k always continues as
    lane -k; anything else would have ended the chain.
    """
    for previous, following in zip(road.lane_sections, road.lane_sections[1:], strict=False):
        by_id = {lane.lane_id: lane for lane in following.lanes}
        for lane in previous.lanes:
            counterpart = by_id.get(lane.lane_id)
            if counterpart is not None:
                lane.successor = counterpart.lane_id
                counterpart.predecessor = lane.lane_id


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
