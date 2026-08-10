"""Turning branch points into OpenDRIVE junctions.

A lanelet2 branch is just several lanelets sharing an end node; OpenDRIVE wants
that expressed as a `<junction>` with an incoming road and a set of *connecting*
roads carrying `junction="<id>"`.

The nice property of working from lanelet2 is that the branches already have
exact geometry. Junction builders in the usual OpenDRIVE toolchains *synthesize*
connecting-road geometry from radii and angles, because they are typically fed
only the incoming roads. Here each branch lanelet is already a road with real
coordinates, so it simply becomes the connecting road unchanged -- nothing is
invented, and the junction is exact.

Two shapes are recognised, and they are mirror images:

* **Divergence** -- one road ends where several begin. The stem is the incoming
  road; each branch becomes a connecting road.
* **Convergence** -- several roads end where one begins. Each branch is an
  incoming road; the stem becomes the connecting road.

Anything more tangled (a branch that is itself a merge, overlapping lanelets in
an intersection interior) is left alone and reported, because guessing at it
would produce a junction that looks authoritative and is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..diagnostics import I_JUNCTION_SKIPPED, DiagnosticBag
from ..ir.model import MapIR
from ..odr.model import ConnectionSpec, JunctionSpec, LinkSpec, RoadSpec
from ..topology import grouping, relations


@dataclass
class _Site:
    kind: str
    """"diverge" or "converge"."""
    stem: int
    """Chain index of the single road."""
    branches: list[int]
    """Chain indices of the several roads."""


def _unique(values: list[int]) -> list[int]:
    """Order-preserving dedupe, so junction ids are stable across runs."""
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _chain_of_lanelet(network: grouping.Network) -> dict[int, int]:
    return {
        lanelet: index
        for index, chain in enumerate(network.chains)
        for lanelet in chain.lanelet_indices
    }


def find_sites(network: grouping.Network, rels: relations.Relations) -> list[_Site]:
    owner = _chain_of_lanelet(network)
    sites: list[_Site] = []

    for index, chain in enumerate(network.chains):
        if not chain.groups:
            continue

        following = _unique(
            [
                owner[s]
                for member in chain.groups[-1].members
                for s in rels.successor_of(member)
                if s in owner
            ]
        )
        if len(following) > 1:
            sites.append(_Site("diverge", index, following))

        preceding = _unique(
            [
                owner[p]
                for member in chain.groups[0].members
                for p in rels.predecessor_of(member)
                if p in owner
            ]
        )
        if len(preceding) > 1:
            sites.append(_Site("converge", index, preceding))

    return sites


def _lane_of_lanelet(road: RoadSpec, *, first: bool) -> dict[int, int]:
    """lanelet2 id -> OpenDRIVE lane id, in the road's first or last section."""
    if not road.lane_sections:
        return {}
    section = road.lane_sections[0 if first else -1]
    return {lane.lanelet2_id: lane.lane_id for lane in section.lanes}


def build(
    network: grouping.Network,
    rels: relations.Relations,
    roads: list[RoadSpec | None],
    ir: MapIR,
    bag: DiagnosticBag,
) -> list[JunctionSpec]:
    """Attach junctions to `roads` in place and return them."""
    owner = _chain_of_lanelet(network)
    junctions: list[JunctionSpec] = []
    claimed: set[int] = set()

    for site in find_sites(network, rels):
        stem = roads[site.stem]
        branches = [(index, roads[index]) for index in site.branches]
        if stem is None or any(road is None for _index, road in branches):
            continue

        # A road can only belong to one junction; a second claim would overwrite
        # its junction id and silently corrupt the first.
        involved = {site.stem, *site.branches}
        if involved & claimed:
            names = ", ".join(f"#{i}" for i in sorted(involved & claimed))
            bag.info(
                I_JUNCTION_SKIPPED,
                f"road(s) {names} already belong to a junction; the overlapping "
                "branch point was left unconnected rather than reassigned",
            )
            continue
        claimed |= involved

        junction_id = len(junctions) + 1
        junction = JunctionSpec(junction_id=junction_id, name=f"junction_{junction_id}")

        if site.kind == "diverge":
            _wire_diverge(junction, stem, branches, network, rels, owner, site, ir)
        else:
            _wire_converge(junction, stem, branches, network, rels, owner, site, ir)

        junctions.append(junction)

    return junctions


def _wire_diverge(
    junction: JunctionSpec,
    stem: RoadSpec,
    branches: list[tuple[int, RoadSpec]],
    network: grouping.Network,
    rels: relations.Relations,
    owner: dict[int, int],
    site: _Site,
    ir: MapIR,
) -> None:
    stem.successor = LinkSpec("junction", junction.junction_id)
    stem_lanes = _lane_of_lanelet(stem, first=False)

    for chain_index, branch in branches:
        branch.junction = junction.junction_id
        branch.predecessor = LinkSpec("road", stem.road_id, "end")
        branch_lanes = _lane_of_lanelet(branch, first=True)

        links: list[tuple[int, int]] = []
        for member in network.chains[site.stem].groups[-1].members:
            for successor in rels.successor_of(member):
                if owner.get(successor) != chain_index:
                    continue
                incoming = stem_lanes.get(ir.lanelets[member].lanelet2_id)
                outgoing = branch_lanes.get(ir.lanelets[successor].lanelet2_id)
                if incoming is not None and outgoing is not None:
                    links.append((incoming, outgoing))

        junction.connections.append(
            ConnectionSpec(
                incoming_road=stem.road_id,
                connecting_road=branch.road_id,
                contact_point="start",
                lane_links=links,
            )
        )


def _wire_converge(
    junction: JunctionSpec,
    stem: RoadSpec,
    branches: list[tuple[int, RoadSpec]],
    network: grouping.Network,
    rels: relations.Relations,
    owner: dict[int, int],
    site: _Site,
    ir: MapIR,
) -> None:
    # Mirror image: the several roads are incoming, the single one connects them.
    stem.junction = junction.junction_id
    stem_lanes = _lane_of_lanelet(stem, first=True)

    for chain_index, branch in branches:
        branch.successor = LinkSpec("junction", junction.junction_id)
        branch_lanes = _lane_of_lanelet(branch, first=False)

        links: list[tuple[int, int]] = []
        for member in network.chains[site.stem].groups[0].members:
            for predecessor in rels.predecessor_of(member):
                if owner.get(predecessor) != chain_index:
                    continue
                incoming = branch_lanes.get(ir.lanelets[predecessor].lanelet2_id)
                outgoing = stem_lanes.get(ir.lanelets[member].lanelet2_id)
                if incoming is not None and outgoing is not None:
                    links.append((incoming, outgoing))

        junction.connections.append(
            ConnectionSpec(
                incoming_road=branch.road_id,
                connecting_road=stem.road_id,
                contact_point="start",
                lane_links=links,
            )
        )
