"""Decomposing lanelets into OpenDRIVE roads.

Two orthogonal groupings, applied in order:

* **Lateral** -- lanelets that share boundaries side by side form a *lane group*,
  ordered left to right. One group is one cross-section.
* **Longitudinal** -- groups that follow one another one-for-one form a *road
  chain*. Each maximal chain becomes one `<road>`; each group inside it becomes
  one `<laneSection>` at the accumulated `s`.

That mapping is not a convenience, it is what the OpenDRIVE constructs mean: a
multi-`laneSection` road *is* a run of consecutive cross-sections with the same
lane count.

A chain stops wherever the one-for-one correspondence does: at a branch, at a
merge, or at a change in lane count. In this release those points end a road and
are reported; joining them through junctions is the next phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir.model import LaneletIR
from .relations import Relations


@dataclass
class LaneGroup:
    """One cross-section: lanelet indices ordered left to right."""

    members: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.members)


@dataclass
class RoadChain:
    """A run of cross-sections that becomes a single `<road>`."""

    groups: list[LaneGroup] = field(default_factory=list)

    @property
    def lanelet_indices(self) -> list[int]:
        return [index for group in self.groups for index in group.members]

    @property
    def width(self) -> int:
        return len(self.groups[0]) if self.groups else 0


@dataclass
class Network:
    groups: list[LaneGroup] = field(default_factory=list)
    chains: list[RoadChain] = field(default_factory=list)
    group_of: dict[int, int] = field(default_factory=dict)
    """Lanelet index -> index into `groups`."""
    branch_lanelets: list[int] = field(default_factory=list)
    two_way_pairs: list[tuple[int, int]] = field(default_factory=list)


def build(lanelets: list[LaneletIR], relations: Relations) -> Network:
    network = Network()
    network.two_way_pairs = sorted(relations.antiparallel)
    _build_groups(lanelets, relations, network)
    _build_chains(relations, network)
    network.branch_lanelets = sorted(i for i in range(len(lanelets)) if relations.is_branch(i))
    return network


def _same_direction_neighbour(relations: Relations, index: int) -> int | None:
    """The lanelet immediately right of `index`, if it travels the same way.

    Opposing-direction neighbours share a boundary too, but they belong on the
    other side of the reference line as positive-id lanes. That is Phase 2, so
    here they simply do not group.
    """
    neighbour = relations.right_of.get(index)
    if neighbour is None:
        return None
    pair = (min(index, neighbour), max(index, neighbour))
    if pair in relations.antiparallel:
        return None
    return neighbour


def _build_groups(lanelets: list[LaneletIR], relations: Relations, network: Network) -> None:
    assigned: set[int] = set()

    def leftmost(index: int) -> int:
        seen = {index}
        current = index
        while True:
            left = relations.left_of.get(current)
            if left is None or left in seen:
                return current
            pair = (min(current, left), max(current, left))
            if pair in relations.antiparallel:
                return current
            seen.add(left)
            current = left

    for index in range(len(lanelets)):
        if index in assigned:
            continue
        start = leftmost(index)
        if start in assigned:
            continue

        members: list[int] = []
        current: int | None = start
        while current is not None and current not in assigned:
            members.append(current)
            assigned.add(current)
            current = _same_direction_neighbour(relations, current)

        group = LaneGroup(members)
        network.group_of.update({member: len(network.groups) for member in members})
        network.groups.append(group)


def _next_group(relations: Relations, network: Network, group_index: int) -> int | None:
    """The group that continues this one, if the correspondence is exactly 1:1.

    Every condition here is load-bearing. Anything looser and two roads that
    merely touch would be welded into one road with a discontinuous cross-section.
    """
    group = network.groups[group_index]

    successors: list[int] = []
    for member in group.members:
        following = relations.successor_of(member)
        if len(following) != 1:
            return None
        successors.append(following[0])

    candidate_indices = {network.group_of.get(s) for s in successors}
    if len(candidate_indices) != 1:
        return None
    candidate = candidate_indices.pop()
    if candidate is None or candidate == group_index:
        return None

    target = network.groups[candidate]
    # Same width, same lateral order, and nothing else feeding into it.
    if target.members != successors:
        return None
    return (
        candidate
        if all(len(relations.predecessor_of(member)) == 1 for member in target.members)
        else None
    )


def _build_chains(relations: Relations, network: Network) -> None:
    next_of: dict[int, int] = {}
    for group_index in range(len(network.groups)):
        following = _next_group(relations, network, group_index)
        if following is not None:
            next_of[group_index] = following

    previous_of: dict[int, int] = {}
    for source, target in next_of.items():
        # A group reachable from two places starts its own chain instead.
        if target in previous_of:
            previous_of[target] = -1
        else:
            previous_of[target] = source

    visited: set[int] = set()
    for group_index in range(len(network.groups)):
        if group_index in visited:
            continue
        if previous_of.get(group_index, -1) != -1:
            continue  # not a chain head; it will be reached from its predecessor

        chain = RoadChain()
        current: int | None = group_index
        while current is not None and current not in visited:
            visited.add(current)
            chain.groups.append(network.groups[current])
            following = next_of.get(current)
            # Only continue when the successor is reached solely from here.
            current = following if previous_of.get(following, -1) == current else None
        network.chains.append(chain)

    # Any group left over sits inside a cycle; give each its own single-section road.
    for group_index in range(len(network.groups)):
        if group_index not in visited:
            visited.add(group_index)
            network.chains.append(RoadChain([network.groups[group_index]]))
