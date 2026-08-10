"""Decomposing lanelets into OpenDRIVE roads.

Two orthogonal groupings, applied in order:

* **Lateral** -- lanelets that share boundaries side by side form a *lane group*,
  ordered left to right. One group is one cross-section.
* **Longitudinal** -- groups that follow one another unambiguously form a *road
  chain*. Each maximal chain becomes one `<road>`; each group inside it becomes
  one `<laneSection>` at the accumulated `s`. The lane *count* may change from
  one section to the next -- that is exactly what a widening or a lane drop is,
  and OpenDRIVE expresses it by lane sections of differing width linked lane by
  lane.

That mapping is not a convenience, it is what the OpenDRIVE constructs mean: a
multi-`laneSection` road *is* a run of consecutive cross-sections with the same
lane count.

A chain stops where the correspondence becomes ambiguous: at a branch, at a
merge, or where lateral order would have to cross. Those points end a road, and
`mapping/junctions.py` joins the resulting roads through a `<junction>`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir.model import LaneletIR
from .relations import Relations


@dataclass
class LaneGroup:
    """One cross-section: lanelet indices ordered left to right.

    "Left" is relative to the *group's* forward direction, which is the
    direction of its first member. A member travelling the other way is marked
    in `reversed_`, and its own left/right naming is mirrored relative to the
    group -- what it calls its right edge is the group's left edge.
    """

    members: list[int] = field(default_factory=list)
    reversed_: list[bool] = field(default_factory=list)
    """Per member: does it travel against the group's forward direction?"""

    def __len__(self) -> int:
        return len(self.members)

    def __post_init__(self) -> None:
        if not self.reversed_:
            self.reversed_ = [False] * len(self.members)

    @property
    def two_way(self) -> bool:
        return any(self.reversed_)

    def is_reversed(self, member: int) -> bool:
        return self.reversed_[self.members.index(member)]


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
    network.two_way_pairs = sorted(relations.opposing)
    _build_groups(lanelets, relations, network)
    _build_chains(relations, network)
    network.branch_lanelets = sorted(i for i in range(len(lanelets)) if relations.is_branch(i))
    return network


def _neighbour_chain(relations: Relations, start: int) -> list[tuple[int, bool]]:
    """Walk right from `start`, crossing into opposing traffic when it is there.

    Returns `(index, reversed)` pairs. A two-way road is one cross-section: the
    opposing carriageway is reached through the boundary the two share, and from
    there the walk continues in *its* left direction, because right-for-them is
    left-for-us.
    """
    out: list[tuple[int, bool]] = [(start, False)]
    seen = {start}

    current, flipped = start, False
    while True:
        # Within one carriageway, "further right" is right_of; once flipped, the
        # group's rightward direction is that lanelet's own left.
        following = (relations.left_of if flipped else relations.right_of).get(current)
        if following is None:
            # Step across to the opposing carriageway, if there is one.
            following = relations.opposing_of.get(current)
            if following is None or following in seen:
                return out
            flipped = not flipped
        if following in seen:
            return out
        seen.add(following)
        out.append((following, flipped))
        current = following


def _leftmost(relations: Relations, index: int) -> tuple[int, bool]:
    """Walk left from `index` to the edge of its cross-section."""
    current, flipped = index, False
    seen = {index}
    while True:
        preceding = (relations.right_of if flipped else relations.left_of).get(current)
        if preceding is None:
            preceding = relations.opposing_of.get(current)
            if preceding is None or preceding in seen:
                return current, flipped
            # Crossing the centre line flips the sense of left and right.
            flipped = not flipped
        if preceding in seen:
            return current, flipped
        seen.add(preceding)
        current = preceding


def _build_groups(lanelets: list[LaneletIR], relations: Relations, network: Network) -> None:
    assigned: set[int] = set()

    for index in range(len(lanelets)):
        if index in assigned:
            continue
        start, start_flipped = _leftmost(relations, index)
        if start in assigned:
            continue

        walk = [
            (member, flipped)
            for member, flipped in _neighbour_chain(relations, start)
            if member not in assigned
        ]
        if not walk:
            continue

        # Either direction is a valid s-axis for a two-way road, so the only
        # thing that matters is picking one deterministically: the direction of
        # the lanelet that seeded this group, which is the first in IR order not
        # already claimed. `start_flipped` says whether the walk's origin faces
        # the other way, so subtracting it re-expresses every flag in the
        # seed's terms.
        forward_flip = start_flipped
        members = [member for member, _ in walk]
        flags = [flipped != forward_flip for _, flipped in walk]

        for member in members:
            assigned.add(member)

        group = LaneGroup(members, flags)
        network.group_of.update({member: len(network.groups) for member in members})
        network.groups.append(group)


def _next_group(relations: Relations, network: Network, group_index: int) -> int | None:
    """The group that continues this one, if the correspondence is exactly 1:1.

    Every condition here is load-bearing. Anything looser and two roads that
    merely touch would be welded into one road with a discontinuous cross-section.
    """
    group = network.groups[group_index]

    # A member with no successor is a lane that simply ends -- a lane drop, which
    # is a lane-section change, not a branch. A member with several is a genuine
    # branch and belongs to a junction.
    # For a member travelling against the group, "onward along the road" is its
    # *predecessor*: it drives towards the road's start, not away from it.
    def onward(member: int, reversed_: bool) -> list[int]:
        return relations.predecessor_of(member) if reversed_ else relations.successor_of(member)

    def backward(member: int, reversed_: bool) -> list[int]:
        return relations.successor_of(member) if reversed_ else relations.predecessor_of(member)

    successors: list[int] = []
    for position, member in enumerate(group.members):
        following = onward(member, group.reversed_[position])
        if len(following) > 1:
            return None
        successors.extend(following)
    if not successors:
        return None

    candidate_indices = {network.group_of.get(s) for s in successors}
    if len(candidate_indices) != 1:
        return None
    candidate = candidate_indices.pop()
    if candidate is None or candidate == group_index:
        return None

    target = network.groups[candidate]

    # Lateral order must survive: lanes may appear or disappear, but they may
    # not cross over one another, or the lane links would be nonsense.
    carried = set(successors)
    if successors != [member for member in target.members if member in carried]:
        return None

    # Nothing outside this group may feed the target, or the join is a merge and
    # belongs in a junction rather than inside one road.
    for position, member in enumerate(target.members):
        preceding = backward(member, target.reversed_[position])
        if len(preceding) > 1:
            return None
        if preceding and network.group_of.get(preceding[0]) != group_index:
            return None

    # A two-way road must keep facing the same way along its whole length, or the
    # lane signs would flip mid-road.
    if group.two_way or target.two_way:
        carried_flags = {
            member: group.reversed_[group.members.index(member)] for member in group.members
        }
        for position, member in enumerate(target.members):
            source = backward(member, target.reversed_[position])
            if source and carried_flags.get(source[0]) not in (None, target.reversed_[position]):
                return None

    return candidate


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
