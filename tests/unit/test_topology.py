"""Topology inference, on hand-built maps with known answers.

lanelet2 stores no relations, so everything here is derived from geometry and
object identity. These tests pin the derivation: which lanelets follow which,
which sit side by side, and how that becomes roads and lane sections.
"""

from __future__ import annotations

from opendrive_transpiler.config import TranspileOptions
from opendrive_transpiler.diagnostics import DiagnosticBag
from opendrive_transpiler.frontend.interp import execute
from opendrive_transpiler.frontend.loader import parse_source
from opendrive_transpiler.ir.model import build_ir
from opendrive_transpiler.topology import grouping, relations
from opendrive_transpiler.topology.index import NodeIndex

PRELUDE = "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"


def analyse(body: str, tolerance: float = 1e-3):
    options = TranspileOptions(strict=False, point_tolerance=tolerance)
    bag = DiagnosticBag(strict=False)
    module = parse_source(PRELUDE + body, "<test>", bag)
    registry = execute(module, "<test>", bag, options)
    ir = build_ir(registry, bag, options)
    index = NodeIndex(ir.lanelets, tolerance)
    rels = relations.infer(ir.lanelets, index)
    return ir, index, rels, grouping.build(ir.lanelets, rels)


CHAIN = """
left = [Point3d(getId(), i * 10.0, 1.0, 0.0) for i in range(5)]
right = [Point3d(getId(), i * 10.0, -1.0, 0.0) for i in range(5)]
for i in range(4):
    ll = Lanelet(getId(), LineString3d(getId(), [left[i], left[i + 1]]),
                 LineString3d(getId(), [right[i], right[i + 1]]))
    ll.attributes['subtype'] = 'road'
"""

SIDE_BY_SIDE = """
top = [Point3d(getId(), i * 20.0, 3.0, 0.0) for i in range(3)]
mid = [Point3d(getId(), i * 20.0, 0.0, 0.0) for i in range(3)]
bottom = [Point3d(getId(), i * 20.0, -3.0, 0.0) for i in range(3)]
for i in range(2):
    shared = LineString3d(getId(), [mid[i], mid[i + 1]])
    a = Lanelet(getId(), LineString3d(getId(), [top[i], top[i + 1]]), shared)
    b = Lanelet(getId(), shared, LineString3d(getId(), [bottom[i], bottom[i + 1]]))
    a.attributes['subtype'] = 'road'
    b.attributes['subtype'] = 'road'
"""

Y_SPLIT = """
sl = [Point3d(getId(), 0.0, 1.5, 0.0), Point3d(getId(), 30.0, 1.5, 0.0)]
sr = [Point3d(getId(), 0.0, -1.5, 0.0), Point3d(getId(), 30.0, -1.5, 0.0)]
stem = Lanelet(getId(), LineString3d(getId(), sl), LineString3d(getId(), sr))
a = Lanelet(getId(), LineString3d(getId(), [sl[1], Point3d(getId(), 60.0, 1.5, 0.0)]),
            LineString3d(getId(), [sr[1], Point3d(getId(), 60.0, -1.5, 0.0)]))
b = Lanelet(getId(), LineString3d(getId(), [sl[1], Point3d(getId(), 60.0, 12.0, 0.0)]),
            LineString3d(getId(), [sr[1], Point3d(getId(), 60.0, 9.0, 0.0)]))
for ll in (stem, a, b):
    ll.attributes['subtype'] = 'road'
"""


# --------------------------------------------------------------------------
# Node identity
# --------------------------------------------------------------------------


def test_shared_point_objects_become_one_node():
    _ir, index, _rels, _net = analyse(CHAIN)
    # 4 lanelets x 2 bounds x 2 points = 16 handles over 10 distinct nodes.
    assert index.node_count == 10


def test_coincident_but_distinct_points_also_merge():
    """Maps written by hand repeat coordinates instead of sharing objects."""
    _ir, _index, rels, net = analyse(
        """
a = Lanelet(getId(), LineString3d(getId(), [Point3d(getId(), 0.0, 1.0, 0.0),
                                            Point3d(getId(), 10.0, 1.0, 0.0)]),
            LineString3d(getId(), [Point3d(getId(), 0.0, -1.0, 0.0),
                                   Point3d(getId(), 10.0, -1.0, 0.0)]))
b = Lanelet(getId(), LineString3d(getId(), [Point3d(getId(), 10.0, 1.0, 0.0),
                                            Point3d(getId(), 20.0, 1.0, 0.0)]),
            LineString3d(getId(), [Point3d(getId(), 10.0, -1.0, 0.0),
                                   Point3d(getId(), 20.0, -1.0, 0.0)]))
"""
    )
    assert rels.successor_of(0) == [1]
    assert len(net.chains) == 1


def test_points_further_apart_than_the_tolerance_do_not_merge():
    _ir, _index, rels, _net = analyse(
        """
a = Lanelet(getId(), LineString3d(getId(), [Point3d(getId(), 0.0, 1.0, 0.0),
                                            Point3d(getId(), 10.0, 1.0, 0.0)]),
            LineString3d(getId(), [Point3d(getId(), 0.0, -1.0, 0.0),
                                   Point3d(getId(), 10.0, -1.0, 0.0)]))
b = Lanelet(getId(), LineString3d(getId(), [Point3d(getId(), 10.5, 1.0, 0.0),
                                            Point3d(getId(), 20.0, 1.0, 0.0)]),
            LineString3d(getId(), [Point3d(getId(), 10.5, -1.0, 0.0),
                                   Point3d(getId(), 20.0, -1.0, 0.0)]))
"""
    )
    assert rels.successor_of(0) == []


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------


def test_chain_successors_are_linear():
    _ir, _index, rels, _net = analyse(CHAIN)
    assert [rels.successor_of(i) for i in range(4)] == [[1], [2], [3], []]
    assert [rels.predecessor_of(i) for i in range(4)] == [[], [0], [1], [2]]
    assert not any(rels.is_branch(i) for i in range(4))


def test_a_shared_boundary_makes_lanelets_adjacent():
    _ir, _index, rels, _net = analyse(SIDE_BY_SIDE)
    assert rels.right_of == {0: 1, 2: 3}
    assert rels.left_of == {1: 0, 3: 2}


def test_a_split_is_reported_as_a_branch():
    _ir, _index, rels, net = analyse(Y_SPLIT)
    assert sorted(rels.successor_of(0)) == [1, 2]
    assert rels.is_branch(0)
    assert net.branch_lanelets == [0]


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def test_a_chain_becomes_one_road_of_many_sections():
    _ir, _index, _rels, net = analyse(CHAIN)
    assert [g.members for g in net.groups] == [[0], [1], [2], [3]]
    assert len(net.chains) == 1
    assert [g.members for g in net.chains[0].groups] == [[0], [1], [2], [3]]


def test_parallel_lanes_become_one_road_of_wide_sections():
    _ir, _index, _rels, net = analyse(SIDE_BY_SIDE)
    assert [g.members for g in net.groups] == [[0, 1], [2, 3]]
    assert len(net.chains) == 1
    assert net.chains[0].width == 2


def test_a_branch_ends_the_chain_rather_than_welding_roads_together():
    _ir, _index, _rels, net = analyse(Y_SPLIT)
    assert len(net.chains) == 3
    assert all(len(chain.groups) == 1 for chain in net.chains)


def test_lane_groups_are_ordered_left_to_right():
    ir, _index, _rels, net = analyse(SIDE_BY_SIDE)
    group = net.groups[0]
    left_lanelet = ir.lanelets[group.members[0]]
    right_lanelet = ir.lanelets[group.members[1]]
    # The leftmost lanelet's outer bound is the one at the greatest y.
    assert max(p.y for p in left_lanelet.left.points) > max(p.y for p in right_lanelet.left.points)


def test_an_isolated_lanelet_still_becomes_a_road():
    _ir, _index, _rels, net = analyse(
        """
ll = Lanelet(getId(), LineString3d(getId(), [Point3d(getId(), 0.0, 1.0, 0.0),
                                             Point3d(getId(), 10.0, 1.0, 0.0)]),
             LineString3d(getId(), [Point3d(getId(), 0.0, -1.0, 0.0),
                                    Point3d(getId(), 10.0, -1.0, 0.0)]))
"""
    )
    assert len(net.chains) == 1
    assert net.chains[0].lanelet_indices == [0]


# --------------------------------------------------------------------------
# Direction: geometry, not traversal order
# --------------------------------------------------------------------------

TWO_WAY = """
centre = [Point3d(getId(), i * 20.0, 0.0, 0.0) for i in range(3)]
south = [Point3d(getId(), i * 20.0, -3.5, 0.0) for i in range(3)]
north = [Point3d(getId(), i * 20.0, 3.5, 0.0) for i in range(3)]
shared = LineString3d(getId(), centre)
# Each lanelet has the centre on ITS OWN left, so both use it as leftBound --
# one of them reversed. This is what a real two-way road looks like.
east = Lanelet(getId(), shared, LineString3d(getId(), south))
west = Lanelet(getId(), shared.invert(), LineString3d(getId(), list(reversed(north))))
east.attributes['subtype'] = 'road'
west.attributes['subtype'] = 'road'
"""

MALFORMED = """
top = [Point3d(getId(), i * 20.0, 3.5, 0.0) for i in range(3)]
mid = [Point3d(getId(), i * 20.0, 0.0, 0.0) for i in range(3)]
bot = [Point3d(getId(), i * 20.0, -3.5, 0.0) for i in range(3)]
shared = LineString3d(getId(), mid)
a = Lanelet(getId(), LineString3d(getId(), top), shared)
# Left bound runs -x while the right bound runs +x: the lanelet contradicts itself.
b = Lanelet(getId(), shared.invert(), LineString3d(getId(), bot))
a.attributes['subtype'] = 'road'
b.attributes['subtype'] = 'road'
"""


def test_a_two_way_pair_becomes_one_cross_section():
    """The shared centre is the LEFT bound of both, so left-to-left must match."""
    _ir, _index, rels, net = analyse(TWO_WAY)
    assert sorted(rels.opposing) == [(0, 1)]
    assert len(net.groups) == 1, "a two-way road is one cross-section, not two"
    # Exactly one of the two travels against the group's forward direction.
    assert sum(net.groups[0].reversed_) == 1
    assert net.groups[0].two_way


def test_same_direction_lanes_are_not_called_opposing():
    """The regression guard: SIDE_BY_SIDE must stay a plain two-lane road."""
    _ir, _index, rels, net = analyse(SIDE_BY_SIDE)
    assert rels.opposing == set()
    assert [g.members for g in net.groups] == [[0, 1], [2, 3]]
    assert not any(g.two_way for g in net.groups)


def test_travel_direction_averages_both_bounds():
    """Neither bound alone decides; a well-formed lanelet's bounds agree."""
    ir, _index, _rels, _net = analyse(TWO_WAY)
    east, west = ir.lanelets
    assert relations.travel_direction(east)[0] > 0.9
    assert relations.travel_direction(west)[0] < -0.9


def test_a_lanelet_whose_bounds_disagree_is_detected():
    """Its two edges claim opposite directions, so it says nothing about travel."""
    ir, _index, _rels, _net = analyse(MALFORMED)
    disagree = [relations.bounds_disagree(lanelet) for lanelet in ir.lanelets]
    assert disagree.count(True) == 1


def test_direction_survives_a_reversed_shared_boundary_on_a_curve():
    """A curved two-way pair must still be recognised, not just an axis-aligned one."""
    _ir, _index, rels, net = analyse(
        """
import math
inner, outer, mid = [], [], []
for step in range(7):
    angle = math.radians(90.0 * step / 6)
    mid.append(Point3d(getId(), 40.0 * math.cos(angle), 40.0 * math.sin(angle), 0.0))
    inner.append(Point3d(getId(), 36.5 * math.cos(angle), 36.5 * math.sin(angle), 0.0))
    outer.append(Point3d(getId(), 43.5 * math.cos(angle), 43.5 * math.sin(angle), 0.0))
shared = LineString3d(getId(), mid)
a = Lanelet(getId(), shared, LineString3d(getId(), inner))
b = Lanelet(getId(), shared.invert(), LineString3d(getId(), list(reversed(outer))))
a.attributes['subtype'] = 'road'
b.attributes['subtype'] = 'road'
"""
    )
    assert sorted(rels.opposing) == [(0, 1)]
    assert len(net.groups) == 1


# --------------------------------------------------------------------------
# Bounds that disagree about where the lanelet is
# --------------------------------------------------------------------------
# Everything projected onto a road is clamped to its reference line, so a lanelet
# whose bounds cover different stretches truncates whatever is measured against
# it. Two ratios have to be large together, because either alone is ordinary
# geometry -- these cases pin both halves of that.


def misaligned(body: str) -> list[bool]:
    ir, _index, _rels, _network = analyse(body)
    return [relations.bounds_misaligned(ll) for ll in ir.lanelets]


def test_a_bound_running_past_the_other_is_caught():
    """The case that prompted this: one bound ends at 20 m, the other at 40."""
    assert misaligned("""
a = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 2.0, 0.0),
                           Point3d(getId(), 20.0, 2.0, 0.0)]),
    LineString3d(getId(), [Point3d(getId(), 0.0, -2.0, 0.0),
                           Point3d(getId(), 40.0, -2.0, 0.0)]))
a.attributes['subtype'] = 'road'
""") == [True]


def test_a_merge_taper_is_not_misaligned():
    """Width ratio 7, length ratio 1.003: it narrows, it does not overshoot."""
    assert misaligned("""
a = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 0.0, 0.0),
                           Point3d(getId(), 40.0, 0.0, 0.0)]),
    LineString3d(getId(), [Point3d(getId(), 0.0, -3.5, 0.0),
                           Point3d(getId(), 40.0, -0.5, 0.0)]))
a.attributes['subtype'] = 'road'
""") == [False]


def test_a_turn_tight_enough_to_double_the_outer_edge_is_not_misaligned():
    """Length ratio 2, width ratio 1: a pivot is odd, but self-consistent."""
    assert misaligned("""
import math
inner = [Point3d(getId(), 3.5 * math.cos(i * math.pi / 16),
                 3.5 * math.sin(i * math.pi / 16), 0.0) for i in range(9)]
outer = [Point3d(getId(), 7.0 * math.cos(i * math.pi / 16),
                 7.0 * math.sin(i * math.pi / 16), 0.0) for i in range(9)]
a = Lanelet(getId(), LineString3d(getId(), inner), LineString3d(getId(), outer))
a.attributes['subtype'] = 'road'
""") == [False]


def test_an_ordinary_curve_is_not_misaligned():
    assert misaligned("""
import math
inner = [Point3d(getId(), 10.0 * math.cos(i * math.pi / 16),
                 10.0 * math.sin(i * math.pi / 16), 0.0) for i in range(9)]
outer = [Point3d(getId(), 13.5 * math.cos(i * math.pi / 16),
                 13.5 * math.sin(i * math.pi / 16), 0.0) for i in range(9)]
a = Lanelet(getId(), LineString3d(getId(), inner), LineString3d(getId(), outer))
a.attributes['subtype'] = 'road'
""") == [False]


def test_a_short_wide_lanelet_is_not_misaligned():
    """A crossing is wider than it is long, and perfectly well formed."""
    assert misaligned("""
a = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 5.0, 0.0),
                           Point3d(getId(), 0.0, -5.0, 0.0)]),
    LineString3d(getId(), [Point3d(getId(), 4.0, 5.0, 0.0),
                           Point3d(getId(), 4.0, -5.0, 0.0)]))
a.attributes['subtype'] = 'crosswalk'
""") == [False]


def test_a_single_point_bound_never_reaches_this_check():
    """It is dropped as degenerate earlier, with its own diagnostic."""
    assert (
        misaligned("""
p = Point3d(getId(), 0.0, -2.0, 0.0)
a = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 2.0, 0.0),
                           Point3d(getId(), 40.0, 2.0, 0.0)]),
    LineString3d(getId(), [p]))
a.attributes['subtype'] = 'road'
""")
        == []
    )


def test_a_zero_length_bound_is_not_reported_as_misaligned():
    """Dividing by it would raise; the guard has to come first."""
    ir, _index, _rels, _network = analyse("""
a = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 2.0, 0.0),
                           Point3d(getId(), 40.0, 2.0, 0.0)]),
    LineString3d(getId(), [Point3d(getId(), 0.0, -2.0, 0.0),
                           Point3d(getId(), 0.0, -2.0, 0.0)]))
a.attributes['subtype'] = 'road'
""")
    for lanelet in ir.lanelets:
        assert relations.bounds_misaligned(lanelet) is False
