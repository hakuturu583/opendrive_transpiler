"""Choosing which boundary of each cross-section carries the reference line.

`W512` says a join does not meet. This is the machinery that answers what could
be done about it: an assignment of a boundary index to every cross-section such
that, at each handover, the boundary one side ends on is the boundary the other
begins on.

It changes nothing that is emitted yet. What it produces is the number the
decision needs -- on the Lanelet2 Karlsruhe example, 17 of the 25 unmet handovers
can be closed, and it costs 65 cross-sections moving off their leftmost boundary
because moving one forces every road along its chain to move with it.
"""

from __future__ import annotations

from opendrive_transpiler.topology.contiguity import Edge, corners_of, solve

TOL = 1e-3


def stack(*ys: float, x0: float = 0.0, x1: float = 10.0) -> list[list[tuple[float, float, float]]]:
    """A cross-section as horizontal boundaries at the given lateral offsets."""
    return [[(x0, y, 0.0), (x1, y, 0.0)] for y in ys]


def graph(stacks, edges):
    ends = [corners_of(s, "end", TOL) for s in stacks]
    starts = [corners_of(s, "start", TOL) for s in stacks]
    return solve(ends, starts, edges)


# --------------------------------------------------------------------------
# Leaving well alone
# --------------------------------------------------------------------------


def test_a_handover_that_already_meets_moves_nothing():
    """Two cross-sections of the same shape, end to start."""
    a = stack(3.5, 0.0, x0=0.0, x1=10.0)
    b = stack(3.5, 0.0, x0=10.0, x1=20.0)
    result = graph([a, b], [Edge(0, 1, within_road=False)])
    assert result.satisfied == [True]
    assert result.chosen == [0, 0], "the leftmost boundary already lines up"
    assert result.interior == []


def test_a_cross_section_with_no_handovers_is_untouched():
    result = graph([stack(3.5, 0.0)], [])
    assert result.chosen == [0]


# --------------------------------------------------------------------------
# The case W512 reports
# --------------------------------------------------------------------------


def test_a_widening_is_closed_by_moving_the_wider_side_inward():
    """One lane becoming two, the new lane appearing on the left.

    The narrow road's only boundary pair ends at y = 3.5 and y = 0; the wide one
    begins at 7.0, 3.5 and 0. Leftmost against leftmost is 3.5 against 7.0 and
    does not meet -- but the wide road's *second* boundary begins at 3.5.
    """
    narrow = stack(3.5, 0.0, x0=0.0, x1=10.0)
    wide = stack(7.0, 3.5, 0.0, x0=10.0, x1=20.0)
    result = graph([narrow, wide], [Edge(0, 1, within_road=False)])
    assert result.satisfied == [True]
    assert result.chosen == [0, 1], "the wide road takes its middle boundary"
    assert result.interior == [1]


def test_the_cheaper_side_is_the_one_that_moves():
    """Both sides could move; the assignment prefers the one that stays put."""
    narrow = stack(3.5, 0.0, x0=0.0, x1=10.0)
    wide = stack(7.0, 3.5, 0.0, x0=10.0, x1=20.0)
    result = graph([narrow, wide], [Edge(0, 1, within_road=False)])
    assert result.chosen[0] == 0, "nothing forces the narrow road off its own bound"


def test_a_choice_propagates_down_the_chain():
    """This is the cost, and it is not local.

    Moving one cross-section's reference line forces the next one along to move
    too, or the handover between *them* stops meeting. A single widening at the
    head of a chain pulls every road behind it.
    """
    wide = stack(7.0, 3.5, 0.0, x0=0.0, x1=10.0)
    middle = stack(7.0, 3.5, 0.0, x0=10.0, x1=20.0)
    narrow = stack(3.5, 0.0, x0=20.0, x1=30.0)
    result = graph(
        [wide, middle, narrow],
        [Edge(0, 1, within_road=True), Edge(1, 2, within_road=False)],
    )
    assert all(result.satisfied)
    assert result.chosen == [1, 1, 0], "both wide sections drop to the shared bound"


# --------------------------------------------------------------------------
# What it may not do
# --------------------------------------------------------------------------


def test_an_impossible_handover_is_left_unsatisfied_rather_than_forced():
    """Two cross-sections that share no corner at all: nothing to choose."""
    a = stack(3.5, 0.0, x0=0.0, x1=10.0)
    b = stack(99.0, 96.0, x0=10.0, x1=20.0)
    result = graph([a, b], [Edge(0, 1, within_road=False)])
    assert result.satisfied == [False]
    assert result.interior == [], "and it did not move anything for nothing"


def test_an_impossible_handover_does_not_drag_a_solvable_one_down():
    solvable = [stack(3.5, 0.0, x0=0.0, x1=10.0), stack(7.0, 3.5, 0.0, x0=10.0, x1=20.0)]
    hopeless = [stack(50.0, 47.0, x0=0.0, x1=10.0), stack(99.0, 96.0, x0=10.0, x1=20.0)]
    result = graph(
        solvable + hopeless,
        [Edge(0, 1, within_road=False), Edge(2, 3, within_road=False)],
    )
    assert result.satisfied == [True, False]


def test_a_degenerate_boundary_is_not_a_corner():
    """A bound with fewer than two points has no end to hand anything over at."""
    a = [[(0.0, 3.5, 0.0), (10.0, 3.5, 0.0)], [(10.0, 0.0, 0.0)]]
    b = stack(3.5, 0.0, x0=10.0, x1=20.0)
    result = graph([a, b], [Edge(0, 1, within_road=False)])
    assert result.satisfied == [True], "the surviving boundary still lines up"


def test_the_solver_never_reports_more_than_it_achieves():
    """`met` has to be recomputed from the final assignment, not accumulated."""
    a = stack(3.5, 0.0, x0=0.0, x1=10.0)
    b = stack(7.0, 3.5, 0.0, x0=10.0, x1=20.0)
    c = stack(99.0, 96.0, x0=20.0, x1=30.0)
    edges = [Edge(0, 1, within_road=False), Edge(1, 2, within_road=False)]
    result = graph([a, b, c], edges)
    recomputed = sum(
        1
        for edge, ok in zip(edges, result.satisfied, strict=True)
        if ok and edge.left != edge.right
    )
    assert result.met == recomputed
