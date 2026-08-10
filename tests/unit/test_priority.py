"""Right-of-way rules becoming junction `<priority>`.

`RightOfWay` is the only thing in lanelet2 that ranks lanelets against each
other, and `<priority high low>` is where OpenDRIVE puts that ranking. It names
*connecting* roads, which is the constraint these tests pin: a ranking exists
only where a junction has more than one connecting road to rank.
"""

from __future__ import annotations

from opendrive_transpiler import TranspileOptions, transpile_source

PRELUDE = (
    "from lanelet2.core import (\n"
    "    AllWayStop,\n"
    "    Lanelet,\n"
    "    LineString3d,\n"
    "    Point3d,\n"
    "    RightOfWay,\n"
    "    getId,\n"
    ")\n"
)

# One road ending where two begin: the stem is incoming, the branches connect.
DIVERGE = """
stem_left = [Point3d(getId(), 0.0, 1.5, 0.0), Point3d(getId(), 30.0, 1.5, 0.0)]
stem_right = [Point3d(getId(), 0.0, -1.5, 0.0), Point3d(getId(), 30.0, -1.5, 0.0)]
stem = Lanelet(getId(), LineString3d(getId(), stem_left),
               LineString3d(getId(), stem_right))
stem.attributes['subtype'] = 'road'

through = Lanelet(getId(),
    LineString3d(getId(), [stem_left[1], Point3d(getId(), 60.0, 1.5, 0.0)]),
    LineString3d(getId(), [stem_right[1], Point3d(getId(), 60.0, -1.5, 0.0)]))
through.attributes['subtype'] = 'road'

side = Lanelet(getId(),
    LineString3d(getId(), [stem_left[1], Point3d(getId(), 60.0, 12.0, 0.0)]),
    LineString3d(getId(), [stem_right[1], Point3d(getId(), 60.0, 9.0, 0.0)]))
side.attributes['subtype'] = 'road'
"""

# Two roads ending where one begins. There is a single connecting road, so
# nothing to rank.
CONVERGE = """
tip_left = Point3d(getId(), 30.0, 1.5, 0.0)
tip_right = Point3d(getId(), 30.0, -1.5, 0.0)

through = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 1.5, 0.0), tip_left]),
    LineString3d(getId(), [Point3d(getId(), 0.0, -1.5, 0.0), tip_right]))
through.attributes['subtype'] = 'road'

side = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 12.0, 0.0), tip_left]),
    LineString3d(getId(), [Point3d(getId(), 0.0, 9.0, 0.0), tip_right]))
side.attributes['subtype'] = 'road'

stem = Lanelet(getId(),
    LineString3d(getId(), [tip_left, Point3d(getId(), 60.0, 1.5, 0.0)]),
    LineString3d(getId(), [tip_right, Point3d(getId(), 60.0, -1.5, 0.0)]))
stem.attributes['subtype'] = 'road'
"""


def convert(body: str, **kwargs):
    options = TranspileOptions(strict=False, name="priority", **kwargs)
    return transpile_source(PRELUDE + body, "priority.py", options=options)


def priorities(result) -> list[tuple[int, int]]:
    return [(p.high, p.low) for junction in result.model.junctions for p in junction.priorities]


def codes(result) -> set[str]:
    return {d.code for d in result.diagnostics}


def road_of(result, lanelet_index: int) -> int:
    """Which OpenDRIVE road carries the nth lanelet the script built."""
    ids = sorted({i for road in result.model.roads for i in road.lanelet2_ids})
    target = ids[lanelet_index]
    return next(road.road_id for road in result.model.roads if target in road.lanelet2_ids)


# --------------------------------------------------------------------------


def test_a_right_of_way_ranks_the_two_branches():
    result = convert(
        DIVERGE + "RightOfWay(getId(), [], [through], [side], None)\n",
    )
    assert len(result.model.junctions) == 1
    high, low = priorities(result)[0]
    # The through lane goes first; the side road yields to it.
    assert (high, low) == (road_of(result, 1), road_of(result, 2))


def test_the_ranking_follows_the_roles_rather_than_the_order():
    """Swapping the roles must swap high and low, not just relabel them."""
    forward = priorities(convert(DIVERGE + "RightOfWay(getId(), [], [through], [side], None)\n"))
    reversed_ = priorities(convert(DIVERGE + "RightOfWay(getId(), [], [side], [through], None)\n"))
    assert forward and reversed_
    assert forward[0] == tuple(reversed(reversed_[0]))


def test_a_converted_rule_is_not_reported_as_dropped():
    """A rule that became a <priority> did convert, even though it is no signal."""
    result = convert(DIVERGE + "RightOfWay(getId(), [], [through], [side], None)\n")
    assert "LL2ODR-I902" not in codes(result)
    assert "LL2ODR-I905" not in codes(result)


def test_a_convergence_has_nothing_to_rank_and_says_so():
    """One connecting road cannot be ranked against itself."""
    result = convert(CONVERGE + "RightOfWay(getId(), [], [through], [side], None)\n")
    assert priorities(result) == []
    assert "LL2ODR-I905" in codes(result)


def test_a_rule_over_lanelets_outside_any_junction_is_reported():
    result = convert(
        "a = Lanelet(getId(), "
        "LineString3d(getId(), [Point3d(getId(), 0.0, 2.0, 0.0), "
        "Point3d(getId(), 40.0, 2.0, 0.0)]), "
        "LineString3d(getId(), [Point3d(getId(), 0.0, -2.0, 0.0), "
        "Point3d(getId(), 40.0, -2.0, 0.0)]))\n"
        "a.attributes['subtype'] = 'road'\n"
        "RightOfWay(getId(), [], [a], [], None)\n"
    )
    assert priorities(result) == []
    assert "LL2ODR-I905" in codes(result)


def test_an_all_way_stop_is_not_a_priority():
    """Every approach yields to every other, so there is no ranking to emit."""
    result = convert(DIVERGE + "AllWayStop(getId(), [], [through, side], [])\n")
    assert priorities(result) == []
    # ...and it is not silently forgotten either.
    assert "LL2ODR-I902" in codes(result)


def test_priority_reaches_the_generated_code():
    result = convert(DIVERGE + "RightOfWay(getId(), [], [through], [side], None)\n")
    assert "_JunctionWithPriority" in result.code
    assert "priorities.append" in result.code
    # The shim needs ElementTree, and it must only appear when it is used.
    assert "import xml.etree.ElementTree as ET" in result.code


def test_no_shim_is_emitted_when_nothing_needs_it():
    result = convert(DIVERGE)
    assert "_JunctionWithPriority" not in result.code
    assert "ElementTree" not in result.code


def test_disabling_junctions_disables_priority_too():
    result = convert(
        DIVERGE + "RightOfWay(getId(), [], [through], [side], None)\n", junctions=False
    )
    assert result.model.junctions == []
    assert priorities(result) == []
