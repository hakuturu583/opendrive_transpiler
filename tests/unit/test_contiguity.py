"""Whether the roads the file links actually meet, and saying so when they do not.

OpenDRIVE expects a road and its successor -- and an incoming road and the
connecting road it enters a junction by -- to be geometrically contiguous:
driving off the end of one puts you on the start of the next. Stating the link
and meeting at it are separate claims, and until `W512` only the first was ever
checked. On the Lanelet2 Karlsruhe example that left 34 joins up to 12.5 m apart
sitting behind a connectivity score of 310 successions out of 327.

They come apart because of the reference line this emits, which is the leftmost
boundary of the cross-section. Where the cross-section changes lane count across
a join, the leftmost boundary is a different physical line on either side, and
the two ends sit a few lane widths apart. Every one of the 45 joins on that map
that keeps its lane count is exact; every gap is a join that does not.
"""

from __future__ import annotations

import math
import re

import pytest

from opendrive_transpiler import TranspileOptions, transpile_source


def convert(source: str, **kwargs):
    return transpile_source(
        source, "t.py", options=TranspileOptions(strict=False, name="t", **kwargs)
    )


def notices(result, code: str = "LL2ODR-W512") -> list[str]:
    return [d.message for d in result.diagnostics if d.code == code]


def start_of(road) -> tuple[float, float]:
    return road.geometries[0].x, road.geometries[0].y


def end_of(road) -> tuple[float, float]:
    last = road.geometries[-1]
    return (
        last.x + last.length * math.cos(last.hdg),
        last.y + last.length * math.sin(last.hdg),
    )


# Two single-lane approaches merging into one two-lane road, which is the shape
# the `merge` fixture is built from and which the Karlsruhe map has six of. The
# straight approach follows the y = 3 boundary; downstream the cross-section
# gains a lane on the left, so the merged road follows y = 6 instead.
MERGE = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
low = Point3d(getId(), 30.0, 0.0, 0.0)
mid = Point3d(getId(), 30.0, 3.0, 0.0)
high = Point3d(getId(), 30.0, 6.0, 0.0)
straight = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 3.0, 0.0), mid]),
    LineString3d(getId(), [Point3d(getId(), 0.0, 0.0, 0.0), low]))
slip = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 13.0, 0.0), high]),
    LineString3d(getId(), [Point3d(getId(), 0.0, 10.0, 0.0), mid]))
centre = LineString3d(getId(), [mid, Point3d(getId(), 60.0, 3.0, 0.0)])
inner = Lanelet(getId(), centre,
    LineString3d(getId(), [low, Point3d(getId(), 60.0, 0.0, 0.0)]))
outer = Lanelet(getId(),
    LineString3d(getId(), [high, Point3d(getId(), 60.0, 6.0, 0.0)]), centre)
for lanelet in (straight, slip, inner, outer):
    lanelet.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([straight, slip, inner, outer])
"""

# A plain chain: two lanelets, one behind the other, sharing their joint. The
# cross-section never changes, so there is nothing for the reference line to
# step across.
CHAIN = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
a = Point3d(getId(), 30.0, 0.0, 0.0)
b = Point3d(getId(), 30.0, 3.5, 0.0)
first = Lanelet(getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 3.5, 0.0), b]),
    LineString3d(getId(), [Point3d(getId(), 0.0, 0.0, 0.0), a]))
second = Lanelet(getId(),
    LineString3d(getId(), [b, Point3d(getId(), 60.0, 3.5, 0.0)]),
    LineString3d(getId(), [a, Point3d(getId(), 60.0, 0.0, 0.0)]))
for lanelet in (first, second):
    lanelet.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([first, second])
"""


# --------------------------------------------------------------------------
# What it catches
# --------------------------------------------------------------------------


def test_a_join_where_the_cross_section_changes_is_reported():
    assert notices(convert(MERGE)), "the merged road follows a boundary 3 m off the approach's"


def test_the_report_gives_the_distance_and_the_reason():
    message = notices(convert(MERGE))[0]
    assert re.search(r"are 3(\.\d+)? m apart", message), message
    assert "cross-section changes across the join" in message


def test_the_reported_distance_is_the_real_one():
    """A number that is not measured from the emitted geometry is worse than none."""
    result = convert(MERGE)
    roads = {road.road_id: road for road in result.model.roads}
    message = notices(result)[0]
    a, b = (int(n) for n in re.findall(r"road (\d+)", message)[:2])
    stated = float(re.search(r"are ([\d.]+) m apart", message).group(1))

    # One of the four end-to-start pairings is the one being reported.
    pairs = [
        math.dist(end_of(roads[a]), start_of(roads[b])),
        math.dist(end_of(roads[b]), start_of(roads[a])),
        math.dist(start_of(roads[a]), start_of(roads[b])),
        math.dist(end_of(roads[a]), end_of(roads[b])),
    ]
    assert any(abs(stated - p) < 0.01 for p in pairs), (stated, pairs)


def test_only_the_join_that_steps_is_reported():
    """The slip road already meets the merged road, and must not be named."""
    messages = notices(convert(MERGE))
    assert len(messages) == 1, messages


# --------------------------------------------------------------------------
# What it must leave alone
# --------------------------------------------------------------------------


def test_a_chain_with_an_unchanging_cross_section_says_nothing():
    result = convert(CHAIN)
    assert not notices(result)
    assert len(result.model.roads) == 1, "and it is one road, so there is no join at all"


@pytest.mark.parametrize("reference_line", ["left-bound", "centerline", "auto"])
def test_a_road_is_never_reported_against_itself(reference_line: str):
    for message in notices(convert(MERGE, reference_line=reference_line)):
        a, b = re.findall(r"road (\d+)", message)[:2]
        assert a != b


def test_turning_junctions_off_leaves_only_the_road_to_road_joins():
    """A junction link that no longer exists may not still be reported."""
    for message in notices(convert(MERGE, junctions=False)):
        assert "through junction" not in message


def test_the_same_pair_is_reported_once():
    """Both roads name the join, and a junction names it a third time."""
    messages = notices(convert(MERGE))
    pairs = [tuple(sorted(re.findall(r"road (\d+)", m)[:2])) for m in messages]
    assert len(pairs) == len(set(pairs))
