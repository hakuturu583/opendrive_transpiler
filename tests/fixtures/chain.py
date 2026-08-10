"""A run of lanelets sharing end points, so `follows` holds between them.

Derived from simple_lanelet2's tests/cases/0700_routing.py (BSD-3-Clause), with
the project-internal test harness removed. Exercises list comprehensions,
default arguments, an `if`, a `for` loop and post-construction attribute
assignment -- the constructs a real map-building script actually uses.
"""

from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId


def chain(count, y=0.0, subtype="road", shared_left=None):
    left = [Point3d(getId(), i * 10.0, y + 1.0, 0.0) for i in range(count + 1)]
    right = [Point3d(getId(), i * 10.0, y - 1.0, 0.0) for i in range(count + 1)]
    if shared_left is not None:
        right = shared_left
    lanelets = []
    for i in range(count):
        ll = Lanelet(
            getId(),
            LineString3d(getId(), [left[i], left[i + 1]]),
            LineString3d(getId(), [right[i], right[i + 1]]),
        )
        ll.attributes["subtype"] = subtype
        lanelets.append(ll)
    return lanelets, left, right


lanelets, _left, _right = chain(4)
lanelet_map = createMapFromLanelets(lanelets)
