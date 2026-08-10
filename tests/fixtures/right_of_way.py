"""A divergence where one branch has right of way over the other.

The `RightOfWay` regulatory element is the only thing in lanelet2 that ranks
lanelets against each other, and OpenDRIVE's `<priority>` is where that ranking
belongs -- naming the junction's connecting roads, which here are the two
branches.
"""

from lanelet2.core import (
    Lanelet,
    LineString3d,
    Point3d,
    RightOfWay,
    createMapFromLanelets,
    getId,
)

stem_left = [Point3d(getId(), 0.0, 1.5, 0.0), Point3d(getId(), 30.0, 1.5, 0.0)]
stem_right = [Point3d(getId(), 0.0, -1.5, 0.0), Point3d(getId(), 30.0, -1.5, 0.0)]

stem = Lanelet(
    getId(),
    LineString3d(getId(), stem_left),
    LineString3d(getId(), stem_right),
)
stem.attributes["subtype"] = "road"

# The through lane carries the traffic that does not stop.
through = Lanelet(
    getId(),
    LineString3d(getId(), [stem_left[1], Point3d(getId(), 60.0, 1.5, 0.0)]),
    LineString3d(getId(), [stem_right[1], Point3d(getId(), 60.0, -1.5, 0.0)]),
)
through.attributes["subtype"] = "road"

# The side road has to give way to it.
side = Lanelet(
    getId(),
    LineString3d(getId(), [stem_left[1], Point3d(getId(), 60.0, 12.0, 0.0)]),
    LineString3d(getId(), [stem_right[1], Point3d(getId(), 60.0, 9.0, 0.0)]),
)
side.attributes["subtype"] = "road"

stop_line = LineString3d(
    getId(),
    [Point3d(getId(), 31.0, 9.0, 0.0), Point3d(getId(), 31.0, 12.0, 0.0)],
)

priority = RightOfWay(getId(), [], [through], [side], stop_line)
priority.attributes["subtype"] = "right_of_way"

lanelet_map = createMapFromLanelets([stem, through, side])
