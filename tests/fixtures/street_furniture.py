"""A street with a guard rail down one side and a crosswalk across it.

Both are things lanelet2 states plainly that have no lane equivalent: a barrier
is not a painted line, and a crosswalk is a marking across a carriageway rather
than a carriageway of its own. Both become `<object>` outlines.
"""

from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId

left = LineString3d(
    getId(),
    [Point3d(getId(), 0.0, 2.0, 0.0), Point3d(getId(), 40.0, 2.0, 0.0)],
)
left.attributes["type"] = "line_thin"
left.attributes["subtype"] = "solid"

# The verge side is protected rather than marked.
right = LineString3d(
    getId(),
    [
        Point3d(getId(), 0.0, -2.0, 0.0),
        Point3d(getId(), 20.0, -2.0, 0.0),
        Point3d(getId(), 40.0, -2.0, 0.0),
    ],
)
right.attributes["type"] = "guard_rail"

street = Lanelet(getId(), left, right)
street.attributes["subtype"] = "road"
street.attributes["location"] = "urban"

# A crossing, at right angles to the street and wider than it.
crossing = Lanelet(
    getId(),
    LineString3d(
        getId(),
        [Point3d(getId(), 8.0, 5.0, 0.0), Point3d(getId(), 8.0, -5.0, 0.0)],
    ),
    LineString3d(
        getId(),
        [Point3d(getId(), 12.0, 5.0, 0.0), Point3d(getId(), 12.0, -5.0, 0.0)],
    ),
)
crossing.attributes["subtype"] = "crosswalk"

lanelet_map = createMapFromLanelets([street, crossing])
