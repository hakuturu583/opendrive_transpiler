"""Two single-lane roads merging into one two-lane road.

This is the shape the backend's own lane linking refuses: it raises
`NotSameAmountOfLanesError` when two connected roads carry different lane counts.
The correspondence is unambiguous here -- each incoming lane continues into
exactly one outgoing lane -- so the links are written down from the lanelet
topology instead of being inferred from geometry.

Taken from the real Karlsruhe map, where it occurs six times.
"""

from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId

# The joint where everything meets, shared by identity so succession is exact.
joint_low = Point3d(getId(), 30.0, 0.0, 0.0)
joint_mid = Point3d(getId(), 30.0, 3.0, 0.0)
joint_high = Point3d(getId(), 30.0, 6.0, 0.0)

# The straight approach, along y = 0..3.
straight = Lanelet(
    getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 3.0, 0.0), joint_mid]),
    LineString3d(getId(), [Point3d(getId(), 0.0, 0.0, 0.0), joint_low]),
)
straight.attributes["subtype"] = "road"

# The slip road, converging from above. Its bounds are diagonal, so it shares no
# boundary with the straight approach and stays a road of its own.
slip = Lanelet(
    getId(),
    LineString3d(getId(), [Point3d(getId(), 0.0, 13.0, 0.0), joint_high]),
    LineString3d(getId(), [Point3d(getId(), 0.0, 10.0, 0.0), joint_mid]),
)
slip.attributes["subtype"] = "road"

# Downstream the two run side by side, sharing the y = 3 boundary, so they become
# one two-lane road.
shared_centre = LineString3d(getId(), [joint_mid, Point3d(getId(), 60.0, 3.0, 0.0)])

inner = Lanelet(
    getId(),
    shared_centre,
    LineString3d(getId(), [joint_low, Point3d(getId(), 60.0, 0.0, 0.0)]),
)
inner.attributes["subtype"] = "road"

outer = Lanelet(
    getId(),
    LineString3d(getId(), [joint_high, Point3d(getId(), 60.0, 6.0, 0.0)]),
    shared_centre,
)
outer.attributes["subtype"] = "road"

lanelet_map = createMapFromLanelets([straight, slip, inner, outer])
