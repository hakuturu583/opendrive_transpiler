"""One lanelet splitting into two.

A branch has no unambiguous road link without junction support, so each side
should become its own road and the transpiler should say why.
"""

from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId

stem_left = [Point3d(getId(), 0.0, 1.5, 0.0), Point3d(getId(), 30.0, 1.5, 0.0)]
stem_right = [Point3d(getId(), 0.0, -1.5, 0.0), Point3d(getId(), 30.0, -1.5, 0.0)]

stem = Lanelet(
    getId(),
    LineString3d(getId(), stem_left),
    LineString3d(getId(), stem_right),
)
stem.attributes["subtype"] = "road"

straight = Lanelet(
    getId(),
    LineString3d(getId(), [stem_left[1], Point3d(getId(), 60.0, 1.5, 0.0)]),
    LineString3d(getId(), [stem_right[1], Point3d(getId(), 60.0, -1.5, 0.0)]),
)
straight.attributes["subtype"] = "road"

ramp = Lanelet(
    getId(),
    LineString3d(getId(), [stem_left[1], Point3d(getId(), 60.0, 12.0, 0.0)]),
    LineString3d(getId(), [stem_right[1], Point3d(getId(), 60.0, 9.0, 0.0)]),
)
ramp.attributes["subtype"] = "exit"

lanelet_map = createMapFromLanelets([stem, straight, ramp])
