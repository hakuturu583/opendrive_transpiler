"""Two lanes that become one: a lane drop inside a single road.

OpenDRIVE expresses this directly as consecutive lane sections of differing
width, linked lane by lane -- so this must come out as *one* road with two
sections, not two roads joined by a junction. The surviving lane keeps its
identity across the drop.
"""

from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId

top = [Point3d(getId(), i * 30.0, 3.5, 0.0) for i in range(3)]
middle = [Point3d(getId(), i * 30.0, 0.0, 0.0) for i in range(3)]
bottom = [Point3d(getId(), i * 30.0, -3.5, 0.0) for i in range(3)]

# Section 0: two lanes side by side.
shared = LineString3d(getId(), [middle[0], middle[1]])
shared.attributes["type"] = "line_thin"
shared.attributes["subtype"] = "dashed"

fast = Lanelet(getId(), LineString3d(getId(), [top[0], top[1]]), shared)
slow = Lanelet(getId(), shared, LineString3d(getId(), [bottom[0], bottom[1]]))

# Section 1: only the left-hand lane continues.
onward = Lanelet(
    getId(),
    LineString3d(getId(), [top[1], top[2]]),
    LineString3d(getId(), [middle[1], middle[2]]),
)

for lanelet in (fast, slow, onward):
    lanelet.attributes["subtype"] = "road"

lanelet_map = createMapFromLanelets([fast, slow, onward])
