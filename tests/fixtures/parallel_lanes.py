"""Two lanes side by side, sharing their middle boundary, two sections long.

Boundary sharing is the primary adjacency signal, so this should become a single
road with two lane sections of two lanes each -- not four separate roads.
"""

from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId

top = [Point3d(getId(), i * 25.0, 3.5, 0.0) for i in range(3)]
middle = [Point3d(getId(), i * 25.0, 0.0, 0.0) for i in range(3)]
bottom = [Point3d(getId(), i * 25.0, -3.5, 0.0) for i in range(3)]

lanelets = []
for i in range(2):
    shared = LineString3d(getId(), [middle[i], middle[i + 1]])
    shared.attributes["type"] = "line_thin"
    shared.attributes["subtype"] = "dashed"

    outer_left = LineString3d(getId(), [top[i], top[i + 1]])
    outer_left.attributes["type"] = "line_thin"
    outer_left.attributes["subtype"] = "solid"

    outer_right = LineString3d(getId(), [bottom[i], bottom[i + 1]])
    outer_right.attributes["type"] = "line_thin"
    outer_right.attributes["subtype"] = "solid"

    fast = Lanelet(getId(), outer_left, shared)
    slow = Lanelet(getId(), shared, outer_right)
    for lanelet in (fast, slow):
        lanelet.attributes["subtype"] = "highway"
        lanelet.attributes["location"] = "nonurban"
        lanelet.attributes["speed_limit"] = "100 km/h"
        lanelets.append(lanelet)

lanelet_map = createMapFromLanelets(lanelets)
