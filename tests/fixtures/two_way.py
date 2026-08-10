"""A two-way road: one lane each way, sharing the centre line.

Each lanelet names its bounds relative to its *own* direction of travel, so the
shared centre line is the **left** bound of both of them, stored in opposite
order. That is what a real two-way lanelet2 road looks like, and it must come out
as a single road with lane +1 running against the s-axis and lane -1 with it.
"""

from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId

centre = [Point3d(getId(), i * 25.0, 0.0, 0.0) for i in range(3)]
south = [Point3d(getId(), i * 25.0, -3.5, 0.0) for i in range(3)]
north = [Point3d(getId(), i * 25.0, 3.5, 0.0) for i in range(3)]

centre_line = LineString3d(getId(), centre)
centre_line.attributes["type"] = "line_thin"
centre_line.attributes["subtype"] = "solid_solid"

southbound_edge = LineString3d(getId(), south)
southbound_edge.attributes["type"] = "line_thin"
southbound_edge.attributes["subtype"] = "solid"

northbound_edge = LineString3d(getId(), list(reversed(north)))
northbound_edge.attributes["type"] = "line_thin"
northbound_edge.attributes["subtype"] = "solid"

# Travels +x, occupying y in [-3.5, 0]: the centre is on its left.
eastbound = Lanelet(getId(), centre_line, southbound_edge)

# Travels -x, occupying y in [0, 3.5]: the centre is on its left too, reversed.
westbound = Lanelet(getId(), centre_line.invert(), northbound_edge)

for lanelet in (eastbound, westbound):
    lanelet.attributes["subtype"] = "road"
    lanelet.attributes["location"] = "urban"

lanelet_map = createMapFromLanelets([eastbound, westbound])
