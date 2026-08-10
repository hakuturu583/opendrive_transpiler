"""A map built alongside a UTM projector, as scripts that write OSM do.

The projector's origin is what reaches <header><geoReference>. The write() call
is a no-op for us -- we convert the map the script builds, not a file it wrote.

Derived from simple_lanelet2's tests/cases/0310_io_roundtrip.py (BSD-3-Clause).
"""

from lanelet2.core import (
    AttributeMap,
    Lanelet,
    LineString3d,
    Point3d,
    TrafficLight,
    createMapFromLanelets,
)
from lanelet2.io import Origin, write
from lanelet2.projection import UtmProjector

points = [Point3d(10 + i, float(i) * 12.0, float(i % 2), 0.0) for i in range(4)]

left = LineString3d(20, [points[0], points[1]])
left.attributes["type"] = "line_thin"
left.attributes["subtype"] = "solid"

right = LineString3d(21, [points[2], points[3]])
right.attributes["type"] = "line_thin"
right.attributes["subtype"] = "dashed"

lanelet = Lanelet(30, left, right)
lanelet.attributes["subtype"] = "road"
lanelet.addRegulatoryElement(
    TrafficLight(50, AttributeMap(), [LineString3d(22, [points[0], points[3]])])
)

lanelet_map = createMapFromLanelets([lanelet])

origin = Origin(49.0, 8.4, 0.0)
projector = UtmProjector(origin)
write("out.osm", lanelet_map, projector)
