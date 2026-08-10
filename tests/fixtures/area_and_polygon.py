"""A road alongside map furniture: an Area with a hole, and a standalone Polygon.

Neither has an OpenDRIVE road equivalent, so neither is converted yet. What this
fixture pins is that they are *reported* rather than dropped in silence -- the
road still converts, and the run says exactly what it left behind.

Derived from simple_lanelet2's tests/cases/0230_area.py and
0310_io_roundtrip.py (BSD-3-Clause), with the test harness removed.
"""

from lanelet2.core import (
    Area,
    AttributeMap,
    Lanelet,
    LineString3d,
    Point3d,
    Polygon3d,
    createMapFromLanelets,
    getId,
)


def ring(corners):
    """One closed ring, as the list of line strings lanelet2 expects."""
    points = [Point3d(getId(), x, y, 0.0) for x, y in corners]
    return [LineString3d(getId(), [points[i], points[(i + 1) % len(points)]])
            for i in range(len(points))]


# A plain road, so there is something to actually convert.
left = LineString3d(getId(), [Point3d(getId(), 0.0, 2.0, 0.0), Point3d(getId(), 40.0, 2.0, 0.0)])
left.attributes["type"] = "line_thin"
left.attributes["subtype"] = "solid"
right = LineString3d(getId(), [Point3d(getId(), 0.0, -2.0, 0.0), Point3d(getId(), 40.0, -2.0, 0.0)])
right.attributes["type"] = "curbstone"
right.attributes["subtype"] = "low"

road = Lanelet(getId(), left, right)
road.attributes["subtype"] = "road"
road.attributes["location"] = "urban"

lanelet_map = createMapFromLanelets([road])

# A parking area with a planted island in the middle of it.
parking = Area(
    getId(),
    ring([(5.0, -6.0), (25.0, -6.0), (25.0, -14.0), (5.0, -14.0)]),
    [ring([(12.0, -9.0), (18.0, -9.0), (18.0, -11.0), (12.0, -11.0)])],
    AttributeMap({"subtype": "parking"}),
)
lanelet_map.add(parking)

# A standalone polygon, the way a script marks out a painted zone.
zone = Polygon3d(
    getId(),
    [
        Point3d(getId(), 30.0, 4.0, 0.0),
        Point3d(getId(), 36.0, 4.0, 0.0),
        Point3d(getId(), 36.0, 8.0, 0.0),
    ],
)
zone.attributes["type"] = "parking"
lanelet_map.add(zone)
