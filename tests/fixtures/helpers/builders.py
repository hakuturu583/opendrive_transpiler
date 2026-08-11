"""Map-building helpers, factored out the way real generator scripts factor them."""

from lanelet2.core import Lanelet, LineString3d, Point3d, getId

LANE_WIDTH = 3.0


def straight_lanelet(y_offset=0.0, length=20.0):
    left = LineString3d(
        getId(),
        [
            Point3d(getId(), 0.0, y_offset + LANE_WIDTH, 0.0),
            Point3d(getId(), length, y_offset + LANE_WIDTH, 0.0),
        ],
    )
    right = LineString3d(
        getId(),
        [Point3d(getId(), 0.0, y_offset, 0.0), Point3d(getId(), length, y_offset, 0.0)],
    )
    lanelet = Lanelet(getId(), left, right)
    lanelet.attributes["subtype"] = "road"
    return lanelet
