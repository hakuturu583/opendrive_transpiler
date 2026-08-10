"""A curving road whose width tapers, plus elevation.

The planView is emitted as one <line> per polyline segment, so every vertex here
should survive into the .xodr exactly. The taper exercises the piecewise-linear
width profile, and the rising z the elevation profile.
"""

import math

from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId

RADIUS = 40.0
STEPS = 12

left_points = []
right_points = []
for step in range(STEPS + 1):
    angle = math.radians(90.0 * step / STEPS)
    half_width = 1.5 + 1.0 * step / STEPS  # 3.0 m widening to 5.0 m
    z = 0.5 * step
    # Travel runs counter-clockwise, so the centre of the arc -- the smaller
    # radius -- is on the left.
    left_points.append(
        Point3d(
            getId(),
            (RADIUS - half_width) * math.cos(angle),
            (RADIUS - half_width) * math.sin(angle),
            z,
        )
    )
    right_points.append(
        Point3d(
            getId(),
            (RADIUS + half_width) * math.cos(angle),
            (RADIUS + half_width) * math.sin(angle),
            z,
        )
    )

left = LineString3d(getId(), left_points)
left.attributes["type"] = "line_thin"
left.attributes["subtype"] = "solid"
right = LineString3d(getId(), right_points)
right.attributes["type"] = "curbstone"
right.attributes["subtype"] = "high"

lanelet = Lanelet(getId(), left, right)
lanelet.attributes["subtype"] = "road"
lanelet.attributes["location"] = "urban"
lanelet.attributes["speed_limit"] = "50"

lanelet_map = createMapFromLanelets([lanelet])
