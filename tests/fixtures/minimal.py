"""The canonical five-line lanelet2 example, from simple_lanelet2's README.

Note that `left` is at y=0 and `right` at y=1, so the bounds are geometrically
swapped relative to their names -- the transpiler should notice and say so.
"""

import lanelet2
from lanelet2.core import Lanelet, LineString3d, Point3d, getId

left = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 1, 0, 0)])
right = LineString3d(getId(), [Point3d(getId(), 0, 1, 0), Point3d(getId(), 1, 1, 0)])
lanelet = Lanelet(getId(), left, right)
