"""A script whose map-building lives in a helper package next to it.

Factoring node and lanelet factories into a `helpers` module is idiomatic, and
leaving those imports unresolved meant such a script converted to nothing at all.
The helper is interpreted the same symbolic way as this file -- never executed.
"""

import pathlib

from helpers import straight_lanelet
from lanelet2.core import createMapFromLanelets

# Scripts reach for __file__ to place their output. Irrelevant to the conversion,
# but it has to resolve.
OUTPUT_DIR = pathlib.Path(__file__).parent

lanelet_map = createMapFromLanelets([straight_lanelet(), straight_lanelet(y_offset=3.0)])
