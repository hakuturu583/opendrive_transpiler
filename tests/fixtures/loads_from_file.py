"""A script that reads its map from disk instead of building it.

There is no AST to translate here -- the map exists only at runtime -- so this
must be reported rather than silently converted to an empty network.
"""

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector

projector = UtmProjector(Origin(49.0, 8.4, 0.0))
lanelet_map = load("mapping_example.osm", projector)
