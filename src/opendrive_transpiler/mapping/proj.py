"""A pure-Python UTM forward projection, for georeferencing metadata only.

`UtmProjector(origin, useOffset=True)` -- lanelet2's default -- subtracts the
origin's easting and northing, so map coordinates are metres relative to the
origin rather than absolute UTM. A `<geoReference>` that named only the zone
would therefore be wrong by a few hundred kilometres.

Getting it right needs the origin's actual easting and northing, which needs a
real projection. This is the Krüger series that PROJ and GeographicLib use,
truncated at the fourth order -- sub-millimetre within a UTM zone, and about
forty lines of `math`, so the zero-dependency core survives.

Nothing here touches the geometry: the map is already in metres. This only
decides what the header says the coordinates mean.
"""

from __future__ import annotations

import math

# WGS84.
_A = 6378137.0
_F = 1.0 / 298.257223563
_K0 = 0.9996
_FALSE_EASTING = 500000.0
_FALSE_NORTHING_SOUTH = 10000000.0

_N = _F / (2.0 - _F)
_RADIUS = (_A / (1.0 + _N)) * (1.0 + _N**2 / 4.0 + _N**4 / 64.0)

# Krüger series coefficients alpha_1..alpha_3.
_ALPHA = (
    _N / 2.0 - 2.0 * _N**2 / 3.0 + 5.0 * _N**3 / 16.0,
    13.0 * _N**2 / 48.0 - 3.0 * _N**3 / 5.0,
    61.0 * _N**3 / 240.0,
)


def utm_zone(longitude: float) -> int:
    """UTM zone number for a longitude, 1..60."""
    zone = math.floor((longitude + 180.0) / 6.0) + 1
    # +180 lands exactly on the wrap; clamp rather than produce zone 61.
    return min(max(zone, 1), 60)


def central_meridian(zone: int) -> float:
    return (zone - 1) * 6.0 - 180.0 + 3.0


def utm_forward(latitude: float, longitude: float, zone: int | None = None) -> tuple[float, float]:
    """Project WGS84 degrees to UTM `(easting, northing)` in metres.

    The zone may be forced, which matters because lanelet2 pins the zone from
    the origin and keeps using it for points that have drifted into the next one.
    """
    zone = utm_zone(longitude) if zone is None else zone
    phi = math.radians(latitude)
    delta = math.radians(longitude - central_meridian(zone))

    root_n = 2.0 * math.sqrt(_N) / (1.0 + _N)
    t = math.sinh(math.atanh(math.sin(phi)) - root_n * math.atanh(root_n * math.sin(phi)))
    xi = math.atan2(t, math.cos(delta))
    eta = math.atanh(math.sin(delta) / math.hypot(1.0, t))

    easting = eta
    northing = xi
    for index, alpha in enumerate(_ALPHA, start=1):
        easting += alpha * math.cos(2.0 * index * xi) * math.sinh(2.0 * index * eta)
        northing += alpha * math.sin(2.0 * index * xi) * math.cosh(2.0 * index * eta)

    easting = _FALSE_EASTING + _K0 * _RADIUS * easting
    northing = _K0 * _RADIUS * northing
    if latitude < 0.0:
        northing += _FALSE_NORTHING_SOUTH
    return easting, northing


def utm_offsets(latitude: float, longitude: float) -> tuple[float, float]:
    """PROJ `+x_0` / `+y_0` that reproduce lanelet2's `useOffset=True` frame.

    lanelet2 emits `x = easting - origin_easting`. PROJ computes
    `x = k0 * X + x_0`, and plain UTM uses `x_0 = 500000`, so the offset that
    matches is `500000 - origin_easting`; likewise for northing.
    """
    easting, northing = utm_forward(latitude, longitude)
    base_northing = _FALSE_NORTHING_SOUTH if latitude < 0.0 else 0.0
    return _FALSE_EASTING - easting, base_northing - northing


def mgrs_square_offsets(latitude: float, longitude: float) -> tuple[float, float]:
    """PROJ `+x_0` / `+y_0` for MGRS coordinates within their 100 km square.

    Autoware's MGRS projector reports metres inside a 100 km grid square, which
    is UTM truncated to the square's south-west corner.
    """
    easting, northing = utm_forward(latitude, longitude)
    square_east = math.floor(easting / 100000.0) * 100000.0
    square_north = math.floor(northing / 100000.0) * 100000.0
    base_northing = _FALSE_NORTHING_SOUTH if latitude < 0.0 else 0.0
    return _FALSE_EASTING - square_east, base_northing - square_north
