"""Pure-Python projections, in `math` only, so the zero-dependency core survives.

Two jobs live here, and they are not the same job.

**UTM forward, for georeferencing metadata.** `UtmProjector(origin,
useOffset=True)` -- lanelet2's default -- subtracts the origin's easting and
northing, so map coordinates are metres relative to the origin rather than
absolute UTM. A `<geoReference>` that named only the zone would be wrong by a few
hundred kilometres. Getting it right needs the origin's actual easting and
northing, which needs a real projection: this is the Krüger series that PROJ and
GeographicLib use, truncated at the fourth order -- sub-millimetre within a zone.
Nothing in that path touches the geometry; the map is already in metres, and the
projection only decides what the header says those metres mean.

**ECEF and the local tangent plane, which genuinely move points.**
`GeocentricProjector` is the one projector whose output is *not* a planar metre
frame: it emits earth-centred XYZ, where a road near Tokyo sits at around
(-3.96e6, 3.35e6, 3.70e6) and "up" is a mixture of all three axes. Feeding those
x and y straight into a plan view would foreshorten every road and tilt every
flat one, so such a map is rotated onto a tangent plane first. That is what
`ecef_to_geodetic` and `enu_basis` are for.
"""

from __future__ import annotations

import math
import re

from ..geometry.vec import Vec3

# WGS84.
_A = 6378137.0
_F = 1.0 / 298.257223563
_K0 = 0.9996
_FALSE_EASTING = 500000.0
_FALSE_NORTHING_SOUTH = 10000000.0

_B = _A * (1.0 - _F)
_E2 = _F * (2.0 - _F)
_EP2 = _E2 / (1.0 - _E2)

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


# --------------------------------------------------------------------------
# Earth-centred coordinates and the local tangent plane
# --------------------------------------------------------------------------


def geodetic_to_ecef(latitude: float, longitude: float, altitude: float = 0.0) -> Vec3:
    """WGS84 degrees and metres to earth-centred, earth-fixed XYZ."""
    phi = math.radians(latitude)
    lam = math.radians(longitude)
    sin_phi = math.sin(phi)
    # Radius of curvature in the prime vertical.
    nu = _A / math.sqrt(1.0 - _E2 * sin_phi * sin_phi)
    return (
        (nu + altitude) * math.cos(phi) * math.cos(lam),
        (nu + altitude) * math.cos(phi) * math.sin(lam),
        (nu * (1.0 - _E2) + altitude) * sin_phi,
    )


def ecef_to_geodetic(x: float, y: float, z: float) -> Vec3:
    """Earth-centred XYZ back to WGS84 `(latitude, longitude, altitude)`.

    Bowring's closed form. Iterating instead would converge too, but a closed
    form has no tolerance to tune and no way to stop short of the answer; the
    residual here is well under a micrometre for anything on or near the surface.
    """
    longitude = math.atan2(y, x)
    p = math.hypot(x, y)
    if p == 0.0:
        # On the axis: latitude is +-90 and longitude is arbitrary, so say 0.
        pole = math.copysign(90.0, z or 1.0)
        return pole, 0.0, abs(z) - _B

    theta = math.atan2(z * _A, p * _B)
    latitude = math.atan2(
        z + _EP2 * _B * math.sin(theta) ** 3,
        p - _E2 * _A * math.cos(theta) ** 3,
    )
    sin_lat = math.sin(latitude)
    nu = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    altitude = p / math.cos(latitude) - nu
    return math.degrees(latitude), math.degrees(longitude), altitude


def enu_basis(latitude: float, longitude: float) -> tuple[Vec3, Vec3, Vec3]:
    """The east, north and up unit vectors at a point, in ECEF axes.

    Returned as rows, so `east . d` is the eastward component of an ECEF offset.
    """
    phi = math.radians(latitude)
    lam = math.radians(longitude)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    return (
        (-sin_lam, cos_lam, 0.0),
        (-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi),
        (cos_phi * cos_lam, cos_phi * sin_lam, sin_phi),
    )


def ecef_to_enu(point: Vec3, anchor: Vec3, basis: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    """One ECEF point as east/north/up metres about `anchor`.

    The basis is passed in rather than recomputed because a map rotates tens of
    thousands of points through the same one.
    """
    dx = point[0] - anchor[0]
    dy = point[1] - anchor[1]
    dz = point[2] - anchor[2]
    return tuple(row[0] * dx + row[1] * dy + row[2] * dz for row in basis)  # type: ignore[return-value]


# --------------------------------------------------------------------------
# MGRS grid squares
# --------------------------------------------------------------------------
# Autoware georeferences by naming a 100 km grid square rather than an origin,
# so the square's own south-west corner is the frame the coordinates are in.

# I and O are omitted throughout, because they read as 1 and 0.
_MGRS_COLUMNS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
_MGRS_ROWS = "ABCDEFGHJKLMNPQRSTUV"
_MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"

_MGRS_PATTERN = re.compile(
    r"^\s*(?P<zone>[0-9]{1,2})\s*(?P<band>[C-HJ-NP-X])\s*"
    r"(?P<column>[A-HJ-NP-Z])(?P<row>[A-HJ-NP-V])\s*$",
    re.IGNORECASE,
)


def band_latitude(band: str) -> tuple[float, float]:
    """The latitude range a band letter covers. X is 12 degrees tall, not 8."""
    index = _MGRS_BANDS.index(band.upper())
    south = -80.0 + 8.0 * index
    return south, (south + 12.0 if band.upper() == "X" else south + 8.0)


def mgrs_square_corner(code: str) -> tuple[int, float, float, float] | None:
    """Decode an MGRS grid square to `(zone, easting, northing, latitude)`.

    Easting and northing are the square's south-west corner in that zone's UTM.
    The row letters repeat every 2000 km, so the band letter is what says which
    repetition is meant -- without it the answer is ambiguous by whole continents.

    Returns None if the code is not a well-formed 100 km square designator.
    """
    match = _MGRS_PATTERN.match(code or "")
    if match is None:
        return None

    zone = int(match.group("zone"))
    if not 1 <= zone <= 60:
        return None
    band = match.group("band").upper()
    column = match.group("column").upper()
    row = match.group("row").upper()

    columns = _MGRS_COLUMNS[(zone - 1) % 3]
    if column not in columns:
        return None
    easting = (columns.index(column) + 1) * 100000.0

    # Even zones start their row lettering five letters up the alphabet.
    offset = 0 if zone % 2 else 5
    northing = ((_MGRS_ROWS.index(row) - offset) % 20) * 100000.0

    # Lift it into the band by whole 2000 km cycles.
    south, north = band_latitude(band)
    floor_northing = utm_forward(south, central_meridian(zone), zone)[1]
    while northing < floor_northing - 100000.0:
        northing += 2000000.0

    return zone, easting, northing, (south + north) / 2.0


def mgrs_code_offsets(code: str) -> tuple[int, float, float, float] | None:
    """PROJ `+x_0` / `+y_0` for coordinates measured inside a named grid square.

    Returns `(zone, x_0, y_0, latitude)`, or None if the code is malformed.
    """
    decoded = mgrs_square_corner(code)
    if decoded is None:
        return None
    zone, easting, northing, latitude = decoded
    base_northing = _FALSE_NORTHING_SOUTH if latitude < 0.0 else 0.0
    return zone, _FALSE_EASTING - easting, base_northing - northing, latitude
