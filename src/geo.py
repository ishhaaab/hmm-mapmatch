"""Shared geospatial helpers (metres / degrees only, no shapely).

All angles are in degrees unless noted; lat/lon are EPSG:4326; distances are
metres. The road model is consumed as straight segments between edge endpoint
nodes (see graph.py), so projection works on endpoint pairs rather than full
line geometry.
"""
from __future__ import annotations

import math

M_PER_DEG_LAT = 111_320.0
EARTH_RADIUS_M = 6_371_000.0


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to point 2.

    Returns [0, 360) clockwise from north; degenerate (zero-length) input
    returns 0.0.
    """
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two degree points."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(max(a, 0.0), 1.0)))


def project_to_segment(
    lat: float,
    lon: float,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
):
    """Project point P onto straight segment A->B in a local planar frame.

    Returns (frac, dist_m, lat_f, lon_f):
      frac   — clamped along-segment fraction in [0, 1] (0 at A, 1 at B);
      dist_m — perpendicular distance from P to the clamped foot, metres;
      (lat_f, lon_f) — foot coordinates, degrees.

    A degenerate (zero-length) segment returns frac=0, dist = haversine(P, A),
    foot = A.

    This is the map-matching geometry primitive: `dist_m` is the emission
    argument, and `frac` seeds the route-distance computation (partial edge
    distance + shortest network path + partial edge distance).
    """
    lon_a, lon_b, lon = float(lon_a), float(lon_b), float(lon)
    lat_a, lat_b, lat = float(lat_a), float(lat_b), float(lat)

    east_scale = M_PER_DEG_LAT * math.cos(math.radians(lat_a))
    abx = (lon_b - lon_a) * east_scale
    aby = (lat_b - lat_a) * M_PER_DEG_LAT
    apx = (lon - lon_a) * east_scale
    apy = (lat - lat_a) * M_PER_DEG_LAT

    len2 = abx * abx + aby * aby
    if len2 <= 0.0:
        return 0.0, haversine_m(lat, lon, lat_a, lon_a), lat_a, lon_a

    frac = min(max((apx * abx + apy * aby) / len2, 0.0), 1.0)
    fx = apx - frac * abx
    fy = apy - frac * aby
    return (
        frac,
        math.sqrt(fx * fx + fy * fy),
        lat_a + frac * (lat_b - lat_a),
        lon_a + frac * (lon_b - lon_a),
    )