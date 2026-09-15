"""Route sampler + GPS noise model.

Sample shortest-path routes on the graph, corrupt them with a documented
noise model (Gaussian jitter + dropouts + tunnel gaps). The clean route,
including the exact edge ids traversed, is kept as stable ground truth even
when nearby roads are geometrically ambiguous. Seed every RNG.

Units: coordinates in degrees (lat/lon), distances in metres, times in
seconds, speeds in metres/second.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import networkx as nx
import numpy as np

from src.geo import M_PER_DEG_LAT

STEP_M = 10.0  # dense centreline spacing, metres
SPEED_MPS = 8.0  # simulated travel speed, metres/second (~29 km/h)
FIX_HZ = 1.0  # GPS fix rate, fixes/second
GAP_S = 60.0  # tunnel-gap length, seconds
MIN_ROUTE_M = 2_000.0  # shortest accepted route, metres
MAX_ROUTE_M = 5_000.0  # longest accepted route, metres
MAX_TRIES = 200  # OD-pair attempts per requested route


def _edge_length(d: dict[str, Any]) -> float:
    """Edge length in metres; accepts osmnx ('length') or rebuilt ('length_m')."""
    return float(d.get("length_m", d.get("length", 0.0)))


def _nx_weight(u: int, v: int, d: Mapping[Any, Any]) -> float:
    """NetworkX weight callable for simple and multi-edge graphs.

    For a MultiDiGraph, NetworkX passes the full ``key -> attributes`` mapping
    rather than one edge's attributes. Returning zero for that mapping makes
    every route an unweighted path, so explicitly select the shortest parallel
    edge here.
    """
    if "length_m" in d or "length" in d:
        return _edge_length(dict(d))
    lengths = [
        _edge_length(dict(attrs)) for attrs in d.values() if isinstance(attrs, Mapping)
    ]
    if not lengths:
        raise ValueError(f"edge {u}->{v} has no usable length attribute")
    return min(lengths)


def _densify(
    G: nx.MultiDiGraph, nodes: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a node path into a dense centreline polyline.

    Inputs: graph with node x (lon, degrees) / y (lat, degrees); node-id path.
    Outputs: (lat, lon) arrays in degrees + cumulative distance in metres,
    spaced ~STEP_M apart, including both endpoints.
    """
    if len(nodes) < 2:
        raise ValueError("a route must contain at least two nodes")
    lats = np.array([float(G.nodes[n]["y"]) for n in nodes])
    lons = np.array([float(G.nodes[n]["x"]) for n in nodes])
    seg_m = np.array(
        [min(_edge_length(d) for d in G[u][v].values()) for u, v in pairwise(nodes)],
        dtype=float,
    )
    if np.any(~np.isfinite(seg_m)) or np.any(seg_m <= 0.0):
        raise ValueError("route edges must have finite, positive lengths")
    cum = np.concatenate([[0.0], np.cumsum(seg_m)])
    total = float(cum[-1])
    n_steps = max(int(total / STEP_M), 1)
    s = np.linspace(0.0, total, n_steps + 1)
    return (
        np.interp(s, cum, lats),
        np.interp(s, cum, lons),
        s,
    )


def sample_route(G: nx.MultiDiGraph, n: int = 1, seed: int = 0) -> list[dict[str, Any]]:
    """Sample n shortest-path routes of 2-5 km each.

    Inputs: routable MultiDiGraph (edge weights in metres), n routes,
    seed (int, drives ALL randomness here).
    Output: list of route dicts with keys nodes (list[int]), segments
    (list[(u, v)]), edge_ids (list[int], the exact edge key/edge_id chosen
    per hop, including the selected parallel edge), length_m (float), lat/lon
    (deg arrays of the dense centreline), cumdist_m (metres array). Raises
    RuntimeError if too few OD pairs fall in the 2-5 km band.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if len(G) < 2:
        raise ValueError("route graph must contain at least two nodes")
    rng = np.random.default_rng(seed)
    node_ids = list(G.nodes)
    routes: list[dict[str, Any]] = []
    tries = 0
    while len(routes) < n and tries < MAX_TRIES * n:
        tries += 1
        origin = node_ids[rng.integers(len(node_ids))]
        dest = node_ids[rng.integers(len(node_ids))]
        if dest == origin:
            continue
        try:
            path = nx.shortest_path(G, origin, dest, weight=_nx_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        hops = [(int(path[i]), int(path[i + 1])) for i in range(len(path) - 1)]
        length_m = sum(min(_edge_length(d) for d in G[u][v].values()) for u, v in hops)
        if not (MIN_ROUTE_M <= length_m <= MAX_ROUTE_M):
            continue
        # Exact edge ids per hop: the min-length parallel edge is the one
        # dijkstra picks (multi-edges between the same node pair), so this
        # stays consistent with length_m and gives an exact ground-truth id.
        edge_ids = []
        for u, v in hops:
            best_key, best_d = min(
                G[u][v].items(),
                key=lambda item: (
                    _edge_length(item[1]),
                    int(item[1].get("edge_id", item[0])),
                ),
            )
            edge_ids.append(int(best_d.get("edge_id", best_key)))
        lat, lon, cum = _densify(G, path)
        routes.append(
            {
                "nodes": [int(v) for v in path],
                "segments": hops,
                "edge_ids": edge_ids,
                "length_m": float(length_m),
                "lat": lat,
                "lon": lon,
                "cumdist_m": cum,
            }
        )
    if len(routes) < n:
        raise RuntimeError(
            f"only {len(routes)}/{n} routes in the 2-5 km band after "
            f"{tries} tries; graph may be too small"
        )
    return routes


def corrupt(
    route: dict[str, Any],
    sigma_m: float = 15.0,
    dropout_p: float = 0.1,
    gaps: int = 1,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Corrupt a clean route into noisy GPS fixes + untouched ground truth.

    Inputs: route (one sample_route dict; centreline degrees, cumdist
    metres), sigma_m (jitter std, metres; supports 15/25/40), dropout_p
    (independent per-fix drop probability, ~0.1), gaps (count of ~60 s
    contiguous tunnel outages), seed (int, drives ALL randomness here).
    Outputs: (fixes, truth); fixes has lat/lon (degrees) + t (seconds) for
    surviving fixes only; truth has lat/lon/t for the full 1 Hz clean run.
    truth is never jittered or dropped.
    """
    if not np.isfinite(sigma_m) or sigma_m < 0.0:
        raise ValueError(f"sigma_m must be finite and >= 0, got {sigma_m}")
    if not np.isfinite(dropout_p) or not 0.0 <= dropout_p <= 1.0:
        raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")
    if gaps < 0:
        raise ValueError(f"gaps must be >= 0, got {gaps}")
    try:
        route_lat = np.asarray(route["lat"], dtype=float)
        route_lon = np.asarray(route["lon"], dtype=float)
        route_distance = np.asarray(route["cumdist_m"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"route is missing required field {exc.args[0]!r}") from exc
    if any(values.ndim != 1 for values in (route_lat, route_lon, route_distance)):
        raise ValueError("route lat/lon/cumdist_m values must be one-dimensional")
    if not (
        len(route_lat) == len(route_lon) == len(route_distance) and len(route_lat) >= 2
    ):
        raise ValueError("route lat/lon/cumdist_m arrays must have equal length >= 2")
    if np.any(~np.isfinite(route_lat)) or np.any(~np.isfinite(route_lon)):
        raise ValueError("route coordinates must be finite")
    if (
        np.any(~np.isfinite(route_distance))
        or route_distance[0] != 0.0
        or np.any(np.diff(route_distance) <= 0.0)
    ):
        raise ValueError("route cumulative distance must start at zero and increase")

    rng = np.random.default_rng(seed)
    duration_s = float(route_distance[-1]) / SPEED_MPS
    t = np.arange(0.0, duration_s, 1.0 / FIX_HZ)
    s = t * SPEED_MPS
    truth = {
        "lat": np.interp(s, route_distance, route_lat),
        "lon": np.interp(s, route_distance, route_lon),
        "t": t,
    }

    east_m = rng.normal(0.0, sigma_m, size=t.shape)
    north_m = rng.normal(0.0, sigma_m, size=t.shape)
    lat_rad = np.radians(truth["lat"])
    noisy_lat = truth["lat"] + north_m / M_PER_DEG_LAT
    noisy_lon = truth["lon"] + east_m / (M_PER_DEG_LAT * np.cos(lat_rad))

    keep = rng.random(t.shape) >= dropout_p
    gap_len = int(GAP_S * FIX_HZ)
    for _ in range(gaps):
        if len(t) > gap_len:
            start = rng.integers(0, len(t) - gap_len + 1)
            keep[start : start + gap_len] = False
    if not np.any(keep):  # degenerate: keep the middle fix
        keep[len(keep) // 2] = True
    fixes = {"lat": noisy_lat[keep], "lon": noisy_lon[keep], "t": t[keep]}
    return fixes, truth
