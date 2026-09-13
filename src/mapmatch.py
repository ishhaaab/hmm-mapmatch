"""Full HMM map-matching pipeline: candidates -> emission/transition -> Viterbi.

`match_trace` wires the pieces from hmm.py and candidates.py together for a
whole noisy trace. The two Phase-2 baselines the project must beat (raw
haversine polyline, nearest-segment snap) live here too, along with the
route-distance terms shared by the transition model and the baselines.

Units: coordinates in degrees (lat/lon), distances in metres, times in
seconds; the road network is the straight-segment model documented in
graph.py.

Route-distance rule of thumb (the classic bug): the driving distance runs
from the *projection foot on edge A* to the *projection foot on edge B* —
partial distance to A's exit node + shortest network path + partial distance
from B's entry node to the foot. Consecutive fixes on the same edge just use
|frac diff| x length. The three helper-terms below encode this rule once, so
the standalone `route_distance_between` and the cache-aware
`_transition_matrix` cannot drift apart.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np

from src.candidates import Candidate, CandidateGrid
from src.geo import haversine_m
from src.hmm import emission_logprob, transition_logprob, viterbi

ROUTE_CUTOFF_M = 2_000.0  # network distances beyond this are treated as impossible


def _along_edge_distance(
    cand_a: Candidate, cand_b: Candidate, edge_lengths: Dict[int, float]
) -> float:
    """Driving distance between two feet that lie on the same edge (metres)."""
    return abs(cand_b.frac - cand_a.frac) * edge_lengths[cand_a.edge_id]


def _remaining_to_exit(cand: Candidate, edge_lengths: Dict[int, float]) -> float:
    """Distance from cand's foot along travel direction to its edge's exit."""
    return (1.0 - cand.frac) * edge_lengths[cand.edge_id]


def _entry_to_foot(cand: Candidate, edge_lengths: Dict[int, float]) -> float:
    """Distance from cand's edge's entry node to cand's foot."""
    return cand.frac * edge_lengths[cand.edge_id]


def route_distance_between(
    cand_a: Candidate,
    cand_b: Candidate,
    edge_lengths: Dict[int, float],
    graph: nx.MultiDiGraph,
) -> float:
    """Driving distance between two candidate projection points (metres).

    Same edge: |frac_b - frac_a| x length (motion along that edge).
    Different edges: remaining distance to edge A's exit node + shortest
    network path to edge B's entry node + distance to the foot on edge B.
    Returns inf when no network path exists.
    """
    if cand_a.edge_id == cand_b.edge_id:
        return _along_edge_distance(cand_a, cand_b, edge_lengths)

    try:
        net = nx.shortest_path_length(graph, cand_a.v, cand_b.u, weight="length_m")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")

    return (
        _remaining_to_exit(cand_a, edge_lengths)
        + net
        + _entry_to_foot(cand_b, edge_lengths)
    )


def _emission_matrix(cands: Sequence[Candidate], sigma: float) -> np.ndarray:
    """Log emission vector over one fix's candidate set."""
    return np.array([emission_logprob(c.dist_m, sigma) for c in cands])


def _transition_matrix(
    cands_prev: Sequence[Candidate],
    cands_curr: Sequence[Candidate],
    euclid_m: float,
    edge_lengths: Dict[int, float],
    graph: nx.MultiDiGraph,
    beta: float,
    net_cache: Dict[int, Dict[int, float]],
) -> np.ndarray:
    """(S_{t-1}, S_t) log transition matrix between two consecutive fixes.

    net_cache: node -> {node: driving distance} memo shared across the whole
    trace; single-source Dijkstra is run once per unique exit node. Uses the
    same route-distance terms as `route_distance_between`, but resolves the
    network segment from the cache instead of one full Dijkstra per pair.
    """
    S_prev, S_curr = len(cands_prev), len(cands_curr)
    trans = np.zeros((S_prev, S_curr))

    net_from: Dict[int, Dict[int, float]] = {}
    for c in cands_prev:
        if c.v in net_from:
            continue
        if c.v not in net_cache:
            try:
                net_cache[c.v] = nx.single_source_dijkstra_path_length(
                    graph, c.v, weight="length_m", cutoff=ROUTE_CUTOFF_M
                )
            except nx.NodeNotFound:
                net_cache[c.v] = {}
        net_from[c.v] = net_cache[c.v]

    for j, cj in enumerate(cands_prev):
        for k, ck in enumerate(cands_curr):
            if cj.edge_id == ck.edge_id:
                rd = _along_edge_distance(cj, ck, edge_lengths)
            else:
                rd = (
                    _remaining_to_exit(cj, edge_lengths)
                    + net_from[cj.v].get(ck.u, float("inf"))
                    + _entry_to_foot(ck, edge_lengths)
                )
            trans[j, k] = transition_logprob(rd, euclid_m, beta)
    return trans


def decode_trace(
    fixes: Mapping[str, np.ndarray],
    grid: CandidateGrid,
    graph: nx.MultiDiGraph,
    edge_lengths: Dict[int, float],
    sigma: float,
    beta: float,
) -> Tuple[List[Candidate], np.ndarray]:
    """Decode one noisy trace; return (matched candidates, kept fix indices).

    fixes: dict with 'lat'/'lon'/'t' arrays (degrees / metres / seconds).
    Fixes with no candidate in range are skipped (sparse GPS — the decoder
    cannot invent states), so the returned candidates align with `indices`.
    """
    lat, lon = fixes["lat"], fixes["lon"]
    T = len(lat)
    if T == 0:
        return [], np.array([], dtype=int)

    cands_per_fix = [grid.query(float(lat[t]), float(lon[t])) for t in range(T)]
    keep = np.array([t for t in range(T) if cands_per_fix[t]], dtype=int)
    if len(keep) == 0:
        return [], keep
    cands_per_fix = [cands_per_fix[t] for t in keep]
    lat_kept, lon_kept = lat[keep], lon[keep]

    emissions = [_emission_matrix(cands, sigma) for cands in cands_per_fix]

    net_cache: Dict[int, Dict[int, float]] = {}
    transitions = [
        _transition_matrix(
            cands_per_fix[t - 1], cands_per_fix[t],
            haversine_m(lat_kept[t - 1], lon_kept[t - 1],
                        lat_kept[t], lon_kept[t]),
            edge_lengths, graph, beta, net_cache,
        )
        for t in range(1, len(cands_per_fix))
    ]

    s0 = len(cands_per_fix[0])
    starts = np.full(s0, -math.log(s0))
    path = viterbi(emissions, transitions, starts)
    return [cands_per_fix[t][path[t]] for t in range(len(path))], keep


def match_trace(
    fixes: Mapping[str, np.ndarray],
    grid: CandidateGrid,
    graph: nx.MultiDiGraph,
    edge_lengths: Dict[int, float],
    sigma: float,
    beta: float,
) -> List[Candidate]:
    """Decode one noisy trace into the list of matched Candidate objects.

    Convenience over `decode_trace` for callers that do not need the fix
    index alignment (e.g. `refit_sigma_beta` in em.py does).
    """
    return decode_trace(fixes, grid, graph, edge_lengths, sigma, beta)[0]


def nearest_snap(
    fixes: Mapping[str, np.ndarray], grid: CandidateGrid
) -> List[Candidate]:
    """Baseline: assign each fix to its closest candidate, independently."""
    out = []
    for lat, lon in zip(fixes["lat"], fixes["lon"]):
        cands = grid.query(float(lat), float(lon))
        if cands:
            out.append(cands[0])  # CandidateGrid.query sorts nearest first
    return out


def raw_haversine_length(fixes: Mapping[str, np.ndarray]) -> float:
    """Baseline: naive polyline length of the noisy fixes (metres)."""
    lat, lon = fixes["lat"], fixes["lon"]
    return sum(
        haversine_m(lat[t - 1], lon[t - 1], lat[t], lon[t])
        for t in range(1, len(lat))
    )


def path_route_distance(
    path: Sequence[Candidate],
    edge_lengths: Dict[int, float],
    graph: nx.MultiDiGraph,
) -> float:
    """Total driving distance implied by a matched candidate path (metres)."""
    return sum(
        route_distance_between(a, b, edge_lengths, graph)
        for a, b in zip(path, path[1:])
    )