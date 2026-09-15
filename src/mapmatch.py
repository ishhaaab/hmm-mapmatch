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
from B's entry node to the foot. On the same directed edge, progress is
constrained to be non-negative so GPS projection jitter is not interpreted as
physical reverse travel. The three helper-terms below encode this rule once, so
the standalone `route_distance_between` and the cache-aware
`_transition_matrix` cannot drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import NamedTuple

import networkx as nx
import numpy as np

from src.candidates import Candidate, CandidateGrid
from src.geo import haversine_m
from src.hmm import emission_logprob, posterior_marginals, transition_logprob, viterbi

ROUTE_DETOUR_ALLOWANCE_M = 2_000.0
U_TURN_LOG_PENALTY = 30.0
DEFAULT_OBSERVATION_INTERVAL_S = 5.0


class MatchedRoute(NamedTuple):
    """Network route reconstructed from a sequence of matched candidates."""

    edge_ids: list[int]
    total_m: float
    streets_m: float
    connectors_m: float


def thin_trace(
    fixes: Mapping[str, np.ndarray],
    min_interval_s: float = DEFAULT_OBSERVATION_INTERVAL_S,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Keep temporally separated observations and always retain both ends.

    At 1 Hz, 15 m GPS noise is larger than typical between-fix movement. The
    N&K transition model then explains jitter rather than vehicle motion.
    Temporal thinning is shared preprocessing for the HMM and its baselines.
    Returns the sampled trace and its indices into the input trace.
    """
    min_interval_s = float(min_interval_s)
    if not math.isfinite(min_interval_s) or min_interval_s < 0.0:
        raise ValueError(
            f"min_interval_s must be finite and >= 0, got {min_interval_s}"
        )
    try:
        arrays = {
            name: np.asarray(fixes[name], dtype=float) for name in ("lat", "lon", "t")
        }
    except KeyError as exc:
        raise ValueError(f"fixes is missing required field {exc.args[0]!r}") from exc
    if any(values.ndim != 1 for values in arrays.values()):
        raise ValueError("fix lat/lon/t values must be one-dimensional")
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("fix lat/lon/t arrays must have equal lengths")
    timestamps = arrays["t"]
    if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("fix timestamps must be finite and strictly increasing")
    if len(timestamps) <= 1 or min_interval_s == 0.0:
        indices = np.arange(len(timestamps), dtype=int)
    else:
        selected = [0]
        for index in range(1, len(timestamps) - 1):
            if timestamps[index] - timestamps[selected[-1]] >= min_interval_s:
                selected.append(index)
        if selected[-1] != len(timestamps) - 1:
            selected.append(len(timestamps) - 1)
        indices = np.asarray(selected, dtype=int)
    return {name: values[indices] for name, values in arrays.items()}, indices


def _along_edge_distance(
    cand_a: Candidate, cand_b: Candidate, edge_lengths: dict[int, float]
) -> float:
    """Non-negative progress between two feet on the same directed edge."""
    return max(0.0, cand_b.frac - cand_a.frac) * edge_lengths[cand_a.edge_id]


def _remaining_to_exit(cand: Candidate, edge_lengths: dict[int, float]) -> float:
    """Distance from cand's foot along travel direction to its edge's exit."""
    return (1.0 - cand.frac) * edge_lengths[cand.edge_id]


def _entry_to_foot(cand: Candidate, edge_lengths: dict[int, float]) -> float:
    """Distance from cand's edge's entry node to cand's foot."""
    return cand.frac * edge_lengths[cand.edge_id]


def _is_immediate_uturn(cand_a: Candidate, cand_b: Candidate) -> bool:
    """Whether two directed candidates are opposite sides of one segment."""
    return cand_a.u == cand_b.v and cand_a.v == cand_b.u


def route_distance_between(
    cand_a: Candidate,
    cand_b: Candidate,
    edge_lengths: dict[int, float],
    graph: nx.MultiDiGraph,
) -> float:
    """Driving distance between two candidate projection points (metres).

    Same directed edge: max(0, frac_b - frac_a) x length. A backwards change
    in projection is treated as GPS jitter, not physical reverse travel; real
    reverse travel uses the separately directed opposite edge.
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
    edge_lengths: dict[int, float],
    graph: nx.MultiDiGraph,
    beta: float,
    net_cache: dict[int, tuple[float, dict[int, float]]],
) -> np.ndarray:
    """(S_{t-1}, S_t) log transition matrix between two consecutive fixes.

    net_cache maps a source node to (cutoff, distances). A source is recomputed
    if a later observation needs a larger cutoff, avoiding false impossibility
    from a previously truncated search.
    """
    S_prev, S_curr = len(cands_prev), len(cands_curr)
    trans = np.zeros((S_prev, S_curr))
    cutoff = euclid_m + ROUTE_DETOUR_ALLOWANCE_M

    net_from: dict[int, dict[int, float]] = {}
    for c in cands_prev:
        if c.v in net_from:
            continue
        cached = net_cache.get(c.v)
        if cached is None or cached[0] < cutoff:
            try:
                distances = nx.single_source_dijkstra_path_length(
                    graph, c.v, weight="length_m", cutoff=cutoff
                )
            except nx.NodeNotFound:
                distances = {}
            net_cache[c.v] = (cutoff, distances)
        net_from[c.v] = net_cache[c.v][1]

    for j, cj in enumerate(cands_prev):
        for k, ck in enumerate(cands_curr):
            if cj.edge_id == ck.edge_id:
                rd = _along_edge_distance(cj, ck, edge_lengths)
            else:
                network_m = net_from[cj.v].get(ck.u, float("inf"))
                if network_m > cutoff:
                    network_m = float("inf")
                rd = (
                    _remaining_to_exit(cj, edge_lengths)
                    + network_m
                    + _entry_to_foot(ck, edge_lengths)
                )
            trans[j, k] = transition_logprob(rd, euclid_m, beta)
            if _is_immediate_uturn(cj, ck):
                trans[j, k] -= U_TURN_LOG_PENALTY
    return trans


def _build_trellis(
    fixes: Mapping[str, np.ndarray],
    grid: CandidateGrid,
    graph: nx.MultiDiGraph,
    edge_lengths: dict[int, float],
    sigma: float,
    beta: float,
) -> tuple[
    list[list[Candidate]],
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
]:
    """Validate fixes and build the variable-width HMM trellis once."""
    try:
        lat = np.asarray(fixes["lat"], dtype=float)
        lon = np.asarray(fixes["lon"], dtype=float)
        timestamps = np.asarray(fixes["t"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"fixes is missing required field {exc.args[0]!r}") from exc
    if lat.ndim != 1 or lon.ndim != 1 or timestamps.ndim != 1:
        raise ValueError("fix lat/lon/t values must be one-dimensional")
    if not (len(lat) == len(lon) == len(timestamps)):
        raise ValueError("fix lat/lon/t arrays must have equal lengths")
    if np.any(~np.isfinite(lat)) or np.any((lat < -90.0) | (lat > 90.0)):
        raise ValueError("fix latitudes must be finite and in [-90, 90]")
    if np.any(~np.isfinite(lon)) or np.any((lon < -180.0) | (lon > 180.0)):
        raise ValueError("fix longitudes must be finite and in [-180, 180]")
    if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("fix timestamps must be finite and strictly increasing")
    T = len(lat)
    if T == 0:
        return [], np.array([], dtype=int), [], [], np.array([], dtype=float)

    cands_per_fix = [grid.query(float(lat[t]), float(lon[t])) for t in range(T)]
    keep = np.array([t for t in range(T) if cands_per_fix[t]], dtype=int)
    if len(keep) == 0:
        return [], keep, [], [], np.array([], dtype=float)
    cands_per_fix = [cands_per_fix[t] for t in keep]
    lat_kept, lon_kept = lat[keep], lon[keep]

    emissions = [_emission_matrix(cands, sigma) for cands in cands_per_fix]

    net_cache: dict[int, tuple[float, dict[int, float]]] = {}
    transitions = [
        _transition_matrix(
            cands_per_fix[t - 1],
            cands_per_fix[t],
            haversine_m(lat_kept[t - 1], lon_kept[t - 1], lat_kept[t], lon_kept[t]),
            edge_lengths,
            graph,
            beta,
            net_cache,
        )
        for t in range(1, len(cands_per_fix))
    ]

    s0 = len(cands_per_fix[0])
    starts = np.full(s0, -math.log(s0))
    return cands_per_fix, keep, emissions, transitions, starts


def decode_trace(
    fixes: Mapping[str, np.ndarray],
    grid: CandidateGrid,
    graph: nx.MultiDiGraph,
    edge_lengths: dict[int, float],
    sigma: float,
    beta: float,
) -> tuple[list[Candidate], np.ndarray]:
    """Decode one noisy trace; return (matched candidates, kept fix indices).

    Fixes with no candidate in range are skipped because the decoder cannot
    invent states. Returned candidates therefore align with ``indices``.
    """
    cands_per_fix, keep, emissions, transitions, starts = _build_trellis(
        fixes, grid, graph, edge_lengths, sigma, beta
    )
    if not cands_per_fix:
        return [], keep
    path = viterbi(emissions, transitions, starts)
    return [cands_per_fix[t][path[t]] for t in range(len(path))], keep


def decode_trace_with_confidence(
    fixes: Mapping[str, np.ndarray],
    grid: CandidateGrid,
    graph: nx.MultiDiGraph,
    edge_lengths: dict[int, float],
    sigma: float,
    beta: float,
) -> tuple[list[Candidate], np.ndarray, float]:
    """Decode a trace and return a mean posterior margin in ``[0, 1]``.

    The margin at each retained fix is the posterior mass of the Viterbi state
    minus the largest alternative state's mass. A negative margin means the
    sequence-optimal state is not the marginally most likely state and is
    conservatively counted as zero.
    """
    cands_per_fix, keep, emissions, transitions, starts = _build_trellis(
        fixes, grid, graph, edge_lengths, sigma, beta
    )
    if not cands_per_fix:
        return [], keep, 0.0
    indices = viterbi(emissions, transitions, starts)
    matched = [cands_per_fix[t][indices[t]] for t in range(len(indices))]
    posteriors = posterior_marginals(emissions, transitions, starts)
    margins = []
    for posterior, selected in zip(posteriors, indices):
        selected_p = float(posterior[selected])
        if len(posterior) == 1:
            margins.append(selected_p)
        else:
            alternative = float(np.max(np.delete(posterior, selected)))
            margins.append(max(0.0, selected_p - alternative))
    confidence = float(np.mean(margins))
    return matched, keep, min(1.0, max(0.0, confidence))


def match_trace(
    fixes: Mapping[str, np.ndarray],
    grid: CandidateGrid,
    graph: nx.MultiDiGraph,
    edge_lengths: dict[int, float],
    sigma: float,
    beta: float,
) -> list[Candidate]:
    """Decode one noisy trace into the list of matched Candidate objects.

    Convenience over `decode_trace` for callers that do not need the fix
    index alignment (e.g. `refit_sigma_beta` in em.py does).
    """
    return decode_trace(fixes, grid, graph, edge_lengths, sigma, beta)[0]


def nearest_snap(
    fixes: Mapping[str, np.ndarray], grid: CandidateGrid
) -> list[Candidate]:
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
        haversine_m(lat[t - 1], lon[t - 1], lat[t], lon[t]) for t in range(1, len(lat))
    )


def path_route_distance(
    path: Sequence[Candidate],
    edge_lengths: dict[int, float],
    graph: nx.MultiDiGraph,
) -> float:
    """Total driving distance implied by a matched candidate path (metres)."""
    return sum(
        route_distance_between(a, b, edge_lengths, graph) for a, b in pairwise(path)
    )


def reconstruct_route(
    path: Sequence[Candidate],
    edge_lengths: dict[int, float],
    graph: nx.MultiDiGraph,
) -> MatchedRoute:
    """Reconstruct a stable network route from per-fix candidate states.

    Consecutive observations on one edge form one run, so along-edge GPS
    jitter is not counted repeatedly. Candidate-edge switches are connected by
    weighted shortest paths. Raises ``ValueError`` if a switch is unroutable.
    """
    if not path:
        return MatchedRoute([], 0.0, 0.0, 0.0)

    runs: list[tuple[Candidate, Candidate]] = []
    for candidate in path:
        if runs and candidate.edge_id == runs[-1][0].edge_id:
            runs[-1] = (runs[-1][0], candidate)
        else:
            runs.append((candidate, candidate))

    edge_ids = [int(runs[0][0].edge_id)]
    if len(runs) == 1:
        first, last = runs[0]
        distance = _along_edge_distance(first, last, edge_lengths)
        return MatchedRoute(edge_ids, float(distance), float(distance), 0.0)

    streets = _remaining_to_exit(runs[0][0], edge_lengths)
    streets += sum(edge_lengths[first.edge_id] for first, _ in runs[1:-1])
    streets += _entry_to_foot(runs[-1][1], edge_lengths)

    connectors = 0.0
    for (_, previous), (current, _) in pairwise(runs):
        if previous.v != current.u:
            try:
                nodes = nx.shortest_path(
                    graph, previous.v, current.u, weight="length_m"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
                raise ValueError(
                    f"no route from edge {previous.edge_id} to {current.edge_id}"
                ) from exc
            for u, v in pairwise(nodes):
                key, attrs = min(
                    graph[u][v].items(),
                    key=lambda item: (
                        float(item[1]["length_m"]),
                        int(item[1].get("edge_id", item[0])),
                    ),
                )
                connectors += float(attrs["length_m"])
                connector_id = int(attrs.get("edge_id", key))
                if connector_id != edge_ids[-1]:
                    edge_ids.append(connector_id)
        if current.edge_id != edge_ids[-1]:
            edge_ids.append(int(current.edge_id))

    return MatchedRoute(
        edge_ids,
        float(streets + connectors),
        float(streets),
        float(connectors),
    )
