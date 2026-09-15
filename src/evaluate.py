"""Evaluation: match rate, distance error, ETA-MAE delta, slices.

Route quality is measured over the reconstructed network edge sequence against
``synthesize.sample_route`` ground truth. Recall is retained as ``match_rate``
for continuity, while precision, F1, and order-aware LCS prevent a route with
many extra edges from looking perfect.

Units: distances in metres, times in seconds, speeds in metres/second,
angles in degrees. The ETA model is deliberately trivial — ETA = route
distance / constant speed — so the "downstream win" isolates the value of
a better distance estimate, not a better ETA model.

``hmm_len_m`` is the production route length: consecutive states on one edge
are collapsed and edge switches are joined through the network. The old
per-fix projection-foot sum is retained only as ``hmm_feet_len_m`` diagnostics.
Both the HMM and baselines receive the same temporal thinning.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from src.candidates import Candidate, CandidateGrid
from src.em import MODEL_PATH
from src.graph import as_routing_graph, load_graph
from src.mapmatch import (
    DEFAULT_OBSERVATION_INTERVAL_S,
    match_trace,
    nearest_snap,
    path_route_distance,
    raw_haversine_length,
    reconstruct_route,
    thin_trace,
)
from src.synthesize import SPEED_MPS, corrupt, sample_route

DEFAULT_SIGMAS = (15, 25, 40)
DEFAULT_DROPOUTS = (0.0, 0.1)
DEFAULT_ROUTE_SEEDS = (7, 8, 9, 10, 11)
DEFAULT_CORRUPT_SEED = 3
DEFAULT_GAPS = 1


def _calibrated_parameter(name: str, fallback: float) -> float:
    """Read one deployment parameter, with a fallback for partial checkouts."""
    try:
        return float(json.loads(MODEL_PATH.read_text(encoding="utf-8"))[name])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


DEFAULT_MODEL_SIGMA = _calibrated_parameter("sigma_m", 15.0)
DEFAULT_BETA = _calibrated_parameter("beta_m", 0.5)
DEFAULT_CANDIDATE_RADIUS = _calibrated_parameter("candidate_radius_m", 120.0)
DEFAULT_MAX_CANDIDATES = int(_calibrated_parameter("max_candidates", 40.0))


def _mean(rows: Sequence[dict], key: str) -> float:
    """Arithmetic mean of one numeric field in a non-empty row collection."""
    return float(sum(row[key] for row in rows) / len(rows))


def _edge_ids(seq: Sequence[Any]) -> list[int]:
    """Extract edge ids from ints or Candidate records."""
    out = []
    for c in seq:
        if isinstance(c, Candidate) or hasattr(c, "edge_id"):
            out.append(int(c.edge_id))
        else:
            out.append(int(c))
    return out


def route_match_rate(matched_ids: Sequence[Any], truth_ids: Sequence[Any]) -> float:
    """Recall of unique ground-truth edge ids.

    Inputs: matched edge ids (Viterbi path order, repeats allowed),
    truth edge ids (sample_route's `edge_ids`). Order-insensitive; the
    order-aware companion is `lcs_match_rate`. Empty truth returns 0.0.
    """
    truth = _edge_ids(truth_ids)
    if not truth:
        return 0.0
    matched = set(_edge_ids(matched_ids))
    return len(matched & set(truth)) / len(set(truth))


def route_precision(matched_ids: Sequence[Any], truth_ids: Sequence[Any]) -> float:
    """Precision of unique matched edge ids; empty matches return 0.0."""
    matched = set(_edge_ids(matched_ids))
    if not matched:
        return 0.0
    return len(matched & set(_edge_ids(truth_ids))) / len(matched)


def route_f1(matched_ids: Sequence[Any], truth_ids: Sequence[Any]) -> float:
    """Harmonic mean of unique-edge precision and recall."""
    precision = route_precision(matched_ids, truth_ids)
    recall = route_match_rate(matched_ids, truth_ids)
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )


def _lcs_len(a: list[int], b: list[int]) -> int:
    """Longest-common-subsequence length (two-row DP, O(N*M) time)."""
    if not a or not b:
        return 0
    # Iterate over the shorter sequence in the inner loop.
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _collapse_runs(ids: list[int]) -> list[int]:
    """Collapse consecutive duplicates (per-fix path -> street sequence)."""
    runs: list[int] = []
    for e in ids:
        if not runs or e != runs[-1]:
            runs.append(e)
    return runs


def lcs_match_rate(matched_ids: Sequence[Any], truth_ids: Sequence[Any]) -> float:
    """Order-aware match fraction: LCS(collapsed matched, truth) / |truth|.

    Collapsing consecutive repeats first so a 400-fix path over 40 edges
    is compared as the ~40-street sequence the decoder actually chose.
    Empty truth returns 0.0.
    """
    truth = _edge_ids(truth_ids)
    if not truth:
        return 0.0
    matched_runs = _collapse_runs(_edge_ids(matched_ids))
    return _lcs_len(matched_runs, truth) / len(truth)


def walk_route_distance(
    path: Sequence[Candidate],
    edge_lengths: dict[int, float],
    graph: nx.MultiDiGraph,
) -> dict[str, float]:
    """Junction-level driven length of a matched street sequence.

    Each maximal run of equal edges is priced once (first foot -> exit on
    the first edge, full lengths in the middle, entry -> last foot on the
    final edge); every road switch is bridged with a real shortest path.
    Returns {"total_m", "streets_m", "bridges_m"}. Empty/single-fix paths
    return zero distance; an unroutable switch raises ``ValueError``.
    """
    route = reconstruct_route(path, edge_lengths, graph)
    return {
        "total_m": route.total_m,
        "streets_m": route.streets_m,
        "bridges_m": route.connectors_m,
    }


def evaluate(
    matched,
    ground_truth,
    matched_length_m: float | None = None,
    true_length_m: float | None = None,
    raw_length_m: float | None = None,
    speed_mps: float = SPEED_MPS,
) -> dict:
    """Metrics on edge-id ground truth.

    matched: list of edge ids from the Viterbi path (or Candidate records);
    ground_truth: edge_ids list from synthesize.sample_route. Returns dict with:
      match_rate — |matched ∩ truth| / |truth| over edge ids;
          order-aware alternative: longest-common-subsequence fraction.
      distance_error_m — |matched route length - true length|.
      eta_mae_delta_s — ETA MAE using raw vs matched distance.
    Length/ETA fields are populated only when the corresponding lengths are
    supplied (see `evaluate_trace` for the full pipeline version that
    computes them); match rates are always computed.
    """
    m_ids = _edge_ids(matched)
    t_ids = _edge_ids(ground_truth)
    if not math.isfinite(speed_mps) or speed_mps <= 0.0:
        raise ValueError(f"speed_mps must be finite and > 0, got {speed_mps}")
    out: dict = {
        "match_rate": route_match_rate(m_ids, t_ids),
        "edge_precision": route_precision(m_ids, t_ids),
        "edge_f1": route_f1(m_ids, t_ids),
        "lcs_rate": lcs_match_rate(m_ids, t_ids),
        "distance_error_m": None,
        "eta_hmm_err_s": None,
        "eta_raw_err_s": None,
        "eta_mae_delta_s": None,
    }
    if matched_length_m is not None and true_length_m is not None:
        lengths = (matched_length_m, true_length_m)
        if any(not math.isfinite(value) or value < 0.0 for value in lengths):
            raise ValueError("route lengths must be finite and >= 0")
        out["distance_error_m"] = abs(float(matched_length_m) - float(true_length_m))
        out["eta_hmm_err_s"] = out["distance_error_m"] / float(speed_mps)
        if raw_length_m is not None:
            if not math.isfinite(raw_length_m) or raw_length_m < 0.0:
                raise ValueError("raw_length_m must be finite and >= 0")
            raw_err = abs(float(raw_length_m) - float(true_length_m))
            out["eta_raw_err_s"] = raw_err / float(speed_mps)
            out["eta_mae_delta_s"] = out["eta_raw_err_s"] - out["eta_hmm_err_s"]
    return out


def evaluate_trace(
    matched: Sequence[Candidate],
    snap: Sequence[Candidate],
    fixes,
    route,
    edge_lengths: dict[int, float],
    graph: nx.MultiDiGraph,
    speed_mps: float = SPEED_MPS,
) -> dict:
    """Full per-route metrics: match rates, feet/walk lengths, ETA deltas.

    matched/snap: decoded Candidate paths (Viterbi / nearest-snap baseline).
    fixes: corrupt() output (lat/lon/t of surviving fixes). route: one
    sample_route dict (edge_ids + length_m ground truth). Returns a flat
    dict with per-route scalars (all floats except ids/seeds filled by the
    caller in run_noise_grid).
    """
    truth_ids = list(route["edge_ids"])
    true_len = float(route["length_m"])
    hmm_feet_len = (
        float(path_route_distance(matched, edge_lengths, graph))
        if len(matched) >= 2
        else 0.0
    )
    snap_feet_len = (
        float(path_route_distance(snap, edge_lengths, graph)) if len(snap) >= 2 else 0.0
    )
    raw_len = float(raw_haversine_length(fixes)) if len(fixes["t"]) >= 2 else 0.0
    matched_route = reconstruct_route(matched, edge_lengths, graph)
    snap_route = reconstruct_route(snap, edge_lengths, graph)
    hmm_len = matched_route.total_m
    snap_len = snap_route.total_m
    base = evaluate(
        matched_route.edge_ids, truth_ids, hmm_len, true_len, raw_len, speed_mps
    )
    snap_err = abs(snap_len - true_len)
    raw_err = abs(raw_len - true_len)
    return {
        "match_rate": base["match_rate"],
        "edge_precision": base["edge_precision"],
        "edge_f1": base["edge_f1"],
        "lcs_rate": base["lcs_rate"],
        "true_len_m": true_len,
        "hmm_len_m": hmm_len,
        "snap_len_m": snap_len,
        "raw_len_m": raw_len,
        "hmm_feet_len_m": hmm_feet_len,
        "snap_feet_len_m": snap_feet_len,
        "walk_len_m": matched_route.total_m,
        "walk_streets_m": matched_route.streets_m,
        "walk_bridges_m": matched_route.connectors_m,
        "feet_ratio": hmm_feet_len / true_len if true_len else 0.0,
        "walk_ratio": matched_route.total_m / true_len if true_len else 0.0,
        "hmm_err_m": base["distance_error_m"],
        "snap_err_m": snap_err,
        "raw_err_m": raw_err,
        "eta_hmm_err_s": base["eta_hmm_err_s"],
        "eta_raw_err_s": base["eta_raw_err_s"],
        "eta_mae_delta_s": base["eta_mae_delta_s"],
        "eta_snap_err_s": snap_err / float(speed_mps),
        "hmm_beats_snap": bool(base["distance_error_m"] < snap_err),
        "hmm_beats_raw": bool(base["distance_error_m"] < raw_err),
        "n_fixes": len(fixes["t"]),
        "n_truth_edges": len(truth_ids),
    }


def run_noise_grid(
    sigmas=DEFAULT_SIGMAS,
    dropouts=DEFAULT_DROPOUTS,
    out: Path = Path("results/metrics.json"),
    n_routes: int = 5,
    route_seeds=DEFAULT_ROUTE_SEEDS,
    corrupt_seed: int = DEFAULT_CORRUPT_SEED,
    gaps: int = DEFAULT_GAPS,
    model_sigma: float = DEFAULT_MODEL_SIGMA,
    beta: float = DEFAULT_BETA,
    candidate_radius_m: float = DEFAULT_CANDIDATE_RADIUS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    speed_mps: float = SPEED_MPS,
    observation_interval_s: float = DEFAULT_OBSERVATION_INTERVAL_S,
) -> dict:
    """Run sigma x dropout grid, persist JSON.

    Each cell changes the generating noise and dropout while decoding with one
    fixed, calibrated ``model_sigma`` and beta. This mirrors deployment and
    avoids leaking the simulated noise label into inference. All RNG is seeded;
    pass another ``corrupt_seed`` for a second-seed robustness check.
    """
    sigmas = tuple(float(s) for s in sigmas)
    dropouts = tuple(float(d) for d in dropouts)
    if not sigmas or any(not math.isfinite(s) or s <= 0.0 for s in sigmas):
        raise ValueError("sigmas must contain finite values > 0")
    if not dropouts or any(
        not math.isfinite(d) or not 0.0 <= d <= 1.0 for d in dropouts
    ):
        raise ValueError("dropouts must contain values in [0, 1]")
    if not math.isfinite(model_sigma) or model_sigma <= 0.0:
        raise ValueError(f"model_sigma must be finite and > 0, got {model_sigma}")
    if not math.isfinite(candidate_radius_m) or candidate_radius_m <= 0.0:
        raise ValueError(
            f"candidate_radius_m must be finite and > 0, got {candidate_radius_m}"
        )
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates <= 0
    ):
        raise ValueError(
            f"max_candidates must be a positive integer, got {max_candidates}"
        )
    if n_routes <= 0:
        raise ValueError(f"n_routes must be > 0, got {n_routes}")
    seeds = list(route_seeds)[:n_routes]
    if len(seeds) != n_routes:
        raise ValueError("route_seeds must contain at least n_routes values")
    nodes, edges = load_graph()
    graph = as_routing_graph(nodes, edges)
    edge_lengths: dict[int, float] = dict(zip(edges["edge_id"], edges["length_m"]))
    grid = CandidateGrid(nodes, edges, candidate_radius_m, max_candidates)

    route_rows: list[dict] = []
    cells: list[dict] = []
    for sigma in sigmas:
        for dropout in dropouts:
            per: list[dict] = []
            for route_index, rs in enumerate(seeds):
                route = sample_route(graph, n=1, seed=int(rs))[0]
                noise_seed = int(
                    np.random.SeedSequence(
                        [int(corrupt_seed), int(rs), route_index]
                    ).generate_state(1)[0]
                )
                fixes, _truth = corrupt(
                    route,
                    sigma_m=sigma,
                    dropout_p=dropout,
                    gaps=gaps,
                    seed=noise_seed,
                )
                sampled_fixes, _ = thin_trace(fixes, observation_interval_s)
                matched = match_trace(
                    sampled_fixes,
                    grid,
                    graph,
                    edge_lengths,
                    sigma=model_sigma,
                    beta=beta,
                )
                snap = nearest_snap(sampled_fixes, grid)
                m = evaluate_trace(
                    matched,
                    snap,
                    sampled_fixes,
                    route,
                    edge_lengths,
                    graph,
                    speed_mps,
                )
                m.update(
                    {
                        "sigma_m": sigma,
                        "dropout_p": dropout,
                        "route_seed": int(rs),
                        "noise_seed": noise_seed,
                        "n_input_fixes": len(fixes["t"]),
                    }
                )
                per.append(m)
                route_rows.append(m)
            n = len(per)
            cells.append(
                {
                    "sigma_m": sigma,
                    "dropout_p": dropout,
                    "n_routes": n,
                    "mean_match_rate": _mean(per, "match_rate"),
                    "mean_edge_precision": _mean(per, "edge_precision"),
                    "mean_edge_f1": _mean(per, "edge_f1"),
                    "mean_lcs_rate": _mean(per, "lcs_rate"),
                    "mean_hmm_err_m": _mean(per, "hmm_err_m"),
                    "mean_snap_err_m": _mean(per, "snap_err_m"),
                    "mean_raw_err_m": _mean(per, "raw_err_m"),
                    "mean_eta_hmm_s": _mean(per, "eta_hmm_err_s"),
                    "mean_eta_raw_s": _mean(per, "eta_raw_err_s"),
                    "mean_eta_delta_s": _mean(per, "eta_mae_delta_s"),
                    "frac_hmm_beats_snap": sum(1 for r in per if r["hmm_beats_snap"])
                    / n,
                    "frac_hmm_beats_raw": sum(1 for r in per if r["hmm_beats_raw"]) / n,
                    "mean_feet_ratio": _mean(per, "feet_ratio"),
                    "mean_walk_ratio": _mean(per, "walk_ratio"),
                }
            )
    results = {
        "config": {
            "sigmas": list(sigmas),
            "dropouts": list(dropouts),
            "route_seeds": seeds,
            "corrupt_seed": int(corrupt_seed),
            "gaps": int(gaps),
            "model_sigma_m": float(model_sigma),
            "candidate_radius_m": float(candidate_radius_m),
            "max_candidates": int(max_candidates),
            "beta": float(beta),
            "speed_mps": float(speed_mps),
            "observation_interval_s": float(observation_interval_s),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        },
        "cells": cells,
        "routes": route_rows,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


if __name__ == "__main__":
    res = run_noise_grid()
    for c in res["cells"]:
        print(
            f"sigma={c['sigma_m']:.0f} dropout={c['dropout_p']:.1f} "
            f"match={c['mean_match_rate']:.3f} lcs={c['mean_lcs_rate']:.3f} "
            f"hmm_err={c['mean_hmm_err_m']:.0f}m snap_err={c['mean_snap_err_m']:.0f}m "
            f"raw_err={c['mean_raw_err_m']:.0f}m eta_delta={c['mean_eta_delta_s']:.0f}s "
            f"beats_snap={c['frac_hmm_beats_snap']:.2f} beats_raw={c['frac_hmm_beats_raw']:.2f} "
            f"feet={c['mean_feet_ratio']:.2f}x walk={c['mean_walk_ratio']:.2f}x"
        )
    print(json.dumps({"wrote": str(Path("results/metrics.json"))}))
