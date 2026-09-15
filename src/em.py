"""Learn the emission sigma and transition beta from matched traces.

This is a hard-assignment refit, not textbook EM: the E-step is a Viterbi
(hard) decode rather than a soft forward/backward posterior, and each M-step
is a closed-form refit. Iterate `iters` times to converge. (Newson & Krumm
normally *measure* sigma_z from the GPS device spec; here we fit it from
data instead, which is the point of this module.)

M-step estimators are exact under the model:
  - sigma <- RMSE of perpendicular fix->segment residuals (the MLE for a
    Gaussian emission; the residuals are |N(0, sigma)|).
  - beta  <- mean(|route_dist - great_circle_dist|) (the MLE for the
    exponential transition's mean).

Known bias: the residuals come from the hard-assigned matches, and a small
fraction of those land on the wrong (parallel) street, so sigma is pulled
below the true generating value. The acceptance test only bounds the error;
it does not validate the estimator on clean matches.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import networkx as nx
import numpy as np

from src.candidates import CandidateGrid
from src.geo import haversine_m
from src.graph import as_routing_graph, load_graph
from src.mapmatch import (
    DEFAULT_OBSERVATION_INTERVAL_S,
    decode_trace,
    route_distance_between,
    thin_trace,
)
from src.synthesize import corrupt, sample_route

PARAM_FLOOR_M = 1e-3
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "params.json"
CALIBRATION_ROUTE_SEEDS = (20, 21, 22, 23, 24)
DEPLOYMENT_CANDIDATE_RADIUS_M = 120.0
DEPLOYMENT_MAX_CANDIDATES = 40


class FitParams(TypedDict):
    """Refit result: the two HMM noise parameters (sigma in metres, beta in
    metres — the exponential transition's mean)."""

    sigma: float
    beta: float


FixedTrace = Mapping[str, Mapping[str, np.ndarray]]  # {'fixes': {...}} wrapper


def refit_sigma_beta(
    traces: Sequence[FixedTrace],
    grid: CandidateGrid,
    graph: nx.MultiDiGraph,
    edge_lengths: dict[int, float],
    init_sigma: float = 20.0,
    init_beta: float = 0.5,
    iters: int = 1,
) -> FitParams:
    """Refit sigma/beta from matched traces via hard-assignment iterations.

    traces: sequence of dicts with 'fixes' (synthesize.corrupt output:
        lat/lon/t arrays); ground-truth is not needed by the fit.
    grid: prebuilt CandidateGrid; graph: routing graph; edge_lengths:
        edge_id -> metres.
    Each iteration, under the current (sigma, beta): decode every trace with
    the full pipeline (E-step) and collect the residuals and gaps from the
    hard assignments, then apply the closed-form M-step above. `iters`=0
    returns the init values unchanged.
    """
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    sigma, beta = float(init_sigma), float(init_beta)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"init_sigma must be finite and > 0, got {init_sigma}")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"init_beta must be finite and > 0, got {init_beta}")
    for _ in range(iters):
        residuals: list[float] = []
        gaps: list[float] = []
        for trace in traces:
            fixes = trace["fixes"]
            matched, keep = decode_trace(fixes, grid, graph, edge_lengths, sigma, beta)
            residuals.extend(c.dist_m for c in matched)
            for t in range(1, len(matched)):
                cand_a, cand_b = matched[t - 1], matched[t]
                rd = route_distance_between(cand_a, cand_b, edge_lengths, graph)
                i, j = keep[t - 1], keep[t]
                ed = haversine_m(
                    fixes["lat"][i], fixes["lon"][i], fixes["lat"][j], fixes["lon"][j]
                )
                gap = abs(rd - ed)
                if np.isfinite(gap):
                    gaps.append(gap)
        if not residuals:
            raise ValueError(
                "no trace fixes could be matched; parameters cannot be refit"
            )
        sigma = max(
            PARAM_FLOOR_M,
            float(np.sqrt(np.mean(np.asarray(residuals) ** 2))),
        )
        if gaps:
            beta = max(PARAM_FLOOR_M, float(np.mean(gaps)))
    return {"sigma": sigma, "beta": beta}


def calibrate_synthetic_model(
    out: Path = MODEL_PATH,
    route_seeds: Sequence[int] = CALIBRATION_ROUTE_SEEDS,
    corrupt_seed: int = 101,
    generating_sigma_m: float = 15.0,
    dropout_p: float = 0.1,
    gaps: int = 1,
    observation_interval_s: float = DEFAULT_OBSERVATION_INTERVAL_S,
    candidate_radius_m: float = DEPLOYMENT_CANDIDATE_RADIUS_M,
    max_candidates: int = DEPLOYMENT_MAX_CANDIDATES,
    iters: int = 3,
) -> dict:
    """Fit and persist deterministic startup parameters on synthetic routes.

    Calibration seeds are disjoint from the evaluation seeds in evaluate.py.
    This artifact is a reproducible deployment default, not evidence that the
    parameters transfer to real GPS data.
    """
    if not route_seeds:
        raise ValueError("route_seeds must not be empty")
    nodes, edges = load_graph()
    graph = as_routing_graph(nodes, edges)
    edge_lengths = {
        int(edge_id): float(length_m)
        for edge_id, length_m in zip(edges["edge_id"], edges["length_m"])
    }
    traces = []
    noise_seeds = []
    for index, route_seed in enumerate(route_seeds):
        route = sample_route(graph, n=1, seed=int(route_seed))[0]
        noise_seed = int(
            np.random.SeedSequence(
                [int(corrupt_seed), int(route_seed), index]
            ).generate_state(1)[0]
        )
        fixes, _ = corrupt(
            route,
            sigma_m=generating_sigma_m,
            dropout_p=dropout_p,
            gaps=gaps,
            seed=noise_seed,
        )
        sampled, _ = thin_trace(fixes, observation_interval_s)
        traces.append({"fixes": sampled})
        noise_seeds.append(noise_seed)

    fitted = refit_sigma_beta(
        traces,
        CandidateGrid(nodes, edges, candidate_radius_m, max_candidates),
        graph,
        edge_lengths,
        init_sigma=20.0,
        init_beta=5.0,
        iters=iters,
    )
    artifact = {
        "schema_version": 1,
        "sigma_m": fitted["sigma"],
        "beta_m": fitted["beta"],
        "observation_interval_s": float(observation_interval_s),
        "candidate_radius_m": float(candidate_radius_m),
        "max_candidates": int(max_candidates),
        "training": {
            "kind": "synthetic_hard_viterbi",
            "route_seeds": [int(seed) for seed in route_seeds],
            "noise_seeds": noise_seeds,
            "generating_sigma_m": float(generating_sigma_m),
            "dropout_p": float(dropout_p),
            "gaps": int(gaps),
            "iterations": int(iters),
        },
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


if __name__ == "__main__":
    print(json.dumps(calibrate_synthetic_model(), indent=2))
