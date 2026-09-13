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

from typing import Dict, List, Mapping, Sequence, TypedDict

import networkx as nx
import numpy as np

from src.candidates import CandidateGrid
from src.geo import haversine_m
from src.mapmatch import decode_trace, route_distance_between


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
    edge_lengths: Dict[int, float],
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
    sigma, beta = float(init_sigma), float(init_beta)
    for _ in range(iters):
        residuals: List[float] = []
        gaps: List[float] = []
        for trace in traces:
            fixes = trace["fixes"]
            matched, keep = decode_trace(fixes, grid, graph, edge_lengths,
                                         sigma, beta)
            if len(matched) < 2:
                continue
            residuals.extend(c.dist_m for c in matched)
            for t in range(1, len(matched)):
                cand_a, cand_b = matched[t - 1], matched[t]
                rd = route_distance_between(cand_a, cand_b, edge_lengths, graph)
                i, j = keep[t - 1], keep[t]
                ed = haversine_m(fixes["lat"][i], fixes["lon"][i],
                                 fixes["lat"][j], fixes["lon"][j])
                gaps.append(abs(rd - ed))
        if residuals:
            sigma = float(np.sqrt(np.mean(np.asarray(residuals) ** 2)))
        if gaps:
            beta = float(np.mean(gaps))
    return {"sigma": sigma, "beta": beta}