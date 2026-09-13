"""HMM emission, transition, Viterbi — NumPy only, log-space.

Emission (Newson & Krumm 2009): Gaussian on perpendicular distance
fix->segment (sigma):
    p(z|r) = (1/sqrt(2*pi*sigma^2)) * exp(-(dist/sigma)^2 / 2)

Transition (Newson & Krumm 2009): exponential penalty on
d = |route_dist - great_circle_dist| (beta):
    p(d) = (1/beta) * exp(-d/beta)
where route_dist is the shortest driving distance between the two candidate
projection points and great_circle_dist the straight-line distance between
the two fixes. The 1/beta normalisers are constant across paths and cancel in
a log-space Viterbi argmax; they are kept here for exactness and testability.

Trellis shape: in map matching every fix has its own candidate set, so the
textbook T x S emission array and single S x S transition matrix do not
apply. viterbi() therefore takes per-step arrays whose widths vary — see the
signature.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def emission_logprob(dist_m: float, sigma: float) -> float:
    """Log of the Gaussian emission on perpendicular distance.

    dist_m: perpendicular distance fix->segment in metres; sigma: GPS noise
    std in metres (must be > 0). Exact N&K emission in log space.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return float(-0.5 * ((dist_m / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2)))


def transition_logprob(route_dist: float, euclid_dist: float,
                       beta: float) -> float:
    """Log of the N&K transition penalty.

    d = |route_dist - euclid_dist| in metres: route_dist is the shortest
    driving distance between the candidate projection points, euclid_dist the
    great-circle distance between the two GPS fixes. beta (metres) must be
    > 0. Includes the constant -log(beta) normaliser.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    d = abs(route_dist - euclid_dist)
    return float(-np.log(beta) - d / beta)


def viterbi(
    emissions: Sequence[np.ndarray],
    transitions: Sequence[np.ndarray],
    starts: np.ndarray,
) -> List[int]:
    """Most-likely state sequence over a variable-width trellis (log space).

    emissions[t]: (S_t,) log emission probability per candidate of fix t.
    transitions[t]: (S_{t-1}, S_t) log transition probability from state
        s_{t-1} to s_t, e.g. transition_logprob at the two candidate
        projection points.
    starts: (S_0,) log probability of each initial state (flat prior is fine).

    Returns the index of the winning state per timestep. Runs entirely in
    log space (raw probabilities underflow); O(T * S_max^2).
    """
    T = len(emissions)
    if T == 0:
        return []
    for t in range(T):
        if len(emissions[t]) == 0:
            raise ValueError(f"empty state set at step {t}: no candidates")
    for t in range(1, T):
        exp_shape = (len(emissions[t - 1]), len(emissions[t]))
        if transitions[t - 1].shape != exp_shape:
            raise ValueError(
                f"transition[{t - 1}] shape {transitions[t - 1].shape} != {exp_shape}"
            )

    # delta[t][s] = log prob of the best path ending in state s at time t;
    # psi[t][s] = argmax predecessor of state s at time t (backtrack pointers).
    delta = [None] * T
    psi = [None] * T
    delta[0] = starts + emissions[0]

    for t in range(1, T):
        score = delta[t - 1][:, np.newaxis] + transitions[t - 1]  # (S_{t-1}, S_t)
        psi[t] = np.argmax(score, axis=0)
        delta[t] = np.max(score, axis=0) + emissions[t]

    path = [0] * T
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = int(psi[t + 1][path[t + 1]])
    return path