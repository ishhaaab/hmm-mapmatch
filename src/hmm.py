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

import math
from collections.abc import Sequence

import numpy as np


def emission_logprob(dist_m: float, sigma: float) -> float:
    """Log of the Gaussian emission on perpendicular distance.

    dist_m: perpendicular distance fix->segment in metres; sigma: GPS noise
    std in metres (must be > 0). Exact N&K emission in log space.
    """
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not math.isfinite(dist_m) or dist_m < 0.0:
        raise ValueError(f"dist_m must be finite and >= 0, got {dist_m}")
    return float(-0.5 * ((dist_m / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2)))


def transition_logprob(route_dist: float, euclid_dist: float, beta: float) -> float:
    """Log of the N&K transition penalty.

    d = |route_dist - euclid_dist| in metres: route_dist is the shortest
    driving distance between the candidate projection points, euclid_dist the
    great-circle distance between the two GPS fixes. beta (metres) must be
    > 0. Includes the constant -log(beta) normaliser.
    """
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if math.isnan(route_dist) or route_dist < 0.0:
        raise ValueError(f"route_dist must be >= 0, got {route_dist}")
    if not math.isfinite(euclid_dist) or euclid_dist < 0.0:
        raise ValueError(f"euclid_dist must be finite and >= 0, got {euclid_dist}")
    d = abs(route_dist - euclid_dist)
    return float(-np.log(beta) - d / beta)


def viterbi(
    emissions: Sequence[np.ndarray],
    transitions: Sequence[np.ndarray],
    starts: np.ndarray,
) -> list[int]:
    """Most-likely state sequence over a variable-width trellis (log space).

    emissions[t]: (S_t,) log emission probability per candidate of fix t.
    transitions[t]: (S_{t-1}, S_t) log transition probability from state
        s_{t-1} to s_t, e.g. transition_logprob at the two candidate
        projection points.
    starts: (S_0,) log probability of each initial state (flat prior is fine).

    Returns the index of the winning state per timestep. Runs entirely in
    log space (raw probabilities underflow); O(T * S_max^2).
    """
    emissions = [np.asarray(values, dtype=float) for values in emissions]
    transitions = [np.asarray(values, dtype=float) for values in transitions]
    T = len(emissions)
    if T == 0:
        if len(transitions) != 0 or np.asarray(starts).size != 0:
            raise ValueError("an empty trellis requires empty transitions and starts")
        return []
    if len(transitions) != T - 1:
        raise ValueError(
            f"expected {T - 1} transition matrices, got {len(transitions)}"
        )
    for t in range(T):
        emission = emissions[t]
        if emission.ndim != 1 or len(emission) == 0:
            raise ValueError(
                f"emissions[{t}] must be a non-empty one-dimensional state set"
            )
        if np.any(np.isnan(emission)) or np.any(np.isposinf(emission)):
            raise ValueError(f"emissions[{t}] contains NaN/+inf")
    starts = np.asarray(starts, dtype=float)
    if starts.ndim != 1 or starts.shape != (len(emissions[0]),):
        raise ValueError(f"starts shape {starts.shape} != ({len(emissions[0])},)")
    if np.any(np.isnan(starts)) or np.any(np.isposinf(starts)):
        raise ValueError("starts may contain finite values or -inf, not NaN/+inf")
    for t in range(1, T):
        exp_shape = (len(emissions[t - 1]), len(emissions[t]))
        if transitions[t - 1].shape != exp_shape:
            raise ValueError(
                f"transition[{t - 1}] shape {transitions[t - 1].shape} != {exp_shape}"
            )
        if np.any(np.isnan(transitions[t - 1])) or np.any(
            np.isposinf(transitions[t - 1])
        ):
            raise ValueError(f"transitions[{t - 1}] contains NaN/+inf")

    # delta[t][s] = log prob of the best path ending in state s at time t;
    # psi[t][s] = argmax predecessor of state s at time t (backtrack pointers).
    delta = [None] * T
    psi = [None] * T
    delta[0] = starts + emissions[0]
    if not np.any(np.isfinite(delta[0])):
        raise ValueError("no viable state at step 0")

    for t in range(1, T):
        score = delta[t - 1][:, np.newaxis] + transitions[t - 1]  # (S_{t-1}, S_t)
        psi[t] = np.argmax(score, axis=0)
        delta[t] = np.max(score, axis=0) + emissions[t]
        if not np.any(np.isfinite(delta[t])):
            raise ValueError(f"no viable state sequence reaches step {t}")

    path = [0] * T
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = int(psi[t + 1][path[t + 1]])
    return path


def _logsumexp(values: np.ndarray, axis=None) -> np.ndarray:
    """Stable log(sum(exp(values))) with all-impossible slices preserved."""
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    with np.errstate(invalid="ignore", divide="ignore"):
        shifted = np.where(finite, values - maximum, -np.inf)
        summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
        result = np.where(finite, maximum + np.log(summed), -np.inf)
    if axis is None:
        return np.asarray(result).reshape(())
    return np.squeeze(result, axis=axis)


def posterior_marginals(
    emissions: Sequence[np.ndarray],
    transitions: Sequence[np.ndarray],
    starts: np.ndarray,
) -> list[np.ndarray]:
    """Per-step state posteriors for the same variable-width HMM trellis.

    This is a log-space forward/backward pass. The returned vectors each sum
    to one and are conditional on the candidate sets supplied by the caller.
    """
    path = viterbi(emissions, transitions, starts)
    if not path:
        return []

    emissions = [np.asarray(values, dtype=float) for values in emissions]
    transitions = [np.asarray(values, dtype=float) for values in transitions]
    T = len(emissions)
    alpha: list[np.ndarray] = [np.empty(0)] * T
    alpha[0] = np.asarray(starts, dtype=float) + emissions[0]
    for t in range(1, T):
        scores = alpha[t - 1][:, np.newaxis] + transitions[t - 1]
        alpha[t] = emissions[t] + _logsumexp(scores, axis=0)

    backward: list[np.ndarray] = [np.empty(0)] * T
    backward[-1] = np.zeros(len(emissions[-1]), dtype=float)
    for t in range(T - 2, -1, -1):
        scores = (
            transitions[t]
            + emissions[t + 1][np.newaxis, :]
            + backward[t + 1][np.newaxis, :]
        )
        backward[t] = _logsumexp(scores, axis=1)

    log_evidence = float(_logsumexp(alpha[-1]))
    return [np.exp(alpha[t] + backward[t] - log_evidence) for t in range(T)]
