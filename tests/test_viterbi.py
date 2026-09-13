"""Tests for the HMM core: emission/transition math + Viterbi decoding.

- emission/transition: exact closed-form N&K formula checks.
- viterbi: hand-computed trellis whose optimal path is known
  independently (worked out on paper from the raw probabilities), plus the
  variable-width-trellis contract used by map matching.
"""
import math

import numpy as np
import pytest

from src.hmm import emission_logprob, transition_logprob, viterbi


# ---------------------------------------------------------------- emission --

def test_emission_matches_gaussian_formula():
    for sigma in (15.0, 25.0, 40.0):
        lp = emission_logprob(0.0, sigma)
        assert lp == pytest.approx(-0.5 * math.log(2 * math.pi * sigma**2))


def test_emission_decreases_with_distance():
    assert emission_logprob(10.0, 15.0) < emission_logprob(0.0, 15.0)
    assert emission_logprob(30.0, 15.0) < emission_logprob(10.0, 15.0)


def test_emission_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        emission_logprob(1.0, 0.0)


# -------------------------------------------------------------- transition --

def test_transition_matches_paper_formula():
    beta = 0.5
    assert transition_logprob(0.0, 0.0, beta) == pytest.approx(-math.log(beta))


def test_transition_decreases_with_gap_and_is_symmetric():
    b = 0.7
    small = transition_logprob(100.0, 90.0, b)
    large = transition_logprob(200.0, 90.0, b)
    assert small == pytest.approx(transition_logprob(90.0, 100.0, b))
    assert large < small


def test_transition_rejects_nonpositive_beta():
    with pytest.raises(ValueError):
        transition_logprob(1.0, 1.0, 0.0)


# ---------------------------------------------------------------- viterbi ---

def _constant_two_state_trellis():
    """T=3, S=2 with transitions that favour state switching.

    Emission:  fix0 prefers state0, fix1 prefers state1, fix2 prefers state0.
    Optimal path [0,1,0] has probability 0.5*0.8 * 0.7*0.9 * 0.7*0.7
    = 0.12348 — computed by enumerating all 8 paths by hand.
    """
    emissions = [
        np.array([math.log(0.8), math.log(0.1)]),
        np.array([math.log(0.1), math.log(0.9)]),
        np.array([math.log(0.7), math.log(0.2)]),
    ]
    switch = np.array(
        [
            [math.log(0.3), math.log(0.7)],  # from state 0: stay/switch
            [math.log(0.7), math.log(0.3)],  # from state 1: switch/stay
        ]
    )
    transitions = [switch, switch]
    starts = np.array([math.log(0.5), math.log(0.5)])
    return emissions, transitions, starts


def test_viterbi_recovers_known_optimal_path():
    emissions, transitions, starts = _constant_two_state_trellis()
    assert viterbi(emissions, transitions, starts) == [0, 1, 0]


def test_viterbi_returns_most_likely_sequence():
    emissions, transitions, starts = _constant_two_state_trellis()
    path = viterbi(emissions, transitions, starts)
    # Recompute the total log-probability of the returned path directly.
    lp = starts[path[0]] + emissions[0][path[0]]
    for t in range(1, len(emissions)):
        lp += transitions[t - 1][path[t - 1], path[t]] + emissions[t][path[t]]
    assert lp == pytest.approx(math.log(0.12348), rel=1e-9)
    assert lp == pytest.approx(max(
        starts[i] + emissions[0][i]
        + transitions[0][i, j] + emissions[1][j]
        + transitions[1][j, k] + emissions[2][k]
        for i in range(2) for j in range(2) for k in range(2)
    ))


def test_viterbi_single_timestep():
    emissions = [np.array([math.log(0.2), math.log(0.8)])]
    starts = np.array([math.log(0.5), math.log(0.5)])
    assert viterbi(emissions, [], starts) == [1]


def test_viterbi_variable_width_trellis():
    # Widths may differ per step: S_0=2, S_1=3. All transitions are equal,
    # so the path is driven purely by the emissions and initial prior.
    emissions = [np.array([-1.0, -2.0]), np.array([-3.0, -4.0, -5.0])]
    transitions = [np.full((2, 3), -1.0)]
    starts = np.array([0.0, 0.0])
    path = viterbi(emissions, transitions, starts)
    assert path == [0, 0]
    assert len(path) == 2


def test_viterbi_impossible_transition_is_avoided():
    # From state 0 you cannot reach state 1 (-inf); the decoder must use
    # state 0 at the first step even though its emission is worse.
    emissions = [np.array([-1.0, -2.0]), np.array([-2.0, -1.0])]
    transitions = [np.array([[-1.0, -np.inf], [-1.0, -1.0]])]
    starts = np.array([0.0, 0.0])
    assert viterbi(emissions, transitions, starts) == [0, 0]