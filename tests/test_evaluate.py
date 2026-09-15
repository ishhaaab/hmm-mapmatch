"""Unit tests for evaluation definitions and input validation."""

import pytest

from src.evaluate import evaluate, route_f1, route_match_rate, route_precision


def test_edge_scores_penalize_missing_and_extra_edges():
    matched = [1, 2, 9]
    truth = [1, 2, 3]
    assert route_match_rate(matched, truth) == pytest.approx(2.0 / 3.0)
    assert route_precision(matched, truth) == pytest.approx(2.0 / 3.0)
    assert route_f1(matched, truth) == pytest.approx(2.0 / 3.0)


def test_evaluate_computes_distance_equivalent_eta_error():
    metrics = evaluate(
        [1, 2],
        [1, 2],
        matched_length_m=120.0,
        true_length_m=100.0,
        raw_length_m=140.0,
        speed_mps=10.0,
    )
    assert metrics["distance_error_m"] == 20.0
    assert metrics["eta_hmm_err_s"] == 2.0
    assert metrics["eta_raw_err_s"] == 4.0
    assert metrics["eta_mae_delta_s"] == 2.0


def test_evaluate_rejects_invalid_speed_or_lengths():
    with pytest.raises(ValueError, match="speed_mps"):
        evaluate([], [], speed_mps=0.0)
    with pytest.raises(ValueError, match="route lengths"):
        evaluate([], [], matched_length_m=-1.0, true_length_m=1.0)
