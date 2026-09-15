"""Unit tests for evaluation definitions and input validation."""

import json

import pytest

import src.evaluate as evaluation
from src.evaluate import (
    DEFAULT_N_ROUTES,
    DEFAULT_ROUTE_SEEDS,
    bootstrap_mean_intervals,
    evaluate,
    route_f1,
    route_match_rate,
    route_precision,
)


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


def test_default_benchmark_has_100_routes_disjoint_from_calibration():
    assert DEFAULT_N_ROUTES == 100
    assert len(DEFAULT_ROUTE_SEEDS) == len(set(DEFAULT_ROUTE_SEEDS)) == 100
    assert set(DEFAULT_ROUTE_SEEDS).isdisjoint({20, 21, 22, 23, 24})


def test_bootstrap_mean_intervals_are_deterministic_and_paired():
    rows = [
        {"score": 0.0, "scaled_score": 0.0},
        {"score": 1.0, "scaled_score": 10.0},
        {"score": 2.0, "scaled_score": 20.0},
        {"score": 3.0, "scaled_score": 30.0},
    ]
    metrics = {"mean_score": "score", "mean_scaled_score": "scaled_score"}

    first = bootstrap_mean_intervals(rows, metrics, n_resamples=2_000, seed=9)
    second = bootstrap_mean_intervals(rows, metrics, n_resamples=2_000, seed=9)

    assert first == second
    assert first["mean_score"]["lower"] < 1.5 < first["mean_score"]["upper"]
    assert first["mean_scaled_score"]["lower"] == pytest.approx(
        10.0 * first["mean_score"]["lower"]
    )
    assert first["mean_scaled_score"]["upper"] == pytest.approx(
        10.0 * first["mean_score"]["upper"]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"n_resamples": 0}, "n_resamples"),
    ],
)
def test_bootstrap_mean_intervals_reject_invalid_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_mean_intervals([{"score": 1.0}], {"mean_score": "score"}, **kwargs)


def test_noise_grid_persists_bootstrap_metadata_and_cell_intervals(
    monkeypatch, tmp_path
):
    fixes = {"lat": [0.0, 0.0], "lon": [0.0, 0.0], "t": [0.0, 5.0]}
    route_metrics = {field: 1.0 for field in evaluation.CELL_AGGREGATES.values()}
    route_metrics["hmm_beats_snap"] = True
    route_metrics["hmm_beats_raw"] = True

    monkeypatch.setattr(
        evaluation,
        "load_graph",
        lambda: ([object()], {"edge_id": [1], "length_m": [100.0]}),
    )
    monkeypatch.setattr(evaluation, "as_routing_graph", lambda nodes, edges: object())
    monkeypatch.setattr(evaluation, "CandidateGrid", lambda *args: object())
    monkeypatch.setattr(
        evaluation,
        "sample_route",
        lambda graph, n, seed: [{"edge_ids": [1], "length_m": 100.0}],
    )
    monkeypatch.setattr(evaluation, "corrupt", lambda route, **kwargs: (fixes, fixes))
    monkeypatch.setattr(
        evaluation, "thin_trace", lambda trace, interval: (trace, [0, 1])
    )
    monkeypatch.setattr(evaluation, "match_trace", lambda *args, **kwargs: [])
    monkeypatch.setattr(evaluation, "nearest_snap", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        evaluation,
        "evaluate_trace",
        lambda *args, **kwargs: route_metrics.copy(),
    )

    output = tmp_path / "metrics.json"
    result = evaluation.run_noise_grid(
        sigmas=(15.0,),
        dropouts=(0.0,),
        out=output,
        n_routes=2,
        route_seeds=(7, 8),
        bootstrap_resamples=100,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["config"]["bootstrap"] == {
        "method": "percentile_bootstrap",
        "sampling_unit": "route",
        "confidence_level": 0.95,
        "n_resamples": 100,
        "seed": 17,
        "paired_across_cells": True,
    }
    cell = result["cells"][0]
    assert cell["n_routes"] == 2
    assert set(cell["confidence_intervals"]) == set(evaluation.CELL_AGGREGATES)
    assert cell["confidence_intervals"]["mean_edge_f1"] == {
        "lower": 1.0,
        "upper": 1.0,
    }
