"""API contract and startup-state tests on a tiny persisted graph."""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.geo import M_PER_DEG_LAT
from src.graph import save_graph


@pytest.fixture()
def client(tmp_path: Path):
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2],
            "lat": [0.0, 0.0, 0.0],
            "lon": [0.0, 0.001, 0.002],
        }
    )
    length = 0.001 * M_PER_DEG_LAT
    edges = pd.DataFrame(
        {
            "edge_id": [0, 1],
            "u": [0, 1],
            "v": [1, 2],
            "length_m": [length, length],
            "bearing": [90.0, 90.0],
        }
    )
    save_graph(nodes, edges, tmp_path)
    model_path = tmp_path / "params.json"
    model_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sigma_m": 5.0,
                "beta_m": 1.0,
                "candidate_radius_m": 20.0,
                "max_candidates": 8,
                "observation_interval_s": 0.0,
            }
        ),
        encoding="utf-8",
    )
    app = create_app(data_dir=tmp_path, model_path=model_path)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert Path(body.pop("model_source")).name == "params.json"
    assert body == {
        "status": "ok",
        "nodes": 3,
        "edges": 2,
        "sigma_m": 5.0,
        "beta_m": 1.0,
        "observation_interval_s": 0.0,
        "candidate_radius_m": 20.0,
        "max_candidates": 8,
    }


def test_match_returns_reconstructed_route_and_confidence(client: TestClient):
    response = client.post(
        "/match",
        json={
            "points": [
                {"lat": 0.0, "lon": 0.00025, "t": 0.0},
                {"lat": 0.0, "lon": 0.00175, "t": 5.0},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["edge_ids"] == [0, 1]
    assert body["route_distance_m"] == pytest.approx(1.5 * 0.001 * M_PER_DEG_LAT)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["input_points"] == body["sampled_points"] == 2
    assert body["matched_points"] == 2
    assert body["skipped_points"] == 0
    assert body["latency_ms"] >= 0.0


def test_match_rejects_nonincreasing_timestamps(client: TestClient):
    response = client.post(
        "/match",
        json={
            "points": [
                {"lat": 0.0, "lon": 0.00025, "t": 1.0},
                {"lat": 0.0, "lon": 0.00175, "t": 1.0},
            ]
        },
    )
    assert response.status_code == 422


def test_match_rejects_boolean_coordinates(client: TestClient):
    response = client.post(
        "/match",
        json={
            "points": [
                {"lat": True, "lon": 0.00025, "t": 0.0},
                {"lat": 0.0, "lon": 0.00175, "t": 5.0},
            ]
        },
    )
    assert response.status_code == 422


def test_match_returns_422_when_points_have_no_candidates(client: TestClient):
    response = client.post(
        "/match",
        json={
            "points": [
                {"lat": 1.0, "lon": 1.0, "t": 0.0},
                {"lat": 1.001, "lon": 1.001, "t": 5.0},
            ]
        },
    )
    assert response.status_code == 422


def test_match_rejects_too_many_sampled_points(client: TestClient):
    response = client.post(
        "/match",
        json={
            "points": [
                {"lat": 0.0, "lon": 0.0005, "t": float(index)} for index in range(1_001)
            ]
        },
    )
    assert response.status_code == 422
    assert "more than 1000 sampled points" in response.json()["detail"]


def test_explicit_missing_model_path_is_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="model parameters not found"):
        create_app(model_path=tmp_path / "missing.json")


def test_unknown_model_schema_is_rejected(tmp_path: Path):
    model_path = tmp_path / "params.json"
    model_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        create_app(model_path=model_path)
