"""Phase 1 smoke tests: graph files, sampler, noise model.

Covers the DONE criteria: graph parquet exists with correct columns, route
lengths are positive, corrupt() preserves ground truth, and outputs are
reproducible for the same seed. No Viterbi/EM coverage here (Phase 2+).
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.graph import as_routing_graph, load_graph
from src.synthesize import corrupt, sample_route

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_graph_files_exist_with_correct_columns():
    assert (PROCESSED / "nodes.parquet").exists()
    assert (PROCESSED / "edges.parquet").exists()
    nodes = pq.read_table(PROCESSED / "nodes.parquet")
    edges = pq.read_table(PROCESSED / "edges.parquet")
    assert nodes.schema.names == ["node_id", "lat", "lon"]
    assert edges.schema.names == ["edge_id", "u", "v", "length_m", "bearing"]
    assert nodes.num_rows > 0
    assert edges.num_rows > 0


def test_sample_route_positive_length():
    nodes, edges = load_graph()
    G = as_routing_graph(nodes, edges)
    routes = sample_route(G, n=2, seed=7)
    assert len(routes) == 2
    for route in routes:
        assert 2_000.0 <= route["length_m"] <= 5_000.0
        assert len(route["segments"]) > 0
        assert len(route["edge_ids"]) == len(route["segments"])
        assert all(isinstance(e, int) for e in route["edge_ids"])
        assert len(route["lat"]) == len(route["lon"]) == len(route["cumdist_m"])


def test_corrupt_preserves_ground_truth():
    nodes, edges = load_graph()
    G = as_routing_graph(nodes, edges)
    (route,) = sample_route(G, n=1, seed=7)
    clean_lat = route["lat"].copy()
    clean_lon = route["lon"].copy()
    fixes, truth = corrupt(route, sigma_m=25.0, dropout_p=0.1, gaps=1, seed=3)
    # Ground-truth inputs untouched, truth complete at 1 Hz, fixes a subset.
    np.testing.assert_array_equal(route["lat"], clean_lat)
    np.testing.assert_array_equal(route["lon"], clean_lon)
    assert len(truth["t"]) > len(fixes["t"]) > 0
    assert set(fixes.keys()) == {"lat", "lon", "t"}
    assert set(truth.keys()) == {"lat", "lon", "t"}
    assert np.all(np.diff(truth["t"]) > 0)  # strictly increasing timestamps


def test_outputs_reproducible_for_same_seed():
    nodes, edges = load_graph()
    G = as_routing_graph(nodes, edges)
    r1 = sample_route(G, n=1, seed=42)[0]
    r2 = sample_route(G, n=1, seed=42)[0]
    assert r1["nodes"] == r2["nodes"]
    f1, t1 = corrupt(r1, sigma_m=15.0, dropout_p=0.1, gaps=1, seed=11)
    f2, _ = corrupt(r2, sigma_m=15.0, dropout_p=0.1, gaps=1, seed=11)
    np.testing.assert_array_equal(f1["lat"], f2["lat"])
    np.testing.assert_array_equal(f1["lon"], f2["lon"])
    np.testing.assert_array_equal(f1["t"], f2["t"])
    np.testing.assert_array_equal(t1["lat"], corrupt(r1, seed=99)[1]["lat"])
