"""Tests for CandidateGrid — spatial index + per-fix candidate search.

Uses a tiny hand-built ~111 m square graph (nodes/e the table schemas from
graph.py) so expected distances and projection fractions are exact.
"""
import pandas as pd
import pytest

from src.candidates import CandidateGrid, _default_radius
from src.geo import M_PER_DEG_LAT


def _square_graph():
    """~111 m square: e0 bottom (0->1), e1 top (2->3), e2 left (0->2),
    e3 right (1->3). Node 0 at origin, lat rows climb north, lon east."""
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "lat": [0.0, 0.0, 0.001, 0.001],
            "lon": [0.0, 0.001, 0.0, 0.001],
        }
    )
    side_m = 0.001 * M_PER_DEG_LAT
    edges = pd.DataFrame(
        {
            "edge_id": [0, 1, 2, 3],
            "u": [0, 2, 0, 1],
            "v": [1, 3, 2, 3],
            "length_m": [side_m, side_m, side_m, side_m],
            "bearing": [90.0, 90.0, 0.0, 0.0],
        }
    )
    return nodes, edges


def test_grid_finds_nearest_edge_only():
    nodes, edges = _square_graph()
    grid = CandidateGrid(nodes, edges, radius_m=40.0)
    # 11.13 m north of the bottom edge's midpoint -> only edge 0 qualifies.
    cands = grid.query(0.0001, 0.0005)
    assert len(cands) == 1
    c = cands[0]
    assert c.edge_id == 0
    assert c.frac == pytest.approx(0.5, abs=1e-9)
    assert c.dist_m == pytest.approx(0.0001 * M_PER_DEG_LAT, rel=1e-6)


def test_grid_returns_all_edges_within_radius():
    nodes, edges = _square_graph()
    grid = CandidateGrid(nodes, edges, radius_m=100.0)
    # Square centre is ~55.7 m from every side -> all four edges qualify.
    cands = grid.query(0.0005, 0.0005)
    assert sorted(c.edge_id for c in cands) == [0, 1, 2, 3]
    for c in cands:
        assert c.frac == pytest.approx(0.5, abs=1e-9)


def test_grid_clamps_projection_past_endpoint():
    nodes, edges = _square_graph()
    grid = CandidateGrid(nodes, edges, radius_m=150.0)
    # Query east of the square: only the two edges facing it qualify, and
    # both projections clamp onto an endpoint (e0's far end frac=1.0; e3's
    # start, i.e. shared node 1, frac=0.0). e1/e2 are >150 m away.
    cands = grid.query(0.0, 0.002)
    by_id = {c.edge_id: c for c in cands}
    assert set(by_id) == {0, 3}
    assert by_id[0].frac == pytest.approx(1.0)
    assert by_id[3].frac == pytest.approx(0.0)
    expected = 0.001 * M_PER_DEG_LAT
    assert by_id[0].dist_m == pytest.approx(expected, rel=1e-6)
    assert by_id[3].dist_m == pytest.approx(expected, rel=1e-6)


def test_grid_empty_when_nothing_within_radius():
    nodes, edges = _square_graph()
    grid = CandidateGrid(nodes, edges, radius_m=10.0)
    assert grid.query(0.0005, 0.0005) == []  # ~55.7 m from every side


def test_candidates_on_edge_have_zero_distance():
    nodes, edges = _square_graph()
    grid = CandidateGrid(nodes, edges, radius_m=40.0)
    c = grid.query(0.0, 0.0005)[0]
    assert c.dist_m == pytest.approx(0.0, abs=1e-6)
    assert c.frac == pytest.approx(0.5, abs=1e-9)


def test_default_radius_scales_with_sigma():
    assert _default_radius(15.0) == 50.0
    assert _default_radius(25.0) == 75.0
    assert _default_radius(40.0) == 120.0
    assert _default_radius(10.0) == 50.0  # floor keeps a sane minimum