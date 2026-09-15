"""Phase-2 acceptance tests on real data.

BUILD_GUIDE (Phase 2 check): at sigma=15 m, route match rate > 85% and the
HMM beats both baselines (raw haversine polyline, nearest-segment snap) on
route distance error. Plus: the hard-assignment refit recovers the
generating sigma from a badly-wrong init.

All randomness is seeded (routes + corruption), so these are reproducible.
"""

import networkx as nx
import numpy as np
import pytest

from src.candidates import Candidate, CandidateGrid, _default_radius
from src.em import refit_sigma_beta
from src.evaluate import (
    DEFAULT_BETA,
    DEFAULT_CANDIDATE_RADIUS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MODEL_SIGMA,
)
from src.graph import as_routing_graph, load_graph
from src.mapmatch import (
    _transition_matrix,
    match_trace,
    nearest_snap,
    raw_haversine_length,
    reconstruct_route,
    route_distance_between,
    thin_trace,
)
from src.synthesize import corrupt, sample_route


@pytest.fixture(scope="module")
def setup():
    nodes, edges = load_graph()
    G = as_routing_graph(nodes, edges)
    edge_lengths = dict(zip(edges["edge_id"], edges["length_m"]))
    grid = CandidateGrid(nodes, edges, _default_radius(15.0))
    return {
        "nodes": nodes,
        "edges": edges,
        "G": G,
        "edge_lengths": edge_lengths,
        "grid": grid,
        "production_grid": CandidateGrid(
            nodes, edges, DEFAULT_CANDIDATE_RADIUS, DEFAULT_MAX_CANDIDATES
        ),
    }


def test_phase2_match_rate_above_85(setup):
    """BUILD_GUIDE check on the HMM core.

    Measured on dense GPS (gaps=0) to isolate the decoder. Mean over ten
    seeded routes.
    """
    rates = []
    for seed in range(10):
        route = sample_route(setup["G"], n=1, seed=seed)[0]
        fixes, _truth = corrupt(route, sigma_m=15.0, dropout_p=0.1, gaps=0, seed=3)
        matched = match_trace(
            fixes,
            setup["grid"],
            setup["G"],
            setup["edge_lengths"],
            sigma=15.0,
            beta=0.5,
        )
        truth_ids = set(route["edge_ids"])
        matched_ids = {c.edge_id for c in matched}
        rates.append(len(matched_ids & truth_ids) / len(truth_ids))

    assert len(rates) == 10
    assert sum(rates) / len(rates) > 0.85


def test_phase2_hmm_beats_both_baselines(setup):
    """Production preprocessing beats both baselines on mean distance error."""
    batch_rates = []
    hmm_errors = []
    snap_errors = []
    raw_errors = []
    for route_seed in (7, 8, 9, 10, 11):
        route = sample_route(setup["G"], n=1, seed=route_seed)[0]
        fixes, _truth = corrupt(route, sigma_m=15.0, dropout_p=0.1, gaps=1, seed=3)
        true_len = route["length_m"]

        sampled, _ = thin_trace(fixes)
        matched = match_trace(
            sampled,
            setup["production_grid"],
            setup["G"],
            setup["edge_lengths"],
            sigma=DEFAULT_MODEL_SIGMA,
            beta=DEFAULT_BETA,
        )
        matched_route = reconstruct_route(matched, setup["edge_lengths"], setup["G"])
        hmm_err = abs(matched_route.total_m - true_len)

        snap = nearest_snap(sampled, setup["production_grid"])
        snap_err = abs(
            reconstruct_route(snap, setup["edge_lengths"], setup["G"]).total_m
            - true_len
        )

        raw_err = abs(raw_haversine_length(sampled) - true_len)

        # The blind gap is bridged by network consistency; the reconstructed
        # route should remain recognisable.
        truth_ids = set(route["edge_ids"])
        matched_ids = set(matched_route.edge_ids)
        batch_rates.append(len(matched_ids & truth_ids) / len(truth_ids))
        hmm_errors.append(hmm_err)
        snap_errors.append(snap_err)
        raw_errors.append(raw_err)

    assert sum(batch_rates) / len(batch_rates) > 0.90
    assert sum(hmm_errors) / len(hmm_errors) < sum(snap_errors) / len(snap_errors)
    assert sum(hmm_errors) / len(hmm_errors) < sum(raw_errors) / len(raw_errors)


def test_same_edge_projection_jitter_does_not_add_reverse_distance(setup):
    edge_id = int(setup["edges"].iloc[0]["edge_id"])
    row = setup["edges"].iloc[0]
    forward = Candidate(edge_id, int(row.u), int(row.v), 0.0, 0.8)
    jittered_back = Candidate(edge_id, int(row.u), int(row.v), 0.0, 0.2)
    assert (
        route_distance_between(
            forward, jittered_back, setup["edge_lengths"], setup["G"]
        )
        == 0.0
    )


def test_phase2_reconstructed_length_is_honest(setup):
    """The production route reconstruction remains close under full noise."""
    route = sample_route(setup["G"], n=1, seed=7)[0]
    true_len = route["length_m"]
    for sigma, dropout, gaps in ((0.01, 0.0, 0), (15.0, 0.1, 1)):
        fixes, _truth = corrupt(
            route, sigma_m=sigma, dropout_p=dropout, gaps=gaps, seed=3
        )
        sampled, _ = thin_trace(fixes)
        matched = match_trace(
            sampled,
            setup["production_grid"],
            setup["G"],
            setup["edge_lengths"],
            sigma=DEFAULT_MODEL_SIGMA,
            beta=DEFAULT_BETA,
        )
        ratio = (
            reconstruct_route(matched, setup["edge_lengths"], setup["G"]).total_m
            / true_len
        )
        assert 0.85 < ratio < 1.25


def test_transition_cache_respects_each_gap_cutoff():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, length_m=1_500.0)
    graph.add_edge(1, 2, length_m=1_000.0)
    previous = [Candidate(10, 9, 0, 0.0, 1.0)]
    current = [Candidate(11, 2, 3, 0.0, 0.0)]
    lengths = {10: 1.0, 11: 1.0}
    cache = {}

    short_gap = _transition_matrix(previous, current, 0.0, lengths, graph, 1.0, cache)
    long_gap = _transition_matrix(
        previous, current, 1_000.0, lengths, graph, 1.0, cache
    )

    assert short_gap[0, 0] == -float("inf")
    assert np.isfinite(long_gap[0, 0])

    cache = {}
    long_gap = _transition_matrix(
        previous, current, 1_000.0, lengths, graph, 1.0, cache
    )
    short_gap = _transition_matrix(previous, current, 0.0, lengths, graph, 1.0, cache)

    assert np.isfinite(long_gap[0, 0])
    assert short_gap[0, 0] == -float("inf")


def test_reconstruction_selects_equal_parallel_connector_deterministically():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 2, key=9, edge_id=9, length_m=50.0)
    graph.add_edge(0, 2, key=5, edge_id=5, length_m=50.0)
    path = [
        Candidate(10, -1, 0, 0.0, 1.0),
        Candidate(20, 2, 3, 0.0, 0.0),
    ]

    route = reconstruct_route(path, {10: 100.0, 20: 100.0}, graph)

    assert route.edge_ids == [10, 5, 20]
    assert route.total_m == 50.0


def test_refit_recovers_generating_sigma(setup):
    route = sample_route(setup["G"], n=1, seed=31)[0]
    fixes, _truth = corrupt(route, sigma_m=15.0, dropout_p=0.0, gaps=0, seed=5)

    fitted = refit_sigma_beta(
        [{"fixes": fixes, "truth": route}],
        setup["grid"],
        setup["G"],
        setup["edge_lengths"],
        init_sigma=5.0,
        init_beta=0.05,
        iters=2,
    )
    assert abs(fitted["sigma"] - 15.0) < 5.0
    assert fitted["beta"] > 0.0
