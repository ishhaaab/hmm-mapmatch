"""Evaluation: match rate, distance error, ETA-MAE delta, slices.

Route match rate is defined over **edge ids** (ground truth from
synthesize.sample_route's `edge_ids` vs the Viterbi path's candidate edge
ids), so it stays unambiguous on parallel roads.
"""
import json
from pathlib import Path


def evaluate(matched, ground_truth) -> dict:
    """Metrics on edge-id ground truth.

    matched: list of edge ids from the Viterbi path; ground_truth: edge_ids
    list from synthesize.sample_route. Returns dict with:
      match_rate — |matched ∩ truth| / |truth| over edge ids;
          order-aware alternative: longest-common-subsequence fraction.
      distance_error_m — |matched route length - true length|.
      eta_mae_delta_s — ETA MAE using raw vs matched distance.
    Implemented in Phase 3.
    """
    raise NotImplementedError("Phase 3")


def run_noise_grid(sigmas=(15, 25, 40), dropouts=(0.0, 0.1),
                   out: Path = Path("results/metrics.json")) -> dict:
    """Run sigma x dropout grid, persist JSON. Implemented in Phase 3."""
    raise NotImplementedError("Phase 3")


if __name__ == "__main__":
    print(json.dumps({"status": "skeleton — implement Phase 3"}))