"""Candidate road segments per GPS fix.

For each fix, find the road segments that could plausibly have produced it
and project the fix onto each one.

Design contract (used by hmm.viterbi via the matched pipeline):
- The road network is modelled as straight segments between the edge
  endpoint nodes (nodes table) — see graph.py module docs.
- The candidate radius is derived from the sensor noise sigma (default
  max(3*sigma, 50) m) so the state space is never truncated at high noise.
  A fixed 75 m radius would cut off the top of the sigma=40 m grid;
  Newson & Krumm use 200 m.
- Every candidate carries the clamped perpendicular projection: `dist_m`
  feeds the emission, `frac` lets the transition assemble the route distance
  (partial edge distance + shortest network path + partial edge distance).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

import pandas as pd

from src.geo import M_PER_DEG_LAT, project_to_segment


class Candidate(NamedTuple):
    """One plausible road segment for one GPS fix.

    edge_id: id in the edges table; u, v: endpoint node ids.
    dist_m: perpendicular distance from fix to the segment (the emission
        argument), metres.
    frac: clamped projection fraction along u->v in [0, 1]; used to compute
        the route distance from here to the next fix's candidate.
    """

    edge_id: int
    u: int
    v: int
    dist_m: float
    frac: float


def _default_radius(sigma_m: float) -> float:
    """Candidate radius that does not truncate the true state space."""
    return max(3.0 * sigma_m, 50.0)


class CandidateGrid:
    """Spatial index for fast candidate-edge lookup via grid bucketing.

    Build once per graph + radius, then call query() per fix. Cell size
    equals the search radius (lat-direction), so every edge whose
    perpendicular distance is within the radius is guaranteed to appear in
    one of the 3x3 neighbourhood cells of the query point.

    Edges are straight segments between endpoint nodes; an edge is inserted
    into every cell its bounding box overlaps.
    """

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame,
                 radius_m: float) -> None:
        self.radius_m = float(radius_m)
        self.cell_size_deg = self.radius_m / M_PER_DEG_LAT

        node_pos: Dict[int, Tuple[float, float]] = {
            int(row.node_id): (float(row.lat), float(row.lon))
            for row in nodes.itertuples(index=False)
        }

        self._cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self._edge_info: List[Tuple[int, int, int, float, float, float, float]] = []
        for row in edges.itertuples(index=False):
            edge_id, u, v = int(row.edge_id), int(row.u), int(row.v)
            lat_a, lon_a = node_pos[u]
            lat_b, lon_b = node_pos[v]
            c_min = self._cell(min(lat_a, lat_b), min(lon_a, lon_b))
            c_max = self._cell(max(lat_a, lat_b), max(lon_a, lon_b))
            self._edge_info.append((edge_id, u, v, lat_a, lon_a, lat_b, lon_b))
            idx = len(self._edge_info) - 1
            for ci in range(c_min[0], c_max[0] + 1):
                for cj in range(c_min[1], c_max[1] + 1):
                    self._cells[(ci, cj)].append(idx)

    def _cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Grid cell indices for a degree coordinate."""
        return (
            math.floor(lat / self.cell_size_deg),
            math.floor(lon / self.cell_size_deg),
        )

    def query(self, lat: float, lon: float) -> List[Candidate]:
        """Return every edge within radius_m of (lat, lon), projected.

        Results are sorted by dist_m (nearest first) for determinism.
        """
        clat, clon = self._cell(lat, lon)
        out: List[Candidate] = []
        seen = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for idx in self._cells.get((clat + di, clon + dj), ()):
                    if idx in seen:
                        continue
                    seen.add(idx)
                    edge_id, u, v, lat_a, lon_a, lat_b, lon_b = self._edge_info[idx]
                    frac, dist_m, _, _ = project_to_segment(
                        lat, lon, lat_a, lon_a, lat_b, lon_b
                    )
                    if dist_m <= self.radius_m:
                        out.append(Candidate(edge_id, u, v, dist_m, frac))
        out.sort(key=lambda c: c.dist_m)
        return out


def candidates_for_fix(
    fix: Mapping[str, float],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    radius_m: Optional[float] = None,
    sigma_m: float = 15.0,
) -> List[Candidate]:
    """Return Candidate records for every edge within radius_m of `fix`.

    fix: {"lat": ..., "lon": ...} in degrees (t is not needed for the
        search). nodes/edges: tables as persisted by graph.save_graph.
    radius_m: search radius; defaults to _default_radius(sigma_m) so the
        candidate set stays complete across the sigma = 15/25/40 grid.
    Convenience wrapper — builds a fresh CandidateGrid per call. For batch
    processing (a whole trace), build one CandidateGrid and call query().
    """
    if radius_m is None:
        radius_m = _default_radius(sigma_m)
    grid = CandidateGrid(nodes, edges, radius_m)
    return grid.query(float(fix["lat"]), float(fix["lon"]))