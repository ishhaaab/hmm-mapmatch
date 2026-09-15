"""Candidate road segments per GPS fix.

For each fix, find the road segments that could plausibly have produced it
and project the fix onto each one.

Design contract (used by hmm.viterbi via the matched pipeline):
- The road network is modelled as straight segments between the edge
  endpoint nodes (nodes table) — see graph.py module docs.
- The convenience function derives its radius from sensor noise sigma. The
  deployed matcher instead uses a fitted 120 m radius and caps each fix at the
  nearest 40 candidates to bound the quadratic trellis cost.
- Every candidate carries the clamped perpendicular projection: `dist_m`
  feeds the emission, `frac` lets the transition assemble the route distance
  (partial edge distance + shortest network path + partial edge distance).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import NamedTuple

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

    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        radius_m: float,
        max_candidates: int | None = None,
    ) -> None:
        required_nodes = {"node_id", "lat", "lon"}
        required_edges = {"edge_id", "u", "v", "length_m"}
        if not required_nodes.issubset(nodes.columns):
            raise ValueError(
                f"nodes table is missing {sorted(required_nodes - set(nodes.columns))}"
            )
        if not required_edges.issubset(edges.columns):
            raise ValueError(
                f"edges table is missing {sorted(required_edges - set(edges.columns))}"
            )
        self.radius_m = float(radius_m)
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be finite and > 0, got {radius_m}")
        if nodes.empty:
            raise ValueError("nodes must not be empty")
        if edges.empty:
            raise ValueError("edges must not be empty")
        if max_candidates is not None and (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            raise ValueError(
                "max_candidates must be a positive integer or None, "
                f"got {max_candidates}"
            )
        self.max_candidates = (
            int(max_candidates) if max_candidates is not None else None
        )
        node_pos: dict[int, tuple[float, float]] = {}
        for row in nodes.itertuples(index=False):
            node_id = int(row.node_id)
            if node_id in node_pos:
                raise ValueError(f"duplicate node_id {node_id}")
            lat, lon = float(row.lat), float(row.lon)
            if not math.isfinite(lat) or not -90.0 <= lat <= 90.0:
                raise ValueError(f"node {node_id} has invalid latitude {lat}")
            if not math.isfinite(lon) or not -180.0 <= lon <= 180.0:
                raise ValueError(f"node {node_id} has invalid longitude {lon}")
            node_pos[node_id] = (lat, lon)

        self.lat_cell_size_deg = self.radius_m / M_PER_DEG_LAT
        max_abs_lat = max(abs(lat) for lat, _ in node_pos.values())
        max_abs_lat = min(89.999999, max_abs_lat + self.lat_cell_size_deg)
        min_lon_scale = M_PER_DEG_LAT * math.cos(math.radians(max_abs_lat))
        self.lon_cell_size_deg = self.radius_m / min_lon_scale

        # Endpoint-only geometry cannot distinguish parallel edges in the same
        # direction. Routing always selects the shortest, so longer copies are
        # dominated states and should not enter the trellis.
        best_edges = {}
        seen_edge_ids = set()
        for row in edges.itertuples(index=False):
            edge_id, u, v = int(row.edge_id), int(row.u), int(row.v)
            if edge_id in seen_edge_ids:
                raise ValueError(f"duplicate edge_id {edge_id}")
            seen_edge_ids.add(edge_id)
            if u not in node_pos or v not in node_pos:
                raise ValueError(f"edge {edge_id} references a missing node")
            length_m = float(row.length_m)
            if not math.isfinite(length_m) or length_m <= 0.0:
                raise ValueError(f"edge {edge_id} must have a finite, positive length")
            key = (u, v)
            incumbent = best_edges.get(key)
            if incumbent is None or (length_m, edge_id) < incumbent[:2]:
                best_edges[key] = (length_m, edge_id, row)

        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._edge_info: list[tuple[int, int, int, float, float, float, float]] = []
        for _, _, row in sorted(best_edges.values(), key=lambda item: item[1]):
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

    def _cell(self, lat: float, lon: float) -> tuple[int, int]:
        """Grid cell indices for a degree coordinate."""
        return (
            math.floor(lat / self.lat_cell_size_deg),
            math.floor(lon / self.lon_cell_size_deg),
        )

    def query(self, lat: float, lon: float) -> list[Candidate]:
        """Return projected edges within radius_m of (lat, lon).

        Results are sorted by dist_m (nearest first) for determinism, then
        truncated to ``max_candidates`` when a cap was configured.
        """
        if not math.isfinite(lat) or not -90.0 <= lat <= 90.0:
            raise ValueError(f"lat must be finite and in [-90, 90], got {lat}")
        if not math.isfinite(lon) or not -180.0 <= lon <= 180.0:
            raise ValueError(f"lon must be finite and in [-180, 180], got {lon}")
        clat, clon = self._cell(lat, lon)
        out: list[Candidate] = []
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
        out.sort(key=lambda c: (c.dist_m, c.edge_id, c.u, c.v))
        return out[: self.max_candidates]


def candidates_for_fix(
    fix: Mapping[str, float],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    radius_m: float | None = None,
    sigma_m: float = 15.0,
) -> list[Candidate]:
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
