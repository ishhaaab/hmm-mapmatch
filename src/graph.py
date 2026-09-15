"""OSM road-graph download + parquet persist.

Phase 1 entry point. Keep the place query small (a neighbourhood, not a city)
so the graph stays manageable (target 15-25k edges).

Units: node coordinates in degrees (lat/lon, EPSG:4326), edge lengths in
metres, bearings in degrees clockwise from north.

Geometry model: every edge is a straight segment between its endpoint nodes
(node positions come from the nodes table). OSMnx simplified edges can span
intermediate shape points; the straight-segment approximation is consistent
with the densifier and bearing computations and adequate at this scale. If
sub-segment precision is ever needed, persist each edge's OSM `geometry`
shape points instead.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import networkx as nx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.geo import bearing_deg

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

PLACE = "HSR Layout, Bengaluru, India"

# HSR Layout + surroundings (Koramangala / Bellandur corridor). Target 15-25k
# edges; still neighbourhood-scale, NOT all of Bangalore.
# Tuple order for osmnx 2.x graph_from_bbox: (west, south, east, north).
HSR_BBOX = (77.605, 12.885, 77.675, 12.935)

NODES_SCHEMA = pa.schema(
    [
        ("node_id", pa.int64()),
        ("lat", pa.float64()),  # degrees
        ("lon", pa.float64()),  # degrees
    ]
)
EDGES_SCHEMA = pa.schema(
    [
        ("edge_id", pa.int64()),
        ("u", pa.int64()),
        ("v", pa.int64()),
        ("length_m", pa.float64()),  # metres
        ("bearing", pa.float64()),  # degrees clockwise from north
    ]
)


def fetch_graph(place: str = PLACE, network_type: str = "drive") -> nx.MultiDiGraph:
    """Download the drivable street network and keep its routable core.

    Input: place (Nominatim query, e.g. a neighbourhood, NOT a whole city),
    network_type (OSMnx filter, default 'drive').
    Output: osmnx MultiDiGraph restricted to its largest strongly-connected
    component so random origin/destination pairs are routable.
    The default place falls back to a fixed HSR bounding box for known OSM or
    HTTP response failures. Programming and validation errors are not hidden.
    """
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
    from requests import RequestException

    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.overpass_rate_limit = True
    try:
        G = ox.graph_from_place(
            place, network_type=network_type, simplify=True, retain_all=False
        )
    except (
        InsufficientResponseError,
        ResponseStatusCodeError,
        RequestException,
    ) as exc:
        if place != PLACE:
            raise
        warnings.warn(
            f"place lookup failed ({exc!r}); falling back to the HSR bounding box",
            RuntimeWarning,
            stacklevel=2,
        )
        G = ox.graph_from_bbox(
            HSR_BBOX, network_type=network_type, simplify=True, retain_all=False
        )
    G = largest_strongly_connected(G)
    return G


def largest_strongly_connected(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Restrict G to its largest strongly-connected component (a copy).

    Input: directed MultiDiGraph. Output: subgraph copy on the largest SCC,
    so every node can reach every other node (routability).
    """
    if len(G) == 0:
        raise ValueError("cannot select a connected component from an empty graph")
    largest = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(largest).copy()


def graph_to_tables(G: nx.MultiDiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert an osmnx graph into nodes/edges tables.

    Input: MultiDiGraph with per-node x (lon, degrees) / y (lat, degrees)
    and per-edge length (metres).
    Outputs: nodes(node_id, lat, lon), edges(edge_id, u, v, length_m,
    bearing) with edge_id assigned sequentially in iteration order.
    """
    node_rows = [
        {"node_id": int(n), "lat": float(d["y"]), "lon": float(d["x"])}
        for n, d in G.nodes(data=True)
    ]
    nodes = pd.DataFrame(node_rows, columns=["node_id", "lat", "lon"])
    nodes = nodes.sort_values("node_id").reset_index(drop=True)

    edge_rows = []
    for edge_id, (u, v, _key, d) in enumerate(G.edges(keys=True, data=True)):
        du = G.nodes[u]
        dv = G.nodes[v]
        length_m = float(d.get("length", 0.0))
        if not math.isfinite(length_m) or length_m <= 0.0:
            raise ValueError(f"edge {u}->{v} has invalid length {length_m}")
        edge_rows.append(
            {
                "edge_id": int(edge_id),
                "u": int(u),
                "v": int(v),
                "length_m": length_m,
                "bearing": bearing_deg(du["y"], du["x"], dv["y"], dv["x"]),
            }
        )
    edges = pd.DataFrame(
        edge_rows, columns=["edge_id", "u", "v", "length_m", "bearing"]
    )
    # Edge ids are sequential-in-iteration-order; sorting makes the row order
    # deterministic even if osmnx iteration order ever varies between runs.
    edges = edges.sort_values("edge_id").reset_index(drop=True)
    return nodes, edges


def download_graph(place: str = PLACE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the graph for `place` and return it as tables.

    Input: place (Nominatim query string, degrees-based lookup server-side).
    Outputs: (nodes, edges) DataFrames with the schemas documented above.
    """
    G = fetch_graph(place)
    return graph_to_tables(G)


def save_graph(
    nodes: pd.DataFrame, edges: pd.DataFrame, out_dir: Path = PROCESSED
) -> None:
    """Persist nodes/edges to nodes.parquet + edges.parquet.

    Inputs: DataFrames with exactly the NODES/EDGES columns; out_dir.
    Writes parquet with explicit pyarrow schemas (no pandas one-liners),
    so column names, order, and dtypes are enforced at write time.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_table = pa.Table.from_arrays(
        [
            pa.array(nodes["node_id"].to_numpy(dtype="int64"), type=pa.int64()),
            pa.array(nodes["lat"].to_numpy(dtype="float64"), type=pa.float64()),
            pa.array(nodes["lon"].to_numpy(dtype="float64"), type=pa.float64()),
        ],
        schema=NODES_SCHEMA,
    )
    edges_table = pa.Table.from_arrays(
        [
            pa.array(edges["edge_id"].to_numpy(dtype="int64"), type=pa.int64()),
            pa.array(edges["u"].to_numpy(dtype="int64"), type=pa.int64()),
            pa.array(edges["v"].to_numpy(dtype="int64"), type=pa.int64()),
            pa.array(edges["length_m"].to_numpy(dtype="float64"), type=pa.float64()),
            pa.array(edges["bearing"].to_numpy(dtype="float64"), type=pa.float64()),
        ],
        schema=EDGES_SCHEMA,
    )
    pq.write_table(nodes_table, out_dir / "nodes.parquet")
    pq.write_table(edges_table, out_dir / "edges.parquet")


def load_graph(
    out_dir: Path = PROCESSED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read nodes.parquet + edges.parquet back, validating columns.

    Input: out_dir holding the parquet files. Outputs: (nodes, edges).
    Raises FileNotFoundError if either file is missing, ValueError on
    schema mismatch.
    """
    out_dir = Path(out_dir)
    nodes_path = out_dir / "nodes.parquet"
    edges_path = out_dir / "edges.parquet"
    if not nodes_path.exists():
        raise FileNotFoundError(f"missing {nodes_path}; run graph.main() first")
    if not edges_path.exists():
        raise FileNotFoundError(f"missing {edges_path}; run graph.main() first")
    nodes = pq.read_table(nodes_path, schema=NODES_SCHEMA).to_pandas()
    edges = pq.read_table(edges_path, schema=EDGES_SCHEMA).to_pandas()
    if list(nodes.columns) != ["node_id", "lat", "lon"]:
        raise ValueError(f"bad nodes.parquet columns: {list(nodes.columns)}")
    if list(edges.columns) != ["edge_id", "u", "v", "length_m", "bearing"]:
        raise ValueError(f"bad edges.parquet columns: {list(edges.columns)}")
    return nodes, edges


def as_routing_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.MultiDiGraph:
    """Rebuild a weighted MultiDiGraph from the persisted tables.

    Inputs: nodes/edges tables as written by save_graph. Output:
    MultiDiGraph with node x/y (degrees) and edge length_m (metres)
    weights, suitable for shortest-path sampling. No network access.
    """
    required_node_columns = {"node_id", "lat", "lon"}
    required_edge_columns = {"edge_id", "u", "v", "length_m"}
    if not required_node_columns.issubset(nodes.columns):
        missing = sorted(required_node_columns - set(nodes.columns))
        raise ValueError(f"nodes table is missing {missing}")
    if not required_edge_columns.issubset(edges.columns):
        missing = sorted(required_edge_columns - set(edges.columns))
        raise ValueError(f"edges table is missing {missing}")
    if nodes["node_id"].duplicated().any():
        raise ValueError("nodes table contains duplicate node_id values")
    if edges["edge_id"].duplicated().any():
        raise ValueError("edges table contains duplicate edge_id values")

    G: nx.MultiDiGraph = nx.MultiDiGraph()
    for row in nodes.itertuples(index=False):
        G.add_node(int(row.node_id), x=float(row.lon), y=float(row.lat))
    for row in edges.itertuples(index=False):
        if int(row.u) not in G or int(row.v) not in G:
            raise ValueError(f"edge {row.edge_id} references a missing node")
        if not math.isfinite(float(row.length_m)) or float(row.length_m) <= 0.0:
            raise ValueError(f"edge {row.edge_id} has an invalid length")
        G.add_edge(
            int(row.u),
            int(row.v),
            key=int(row.edge_id),
            edge_id=int(row.edge_id),
            length_m=float(row.length_m),
        )
    return G


def main(place: str = PLACE) -> None:
    """Download the graph, persist it, and print counts."""
    nodes, edges = download_graph(place)
    save_graph(nodes, edges)
    print(f"place: {place}")
    print(f"nodes: {len(nodes)}")
    print(f"edges: {len(edges)}")
    print(f"edge-km: {edges['length_m'].sum() / 1000.0:.1f}")
    print(f"wrote: {PROCESSED / 'nodes.parquet'}")
    print(f"wrote: {PROCESSED / 'edges.parquet'}")


if __name__ == "__main__":
    main()
