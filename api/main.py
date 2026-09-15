"""FastAPI serving layer for the HMM map matcher."""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import networkx as nx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.candidates import CandidateGrid, _default_radius
from src.em import MODEL_PATH
from src.graph import PROCESSED, as_routing_graph, load_graph
from src.mapmatch import (
    DEFAULT_OBSERVATION_INTERVAL_S,
    decode_trace_with_confidence,
    reconstruct_route,
    thin_trace,
)

MAX_POINTS = 5_000
MAX_SAMPLED_POINTS = 1_000
DEFAULT_SIGMA_M = 15.0
DEFAULT_BETA_M = 0.5


class Fix(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    t: float = Field(..., ge=0.0)


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[Fix] = Field(..., min_length=2, max_length=MAX_POINTS)

    @model_validator(mode="after")
    def timestamps_are_increasing(self) -> MatchRequest:
        timestamps = [point.t for point in self.points]
        if any(current <= previous for previous, current in pairwise(timestamps)):
            raise ValueError("point timestamps must be strictly increasing")
        return self


class MatchResponse(BaseModel):
    edge_ids: list[int]
    route_distance_m: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    input_points: int = Field(..., ge=2)
    sampled_points: int = Field(..., ge=2)
    matched_points: int = Field(..., ge=2)
    skipped_points: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: str
    nodes: int = Field(..., gt=0)
    edges: int = Field(..., gt=0)
    sigma_m: float = Field(..., gt=0.0)
    beta_m: float = Field(..., gt=0.0)
    observation_interval_s: float = Field(..., ge=0.0)
    candidate_radius_m: float = Field(..., gt=0.0)
    max_candidates: int = Field(..., gt=0)
    model_source: str


@dataclass(frozen=True)
class MatcherState:
    graph: nx.MultiDiGraph
    grid: CandidateGrid
    edge_lengths: dict[int, float]
    sigma_m: float
    beta_m: float
    node_count: int
    edge_count: int
    observation_interval_s: float
    candidate_radius_m: float
    max_candidates: int
    model_source: str


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return value


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def create_app(
    data_dir: Path | None = None,
    model_path: Path | None = None,
    sigma_m: float | None = None,
    beta_m: float | None = None,
    candidate_radius_m: float | None = None,
    max_candidates: int | None = None,
    observation_interval_s: float | None = None,
) -> FastAPI:
    """Create an app whose immutable matcher state is loaded at startup."""
    configured_model_path = Path(
        model_path or os.getenv("HMM_MODEL_PATH", str(MODEL_PATH))
    )
    explicit_model_path = model_path is not None or "HMM_MODEL_PATH" in os.environ
    model_values = {}
    if configured_model_path.exists():
        try:
            model_values = json.loads(configured_model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read model parameters: {exc}") from exc
        if not isinstance(model_values, dict):
            raise ValueError("model parameters must be a JSON object")
        if (
            type(model_values.get("schema_version")) is not int
            or model_values.get("schema_version") != 1
        ):
            raise ValueError("unsupported model parameter schema_version")
        required = {
            "sigma_m",
            "beta_m",
            "candidate_radius_m",
            "observation_interval_s",
            "max_candidates",
        }
        if missing := required - model_values.keys():
            raise ValueError(f"model parameters are missing {sorted(missing)}")
    elif explicit_model_path:
        raise FileNotFoundError(f"model parameters not found: {configured_model_path}")
    configured_data_dir = Path(data_dir or os.getenv("HMM_DATA_DIR", str(PROCESSED)))
    configured_sigma = _positive(
        sigma_m
        if sigma_m is not None
        else os.getenv("HMM_SIGMA_M", model_values.get("sigma_m", DEFAULT_SIGMA_M)),
        "sigma_m",
    )
    configured_beta = _positive(
        beta_m
        if beta_m is not None
        else os.getenv("HMM_BETA_M", model_values.get("beta_m", DEFAULT_BETA_M)),
        "beta_m",
    )
    configured_radius = _positive(
        candidate_radius_m
        if candidate_radius_m is not None
        else os.getenv(
            "HMM_CANDIDATE_RADIUS_M",
            model_values.get("candidate_radius_m", _default_radius(configured_sigma)),
        ),
        "candidate_radius_m",
    )
    configured_interval = _nonnegative(
        observation_interval_s
        if observation_interval_s is not None
        else os.getenv(
            "HMM_OBSERVATION_INTERVAL_S",
            model_values.get("observation_interval_s", DEFAULT_OBSERVATION_INTERVAL_S),
        ),
        "observation_interval_s",
    )
    configured_max_candidates = _positive_int(
        max_candidates
        if max_candidates is not None
        else os.getenv("HMM_MAX_CANDIDATES", model_values.get("max_candidates", 40)),
        "max_candidates",
    )

    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        nodes, edges = load_graph(configured_data_dir)
        graph = as_routing_graph(nodes, edges)
        edge_lengths = {
            int(edge_id): float(length_m)
            for edge_id, length_m in zip(edges["edge_id"], edges["length_m"])
        }
        instance.state.matcher = MatcherState(
            graph=graph,
            grid=CandidateGrid(
                nodes, edges, configured_radius, configured_max_candidates
            ),
            edge_lengths=edge_lengths,
            sigma_m=configured_sigma,
            beta_m=configured_beta,
            node_count=len(nodes),
            edge_count=len(edges),
            observation_interval_s=configured_interval,
            candidate_radius_m=configured_radius,
            max_candidates=configured_max_candidates,
            model_source=(
                str(configured_model_path) if model_values else "built-in defaults"
            ),
        )
        yield

    application = FastAPI(
        title="hmm-mapmatch",
        version="0.4.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        state: MatcherState = request.app.state.matcher
        return HealthResponse(
            status="ok",
            nodes=state.node_count,
            edges=state.edge_count,
            sigma_m=state.sigma_m,
            beta_m=state.beta_m,
            observation_interval_s=state.observation_interval_s,
            candidate_radius_m=state.candidate_radius_m,
            max_candidates=state.max_candidates,
            model_source=state.model_source,
        )

    @application.post("/match", response_model=MatchResponse)
    def match(payload: MatchRequest, request: Request) -> MatchResponse:
        started = time.perf_counter()
        state: MatcherState = request.app.state.matcher
        fixes = {
            "lat": np.asarray([point.lat for point in payload.points], dtype=float),
            "lon": np.asarray([point.lon for point in payload.points], dtype=float),
            "t": np.asarray([point.t for point in payload.points], dtype=float),
        }
        try:
            sampled_fixes, _ = thin_trace(fixes, state.observation_interval_s)
            if len(sampled_fixes["t"]) > MAX_SAMPLED_POINTS:
                raise ValueError(
                    f"trace has more than {MAX_SAMPLED_POINTS} sampled points"
                )
            matched, kept, confidence = decode_trace_with_confidence(
                sampled_fixes,
                state.grid,
                state.graph,
                state.edge_lengths,
                state.sigma_m,
                state.beta_m,
            )
            if len(matched) < 2:
                raise ValueError("fewer than two points have road candidates")
            route = reconstruct_route(matched, state.edge_lengths, state.graph)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return MatchResponse(
            edge_ids=route.edge_ids,
            route_distance_m=route.total_m,
            confidence=confidence,
            input_points=len(payload.points),
            sampled_points=len(sampled_fixes["t"]),
            matched_points=len(matched),
            skipped_points=len(sampled_fixes["t"]) - len(kept),
            latency_ms=(time.perf_counter() - started) * 1_000.0,
        )

    return application


app = create_app()
