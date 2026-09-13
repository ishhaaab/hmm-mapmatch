"""FastAPI: POST /match + GET /health.

Loads graph + fitted sigma/beta at startup, not per request.
"""
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="hmm-mapmatch")


class Fix(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    t: float = Field(..., ge=0.0)


class MatchRequest(BaseModel):
    points: List[Fix] = Field(..., min_length=2)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
def match(req: MatchRequest):
    """Phase 4: run candidates -> viterbi -> response."""
    raise NotImplementedError("Phase 4: implement /match")