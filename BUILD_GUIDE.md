# Build Guide

## Implemented

### Phase 1: Graph and Synthesis

- Download and persist a directed OSM graph.
- Sample seeded, length-weighted shortest routes.
- Generate internally consistent 1 Hz truth with Gaussian jitter, dropout, and
  tunnel gaps.
- Include processed graph data for offline tests and Docker startup.

### Phase 2: HMM Core

- Use a latitude-aware candidate grid with a fixed deployment search radius
  and bounded candidate count.
- Score Gaussian emissions and Newson-Krumm distance-discrepancy transitions.
- Decode variable-width candidate sets in log space.
- Reject impossible trellises and invalid numerical inputs.
- Fit sigma and beta with deterministic hard Viterbi training.

### Phase 3: Evaluation

- Apply shared temporal preprocessing to all methods.
- Reconstruct route edges and distance through the directed graph.
- Report edge recall, precision, F1, LCS recall, distance error, and
  time-equivalent distance error.
- Run a fixed-parameter sigma/dropout grid with route-specific corruption RNGs.
- Evaluate 100 unique route seeds per cell with paired, route-level percentile
  bootstrap confidence intervals.
- Verify behavior with a second corruption seed.

### Phase 4: Serving Foundation

- Load graph, candidate index, and `model/params.json` once at startup.
- Validate request coordinates, size, and timestamp order.
- Return route IDs, distance, posterior margin, point counts, and latency.
- Package the graph/model in a non-root Docker image with a readiness health
  check and a small API-only dependency set.

## Next Checks

1. Add real labeled traces and a geographically separate holdout.
2. Benchmark p50/p90/p99 latency and concurrent throughput.
3. Calibrate confidence and add a low-confidence abstention response.
4. Persist full OSM geometry, then re-run all metrics because candidate identity
   and projection distances will change.
5. Define monitoring and retraining only after a real feedback/label source is
   available.

## Commands

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m src.em
.venv\Scripts\python -m src.evaluate
docker build -t hmm-mapmatch .
docker run --rm -p 8000:8000 hmm-mapmatch
```
