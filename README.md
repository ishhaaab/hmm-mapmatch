# HMM Map-Matching from Scratch

Snapping noisy GPS traces to the OSM road network with a hand-built Hidden
Markov Model + Viterbi decoder. No `hmmlearn`, NumPy only.

## Status

Viterbi core implemented. `candidates` buckets
road segments into a grid, `hmm` decodes the trellis with log
probabilities, `mapmatch` connects the pieces, `em` refits sigma and beta
from matched traces. Phases 3 and 4 (the noise grid in `evaluate.py` and
the `/match` API) are not implemented yet.

Measured on the real HSR Layout graph at sigma 15 m (1 Hz fixes, 8 m/s):

- Dense GPS, no tunnel gap: mean route match rate 0.90 over 10 routes
  (test floor > 0.85).
- Full noise model, 10% dropout plus a 60 s tunnel gap: mean 0.82 over 15
  routes (test floor > 0.70). The gap hides ~11% of the route under
  missing data, and the decoder bridges the gap with momentum.
- Route-distance error: the HMM beats both baselines on every one of 5
  full-noise routes tested. Nearest-snap is far off, and raw haversine
  overshoots the true length by ~2.7×.

There's a known issue: the HMM's own route length runs about 2.2× the true
length at sigma 15 m. See `issues.md`.

Reproduction: run `pytest tests/`. The match rate counts edge ids
strictly, so the opposite direction of a bidirectional street counts as a
miss.

## Quickstart (once Phase 0 is done)

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
pytest tests/
uvicorn api.main:app --reload
```

## Layout

- `src/graph.py`: OSMnx download + parquet persist
- `src/geo.py`: shared geometry: haversine, bearing, point-to-segment projection
- `src/synthesize.py`: route sampler + noise model (edge-id ground truth)
- `src/candidates.py`: candidate segments per fix (grid-bucketed spatial index)
- `src/hmm.py`: emission, transition, Viterbi (log space)
- `src/mapmatch.py`: pipeline + baselines
- `src/em.py`: refit sigma/beta from matched traces
- `src/evaluate.py`: match rate, distance error, slices
- `api/main.py`: `POST /match`
- `issues.md`: known problems
- `results/metrics.json`: noise-grid results