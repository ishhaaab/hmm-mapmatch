# HMM Map-Matching from Scratch

Snapping noisy GPS traces to the OSM road network with a hand-built Hidden
Markov Model + Viterbi decoder. No `hmmlearn`, NumPy only.

## Status

Viterbi core implemented: `candidates` uses a
grid-bucketed spatial index, `hmm` decodes the variable-width trellis in
log space, `mapmatch` wires the whole pipeline, `em` refits sigma/beta.
Phases 3–4 (grid eval, `api/main.py`) are not yet implemented.

Measured on the real HSR Layout graph at σ = 15 m (1 Hz fixes, 8 m/s):

- Dense GPS (w/o tunnel gap): **mean route match rate 0.90** over 20 routes.
- Full noise model (10% dropouts + 60 s tunnel gap): mean 0.82 — the gap
  hides ~11% of the route under missing data that no decoder can observe;
  the gap is bridged by momentum.
- Route-distance error always beats both baselines (nearest-segment snap is
  catastrophically off; raw haversine overshoots by ~1.5× the true length).

Details and reproduction in `tests/test_phase2.py`. Edge-id match rate is
strict: the opposite direction of a bidirectional street counts as a miss.

## Quickstart (once Phase 0 is done)

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
pytest tests/
uvicorn api.main:app --reload
```

## Layout

- `src/graph.py` — OSMnx download + parquet persist
- `src/geo.py` — shared geometry: haversine, bearing, point→segment projection
- `src/synthesize.py` — route sampler + noise model (edge-id ground truth)
- `src/candidates.py` — candidate segments per fix (grid-bucketed spatial index)
- `src/hmm.py` — emission, transition, Viterbi (log space)
- `src/mapmatch.py` — pipeline: candidates → emission/transition → Viterbi, + baselines
- `src/em.py` — refit sigma/beta from matched traces
- `src/evaluate.py` — match rate, distance error, slices
- `api/main.py` — `POST /match`
- `results/metrics.json` — noise-grid results
