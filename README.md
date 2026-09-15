# HMM Map Matching from Scratch

A NumPy implementation of Hidden Markov Model map matching on an OpenStreetMap
road graph. It generates reproducible noisy GPS traces, finds nearby road
candidates, decodes the most likely sequence with Viterbi, reconstructs the
network route, evaluates it against exact synthetic ground truth, and serves it
through FastAPI.

## Status

Phases 1-3 are implemented. Phase 4 now has a startup-loaded `POST /match` API,
health reporting, a fitted parameter artifact, and Docker packaging. Real-data
validation, parameter monitoring, and scheduled retraining remain roadmap work;
MLflow, Evidently, and DuckDB are not claimed as implemented.

## Model

- Hidden state: a directed road segment near each retained GPS fix.
- Emission: Gaussian log density on perpendicular fix-to-segment distance.
- Transition: Newson-Krumm exponential likelihood on the difference between
  network distance and haversine displacement.
- Stabilization: non-negative progress on one directed edge and a fixed
  immediate U-turn penalty prevent projection jitter from becoming fake travel.
- Inference: variable-width, log-space Viterbi. An all-impossible trellis is an
  error rather than an arbitrary state-zero path.
- Confidence: mean forward/backward posterior margin of the Viterbi state over
  its strongest alternative. It is conditional on the generated candidate set
  and is not calibrated as a real-world correctness probability.
- Preprocessing: all compared methods use a 5 s observation interval. At 1 Hz,
  15 m noise is larger than typical 8 m vehicle movement, which violates the
  useful operating regime of the transition model.
- Route output: consecutive states on one edge are collapsed and edge changes
  are joined with weighted shortest paths. The old projection-foot sum remains
  a diagnostic only.

`src/em.py` performs hard Viterbi training, not soft EM. The checked-in model
artifact was fitted on synthetic route seeds 20-24 and is separate from the
evaluation seeds. Candidate sets are capped at the nearest 40 states to bound
the quadratic trellis cost.

## Results

`results/metrics.json` contains five deterministic routes per cell, each with a
60 s gap. The deployed model stays fixed at sigma 15.41 m, beta 6.12 m, a 120 m
candidate radius, 40 candidates per fix, and a 5 s observation interval; the
generating noise label is not passed to the decoder.

| Noise | Dropout | Edge recall | Edge F1 | HMM error | Raw error | Route length / true |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 m | 0% | 0.979 | 0.925 | 173 m | 326 m | 1.05x |
| 15 m | 10% | 0.984 | 0.921 | 148 m | 262 m | 1.04x |
| 25 m | 0% | 0.931 | 0.764 | 952 m | 1,330 m | 1.25x |
| 25 m | 10% | 0.904 | 0.757 | 757 m | 1,181 m | 1.21x |
| 40 m | 0% | 0.877 | 0.623 | 2,675 m | 3,214 m | 1.71x |
| 40 m | 10% | 0.862 | 0.604 | 2,457 m | 2,927 m | 1.67x |

The HMM beats raw haversine on mean distance error in every cell, but not on
every individual route. Recall alone is optimistic at high noise, so F1 is the
headline route metric. These are synthetic results over a small sample, not a
production accuracy claim.

## Run

Python 3.11 or newer is required. The processed HSR Layout graph is included so
tests and the API work from a clean checkout.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m src.em
.venv\Scripts\python -m src.evaluate
.venv\Scripts\uvicorn api.main:app --reload
```

Example request:

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{"points":[{"lat":12.9031118,"lon":77.6393678,"t":0},{"lat":12.9031242,"lon":77.6390035,"t":5},{"lat":12.9031366,"lon":77.6386391,"t":10}]}'
```

The response contains reconstructed `edge_ids`, `route_distance_m`, posterior
margin `confidence`, point counts, and compute `latency_ms`. Timestamps must be
strictly increasing. Requests are limited to 5,000 input points and 1,000 points
after temporal thinning. Configuration can be overridden with `HMM_DATA_DIR`,
`HMM_MODEL_PATH`, `HMM_SIGMA_M`, `HMM_BETA_M`,
`HMM_CANDIDATE_RADIUS_M`, `HMM_MAX_CANDIDATES`, and
`HMM_OBSERVATION_INTERVAL_S`.

```bash
docker build -t hmm-mapmatch .
docker run --rm -p 8000:8000 hmm-mapmatch
```

## Layout

- `src/graph.py`: OSMnx download, schema validation, and parquet persistence
- `src/geo.py`: haversine, bearing, and local segment projection
- `src/synthesize.py`: weighted shortest-route sampler and GPS noise model
- `src/candidates.py`: latitude-aware grid candidate index
- `src/hmm.py`: HMM likelihoods, Viterbi, and forward/backward inference
- `src/mapmatch.py`: trace preprocessing, decoding, and route reconstruction
- `src/em.py`: hard-Viterbi calibration and model artifact generation
- `src/evaluate.py`: route, distance, and time-equivalent error metrics
- `api/main.py`: startup-loaded FastAPI service

The included road-network extract is derived from OpenStreetMap data and must
be used with OpenStreetMap attribution under the ODbL. Copyright OpenStreetMap
contributors.
