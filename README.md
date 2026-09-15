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

`results/metrics.json` contains 100 deterministic routes per cell, each with a
60 s gap, plus paired 95% route-level percentile bootstrap intervals from
10,000 resamples. The deployed model stays fixed at sigma 15.41 m, beta 6.12 m,
a 120 m candidate radius, 40 candidates per fix, and a 5 s observation interval;
the generating noise label is not passed to the decoder.

| Noise | Dropout | Edge recall (95% CI) | Edge F1 (95% CI) | HMM error (95% CI) | Raw error (95% CI) | Route length / true |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 m | 0% | 0.977 [0.971, 0.982] | 0.923 [0.915, 0.931] | 141 m [123, 160] | 320 m [283, 358] | 1.04x |
| 15 m | 10% | 0.975 [0.968, 0.981] | 0.922 [0.913, 0.931] | 137 m [121, 155] | 293 m [261, 327] | 1.04x |
| 25 m | 0% | 0.953 [0.944, 0.962] | 0.798 [0.786, 0.809] | 806 m [743, 873] | 1,247 m [1,155, 1,343] | 1.22x |
| 25 m | 10% | 0.947 [0.937, 0.957] | 0.792 [0.779, 0.805] | 777 m [712, 842] | 1,169 m [1,087, 1,254] | 1.22x |
| 40 m | 0% | 0.897 [0.882, 0.912] | 0.634 [0.618, 0.649] | 2,564 m [2,380, 2,758] | 3,025 m [2,823, 3,230] | 1.70x |
| 40 m | 10% | 0.893 [0.878, 0.907] | 0.634 [0.618, 0.649] | 2,424 m [2,255, 2,601] | 2,863 m [2,682, 3,045] | 1.67x |

Recall alone is optimistic at high noise, so F1 is the headline route metric.
Mean HMM distance error is below raw haversine in every cell, although the HMM
beats raw on only 75-94% of individual routes depending on the cell.
The intervals quantify route-sampling uncertainty within this synthetic setup;
they do not establish transfer to real traces or another region and are not a
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
