# Project Design

## Objective

Map noisy, timestamped GPS fixes to a directed road network without relying on
an HMM library. The project is designed to make each probabilistic and routing
assumption visible and testable.

## Data Model

The persisted graph has:

- `nodes(node_id, lat, lon)`
- `edges(edge_id, u, v, length_m, bearing)`

Edges use their OSM length but are projected as straight endpoint segments.
Parallel edges with the same directed endpoints are geometrically
indistinguishable after persistence, so candidate generation keeps the shortest
one, matching NetworkX routing behavior. Full OSM edge geometry is a planned
schema improvement.

Synthetic routes are true `length_m`-weighted shortest paths. Their cumulative
distance, timestamps, edge IDs, and reported route length use the same edge
length model. Corruption adds isotropic Gaussian position noise, independent
dropout, and seeded contiguous gaps.

## Probabilistic Model

For candidate segment `r` and observation `z`, the emission score is:

```text
log p(z | r) = -0.5 * ((distance(z, r) / sigma)^2 + log(2*pi*sigma^2))
```

For consecutive observations and candidates, the base transition score is:

```text
delta = |network_distance(candidate_a, candidate_b)
         - haversine(observation_a, observation_b)|
log p(delta) = -log(beta) - delta / beta
```

The implementation adds two explicit operational assumptions:

- projected progress cannot be negative while remaining on one directed edge;
- switching immediately to the reverse directed edge receives a fixed log
  penalty.

These prevent high-frequency GPS jitter from repeatedly creating U-turns. A
5-second sampling interval is applied before matching and to both baselines.
The timestamp controls sampling only; there is no velocity, acceleration, or
heading state, so the implementation must not be described as using momentum.

Viterbi computes the maximum-score path in log space. Forward/backward computes
state marginals for confidence. Since candidate generation truncates the state
space and the model is fitted on synthetic data, confidence is conditional and
not externally calibrated.

## Parameter Fitting

`refit_sigma_beta` is hard Viterbi training:

- decode under current parameters;
- set sigma to the RMSE of matched perpendicular residuals;
- set beta to the mean absolute network-versus-haversine discrepancy;
- repeat.

This is not textbook soft EM because it does not take expectations under the
full posterior. `python -m src.em` writes `model/params.json` using route seeds
20-24. Evaluation uses route seeds 7-11 to avoid direct train/evaluation reuse.

## Evaluation

Primary route metrics are unique-edge recall, precision, F1, and an order-aware
LCS recall. Distance uses the reconstructed network route, not the sum of every
noisy projection movement. Both HMM and baselines see the same sampled trace.

The reported time metric is distance absolute error divided by the simulator's
constant speed. It is a time-equivalent distance error, not validation of an ETA
prediction model.

The noise grid keeps deployment parameters fixed while changing generating
noise and dropout. This avoids oracle knowledge of the test noise. Each cell
currently has only five synthetic routes, so results need confidence intervals,
more seeds, another region, and real audited traces before any production claim.

## Serving Contract

`POST /match` accepts 2-5,000 points with finite coordinates and strictly
increasing nonnegative timestamps. To bound decode cost, no more than 1,000
points may remain after temporal thinning. It returns:

- reconstructed edge IDs, including network connectors;
- reconstructed route distance in metres;
- mean posterior margin confidence;
- input, sampled, matched, and candidate-miss counts;
- server compute latency in milliseconds.

Graph, the 120 m index, the 40-candidate cap, and fitted parameters load once
during application startup.
Invalid or unroutable traces return HTTP 422. `/health` reports loaded graph and
model configuration, so a process without usable startup data cannot report
healthy.

## Remaining Work

1. Preserve full OSM edge geometry and stable OSM identifiers.
2. Evaluate on map-aligned real traces and a second geographic region.
3. Calibrate confidence and define abstention thresholds.
4. Add heading/speed-aware transitions instead of relying only on thinning.
5. Add benchmark uncertainty, latency percentiles, and concurrency/load tests.
6. Add monitoring and scheduled recalibration only after real labels exist.
