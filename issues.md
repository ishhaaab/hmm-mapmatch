# Issues and Limitations

## Fixed

- `sample_route` previously gave every `MultiDiGraph` edge weight zero because
  its callback misread NetworkX's parallel-edge mapping. Generated benchmark
  routes are now verified against weighted shortest-path distance.
- Synthetic timestamps used endpoint-chord distance while route truth used OSM
  edge length. Densification now uses the same edge-length cumulative axis.
- Longitude grid cells previously caused in-radius false negatives away from the
  equator. The index now uses a conservative latitude-aware longitude scale.
- An all-`-inf` Viterbi timestep previously returned arbitrary state zero. It now
  fails explicitly, and malformed arrays are validated.
- Independent snapping and the HMM were evaluated with a projection-foot sum
  that counted high-frequency GPS wobble as travel. Production evaluation now
  uses shared 5 s preprocessing and reconstructed network routes.
- All routes previously reused the same corruption RNG stream. Each route now
  receives a deterministic independent stream while cells remain paired.
- `/match` previously returned HTTP 500 for every valid request while `/health`
  still reported success. Matching, startup state, response schemas, and model
  health metadata are implemented.
- Processed graph data, project documentation, and metrics were omitted from a
  clean clone by broad ignore rules. They are now included explicitly.
- The six-cell noise benchmark now uses 100 unique routes instead of five and
  reports paired, route-level percentile bootstrap confidence intervals.

## Open Limitations

### Endpoint-only road geometry

OSM edge geometry and stable OSM identifiers are discarded. Curved roads are
approximated by endpoint segments, and same-direction parallel edges with equal
endpoints cannot be distinguished. Candidate generation keeps only the shortest
such edge. Full geometry must be persisted before claiming road-level accuracy
on complex junctions.

### High-noise false positives

At 40 m generating noise, edge recall remains about 0.89-0.90 but precision is
about 0.49, producing F1 around 0.63. The reconstructed route is about
1.67-1.70 times the true distance. A heading/speed model and real calibration
data are more defensible next steps than tuning against the synthetic benchmark.

### Synthetic-only calibration

The checked-in sigma and beta are fitted on five synthetic routes. They are
useful reproducible defaults, not proof of transfer to phone GPS. Confidence is
also conditional on the candidate set and has no empirical calibration curve.

### Synthetic benchmark scope

Each noise-grid cell has 100 routes with 95% bootstrap intervals. This reduces
sampling uncertainty but does not address transfer beyond one synthetic road
graph and noise model. The next evaluation needs a second region and real
labels.

### Simplified temporal model

Timestamps are validated and used for thinning, but transition scores do not
model elapsed time, speed, acceleration, or heading. The matcher bridges gaps
through spatial network consistency, not momentum.

### Hard training, not soft EM

`refit_sigma_beta` uses Viterbi assignments. A true EM implementation would use
forward/backward expectations and needs careful treatment of candidate-set
normalization. The current name is retained for project continuity, but all
documentation calls the method hard Viterbi training.
