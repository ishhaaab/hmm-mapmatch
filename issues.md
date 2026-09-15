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

## Open Limitations

### Endpoint-only road geometry

OSM edge geometry and stable OSM identifiers are discarded. Curved roads are
approximated by endpoint segments, and same-direction parallel edges with equal
endpoints cannot be distinguished. Candidate generation keeps only the shortest
such edge. Full geometry must be persisted before claiming road-level accuracy
on complex junctions.

### High-noise false positives

At 40 m generating noise, edge recall remains about 0.86-0.88 but precision is
much lower, producing F1 around 0.60-0.62. The reconstructed route is about
1.67-1.72 times the true distance. A heading/speed model and real calibration
data are more defensible next steps than tuning against the five evaluation
seeds.

### Synthetic-only calibration

The checked-in sigma and beta are fitted on five synthetic routes. They are
useful reproducible defaults, not proof of transfer to phone GPS. Confidence is
also conditional on the candidate set and has no empirical calibration curve.

### Small benchmark

Each noise-grid cell has five routes. A second corruption seed preserves the
same broad degradation pattern, but this sample is too small for strong claims.
The next evaluation should use more seeds, confidence intervals, a second
region, and real labels.

### Simplified temporal model

Timestamps are validated and used for thinning, but transition scores do not
model elapsed time, speed, acceleration, or heading. The matcher bridges gaps
through spatial network consistency, not momentum.

### Hard training, not soft EM

`refit_sigma_beta` uses Viterbi assignments. A true EM implementation would use
forward/backward expectations and needs careful treatment of candidate-set
normalization. The current name is retained for project continuity, but all
documentation calls the method hard Viterbi training.
