# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Objective metrics emit canonical provenance keys only (atomic objective atoms; no raw component keys)
For registry-defined coord-side objectives, trainers MUST emit canonical
adjacent-repulsion atom keys only when the term is enabled and non-zero.

Normative behavior:

- Stage-2 MAY emit:
  - `loss/B_coord/adjacent_repulsion`
  for rollout-context provenance in Stage-2,
- Stage-1 MUST NOT invent a second unrelated Stage-1-only atom namespace for
  the same term and MAY instead emit its contribution through the existing
  Stage-1 coord auxiliary family,
- raw module-local adjacent metrics MAY exist internally, but
  `loss/B_coord/adjacent_repulsion` is the only canonical Stage-2 atom emitted
  by this change.

#### Scenario: Enabled Stage-2 adjacent repulsion emits canonical provenance keys
- **WHEN** Stage-2 adjacent repulsion contributes to the objective
- **THEN** the emitted canonical Stage-2 atom key is
  `loss/B_coord/adjacent_repulsion`
- **AND** no raw alias key is emitted for the same Stage-2 contribution.

### Requirement: Coord distribution diagnostics are provenance-split in Stage-2 two-channel
When adjacent repulsion is enabled, the trainer metrics contract SHALL expose
adjacent-pair diagnostics in the existing coord diagnostics families.

Normative behavior:

- Stage-1 diagnostics MAY include:
  - `coord_diag/adjacent_repulsion_pair_count`
  - `coord_diag/adjacent_repulsion_applied_count`
  - `coord_diag/adjacent_repulsion_copy_score_mean`
- Stage-2 diagnostics MAY include:
  - `coord_diag/B/adjacent_repulsion_pair_count` for rollout-context provenance
  - matching `*_applied_count` keys under the same families,
  - matching `*_copy_score_mean` keys under the same families,
- count-like adjacent diagnostics MUST aggregate additively across micro-steps.

#### Scenario: Adjacent-pair diagnostics are aggregation-safe
- **WHEN** adjacent-repulsion pair counts are emitted from multiple micro-steps
- **THEN** the finalized optimizer-step counts are additive totals
- **AND** they are not mean-diluted gauges.
