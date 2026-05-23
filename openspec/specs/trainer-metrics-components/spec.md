## Purpose
Define trainer metric component naming, aggregation, and duplicate-collapse
diagnostic contracts.

## Requirements

### Requirement: Channel-B duplicate-collapse metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose duplicate-collapse diagnostics with
explicit gauge-vs-counter naming and one canonical duplicate-control family.

Normative gauges:
- `dup/raw/max_desc_count`
- `dup/raw/saturation_rate`
- `dup/raw/duplicate_like_max_cluster_size`
- `dup/raw/desc_entropy`

Normative raw pathology counters:
- `dup/raw/near_iou90_pairs_same_desc_count`
- `dup/raw/near_iou90_pairs_any_desc_count`

Normative policy-action counters:
- `stage2_ab/channel_b/dup/N_clusters_total`
- `stage2_ab/channel_b/dup/N_clusters_exempt`
- `stage2_ab/channel_b/dup/N_clusters_suppressed`
- `stage2_ab/channel_b/dup/N_objects_suppressed`
- `stage2_ab/channel_b/dup/N_duplicate_control_first_divergence_boundaries`
- `stage2_ab/channel_b/dup/N_duplicate_control_first_divergence_skipped_no_divergence`

Normative behavior:
- Count-like metrics MUST use `/N_`, `_count`, `_total`, `_sum`, `_num`, or
  `_den` naming so optimizer-step aggregation treats them as additive totals.
- Gauge-like metrics MUST remain mean-like and MUST NOT masquerade as counters.
- `dup/raw/*` gauges MUST finalize as weighted means across micro-steps.
- `dup/raw/*_count` metrics and `stage2_ab/channel_b/dup/N_*` metrics MUST
  remain additive totals across micro-steps.
- `dup/raw/duplicate_like_max_cluster_size`,
  `dup/raw/desc_entropy`, and `stage2_ab/channel_b/dup/N_clusters_total` MUST
  be derived from the same cluster-aware duplicate-control relation used by
  Channel-B runtime preparation.
- raw pathology metrics and policy-action counters MUST stay distinct:
  - `dup/raw/*` describes the pre-policy current-attempt duplicate state,
  - `stage2_ab/channel_b/dup/N_*` describes current duplicate-control
    diagnostic decisions and metadata.
- duplicate-control counters MUST remain diagnostic metadata only.
  Duplicate-burst UL is not part of the current canonical objective list; only
  retired legacy loss scalars are tied to the retired objective module.

#### Scenario: Duplicate counters aggregate additively across micro-steps
- **WHEN** duplicate count-like metrics are emitted from multiple micro-steps
  in one optimizer step
- **THEN** the finalized step metric is the additive total
- **AND** the result is not diluted by mean-style aggregation.

#### Scenario: Cluster-aware duplicate metrics share the runtime detector
- **WHEN** Channel-B training emits duplicate-collapse diagnostics for a step
- **THEN** cluster-aware gauges and counters are derived from the same
  duplicate-like grouping used for filtering and diagnostic bookkeeping
- **AND** operators do not need to reconcile separate duplicate definitions for
  metrics versus training behavior.

#### Scenario: Raw duplicate gauges remain mean-like across micro-steps
- **WHEN** `dup/raw/duplicate_like_max_cluster_size` and `dup/raw/desc_entropy`
  are emitted from multiple micro-steps in one optimizer step
- **THEN** the finalized step metrics are weighted means rather than additive
  sums
- **AND** the policy-action counters remain additive in the same step.
