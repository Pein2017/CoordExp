## MODIFIED Requirements

### Requirement: Channel-B duplicate-collapse metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose duplicate-collapse diagnostics with
explicit gauge-vs-counter naming.

Normative gauges:
- `dup/max_desc_count`
- `dup/saturation_rate`
- `dup/duplicate_like_max_cluster_size`
- `dup/desc_entropy`

Normative count-like metrics:
- `dup/near_iou90_pairs_same_desc_count`
- `dup/near_iou90_pairs_any_desc_count`
- `stage2_ab/channel_b/dup/N_raw_bbox_valid`
- `stage2_ab/channel_b/dup/N_clean_accepted`
- `stage2_ab/channel_b/dup/N_duplicates`
- `stage2_ab/channel_b/dup/N_duplicate_bursts`
- `stage2_ab/channel_b/dup/N_duplicate_like_clusters`
- `stage2_ab/channel_b/dup/N_ul_boundaries`
- `stage2_ab/channel_b/dup/N_ul_skipped_no_divergence`

Normative behavior:
- Count-like metrics MUST use `/N_`, `_count`, `_total`, `_sum`, `_num`, or
  `_den` naming so optimizer-step aggregation treats them as additive totals.
- Gauge-like metrics MUST remain mean-like and MUST NOT masquerade as counters.
- `dup/duplicate_like_max_cluster_size` and
  `stage2_ab/channel_b/dup/N_duplicate_like_clusters` MUST be derived from the
  same cluster-aware duplicate-targeting relation used by Channel-B runtime
  preparation.

#### Scenario: Duplicate counters aggregate additively across micro-steps
- **WHEN** duplicate count-like metrics are emitted from multiple micro-steps
  in one optimizer step
- **THEN** the finalized step metric is the additive total
- **AND** the result is not diluted by mean-style aggregation.

#### Scenario: Cluster-aware duplicate metrics share the runtime detector
- **WHEN** Channel-B training emits duplicate-collapse diagnostics for a step
- **THEN** cluster-aware gauges and counters are derived from the same
  duplicate-like grouping used to build suppression targets
- **AND** operators do not need to reconcile separate duplicate definitions for
  metrics versus training behavior.
