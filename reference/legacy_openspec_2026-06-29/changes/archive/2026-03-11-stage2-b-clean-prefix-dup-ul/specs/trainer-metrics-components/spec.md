# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Stable metric and batch key names
The canonical metric docs SHALL include the clean-prefix Channel-B duplicate-ul and duplicate-collapse metrics introduced by this change.

Normative behavior:
- `docs/training/METRICS.md` MUST define the canonical training keys added by this contract.
- `docs/training/STAGE2_RUNBOOK.md` MUST define the corresponding Channel-B behavior and interpretation.
- Removed raw-prefix wording and removed legacy metric names MUST NOT linger in the canonical docs after implementation lands.

#### Scenario: Canonical duplicate metrics are documented
- **GIVEN** a training run after the clean-prefix Channel-B feature lands
- **WHEN** duplicate-ul and duplicate-collapse metrics are emitted
- **THEN** their canonical key names are documented in `docs/training/METRICS.md`
- **AND** the Channel-B contract that produces them is documented in `docs/training/STAGE2_RUNBOOK.md`.

### Requirement: Objective metrics emit canonical provenance keys only (atomic objective atoms; no raw component keys)
For registry-defined objective modules, trainers MUST emit only atomic objective contributions under canonical `loss/<provenance>/<atom>` keys.

Normative behavior for this change:
- Channel-B rollout-context objective atoms MUST now include:
  - `loss/B_rollout_text/duplicate_ul`
- Raw component loss aliases remain disallowed.

#### Scenario: Channel-B emits duplicate_ul as a canonical objective atom
- **WHEN** a Channel-B training step applies duplicate unlikelihood
- **THEN** the emitted objective key is `loss/B_rollout_text/duplicate_ul`
- **AND** no raw alias key for duplicate-unlikelihood is emitted.

## ADDED Requirements

### Requirement: Channel-B duplicate-collapse metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose duplicate-collapse diagnostics with explicit gauge-vs-counter naming.

Normative gauges:
- `dup/max_desc_count`
- `dup/saturation_rate`

Normative count-like metrics:
- `dup/near_iou90_pairs_same_desc_count`
- `dup/near_iou90_pairs_any_desc_count`
- `stage2_ab/channel_b/dup/N_raw_bbox_valid`
- `stage2_ab/channel_b/dup/N_clean_accepted`
- `stage2_ab/channel_b/dup/N_duplicates`
- `stage2_ab/channel_b/dup/N_duplicate_bursts`
- `stage2_ab/channel_b/dup/N_ul_boundaries`
- `stage2_ab/channel_b/dup/N_ul_skipped_no_divergence`

Normative behavior:
- Count-like metrics MUST use `/N_`, `_count`, `_total`, `_sum`, `_num`, or `_den` naming so optimizer-step aggregation treats them as additive totals.
- Gauge-like metrics MUST remain mean-like and MUST NOT masquerade as counters.

#### Scenario: Duplicate counters aggregate additively across micro-steps
- **WHEN** duplicate count-like metrics are emitted from multiple micro-steps in one optimizer step
- **THEN** the finalized step metric is the additive total
- **AND** the result is not diluted by mean-style aggregation.
