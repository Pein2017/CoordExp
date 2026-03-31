# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Stable metric and batch key names
The canonical metric docs SHALL include the pseudo-positive Channel-B triage metrics introduced by this change.

Normative behavior:

- `docs/training/METRICS.md` MUST define the canonical training keys added by this contract,
- `docs/training/STAGE2_RUNBOOK.md` MUST define the corresponding Channel-B pseudo-positive behavior and interpretation,
- removed blanket-neutral wording and removed legacy metric names MUST NOT linger in the canonical docs after implementation lands.

#### Scenario: Canonical pseudo-positive metrics are documented
- **GIVEN** a training run after the pseudo-positive Channel-B feature lands
- **WHEN** pseudo-positive triage metrics are emitted
- **THEN** their canonical key names are documented in `docs/training/METRICS.md`
- **AND** the Channel-B contract that produces them is documented in `docs/training/STAGE2_RUNBOOK.md`.

### Requirement: Channel-B triage metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose pseudo-positive triage bookkeeping separately from legacy duplicate diagnostics.

Normative count-like metrics introduced by this change:

- `train/triage/pseudo_positive_candidate_count`
- `train/triage/pseudo_positive_subthreshold_count`
- `train/triage/pseudo_positive_selected_count`
- `train/triage/pseudo_positive_cluster_demoted_count`
- `train/triage/anchor_preparation_dropped_count`

Normative numerator / denominator metrics introduced by this change:

- `train/triage/pseudo_positive_support_rate_num`
- `train/triage/pseudo_positive_support_rate_den`
- `train/triage/pseudo_positive_selected_support_rate_num`
- `train/triage/pseudo_positive_selected_support_rate_den`

Normative behavior:

- pre-existing canonical Channel-B metrics from the base metrics contract remain in force unchanged,
- in pseudo-positive-enabled runs, the pre-existing `train/triage/recovered_ground_truth_rate_num` and `train/triage/recovered_ground_truth_rate_den` MUST serve as the canonical recovered-GT support-rate numerator / denominator pair rather than introducing a second recovered-GT rate family,
- in pseudo-positive-enabled runs, the pre-existing `train/triage/unlabeled_consistent_count` MUST remain the canonical total shielded-anchor count and MUST equal `train/triage/pseudo_positive_subthreshold_count + train/triage/pseudo_positive_cluster_demoted_count`,
- in pseudo-positive-enabled runs, legacy singular explorer metric families such as `rollout/explorer/*` MUST remain defined as mean-over-valid-explorer-view summaries rather than as one arbitrarily chosen explorer view,
- `stage2/raw_rollouts` MUST continue to count total raw rollout trajectories across the anchor rollout plus all explorer rollouts,
- `rollout/parse_truncated_rate` MUST continue to represent the all-rollout parse-truncated ratio over those total raw rollouts,
- explorer decode-profile metrics such as `rollout/explorer/temperature`, `rollout/explorer/do_sample`, `rollout/explorer/top_p`, and `rollout/explorer/top_k` MUST continue to report the shared explorer decode profile in pseudo-positive-enabled runs,
- count-like metrics MUST aggregate additively across micro-steps,
- numerator / denominator metrics MUST aggregate additively across micro-steps,
- `train/triage/unlabeled_consistent_count` MUST NOT be reinterpreted as a synonym for `train/triage/pseudo_positive_subthreshold_count`,
- explorer-preparation aborts are failure incidents rather than finalized optimizer-step metrics and MUST be surfaced through run-failure telemetry or ablation reporting instead of a normal `train/triage/*` key,
- monitor-dump payloads MAY mirror the pseudo-positive triage bookkeeping, but per-sample Channel-B metadata remains the canonical audit carrier.

#### Scenario: Pseudo-positive support-rate numerators and denominators aggregate additively
- **WHEN** multiple micro-steps emit pseudo-positive support-rate numerators and denominators within one optimizer step
- **THEN** the finalized step metrics are additive totals rather than mean-diluted gauges
- **AND** the resulting rates remain comparable across runs with different `num_rollouts` values.
