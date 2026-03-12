# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical metric docs and logs reflect v3 triage semantics
The canonical metric docs SHALL describe the v3 K=2 triage metrics and the broadened meaning of `train/optimization/loss_dead_anchor_suppression`.

Normative behavior:

- `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS.md` MUST describe the anchor/explorer K=2 target-construction flow,
- the docs MUST state that `train/optimization/loss_dead_anchor_suppression` covers dead anchor-side continuation suppression in the v3 contract,
- legacy duplicate metrics MAY remain documented as supporting diagnostics.

#### Scenario: loss_dead_anchor_suppression docs reflect dead-anchor suppression semantics
- **WHEN** v3 triage lands
- **THEN** the canonical docs describe `train/optimization/loss_dead_anchor_suppression` as the dead-anchor continuation suppression atom for Channel-B
- **AND** they do not describe it as same-desc duplicate-only cleanup.

## ADDED Requirements

### Requirement: Channel-B triage metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose the v3 triage bookkeeping separately from legacy duplicate diagnostics.

Normative count-like metrics:

- `train/triage/gt_backed_count`
- `train/triage/unlabeled_consistent_count`
- `train/triage/dead_anchor_count`
- `train/triage/explorer_only_dead_count`
- `train/triage/recovered_ground_truth_count`

Normative numerator / denominator metrics:

- `train/triage/recovered_ground_truth_rate_num`
- `train/triage/recovered_ground_truth_rate_den`
- `train/triage/dead_anchor_rate_num`
- `train/triage/dead_anchor_rate_den`

Normative behavior:

- count-like metrics MUST aggregate additively across micro-steps,
- numerator / denominator metrics MUST aggregate additively across micro-steps,
- monitor-dump payloads SHOULD expose both rollout views plus the final triage decision for high-signal samples.

#### Scenario: Triage counts aggregate additively
- **WHEN** multiple micro-steps emit triage count metrics within one optimizer step
- **THEN** the finalized step metrics are additive totals rather than mean-diluted gauges.
