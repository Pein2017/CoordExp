# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical metric docs and logs reflect v3 triage semantics
The canonical metric docs SHALL describe the v3 K=2 triage metrics and the broadened meaning of `loss/B_rollout_text/duplicate_ul`.

Normative behavior:

- `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS.md` MUST describe the anchor/explorer K=2 target-construction flow,
- the docs MUST state that `loss/B_rollout_text/duplicate_ul` covers dead anchor-side continuation suppression in the v3 contract,
- legacy duplicate metrics MAY remain documented as supporting diagnostics.

#### Scenario: duplicate_ul docs reflect dead-anchor suppression semantics
- **WHEN** v3 triage lands
- **THEN** the canonical docs describe `loss/B_rollout_text/duplicate_ul` as the dead-anchor continuation suppression atom for Channel-B
- **AND** they do not describe it as same-desc duplicate-only cleanup.

## ADDED Requirements

### Requirement: Channel-B triage metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose the v3 triage bookkeeping separately from legacy duplicate diagnostics.

Normative count-like metrics:

- `stage2_ab/channel_b/triage/N_anchor_gt_backed`
- `stage2_ab/channel_b/triage/N_shielded_anchor`
- `stage2_ab/channel_b/triage/N_dead_anchor`
- `stage2_ab/channel_b/triage/N_dead_explorer`
- `stage2_ab/channel_b/triage/N_recovered_gt`

Normative numerator / denominator metrics:

- `stage2_ab/channel_b/triage/recovered_gt_num`
- `stage2_ab/channel_b/triage/recovered_gt_den`
- `stage2_ab/channel_b/triage/dead_anchor_num`
- `stage2_ab/channel_b/triage/dead_anchor_den`

Normative behavior:

- count-like metrics MUST aggregate additively across micro-steps,
- numerator / denominator metrics MUST aggregate additively across micro-steps,
- monitor-dump payloads SHOULD expose both rollout views plus the final triage decision for high-signal samples.

#### Scenario: Triage counts aggregate additively
- **WHEN** multiple micro-steps emit triage count metrics within one optimizer step
- **THEN** the finalized step metrics are additive totals rather than mean-diluted gauges.
