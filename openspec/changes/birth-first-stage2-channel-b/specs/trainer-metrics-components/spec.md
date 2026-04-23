## ADDED Requirements

### Requirement: Channel-B birth-first metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose birth-first diagnostics separately from duplicate-collapse diagnostics so decision runs can distinguish object-birth learning from anti-repeat behavior.

Normative mean-like rollout-text atom:
- `loss/B_rollout_text/continue_over_eos_margin`

Normative triage counters:
- `train/triage/support_positive_shielded_count`
- `train/triage/neutral_shielded_count`
- `train/triage/recovered_ground_truth_count`
- `train/triage/recovered_ground_truth_rate_num`
- `train/triage/recovered_ground_truth_rate_den`

Normative policy counters:
- `stage2_ab/channel_b/birth_first/N_continue_over_eos_boundaries`
- `stage2_ab/channel_b/birth_first/N_continue_over_eos_skipped_no_recovered_boundary`

Normative behavior:
- Count-like birth-first metrics MUST use `/N_`, `_count`, `_total`, `_sum`, `_num`, or `_den` naming so optimizer-step aggregation treats them as additive totals.
- `loss/B_rollout_text/continue_over_eos_margin` MUST remain mean-like across eligible recovered boundaries and MUST NOT be emitted as a raw sum.
- `train/triage/support_positive_shielded_count` and `train/triage/neutral_shielded_count` MUST be derived from the same retained-anchor partition used to build the final clean target.
- `train/triage/recovered_ground_truth_count` MUST count recovered GT objects that stay on the weighted FN path in the current step.
- `train/triage/recovered_ground_truth_rate_num` and `train/triage/recovered_ground_truth_rate_den` MUST remain the additive numerator/denominator surfaces for comparing recovered-GT opportunity rates across runs.
- `stage2_ab/channel_b/birth_first/N_continue_over_eos_boundaries` MUST count eligible recovered boundaries that actually emitted a continue-over-EOS term.
- `stage2_ab/channel_b/birth_first/N_continue_over_eos_skipped_no_recovered_boundary` MUST remain an additive policy counter describing the no-target case rather than a mean-like rollout gauge.

#### Scenario: Birth-first counters aggregate additively across micro-steps
- **WHEN** `train/triage/*_count` or `stage2_ab/channel_b/birth_first/N_*` metrics are emitted from multiple micro-steps in one optimizer step
- **THEN** the finalized step metric is the additive total
- **AND** the result is not diluted by mean-style aggregation.

#### Scenario: Continue-over-EOS atom remains mean-like across micro-steps
- **WHEN** `loss/B_rollout_text/continue_over_eos_margin` is emitted from multiple micro-steps in one optimizer step
- **THEN** the finalized step metric is the weighted mean over eligible recovered boundaries
- **AND** it is comparable across different packing and grad-accum settings.

#### Scenario: Support-positive and neutral shield counts share the runtime partition
- **WHEN** Channel-B birth-first metrics are emitted for one optimizer step
- **THEN** `train/triage/support_positive_shielded_count` and `train/triage/neutral_shielded_count` are derived from the same retained-anchor partition used by the target builder
- **AND** operators do not need a second diagnostic definition to interpret the logged counts.
