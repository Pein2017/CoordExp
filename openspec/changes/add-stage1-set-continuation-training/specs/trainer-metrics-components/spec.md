## ADDED Requirements

### Requirement: Stage-1 set-continuation metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose set-continuation mechanism diagnostics
with explicit scope, count-vs-gauge naming, and variant-specific documentation.

Normative behavior:
- ordinary Stage-1 metric parity remains unchanged when
  `custom.trainer_variant` is not `stage1_set_continuation`,
- set-continuation MP/PEM/structural-close metrics are variant-specific,
- count-like metrics MUST use `/N_`, `_count`, `_total`, `_sum`, `_num`, or
  `_den` naming or be explicitly registered as additive totals,
- gauge-like metrics MUST remain mean-like and MUST NOT masquerade as counts,
- logZ metrics MUST distinguish scored-candidate raw mass, exact remaining-set
  mass, and estimated remaining-set mass,
- structural-close metrics MUST distinguish close-start branchpoint probability
  from full closure-sequence log probability,
- candidate logprob metrics MUST include length and per-token diagnostics.
- branch-runtime metrics MUST distinguish retained-graph versus
  checkpoint/recompute execution versus smart batched exact execution when the
  set-continuation trainer is active,
- DDP synchronization metrics MUST distinguish max-count candidate padding from
  no-padding execution and record local versus cross-rank candidate pressure,
- objective-fidelity metrics MUST distinguish exact MP samples from samples
  handled by configured approximate fallback,
- fallback reason metrics MUST be emitted whenever the trainer changes from an
  exact candidate plan to approximate subsampling.
- smart branch batching metrics MUST expose scheduler choice, branch-batch row
  pressure, token volume, and padding waste so production runs can distinguish
  useful GPU utilization from padded waste.
- bidirectional gate token-count metrics MUST use `_count` suffixes so additive
  aggregation semantics are explicit.

Required variant-specific compact v2 families include:
- `loss/candidate_balanced`, `loss/schema_open`,
  `loss/json_structural`, `loss/anti_close_start`,
  `loss/weak_schema_close`, `loss/coord_gate`, `loss/text_gate`,
- `mp/num_prefix_objects`, `mp/num_remaining_objects`,
  `mp/num_candidates_scored`, `mp/candidate_tokens_scored_mean`,
  `mp/schema_open_tokens_scored_mean`,
  `mp/json_structural_tokens_scored_mean`,
  `mp/annotation_completeness_weight_mean`,
  `mp/final_close_weight_mean`, `mp/tail_positive_samples`,
  `mp/final_gt_object_scored_samples`,
- `mp/objective_fidelity_exact_samples`, `mp/fallback_applied_samples`,
  `mp/objective_contributing_samples`,
- `mp/selected_mode_empty_prefix`, `mp/selected_mode_full_prefix`,
- `stop/p_close_start_when_remaining_exists`,
  `stop/p_continue_start_when_remaining_exists`,
  `stop/p_close_start_when_remaining_empty`.
- `gate/coord_slot_coord_mass_mean`, `gate/text_slot_coord_mass_mean`,
  `gate/coord_tokens_count`, `gate/text_tokens_count`.

#### Scenario: Ordinary Stage-1 does not emit MP metrics
- **GIVEN** ordinary Stage-1 SFT
- **WHEN** metrics are reported
- **THEN** set-continuation `mp/*` and structural-close `stop/*` metrics are
  not emitted unless the set-continuation variant is active.

#### Scenario: Set-continuation counters aggregate safely
- **GIVEN** set-continuation metrics are emitted across multiple micro-steps
- **WHEN** optimizer-step metrics are finalized
- **THEN** additive totals and mean-like gauges are aggregated according to
  their registered semantics.

#### Scenario: Approximate fallback is visible in metrics
- **GIVEN** a set-continuation sample exceeds an authored train-forward budget
- **WHEN** the trainer falls back to uniform candidate subsampling
- **THEN** objective-fidelity metrics count the sample as approximate
- **AND** at least one fallback-reason metric identifies why exact scoring was
  not used.

#### Scenario: Smart branch batching telemetry is visible
- **GIVEN** the set-continuation trainer uses
  `train_forward.branch_runtime.mode=smart_batched_exact`
- **WHEN** candidate branches are scored
- **THEN** metrics include the number of branch batches, rows per branch batch,
  token volume, padding fraction, and scheduler code
- **AND** objective-fidelity metrics continue to report exact selected-candidate
  execution unless an explicit approximate fallback changes the candidate set.

#### Scenario: Bidirectional gate metrics are compact and scoped
- **GIVEN** bidirectional token gating is enabled for set-continuation training
- **WHEN** metrics are reported
- **THEN** logs include finite `loss/coord_gate` and `loss/text_gate` values
- **AND** logs include coord/text gate token counts and coord-mass gauges
- **AND** ordinary Stage-1 SFT does not emit those gate keys.
