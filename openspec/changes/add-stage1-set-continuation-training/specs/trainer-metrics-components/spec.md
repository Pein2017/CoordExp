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

Required variant-specific families include:
- `loss/mp`, `loss/mp_diagnostic`, `loss/pem`,
  `loss/anti_close_start`, `loss/weak_schema_close`,
- `mp/logZ_scored_raw`, `mp/logZ_remaining_exact`,
  `mp/logZ_remaining_est`, `mp/logZ_estimator`,
- `mp/candidate_entry_tokens_*`, `mp/candidate_logprob_sum_*`,
  `mp/candidate_logprob_per_token_*`,
- `mp/candidate_coord_token_fraction_*`,
  `mp/candidate_logprob_per_coord_token_*`,
  `mp/candidate_logprob_per_noncoord_token_*`,
  `mp/valid_length_corr_samples`,
- `mp/branch_forwards_per_sample`, `mp/total_candidate_tokens_scored`,
  `mp/repeated_forward_token_ratio_vs_baseline`,
- `mp/branch_runtime_mode`, `mp/checkpointed_branch_forwards`,
  `mp/retained_graph_branch_forwards`,
- `mp/smart_batched_branch_forwards`, `mp/branch_batch_count`,
  `mp/branch_batch_rows_mean`, `mp/branch_batch_rows_max`,
  `mp/branch_batch_tokens_mean`, `mp/branch_batch_tokens_max`,
  `mp/branch_batch_padding_fraction`, `mp/branch_batch_scheduler`,
- `mp/ddp_candidate_padding_policy`,
  `mp/ddp_candidate_forward_local_count`,
  `mp/ddp_candidate_forward_max_count`,
  `mp/ddp_candidate_padding_forwards`,
- `mp/objective_fidelity_exact_samples`,
  `mp/objective_fidelity_approx_samples`, `mp/fallback_applied_samples`,
  `mp/fallback_reason_candidate_budget`,
  `mp/fallback_reason_token_budget`,
  `mp/fallback_reason_memory_budget`,
- `mp/prefix_encoding_cache_hits`, `mp/prefix_encoding_cache_misses`,
- `stop/p_close_start_when_remaining_exists`,
  `stop/p_continue_start_when_remaining_exists`,
  `stop/p_close_start_when_remaining_empty`,
  `stop/logp_close_sequence_when_remaining_empty`,
  `stop/p_final_schema_token_teacher_forced`.

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
