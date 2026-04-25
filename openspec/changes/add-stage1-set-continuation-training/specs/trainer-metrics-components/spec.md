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
