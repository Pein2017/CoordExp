# trainer-metrics-components Specification (Delta)

## ADDED Requirements

### Requirement: Objective metrics emit canonical provenance keys only (atomic objective atoms; no raw component keys)
For registry-defined objective modules, trainers MUST emit only **atomic objective contributions** under canonical `loss/<provenance>/<atom>` keys and MUST NOT emit raw component loss keys by default.

Definitions:
- An "objective atom" is a post-weighting contribution used in the trainer's total loss.
- For multi-term modules (notably bbox geometry + coord regularization), objective atoms are emitted per sub-term (no pre-summed aggregates).
- "Provenance" encodes which forward/context produced the objective for Stage-2 AB.

Normative behavior:
- Stage-2 AB and rollout-aligned trainers MUST emit only the following objective keys (minimum set), and only when the effective weight is non-zero:
  - Channel-A:
    - `loss/A1_text/{struct_ce,desc_ce}` (GT-anchor forward; token CE objective atoms)
    - `loss/A2_text/struct_ce` (final self-context forward; optional struct/EOS CE stabilizer atom)
    - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}` (final self-context forward; geometry objective atoms)
    - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` (final self-context forward; coord_reg objective atoms)
  - Channel-B (rollout context):
    - `loss/B_rollout_text/{struct_ce,desc_ce}` (rollout-context forward; token CE objective atoms)
    - `loss/B_coord/{bbox_smoothl1,bbox_ciou}` (rollout-context forward; geometry objective atoms)
    - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` (rollout-context forward; coord_reg objective atoms)
- Raw/duplicate loss keys MUST NOT be emitted by default (non-exhaustive):
  - Per-component registry metrics: `loss/token_ce`, `loss/struct_ce`, `loss/desc_ce`, `loss/geo`, `loss/coord_reg`, `loss/coord_gate`, `loss/text_gate`
  - Legacy objective suffixes: `loss/token_ce_obj`, `loss/bbox_geo_obj`, `loss/coord_reg_obj`
  - Trainer-specific aliases: `loss/ce`, `loss/coord`, `loss/coord_prefix`, `loss/coord_tail`

#### Scenario: Canonical-only objective keys
- **WHEN** a Stage-2 or rollout-aligned training step emits objective metrics
- **THEN** emitted keys include only canonical `loss/<provenance>/<module>` keys for registry-defined objective modules
- **AND** raw component loss keys and legacy aliases are absent.

### Requirement: Coord distribution diagnostics are provenance-split in Stage-2 two-channel
When Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) emits coord-distribution diagnostics, it MUST attribute them to the forward surface that produced the logits used for the coord-vocab slice.

Rationale:
- Channel-A runs two teacher-forced forwards (`A1` and `A2`) when `n_softctx_iter >= 2`.
- Stage-1 runs only one forward and reports coord distribution monitors without forward provenance.
- Provenance-splitting makes Stage-2 comparable to Stage-1 and makes self-context drift visible.

Normative behavior:
- Stage-2 two-channel MUST emit coord-vocab distribution monitors under:
  - `coord_diag/A1/*`: computed from Channel-A **A1** logits (GT anchor forward; `it==0`).
  - `coord_diag/A2/*`: computed from Channel-A **A2** logits (final softctx forward; `it==n_softctx_iter-1`), emitted only when `n_softctx_iter > 1`.
  - `coord_diag/B/*`: computed from Channel-B logits (rollout-context forward).
- The set of per-provenance keys SHOULD include (non-exhaustive):
  - `coord_diag/<prov>/coord_tokens_total`
  - `coord_diag/<prov>/{acc_top5,p_gt_mean,margin_mean,expected_bin_mae,expected_bin_abs_err_p90,coord_vocab_mass_mean}`
- Stage-2 two-channel MUST NOT emit bare `coord_diag/*` keys for these monitors (ambiguous provenance is disallowed).

### Requirement: Rollout-only metrics are sparse-emitted
Trainers MUST NOT emit rollout-specific monitor metrics on steps where no rollout was executed.

Normative behavior:
- “Rollout executed” MUST be determined by runtime evidence (e.g., non-zero rollout generation time, non-zero parsed rollout length, or equivalent authoritative signal), not merely by decode configuration.
- For Stage-2 AB Channel-B, `stage2/raw_rollouts > 0` SHOULD be treated as the authoritative runtime signal that a rollout was executed for the step.
- When rollout was not executed, the trainer MUST omit (not emit with `0.0`) rollout-specific keys, including (non-exhaustive):
  - `rollout/precision`, `rollout/recall`, `rollout/f1`
  - `rollout/*` parse/gating/length/coverage diagnostics
  - `time/rollout_generate_s`, `time/rollout_parse_match_s`, `time/rollout_teacher_encode_s`

#### Scenario: A-only Stage-2 does not spam rollout zeros
- **WHEN** Stage-2 AB runs with `stage2_ab.schedule.b_ratio = 0.0`
- **AND** no rollout is executed for the current optimizer step
- **THEN** the emitted training log line contains no `rollout/*` scalar keys (they are absent rather than constant zeros).

### Requirement: Zero-valued timing keys are sparse-emitted
To reduce constant-noise monitors, trainers SHOULD omit timing keys that are identically `0.0` for the current run.

Normative behavior:
- `time/mask_build_s` MUST be omitted when it is not measured by the current trainer (`0.0` placeholder values are disallowed).

## REMOVED Requirements

### Requirement: Metric key schema remains backward compatible during refactor
**Reason**: This change intentionally breaks legacy loss metric aliases to eliminate drift and enforce a single canonical loss registry contract.

**Migration**: Update dashboards/scripts to consume objective-only provenance keys (atomic objective atoms), for example:
- Channel-A:
  - `loss/A1_text/{struct_ce,desc_ce}`
  - `loss/A2_text/struct_ce`
  - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}`
  - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
- Channel-B:
  - `loss/B_rollout_text/{struct_ce,desc_ce}`
  - `loss/B_coord/{bbox_smoothl1,bbox_ciou}`
  - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`

Stop relying on any raw per-component `loss/*` keys or legacy alias keys.
