# stage2-ab-training Specification (Delta)

## ADDED Requirements

### Requirement: Stage-2 AB objective weights are pipeline-only (no flat objective knobs)
When `custom.trainer_variant: stage2_two_channel`, the Stage-2 AB objective MUST be fully determined by the declared module pipeline under:
- `stage2_ab.pipeline.objective[]` and `stage2_ab.pipeline.diagnostics[]`.

Normative behavior:
- `stage2_ab.pipeline` MUST be present (no implicit default objective manifest).
- Flat objective knobs are **removed** and MUST be rejected when present, including (non-exhaustive):
  - `stage2_ab.desc_ce_weight`
  - `stage2_ab.fmt_struct_ce_weight`
  - `stage2_ab.bbox_smoothl1_weight`
  - `stage2_ab.bbox_ciou_weight`
  - `stage2_ab.coord_ce_weight`
  - `stage2_ab.coord_el1_weight`
  - `stage2_ab.coord_ehuber_weight`
  - `stage2_ab.coord_entropy_weight`
  - `stage2_ab.coord_gate_weight`
  - `stage2_ab.text_gate_weight`
- Legacy aux-loss config surfaces MUST be rejected for Stage-2 AB, including `custom.coord_soft_ce_w1.*`.

#### Scenario: Missing pipeline fails fast
- **WHEN** a Stage-2 AB config sets `custom.trainer_variant: stage2_two_channel`
- **AND** `stage2_ab.pipeline` is absent
- **THEN** config loading fails fast before trainer init
- **AND** the error indicates `stage2_ab.pipeline` is required.

#### Scenario: Flat objective knob fails fast
- **WHEN** a Stage-2 AB config defines `stage2_ab.desc_ce_weight`
- **THEN** config loading fails fast
- **AND** the error indicates loss weights must be expressed via `stage2_ab.pipeline.objective[*].config`.

### Requirement: Stage-2 AB pipeline specs are explicit and complete (no implicit defaults)
Stage-2 AB pipeline module specs MUST be authored with explicit fields and complete module configs to prevent silent drift from default injection.

Normative behavior:
- Each entry in `stage2_ab.pipeline.objective[]` and `stage2_ab.pipeline.diagnostics[]` MUST include:
  - `name`, `enabled`, `weight`, `channels`, `config`.
- `channels` MUST be explicitly authored as a subset of `{A,B}`.
- `config` MUST include exactly the allowlisted keys for the referenced module:
  - missing required keys MUST fail fast (no implicit defaults),
  - unknown keys MUST fail fast (no escape-hatch aliases).

#### Scenario: Missing module spec field fails fast
- **WHEN** a Stage-2 AB config omits `stage2_ab.pipeline.objective[i].channels`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the missing required field and its full path.

### Requirement: Stage-2 AB module configs are strict and canonical (no aliases)
Stage-2 AB pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

Normative behavior:
- `bbox_geo.config` MUST accept only:
  - `smoothl1_weight`
  - `ciou_weight`
- `coord_reg.config` MUST accept only canonical keys, including:
  - `coord_ce_weight`
  - `soft_ce_weight`
  - `w1_weight`
  - `coord_gate_weight`
  - `text_gate_weight`
  - `temperature`
  - `target_sigma`
  - `target_truncate`
- Legacy alias keys (e.g., `bbox_smoothl1_weight`, `bbox_ciou_weight`, `coord_soft_ce_weight`, `coord_w1_weight`) MUST be rejected.

#### Scenario: Alias key in module config fails fast
- **WHEN** `stage2_ab.pipeline.objective[*].name=bbox_geo`
- **AND** the module config contains `bbox_smoothl1_weight`
- **THEN** configuration parsing fails fast
- **AND** the error indicates `smoothl1_weight` is the only accepted key.

### Requirement: Stage-2 AB supports text_gate via coord_reg module config
Stage-2 AB MUST support `text_gate` as part of `coord_reg` with a typed weight:
- `stage2_ab.pipeline.objective[*].config.text_gate_weight`

Normative behavior:
- `text_gate_weight > 0` MUST introduce a non-zero `text_gate` contribution when coord-vocab mass appears at supervised text positions (subject to registry masking).

#### Scenario: Non-zero text_gate_weight is effective
- **WHEN** `coord_reg.config.text_gate_weight > 0`
- **AND** the model places substantial coord-vocab probability mass at supervised `type=struct|desc` positions
- **THEN** the emitted `text_gate` objective atom increases relative to the same run with `text_gate_weight = 0`:
  - Channel-A: `loss/A2_coord/text_gate`
  - Channel-B: `loss/B_coord/text_gate`
- **AND** the increase is attributable to the `text_gate` sub-term inside `coord_reg`.

### Requirement: Coord diagnostics are attributed to A1 vs A2 logits in Channel-A
Stage-2 AB SHOULD provide coord-distribution monitors that let operators compare the GT-anchor logits (`A1`) versus the final softctx logits (`A2`) on the same GT coord-token positions.

Normative behavior:
- When `coord_diag` diagnostics module is enabled (non-zero effective weight for the current channel), the trainer MUST emit coord distribution monitor keys with explicit forward provenance:
  - `coord_diag/A1/*`: computed from the Channel-A A1 logits (`logits_a1`, `it==0`).
  - `coord_diag/A2/*`: computed from the Channel-A final softctx logits (`it==n_softctx_iter-1`), emitted only when `n_softctx_iter > 1`.
- The monitor set SHOULD include at least:
  - `coord_diag/<prov>/acc_top5`
  - `coord_diag/<prov>/p_gt_mean`
  - `coord_diag/<prov>/expected_bin_mae`
- These diagnostics MUST NOT affect the training objective (they are monitors only).
- The trainer MUST NOT emit ambiguous bare `coord_diag/*` keys for these monitors in Stage-2 AB logs.

## MODIFIED Requirements

### Requirement: Hybrid objective preserves JSON structure CE and adds bbox geometry losses
The Stage-2 AB trainer MUST compute a hybrid objective consistent with the unified teacher-forcing loss registry.

Channel-A:
- **Token CE anchor at GT context**:
  - `struct_ce` and `desc_ce` MUST be computed from the teacher-forced logits of the GT anchor forward (context `gt`).
  - `coord` tokens MUST NOT contribute to CE (they are masked out), to avoid double-supervision.
- **Optional self-context format stabilizer**:
  - The trainer MAY compute an additional small-weight self-context CE stabilizer from `context=self_context` logits restricted to token types `struct|eos` only.
  - Self-context `desc` tokens MUST have CE weight `0` (token_ce is struct/EOS-only in self_context).
  - Self-context `coord` tokens MUST be excluded from token CE (coord-slot supervision is expressed via coord_reg sub-terms).
- **Geometry + coord regularizers from self-context logits**:
  - `geo` and `coord_reg` MUST be computed from the final-iteration logits `z^(n_softctx_iter-1)` (context `self_context`).

Channel-B (rollout context):
- **FP-neutral + EOS-enforced supervision**:
  - FP spans MUST be excluded from CE, geometry, and coord_reg terms.
  - Matched prefix objects MUST receive struct-only CE (`desc_ce` disabled).
  - FN injected objects MUST receive struct+desc CE by default (desc weight configurable).
  - The top-level JSON closing brace `}` and `<|im_end|>` MUST remain supervised (EOS-enforced closure).
- **Matched/FN-only geometry**:
  - Geometry losses MUST be computed only for matched and FN-injected objects.
  - Unmatched predicted objects (FP) MUST NOT receive geometric gradients.
- **FN append always**:
  - FN objects MUST be appended so they are supervised even when missing from the rollout.
  - If strict validation yields `N_valid_pred == 0`, the trainer MUST treat all GT objects as FN and append them.

Configurable supervision weights:
- Loss weights MUST be expressed via `stage2_ab.pipeline.objective[*].config`:
  - `token_ce.config.desc_ce_weight` (Channel-A tail desc)
  - `token_ce.config.self_context_struct_ce_weight` (Channel-A optional stabilizer)
  - `token_ce.config.rollout_fn_desc_weight` (Channel-B FN desc weight)
  - `token_ce.config.rollout_matched_prefix_struct_weight` (Channel-B matched-prefix struct weight)
  - `token_ce.config.rollout_drop_invalid_struct_ce_multiplier` (optional upweight on rollout invalid drops)
  - `bbox_geo.config.smoothl1_weight`, `bbox_geo.config.ciou_weight`
  - `coord_reg.config.coord_ce_weight` (optional hard coord-token CE atom `coord_token_ce`)
  - `coord_reg.config.coord_gate_weight`, `coord_reg.config.text_gate_weight` (and other coord_reg sub-terms)

Geometry decoding and stability:
- Bbox geometry losses MUST decode coordinates from coord-token distributions via the configured decode mode (expectation or ST) and MUST operate in normalized `[0, 1]`.
- The trainer MUST canonicalize predicted boxes before IoU-based terms and MUST fence NaN/Inf values.

#### Scenario: Channel-A desc CE is anchored to GT logits
- **WHEN** Channel-A executes with `n_softctx_iter >= 2`
- **THEN** `loss/A1_text/{struct_ce,desc_ce}` are computed from `context=gt` (A1) logits
- **AND** `loss/A2_text/struct_ce` is computed from `context=self_context` (final-iteration A2) logits when `token_ce.config.self_context_struct_ce_weight > 0`
- **AND** `loss/A2_coord/{bbox_smoothl1,bbox_ciou}` are computed from `context=self_context` (final-iteration A2) logits (when enabled).

#### Scenario: Channel-B is FP-neutral and EOS-enforced
- **WHEN** Channel-B constructs rollout-context masks for a batch containing matched, FP, and FN objects
- **THEN** FP spans do not contribute to `struct_ce`, `desc_ce`, `geo`, or `coord_reg`
- **AND** the top-level `}` and `<|im_end|>` remain supervised under `struct_ce`.

#### Scenario: Matched-prefix desc is excluded by default
- **WHEN** Channel-B includes a matched prefix object with a `desc` span
- **THEN** those matched-prefix `desc` tokens do not contribute to `loss/desc_ce`
- **AND** FN-injected `desc` tokens contribute according to `token_ce.config.rollout_fn_desc_weight`.
