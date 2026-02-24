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
- **THEN** the run emits a positive `loss/text_gate`
- **AND** the total `loss/coord_reg` increases relative to the same run with `text_gate_weight = 0`.

## MODIFIED Requirements

### Requirement: Hybrid objective preserves JSON structure CE and adds bbox geometry losses
The Stage-2 AB trainer MUST compute a hybrid objective consistent with the unified teacher-forcing loss registry.

Channel-A:
- **Token CE anchor at GT context**:
  - `struct_ce` and `desc_ce` MUST be computed from the teacher-forced logits of the GT anchor forward (context `gt`).
  - `coord` tokens MUST NOT contribute to CE (they are masked out), to avoid double-supervision.
- **Optional self-context format stabilizer**:
  - The trainer MAY compute an additional small-weight self-context CE stabilizer from `context=self_context` logits restricted to token types `struct|eos` only.
  - Self-context `desc` tokens MUST have CE weight `0`.
  - Self-context `coord` tokens MUST have CE weight `0`.
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
  - `coord_reg.config.coord_gate_weight`, `coord_reg.config.text_gate_weight` (and other coord_reg sub-terms)

Geometry decoding and stability:
- Bbox geometry losses MUST decode coordinates from coord-token distributions via the configured decode mode (expectation or ST) and MUST operate in normalized `[0, 1]`.
- The trainer MUST canonicalize predicted boxes before IoU-based terms and MUST fence NaN/Inf values.

#### Scenario: Channel-A desc CE is anchored to GT logits
- **WHEN** Channel-A executes with `n_softctx_iter >= 2`
- **THEN** `loss/desc_ce` is computed from `context=gt` logits
- **AND** `geo` is computed from `context=self_context` logits.

#### Scenario: Channel-B is FP-neutral and EOS-enforced
- **WHEN** Channel-B constructs rollout-context masks for a batch containing matched, FP, and FN objects
- **THEN** FP spans do not contribute to `struct_ce`, `desc_ce`, `geo`, or `coord_reg`
- **AND** the top-level `}` and `<|im_end|>` remain supervised under `struct_ce`.

#### Scenario: Matched-prefix desc is excluded by default
- **WHEN** Channel-B includes a matched prefix object with a `desc` span
- **THEN** those matched-prefix `desc` tokens do not contribute to `loss/desc_ce`
- **AND** FN-injected `desc` tokens contribute according to `token_ce.config.rollout_fn_desc_weight`.

