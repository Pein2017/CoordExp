# stage2-ab-training Spec Delta

This is a delta spec for change `align-stage2-full-idea-contracts-2026-02-09`.

## MODIFIED Requirements

### Requirement: Hybrid objective preserves JSON structure CE and adds bbox geometry losses
The Stage-2 AB trainer MUST compute a hybrid objective with:

Channel-A:
- **Token CE anchor at A1**:
  - CE on non-coord tokens MUST be computed from the teacher-forced logits of the first forward (`z^(0)`; GT context).
  - Coord tokens MUST NOT contribute to CE (they are masked out), to avoid double-supervision.
- **Geometry + distribution regularizers from final softctx logits**:
  - Geometry losses and any distribution-level losses MUST be computed from the final-iteration logits `z^(n_softctx_iter-1)`.

Channel-B:
- **Unified one-pass target over rollout prefix + FN injection**:
  - The trainer MUST construct one teacher-forced assistant target by taking the retained rollout prefix and injecting FN objects inside the top-level JSON object before final closure.
  - FN injection MUST locate the outermost top-level JSON close brace `}` using brace-depth scan (or an equivalent deterministic parser), not by "last `}` token" heuristics.
  - If the retained prefix body already contains at least one object entry, the trainer MUST insert a comma before the first FN entry; otherwise it MUST inject FN entries without a leading comma.
- **CE masking policy (hard rules)**:
  - Matched prefix objects: structure CE MUST be enabled; desc CE MUST be disabled; coord CE MUST be disabled.
  - FP prefix objects: structure/desc/coord CE MUST all be masked out.
  - FN-injected objects: structure CE MUST be enabled; desc CE MUST be enabled; coord CE MUST be disabled.
- **Geometry policy**:
  - Geometry losses MUST be computed for both:
    - matched prefix objects (`L_geo_matched`), and
    - FN-injected objects (`L_geo_FN`).
  - FP objects MUST NOT contribute geometry loss.
- **Deterministic key allocation**:
  - FN key assignment MUST start from `start_id = max_object_index_in_prefix + 1` (or `1` when none).
  - `max_object_index_in_prefix` MUST be computed from all retained prefix keys matching `object_{n}`, including keys from entries later dropped by strict validation/matching; malformed keys are ignored.
- **Stop/closure CE is supervised**:
  - Channel-B CE MUST supervise stop/continue via token-level supervision on:
    - the same outermost top-level `}` token used as the FN injection anchor, and
    - `<|im_end|>`.
  - Stop-neutral masking MUST NOT be applied in Stage-2 AB Channel-B.
  - This requirement is intentionally aligned with
    `openspec/changes/remove-stage2-ab-stop-neutral/specs/stage2-ab-training/spec.md`.

Configurable desc supervision (both channels):
- `stage2_ab.desc_ce_weight` MUST be a float `>= 0` and applies to desc value tokens by default.
- Channel-B MAY additionally provide `stage2_ab.channel_b.desc_ce_weight_matched` as the weight for matched-object desc tokens (distinct from FN-appended desc tokens).

Bbox geometry losses (both channels) are computed from coord distributions:
- The trainer MUST decode coordinates from coord-token distributions via CoordExp expectation decoding (not argmax):
  - Let bins `k ∈ {0..999}` correspond to the coord-token sub-vocabulary.
  - Let `p(k)` be the softmax distribution over these 1000 bins for a coord slot, taken from the standard causal shift:
    - For a coord token at input position `p`, `p(k)` MUST be computed from logits at position `p-1` (consistent with CE).
  - The decoded normalized coordinate MUST be: `c_hat = Σ_k p(k) * (k/999)` in `[0, 1]`.
- The trainer MUST compute bbox losses in normalized coordinate space `[0, 1]`:
  - GT bbox ints in `[0, 999]` MUST be converted to floats by dividing by `999`.
  - Predicted bbox coords MUST be the decoded normalized floats from `c_hat` above.
- Geometry loss MUST use logits from:
  - the final iteration `z^(n_softctx_iter-1)` in Channel-A, and
  - the Channel-B teacher-forced logits under the unified rollout+FN sequence.

Loss form (normative):
- The trainer MUST use SmoothL1 (Huber) + CIoU as the bbox regression terms.
- The trainer MUST NOT use GIoU in Stage-2 AB.

Numerical stability (normative):
- The trainer MUST canonicalize predicted boxes before CIoU:
  - `(x1,x2) := (min(x1,x2), max(x1,x2))`, `(y1,y2) := (min(y1,y2), max(y1,y2))`.
- The trainer MUST ensure the geometry losses do not produce NaNs/Infs, including early training when predictions are degenerate.

Efficiency rule (normative):
- If Channel-B has no valid matched pairs for a sample/batch, the trainer MUST still run the unified one-pass Channel-B forward (with FN injection and configured CE masks) and MUST NOT require a separate geo-only pass.

#### Scenario: Channel-B geometry includes matched and FN but excludes FP
- **GIVEN** Channel-B where matching yields non-empty matched, FP, and FN sets
- **WHEN** Channel-B losses are computed
- **THEN** geometry loss is accumulated for matched and FN-injected objects
- **AND** FP objects contribute zero geometry loss.

#### Scenario: Start key avoids collision even when highest key object is invalid
- **GIVEN** retained rollout prefix contains keys `object_2` (valid) and `object_7` (invalid and dropped by strict validation)
- **WHEN** FN entries are injected
- **THEN** `max_object_index_in_prefix` is `7`
- **AND** FN key assignment starts at `object_8`.

#### Scenario: Closure-supervision brace target is the same brace used for injection
- **GIVEN** Channel-B injects FN entries before the outermost close brace resolved by brace-depth scan
- **WHEN** CE masks are produced
- **THEN** that same outermost close brace token position remains CE-supervised
- **AND** `<|im_end|>` remains CE-supervised.

#### Scenario: CE masking follows matched/FP/FN policy
- **GIVEN** Channel-B contains one matched object, one FP object, and one FN-injected object
- **WHEN** CE weights are materialized
- **THEN** matched structure tokens are supervised while matched desc tokens are masked
- **AND** FP structure/desc/coord tokens are all masked
- **AND** FN-injected structure and desc tokens are supervised.

## ADDED Requirements

### Requirement: Unified Channel-B is the default contract and reordered_gt_sft is legacy opt-in
For Stage-2 AB, Unified Channel-B semantics SHALL be the normative default behavior.

Legacy `reordered_gt_sft` behavior SHALL be treated as experimental/ablation-only:
- it MUST NOT be the default path,
- it MAY be enabled only via explicit opt-in configuration,
- it MUST be documented as legacy behavior when enabled.

#### Scenario: Default Stage-2 AB run uses unified Channel-B semantics
- **GIVEN** a Stage-2 AB config that does not explicitly opt into legacy `reordered_gt_sft`
- **WHEN** Channel-B behavior is materialized
- **THEN** the trainer uses unified rollout-prefix + FN-injection semantics
- **AND** legacy `reordered_gt_sft` behavior is not selected by default.

#### Scenario: Legacy reordered_gt_sft requires explicit opt-in
- **GIVEN** a Stage-2 AB run where legacy `reordered_gt_sft` mode is enabled
- **WHEN** configuration is validated and training starts
- **THEN** the mode is treated as explicit ablation/legacy behavior
- **AND** the run does not claim unified-default Channel-B semantics.
