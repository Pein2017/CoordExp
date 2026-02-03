# stage2-ab-training Specification

## MODIFIED Requirements

### Requirement: Stage-2 AB trainer variant is selectable via YAML
When training config sets `custom.trainer_variant: stage2_ab_training`, the system SHALL use the Stage-2 AB trainer implementation.

The trainer MUST be configurable via YAML and MUST NOT require new CLI flags.

Canonical config location (typed):
- Stage-2 AB knobs MUST be expressed under the top-level `stage2_ab` mapping (parallel to `training` and `custom`).
- Unknown keys under `stage2_ab` MUST fail fast with actionable guidance (to avoid silent drift from typos).

#### Scenario: Selecting the trainer variant with typed `stage2_ab`
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** a top-level `stage2_ab` mapping is provided
- **WHEN** training starts
- **THEN** the Stage-2 AB trainer is constructed and used for training
- **AND** the trainer reads Stage-2 AB knobs from `stage2_ab`.

#### Scenario: Unknown stage2_ab keys fail fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** a top-level `stage2_ab` mapping contains an unknown key (e.g., a typo)
- **WHEN** training starts
- **THEN** configuration parsing fails fast with guidance to fix/remove the unknown key.

### Requirement: Channel selection is deterministic and step-driven
The trainer SHALL choose between Channel-A and Channel-B **deterministically** as a function of (`global_step`, `stage2_ab.schedule.b_ratio`), with a deterministic override to Channel-B when rollout-buffer reuse is active.

Definition (normative):
- `global_step` MUST refer to the **optimizer-step** counter (post gradient-accumulation), i.e. the value that increments exactly once per optimizer update.
- The selected channel for a given `global_step` MUST remain fixed for the entire accumulation window (all micro-batches that contribute to that optimizer update).
- On resume from checkpoint, the schedule MUST continue from the restored `global_step` (no re-randomization).

Schedule definition (normative minimum):
- `stage2_ab.schedule.b_ratio` MUST be a float in `[0.0, 1.0]`.
- `stage2_ab.schedule.b_ratio` MUST be explicitly provided (no implicit default).
- Let optimizer step be `s` (0-indexed).
- The trainer MUST select Channel-B at step `s` iff:
  - `floor((s+1) * b_ratio) > floor(s * b_ratio)`.
  - Otherwise it MUST select Channel-A.

Special cases:
- If `b_ratio == 0.0`, the trainer MUST always select Channel-A (unless buffer reuse overrides to B).
- If `b_ratio == 1.0`, the trainer MUST always select Channel-B.

Legacy schedule handling (normative):
- The legacy list-based schedule knob `schedule.pattern` is not supported.
- If a config provides `stage2_ab.schedule.pattern`, configuration parsing MUST fail fast with guidance to migrate to `stage2_ab.schedule.b_ratio`.

Buffer reuse override (normative):
- Stage-2 AB MAY reuse a buffered Channel-B batch across multiple optimizer steps as an optimization when rollout buffering is enabled under `custom.extra.rollout_matching.rollout_buffer`.
- If rollout buffering is enabled with `custom.extra.rollout_matching.rollout_buffer.m_steps > 1`, and the trainer is reusing a buffered Channel-B batch for additional optimizer steps (“M-steps” reuse), then:
  - the trainer MUST override the channel selection to Channel-B for those reuse steps, regardless of `b_ratio`,
  - and the override MUST be deterministic as a function of the buffer’s reuse state and `global_step`.
  - the trainer SHOULD expose a stable boolean metric/log key indicating reuse is active for the step (e.g., `stage2_ab/channel_b/is_reuse_step`).

#### Scenario: b_ratio=0.5 alternates deterministically by optimizer step
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.5`
- **WHEN** the trainer selects channels for `global_step` `s = 0, 1, 2, 3`
- **THEN** it selects Channel-A at steps `0` and `2`
- **AND** it selects Channel-B at steps `1` and `3`.

#### Scenario: b_ratio edge cases are deterministic
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.0`
- **WHEN** the trainer selects channels for any `global_step`
- **THEN** it always selects Channel-A (unless buffer reuse overrides to B).

- **GIVEN** `stage2_ab.schedule.b_ratio: 1.0`
- **WHEN** the trainer selects channels for any `global_step`
- **THEN** it always selects Channel-B.

#### Scenario: Channel selection continues across checkpoint resume
- **GIVEN** a run that has completed optimizer step `global_step = s`
- **WHEN** training resumes from a checkpoint that restores `global_step = s`
- **THEN** the channel selected for step `s` is identical to the pre-resume selection for step `s`.

#### Scenario: stage2_ab.schedule.pattern fails fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** `stage2_ab.schedule.pattern: ["A","B"]` is provided
- **WHEN** config is parsed/materialized
- **THEN** it fails fast with guidance to use `stage2_ab.schedule.b_ratio`.

#### Scenario: Missing b_ratio fails fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** `stage2_ab.schedule.b_ratio` is not provided
- **WHEN** config is parsed/materialized
- **THEN** it fails fast with guidance to set `stage2_ab.schedule.b_ratio`.

#### Scenario: Rollout buffer reuse forces Channel-B
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.0` (Channel-A would be selected)
- **AND** rollout buffering is enabled with `custom.extra.rollout_matching.rollout_buffer.m_steps > 1`
- **AND** the trainer is in a reuse step (reusing a buffered Channel-B batch)
- **WHEN** the trainer selects the channel for that optimizer step
- **THEN** it selects Channel-B (buffer reuse override).

### Requirement: Bbox-only v1 guardrails are enforced
The Stage-2 AB trainer MUST enforce bbox-only v1 guardrails on both GT objects and predicted rollout objects.

Stage-2 AB is bbox-only v1:
- GT objects MUST contain exactly one geometry field and it MUST be `bbox_2d` with exactly 4 coordinates.
  - If any GT object contains `poly` or any other geometry key, the trainer MUST fail fast with an error.
  - If any GT object contains malformed `bbox_2d` (wrong length, non-decodable values, or out-of-range values outside `[0, 999]` in norm1000 space), the trainer MUST fail fast.
    - Coercion contract (normative): values MUST be parsed as `int(round(float(x)))`; any value that cannot be coerced or lands outside `[0, 999]` is invalid.
  - If any GT bbox is invalid in ordering (`x2 < x1` or `y2 < y1`), the trainer MUST fail fast.

Predicted rollout objects MUST be filtered deterministically (no repair, no conversion):
- Instances whose geometry is not `bbox_2d` (including `poly`) MUST be dropped.
- Instances with missing/empty `desc` MUST be dropped.
- Instances with invalid bbox arity or invalid coord values MUST be dropped.

Diagnostics (normative):
- The trainer MUST expose strict-drop counters:
  - `N_valid_pred` (valid predicted instances kept),
  - `N_drop_invalid` (instances dropped by strict validation),
  - and reason buckets at minimum including: `missing_desc`, `missing_geom`, `wrong_arity`, `non_coord_token`, `poly_unsupported`, `unknown_geom`, `bbox_invalid`, `key_invalid`.
- The trainer MUST emit these counters as stable training metrics/log keys at least once per optimizer step that runs Channel-B.

#### Scenario: Drop diagnostics include valid and invalid counts
- **GIVEN** a Channel-B rollout containing a mix of valid bbox instances and invalid instances
- **WHEN** the trainer applies strict validation
- **THEN** invalid instances are dropped deterministically (no repair)
- **AND** the trainer exposes `N_valid_pred` and `N_drop_invalid` with at least one reason bucket incremented.

### Requirement: Channel-A performs iterative soft self-context via N× full-forwards (no rollout)
Channel-A MUST implement iterative soft self-context using `stage2_ab.n_softctx_iter` full forward passes:
- `stage2_ab.n_softctx_iter` MUST be an integer `>= 1`.
- The iteration index `m` ranges over `m = 0..n_softctx_iter-1`.
- For `n_softctx_iter = 1`, Channel-A MUST reduce to a single teacher-forced forward (pure TF baseline).
- For `n_softctx_iter > 1`, Channel-A MUST:
  - Run a teacher-forced forward to obtain logits for coord slots.
  - Construct coord-slot **soft embeddings** as the expectation over the coord-token sub-vocabulary.
  - Update coord-slot embeddings and re-run a full forward, repeating until `m = n_softctx_iter-1`.

The trainer MUST use the **final-iteration** logits `z^(n_softctx_iter-1)` for geometry decoding and loss computation.

Gradient semantics (normative):
- The default gradient mode MUST be fully unrolled:
  - `stage2_ab.softctx_grad_mode: "unroll"` MUST record gradients through all softctx iterations
  - and MUST NOT detach the expected coord embeddings used to update coord-slot inputs.
- An explicit EM-style fallback MAY be provided for ablations:
  - `stage2_ab.softctx_grad_mode: "em_detach"` MAY detach the expected coord embeddings (or equivalently disable grad recording for embedding updates) to reduce memory.

Causal shift convention (normative):
- For a causal LM, logits at sequence position `t` predict the token at position `t+1` (the standard shift used for CE).
- Therefore, when updating a coord-slot embedding at token position `p`, the trainer MUST use the coord distribution read from logits at position `p-1` (for `p > 0`) from the previous iteration.

#### Scenario: Unroll mode does not detach expected coord embeddings
- **GIVEN** `stage2_ab.n_softctx_iter: 2`
- **AND** `stage2_ab.softctx_grad_mode: unroll`
- **WHEN** Channel-A runs the softctx loop
- **THEN** it executes two full forward passes
- **AND** it does not detach the expected coord embeddings used to update coord-slot inputs.

### Requirement: Hybrid objective preserves JSON structure CE and adds bbox geometry losses
The Stage-2 AB trainer MUST compute a hybrid objective with:

Channel-A:
- **Token CE anchor at A1**:
  - CE on non-coord tokens MUST be computed from the teacher-forced logits of the first forward (`z^(0)`; GT context).
  - Coord tokens MUST NOT contribute to CE (they are masked out), to avoid double-supervision.
- **Geometry + distribution regularizers from final softctx logits**:
  - Geometry losses and any distribution-level losses MUST be computed from the final-iteration logits `z^(n_softctx_iter-1)`.

Channel-B:
- **Matched-only geometry**:
  - Geometry losses MUST be computed only for matched `(pred_i -> gt_j)` pairs.
  - Unmatched predicted objects (FP under Hungarian) MUST NOT receive geometric gradients.
- **Stop-neutral CE**:
  - Channel-B CE MUST NOT supervise the stop/continue decision.
  - The trainer MUST mask CE on:
    - the top-level JSON closing brace `}` (the brace that closes the outermost assistant JSON object), and
    - `<|im_end|>` (the only turn-end token).
  - Top-level brace identification MUST be robust and deterministic:
    - the trainer MUST identify the token position of the `}` that closes the **outermost** JSON object in the rendered `y_GT_reordered` assistant span,
    - and MUST NOT rely on “the last `}` token id in the whole sequence” without verifying it corresponds to the outermost close brace of the assistant JSON.
    - A compliant approach is to decode the assistant-span token pieces and locate the outermost close brace via a brace-depth scan, then map the character span back to token positions.
- **FN append always**:
  - FN objects MUST be appended to the B3 target so they are supervised even when they were missing from rollout.
  - If `N_valid_pred == 0` after strict validation, the trainer MUST fall back to `y_GT_reordered := y_GT_canonical` (Stage-1 canonical GT order), which is equivalent to “FN append all GT objects”.
  - Optional weak correction: when `N_drop_invalid > 0`, the trainer MAY upweight Channel-B’s B3 structure-token CE weights to discourage “escaping supervision via invalid instances”.
    - This upweight MUST be controlled by `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier` (float).
    - The multiplier MUST default to `1.0` (no effect) and MUST be constrained to a safe range `[1.0, 4.0]` (clamp or fail fast).
    - “Structure-token CE weights” refers to Channel-B CE-supervised tokens excluding:
      - coord tokens,
      - desc value tokens, and
      - stop-neutral masked token positions (`}` and `<|im_end|>`).

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
  - the Channel-B teacher-forced logits under the rollout scaffold (or B2 refined logits when enabled).

Loss form (normative):
- The trainer MUST use SmoothL1 (Huber) + CIoU as the bbox regression terms.
- The trainer MUST NOT use GIoU in Stage-2 AB.

Numerical stability (normative):
- The trainer MUST canonicalize predicted boxes before CIoU:
  - `(x1,x2) := (min(x1,x2), max(x1,x2))`, `(y1,y2) := (min(y1,y2), max(y1,y2))`.
- The trainer MUST ensure the geometry losses do not produce NaNs/Infs, including early training when predictions are degenerate.

Efficiency rule (normative):
- If Channel-B has no valid matched pairs for a sample/batch, the trainer MUST skip the B2 forward (geo-only) and run B3 only.

#### Scenario: Channel-B is stop-neutral for `}` and `<|im_end|>`
- **GIVEN** Channel-B builds a teacher-forced target that ends with a top-level `}` followed by `<|im_end|>`
- **WHEN** the trainer builds CE labels/weights for Channel-B
- **THEN** it masks CE on that top-level `}` token position
- **AND** it masks CE on `<|im_end|>`.

#### Scenario: Dropped invalid instances may upweight B3 structure CE
- **GIVEN** a Channel-B rollout with `N_drop_invalid > 0`
- **AND** `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier: 1.5`
- **WHEN** Channel-B builds CE weights for structure tokens in B3
- **THEN** it MAY multiply the structure-token CE weights by `1.5` (bounded) for that sample/window.

## ADDED Requirements

### Requirement: Channel-B supports semantic-tolerant matched desc supervision
Channel-B desc supervision MUST support a semantic tolerance mode for matched objects:
- For each matched pair `(pred_i -> gt_j)`, compute a similarity score between:
  - the predicted description string `pred_desc_i` from rollout parsing, and
  - the GT description string `gt_desc_j` from the dataset.
- If similarity is at least a configurable threshold, the trainer MUST treat the predicted desc as acceptable and MUST NOT penalize the matched object’s GT desc token positions in CE (i.e., weight 0 / masked).
- If similarity is below the threshold, the trainer MUST apply a (small) desc CE weight to pull toward GT.

Scope constraints (normative):
- Semantic tolerance MUST apply only to **matched** objects.
- FN-appended objects MUST be supervised normally (no semantic gating), since they have no competing predicted description.

Configuration (normative):
- `stage2_ab.channel_b.semantic_desc_gate.enabled` MUST accept a boolean and MUST default to `true`.
- `stage2_ab.channel_b.semantic_desc_gate.threshold` MUST accept a float in `[0.0, 1.0]` and MUST default to `0.5`.
- `stage2_ab.channel_b.semantic_desc_gate.model_name_or_path` MUST accept a string and MUST default to `sentence-transformers/all-MiniLM-L6-v2`.
- If semantic gating is enabled, `stage2_ab.channel_b.semantic_desc_gate.revision` MUST be provided as a string to pin the embedding model version.
- If semantic gating is enabled, the resolved model identity MUST be logged for reproducibility (including the provided `revision`).
- If semantic gating is enabled but the sentence-transformer dependency or the specified model weights are not available at runtime, the trainer MUST NOT fail fast and MUST instead:
  - disable semantic gating for the affected step/run (treating matched desc tokens as not semantically acceptable unless they match the normal non-gated rules), and
  - emit a stable warning log at least once describing that semantic gating is disabled due to missing dependency/weights, and
  - expose a stable boolean metric/log key indicating whether semantic gating is active for the step (e.g., `stage2_ab/channel_b/semantic_desc_gate/is_active`).
Performance (non-normative guidance):
- The implementation SHOULD compute sentence embeddings in a batched manner per optimizer step (or per micro-batch) and MAY cache embeddings within the step to bound overhead without changing semantics.

#### Scenario: Semantically close matched desc is not penalized
- **GIVEN** a matched object whose rollout desc is semantically close to GT (similarity ≥ threshold)
- **WHEN** Channel-B builds CE labels/weights for the matched object’s GT desc tokens
- **THEN** those desc token positions are masked (weight 0) so the model is not forced to match the GT string exactly.

#### Scenario: Semantic gate is disabled when the model is unavailable
- **GIVEN** `stage2_ab.channel_b.semantic_desc_gate.enabled: true`
- **AND** `stage2_ab.channel_b.semantic_desc_gate.model_name_or_path: "/path/does/not/exist"`
- **AND** `stage2_ab.channel_b.semantic_desc_gate.revision: "pinned"`
- **WHEN** training starts and Channel-B attempts to compute semantic gating
- **THEN** the trainer continues without semantic gating for that step/run
- **AND** it emits a warning that semantic gating is disabled due to missing dependency/weights
- **AND** it exposes `stage2_ab/channel_b/semantic_desc_gate/is_active = false`.

#### Scenario: Semantic gate requires a pinned revision/version
- **GIVEN** `stage2_ab.channel_b.semantic_desc_gate.enabled: true`
- **AND** `stage2_ab.channel_b.semantic_desc_gate.model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"`
- **AND** `stage2_ab.channel_b.semantic_desc_gate.revision` is not provided
- **WHEN** training starts and Channel-B attempts to load the semantic gate model
- **THEN** the trainer fails fast with guidance to provide `stage2_ab.channel_b.semantic_desc_gate.revision`.

### Requirement: Deprecated legacy coord-loss knobs are silently ignored
To enable config refactors without blocking training runs, the configuration system MUST silently ignore deprecated legacy coord-loss knobs under `custom.*` that are no longer supported by the project’s coord-loss contract.

Normative minimum:
- If `custom.coord_loss` is present in a YAML config, configuration parsing MUST NOT raise, and the value MUST be ignored.

#### Scenario: custom.coord_loss does not hard error
- **GIVEN** a config that includes `custom.coord_loss` (legacy)
- **WHEN** configuration is parsed
- **THEN** parsing succeeds and the legacy field is ignored.
