# stage2-ab-training Spec Delta

This is a delta spec for change `remove-stage2-ab-stop-neutral`.

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
- **Matched-only geometry**:
  - Geometry losses MUST be computed only for matched `(pred_i -> gt_j)` pairs.
  - Unmatched predicted objects (FP under Hungarian) MUST NOT receive geometric gradients.
- **Stop/closure CE is supervised**:
  - Channel-B CE MUST supervise stop/continue only via token-level supervision on:
    - the top-level JSON closing brace `}` (the brace that closes the outermost assistant JSON object), and
    - `<|im_end|>` (the only turn-end token).
  - Stop-neutral masking MUST NOT be applied in Stage-2 AB Channel-B.
  - Stop-neutral config keys MUST NOT be accepted (no legacy stop-neutral knobs are supported under the typed `stage2_ab` schema).
  - Top-level brace identification MUST be robust and deterministic:
    - the trainer MUST identify the token position of the `}` that closes the **outermost** JSON object in the rendered `y_GT_reordered` assistant span,
    - and MUST NOT rely on "the last `}` token id in the whole sequence" without verifying it corresponds to the outermost close brace of the assistant JSON.
    - A compliant approach is to decode the assistant-span token pieces and locate the outermost close brace via a brace-depth scan that is string/escape-aware (i.e., braces inside JSON string literals do not affect brace depth), then map the character span back to token positions.
  - If stop/closure marker positions cannot be located deterministically for a sample (e.g., truncation or prefix/tail misalignment), the trainer MUST drop that sample from Channel-B supervision for that step and MUST increment `stage2_ab/channel_b/closure_supervision/N_drop` (as an entry in `trainer_metrics.metrics` per `src/metrics/payload_contract.py`).
    - `stage2_ab/channel_b/closure_supervision/N_drop` MUST be a non-negative global counter for the step (a compliant approach is a global sum over rank-local counts).
    - If all Channel-B samples in a step are dropped for closure-marker resolution (no valid samples remain for Channel-B supervision in that step), the trainer MUST fail fast with actionable diagnostics.
  - The trainer MUST emit `rollout/parse_truncated_rate` (as an entry in `trainer_metrics.metrics` per `src/metrics/payload_contract.py`) for Channel-B steps as a global aggregate after grad-accum aggregation and DDP all-reduce.
    - `rollout/parse_truncated_rate` is defined as `(num_truncated_samples / num_rollout_samples)` for the step (0 when `num_rollout_samples == 0`).
  - All metrics emitted via the neutral trainer-metrics payload `metrics` map (see `src/metrics/payload_contract.py`) MUST be emitted as **global** aggregates after grad-accum aggregation and DDP all-reduce (no rank-local training metrics).
    - counters: global sum,
    - wall-time seconds (e.g., `time/*_s`): global max,
    - rates: ratio of globally-summed numerator/denominator (never mean of rank-local ratios).
- **FN append always**:
  - FN objects MUST be appended to the B3 target so they are supervised even when they were missing from rollout.
  - If `N_valid_pred == 0` after strict validation, the trainer MUST fall back to `y_GT_reordered := y_GT_canonical` (Stage-1 canonical GT order), which is equivalent to "FN append all GT objects".
  - Optional weak correction: when `N_drop_invalid > 0`, the trainer MAY upweight Channel-B's B3 structure-token CE weights to discourage "escaping supervision via invalid instances".
    - This upweight MUST be controlled by `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier` (float).
    - The multiplier MUST default to `1.0` (no effect) and MUST be constrained to a safe range `[1.0, 4.0]` (clamp or fail fast).
    - "Structure-token CE weights" refers to Channel-B CE-supervised tokens excluding:
      - coord tokens, and
      - desc value tokens.
      - This includes top-level stop/closure tokens (`}` and `<|im_end|>`).

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

#### Scenario: Channel-B supervises top-level closure and `<|im_end|>`
- **GIVEN** Channel-B builds a teacher-forced target that ends with a top-level `}` followed by `<|im_end|>`
- **WHEN** the trainer builds CE labels/weights for Channel-B
- **THEN** it keeps CE supervision on that top-level `}` token position
- **AND** it keeps CE supervision on `<|im_end|>`.

#### Scenario: Stop-neutral masking is not applied
- **GIVEN** Stage-2 AB Channel-B configuration
- **WHEN** CE masks are constructed for Channel-B
- **THEN** top-level `}` and `<|im_end|>` are not masked out by any stop-neutral branch
- **AND** FP-neutral masking remains limited to unmatched predicted object spans.

#### Scenario: Legacy stop-neutral config keys fail fast
- **GIVEN** Stage-2 AB config includes legacy stop-neutral keys under Channel-B
- **WHEN** trainer configuration is validated
- **THEN** startup fails fast before training
- **AND** the error indicates stop-neutral knobs are unsupported under the typed contract.

#### Scenario: Closure marker resolution failure is dropped and counted
- **GIVEN** a Channel-B sample where the trainer cannot deterministically locate the outermost `}` / `<|im_end|>` marker positions (e.g., truncation)
- **WHEN** the trainer constructs CE labels/weights for Channel-B
- **THEN** it drops the sample from Channel-B supervision for that step
- **AND** it increments `stage2_ab/channel_b/closure_supervision/N_drop`.

#### Scenario: Dropped invalid instances may upweight B3 structure CE
- **GIVEN** a Channel-B rollout with `N_drop_invalid > 0`
- **AND** `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier: 1.5`
- **WHEN** Channel-B builds CE weights for structure tokens in B3
- **THEN** it MAY multiply the structure-token CE weights by `1.5` (bounded) for that sample/window.

#### Scenario: No valid predictions fall back to canonical GT order
- **GIVEN** strict validation yields `N_valid_pred == 0`
- **WHEN** Channel-B builds `y_GT_reordered` for B3
- **THEN** it sets `y_GT_reordered := y_GT_canonical`
- **AND** this is equivalent to appending all GT objects as FN-supervised targets.

#### Scenario: Out-of-range struct CE multiplier is handled safely
- **GIVEN** `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier` is outside `[1.0, 4.0]`
- **WHEN** trainer parses Channel-B config
- **THEN** it clamps the value into `[1.0, 4.0]` or fails fast (implementation choice)
- **AND** it MUST NOT run with an effective multiplier outside the safe range `[1.0, 4.0]`.

#### Scenario: B2 forward is skipped when there are no valid matched pairs
- **GIVEN** Channel-B sample/batch has zero valid matched pairs
- **WHEN** trainer executes Channel-B steps
- **THEN** it skips B2 geo-only forward
- **AND** runs B3 only.
