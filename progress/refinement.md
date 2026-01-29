# Stage-2 AB Pre-Implementation Refinements (Semantics + Guardrails)

This document records the refinements/decisions from the Stage-2 AB semantic audit discussion (Jan 28, 2026).
Stage-2 AB is not implemented in-repo yet; these notes are intended to be applied as OpenSpec deltas and to
guide implementation without accidental semantic drift.

Related context:
- Deep tensor-flow audit notes (existing Stage-2 rollout-matching + packing/mRoPE hazards): `progress/analysis-report.md`
- Algorithm intent: `progress/full_idea.md`
- Pending change contract baseline (to be updated by another agent): `openspec/changes/2026-01-27-add-stage2-ab-iterative-softctx-bboxonly/`

## 0. Decisions (Locked)

### 0.1 Qwen3-VL multimodal + Channel-A iterative `inputs_embeds`: add a guardrail (required)

Background:
- HF Qwen3-VL enforces XOR: exactly one of `input_ids` or `inputs_embeds` may be passed per forward.
- If `input_ids is None`, Qwen3-VL detects image/video placeholder positions by comparing rows of
  `inputs_embeds` to the embedding of `image_token_id`/`video_token_id` (embedding-equality based placeholder mask).
- During forward, Qwen3-VL inserts visual features via `masked_scatter` into the placeholder positions.

Risk:
- Even if Channel-A only intends to modify bbox coord-token slots, iterative loops can accidentally reuse
  the post-forward `inputs_embeds` (already containing visual features) for the next iteration. That breaks
  placeholder detection and can silently corrupt multimodal semantics or raise feature/token count mismatch errors.

Refinement / guardrail:
- For every softctx iteration forward, build a fresh base `inputs_embeds` from teacher-forced `input_ids`
  (embedding module call), then replace only coord-slot rows; never reuse post-forward `inputs_embeds`.
- Add an explicit runtime assertion (debug mode) that placeholder-token embedding rows are unchanged bitwise.
  (Positions come from `input_ids` for bookkeeping; the model forward still receives `input_ids=None`.)
- Do not change model train/eval mode inside the softctx loop; only grad recording changes per iteration.
- Do not use or persist kv-cache across Channel-A iterations.

### 0.2 Autograd boundary is a MUST-TEST contract (required)

Channel-A intent:
- Iterations `m = 0 .. n_softctx_iter-2` run under `torch.no_grad()` (including softmax/expectation used for
  the next-iteration coord-slot embeddings).
- Final iteration `m = n_softctx_iter-1` runs with grad and supplies logits used for geometry decoding and loss.
- Softctx expected embeddings are treated as E-step artifacts: stop-grad / detach is applied so no unwanted
  graph spans earlier iterations.

Refinement:
- Add deterministic CPU-friendly tests (see Section 2) that assert:
  - expected parameters have finite grads after backward, and
  - rollout / Channel-B recording and intermediate Channel-A iterations do not create grad history.

### 0.3 Causal shift / off-by-one is fragile; verify explicitly (required)

Normative convention:
- For a coord token at input position `p`, read its distribution from logits position `p-1` (standard next-token shift).

Refinement:
- Add tests that explicitly verify the gather indices match `p-1` under both non-packed and packed-sequence inputs,
  and that no cross-segment leakage occurs at packed boundaries.

### 0.4 Packing + Qwen mRoPE must be tested comprehensively (required)

Background:
- With padding-free packing, correctness depends on consistent `text_position_ids` boundaries and Qwen mRoPE handling.
- Existing Stage-2 rollout-matching trainer already contains a necessary fix: rebuild 4-row `position_ids` as
  `[text_position_ids; mRoPE(t/h/w)]` before model forward to avoid silent packed-boundary failures.

Refinement:
- Stage-2 AB MUST apply the same 4-row position-id contract for every forward in both channels,
  including every softctx iteration forward in Channel-A.
- Add tests to ensure every iteration sees correct `position_ids` layout and that forward does not regress under packing.

### 0.5 Channel-B rollout should support stochastic decoding with seed (HF and vLLM) (required)

Decision:
- Allow non-zero temperature / sampling for generalization in dense detection tasks.
- Keep it reproducible: stochastic rollouts MUST be seeded and seed metadata logged.

Refinement:
- Extend/clarify rollout seeding contract to cover *sampling* in both backends:
  - Define `rollout_seed_base = (training_seed + global_step * 1000003) & 0x7FFFFFFF`.
  - Derive deterministic per-batch/per-request seeds from `rollout_seed_base` (stable plan; log it).
  - For HF backend: pass a `torch.Generator` via `generate(..., generator=gen)` (sampling uses RNG),
    seeded from `rollout_seed_base` plus a stable offset (e.g., per microbatch start index).
  - For vLLM backend: pass seed in RequestConfig (or equivalent) using the same seed plan.

### 0.6 Coordinate normalization constitution: bins are 0..999; normalized float uses `/999` (required)

Decision:
- Coord tokens represent integer bins `k ∈ [0, 999]` (inclusive).
- Canonical encode/decode for Stage-2 AB geometry decoding and losses:
  - Encode (float->bin): `k = clamp(round(999*c), 0, 999)`
  - Decode (bin->float): `c = k / 999`
- Therefore, normalized float values MUST treat `k=999` as `1.0`.

Why `/999` (not `/1000`):
- `/999` makes the mapping consistent and invertible with `round(999*c)`; `/1000` makes 999 map to 0.999,
  which systematically shrinks edge coordinates and breaks the intended contract for geometry losses and decoding.

Hard rule (repo-wide): never use `/1000` as a *normalization denominator* for coord bins
- If a value is a coord bin index `k ∈ [0, 999]`, then any “normalized coord float” MUST be `k/999`.
- Do **not** use `k/1000` anywhere (including “helper metadata”, eval, matching heuristics, visualization, etc.).
- If a downstream step needs coordinates strictly inside a raster grid `[0, R)`, do not reintroduce `/1000`.
  Instead, use the `/999` normalized float and project to an in-bounds raster coordinate:
  - `u = k/999`  in `[0, 1]`
  - `x_raster = u * (R - 1)`  in `[0, R-1]` (safe for libraries expecting `[0, R)`).

Clarification: “1000” is still valid as the number of bins `K=1000` (because bins are 0..999),
but **1000 MUST NOT be used as the normalization denominator**.

Important repo-wide footgun (current state):
- The codebase currently contains *both* conventions:
  - Some utilities/metadata normalize by `/1000` (e.g., `_coord_token_norm` helpers).
  - Pixel conversion utilities use `/999` (via `MAX_BIN=999`), which maps 999 to the image boundary.

Refinement:
- Unify the project-wide “bin -> normalized float” convention to `/999`.
- Remove `/1000`-based “normalized coord” helpers/metadata entirely (no exceptions for internal rasterization).

## 1. Proposal-Level Deltas to Apply to OpenSpec

### 1.1 Deterministic schedule semantics remain, but stochastic rollout is permitted (seeded)

Update/extend the spec to:
- Allow `temperature>0` and sampling for both HF and vLLM backends, while requiring:
  - a deterministic seed plan derived from `(training_seed, global_step, stable_request_index)`; and
  - logging of `rollout_seed_base` and the effective seed plan at least once per optimizer step when fresh rollouts occur.

Rationale:
- Supports exploration/generalization for dense detection while preserving research reproducibility.

### 1.2 Coordinate normalization: standardize to `/999` everywhere for bin->float

Update/extend the spec and repo conventions to:
- Define the canonical bin->float normalization as `k/999` (inclusive endpoint).
- Ban `/1000` in any bin->float mapping (including “metadata normalization”, eval, matching, and visualization).
- (Optional but strongly recommended) Any “normalized bin distance” metric (e.g., W1 distance) SHOULD normalize by
  `MAX_BIN=999` (or `K-1`) so that the maximum possible distance maps cleanly to `1.0` under normalized scale.

Rationale:
- Prevents systematic edge shrinkage and mismatched encode/decode contracts; matches Stage-2 AB intended losses.

### 1.3 Channel-A multimodal safety guardrails (Qwen3-VL specific)

Update/extend the spec to explicitly require:
- Fresh base `inputs_embeds` each iteration (no reuse after model-internal feature insertion).
- Placeholder-token embedding rows remain bitwise unchanged except coord-slot rows.
- No kv-cache reuse across softctx iterations.

Rationale:
- HF Qwen3-VL placeholder detection depends on embedding equality when `input_ids=None`.

### 1.4 Identity collator + trainer-side encoding is the default integration choice

Decision:
- Use identity data collator and do encoding inside the trainer to guarantee raw fields are available for Channel-B
  and to keep both channels consistent (simplest integration surface).

Packing:
- Require `training.packing=true` for Stage-2 AB efficiency.
- Stage-2 AB must follow Stage-2 rollout-matching semantics:
  - dataset-level packing wrappers disabled;
  - trainer performs dynamic packing after rollout (Channel-B) and equivalent packing for Channel-A;
  - every forward uses correct packing metadata and Qwen 4-row `position_ids` contract.

## 2. Test Plan (CPU-friendly, deterministic; to be placed under `temp/`)

These tests are intended to surface semantic bugs early; they are not coverage-oriented.

### 2.1 Autograd contract test (required)
- Minimal forward + loss under Channel-A and Channel-B.
- Assert expected trainable parameters receive finite, non-null grads after `.backward()`.

### 2.2 No-grad boundary test (required)
- Assert that:
  - Channel-A iterations `0..n_softctx_iter-2` do not build autograd graphs, and
  - Channel-B rollout / parsing / recording does not attach grad history.
- Include a lightweight check against unexpected graph retention (e.g., tensor `.grad_fn is None` where required).

### 2.3 Geometry contract test (required)
- With `do_resize=false`, assert coordinate tensors preserve shape/order/values end-to-end (bbox_2d is `[x1,y1,x2,y2]`).
- Assert normalization constitution uses `/999` (k=999 -> 1.0) and never `/1000`.

### 2.4 Shift + packing/mRoPE invariants test (required)
- Verify coord-slot distribution is read from logits at `p-1` (causal shift).
- Under packing, verify no cross-segment leakage at packed boundaries.
- Verify every iteration forward uses correct 4-row `position_ids` with `text_position_ids` as row0.

### 2.5 Determinism sanity (optional, but recommended)
- Same (training seed, global_step, stable request index) produces identical rollouts/artifacts under sampling
  for both HF and vLLM backends (given same model weights).

## 3. Concrete Repo Touchpoints (for the implementer)

These are known footgun locations to update when applying the refinements:

- `/1000` normalization helpers / occurrences to remove:
  - `src/coord_tokens/codec.py` (`normalized_from_ints`)
  - `src/coord_tokens/validator.py` (`_coord_token_norm` metadata)
  - `src/datasets/preprocessors/augmentation.py` (restoring `_coord_token_norm`)
  - tests referencing `/1000` (e.g., `tests/coord_tokens/test_augmentation_roundtrip.py`)
  - `src/trainers/rollout_matching_sft.py` (`_mask_iou_norm1000` projects with `R/1000` today; re-express with `/999` + `(R-1)` scaling)
  - `scripts/analysis/rollout_backend_bench/benchmark_rollout_backends.py` (bin->pixel projection uses `/1000` today)
  - `public_data/scripts/filter_repetitive_samples.py` (grid projection uses `/1000` today)
  - `public_data/scripts/convert_to_coord_tokens.py` (binning uses `*1000.0` today; must become `*999.0`)
  - OpenSpec / docs references that still claim `k/1000` normalization (must be updated by the spec agent):
    - `openspec/specs/coord-token-mode/spec.md`
    - archived change docs that mention `k/1000`

- Qwen3-VL packing/mRoPE contract precedent:
  - `src/trainers/rollout_matching_sft.py` (4-row `position_ids` reconstruction before forward)

- Seed plan precedent:
  - `src/trainers/rollout_matching_sft.py` (`_derive_rollout_seed_base`)
  - extend for HF sampling via `generate(..., generator=...)` and for vLLM sampling via RequestConfig seed.
