## Context
Stage-1 provides a teacher-forced baseline. Stage-2 aims to reduce teacher-forcing mismatch by introducing self-conditioning effects through two complementary channels:
- **Channel-A (hot)**: iterative *soft* self-context without autoregressive rollout.
- **Channel-B (cold)**: deterministic self-rollout + strict parse/match + teacher-forced learning.

Stage-2 AB assumes the model is initialized from a Stage-1 pretrained checkpoint. Therefore, in normal runs rollouts are expected to be relatively stable/syntactically well-formed, but still imperfect (e.g., missed objects, inaccurate localization). Deterministic invalid-rollout fallback is still required for robustness and reproducibility.

This change proposal is **bbox-only v1**. Polygons (`poly`) are out of scope and must be filtered out upstream; Stage-2 fails fast if GT contains `poly`. The design should remain scalable to `poly` in later versions.

Constraints:
- JSON-only assistant schema (no `<obj><box>` tags).
- Fully deterministic scheduling and rollout seeding.
- Must remain compatible with Qwen3-VL (dense) multimodal forward semantics.
- Must not modify upstream HF model files (e.g., `modeling_qwen3_vl.py`).

## Goals / Non-Goals
Goals:
- Provide a new trainer variant for Stage-2 AB training with deterministic channel scheduling.
- Implement Channel-A iterative soft self-context using `n_softctx_iter` full-forwards (no sampling).
- Reuse existing rollout-matching infrastructure for Channel-B.
- Add decoded-geometry bbox losses (L1 + GIoU) using CoordExp expectation decoding (`c_hat = E[k]/999`) in normalized `[0, 1]` space.
- Keep JSON structure CE while allowing configurable `desc` CE weight / mask.

Non-Goals:
- Adding DETR heads or changing the model architecture.
- Supporting `poly` rollouts/losses in this v1 (no poly->bbox conversion).
- Making vLLM server-mode work under multi-process learners (existing restriction remains).

## Decisions
### Decision: New trainer variant (do not change `rollout_matching_sft` behavior)
We introduce a new `custom.trainer_variant` (name TBD in implementation, e.g. `stage2_ab_training`) instead of extending `rollout_matching_sft`.
Rationale:
- The existing `rollout-matching-sft` spec explicitly disallows decoded-coordinate losses; Stage-2 AB requires them.
- Keeping the existing trainer stable avoids breaking prior configs and archived specs.

### Decision: Qwen3-VL compatibility via `inputs_embeds` (Channel-A iterative forwards)
Qwen3-VL requires `input_ids` XOR `inputs_embeds`. For Channel-A iterative soft self-context we need to alter coord-slot embeddings.
Approach:
- Build the initial `inputs_embeds` from the model's input embedding layer using the teacher-forced `input_ids`.
- Update *only* coord-slot embeddings during the softctx loop; do not touch vision placeholder embeddings.
- Call the model with `inputs_embeds` and `input_ids=None`.

Compatibility note:
- When `input_ids=None`, Qwen3-VL detects `<image>` placeholders by **exact embedding equality** (bitwise). Therefore we must start from the original embedding outputs and avoid perturbing placeholder slots; only scatter updates into coord-slot positions.
- Multimodal safety guardrail: for **every** softctx iteration forward, rebuild a **fresh** base `inputs_embeds` by calling the embedding module on the teacher-forced `input_ids`, then scatter-update only the coord-slot rows. Never reuse a post-forward `inputs_embeds` tensor (it may already have visual features inserted).
- Cache safety: do not carry `past_key_values` across iterations; force `use_cache=False` for Channel-A training forwards. (This is required because generation uses in-place cache updates and is not safe to reuse for iterative training-style forwards.)
- Optional debug assertion: for multimodal batches, assert placeholder-row embeddings are bitwise unchanged between base embeddings and the scattered embeddings (only coord rows may differ).

### Decision: Memory control via default `no_grad` for early iterations
Channel-A does `n_softctx_iter` full-forwards. To keep memory predictable:
- Iterations `0..n_softctx_iter-2` run under `torch.no_grad()` (no activation saving).
- The final iteration runs with grad and produces the logits used for decoding + loss.
Additional guardrails:
- Soft expected embeddings are treated as E-step artifacts (detach/stop-grad).
- Do not toggle `model.train()`/`model.eval()` inside the softctx loop; only gradient recording changes per iteration.

### Decision: Bbox-only v1 guardrails
- GT objects must be `bbox_2d` only (length=4, values decodable and within `[0, 999]`, and valid ordering `x1<=x2,y1<=y2`). Any GT non-bbox geometry or malformed bbox fails fast.
- If rollout parsing yields predicted non-bbox objects (including `poly`) or malformed bboxes, drop them deterministically and emit per-type counters.
- If Channel-B rollout parsing is invalid (e.g., missing top-level `{`), fall back deterministically to the empty prefix `{`, record an invalid-rollout counter, and continue training via FN append.
- Do not implement `poly -> bbox` conversions.

### Decision: Coord quantization uses `k/999` only
All coord bin encode/decode logic is unified as:
- Encode: `k = clamp(round(999*c), 0, 999)`
- Decode: `c = k/999`
No `1000`-based normalization is allowed.

Repo-wide consistency requirement:
- Any `/1000`-based "bin -> normalized float" helpers/metadata MUST be removed or updated to `/999` as part of implementing this change (including tests, scripts, and docs).
- Rationale: Stage-2 AB geometry losses and CoordExp expectation decoding rely on `k=999 -> 1.0`. Keeping `/1000` in any part of the pipeline creates silent scale drift and breaks comparability.

## Risks / Trade-offs
- **Iterative forwards cost**: Channel-A adds extra forward passes. Mitigation: default `no_grad` for early iterations; keep `n_softctx_iter` small (ablation: 1/2/3).
- **Qwen3-VL placeholder detection**: Using `inputs_embeds` requires preserving placeholder embeddings exactly. Mitigation: update only coord slots; add a runtime assertion that placeholder mask counts match visual feature length.
- **Packing interactions**: Extra forwards must preserve correct `position_ids` semantics under padding-free packing. Mitigation: reuse the existing Qwen3-VL `position_ids` handling pattern from `rollout_matching_sft` (4-row position_ids with `text_position_ids` leading row).
- **Packed-boundary correctness hazard** (FlashAttention-2 padding-free): if the wrong row is treated as `text_position_ids`, boundaries may be silently corrupted. Mitigation: require 4-row `position_ids` for every forward (including each Channel-A iteration) and add a deterministic regression test / audit script.
- **KV cache leakage across softctx iterations**: accidental cache reuse can produce stale attention and nondeterministic behavior. Mitigation: hard-set `use_cache=False`, never pass/keep `past_key_values`, and assert `outputs.past_key_values is None` in training forwards.
- **Logits slicing footgun** (`logits_to_keep`): if the forward path slices logits, coord decoding and CE alignment can silently break. Mitigation: enforce full-sequence logits for all training forwards (or fail fast).

## Migration Plan
- Add the new trainer variant without altering `rollout_matching_sft`.
- Provide YAML examples that can run:
  - Channel-A only (no rollout) for smoke testing.
  - Channel-B only (pure rollout-matching + bbox geo loss) for comparison.
  - Mixed schedule for full Stage-2.

## Open Questions
- Final naming of the new trainer variant and YAML key namespace (recommend `custom.extra.stage2_ab`).
- Whether geometry loss should be applied only on the final iteration logits (default) or also intermediate iterations (optional).
- For Channel-B rollouts, if stochastic decoding is enabled under the HF backend, we will use global seeding (e.g., `seed_everything(rollout_seed_base)`) as a best-effort reproducibility mechanism. Strict per-request determinism under HF sampling is out of scope for v1 landing; use the vLLM backend for seeded per-request sampling if strictness is required.
