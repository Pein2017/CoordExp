# Change: Add Stage-2 AB training (iterative soft self-context Channel-A + rollout Channel-B), bbox-only v1

## Why
Stage-2 needs a reproducible, paper-ready training loop that combines (A) a no-rollout proxy for self-conditioning and (B) on-policy rollout self-conditioning, while staying compatible with the existing JSON-only data contract and Qwen3-VL (dense) forward semantics.

Stage-2 is intended to start from a Stage-1 pretrained checkpoint, so rollouts are expected to be relatively stable (though still imperfect, e.g. missing objects). The Stage-2 infra should therefore optimize for determinism and strict alignment rather than “repairing” broken outputs.

## What Changes
- Add a new training capability for Stage-2 with a deterministic two-channel schedule:
  - **Channel-A (hot)**: iterative soft self-context via `n_softctx_iter` full-forwards (no autoregressive rollout).
  - **Channel-B (cold)**: deterministic self-rollout + strict parse/match + teacher-forced training.
- Introduce a bbox-only v1 guardrail: Stage-2 fails fast if GT contains `poly`; no poly->bbox conversion.
- Compute a hybrid objective that preserves JSON-structure CE while adding decoded-geometry losses on bboxes (L1 + GIoU), using CoordExp expectation decoding with `k/999`.
- As a prerequisite for "globally consistent" geometry semantics, remove `/1000`-based coord-bin normalization across the codebase and unify all bin->normalized-float conversions to `k/999` (including tests, scripts, and docs).
- Keep compatibility with existing `rollout-matching-sft` infra (vLLM backends, post-rollout packing, rollout buffer offload) by reusing configs and utilities, without changing the existing `rollout_matching_sft` trainer behavior.
- Maintain full compatibility with ms-swift and Transformers by relying on their standard trainer/checkpoint/forward contracts (no upstream patches).

## Impact
- Affected specs:
  - **ADDED**: `stage2-ab-training` (new capability).
  - No changes to existing `rollout-matching-sft` requirements (kept stable).
- Affected code (planned; not implemented in this change proposal):
  - New trainer variant (e.g., `custom.trainer_variant: stage2_ab_training`).
  - Shared geometry-loss utilities (bbox-only v1).
  - Reuse existing rollout backend / packing / buffering code paths for Channel-B.
