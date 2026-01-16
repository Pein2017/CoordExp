## Context
CoordExp currently trains a Qwen3-VL family V-LLM to emit **JSON-only** structured detections with embedded `<|coord_k|>` tokens. Baseline training (alias: `stage_1`) uses:
- standard full-vocab CE for **non-coord** tokens (JSON structure + text), and
- distributional coord-token supervision (`softCE + W1 + gate`) for **coord** token positions only.

**Prerequisite (stage_1):** stage_2 rollout-matching assumes the model already produces valid JSON and `<|coord_*|>` tokens reliably. Run rollout-matching only after stage_1 output stability is achieved.

Rollout-matching training (alias: `stage_2`, motivated in `progress/full_idea.md`) requires an on-policy rollout to infer a latent alignment (matching) and then update the model in a way that is consistent with the rollout context.

This change introduces an explicit rollout-matching training infrastructure under the existing `src/sft.py` entrypoint, implemented as a trainer variant.

## Goals / Non-Goals
Goals:
- Provide a rollout-matching training loop that performs: rollout → strict parse → match → single teacher-forced forward → masked losses.
- Keep supervision **token-distributional** at coord positions (`softCE + W1 + gate`); keep non-coord supervision as standard CE.
- Keep the implementation YAML-first and minimally invasive to the existing baseline training entrypoint.
- Make failure modes explicit and safe (strict object dropping; mandatory FN append for recall recovery; fail-fast for unsupported modes like packing).

Non-goals:
- No expectation-decoding or continuous L1/GIoU losses.
- No output format change away from JSON.
- No DETR-style heads or query decoders.
- No IoU/GIoU/maskIoU metric logging (geometry scores are internal to matching only).

## Decisions
- Decision: Implement rollout-matching training as a trainer variant selected by `custom.trainer_variant: rollout_matching_sft`.
  - Rationale: this keeps the new capability isolated from baseline training and minimizes upstream divergence.

- Decision: Use ms-swift / transformers generation for the MVP rollout backend.
  - Rationale: it is guaranteed compatible with multimodal Qwen3-VL and deepspeed setups via `unwrap_model_for_generation` and `template.generate_context`.
  - Follow-up: keep an internal abstraction so vLLM can be added later as an optional backend.

- Decision: Use an identity/pass-through collator for rollout-matching and re-encode inside the trainer.
  - Rationale: the trainer needs `messages`, `assistant_payload`, and GT `objects` to generate rollouts, run strict parsing/matching, and construct the appended GT tail; this matches ms-swift rollout trainer patterns (e.g. GKD/GRPO).

- Decision: Converge rollout-matching into a single-path training construction.
  - Definition: for each sample, construct a single teacher-forced assistant token sequence:
    - `Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`
    - `Y_rollout_prefix` is a prefix of the rollout assistant `response_token_ids` with **suffix-only trimming** allowed to:
      - treat `<|im_end|>` as a hard stop (strip it even when fused),
      - drop a trailing incomplete / invalid suffix when the rollout is truncated mid-object,
      - drop the final top-level JSON `}` so the JSON object is open for append.
    - `SerializeAppend(FN_gt_objects)` emits the unmatched GT objects as an **append fragment** (comma-separated `"object_{n}": {...}` entries) and then closes the top-level JSON with a final `}` before EOS.
  - Definition: compute a single total loss from one forward pass on `Y_train` using per-token supervision masks.
  - Rationale: eliminates divergence between multiple forward passes / multiple targets; makes loss accounting and batching simpler.

- Decision: Treat the rollout prefix as immutable in token space (no re-serialization).
  - Rationale: any decode+re-encode can change tokenization and break coord-slot index alignment; using `response_token_ids` as the prefix preserves exact token boundaries.
  - Constraint: suffix-only trimming is allowed, but tokens before the trim boundary MUST remain unchanged.
  - Constraint: utilities MUST NOT sort keys or pretty-print the rollout prefix; predicted order is defined by raw text appearance.

- Decision: Enforce strict parsing (no repair) and object-level dropping.
  - Rationale: “repair” changes tokens and silently shifts supervision; strict object dropping keeps `Y_rollout` unchanged while still enabling training via FN append.

- Decision: Ignore `desc` value tokens for CE during rollout-matching (both prefix and appended tail).
  - Rationale: the GT label text can be incomplete/mislabeled and the model’s rollout text can be more specific; forcing desc tokens risks amplifying noise and harming stability. Stage_2 focuses on geometry alignment + recall recovery via FN append.
  - Note: JSON structure tokens remain supervised via CE in the appended tail; coord slots remain supervised via `softCE + W1 + gate`.

- Decision: Compute coord-slot indices via token-aligned parsing, not by searching the decoded text.
  - Rationale: repeated `<|coord_k|>` values are common; string search is ambiguous and non-deterministic.

- Decision: Use maskIoU-based Hungarian matching with dummy augmentation and pre-assignment gating (MVP baseline).
  - Rationale: geometry-only matching is available for bbox/poly, and gating prevents early training from being dominated by wrong matches.

- Decision: Add poly self-context supervision via Sinkhorn OT + barycentric projection ONLY (no mixture).
  - Rationale: poly vertices do not have a fixed semantic order; OT provides a stable alignment while staying within token-distributional coord supervision.

- Decision: Fail fast when `training.packing: true` under rollout-matching.
  - Rationale: ms-swift rollout generation paths do not support packing/padding-free; failing fast avoids silent mis-training.

## Alternatives Considered
- Implement rollout-matching training as a second entrypoint (`src/sft_rollout_matching.py`) instead of a trainer variant.
  - Pros: fewer conditionals in `src/sft.py`.
  - Cons: duplicates config/plumbing and diverges from the project’s “one entrypoint” practice.

- Use a two-path design (“reordered-GT SFT” + “self-context forward”) instead of a single-path masked-loss design.
  - Pros: conceptually separates “learn format/text” and “calibrate coords”.
  - Cons: duplicates forward passes and complicates batching; encourages accidental divergence in masking/targets across paths.

- Use vLLM as the rollout backend from day one.
  - Pros: faster rollouts.
  - Cons: more complex weight sync + multimodal edge cases; higher risk for the MVP.

## Risks / Trade-offs
- Risk: Aligning per-object coord token positions to token indices may be brittle.
  - Mitigation: strict token-aligned parsing; exclude ambiguous objects from self-context supervision; FN append tail ensures the sample still provides teacher-forced signal.

- Risk: Rollout-matching increases compute (extra forward(s) + generation).
  - Mitigation: keep matching and OT alignment lightweight (candidate pruning); add a `generate_every`-style knob later if needed (not required for MVP).

- Risk: Wrong matches early can harm training.
  - Mitigation: strict pre-assignment gating (`maskIoU < threshold` treated as infeasible) and bounded debug counters (no geometry metric logging).

## Migration Plan
- Add the trainer variant behind `custom.trainer_variant`.
- Add a rollout-matching YAML template and a small smoke-test recipe (single GPU, small limit, deterministic seed).
- Validate with a short run that:
  - strict parsing yields a non-trivial valid-object rate,
  - match rate is reasonable,
  - losses are finite and decreasing.

## Open Questions
- What is the most robust token-aligned parsing strategy to guarantee coord-slot indices for JSON without relying on string search?
- When adding vLLM, do we accept slightly stale weights between sync steps, or must rollouts reflect every gradient update?
