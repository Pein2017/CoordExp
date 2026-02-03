## Context
Stage-2 AB training is the main EM-ish training loop for CoordExp, combining:
- Channel-A: iterative soft self-context (no rollout) for high-throughput geometry calibration, and
- Channel-B: sparse rollouts + matching for set-level alignment under “true self-context”.

`progress/full_idea.md` is treated as the governing constitution for semantics. This change proposal updates the OpenSpec contract so the repository’s training behavior, configs, and docs can be refactored to match that constitution.

Key constraints:
- Config-first: do not introduce new CLI flags.
- Preserve Qwen3-VL chat template compatibility; do not edit upstream HF model files.
- Training uses `do_resize=false` and preserves geometry ordering (coord slots are never dropped/reordered).

## Goals / Non-Goals
Goals:
- Make Stage-2 AB behavior fully determined by explicit, typed YAML config (no “bag of knobs” in `custom.extra`).
- Make Channel-A default **fully unrolled** (no detach in the softctx loop), to test real chain-differentiable credit assignment.
- Make Channel-B **FP-neutral** in geometry and **stop-neutral** in CE, to avoid suppressing unlabeled objects.
- Make bbox loss consistent and stable: **SmoothL1 + CIoU** with canonicalization; remove GIoU.
- Make “strict drop” on invalid rollout instances visible via diagnostics, and ensure invalid instances do not “escape” correction.
- Keep deterministic scheduling and reproducibility across resume.

Non-Goals:
- Adding polygon (poly) training losses or changing poly parsing beyond bbox-only v1.
- Migrating `custom.extra.rollout_matching` to a new top-level typed config in this change (future refactor).
- Introducing RL / reward optimization loops.

## Decisions

### 1) Channel schedule (`b_ratio`) is deterministic and step-driven
Replace `schedule.pattern` with `stage2_ab.schedule.b_ratio: float ∈ [0,1]`.

Decision: use a Bresenham-style ratio schedule:
- Let optimizer step be `s` (0-indexed).
- Select Channel-B iff `floor((s+1) * b_ratio) > floor(s * b_ratio)`, else Channel-A.

This yields:
- exact long-run proportion (up to rounding),
- even spacing for small ratios,
- strict determinism from `global_step` only (no extra state).

### 2) Channel-A “unroll” is the default grad semantics
Default `stage2_ab.softctx_grad_mode = "unroll"`:
- no `no_grad` early iterations,
- no `detach()` of expected coord embeddings anywhere,
- gradients may flow across iterations and across coord slots.

Provide an explicit fallback mode (`"em_detach"`) as an opt-in ablation, but it is not the default.

### 3) CE anchor split for Channel-A
Channel-A computes:
- CE (non-coord tokens, including desc/text/structure) on the **A1** teacher-forced logits `z^(0)` (GT context),
- geometry + distribution regularizers from the final softctx logits `z^(n_softctx_iter-1)`.

Rationale: “CE@A1, geo@final” matches the constitution and avoids conflating text supervision with softctx coordinate refinement.

### 4) Channel-B FP-neutral + stop-neutral
Channel-B is defined as:
- matched-only geometric gradients (FP objects get no geometry loss),
- stop-neutral CE (mask top-level `}` and `<|im_end|>` from Channel-B CE),
- FN append always.

Implementation details that must be specified:
- `<|im_end|>` is the only turn-end token (EOS).
- The “top-level `}`” to mask must be located by brace-stack parsing on the rendered JSON text (outermost close brace), not by “last `}` token id”.

### 5) Strict-drop diagnostics + weak correction
Invalid predicted instances are dropped deterministically (no repair), but must produce:
- `N_valid_pred`, `N_drop_invalid`, and per-reason buckets.

To avoid “escape via invalid instances”, when `N_drop_invalid > 0` the system may upweight the B3 structure CE term (small, bounded).

### 6) Semantic-tolerant matched desc supervision
For matched pairs `(pred_i -> gt_j)`:
- compute sentence embedding similarity between `pred_desc_i` (from rollout) and `gt_desc_j`,
- if similarity ≥ threshold: treat as correct (mask GT desc token positions in CE),
- else: apply a small desc CE weight to pull toward GT.

FN-appended objects are supervised normally (no semantic gating).

The sentence-transformer model is configured by path/name + a pinned revision. If semantic gating is enabled but the dependency or model weights are unavailable at runtime, the trainer does not fail fast and continues with semantic gating disabled (with a warning + an `is_active` metric/log key).

### 7) CIoU + SmoothL1 is mandatory; remove GIoU
Stage-2 AB bbox geometry loss is:
- SmoothL1 (Huber) on normalized coords, plus
- CIoU on canonicalized boxes.

All boxes must be canonicalized before CIoU to avoid NaNs and nonsensical gradients early.

## Risks / Trade-offs
- **Compute/memory:** full unroll increases memory vs EM-detach. Mitigation: keep `n_softctx_iter` small by default (2), use gradient checkpointing if needed, and provide explicit `em_detach` ablation.
- **Stop-neutral Channel-B:** Channel-B no longer teaches stopping; Channel-A remains responsible for closure supervision. This is intentional for unlabeled-aware FP neutrality.
- **Semantic gating cost:** sentence-transformer adds CPU overhead; mitigate by batching + caching embeddings per step, and by making it configurable (on/off).
- **Config migration:** moving to top-level typed `stage2_ab` is breaking; mitigate by migrating repo configs in-tree and removing legacy knobs from docs/configs.

## Migration Plan
1) Introduce `stage2_ab` typed config parsing (top-level only).
2) Update trainers to read only the typed representation.
3) Update `configs/stage2_ab/**` to the new schema.
4) Remove legacy Stage-2 AB config knobs and docs references.
5) Update tests to reflect the new semantics and knobs.

## Open Questions
- None in this proposal; the constitution decisions have been finalized in `progress/full_idea.md`.
