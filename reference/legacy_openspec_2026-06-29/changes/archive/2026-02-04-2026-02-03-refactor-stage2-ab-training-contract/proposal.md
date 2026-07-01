# Change: Refactor Stage-2 AB training contract (unroll Channel-A, FP/stop-neutral Channel-B, typed config, CIoU)

## Why
Stage-2 AB is the core “paper-ready” training loop for CoordExp, but the current implementation and its YAML/config contract have drifted from the governing constitution in `progress/full_idea.md` (especially around Channel-A unroll, Channel-B FP/stop neutrality, and geometry loss defaults).

This change proposal defines a single, coherent training contract so that:
- every behavior is driven by explicit YAML knobs (typed schema),
- training semantics are reproducible across runs/checkpoints,
- and Stage-2 AB aligns with the updated “FP-neutral + stop-neutral Channel-B” philosophy without sacrificing Stage-1 stability.

## What Changes
- **BREAKING (config):** Introduce a typed top-level `stage2_ab` config section (parallel to `training` and `custom`).
- Remove all legacy Stage-2 AB config compatibility behaviors (treat legacy knobs as if they never existed).
- Replace the current `schedule.pattern: ["A","B",...]` with a deterministic float schedule: `stage2_ab.schedule.b_ratio: float`.
- Align Channel-A with the constitution:
  - **Default** `softctx_grad_mode: unroll` (no detach anywhere in the soft self-context loop).
  - Compute CE anchor at **A1** (teacher-forced logits), and compute geometry from the final softctx iteration logits.
- Align Channel-B with the constitution:
  - **FP-neutral geometry:** matched-only geometric gradients; unmatched predicted objects receive no geometry loss.
  - **Stop-neutral CE:** mask the top-level JSON closing brace `}` and `<|im_end|>` from Channel-B CE so Channel-B does not supervise stop/continue decisions.
  - **FN append always:** FN objects are always appended in the B3 target, regardless of rollout quality.
  - **Strict-drop diagnostics:** invalid predicted object instances are dropped deterministically (no repair) but produce explicit diagnostics (counts + reason buckets) and may upweight B3 structure CE.
  - **Semantic-tolerant matched desc supervision:** apply desc CE in predicted order but gate matched objects by sentence-embedding similarity (do not penalize near-matches; correct only when semantically far).
- Replace bbox geometry losses:
  - **Remove GIoU** from Stage-2 AB losses and configs.
  - **Mandate CIoU + SmoothL1** for bbox regression, with explicit box canonicalization to avoid NaNs.
- Config parsing hygiene:
  - Remove hard errors for deprecated legacy coord-loss knobs (silently ignore them).

## Impact
- Affected specs:
  - **MODIFIED**: `stage2-ab-training` (update requirements to match `progress/full_idea.md`).
  - Note: `stage2-ab-training` was introduced in change `2026-01-27-add-stage2-ab-iterative-softctx-bboxonly` and is not yet present under `openspec/specs/`. When archiving changes into `openspec/specs/`, archive `2026-01-27-...` first, then apply this refactor change.
- Affected code (planned; not implemented in this proposal):
  - `src/trainers/stage2_ab_training.py` (core behavior changes: unroll, stop-neutral, semantic gating, CIoU, CE anchor split).
  - `src/config/schema.py`, `src/config/loader.py`, `src/sft.py` (typed `stage2_ab` config; remove legacy Stage-2 AB knobs).
  - `configs/stage2_ab/**` and `docs/training/STAGE2_RUNBOOK.md` (migration to new config contract and updated semantics).
