## Context

This change is an audit + hardening pass over the end-to-end training pipeline described in:
- `progress/full_idea.md` (semantic baseline),
- Stage-1 launcher: `scripts/train.sh` + `configs/stage1/ablation/geometry_first_coco80.yaml`,
- Stage-2 AB server-mode launcher: `scripts/train_stage2.sh` + `configs/stage2_ab/prod/ab_mixed.yaml`.

Stage-2 AB adds an external vLLM rollout server (via `swift rollout`) plus packed teacher-forced learning
under two channels (A/B). Correctness depends on strict boundaries:
data contract -> chat-template render -> tokenize -> pack -> forward/loss masks -> metrics/eval -> artifacts.

This pipeline also depends on upstream library contracts (ms-swift, transformers, vLLM, torch). The audit
treats those dependencies as explicit integration surfaces: we verify versions/provenance are captured and
we add CPU-only contract tests for the specific upstream APIs/protocols CoordExp relies on.

Constraints (non-negotiable):
- Config-first (YAML); avoid adding new CLI hyperparameter flags.
- Preserve geometry invariants (never drop/reorder coords); training uses `do_resize=false`.
- Keep Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model internals (e.g., `modeling_qwen3_vl.py`).

Operational constraint for this investigation:
- GPU smoke runs are temporarily unavailable; we will prioritize CPU-only checks + unit tests now and defer
  GPU-dependent end-to-end runs as explicit tasks.

## Goals / Non-Goals

**Goals:**
- Produce a pipeline map (data -> transforms/packing -> training/inference -> artifacts) grounded in the
  operational entrypoints and the actual module owners in `src/`.
- Run a comprehensive audit/diagnosis pass with evidence handles (file path + symbol/line, config keys,
  or a test/command) for each risk area.
- Apply minimal fixes that improve correctness/reproducibility/eval validity, and add CPU-only tests that
  lock down invariants so regressions are caught early.
- Ensure Stage-2 AB rollout behavior is diagnosable with stable metrics and that known failure modes have
  deterministic fallbacks (no silent objective drift).

**Non-Goals:**
- No large architecture forks (no DETR-style heads, no bespoke RL loops, no custom vision backbones).
- No behavior changes that require new CLI flags.
- No broad refactors unrelated to correctness/reproducibility/eval validity.

## Decisions

1) **Audit method: breadth pass -> depth pass -> hardening**
   - Breadth: identify the true operational flow from configs + launchers + entrypoint wiring.
   - Depth: deep-dive the highest-risk boundaries (packing masks, template/tokenizer alignment, Stage-2
     rollout parsing + matching + FN injection, loss masking/weighting, checkpoint artifact contents).
   - Hardening: implement the smallest fixes + tests to prevent recurrence.

2) **Evidence gate**
   - No audit claim is “done” unless it has a concrete handle:
     - `path:line` evidence, or
     - a config key path, or
     - a test/command with an expected outcome.

3) **CPU-first verification now; GPU smokes later**
   - Add/extend unit tests and lightweight tools that validate invariants without GPUs.
   - Encode GPU-dependent operational smokes as deferred tasks with exact commands and pass criteria.

4) **Spec-aligned, config-first changes**
   - Prefer tightening validation and adding tests over changing semantics.
   - If a true requirement change is needed, capture it explicitly in a follow-up spec change.

## Risks / Trade-offs

- **[Risk] Some failures only reproduce under GPU/vLLM load (deadlocks, throughput collapse).**
  -> Mitigation: strengthen CPU contract tests (DDP control-flow invariants, request chunking, parsing)
  and defer a small number of operational smokes with explicit acceptance checks.

- **[Risk] Making validation stricter can cause existing runs to fail earlier.**
  -> Mitigation: limit fail-fast to objective-changing or eval-invalidating cases, and ensure errors are
  actionable (full dotted-path keys, suggested remediation).

- **[Risk] Adding diagnostics can bloat logs or add overhead.**
  -> Mitigation: keep diagnostics aggregate-only, bounded, and best-effort unless they affect the loss.
