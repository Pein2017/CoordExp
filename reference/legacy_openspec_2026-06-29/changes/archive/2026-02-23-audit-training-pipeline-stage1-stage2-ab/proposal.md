## Why

CoordExp stage_1 + stage_2 AB training spans strict JSONL contracts, Qwen3-VL chat templating (CoordJSON),
packing, custom coord-token objectives, and Stage-2 rollout-matching with an external vLLM server.
Small mismatches (tokenizer/template drift, packing masks, objective masking, rollout parsing/closure,
checkpoint contents) can silently invalidate learning dynamics or evaluation.

This change introduces a comprehensive, evidence-backed audit and a hardening pass (minimal fixes + tests)
to ensure the implemented pipeline matches `progress/full_idea.md` and existing OpenSpec contracts for
reproducible, paper-ready runs.

## What Changes

- Add a comprehensive end-to-end audit checklist scoped to the operational entrypoints:
  - `scripts/train.sh` (stage_1)
  - `scripts/train_stage2.sh` (stage_2 AB server-mode launcher)
  - `configs/stage1/ablation/geometry_first_coco80.yaml`
  - `configs/stage2_ab/prod/ab_mixed.yaml`
- Perform a diagnosis pass across:
  - data ingestion/cooking,
  - chat-template rendering/tokenizer compatibility,
  - packing + masking + attention,
  - forward/loss composition (Stage-1 and Stage-2 A/B),
  - metrics/eval triggers,
  - checkpoint artifact contents,
  - Stage-2 AB vLLM server-mode stability + performance.
- Expand the checklist to cover upstream integration contracts (ms-swift / transformers / vLLM / torch),
  and record upstream provenance in run artifacts so results remain auditable across dependency drift.
- Apply minimal code changes to fix correctness/reproducibility/eval-validity issues discovered, prioritizing:
  - fail-fast for objective-changing misconfig/misalignment,
  - improved diagnostics where ambiguity currently hides issues,
  - unit tests that lock down invariants without requiring GPUs.
- Defer GPU-dependent operational smoke runs until resources are available; encode them as explicit follow-up tasks.

## Capabilities

### New Capabilities

- `training-pipeline-audit`: A reproducibility-grade audit + verification harness for stage_1 + stage_2 AB
  training that codifies the invariants implied by `progress/full_idea.md` and existing OpenSpec specs,
  and requires CPU-only unit-test coverage for the highest-risk boundaries.

### Modified Capabilities

- (none; this change aims to align implementation with existing capability specs. If requirement changes
  are needed, open a follow-up OpenSpec change.)

## Impact

- Code: `src/sft.py`, `src/config/*`, `src/datasets/*`, `src/trainers/*`, `src/metrics/*`,
  plus targeted scripts/tooling under `scripts/tools/`.
- Tests: add/extend unit tests under `tests/` to cover packing/templating/loss-masking/rollout contracts.
- Run artifacts: extend run manifest metadata to include upstream dependency provenance and rollout-server launch flags.
- Docs: reconcile operator-facing entrypoints (e.g., server-mode launcher naming) so docs match reality.
- No new CLI flags; runtime behavior remains YAML-driven.
