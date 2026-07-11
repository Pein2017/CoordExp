---
doc_id: docs.training.stage2-runbook
layer: docs
doc_type: historical-runbook
status: historical-reference
domain: training
summary: Historical YAML-first runbook for legacy Stage-2 rollout-correction training and old-run interpretation.
updated: 2026-07-11
---

# Historical Stage-2 Training Runbook

> This page is historical MS-Swift/mainline material. It is preserved for
> contract archaeology and old-run interpretation, not as a current `main`
> entrypoint or launch authorization.

Current implementation work starts from [`../COORDEXP_SWIFT.md`](../COORDEXP_SWIFT.md),
the current source tree, current configs, and the relevant stable
`coordexp-swift-*` specs. The current Swift path does not route through the old
Stage-2 trainer, rollout server, or `src/infer/` package described by the legacy
runbook.

## Historical contract snapshot

The legacy route used `pipeline.id: stage2_rollout_correction`, rollout-prefix
plus GT/residual correction, YAML-first settings, and old trainer/backend
modules. Those names remain in archived configs and historical artifacts; their
presence here does not make them supported current behavior.

Historical source/config handles include:

- `configs/stage2/rollout_correction/`;
- `src/sft.py`;
- `src/trainers/stage2_rollout_correction.py`;
- `src/trainers/rollout_aligned_targets.py`;
- `src/trainers/rollout_aligned_evaluator.py`;
- `src/infer/` and its legacy backend/dispatch modules;
- `scripts/train_stage2.sh`.

## Historical references

- [`../history/training/STAGE2_DESIGN.md`](../history/training/STAGE2_DESIGN.md)
  for preserved design rationale;
- [`../history/superpowers/README.md`](../history/superpowers/README.md) for
  dated historical plans and handoffs;
- [`../../openspec/changes/archive/`](../../openspec/changes/archive/) for
  one-time archived contract projects.

Do not use an old Stage-2 document to infer current config schemas, backend
availability, checkpoint-resume semantics, or metric comparability. Verify
those claims against current source and stable specs before interpreting a
result.
