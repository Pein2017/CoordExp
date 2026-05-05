---
doc_id: docs.training.index
layer: docs
doc_type: router
status: canonical
domain: training
summary: Router for Stage-1 and Stage-2 training documentation, metrics, and runbooks.
tags: [training, stage1, stage2]
updated: 2026-05-05
---

# Training Docs

Open this folder when you need current training behavior, recommended configs,
or metric interpretation.

## Current Training Surface Matrix

| Surface | Status | Primary config / route | Packing status | Notes |
|---|---|---|---|---|
| Stage-1 baseline SFT | Current baseline | `configs/stage1/sft_base.yaml` and shared Stage-1 profiles | Static packing where supported | Teacher-forced baseline without rollout-aware matching. |
| Stage-1 set-continuation ET-RMP-CE | Current legacy-compatible continuation surface | `configs/stage1/set_continuation/production.yaml` | Packing and eval packing disabled/rejected | Uses full-suffix teacher-forced rows and ET-RMP branch supervision; recorded metrics must keep exact scope labels. |
| Stage-1 compact recursive detection | Canonical latest compact detection | `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`; runtime policy in `src/detection/runtime.py` | Packing/cache fail fast for latest compact recursive CE surfaces until sidecar target-position offset rewriting is implemented and validated | Uses `LatestDetectionTrainingConfig` top-level sections; separate from set-continuation ET-RMP-CE and legacy Stage-1 SFT. |
| Stage-1 compact detection bridge | Legacy bridge only | `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml` | Legacy SFT smoke surface; not a latest packing example | Uses legacy `TrainingConfig` plus `custom.detection_sequence_format`; do not use as a latest-schema example. |
| Stage-2 two-channel | Active Stage-2 operator path | `configs/stage2_two_channel/` | Post-rollout trainer packing when configured; rollout generation remains unpacked | YAML-first Channel-A plus clean-prefix Channel-B training. |
| Stage-2 rollout-aligned | Supported compatibility variant | `custom.trainer_variant: stage2_rollout_aligned` with `rollout_matching.pipeline.*` | Compatibility path | Do not author `stage2_ab.pipeline.*` for this variant. |
| Runtime fusion config | Dormant legacy surface | `configs/fusion/` examples only | Not part of supported training authoring | Merge JSONLs offline for multi-dataset training today. |

## Read Order

1. [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md) for Stage-1 objective/status, retired candidate-branch objectives, ET-RMP-CE, and compact recursive detection distinctions
2. [../data/PACKING.md](../data/PACKING.md) for the current packing matrix, hard-cap behavior, and surface-specific packing support
3. [STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md) for current Stage-2 workflows, launcher patterns, and historical-context pointers
4. [LVIS.md](LVIS.md) for LVIS-specific dataset, prompt, Stage-2, and evaluation semantics
5. [METRICS.md](METRICS.md) for loss-key and logging interpretation
6. [`stage2-ab-training/spec.md`](../../openspec/specs/stage2-ab-training/spec.md) when exact `stage2_two_channel` stable contract semantics matter
7. [`rollout-matching-sft/spec.md`](../../openspec/specs/rollout-matching-sft/spec.md) when working on the supported `stage2_rollout_aligned` variant
8. [`runtime-architecture-refactor-program/spec.md`](../../openspec/specs/runtime-architecture-refactor-program/spec.md) when the question is about runtime ownership seams or compatibility-preserving refactors

## Compact Detection Sequence Contracts

These are source-owned contracts for the compact detection sequence work. They
document implemented entrypoints and compatibility seams only; they do not imply
that a benchmark, smoke, or validation run has completed.

- Canonical latest compact detection lives under
  `configs/stage1/recursive_detection_ce_latest/` and parses through
  `LatestDetectionTrainingConfig`.
- Latest authoring snippets live under `configs/_shared/latest_detection/` and
  use top-level `data`, `prompt`, `detection_template`, `token_rows`,
  `objective`, `packing`, `evaluation`, and `validation`. They are not consumed
  by canonical launch configs until the relevant `extends` chains are migrated.
- `configs/stage1/compact_detection_sequence/` is a legacy bridge around
  `TrainingConfig` plus `custom.detection_sequence_format`.
- Stage-1 set-continuation ET-RMP-CE is a separate continuation surface rooted
  at `configs/stage1/set_continuation/production.yaml`; it is not latest
  compact recursive detection.
- Strict template owner: `src/detection/template.py`.
- Factory-visible strict template IDs: `stage1_json_pretty` and `compact_full`.
- Compatibility facade: `src/common/detection_sequence.py`.
- Low-level stdlib-only row helper: `src/common/detection_compact_rows.py`.
- Helper/compatibility formats stay helper-only: `compact_no_desc`, `compact_no_bbox`, and `compact_min`.
- Compatible parsing returns `None` for malformed rows instead of raising through the common facade.
- Generation suffix handling preserves desc control characters while keeping the existing suffix behavior.

Latest compact recursive detection runtime policy is centralized in
`src/detection/runtime.py`:

- latest detection runtime support and preflight checks,
- recursive CE runtime config resolution,
- prompt/mode/custom shim resolution,
- `build_latest_detection_dataset`,
- packing/cache fail-fast policy for latest compact recursive CE surfaces.

`src/sft.py` delegates these policies to `src/detection/runtime.py` and keeps
backward-compatible private aliases for older imports. This extraction did not
introduce new CLI flags or config schema keys.

## Page Roles

- [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md)
  - Stage-1 objective surfaces, coord-token training details, ET-RMP-CE, compact recursive detection status, and retired candidate-objective boundaries
- [../data/PACKING.md](../data/PACKING.md)
  - surface-specific packing matrix, Stage-1 static packing contract, hard length cap, and fail-fast behavior for overlength atomic samples
- [STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md)
  - YAML-first runbook, smoke workflow, server-mode launcher entrypoints, and the active `stage2_two_channel` path
- [LVIS.md](LVIS.md)
  - LVIS federated-label design note plus migration guide for Stage-1, Stage-2, and evaluation
- [METRICS.md](METRICS.md)
  - canonical training metric and loss interpretation; metric claims must include exact scope

## Use This Router For

- "How does current Stage-2 work?"
- "Which Stage-1 surface should I run or compare?"
- "What should I read before touching Stage-1 or Stage-2 configs?"
- "What is the current Stage-1 packing and `global_max_length` contract?"
- "Which Stage-1 set-continuation objectives are retired versus promoted?"
- "How do I distinguish ET-RMP, compact recursive detection, and baseline SFT evidence?"

## Code Handles

- `src/sft.py`
- `src/config/schema.py::LatestDetectionTrainingConfig`
- `src/detection/runtime.py`
- `src/detection/template.py`
- `src/common/detection_sequence.py`
- `src/common/detection_compact_rows.py`
- `src/bootstrap/`
- `src/trainers/stage1_set_continuation/`
- `configs/stage1/set_continuation/`
- `configs/stage1/recursive_detection_ce_latest/`
- `configs/stage1/compact_detection_sequence/`
- `configs/_shared/latest_detection/` authoring snippets, not current launch inheritance
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/`
- `src/trainers/stage2_rollout_aligned.py`
- `src/trainers/rollout_aligned_targets.py`
- `src/trainers/rollout_aligned_evaluator.py`
- `src/trainers/rollout_runtime/`
- `src/trainers/rollout_matching/`
- `src/trainers/teacher_forcing/`
- `src/launchers/stage2_vllm_server.py`
- `configs/_shared/datasets/`
- `configs/_shared/prompts/`
- `configs/stage1/`
- `configs/stage2_two_channel/`
