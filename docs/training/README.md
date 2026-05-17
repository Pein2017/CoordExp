---
doc_id: docs.training.index
layer: docs
doc_type: router
status: canonical
domain: training
summary: Router for Stage-1 and Stage-2 training documentation, metrics, and runbooks.
tags: [training, stage1, stage2]
updated: 2026-05-16
---

# Training Docs

Open this folder when you need current training behavior, recommended configs,
or metric interpretation.

## Current Training Surface Matrix

The unified training architecture is currently a guarded shadow contract, not a
wholesale replacement for every live launcher. Use it as the current design
direction and validation surface when adding new training behavior. It has
closed top-level domains:

```text
run, surface, data, template, supervision, objectives, observability, artifacts, runtime
```

`experimental` is the only optional top-level domain and requires an explicit
owner/expiry/opt-in. Supported `surface.id` values are:

- `stage1_json_ce`: JSON chat CE baseline.
- `stage1_compact_trie_ce`: primary Stage-1 compact-full direction with
  token-span supervision and trie/coordinate objectives.
- `stage2_two_channel`: Stage-2 two-channel shadow architecture.

| Surface | Status | Primary config / route | Packing status | Notes |
|---|---|---|---|---|
| Stage-1 JSON CE | Current baseline and shadow `surface.id: stage1_json_ce` | `configs/stage1/sft_base.yaml`, shared Stage-1 profiles, and `src/training/pipelines/stage1_json_ce.py` | Static packing where supported | JSON chat CE remains the baseline/regression surface; do not treat it as the compact-full target architecture. |
| Stage-1 compact recursive detection | Primary Stage-1 direction and shadow `surface.id: stage1_compact_trie_ce` | `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`; runtime policy in `src/detection/runtime.py`; shadow pipeline in `src/training/pipelines/stage1_compact_trie_ce.py` | Packing/cache fail fast for latest compact recursive CE surfaces until sidecar target-position offset rewriting is implemented and validated | Compact-full is the default Stage-1 direction: token-span supervision plus `token_ce`, `trie_ce`, `coord_soft_ce`, and `box_regression` objective profiles. Disabled objectives stay explicit in the profile. |
| Stage-1 compact recursive detection geometry-aware softCE | A5/A6 ablation candidates | `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml`; `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml` | Same as latest compact recursive CE | Both extend the A2/support2 comparator, keep compact-full/data/model/optimizer/schedule unchanged, and replace coordinate hard CE with `objective.coord_soft_ce`; production intent is two concurrent 4-GPU jobs, one for A5 and one for A6, after tiny/DDP4 smoke and risk audit. |
| Stage-1 compact prefix roll-in ET-RMP-CE | E1 ablation/smoke route | `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml` | Packing/cache disabled; recursive sidecar offset rewriting is not implemented | Compact-full only. EOS supervision uses ordinary teacher-forced `<|im_end|>` CE. |
| Stage-1 compact detection bridge | Legacy bridge only | `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml` | Legacy SFT smoke surface; not a latest packing example | Uses legacy `TrainingConfig` plus `custom.detection_sequence_format`; do not use as a latest-schema example. |
| Stage-2 two-channel | Active Stage-2 operator path and shadow `surface.id: stage2_two_channel` | `configs/stage2_two_channel/`; shadow pipeline in `src/training/pipelines/stage2_two_channel.py` | Post-rollout trainer packing when configured; rollout generation remains unpacked | YAML-first Channel-A plus clean-prefix Channel-B training. New planning direction is duplicate filtering before greedy-IoU assignment and Channel-B false-negative insertion. |
| Stage-2 rollout-aligned | Supported compatibility variant | `custom.trainer_variant: stage2_rollout_aligned` with `rollout_matching.pipeline.*` | Compatibility path | Do not author `stage2_ab.pipeline.*` for this variant. |
| Runtime fusion config | Dormant legacy surface | `configs/fusion/` examples only | Not part of supported training authoring | Merge JSONLs offline for multi-dataset training today. |

Current cleanup decisions:

- New shadow surface configs reject removed training mechanisms anywhere in the
  payload, including duplicate-burst unlikelihood, adjacent repulsion,
  EOS-loosen/trust/weighted-loss variants, continuation forcing, separator
  forcing, and stop-signal gate/damping variants.
- Historical diagnostics, old artifacts, absence tests, and compatibility
  readers may still mention those names. Current guidance must not recommend
  them as active training strategy.
- Objective profiles are keyed in YAML-like authoring, but resolve in canonical
  order: `token_ce`, `trie_ce`, `coord_soft_ce`, `box_regression`.

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
- `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml` remains the random-permutation ET-RMP-CE production baseline/comparator.
- `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml` and `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml` are unlaunched geometry-aware coordinate softCE ablation candidates that preserve the A2/support2 setup except for `objective.coord_soft_ce` and run identity.
- `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml` is the first `prefix_rollin_et_rmp_ce` ablation route; do not describe it as production-ready.
- `configs/stage1/compact_detection_sequence/` is a legacy bridge around
  `TrainingConfig` plus `custom.detection_sequence_format`.
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

`src/sft.py` delegates these policies to `src/detection/runtime.py`; removed
legacy set-continuation routes are rejection-only and are not executable
compatibility aliases. This surface remains config-first: prefix-rollin adds
latest-detection objective subkeys, but no new CLI flags.

## Page Roles

- [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md)
  - Stage-1 objective surfaces, coord-token training details, compact recursive detection status, and retired candidate-objective boundaries
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
- "How do I distinguish compact recursive detection and baseline SFT evidence?"

## Code Handles

- `src/sft.py`
- `src/config/schema.py::LatestDetectionTrainingConfig`
- `src/detection/runtime.py`
- `src/detection/template.py`
- `src/common/detection_sequence.py`
- `src/common/detection_compact_rows.py`
- `src/bootstrap/`
- `src/training/surfaces.py`
- `src/training/pipelines/`
- `src/training/objectives/`
- `src/training/observability/`
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
