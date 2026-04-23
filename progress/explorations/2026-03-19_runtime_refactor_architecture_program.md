---
doc_id: progress.explorations.runtime-refactor-program-2026-03-19
layer: progress
doc_type: exploration
status: completed-exploration
domain: repo-architecture
summary: Consolidated architecture diagnosis and completion checkpoint for the March 2026 runtime refactor program.
updated: 2026-04-23
---

# Runtime Refactor Architecture Program (2026-03-19)

This note merges the earlier architecture diagnosis draft and the later
completion checkpoint into one historical record for the March 2026 runtime
refactor program.

Prefer the current docs/spec layer for live structure:

- `docs/PROJECT_CONTEXT.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/IMPLEMENTATION_MAP.md`
- `openspec/specs/runtime-architecture-refactor-program/spec.md`

Use this note when you want the historical before/after story:

- what the risky seams were before the refactor,
- which components were actually extracted,
- and which validation bundles stayed green before the change was declared
  complete.

## Diagnosis Snapshot

The refactor was motivated by one dominant pattern: too much runtime,
target-building, and evaluation logic was concentrated in a few oversized
orchestration modules.

Highest-risk components at diagnosis time:

1. `src/trainers/stage2_rollout_aligned.py`
   - rollout backend lifecycle
   - target construction
   - evaluation orchestration
   - compatibility adapters used by other trainers
2. `src/trainers/stage2_two_channel.py`
   - Channel-B rollout prep and triage
   - multi-objective loss routing
   - mixed runtime and research semantics
3. `src/sft.py`
   - bootstrap, trainer setup, manifest projection, and provenance writing
4. `src/infer/engine.py` and `src/eval/detection.py`
   - broad orchestration surfaces that mixed backend-specific logic with
     artifact writing

The main design lesson from the diagnosis pass was to extract by
responsibility, not by arbitrary line count:

- shared rollout runtime
- rollout-aligned target/eval helpers
- bootstrap/provenance seams
- infer/eval artifact orchestration

## What Landed

By the completion pass on 2026-03-19, the refactor had landed the following
component splits while preserving public behavior.

### Shared Rollout Runtime

- `src/trainers/rollout_runtime/dispatch.py`
- `src/trainers/rollout_runtime/vllm_config.py`
- `src/trainers/rollout_runtime/vllm_engine.py`
- `src/trainers/rollout_runtime/vllm_server.py`
- `src/trainers/rollout_runtime/vllm_infer.py`

These modules took ownership of shared rollout backend dispatch, vLLM local and
server preparation, request planning, and sync helpers while keeping
trainer-facing adapters intact.

### Rollout-Aligned Target And Eval Extraction

- `src/trainers/rollout_aligned_targets.py`
- `src/trainers/rollout_aligned_evaluator.py`

The trainer kept compatibility wrappers, but target construction helpers and
evaluation reduction / artifact logic moved out of the main trainer body.

### Stage-2 Two-Channel Decomposition

- `src/trainers/stage2_two_channel/types.py`
- `src/trainers/stage2_two_channel/target_builder.py`
- `src/trainers/stage2_two_channel/rollout_views.py`
- `src/trainers/stage2_two_channel/objective_runner.py`
- `src/trainers/stage2_two_channel/coordination.py`
- `src/trainers/stage2_two_channel/executors.py`

This preserved the public trainer surface while separating payload types,
Channel-B target building, objective execution, and threaded coordination.

### Bootstrap, Manifest, And Provenance Ownership

- `src/bootstrap/pipeline_manifest.py`
- `src/bootstrap/trainer_setup.py`
- `src/bootstrap/run_metadata.py`

`src/sft.py` stayed the entrypoint, but manifest assembly, trainer
construction, and provenance writing no longer had to live inline.

### Infer And Eval Artifact Seams

- `src/infer/artifacts.py`
- `src/infer/backends.py`
- `src/eval/artifacts.py`
- `src/eval/orchestration.py`

The public infer/eval entrypoints were preserved while deeper artifact-writing
and orchestration responsibilities moved into more focused helpers.

## Key Preserved Components

The refactor intentionally kept these compatibility surfaces stable:

- trainer-facing rollout adapters in `stage2_rollout_aligned.py`
- rollout-aligned metric names and artifact names
- infer summary / output JSONL contracts
- detection evaluator output artifacts such as `metrics.json`,
  `per_image.json`, `per_class.csv`, `coco_gt.json`, `coco_preds.json`, and
  `vis_resources/gt_vs_pred.jsonl`
- operator-facing CLI and runbook entrypoints

## Validation Evidence

The completion pass stayed green on targeted parity bundles rather than a new
ad hoc smoke workflow.

High-signal validation bundles:

- `tests/test_stage2_rollout_aligned.py` plus rollout/runtime parity bundles
- `tests/test_stage2_ab_training.py`
- `tests/test_stage2_two_channel_training.py`
- `tests/test_stage2_ab_vllm_server_mode_smoke.py`
- `tests/test_unified_infer_pipeline.py`
- `tests/test_detection_eval_output_parity.py`
- `tests/test_detection_eval_ingestion_diagnostics.py`
- `tests/test_run_manifest_files.py`
- `tests/test_dependency_provenance.py`
- `tests/test_launcher_metadata_env.py`

Late completion summaries recorded:

- `172 passed, 2 skipped` on the rollout-aligned parity bundle
- `201 passed` on config / manifest / infer / eval artifact bundles
- `421 passed, 4 skipped` on the broader final parity bundle
- `openspec validate refactor-core-runtime-architecture --type change --strict --no-interactive`

## Retrieval Guidance

If you are asking:

- "what is the current runtime contract?"
  - read `docs/` and `openspec/specs/`
- "why were these seams extracted?"
  - read this note
- "where should I look first for repo/runtime historical architecture work?"
  - start with [README.md](README.md) in this folder
