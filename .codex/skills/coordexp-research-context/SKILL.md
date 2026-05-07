---
name: coordexp-research-context
description: "Use when CoordExp work needs broad research context, design history, empirical evidence, benchmark provenance, diagnostics, or a current-vs-historical read before implementation or audit."
---

# CoordExp Research Context

Use this skill to build a compact context pack. It should connect current behavior to historical evidence without turning progress notes into contract truth.

## Authority Model

Current behavior comes from the repo docs spine:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. relevant domain docs under `docs/`
5. `openspec/specs/` only for stable compatibility-sensitive contracts
6. `openspec/changes/<active-change>/` only when explicitly in scope
7. `progress/` only for historical evidence, diagnostics, benchmark reports, or design derivation

Routing helpers:

- `docs/AGENT_INDEX.md` and `docs/catalog.yaml` for current routes
- `progress/index.yaml` and `progress/README.md` for history/evidence routes
- `docs/ARTIFACTS.md` for artifact and provenance questions

If current docs and progress disagree, answer current behavior from `docs/` and use `progress/` to explain how the project got there.

## When To Use

Use this for:

- current-vs-historical reads before a nontrivial change
- benchmark provenance, score comparisons, and checkpoint interpretation
- mechanism diagnosis and failure-history lookup
- research design lineage, abandoned directions, and why a guardrail exists
- preparing an audit or implementation context pack

Do not use it as a substitute for `coordexp-codebase` when the task is just finding current code entrypoints.

## Context Pack Contract

Produce a short pack with:

- Current answer: what is true now and which docs/specs make it authoritative.
- Code/config handles: exact files, symbols, configs, and artifact names likely to matter.
- Evidence: only the progress notes, benchmark summaries, manifests, or artifacts needed for the question.
- Scope labels: `tiny`, `val200`, `limit=200`, first-200, full-val, proxy view, raw-text, coord-token, bbox format, checkpoint id, launch shape.
- Risks and open questions: only the smallest set that blocks safe action.
- Search seeds: 2-5 targeted `rg` patterns when another agent needs to continue.

Keep it concise. Link to long notes instead of restating them.

## Progress Routing

Use `progress/` by evidence type:

- `progress/benchmarks/`: measured run comparisons, checkpoint selection, scoreboards, evaluation sweeps.
- `progress/diagnostics/`: root-cause analysis, mechanism studies, threshold sweeps, failure interpretation, operator notes.
- `progress/explorations/`: architecture and implementation-planning history.
- `progress/directions/`: historical research directions and Stage-2 lineage.
- `progress/pretrain/`: Stage-1 foundation and pretraining history.
- `progress/audits/`: temporary or removable audit evidence. Use only as supporting history; do not promote it into a durable codebase reference unless the user explicitly asks.

Prefer one canonical note per cluster. Treat copied artifact-side markdown and superseded notes as provenance beneath a parent note.

## High-Signal Code Handles

General runtime:

- `src/sft.py`
- `src/training_runtime/plan.py::resolve_training_runtime_plan`
- `src/bootstrap/experiment_manifest.py`
- `src/bootstrap/pipeline_manifest.py`
- `src/bootstrap/run_metadata.py`
- `src/bootstrap/trainer_setup.py`
- `src/config/loader.py`
- `src/config/schema.py`

Data and geometry:

- `src/datasets/geometry.py`
- `src/datasets/dense_caption.py`
- `src/datasets/builders/jsonlines.py`
- `src/datasets/encoded_sample_cache.py`
- `src/detection/dataset.py::DetectionTrainingDataset`
- `src/detection/packing.py`

Stage-1 compact/latest detection:

- `src/config/schema.py::LatestDetectionTrainingConfig`
- `src/config/schema.py::DetectionObjectiveConfig`
- `src/sft.py::_resolve_recursive_detection_ce_cfg`
- `src/sft.py::_assert_latest_detection_runtime_supported`
- `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`
- `configs/stage1/recursive_detection_ce_latest/smoke/`

Stage-1 set-continuation:

- `src/trainers/stage1_set_continuation/`
- `src/data_collators/stage1_set_continuation_collator.py`
- `configs/stage1/set_continuation/production.yaml`

Stage-2:

- `src/trainers/stage2_coordination.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/`
- `src/trainers/stage2_ab/`
- `src/trainers/stage2_rollout_aligned.py`
- `src/trainers/rollout_matching/`
- `src/trainers/rollout_runtime/`
- `src/trainers/teacher_forcing/`
- `src/launchers/stage2_vllm_server.py`

Infer/eval:

- `src/infer/pipeline.py::run_pipeline`
- `src/infer/engine.py::InferenceEngine.infer`
- `src/infer/backends.py`
- `src/infer/artifacts.py`
- `src/eval/detection.py::evaluate_and_save`
- `src/eval/confidence_postop.py`
- `src/eval/bbox_confidence.py`
- `src/eval/proxy_eval_bundle.py`
- `src/eval/artifacts.py`

## Benchmark and Evidence Rules

- Never compare `val200`, `limit=200`, proxy, first-200, and full-val as if they are the same scope.
- Report raw-text vs coord-token and bbox serialization when it affects interpretation.
- Preserve failed or repaired runs as provenance, not as the headline result.
- Prefer durable summaries and manifests over transient logs: `summary.json`, `metrics.json`, `proxy_eval_bundle_summary.json`, `timing_summary.json`, `resolved_config.json`, `run_metadata.json`, `pipeline_manifest.json`.
- For shard/fanout recovery, identify missing or failed shards and merged summaries instead of trusting a single success line.

## Helpful Reference

Open `references/grep-seeds.md` only when you need repo-wide search seeds.
