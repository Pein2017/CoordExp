---
name: coordexp-research-context
description: Use when CoordExp work needs broad research context, design history, empirical evidence, benchmark provenance, diagnostics, or a current-vs-historical read before implementation or audit.
---

# CoordExp Research Context

Use this skill to build a compact context pack from the current docs/spec layer plus the historical `progress/` layer.

## Primary Entry Points

1. `docs/AGENT_INDEX.md`
2. `docs/catalog.yaml`
3. `progress/index.yaml`
4. `docs/PROJECT_CONTEXT.md`
5. `openspec/specs/runtime-architecture-refactor-program/spec.md` for current runtime decomposition questions
6. `openspec/specs/rollout-matching-sft/spec.md` when the question is about the supported rollout-aligned Stage-2 variant
7. `progress/README.md`, then the matching category router under `progress/`

## Read Order

Start here:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. the relevant domain router under `docs/`
5. relevant `openspec/specs/*.md`
6. `openspec/specs/runtime-architecture-refactor-program/spec.md` when the question is about module ownership, launchers, artifacts, or runtime seams
7. `progress/index.yaml`
8. `progress/README.md`
9. the category router that matches the question
10. the specific historical notes you need

## Precedence

When two sources disagree, use:

1. `openspec/specs/`
2. `docs/`
3. `openspec/changes/<active-change>/`
4. `progress/`

## What To Produce

Build a short context pack with:

- current authoritative docs/specs
- current architecture/runtime seams (for example `src/bootstrap/`, `src/config/loader.py::ConfigLoader`, `src/datasets/geometry.py::{geometry_from_dict,transform_geometry}`, `src/trainers/stage2_coordination.py`, `src/trainers/stage2_two_channel.py::Stage2ABTrainingTrainer`, `src/trainers/stage2_rollout_aligned.py::RolloutMatchingSFTTrainer`, `src/trainers/{rollout_aligned_targets.py,rollout_aligned_evaluator.py}`, `src/trainers/rollout_runtime/`, `src/infer/pipeline.py::run_pipeline`, `src/infer/engine.py::InferenceEngine.infer`, `src/infer/artifacts.py`, `src/eval/detection.py::evaluate_and_save`, `src/eval/artifacts.py`)
- relevant implementation files or configs
- the exact historical notes that matter, with scope labels such as `val200`, `limit=200`, full-val, raw-text, coord-token, checkpoint ids, and GPU launch shape when relevant
- a small set of grep seeds for narrowing

## Progress Usage

Use `progress/` for:

- historical design directions
- diagnostics and audits
- benchmark evidence
- pretraining background

Do not use it as the primary source for current behavior.

Route progress notes by question type:

- `progress/diagnostics/` for root-cause analysis, mechanism studies, threshold sweeps, failure interpretation, and operator notes.
- `progress/benchmarks/` for measured run comparisons, checkpoint selection notes, scoreboards, and evaluation sweeps.
- `progress/explorations/` for architecture and implementation-planning history.
- `progress/directions/` for historical research direction and Stage-2 lineage.
- `progress/pretrain/` for Stage-1 foundation history.

## Research Workflow Patterns

- Keep one canonical note per cluster. Use supporting notes and copied artifact-side markdown only as evidence beneath the parent note.
- For benchmark claims, report the exact scope and surface. Do not compare `val200`, `limit=200`, proxy views, and full-val as if they were the same result.
- Prefer durable summary artifacts such as `summary.json`, `proxy_eval_bundle_summary.json`, `metrics.json`, `timing_summary.json`, manifests, and copied progress artifacts over stale runtime paths.
- When a run is repaired or rerun, identify the valid cell and keep the failed first attempt as provenance, not as the result.
- For shard/fanout recovery, rerun only missing or failed shards when possible, then verify merged summaries/manifests rather than trusting a log line.
- After extracting durable results from `temp/` or transient worktrees, clean up scratch when asked or when it is clearly no longer needed, while preserving stable artifact roots and provenance-bearing outputs.

## Helpful Reference

Open `references/grep-seeds.md` only when you need repo-wide search seeds.
