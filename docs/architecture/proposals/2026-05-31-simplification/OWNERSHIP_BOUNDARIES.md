---
doc_id: docs.architecture.ownership-boundaries
layer: docs
doc_type: architecture-proposal
status: proposal
domain: architecture
summary: Proposed subsystem ownership boundaries for CoordExp simplification.
updated: 2026-05-31
---

# CoordExp Subsystem Ownership Boundaries

## Purpose

This document defines proposed ownership boundaries for major CoordExp subsystems. It is intended to help implementation agents decide where new logic should live and where legacy logic should be migrated or retired.

The guiding rule is:

> A module may orchestrate across concepts, but each concept must have exactly one canonical owner.

## Lifecycle labels

All major mechanisms, entrypoints, config keys, and artifact names should be classified as one of the following.

| Label | Meaning | Allowed in active configs? | Requires owner/expiry? |
| --- | --- | --- | --- |
| `ACTIVE` | Canonical, documented, tested, reportable path. | Yes | Owner required. Expiry not required. |
| `COMPATIBILITY` | Temporary bridge for legacy configs, artifacts, or tests. | No, unless explicitly opted in. | Owner and expiry required. |
| `RETIRED` | Historical only. May appear in archived docs or explicit rejection tests. | No | No active owner. |

Compatibility code without an expiry should be treated as architecture debt.

## Proposed canonical owners

### 1. Training pipeline

**Canonical owner:** `src/training/`

**Current important files:**

```text
src/sft.py
src/training/surfaces.py
src/training/pipelines/
src/training_runtime/plan.py
src/training_runtime/preflight.py
src/bootstrap/
src/trainers/
```

**Should own:**

- Training surface resolution.
- Pipeline selection.
- Surface-specific dataset construction.
- Trainer class selection.
- Surface-specific collator choice.
- Packing/cache eligibility.
- Runtime preflight policy.
- token_embeddings_adapter / trainable token-row setup, once extracted from entrypoint code.
- Training artifact and manifest emission.

**Should not own:**

- Raw YAML parsing details.
- Detection sequence rendering/parsing internals.
- Image path semantics beyond consuming a resolved data plan.
- Backend decode implementation.
- Offline evaluation metric semantics.

**Boundary rule:**

A new training behavior should first declare which surface it belongs to. If it does not fit `stage1_json_ce`, `stage1_compact_trie_ce`, or `stage2_rollout_correction`, it must be behind an explicit `experimental` block with owner, expiry, and opt-in.

**Recommended target:**

```text
src/training/
  entrypoint.py
  plan.py
  runner.py
  surfaces.py
  pipelines/
    stage1_json_ce.py
    stage1_compact_trie_ce.py
    stage2_rollout_correction.py
  runtime/
    packing.py
    cache.py
    checkpointing.py
    preflight.py
  adapters/
    token_embeddings_adapter.py
    token_rows.py
  artifacts.py
  ms_swift_projection.py
```

### 2. Inference, decoding, and rollout pipeline

**Canonical owner:** `src/infer/`

**Current important files:**

```text
src/infer/pipeline.py
src/infer/runtime.py
src/infer/backend.py
src/infer/backend_sync.py
src/infer/backend_vllm_server.py
src/infer/constraints.py
src/infer/artifacts.py
scripts/run_infer.py
src/trainers/stage2_rollout_runtime.py
```

**Should own:**

- Prompt policy for inference and rollout decode.
- `DetectionDecodeRequest` and successor batch request types.
- Decode policy fingerprints.
- Model identity fingerprints.
- HF/vLLM backend projection.
- Trace normalization.
- Offline inference artifact loop.
- Shared rollout decode APIs used by Stage-2.
- vLLM server launch/sync helpers, if separated from CLI.

**Should not own:**

- Training objective construction.
- Stage-2 assignment or correction target logic.
- Dataset JSONL preprocessing.
- Offline metric definitions.

**Boundary rule:**

Stage-2 rollout should call `src.infer` through explicit request objects and backend specs. It should not pass an entire trainer object as the implicit source of runtime policy after migration.

**Recommended target:**

```text
src/infer/
  pipeline.py          # infer/eval/vis pipeline orchestration
  runtime.py           # DecodeRuntime, requests, results, fingerprints
  prompt.py            # prompt policy and parity checks
  backend.py           # HF/vLLM backend-neutral projection
  backend_hf.py        # optional split if backend.py stays too large
  backend_vllm.py      # optional split
  rollout.py           # rollout-facing decode adapter
  launch.py            # local vLLM server lifecycle
  artifacts.py
  constraints.py
```

### 3. Dataset and preprocessing pipeline

**Canonical owner:** `src/data/` after migration. Current transitional owner is `src/datasets/` plus `public_data/` scripts.

**Current important files:**

```text
public_data/scripts/
public_data/view_contracts.py
src/datasets/dense_caption.py
src/datasets/builders/jsonlines.py
src/datasets/geometry.py
src/detection/dataset.py
src/common/paths.py
```

**Should own:**

- Strict JSONL loading.
- DataView metadata resolution.
- Image-root resolution.
- Offline geometry contract validation.
- Canonical record IR.
- Dataset sampling and deterministic row ordering.
- Dataset-level cache fingerprints, once separated from training runtime policy.

**Should not own:**

- Runtime image resizing.
- Runtime bbox conversion between parameterizations.
- Prompt text selection beyond passing template IDs or prompt policy references.
- Training loss semantics.
- Evaluation metrics.

**Boundary rule:**

Prepared data must be validated, not repaired. If a record does not match the configured coordinate surface, bbox format, image dimensions, or metadata contract, runtime should fail before training/inference.

**Recommended target:**

```text
src/data/
  records.py          # CoordExpRecord, DetectionObject, GeometryValue
  jsonl.py            # strict JSONL loader and diagnostics
  views.py            # DataView metadata and image-root resolution
  image_paths.py      # canonical path resolver
  geometry.py         # validation and normalization checks
  dataset.py          # active dataset wrapper
  sampling.py
  cache.py            # dataset cache fingerprints, if not training-specific
```

### 4. Detection templates and sequence semantics

**Canonical owner:** `src/detection/template.py` and focused submodules under `src/detection/`.

**Current important files:**

```text
src/detection/template.py
src/common/detection_sequence.py
src/common/detection_compact_rows.py
src/detection/teacher_forcing/compact_full_policy.py
src/detection/tokenization.py
src/detection/token_types.py
```

**Should own:**

- Template IDs.
- Render/parse behavior.
- Token-role taxonomy.
- Render span events.
- Terminal and stop-marker semantics.
- Compact row render/parser definitions.
- Object field order semantics.

**Should not own:**

- Dataset file loading.
- Training pipeline selection.
- Decode backend policy.
- Evaluation metric computation.

**Boundary rule:**

All assistant serialization must pass through the canonical template owner. Helper formats such as `compact_no_desc`, `compact_no_bbox`, or `compact_min` should remain compatibility/helper formats and should not become strict factory IDs unless explicitly promoted.

**Recommended target:**

```text
src/detection/
  template.py
  templates/
    stage1_json_pretty.py
    compact_full.py
  spans.py
  parse.py
  token_roles.py
  compact_rows.py
```

### 5. Evaluation and metrics

**Canonical owners:** `src/eval/` for evaluation behavior, `src/metrics/` for metric identity and reduction.

**Current important files:**

```text
src/eval/detection.py
src/eval/detection_orchestrator.py
src/eval/detection_records.py
src/eval/detection_coco.py
src/eval/detection_lvis.py
src/eval/detection_f1ish.py
src/eval/detection_duplicate_guard.py
src/eval/artifacts.py
src/metrics/events.py
src/trainers/metrics/
```

**Should own:**

- Detection artifact ingestion.
- COCO/LVIS/F1-ish evaluation behavior.
- Duplicate guard reports.
- Per-image reports.
- Metric event creation for offline evaluation.
- Metric reducers, aliases, and compatibility flat names.

**Should not own:**

- Training forward-pass loss logic.
- Prompt construction.
- Decode backend execution.
- Dataset preprocessing.

**Boundary rule:**

Evaluation should consume canonical artifacts, not infer execution intent from arbitrary file names or legacy aliases. Compatibility artifact names should be explicit and temporary.

**Recommended target:**

```text
src/eval/
  detection.py              # compatibility facade only, eventually thin
  detection_orchestrator.py # durable orchestration
  records.py
  coco.py
  lvis.py
  f1ish.py
  duplicate_guard.py
  reports.py
  artifacts.py

src/metrics/
  events.py
  reducers.py
  aliases.py
  sinks.py
```

### 6. Configuration system

**Canonical owner:** `src/config/`

**Current important files:**

```text
src/config/loader.py
src/config/schema.py
src/training/surfaces.py
src/config/prompts.py
configs/
```

**Should own:**

- YAML loading and inheritance.
- Strict schema validation.
- Prompt policy resolution from config references.
- Projection into a typed `RunPlan`.
- Projection into ms-swift arguments at the boundary.
- Rejection of retired config keys.

**Should not own:**

- Runtime training execution.
- Dataset cache building.
- Decode backend calls.
- Evaluation metric computation.

**Boundary rule:**

Raw YAML dictionaries should not flow into deep execution code. Load config once, validate once, then pass typed plans.

**Recommended target public schema:**

```yaml
run:
surface:
data:
template:
supervision:
objectives:
observability:
artifacts:
runtime:
experimental:
```

**Compatibility handling:**

`custom.*` should become a projection detail rather than a public architectural center. During migration, legacy configs may continue to load, but new official configs should prefer the domain schema.

### 7. Experiment-specific code

**Canonical owner:** no permanent owner unless promoted to an active surface or objective. Temporary experiments should live behind `experimental` config with owner and expiry.

**Should own:**

- Experiment notes.
- Temporary opt-in flags.
- Migration-only adapters.
- Archived evidence and diagnostics.

**Should not own:**

- Default training behavior.
- Default inference behavior.
- Default metric names.
- Canonical data contracts.

**Boundary rule:**

An experiment may introduce a temporary path only if it declares:

```yaml
experimental:
  owner: ...
  expiry: ...
  notes: ...
  surface_or_pipeline_opt_in: true
```

If the experiment becomes the new default, it must be promoted into an active surface/objective/template/backend and removed from `experimental`.

## Active / compatibility / retired classification proposal

### Active

```text
Training surfaces:
- stage1_json_ce
- stage1_compact_trie_ce
- stage2_rollout_correction

Templates:
- stage1_json_pretty
- compact_full

Artifacts:
- gt_vs_pred.jsonl
- gt_vs_pred_scored.jsonl
- metrics.json
- summary.json
- provenance sidecars

Entry points:
- python -m src.sft --config ...
- scripts/run_infer.py --config ...
- scripts/evaluate_detection.py --config ...
- scripts/postop_confidence.py --config ...
```

### Compatibility

```text
- legacy run_infer CLI flags
- pred.jsonl fallback alias
- src.eval.detection facade behavior
- src.common.detection_sequence facade behavior
- compact_no_desc / compact_no_bbox / compact_min helper formats
- custom.* public config keys during migration
- dense_caption dataset class names during data-IR migration
```

### Retired

```text
- stage2_ab_training
- stage2_two_channel
- rollout_matching_sft
- stage2_rollout_aligned
- stage2_rollout_runtime as a public variant name
- runtime custom.fusion_config as a supported multi-dataset path
- dynamic pairing
- runtime image resizing for coordinate data
- runtime bbox conversion for prepared alternate bbox branches
- removed stop-gate / EOS-loosen / adjacent-repulsion / duplicate-burst-unlikelihood mechanisms
```

## Decision checklist for implementation agents

Before adding or changing code, answer these questions:

1. Which active surface owns this behavior?
2. Which canonical concept is being modified?
3. Is this active behavior, compatibility behavior, or an experiment?
4. Is there exactly one owner for the concept?
5. Does this introduce a second execution path for an existing concept?
6. Does this move raw YAML deeper into runtime code?
7. Does this silently repair data that should fail validation?
8. Does this create a new flat metric name without a `MetricEvent` identity?
9. Does this add a new config key without schema validation and tests?
10. Does this depend on a legacy artifact name without explicit compatibility gating?

If the answer to any of these is unclear, stop and update the plan before modifying source code.
