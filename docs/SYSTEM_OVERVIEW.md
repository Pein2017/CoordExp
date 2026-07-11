---
doc_id: docs.system-overview
layer: docs
doc_type: overview
status: canonical
domain: repo
summary: End-to-end current flow from data intake to CoordExp-Swift training, inference, evaluation, and artifacts.
updated: 2026-07-11
---

# System Overview

This page explains the current CoordExp-Swift flow. It is an explanatory
operator guide; exact compatibility-sensitive semantics belong to
`openspec/specs/`, and historical MS-Swift routes belong in historical docs.

## Flow at a glance

```text
validated coord JSONL + images
  -> src/data/
  -> src/templates/
  -> src/qwen/ encoding and forward helpers
  -> src/packing/ + src/supervision/
  -> src/losses/ + src/runtime/
  -> src/training/supervised_trainer.py
  -> src/artifacts/

inference config + checkpoint composition
  -> src/infer.py -> src/inference/
  -> raw/scored/provenance artifacts
  -> src/eval/detection_consumer.py
  -> COCO artifacts + metrics.json + evaluation_receipt.json
```

Current public entrypoints are `src/train.py`, `src/infer.py`, and the wrapper
`scripts/evaluate_detection.py` for scored detection evaluation. Current config
roots are under `configs/coordexp_swift/`.

## Data and encoding

`src/data/` owns typed raw examples, image references, dimensions, object
identity, descriptions, and geometry validation. `src/templates/` owns prompt
and assistant rendering plus semantic spans. `src/qwen/encoding.py` applies the
chat template, tokenizes with offsets, accounts for image-pad positions, and
aligns spans to physical token positions.

The source-order contract is explicit: `source_order` preserves authored order;
`geo_sorted` asserts that authored rows are already top-to-bottom then
left-to-right. It is not an implicit sort. Offline preparation and data
contract details live in [`data/CONTRACT.md`](data/CONTRACT.md) and
[`data/PREPARATION.md`](data/PREPARATION.md).

## Packing, supervision, and loss

`src/packing/planner.py` builds no-padding concatenative segments under
`global_max_length`; `src/packing/supervision.py` remaps logical supervision
spans to packed positions. `src/supervision/tokens.py` carries token type,
object/field identity, target position, causal-logit position, and optional
coordinate targets.

`src/losses/runner.py` assembles the configured terms and emits loss,
denominator, and finite-status diagnostics. The current source route exposes
base CE, optional token-type gating, and optional coordinate Gaussian/RPS; do
not describe old recursive-detection or rollout-matching trainers as current
Swift V1 ownership.

## Training assembly and runtime

`src/train.py` requires a config and delegates to
`src/training/pipeline.py`. The pipeline resolves the typed config, initializes
the run artifact manager, loads Qwen, installs adapters and selected-token
embedding deltas, builds the pack cache and schedule, assembles losses and
optimizer/scheduler, creates the runtime, registers eval/checkpoint handlers,
and runs `SupervisedTrainer`.

`src/training/supervised_trainer.py` owns planned-step and micro-step iteration
through explicit runtime and loss interfaces. `src/runtime/train_runtime.py`
owns device movement, accumulation, distributed operations, finite gates,
gradient clipping, optimizer/scheduler stepping, and safe artifact writes.

## Inference and evaluation

`src/infer.py` delegates to `src/inference/pipeline.py`. The pipeline resolves
the inference config, writes resolved config artifacts, loads input rows,
plans optional data-parallel shards, runs direct inference, writes shard
artifacts, and merges them. `src/inference/runtime.py` composes the base Qwen
model, optional adapter, optional selected-token embedding delta, and validated
checkpoint handoff identity. `src/inference/backend.py` provides the current HF
generation adapter and trace normalization boundary.

`src/inference/parsing.py` owns best-effort parser diagnostics and
`src/inference/scoring.py` owns selected-token scoring. The evaluator does not
reparse raw text. `src/eval/detection_consumer.py` validates raw/scored row and
provenance binding, converts inline GT norm1000 boxes to pixel `xyxy`, keeps
empty-prediction rows visible as false negatives, and writes COCO artifacts and
mAP/mRecall metrics.

## Artifact and handoff flow

Training artifacts are initialized by `src/artifacts/manager.py`; checkpoints
are written by `src/artifacts/checkpoints.py`; handoff validation is read-only
in `src/artifacts/checkpoint_handoff.py`. Inference artifacts are written by
`src/inference/artifacts.py`; evaluation artifacts are written by
`src/eval/detection_consumer.py`. The canonical inventory is
[`ARTIFACTS.md`](ARTIFACTS.md).

The handoff binds model, processor, template, adapter, selected-token payload,
and resolved-config identities. It is not an exact optimizer/scheduler/scaler/
dataloader/iterator/RNG resume contract.

## Authority and historical boundary

Use [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for precedence,
[`IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md) for targeted source/test
routing, and the relevant `coordexp-swift-*` stable spec for normative details.
Old `src/sft.py`, `src/trainers/`, `src/datasets/`, `src/detection/`, and
`src/infer/` references are historical or comparator-only. Old plans and
architecture proposals are not current behavior authority and are routed by
[`architecture/README.md`](architecture/README.md).
