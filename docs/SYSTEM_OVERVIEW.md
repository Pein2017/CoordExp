---
doc_id: docs.system-overview
layer: docs
doc_type: overview
status: canonical
domain: repo
summary: End-to-end current flow from data intake to CoordExp-Swift training, inference, evaluation, and artifacts.
updated: 2026-08-20
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
  -> src/losses/ + Accelerate replicated DDP + src/runtime/
  -> src/training/supervised_trainer.py
  -> rank-zero RunWriter + synchronized CheckpointWriter (+ opt-in exact state)

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

`src/losses/runner.py` assembles the configured terms and emits raw/weighted
values, denominators, and finite-status diagnostics over the closed term
inventory declared in `src/losses/bindings.py`. The supervised objective is
SFT-only and strict: protected base CE plus the protected token-type gate,
with coordinate Gaussian/RPS as a typed optional auxiliary. Each term declares
one zero policy — base CE forbids a zero weight, the gate's named
zero-weight ablation keeps a detached diagnostic outside the autograd graph
while remaining part of the all-rank pre-backward safety decision, and the
coordinate auxiliary is omitted entirely at weight zero. See
[`COORDEXP_SWIFT.md`](COORDEXP_SWIFT.md#supervised-loss-contract) for the
authored shape. Do not describe old recursive-detection or rollout-matching
trainers as current Swift V1 ownership.

## Training assembly and runtime

`src/train.py` requires a config and delegates to the training facade
`src/training/pipeline.py`. The facade builds an immutable model-free
execution plan (`src/training/execution_plan.py`), opens the pre-model rank
control plane (`src/training/control_plane.py`), initializes the rank-zero
`RunWriter`, admits cache preparation and hydration
(`src/training/cache_workflow.py` with `src/training/cache_contract.py`), and
constructs exactly one `TrainingSession` (`src/training/session.py`). The
session loads Qwen, installs adapters and selected-token embedding deltas,
assembles losses and optimizer/scheduler, creates the Accelerate runtime,
owns exact-resume choreography and the forward-input provider lifetime,
registers eval/checkpoint handlers, and runs `SupervisedTrainer`.
`src/training/reporting.py` owns completed-step rows.

One completed observation passes through four owners and no generic
coordinator. `src/runtime/optimizer_boundary.py` converges the all-rank
optimizer-boundary decision and returns the single update receipt that
distinguishes a wrapper attempt from an applied update.
`src/runtime/metrics.py` reduces the step's scalars across ranks under
producer-declared reducers, failing closed on any metric with no declared
reducer instead of averaging it. `src/training/reporting.py` builds the
canonical row, and `src/artifacts/observation_publisher.py` publishes it on
rank zero and only then feeds the derived console and TensorBoard sinks.
`observability.steps` is a required presentation interval that controls those
sinks alone; it never suppresses a canonical row.

`src/training/supervised_trainer.py` owns planned-step and micro-step iteration
through explicit runtime and loss interfaces. The only training backend is
Accelerate replicated DDP with one process per rank; there are no separate
single-process or DeepSpeed modes. `src/runtime/train_runtime.py` owns device
movement, accumulation, distributed operations, finite gates,
gradient clipping, optimizer/scheduler stepping, and safe artifact writes.

Forward-input preparation is owned by
`src/training/forward_input_provider.py` and selected only by the strict
config field `training.forward_input_provider_mode`. `synchronous` is the
default reference path; `overlapped` is an explicit experimental selection
with depth-one, CPU-only producer semantics and identical prepared inputs.
No environment variable may replace the authored mode.

## Inference and evaluation

`src/infer.py` delegates to `src/inference/pipeline.py`. The pipeline resolves
the inference config, writes resolved config artifacts, loads input rows,
plans optional data-parallel shards, runs direct inference, writes shard
artifacts, and merges them. `src/inference/runtime.py` loads the shared
processor-only frontend and projects strict config into a backend-neutral
launch contract. `src/inference/backend.py` owns semantic requests/results,
likelihood semantics, validation, and session lifecycle.
`src/inference/hf_backend.py` dynamically composes the base Qwen model,
optional DoRA adapter, and optional selected-token embedding delta.
`src/inference/execution_model.py` materializes the same composition into an
immutable snapshot for `src/inference/vllm_backend.py`. Both backends preserve
the same prompt, parser, scoring, artifact, and evaluator contracts. vLLM
launch authorization comes from current snapshot validation, engine
construction, and live decode evidence. Historical qualification receipts and
FP32 HF/vLLM comparisons are optional diagnostics for the claims they measure.
The execution snapshot independently binds DoRA merge and selected-token delta
fold outcomes. vLLM shard receipts retain live-decode and cleanup observations,
which are aggregated per rank instead of entering the cross-rank semantic
identity comparison.

`src/inference/parsing.py` owns best-effort parser diagnostics and
`src/inference/scoring.py` owns selected-token scoring. The evaluator does not
reparse raw text. `src/eval/detection_consumer.py` validates raw/scored row and
provenance binding, converts inline GT norm1000 boxes to pixel `xyxy`, keeps
empty-prediction rows visible as false negatives, and writes COCO artifacts and
mAP/mRecall metrics.

## Artifact and handoff flow

Training artifacts are initialized and finalized on rank zero by
`src/artifacts/run_writer.py`; every rank participates in the synchronized
`CheckpointWriter` choreography in `src/artifacts/checkpoints.py`, while rank
zero stages and publishes the payload. `src/artifacts/checkpoint_payload.py`
binds that inference payload, and `src/artifacts/training_state.py` publishes
the opt-in exact training-state sibling through the checkpoint callback seam.
Inference artifacts are written by
`src/inference/artifacts.py`; evaluation artifacts are written by
`src/eval/detection_consumer.py`. The canonical inventory is
[`ARTIFACTS.md`](ARTIFACTS.md).

Scalar observation is JSONL-first: `logging.jsonl` is the durable authority,
and the rank-zero console line and the run-local `tensorboard/` event files are
derived from a row that is already published. The console ETA is an approximate
estimate that is never persisted or restored. A derived-sink failure is
latched with one bounded warning and never invalidates a published row. A
terminal optimizer boundary publishes one row at its current planned-step id
and then fails the run without publishing exact-resume state, a checkpoint, a
best-selector update, or a successful final artifact.

The minimal inference payload is a standard PEFT adapter plus an optional
separate selected-token embedding delta, both loaded by explicit inference
paths and bound by an inference payload manifest; it is always published.
Exact training continuation is a separate opt-in `training_state/` sibling,
disabled by default and byte-identical to the previous behavior when disabled.
When enabled, the inference payload commits first, the sibling is published
next, and `final.json`/`best.json` update only after every publication required
by the selected checkpoint mode has committed. Continuation is supported only
at an optimizer-step save boundary with the same world size and rank map, where
it restores step, pack cursor, optimizer, scheduler, scaler, and RNG state;
admission is fail-closed before any mutable restore and inference readers
ignore the sibling. Cross-world-size and mid-accumulation resume are
unsupported, and this is neither a performance claim nor a production-launch
claim.

Pack cache v3 remains outside the run tree. Preparation publishes only to a
previously absent version/fingerprint target; it does not repair, replace,
delete, or garbage-collect an existing cache. Only compact train/eval cache
bindings are retained in `run.json`.

## Authority and historical boundary

Use [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for precedence,
[`IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md) for targeted source/test
routing, and the relevant `coordexp-swift-*` stable spec for normative details.
Old `src/sft.py`, `src/trainers/`, `src/datasets/`, `src/detection/`, and
`src/infer/` references are historical or comparator-only. Old plans and
architecture proposals are not current behavior authority and are routed by
[`architecture/README.md`](architecture/README.md).
