---
doc_id: docs.coordexp-swift
layer: docs
doc_type: canonical-implementation-guide
status: canonical
domain: repo
summary: Current routing guide for the CoordExp-Swift training, inference, evaluation, and artifact infrastructure.
tags: [coordexp-swift, training, inference, eval, routing]
updated: 2026-07-11
---

# CoordExp-Swift Canonical Infrastructure

CoordExp-Swift is the current implementation route on repository `main`. This
page describes the live source and config ownership; stable compatibility
semantics belong to the linked `coordexp-swift-*` OpenSpecs.

The public entrypoints are `src/train.py` and `src/infer.py`. Active feature
work may be developed in a named worktree, but a worktree or an old branch does
not change the authority of the current `main` checkout. See
[`BRANCH_AND_WORKTREE_POLICY.md`](BRANCH_AND_WORKTREE_POLICY.md) for checkout
boundaries.

## Current source topology

```text
training config
  -> src/train.py
  -> src/training/pipeline.py
  -> src/training/supervised_trainer.py
  -> src/data -> src/templates -> src/qwen -> src/packing
  -> src/supervision -> src/losses -> src/runtime -> src/artifacts

inference config
  -> src/infer.py
  -> src/inference/pipeline.py
  -> src/inference/runtime.py / src/inference/backend.py
  -> src/inference/artifacts.py
  -> src/eval/detection_consumer.py
```

The source map is intentionally a set of real modules, not a proposed
framework. Current ownership is:

| Surface | Owner | Responsibility |
| --- | --- | --- |
| Entry and assembly | `src/train.py`, `src/training/pipeline.py` | Resolve config and assemble model, adapters, embedding deltas, pack cache, schedule, losses, optimizer/scheduler, runtime, eval, checkpoints, and artifacts |
| Planned-step training | `src/training/supervised_trainer.py` | Iterate micro-steps and planned steps through explicit runtime and loss interfaces |
| Config | `src/config/loader.py`, `src/config/models.py`, `src/config/resolve.py` | YAML extends resolution, strict typed validation, path resolution, and resolved-config fingerprinting |
| Data and geometry | `src/data/` | Validate JSONL examples, images, object identity, descriptions, dimensions, and geometry |
| Template and spans | `src/templates/` | Render prompts/assistant content and expose semantic supervision spans |
| Qwen boundary | `src/qwen/` | Load model/processor/tokenizer, encode chat/image inputs, build positions, and run forward helpers |
| Packing and supervision | `src/packing/`, `src/supervision/` | Concatenate no-padding segments and map logical token atoms to physical positions |
| Losses | `src/losses/` | Assemble CE, token-type gating, optional coordinate Gaussian/RPS, normalization, and diagnostics |
| Runtime and optimization | `src/runtime/`, `src/optim/`, `src/adapters/` | Device/distributed operations, finite gates, optimizer/scheduler steps, adapter and selected-token trainable surfaces |
| Training artifacts | `src/artifacts/` | Run manifest, resolved config, metric streams, eval-forward summaries, checkpoints, and handoff identity |
| Inference | `src/infer.py`, `src/inference/` | Resolve infer config, compose the model, decode, parse, score, shard, merge, and write provenance-bearing artifacts |
| Detection evaluation | `src/eval/detection_consumer.py` | Validate raw/scored binding, normalize geometry units, write COCO artifacts, and emit mAP/mRecall metrics |

There is no current `src/infer/` package route. Do not document or create one
as a sibling of `src/inference/`.

## Config routes

Current config roots are:

- `configs/coordexp_swift/prod/` for production-shaped training configs;
- `configs/coordexp_swift/smoke/` for small training checks;
- `configs/coordexp_swift/infer/` for inference configs;
- `configs/coordexp_swift/deepspeed/` for the DeepSpeed helper config.

Config loading is strict and schema-first. A runnable training config declares
`schema_version: 1`; extends resolution, path origins, resolved values, and the
config fingerprint are part of the runtime evidence. Training runtime choices
are explicit (`single`, `accelerate`, or `deepspeed` where supported by the
schema). A helper config does not by itself establish production benchmark
support.

## Semantic boundaries that docs must preserve

- `source_order` preserves authored object order. `geo_sorted` validates the
  authored top-to-bottom/left-to-right order; it does not silently sort rows.
- `src/data/` validates raw records, `src/templates/` renders semantic spans,
  `src/qwen/encoding.py` aligns spans with physical Qwen tokens, and
  `src/packing/` remaps them into packed positions. Do not collapse these into
  one generic dataset owner.
- The active loss assembly is owned by `src/losses/runner.py`. Current source
  terms are base CE, optional token-type gating, and optional coordinate
  Gaussian/RPS; normalization and finite diagnostics are explicit.
- Inference score-bearing artifacts require backend-neutral trace evidence,
  selected-token score provenance, row binding, and identity fingerprints.
  Predictions alone are not sufficient COCO evidence.
- Swift GT boxes are inline norm1000 `xyxy`; scored predictions are parser-
  normalized pixel `xyxy`. The evaluator converts the GT side and rejects
  mixed-unit COCO sidecars.
- `checkpoint_handoff.json` is an identity and payload-composition seam. V1
  checkpoint metadata records optimizer, scheduler, scaler, dataloader,
  iterator, and RNG resume state as not saved; handoff is not exact training
  continuation.

Stable semantics are owned by these specs:

- [config runtime](../openspec/specs/coordexp-swift-config-runtime/spec.md)
- [data/template/encoding](../openspec/specs/coordexp-swift-data-template-encoding/spec.md)
- [packing/forward](../openspec/specs/coordexp-swift-packing-forward/spec.md)
- [supervision/losses](../openspec/specs/coordexp-swift-supervision-losses/spec.md)
- [training artifacts](../openspec/specs/coordexp-swift-training-artifacts/spec.md)
- [checkpoint handoff](../openspec/specs/coordexp-swift-checkpoint-handoff-readiness/spec.md)
- [inference pipeline](../openspec/specs/coordexp-swift-infer-pipeline/spec.md)
- [inference backend trace](../openspec/specs/coordexp-swift-infer-backend-trace/spec.md)
- [inference scoring artifacts](../openspec/specs/coordexp-swift-infer-scoring-artifacts/spec.md)
- [detection evaluator](../openspec/specs/coordexp-swift-detection-evaluator/spec.md)

## Current evaluation boundary

The implemented inference backend is HF generation through
`src/inference/backend.py`. vLLM fields are reserved and validated as
unimplemented in this source route. The direct evaluator consumes
`gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, and the scored provenance sidecar
from the same artifact directory, then writes COCO artifacts and metrics.

Tiny and two-row runs are implementation checks. A prior val200 result may be
useful as historical evidence, but its output directory is not part of every
checkout; reproduce or cite its receipt before making a current benchmark
claim. Full validation-dataset evaluation and official test-dev submission are
separate workflows.

## Historical boundaries

Old MS-Swift/mainline paths such as `src/sft.py`, `src/trainers/`,
`src/datasets/`, `src/detection/`, `src/infer/`, `configs/stage1/`, and
`configs/stage2/` remain in historical docs, archived configs, tests, and old
run evidence. They are not current Swift ownership. Completed rebuild changes
are preserved under `openspec/changes/archive/`; a named
`openspec/changes/<change>/` directory is the sole local workspace for bounded
code/config/docs or architecture work that benefits from durable lifecycle.
Delta specs are included only when a stable compatibility-sensitive contract
changes; internal refactors do not require invented normative deltas.
