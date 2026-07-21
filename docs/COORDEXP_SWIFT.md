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
| Runtime and optimization | `src/runtime/`, `src/optim/`, `src/adapters/` | Accelerate replicated-DDP operations, finite gates, optimizer/scheduler steps, adapter and selected-token trainable surfaces |
| Training artifacts | `src/artifacts/run_writer.py`, `src/artifacts/checkpoints.py` | Rank-zero run/config/log ownership and synchronized staged adapter-plus-delta checkpoints |
| Inference | `src/infer.py`, `src/inference/` | Resolve infer config, compose the model, decode, parse, score, shard, merge, and write provenance-bearing artifacts |
| Detection evaluation | `src/eval/detection_consumer.py` | Validate raw/scored binding, normalize geometry units, write COCO artifacts, and emit mAP/mRecall metrics |

There is no current `src/infer/` package route. Do not document or create one
as a sibling of `src/inference/`.

The stable contracts for these boundaries are
[`coordexp-swift-training-artifacts`](../openspec/specs/coordexp-swift-training-artifacts/spec.md)
and
[`coordexp-swift-infer-config-runtime`](../openspec/specs/coordexp-swift-infer-config-runtime/spec.md).

## Config routes

Current config roots are:

- `configs/coordexp_swift/prod/` for production-shaped training configs;
- `configs/coordexp_swift/smoke/` for small training checks;
- `configs/coordexp_swift/infer/` for inference configs.

Config loading is strict and schema-first. A runnable training config declares
`schema_version: 1`; extends resolution, path origins, resolved values, and the
config fingerprint are part of the runtime evidence. The training runtime is
Accelerate-only: one process per rank, replicated DDP, with no separate
single-process or DeepSpeed backend mode.

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
- Checkpoints contain a standard PEFT adapter and, when configured, a separate
  selected-token embedding delta. Inference loads both through explicit paths.
  No exact optimizer, scheduler, scaler, dataloader, iterator, or RNG training
  continuation is provided.
- Pack cache v2 is a rebuild-only internal cache outside the run tree. The run
  records only compact immutable train/eval materialization bindings.

Stable semantics are owned by these specs:

- [config runtime](../openspec/specs/coordexp-swift-config-runtime/spec.md)
- [data/template/encoding](../openspec/specs/coordexp-swift-data-template-encoding/spec.md)
- [packing/forward](../openspec/specs/coordexp-swift-packing-forward/spec.md)
- [supervision/losses](../openspec/specs/coordexp-swift-supervision-losses/spec.md)
- [adapter and selected-token payloads](../openspec/specs/coordexp-swift-adapters-embeddings-optim/spec.md)
- [inference pipeline](../openspec/specs/coordexp-swift-infer-pipeline/spec.md)
- [inference backend trace](../openspec/specs/coordexp-swift-infer-backend-trace/spec.md)
- [inference execution model](../openspec/specs/coordexp-swift-infer-execution-model/spec.md)
- [inference scoring artifacts](../openspec/specs/coordexp-swift-infer-scoring-artifacts/spec.md)
- [detection evaluator](../openspec/specs/coordexp-swift-detection-evaluator/spec.md)

## Current evaluation boundary

Inference supports both dynamic HF and offline vLLM through one backend-neutral
session contract. HF remains the direct compatibility path for base plus DoRA
plus selected-token embedding delta. vLLM uses a content-addressed materialized
execution model because upstream vLLM does not load this DoRA composition
directly. The materializer validates the current base, DoRA, selected-token
delta, tied-weight structure, and published snapshot bytes before vLLM loads.
Optional materialized-HF and FP32 comparisons isolate composition or numerical
differences when a research claim needs them; they are not launch receipts.
BF16 vLLM is the normal high-throughput path. The backend session consumes
semantic multimodal decode requests and does not depend on evaluator or
inference-artifact orchestration. Future GRPO
or other post-training rollout code should reuse that session, execution-model,
likelihood, and worker-lifecycle boundary rather than introduce a second vLLM
engine wrapper. No GRPO trainer is implemented by this inference change.

The direct evaluator consumes `gt_vs_pred.jsonl`,
`gt_vs_pred_scored.jsonl`, `pred_token_trace.jsonl`, and the scored provenance
sidecar from the same artifact directory. It replays every prediction score
from the trace and checks manifest/provenance identity before writing COCO
artifacts and metrics.

The runtime records a compact `runtime_preflight`: installed vLLM version,
effective engine settings, execution-model identity, and optional historical
source evidence. Historical application-source hashes, exact engine values,
concurrency receipts, and composition-comparison sidecars are diagnostic only.
Current engine construction and the first contract-valid real multimodal
decode establish operational support. Raw likelihood uses a fresh live replay
and remains fail-closed on prompt, token, stop, length, finiteness, or alignment
mismatch. The older strict qualification probes remain available for explicit
repeatability, parity, or version studies.

Unknown vLLM versions may attempt policy-only inference, but raw-model
likelihood remains version-qualified because replay alignment cannot prove
where that version captures logits relative to the forcing processor. Every
composed execution-model receipt proves the configured DoRA merge and selected-
token embedding fold before atomic publication. Rank-local vLLM receipts retain
live-decode and engine-cleanup observations; multi-rank merge compares semantic
settings and aggregates those observations.

All model, data, adapter, and embedding-delta paths resolve to absolute paths
owned by the YAML file that authored them. Missing paths fail with declaring
config and resolved-path evidence; inference never searches another worktree
or output root automatically.
Merged rank-local decode rates are labeled capacity estimates: they assume
perfect rank overlap and are not controller-wall or end-to-end throughput.

The accepted two-rank BF16 production-mimic smoke completed one finite applied
step, one train row, one eval row, and one shared ten-file run tree with no
rank-local trees or legacy artifact families. Its 9,092 non-checkpoint bytes
are bounded implementation evidence, not a benchmark. A prior val200 result
may be useful as historical evidence, but its output directory is not part of
every checkout; reproduce or cite its receipt before making a current
benchmark claim. Full validation-dataset evaluation and official test-dev
submission are separate workflows.

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
