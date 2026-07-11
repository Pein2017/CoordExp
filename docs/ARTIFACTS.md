---
doc_id: docs.artifacts
layer: docs
doc_type: artifacts-reference
status: canonical
domain: repo
summary: Current CoordExp-Swift training, inference, evaluation, checkpoint, and provenance artifacts.
updated: 2026-07-11
---

# Artifacts And Provenance

This page inventories artifacts emitted by the current CoordExp-Swift source.
It summarizes names and ownership; stable compatibility details belong to the
linked OpenSpecs.

## Ownership map

| Surface | Owner |
| --- | --- |
| Training run, resolved config, and logging | `src/artifacts/run_writer.py` (rank zero) |
| Checkpoint payloads and aliases | `src/artifacts/checkpoints.py` (all-rank synchronization; rank-zero publication) |
| Inference rows, traces, manifests, and merge | `src/inference/artifacts.py`, `src/inference/merge.py` |
| Detection evaluation artifacts | `src/eval/detection_consumer.py` |

## Training artifacts

`RunWriter` initializes one shared run directory on rank zero. Current names
include:

- `run.json`: lifecycle, runtime world size, schedule/counters, warnings, and
  compact immutable train/eval cache-materialization bindings;
- `resolved_config.json`: the resolved config and fingerprint evidence;
- `logging.jsonl`: rank-zero-appended train and eval scalar rows;
- `checkpoints/step-<step>/adapter/`: standard staged PEFT adapter payload;
- `checkpoints/step-<step>/special_token_embeddings/`: optional selected-token
  embedding-delta metadata and safetensor payload;
- `checkpoints/final.json` and `checkpoints/best.json`: checkpoint selectors.

The training run does not emit `run_manifest.json`, per-split metric streams,
per-step receipts, `checkpoint_handoff.json`, or `checkpoint-final` aliases.
It also does not save these exact-resume surfaces:

- optimizer;
- scheduler;
- scaler;
- dataloader;
- iterator;
- RNG.

Therefore checkpoints support explicit adapter-plus-delta model composition;
they do not claim exact training-state resume. Pack cache v2 is rebuild-only,
lives outside the run tree, and contributes only compact bindings to `run.json`.

## Inference artifacts

`src/inference/artifacts.py` writes the inference artifact family in a run or
shard directory:

- `gt_vs_pred.jsonl`: raw per-row GT, parsed predictions, parser status, and
  row identity;
- `gt_vs_pred_scored.jsonl`: score-bearing predictions with selected-token
  evidence;
- `gt_vs_pred_scored.jsonl.provenance.json`: raw/scored hashes, row binding,
  model/processor/tokenizer identity, prompt/template/decode fingerprints, and
  score-policy fingerprint;
- `pred_token_trace.jsonl`: generated-token IDs, text, and logprob trace when
  tracing is materialized;
- `parse_diagnostics.jsonl`: parser/drop diagnostics;
- `image_plan.jsonl`: image planning evidence;
- `summary.json`: terminal inference summary;
- `run_manifest.json`: artifact names, identity fingerprints, backend and
  generation policy, loaded base/adapter/delta identity, and terminal/benchmark eligibility
  fields.

Data-parallel inference also writes shard metadata and a merge plan. The merge
must preserve row identity, artifact hashes, and the resolved-config
fingerprints.

## Detection evaluation artifacts

`src/eval/detection_consumer.py` requires the raw and scored artifacts plus the
scored provenance sidecar from the same directory. It writes:

- `metrics.json`;
- `evaluation_receipt.json`;
- `coco_gt.json`;
- `coco_predictions.json`.

The evaluator validates the raw/scored SHA bindings and row IDs before metrics.
It converts inline GT norm1000 coordinate-bin `xyxy` boxes to pixel `xyxy` and
treats scored predictions as already parser-normalized pixel `xyxy`. Mixed-unit
COCO sidecars are invalid. Raw predictions alone are not metric-bearing COCO
evidence.

## Provenance requirements

For a result to be interpreted, retain enough evidence to identify:

- the authored and resolved config;
- the data/image identity and row scope;
- the base model, processor, tokenizer, adapter, and selected-token embedding
  payload identities;
- the template/prompt and generation policy;
- the backend and trace/scoring policy;
- raw/scored artifact hashes and row binding;
- explicit loaded base/adapter/delta identity and evaluator receipt where
  applicable.

Do not reconstruct score-bearing evaluation from a copied prediction file whose
provenance sidecar or source raw artifact is missing.

## Stable contract routes

- [adapter and selected-token payloads](../openspec/specs/coordexp-swift-adapters-embeddings-optim/spec.md)
- [inference scoring artifacts](../openspec/specs/coordexp-swift-infer-scoring-artifacts/spec.md)
- [detection evaluator](../openspec/specs/coordexp-swift-detection-evaluator/spec.md)
- [inference pipeline](../openspec/specs/coordexp-swift-infer-pipeline/spec.md)

## Historical artifact material

Older artifact notes may mention `src/sft.py`, `src/utils/run_manifest.py`,
`src/infer/*`, Stage-2 rollout fields, or files that are not emitted by the
current Swift path. Those references are preserved for old-run interpretation
under [`docs/history/`](history/README.md), archived OpenSpec changes, or
legacy domain routers. They do not override the ownership and names above.
