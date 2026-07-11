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
| Training run directory and metrics | `src/artifacts/manager.py`, `src/artifacts/metric_stream.py` |
| Training config snapshots | `src/config/writer.py` through the artifact manager |
| Checkpoint payloads and aliases | `src/artifacts/checkpoints.py` |
| Checkpoint handoff validation | `src/artifacts/checkpoint_handoff.py` |
| Inference rows, traces, manifests, and merge | `src/inference/artifacts.py`, `src/inference/merge.py` |
| Detection evaluation artifacts | `src/eval/detection_consumer.py` |

## Training artifacts

The run artifact manager initializes a run directory with a `run_manifest.json`
and then registers outputs as they are produced. Current names include:

- `run_manifest.json`: run status, config references, metric streams, eval
  summaries, checkpoints, warnings, and terminal status;
- `configs/resolved.json` and `configs/resolved.yaml`: resolved config
  snapshots and their fingerprinted resolution metadata;
- `metrics/<split>.jsonl`: typed metric stream events;
- `eval/forward/step-<planned_step_id>.json`: forward-only eval summaries;
- `resolved_step_schedule.json`: the resolved planned-step schedule;
- `checkpoints/step-<planned_step_id>/checkpoint.json`: checkpoint metadata;
- `checkpoints/step-<planned_step_id>/checkpoint_handoff.json`: model and
  payload identity handoff;
- `checkpoints/checkpoint-final.json` and
  `checkpoints/best_acc_top1.json`: aliases when the corresponding selection
  applies.

Checkpoint directories may also contain adapter payload files and selected
special-token embedding metadata/tensor payloads. The checkpoint metadata
records these V1 resume fields as `not_saved_v1`:

- optimizer;
- scheduler;
- scaler;
- dataloader;
- iterator;
- RNG.

Therefore a checkpoint handoff supports identity-checked model composition; it
does not claim exact training-state resume.

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
  generation policy, handoff readiness, and terminal/benchmark eligibility
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
- checkpoint handoff identity and evaluator receipt where applicable.

Do not reconstruct score-bearing evaluation from a copied prediction file whose
provenance sidecar or source raw artifact is missing.

## Stable contract routes

- [training artifacts](../openspec/specs/coordexp-swift-training-artifacts/spec.md)
- [checkpoint handoff readiness](../openspec/specs/coordexp-swift-checkpoint-handoff-readiness/spec.md)
- [inference scoring artifacts](../openspec/specs/coordexp-swift-infer-scoring-artifacts/spec.md)
- [detection evaluator](../openspec/specs/coordexp-swift-detection-evaluator/spec.md)
- [inference pipeline](../openspec/specs/coordexp-swift-infer-pipeline/spec.md)

## Historical artifact material

Older artifact notes may mention `src/sft.py`, `src/utils/run_manifest.py`,
`src/infer/*`, Stage-2 rollout fields, or files that are not emitted by the
current Swift path. Those references are preserved for old-run interpretation
under [`docs/history/`](history/README.md), archived OpenSpec changes, or
legacy domain routers. They do not override the ownership and names above.
