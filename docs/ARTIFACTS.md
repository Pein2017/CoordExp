---
doc_id: docs.artifacts
layer: docs
doc_type: artifacts-reference
status: canonical
domain: repo
summary: Current CoordExp-Swift training, inference, evaluation, checkpoint, and provenance artifacts.
updated: 2026-08-20
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
| Inference payload manifest | `src/artifacts/checkpoint_payload.py` |
| Opt-in exact training-state sibling | `src/artifacts/training_state.py` |
| Inference rows, traces, manifests, and merge | `src/inference/artifacts.py`, `src/inference/merge.py` |
| Detection evaluation artifacts | `src/eval/detection_consumer.py` |

Stable training artifact behavior is specified by
[`coordexp-swift-training-artifacts`](../openspec/specs/coordexp-swift-training-artifacts/spec.md);
explicit inference composition is specified by
[`coordexp-swift-infer-config-runtime`](../openspec/specs/coordexp-swift-infer-config-runtime/spec.md).

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
- `checkpoints/step-<step>/inference_payload_manifest.json`: the
  self-authenticating manifest binding those learned inference files;
- `checkpoints/step-<step>/training_state/` with its `manifest.json`: the
  opt-in exact training-state sibling, written only when exact state is
  enabled;
- `checkpoints/final.json` and `checkpoints/best.json`: checkpoint selectors.

The training run does not emit `run_manifest.json`, per-split metric streams,
per-step receipts, `checkpoint_handoff.json`, or `checkpoint-final` aliases.

Train and forward-eval rows in `logging.jsonl` share one loss projection. Every
computed term contributes an explicit field family — `loss/<term>/raw` for the
globally normalized planned-step value before weighting,
`loss/<term>/weighted` for that value times the configured weight, and
`loss/<term>/selected_count` for the atoms it consumed — alongside its existing
segment-count, token-weighted-diagnostic, and `finite/<term>` fields.
`loss/total` is the sum of the weighted objective terms; no field carries the
distributed mean-gradient compensation applied to the differentiable local
contribution. The named zero-weight gate ablation keeps its whole family with a
weighted value of zero, while a zero-weight optional auxiliary contributes no
field at all rather than zero-valued ones. Non-finite computed values are still
serialized as JSON `null` and named in `non_finite_fields`. Rows written before
this projection use the older ambiguous per-term field and are read against the
commit that wrote them; there is no dual-written alias. The authored loss shape
behind these fields is described in
[`COORDEXP_SWIFT.md`](COORDEXP_SWIFT.md#supervised-loss-contract).

A checkpoint carries two independent payload surfaces:

- The minimal inference payload is always published. It holds the adapter,
  the optional selected-token embedding delta, and their manifest, and it never
  contains or requires base-model weights or optimizer, scheduler, scaler, RNG,
  dataloader, iterator, or sampler state to be loadable. Inference consumes
  only these explicit paths and ignores the training-state sibling.
- The exact training-state sibling is opt-in and disabled by default; with it
  disabled no `training_state/` directory or exact-state identity is written
  and checkpoint behavior is unchanged. When enabled, it is published under
  `training_state/` after the inference payload commits, and it supports
  continuation only from an optimizer-step save boundary at the same world size
  and rank map, restoring step, pack cursor, optimizer, scheduler, scaler, and
  RNG state.

Failure semantics are fail-closed. Resume admission authenticates the sibling,
its rank contributions, and the declared compatibility identities before any
mutable state is restored, and rejects world-size drift, an unsupported
accumulation position, an identity mismatch, or an inference-only payload
rather than degrading to partial restore or weights-only loading. A failed or
interrupted exact-state publication leaves no alias or completed event
advertising that step as resumable, while its separately committed inference
payload remains inference-only. `final.json` and `best.json` update only after
every publication required by the selected checkpoint mode has committed.
Cross-world-size and mid-accumulation resume are unsupported, and none of this
is a performance claim or a production-launch claim.

Immutable pack cache v3 lives outside the run tree, publishes only to an absent
semantic fingerprint target, and contributes only compact bindings to
`run.json`.

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
