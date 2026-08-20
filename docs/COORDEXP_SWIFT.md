---
doc_id: docs.coordexp-swift
layer: docs
doc_type: canonical-implementation-guide
status: canonical
domain: repo
summary: Current routing guide for the CoordExp-Swift training, inference, evaluation, and artifact infrastructure.
tags: [coordexp-swift, training, inference, eval, routing]
updated: 2026-08-20
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
| Entry and assembly | `src/train.py`, `src/training/pipeline.py` (facade), `src/training/execution_plan.py`, `src/training/control_plane.py`, `src/training/cache_workflow.py`, `src/training/session.py` | Resolve config, build the immutable model-free plan, converge ranks, admit the pack cache, then assemble model, adapters, embedding deltas, schedule, losses, optimizer/scheduler, runtime, eval, checkpoints, and artifacts inside one `TrainingSession` |
| Planned-step training | `src/training/supervised_trainer.py` | Iterate micro-steps and planned steps through explicit runtime and loss interfaces |
| Config | `src/config/loader.py`, `src/config/models.py`, `src/config/resolve.py` | YAML extends resolution, strict typed validation, path resolution, and resolved-config fingerprinting |
| Data and geometry | `src/data/` | Validate JSONL examples, images, object identity, descriptions, dimensions, and geometry |
| Template and spans | `src/templates/` | Render prompts/assistant content and expose semantic supervision spans |
| Qwen boundary | `src/qwen/` | Load model/processor/tokenizer, encode chat/image inputs, build positions, and run forward helpers |
| Packing and supervision | `src/packing/`, `src/supervision/` | Concatenate no-padding segments and map logical token atoms to physical positions |
| Losses | `src/losses/` | Assemble the protected supervised objective (base CE plus token-type gate) and the typed optional coordinate Gaussian/RPS auxiliary, with `segment_balanced` normalization and finite diagnostics |
| Runtime and optimization | `src/runtime/`, `src/optim/`, `src/adapters/` | Accelerate replicated-DDP operations, finite gates, optimizer/scheduler steps, adapter and selected-token trainable surfaces |
| Training artifacts | `src/artifacts/run_writer.py`, `src/artifacts/checkpoints.py`, `src/artifacts/checkpoint_payload.py`, `src/artifacts/training_state.py`, `src/training/exact_resume.py` | Rank-zero run/config/log ownership, synchronized staged adapter-plus-delta inference payloads, and the opt-in exact training-state sibling and its resume admission |
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

`resume.mode` selects exact training-state behavior: `disabled` (the default)
or `exact_same_world_size`, which additionally requires
`runtime.determinism.mode: strict_cuda_replay_v1`. A null
`resume.checkpoint_dir` under `exact_same_world_size` publishes exact state
without resuming, which is the publish-only control or parent form.

## Supervised loss contract

The supervised objective is strict, closed, and SFT-only. Normative semantics
belong to
[supervision/losses](../openspec/specs/coordexp-swift-supervision-losses/spec.md)
and [config runtime](../openspec/specs/coordexp-swift-config-runtime/spec.md);
this section only routes operators to the shape a current config must author.

```yaml
losses:
  normalizer: segment_balanced
  protected:
    base_ce:
      weight: 1.0
    token_type_gate:
      mode: enabled                # or: zero_weight_ablation
      weight: 0.1                  # enabled pairs only with this value
      groups: [desc_text, schema, coordinate, eos]
  auxiliary:                       # optional block
    coord_gaussian_rps:
      weight: 1.0
```

- **Protected baseline.** `base_ce` is always present with weight exactly
  `1.0`. `token_type_gate` under `mode: enabled` carries the single canonical
  enabled weight shown above, over exactly those four ordered groups. No other
  protected weight, group order, or group set is accepted.
- **Named gate ablation.** The pure-CE contrast is authored as
  `mode: zero_weight_ablation` with weight exactly `0`; a zero weight without
  that named mode is rejected. The mode survives into
  `resolved_config.json`, so an intentional contrast is never confused with an
  accidentally disabled baseline.
- **Typed auxiliary placement.** Coordinate Gaussian/RPS is an optional
  auxiliary, authored only under `losses.auxiliary`. There is no compatibility
  alias for the former protected placement: authoring it under
  `losses.protected` fails strict validation with a migration-oriented error
  rather than being moved silently.

Each term carries one declared zero policy, and each policy has exactly one
runtime path:

- `forbid` (`base_ce`): omission or a zero weight fails config validation
  before loss construction.
- `detached_diagnostic` (`token_type_gate`): in the named ablation the
  denominator and fp32 per-atom math still run, but inside a no-grad boundary.
  Raw, count, and finite diagnostics are retained, the weighted value is a
  literal zero, and no gate tensor enters the optimized objective or the
  autograd graph. Because it stays a protected diagnostic, a non-finite raw
  gate value still participates in the all-rank pre-backward safety decision
  and still blocks backward and the optimizer update.
- `omit` (`coord_gaussian_rps`): at weight zero the term is not instantiated,
  no denominator is built or gathered, its math is never called, and it
  contributes no bundle entry, finite check, or logging field. Its absence is a
  whole missing field family, not a zero-valued one.

Logging rows name the two concepts separately for every computed term:
`loss/<term>/raw` is the globally normalized planned-step value before
weighting, `loss/<term>/weighted` is that value times the configured weight,
and `loss/<term>/selected_count` reports how many atoms the term consumed.
`loss/total` is the sum of the weighted objective terms. Distributed
mean-gradient compensation is applied once to the differentiable local
contribution only and never appears in these reported values. The old
ambiguous per-term field that named the weighted value is gone and is not
dual-written; historical JSONL keeps the schema of the commit that wrote it.

## Semantic boundaries that docs must preserve

- `source_order` preserves authored object order. `geo_sorted` validates the
  authored top-to-bottom/left-to-right order; it does not silently sort rows.
- `src/data/` validates raw records, `src/templates/` renders semantic spans,
  `src/qwen/encoding.py` aligns spans with physical Qwen tokens, and
  `src/packing/` remaps them into packed positions. Do not collapse these into
  one generic dataset owner.
- The active loss assembly is owned by `src/losses/runner.py` over the closed
  term inventory in `src/losses/bindings.py`. The supervised objective is
  SFT-only: there is no rollout-derived, hidden-state, policy, value, KL, or
  reward composition, and no import path, callable, or dynamic registry may
  select a term. See [Supervised loss contract](#supervised-loss-contract).
- Inference score-bearing artifacts require backend-neutral trace evidence,
  selected-token score provenance, row binding, and identity fingerprints.
  Predictions alone are not sufficient COCO evidence.
- Swift GT boxes are inline norm1000 `xyxy`; scored predictions are parser-
  normalized pixel `xyxy`. The evaluator converts the GT side and rejects
  mixed-unit COCO sidecars.
- Checkpoints always publish the minimal inference payload: a standard PEFT
  adapter and, when configured, a separate selected-token embedding delta,
  loaded through explicit paths. Exact training continuation is a bounded
  opt-in `training_state/` sibling, disabled by default; when enabled it
  restores step, pack cursor, optimizer, scheduler, scaler, and RNG state, but
  only at an optimizer-step save boundary with the same world size and rank
  map. Admission is fail-closed on any world-size, accumulation-position,
  identity, or inference-only-payload mismatch before mutable restore;
  inference never reads the sibling; and aliases update only after every
  required publication commits. Cross-world-size and mid-accumulation resume
  are unsupported, and this is neither a performance claim nor a
  production-launch claim.
- Pack cache v3 is an immutable internal cache outside the run tree. `Rebuild`
  means publishing only to a previously absent version/fingerprint target; the
  normal path never repairs, replaces, deletes, or garbage-collects an existing
  target. The run records only compact train/eval materialization bindings.
  Cache identity binds content and declared producer sources rather than file
  `mtime`, so touching a dataset file without changing its bytes reuses the
  existing cache. Before Accelerate or model construction, distributed startup
  validates the manifest and current rank's required train chunks plus every
  eval payload. Loaded train chunks receive full digest and restricted-payload
  validation, while manifest declaration checks cover every declared chunk.
- Completed training-step rows additionally carry `step_duration_seconds`,
  `input_build_seconds`, and `input_wait_seconds` (max-reduced across ranks;
  additive fields only, never a replacement for an existing row key). For
  end-to-end step wall-clock reading, use `step_duration_seconds`;
  `input_build_seconds` describes CPU-only input construction under every
  provider mode and excludes the device transfer, so it is not a substitute
  for `step_duration_seconds` and is not comparable across changes to how
  input construction is split into phases. Forward-input preparation supports
  exactly two modes. `synchronous` is the shipped default and the reference
  implementation; `overlapped` is an explicit experimental selection that adds
  a bounded, depth-one-ahead CPU producer and is proven to prepare
  byte-identical inputs. Single-GPU measurement has not shown a wall-clock win
  for the overlapped mode, so it is not the shipped default.
- Rank-sharded eval reduction (disjoint pack sharding with exact cross-rank
  aggregation) is the shipped multi-rank default after row-exact fixture
  coverage and an exact 8-rank wall-clock/checkpoint-selector replay. Eval
  still falls back to replicated execution when there are fewer packs than
  ranks or only one rank.
  The forward-input-provider mode is selected only by the strict config field
  `training.forward_input_provider_mode`; no environment variable can replace
  it, and an unsupported value fails strict config validation rather than
  falling back. The eval reduction mode remains selectable only through an
  internal, debug/measurement-only environment variable, not public YAML/CLI
  configuration surface — no new public knob was added by these changes.

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
