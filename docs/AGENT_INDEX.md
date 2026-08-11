---
doc_id: docs.agent-index
layer: docs
doc_type: agent-router
status: canonical
domain: repo
summary: Agent-first retrieval guide for current CoordExp documentation and historical evidence.
tags: [agents, retrieval, docs]
updated: 2026-07-11
---

# Agent Index

Use this page only when the user-named evidence does not reveal the current
owner. Search the machine-readable [catalog](catalog.yaml) or the query routes
below, then load the smallest owning document, contract, code, or artifact set.
Do not read the catalog or this index's targets wholesale.

## Task-local routing

1. Inspect the exact path, artifact, config, diff, run, or question named by the
   user.
2. If ownership remains unclear, search [catalog.yaml](catalog.yaml) by domain,
   path, or status, or use the query routes below.
3. Read only the selected owner and the contracts or evidence required by the
   task.
4. Enter stable specs for compatibility semantics, active changes only when
   named, `research/` for current interpretation, and history only for an
   explicitly historical question.

## Authority quick reference

- `docs/` is current operator-facing truth for routes, workflows, and recommended practice.
- `openspec/specs/` is stable compatibility authority for config/schema, training/eval behavior, artifacts, cache identity, and normative metrics.
- `openspec/changes/<change>/` is the sole local workspace for bounded code,
  config, docs, or architecture work that benefits from durable
  proposal/design/tasks/apply/verify/archive lifecycle. Delta `specs/` belong
  there only when a stable compatibility-sensitive contract changes; internal
  refactors do not require invented normative deltas.
- `research/` contains active research interpretation.
- `docs/history/` contains non-normative provenance for superseded plans and old
  implementation history.
- `progress/` is a legacy evidence archive. Do not use it for current behavior
  when a current doc or stable spec exists.
- Architecture proposals and one-time project plans are non-normative. They
  describe reasoning and sequencing, not implementation authorization.
- Superpowers design and execution plans are non-normative sequencing artifacts.
  For an OpenSpec-owned change, they link to rather than replace its authority.
- When an OpenSpec change uses PWSG, it is internal sequencing discipline:
  Program/change, Wave/task group, Slice/task, Gate (lead-owned executable
  verification, plus one independent audit only for a frozen high-risk or
  decision-bearing target).

## Current Swift route

Current training and inference entrypoints are:

- `src/train.py` -> `src/training/pipeline.py` ->
  `src/training/supervised_trainer.py`
- `src/infer.py` -> `src/inference/pipeline.py` ->
  `src/inference/runtime.py` / `src/inference/backend.py`
- `scripts/evaluate_detection.py` ->
  `src/eval/detection_consumer.py` for scored detection evaluation

Current config roots are:

- `configs/coordexp_swift/prod/`
- `configs/coordexp_swift/smoke/`
- `configs/coordexp_swift/infer/`

Training is Accelerate-only replicated DDP. Route training artifacts through
`src/artifacts/run_writer.py` and `src/artifacts/checkpoints.py`; removed
manager, metric-stream, and checkpoint-handoff modules are not current routes.

Do not route current work through `src/sft.py`, `src/trainers/`,
`src/datasets/`, `src/detection/`, or the old `src/infer/` package. Those names
remain in historical docs, archived configs, tests, and old-run evidence.

## Stable Swift contract routes

Use the smallest relevant spec family:

- Config and data: [`coordexp-swift-config-runtime`](../openspec/specs/coordexp-swift-config-runtime/spec.md), [`coordexp-swift-data-template-encoding`](../openspec/specs/coordexp-swift-data-template-encoding/spec.md)
- Packing, forward, and losses: [`coordexp-swift-packing-forward`](../openspec/specs/coordexp-swift-packing-forward/spec.md), [`coordexp-swift-supervision-losses`](../openspec/specs/coordexp-swift-supervision-losses/spec.md), [`coordexp-swift-pack-cache-semantic-identity`](../openspec/specs/coordexp-swift-pack-cache-semantic-identity/spec.md)
- Trainable payloads and training artifacts: [`coordexp-swift-adapters-embeddings-optim`](../openspec/specs/coordexp-swift-adapters-embeddings-optim/spec.md), [`coordexp-swift-training-artifacts`](../openspec/specs/coordexp-swift-training-artifacts/spec.md)
- Inference and evaluation: [`coordexp-swift-infer-config-runtime`](../openspec/specs/coordexp-swift-infer-config-runtime/spec.md), [`coordexp-swift-infer-pipeline`](../openspec/specs/coordexp-swift-infer-pipeline/spec.md), [`coordexp-swift-infer-backend-trace`](../openspec/specs/coordexp-swift-infer-backend-trace/spec.md), [`coordexp-swift-infer-scoring-artifacts`](../openspec/specs/coordexp-swift-infer-scoring-artifacts/spec.md), [`coordexp-swift-detection-evaluator`](../openspec/specs/coordexp-swift-detection-evaluator/spec.md)

The remaining `coordexp-swift-*` specs are reachable from the
[`openspec/specs/`](../openspec/specs/) directory. Do not invent a missing
pre-promotion spec path to make a link look normative.

## Query routing

- Current architecture or source ownership: [COORDEXP_SWIFT.md](COORDEXP_SWIFT.md), [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md), [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
- Data and preprocessing: [docs/data/README.md](data/README.md), [docs/data/CONTRACT.md](data/CONTRACT.md), [docs/data/PREPARATION.md](data/PREPARATION.md), [docs/data/PACKING.md](data/PACKING.md)
- Current inference/evaluation: [docs/eval/README.md](eval/README.md), [docs/eval/WORKFLOW.md](eval/WORKFLOW.md), [ARTIFACTS.md](ARTIFACTS.md)
- Training-history interpretation: [docs/training/README.md](training/README.md), marked as a legacy router for old MS-Swift/mainline runs
- Standards: [docs/standards/README.md](standards/README.md)
- Accepted architecture: [docs/architecture/README.md](architecture/README.md)

## Historical-material rule

`docs/history/`, `progress/`, archived OpenSpec changes, and old worktrees may explain why a
direction exists. They do not establish current behavior. Preserve their
provenance, label their scope, and do not revive the removed legacy repo-local
super-power workflow or turn a proposal into a runtime framework. This does not
prohibit the installed Superpowers plugin when explicitly invoked as
non-normative execution discipline for a current change.

## High-signal searches

```bash
rg -n "src/train.py|src/infer.py|src/inference|detection_consumer|coordexp_swift" docs openspec configs src tests
rg -n "checkpoint_handoff|resume_state|resolved_config|gt_vs_pred_scored|evaluation_receipt" docs openspec src tests
rg -n "src/sft.py|src/trainers|src/datasets|src/detection|src/infer/|configs/stage1|configs/stage2" docs openspec configs src tests
```
