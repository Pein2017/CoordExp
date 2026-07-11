---
doc_id: docs.project-context
layer: docs
doc_type: root-context
status: canonical
domain: repo
summary: Defines documentation ownership, contract authority, and the universal read order for CoordExp.
tags: [precedence, docs, agents]
updated: 2026-07-11
---

# Project Context & Documentation Authority

This page defines how to interpret repository documentation at the current
fixed point. It is a routing and authority contract, not a description of every
historical implementation that remains in the tree.

## Fixed point

Unless a task names another checkout, the canonical evidence root is the
current `/data/CoordExp` checkout on `main`. Git state, executable source,
configs, tests, artifacts, and stable specs outrank an older document or a
different worktree. See [`BRANCH_AND_WORKTREE_POLICY.md`](BRANCH_AND_WORKTREE_POLICY.md)
for branch and worktree boundaries. Pin a commit in a dated audit or handoff,
not in this evergreen router.

CoordExp-Swift is the current implementation on `main`. The development
worktree at `/data/CoordExp/.worktrees/CoordExp-swift` is a separate checkout
for active feature work; it is not a substitute for the fixed point when
answering a question about `main`.

## Authority model

Use these layers in order, with the narrower layer winning for the question it
owns:

| Layer | Authority | Owns | Does not own |
| --- | --- | --- | --- |
| Current operator docs | `docs/` | Current routes, workflows, ownership, and recommended practice | Normative contract details that belong in stable specs |
| Stable compatibility contracts | `openspec/specs/` | Supported config/schema, training/eval semantics, artifacts, cache identity, and normative metrics | General roadmap or historical explanation |
| Active code-change workspace | `openspec/changes/<change>/` | The sole local workspace for bounded code/config/docs work that benefits from durable proposal/design/tasks/apply/verify/archive lifecycle, including architectural refactors and internal implementation changes | Accepted current behavior before the change is implemented and verified; delta specs are conditional on a stable compatibility-sensitive contract change |
| Active research knowledge | `research/` | Current interpretation, investigations, and durable empirical reasoning | Operator instructions or implementation authority |
| Historical provenance | `docs/history/` | Superseded plans, migrations, old architecture reasoning, and provenance | Current behavior |
| Legacy evidence archive | `progress/` | Dated diagnostics, benchmarks, failed directions, and historical derivations | New canonical docs or current implementation claims |

Do not duplicate a stable requirement in several canonical pages. A current doc
should summarize and link to the owning spec; a proposal may explain why a
direction is useful, but it must not silently become a contract.

## Current implementation spine

The live Swift route is:

```text
configs/coordexp_swift/
  -> src/train.py -> src/training/pipeline.py
  -> src/training/supervised_trainer.py
  -> src/data -> src/templates -> src/qwen -> src/packing
  -> src/supervision -> src/losses -> src/runtime -> src/artifacts

configs/coordexp_swift/infer/
  -> src/infer.py -> src/inference/
  -> scored inference artifacts -> src/eval/detection_consumer.py
```

The old `src/sft.py`, `src/trainers/`, `src/datasets/`, `src/detection/`, and
`src/infer/` package references that appear in legacy docs are historical or
compatibility evidence. They are not current Swift entrypoints. Existing
`configs/stage1/`, `configs/stage2/`, and `configs/archive/` trees must be
treated according to their catalog status and should not be presented as the
default `main` route.

## Universal read order

For a new repository question:

1. [`docs/AGENT_INDEX.md`](AGENT_INDEX.md) and [`docs/catalog.yaml`](catalog.yaml)
   for retrieval routes and the machine-readable inventory.
2. [`docs/BRANCH_AND_WORKTREE_POLICY.md`](BRANCH_AND_WORKTREE_POLICY.md) for
   checkout and branch scope.
3. [`docs/COORDEXP_SWIFT.md`](COORDEXP_SWIFT.md) for the current implementation
   spine and current evidence boundaries.
4. [`docs/SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md) for the end-to-end flow.
5. [`docs/IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md) for source ownership
   and targeted verification.
6. The relevant domain router under `docs/`.
7. The exact `openspec/specs/coordexp-swift-*` contract only when stable
   compatibility semantics matter.
8. The named `openspec/changes/<change>/` workspace when work is being carried
   through its durable proposal/design/tasks/apply/verify/archive lifecycle.
9. `research/` for active research interpretation, then `docs/history/` or
   `progress/` only for explicitly historical questions.

## Authoring and lifecycle rules

- Put current operator-facing behavior and workflows in `docs/`.
- Put stable compatibility-sensitive requirements in `openspec/specs/`.
- Use a named `openspec/changes/<change>/` directory as the sole local active
  code-change workspace for bounded work that benefits from durable
  proposal/design/tasks/apply/verify/archive lifecycle, including architectural
  refactors and internal implementation changes. Include or modify delta
  `specs/` only when a stable compatibility-sensitive contract changes; do not
  invent normative deltas for internal refactors.
- Put research interpretation in `research/`; keep `progress/` read-only and
  historical.
- Keep architecture proposals and one-time project plans non-normative. They
  may record design reasoning and sequencing, but they do not authorize
  implementation or override code/spec evidence. The canonical architecture
  snapshot does not own dynamic program state.
- PWSG is sequencing discipline inside an OpenSpec change, not a docs
  directory: Program/change, Wave/task group, Slice/task, Gate (verify + audit).
- Preserve old proposals and plans as evidence. Mark them historical or
  superseded, and route completed material to `docs/history/` when it can be
  moved without breaking provenance links.
- Do not edit a stable spec merely to repair a documentation link. If code and a
  stable spec genuinely disagree, record the semantic conflict and stop before
  choosing a new contract.

## Current validation boundary

The current source and stable specs support the Swift route described above.
Inference has an implemented HF generation backend; vLLM fields are reserved
and validated as unavailable in the current implementation. Checkpoint handoff
identity is validated, but V1 checkpoint artifacts explicitly do not provide
exact optimizer, scheduler, scaler, dataloader, iterator, or RNG training-state
resume.

The fixed val200 inference/evaluation receipt is a historical, scope-labeled
validation handle, not a live artifact in every checkout. Tiny smokes are
implementation checks. A full validation-dataset run is optional unless a task
explicitly requests it. Do not turn these boundaries into broader readiness
claims without fresh artifacts.
