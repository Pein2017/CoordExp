---
doc_id: docs.architecture.index
layer: docs
doc_type: architecture-router
status: canonical-router
domain: architecture
summary: Stable router for the accepted current CoordExp architecture.
updated: 2026-07-11
---

# Accepted Architecture

This directory is the stable snapshot of accepted current architecture. It is
not a home for dynamic progress, implementation plans, one-time project specs,
historical proposals, or a runtime framework. Historical architecture material
is quarantined under [`../history/architecture/`](../history/architecture/README.md).

## Authority boundaries

| Surface | Authority | Purpose |
| --- | --- | --- |
| `docs/` | Current canonical operator-facing architecture and workflow | Describes what the current `main` implementation does and how to use it |
| `openspec/specs/` | Stable compatibility-sensitive contract | Owns normative config/schema, training/eval, artifact, cache, and metric requirements |
| `openspec/changes/<change>/` | Sole local active code-change workspace | Owns durable proposal/design/tasks/apply/verify/archive lifecycle for bounded code, config, docs, architectural-refactor, or internal-implementation work; delta specs are included only when a stable compatibility-sensitive contract changes |
| `docs/architecture/` | Stable accepted-architecture snapshot | Records accepted current ownership only |
| `docs/history/` | Historical/provenance layer | Holds completed or superseded proposal material, old plans, and migration evidence |

An architecture proposal may explain a seam or recommend a future direction. It
does not authorize implementation, override executable source, or duplicate
normative requirements from an OpenSpec. When a proposal and current code or a
stable spec disagree, record the conflict and resolve the contract separately.

## Current accepted architecture

The current architecture is described by this canonical docs chain:

1. [`../PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md)
2. [`../COORDEXP_SWIFT.md`](../COORDEXP_SWIFT.md)
3. [`../SYSTEM_OVERVIEW.md`](../SYSTEM_OVERVIEW.md)
4. [`../IMPLEMENTATION_MAP.md`](../IMPLEMENTATION_MAP.md)
5. the relevant `coordexp-swift-*` stable specs under
   [`../../openspec/specs/`](../../openspec/specs/)

Those pages describe the live `src/train.py` / `src/infer.py` route, current
ownership seams, and contract links. This README and the linked current docs
are the evergreen architecture surfaces. A proposal front matter status alone
does not make it current architecture authority.

## Change and history lifecycle

New architecture proposals, designs, tasks, and implementation state belong in
a named `openspec/changes/<change>/` workspace. Accepted outcomes are reflected
back into current docs and stable specs; the change is then archived by
OpenSpec. Older blueprint, decision-log, lifecycle-registry, refactoring-program,
and super-power material is retained only under [`../history/`](../history/README.md)
for explicit historical reconstruction.

PWSG is sequencing discipline inside a named OpenSpec change, not a directory
under `docs/`: Program/change, Wave/task group, Slice/task, Gate (verify + audit).
The stable architecture snapshot contains no active implementation state.

## Historical material

Historical architecture reviews and proposals are intentionally absent from
this directory. Use the [history router](../history/README.md) only when a task
explicitly asks for provenance or reconstruction.

## Review rule

When deciding whether an accepted architecture statement represents a real
module owner, interface, or seam, use the live source and tests with the
codebase-design vocabulary: name the owner, caller knowledge, invariant, and
verification surface. Do not add an abstraction merely because a proposal
names one.
