# Design: Training Runtime Architecture Refactor

## Context

The previous parallel worktrees answered different parts of the problem.

The training-runtime integration branch found the right conservative starting
point: make training variant ownership explicit before moving large code. Its
best ideas are `TrainingPipelinePlan`, a derived trainer runtime profile, and
setup-only parity gates.

The old agent-research runtime branch explored a future control plane:
missions, recipes, runs, stage results, run review, summaries, and smoke/scale
gates. Those ideas are valuable as architecture direction, but the dirty
implementation should not be treated as code to merge. It was broad, stale,
and included first-class memory/ledger concepts that must be narrowed to fit
CoordExp's repo-safety guardrails.

This change keeps the ideas and discards the implementation patches.

## Design Principles

- OpenSpec drives code. Do not port implementation files from the deprecated
  worktrees directly.
- The first slice is compatibility-preserving training setup architecture, not
  a new runtime.
- The training runtime architecture must be import-safe at the planning layer:
  no trainer classes, torch, transformers, ms-swift, vLLM, datasets, or config
  loaders.
- There is one source of truth for trainer-variant policy.
- Existing configs, CLI surfaces, geometry contracts, metric keys, and
  artifacts remain stable unless a later OpenSpec delta explicitly changes
  them.
- Future mission/recipe/run concepts must be explicit, repo-visible, and
  evidence-backed. They must not become hidden agent memory.

## Architecture

### 1. Training Pipeline Plan

The first code slice should introduce an import-safe training plan that answers
setup-time ownership questions before trainer construction:

- whether raw sample metadata must survive collation;
- whether dataset static packing is eligible;
- whether post-rollout packing is trainer-owned;
- which collator family is required;
- whether ordinary one-sequence Stage-1 mixins may wrap the trainer;
- which objective pipeline namespace is required;
- whether removed trainer variants must fail with replacement guidance.

The plan is a routing contract, not a new training framework.

### 2. Trainer Runtime Profile Facade

Trainer/bootstrap-adjacent code may expose a runtime-profile facade for
readability and compatibility. That facade derives from the training plan. It
must not maintain a second table of variant policy.

The facade may describe:

- runtime stage;
- rollout-runtime ownership;
- explicit-pipeline requirement;
- mixin exclusion;
- packing and encoded-cache policy;
- collator family;
- manifest family.

Unknown non-empty trainer variants may keep generic profile behavior so legacy
trainer-factory routing can continue.

### 3. Stage-1 Ownership

Ordinary Stage-1 SFT remains the dataset static-packing and ordinary-mixin
path. `training.packing=true` keeps its current dataset-level meaning.

Stage-1 set-continuation remains branch-owned:

- raw sample metadata survives collation;
- dataset packing and eval packing remain rejected;
- the set-continuation collator is selected;
- ordinary one-sequence Stage-1 mixins are excluded;
- the checked-in branch-scoring runtime stays on repeated independent forwards
  or their current `smart_batched_exact` grouping; packed-varlen branch
  execution remains experimental, not a replacement for the v1 production
  path;
- branch scoring, candidate construction, loss math, and bidirectional-token
  semantics remain where they are until dedicated parity tests exist.

### 4. Stage-2 Ownership

Stage-2 two-channel and rollout-aligned paths remain trainer-owned rollout
paths:

- raw sample metadata survives setup;
- dataset static packing is disabled;
- `training.packing=true` means trainer-owned dynamic post-rollout packing;
- identity collation is selected;
- ordinary Stage-1 mixins are excluded;
- `stage2_two_channel` requires `stage2_ab.pipeline`;
- `stage2_rollout_aligned` requires `rollout_matching.pipeline`;
- both Stage-2 variants require top-level `rollout_matching` runtime settings.

Target construction, rollout matching, duplicate control, pseudo-positive
logic, and teacher-forcing objective execution are math-bearing surfaces and
must not move in the setup-only slice.

### 5. Recipe And Run Orchestration Direction

After the training setup seam is stable, later slices may introduce an
agent-friendly orchestration layer. The useful shape from the deprecated
agent-runtime branch is:

```text
research intent -> mission -> recipe/profile -> run -> stage results -> review
```

The first orchestration design should stay thin:

- a mission captures intent, constraints, selected recipe/profile, and next
  action;
- a recipe declares stage identities and dependencies;
- a run records one concrete execution attempt;
- each stage emits one stage-result record;
- run review and summary are derived from canonical evidence;
- smoke and scale are profiles with explicit gates, not informal convention.

The orchestration layer must cite existing authoritative artifacts rather than
replacing them prematurely:

- training resolved config and runtime manifests;
- pipeline manifests;
- run metadata;
- infer/eval resolved configs and output manifests;
- metrics and visualization artifacts.

### 6. Evidence And Memory Boundaries

The deprecated agent-runtime branch used `research-memory-ledger` language. In
this restart, that idea is narrowed:

- mission, run, stage-result, and note records may exist only as explicit
  repo-visible artifacts introduced by OpenSpec;
- authoritative execution truth lives in run and stage artifacts, not in chat
  memory or hidden state;
- summaries and notes are derived orientation layers and must link back to
  exact evidence;
- no hidden agent-only persistence layer may be introduced.

## Phasing

1. Draft this spec-only branch.
2. Remove deprecated worktrees and branches that contain imperfect code
   implementations.
3. Rebase or recreate implementation from this OpenSpec.
4. Implement training setup ownership first.
5. Add recipe/run orchestration only after the setup seam is stable.

## Rejected Alternatives

### Keep the current implementation branch

Rejected because it already contains code changes. The user wants to restart
from OpenSpec and rebase implementation cleanly.

### Merge the old agent-runtime dirty scaffold

Rejected because it is broad, stale, and includes memory/ledger ideas that
need a narrower safety contract before implementation.

### Split training setup and agent orchestration into unrelated changes now

Rejected for this branch because the goal is to preserve both idea sets in one
clean planning surface. Implementation can still split the work into separate
code slices later.
