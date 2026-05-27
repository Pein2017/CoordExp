# Refactor Training Runtime Architecture

## Why

CoordExp now has two useful but imperfect design sources:

- `feat/training-runtime-pipeline-integration` proved a concrete
  Stage-1/Stage-2 setup-ownership seam, but it mixed the desired OpenSpec with
  implementation code that still needs a cleaner rebase.
- `feat/agent-research-runtime` explored an agent-first mission / recipe /
  run / review control plane, but its dirty implementation scaffold was broad,
  stale, and not appropriate to merge into the current training refactor.

This change intentionally deprecates those code implementations while keeping
their useful specification ideas. The goal is to restart from a single
spec-first branch where implementation can proceed from OpenSpec to code in
small, reviewable slices.

## What Changes

- Define one training-runtime architecture program that starts with the
  conservative Stage-1/Stage-2 setup ownership seam.
- Preserve the import-safe training-pipeline plan idea from
  `feat/training-runtime-pipeline-integration`.
- Preserve the trainer-facing runtime-profile facade idea, but require it to
  derive from the same training plan rather than maintaining a second policy
  registry.
- Preserve the useful agent-runtime ideas as orchestration requirements:
  mission intent, recipe stage graphs, run evidence, stage-result records,
  summaries, and smoke/scale gates.
- Narrow the old agent-runtime memory idea so it cannot introduce hidden or
  agent-only persistence. Any mission/run/note records must be explicit
  repo-visible artifacts governed by OpenSpec and tied to authoritative run
  evidence.
- Require phased implementation:
  - first, setup-only training contracts;
  - then bootstrap extraction;
  - then manifest and artifact ownership;
  - only later, recipe/run orchestration;
  - math-bearing or target-construction moves require dedicated parity tests.

## Capabilities

### New Capabilities

- `training-runtime-architecture`: defines the phased architecture for
  training setup ownership, trainer runtime profiles, and future recipe/run
  orchestration.

### Modified Capabilities

- `runtime-architecture-refactor-program`: receives the stricter
  OpenSpec-to-code order and deprecates the old dirty V2 implementation
  scaffolds as code sources.
- `stage2-ab-training`: keeps Stage-2 pipeline namespace ownership explicit
  while setup policy moves behind the training runtime architecture seam.

## Impact

- Affected future code:
  - `src/training_pipeline/`
  - `src/trainers/runtime_contract.py`
  - `src/bootstrap/`
  - `src/sft.py`
  - future recipe/run orchestration surfaces only after the setup seam lands
- Affected docs/specs:
  - this OpenSpec change only in the first branch
  - implementation docs updated later with the code slices they describe
- Repro/eval impact:
  - the first implementation slice must be behavior-preserving;
  - later recipe/run orchestration may add artifacts, but must not replace
    existing authoritative training/inference/evaluation artifacts without an
    explicit compatibility and migration plan.
