## Why

The unified-training-infra-refactor branch has made useful progress toward a
shared training and inference architecture, but the current code still carries
several transitional ownership leaks:

- `src/sft.py` hand-assembles Stage-2 runtime meaning from multiple config
  namespaces, then injects mutable trainer attributes.
- `Stage2RolloutRuntime` remains a broad class for backend lifecycle, eval,
  packing, rollout observability, and trainer-owned concerns.
- Stage-2 rollout-correction target construction, DDP coordination, rollout
  dispatch, target IR assembly, and loss metadata are still coupled through
  large functions and owner-like access.
- `src/infer` is the intended shared runtime root, but prompt/backend/artifact
  helpers still depend on broad `owner` objects instead of narrow resolved
  runtime facts.
- offline inference, train-time eval, score provenance, parser strictness, and
  Stage-2 eval artifacts have overlapping writers and provenance paths.
- retired A/B, Channel-B, `rollout_matching.pipeline`, and shadow pipeline
  concepts still appear in docs/tests/code paths strongly enough to guide future
  work in the wrong direction.

This change makes the architecture boundary hardening itself a stable contract.
It is intentionally scoped to the runtime-boundary hardening slice: shared
inference ownership, Stage-2 rollout consumption of that runtime, strict
prompt/decode/provenance/eval boundaries, and deletion or demotion of
overlapping active authority. Broader config cleanup, fusion-surface cleanup,
or research-objective changes are out of scope unless a later review proves
they are direct implementation prerequisites.

This proposal does not approve production implementation. It defines the
OpenSpec contract and associated Superpowers proposal-review surfaces that must
converge before code movement begins.

## What Changes

- Require Stage-2 runtime/config projection to be resolved once before trainer
  wiring, with `src/sft.py` reduced toward a launcher over typed runtime state.
- Require Stage-2 rollout-correction target construction to be isolatable from
  rollout backend lifecycle, DDP coordination, and trainer step execution.
- Require `Stage2RolloutRuntime` to be reduced to a trainer-owned facade or
  explicitly marked as temporary migration scaffolding; prompt/backend/decode,
  trace, parser, and artifact behavior must live under the shared runtime or
  dedicated trainer-owned seams.
- Require shared inference helpers to consume narrow resolved facts instead of
  probing arbitrary trainer/offline-owner attributes.
- Require one authoritative score/provenance writer for raw/scored artifacts,
  train-time eval artifacts, parser policy, and metric-bearing status.
- Require architecture gates for legacy import surfaces, broad owner coupling,
  unknown trainer variants, retired spec authority, and A/B or Channel-B
  compatibility surfaces.
- Require Stage-2 metric-bearing eval artifacts to fail before materialization
  when exact source image identity, dimensions, parser strictness, or prompt
  provenance are missing or synthetic.
- Associate this OpenSpec change with repo-local Superpowers design and plan
  documents:
  - `docs/superpowers/specs/2026-05-27-unified-training-runtime-boundaries-design.md`
  - `docs/superpowers/plans/2026-05-27-unified-training-runtime-boundaries-proposal-review.md`

## Current Authority And History

Current authority comes from stable specs under `openspec/specs/`, especially
`runtime-architecture-refactor-program`, `shared-inference-runtime`,
`stage2-rollout-correction`, `inference-pipeline`,
`training-config-hierarchy`, and `training-pipeline-audit`.

The archived `openspec/changes/archive/2026-05-27-unify-inference-runtime/`
change is historical provenance only. This proposal may reuse lessons from that
archive, but implementation must target stable specs plus this active change,
not the archived change directory.

## Capabilities

### Modified Capabilities

- `runtime-architecture-refactor-program`: hardens the deletion-test and
  boundary ownership requirements for runtime-critical refactors.
- `stage2-rollout-correction`: clarifies the target-construction, runtime
  facade, and obsolete A/B naming deletion contracts.
- `shared-inference-runtime`: narrows allowed shared-runtime caller boundaries
  and constrains owner-like adapters.
- `inference-pipeline`: single-owns score/provenance semantics and parser
  metric-bearing status across offline and train-time eval artifacts.
- `training-config-hierarchy`: requires fail-fast trainer variant resolution
  and explicit Stage-2 runtime projection provenance.
- `training-pipeline-audit`: distinguishes active runtime surfaces from shadow
  or audit-only descriptors.

## Impact

- Affected code during eventual implementation:
  - `src/sft.py`
  - `src/training_runtime/`
  - `src/training/surfaces.py`
  - `src/training/pipelines/`
  - `src/config/`
  - `src/bootstrap/`
  - `src/infer/`
  - `src/trainers/stage2_rollout_runtime.py`
  - `src/trainers/stage2_rollout_correction_impl.py`
  - `src/trainers/rollout_correction/`
  - `src/trainers/rollout_aligned_evaluator.py`
  - `src/eval/`
- Affected docs/specs during eventual implementation:
  - `docs/AGENT_INDEX.md`
  - `docs/IMPLEMENTATION_MAP.md`
  - `docs/SYSTEM_OVERVIEW.md`
  - `docs/ARTIFACTS.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/eval/WORKFLOW.md`
  - `docs/catalog.yaml`
  - active OpenSpec specs named above
- Affected tests during eventual implementation:
  - Stage-2 rollout-correction contract/runtime/packing/DDP tests
  - training runtime profile and SFT integration tests
  - infer layout/import gates
  - unified infer pipeline, parser, artifact, and provenance tests
  - removed mechanism and legacy surface absence tests

## Current Review State

Subagent review converged on the broad direction, with one scoped
implementation decision from the user on 2026-05-27: the P0 Stage-2
eval-validity hardening is approved as the first implementation slice before
broader training runtime refactoring.

The shared inference/provenance lane returned P0 eval-validity blockers:

- Stage-2 metric-bearing eval can still fabricate or best-effort rewrite image
  paths and dimensions before official artifact/eval materialization.
- Stage-2 official eval can still consume salvage-style parser recoveries as
  metric-bearing predictions.

Those issues are now included as the approved first slice. Broader runtime
projection, target-construction, DDP/packing, shared-runtime owner narrowing,
and A/B or Channel-B deletion work remains blocked until a separate
implementation plan is written and approved.

## Non-Goals

- Do not rename public `infer.*`, `rollout_matching.*`, or
  `stage2_rollout_correction.*` authored YAML namespaces in this change.
- Do not change metric semantics, artifact names, or score meaning without a
  separate explicit spec delta.
- Do not absorb unrelated fusion-config, public-data, or broad config-hierarchy
  cleanup into this runtime-boundary hardening change.
- Do not move Stage-2 residual target construction, duplicate filtering,
  assignment, DDP coordination, loss execution, or training metrics into
  `src/infer`.
- Do not introduce a heavy top-level training framework or inheritance-heavy
  pipeline base.
- Do not use benchmark improvement as an OpenSpec validity gate.
- Do not implement production code until the user explicitly approves the
  implementation phase after subagent review convergence.
