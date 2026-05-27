# Unified Training Runtime Boundaries Design

Status: proposal design; P0 eval-validity slice approved first, broader refactor not approved for implementation.

Date: 2026-05-27

Owner: CoordExp training runtime, Stage-2 rollout correction, shared inference, and artifact provenance

OpenSpec change:
`openspec/changes/harden-unified-training-runtime-boundaries/`

Associated plan:
`docs/superpowers/plans/2026-05-27-unified-training-runtime-boundaries-proposal-review.md`

## Purpose

This design turns the read-only architecture audit of
`codex/unified-training-infra-refactor` into a reviewable implementation
direction. The goal is to harden the current unified training/inference refactor
by moving real responsibilities behind deeper boundaries:

- runtime projection before trainer wiring;
- Stage-2 target construction separate from rollout/backend/DDP execution;
- shared inference adapters with narrow resolved inputs;
- single-owned raw/scored artifact provenance and parser metric-bearing status;
- deletion gates for retired Stage-2 and shadow pipeline authority.

This document records one scoped implementation decision: the P0
metric-bearing Stage-2 eval-validity hardening slice is approved to go first.
Broader runtime projection, target-construction, DDP/packing, shared-runtime
owner narrowing, and A/B deletion work still require a separate implementation
plan and approval.

## Current Contract Anchors

| Anchor | Role |
|---|---|
| `openspec/specs/runtime-architecture-refactor-program/spec.md` | Current umbrella contract for runtime-critical refactors and deletion of overlapping layouts. |
| `openspec/specs/shared-inference-runtime/spec.md` | Current shared prompt/decode/backend/trace/parser/provenance contract; its Purpose still needs spec hygiene. |
| `openspec/specs/stage2-rollout-correction/spec.md` | Current active Stage-2 public surface: `stage2_rollout_correction`, no Stage2-AB/two-channel public semantics. |
| `openspec/specs/inference-engine/spec.md` | Current canonical offline inference artifact and prediction semantics. |
| `openspec/specs/inference-pipeline/spec.md` | Current infer/eval/vis pipeline, raw/scored artifact, and provenance contract. |
| `openspec/specs/training-config-hierarchy/spec.md` | Current config inheritance and representative-leaf parity guardrails. |
| `openspec/specs/training-pipeline-audit/spec.md` | Current training entrypoint and pipeline audit guardrails. |
| `docs/ARTIFACTS.md` | Current artifact family and provenance documentation. |
| `docs/training/STAGE2_RUNBOOK.md` | Current Stage-2 operational route and authored config namespace guidance. |
| `docs/eval/WORKFLOW.md` | Current eval workflow contract. |

Historical references such as
`openspec/changes/archive/2026-05-27-unify-inference-runtime/` and older
Superpowers training architecture docs are provenance only. They are not
implementation targets unless a new active OpenSpec change reopens them.

## Diagnosis

The current branch has useful new seams, but several are still shallow:

- `src/sft.py` still interprets and injects too much runtime meaning.
- `Stage2RolloutRuntime` still owns too many shared-runtime and trainer
  concerns.
- Stage-2 target construction passes large argument bundles and depends on
  full trainer/runtime context.
- `src/infer` modules still consume broad `owner` shapes.
- raw/scored provenance, train-time eval artifacts, and parser strictness have
  overlapping writers.
- retired A/B, Channel-B, and shadow-pipeline concepts can still steer future
  work through docs/tests/spec references.

The deletion test is the guiding standard: if deleting a module removes only a
label while behavior remains in a broader caller, that boundary is not deep
enough.

## Accepted Direction For Proposal

- Keep authored public YAML namespaces stable:
  - offline inference uses `infer.*`;
  - Stage-2 runtime/backend/decode/eval uses `rollout_matching.*`;
  - Stage-2 objective/correction uses `stage2_rollout_correction.*`.
- Resolve Stage-2 runtime state once before trainer construction.
- Keep Stage-2 target/loss/DDP semantics trainer-owned, not in `src/infer`.
- Confine broad owner compatibility to explicit edge adapters.
- Move score/provenance semantics into one writer/validator.
- Treat docs/spec/catalog authority as part of architecture, not as cleanup
  decoration.
- Add architecture gates after each boundary moves, not before.

## OpenSpec Impact Matrix

| Category | Contract impact |
|---|---|
| Preserve | Public `infer.*`, `rollout_matching.*`, and `stage2_rollout_correction.*` namespaces; artifact names; raw/scored separation; metric semantics; geometry and `do_resize=false` invariants. |
| Potential Delta | Runtime-boundary deletion gates; unknown non-empty trainer variant rejection; Stage-2 target-construction isolation; owner-coupling gates; score/provenance single-writer requirements. |
| P0 prerequisite | Stage-2 metric-bearing eval must fail on fabricated geometry, salvage parser recovery, or synthetic prompt provenance before official artifact materialization. |
| Spec Hygiene | Repair `shared-inference-runtime` Purpose placeholder; demote retired specs/progress notes from current routing; clarify archived `unify-inference-runtime` as history. |
| Out Of Scope | Fusion-config removal, public-data refactors, benchmark promotion gates, stable metric redefinition, public namespace renames, reintroducing Stage2-AB/two-channel semantics. |

## Subagent Convergence Gate

Implementation stays blocked until all review lanes return or the user
explicitly waives a lane:

| Lane | Scope | Required convergence |
|---|---|---|
| OpenSpec governance | Capability scope, stable contracts, split-vs-one-change decision | No unresolved P0/P1 scope or authority blockers. |
| Stage-2 trainer boundary | Runtime projection, target construction, DDP/packing, retired naming | Agrees target/loss/DDP remain trainer-owned and old Stage2-AB vocabulary is historical. |
| Shared inference/provenance | Owner coupling, parser strictness, raw/scored artifacts, score provenance | Agrees shared runtime consumes resolved facts and provenance is single-owned. |
| Superpowers docs | Proposal-only doc shape and approval gating | Confirms docs cannot be mistaken for implementation approval. |

## Implementation Boundary

This proposal and its review plan do not authorize code, config, stable docs
route, or stable spec edits beyond proposal artifacts. A future implementation
phase requires:

- explicit user approval;
- subagent convergence with no unresolved P0/P1 blockers;
- an active OpenSpec change when stable behavior changes;
- a separate implementation plan with exact files, tests, and verification;
- `openspec validate <change-id> --strict` in an environment where the CLI is
  available.

## Non-Goals

- Do not rename public config namespaces in this change.
- Do not alter metric semantics, artifact names, or score meaning.
- Do not move Stage-2 target construction, assignment, duplicate filtering,
  DDP coordination, loss execution, or training metrics into `src/infer`.
- Do not add a heavy training pipeline base class.
- Do not preserve old A/B or Channel-B semantics as active behavior.
- Do not implement production code before user approval.

## Target Responsibility Map

| Concern | Target owner |
|---|---|
| Authored YAML parsing | `src/config/*` |
| Runtime surface policy | `src/training_runtime/*` plus named `Stage2RuntimeProjection` or equivalent |
| Launcher orchestration | `src/sft.py` |
| Stage-2 rollout generation | shared inference runtime plus trainer adapter |
| Stage-2 residual targets | trainer-owned rollout-correction target modules |
| DDP and post-rollout packing | trainer-owned coordination modules |
| Prompt/decode/backend/parser | `src/infer/*` with narrow caller adapters |
| Score/provenance comparability | shared artifact/provenance writer |
| COCO/LVIS metric math | `src/eval/*` |
| Architecture gates | focused tests under `tests/` |

## Proposed Implementation Waves

### Wave 0: Convergence Gate

- Draft OpenSpec and Superpowers documents.
- Review with independent subagents.
- Revise until the reviewers agree on scope and anti-goals.
- Wait for explicit implementation approval.

### Wave 0.5: Eval-Validity Prerequisite

User decision on 2026-05-27: approved as the first implementation slice before
broader training runtime refactoring.

- Add/require tests that Stage-2 metric-bearing eval fails before writing
  official artifacts when source image path, image ID, width, height, or source
  identity is missing.
- Add/require tests that salvage parser recoveries are diagnostic-only and
  cannot enter `gt_vs_pred*.jsonl`, confidence post-op, COCO/LVIS/mAP, or
  comparable reports.
- Add/require tests that Stage-2 eval score provenance is derived from the
  actual prompt bundle/decode request/model/score policy, not synthetic summary
  fields.
- Land this as a narrow shared inference/provenance implementation slice before
  broader training runtime hardening.

### Wave 1: Characterization And Config Projection

- Add fail-fast tests for unknown trainer variants.
- Add runtime projection tests for authored/effective Stage-2 policies.
- Define a named Stage-2 projection payload with authored source,
  compatibility fallback source, train/eval prompt policy, decode/backend/eval
  policy, packing settings, geometry/order policy, manifest payloads, and
  policy provenance.
- Move Stage-2 projection out of ad hoc `sft.py` attr injection.

### Wave 2: Stage-2 Trainer Boundary

- Characterize target construction inputs/outputs.
- Keep the target-construction payload limited to parsed rollout attempts, GT
  state, resolved correction policy, assignment/duplicate decisions, residual
  events, supervision metadata, target IR ingredients, and diagnostics.
- Deepen target construction away from backend lifecycle and DDP.
- Narrow packing/DDP owner access.
- Convert retired A/B and Channel-B tests to absence/rejection tests.
- Classify old A/B/Channel name matches by taxonomy before deletion:
  active public surface, private implementation identifier, rejection/absence
  test, migration adapter, historical fixture, or archive.

### Wave 3: Shared Inference And Provenance Boundary

- Confine owner introspection to adapters.
- Route prompt/decode/backend/artifact helpers through resolved facts.
- Single-own score provenance and metric-bearing parser policy.
- Preserve raw/scored artifact split and train-time eval artifact names.

### Wave 4: Deletion And Governance Gates

- Delete or quarantine dead manifest branches and retired naming.
- Demote retired specs/progress notes from active routing.
- Tighten import/search/owner gates to match final boundaries.

## Review Questions

- Should Wave 1 and Wave 2 be one implementation branch, or split after
  runtime projection lands?
- Should inference owner narrowing proceed under this same change or as a
  second branch governed by the same OpenSpec?
- Should docs/spec authority cleanup happen before code movement or with the
  deletion-gate wave?
