# Unified Training Runtime Boundaries Proposal Review Plan

> **Proposal review only. Do not implement production code, configs, stable docs routes, or stable OpenSpec changes from this file.** This plan ends with a user decision packet, not code execution.

**Goal:** Review and converge the OpenSpec proposal for hardening CoordExp unified training runtime boundaries, with the P0 eval-validity slice approved as the first implementation slice.

**Architecture:** This is a proposal gate around runtime projection, Stage-2 target construction, shared inference owner boundaries, artifact provenance, and deletion gates. Public YAML namespaces, artifact names, metric semantics, and current Stage-2 public surface names remain stable unless a later approved OpenSpec explicitly changes them.

**Tech Stack:** OpenSpec, Superpowers proposal docs, CoordExp typed YAML configs, Stage-2 rollout-correction contracts, shared inference runtime contracts, pytest search/contract gates after approval.

---

## Governing Artifacts

Proposal artifacts:

- `openspec/changes/harden-unified-training-runtime-boundaries/`
- `docs/superpowers/specs/2026-05-27-unified-training-runtime-boundaries-design.md`
- `docs/superpowers/plans/2026-05-27-unified-training-runtime-boundaries-proposal-review.md`

Current authority:

- `openspec/specs/runtime-architecture-refactor-program/spec.md`
- `openspec/specs/shared-inference-runtime/spec.md`
- `openspec/specs/stage2-rollout-correction/spec.md`
- `openspec/specs/inference-engine/spec.md`
- `openspec/specs/inference-pipeline/spec.md`
- `openspec/specs/training-config-hierarchy/spec.md`
- `openspec/specs/training-pipeline-audit/spec.md`
- `docs/ARTIFACTS.md`
- `docs/training/STAGE2_RUNBOOK.md`
- `docs/eval/WORKFLOW.md`

Historical provenance only:

- `openspec/changes/archive/2026-05-27-unify-inference-runtime/`
- older Superpowers architecture docs that use Stage2-AB or two-channel
  vocabulary.

## Approval Boundary

- [x] User approved the P0 eval-validity implementation scope as the first
  slice.
- [x] User approved broader runtime-boundary implementation scope.
- [x] Subagent convergence register is complete, with P0 accepted into the
  first implementation slice and remaining P1s carried as sequencing/spec
  precision work.
- [x] Required active OpenSpec change exists if stable behavior changes.
- [x] OpenSpec validation command and target are named.
- [x] A separate P0 implementation plan exists with exact files, tests, and
  verification before production code edits.

P0 production-code implementation is now approved and scoped by
`docs/superpowers/plans/2026-05-27-p0-stage2-eval-validity-implementation.md`.
Broader refactor implementation has since been approved and is tracked in the
OpenSpec task list.

## Read-Only Review Tasks

- [ ] **Task 1: Verify current authority**

Run:

```bash
rg -n "stage2_rollout_correction|rollout_matching|shared runtime|metric_bearing|score_policy_fingerprint" openspec/specs docs/ARTIFACTS.md docs/training/STAGE2_RUNBOOK.md docs/eval/WORKFLOW.md
```

Expected: review notes distinguish stable specs from archived changes and older
Superpowers/progress history.

- [ ] **Task 2: Compare proposal against stable specs**

Run:

```bash
rg -n "Requirement:|Non-Goals|Current Authority|Current Contract Anchors|OpenSpec Impact Matrix" openspec/changes/harden-unified-training-runtime-boundaries docs/superpowers/specs/2026-05-27-unified-training-runtime-boundaries-design.md
```

Expected: every proposed requirement maps to runtime-boundary hardening or spec
hygiene, not unrelated config/fusion cleanup.

- [ ] **Task 3: Collect subagent findings**

Required lanes:

- OpenSpec governance and capability scope.
- Stage-2 trainer/runtime boundary.
- Shared inference and artifact provenance.
- Superpowers proposal-doc shape.

Expected: all lanes return, or the user explicitly waives a missing lane.

- [ ] **Task 4: Revise proposal artifacts**

Apply only proposal/spec/design/plan changes. Do not edit production code,
configs, stable docs routes, or stable specs as part of this proposal-review
plan.

- [ ] **Task 5: Produce decision packet**

Return:

- changed proposal files;
- convergence status by lane;
- unresolved P0/P1 issues if any;
- skipped validation and reason;
- exact hold point before implementation.

## Subagent Convergence Register

| Lane | Agent | Status | P0/P1 blockers | Required doc changes | Main-session disposition |
|---|---|---|---|---|---|
| OpenSpec governance | Bacon | Returned | P1 stable Stage-2 spec promotion gap; P1 `shared-inference-runtime` Purpose placeholder; P1 upstream API/provenance contract coverage location | Keep this as runtime-boundary hardening only; use stable specs as authority; split unrelated config/fusion cleanup | Incorporated; follow-up hygiene remains in later cleanup slices. |
| Stage-2 trainer/runtime boundary | Heisenberg / Maxwell | Heisenberg errored remotely; Maxwell returned | P1 projection payload needs naming; P1 target-construction payload needs explicit exclusions; P1 DDP/packing owner contract needs characterization; P1 A/B deletion taxonomy needs precision; P1 proposal artifacts must stay separate from dirty production edits | Add named `Stage2RuntimeProjection`/equivalent contract, target I/O boundary, DDP/packing contract, A/B taxonomy, and implementation split guidance | Incorporated; qualified convergence on inclusion, sequencing still requires user decision. |
| Shared inference/provenance boundary | Carver | Returned | P0 fabricated/best-effort Stage-2 eval geometry; P0 salvage parser in metric-bearing eval; P1 synthetic prompt provenance; P1 owner-shaped `src/infer`; P1 stable ownership disagreement | Add metric-bearing strictness, exact source geometry, real prompt provenance, owner-boundary requirements; recommend split inference/provenance from training hardening | P0 incorporated; P1 owner-boundary work remains in the shared inference slice. |
| Superpowers doc shape | Beauvoir | Returned | P1 proposal docs can look executable; P1 archived OpenSpec path drift; P1 older two-channel vocabulary conflicts with stable Stage-2 | Rename plan to proposal-review, pin stable specs, mark archive historical, add approval boundary and convergence register | Incorporated. |

## Historical No-Code Stop Condition

This plan is complete when the user receives a decision packet and can choose
whether to approve implementation scope, request more proposal changes, split
the OpenSpec change, or abandon the change.

Current implementation state:

- Shared inference/provenance returned P0 blockers; the user approved them as
  the first implementation slice for metric-bearing Stage-2 eval validity.
- Stage-2 trainer/runtime review returned qualified convergence that runtime
  projection, target construction, DDP/packing, and A/B/channel deletion belong
  under the umbrella, but require sequential implementation slices and tighter
  payload/taxonomy definitions.
- P0 implementation landed first and is committed.
- Stage-2 runtime projection now has a named `Stage2RuntimeProjection` module,
  with unknown non-empty trainer variants rejected unless explicitly registered
  as extension-style variants.
- Remaining work proceeds through the target-construction, DDP/packing,
  shared-inference/provenance, and cleanup/search-gate slices.

## Implementation Notes

The proposal-review plan remains historical evidence for convergence. Active
implementation progress is tracked in
`openspec/changes/harden-unified-training-runtime-boundaries/tasks.md`.
