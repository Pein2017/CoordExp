## 1. Spec Approval Gate

- [ ] Review `proposal.md`, `design.md`, and delta specs with subagent audits.
- [ ] Refine terminology, config names, target IR fields, parser policy, and
  metric namespaces from audit feedback.
- [ ] Obtain explicit user approval before implementation planning.

## 2. Implementation Roadmap Gate

- [ ] After spec approval, switch to the super-power planning workflow.
- [ ] Produce a staged implementation roadmap with code ownership, deletion
  plan, tests, smoke runs, and rollback strategy.
- [ ] Run subagent review/audit of the implementation roadmap.
- [ ] Obtain explicit user approval before real code implementation.

## 3. Implementation Scope Placeholder

- [ ] Implement only after the approval gates above.
- [ ] Keep implementation aligned with this OpenSpec and the approved
  super-power roadmap.

## 4. Draft Implementation Phases For Roadmap Review

- [ ] Add `src/training/teacher_forcing/` IR/helpers, register the concrete
  objective module through `src/training/objectives/`, extend
  `src/training/supervision/`, and add runner validation/unit tests before
  deleting old objective code.
- [ ] Wire `teacher_forcing_target_ir` through dataset sidecar filtering,
  collator/batch-extras handling, model-input bundling, trainer bridges, and
  loss-side stripping; verify it never reaches model forward.
- [ ] Implement Stage-1 latest compact `objective.id: teacher_forcing` with
  `profile: hard_sft` first, then valid-set and coverage profiles.
- [ ] Add config migration failures for old Stage-1 ids and stale support,
  balance, recursive, bbox, geometry, duplicate, and coord-reg keys.
- [ ] Wire Stage-2 adapters to emit the shared target IR without binding core
  objective math to the Stage-2 trainer implementation.
- [ ] Resolve Stage-2 packing explicitly: v1 either fails fast for
  `objective.id: teacher_forcing` with `training.packing=true`, or implements
  and tests exact segment-local to packed atom-position mapping before enabling
  packed forwards.
- [ ] Replace active compact-full training serialization with marker-delimited
  no-newline rendering through a policy-aware `compact_full` path while
  preserving legacy newline parsing/rendering only for explicitly historical
  inference/eval compatibility.
- [ ] Remove old exports/imports only after replacement Stage-1 hard-SFT,
  Stage-1 valid-set, and Stage-2 adapter tests pass.

## 5. Draft Verification Matrix For Roadmap Review

- [ ] Config parse tests for `teacher_forcing/hard_sft`, pure valid-set,
  coverage-regularized valid-set, and old-id rejection.
- [ ] Objective runner tests for selected-token mismatch, causal-position
  mismatch, rank-2 logits rejection, sliced logits or `logits_to_keep`, and
  padding-mask mismatch.
- [ ] Stage-2 packing tests proving either fail-fast rejection before model
  forward or exact packed atom-position mapping while preserving flash-attention
  kwargs.
- [ ] Sidecar bridge tests proving `teacher_forcing_target_ir` survives
  collation/stash and is stripped from model-forward kwargs while attention and
  flash-attention kwargs remain forwarded.
- [ ] Target-builder tests for singleton hard SFT, ambiguous valid-set marginal,
  coverage disabled/enabled, mixed `{TEXT, SCHEMA}` description-boundary atoms,
  coordinate-onset filtering, and overlength/empty-list rejection counters.
- [ ] Parser and serialization tests for policy-aware `compact_full`:
  marker-delimited no-newline training rendering, strict newline rejection,
  legacy-compatible newline parsing for historical outputs, and all stable
  strict parse error codes.
- [ ] Metric-contract tests for the minimum `teacher_forcing/...` and
  `infer/parse/compact_full/...` leaf keys.
- [ ] Small smoke checks before any production training.
