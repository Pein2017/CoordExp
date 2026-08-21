## Context

The accepted training path already separates cache identity, loss composition,
runtime finite gates, and artifact reporting. This repair keeps those owners
and closes three post-archive conformance gaps without introducing a new
coordinator or compatibility layer. See `proposal.md` for motivation and the
delta spec for the only stable behavior change.

The affected runtime is Accelerate replicated DDP. Rank-local failures that
can occur beside a collective must become data in the existing all-rank
decision before any rank raises. Pack-cache identity remains immutable and
source-bound, so adding a previously missing payload owner intentionally turns
over the semantic fingerprint.

## Goals / Non-Goals

**Goals:**

- Give micro-step payload composition one small, source-hashed determinant
  owner.
- Reject declared fp16 without an active GradScaler at admission, with a
  consensus-level optimizer-boundary backstop for later drift.
- Preserve the existing zero-eligible failure semantics while moving the
  failure after the denominator gather at world size greater than one.
- Keep the P2 dispositions and verification evidence truthful and bounded.

**Non-Goals:**

- Do not change which zero-eligible planned steps fail.
- Do not rebuild or delete a pack cache in this change.
- Do not redesign recursive immutable value types for the execution plan.
- Do not decide exact-resume semantics for a crash-dangled active phase.

## Decisions

### 1. Extract a narrow micro-step assembler determinant owner

`src/training/micro_step_assembler.py` owns only the final composition of a
`SupervisedMicroStep`: metadata, resolved token identity, runtime precision,
and FA2 proof controls. `cache_workflow.py` continues to own orchestration and
calls the assembler after encoding, packing, and supervision are ready.
`pack_cache.py` binds the assembler file's source hash as a determinant.

This is preferred over restoring the whole cache workflow as an owner because
that would invalidate caches for unrelated orchestration edits. The extraction
is guarded by a deterministic canonical projection of the semantic payload and
readable field assertions. Production pickle chunk bytes are not used as the
cross-run equivalence oracle because PyTorch storage identifiers make them
non-deterministic; their per-publication digest remains the integrity check for
the artifact that was actually written.

Observable receipts: determinant-registry tests, the canonical payload golden
test, and the next training launch's new pack-cache fingerprint.

### 2. Use admission plus consensus for the fp16 scaler contract

The post-Accelerator admission phase validates that a resolved fp16 config can
reach one active, enabled GradScaler. The check runs inside
`RankControlPlane.converge`, so a rank-local validation error is converted into
the existing shared phase outcome before initialized training proceeds.

`TrainRuntime` also carries the resolved fp16 declaration in every gradient
finite report. If the scaler later becomes unreachable, the already-existing
all-rank optimizer-boundary reduction produces the terminal
`pre_wrapper_fp16_scaler_missing` decision instead of falling through to the
bf16/non-scaler apply path. A separate collective or fourth normal optimizer
action was rejected because the existing report contains all required facts.

Observable receipts: the real Accelerate launch-refusal probe, runtime
finite-gate tests, the typed terminal receipt, and the terminal logging row.

### 3. Carry zero eligible counts through the denominator gather

Local denominator construction permits zero only as an intermediate carrier.
At world size one, or when no collective is available, the existing local
typed failure remains. At world size greater than one, every rank serializes
its local counts, enters the existing denominator gather, and then derives the
same first canonical zero-eligible term and rank set from the gathered payloads
before raising.

Allowing a globally positive denominator to rescue a locally zero rank was
rejected because it changes the configured loss semantics. Adding a second
error-consensus collective was rejected because the denominator gather already
contains the decision facts.

Observable receipts: the real two-rank Gloo probe, identical rank error
contexts, and the single-rank compatibility tests.

### 4. Treat execution-plan immutability as partially mitigated, not fixed

`ResolvedTrainConfig` now severs constructor aliases and returns a deep copy
from `to_artifact_dict`; this closes the known artifact-consumer alias. It does
not make `config_dict` recursively immutable, and the execution plan's nested
measurement values remain shallow. This change therefore records P2-2 as
partial/deferred rather than claiming a completed architectural invariant.

A recursively immutable, deepcopy- and pickle-compatible value representation
would affect wider config and execution-plan consumers. That work belongs to a
separate architecture change owned by `src/config/models.py` and
`src/training/execution_plan.py`, with direct-mutation and nested-alias
acceptance tests.

## Risks / Trade-offs

- [New determinant turns over the fingerprint] -> Permit one normal immutable
  cache publication on the next authorized training launch; retain old caches
  as evidence for the code that produced them.
- [CPU or otherwise scaler-less fp16 launches now fail] -> Treat this as the
  intended fail-closed contract; bf16 and active-scaler fp16 controls remain
  accepted.
- [Canonical payload projection is not the production chunk byte stream] ->
  Name it precisely in receipts and keep production chunk digests scoped to
  integrity within one publication.
- [Execution-plan values remain shallowly mutable] -> Record P2-2 as deferred
  with exact owners and acceptance tests; do not hide it behind `frozen=True`.
- [Crash-dangled active-phase reconciliation remains undefined] -> Keep P2-3
  deferred to a resume-semantics change rather than guessing here.

## Migration Plan

1. Land the three repairs and their focused probes as one revertible commit.
2. Complete independent acceptance against the committed target.
3. Sync the single fp16 delta scenario and archive this change only after the
   OpenSpec artifacts and close-out receipts are coherent.
4. On the next authorized training launch, allow the new determinant set to
   publish once under its new fingerprint. Do not repair or overwrite an old
   fingerprint target.
5. Roll back by reverting the implementation commit and this close-out commit;
   old immutable caches remain untouched.
