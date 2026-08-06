## 1. Freeze Consumer Evidence And Red Tests

- [ ] 1.1 Record the active consumer, merger, analyzer, tests, sealed plan, and
  census byte digests in a strict source-binding receipt; record the active
  `research-probes` Git status without staging, editing, or claiming a commit
  identity for its untracked consumer files.
- [ ] 1.2 Add failing journal interface tests for validated record payload
  views, started/failed/unfinished attempt views, last-durable-record identity,
  corrupt persisted evidence, defensive immutability, and unchanged journal
  schema version 1.
- [ ] 1.3 Add failing consumer-adapter tests for separate logical plan and
  physical schedule identities, exact continuation, no automatic retry, legacy
  receipt materialization, and active-unit/sealed-root non-mutation.

## 2. Journal Read And Diagnostic Surface

- [ ] 2.1 Add immutable public record and attempt view types plus a complete
  validated journal inspection result through the existing internal inspection
  path; return opaque payloads defensively and expose no research semantics.
- [ ] 2.2 Expose last durable sequence, work-item identifier, attempt identifier,
  and record digest while keeping OS exit code and signal outside the journal
  owner.
- [ ] 2.3 Pass the focused artifact suite and journal corruption/failure-injection
  tests; verify old journal fixture bytes, plan/record/terminal schemas, and
  writer behavior are unchanged.

## 3. Logical Plan And Cost-Aware Schedule

- [ ] 3.1 Implement the consumer-specific adapter under `scripts/research/`
  with a strict logical context projection that preserves every sealed context
  identity, candidate order, cost, and legacy receipt shard.
- [ ] 3.2 Implement the versioned deterministic LPT schedule using
  `(-scalar_equivalent_forward_count, context_id)` ordering and
  `(slot_cost, slot_count, slot_index)` placement ties; bind canonical schedule
  bytes and refuse schedule drift on continuation.
- [ ] 3.3 Materialize a CPU schedule mechanics receipt for the accepted
  200-context plan and verify eight 25-context slots, total cost 77,428,
  per-slot costs `(9675, 9675, 9675, 9683, 9678, 9681, 9672, 9689)`, maximum
  cost 9,689, and unchanged logical/legacy partition identities.

## 4. Per-Context Durable Execution And Explicit Continuation

- [ ] 4.1 Create one exact-identity execution journal per physical slot, bind
  plan/schedule/consumer/adapter/model/config/runtime identities, and fail all
  identity or root conflicts before model loading.
- [ ] 4.2 Adapt the existing support-context scoring boundary so each complete
  legacy observation mapping is strictly validated and journaled before the
  next context begins; do not journal each scalar forward.
- [ ] 4.3 Add an explicit continuation path that opens only the exact schedule,
  starts a new attempt, executes only missing contexts in schedule order, and
  never retries an accepted success, failure, unmatched, or other opaque
  outcome.
- [ ] 4.4 Add scorer/model cleanup and best-effort `SIGTERM` attempt-failure
  handling without treating signal handling as authoritative process-exit
  evidence or automatically launching a successor.

## 5. Legacy Receipt Adapter And Compatibility Gate

- [ ] 5.1 Implement a pure materializer that loads all validated slot records,
  regroups them by the sealed legacy `shard_index`, restores sealed plan order,
  and constructs the existing shard receipt schema without attempt-dependent
  fields.
- [ ] 5.2 Refuse terminal receipt publication for missing, duplicate, foreign,
  invalid, or merger-ineligible observations; publish mechanics diagnostics
  without assigning scientific meaning.
- [ ] 5.3 Prove all eight materialized receipts pass the current unchanged
  merger and analyzer contracts and preserve their existing ledger denominator,
  support rule, calibration, checkpoint, prefix, candidate, and outcome
  semantics.

## 6. Attempt And Exit Diagnostics

- [ ] 6.1 Add a parent launcher that records child PID, physical slot, attempt
  identifier, return code, terminating signal, plan/schedule identities,
  accepted/missing counts, last durable record, and temporary-path diagnostics
  in write-once mechanics receipts.
- [ ] 6.2 Cover normal exit, Python exception, explicit `SIGTERM`, attempt start
  without outcome, and diagnostic publication failure; prove none triggers an
  automatic retry or erases an earlier attempt.
- [ ] 6.3 Keep process/attempt diagnostics out of legacy shard receipts and add
  residue tests preventing mechanics fields from reaching merger/analyzer
  science surfaces.

## 7. Deterministic Interruption Equivalence Gate

- [ ] 7.1 Execute a production-shaped deterministic fixture once
  uninterrupted and once with a fresh-process interruption after at least one
  durable context followed by explicit continuation.
- [ ] 7.2 Require byte-for-byte equality and equal SHA-256 for every canonical
  legacy terminal shard receipt, successful unchanged merger validation, and
  no re-execution of accepted contexts; separately require distinct truthful
  attempt/exit histories.
- [ ] 7.3 Preserve a strict CPU equivalence receipt with input identities,
  interruption boundary, missing-set transitions, record hashes, terminal
  receipt hashes, merger verdict, and claim boundary.

## 8. Adoption Surface Without Active-Unit Mutation

- [ ] 8.1 Write a compact interface note describing logical plan, physical
  schedule, slot journals, explicit continuation, terminal materialization,
  diagnostics, failure states, and operator commands.
- [ ] 8.2 Produce an exact-digest-bound adapter patch or consumer example for
  `run_natural_boundary_support_completion.py`; make digest mismatch fail
  visibly and do not apply it to the active `research-probes` worktree.
- [ ] 8.3 Recheck the active worktree and all named sealed roots against the
  pre-change receipt and record that this implementation made no edits,
  merges, cherry-picks, recovery writes, or reinterpretation there.

## 9. Single-GPU Real SIGTERM Mechanics Gate

- [ ] 9.1 Check live GPU ownership and headroom, select one explicit device,
  create a fresh bounded mechanics root, and bind exact source, plan, schedule,
  model, config, runtime, and device identities before model loading.
- [ ] 9.2 Launch the real consumer path, wait until at least one context record
  is durably accepted, send external `SIGTERM`, verify the recorded child exit
  and unchanged durable bytes, then perform a separately invoked exact-identity
  continuation that executes only the missing bounded contexts.
- [ ] 9.3 Materialize the bounded terminal receipt, close the scorer cleanly,
  and preserve a mechanics-only GPU receipt containing signal, attempts,
  last-durable record, missing-set transition, continuation, terminal hashes,
  device/runtime evidence, and explicit exclusion of scientific claims.

## 10. Final Verification, Audit, And Local Commit

- [ ] 10.1 Run focused journal and research-adapter tests, relevant inference
  artifact regressions, immutable-schema checks, formatting/type checks, strict
  OpenSpec validation, and staged-diff/credential scans.
- [ ] 10.2 Run separate standards and intent-contract audits; resolve every P0
  or P1 finding before acceptance and verify the change did not become a
  generic probe framework or alter scientific semantics.
- [ ] 10.3 Freeze a self-excluding fixed-tree manifest over implementation,
  interface note, adapter patch/example, tests, CPU/GPU mechanics receipts, and
  audit verdicts; verify it from a fresh process.
- [ ] 10.4 Stage only the explicit change-owned paths and create a local fixed
  commit identity. Record the commit and verification summary without push,
  merge, cherry-pick, or active-unit mutation.
