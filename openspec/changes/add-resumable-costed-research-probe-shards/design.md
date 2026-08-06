## Context

See `proposal.md` for motivation. The current consumer exists as untracked work
in the active sibling `research-probes` worktree, so there is no honest Git
commit identity for it. This change therefore binds the exact consumer and
merger file digests recorded in the proposal and delivers an adapter patch or
consumer example without editing that worktree.

The consumer's current `execute_shard` holds all observation mappings in memory
and calls `write_shard_artifacts` only after scoring the entire legacy shard.
Its resume input is another whole-shard receipt. A signal before return can
therefore leave no resumable receipt. The accepted execution journal already
supports strict plan admission, single-writer record publication, attempts,
exact-identity opening, and terminal completion, but its public snapshot exposes
only completed work-item identifiers rather than validated record payloads or
attempt diagnostics.

The sealed logical plan has 200 contexts and 77,428 scalar-equivalent forwards.
Its embedded legacy shard costs are `(6757, 7012, 11819, 8208, 9845, 8511,
13867, 11409)`. Applying the selected deterministic schedule gives
`(9675, 9675, 9675, 9683, 9678, 9681, 9672, 9689)`, so the maximum declared
load falls from 13,867 to 9,689 without changing any context.

## Goals / Non-Goals

**Goals:**

- Make one support context the only durability and continuation unit.
- Use all eight physical slots with deterministic declared-cost balance while
  leaving the sealed plan and legacy receipt partition untouched.
- Preserve the existing merger/analyzer surface by materializing the same
  terminal shard receipt schema from journal payloads.
- Make interruption diagnosis mechanical and sufficient to answer which
  attempt exited and which record was last durable.
- Deliver a self-contained, fixed worktree commit and an explicit adoption
  patch rather than changing the active research unit.

**Non-Goals:**

- A generic research-probe DAG, distributed scheduler, queue, retry service,
  checkpoint-recovery system, or scientific result registry.
- Candidate-level durability, automatic retry, automatic continuation, or
  dynamic reassignment after a schedule is admitted.
- Changes to support features, thresholds, outcome statuses, denominators,
  model execution, candidate batching, merger logic, or analyzer logic.
- Recovery or reinterpretation of old v3/v4 process memory or sealed roots.

## Decisions

### 1. Add only validated read projections to the journal owner

Extend the journal's public read surface with immutable record and attempt
views plus a last-durable-record projection. Reuse the same internal inspection
path that validates the entire journal; do not let the adapter enumerate record
files or private helpers. The on-disk schema and writer behavior remain at
version 1.

The projection returns caller payloads opaquely. It deliberately cannot know
an OS return code or signal; the parent launcher owns that fact and joins it to
the journal attempt identity in a separate exit receipt.

**Alternative considered: let the adapter read `records/*.json` directly.**
Rejected because it would couple a second real consumer to private file naming
and could accidentally trust a subset without validating attempts, sequences,
or the terminal.

**Alternative considered: add scheduling and research vocabulary to the
journal.** Rejected because the journal remains a mechanics-only owner.

### 2. Use one journal per physical worker slot

The controller materializes one global logical-plan receipt and one physical
schedule receipt, then creates one journal root per physical slot. Each slot
journal binds the global plan fingerprint, schedule fingerprint, exact
execution identity, and its ordered assigned context identifiers. One worker
process owns that journal's writer lock and one GPU. Eight slots can therefore
run concurrently without inventing multi-writer journal semantics or a result
broker.

On explicit continuation, the same slot opens the same exact-identity journal,
creates a new attempt, and iterates its schedule order while skipping accepted
context identifiers. A completed context payload is never retried. A signal
inside a context may lose that context's in-memory candidate scores, but no
earlier context record.

**Alternative considered: one global journal with eight workers.** Rejected
because the current journal intentionally holds one exclusive writer lock for
its lifetime; adding IPC or multi-writer ordering would be a larger framework.

**Alternative considered: journal each scalar forward.** Rejected because the
accepted durability unit is a context and per-forward fsync would enlarge cost
and schema without decision value.

### 3. Derive a stable LPT schedule without resealing the logical plan

Build a separate canonical schedule artifact. Sort all logical contexts by
`(-scalar_equivalent_forward_count, context_id)`. For each context, choose the
slot minimizing `(current_total_cost, current_context_count, slot_index)` and
append it to that slot's execution order. Bind algorithm name/version, slot
count, logical plan file/content digests, ordered assignments, and totals.

The sealed plan's `context[*].shard_index` becomes legacy receipt-partition
metadata only; it is neither rewritten nor used as physical placement. The
first version fixes eight slots because that is the accepted plan and available
execution shape. Changing the slot count or algorithm creates a different
schedule and cannot continue an old journal.

**Alternative considered: continue using hash shards.** Rejected because the
real plan shows a 30.129% maximum declared-cost reduction from the separate
schedule, directly improving time to the last required context.

**Alternative considered: measured adaptive scheduling.** Rejected because it
would make placement attempt-dependent and complicate exact continuation.

### 4. Store complete legacy observation payloads as journal work items

The adapter calls the consumer's existing context scorer and support-feature
function. After a context completes, it validates and appends the exact
observation mapping that the existing shard receipt would have contained. The
journal does not know whether `status`, `support_features`, or unmatched values
are scientifically acceptable.

An accepted failure-shaped payload is still a completed journal work item and
is not retried. Receipt materialization can later refuse a completed legacy
receipt under the existing contract; remediation requires a user-owned new
execution identity rather than silent replacement.

**Alternative considered: persist candidate scores incrementally and resume
mid-context.** Deferred because the user selected context granularity and the
current scientific consumer already defines context-level observation
validation.

### 5. Materialize legacy receipt shards as a pure terminal projection

After every slot journal is mechanically terminal, load all validated record
views, index by context identifier, and regroup by the immutable legacy
`shard_index`. For each of the eight legacy shards, order observations exactly
as the existing `shard_contexts` contract expects and construct the current
receipt schema. Run the existing merger validator unchanged before publishing
write-once receipt bytes.

Do not copy physical slot, attempt, exit, or resume counters into the legacy
receipt. Those values are path-dependent mechanics. The materialized receipt
contains only stable plan bindings, observations, denominators, lineage, and
the fields the current merger accepts. Consequently identical observation
payloads produce identical receipt bytes whether they came from one attempt or
several.

**Alternative considered: modify the merger to read journals.** Rejected
because it would expand a scientifically sensitive consumer and break the
requested compatibility boundary.

**Alternative considered: preserve the old attempt/reused counters in the
terminal receipt.** Rejected because they make terminal bytes depend on
interruption history; the separate mechanics receipt is their new owner.

### 6. Join process exit with journal state outside scientific artifacts

The parent launcher knows the child PID, return code, and terminating signal.
After every exit it closes no journal bytes itself; it performs validated
read-only inspection and publishes a write-once mechanics exit receipt binding:

- physical slot and schedule fingerprint;
- attempt identifier and child-process identity;
- return code and signal;
- accepted and missing context counts;
- last durable sequence, work-item identifier, and record digest;
- journal plan/terminal state and temporary-path diagnostics.

A `SIGTERM` handler in the worker may best-effort publish an attempt failure and
close the model, but launcher observation remains authoritative for the OS exit.
If the process cannot publish an outcome, the attempt start remains visibly
unfinished. Neither path launches a successor automatically.

### 7. Separate deterministic equivalence from live mechanics evidence

The primary equivalence fixture uses deterministic context observations over a
production-shaped logical plan. It executes once uninterrupted and once with a
fresh-process interruption boundary, then requires byte equality for all eight
legacy terminal receipts and successful validation by the unchanged merger.
It also requires journal and exit diagnostics to differ in the expected
attempt-history fields.

The live gate uses one GPU and a bounded fresh mechanics root. An external
parent waits until at least one context record is durable, sends real
`SIGTERM`, records the child exit, then starts a separately requested
exact-identity continuation. It proves only signal, durability, missing-set,
model-lifecycle, and terminal-materialization mechanics; it is not a support
measurement and never writes an old sealed root.

## Interface Sketch

The consumer-specific surface remains small and mapping-based:

```python
schedule = plan_physical_slots(
    logical_contexts,
    slot_count=8,
    cost_field="scalar_equivalent_forward_count",
)

with open_slot_journal(
    root=fresh_slot_root,
    execution_identity=identity,
    logical_plan=logical_plan_reference,
    schedule=schedule,
    slot_index=slot_index,
) as slot:
    attempt = slot.start_attempt()
    for context in slot.missing_contexts():
        observation = execute_existing_support_context(context)
        slot.append_context(observation, attempt_id=attempt)

receipts = materialize_legacy_shard_receipts(
    logical_plan=logical_plan,
    schedule=schedule,
    slot_journals=slot_journal_roots,
)
```

The adoption patch supplies the glue from the named runner's current context
scoring loop into this interface. It does not copy the experiment's support
rule into stable infrastructure.

## Risks / Trade-offs

- **[A context is expensive enough that mid-context loss still hurts]** → Keep
  the user-selected context boundary, expose last durable context clearly, and
  defer candidate-level journaling until a real need justifies it.
- **[LPT cost is only a forward-count proxy]** → Label the schedule cost unit
  explicitly, preserve deterministic placement, and record measured slot wall
  time diagnostically without adapting the schedule.
- **[The active consumer changes before adoption]** → Bind its exact source
  digest; reject or regenerate the patch rather than applying fuzzily.
- **[A failed payload becomes mechanically complete]** → Keep journal
  completeness separate from legacy receipt eligibility; never retry or
  replace the accepted record automatically.
- **[Journal read projections leak mutable internals]** → Return frozen value
  objects or defensive copies after full validation and keep filesystem paths
  diagnostic-only.
- **[Signal arrives during atomic publication]** → Rely on temp-file,
  `fsync`, exclusive publish, and full reload semantics; only a validated final
  record counts as durable.
- **[Mechanics smoke is mistaken for research evidence]** → Use fresh roots,
  mechanics-only schema names, explicit claim exclusions, and no research
  result publication.

## Migration Plan

1. Implement and test the read-only journal projections without changing disk
   schema or writer behavior.
2. Implement the consumer-specific logical-plan and schedule adapter, then
   freeze a deterministic schedule receipt for the accepted 200-context plan.
3. Add per-slot journal execution and explicit continuation with fake scorers;
   verify no accepted context is re-executed.
4. Add pure legacy receipt materialization and run the unchanged merger and
   analyzer contract tests against its output.
5. Produce an exact-digest-bound patch or consumer example for the active
   runner without editing that worktree.
6. Pass focused CPU tests, full deterministic uninterrupted-versus-resumed
   receipt equivalence, strict OpenSpec validation, and an independent boundary
   audit.
7. Run the separately authorized single-GPU real `SIGTERM` continuation smoke
   into a new mechanics root and preserve its receipts under this change.
8. Fix all implementation, interface, patch, tests, and receipts in one local
   worktree commit. Do not push or adopt into the active unit.

Rollback removes the new adapter and read projections while leaving previously
written version-1 journal roots readable through the unchanged disk schema.
It never deletes or rewrites old sealed roots or active research files.
