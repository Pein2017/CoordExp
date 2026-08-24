## Context

See [proposal.md](proposal.md) for motivation. The existing admission manifest
already captures typed file, directory, executable, and strict-value identity,
and it revalidates those bindings while accepting stages. The production-shaped
vertical driver, however, invokes its worker before it appends and revalidates
the `vertical_smoke` stage. The former branch/commit history previously checked
out in the fixed `/data/CoordExp/.worktrees/research-probe-infras` directory is
retained as provenance only. That existing directory is the active integration
lane, is never moved, recreated, or retired, and its current successor branch
is based on clean `research-probes` commit
`67ad6586bcf3ba5583de5d5d1498b57cfa8c75f5`.

## Goals / Non-Goals

**Goals:**

- Make the research target that a consumer actually executes explicit and
  mechanically revalidatable before any model-launch action.
- Preserve a durable CPU preflight if a later target check fails, while making
  the failed continuation non-launching and non-finalizing.
- Reuse the existing admission/journal seam through both current consumers
  without absorbing their scientific vocabulary or result fields.

**Non-Goals:**

- A generic runner, scheduler, DAG, retry framework, plugin, or artifact
  abstraction.
- A full-tree source snapshot, broad source migration, TensorFlow dependency,
  data-manipulation framework, or a change to model/experiment semantics.
- GPU execution, bulk research launch, or reinterpretation of sealed receipts.
- Moving, recreating, or retiring either fixed research worktree directory.

## Decisions

### Bind a clean worktree plus declared effective inputs

The admission owner will introduce one opaque target-tree identity whose
observable projection contains a resolved root, full Git commit, clean-status
result, and named typed identities for exactly the files and runtime/config
inputs the consumer declares it will use. It is admission-owned outer metadata,
not another `BindingManifest` kind; its named effective inputs remain the
existing closed typed bindings. The consumer remains responsible for declaring
its executable surface; the shared owner validates identity, durability, and
failure behavior.

An alternative was commit-only binding. It is insufficient because a dirty
target or a loaded input outside the intended declaration can differ while a
commit string remains unchanged. A second alternative was digesting the entire
worktree. It would increase cost and false invalidation without proving that an
undeclared runtime input is the one the launcher will use. The selected hybrid
is narrow: clean Git state closes undeclared tracked/untracked drift, while
typed effective inputs close execution-critical resolution drift.

### Put revalidation at the pre-model choke point

The vertical consumer will call an admission-owned revalidation operation after
CPU evidence exists but immediately before `launch_slot_worker` or an
equivalent model/launcher boundary. The operation must occur before subprocess
creation, model load, GPU allocation, vertical output publication, and any
consumer finalization. Its failure leaves the CPU journal durable and does not
append a partial vertical record.

Revalidating only at `append_stage` is rejected because it is post-launch.
Revalidating opportunistically in helpers is rejected because an alternate
consumer path could bypass it. A new third journal stage is rejected because
the stable two-stage evidence contract remains sufficient; this is a gate
inside the existing vertical transition, not additional evidence.

### Make dirty targets fail closed in v1

The user-selected policy is to reject any dirty target. Supporting a dirty
target would require a complete loaded-byte closure and a policy for generated
or ambient imports, expanding the seam before a demonstrated need. A probe
that intentionally works from dirty code remains experiment-local and cannot
claim a reusable admission certificate.

This is an execution policy, not a demand that every shared checkout always be
clean. A planning-dirty `research-probes` checkout simply cannot be the target
of that reusable admission at that moment. The normal model-launch target is a
clean `probe/<ticket>` worktree forked from an immutable research baseline;
current fixed consumers continue to declare their actual target explicitly.

### Keep sources of scientific meaning caller-owned

The shared layer will not model cohorts, prefixes, interventions, owner
matching, metrics, outcome payloads, TensorFlow backends, or stop rules. It
will bind their consumer-owned files as opaque inputs when declared. Existing
sealed receipts remain historical mechanics evidence and are not revalidated or
rewritten.

## Risks / Trade-offs

- [A clean target changes during a long setup] → Revalidation at the last
  pre-model boundary avoids model cost; the user explicitly starts a new
  admission rather than automatic continuation.
- [A consumer omits an effective input] → Require each consumer to enumerate
  its entrypoint, producer, validator, config/runtime inputs and use focused
  sentinel tests that mutate each category.
- [Git identity is unavailable in a target] → Fail closed; do not substitute a
  cwd or another worktree.
- [Directory/branch topology changes] → Bind paths by resolved identity and
  commit, not an assumed branch name; retain the user's fixed worktree
  directories without treating historical branch names as runtime authority.

## Migration Plan

1. Add target-binding capture/revalidation and deterministic failure tests in
   the shared admission owner.
2. Adapt the current natural-boundary support and K10-H20 crossover consumers
   to declare their exact target/effective inputs and run their CPU round trips.
3. Put the pre-model check in the production-shaped support vertical path and
   prove a mutation sentinel cannot reach its launcher.
4. Review the frozen infra diff; only the infra change's separately authorized
   integration gate may merge it into `research-probes`.
5. Re-run target-bound validation from the actual merged `research-probes`
   tree. A bounded GPU smoke, if needed by the existing stable gate, remains a
   separate user-authorized execution decision.
