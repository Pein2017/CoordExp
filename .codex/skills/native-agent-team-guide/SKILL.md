---
name: native-agent-team-guidance
description: Coordinate an authorized native team with a decision-owning lead and bounded execution workers; use for delegation, asking parent, peer interfaces, milestone reports, and durable runtime handoff.
---

# Native Agent Team Guide

Use the role matching the current thread: main lead or subagent worker.
The [agent contract](../../AGENTS.md) owns authority and delegation permission.
This skill grants neither extra agents nor research launches.

## Priority: no polling

Use event-driven, long waits within actual tool and instruction limits, not
status/log/sleep loops. Do useful independent work while waiting; when only a
long external job remains, use the durable handoff below. Root owns wake-me-up;
workers do not remain active solely to monitor. Reconcile existing monitors
before transfer; never duplicate or silently abandon one.

## Main lead/thread

### Choose and bound the work

Keep small or tightly coupled work local. Use a few execution workers only when
independent work saves more than briefing, integration and acceptance cost.
Reconcile existing workers first; reuse a suitable worker through ordinary
corrections. Keep one writer per surface. Nested delegation requires explicit
applicable authorization and a concrete benefit; workers are not schedulers.

The lead owns direction, minimal implementation approach, shared interfaces,
acceptance, runtime continuation and scientific interpretation. Workers execute
bounded outcomes including local diagnosis, implementation and relevant checks.
Broader design or research autonomy must be explicit in the assignment.

Before splitting coupled work, settle producer/consumer entrypoints, necessary
artifact fields/paths and release authority using the existing contract. Resolve
unknown interfaces locally or through a bounded investigation before dependent
implementation; do not invent a schema merely to coordinate workers.

A brief needs only execution-changing facts:

```text
outcome and non-goals; cwd and owned paths;
existing entrypoint/pattern and chosen approach, or the unresolved question;
frozen behavior, interfaces and permissions;
acceptance evidence and stop condition;
relevant peers; runtime handoff to root when applicable.
```

Choose fork context from actual dependencies. Inherited history is a snapshot;
send later changes explicitly. A strong lead should supply a useful implementation
starting point without prescribing every local edit or duplicating worker work.
For pagination, resource budgets, recovery or state transitions, include a few
decision-bearing counterexamples with the acceptance criteria. The worker turns
them into caller-facing tests; concrete fixtures belong with the code, not here.

### Route models and intervene

Prefer Luna or Terra for bounded execution with a clear verifier when fit is
plausible. The lead may execute difficult, coupled work directly. Sol is an
available alternative, not the default autonomous package owner. Check callable
models, supported effort and fork inheritance; follow explicit user choices.
Do not default to max effort or maintain a fixed model/effort ladder. Select
based on uncertainty, risk, observed corrections and total cost to acceptance.

Answer worker questions with the smallest decision promptly. Check whether the
brief caused the detour; narrow scope, choose the missing seam or take over
uncertain design instead of repeatedly returning the same misunderstanding.
User-owned meaning still belongs with the user.

Observe concrete expansion signals: copied runners, new controllers for local
changes, repeated schema rebinding, wrappers around private internals, or tests
mostly validating new bookkeeping. Ask what real failure each addition closes.
Do not equate worker busyness, lines written or receipts produced with progress.

Preserve old evidence without automatically freezing every reusable source file.
Choose snapshots or a narrow versioned seam when sufficient. Never alter code
currently bound to a running job or silently relabel old results after a change.

### Accept and learn

Inspect the fixed candidate's exact diff and smallest decision-bearing check;
completion notifications and worker self-report are not acceptance. Keep execution validity separate
from scientific success. Repair a reducer using retained valid outputs rather
than rerunning expensive model work. Bundle blocking corrections; do not add
another broad review after an unchanged contract has passed its checks.

Record only observations that could change the next assignment in the existing
task record: failure, brief contribution, correction/takeover, and acceptance.
Do not create a routine interview, ledger, benchmark or extra reviewer per task.
If a cost comparison is actually requested, the optional
[evidence workflow](references/evidence-workflow.md) supports existing records.
Count lead/rework costs; unknown costs remain unknown. One task does not establish
a model's capability ceiling or specialty. Persistent memory changes still need
user authorization.

## Subagent worker

### Execute and ask parent early

Own bounded implementation and ordinary repairs within the accepted approach,
not cross-task architecture, scientific meaning or new gates. Discover simple
local facts directly; ask parent before prolonged searching, guessing a shared
API, changing an invariant, adding a release schema or expanding scope. Send
known facts, the precise uncertainty and a minimal option if evident, not a full
alternative design. While awaiting a ruling, do only independent work; otherwise
return the question and resumption point. Silence is not approval.

### Keep the implementation small

- Locate the real entrypoint and closest existing pattern before adding modules.
- Prefer a direct change or thin reuse over a copied runner or parallel framework.
- Ask parent before cloning function globals, monkeypatching private internals,
  or introducing a controller/adapter/schema stack for a local behavior change.
- Explain the actual blocking case for new machinery. Do not manufacture a
  compatibility layer for an interface with no consumer obligation.
- Verify the requested behavior at its real caller/consumer. Keep identity,
  masking, loss accounting and recovery invariants; avoid tests that merely
  mirror newly invented wrappers. Stop when the bounded checks pass.

### Report and coordinate

Send parent a short milestone report when a real check completes, scope starts
to expand, an uncertainty blocks work, a durable job launches, or work fails.
Routine keystrokes and elapsed time need no report. Use:

```text
finding/result -> evidence path -> impact -> action needed (or none)
```

Final return includes outcome, changed paths, checks, unresolved questions and
any external job handoff. Native final completion already reaches the parent;
do not duplicate it with an identical message. Keep raw logs in artifacts.
After returning a candidate, stop writing so the lead reviews a stable target.
If another correction is needed, notify the lead before editing, agree ownership,
then return the updated candidate with affected checks rerun. The lead may take
over a bounded repair; never write concurrently with its acceptance work.

Peers may exchange facts within the frozen contract, not grant authority or
change shared interfaces. Summarize decision-changing exchanges to parent;
affected peers must acknowledge interface changes before dependent work resumes.
Shared files alone are not notification; ordinary evidence needs no handshake.

Use send_message for active peers/parent. It does not start an idle thread;
followup_task resumes a non-root worker. Roster lookup is for reconciliation,
not polling. Interrupting an agent does not establish that its external job
has stopped; reconcile process ownership before any replacement launch.

## Durable job handoff

The worker may launch only within its authorization. Verify that the producer
survives tool/agent return, saves completed units, and exposes failure as well as
success. Use the existing durable runtime rather than building a new scheduler.

Send root the exact command/run identity, PID and tmux/session identity, stable
log and result locations, terminal success/failure signals, checks already done,
and next action. Then return; job completion remains outstanding.

Root verifies the handoff and arms one wake-me-up monitor for its own thread,
covering success, failure and producer exit as supported. Confirm state=armed
before ending the lead turn. If arming fails, resolve the handoff explicitly;
do not claim that a wake will occur. Follow the plugin's delivery/status rules
when awakened, inspect terminal artifacts, and accept or resume missing work.
Do not rerun completed units merely because the delivering agent returned.
