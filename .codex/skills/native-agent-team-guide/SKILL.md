---
name: native-agent-team-guidance
description: Coordinate an authorized native team with a decision-owning lead and bounded execution workers; use for delegation, asking parent, peer interfaces, milestone reports, and durable runtime handoff.
---

# Native Agent Team Guide

Use the current role: research-lead-main, research-worker-main, or execution subagent.
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
corrections. Keep one writer per surface. A research-worker-main may implement
directly or coordinate execution children within its assigned package; it does
not independently schedule the research program.

The lead owns direction, minimal implementation approach, shared interfaces,
acceptance, runtime continuation and scientific interpretation. Workers execute
bounded outcomes including local diagnosis, implementation and relevant checks.
Broader design or research autonomy must be explicit in the assignment.

Assign an independently verifiable outcome, not just a set of files. Establish
known dependencies and owners before dispatch; leave local implementation choices
to the worker. Resolve uncertainty that blocks dependent work first, without
trying to predict every overlap or inventing interfaces just to divide tasks.
Prefer packages that deliver a complete, independently verifiable behavior
through the real consumer. Declare only actual blocking dependencies; do not
split by layers merely to create parallel work. For a mechanical migration that
cannot stay valid in slices, choose a coherent integration boundary and its
verification; add transitional compatibility only when it is actually needed.
Assignments may change as facts emerge; make the new ownership explicit.
For research or review assignments, name what next action each plausible result
changes. If none changes, do not delegate; unresolved findings stay `HOLD` and
do not become new mandatory gates.
For an already frozen CoordExp probe, bind the assignment to the
[Frozen Probe Execution Packet](../research-flow/references/probe-execution-packet.md)
rather than sending broad research history.

A brief needs only execution-changing facts:

```text
outcome and non-goals; cwd and owned paths;
existing entrypoint/pattern and chosen approach, or the unresolved question;
authoritative inputs and consumer schema, including valid empty/HOLD states;
frozen behavior, interfaces, permissions and explicit user resource limits if any;
acceptance evidence and stop condition;
relevant peers; runtime handoff to root when applicable.
```

Choose fork context from actual dependencies. Inherited history is a snapshot;
send later changes explicitly. A strong lead should supply a useful implementation
starting point without prescribing every local edit or duplicating worker work.
Include decision-bearing counterexamples where correctness is fragile. Workers
verify the assigned outcome at its real consumer, not only their edited files.

### Route models and intervene

Before selecting models, effort or fork context, read
[Model routing](../../AGENTS.md#model-routing) and
[Agent topology and delegation](../../AGENTS.md#agent-topology-and-delegation).
For pairing with a persistent worker and dispatching its assignment, read
[lead-worker](../lead-worker/SKILL.md).

Answer worker questions with the smallest decision promptly. Check whether the
brief caused the detour; narrow scope, choose the missing seam or take over
uncertain design instead of repeatedly returning the same misunderstanding.
After a demonstrated semantic misunderstanding, clarify the governing invariant
or take over the coupled part; increasing effort alone is not a correction.
User-owned meaning still belongs with the user.

Intervene when work no longer advances the assigned outcome or coordination
cost exceeds useful independence. Adjust the split or take over rather than
adding process. Preserve evidence identity: coordinate changes to shared inputs
used by running work, and do not treat earlier checks as proof of a changed
candidate. This does not require serializing independent work.

### Accept and learn

Inspect the fixed candidate's exact diff and smallest decision-bearing check;
completion notifications and worker self-report are not acceptance. Keep
execution validity separate from scientific success. Reuse valid evidence rather
than repeating completed work. Bundle blocking corrections; do not add
another broad review after an unchanged contract has passed its checks.
For decision-bearing semantics, include a source-grounded check that distinguishes
the intended interpretation from the nearest plausible wrong one. A self-check
using the implementation's own assumed constants is insufficient evidence.

Record only observations that could change the next assignment in the existing
task record: failure, brief contribution, correction/takeover, and acceptance.
Do not create a routine interview, ledger, benchmark or extra reviewer per task.
If a cost comparison is actually requested, the optional
[evidence workflow](references/evidence-workflow.md) supports existing records.
Evaluate total cost through acceptance, including lead corrections and rework;
unknown costs remain unknown. One task does not establish
a model's capability ceiling or specialty. Persistent memory changes still need
user authorization.

## Subagent worker

### Execute with local ownership

Own local investigation, implementation, tests and directed repairs within the
assignment. On failure or a decision-bearing event, use the escalation policy in
[Checkpoints and waiting](../../AGENTS.md#checkpoints-and-waiting) and the report
format below. Reuse the real entrypoint and existing patterns; choose the
smallest coherent solution. Do not ask the lead to decide facts you can cheaply discover.
Ask when uncertainty affects shared commitments, scope, authority or the accepted
approach, rather than guessing or pursuing a prolonged detour. Send known facts,
the precise question and a minimal option if evident. While waiting, continue
independent work; otherwise return a clear resumption point. Silence is not
approval. Stop when the assigned outcome and relevant checks are complete.

Load decision-bearing constants, identities and denominators from the bound
source artifacts rather than reconstructing them from remembered prose. If the
sources conflict on research meaning, show the conflict to the parent before
implementing an interpretation.

### Coordinate directly

Contact relevant peers directly for facts, dependencies and local coordination;
the lead need not relay every message. Share changes that affect another owner
early, with the affected surface and what action is needed. Ordinary findings
need no acknowledgement ceremony.

When work overlaps, agree who owns the shared change and what the dependent
worker can rely on before proceeding on that surface. Notification alone does
not resolve conflicting writes or assumptions. Keep one writer per surface;
handoffs identify the current candidate, remaining work and new owner. Peers may
coordinate within their assignments, not grant permissions or silently change
shared contracts. Send ownership changes and decision-changing agreements to the
lead; unresolved conflicts or changes to overall scope/architecture return to
the lead. Unaffected work can continue.

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
