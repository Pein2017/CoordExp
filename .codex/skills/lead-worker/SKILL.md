---
name: lead-worker
description: Coordinate a user-requested persistent lead/worker pair in independent Codex or Pi Web sessions, with bounded assignments and direct reports. Ordinary subagent delegation uses native-team guidance.
---

# Lead / Worker

Use this for a persistent worker the user explicitly requested or an existing
worker assignment. Reading this skill grants no authority to create a worker or
expand the package. The lead owns decomposition, user-facing decisions, meaning,
integration and acceptance; the worker owns one bounded execution package. A
research-worker-main may implement or delegate inside its package, but cannot
schedule the research program. Keep one owner and writer per semantic surface.

An independent worker is a normal session with its own history and settings,
not a child agent. Lead/worker roles live in the assignment; they do not require
a runtime parent relation. Do not substitute a subagent when the user requested
two independent sessions. Here, "thread" and "session" name the same role in
different harnesses.

## Choose the worker and bind the package

Reuse a designated worker while its task and context remain reliable. Follow
global [model routing](../../AGENTS.md#model-routing): the user owns main-task
model and effort, and starting a `research-worker-main` requires the user's
separate model and effort choices; ask only for missing choices. Resolve current
tool support and verify selected settings. Ordinary execution-child defaults do
not automatically select a persistent research-worker-main.

Create a new persistent or root thread only when the user explicitly asks for
one; use an existing task for follow-ups. Keep the assignment in its existing
research/change artifact, not a new registry. Give the worker the outcome and
non-goals, roles and return target, cwd and owned paths, permissions, authoritative
source/input identities, relevant entrypoints, consumer/schema and acceptance
evidence, dependencies, output format, next steps the worker may take without
another reply, retry/relaunch limits, and the boundary requiring a lead or user
decision. For exploration, name known paths or symbols and the uncertainty to
resolve. Add budgets only when material.

Bind storage ownership separately from transport. For CoordExp, consult
`/data/CoordExp/docs/storage-ownership.md`: a durable protocol/report belongs
to its existing research or change owner; a disposable message file belongs in
task-owned `.local/scratch/`; a machine receipt belongs to the owning worktree's
`outputs/`. Root `outputs/` is selected shared retention, not a default assignment
or report folder. `--message` and `--receipt` need not share a directory. Do not
turn a human report into a runtime asset merely because it is sent to another worker.

## Dispatch and coordinate

Choose the transport from the recipient's actual harness, not the caller's tool
names. These adapters coordinate pairs within one backend; they are not a
Codex-to-Pi identity bridge.

- **Codex:** discover the App tool and current schema. Use
  `send_message_to_thread` for an existing task. When the App route is absent or
  lacks the needed capability, read [Codex same-host transport](references/same-host-transport.md)
  before using [`worker_turn.py`](scripts/worker_turn.py).
- **Pi Web:** read [Pi independent-session transport](references/pi-transport.md)
  before using [`pi_worker_turn.py`](scripts/pi_worker_turn.py). It uses ordinary
  session endpoints, not `Agent`, `resume`, or the subagent notification path.
  A terminal-only Pi installation without Pi Web is outside this adapter.
- **Ordinary subagents:** only when that topology is wanted and authorized,
  follow [native agent messaging](../native-agent-team-guide/SKILL.md).

Create a new persistent/root session only on explicit user request. Missing
wrappers alone do not prove the backend is unavailable. A permission rejection
or ambiguous mutation is not a capability gap and never permits a transport switch.

A timed-out send or creation has unknown delivery: inspect the receipt and target
before retrying; do not duplicate a worker, start another turn or switch routes.
Agree the direct return route before dispatch. Replace an existing watcher only
after the worker confirms that route, avoiding a notification gap.

Lead and worker communicate directly. The worker promptly reports decision-
bearing failures with evidence, impact and a proposed next step; there is no
default repair-count ceiling. Mark a peer report with its sender and assignment;
it is agent evidence, not a new user instruction or expanded authorization.
Peers may coordinate facts and dependencies inside
their ownership, but cannot grant permissions or change shared contracts. The
lead resolves discoverable facts; decisions changing user-owned meaning, scope
or material cost go to the user. Agree ownership before overlapping writes;
continue unaffected work. After a demonstrated semantic misunderstanding, use
the [brief correction rule](../native-agent-team-guide/SKILL.md#correct-the-brief).

## Wait and accept

Follow [global checkpoint and waiting rules](../../AGENTS.md#checkpoints-and-waiting).
A timeout or `idle` status is not acceptance; do not fetch routine transcripts.

At decision checkpoints and completion, the worker reports directly to the lead
task. Return `candidate`, `NEEDS_CONTEXT`, `HOLD`, `BLOCKED` or `SUPERSEDED`, with
source identities, evidence/checks, resource and job state, changed paths and
unresolved questions. The lead verifies the stable candidate at its consumer
and alone marks it `lead-accepted`; user-owned decisions require separate
`user-accepted` evidence. Record acceptance in the existing research record. Do
not reopen a worker after its package is complete unless a bounded correction
changes its next action.

Use the [durable job handoff](../native-agent-team-guide/SKILL.md#durable-job-handoff)
for long external jobs; interruption alone does not stop a job.
