---
name: lead-worker
description: Coordinate a user-requested persistent lead/worker pair in independent Codex or Pi Web sessions, with bounded assignments and direct reports. Ordinary subagent delegation uses native-team guidance.
---

# Lead / Worker

Use this skill only for a persistent worker the user explicitly requested or an
existing worker assignment. Reading it does not authorize creating a worker or
expanding a package. A persistent worker is an ordinary independent session,
not a child agent; role labels belong in the assignment. Keep one owner and
writer per semantic surface.

## Bind the package

Follow the [native-team outcome-package and continuation rules](../native-agent-team-guide/SKILL.md#assign-and-choose-the-team)
for scope, acceptance, repair and escalation. The assignment should also make
its collaboration shape explicit: name relevant peers and dependencies, where
the result/report goes, what unaffected work may continue, and what change needs
the lead or user. Use the concise [coordination examples](references/coordination-examples.md)
rather than inventing a second team policy.

Reuse a designated worker while its task and context remain reliable. Create a
new persistent/root session only on explicit user request; choose model and
effort under the [user contract](../../AGENTS.md#model-routing), and ask only
for missing user-owned choices. Keep assignments in the existing research or
change owner, not a new registry.

For CoordExp, use the existing task/report owner and follow
[`docs/storage-ownership.md`](../../../docs/storage-ownership.md) for message,
report and receipt placement. `--message` and `--receipt` have different roles
and may have different owners or locations; do not treat either as a shared
asset or put a report in root `outputs/` by default.

## Choose transport by recipient backend

The recipient's harness determines the transport and ID namespace; the caller's
harness does not. These routes do not translate identities across backends.

- **Codex recipient:** discover the current native App tool/schema and retain
  its `send_message_to_thread` route for an existing task. When that route is
  absent or lacks the needed capability, use Codex's same-host `worker_turn.py`
  only as described in
  [same-host transport](references/same-host-transport.md). Pi peer flags are
  not Codex-to-Codex features.
- **Pi Web recipient:** any caller harness, including Codex, may invoke
  [`pi_worker_turn.py`](scripts/pi_worker_turn.py) for a Pi Web recipient. Supply
  actual Pi Web session IDs and the recipient's physical cwd (or explicitly
  selected Pi participant aliases); a Codex thread ID is not a Pi ID and cannot
  be bridged or substituted. Read [Pi transport](references/pi-transport.md)
  before use. It targets one ordinary Pi Web session at a time on the declared
  endpoint; it is not `Agent`, `resume`, or subagent messaging.
- **Ordinary child agents:** use the current harness's native agent tools and
  [native-team guidance](../native-agent-team-guide/SKILL.md), only when that
  topology is wanted and authorized.

A missing wrapper alone does not prove a backend is unavailable. A permission
rejection or ambiguous mutation is not a capability gap and never permits
switching transports. Establish a direct report route in the assignment. A send
receipt records submission status, not work completion or acceptance.

## Return and accept

Use the shared [checkpoint and waiting rules](../../AGENTS.md#checkpoints-and-waiting)
and [native-team report and acceptance contract](../native-agent-team-guide/SKILL.md#report-and-accept).
A timeout or idle state is not acceptance; do not fetch routine transcripts.
For research, record accepted findings and limits at their existing owner.
Long external jobs use the guide's [durable handoff](../native-agent-team-guide/SKILL.md#durable-job-handoff);
interruption alone does not stop a job.
