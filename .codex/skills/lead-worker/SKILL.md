---
name: lead-worker
description: Create user-requested persistent workers and exchange bounded assignments or direct reports with user-selected model and effort. Use for a lead/worker split, reuse of an existing worker, or a same-host fallback when Codex App creation or messaging tools are absent; ordinary parallel subtasks do not require this skill.
---

# Lead / Worker

Keep one decision owner and one persistent execution owner per package. The user
converses with the lead. The worker owns implementation and evidence production;
the lead owns interpretation, acceptance, and the next assignment.

## Bind roles, pair and user choices

The lead normally invokes this skill proactively when the authorized work fits
this split. A worker reading it remains the worker; reading does not authorize
creating another main or changing research direction.

Before binding or dispatching the pair, read the [agent contract's Model routing](../../AGENTS.md#model-routing)
for user model/effort selection and role boundaries, and
[Agent topology and delegation](../../AGENTS.md#agent-topology-and-delegation)
for fork choice. Record the selected settings in the assignment and compare them
with actual task settings; resolve mismatches through an authorized route.

- Reuse a suitable designated worker. Record the pair and assignment in the
  existing task artifact, not another registry. Create a sidebar task only when
  the user explicitly requests one; native subagents serve temporary subtasks,
  not a requested persistent sidebar worker.

## Keep the lead useful and small

Keep discussion, decisions and independent acceptance with the lead; delegate
the agreed execution package. Unless the user requires decision-only work,
resolve cheap facts or calculations needed to frame or accept that package
locally when cheaper than dispatch. Do not repeat the worker's investigation
or enter its owned write surfaces.
Give the user the changed evidence, decision and next action; link the retained
report for details. Combine related progress into one update rather than
relaying worker reports or narrating each check and handoff.

The worker sends a self-contained report directly to the lead task at a
decision-bearing checkpoint or completion. Include status, the result or exact
question, key evidence/checks, resource and job state, and relevant source paths.
Provide immutable identities/hashes when acceptance or downstream work depends
on them; a routine question needs no new artifact. A bare "done" or wake pointer
is insufficient. The lead inspects the exact artifacts needed for acceptance;
do not fetch the worker chat again to recover a report already delivered. Use
history only for a specific missing fact or ambiguous delivery. Reuse the worker while its ownership
and context remain sound.

Give the worker execution-changing facts, preferably through an existing file:

```text
Assignment and source of current authorization:
Roles, lead/worker task identities, user-selected worker model and effort (required):
Direct return target and supported message route:
Applicable AGENTS/skills and authoritative task artifacts to read:
Outcome / non-goals:
Cwd, owned write surfaces, shared inputs:
Frozen question or behavior, decisive sources and constraints:
Blocking decisions/tasks and independently verifiable delivered behavior:
Acceptance evidence, explicit user resource limits if any, stop rule:
Return candidate + changed paths + checks + unresolved questions + job status.
Stop after the package; no self-acceptance or next-round launch.
```

The worker may make reversible implementation choices inside this boundary.
It returns scope/meaning/cost conflicts to the lead, with evidence and one clear
question. The lead resolves discoverable facts and preserves user-owned choices.
For a failed attempt or decision-bearing event, follow
[Checkpoints and waiting](../../AGENTS.md#checkpoints-and-waiting). Send the
supervising main the evidence, impact and proposed next step; identify any
research-lead ruling needed before dependent work resumes.

## Dispatch using the smallest available transport

Communicate directly in both directions: lead assignments/rulings to the worker,
worker reports/questions to the lead. First discover the relevant App tool:
`create_thread` for an explicitly requested new task, `send_message_to_thread`
for an existing task. Check its current schema and model-setting behavior.
For native subagents, follow the messaging and interruption rules in
[native-agent-team-guidance](../native-agent-team-guide/SKILL.md#report-and-coordinate).

If the relevant App tool is absent or lacks a required capability, use the
bundled [scripts/worker_turn.py](scripts/worker_turn.py) through this installation's
same-host App Server socket. Read [Same-host transport](references/same-host-transport.md)
for creation, inspection, delivery and recovery commands before using that route.
Missing wrappers do not establish missing backend capability; do not hand the
user a manual copy/paste task before checking this
route. A permission rejection or an ambiguous mutation is not a capability gap
and must not trigger an alternate-path retry. This fallback does not discover
other hosts, change global configuration, or replace a requested sidebar worker
with a temporary subagent.

A send or creation timeout is **unknown delivery**: inspect the receipt and
target before retrying. Do not duplicate creation, start another turn, or switch
transport after an ambiguous mutation. Submission is not completion.

## Completion, acceptance and continued conversation

Agree the direct return route before the lead ends its turn. The worker sends
its report to that task before ending at a checkpoint; an idle lead receives a
new turn, while an active lead receives steering. Do not require wake-me-up,
log markers, or a wake-then-read-chat chain for ordinary lead/worker exchange.
When replacing an existing watcher, verify the worker received the direct-return
instruction before cancelling that watcher to avoid a notification gap.

Continue independent lead discussion while the worker runs. Compact
`wait_threads` events are optional for an active wait or delivery reconciliation;
keep cursors and do not poll or fetch routine transcripts. Native subagents keep
their native completion messages. If no direct route is available, report the
limitation rather than promising an automatic return. Transport submission,
completion notifications and `idle` are not scientific acceptance.

Accept a stable candidate under
[Acceptance and review](../../AGENTS.md#acceptance-and-review).
Use the existing research-flow or project contract for scientific scope and
technical versus scientific disposition.

Record acceptance in the existing research record and receipt, then tell the
user. If the worker has stopped at its package boundary, close without reopening
it for a conclusion or acknowledgment. Send an active worker only rulings that
change its next action; carry accepted results needed later as pointers and
deltas in the next assignment. Chat is transport, not another progress ledger.
Then close, send one bounded correction, or choose the next package inside the
user's current authority. Stop at the declared stop condition, exhausted budget,
convergence, or a material user-owned decision. Do not convert permission to
execute one package into worker authority to run the research program.
