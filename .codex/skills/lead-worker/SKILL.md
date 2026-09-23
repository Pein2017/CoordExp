---
name: lead-worker
description: Exchange bounded assignments and direct reports between a research lead and persistent worker with user-selected model and effort. The lead invokes this workflow; the worker reads it for its role. Use for a lead/worker split or reuse of an existing worker; ordinary parallel subtasks do not require this skill.
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

Discuss alternatives, define the decision and answer the user in the lead.
Delegate hands-on exploration, implementation, runtime operation and local
checks as one coherent package. Direct lead work is appropriate for the small
independent verification needed to accept the package.

The worker sends a self-contained report directly to the lead task at a
decision-bearing checkpoint or completion. Include status, the result or exact
question, key evidence/checks, resource and job state, and the stable artifact
path/hash. A bare "done" or wake pointer is insufficient. The lead reads that
message and inspects the exact artifacts needed for acceptance; do not fetch the
worker chat again to recover a report already delivered. Use history only for a
specific missing fact or ambiguous delivery. Reuse the worker while its ownership
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
worker reports/questions to the lead. Use an exposed task-message tool when
available and authorized; check its schema and model-setting behavior. An active
recipient receives compatible steering, not a duplicate turn. Native subagents
use `send_message` while active and `followup_task` while idle.

For an **existing same-host sidebar pair**, when the message tool is absent,
this installation has an established App Server Unix-socket route. The bundled
[scripts/worker_turn.py](scripts/worker_turn.py) reuses it. It does not create
threads or change model settings, permissions, cwd, or global configuration.
Inspect without starting anything:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py \
  --lead-thread LEAD_UUID --worker-thread WORKER_UUID --cwd /absolute/worktree
```

Send an authorized assignment/ruling to the worker from a UTF-8 file:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py \
  --lead-thread LEAD_UUID --worker-thread WORKER_UUID --cwd /absolute/worktree \
  --send --message /absolute/assignment.txt --receipt /absolute/dispatch.json
```

The worker returns its self-contained report using the same pair and `--to lead`:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py \
  --lead-thread LEAD_UUID --worker-thread WORKER_UUID --cwd /absolute/worktree \
  --to lead --send --message /absolute/report.txt --receipt /absolute/return.json
```

The default target is `worker`; `--to lead` reverses delivery without swapping
roles. The helper preserves model/effort, resumes an unloaded recipient without
overrides, and sends once. For an active recipient it reads only the latest turn
metadata (`thread/turns/list`, `itemsView=notLoaded`) and calls `turn/steer` with
the exact `expectedTurnId`. Otherwise it uses `turn/start` on an idle recipient.
It refuses unavailable targets, changed recipient settings, and reused receipt
paths. A stale active-turn precondition never falls back to starting another turn.
Each message has one sender and one receipt; this is not an atomic lock against
a concurrent UI operator. Model/effort choices remain assignment requirements,
not hardcoded transport allowlists.

A send timeout is **unknown delivery**, not permission to resend: inspect the
receipt and target task before any retry. Do not run `codex exec resume` beside
an App-owned active worker. If this local socket or its protocol is unavailable,
return the ready message and the specific transport gap; do not invent a
background runner or silently substitute a watcher. The helper is local-only,
not a remote-host discovery layer.

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

Require a stable candidate. Inspect exact outputs/diffs and replay the smallest
independent acceptance check. Worker reports use `candidate`, `NEEDS_CONTEXT`,
`HOLD` or `BLOCKED`; only the lead issues `lead-accepted`. Keep technical validity
separate from whether a research hypothesis was supported. Use the existing
research-flow or project contract for scientific scope, not a duplicate protocol.

Then either close, send one bounded correction, or choose the next package inside
the user's current authority. Stop at the declared stop condition, exhausted
budget, convergence, or a material user-owned decision. Do not convert permission
to execute one package into worker authority to run the research program.
