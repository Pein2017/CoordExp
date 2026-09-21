---
name: lead-worker
description: Let a research lead dispatch bounded execution to a persistent worker with user-selected model and effort. The lead invokes this workflow; the worker reads it for its role. Use for a lead/worker split or reuse of an existing worker; ordinary parallel subtasks do not require this skill.
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

Read compact worker status and final reports first. Expand exact artifacts only
for a decision-bearing uncertainty. Never ingest the entire worker transcript
or routine tool logs just to stay informed. Contexts are separate, but returned
summaries and evidence still consume lead context; this is not zero-cost or a
security boundary. Reuse the worker while its ownership and context remain sound.

Give the worker execution-changing facts, preferably through an existing file:

```text
Assignment and source of current authorization:
Roles, lead/worker task identities, user-selected worker model and effort (required):
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

Use an exposed task-message tool when available and authorized. Check its actual
schema and model-setting behavior. An active worker receives compatible steering
through a supported steering surface; do not start a duplicate turn. Native
subagents use `send_message` while active and `followup_task` while idle.

For an **existing same-host sidebar worker**, when the message tool is absent,
this installation has an established App Server Unix-socket route. The bundled
[scripts/worker_turn.py](scripts/worker_turn.py) reuses it. It does not create
threads or change model settings, permissions, cwd, or global configuration.
Inspect without starting anything:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py \
  --lead-thread LEAD_UUID --worker-thread WORKER_UUID --cwd /absolute/worktree
```

Only after an execution assignment is authorized, send its UTF-8 text file:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py \
  --lead-thread LEAD_UUID --worker-thread WORKER_UUID --cwd /absolute/worktree \
  --send --message /absolute/assignment.txt --receipt /absolute/dispatch.json
```

The helper is model/effort-neutral and requires an idle or unloaded worker.
It resumes an unloaded worker without setting overrides, checks again,
and sends once. It refuses a reused receipt path. One dispatcher owns the pair;
this preflight is not an atomic lock against another UI operator. User model
and effort choices are assignment/prompt requirements, not transport allowlists.
The lead checks them before dispatch and uses an authorized configuration route
if needed; the helper never overrides them.

A send timeout is **unknown delivery**, not permission to resend: inspect the
receipt and target task before any retry. Do not run `codex exec resume` beside
an App-owned active worker. If this local socket or its protocol is unavailable,
return the ready assignment and the specific transport gap; do not invent a
background runner. The helper is local-only, not a remote-host discovery layer.

## Completion, acceptance and continued conversation

Use compact `wait_threads` snapshots/events for a sidebar worker and native
completion events for subagents. Keep current cursors. Do not poll logs or paste
unchanged status. Continue independent lead discussion while execution runs.

If returning while work remains, discover and read the currently installed
`codex-wake-me-up:wake-me-up` skill only when its trigger applies; verify an armed handoff before promising an
automatic return. Otherwise state clearly that the lead must be resumed manually.
Completion notifications, queue acceptance and `idle` are not scientific success.

Require a stable candidate. Inspect exact outputs/diffs and replay the smallest
independent acceptance check. Worker reports use `candidate`, `NEEDS_CONTEXT`,
`HOLD` or `BLOCKED`; only the lead issues `lead-accepted`. Keep technical validity
separate from whether a research hypothesis was supported. Use the existing
research-flow or project contract for scientific scope, not a duplicate protocol.

Then either close, send one bounded correction, or choose the next package inside
the user's current authority. Stop at the declared stop condition, exhausted
budget, convergence, or a material user-owned decision. Do not convert permission
to execute one package into worker authority to run the research program.
