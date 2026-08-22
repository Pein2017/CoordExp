---
name: wake-me-up
description: When a same-host GPU, tmux, PID, or log-producing job — or another local Codex thread — will block the current Codex task for at least 15 minutes and no useful independent work remains, arm one guarded host-local monitor instead of sleeping or polling. Prefer goal-independent `wait_for_event`; use explicit goal delivery only when pause/reactivation semantics are required.
---

# Wake Me Up

Use this only after the external job is launched, its same-host identity is
known, it is expected to block for at least 15 minutes, and every remaining
step depends on it. Do useful independent work first.

1. Call `wait_for_event` once with the exact local task ID, a stable
   `idempotency_key`, one typed condition, and bounded expiry. This is the
   default whether the task goal is null, active, paused, or blocked. Never call
   `create_goal` or mutate a goal to satisfy monitoring.
2. Registration is a local same-user best-effort target, not proof that the
   supplied ID is the caller. It authorizes one later billed FIFO turn. If the
   exact task is stored but unloaded, it also authorizes resume of that task and
   release of user items already ahead of the pointer; status records those item
   identities before resume.
3. Use `defer_goal_until_event` only when an eligible active or paused goal
   explicitly needs pause/reactivation semantics. `wake_me_up` and
   `wake_me_up_defer` are legacy goal aliases, never fallbacks for unavailable
   thread delivery. No operation creates a goal.
4. Choose one typed condition: prefer `receipt_success`; otherwise use
   `log_pattern`, `thread_idle`, `tmux_exit`, `pid_exit`, `gpu_stable`, or
   `time`. Thread delivery may carry any typed witness without upgrading it to
   success. Goal delivery still requires deliberate heuristic authorization.
5. Report the monitor ID, condition, delivery kind, and expiry, then end the
   current turn. The daemon owns observation after the prompt registration
   response. A 30-second MCP timeout is not the condition lifetime.

If experimental queue capability, exact-target read, daemon source/epoch, or
`thread.can_accept_direct_input` preflight fails, report `ThreadDelivery`
unavailable. Do not create a goal, choose a legacy alias, start a CLI-resume
process, or substitute ordinary turn start, steer, inject, Desktop messaging,
`thread/queue/start`, or another app-server/Core writer.

The pointer queue is shared user state and not exactly-once. The plugin makes
one queue-add attempt, then reconciles exact history, exact queue, and 60 seconds
of online absence. User edit/delete/reorder, interruption, queue capacity, crash,
archive, and storage ambiguity remain visible; never re-add blindly or claim a
delivery deadline.

## Producer terminal events

When the existing command or worker harness supports a terminal-event
publisher, reserve its expiring, single-use capability before launch, hand the
capability to that harness, launch through the harness, and then bind it in the
monitor registration or in `wake_me_up_defer`. A fast producer may publish
before binding; if binding commits before the pre-bind deadline, it retains
that immutable event for the monitor lifecycle. End the lead turn immediately
after a successful bind/defer. The producer may send bounded heartbeats and
then publish exactly one terminal event; after the wake, make one status call
and independently review any worker candidate.

Use the frozen MCP operations `wake_me_up_event_reserve`,
`wake_me_up_event_status`, `wake_me_up_event_cancel`,
`wake_me_up_event_heartbeat`, and `wake_me_up_event_publish`. Reserve,
heartbeat, and publish take only a path to a current-user-owned mode-`0600`
regular JSON file of at most 64 KiB; never place the token in a CLI/MCP
argument. The equivalent CLI commands are `event-reserve --payload`,
`event-status --reservation-id`, `event-cancel --reservation-id`,
`event-heartbeat --payload`, and `event-publish --payload`.

The raw publish token is available only on first creation. Capture that first
response or request a mode-`0600` `publisher_descriptor_path`; an idempotent
retry returns identity and fingerprint, never the bearer again. Native
subagents and HarnessDock workers may call the reference
`publish_worker_terminal_from_descriptor` adapter with the same
`worker_terminal` envelope. The plugin hands this capability to an existing
harness; it never launches commands, runs raw shell predicates, becomes a job
scheduler, or adds another wake claimant.

Keep the two condition leaves and their meanings separate:

- `command_terminal` covers `succeeded`, `failed`, `cancelled`, and `signaled`.
  Exit code zero is only a bounded command process outcome.
- `worker_terminal` covers `delivered`, `blocked`, `failed`, and `cancelled`.
  `delivered` carries one candidate commit for review. `git commit` exit 0 is
  not delivery acceptance, and even valid Git attestation is not lead or task
  acceptance.

All worker terminal outcomes wake when bound. Invalid, missing, baseline-
mismatched, out-of-scope, or errored delivery attestation still wakes the lead
with the failure evidence; never treat it as success or wait for expiry. The
plugin performs no Git watcher, merge, cherry-pick, revert, stage, commit, or
push operation.

Use `heartbeat_stale` only as heuristic liveness evidence. It means accepted
heartbeats stopped advancing under the declared interval, not that a worker
failed. It is not a progress stream and is unknown when the required initial
heartbeat or identity is unavailable. The normal heuristic authorization and
`unauthorized_evidence` rules still apply.

After a wake pointer or guarded goal activation, call `wake_me_up_status`
exactly once. Independently review the
returned worker candidate and attestation before deciding whether to integrate
anything. A terminal event authorizes the guarded wake, never acceptance.

Cancellation or expiry makes an unbound reservation unusable; a bound
reservation cannot be rebound or reused after monitor cancellation, expiry,
pause failure, activation failure, or daemon restart. Preserve the exact
at-most-once claim/activation behavior and do not retry or automatically
re-arm. Keep publish tokens, raw command arguments, and sensitive status data
redacted; use bounded evidence and fingerprints.

Reserving/binding does not authorize installation, plugin cutover, daemon
restart, live or paid continuation, or material command/worker/model spend.
Those remain explicit operator actions with their existing verification and
restart boundaries.

Before an authorized rollback to an older event epoch, run the still-current
binary's `event-compatibility-check --supported-event-epoch <target>` and stop
if it refuses. A nonterminal event monitor remains a refusal even when its
reservation row is missing or corrupt; completed event history alone does not.
Never assume the older binary can enforce a compatibility guard that did not
yet exist.

## Expiry itself wakes a deferred goal

A deferred monitor that armed successfully carries one guarantee: the goal is
woken at the latest at its expiry, as long as the daemon lives, the target
guard still holds, and the target is observed idle at least once after expiry.
That last condition is what the idle barrier waits for: a thread stuck in a
non-idle state (for example `systemError`) is never woken, so treat a monitor
still `armed` well past its expiry as a stuck target, not a slow one.

Do **not** wrap a condition in `any(condition, time)` as a deadline backstop —
that only spends the single wake earlier and loses the reason. The terminal
receipt records why it woke: `condition`, `expired`, `unauthorized_evidence`,
or `observer_failed`.

## Cover failure signatures in every log watch

A `log_pattern` watch MUST include the failure signatures you would act on,
not only the success line. A success-only watch stays silent through a
crashloop, and silence is indistinguishable from still-running.

```json
{
  "type": "log_pattern",
  "path": "/abs/path/to/train.log",
  "patterns": [
    {"name": "ready", "regex": "Ready in [0-9.]+s"},
    {"name": "crash", "regex": "Traceback|CUDA out of memory|Killed|FAILED"}
  ]
}
```

Only bytes appended after arming are scanned, and a rotated or truncated file
becomes `unknown`, never a completion. Compose with `any(log_pattern,
pid_exit)` so a silent exit or a hang is covered too. Prefer local paths.

## Wait on another local Codex thread

`{"type": "thread_idle", "thread_id": "<child-thread-id>"}` waits for a
locally loaded thread — typically a subagent you just launched — to end its
turn. This replaces long-yield polling entirely. A child that is already idle
at registration is rejected: handle its result in the current turn instead.
Idle means the turn ended, not that the child succeeded; read its actual
output. Compose with `all(...)` to wait for several children.

## A wake is not a success claim

The witness decides, not the fact that you woke. Only a `receipt_success`
witness is authoritative task success; a log line, an idle child, an exited
PID, a freed GPU, and a reached deadline are all heuristic. A wake labelled
`unauthorized_evidence` means the evidence fired without your explicit
authorization — verify before acting on it as completion.

## After a wake: exactly one status call

Call `wake_me_up_status` once with the monitor ID and act on what it returns.
The report carries the wake reason, the satisfying witness or failure detail,
the matched-line `journal_tail`, and the wait statistics. Do not re-read the
log when the journal already answers, and do not re-verify a receipt wake.

For a per-occurrence loop, re-register after handling the event: a fresh
`wait_for_event` (or explicit goal defer when needed) with a new
`idempotency_key` and `rearm_of` set to the
monitor that just fired. Lineage is archival only — every safety check runs
fresh — and it makes the loop's cost auditable. There is no multi-shot
schedule and nothing re-arms automatically.

Fail closed: if capability, target eligibility, watcher readiness, queue
admission, pause delivery, or goal identity is uncertain, do not retry,
compensate, switch delivery kinds, or arm automatically. Never steer an active
turn or target another task.
