# Same-host worker transport

Read this only when the App tool is absent or lacks the required capability.
The [lead-worker skill](../SKILL.md) owns authorization, roles and acceptance.

### Create and start a requested worker

Resolve the user-selected model and effort against the live model catalog; do
not substitute defaults. Pass the ready assignment as a UTF-8 file:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py \
  --lead-thread LEAD_UUID --create --name WORKER_NAME \
  --model MODEL_ID --effort EFFORT --cwd /absolute/worktree \
  --send --message /absolute/assignment.txt --receipt /absolute/create.json
```

Creation and first input use **one connection**: `thread/start`, verify the
returned settings, `thread/name/set`, then `turn/start`. A create-only empty
task may have no saved rollout and cannot be resumed after disconnect. This
helper therefore requires an initial message; it never changes the lead's model
or silently reconfigures an existing worker. It appends the pair's task IDs and
a direct-return command to the assignment, so the new worker can report back
without reconstructing its identity. Record the returned UUID in the
existing assignment/state owner. A receipt's `submitted` status means the first
turn was accepted, not that the worker finished or research was accepted.

After submission, use one bounded App snapshot or the worker's direct entry
report to verify actual progress and visibility. Immediate `thread/read` can
race initial rollout persistence; an empty-rollout read error does not undo a
successful `turn/start` and never authorizes another creation or send.

### Inspect or message an existing pair

Sending already inspects the pair. Use standalone inspection for reconciliation
or when no message is ready, without starting anything:

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
receipt and target task before any retry. The same applies to a creation timeout:
the task may exist even if its UUID was not returned. The creation receipt keeps
any known UUID and the last phase; a naming or first-send failure is not license
to create a replacement automatically. Do not run `codex exec resume` beside an
App-owned active worker. If this local socket or its protocol is unavailable,
return the ready message and the exact transport gap; do not invent a background
runner or silently substitute a watcher.
