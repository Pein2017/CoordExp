# Direct lead/worker reporting — user-authorized transport update

The user explicitly requested direct session-to-session communication and removal
of the wake-me-up then read-chat loop. This supersedes only the terminal-log-marker
return instructions in unit.md and lead-ruling-01.md. Scientific scope, Lane B
HOLD, Lane A qualification gate, ownership and budgets are unchanged.

Read the updated `/data/CoordExp/.codex/skills/lead-worker/SKILL.md`. Send reports
directly to lead `01a0c1f3-dbef-7b63-b2da-8dc7072cea8d` from worker
`01a0c726-ad7c-7cc0-89b7-d76ac6fcf027` using the supported task-message tool or:

```sh
python /data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py \
  --lead-thread 01a0c1f3-dbef-7b63-b2da-8dc7072cea8d \
  --worker-thread 01a0c726-ad7c-7cc0-89b7-d76ac6fcf027 \
  --cwd /data/CoordExp/.worktrees/research-probes --to lead --send \
  --message /absolute/UTF8/report.txt --receipt /absolute/fresh-return-receipt.json
```

The helper automatically steers an active recipient or starts a turn for an idle
recipient. Keep stable role identities; do not swap lead/worker IDs. Never retry
unknown delivery automatically. It does not change either model or effort.

Send one short acknowledgement now through that direct route: identify yourself
as the worker, confirm receipt of this communication update, give current
qualification status and any immediate blocker. This ACK checks transport only;
do not replay model work or wait for qualification completion to acknowledge.
Preserve all running jobs and continue the already assigned work.

Future checkpoint/candidate messages must include status, the result or exact
question, key checks, costs, job state and the stable artifact path/hash. Send
the report itself, not merely a wake pointer or a request to read your chat.
The lead will inspect the named artifacts independently. Stop at the existing
qualification gate; this update is not broad-training acceptance.

Do not append another wake marker or arm a worker-report watcher. The lead will
cancel the existing monitor after the direct-return instruction is acknowledged.
Save concise reports within the assigned output root as `.txt`/JSON evidence or
in the owned research notes; do not put code or Markdown in outputs.
