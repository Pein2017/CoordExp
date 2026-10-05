# Pi Web independent-session transport

Use this for a user-requested pair of ordinary Pi Web sessions. The
[lead-worker skill](../SKILL.md) owns roles and authorization. The helper is
[`pi_worker_turn.py`](../scripts/pi_worker_turn.py); it does not call Pi Web's
`Agent` tools or assign a parent/subagent relation. It requires the local Pi Web
HTTP service; it is not a terminal-Pi RPC adapter.

## Resolve the pair and backend

Use the lead's actual Pi session ID, not a Codex thread ID, worker ID or filename
guessed from a timestamp. Start the lead as a normal Pi Web conversation. The
worker is another normal session with independent history and settings. Do not
open the same session in a separate CLI writer while Pi Web owns it.

Resolve the current loopback service URL from its runtime owner/listener. Pass
it explicitly with `--base-url`; do not reuse a historical port or a public
Cloudflare/browser URL. This helper bypasses ambient HTTP proxies and refuses
redirects. If the backend needs authentication or is unavailable, report that
gap rather than changing routing, credentials or services.

Choose worker provider, model and effort with the user, verify them against the
current model catalog, and give the worker an already prepared bounded task.
Do not substitute a cheaper/default model for a selected persistent worker.

## Create and start an independent worker

Use a UTF-8 assignment and a fresh receipt path owned by this task. The URL and
IDs below are variables to resolve, not defaults:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --base-url "$PI_WEB_URL" --lead-session "$LEAD_SESSION_ID" \
  --create --name "$WORKER_NAME" --provider "$WORKER_PROVIDER" \
  --model "$WORKER_MODEL" --effort "$WORKER_EFFORT" --cwd /absolute/worktree \
  --send --message /absolute/assignment.txt --receipt /absolute/create.json
```

The helper creates a normal session, verifies its identity/settings, names it,
and submits the first task. Its assignment includes the pair's IDs and a return
command. Keep the returned worker ID at the existing task/research owner. The
receipt acknowledges submission, not successful execution or acceptance.

Creation is several HTTP operations, not a transaction. If naming or first input
fails after creation, keep the known worker ID and reconcile it; do not create
another worker. A create timeout can leave a session whose ID was not returned.
Inspect the receipt and the session list before deciding recovery. The helper
never automatically retries or treats `accepted:false` as proof that no session
was created.

## Inspect, dispatch and return

Inspect the pair without starting a model turn:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --base-url "$PI_WEB_URL" --lead-session "$LEAD_SESSION_ID" \
  --worker-session "$WORKER_SESSION_ID" --cwd /absolute/worktree
```

Add `--send --message /absolute/assignment.txt --receipt /absolute/dispatch.json`
to send to the worker. For a direct report back to the lead, use the same pair:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --base-url "$PI_WEB_URL" --lead-session "$LEAD_SESSION_ID" \
  --worker-session "$WORKER_SESSION_ID" --cwd /absolute/worktree \
  --to lead --send --message /absolute/report.txt --receipt /absolute/return.json
```

Default delivery starts an idle recipient or queues a follow-up for a busy one;
it does not interrupt the current task. An unloaded persisted recipient may be
reopened for sending without changing model/effort. Read-only inspection does
not reopen it. Both sessions must be ordinary sessions in the declared physical
checkout; the helper refuses child/fork relations and identical IDs.

Every message is marked as a peer dispatch/report, with sender, recipient and
dispatch identity. This is transport metadata, not new user authorization. The
worker explicitly returns its report through this command; ordinary sessions do
not get the subagent runtime's automatic completion notification. A crashed
worker cannot promise a return. Check its state and terminal evidence when a
report is missing, rather than treating silence as completion.

## Delivery and recovery limits

Each send uses one fresh receipt and makes no automatic retry. A timeout after
submission is **unknown delivery**. Reconcile the target and receipt before any
new send; never start a second writer with `pi --session` as a recovery shortcut.
Pi Web's API has no caller idempotency key or expected-turn compare-and-swap, so
this client is not an exactly-once queue or an atomic lock against the UI. Avoid
concurrent manual configuration changes or competing senders for the same task.
Do not describe a queued follow-up as an exact-turn steer or completed work.

Keep inspection compact. Read the worker's final report and decision-bearing
evidence, not repeated full transcripts. This transport does not promise
automatic job recovery after a server restart; session persistence, live jobs
and completion reports are separate facts to reconcile.
