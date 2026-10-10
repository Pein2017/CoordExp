# Pi Web independent-session transport

Use this for an explicitly selected pair of ordinary Pi Web sessions. The
[lead-worker skill](../SKILL.md) owns roles and authorization. The helper is
[`pi_worker_turn.py`](../scripts/pi_worker_turn.py); it does not call Pi Web's
`Agent` tools or assign a parent/subagent relation. The caller may use any
harness, including Codex, but the recipient must be a Pi Web session: session
IDs belong to that destination backend. This does not translate Codex thread
IDs, provide Codex-to-Codex routing, or bridge separate Pi Web services or
machines. The helper requires the local Pi Web HTTP service; it is not a
terminal-Pi RPC adapter and does not create a shared-file mailbox.

## Resolve the legacy pair and backend

For legacy pair/create commands, use the lead's actual Pi session ID, not a
Codex thread ID, worker ID or filename guessed from a timestamp. Start the lead
as a normal Pi Web conversation. The worker is another normal session with
independent history and settings. Do not open the same session in a separate
CLI writer while Pi Web owns it.

The user-selected local default is `http://127.0.0.1:12345`; omit `--base-url`
for normal CoordExp use. No port discovery or listener check is needed before
calling the helper. An explicit `--base-url` may override it for a deliberately
selected local backend or testing; never use a public Cloudflare/browser URL.
If the endpoint is unavailable, report the error instead of scanning for another
service. Use the lead's actual session ID from its current caller context.
Do not guess it from session lists.

Prepare the bounded assignment and use the create command below directly when
the user has selected provider/model/effort. **Creation now checks the current
model catalog before any mutation**, then verifies the created session's actual
settings. No separate catalog query, source search, or HTTP assembly is needed.
Do not substitute a default model or silently lower effort.

Only when choosing a model or diagnosing a rejected selection, list choices:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --cwd /absolute/worktree --list-models
```

This returns only provider/model/effort choices and creates no session or model
turn. Internally it uses `GET /api/models?cwd=<URL-encoded physical cwd>`; callers
need not interpret the raw API schema. A catalog error or unavailable selection
stops creation before any POST. Existing-session inspection/send does not need
the catalog. `--cwd` remains the shorthand for one shared directory. For distinct
worktrees use `--lead-cwd` and `--worker-cwd`; each is resolved locally and checked
against that session's own Pi Web metadata. Model lookup and worker creation use
`--worker-cwd`. `--list-models` accepts `--worker-cwd` when you want to select the
worker context explicitly.

Normal use is the documented command, not reading the helper or searching Pi
Web source. Once the command succeeds, return to the assignment and its receipt.
For an unavailable endpoint, unsupported selection or contradictory response,
report the specific failure and whether creation/delivery may have occurred;
do not launch broad source exploration or change services, credentials or routing.
Use targeted source inspection only for an actual transport debugging task.
The helper bypasses ambient HTTP proxies and refuses redirects.

## Route one selected peer

The legacy lead/worker flags below remain available. For an existing Pi Web
peer without lead/worker roles, use one of these mutually exclusive forms. Both
selected sessions must be distinct ordinary root sessions visible through the
same declared endpoint. Explicit IDs are Pi Web IDs, and each explicit cwd must
be an absolute physical directory that matches that session. Without `--send`,
the helper only inspects; it does not start a turn. These routes address one
recipient only.

**Explicit Pi Web IDs and directories:**

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --sender "$SENDER_PI_SESSION_ID" --recipient "$RECIPIENT_PI_SESSION_ID" \
  --sender-cwd /absolute/sender-worktree \
  --recipient-cwd /absolute/recipient-worktree \
  --send --text 'Schema field is ready; see the task report.' \
  --receipt /absolute/task-owned/dispatch.json
```

**Aliases from a caller-selected task state:**

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --participants-state /absolute/task/state.json \
  --sender gradient-worker --recipient runtime-worker \
  --send --text 'Schema field is ready; see the task report.' \
  --receipt /absolute/task-owned/dispatch.json
```

The state contains a transport-owned version-1 section; other task/scientific
fields are ignored and the file is read-only to the helper:

```json
{
  "pi_peer_transport": {
    "version": 1,
    "participants": {
      "gradient-worker": {
        "session_id": "<canonical Pi Web session UUID>",
        "cwd": "/absolute/physical/gradient-worktree"
      },
      "runtime-worker": {
        "session_id": "<canonical Pi Web session UUID>",
        "cwd": "/absolute/physical/runtime-worktree"
      }
    }
  }
}
```

Aliases select exact Pi Web session IDs and cwd bindings; they are not
identities, permissions or inferred role names. Do not combine alias and
explicit-ID routing, override alias cwd values, mix new peer flags with legacy
pair/create/model-list modes, or request multiple recipients. Invalid bindings
fail before API mutation. A receipt records the resolved IDs and directories
alongside any alias labels.

## Create and start an independent worker

Use a UTF-8 assignment and a fresh receipt path owned by this task. The session
IDs below must be the actual pair, not example IDs:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --lead-session "$LEAD_SESSION_ID" \
  --create --name "$WORKER_NAME" --provider "$WORKER_PROVIDER" \
  --model "$WORKER_MODEL" --effort "$WORKER_EFFORT" --cwd /absolute/worktree \
  --send --message /absolute/assignment.txt --receipt /absolute/create.json
```

The helper validates the selected model/effort, creates a normal session in
`--worker-cwd`, verifies its identity/settings, names it, and submits the first
task. Its assignment includes the pair's IDs, both expected working directories,
and a matching return command tagged with the assignment's dispatch UUID. When
both sessions use exactly the same working directory, `--cwd` remains a concise
shorthand. Keep the returned worker ID at the existing task/research owner.
The receipt acknowledges submission, not successful execution or acceptance.

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
  --lead-session "$LEAD_SESSION_ID" \
  --worker-session "$WORKER_SESSION_ID" --cwd /absolute/worktree
```

Add `--send --message /absolute/assignment.txt --receipt /absolute/dispatch.json`
to send to the worker. For a direct report back to the lead, use the same pair:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --lead-session "$LEAD_SESSION_ID" \
  --worker-session "$WORKER_SESSION_ID" --cwd /absolute/worktree \
  --to lead --send --message /absolute/report.txt --receipt /absolute/return.json
```

For distinct worktrees, pass both expectations; short messages may use `--text`
instead of a file. `--reply-to` optionally tags a message with a previous dispatch
UUID for conversational correlation (it does not change routing or delivery):

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --lead-session "$LEAD_SESSION_ID" --worker-session "$WORKER_SESSION_ID" \
  --lead-cwd /absolute/lead-worktree --worker-cwd /absolute/worker-worktree \
  --to worker --send --text "Please confirm you received the update." \
  --reply-to "$PREVIOUS_DISPATCH_ID" --receipt /absolute/reply.json
```

`--message` and `--text` are mutually exclusive. `--message FILE` reads and
sends the entire UTF-8 file body; it is not a report attachment or pointer. Use
`--text` for a short brief and optional `--report-path` for a literal absolute
report reference. `--report-section`, `--revision` and `--supersedes` add
reference metadata (`--report-section` requires `--report-path`). The helper
does not open, stat, hash, canonicalize or inject the referenced report, so a
syntactically valid but unavailable path is not an existence or verification
claim. These fields do not impose a word limit, change authority, or select a
delivery mode. The receipt retains routing and reference metadata; the body and
full envelope have separate hashes.

For example, a direct peer brief may point at one affected report section:

```bash
python /data/CoordExp/.codex/skills/lead-worker/scripts/pi_worker_turn.py \
  --sender "$SENDER_PI_SESSION_ID" --recipient "$RECIPIENT_PI_SESSION_ID" \
  --sender-cwd /absolute/sender-worktree \
  --recipient-cwd /absolute/recipient-worktree \
  --send --text 'Use the corrected presentation table; keyed data are unchanged.' \
  --report-path /absolute/task/reports/ce-presentation.md \
  --report-section summary-table --revision r2 \
  --supersedes 'r1 presentation table only' \
  --receipt /absolute/task-owned/correction.json
```

The helper retains explicit session-ID addressing, a fresh dispatch UUID, an
authority-neutral envelope, a content-hash receipt and the existing
queue/steering modes. Generated return commands tag their reply with the
original dispatch UUID; this is conversation correlation, not server-side
exactly-once delivery. The helper does not resolve recipients by display name or
broadcast to every session.

Choose delivery explicitly when needed, in either direction:

- `--delivery queue` (default): starts a turn in an idle recipient with the
  queued message, or leaves a follow-up for after its current work when busy.
  Use for results, candidates and routine questions.
- `--delivery steering`: starts a turn in an idle recipient, or injects a
  steering message at a subsequent agent processing boundary while it is busy.
  Use for an urgent correction to ongoing work, naming the affected action and
  new evidence. It does not cancel an external job or guarantee immediate or
  exact-turn interruption.

An accepted/queued receipt means the server accepted the prompt submission; it
does not mean the recipient read, consumed or accepted the message, completed
the requested work, or accepted a result. Queue is not a passive notification,
and steering is not cancellation.

For example, append `--delivery steering` to the return command above to steer
the lead; omit `--to lead` to steer the worker. These are cross-session peer
messages, not `steer_subagent` calls. The requested mode is saved in the receipt
and command result; acknowledgment still means accepted/queued, not consumed.

An unloaded persisted recipient may be
reopened for sending without changing model/effort. Read-only inspection does
not reopen it. Both sessions must be ordinary independent root sessions on the
same declared Pi Web endpoint. Their expected physical directories may differ,
but each must resolve locally and match its own session metadata. The helper
refuses child/fork relations and identical IDs.

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
