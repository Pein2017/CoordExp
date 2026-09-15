# Wake registration diagnosis v1

Date: 2026-09-14 UTC

Scope: read-only diagnosis; no wake was armed, no runtime/database/config was modified, and no producer was touched.

## Decision

The failed wake was not an unavailable-plugin or malformed-condition failure. The installed `codex-wake-me-up@coordexp-local` received the call, passed daemon readiness, and then exhausted its fixed 10-second app-server RPC budget while resolving the delivery target with `thread/read(includeTurns=false)` for root thread `01a06f6d-2336-7670-a929-8c4d56ed54ba`. Registration stopped before condition preparation and before creation of a monitor row.

The established cause boundary is a thread-specific app-server read failure, not a general Wake Me Up failure. This task also has a projection cursor that is incompatible with its current physical rollout segment under the locally available thread-store semantics. That is a concrete anomaly, but its causality for `thread/read(includeTurns=false)` is unproven: the available read path can obtain no-history thread metadata and a preview directly from the rollout rather than consulting projected turns. Remaining explanations include current-rollout summary parsing, live-thread contention, projection handling in the deployed build, or another app-server reader component.

## Original failed call

Canonical evidence: `/data/CoordExp/.codex/sessions/2026/09/08/rollout-2026-09-08T16-19-27-01a06f6d-2336-7670-a929-8c4d56ed54ba_01a081d1-3818-7472-8cc9-b8774a831903.jsonl`, physical line 21387, ordinal 33161, timestamp `2026-09-13T07:20:23.950Z`, turn `01a09959-20da-7ca0-a40b-32a7a7093604`.

```json
{
  "condition": {
    "type": "all",
    "children": [
      {"type": "native_worker_terminal", "task_name": "/root/astra_low_eval_materialization_finish"},
      {"type": "native_worker_terminal", "task_name": "/root/replay_throughput"}
    ]
  },
  "expires_in_seconds": 14400,
  "idempotency_key": "owner-successor-scale-20260913-supply-throughput-v1"
}
```

Result: `status=failed`, duration `10.227683802s`, error `thread delivery registration failed: stage=resolve_delivery_target error=app-server thread/read timed out`. Both task names occur as canonical `SubAgentActivity.agent_path` values in the same raw rollout. The wake ledger contains no row for this idempotency key.

Installed-source boundary:

- `/data/CoordExp/.codex/plugins/cache/coordexp-local/codex-wake-me-up/0.1.0+codex.20260909065612/src/codex_wake_me_up/service.py:462-469` performs `resolve_thread_delivery` before condition preparation or `ledger.create_or_get`.
- `/data/CoordExp/.codex/plugins/cache/coordexp-local/codex-wake-me-up/0.1.0+codex.20260909065612/src/codex_wake_me_up/app_server.py:407-449` resolves root ancestry through `thread/read(includeTurns=false)`.
- `/data/CoordExp/.codex/plugins/cache/coordexp-local/codex-wake-me-up/0.1.0+codex.20260909065612/src/codex_wake_me_up/app_server.py:68-77,174-213` defines the narrow app-server client, fixes the request timeout at 10 seconds, and maps expiry to `app-server <method> timed out`.
- The client connects to `/data/CoordExp/.codex/app-server-control/app-server-control.sock`; the active app-server executable observed during diagnosis was `/data/CoordExp/.codex/packages/standalone/releases/0.154.0-x86_64-unknown-linux-musl/bin/codex`.
- The read-only monitor lookup used `/data/CoordExp/.codex/runtime/codex-wake-me-up/monitors.sqlite3`; task metadata and projection state came from `/data/CoordExp/.codex/state_5.sqlite` and `/data/CoordExp/.codex/thread_history_1.sqlite`.

## Task-specific control evidence

A single paired live read using the installed plugin client reproduced the split:

- this root: `thread/read(includeTurns=false)` timed out in `10.107s`;
- recent control thread `01a099eb-db6b-7812-9006-73a363c9884a`: the same read returned in `0.179s` after `0.110s` connect/initialize.

The control thread had also armed monitor `ad6116ea-7eb1-4119-8eb6-1677efe8cb9e` successfully on 2026-09-13: an `any(log_pattern, all(pid_exit, pid_exit))` registration returned `state=armed` in `0.282928103s` while that thread was active. Thus plugin availability, condition composition, and an active turn are not sufficient explanations.

Current projection evidence establishes a separate thread-history anomaly:

- this root rollout is currently about 510.5 MB and physical ordinal 36171, while `thread_history_projection_state` remains at byte offset `3419709`, ordinal `226`, with only 2 projected turns and 71 items;
- the stored offset is not a JSONL record boundary: it is 2,804,083 bytes inside a 3,504,315-byte `compacted` record whose physical ordinal is 11963;
- this continuation segment begins at ordinal 11775 and its `history_base` points to a different physical segment (`01a07bf3-04ed-7643-9f68-c95bdf71327f`);
- the successful control projection is at exact rollout EOF with 6 turns and 1,111 items.

The cursor cannot legitimately be interpreted as an ancestral-segment byte address under the available local projection implementation. `/data/CoordExp/external/harness/codex/codex-rs/thread-store/src/local/thread_history_materialization.rs:35-68` loads projection state by the requested/current thread ID and passes `next_byte_offset` directly to the supplied current `rollout_path`; `:85-133` seeks that same path at the stored offset. It separately derives the starting ordinal from `SessionMeta.history_base`. `/data/CoordExp/external/harness/codex/codex-rs/thread-store/src/local/thread_history.rs:54-100` confirms the state contains no segment identity, only `{thread_id, next_byte_offset, next_ordinal}`. The installed 0.154.0 binary embeds the same projection SQL and diagnostic strings, but no exact source-commit provenance was available, so deployed byte-for-byte equivalence is not claimed.

This does **not** prove the cursor anomaly caused the wake failure. `/data/CoordExp/external/harness/codex/codex-rs/thread-store/src/local/read_thread.rs:32-75` shows a no-history read can use SQLite metadata and then opportunistically derive a richer summary from the current rollout; `/data/CoordExp/external/harness/codex/codex-rs/rollout/src/list.rs:772-783,1112-1218` implements that bounded head/preview scan. This supports the narrower conclusion: the timeout is in this task's app-server read path, while projection-versus-rollout-summary causality remains unresolved.

## Smallest next check and repair boundary

No action is needed for the current research run: retain the user-directed tmux/log/terminal-receipt handoff and inspect only after the user returns.

If a wake repair is later authorized, first repeat exactly one paired `thread/read(includeTurns=false)` after this root is idle. Persistence would separate the durable thread/history path from live-turn contention. Only then profile the no-history read boundary to distinguish current-rollout summary parsing from projection work; do not assume projection rebuilding alone will repair Wake Me Up. If the projection is repaired independently, scope it to this thread under byte-exact backup and validate physical EOF, raw/projected IDs and counts, and `PRAGMA integrity_check`. Rerun the direct read before attempting a fresh monitor, and do not replay the failed idempotency key. Increasing `expires_in_seconds` cannot affect this pre-admission 10-second RPC, and merely increasing the client timeout would mask rather than localize the read failure.
