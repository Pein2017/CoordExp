---
title: Stateful Pi RPC Thread Pilot
description: Test whether a Codex-independent supervisor can preserve one Pi conversation across turns and process restarts.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: pilot_only
unit_id: 2026-07-23-stateful-rpc-thread-pilot
topic: pi-lightweight-worker-ablation
status: complete
evidence_status: verified_bounded
updated: 2026-07-23
---

# Stateful Pi RPC Thread Pilot

## Question

Can Pi act as a reusable logical worker thread, rather than a one-shot worker,
without coupling the implementation to Codex Command Line Interface internals?

## Contract

The supported pilot entry point is
`.pi-worker/home/pi-worker/pi_worker_thread.py`. It starts one long-lived Pi
`--mode rpc` process, binds one persistent Pi session to one prepared sandbox,
and exchanges line-feed-delimited JavaScript Object Notation (`JSONL`) with its
caller. The supervisor imports no Codex package and does not assume a native
Multi-Agent Version 2 transport.

One thread directory owns one immutable tuple:

- thread specification digest;
- runtime specification digest;
- sandbox root and workspace;
- Pi session identifier.

Restarting the supervisor with the same specifications and thread directory
reopens that session. Changing the model, reasoning, sandbox, workspace, proxy,
or state paths requires a new thread directory. Concurrent supervisors for the
same thread are rejected with a filesystem lock.

The caller may send `prompt`, `steer`, `follow_up`, `abort`, `compact`,
`get_state`, `get_messages`, `get_stats`, `get_last`, `get_entries`, `status`,
or `close`. `request_id` is the canonical correlation field; `id` remains a
short compatibility alias. The supervisor emits normalized `thread_ready`,
`thread_response`, `thread_event`, `thread_status`, `thread_protocol_error`,
and `thread_exited` records. Complete raw Pi events are retained even when the
caller requests only lifecycle events.

`medium`, `high`, and `xhigh` reasoning are allowed. `max` is rejected. Project
extensions, skills, prompt templates, and context files are disabled so that
conversation continuity comes from the named Pi session rather than ambient
project discovery. Home, authentication, agent, session, manifest, and log
state remain worktree-local. Provider traffic uses the explicit local proxy at
`127.0.0.1:9090`.

## Multi-Agent Version 2 Placement

Pi remains a logical child below a native Codex owner; it is not represented as
a native Multi-Agent Version 2 node. A generic Codex subagent can be assigned a
self-contained foreman brief with `fork_turns: none`. That foreman starts this
supervisor as a subprocess, sends commands, waits for `agent_settled`, and
returns compact receipts to the parent. The subprocess protocol therefore
survives Codex Command Line Interface upgrades unless the generic process or
subagent surfaces themselves change.

The native parent continues to own task selection, workspace preparation,
scientific interpretation, independent verification, and final acceptance.
Pi may retain task-local dialogue across delegated turns, but it must not share
concurrent writes with another worker.

## Example Lifecycle

Start or resume a thread:

```bash
conda run --no-capture-output -n ms python \
  .pi-worker/home/pi-worker/pi_worker_thread.py \
  --thread-spec .pi-worker/specs/my-thread.json \
  --runtime-spec .pi-worker/specs/my-runtime.json \
  --thread-dir .pi-worker/threads/my-thread
```

Keep the subprocess standard input open. After `thread_ready`, send one JSON
object per line:

```json
{"type":"prompt","request_id":"turn-1","message":"Inspect the bounded task and report the result."}
{"type":"get_stats","request_id":"stats-1"}
{"type":"close","request_id":"close-1"}
```

Wait for `agent_settled` before treating a prompt as complete. Await requested
responses before `close`; `close` intentionally refuses an active generation
unless `force` is true. A later start with the same three arguments emits
`thread_ready` with the same `session_id` and `resumed: true` when prior
messages exist.

## Non-goals

- no Codex core, plugin, or Command Line Interface patch;
- no claim that Pi is a native Multi-Agent Version 2 child;
- no automatic cross-workspace session migration;
- no shared concurrent writer;
- no default Pi route or scientific-decision delegation from this pilot.

## Verification

Focused tests cover reasoning bounds, persistent session command construction,
stable command translation, request identifier compatibility, strict
line-feed framing, and local proxy validation. The real smoke uses the closed
Stage 0 deterministic aggregation fixture and tests three dialogue turns,
including one supervisor process restart. See `results.md`.
