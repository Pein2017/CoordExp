---
title: Stateful Pi RPC Thread Pilot Results
description: Evidence from a three-turn Pi conversation with a supervisor restart between turns two and three.
type: investigation
role: result
authority: non_normative_research
architecture_promotion_status: pilot_only
unit_id: 2026-07-23-stateful-rpc-thread-pilot
topic: pi-lightweight-worker-ablation
status: complete
evidence_status: verified_bounded
updated: 2026-07-23
---

# Stateful Pi RPC Thread Pilot Results

## Verdict

The reusable-thread mechanism works for the bounded pilot. One Luna-medium Pi
session preserved task results across two ordinary turns and a supervisor
process restart. The second and third turns recovered prior values without
reading files or invoking tools. This establishes functional conversation
continuity for one persistent Pi session; it does not establish reliability at
scale or promote Pi to a default worker route.

## Evidence Scope

- Pi version: `0.81.1`.
- Model: `gpt-5.6-luna` through provider `openai-codex`.
- Reasoning: `medium`.
- Proxy: `http://127.0.0.1:9090`.
- Fixture: closed Stage 0 Task 4 deterministic aggregation workspace.
- Thread identifier: `stateful-task04-smoke`.
- Pi session identifier: `8eb2c317-1070-465b-a9b6-50d76485e9ba`.
- Persistent session file:
  `.pi-worker/sessions/2026-07-23T02-14-34-968Z_8eb2c317-1070-465b-a9b6-50d76485e9ba.jsonl`.
- Supervisor receipt directory:
  `.pi-worker/threads/stateful-task04-smoke/`.

The `.pi-worker` evidence is intentionally ignored worktree-local runtime
state. The tracked result records the decision-changing observations without
promoting credentials or raw session content into Git.

## Observations

Turn one read `task.md` and `evidence/budget4.jsonl` and returned the exact
expected aggregate:

```json
{"row_count":12,"total_C_g_lower":42,"total_C_best_lower":45,"total_C_union_lower":54,"total_best_gain":3,"total_union_only_gain":9,"positive_best_gain_image_ids":["5001","13348","14038"],"positive_union_only_gain_image_ids":["1584","7511","13348","13923","14439","16228"]}
```

Turn two was instructed to use only established conversation state. It made no
tool call and returned:

```json
{"previous_row_count":12,"previous_total_C_union_lower":54}
```

The supervisor was then closed and restarted with the same specifications and
thread directory. Startup returned `resumed: true`, the same session
identifier, and `messageCount: 8`. Turn three was again forbidden from reading
files or calling tools and returned:

```json
{"total_best_gain":3,"total_union_only_gain":9}
```

After turn three, `messageCount` was `10`. Session statistics reported three
user messages, five assistant messages, two tool calls, two tool results,
`8,852` total billed/cache-accounting tokens, and provider-reported total cost
`0.0099246`. The only two tool calls occurred in turn one. The raw event stream
contains three `agent_settled` events and exactly two
`tool_execution_start` events.

The third provider request first timed out while attempting WebSocket transport
and transparently fell back to Server-Sent Events. The answer completed
successfully. This is a transport-resilience observation, not evidence that
the WebSocket path is reliable through the proxy.

## Interpretation

The session file, not the live supervisor process, is the continuity owner.
That permits a native Codex foreman to stop and later resume a Pi worker while
retaining task-local dialogue. Binding the manifest to exact thread and runtime
digests prevents accidental reuse against a different sandbox or model.

The result supports adopting the stateful supervisor as the main Pi pilot
entry point. It does not overturn the Stage 0 boundary: Codex still owns final
verification and scientific judgment, and Pi must use a prepared isolated
workspace rather than the active dirty checkout.

## Verification Commands

```bash
conda run --no-capture-output -n ms pytest -q \
  .pi-worker/home/pi-worker/test_pi_worker_thread.py

conda run --no-capture-output -n ms python -m py_compile \
  .pi-worker/home/pi-worker/pi_worker_thread.py \
  .pi-worker/home/pi-worker/pi_worker_foreman.py

conda run --no-capture-output -n ms python \
  scripts/research/check_research_graph.py
```
