# Pi Stateful RPC Thread Pilot

The main Pi pilot path is now a reusable conversation thread rather than a
one-shot process. `.pi-worker/home/pi-worker/pi_worker_thread.py` owns a thin JSONL
protocol over Pi `--mode rpc`, imports no Codex code, and binds one persistent
Pi session to one exact thread specification, runtime specification, sandbox,
and workspace. Restarting the supervisor with the same thread directory
resumes the session; a changed model or runtime requires a new thread.

The real Luna-medium smoke completed three turns against closed Stage 0 Task 4.
Turn one read the task and evidence. Turn two recalled row count `12` and union
total `54` without tool calls. After the supervisor was stopped and restarted,
startup reported the same session identifier, `resumed: true`, and eight prior
messages. Turn three recalled best gain `3` and union-only gain `9` without
tool calls. The final session contained three user messages and only the two
first-turn read calls.

The supported orchestration placement is a logical Pi child below a generic
native Codex foreman. The foreman can be spawned with `fork_turns: none`, keep
the supervisor subprocess standard input open, wait for `agent_settled`, and
return receipts. Pi is not a native Multi-Agent Version 2 node. Codex retains
workspace preparation, independent verification, scientific interpretation,
and final acceptance. No concurrent shared writes are allowed.

All persistent home, authentication, session, thread, and log state stays
under the worktree-local `.pi-worker/` tree. Provider traffic uses the explicit
port-9090 local proxy. The protocol permits `medium`, `high`, and `xhigh` and
rejects `max`.

Authoritative pilot records:

- `research/investigations/pi-lightweight-worker-ablation/experiments/2026-07-23-stateful-rpc-thread-pilot/unit.md`
- `research/investigations/pi-lightweight-worker-ablation/experiments/2026-07-23-stateful-rpc-thread-pilot/results.md`
