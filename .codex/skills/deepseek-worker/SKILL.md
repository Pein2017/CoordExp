---
name: deepseek-worker
description: Create or continue a persistent DeepSeek V4.1-Flash Codex worker through CLI profiles when the user selects DeepSeek for a bounded research or engineering trial. Keep the GPT lead responsible for scope and acceptance.
---

# DeepSeek Worker

Use the existing Codex runtime and shared configuration. The GPT lead stays in
its current task; the DeepSeek worker has its own persistent conversation.
This is the tested alternative to cross-provider native subagents on Codex 0.154.0.

## Select and brief

- Use `/data/CoordExp/.codex` as the existing `CODEX_HOME`; do not create a second
  configuration or copy AGENTS, skills, MCP definitions, or credentials.
- Choose `ds-flash-low`, `ds-flash-high`, or `ds-flash-max`. All use API model
  `deepseek-flash`; Default and Plan effort match the profile suffix. Start a
  simple capability trial at low unless the user or task calls for another tier.
  Do not change effort or provider globally.
- The calling process needs `DEEPSEEK_API_KEY`. Check presence only; never print
  the value or place it in a prompt, command argument, log, or skill file.
- Give the worker the question, exact cwd/artifacts, owned read/write scope,
  evidence to return, and stopping condition. For research, keep the agreed
  contrast and claim boundary. A trial is not permission for costly experiments.
- Ask for a compact candidate result with concrete evidence, uncertainty and
  artifact paths. Keep the lead's independent acceptance responsibility.
- Prefer an existing worker whose context and scope still fit. Before resuming,
  reconcile ownership and confirm it is idle; do not race Desktop input, another
  lead, and CLI resume against the same task. Do not delegate further by default.

## Create through CLI

Use the actual task cwd, not the example merely because it is shown here. Prepare
the bounded brief in `request.txt` using the normal file-editing tool. Store the
run directory and command handle so a long call is joined rather than relaunched.

```bash
worker_profile=ds-flash-low
worker_cwd=/data/CoordExp/.worktrees/research-probes
worker_run=$(mktemp -d /tmp/deepseek-worker.XXXXXX)
# Prepare "$worker_run/request.txt" before the following call.
codex exec --profile "$worker_profile" -C "$worker_cwd" --json \
  -o "$worker_run/reply.txt" - < "$worker_run/request.txt" \
  > "$worker_run/events.jsonl" 2> "$worker_run/stderr.log"
```

Do not add `--ephemeral`: the resulting session must survive for follow-up work.
Preserve the process exit status. Extract the exact session ID from the event,
not from a title, guessed UUID, or `--last`:

```bash
jq -er 'select(.type == "thread.started") | .thread_id' "$worker_run/events.jsonl"
```

Return a small receipt: thread ID, profile, cwd, run path, exit status, and result
status. Keep that receipt in the lead's existing task record when continuity is
needed; shell variables do not survive every tool invocation.

## Send a follow-up to the same worker

Resolve `worker_thread` and `worker_profile` from that receipt. Use a fresh run
directory for each turn so previous evidence is retained; prepare its request
file before executing. Pass the profile on every resume to load the provider
definition and model catalog again.

```bash
codex --profile "$worker_profile" exec resume "$worker_thread" --json \
  -o "$worker_run/reply.txt" - < "$worker_run/request.txt" \
  > "$worker_run/events.jsonl" 2> "$worker_run/stderr.log"
```

For an existing task, keep its recorded cwd unless the assignment explicitly
changes it. Use one live invocation per worker and wait on its command handle.
If context or ownership no longer fits, start a new worker with an explicit brief.

## Accept the turn, then assess the work

Check the command exit status, `turn.completed` versus failure/error events, and
the final response. An ID, `thread.started`, an idle process, or a zero exit alone
does not establish success. Inspect error items: Codex can encode startup feature
warnings as error items even on an otherwise completed turn.

When route or effort is in doubt, inspect persisted thread metadata or the App
Server resume response: expected provider `deepseek`, model `deepseek-flash`, and
the selected effort. Do not use the model's self-description as route evidence.
On auth/config/transport failure, report the exact sanitized error and stop;
do not silently substitute GPT, loop retries, or change shared configuration.

The lead checks decision-bearing files, commands or calculations before accepting
the worker's result. Distinguish observation from inference. For early research
trials, note task type, tier, useful output and correction burden in the existing
research record; do not infer broad model quality from a greeting or one success.

## Desktop and App Server are optional

CLI sessions have source `exec` and can be absent from the Desktop sidebar. A
successful CLI worker does not require a sidebar entry. If the user requests
Desktop visibility or App Server communication, read the existing
[DeepSeek handoff](/data/CoordExp/.codex/model-catalogs/README-deepseek.md).
It documents the supported history-preserving fork, provider registration,
startup catalog requirement and minimal profile-expansion launcher. Never rewrite
session source fields directly or treat `agents.send_message` as a CLI transport.

Shared AGENTS, skills, permissions and MCP configuration are inherited. Low has
real shell and Serena MCP acceptance evidence. This does not prove every tool,
Desktop-only dynamic capability, memory injection, or high/max model quality.
Native provider web search is disabled in these profiles. The Astra-derived
model instruction template is a snapshot, with a DeepSeek identity adjustment.
