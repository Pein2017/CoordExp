---
name: handoff
description: Use when the user asks for a handoff, continuation prompt, compact session summary, another-agent brief, or cross-machine execution note.
---

# Handoff

Write a continuation document containing only the state a fresh agent or another machine needs.

## Output Target

- If the user gives a path, write there.
- For an ephemeral same-session or side-agent transfer, return the handoff
  inline or use a temporary Markdown file from `mktemp -t handoff-XXXXXX.md`.
- For a durable repo-local handoff, use the user-named tracked path or the
  repository's established handoff/memory policy. A temporary file is not
  durable until the user copies it to a persistent destination.
- For a cross-machine handoff without a named destination, use a temporary file
  for manual transfer and state that persistence is still the user's
  responsibility.
- Read the target path before writing if it already exists.

## Fresh-Session Contract

Lead with the state that prevents a fresh agent from restarting a superseded
route:

1. current objective and decision-owning outcome;
2. current decision and the evidence that changed it;
3. superseded or rejected directions, including the condition for reopening;
4. next discriminator or executable action and its stop condition;
5. minimum reading path in authority order.

Also identify volatile facts that the fresh agent must reverify, the first safe
command or action, and its expected success and failure signals. Do not make the
new agent rediscover whether a process, artifact, worktree, or authorization is
still live.

For research handoffs, compare these fields with the owning result, current
decision or compass, and project memory before writing. Resolve contradictions
at their owner rather than explaining around them in the handoff. Treat the
handoff as a continuation router, not a third scientific authority.

## CoordExp Content

Include:

- repo root, branch if relevant, and dirty-file scope;
- exact artifact/config/checkpoint paths that matter;
- commands already run and their outcomes;
- current evidence scope and the strongest claim that is and is not supported;
- unresolved decisions, blockers, and recommended next action;
- which skills or repo docs the next agent should use;
- verification that still needs to run.

Do not duplicate large artifacts, PRDs, plans, metrics, transcripts, or docs.
Link exact paths instead; repo files, docs, and artifacts remain executable
truth. If a large external context packet matters, include one path and content
hash plus its distilled decision impact, not the full text.

Before reporting a durable handoff complete, verify that linked local files
exist, ignored-file behavior matches the intended persistence, and the fresh
agent can identify one current route without reading the prior conversation.

## Common CoordExp Templates

Public data provenance handoff:

- manifest path under `manifests/public_data_provenance/`;
- processed root and whether it exists locally;
- raw dataset prerequisite;
- exact regeneration command from the manifest;
- `python -m pytest tests/test_public_data_provenance_manifests.py -q` result or pending status;
- reminder that routine recovery is regenerate-from-raw-plus-manifest, not Baidu sync.

Baidu artifact transfer handoff:

- local artifact root and intended remote `/CoordExp/outputs/...` path;
- BaiduPCS-Go binary path and login status;
- tmux session/log path if already running;
- unsafe filename mapping manifest, if created;
- post-transfer checks: shard/index/tokenizer/config files, file counts, sizes, and representative `resolved_config.json`.
