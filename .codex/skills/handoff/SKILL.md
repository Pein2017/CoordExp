---
name: handoff
description: Use when the user asks for a handoff, continuation prompt, compact session summary, another-agent brief, or cross-machine execution note.
---

# Handoff

Write a concise continuation document for a fresh agent or another machine.

## Output Target

- If the user gives a path, write there.
- Otherwise save to a temporary Markdown file from `mktemp -t handoff-XXXXXX.md`.
- Read the target path before writing if it already exists.

## CoordExp Content

Include only portable, actionable state:

- repo root, branch if relevant, and dirty-file scope;
- exact artifact/config/checkpoint paths that matter;
- commands already run and their outcomes;
- unresolved decisions, blockers, and recommended next action;
- which skills or repo docs the next agent should use;
- verification that still needs to run.

Do not duplicate large artifacts, PRDs, plans, metrics, or docs. Link exact paths instead. Keep references short and navigational; repo files, docs, and artifacts remain executable truth.

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
