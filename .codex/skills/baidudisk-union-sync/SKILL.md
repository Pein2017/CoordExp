---
name: baidudisk-union-sync
description: Use when syncing large local artifact directories with Baidu Netdisk as a Git-like remote where additions are pushed and pulled automatically, but deletes and overwrites are never automated. Provides a reusable append-only union-sync workflow, BaiduPCS-Go wrapper script, conflict detection, manifests, staging, and handoff guidance for multiple machines or agent environments.
---

# BaiduDisk Union Sync

Use this skill when a user wants Baidu Netdisk to behave like a lightweight
large-asset remote across multiple machines:

- local-only files should be uploaded to the remote
- remote-only files should be downloaded locally
- same-path different-content files should be reported as conflicts
- automatic sync must not delete anything
- automatic sync must not overwrite anything

This is an **append-only union sync**, not a mirror. Treat deletion as a manual
maintenance action.

## Core Model

Think of Baidu Netdisk as a large-asset union remote:

```text
node A local tree + node B local tree + remote tree => union of known files
```

The sync loop is intentionally conservative:

1. `pull`: download remote files into a local staging directory, then merge into
   the target local tree with ignore-existing semantics.
2. `push`: upload the local tree with skip-existing semantics.
3. `manifest`: write append-only per-node manifests so future agents can detect
   conflicts and understand provenance.

Never use mirror/delete behavior in automated jobs.

## Bundled Script

Use the bundled Python script:

```bash
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py --help
```

The script has no third-party Python dependency. It shells out to:

- `BaiduPCS-Go`
- `rsync` for safe local staging merges during pull

For a first run, copy and edit the bundled config template:

```bash
cp .codex/skills/baidudisk-union-sync/references/config-template.json \
  temp/baidudisk-union-sync/config.json
```

Then run:

```bash
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py \
  --config temp/baidudisk-union-sync/config.json doctor
```

```bash
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py \
  --config temp/baidudisk-union-sync/config.json status outputs
```

Use `--apply` for commands that transfer files:

```bash
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py \
  --config temp/baidudisk-union-sync/config.json sync outputs --apply
```

Without `--apply`, transfer commands are dry-runs.

## Recommended Workflow

For a new machine:

1. Ensure the same Git branch has this skill directory.
2. Install or copy `BaiduPCS-Go`.
3. Log in to BaiduPCS-Go with browser cookies.
4. Copy `references/config-template.json` to a local ignored path.
5. Set a stable `node_id` or let the script create one.
6. Run `doctor`.
7. Run `status`.
8. Run `sync ROOT --apply` inside `tmux` for large trees.

Example:

```bash
tmux new -s baidudisk_union_sync_outputs
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py \
  --config temp/baidudisk-union-sync/config.json sync outputs --apply
```

## Safety Rules

- Use `--policy skip` for automated uploads.
- Use no-overwrite downloads into staging.
- Merge staging with `rsync --ignore-existing`.
- Stop on conflicts unless the user explicitly asks for an advanced manual
  recovery.
- Reject symlinks and special files by default.
- Reject unsafe Baidu/Windows-hostile filenames by default.
- Do not add `--delete`, `--ow`, `overwrite`, or mirror semantics to timers.

## Deletes

Deletes are manual by design.

Recommended manual-delete playbook:

1. Stop any sync timer or tmux sync loop on every active node.
2. Delete the target path locally where appropriate.
3. Delete the same path on Baidu Netdisk manually.
4. Add the path to a shared denylist if old nodes might still have a copy.
5. Restart sync only after every node has the updated rule or local deletion.

The denylist prevents re-upload; it does not delete anything.

## When To Read References

- Read `references/semantics.md` when designing policy, explaining behavior, or
  deciding how to handle conflicts and deletes.
- Read `references/config-template.json` when bootstrapping a new repo or node.

## Relationship To Other Baidu Skills

If a repo also has a simpler BaiduPCS-Go upload/download skill, use that skill
for one-off transfer mechanics. Use this skill when the task is cross-machine
sync with Git-like union semantics.
