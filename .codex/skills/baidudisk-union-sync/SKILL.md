---
name: baidudisk-union-sync
description: Use when syncing large artifact trees across machines through Baidu Netdisk with append-only union semantics and conflict detection.
---

# BaiduDisk Union Sync

Use this when Baidu Netdisk should act like a conservative large-asset remote across machines. The invariant is append-only union sync: add missing files, detect conflicts, never delete, never overwrite.

## Core Model

```text
node A tree + node B tree + remote tree => union of known files
```

- `pull`: download remote-only files into staging, then merge locally with ignore-existing behavior.
- `push`: upload local-only files with skip-existing behavior.
- `manifest`: record per-node state so future runs can detect same-path/different-content conflicts.

Automated jobs must not use mirror/delete/overwrite behavior.

## Script

```bash
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py --help
cp .codex/skills/baidudisk-union-sync/references/config-template.json temp/baidudisk-union-sync/config.json
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py --config temp/baidudisk-union-sync/config.json doctor
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py --config temp/baidudisk-union-sync/config.json status outputs
```

Transfers are dry-run unless `--apply` is present:

```bash
tmux new -s baidudisk_union_sync_outputs
python .codex/skills/baidudisk-union-sync/scripts/baidu_union_sync.py \
  --config temp/baidudisk-union-sync/config.json sync outputs --apply
```

## Operator Preflight

- Sync or intentionally snapshot Git state before long `outputs/` transfers, especially from `main`.
- Discover the actual `BaiduPCS-Go` binary and confirm the remote root with `BaiduPCS-Go ls /CoordExp/outputs` before trusting status output.
- Confirm the exact artifact family before transfer: list the remote parent, inspect names such as `gaussian`, `iou`, `ciou`, or `mix0p*`, and check `resolved_config.json` when variants are easy to confuse.
- Treat `remote_manifest_count: 0` in a dry run as "remote state was not proven", not as proof of a conflict-free remote tree.

## Safety Rules

- Use `--policy skip` for automated uploads.
- Pull into staging and merge with `rsync --ignore-existing`.
- Stop on conflicts unless the user explicitly requests manual recovery.
- Reject symlinks, special files, and unsafe Baidu/Windows-hostile names by default.
- Do not add `--delete`, `--ow`, overwrite, or mirror semantics to timers.
- For deletes, stop all sync loops, delete manually on every relevant surface, then add a denylist rule before restarting old nodes.

## References

- `references/semantics.md`: policy, conflict, and delete semantics.
- `references/config-template.json`: portable node/root config.

Use `baidupcsgo-upload` for one-off uploads/downloads. Use this skill for cross-machine union sync.
