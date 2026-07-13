---
name: baidu-netdisk-transfer
description: Use when Baidu Netdisk work needs BaiduPCS-Go one-off upload/download, browser-cookie login, tmux survival, bypy failure recovery, or append-only union sync with conflict detection.
---

# Baidu Netdisk Transfer

Use one skill for Baidu Netdisk operations. Pick the mode before launching
commands because the overwrite policy is different.

## Mode

- `mode=one-off`: use BaiduPCS-Go for uploads/downloads, cookie login, tmux,
  bypy failure recovery, or manual replacement.
- `mode=union-sync`: use append-only sync for large artifact trees shared across
  machines. Add missing files, detect conflicts, never delete, never overwrite.

Require fresh explicit user approval before any remote or local overwrite,
replacement, or deletion. Mode selection does not carry that approval forward.

## One-Off Transfer

Use for Ubuntu-based BaiduPCS-Go transfers, especially when `bypy` fails with
`Slice MD5 mismatch`, `31064 file is not authorized`, app-root confusion, or
large directory transfers.

Non-obvious facts:

- `BaiduPCS-Go` sees Netdisk root `/`; `bypy` uses an app sandbox.
- Create remote directories under the real root before upload; ignore `31061`.
- Preserve the intended repo-relative remote layout unless the user gives a
  different root.
- Upload safest default: `--norapid -p 1 -l 1 --retry 8`.
- The helpers default to non-overwrite behavior. Use
  `BAIDUPCS_UPLOAD_POLICY=overwrite` or `BAIDUPCS_DOWNLOAD_OVERWRITE=1` only
  after fresh explicit approval for the named paths.
- Large transfers should run in `tmux`.

Scripts:

```bash
bash .codex/skills/baidu-netdisk-transfer/scripts/install_baidupcsgo.sh
bash .codex/skills/baidu-netdisk-transfer/scripts/upload_dir.sh <local_dir> <remote_dir> <BaiduPCS-Go>
bash .codex/skills/baidu-netdisk-transfer/scripts/download_dir.sh <remote_dir> <local_parent> <BaiduPCS-Go>
```

Login with browser cookies when available:

```bash
COOKIE=$(tr -d '\n' < baidu_net_cookie.txt)
/abs/path/to/BaiduPCS-Go login --cookies="$COOKIE"
/abs/path/to/BaiduPCS-Go quota
/abs/path/to/BaiduPCS-Go ls /
```

Before large uploads, scan for Baidu/Windows-hostile names such as `:`, `>`,
control characters, or visual arrows embedded in generated figure names. If a
small set of files blocks transfer, rename them and write a mapping manifest
near the artifact, for example `outputs/_baidu_filename_mapping/`.

## Union Sync

For append-only multi-node synchronization, read
[semantics.md](references/semantics.md) before running `status`, `push`, `pull`,
`sync`, conflict recovery, or deletion maintenance. It owns the command
semantics, manifest continuity, conflict policy, filename policy, and delete
procedure.

```bash
python .codex/skills/baidu-netdisk-transfer/scripts/baidu_union_sync.py --help
cp .codex/skills/baidu-netdisk-transfer/references/config-template.json temp/baidu-netdisk-transfer/config.json
python .codex/skills/baidu-netdisk-transfer/scripts/baidu_union_sync.py --config temp/baidu-netdisk-transfer/config.json doctor
python .codex/skills/baidu-netdisk-transfer/scripts/baidu_union_sync.py --config temp/baidu-netdisk-transfer/config.json status outputs
```

The preflight is complete when `doctor` passes and `status` reports no
unresolved same-path/different-content conflict. Transfers remain dry-run until
an authorized command includes `--apply`.

## Failure Triage

- Upload fails after login: use `--norapid`, keep `-p 1 -l 1`, verify quota and
  real-root path.
- Need speed: raise `BAIDUPCS_UPLOAD_PARALLEL_FILES` before per-file threads.
- Download completes but files are not where expected: inspect
  account-prefixed BaiduPCS-Go staging directories.
- Existing `bypy` files are invisible: recreate the target path under `/` and
  transfer with BaiduPCS-Go.
- ETA questions need live evidence: combine current shard progress, local file
  size, and remote `BaiduPCS-Go ls` visibility rather than pane output alone.

`references/config-template.json` is the portable node/root config.
