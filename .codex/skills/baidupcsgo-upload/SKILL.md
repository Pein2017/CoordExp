---
name: baidupcsgo-upload
description: Use when Baidu Netdisk transfers need BaiduPCS-Go, browser-cookie login, tmux survival, or bypy large-file failure recovery.
---

# BaiduPCS-Go Transfer

Use for Ubuntu-based Baidu Netdisk uploads/downloads, especially when `bypy` fails with `Slice MD5 mismatch`, `31064 file is not authorized`, app-root confusion, or large directory transfers.

## Non-Obvious Facts

- `BaiduPCS-Go` sees Netdisk root `/`; `bypy` uses an app sandbox.
- Create remote directories under the real root before uploading; ignore `31061 文件已存在`.
- Preserve the intended repo-relative remote layout unless the user gives a different root.
- Upload safest default: `--norapid -p 1 -l 1 --retry 8`.
- With `--norapid`, single-file threading is effectively limited; increase concurrent file count before per-file threads.
- Large transfers should run in `tmux`.
- Downloads may stage under an account-prefixed directory such as `1592545883_Pien1722/...`; treat it as BaiduPCS-Go staging, then verify/merge the intended contents.

## Scripts

```bash
bash .codex/skills/baidupcsgo-upload/scripts/install_baidupcsgo.sh
bash .codex/skills/baidupcsgo-upload/scripts/upload_dir.sh <local_dir> <remote_dir> <BaiduPCS-Go>
bash .codex/skills/baidupcsgo-upload/scripts/download_dir.sh <remote_dir> <local_parent> <BaiduPCS-Go>
```

Install default: `./baidupcsgo`, release `v4.0.1`.

## Login

Prefer browser cookies:

```bash
COOKIE=$(tr -d '\n' < baidu_net_cookie.txt)
/abs/path/to/BaiduPCS-Go login --cookies="$COOKIE"
/abs/path/to/BaiduPCS-Go quota
/abs/path/to/BaiduPCS-Go ls /
```

Do not ask the user to type account credentials in the terminal if cookies are available.

## Upload

```bash
BAIDUPCS_UPLOAD_PARALLEL_FILES=4 \
BAIDUPCS_UPLOAD_FILE_THREADS=1 \
bash .codex/skills/baidupcsgo-upload/scripts/upload_dir.sh \
  /abs/repo/some/subtree/run-a /some/subtree/run-a /abs/path/to/BaiduPCS-Go
```

Use aggressive per-file mode only when throughput matters more than the safer `--norapid` path:

```bash
BAIDUPCS_UPLOAD_NO_RAPID=0 BAIDUPCS_UPLOAD_FILE_THREADS=4 BAIDUPCS_UPLOAD_PARALLEL_FILES=2 \
bash .codex/skills/baidupcsgo-upload/scripts/upload_dir.sh <local_dir> <remote_dir> <BaiduPCS-Go>
```

Verify shard files, tokenizer/config files, and index files first.

## Download

```bash
BAIDUPCS_DOWNLOAD_THREADS=8 BAIDUPCS_DOWNLOAD_PARALLEL_FILES=4 \
bash .codex/skills/baidupcsgo-upload/scripts/download_dir.sh \
  /remote/run-a /abs/local/parent /abs/path/to/BaiduPCS-Go
```

If `locate` mode has authorization issues, try `BAIDUPCS_DOWNLOAD_MODE=pcs`. Reduce `BAIDUPCS_DOWNLOAD_THREADS` before reducing parallel files when unstable.

## Tmux Pattern

```bash
tmux new -s baidupcs_upload
bash .codex/skills/baidupcsgo-upload/scripts/upload_dir.sh <local_dir> <remote_dir> <BaiduPCS-Go>
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t baidupcs_upload`.

## Failure Triage

- Upload fails after login: use `--norapid`, keep `-p 1 -l 1`, verify quota and real-root path.
- Need speed: raise `BAIDUPCS_UPLOAD_PARALLEL_FILES` first.
- Download completes but files are not where expected: inspect the account-prefixed staging tree.
- Existing `bypy` files are invisible: recreate the target path under `/` and transfer with BaiduPCS-Go.
