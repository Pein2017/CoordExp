---
name: baidupcsgo-upload
description: Install and use qjfoidnh/BaiduPCS-Go on Ubuntu to upload files or whole directories to Baidu Netdisk, especially when bypy fails on large files, slice uploads, or app-root path semantics. Use when Codex needs a reproducible CLI workflow for Baidu Netdisk login via browser cookies, tmux-safe long-running uploads, remote directory creation, or verification of uploaded artifacts.
---

# BaiduPCS-Go Upload

Use this skill for Ubuntu-based Baidu Netdisk uploads or downloads when reliability matters more than speed, but expose parallelism when the user wants higher throughput.

Prefer this over `bypy` when:
- `bypy` hits `Slice MD5 mismatch`, `31064 file is not authorized`, or other large-file upload failures
- the task is to upload a whole local directory and keep the original folder structure
- the upload needs to run safely inside `tmux`
- the user can provide browser cookies but does not want to type account credentials into the terminal

## Core facts

- `BaiduPCS-Go` sees the Baidu Netdisk root `/`, not `bypy`'s `/apps/bypy` sandbox.
- A path that exists under `bypy` may still need to be created again under the real root for `BaiduPCS-Go`.
- For large model checkpoints, start with conservative upload settings:
  `--norapid -p 1 -l 1 --retry 8`
- Upload and download both support parallelism. Prefer raising concurrent file count first, then per-file threads.
- For upload, `--norapid` is the safest default, but BaiduPCS-Go notes that disabling rapid upload effectively limits single-file threading to one thread.
- For long uploads, run inside `tmux`.

## 1. Install the binary

Use the bundled installer script:

```bash
bash scripts/install_baidupcsgo.sh
```

The script prints the absolute binary path on success.
Default install directory: `./baidupcsgo`

Default release: `v4.0.1`

## 2. Log in with browser cookies

Prefer cookie login over username/password.

1. Ask the user to export the Cookie header from an already logged-in `pan.baidu.com` browser session.
2. Put the cookie string into a local file such as `baidu_net_cookie.txt`.
3. Run:

```bash
COOKIE=$(tr -d '\n' < baidu_net_cookie.txt)
/absolute/path/to/BaiduPCS-Go login --cookies="$COOKIE"
```

4. Verify:

```bash
/absolute/path/to/BaiduPCS-Go quota
/absolute/path/to/BaiduPCS-Go pwd
/absolute/path/to/BaiduPCS-Go ls /
```

Successful login should show the real Netdisk root and the account quota.

## 3. Create the remote directory tree

Before uploading, explicitly create the remote path if it may not exist:

```bash
/absolute/path/to/BaiduPCS-Go mkdir /output
/absolute/path/to/BaiduPCS-Go mkdir /output/stage1_2b
/absolute/path/to/BaiduPCS-Go mkdir /output/stage1_2b/my-run
```

Ignore `31061 文件已存在`.

## 4. Upload a whole directory

Use the bundled uploader script when the goal is “local directory -> same-named remote directory”.

Example:

```bash
bash scripts/upload_dir.sh \
  /abs/local/output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged \
  /output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged \
  /abs/path/to/BaiduPCS-Go
```

The script:
- ensures the remote parent directories exist
- uploads the directory to `dirname(REMOTE_DIR)` so the original folder name is preserved
- defaults to conservative settings with `--norapid -p 1 -l 1 --retry 8`
- can be tuned with environment variables instead of editing the script

Example with parallel uploads across files while keeping `--norapid`:

```bash
BAIDUPCS_UPLOAD_PARALLEL_FILES=4 \
BAIDUPCS_UPLOAD_FILE_THREADS=1 \
bash scripts/upload_dir.sh /abs/local/dir /output/stage1_2b/my-run /abs/path/to/BaiduPCS-Go
```

Example with more aggressive per-file parallelism:

```bash
BAIDUPCS_UPLOAD_NO_RAPID=0 \
BAIDUPCS_UPLOAD_FILE_THREADS=4 \
BAIDUPCS_UPLOAD_PARALLEL_FILES=2 \
bash scripts/upload_dir.sh /abs/local/dir /output/stage1_2b/my-run /abs/path/to/BaiduPCS-Go
```

Use the aggressive mode only when higher throughput matters more than the safer `--norapid` path.

## 5. Download a whole directory

Use the bundled downloader script when the goal is “remote directory -> local parent directory while preserving the remote folder name”.

Example:

```bash
bash scripts/download_dir.sh \
  /output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged \
  /abs/local/output_cache \
  /abs/path/to/BaiduPCS-Go
```

The script:
- sets `BaiduPCS-Go`'s save directory to the provided local parent directory
- downloads with `--fullpath` so the remote folder structure is preserved
- defaults to parallel download settings `--mode locate -p 4 -l 2 --retry 8 --ow --mtime`

Example with higher download parallelism:

```bash
BAIDUPCS_DOWNLOAD_THREADS=8 \
BAIDUPCS_DOWNLOAD_PARALLEL_FILES=4 \
bash scripts/download_dir.sh /output/stage1_2b/my-run /abs/local/output_cache /abs/path/to/BaiduPCS-Go
```

## 6. Run inside tmux for large transfers

For multi-GB uploads, prefer:

```bash
tmux new -s baidupcs_upload
bash scripts/upload_dir.sh /abs/local/dir /remote/dir /abs/path/to/BaiduPCS-Go
```

For downloads, the same pattern applies:

```bash
tmux new -s baidupcs_download
bash scripts/download_dir.sh /remote/dir /abs/local/parent /abs/path/to/BaiduPCS-Go
```

Detach with `Ctrl-b d`.

Reattach with:

```bash
tmux attach -t baidupcs_upload
```

## 7. Verify completion

Check the target directory:

```bash
/absolute/path/to/BaiduPCS-Go ls /output/stage1_2b/my-run
```

For a large upload, validate the highest-risk artifacts first:
- model shard files such as `model-00001-of-00002.safetensors`
- tokenizer/config files needed for loading
- index files such as `model.safetensors.index.json`

For a large download, validate:
- the expected top-level folder exists under the local parent directory
- shard counts and file sizes match `ls` output from the remote side
- loader-critical files such as tokenizer/config/index files are present

## 8. Failure handling

If login succeeds but uploads fail:
- rerun with `--norapid`
- keep `-p 1 -l 1` before trying higher concurrency
- if you need more throughput, raise `BAIDUPCS_UPLOAD_PARALLEL_FILES` before `BAIDUPCS_UPLOAD_FILE_THREADS`
- verify the remote path is under the real Netdisk root, not `/apps/bypy`
- verify quota with `quota`

If downloads fail:
- try `BAIDUPCS_DOWNLOAD_MODE=pcs` when `locate` hits authorization issues
- reduce `BAIDUPCS_DOWNLOAD_THREADS` and `BAIDUPCS_DOWNLOAD_PARALLEL_FILES` if the transfer becomes unstable
- keep `--mtime` enabled unless you specifically want fresh local timestamps

If `bypy` already uploaded files somewhere else:
- remember that `BaiduPCS-Go` will not show files stored only under `bypy`'s app root
- recreate the intended target path under the real root and upload again

## Scripts

- `scripts/install_baidupcsgo.sh`: download and unpack the tested Linux amd64 release
- `scripts/upload_dir.sh`: create the remote directory chain and upload a local directory with stable defaults plus env-based parallel tuning
- `scripts/download_dir.sh`: download a remote file or directory into a chosen local parent directory with env-based parallel tuning
