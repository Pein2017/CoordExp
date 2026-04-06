---
name: baidupcsgo-upload
description: Install and use qjfoidnh/BaiduPCS-Go on Ubuntu to upload files or whole directories to Baidu Netdisk, especially when bypy fails on large files, slice uploads, or app-root path semantics. Use when Codex needs a reproducible CLI workflow for Baidu Netdisk login via browser cookies, tmux-safe long-running uploads, remote directory creation, or verification of uploaded artifacts.
---

# BaiduPCS-Go Upload

Use this skill for Ubuntu-based Baidu Netdisk uploads when reliability matters more than speed.

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
- forces conservative settings with `--norapid -p 1 -l 1 --retry 8`

## 5. Run inside tmux for large uploads

For multi-GB checkpoints, prefer:

```bash
tmux new -s baidupcs_upload
bash scripts/upload_dir.sh /abs/local/dir /remote/dir /abs/path/to/BaiduPCS-Go
```

Detach with `Ctrl-b d`.

Reattach with:

```bash
tmux attach -t baidupcs_upload
```

## 6. Verify completion

Check the target directory:

```bash
/absolute/path/to/BaiduPCS-Go ls /output/stage1_2b/my-run
```

For a large upload, validate the highest-risk artifacts first:
- model shard files such as `model-00001-of-00002.safetensors`
- tokenizer/config files needed for loading
- index files such as `model.safetensors.index.json`

## 7. Failure handling

If login succeeds but uploads fail:
- rerun with `--norapid`
- keep `-p 1 -l 1` before trying higher concurrency
- verify the remote path is under the real Netdisk root, not `/apps/bypy`
- verify quota with `quota`

If `bypy` already uploaded files somewhere else:
- remember that `BaiduPCS-Go` will not show files stored only under `bypy`'s app root
- recreate the intended target path under the real root and upload again

## Scripts

- `scripts/install_baidupcsgo.sh`: download and unpack the tested Linux amd64 release
- `scripts/upload_dir.sh`: create the remote directory chain and upload a local directory with stable defaults
