---
doc_id: specs.output-sync-public-data-provenance.2026-05-11
layer: superpowers
doc_type: design-spec
status: accepted
domain: artifacts
summary: Lightweight replacement for the reverted all-large-assets Baidu Netdisk sync design.
updated: 2026-05-11
---

# Output Sync And Public Data Provenance Design

## Context

The earlier large-asset sync design treated `public_data/`, `model_cache/`,
and `output/` as equally managed large-file trees. A first dry operational
attempt showed that this was too heavy for the real workflow: scanning and
hashing raw `public_data/` images can dominate the sync before any useful
backup starts.

The corrected model separates assets by regeneration cost:

- `model_cache/` contains pretrained upstream weights and cache files. Each
  machine should download or prepare these locally.
- `public_data/` raw inputs are externally recoverable. Each machine can fetch
  raw data directly.
- processed `public_data/` directories are reproducible outputs of scripts and
  commands. The repository should track how to regenerate them, not mirror
  their file contents through Baidu Netdisk.
- `output/` contains experiment, training, inference, evaluation, and adapter
  artifacts. These are the assets that need Baidu Netdisk backup and cross-node
  synchronization.

## Goals

- Keep `git` as the truth source for code, docs, commands, and provenance.
- Keep Baidu Netdisk focused on `output/` artifacts.
- Preserve repo-relative paths on the remote side.
- Avoid automatic filtering inside `output/`; synced output should err on the
  side of preserving files.
- Make processed data reproducible by recording exact production provenance.
- Keep the replacement thin enough to operate manually through the existing
  BaiduPCS-Go skill while leaving room for a small wrapper later.

## Non-Goals

- Do not sync `model_cache/` through Baidu Netdisk.
- Do not sync raw `public_data/` through Baidu Netdisk.
- Do not sync processed `public_data/` directory contents through Baidu
  Netdisk by default.
- Do not recreate the reverted generic `large_asset_sync` Python framework.
- Do not introduce an automatic rebuild framework for processed data in this
  change.

## Asset Ownership

| Repo path | Ownership | Durable truth | Recovery path |
| --- | --- | --- | --- |
| `model_cache/` | upstream pretrained cache | upstream model source plus local cache policy | download again per machine |
| `public_data/` raw inputs | external datasets | external dataset source plus local notes | download again per machine |
| `public_data/` processed directories | reproducible local products | git-tracked provenance manifest | rerun recorded script/command |
| `output/` | experiment artifacts | local files plus Baidu Netdisk mirror | upload/download via BaiduPCS-Go |

## Public Data Provenance

Processed data directories under `public_data/` should have a small
git-tracked provenance record under:

```text
manifests/public_data_provenance/
```

The provenance tree mirrors the target directory shape. For example:

```text
public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/
manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60_lvis_proxy.json
```

Each record describes how to recreate the processed directory:

- `relative_path`: repo-relative target directory under `public_data/`
- `producer_script`: repo-relative script or module that produced it
- `working_dir`: repo-relative working directory for the command
- `command`: the exact command that produced the directory
- `inputs`: raw data, configs, checkpoints, or manifests consumed
- `key_params`: parameters that materially affect the output
- `code_ref`: optional commit, branch, or note tying the record to code state
- `generated_at_utc`: optional timestamp for the observed generation event
- `notes`: concise operator notes, caveats, or known regeneration constraints

The manifest records are small text artifacts and should be reviewed like code.
They must not become a second copy of the data.

## Output Sync

`output/` is the only default Baidu Netdisk sync surface.

Canonical remote layout:

```text
output/... -> /CoordExp/output/...
```

`output/` sync is intentionally inclusive:

- include training artifacts
- include inference artifacts, especially `output/infer/**`
- include evaluation metrics and JSONL artifacts
- include adapter files and adapter metadata
- include empty files and interrupted files if they exist on disk
- do not apply automatic ignore rules by default

This reflects the operational rule that missing a useful artifact is worse than
backing up a messy one. Directory hygiene remains a human responsibility.

## Duplicate Policy

The long-term standard is no silent overwrite.

Default behavior for a future wrapper:

- if the remote target path does not exist, upload normally
- if the remote target path already exists, fail before upload
- allow overwrite only with an explicit migration or repair flag such as
  `--allow-overwrite`

The first migration from older backup state may use explicit overwrite, but the
steady-state rule is conflict-first.

## Operating Model

Daily usage should be simple:

1. Code and provenance move through `git`.
2. Raw datasets and pretrained caches are prepared independently on each node.
3. Processed `public_data/` directories are regenerated from their provenance
   records when needed.
4. `output/` is backed up to `/CoordExp/output/` through BaiduPCS-Go.
5. Long-running transfers run in `tmux` and use the existing proxy if the node
   is offline except through `127.0.0.1:9090`.

The existing `.codex/skills/baidupcsgo-upload` workflow remains the canonical
manual transfer procedure until a smaller `output/`-only wrapper is added.

## Verification

A useful verification loop is lightweight:

- `git status --short` confirms provenance and docs are tracked.
- `rg "large_asset_sync|large_assets"` confirms the reverted generic sync
  framework is not the active path.
- `BaiduPCS-Go ls /CoordExp/output` checks remote reachability.
- targeted remote directory listing checks the specific `output/` run being
  uploaded or restored.

Do not use full-tree hashing of raw `public_data/` as part of routine sync
verification.
