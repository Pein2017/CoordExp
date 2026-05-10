---
doc_id: docs.standards.large_asset_sync
layer: docs
doc_type: standard
status: canonical
domain: operations
summary: Stable operator workflow for repo-relative large-asset sync across local nodes and Baidu Netdisk.
tags: [standards, artifacts, baidu, sync]
updated: 2026-05-09
---

# Large Asset Sync

This workflow keeps repo-relative large assets aligned across local nodes and
Baidu Netdisk without treating cloud state as the primary source of truth.

## Truth Source

- `git`-tracked manifests under `manifests/large_assets/`
- `publish --execute` is the only command that advances tracked manifest state

## Canonical Remote Root

- `/CoordExp`

Managed remote identity is always derived from the repo-relative path. Do not
author machine-specific absolute paths into manifests or reports.

## Managed Roots

- `public_data/**`
- `model_cache/**`
- `output/**`

Broad `output/**` coverage is filtered by
`manifests/large_assets/ignore.txt` so transient files do not become desired
state by accident.

## Preflight

- Run the CLI through the repo environment:
  `conda run -n ms python scripts/large_asset_sync.py ...`
- BaiduPCS-Go must be installed and authenticated before using the
  remote-facing commands: `scan-remote`, `publish`, and `align-local`.
- `scan-local` is the only command guaranteed to work without cloud access.

## Commands

- `scan-local`
  - scans the managed roots under `--repo-root` and writes a local report
  - does not require live cloud access
  - does not mutate manifests
- `scan-remote`
  - inspects the canonical remote paths corresponding to manifest-tracked files
  - writes remote state plus a manifest-vs-remote diff report
  - does not mutate manifests
- `publish`
  - compares local files against the tracked manifests
  - without `--execute`, writes a plan report only
  - with `--execute`, uploads planned files, verifies remote existence/size,
    then rewrites the tracked manifests
- `align-local`
  - compares local files plus current remote state against the tracked
    manifests
  - without `--execute`, writes a restore plan only
  - with `--execute`, downloads only planned missing or drifted local files

All subcommands require `--policy` and `--report`. `--execute` is opt-in and
never the default.

Example local scan:

```bash
conda run -n ms python scripts/large_asset_sync.py \
  scan-local \
  --repo-root . \
  --policy manifests/large_assets/policy.yaml \
  --report temp/large_asset_sync/local_scan.json
```

## Daily Workflow

1. Produce or update files locally under a managed root.
2. Run `scan-local`.
3. Run `publish --report ...` and inspect the plan.
4. Run `publish --execute ...` only when the plan is correct.
5. Commit the manifest updates.
6. On another node, `git pull`, run `scan-local`, `scan-remote`, and
   `align-local --report ...`.
7. Run `align-local --execute ...` only for the planned missing or drifted
   files.

## Stable Paths

- Policy: `manifests/large_assets/policy.yaml`
- Ignore rules: `manifests/large_assets/ignore.txt`
- Desired-state manifests:
  - `manifests/large_assets/public_data.manifest.json`
  - `manifests/large_assets/model_cache.manifest.json`
  - `manifests/large_assets/output.manifest.json`
- Disposable operator reports:
  - `temp/large_asset_sync/*.json`

Treat `manifests/large_assets/**` as durable repo state and
`temp/large_asset_sync/**` as local operator scratch output.
