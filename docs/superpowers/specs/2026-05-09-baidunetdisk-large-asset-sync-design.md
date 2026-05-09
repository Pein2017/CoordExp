---
title: Baidu Netdisk Large Asset Sync Design
date: 2026-05-09
status: draft
change: baidunetdisk-large-asset-sync
---

# Baidu Netdisk Large Asset Sync Design

## Problem

CoordExp now spans at least two A100 nodes that do not have direct network
reachability to each other. Code can move through `git`, but the large
non-code asset trees that matter for real work do not fit cleanly into the git
workflow:

- `public_data/**`
- `model_cache/**`
- `output/**`

The desired user experience is close to "one machine with two execution
surfaces":

- both nodes should preserve the same repo-relative asset layout;
- Baidu Netdisk should serve as the large-file backup and transfer layer;
- switching nodes should not require a fresh full download from the cloud;
- important sync status should be answerable from metadata and manifests, not
  from repeated bulk transfers.

This design therefore introduces a manifest-driven large-asset sync contract
that complements `git` without changing the existing repo-relative path model.

## Goals

- Treat repo-relative paths as the stable asset identity.
- Make `git` the truth source for what large assets should exist.
- Use Baidu Netdisk as the cross-node backup and transfer substrate.
- Support fast "am I synced?" checks without re-downloading large trees.
- Keep the day-to-day workflow half-automatic:
  - create files locally first;
  - publish in batches when switching nodes or ending a phase;
  - inspect a report before downloading missing files on another node.
- Allow `output/**` to be broadly managed while excluding obvious temporary
  files.

## Non-Goals

- No attempt to turn CoordExp into a full content-addressed object store.
- No requirement that every new file be uploaded immediately after creation.
- No new absolute-path contract in manifests; local roots remain repo-relative.
- No dependence on direct node-to-node communication.
- No guarantee that Baidu Netdisk remote timestamps equal local timestamps.
- No repeated full-tree cloud downloads just to verify parity.

## Approaches Considered

### 1. Directory mirror only

Compare local and remote trees using path existence, file size, and timestamps.

Pros:

- simple mental model;
- no extra repo metadata to maintain.

Cons:

- weak auditability;
- easy to fool with rewritten files;
- remote timestamps are not trustworthy enough to act as the truth source;
- poor fit for multi-node drift diagnosis.

Verdict:

- rejected as too weak for long-lived research assets.

### 2. Git-tracked manifest as truth source

Track expected large assets in small repo-side manifests. Nodes and cloud
storage align to those manifests.

Pros:

- clear authority chain;
- keeps relative paths stable;
- works well when nodes cannot talk directly;
- allows fast verification from metadata with targeted strong checks only when
  needed.

Cons:

- introduces a lightweight manifest management workflow;
- requires intentional publish steps when large assets change.

Verdict:

- recommended.

### 3. Full content-addressed asset store

Store every asset by hash and materialize directory trees as views.

Pros:

- strongest deduplication and integrity model.

Cons:

- too heavyweight for the current CoordExp workflow;
- would materially change how local paths and backups are managed.

Verdict:

- deferred indefinitely.

## Recommended Design

Use `git`-tracked large-asset manifests as the truth source. Preserve the
existing repo-relative layout locally and mirror the same relative layout under
the canonical Baidu Netdisk root `/CoordExp`.

The authority chain is:

```text
git manifests
  -> define what should exist
local node state
  -> defines what this machine currently has
Baidu Netdisk /CoordExp mirror
  -> defines what can be restored onto another machine
```

The sync system is intentionally report-first:

- local scan reports whether a node matches manifest;
- remote scan reports whether Netdisk matches manifest;
- align/download only fetches missing or drifted local files after explicit
  confirmation.

## Stable Path Contract

The managed asset root is the repo root. Asset identity is always the path
relative to the repo root, never an absolute machine path.

Managed roots:

- `public_data/**`
- `model_cache/**`
- `output/**`

Canonical Netdisk mirror root:

- `/CoordExp/public_data/**`
- `/CoordExp/model_cache/**`
- `/CoordExp/output/**`

This means:

- local machine A and local machine B may live at different absolute roots;
- manifests still store only repo-relative paths;
- upload/download tools must preserve relative path exactly.

The older top-level remote paths such as `/output` or `/model_cache` may remain
historically present, but they are not the canonical target for the new sync
contract. New sync operations should converge on `/CoordExp/**`.

## Managed Metadata Model

The design uses three git-tracked manifests plus one policy file:

- `manifests/large_assets/public_data.manifest.json`
- `manifests/large_assets/model_cache.manifest.json`
- `manifests/large_assets/output.manifest.json`
- `manifests/large_assets/policy.yaml`

One optional ignore file controls exclusions:

- `manifests/large_assets/ignore.txt`

One local ephemeral workspace captures per-node snapshots and reports:

- `temp/large_asset_sync/`

The manifests are the durable repo contract. The temp snapshots are local
diagnostic artifacts and do not enter `git`.

## Manifest Schema

Each manifest records the expected files below one managed root. Directory
entries may be omitted if parent path derivation is sufficient; files are the
important contract.

Each file record should include:

- `relative_path`
- `size_bytes`
- `local_mtime_utc`
- `sha256`
- `hash_policy`
- `managed_root`
- `last_verified_utc`

Recommended `hash_policy` values:

- `full`: full-file `sha256` is required and current
- `sampled`: fast path uses metadata, full hash may be deferred until publish
- `none`: reserved for future exceptional cases, not expected in the default
  policy

The manifest should also include a small header with:

- schema version
- generation timestamp
- repo root contract
- canonical remote root `/CoordExp`

## Policy Model

`policy.yaml` should define:

- managed roots
- remote root
- ignore file location
- fast-verify rules
- strong-verify rules
- default hash policy by root

Recommended defaults:

- `public_data/**`: `full`
- `model_cache/**`: `full`
- `output/**`: `sampled` during normal scans, upgraded to `full` on publish for
  changed files

This preserves a strong integrity story for stable datasets and model assets,
while keeping active `output` scans cheap enough for regular use.

## Ignore Rules

`output/**` is broadly managed, but obvious transient files should be excluded.
The ignore file should support simple glob-style patterns.

Recommended initial exclusions:

- `**/*.tmp`
- `**/*.part`
- `**/*.lock`
- `**/.DS_Store`
- `**/tmp/**`
- `**/cache/**`
- `**/*.uploading`
- `**/*.downloading`
- `**/wandb/latest-run/**` if present in future workflows

The default stance is:

- include almost everything;
- exclude only files that are clearly temporary, incomplete, or operationally
  disposable.

## Verification Modes

### Fast verify

Fast verify is the default for daily use and node switching.

Inputs:

- git manifests
- local file metadata
- Netdisk path listings

Checks:

- path presence
- file size
- local mtime drift against manifest for local files only
- remote presence and remote size against manifest for cloud files

Outputs:

- `synced`
- `missing_local`
- `missing_remote`
- `drift_local_metadata`
- `drift_remote_size`
- `unknown_remote_state` if the remote listing is incomplete or fails

Fast verify does not require downloading large files.

### Strong verify

Strong verify is used:

- on first enrollment into manifests;
- on publish for changed files;
- when a fast verify report marks a file as suspicious;
- when the user wants a higher-confidence audit.

Checks:

- recompute `sha256` for target local files;
- compare against manifest;
- optionally sample a small number of downloaded cloud restores in a manual
  audit workflow if remote integrity is ever in doubt.

Because Baidu Netdisk does not provide a trustable remote content hash API in
this workflow, strong verification is local-first. Remote integrity is inferred
from successful upload plus remote path-and-size parity, not from remote hash
recalculation.

## Operational Commands

The implementation should provide four primary command families behind one
repo-local tool, likely a Python entrypoint under `scripts/`.

### 1. Scan local

Purpose:

- inventory managed local files;
- compare local state to manifest;
- optionally compute hashes for changed files.

Typical use:

- before publish;
- after `git pull` on another node;
- before deciding what to download.

### 2. Scan remote

Purpose:

- walk `/CoordExp/**` on Netdisk using `BaiduPCS-Go`;
- compare remote path-and-size state to manifest;
- produce a cloud sync report without downloading payloads.

Typical use:

- before switching nodes;
- after upload batches;
- when diagnosing whether the cloud mirror is complete.

### 3. Publish

Purpose:

- refresh manifest entries from local state for intentional changes;
- upload missing or changed files to canonical remote paths;
- leave repo-relative layout unchanged.

Required behavior:

- stage manifest updates only after the tool has a coherent local view;
- upload to `/CoordExp/<relative_path>`;
- verify remote existence and size after upload;
- write a publish report under `temp/large_asset_sync/`.

Publish is the only path that should intentionally advance the manifest truth.

### 4. Align local

Purpose:

- read manifests and remote scan results;
- report what the current node is missing;
- after explicit confirmation, download only the missing or drifted files.

Required behavior:

- dry-run by default;
- explicit execute flag required for download;
- preserve relative paths exactly when restoring into the repo root;
- avoid touching already-synced files.

This is the critical command family for the "switch machine without full
re-download" experience.

## Report Model

Every scan or publish action should emit a small machine-readable report under
`temp/large_asset_sync/`. Suggested files:

- `local_scan.json`
- `remote_scan.json`
- `publish_report.json`
- `align_plan.json`

Each report should summarize:

- total managed files
- total bytes
- new files
- missing files
- changed files
- ignored files
- suspicious files needing strong verify

The report files are intentionally local and disposable. The durable truth
remains the manifests committed to `git`.

## Lifecycle Workflow

### On the node that produced new large assets

1. create or update files locally under managed roots;
2. run local scan;
3. run publish in plan mode to inspect changes;
4. execute publish to:
   - refresh manifest entries for intended changes
   - upload changed files to `/CoordExp/**`
5. commit the manifest updates to `git`.

### On the other node

1. `git pull`
2. run local scan
3. run remote scan
4. inspect align plan
5. confirm download only for files marked missing or drifted locally

This preserves the chosen half-automatic workflow:

- local-first creation;
- batch publish;
- report before download.

## Failure Handling

If local scan fails:

- do not modify manifests;
- surface the first unreadable or unstable paths;
- allow rerun on a narrower root if needed.

If remote scan fails partially:

- mark affected paths as `unknown_remote_state`;
- do not silently treat them as missing;
- allow re-scan before any destructive action.

If publish uploads some files but not others:

- remote verification should classify partial success explicitly;
- manifest updates should not be committed until the report is reviewed;
- rerun publish should skip already-synced files and continue from the delta.

If align download is interrupted:

- partially restored files should remain excluded until revalidated;
- temp download markers should be ignored by the manifest rules;
- rerun align should continue from the remaining missing set.

## Risks And Guardrails

### Risk: remote timestamps drift from local timestamps

Guardrail:

- do not use remote `mtime` as a truth signal;
- use remote presence and size only in the fast path.

### Risk: `output/**` grows quickly and makes scans expensive

Guardrail:

- default to metadata-first scans;
- hash only changed or suspicious files unless publish requests stronger
  verification.

### Risk: manifest truth advances before cloud upload is actually usable

Guardrail:

- publish must verify remote existence and size before it is treated as
  complete;
- manifest changes should be committed only after publish review.

### Risk: canonical and legacy remote roots diverge

Guardrail:

- all new sync operations target `/CoordExp/**`;
- legacy remote roots are treated as historical leftovers, not active truth.

## Testing And Verification Plan

The implementation plan should require at least:

- unit tests for ignore-rule matching;
- unit tests for manifest diff classification;
- unit tests for relative-path mapping to `/CoordExp/**`;
- unit tests for dry-run align behavior;
- a small integration smoke over a temporary managed subtree with:
  - local scan
  - manifest write
  - remote publish plan
  - remote restore plan

The first production-facing smoke does not need to publish all real assets. It
only needs to prove that one realistic subtree can:

- enroll into manifest;
- map to canonical remote paths;
- produce a correct missing/synced plan on a second local state.

## Initial Implementation Shape

The first implementation should stay narrow:

- one Python tool under `scripts/` for scan/publish/align flows;
- reuse the existing BaiduPCS-Go binary and wrapper scripts where practical;
- no background daemon;
- no always-on filesystem watcher;
- no database.

This keeps the feature compatible with the current repo operating style and
easy to audit.

## Acceptance Criteria

This design is successful when:

- manifests become the clear git-tracked truth for managed large assets;
- both A100 nodes can determine sync state without a full cloud re-download;
- cloud storage mirrors repo-relative paths exactly under `/CoordExp/**`;
- local align defaults to report-first behavior;
- transient `output/**` files do not churn the managed manifests;
- the workflow feels close to a single-machine development experience.

## Assumptions

- Baidu Netdisk space remains sufficient for broad `output/**` coverage.
- Future checkpoints are usually adapter-only and therefore modest enough to
  make broad sync practical.
- `git` remains the authoritative coordination surface for manifests.
- The repo root on each node contains the canonical local relative asset tree.
