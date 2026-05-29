---
doc_id: docs.standards.output_sync_and_data_provenance
layer: docs
doc_type: standard
status: canonical
domain: artifacts
summary: Standard ownership split for output backup, public data provenance, and model cache recovery.
updated: 2026-05-11
---

# Output Sync And Data Provenance

CoordExp treats large non-code assets by whether they are externally
recoverable, reproducible, or experiment-specific.

## Ownership Rules

- `model_cache/` is not a Baidu Netdisk sync surface. It contains upstream
  pretrained assets and local cache state; each machine prepares it locally.
- raw `public_data/` is not a Baidu Netdisk sync surface. Each machine fetches
  raw datasets from the original source or a local mirror.
- processed `public_data/` directories are not mirrored by default. Their
  generation provenance is tracked in git under
  `manifests/public_data_provenance/`.
- `outputs/` is the Baidu Netdisk sync surface. It contains experiment artifacts
  that may be difficult or impossible to regenerate exactly.

The canonical remote layout is:

```text
outputs/... -> /CoordExp/outputs/...
```

Use repo-relative paths when describing artifacts. Avoid machine-specific
absolute paths in durable records unless they are explicitly labeled as local
operator notes.

## Public Data Provenance

Every durable processed directory under `public_data/` should have a matching
provenance JSON file under:

```text
manifests/public_data_provenance/
```

Mirror the `public_data/` path below that directory. Example:

```text
public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/
manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60_lvis_proxy.json
```

The provenance record should answer:

- what directory was produced
- which script or module produced it
- which command was run
- which raw inputs, configs, checkpoints, or side manifests were consumed
- which key parameters materially affect the output
- which code state the record is tied to, when known

These files are small and belong in git. They are not backed up through Baidu
Netdisk as the primary truth source.

## Outputs Backup

Back up `outputs/` inclusively. This includes:

- training run directories
- adapter checkpoints and adapter metadata
- inference artifacts under `outputs/infer/**`
- evaluation outputs, metrics, summaries, and JSONL records
- empty files or interrupted files if they are present on disk

Do not apply automatic ignore rules to `outputs/` by default. Clean up the local
directory manually when something should not be preserved.

When a node still has active training writing to `output_remote/`, do not rename
or delete that tree in place. Instead, absorb it into `outputs/` with a
no-overwrite copy:

```bash
python scripts/absorb_output_remote_into_outputs.py --apply
```

This keeps `output_remote/` intact for the active writer while making
`outputs/` the canonical sync surface for Baidu Netdisk.

For long transfers, use `tmux` and BaiduPCS-Go. On offline nodes that can only
egress through the local proxy, export:

```bash
export http_proxy=http://127.0.0.1:9090
export https_proxy=http://127.0.0.1:9090
export HTTP_PROXY=http://127.0.0.1:9090
export HTTPS_PROXY=http://127.0.0.1:9090
```

Use the repo skill `.codex/skills/baidudisk-union-sync` for append-only
cross-machine sync on `/CoordExp/outputs`, and `.codex/skills/baidupcsgo-upload`
for one-off upload/download recovery.

## Duplicate Policy

The steady-state rule is conflict-first:

- a remote path that already exists should be treated as an error
- overwrite is allowed only when the operator explicitly requests it for a
  migration or repair

This prevents two isolated A100 nodes from silently replacing one another's
artifacts. The first cleanup or migration pass may be run with explicit
overwrite because older remote state predates this standard.

## Quick Checks

Before relying on a processed data directory:

```bash
test -f manifests/public_data_provenance/<dataset>/<processed-dir>.json
```

Before relying on a remote outputs backup:

```bash
./baidupcsgo/BaiduPCS-Go-v4.0.1-linux-amd64/BaiduPCS-Go ls /CoordExp/outputs
```

Use targeted directory listings for the exact run path instead of full-tree
hashing of raw datasets.
