---
doc_id: docs.standards.output_sync_and_data_provenance
layer: docs
doc_type: standard
status: canonical
domain: artifacts
summary: Standard ownership split for local outputs, public data provenance, and model cache recovery.
updated: 2026-10-02
---

# Output Ownership And Data Provenance

CoordExp places non-code assets with their physical owner and records how
derived data was produced.

## Ownership Rules

- `model_cache/` contains upstream pretrained assets and machine-local cache
  state; each machine prepares it locally.
- raw `public_data/` datasets are fetched from their original source or a local
  mirror.
- processed `public_data/` directories are derived data; their generation
  provenance is tracked in git under `manifests/public_data_provenance/`.
- `outputs/` holds artifacts for its owning physical worktree. The repository
  root's `outputs/` is a selected shared-asset store, not a default run
  destination; see [the output storage policy](../OUTPUT_STORAGE_POLICY.md).

This document does not define a remote backup target or namespace.

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
public_data/coco/rescale_32_1024_bbox_len12000/
manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json
```

The provenance record should answer:

- what directory was produced
- which script or module produced it
- which command was run
- which raw inputs, configs, checkpoints, or side manifests were consumed
- which key parameters materially affect the output
- which code state the record is tied to, when known

These files are small and belong in git as the provenance source.

## Output placement

Keep a branch's runs under its physical worktree owner. Copy an artifact into
the root shared-asset store only when it has an explicit cross-worktree use, as
described by [the output storage policy](../OUTPUT_STORAGE_POLICY.md). This
standard does not configure or imply external synchronization.

The existing [`absorb_output_remote_into_outputs.py`](../../scripts/absorb_output_remote_into_outputs.py)
helper performs a local copy. Its source, destination, and report paths default
relative to the current working directory. For a scoped migration, run it from
the physical worktree that owns the artifacts, set its report under task-owned
`.local/scratch/`, and choose the destination under the storage policy. It does
not define a remote backup path.

## Quick Checks

Before relying on a processed data directory:

```bash
test -f manifests/public_data_provenance/<dataset>/<processed-dir>.json
```

Use targeted directory listings for the exact run path instead of full-tree
hashing of raw datasets.
