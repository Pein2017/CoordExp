# Public Data Provenance Manifests

This directory stores small git-tracked provenance records for durable
`public_data/` artifacts.

It does not store data files. It records how processed data was produced so a
new machine can regenerate the directory after preparing raw datasets locally.

Path convention:

```text
public_data/<dataset>/<processed-dir>/
manifests/public_data_provenance/<dataset>/<processed-dir>.json

public_data/<dataset>/images/<image-store>/
manifests/public_data_provenance/<dataset>/images/<image-store>.json

public_data/<dataset>/views/<view-family>/<view-name>/
manifests/public_data_provenance/<dataset>/views/<view-family>/<view-name>.json
```

Each manifest should follow `schema.json` and include the exact production
command whenever possible. Manifests must include `artifact_type`:

- `processed_directory` for legacy processed roots.
- `image_store` for reusable image roots such as
  `public_data/coco/images/res-1024`.
- `annotation_view` for model-facing JSONL views such as
  `public_data/coco/views/coco80/len-12000`.

Materialized training-sample directories and annotation views should include a
JSONL-only checksum block. The checksum scope is intentionally narrow: hash the
model-facing `*.jsonl` files and an aggregate over those file records, but do
not hash raw images, resized images, caches, or whole `public_data/` trees.
Tests verify checksum entries when the artifact root exists locally. If a
checkout does not have generated data, tests skip local file hashing for that
absent root. If the root exists, every listed JSONL and sidecar must exist, and
the local top-level JSONL set must match the manifest checksum block.

Image-store manifests use `checksums: null` by default. Full image-file hashing
is intentionally out of the routine provenance path; it can be added later only
for an explicit freeze or cross-node image-store audit.

Use a fresh materialization command for image-store manifests, such as
`--image-store-mode hardlink`, `reflink`, or `copy`. `reuse-existing` is useful
for local revalidation after the image store already exists, but it is not a
fresh regeneration command.

Annotation-view manifests should also record lightweight sidecar metadata when
available under `metadata`. The Git-tracked manifest should carry small audit
fields inline under `key_params.view_summary`, so review does not require the
ignored local sidecars to be present. For example:

```json
{
  "metadata": {
    "view_metadata": {
      "path": "public_data/coco/views/coco80/len-12000/meta.json",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size_bytes": 123
    },
    "source_comparison": {
      "path": "public_data/coco/views/coco80/len-12000/source_comparison.json",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size_bytes": 123
    },
    "length_stats": {
      "train": {
        "path": "public_data/coco/views/coco80/len-12000/train.length_stats.json",
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "size_bytes": 123
      }
    }
  }
}
```

Derived artifacts at the same image resolution should share the canonical
processed image root instead of copying or relinking images. For example, COCO
1024 length-budget variants keep only JSONL/meta files and write relative image
paths that point back to `public_data/coco/rescale_32_1024_bbox/images/`.
Phase 1 view-architecture artifacts instead resolve JSONL image paths through
the declared image store, for example
`public_data/coco/images/res-1024`.

Minimal example:

```json
{
  "schema_version": 1,
  "artifact_type": "annotation_view",
  "relative_path": "public_data/coco/views/coco80/len-12000",
  "producer_script": "public_data/scripts/build_coco_views.py",
  "working_dir": ".",
  "command": "PYTHONPATH=. conda run -n ms python public_data/scripts/build_coco_views.py --views coco80/len-12000 --max-total-tokens 12000 --image-store-mode reuse-existing",
  "inputs": [
    {
      "kind": "raw_dataset",
      "path": "public_data/coco/raw",
      "notes": "Prepared locally from the COCO source files."
    }
  ],
  "key_params": {
    "image_store": "public_data/coco/images/res-1024",
    "image_path_semantics": "image_store_relative",
    "coordinate_space": "norm1000",
    "coordinate_storage": "integer",
    "length_budget_scope": {
      "rendered_families": ["objects"],
      "excluded_sidecars": ["metadata.supervision.support_objects"]
    },
    "view_summary": {
      "records": 1,
      "rendered_object_count": 1,
      "support_sidecar_count": 0
    },
    "routine_sync_policy": "regenerate_from_raw_plus_manifest"
  },
  "checksums": {
    "scope": "jsonl_training_samples_only",
    "algorithm": "sha256",
    "aggregate_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "aggregate_source": "sorted path sha256 size_bytes records lines",
    "files": [
      {
        "path": "public_data/coco/views/coco80/len-12000/train.jsonl",
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "size_bytes": 123,
        "records": 1
      }
    ]
  },
  "metadata": {
    "view_metadata": {
      "path": "public_data/coco/views/coco80/len-12000/meta.json",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size_bytes": 123
    },
    "source_comparison": {
      "path": "public_data/coco/views/coco80/len-12000/source_comparison.json",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size_bytes": 123
    }
  },
  "code_ref": null,
  "generated_at_utc": null,
  "notes": "Example shape only; replace with the real producer and command."
}
```
