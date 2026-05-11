# Public Data Provenance Manifests

This directory stores small git-tracked provenance records for durable
processed directories under `public_data/`.

It does not store data files. It records how processed data was produced so a
new machine can regenerate the directory after preparing raw datasets locally.

Path convention:

```text
public_data/<dataset>/<processed-dir>/
manifests/public_data_provenance/<dataset>/<processed-dir>.json
```

Each manifest should follow `schema.json` and include the exact production
command whenever possible.

Materialized training-sample directories should also include a JSONL-only
checksum block. The checksum scope is intentionally narrow: hash the
model-facing `*.jsonl` files and an aggregate over those file records, but do
not hash raw images, resized images, caches, or whole `public_data/` trees.

Minimal example:

```json
{
  "schema_version": 1,
  "relative_path": "public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy",
  "producer_script": "scripts/data/prepare_coco_rescaled.py",
  "working_dir": ".",
  "command": "conda run -n ms python scripts/data/prepare_coco_rescaled.py --config configs/data/coco_rescale_32_1024_bbox_max60_lvis_proxy.yaml",
  "inputs": [
    {
      "kind": "raw_dataset",
      "path": "public_data/coco/raw",
      "notes": "Prepared locally from the COCO source files."
    }
  ],
  "key_params": {
    "image_size": 1024,
    "min_size": 32,
    "max_objects": 60
  },
  "checksums": {
    "scope": "jsonl_training_samples_only",
    "algorithm": "sha256",
    "aggregate_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "aggregate_source": "sorted path sha256 size_bytes records lines",
    "files": [
      {
        "path": "public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl",
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "size_bytes": 123,
        "records": 1
      }
    ]
  },
  "code_ref": null,
  "generated_at_utc": null,
  "notes": "Example shape only; replace with the real producer and command."
}
```
