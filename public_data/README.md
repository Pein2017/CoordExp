# Public data

Large datasets and label workspaces remain external. This checkout maintains only
the fixed COCO recovery path used by current configs, not the retired multi-dataset
factory. [Recovery contracts](../manifests/public_data_provenance/README.md) own
exact input/output hashes, the observed annotation delta and restored-image layout.
[Storage policy](../docs/OUTPUT_STORAGE_POLICY.md) owns transfer and retention.

- `coco_records.py`: deterministic raw COCO selection, original grid sizing,
  pixel/norm1000 conversion and content-bound annotation edits.
- `recover_coco.py`: manifest checks, raw ZIP qualification, no-overwrite recovery
  and bounded real-reader/image verification.
- `coco_annotation_delta.json`: only the current observed label edits needed to
  reconstruct consumed bytes; not new label admission or a historical snapshot.

```bash
python -m public_data.recover_coco --help
python -m public_data.recover_coco check
python -m pytest -q tests/data/test_public_data_recovery.py
```

The two current manifests are executable; the other eight identify historical
assets only. There is no legacy shell launcher or automatic archive restoration.
Do not modify existing data, model outputs or labels to make recovery checks pass.
