# Public-data provenance manifests

Each retained processed COCO owner has one JSON manifest validated by
`python -m public_data.provenance`. The schema owner is `schema.json`; there is
no second manifest format or validator.

`regeneration_status: ready` is required for validation success. A manifest on
`hold_unresolved_source_parity` records current content identity but fails
closed until its producer can reproduce that content.

Every manifest names:

- the logical output root and artifact type;
- one retained producer module and a `python -m public_data...` command;
- executable dependencies, upstream inputs, and generation parameters;
- JSONL-only SHA-256 records for model-facing samples; and
- lightweight metadata sidecars when present.

Routine validation never hashes raw or resized image trees. If a processed root
is absent, the validator reports `absent`; this means the manifest closure is
usable but local content was not validated. If a root is present, its complete
top-level JSONL set and declared sidecars must match.

Refinement manifests distinguish live authority files from recorded receipt
identity. Runtime receipts, journals, projects, working data, task indices,
outputs, and tokenizer files are live dependencies and must resolve with their
declared hashes. `historical_training_config` instead uses
`identity_mode: recorded_receipt_only`: its exact `recorded_path` and SHA-256
must match the immutable publication receipt, but the historical file is not
resolved or required in the current checkout. It is provenance evidence, not a
current training or regeneration config.

```bash
python -m public_data.provenance \
  --manifest manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json \
  --repo-root .
```

LVIS currently has raw local inputs but no retained processed materialization,
so no LVIS processed manifest is fabricated. Regenerate a fresh processed LVIS
root through `public_data/run.sh lvis ...`, then add a manifest only after the
artifact exists and its JSONL identities are frozen.
