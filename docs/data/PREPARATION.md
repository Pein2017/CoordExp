# Data preparation and intake

The maintained recovery path is [fixed COCO recovery](../../manifests/public_data_provenance/README.md),
not a universal multi-dataset factory. It reconstructs the exact current consumed
view, including observed annotation edits, from checksum-bound raw archives.
[Storage policy](../OUTPUT_STORAGE_POLICY.md) owns backup and retention; the
[JSONL contract](CONTRACT.md) owns coordinate and image-path interpretation.

```bash
python -m public_data.recover_coco check
python -m public_data.recover_coco verify \
  --manifest manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json \
  --workspace /path/to/restored-workspace
```

The complete restore/dry-run commands and pinned dependencies live in the recovery
README rather than being maintained a second time here. Eight older proxy/view
identities are historical-only; their presence is not a current factory interface.

Before training, verify the exact declared JSONL bytes, image resolution and paths,
coordinate chart/range, object metadata, annotation version and intended order.
A bounded reader/image smoke is a necessary integration check, not full dataset
qualification. Retain pixel and normalized geometry distinctions; no runtime
image resize or silent bbox-chart conversion may repair an incompatible input.

Point the selected current config's `data.train.path` and `data.eval.path` at the
restored absolute inputs, then resolve a new config identity. Do not overwrite an
existing dataset or change an old receipt/hash to disguise a different version.
Model-specific token length is still validated by the current packing pipeline.
The fixed recovery is not a general 12k-token filter or permission to discard
objects/images from another corpus.

For a new dataset or alternate coordinate/label policy, establish a bounded
conversion and validation contract before introducing an entry. The prior LVIS,
proxy, alternate-bbox and legacy tokenizer/factory implementations remain Git
history, not current commands.
