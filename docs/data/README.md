# Data and geometry

Current source validation, image paths and bbox semantics live in `src.data`.
[Contract](CONTRACT.md) and [packing](PACKING.md) describe the supported surfaces.
For concrete frozen populations and external roots use [research assets](../../research/assets.md).

The in-tree legacy dataset conversion factories are retired; their processed
outputs and `manifests/public_data_provenance` are not deleted. New data preparation
must use an explicit current implementation and bind original/derived bytes.
Paths resolve from the JSONL parent. Do not reinterpret pixels as normalized bins,
change label versions silently, or infer negatives from absent COCO annotations.
