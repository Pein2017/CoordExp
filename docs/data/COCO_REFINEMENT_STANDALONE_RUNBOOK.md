---
doc_id: docs.data.coco-refinement-standalone-runbook
layer: docs
doc_type: runbook
status: retired
domain: data
summary: Closed-instance record for the July 17 COCO refinement Gate A.
tags: [data, coco, annotation, standalone, gate-a, retired]
updated: 2026-10-02
---

# Retired COCO Refinement Gate A

The July 17, 2026 Gate A instance is closed. It completed eight generations and
stopped gracefully on 2026-10-02. Its committed edits were accepted in the
four canonical model-facing views below. The five unfinished native Gate A
drafts were discarded. This is a status record, not an operator guide.

- [`train.norm.jsonl`](../../public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl)
- [`val.norm.jsonl`](../../public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl)
- [`train.coord.jsonl`](../../public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl)
- [`val.coord.jsonl`](../../public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl)

The separate five-image Label Studio project is also closed. Its submitted
completions were accepted and published from the [canonical source
package](../../public_data/coco/annotation_sources/label-studio-project3-20260918/completed-annotations.json),
with the [publication receipt](../../public_data/coco/annotation_sources/label-studio-project3-20260918/publication.json).
The full row lists for image IDs `7116`, `351017`, `417044`, `477415`, and
`309264` were installed in `train.norm.jsonl` and `train.coord.jsonl` from the
same byte-qualified candidate payloads. The five rows contain 178 objects
(105 added, 10 deleted, and 7 box edits relative to 83); the other 117,261
train rows are unchanged.

Within `rescale_32_1024_bbox_len12000/`, unsuffixed `train.jsonl` and
`val.jsonl` are the original pixel-coordinate sources and remain unchanged.
The `.norm` and `.coord` files are edited model-facing views; the [shared
dataset config](../../configs/_shared/datasets/coco_1024_bbox_len12000.yaml)
selects `.coord`. The [six-file provenance
manifest](../../manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json)
records their hashes. Original processed sources in
`public_data/coco/rescale_32_1024_bbox/` remain separate.

## Evidence

The [OpenSpec design](../../openspec/changes/coco-refinement/design.md) records
the original change scope. The [Gate A research record](../../research/investigations/coco-refinement-gate-a-20260717.md)
contains historical execution evidence. The [five-image source note](../../public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4/README.md)
records the accepted data result. Storage ownership is defined in
[`OUTPUT_STORAGE_POLICY.md`](../OUTPUT_STORAGE_POLICY.md).
