# Retired five-image Label Studio subproject

Project 3 (`CoordExp COCO refinement - 5-image subproject`) is closed. Its
submitted annotations for COCO image IDs `7116`, `309264`, `351017`, `417044`,
and `477415` were accepted and published in the canonical
[source package](../../annotation_sources/label-studio-project3-20260918/completed-annotations.json)
with its [publication receipt](../../annotation_sources/label-studio-project3-20260918/publication.json).
No editor setup is described here.

The full row lists for all five images were replaced in the edited
`train.norm.jsonl` and `train.coord.jsonl` views using the same byte-qualified
candidate payloads. The five rows contain 178 objects, compared with 83 before
publication: 105 additions, 10 deletions, and 7 box edits. The other 117,261
train rows are unchanged.

## Canonical data roles

The unsuffixed `train.jsonl` and `val.jsonl` in
`public_data/coco/rescale_32_1024_bbox_len12000/` are unchanged original
pixel-coordinate sources. The `.norm` and `.coord` files are edited
model-facing views; the [shared dataset
config](../../../../configs/_shared/datasets/coco_1024_bbox_len12000.yaml)
selects `.coord`. Original processed train/val sources under
`public_data/coco/rescale_32_1024_bbox/` remain separate.

The [six-file provenance
manifest](../../../../manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json)
records hashes for the six train/val files.
