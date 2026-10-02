---
doc_id: docs.data.coco-refinement-runbook
layer: docs
doc_type: runbook
status: retired
domain: data
summary: Recovery notes for the retired Label Studio COCO refinement editor and its pending-draft export.
tags: [data, coco, label-studio, annotation, retired]
updated: 2026-10-01
---

# Retired Label Studio COCO Refinement Notes

The legacy Label Studio editor has been retired. Gate A is the sole active COCO
refinement editor; use
[`COCO_REFINEMENT_STANDALONE_RUNBOOK.md`](COCO_REFINEMENT_STANDALONE_RUNBOOK.md)
for its human-only workflow. Do not start the retired Django/Label Studio
service or its resident ROI model.

The two unfinished drafts are preserved verbatim, with their task rows, linked
completed annotations, and project label configuration, at:

```text
public_data/coco/annotation_drafts/retired-label-studio-20261002/drafts.jsonl
manifests/annotation_drafts/retired_label_studio_20261002.json
```

These are pending drafts for human recovery only. They are not ground truth,
training data, a Gate A input, or permission to import drafts or change labels.
The stopped Label Studio database is retained only until the pending-draft
export is independently verified and the old state is deleted; no post-move
runtime path is assigned. Its captured source path records the former location
under `outputs/label_studio_coco_refinement/`. Do not copy credentials into
`public_data/`.

## Frozen source identity

The legacy editor's source contract remains frozen at train SHA-256
`d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a` and val
SHA-256 `a34afb33c567690f56fa3704e213cfc00dc3c000f018d2bb89a7f60139efd795`.
The current public-data views and old working snapshots do not match these
bytes. The editor must remain retired; do not relax the hashes, overwrite
source views, or rebase the retained drafts without an explicit research
decision. A fresh standalone runtime would fail its source fingerprint check.

The immutable model-facing views remain
`public_data/coco/rescale_32_1024_bbox_len12000/{train,val}.{norm,coord}.jsonl`.
Official raw COCO inputs and shared images remain in their existing
`public_data/coco/raw/` and `public_data/coco/rescale_32_1024_bbox/images/`
owners. Relocation of Gate A's mutable annotation views from its existing
`outputs/coco_refinement/gate-a-20260717/` runtime is explicitly USER-DEFERRED
to the next round; do not treat it as cut over or create a new runtime root.

## Human recovery context

The retired editor contained two managed source projects (train and val) plus
a five-image exploratory subproject (project ID 3). That subproject used image
IDs `7116`, `309264`, `351017`, `417044`, and `477415`. Only two task drafts
were present at retirement; the export records the exact task and project
identity for each one.

The historical project edited official English COCO-80 axis-aligned bounding
boxes over the existing COCO image store. Its source coordinates used integer
norm1000 `xyxy` values on `0..999`. To inspect a draft by hand, read the
preserved `draft.result` together with that record's `task.data`, linked
annotation, and `project_context.label_config`; resolve its image through the
existing public COCO image store. Keep all original strings intact. Do not use
the old exporter to publish a working dataset or feed a draft into Gate A.

Useful historical editor controls for visual review were zoom and zoom
controls, inline labels, a semi-transparent two-pixel rectangle outline, and
an image crosshair. With the rectangle tool active, a drag over an existing
rectangle created a new box; use the normal selection tool to move or resize.
The `Regions` outliner was the reliable way to select a small box covered by a
larger one. These notes describe the retired UI only; they do not change the
pending annotations.
