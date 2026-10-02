---
doc_id: docs.data.coco-refinement-runbook
layer: docs
doc_type: runbook
status: retired
domain: data
summary: Provenance and retirement record for the former Label Studio COCO refinement editor and its submitted annotations.
tags: [data, coco, label-studio, annotation, retired]
updated: 2026-10-02
---

# Retired Label Studio COCO Refinement Record

The legacy Label Studio editor and its runtime snapshots were retired on
2026-10-02. Its five submitted human annotation rows are now published in the
canonical COCO training views. No new Label Studio service or setup is defined
here.

The accepted source records and publication receipt are retained at:

```text
public_data/coco/annotation_sources/label-studio-project3-20260918/completed-annotations.json
public_data/coco/annotation_sources/label-studio-project3-20260918/publication.json
```

The completion source contains only the five submitted rows from project 3:
images `7116`, `309264`, `351017`, `417044`, and `477415`. The publication
receipt is `lead-accepted` and identifies the source and installed files. The
accepted rows add 105 regions, remove 10 prior region IDs, and revise 7 prior
boxes; the five rows contain 83 objects in the previous view and 178 in the
published view. This is adoption of submitted human annotations only. The two
unfinished drafts were not imported.

The current canonical model-facing views are
`public_data/coco/rescale_32_1024_bbox_len12000/{train,val}.{norm,coord}.jsonl`.
The accepted publication updates the five selected train rows in `train.norm`
and `train.coord`; the val views and other rows were preserved. Their current
provenance manifest is
[`rescale_32_1024_bbox_len12000.json`](../../manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json).
Official raw COCO inputs and shared images remain under
`public_data/coco/raw/` and
`public_data/coco/rescale_32_1024_bbox/images/`.

The legacy editor's input snapshots were train SHA-256
`d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a` and val
SHA-256 `a34afb33c567690f56fa3704e213cfc00dc3c000f018d2bb89a7f60139efd795`.
These identify the former editor inputs, not the current published views. The
source database identity is retained in `completed-annotations.json` and its
publication receipt.

The two unfinished draft records were discarded after the submitted
annotations were published. Their sealed export and manifest were removed
without rewriting the manifest; Git history retains the tracked manifest. The
former Label Studio output family was retired after checking its frozen source
identity and open holders. The factual retirement receipt is
[`label-studio-retirement.json`](../../.worktrees/coordexp-infras/outputs/maintenance/output-storage-closeout-20261002/label-studio-retirement.json).
