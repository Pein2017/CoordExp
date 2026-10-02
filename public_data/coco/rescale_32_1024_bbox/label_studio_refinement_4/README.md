# Retired five-image Label Studio subproject

Project 3 (`CoordExp COCO refinement - 5-image subproject`) is retired. Its
five-image scope was COCO image IDs `7116`, `309264`, `351017`, `417044`, and
`477415`. The former local URL is no longer served. Gate A is the sole active
refinement editor.

The two unfinished drafts are preserved with exact task/project context in
[`public_data/coco/annotation_drafts/retired-label-studio-20261002/drafts.jsonl`](../../annotation_drafts/retired-label-studio-20261002/drafts.jsonl).
The hash and row-ID record is
[`manifests/annotation_drafts/retired_label_studio_20261002.json`](../../../../manifests/annotation_drafts/retired_label_studio_20261002.json).
These drafts remain pending. They are not ground truth or training data, and
are not merged into the current train/val views or imported into Gate A.

The historical exporter is retained for source inspection but is not an
operator command. It writes a working dataset from Label Studio state and must
not be used to publish labels or update model-facing views without a separate
explicit decision.

## Historical visual-review notes

The project edited official English COCO-80 axis-aligned rectangles over the
existing image store. Zoom controls stayed enabled; labels appeared inline;
rectangles used a semi-transparent two-pixel outline; and the image crosshair
was enabled. With the rectangle tool active, dragging over an existing box
created a new box. Use the normal selection tool to move or resize. The
`Regions` outliner was the reliable way to select a small box covered by a
larger box. These interaction notes do not modify or interpret the pending
drafts.
