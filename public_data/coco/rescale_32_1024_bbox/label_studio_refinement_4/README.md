# Five-image Label Studio refinement subproject

This directory is the file boundary for Project 3 (`CoordExp COCO refinement - 5-image subproject`).
It contains only image IDs `7116`, `309264`, `351017`, `417044`, and `477415`.

- `source.norm.jsonl`: the five current source rows copied from the latest `train.norm.jsonl`.
- `working.norm.jsonl`: the current GT snapshot; it is refreshed after every successful
  Project 3 `Update` and can contain added, deleted, moved, or relabeled boxes.
- `project_manifest.json`: project, task, source-line, and checksum bindings.
- `export_subproject.py`: fail-closed exporter for the five tasks only.

Open Project 3 at:

`http://127.0.0.1:8080/projects/3/data`

The running local server refreshes the snapshot automatically after `Update`. To
rebuild it manually (for recovery or verification), run:

```bash
BASE_DATA_DIR=/data/CoordExp/outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000/label-studio/state \
LOCAL_FILES_DOCUMENT_ROOT=/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images \
LOCAL_FILES_SERVING_ENABLED=true \
DJANGO_DB=sqlite \
DJANGO_SETTINGS_MODULE=core.settings.label_studio \
PYTHONPATH=/data/CoordExp \
/data/CoordExp/label-studio/.venv/bin/python \
/data/CoordExp/public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4/export_subproject.py
```

The exporter writes only `working.norm.jsonl` in this directory. It preserves
existing COCO annotation IDs, assigns deterministic negative IDs to newly drawn
regions, and treats the live annotation list as authoritative: additions,
deletions, geometry edits, and class edits are all published. It still rejects
tasks outside the five-image subset.

## Frontend interaction profile

Project 3 follows the older refinement frontend's useful interaction cues: zoom
and zoom controls stay enabled, labels are shown inline, rectangles use a
semi-transparent two-pixel outline, and the image crosshair is enabled. When a
rectangle tool is active, a drag over an existing rectangle is routed to the
new-box tool instead of selecting the old box; switch to the normal selection
tool when you need to move or resize an existing box. The editor's `Regions`
(outliner) tab remains the authoritative list for selecting a small box when a
larger box covers it. The legacy relation/group tab is hidden for this project
only; prediction/evidence layers and per-region text controls are intentionally
not imported because this subproject exports rectangle-label results only and
must remain limited to the five original annotations.
