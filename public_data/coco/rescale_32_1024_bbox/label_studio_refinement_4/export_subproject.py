#!/usr/bin/env python3
"""Export only the four-image Label Studio subproject into working.norm.jsonl."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
STATE_ROOT = REPO_ROOT / "outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000/label-studio/state"

os.environ.setdefault("BASE_DATA_DIR", str(STATE_ROOT))
os.environ.setdefault(
    "LOCAL_FILES_DOCUMENT_ROOT",
    str(REPO_ROOT / "public_data/coco/rescale_32_1024_bbox/images"),
)
os.environ.setdefault("LOCAL_FILES_SERVING_ENABLED", "true")
os.environ.setdefault("DJANGO_DB", "sqlite")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "label-studio" / "label_studio"))

import django  # noqa: E402

django.setup()

from projects.models import Project  # noqa: E402
from src.label_studio_coco_refinement.draft_adapter import (  # noqa: E402
    DraftContractError,
    canonicalize_label_studio_draft,
)
from tasks.models import Task  # noqa: E402


def _canonical_line(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest_path = HERE / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_id = manifest.get("project_id")
    image_ids = {int(value) for value in manifest.get("image_ids", [])}
    if project_id != 3 or image_ids != {7116, 351017, 417044, 477415}:
        raise RuntimeError("subproject manifest identity is not the expected four-image scope")

    source_path = HERE / "source.norm.jsonl"
    source_by_image: dict[int, dict] = {}
    for line_number, raw_line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(raw_line)
        image_id = row.get("image_id")
        if not isinstance(image_id, int) or image_id in source_by_image:
            raise RuntimeError(f"invalid or duplicate source image_id at line {line_number}")
        source_by_image[image_id] = row
    if set(source_by_image) != image_ids:
        raise RuntimeError("source file does not contain exactly the four manifest images")

    project = Project.objects.get(pk=project_id)
    tasks = list(Task.objects.filter(project_id=project_id).order_by("data__image_id", "pk"))
    if {task.data.get("image_id") for task in tasks} != image_ids or len(tasks) != 4:
        raise RuntimeError("Project 3 task scope drifted; refusing to export")

    output_rows: list[dict] = []
    changed_images: list[int] = []
    for task in tasks:
        image_id = task.data.get("image_id")
        source_row = source_by_image[image_id]
        annotations = list(task.annotations.order_by("pk"))
        if len(annotations) != 1:
            raise RuntimeError(f"image {image_id} must have exactly one annotation")
        try:
            canonical = canonicalize_label_studio_draft(
                annotations[0].result or [],
                split="train",
                image_id=image_id,
                image_width=source_row["width"],
                image_height=source_row["height"],
            )
        except DraftContractError as exc:
            raise RuntimeError(f"image {image_id} annotation is outside the editable contract: {exc}") from exc

        source_objects = {int(obj["coco_ann_id"]): obj for obj in source_row["objects"]}
        edited_by_id: dict[int, dict] = {}
        for region in canonical.regions:
            annotation_id = region.get("coco_ann_id")
            if not isinstance(annotation_id, int) or annotation_id not in source_objects:
                raise RuntimeError(
                    f"image {image_id} contains a new or unknown coco_ann_id; only original boxes may be edited"
                )
            if annotation_id in edited_by_id:
                raise RuntimeError(f"image {image_id} contains duplicate coco_ann_id {annotation_id}")
            edited_by_id[annotation_id] = {
                "bbox_2d": list(region["bbox_2d"]),
                "desc": region["category_name"],
                "category_id": region["category_id"],
                "category_name": region["category_name"],
                "coco_ann_id": annotation_id,
            }

        edited_objects = [
            edited_by_id[int(original["coco_ann_id"])]
            for original in source_row["objects"]
            if int(original["coco_ann_id"]) in edited_by_id
        ]
        edited_row = dict(source_row)
        edited_row["objects"] = edited_objects
        output_rows.append(edited_row)
        if edited_objects != source_row["objects"]:
            changed_images.append(image_id)

    output_rows.sort(key=lambda row: row["image_id"])
    output_path = HERE / "working.norm.jsonl"
    payload = b"".join(_canonical_line(row) for row in output_rows)
    backup_path = HERE / "working.norm.jsonl.before-export"
    if output_path.exists():
        backup_path.write_bytes(output_path.read_bytes())
    fd, temporary = tempfile.mkstemp(prefix=".working.norm.jsonl.", dir=HERE)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    receipt = {
        "project_id": project_id,
        "image_ids": sorted(image_ids),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "changed_images": sorted(changed_images),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (HERE / "last_export.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
