#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_review_tools import (
    BUNDLE_ROOT,
    PLATFORM_ROOT,
    export_reviewed_objects_from_annotation,
    label_studio_request,
    read_env_file,
    write_json,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export only the reviewed Final layer from a CoordExp Label Studio project."
    )
    parser.add_argument("--project-id", type=int, default=None, help="Label Studio project id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BUNDLE_ROOT / "export",
        help="Directory for reviewed JSONL and audit reports.",
    )
    return parser.parse_args()


def get_project_id(cli_project_id: int | None) -> int:
    if cli_project_id is not None:
        return cli_project_id
    info_path = PLATFORM_ROOT / "final_review_project_info.json"
    if not info_path.exists():
        raise SystemExit("Pass --project-id or create final_review_project_info.json first.")
    return int(json.loads(info_path.read_text())["project_id"])


def fetch_project_tasks(base_url: str, token: str, project_id: int) -> list[dict]:
    response = label_studio_request("GET", f"{base_url}/api/projects/{project_id}/tasks?page_size=100", token)
    if isinstance(response, list):
        return response
    return response.get("results", [])


def select_review_annotation(task: dict) -> dict | None:
    annotations = [ann for ann in task.get("annotations", []) if not ann.get("was_cancelled")]
    if not annotations:
        return None
    return sorted(annotations, key=lambda ann: (ann.get("updated_at") or "", ann.get("id") or 0))[-1]


def main() -> None:
    args = parse_args()
    project_id = get_project_id(args.project_id)
    env = read_env_file(PLATFORM_ROOT / "label_studio_credentials.env")
    base_url = env["LABEL_STUDIO_URL"].rstrip("/")
    token = env["LABEL_STUDIO_API_KEY"]

    tasks = fetch_project_tasks(base_url, token, project_id)
    reviewed_rows: list[dict] = []
    audit_rows: list[dict] = []

    for task in tasks:
        data = task["data"]
        annotation = select_review_annotation(task)
        if annotation is None:
            objects: list[dict] = []
            audit = {
                "exported_final_count": 0,
                "skipped_candidate_count": 0,
                "missing_class_count": 0,
                "skipped_nonfinal_count": 0,
                "missing_annotation": True,
            }
            annotation_id = None
        else:
            objects, audit = export_reviewed_objects_from_annotation(annotation)
            audit["missing_annotation"] = False
            annotation_id = annotation.get("id")

        reviewed_rows.append(
            {
                "sample_id": data["sample_id"],
                "image_rel": data["image_rel"],
                "image_abs": data["image_abs"],
                "width": data["width"],
                "height": data["height"],
                "objects": objects,
            }
        )
        audit_rows.append(
            {
                "sample_id": data["sample_id"],
                "task_id": task["id"],
                "annotation_id": annotation_id,
                "image_rel": data["image_rel"],
                "original_gt_count": data.get("gt_count"),
                "pred_count": data.get("pred_count"),
                "tp_count": data.get("tp_count"),
                "candidate_fp_count": data.get("candidate_fp_count"),
                "exported_final_count": audit["exported_final_count"],
                "skipped_candidate_count": audit["skipped_candidate_count"],
                "missing_class_count": audit["missing_class_count"],
                "skipped_nonfinal_count": audit["skipped_nonfinal_count"],
                "missing_annotation": audit["missing_annotation"],
            }
        )

    output_dir = args.output_dir
    reviewed_path = output_dir / "reviewed_final_objects.jsonl"
    audit_path = output_dir / "reviewed_final_audit.jsonl"
    summary_path = output_dir / "reviewed_final_summary.json"
    write_jsonl(reviewed_path, reviewed_rows)
    write_jsonl(audit_path, audit_rows)

    summary = {
        "project_id": project_id,
        "task_count": len(tasks),
        "sample_count": len(reviewed_rows),
        "exported_object_count": sum(row["exported_final_count"] for row in audit_rows),
        "skipped_candidate_count": sum(row["skipped_candidate_count"] for row in audit_rows),
        "missing_class_count": sum(row["missing_class_count"] for row in audit_rows),
        "missing_annotation_count": sum(1 for row in audit_rows if row["missing_annotation"]),
        "reviewed_jsonl": str(reviewed_path),
        "audit_jsonl": str(audit_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
