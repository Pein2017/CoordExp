#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_review_tools import (
    BUNDLE_ROOT,
    DEFAULT_VARIANT,
    PLATFORM_ROOT,
    build_tasks_from_source,
    generate_label_config,
    iter_projects,
    label_studio_request,
    read_env_file,
    write_json,
)

PROJECT_TITLE = "CoordExp Final Review with Prediction Evidence"
PROJECT_DESCRIPTION = (
    "Final-review workflow: one editable/exportable Final annotation layer plus "
    "a read-only Prediction Evidence layer for desc_first_t07 TP/FP/FN audit."
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and import the CoordExp final-review Label Studio project."
    )
    parser.add_argument(
        "--source-tasks",
        type=Path,
        default=BUNDLE_ROOT / "import/tasks_http_static.json",
        help="Existing source tasks with GT and prediction preannotations.",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help="Prediction model_version to use as the evidence layer.",
    )
    parser.add_argument(
        "--title",
        default=PROJECT_TITLE,
        help="Label Studio project title.",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Create a new project even when the title already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = read_env_file(PLATFORM_ROOT / "label_studio_credentials.env")
    base_url = env["LABEL_STUDIO_URL"].rstrip("/")
    token = env["LABEL_STUDIO_API_KEY"]

    source_tasks = json.loads(args.source_tasks.read_text())
    final_tasks, matching_summary = build_tasks_from_source(source_tasks, args.variant)
    label_config = generate_label_config()

    label_config_path = BUNDLE_ROOT / "label_config_final_review.xml"
    tasks_path = BUNDLE_ROOT / "import/tasks_final_review.json"
    first_task_path = BUNDLE_ROOT / "import/first_task_final_review.json"
    matching_summary_path = BUNDLE_ROOT / "comparison/final_review_matching_summary.jsonl"
    payload_path = BUNDLE_ROOT / "project_payload_final_review.json"

    label_config_path.write_text(label_config, encoding="utf-8")
    write_json(tasks_path, final_tasks)
    write_json(first_task_path, final_tasks[0])
    write_jsonl(matching_summary_path, matching_summary)

    payload = {
        "title": args.title,
        "description": PROJECT_DESCRIPTION,
        "label_config": label_config,
        "show_skip_button": True,
        "enable_empty_annotation": True,
        "maximum_annotations": 1,
        "show_annotation_history": True,
        "show_collab_predictions": False,
    }
    write_json(payload_path, payload)

    projects_before = label_studio_request("GET", f"{base_url}/api/projects", token)
    write_json(PLATFORM_ROOT / "projects_before_final_review_import.json", projects_before or {})
    existing = [p for p in iter_projects(projects_before or []) if p.get("title") == args.title]
    if existing and not args.force_new:
        existing_ids = ", ".join(str(p["id"]) for p in existing)
        raise SystemExit(
            f"Project title already exists: {args.title!r} (ids: {existing_ids}). "
            "Use --force-new to create another independent project."
        )

    project = label_studio_request("POST", f"{base_url}/api/projects", token, payload)
    write_json(PLATFORM_ROOT / "project_create_final_review_response.json", project or {})
    project_id = int(project["id"])

    import_response = label_studio_request("POST", f"{base_url}/api/projects/{project_id}/import", token, final_tasks)
    write_json(PLATFORM_ROOT / "import_final_review_response.json", import_response or {})

    project_after = label_studio_request("GET", f"{base_url}/api/projects/{project_id}", token)
    write_json(PLATFORM_ROOT / "project_final_review_after_import.json", project_after or {})

    tasks_response = label_studio_request(
        "GET", f"{base_url}/api/projects/{project_id}/tasks?page_size=100", token
    )
    tasks = tasks_response if isinstance(tasks_response, list) else tasks_response.get("results", [])
    write_json(PLATFORM_ROOT / "project_final_review_tasks_sample.json", tasks[:1])
    task_id_by_sample_id = {str(task["data"]["sample_id"]): task["id"] for task in tasks}
    source_first_sample_id = str(final_tasks[0]["data"]["sample_id"]) if final_tasks else None
    first_task_id = task_id_by_sample_id.get(source_first_sample_id) if source_first_sample_id else None

    info = {
        "project_id": project_id,
        "project_title": args.title,
        "project_url": f"{base_url}/projects/{project_id}/data",
        "first_sample_task_url": f"{base_url}/projects/{project_id}/data?tab=1&task={first_task_id}",
        "source_first_sample_id": source_first_sample_id,
        "task_count": len(final_tasks),
        "annotation_count": (import_response or {}).get("annotation_count"),
        "prediction_variant": args.variant,
        "iou_threshold": final_tasks[0]["data"]["iou_threshold"] if final_tasks else None,
        "label_config": str(label_config_path),
        "import_file": str(tasks_path),
        "project_payload": str(payload_path),
        "matching_summary": str(matching_summary_path),
        "export_command": (
            f"python {PLATFORM_ROOT / 'export_final_review.py'} "
            f"--project-id {project_id}"
        ),
        "policy": (
            "One COCO sample per task. First image is the editable/exportable Final layer. "
            "Second image is read-only Prediction Evidence. Export reads only final_bbox + final_class."
        ),
        "task_id_by_sample_id": task_id_by_sample_id,
    }
    write_json(PLATFORM_ROOT / "final_review_project_info.json", info)

    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
