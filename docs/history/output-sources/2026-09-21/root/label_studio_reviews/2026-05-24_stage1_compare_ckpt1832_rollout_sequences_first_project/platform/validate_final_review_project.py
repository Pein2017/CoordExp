#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from final_review_tools import (
    ASSET_URL_PREFIX,
    BUNDLE_ROOT,
    COCO_80_CLASS_NAMES,
    PLATFORM_ROOT,
    label_studio_request,
    read_env_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated/imported final-review Label Studio assets.")
    parser.add_argument(
        "--label-config",
        type=Path,
        default=BUNDLE_ROOT / "label_config_final_review.xml",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=BUNDLE_ROOT / "import/tasks_final_review.json",
    )
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=PLATFORM_ROOT / "final_review_validation_report.json",
    )
    parser.add_argument(
        "--check-image-http",
        action="store_true",
        help="Fetch the first task image URL and require HTTP 200.",
    )
    return parser.parse_args()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def label_values(root: ET.Element, control_name: str) -> dict[str, str]:
    for control in root.iter("RectangleLabels"):
        if control.attrib.get("name") != control_name:
            continue
        return {label.attrib["value"]: label.attrib.get("background", "") for label in control.iter("Label")}
    return {}


def taxonomy_choices(root: ET.Element, taxonomy_name: str) -> list[str]:
    for taxonomy in root.iter("Taxonomy"):
        if taxonomy.attrib.get("name") != taxonomy_name:
            continue
        return [choice.attrib["value"] for choice in taxonomy.iter("Choice")]
    return []


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    label_config = args.label_config.read_text(encoding="utf-8")
    root = ET.fromstring(label_config)

    require("rollout_sequences" not in label_config, "label config still references rollout_sequences", failures)
    for forbidden in ["PolygonLabels", "KeyPointLabels", "BrushLabels", "HyperText", "TextArea"]:
        require(f"<{forbidden}" not in label_config, f"forbidden tag present: {forbidden}", failures)
    require(not re.search(r'hotkey="[0-9]"', label_config), "numeric hotkey is assigned", failures)

    images = [node.attrib.get("name") for node in root.iter("Image")]
    require(images == ["final_image", "evidence_image"], f"expected two vertical images, got {images}", failures)

    final_labels = label_values(root, "final_bbox")
    evidence_labels = label_values(root, "evidence_bbox")
    require(final_labels.get("final_object") == "#54A24B", "final_object color mismatch", failures)
    require(final_labels.get("candidate_from_prediction") == "#E45756", "candidate_from_prediction color mismatch", failures)
    require(final_labels.get("needs_review") == "#F2CF5B", "needs_review color mismatch", failures)
    require(evidence_labels.get("evidence_tp") == "#54A24B", "evidence_tp color mismatch", failures)
    require(evidence_labels.get("evidence_fp") == "#E45756", "evidence_fp color mismatch", failures)
    require(evidence_labels.get("evidence_fn") == "#F2CF5B", "evidence_fn color mismatch", failures)

    final_choices = taxonomy_choices(root, "final_class")
    require(final_choices == list(COCO_80_CLASS_NAMES), "final_class taxonomy does not match COCO_80_CLASS_NAMES", failures)
    require(len(final_choices) == 80, f"final_class taxonomy count is {len(final_choices)}", failures)
    require(len(set(final_choices)) == 80, "final_class taxonomy has duplicate classes", failures)

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    for task_idx, task in enumerate(tasks):
        data = task["data"]
        require(data["final_image"].startswith(ASSET_URL_PREFIX), f"task {task_idx} final_image is not same-origin", failures)
        require(data["evidence_image"].startswith(ASSET_URL_PREFIX), f"task {task_idx} evidence_image is not same-origin", failures)
        require(data["final_image"] == data["evidence_image"], f"task {task_idx} image URLs differ", failures)
        require(Path(data["image_abs"]).exists(), f"task {task_idx} image_abs missing: {data['image_abs']}", failures)
        require(data["image_rel"].startswith("images/"), f"task {task_idx} image_rel not preserved", failures)

        results = task["annotations"][0]["result"]
        evidence_boxes = [r for r in results if r.get("from_name") == "evidence_bbox"]
        final_boxes = [r for r in results if r.get("from_name") == "final_bbox"]
        evidence_count = {"evidence_tp": 0, "evidence_fp": 0, "evidence_fn": 0}
        for box in evidence_boxes:
            labels = box.get("value", {}).get("rectanglelabels") or []
            if labels:
                evidence_count[labels[0]] = evidence_count.get(labels[0], 0) + 1
            require(box.get("readonly") is True, f"task {task_idx} evidence box is editable", failures)
        for box in final_boxes:
            require(box.get("readonly") is False, f"task {task_idx} final box is readonly", failures)
        require(evidence_count["evidence_tp"] + evidence_count["evidence_fp"] == data["pred_count"],
                f"task {task_idx} TP+FP != pred_count", failures)
        require(evidence_count["evidence_tp"] + evidence_count["evidence_fn"] == data["gt_count"],
                f"task {task_idx} TP+FN != gt_count", failures)

    project_report = None
    if args.project_id is not None:
        env = read_env_file(PLATFORM_ROOT / "label_studio_credentials.env")
        base_url = env["LABEL_STUDIO_URL"].rstrip("/")
        token = env["LABEL_STUDIO_API_KEY"]
        project = label_studio_request("GET", f"{base_url}/api/projects/{args.project_id}", token)
        project_report = {
            "id": project["id"],
            "title": project["title"],
            "task_number": project.get("task_number"),
            "total_annotations_number": project.get("total_annotations_number"),
        }
        require(
            project["label_config"].strip() == label_config.strip(),
            "imported project label_config differs from generated file",
            failures,
        )

    image_http_status = None
    if args.check_image_http and tasks:
        with urllib.request.urlopen(tasks[0]["data"]["final_image"], timeout=10) as response:
            image_http_status = response.status
        require(image_http_status == 200, f"first image returned HTTP {image_http_status}", failures)

    report = {
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "task_count": len(tasks),
        "coco80_count": len(final_choices),
        "first_image_http_status": image_http_status,
        "project": project_report,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
