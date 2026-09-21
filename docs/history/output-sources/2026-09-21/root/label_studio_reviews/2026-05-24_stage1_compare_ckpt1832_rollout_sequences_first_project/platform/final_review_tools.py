from __future__ import annotations

import json
import os
import importlib.util
import urllib.error
import urllib.request
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path("/data/CoordExp")
BUNDLE_ROOT = REPO_ROOT / "outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project"
PLATFORM_ROOT = BUNDLE_ROOT / "platform"
DEFAULT_VARIANT = "desc_first_t07"
IOU_THRESHOLD = 0.50
ASSET_URL_PREFIX = "http://127.0.0.1:18080/coordexp-assets/"

_PROMPT_VARIANTS_PATH = REPO_ROOT / "src/config/prompt_variants.py"
_PROMPT_VARIANTS_SPEC = importlib.util.spec_from_file_location(
    "coordexp_prompt_variants_for_label_studio", _PROMPT_VARIANTS_PATH
)
if _PROMPT_VARIANTS_SPEC is None or _PROMPT_VARIANTS_SPEC.loader is None:
    raise RuntimeError(f"Unable to load {_PROMPT_VARIANTS_PATH}")
_PROMPT_VARIANTS_MODULE = importlib.util.module_from_spec(_PROMPT_VARIANTS_SPEC)
sys.modules[_PROMPT_VARIANTS_SPEC.name] = _PROMPT_VARIANTS_MODULE
_PROMPT_VARIANTS_SPEC.loader.exec_module(_PROMPT_VARIANTS_MODULE)
COCO_80_CLASS_NAMES = _PROMPT_VARIANTS_MODULE.COCO_80_CLASS_NAMES

COCO_80_CLASS_SET = set(COCO_80_CLASS_NAMES)


@dataclass(frozen=True)
class MatchingResult:
    tp_pairs: list[tuple[int, int]]
    fp_pred_indices: list[int]
    fn_gt_indices: list[int]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_objects(gt_objects: list[dict], pred_objects: list[dict], iou_threshold: float = IOU_THRESHOLD) -> MatchingResult:
    candidates: list[tuple[float, int, int]] = []
    for gt_pos, gt in enumerate(gt_objects):
        for pred_pos, pred in enumerate(pred_objects):
            if gt.get("desc") != pred.get("desc"):
                continue
            iou = bbox_iou(gt["bbox_2d"], pred["bbox_2d"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_pos, pred_pos))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, gt_pos, pred_pos in sorted(candidates, key=lambda item: item[0], reverse=True):
        if gt_pos in matched_gt or pred_pos in matched_pred:
            continue
        matched_gt.add(gt_pos)
        matched_pred.add(pred_pos)
        pairs.append((gt_pos, pred_pos))

    return MatchingResult(
        tp_pairs=sorted(pairs),
        fp_pred_indices=[idx for idx in range(len(pred_objects)) if idx not in matched_pred],
        fn_gt_indices=[idx for idx in range(len(gt_objects)) if idx not in matched_gt],
    )


def percent_box_from_xyxy(bbox: list[float], width: int, height: int) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "x": x1 / width * 100.0,
        "y": y1 / height * 100.0,
        "width": (x2 - x1) / width * 100.0,
        "height": (y2 - y1) / height * 100.0,
        "rotation": 0,
    }


def xyxy_from_percent_box(value: dict, width: int, height: int) -> list[int]:
    x1 = value["x"] / 100.0 * width
    y1 = value["y"] / 100.0 * height
    x2 = (value["x"] + value["width"]) / 100.0 * width
    y2 = (value["y"] + value["height"]) / 100.0 * height
    return [round(x1), round(y1), round(x2), round(y2)]


def _taxonomy_result(region_id: str, from_name: str, to_name: str, desc: str, readonly: bool) -> dict | None:
    if desc not in COCO_80_CLASS_SET:
        return None
    return {
        "id": region_id,
        "from_name": from_name,
        "to_name": to_name,
        "type": "taxonomy",
        "value": {"taxonomy": [[desc]]},
        "readonly": readonly,
    }


def _rectangle_result(
    region_id: str,
    from_name: str,
    to_name: str,
    label: str,
    obj: dict,
    width: int,
    height: int,
    readonly: bool,
) -> dict:
    value = percent_box_from_xyxy(obj["bbox_2d"], width, height)
    value["rectanglelabels"] = [label]
    return {
        "id": region_id,
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "to_name": to_name,
        "from_name": from_name,
        "type": "rectanglelabels",
        "value": value,
        "readonly": readonly,
    }


def image_abs_to_same_origin_url(image_abs: str | Path) -> str:
    image_path = Path(image_abs).resolve()
    rel = image_path.relative_to(REPO_ROOT.resolve())
    return ASSET_URL_PREFIX + str(rel)


def build_final_review_task(
    source_task: dict,
    gt_objects: list[dict],
    pred_objects: list[dict],
    pred_variant: str = DEFAULT_VARIANT,
) -> dict:
    data = source_task["data"]
    width = int(data["width"])
    height = int(data["height"])
    image_url = image_abs_to_same_origin_url(data["image_abs"])
    matching = match_objects(gt_objects, pred_objects)
    tp_pred_indices = {pred_idx for _, pred_idx in matching.tp_pairs}

    results: list[dict] = []
    for gt_pos, gt in enumerate(gt_objects):
        region_id = f"final:gt:{gt_pos}"
        label = "final_object" if gt.get("desc") in COCO_80_CLASS_SET else "needs_review"
        results.append(_rectangle_result(region_id, "final_bbox", "final_image", label, gt, width, height, False))
        class_result = _taxonomy_result(region_id, "final_class", "final_image", gt.get("desc", ""), False)
        if class_result:
            results.append(class_result)

    for pred_pos in matching.fp_pred_indices:
        pred = pred_objects[pred_pos]
        label = "candidate_from_prediction" if pred.get("desc") in COCO_80_CLASS_SET else "needs_review"
        region_id = f"final:candidate:{pred_pos}"
        results.append(_rectangle_result(region_id, "final_bbox", "final_image", label, pred, width, height, False))
        class_result = _taxonomy_result(region_id, "final_class", "final_image", pred.get("desc", ""), False)
        if class_result:
            results.append(class_result)

    for pred_pos, pred in enumerate(pred_objects):
        label = "evidence_tp" if pred_pos in tp_pred_indices else "evidence_fp"
        region_id = f"evidence:pred:{pred_pos}"
        results.append(_rectangle_result(region_id, "evidence_bbox", "evidence_image", label, pred, width, height, True))
        class_result = _taxonomy_result(region_id, "evidence_class", "evidence_image", pred.get("desc", ""), True)
        if class_result:
            results.append(class_result)

    for gt_pos in matching.fn_gt_indices:
        gt = gt_objects[gt_pos]
        region_id = f"evidence:fn:{gt_pos}"
        results.append(_rectangle_result(region_id, "evidence_bbox", "evidence_image", "evidence_fn", gt, width, height, True))
        class_result = _taxonomy_result(region_id, "evidence_class", "evidence_image", gt.get("desc", ""), True)
        if class_result:
            results.append(class_result)

    return {
        "data": {
            "final_image": image_url,
            "evidence_image": image_url,
            "sample_id": data["line_idx"],
            "image_abs": data["image_abs"],
            "image_rel": data["image_rel"],
            "pred_variant": pred_variant,
            "width": width,
            "height": height,
            "gt_count": len(gt_objects),
            "pred_count": len(pred_objects),
            "tp_count": len(matching.tp_pairs),
            "fp_count": len(matching.fp_pred_indices),
            "fn_count": len(matching.fn_gt_indices),
            "candidate_fp_count": len(matching.fp_pred_indices),
            "iou_threshold": IOU_THRESHOLD,
        },
        "annotations": [{"result": results, "was_cancelled": False, "ground_truth": False}],
    }


def extract_objects_from_prediction(prediction: dict) -> list[dict]:
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for result in prediction.get("result", []):
        region_id = str(result["id"])
        if region_id not in grouped:
            grouped[region_id] = {"source_id": region_id}
            order.append(region_id)
        if result.get("type") == "rectanglelabels":
            value = result["value"]
            width = int(result["original_width"])
            height = int(result["original_height"])
            grouped[region_id]["bbox_2d"] = xyxy_from_percent_box(value, width, height)
        elif result.get("type") == "textarea":
            text = result.get("value", {}).get("text") or [""]
            grouped[region_id]["desc"] = text[0]

    objects: list[dict] = []
    for idx, region_id in enumerate(order):
        item = grouped[region_id]
        if "bbox_2d" not in item:
            continue
        item.setdefault("desc", "")
        item["index"] = idx
        objects.append(item)
    return objects


def find_prediction(source_task: dict, model_version: str) -> dict:
    for prediction in source_task.get("predictions", []):
        if prediction.get("model_version") == model_version:
            return prediction
    raise KeyError(f"Prediction model_version not found: {model_version}")


def build_tasks_from_source(source_tasks: list[dict], pred_variant: str = DEFAULT_VARIANT) -> tuple[list[dict], list[dict]]:
    tasks: list[dict] = []
    summaries: list[dict] = []
    for source_task in source_tasks:
        gt_objects = extract_objects_from_prediction(find_prediction(source_task, "source_gt_reference"))
        pred_objects = extract_objects_from_prediction(find_prediction(source_task, pred_variant))
        task = build_final_review_task(source_task, gt_objects, pred_objects, pred_variant)
        tasks.append(task)
        summaries.append(
            {
                "sample_id": task["data"]["sample_id"],
                "image_rel": task["data"]["image_rel"],
                "gt_count": task["data"]["gt_count"],
                "pred_count": task["data"]["pred_count"],
                "tp_count": task["data"]["tp_count"],
                "fp_count": task["data"]["fp_count"],
                "fn_count": task["data"]["fn_count"],
            }
        )
    return tasks, summaries


def generate_label_config() -> str:
    choices = "\n".join(f'    <Choice value="{name}"/>' for name in COCO_80_CLASS_NAMES)
    return f"""<View>
  <Header value="Final / Reviewed Annotation"/>
  <Image name="final_image" value="$final_image" zoom="true" zoomControl="true" rotateControl="false"/>

  <RectangleLabels name="final_bbox" toName="final_image" choice="single" showInline="true" opacity="0.35" strokeWidth="2">
    <Label value="final_object" background="#54A24B" hotkey=""/>
    <Label value="candidate_from_prediction" background="#E45756" hotkey=""/>
    <Label value="needs_review" background="#F2CF5B" hotkey=""/>
  </RectangleLabels>

  <Taxonomy name="final_class" toName="final_image" perRegion="true" leafsOnly="true" required="true" maxWidth="640px" placeholder="Search COCO class...">
{choices}
  </Taxonomy>

  <Header value="Prediction Evidence"/>
  <Image name="evidence_image" value="$evidence_image" zoom="true" zoomControl="true" rotateControl="false"/>

  <RectangleLabels name="evidence_bbox" toName="evidence_image" choice="single" showInline="true" opacity="0.35" strokeWidth="2">
    <Label value="evidence_tp" background="#54A24B" hotkey=""/>
    <Label value="evidence_fp" background="#E45756" hotkey=""/>
    <Label value="evidence_fn" background="#F2CF5B" hotkey=""/>
  </RectangleLabels>

  <Taxonomy name="evidence_class" toName="evidence_image" perRegion="true" leafsOnly="true" maxWidth="640px" placeholder="Evidence class">
{choices}
  </Taxonomy>
</View>
"""


def export_reviewed_objects_from_annotation(annotation: dict) -> tuple[list[dict], dict]:
    class_by_id: dict[str, str] = {}
    for result in annotation.get("result", []):
        if result.get("from_name") != "final_class" or result.get("type") != "taxonomy":
            continue
        taxonomy = result.get("value", {}).get("taxonomy") or []
        if taxonomy and taxonomy[0]:
            class_by_id[str(result["id"])] = taxonomy[0][-1]

    objects: list[dict] = []
    skipped_candidate_count = 0
    missing_class_count = 0
    skipped_nonfinal_count = 0
    for result in annotation.get("result", []):
        if result.get("from_name") != "final_bbox" or result.get("type") != "rectanglelabels":
            continue
        labels = result.get("value", {}).get("rectanglelabels") or []
        label = labels[0] if labels else ""
        if label == "candidate_from_prediction":
            skipped_candidate_count += 1
            continue
        if label != "final_object":
            skipped_nonfinal_count += 1
            continue
        desc = class_by_id.get(str(result["id"]))
        if desc not in COCO_80_CLASS_SET:
            missing_class_count += 1
            continue
        bbox = xyxy_from_percent_box(result["value"], int(result["original_width"]), int(result["original_height"]))
        objects.append({"bbox_2d": bbox, "desc": desc, "source": "reviewed_final"})

    return objects, {
        "exported_final_count": len(objects),
        "skipped_candidate_count": skipped_candidate_count,
        "missing_class_count": missing_class_count,
        "skipped_nonfinal_count": skipped_nonfinal_count,
    }


def read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def label_studio_request(method: str, url: str, token: str, payload: dict | list | None = None) -> dict | list | None:
    data = None
    headers = {"Authorization": f"Token {token}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Label Studio API {method} {url} failed: {exc.code} {detail}") from exc
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def iter_projects(projects_response: dict | list) -> Iterable[dict]:
    if isinstance(projects_response, dict):
        yield from projects_response.get("results", [])
    elif isinstance(projects_response, list):
        yield from projects_response


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
