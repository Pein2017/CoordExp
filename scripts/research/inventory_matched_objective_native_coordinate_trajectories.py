#!/usr/bin/env python3
"""Inventory matched native coordinate trajectories from two objective sources.

This is an analysis-only helper for the six fixed-prompt cases.  It joins the
new pure cross-entropy sampled artifacts with the historical Gaussian/RPS
full-image independent-bagging calls, without attempting to decide whether
two rows describe the same physical instance.  Rows unmatched to official
ground truth are retained as candidates and are never called hallucinations.

The source policy is deliberately strict: sixteen Full-Image K-Rollout
Independent Bagging calls per image, temperature 0.4, top-p 0.95,
repetition penalty 1.0, and a 512-token generation limit.  A mismatch is a
hard error because silently mixing policies would make trajectory comparison
uninterpretable.

Here ``FULL_BAG_K`` names the existing Full-Image Independent Bagging arm with
K equal to 16 calls.  ``IoU`` means intersection over union and is reported
only as a geometric comparison to the reviewed reference box; it is not an
identity or ground-truth decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


FULL_BAG_K = "FULL_BAG_K"
EXPECTED_CALLS = 16
EXPECTED_POLICY = {
    "temperature": 0.4,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}
SCHEMA_VERSION = "matched_objective_native_coordinate_trajectory_inventory.v1"
PURE_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
REVIEW_QUEUE_LIMIT = 512


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _float(value: Any) -> float:
    return float(value)


def _policy_from_config(config: Mapping[str, Any]) -> dict[str, float | int]:
    values = {
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "repetition_penalty": config.get("repetition_penalty"),
        "max_new_tokens": config.get("max_new_tokens"),
    }
    for key, expected in EXPECTED_POLICY.items():
        actual = values[key]
        if actual is None or float(actual) != float(expected):
            raise ValueError(f"sampling policy {key}={actual!r} differs from expected {expected!r}")
    return {key: (int(value) if key == "max_new_tokens" else float(value)) for key, value in values.items()}


def _case_dimensions(case: Mapping[str, Any], *, fallback: Mapping[str, Any] | None = None) -> tuple[int, int]:
    dimensions = case.get("image_dimensions")
    if isinstance(dimensions, Mapping):
        width, height = dimensions.get("width"), dimensions.get("height")
    else:
        width, height = case.get("width"), case.get("height")
    if (width is None or height is None) and fallback is not None:
        width, height = fallback.get("width"), fallback.get("height")
    if width is None or height is None:
        image_path = case.get("image_path")
        if image_path:
            try:
                from PIL import Image

                with Image.open(Path(str(image_path)).expanduser().resolve(strict=True)) as image:
                    width, height = image.size
            except Exception as exc:  # pragma: no cover - only used without metadata
                raise ValueError(f"could not determine dimensions for case {case.get('name')}") from exc
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return width, height


def _reference_bins(case: Mapping[str, Any], width: int, height: int) -> list[int]:
    supplied = case.get("reference_box_xyxy_coordinate_bins")
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)) and len(supplied) == 4:
        return [int(value) for value in supplied]
    box = case.get("reference_box_xyxy")
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        raise ValueError(f"case {case.get('name')} lacks reference_box_xyxy")
    return [
        max(0, min(999, int(round(float(value) * 1000.0 / (width if index % 2 == 0 else height)))))
        for index, value in enumerate(box)
    ]


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != 4 or len(right) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in left)
    a1, b1, a2, b2 = (float(value) for value in right)
    if not all(math.isfinite(value) for value in (*left, *right)):
        return None
    iw, ih = max(0.0, min(x2, a2) - max(x1, a1)), max(0.0, min(y2, b2) - max(y1, b1))
    intersection = iw * ih
    area_left = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_right = max(0.0, a2 - a1) * max(0.0, b2 - b1)
    union = area_left + area_right - intersection
    return None if union <= 0 else intersection / union


def _center_distance(left: Sequence[float], right: Sequence[float], width: int, height: int) -> float | None:
    if len(left) != 4 or len(right) != 4:
        return None
    try:
        lx, ly = (float(left[0]) + float(left[2])) / 2.0, (float(left[1]) + float(left[3])) / 2.0
        rx, ry = (float(right[0]) + float(right[2])) / 2.0, (float(right[1]) + float(right[3])) / 2.0
    except (TypeError, ValueError):
        return None
    return math.hypot((lx - rx) / width, (ly - ry) / height)


def _candidate_record(
    raw: Mapping[str, Any],
    *,
    source: str,
    image_id: str,
    trajectory_index: int,
    width: int,
    height: int,
    reference_box: Sequence[float],
    reference_bins: Sequence[int],
    category: str,
    valid: bool,
    dropped: bool = False,
) -> dict[str, Any] | None:
    raw_category = raw.get("description", raw.get("category_text", raw.get("normalized_category_name", "")))
    normalized = str(raw_category or "").strip().lower()
    if normalized != category:
        return None
    bins_value = raw.get("coord_bins", raw.get("coordinate_bins"))
    bins = None
    if isinstance(bins_value, Sequence) and not isinstance(bins_value, (str, bytes)) and len(bins_value) == 4:
        try:
            bins = [int(value) for value in bins_value]
        except (TypeError, ValueError):
            bins = None
    box_value = raw.get("bbox", raw.get("parsed_bbox_xyxy"))
    box = None
    if isinstance(box_value, Sequence) and not isinstance(box_value, (str, bytes)) and len(box_value) == 4:
        try:
            box = [float(value) for value in box_value]
        except (TypeError, ValueError):
            box = None
    return {
        "candidate_id": str(raw.get("object_span_id") or f"{source}:{image_id}:trajectory-{trajectory_index}:row-{raw.get('generated_order', raw.get('generated_row_index', 'unknown'))}"),
        "source": source,
        "trajectory_index": int(trajectory_index),
        "generated_order": raw.get("generated_order", raw.get("generated_row_index")),
        "category": normalized,
        "coord_bins": bins,
        "pixel_box_xyxy": box,
        "valid": bool(valid),
        "dropped": bool(dropped),
        "reference_iou": None if box is None else _box_iou(box, reference_box),
        "normalized_center_distance": None if box is None else _center_distance(box, reference_box, width, height),
        "per_boundary_bin_error": None if bins is None else [int(bins[index]) - int(reference_bins[index]) for index in range(4)],
        "per_boundary_bin_absolute_error": None if bins is None else [abs(int(bins[index]) - int(reference_bins[index])) for index in range(4)],
    }


def _pure_predictions(rollout: Mapping[str, Any], *, source: str, case: Mapping[str, Any], trajectory_index: int, width: int, height: int, reference_box: Sequence[float], reference_bins: Sequence[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parsed = _mapping(rollout.get("predictions"), "pure rollout predictions")
    category = str(case["category"]).strip().lower()
    accepted: list[dict[str, Any]] = []
    for prediction in parsed.get("predictions", []):
        if isinstance(prediction, Mapping):
            record = _candidate_record(prediction, source=source, image_id=str(case["image_id"]), trajectory_index=trajectory_index, width=width, height=height, reference_box=reference_box, reference_bins=reference_bins, category=category, valid=True)
            if record is not None:
                accepted.append(record)
    dropped: list[dict[str, Any]] = []
    for prediction in parsed.get("dropped_predictions", []):
        if isinstance(prediction, Mapping):
            record = _candidate_record(prediction, source=source, image_id=str(case["image_id"]), trajectory_index=trajectory_index, width=width, height=height, reference_box=reference_box, reference_bins=reference_bins, category=category, valid=False, dropped=True)
            if record is not None:
                dropped.append(record)
    return accepted, dropped


def _bundle_predictions(bundle: Mapping[str, Any], *, source: str, case: Mapping[str, Any], trajectory_index: int, width: int, height: int, reference_box: Sequence[float], reference_bins: Sequence[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category = str(case["category"]).strip().lower()
    accepted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    receipts = bundle.get("parse_score_receipts", [])
    if not isinstance(receipts, list):
        raise ValueError("Gaussian call parse_score_receipts must be a list")
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            continue
        valid = str(receipt.get("parse_status", "")) == "accepted" and str(receipt.get("prediction_validity", "")).startswith("accepted")
        record = _candidate_record(receipt, source=source, image_id=str(case["image_id"]), trajectory_index=trajectory_index, width=width, height=height, reference_box=reference_box, reference_bins=reference_bins, category=category, valid=valid, dropped=not valid)
        if record is not None:
            (accepted if valid else dropped).append(record)
    return accepted, dropped


def _load_pure(path: Path, *, case: Mapping[str, Any]) -> dict[str, Any]:
    document = _mapping(_read_json(path), f"pure artifact {path}")
    if document.get("schema_version") != PURE_SCHEMA_VERSION:
        raise ValueError(f"pure artifact {path} has unexpected schema")
    config = _mapping(document.get("config"), "pure config")
    policy = _policy_from_config(config)
    rollouts = document.get("rollouts")
    if document.get("rollout_count") != EXPECTED_CALLS or not isinstance(rollouts, list) or len(rollouts) != EXPECTED_CALLS:
        raise ValueError(f"pure artifact {path} must contain exactly {EXPECTED_CALLS} rollouts")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != EXPECTED_CALLS or len(set(int(value) for value in seeds)) != EXPECTED_CALLS:
        raise ValueError(f"pure artifact {path} must contain {EXPECTED_CALLS} unique seeds")
    image_id = str(case["image_id"])
    metadata = _mapping(_mapping(document.get("prompt_metadata"), "prompt_metadata").get(f"coco2017_val_{int(image_id):012d}"), "prompt image metadata")
    return {"document": document, "policy": policy, "rollouts": rollouts, "seeds": [int(value) for value in seeds], "metadata": metadata}


def _schedule_requests(schedule: Mapping[str, Any], *, image_id: str) -> list[dict[str, Any]]:
    requests = _mapping(schedule.get("schedule"), "schedule").get("requests")
    if not isinstance(requests, list):
        raise ValueError("schedule.schedule.requests must be a list")
    selected = [dict(_mapping(request, "schedule request")) for request in requests if str(request.get("image_id")) == image_id and _mapping(request.get("arm"), "request arm").get("arm_code") == FULL_BAG_K]
    if len(selected) != EXPECTED_CALLS:
        raise ValueError(f"schedule must contain exactly {EXPECTED_CALLS} FULL_BAG_K requests for image {image_id}, found {len(selected)}")
    selected.sort(key=lambda request: int(request.get("cell_index", -1)))
    if [int(request.get("cell_index", -1)) for request in selected] != list(range(EXPECTED_CALLS)):
        raise ValueError(f"schedule FULL_BAG_K cells for image {image_id} are not exactly 0..15")
    seeds = [int(request.get("sampling_seed")) for request in selected]
    if len(set(seeds)) != EXPECTED_CALLS:
        raise ValueError(f"schedule FULL_BAG_K seeds for image {image_id} are not unique")
    for request in selected:
        arm = _mapping(request.get("arm"), "request arm")
        if int(arm.get("calls_per_image", -1)) != EXPECTED_CALLS or arm.get("history_policy") != "fresh_base_prompt_per_call":
            raise ValueError("schedule FULL_BAG_K request has an incompatible arm")
    return selected


def _bundle_index(calls_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(calls_root.expanduser().resolve(strict=True).rglob("terminal-output-bundle.json")):
        try:
            bundle = _mapping(_read_json(path), str(path))
        except ValueError:
            continue
        request_id = bundle.get("request_id")
        if request_id:
            if str(request_id) in index and index[str(request_id)] != path:
                raise ValueError(f"duplicate bundle request id {request_id}")
            index[str(request_id)] = path
    return index


def _validate_bundle(bundle: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    scheduled = _mapping(bundle.get("scheduled_request"), "bundle scheduled request")
    evidence = _mapping(bundle.get("execution_evidence"), "bundle execution evidence")
    anchor = _mapping(_mapping(bundle.get("decode_result"), "bundle decode result").get("execution_contract_anchor"), "decode execution contract anchor")
    policy = _mapping(anchor.get("decode_generation_policy"), "bundle generation policy")
    for key, expected in EXPECTED_POLICY.items():
        actual = policy.get(key)
        if actual is None or float(actual) != float(expected):
            raise ValueError(f"Gaussian bundle policy {key}={actual!r} differs from expected {expected!r}")
    if str(bundle.get("request_id")) != str(request.get("request_id")) or str(scheduled.get("request_id")) != str(request.get("request_id")):
        raise ValueError("bundle and schedule request ids differ")
    if int(evidence.get("image_id")) != int(request.get("image_id")) or int(evidence.get("sampling_seed")) != int(request.get("sampling_seed")):
        raise ValueError("bundle and schedule image/seed differ")
    if _mapping(evidence.get("arm"), "bundle arm").get("arm_code") != FULL_BAG_K:
        raise ValueError("bundle is not FULL_BAG_K")


def _review_queue(pure: Sequence[Mapping[str, Any]], gaussian: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left in pure:
        for right in gaussian:
            left_iou = left.get("reference_iou")
            right_iou = right.get("reference_iou")
            if left.get("coord_bins") is not None and right.get("coord_bins") is not None:
                coordinate_l1 = sum(abs(int(a) - int(b)) for a, b in zip(left["coord_bins"], right["coord_bins"])) / 4000.0
            else:
                coordinate_l1 = None
            pairs.append({
                "admission": "non_admitting_review_queue_only",
                "pure_candidate_id": left.get("candidate_id"),
                "gaussian_candidate_id": right.get("candidate_id"),
                "reference_proximity": None if left_iou is None or right_iou is None else min(float(left_iou), float(right_iou)),
                "mean_reference_iou": None if left_iou is None or right_iou is None else (float(left_iou) + float(right_iou)) / 2.0,
                "coordinate_l1_distance_normalized": coordinate_l1,
                "note": "This queue does not infer same-owner identity, ground truth, or hallucination status.",
            })
    pairs.sort(key=lambda pair: (-float(pair["reference_proximity"] if pair["reference_proximity"] is not None else -1.0), float(pair["coordinate_l1_distance_normalized"] if pair["coordinate_l1_distance_normalized"] is not None else math.inf)))
    for index, pair in enumerate(pairs, start=1):
        pair["review_rank"] = index
    return pairs


def build_inventory(*, cases_path: Path, pure_artifact_root: Path, gaussian_schedule_path: Path, gaussian_calls_root: Path) -> dict[str, Any]:
    cases_payload = _mapping(_read_json(cases_path), "cases")
    cases = cases_payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases document must contain a non-empty cases list")
    schedule = _mapping(_read_json(gaussian_schedule_path), "Gaussian schedule")
    bundle_index = _bundle_index(gaussian_calls_root)
    result_cases: list[dict[str, Any]] = []
    for case_value in cases:
        case = _mapping(case_value, "case")
        image_id = str(case.get("image_id"))
        pure_path = pure_artifact_root.expanduser().resolve(strict=True) / f"image-{image_id}.json"
        pure = _load_pure(pure_path, case=case)
        fallback = {"width": pure["metadata"].get("width"), "height": pure["metadata"].get("height")}
        width, height = _case_dimensions(case, fallback=fallback)
        reference_box = [float(value) for value in case["reference_box_xyxy"]]
        reference_bins = _reference_bins(case, width, height)
        requests = _schedule_requests(schedule, image_id=image_id)
        schedule_seeds = [int(request["sampling_seed"]) for request in requests]
        if pure["seeds"] != schedule_seeds:
            raise ValueError(f"pure and Gaussian seed order differs for image {image_id}")
        pure_trajectories: list[dict[str, Any]] = []
        gaussian_trajectories: list[dict[str, Any]] = []
        for cell_index, (rollout, request) in enumerate(zip(pure["rollouts"], requests)):
            if int(rollout.get("seed")) != int(request["sampling_seed"]):
                raise ValueError(f"pure rollout seed differs at image {image_id}, cell {cell_index}")
            accepted, dropped = _pure_predictions(rollout, source="pure_cross_entropy", case=case, trajectory_index=cell_index, width=width, height=height, reference_box=reference_box, reference_bins=reference_bins)
            parsed = _mapping(rollout.get("predictions"), "pure predictions")
            pure_trajectories.append({
                "cell_index": cell_index,
                "seed": int(rollout["seed"]),
                "stop_reason": rollout.get("stop_reason"),
                "validity_counts": {"all_valid_predictions": int(parsed.get("valid_prediction_count", len(parsed.get("predictions", [])))), "relevant_category_valid": len(accepted), "relevant_category_dropped": len(dropped), "dropped_prediction_count": int(parsed.get("dropped_prediction_count", len(parsed.get("dropped_predictions", [])))), "stop_count": int(bool(rollout.get("stop_reason")))},
                "predictions": accepted,
                "dropped_predictions": dropped,
                "source_artifact": {"path": str(pure_path), "rollout_index": cell_index, "schema_version": PURE_SCHEMA_VERSION},
            })
            request_id = str(request.get("request_id"))
            bundle_path = bundle_index.get(request_id)
            if bundle_path is None:
                raise ValueError(f"no Gaussian bundle found for request {request_id}")
            bundle = _mapping(_read_json(bundle_path), str(bundle_path))
            _validate_bundle(bundle, request)
            accepted_g, dropped_g = _bundle_predictions(bundle, source="gaussian_ranked_probability_score", case=case, trajectory_index=cell_index, width=width, height=height, reference_box=reference_box, reference_bins=reference_bins)
            diagnostics = _mapping(bundle.get("call_diagnostics"), "Gaussian call diagnostics")
            gaussian_trajectories.append({
                "cell_index": cell_index,
                "seed": int(request["sampling_seed"]),
                "stop_reason": bundle.get("stop_reason"),
                "validity_counts": {"all_valid_predictions": int(diagnostics.get("valid_prediction_count", len(bundle.get("parse_score_receipts", [])))), "relevant_category_valid": len(accepted_g), "relevant_category_dropped": len(dropped_g), "dropped_prediction_count": len(dropped_g), "invalid_row_count": int(diagnostics.get("invalid_row_count", 0)), "malformed_row_count": int(diagnostics.get("malformed_row_count", 0)), "stop_count": int(bool(bundle.get("stop_reason")))},
                "predictions": accepted_g,
                "dropped_predictions": dropped_g,
                "source_artifact": {"path": str(bundle_path), "request_id": request_id, "schedule_index": request.get("schedule_index"), "schema_version": bundle.get("schema_version")},
            })
        review_queue = _review_queue([candidate for trajectory in pure_trajectories for candidate in trajectory["predictions"]], [candidate for trajectory in gaussian_trajectories for candidate in trajectory["predictions"]])
        source_totals = {}
        for source_name, trajectories in (("pure_cross_entropy", pure_trajectories), ("gaussian_ranked_probability_score", gaussian_trajectories)):
            source_totals[source_name] = {
                "trajectory_count": len(trajectories),
                "relevant_category_valid": sum(int(item["validity_counts"]["relevant_category_valid"]) for item in trajectories),
                "relevant_category_dropped": sum(int(item["validity_counts"]["relevant_category_dropped"]) for item in trajectories),
                "stop_count": sum(int(item["validity_counts"]["stop_count"]) for item in trajectories),
            }
        result_cases.append({
            "case_name": case.get("name"),
            "image_id": image_id,
            "category": str(case.get("category")).strip().lower(),
            "image_dimensions": {"width": width, "height": height},
            "reference_object_identifier": case.get("reference_object_identifier"),
            "reference_box_xyxy": reference_box,
            "reference_coordinate_bins": reference_bins,
            "denominators": {"trajectory_count_per_source": EXPECTED_CALLS, "matched_seed_count": EXPECTED_CALLS, "coordinate_bin_range": [0, 999]},
            "source_totals": source_totals,
            "trajectories": {"pure_cross_entropy": pure_trajectories, "gaussian_ranked_probability_score": gaussian_trajectories},
            "candidate_pair_review_queue": review_queue[:REVIEW_QUEUE_LIMIT],
            "candidate_pair_review_queue_total": len(review_queue),
            "candidate_pair_review_queue_truncated": len(review_queue) > REVIEW_QUEUE_LIMIT,
        })
    return {"schema_version": SCHEMA_VERSION, "policy": dict(EXPECTED_POLICY), "case_count": len(result_cases), "cases": result_cases, "provenance": {"cases_path": str(cases_path.resolve()), "pure_artifact_root": str(pure_artifact_root.resolve()), "gaussian_schedule_path": str(gaussian_schedule_path.resolve()), "gaussian_calls_root": str(gaussian_calls_root.resolve())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pure-artifact-root", type=Path, required=True)
    parser.add_argument("--gaussian-schedule", type=Path, required=True)
    parser.add_argument("--gaussian-calls-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"output exists; use --force: {output}")
    inventory = build_inventory(cases_path=args.cases, pure_artifact_root=args.pure_artifact_root, gaussian_schedule_path=args.gaussian_schedule, gaussian_calls_root=args.gaussian_calls_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "case_count": inventory["case_count"], "schema_version": SCHEMA_VERSION}, sort_keys=True))


if __name__ == "__main__":
    main()
