"""Frozen saved-row accounting, independent of experiment orchestration.

Matching is class-agnostic cardinality-first at the caller's threshold. Class
correctness, geometric validity, literal recurrence and IoU recurrence remain
separate. Annotation-unmatched rows never become false physical objects here.
"""
from __future__ import annotations

import collections
import re
from typing import Any, Mapping, Sequence

from src.eval.assignment import global_matches
from src.inference.parsing import parse_compact_object_box_closed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == 4 and len(right) == 4, "box must have four coordinates")
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    iw, ih = max(0.0, min(lx2, rx2) - max(lx1, rx1)), max(0.0, min(ly2, ry2) - max(ly1, ry1))
    inter = iw * ih
    union = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1) + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1) - inter
    return inter / union if union else 0.0

def one_to_one_matches(references: list[Mapping[str, Any]], predictions: list[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Class-agnostic cardinality-first matching on normalized bins."""
    gt = [("*", tuple(ref["reference_coord_bins_1000"])) for ref in references]
    pred = [("*", tuple(item["coord_bins_1000"])) for item in predictions]
    matches = global_matches(gt, pred, threshold)
    return [{"reference_owner_id": references[gi]["owner_id"], "reference_index": gi,
             "prediction_id": predictions[pi]["prediction_id"], "prediction_order": predictions[pi]["generated_order"],
             "iou": overlap} for gi, pi, overlap in matches]

def pairwise_iou95(predictions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            overlap = iou_xyxy(predictions[left]["coord_bins_1000"], predictions[right]["coord_bins_1000"])
            if overlap > 0.95:
                pairs.append({"left_prediction_id": predictions[left]["prediction_id"],
                              "right_prediction_id": predictions[right]["prediction_id"],
                              "left_order": predictions[left]["generated_order"],
                              "right_order": predictions[right]["generated_order"], "iou": overlap})
    return pairs

def _drop_bins(drop: Mapping[str, Any]) -> list[int] | None:
    context = drop.get("context")
    if isinstance(context, Mapping) and isinstance(context.get("bbox"), list):
        return list(context["bbox"])
    texts = [str(span.get("text", "")) for span in drop.get("coord_token_spans", []) if isinstance(span, Mapping)]
    values = [int(match.group(1)) for text in texts for match in [re.fullmatch(r"<\|coord_(\d+)\|>", text)] if match]
    return values if len(values) == 4 else None

def flatten_raw_rows(parsed: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = []
    for item in parsed.get("pred", []):
        bins = item.get("coord_bins")
        valid.append({"prediction_id": f"p{item.get('generated_order')}", "generated_order": item.get("generated_order"),
                      "status": "parsed_valid", "description": item.get("description"),
                      "coord_bins_1000": list(bins) if isinstance(bins, list) else None,
                      "bbox_pixel_xyxy": item.get("bbox"), "raw_span_sha256": item.get("raw_span_sha256"), "raw": item})
    dropped = []
    for item in parsed.get("dropped_predictions", []):
        dropped.append({"prediction_id": f"p{item.get('generated_order')}", "generated_order": item.get("generated_order"),
                        "status": "parser_dropped", "drop_reason": item.get("reason"), "drop_code": item.get("code"),
                        "description": item.get("context", {}).get("description") if isinstance(item.get("context"), Mapping) else None,
                        "coord_bins_1000": _drop_bins(item), "raw_span_sha256": item.get("raw_span_sha256"), "raw": item})
    return valid, dropped

def class_correct(*, target: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool | None:
    if not str(target.get("class_status", "")).startswith("verified_"):
        return None
    return prediction.get("description") == target.get("description")

def ledger_image(
    targets: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *, threshold: float
) -> dict[str, Any]:
    matches = one_to_one_matches(list(targets), list(predictions), threshold)
    target_by_owner = {str(item["owner_id"]): item for item in targets}
    prediction_by_id = {str(item["prediction_id"]): item for item in predictions}
    matched_owner_ids = {str(item["reference_owner_id"]) for item in matches}
    matched_prediction_ids = {str(item["prediction_id"]) for item in matches}
    class_values = [
        class_correct(target=target_by_owner[str(item["reference_owner_id"])], prediction=prediction_by_id[str(item["prediction_id"])])
        for item in matches
    ]
    return {
        "target_count": len(targets),
        "matched_count": len(matches),
        "fn_count": len(targets) - len(matches),
        "fn_rate": (len(targets) - len(matches)) / len(targets) if targets else 0.0,
        "covered_owner_ids": [str(item["reference_owner_id"]) for item in matches],
        "missing_owner_ids": [str(item["owner_id"]) for item in targets if str(item["owner_id"]) not in matched_owner_ids],
        "matched_prediction_ids": [str(item["prediction_id"]) for item in matches],
        "annotation_unmatched_prediction_ids": [str(item["prediction_id"]) for item in predictions if str(item["prediction_id"]) not in matched_prediction_ids],
        "matches": matches,
        "class_correct_count": sum(value is True for value in class_values),
        "class_wrong_count": sum(value is False for value in class_values),
        "class_unknown_count": sum(value is None for value in class_values),
    }

def matchable_rows_with_geometry_debt(parsed: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every native-parser row visible before class-agnostic matching."""

    valid, dropped = flatten_raw_rows(parsed)
    matchable = []
    for row in valid:
        bins = row.get("coord_bins_1000")
        geometry_ok = (
            isinstance(bins, list)
            and len(bins) == 4
            and all(type(value) is int for value in bins)
            and 0 <= bins[0] < bins[2] <= 999
            and 0 <= bins[1] < bins[3] <= 999
        )
        if geometry_ok:
            matchable.append(row)
        else:
            dropped.append(
                {
                    **row,
                    "status": "consumer_geometry_invalid",
                    "drop_reason": "consumer_geometry_invalid_after_native_parse",
                    "drop_code": "evaluation.geometry_invalid",
                }
            )
    return matchable, dropped

def strict_repeat_rows(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Count each later valid row once when any earlier row has IoU strictly >.95."""

    repeats = []
    for index, row in enumerate(predictions):
        overlaps = [
            iou_xyxy(
                row["coord_bins_1000"], previous["coord_bins_1000"]
            )
            for previous in predictions[:index]
        ]
        best = max(overlaps, default=0.0)
        if best > 0.95:
            repeats.append(
                {
                    "prediction_id": str(row["prediction_id"]),
                    "generated_order": int(row["generated_order"]),
                    "best_earlier_iou": best,
                }
            )
    return repeats

PAT = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>"
    + r"<\|coord_(\d+)\|>" * 4
    + r"<\|box_end\|>"
)

def score(raw, case, bank):
    """Return the frozen known-owner, validity, and recurrence accounting."""
    native = parse_compact_object_box_closed(
        raw["text"],
        image_width=case["image_width"],
        image_height=case["image_height"],
        row_id=case["row_id"],
        row_index=case["row_index"],
    ).to_artifact_dict()
    valid, drops = matchable_rows_with_geometry_debt(
        {**native, "pred": native["predictions"]}
    )
    for prediction in valid:
        prediction["prediction_id"] = f"P{prediction['generated_order']}"
    ledger = ledger_image(bank, valid, threshold=0.5)
    owners = {str(owner["owner_id"]): owner for owner in bank}
    predictions = {prediction["prediction_id"]: prediction for prediction in valid}
    ledger["label_string_compatible_matches"] = [
        dict(owner_id=item["reference_owner_id"], prediction_id=item["prediction_id"])
        for item in ledger["matches"]
        if owners[item["reference_owner_id"]]["description"].strip().lower()
        == predictions[item["prediction_id"]]["description"].strip().lower()
    ]
    spans = [
        dict(row=index + 1, description=item[1], box=list(map(int, item.groups()[1:])))
        for index, item in enumerate(PAT.finditer(raw["text"]))
    ]
    counts = collections.Counter((item["description"], tuple(item["box"])) for item in spans)
    seen = set()
    first_repeat = None
    runs = []
    for item in spans:
        key = (item["description"], tuple(item["box"]))
        if key in seen and first_repeat is None:
            first_repeat = item
        seen.add(key)
        if runs and (runs[-1]["description"], runs[-1]["box"]) == (
            item["description"], item["box"]
        ):
            runs[-1]["length"] += 1
        else:
            runs.append(dict(start_row=item["row"], length=1,
                             description=item["description"], box=item["box"]))
    repeats = strict_repeat_rows(valid)
    reasons = collections.Counter(item.get("drop_reason", item.get("reason")) for item in drops)
    coordinates = [token - 151670 for token in raw["token_ids"] if 151670 <= token <= 152669]
    return dict(
        matches=ledger,
        token_count=len(raw["token_ids"]),
        stop=raw["stop"],
        burden=dict(
            complete_rows=len(spans),
            valid=len(valid),
            invalid=sum(not (item["box"][0] < item["box"][2]
                            and item["box"][1] < item["box"][3]) for item in spans),
            malformed=sum(count for reason, count in reasons.items()
                          if "geometry" not in str(reason) and "bbox" not in str(reason)),
            drop_reasons=dict(reasons),
            strict_valid_repeats=len(repeats),
            literal_repeats=sum(count - 1 for count in counts.values()),
            literal_invalid_repeats=sum(
                count - 1 for (_, box), count in counts.items()
                if not (box[0] < box[2] and box[1] < box[3])
            ),
            unknown=len(ledger["annotation_unmatched_prediction_ids"]),
            cap=int(raw["stop"] == "length"),
            eos=int(raw["stop"] == "im_end"),
        ),
        endpoint_occupancy=dict(
            total=len(coordinates),
            zero=coordinates.count(0),
            last=coordinates.count(999),
            fraction=sum(value in [0, 999] for value in coordinates) / len(coordinates)
            if coordinates else None,
            per_role={
                role: dict(total=len(spans),
                           zero=sum(item["box"][index] == 0 for item in spans),
                           last=sum(item["box"][index] == 999 for item in spans))
                for index, role in enumerate(["x1", "y1", "x2", "y2"])
            },
        ),
        first_literal_repeat=first_repeat,
        first_strict_repeat=repeats[0] if repeats else None,
        longest_exact_run=max(runs, key=lambda item: item["length"]) if runs else None,
        exact_runs=runs,
        complete_rows=spans,
        native_parse=native,
        valid_predictions=valid,
        drops=drops,
    )


def termination_metrics(token_count: int, token_ids: Sequence[int], stop_reason: str, *, cap: int, eos_token_id: int = 151645) -> dict[str, Any]:
    """Separate natural EOS from a length cap, including exact-cap cases."""
    natural_eos = stop_reason == "im_end" and bool(token_ids) and token_ids[-1] == eos_token_id
    capped_by_limit = stop_reason in {"length", "max_new_tokens"} and token_count >= cap
    return {"natural_eos": natural_eos, "eos_debt": not natural_eos,
            "capped": capped_by_limit, "capped_by_limit": capped_by_limit, "cap_debt": int(capped_by_limit),
            "excess_token_count": max(0, token_count - cap)}
