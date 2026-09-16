"""CPU-only scoring and visual-review preparation for saved training readbacks.

The selector is diagnostic evidence only. It parses the saved token text with
the native parser, keeps parser drops in the raw-row view, and compares valid
``coord_bins`` (normalized 0..1000) with the bound reference boxes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.training_set_completion import artifacts as artifact_primitives
from src.eval.assignment import global_matches


IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
CAP_DEFAULT = 3084


canonical = artifact_primitives.canonical
digest = artifact_primitives.digest
file_hash = artifact_primitives.file_hash
binding = artifact_primitives.binding


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_unique_image_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    image_ids = [int(row["image_id"]) for row in rows]
    require(len(image_ids) == len(set(image_ids)), "duplicate readback image rows")


def termination_metrics(token_count: int, token_ids: Sequence[int], stop_reason: str, *, cap: int, eos_token_id: int = 151645) -> dict[str, Any]:
    """Separate natural EOS from a length cap, including exact-cap cases."""
    natural_eos = stop_reason == "im_end" and bool(token_ids) and token_ids[-1] == eos_token_id
    capped_by_limit = stop_reason in {"length", "max_new_tokens"} and token_count >= cap
    return {"natural_eos": natural_eos, "eos_debt": not natural_eos,
            "capped": capped_by_limit, "capped_by_limit": capped_by_limit, "cap_debt": int(capped_by_limit),
            "excess_token_count": max(0, token_count - cap)}


def iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    require(len(left) == 4 and len(right) == 4, "box must have four coordinates")
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


def _load_tokenizer(root: str | Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(root), local_files_only=True)


def score_readback(readback: Mapping[str, Any], acquisition: Mapping[str, Any], target: Mapping[str, Any], tokenizer: Any, *, cap: int) -> dict[str, Any]:
    from src.eval.native_rows import native_detection_record as native_record

    records = {int(row["image_id"]): row for row in acquisition["records"]}
    references = {}
    for row in target["records"]:
        if row.get("role") in {"gt_atomic", "new", "prior_non_gt"}:
            references.setdefault(int(row["image_id"]), []).append(row)
    readback_rows = readback.get("rows", [])
    require(isinstance(readback_rows, list), "readback rows must be a list")
    validate_unique_image_rows(readback_rows)
    scored, seen = [], set()
    for saved in readback_rows:
        image_id = int(saved["image_id"])
        require(image_id in records, f"readback image absent from acquisition manifest: {image_id}")
        ids = saved.get("generated_token_ids")
        require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids), f"invalid saved token IDs: {image_id}")
        if saved.get("generated_token_ids_sha256"):
            require(saved["generated_token_ids_sha256"] == digest(ids), f"saved token-ID hash mismatch: {image_id}")
        text = saved.get("raw_decode_text")
        require(isinstance(text, str), f"missing saved raw text: {image_id}")
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        require(decoded == text, f"saved token decode differs from raw text: {image_id}")
        parsed = native_record(text, records[image_id]["case"], records[image_id]["golden"], saved.get("decode_stop_reason", "unknown"))
        valid, dropped = flatten_raw_rows(parsed)
        valid = [row for row in valid if row["coord_bins_1000"] is not None and len(row["coord_bins_1000"]) == 4]
        image_refs = references.get(image_id, [])
        thresholds = {}
        for threshold in (0.5, 0.8):
            matches = one_to_one_matches(image_refs, valid, threshold)
            matched_refs = {item["reference_owner_id"] for item in matches}
            matched_preds = {item["prediction_id"] for item in matches}
            thresholds[str(threshold)] = {"matches": matches,
                "missing_reference_owner_ids": [ref["owner_id"] for ref in image_refs if ref["owner_id"] not in matched_refs],
                "unmatched_predictions": [item["prediction_id"] for item in valid if item["prediction_id"] not in matched_preds]}
        stop = saved.get("decode_stop_reason")
        termination = termination_metrics(len(ids), ids, stop, cap=cap)
        scored.append({"image_id": image_id, "route_id": saved.get("route_id"), "readback_source": saved,
                       "token_decode_match": True,
                       "token_count": len(ids), "cap": cap, **termination,
                       "decode_stop_reason": stop,
                       "reference_count": len(image_refs), "valid_prediction_count": len(valid),
                       "raw_row_count": len(valid) + len(dropped), "raw_row_debt_count": len(dropped),
                       "raw_rows": valid + dropped, "pairwise_iou_gt_0.95": pairwise_iou95(valid),
                       "pairwise_selector_note": "IoU>0.95 viewing aid only; it does not assert physical repeat or owner identity.",
                       "class_agnostic_iou_matching": thresholds,
                       "coordinate_semantics": "pred.coord_bins_1000 and references are normalized 0..1000; parsed.pred.bbox is processed image pixels and is not mixed into matching.",
                       "status": "diagnostic_only_pending_owner_review"})
        seen.add(image_id)
    missing_images = [image_id for image_id in IMAGE_IDS if image_id not in seen]
    return {"schema": "training_set_completion.readback_selector_preparation.v1", "status": "diagnostic_only_pending_owner_review",
            "readback_status": readback.get("status"), "coordinate_domain": "normalized_0_1000_coord_bins_only_for_matching",
            "matching": "class-agnostic deterministic one-to-one IoU at thresholds .50 and .80; unmatched predictions are diagnostic, not FP",
            "expected_image_ids": list(IMAGE_IDS), "readback_image_ids": sorted(seen), "missing_image_ids": missing_images,
            "unsupported_missing_reference_counts": [{"image_id": image_id, "reference_count": len(references.get(image_id, [])), "reason": "no saved readback row; no predictions fabricated"} for image_id in missing_images],
            "rows": scored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--target-owners", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-packet-output", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--cap", type=int, default=CAP_DEFAULT)
    args = parser.parse_args()
    readback, acquisition, target = (json.loads(path.read_text()) for path in (args.readback, args.acquisition_manifest, args.target_owners))
    tokenizer_root = args.tokenizer_root or Path(acquisition["model"]["base_model"]["root"])
    tokenizer = _load_tokenizer(tokenizer_root)
    result = score_readback(readback, acquisition, target, tokenizer, cap=args.cap)
    result["sources"] = {"readback": binding(args.readback), "acquisition_manifest": binding(args.acquisition_manifest), "target_owners": binding(args.target_owners),
                          "tokenizer_root": {"path": str(tokenizer_root.resolve()), "tokenizer_json_sha256": file_hash(tokenizer_root / "tokenizer.json")},
                          "producer": binding(Path(__file__)), "artifact_primitives": binding(Path(artifact_primitives.__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result))
    # Existing packet renderings provide the one-image overlay/crop review
    # surface; this hook records exactly which files the reviewer should open.
    packet_root = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/stage01-review-packets-v1")
    packet_rows = []
    for row in result["rows"][:1]:
        image_id = row["image_id"]; packet = packet_root / f"image-{image_id:012d}" / "packet.json"
        packet_data = json.loads(packet.read_text()) if packet.is_file() else None
        packet_rows.append({"image_id": image_id, "packet_json": binding(packet) if packet.is_file() else None,
                            "original_image": packet_data.get("image") if packet_data else None,
                            "overview_path": packet_data.get("overview_path") if packet_data else None,
                            "gt_overlay_path": packet_data.get("gt_overlay_path") if packet_data else None,
                            "policy_overlay_paths": {p["policy"]: p["overlay_path"] for p in packet_data.get("policies", [])} if packet_data else {},
                            "raw_rows_for_review": row["raw_rows"], "unmatched_predictions": row["class_agnostic_iou_matching"]["0.5"]["unmatched_predictions"],
                            "existing_crop_paths_by_prediction": {item["prediction_id"]: [str(path) for path in sorted((packet.parent / "crops").glob(f"*-p{item['generated_order']}-*.png"))]
                                                                    for item in row["raw_rows"]},
                            "review_boundary": "Open the bound original, GT overlay, four policy overlays, and exact existing crops; this packet makes no owner or FP decision."})
    review_packet = {"schema": "training_set_completion.readback_selector_review_packet.v1", "status": "diagnostic_only_pending_owner_review", "source_scored_json": binding(args.output), "images": packet_rows}
    args.review_packet_output.write_bytes(canonical(review_packet))
    print(json.dumps({"status": result["status"], "readback_images": result["readback_image_ids"], "missing_images": result["missing_image_ids"],
                      "rows": [{"image_id": r["image_id"], "references": r["reference_count"], "valid_predictions": r["valid_prediction_count"], "raw_rows": r["raw_row_count"], "raw_row_debt": r["raw_row_debt_count"], "matched_0.5": len(r["class_agnostic_iou_matching"]["0.5"]["matches"]), "matched_0.8": len(r["class_agnostic_iou_matching"]["0.8"]["matches"])} for r in result["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
