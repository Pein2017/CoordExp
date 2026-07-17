#!/usr/bin/env python3
"""Admit first-row native sibling owners from one merged replay receipt.

This is a deliberately small, offline research helper.  It does not rerun a
model, infer a missing object, or turn an ambiguous geometry into an owner.
Only sampled calls are eligible for owner admission.  Greedy rows are emitted
as a separate control record and never contribute support to an owner.

The matching gate is intentionally stricter than the historical atlas:

* ledger rows must be ``final_state == accepted``;
* image and normalized COCO-80 category must match;
* the maximum intersection-over-union (IoU) must be at least 0.50; and
* a maximum with a top/second IoU margin at most 0.05 is ambiguous.

The variant table preserves the exact generated token span, token-id hash,
raw row text, source bundle, request identifier, integer seed, owner, and all
same-category ledger IoUs.  Seeds are kept as Python integers and serialized
without a floating-point conversion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "native_sibling_first_row_owner_admission.v1"
DEFAULT_IOU_FLOOR = 0.50
DEFAULT_AMBIGUITY_MARGIN = 0.05
DEFAULT_EXPECTED_SAMPLES = 32
SUPPORTED_SAMPLE_COUNTS = frozenset({32, 64})


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_category(value: Any) -> str:
    """Normalize only whitespace/case; do not invent category aliases."""

    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _as_box(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if not (x1 < x2 and y1 < y2):
        return None
    return box


def box_iou(left: Sequence[Any], right: Sequence[Any]) -> float:
    """Return finite axis-aligned box IoU, or zero for malformed boxes."""

    left_box = _as_box(left)
    right_box = _as_box(right)
    if left_box is None or right_box is None:
        return 0.0
    lx1, ly1, lx2, ly2 = left_box
    rx1, ry1, rx2, ry2 = right_box
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def read_accepted_ledger(path: Path) -> list[dict[str, Any]]:
    """Read accepted ledger rows and discard no-owner/malformed entries."""

    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping) or value.get("final_state") != "accepted":
                continue
            owner_id = str(value.get("object_identifier") or "")
            category = _normalize_category(value.get("normalized_category_name"))
            box = _as_box(value.get("source_canvas_box_xyxy"))
            if not owner_id or not category or box is None:
                raise ValueError(f"accepted ledger row {line_number} lacks owner/category/box")
            rows.append(
                {
                    **dict(value),
                    "object_identifier": owner_id,
                    "normalized_category_name": category,
                    "source_canvas_box_xyxy": box,
                }
            )
    return rows


def match_first_row_to_ledger(
    *,
    prediction: Mapping[str, Any] | None,
    image_id: str,
    ledger: Sequence[Mapping[str, Any]],
    iou_floor: float = DEFAULT_IOU_FLOOR,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> dict[str, Any]:
    """Classify one prediction without making a claim on ambiguous geometry."""

    category = _normalize_category(prediction.get("description") if prediction else None)
    box = _as_box(prediction.get("bbox") if prediction else None)
    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for item in ledger:
        if str(item.get("final_state")) != "accepted":
            continue
        if str(item.get("image_id")) != str(image_id):
            continue
        if _normalize_category(item.get("normalized_category_name")) != category:
            continue
        candidate_box = _as_box(item.get("source_canvas_box_xyxy"))
        owner_id = str(item.get("object_identifier") or "")
        if candidate_box is None or not owner_id or box is None:
            continue
        candidates.append((box_iou(box, candidate_box), owner_id, item))
    candidates.sort(key=lambda entry: (-entry[0], entry[1]))
    ious = {owner_id: float(score) for score, owner_id, _ in candidates}
    best_iou = float(candidates[0][0]) if candidates else 0.0
    second_iou = float(candidates[1][0]) if len(candidates) > 1 else 0.0
    margin = float(best_iou - second_iou)
    if box is None or not category:
        status = "unmatched"
        reason = "missing_prediction_box_or_category"
    elif best_iou < float(iou_floor):
        status = "unmatched"
        reason = "maximum_iou_below_floor"
    # Treat the mathematical boundary as ambiguous despite binary floating
    # point representation of decimal IoUs such as 0.95 and 1.00.
    elif margin <= float(ambiguity_margin) + 1e-12:
        status = "ambiguous"
        reason = "top_second_iou_margin_at_or_below_threshold"
    else:
        status = "unique"
        reason = "unique_maximum_iou_owner"
    return {
        "status": status,
        "reason": reason,
        "normalized_category": category,
        "best_iou": best_iou,
        "second_iou": second_iou,
        "top_second_iou_margin": margin,
        "iou_floor": float(iou_floor),
        "ambiguity_margin": float(ambiguity_margin),
        "candidate_count": len(candidates),
        "owner_id": candidates[0][1] if status == "unique" else None,
        "candidate_owner_ids": [owner_id for _, owner_id, _ in candidates],
        "ious_by_owner_id": ious,
    }


def _bundle_payload(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    value = bundle.get("decode_result")
    if not isinstance(value, Mapping):
        raise ValueError("replay bundle lacks decode_result")
    return value


def _first_prediction(bundle: Mapping[str, Any]) -> Mapping[str, Any] | None:
    parse_result = bundle.get("parse_result")
    if not isinstance(parse_result, Mapping):
        return None
    predictions = parse_result.get("predictions")
    if not isinstance(predictions, list):
        return None
    for prediction in predictions:
        if isinstance(prediction, Mapping) and int(prediction.get("generated_order", -1)) == 0:
            return prediction
    return None


def first_complete_row_record(
    *,
    bundle: Mapping[str, Any],
    source_bundle: Path,
    ledger: Sequence[Mapping[str, Any]],
    iou_floor: float = DEFAULT_IOU_FLOOR,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> dict[str, Any]:
    """Extract and classify the first complete generated row in one bundle."""

    spans = bundle.get("complete_row_spans")
    generated = bundle.get("raw_generated_token_ids")
    if not isinstance(spans, list) or not spans or not isinstance(generated, list):
        return {
            "row_status": "no_complete_first_row",
            "source_bundle": str(source_bundle),
            "request_id": str(bundle.get("request_id") or ""),
            "sampling_seed": bundle.get("sampling_seed"),
            "owner_id": None,
        }
    try:
        start, end = int(spans[0][0]), int(spans[0][1])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed first complete row span in {source_bundle}") from exc
    if not (0 <= start < end <= len(generated)):
        raise ValueError(f"first complete row span is outside generated IDs in {source_bundle}")
    row_token_ids = [int(token) for token in generated[start:end]]
    prediction = _first_prediction(bundle)
    raw_text = str(prediction.get("raw_span_text", "")) if prediction else ""
    image_id = str(bundle.get("image_id") or "")
    match = match_first_row_to_ledger(
        prediction=prediction,
        image_id=image_id,
        ledger=ledger,
        iou_floor=iou_floor,
        ambiguity_margin=ambiguity_margin,
    )
    runtime = bundle.get("runtime")
    decode_mode = runtime.get("decode_mode") if isinstance(runtime, Mapping) else ""
    return {
        "row_status": "accepted_first_complete_row" if prediction is not None else "parser_missing_first_row",
        "image_id": image_id,
        "request_id": str(bundle.get("request_id") or ""),
        "sampling_seed": None if bundle.get("sampling_seed") is None else int(bundle["sampling_seed"]),
        "sampling_seed_decimal": (
            None if bundle.get("sampling_seed") is None else str(int(bundle["sampling_seed"]))
        ),
        "decode_mode": str(decode_mode or ""),
        "source_bundle": str(source_bundle),
        "generated_token_span": [start, end],
        "row_token_count": len(row_token_ids),
        "row_token_ids_sha256": _sha256_json(row_token_ids),
        "raw_exact_row_text": raw_text,
        "raw_exact_row_text_sha256": _sha256_text(raw_text),
        "prediction": dict(prediction) if prediction is not None else None,
        "ledger_match": match,
        "owner_id": match.get("owner_id"),
    }


def _resolve_bundle_path(path: str | Path, merged_path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=True)
    return (merged_path.parent / candidate).resolve(strict=True)


def _load_merged(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("calls"), list):
        raise ValueError("merged replay receipt must be an object with calls")
    return value


def _owner_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_samples: int,
) -> list[dict[str, Any]]:
    threshold = 3 if expected_samples == 32 else 4 if expected_samples == 64 else None
    if threshold is None:
        raise ValueError("expected_samples must be 32 or 64")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        match = record.get("ledger_match")
        if isinstance(match, Mapping) and match.get("status") == "unique" and match.get("owner_id"):
            grouped[str(match["owner_id"])].append(record)
    result: list[dict[str, Any]] = []
    for owner_id, owner_records in sorted(grouped.items()):
        hashes = sorted({str(record.get("row_token_ids_sha256")) for record in owner_records})
        support = len(owner_records)
        result.append(
            {
                "owner_id": owner_id,
                "support_count": support,
                "support_denominator": int(expected_samples),
                "support_fraction": float(support / expected_samples),
                "distinct_exact_row_hash_count": len(hashes),
                "distinct_exact_row_hashes": hashes,
                "admitted": bool(support >= threshold and len(hashes) >= 3),
                "support_threshold": threshold,
                "variant_threshold": 3,
            }
        )
    return result


def analyze_merged_receipt(
    *,
    merged_path: Path,
    ledger_path: Path,
    expected_samples: int = DEFAULT_EXPECTED_SAMPLES,
    iou_floor: float = DEFAULT_IOU_FLOOR,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a compact manifest and complete sampled/greedy variant records."""

    if expected_samples not in SUPPORTED_SAMPLE_COUNTS:
        raise ValueError("expected_samples must be 32 or 64")
    merged_path = merged_path.expanduser().resolve(strict=True)
    ledger_path = ledger_path.expanduser().resolve(strict=True)
    merged = _load_merged(merged_path)
    ledger = read_accepted_ledger(ledger_path)
    variants: list[dict[str, Any]] = []
    sampled_records: list[dict[str, Any]] = []
    greedy_records: list[dict[str, Any]] = []
    for call in merged["calls"]:
        if not isinstance(call, Mapping):
            raise ValueError("merged receipt contains a non-object call")
        mode = str(call.get("decode_mode") or "")
        if mode not in {"sampled", "greedy"}:
            continue
        bundle_path = _resolve_bundle_path(str(call.get("bundle_path") or ""), merged_path)
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if not isinstance(bundle, Mapping):
            raise ValueError(f"bundle is not a JSON object: {bundle_path}")
        record = first_complete_row_record(
            bundle=bundle,
            source_bundle=bundle_path,
            ledger=ledger,
            iou_floor=iou_floor,
            ambiguity_margin=ambiguity_margin,
        )
        record["decode_mode"] = mode
        record["sampling_seed"] = (
            None if call.get("sampling_seed") is None else int(call["sampling_seed"])
        )
        record["sampling_seed_decimal"] = (
            None if record["sampling_seed"] is None else str(record["sampling_seed"])
        )
        record["request_id"] = str(call.get("request_id") or record.get("request_id") or "")
        record["call_sampling_seed_matches_bundle"] = (
            mode != "sampled"
            or bundle.get("sampling_seed") is None
            or int(bundle["sampling_seed"]) == record["sampling_seed"]
        )
        variants.append(record)
        (sampled_records if mode == "sampled" else greedy_records).append(record)
    owner_summaries = _owner_summary(sampled_records, expected_samples=expected_samples)
    owner_by_id = {item["owner_id"]: item for item in owner_summaries}
    observed_sample_count = len(sampled_records)
    denominator_matches = observed_sample_count == int(expected_samples)
    for item in owner_summaries:
        if not denominator_matches:
            item["admitted"] = False
            item["admission_blocker"] = "observed_sample_count_does_not_match_expected_denominator"
    for greedy in greedy_records:
        match = greedy.get("ledger_match")
        owner_id = match.get("owner_id") if isinstance(match, Mapping) else None
        greedy["owner_admitted"] = (
            None if owner_id is None else bool(owner_by_id.get(str(owner_id), {}).get("admitted", False))
        )
        greedy["owner_uncovered"] = None
        greedy["owner_uncovered_evidence"] = "not_available_from_merged_receipt"
    status_counts = Counter(str(record.get("ledger_match", {}).get("status")) for record in sampled_records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "full_name": "Native Sibling First-Row Owner Admission",
        "operational_meaning": "Offline sampled first-row owner gate; greedy is a non-admitting control.",
        "merged_receipt": str(merged_path),
        "merged_receipt_sha256": _sha256_file(merged_path),
        "ledger": str(ledger_path),
        "ledger_sha256": _sha256_file(ledger_path),
        "image_ids": sorted({str(record.get("image_id")) for record in sampled_records if record.get("image_id")}),
        "expected_sample_count": int(expected_samples),
        "observed_sample_count": int(observed_sample_count),
        "sample_count_matches_expected": bool(denominator_matches),
        "support_threshold": 3 if expected_samples == 32 else 4,
        "distinct_exact_row_variant_threshold": 3,
        "iou_floor": float(iou_floor),
        "ambiguity_margin": float(ambiguity_margin),
        "sampled_status_counts": dict(sorted(status_counts.items())),
        "owner_admission_ready": bool(denominator_matches),
        "owner_groups": owner_summaries,
        "admitted_owner_ids": sorted(owner_id for owner_id, value in owner_by_id.items() if value.get("admitted")),
        "greedy_control_count": len(greedy_records),
        "greedy_control": [
            {
                "request_id": item.get("request_id"),
                "source_bundle": item.get("source_bundle"),
                "row_status": item.get("row_status"),
                "ledger_status": item.get("ledger_match", {}).get("status"),
                "owner_id": item.get("owner_id"),
                "owner_admitted": item.get("owner_admitted"),
                "owner_uncovered": item.get("owner_uncovered"),
                "owner_uncovered_evidence": item.get("owner_uncovered_evidence"),
            }
            for item in greedy_records
        ],
    }
    return manifest, variants


def _write_variant_tables(
    variants: Sequence[Mapping[str, Any]],
    *,
    jsonl_path: Path,
    csv_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite immutable variant table")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for variant in variants:
            handle.write(json.dumps(dict(variant), ensure_ascii=False, sort_keys=True) + "\n")
    columns = [
        "decode_mode",
        "image_id",
        "request_id",
        "sampling_seed",
        "sampling_seed_decimal",
        "row_status",
        "generated_token_span",
        "row_token_ids_sha256",
        "raw_exact_row_text",
        "raw_exact_row_text_sha256",
        "normalized_category",
        "ledger_status",
        "owner_id",
        "best_iou",
        "second_iou",
        "top_second_iou_margin",
        "ious_by_owner_id",
        "source_bundle",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for variant in variants:
            match = variant.get("ledger_match") if isinstance(variant.get("ledger_match"), Mapping) else {}
            writer.writerow(
                {
                    "decode_mode": variant.get("decode_mode"),
                    "image_id": variant.get("image_id"),
                    "request_id": variant.get("request_id"),
                    "sampling_seed": (
                        "" if variant.get("sampling_seed") is None else str(variant.get("sampling_seed"))
                    ),
                    "sampling_seed_decimal": variant.get("sampling_seed_decimal"),
                    "row_status": variant.get("row_status"),
                    "generated_token_span": json.dumps(variant.get("generated_token_span")),
                    "row_token_ids_sha256": variant.get("row_token_ids_sha256"),
                    "raw_exact_row_text": variant.get("raw_exact_row_text"),
                    "raw_exact_row_text_sha256": variant.get("raw_exact_row_text_sha256"),
                    "normalized_category": match.get("normalized_category"),
                    "ledger_status": match.get("status"),
                    "owner_id": match.get("owner_id"),
                    "best_iou": match.get("best_iou"),
                    "second_iou": match.get("second_iou"),
                    "top_second_iou_margin": match.get("top_second_iou_margin"),
                    "ious_by_owner_id": json.dumps(match.get("ious_by_owner_id", {}), sort_keys=True),
                    "source_bundle": variant.get("source_bundle"),
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-receipt", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Admission manifest JSON path")
    parser.add_argument("--variants-jsonl", type=Path)
    parser.add_argument("--variants-csv", type=Path)
    parser.add_argument("--expected-samples", type=int, default=DEFAULT_EXPECTED_SAMPLES, choices=(32, 64))
    parser.add_argument("--iou-floor", type=float, default=DEFAULT_IOU_FLOOR)
    parser.add_argument("--ambiguity-margin", type=float, default=DEFAULT_AMBIGUITY_MARGIN)
    args = parser.parse_args(argv)
    manifest, variants = analyze_merged_receipt(
        merged_path=args.merged_receipt,
        ledger_path=args.ledger,
        expected_samples=args.expected_samples,
        iou_floor=args.iou_floor,
        ambiguity_margin=args.ambiguity_margin,
    )
    output = args.output.expanduser().resolve()
    jsonl_path = (args.variants_jsonl or output.with_name(output.stem + "-variants.jsonl")).expanduser().resolve()
    csv_path = (args.variants_csv or output.with_name(output.stem + "-variants.csv")).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_variant_tables(variants, jsonl_path=jsonl_path, csv_path=csv_path)
    print(json.dumps({"manifest": str(output), "variants_jsonl": str(jsonl_path), "variants_csv": str(csv_path), "variant_count": len(variants)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
