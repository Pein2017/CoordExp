"""CPU consumer for the deployment-greedy repeat-supply census.

This module deliberately does not load a model.  It consumes the immutable
Stable50 endpoint records (or a later A-owned raw128 subset of that frozen
universe), replays the class-blind native-pixel repeat predicate, and freezes a
small visual-review manifest.  A sub-threshold pair is a review candidate, not
a negative label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

CAP = 3084
EOS = 151645
DUPLICATE_IOU_THRESHOLD = 0.95
REVIEW_LIMIT_PER_STRATUM = 2
DEFAULT_SOURCE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-positive-progress-matched-control/endpoint-preparation/packet.json"
)
DEFAULT_UNIVERSE = DEFAULT_SOURCE
DEFAULT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state/repeat"
)

SCHEMA = "native_owner_scale.repeat.census.v1"
REPLAY_SCHEMA = "native_owner_scale.repeat.replay_packet.v1"
CONSUMER_SCHEMA = "native_owner_scale.repeat.consumer.v1"
LINEAGE_SCHEMA = "native_owner_scale.repeat.lineage.v1"
PROJECTION_SCHEMA = "native_owner_scale.repeat.raw128_projection.v1"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing source: {path}")
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> Any:
    require(path.is_absolute() and path.is_file(), f"missing JSON source: {path}")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def publish(path: Path, value: Any) -> None:
    """Write canonical JSON without silently overwriting a prior receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    if path.exists():
        require(path.read_bytes() == payload, f"occupied output differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()


def _category(obj: Mapping[str, Any] | None) -> str:
    if not isinstance(obj, Mapping):
        return ""
    raw = obj.get("description", obj.get("desc", obj.get("label", obj.get("category", ""))))
    if not isinstance(raw, str):
        return ""
    try:
        from src.eval.detection_categories import normalize_coco_category_name

        return normalize_coco_category_name(raw)
    except Exception:
        return " ".join(raw.strip().lower().split())


def _pixel_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return None
    return box


def _bbox(obj: Mapping[str, Any] | None) -> Any:
    return obj.get("bbox", obj.get("bbox_2d")) if isinstance(obj, Mapping) else None


def iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0.0 else 0.0


def is_strict_repeat(overlap: float) -> bool:
    """The frozen predicate is exclusive: native-pixel IoU must be > .95."""
    return overlap > DUPLICATE_IOU_THRESHOLD


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("example_id", record.get("row_id", record.get("image_id")))
    require(value is not None, "record has no stable example/image identity")
    return str(value)


def _first(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _source_rows(raw: Any) -> tuple[list[dict[str, Any]], str]:
    """Extract parsed complete-row records from a packet or A-owned manifest."""
    if isinstance(raw, list):
        rows = raw
        schema = "jsonl_records"
    elif isinstance(raw, Mapping):
        schema = str(raw.get("schema", "unknown"))
        rows = raw.get("eval_records")
        if rows is None:
            rows = raw.get("records")
        if rows is None:
            rows = raw.get("raw_records")
        require(isinstance(rows, list), "source has no eval_records/records list")
    else:
        raise ValueError("source must be a JSON object or JSONL list")
    require(rows and all(isinstance(row, Mapping) for row in rows), "source records are empty/malformed")
    return [dict(row) for row in rows], schema


def _normalize_source_record(
    record: Mapping[str, Any], *, source_schema: str, source_path: Path
) -> dict[str, Any]:
    rid = _record_id(record)
    parsed = _first(record, "stable_parsed", "parsed", "source_parsed", "raw_parsed")
    require(isinstance(parsed, Mapping), f"{rid}: no parsed complete-row evidence")
    parsed = dict(parsed)
    pred = parsed.get("pred")
    require(isinstance(pred, list), f"{rid}: parsed pred is not a list")
    ids = _first(record, "stable_ids", "action_ids", "token_ids", "generated_token_ids")
    require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids), f"{rid}: token trace missing/invalid")
    score = _first(record, "stable_score", "score")
    score = dict(score) if isinstance(score, Mapping) else {}
    stop = _first(record, "stop_reason", "decode_stop_reason")
    if stop is None:
        stop = score.get("stop_reason", parsed.get("decode_stop_reason"))
    require(stop in ("im_end", "length", "eos", "forced_eos", "forced_im_end"), f"{rid}: unsupported stop reason {stop!r}")
    stop = "im_end" if stop in ("eos", "forced_eos", "forced_im_end") else str(stop)
    if stop == "length":
        require(len(ids) == CAP and EOS not in ids, f"{rid}: capped trace is not exactly {CAP} non-EOS tokens")
    else:
        require(0 < len(ids) <= CAP and ids[-1] == EOS and EOS not in ids[:-1], f"{rid}: EOS trace/budget mismatch")
    width = parsed.get("image_width", parsed.get("width"))
    height = parsed.get("image_height", parsed.get("height"))
    require(type(width) is int and width > 0 and type(height) is int and height > 0, f"{rid}: invalid image dimensions")
    image_path = parsed.get("image_path")
    if image_path is None:
        case = record.get("case")
        image_path = case.get("image_path") if isinstance(case, Mapping) else None
    require(isinstance(image_path, str) and image_path, f"{rid}: image path missing")
    prefix_ids = _first(record, "prefix_ids", "prefix_token_ids") or []
    forced_ids = _first(record, "forced_ids") or []
    require(isinstance(prefix_ids, list) and isinstance(forced_ids, list), f"{rid}: prefix provenance malformed")
    plan = record.get("case", {}).get("image_plan", {}) if isinstance(record.get("case"), Mapping) else {}
    image_content_sha = _first(record, "image_content_sha256")
    executed_media_sha = _first(record, "executed_media_sha256")
    if isinstance(plan, Mapping):
        if image_content_sha is None:
            image_content_sha = _first(plan, "image_content_sha256")
        if executed_media_sha is None:
            executed_media_sha = _first(plan, "executed_media_sha256")
    normalized = {
        "example_id": rid,
        "image_id": int(record.get("image_id", parsed.get("row_id", -1))),
        "split": record.get("split"),
        "image_path": image_path,
        "image_width": width,
        "image_height": height,
        "image_content_sha256": image_content_sha,
        "executed_media_sha256": executed_media_sha,
        "parsed": parsed,
        "token_ids": list(ids),
        "token_ids_sha256": digest(ids),
        "stop_reason": stop,
        "prefix_ids": list(prefix_ids),
        "forced_ids": list(forced_ids),
        "remaining_budget": int(record.get("remaining_budget", CAP - len(prefix_ids) - len(forced_ids))),
        "source_record_digest": digest({
            "example_id": rid,
            "image_id": record.get("image_id"),
            "token_ids": ids,
            "parsed": parsed,
        }),
        "source_schema": source_schema,
        "source_path": str(source_path.resolve()),
        "declared_score": score,
    }
    require(normalized["remaining_budget"] >= 0, f"{rid}: negative remaining budget")
    return normalized


def _load_records(source_path: Path, universe_path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = source_path.resolve()
    raw = load_json(source_path)
    rows, source_schema = _source_rows(raw)
    normalized = [_normalize_source_record(row, source_schema=source_schema, source_path=source_path) for row in rows]
    ids = [_record_id(row) for row in normalized]
    require(len(ids) == len(set(ids)), "duplicate source example IDs")
    universe_binding = binding(universe_path.resolve()) if universe_path is not None else None
    if universe_path is not None and source_path != universe_path.resolve():
        urows, uschema = _source_rows(load_json(universe_path.resolve()))
        universe_ids = {_record_id(row) for row in urows}
        require(set(ids) <= universe_ids, "A raw128 source expands beyond frozen 384 universe")
        require(len(ids) <= len(universe_ids), "source is larger than frozen universe")
        universe_binding["schema"] = uschema
    return normalized, {"source": binding(source_path), "universe": universe_binding}


def _valid_rows(parsed: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    result: list[dict[str, Any]] = []
    invalid = Counter()
    pred = parsed.get("pred", [])
    require(isinstance(pred, list), "parsed pred is not a list")
    for index, obj in enumerate(pred):
        box = _pixel_box(_bbox(obj))
        category = _category(obj)
        if box is None:
            invalid["invalid_geometry_in_parsed"] += 1
            continue
        if not category:
            invalid["invalid_category_in_parsed"] += 1
            continue
        result.append({"pred_index": index, "category": category, "bbox": list(box), "object": obj})
    return result, invalid


def _drop_counts(parsed: Mapping[str, Any]) -> Counter[str]:
    drops = parsed.get("dropped_predictions", [])
    require(isinstance(drops, list), "dropped_predictions is not a list")
    counts = Counter()
    for item in drops:
        reason = item.get("reason") if isinstance(item, Mapping) else None
        if reason == "geometry_invalid":
            counts["parser_geometry_invalid"] += 1
        elif reason:
            counts["parser_malformed"] += 1
        else:
            counts["parser_other_malformed"] += 1
    return counts


def _supported_seed_indices(parsed: Mapping[str, Any], valid: Sequence[Mapping[str, Any]]) -> set[int]:
    gt = parsed.get("gt", [])
    require(isinstance(gt, list), "parsed gt is not a list")
    gt_rows = [(_category(obj), _pixel_box(_bbox(obj))) for obj in gt]
    supported: set[int] = set()
    for row in valid:
        if any(category == row["category"] and box is not None and iou_xyxy(row["bbox"], box) >= 0.5
               for category, box in gt_rows):
            supported.add(int(row["pred_index"]))
    return supported


def _best_previous(
    row: Mapping[str, Any], previous: Sequence[Mapping[str, Any]], *, same_category: bool
) -> dict[str, Any] | None:
    candidates = [
        {
            "iou": iou_xyxy(row["bbox"], earlier["bbox"]),
            "seed_pred_index": int(earlier["pred_index"]),
            "seed_category": earlier["category"],
            "seed_bbox": list(earlier["bbox"]),
        }
        for earlier in previous
        if not same_category or earlier["category"] == row["category"]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda value: (-value["iou"], value["seed_pred_index"]))


def _pair_record(
    current: Mapping[str, Any],
    best: Mapping[str, Any],
    *,
    relation: str,
    seed_support: str,
    later_supported: bool,
) -> dict[str, Any]:
    overlap = float(best["iou"])
    if relation == "subthreshold_same_category_candidate":
        iou_bin = "near" if overlap >= 0.90 else "mid" if overlap >= 0.50 else "low"
    else:
        iou_bin = "strict"
    return {
        "later_pred_index": int(current["pred_index"]),
        "later_category": current["category"],
        "later_bbox": list(current["bbox"]),
        "seed_pred_index": int(best["seed_pred_index"]),
        "seed_category": best["seed_category"],
        "seed_bbox": list(best["seed_bbox"]),
        "best_iou": overlap,
        "seed_support": seed_support,
        "later_support": "supported" if later_supported else "unknown_or_unmatched",
        "relation": relation,
        "iou_bin": iou_bin,
        "negative_authorized": False,
        "review_status": "not_a_label" if relation == "strict_repeat" else "unknown_pending_visual_review",
    }


def _record_census(record: Mapping[str, Any]) -> dict[str, Any]:
    parsed = record["parsed"]
    valid, invalid = _valid_rows(parsed)
    drops = _drop_counts(parsed)
    supported_indices = _supported_seed_indices(parsed, valid)
    strict: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    for position, current in enumerate(valid):
        previous = valid[:position]
        if not previous:
            continue
        any_best = _best_previous(current, previous, same_category=False)
        same_best = _best_previous(current, previous, same_category=True)
        if any_best is not None and is_strict_repeat(float(any_best["iou"])):
            strict.append(_pair_record(
                current, any_best, relation="strict_repeat",
                seed_support="supported" if int(any_best["seed_pred_index"]) in supported_indices else "unknown_or_unmatched",
                later_supported=int(current["pred_index"]) in supported_indices,
            ))
        if same_best is not None and 0.0 < float(same_best["iou"]) <= DUPLICATE_IOU_THRESHOLD:
            drift.append(_pair_record(
                current, same_best, relation="subthreshold_same_category_candidate",
                seed_support="supported" if int(same_best["seed_pred_index"]) in supported_indices else "unknown_or_unmatched",
                later_supported=int(current["pred_index"]) in supported_indices,
            ))
    score = record.get("declared_score", {})
    score = score if isinstance(score, Mapping) else {}
    parsed_status = str(parsed.get("parse_status", "unknown"))
    counts = {
        "parsed_rows": len(parsed.get("pred", [])),
        "valid_complete_rows": len(valid),
        "invalid_geometry_rows": int(invalid["invalid_geometry_in_parsed"] + drops["parser_geometry_invalid"]),
        "invalid_category_rows": int(invalid["invalid_category_in_parsed"]),
        "malformed_rows": int(drops["parser_malformed"] + drops["parser_other_malformed"]),
        "parser_drop_rows": len(parsed.get("dropped_predictions", [])),
        "declared_invalid_predictions": int(score.get("invalid_predictions", 0)),
        "invalid": int(invalid["invalid_geometry_in_parsed"] + invalid["invalid_category_in_parsed"] + drops["parser_geometry_invalid"]),
        "strict_repeat_rows": len(strict),
        "subthreshold_candidates": len(drift),
        "cap": int(record["stop_reason"] == "length" or score.get("cap", 0)),
        "eos": int(record["stop_reason"] == "im_end"),
        "complete_token_length": len(record["token_ids"]),
    }
    return {
        "example_id": record["example_id"],
        "image_id": record["image_id"],
        "split": record.get("split"),
        "image_path": record["image_path"],
        "image_content_sha256": record.get("image_content_sha256"),
        "executed_media_sha256": record.get("executed_media_sha256"),
        "image_width": record["image_width"],
        "image_height": record["image_height"],
        "source_record_digest": record["source_record_digest"],
        "token_ids_sha256": record["token_ids_sha256"],
        "stop_reason": record["stop_reason"],
        "parse_status": parsed_status,
        "counts": counts,
        "strict_repeat_rows": strict,
        "subthreshold_candidates": drift,
    }


def _stratum_key(pair: Mapping[str, Any]) -> tuple[str, str, str]:
    relation = "exact" if pair["relation"] == "strict_repeat" else "drift"
    support = "supported_seed" if pair["seed_support"] == "supported" else "unknown_seed"
    return relation, support, pair["iou_bin"]


def _review_selection(records: Sequence[Mapping[str, Any]], limit: int = REVIEW_LIMIT_PER_STRATUM) -> list[dict[str, Any]]:
    require(limit > 0, "review limit must be positive")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for pair in [*record["strict_repeat_rows"], *record["subthreshold_candidates"]]:
            item = dict(pair)
            item["example_id"] = record["example_id"]
            item["image_id"] = record["image_id"]
            item["image_path"] = record["image_path"]
            item["image_content_sha256"] = record.get("image_content_sha256")
            item["image_width"] = record["image_width"]
            item["image_height"] = record["image_height"]
            item["source_record_digest"] = record["source_record_digest"]
            item["selection_key"] = digest({
                "example_id": record["example_id"],
                "later_pred_index": pair["later_pred_index"],
                "seed_pred_index": pair["seed_pred_index"],
                "relation": pair["relation"],
            })
            grouped[_stratum_key(item)].append(item)
    order = [
        ("exact", "supported_seed", "strict"),
        ("exact", "unknown_seed", "strict"),
        ("drift", "supported_seed", "near"),
        ("drift", "supported_seed", "mid"),
        ("drift", "supported_seed", "low"),
        ("drift", "unknown_seed", "near"),
        ("drift", "unknown_seed", "mid"),
        ("drift", "unknown_seed", "low"),
    ]
    selected: list[dict[str, Any]] = []
    for key in order:
        candidates = sorted(grouped.get(key, []), key=lambda item: item["selection_key"])
        for rank, item in enumerate(candidates[:limit]):
            item = dict(item)
            item["review_unit_id"] = f"{key[0]}-{key[1]}-{key[2]}-{rank:02d}"
            item["stratum"] = {"relation": key[0], "seed": key[1], "iou_bin": key[2]}
            item["selection_rank"] = rank
            item["visual_status"] = "pending_root_review"
            item["negative_authorized"] = False
            selected.append(item)
    return selected


def run_census(source_path: Path, *, universe_path: Path | None = DEFAULT_UNIVERSE, review_limit: int = REVIEW_LIMIT_PER_STRATUM) -> dict[str, Any]:
    records, sources = _load_records(source_path, universe_path)
    per_record = [_record_census(record) for record in records]
    aggregate = Counter()
    for record in per_record:
        aggregate.update(record["counts"])
    selected = _review_selection(per_record, review_limit)
    strict_images = sorted({r["image_id"] for r in per_record if r["counts"]["strict_repeat_rows"]})
    source_schema = records[0]["source_schema"]
    expected_universe = sources.get("universe")
    return {
        "schema": SCHEMA,
        "status": "candidate_cpu_verified",
        "source": sources["source"],
        "frozen_universe": expected_universe,
        "source_schema": source_schema,
        "predicate": {
            "coordinate_space": "native_pixel_xyxy",
            "class_blind": True,
            "threshold": DUPLICATE_IOU_THRESHOLD,
            "comparison": "strict_gt",
            "scope": "later_valid_complete_row_against_any_earlier_valid_complete_row",
            "counting": "one_per_later_row_not_one_per_pair",
        },
        "output_cap": CAP,
        "original_source_population": len(records),
        "counts": {
            **{key: int(value) for key, value in aggregate.items()},
            "source_records": len(records),
            "eligible": len(records),
            "executed": len(records),
            "valid": int(aggregate["valid_complete_rows"]),
            "admitted": int(aggregate["strict_repeat_rows"]),
            "held": int(aggregate["subthreshold_candidates"]),
            "strict_repeat_images": len(strict_images),
            "nominated": len(selected),
            "review_units": len(selected),
        },
        "stage_count_definitions": {
            "eligible": "source-bound parsed records that passed token/EOS/cap/dimension identity checks",
            "executed": "already-existing greedy source records; no new model calls",
            "valid": "complete parsed rows with finite positive native-pixel extent and nonempty class",
            "admitted": "strict repeat rows admitted to this census only; never training/negative labels",
            "held": "all same-category subthreshold overlap candidates held for visual review; not labels",
            "nominated": "bounded deterministic visual-review units",
        },
        "strict_repeat_images": strict_images,
        "records": per_record,
        "review_selection": selected,
        "claim_boundary": (
            "This is deployment-greedy supply and a bounded physical-drift audit. "
            "It does not rerun the 0/768 raw-softmax comparison, test negative-learning efficacy, "
            "or authorize a negative objective. GT-unmatched rows remain unknown."
        ),
    }


def _assert_boundary_invariant() -> dict[str, Any]:
    below = math.nextafter(DUPLICATE_IOU_THRESHOLD, 0.0)
    above = math.nextafter(DUPLICATE_IOU_THRESHOLD, 1.0)
    result = {
        "threshold": DUPLICATE_IOU_THRESHOLD,
        "below": below,
        "equal": DUPLICATE_IOU_THRESHOLD,
        "above": above,
        "below_is_repeat": is_strict_repeat(below),
        "equal_is_repeat": is_strict_repeat(DUPLICATE_IOU_THRESHOLD),
        "above_is_repeat": is_strict_repeat(above),
    }
    require(not result["below_is_repeat"] and not result["equal_is_repeat"] and result["above_is_repeat"],
            "exclusive IoU>.95 boundary invariant failed")
    return result


def write_outputs(source_path: Path, output: Path, *, universe_path: Path | None = DEFAULT_UNIVERSE,
                  review_limit: int = REVIEW_LIMIT_PER_STRATUM, render: bool = True) -> dict[str, Any]:
    census = run_census(source_path, universe_path=universe_path, review_limit=review_limit)
    output = output.resolve()
    census_path = output / "census.json"
    publish(census_path, census)
    replay = {
        "schema": REPLAY_SCHEMA,
        "status": "cpu_ready_no_model_calls",
        "source": census["source"],
        "frozen_universe": census["frozen_universe"],
        "source_population": census["original_source_population"],
        "output_cap": CAP,
        "predicate": census["predicate"],
        "review_manifest": {"path": str((output / "review-manifest.json").resolve()), "units": len(census["review_selection"])},
        "consumer": {
            "module": str(Path(__file__).resolve()),
            "command": f"python {Path(__file__).resolve()} consume --census {census_path}",
            "threshold_boundary_test": "exclusive_gt_0.95",
        },
        "launch_gate": {
            "new_model_calls": False,
            "training": False,
            "new_negative_labels": False,
            "new_sample_expansion": False,
            "gpus_returned": [4, 5],
        },
    }
    publish(output / "replay-packet.json", replay)
    review_manifest = {
        "schema": "native_owner_scale.repeat.review_manifest.v1",
        "source": census["source"],
        "predicate": census["predicate"],
        "selection": census["review_selection"],
        "review_policy": {
            "supported_seed": "seed has same-category GT overlap >= .50; annotation-relative support only",
            "unknown_seed": "seed is GT-unmatched/unknown; do not call hallucinated without visual review",
            "subthreshold": "visually inspect only; never convert to an authorized negative label",
            "full_canvas": True,
            "cropped_short_output_proxy": False,
        },
    }
    publish(output / "review-manifest.json", review_manifest)
    if render:
        render_cards(census["review_selection"], output / "cards")
    return census


def render_cards(selection: Sequence[Mapping[str, Any]], card_dir: Path) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to render full-canvas review cards") from exc
    card_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for item in selection:
        image_path = Path(str(item["image_path"]))
        require(image_path.is_file(), f"review image missing: {image_path}")
        if item.get("image_content_sha256"):
            require(file_hash(image_path) == item["image_content_sha256"], f"review image hash differs: {image_path}")
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            max_width = 1400
            scale = min(1.0, max_width / source.width)
            canvas = source.resize((round(source.width * scale), round(source.height * scale))) if scale < 1.0 else source.copy()
        draw = ImageDraw.Draw(canvas)
        line_width = max(2, round(4 * scale))
        def draw_box(box: Sequence[float], colour: tuple[int, int, int], label: str) -> None:
            coords = tuple(round(float(v) * scale) for v in box)
            draw.rectangle(coords, outline=colour, width=line_width)
            draw.text((coords[0] + 3, coords[1] + 3), label, fill=colour)
        draw_box(item["seed_bbox"], (255, 165, 0), f"seed {item['seed_pred_index']}")
        draw_box(item["later_bbox"], (255, 0, 0), f"later {item['later_pred_index']}")
        header = (
            f"{item['review_unit_id']}  {item['relation']}  IoU={item['best_iou']:.6f}  "
            f"seed={item['seed_support']}  later={item['later_support']}  "
            "(review only; no negative label)"
        )
        header_height = 38
        framed = Image.new("RGB", (canvas.width, canvas.height + header_height), "white")
        framed.paste(canvas, (0, header_height))
        ImageDraw.Draw(framed).text((8, 8), header, fill=(0, 0, 0))
        name = f"{item['review_unit_id']}-{item['example_id']}-{item['later_pred_index']}.png"
        path = card_dir / name
        framed.save(path, format="PNG", optimize=False)
        rendered.append({
            "review_unit_id": item["review_unit_id"],
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
            "full_canvas_size": [framed.width, framed.height],
        })
    publish(card_dir.parent / "cards-manifest.json", {
        "schema": "native_owner_scale.repeat.cards_manifest.v1",
        "cards": rendered,
        "source_review_units": len(selection),
    })
    return rendered


def consume(census_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    census_path = census_path.resolve()
    census = load_json(census_path)
    require(census.get("schema") == SCHEMA, "wrong census schema")
    source = Path(census["source"]["path"])
    require(file_hash(source) == census["source"]["sha256"], "source packet changed since census")
    universe = census.get("frozen_universe")
    universe_path = Path(universe["path"]) if isinstance(universe, Mapping) else None
    if universe_path is not None:
        require(file_hash(universe_path) == universe["sha256"], "frozen universe changed since census")
    fresh = run_census(source, universe_path=universe_path, review_limit=REVIEW_LIMIT_PER_STRATUM)
    require(fresh["predicate"] == census["predicate"], "predicate changed on cold replay")
    require(fresh["counts"] == census["counts"], "aggregate counts changed on cold replay")
    require(fresh["records"] == census["records"], "per-record census changed on cold replay")
    require(fresh["review_selection"] == census["review_selection"], "review selection changed on cold replay")
    boundary = _assert_boundary_invariant()
    result = {
        "schema": CONSUMER_SCHEMA,
        "status": "passed",
        "census": {"path": str(census_path), "sha256": file_hash(census_path)},
        "source": census["source"],
        "counts": census["counts"],
        "threshold_boundary": boundary,
        "consumer_claim": "strict rows are census-only; subthreshold candidates remain unknown review evidence",
    }
    if output_path is None:
        output_path = census_path.parent / "consumer.json"
    publish(output_path.resolve(), result)
    return result


def _load_lineage_tokenizer(source_raw: Any) -> tuple[Any | None, dict[str, Any]]:
    """Load only the local tokenizer needed to map literal spans to token offsets.

    This is intentionally a tokenizer-only operation: no model or adapter is
    loaded and no generation is performed.  If a later source packet does not
    expose a local tokenizer identity, the caller still emits literal-span
    lineage and records the token-position gap instead of guessing.
    """
    model_path: str | None = None
    if isinstance(source_raw, Mapping):
        model = source_raw.get("model")
        if isinstance(model, Mapping):
            value = model.get("base_model_path") or model.get("base_model")
            if isinstance(value, str) and value:
                model_path = value
        if model_path is None:
            config = source_raw.get("config")
            if isinstance(config, Mapping):
                config_model = config.get("model")
                if isinstance(config_model, Mapping):
                    value = config_model.get("base_model") or config_model.get("base_model_path")
                    if isinstance(value, str) and value:
                        model_path = value
    identity: dict[str, Any] = {
        "path": model_path,
        "local_only": True,
        "use_fast": True,
        "status": "unavailable",
    }
    if model_path is None:
        identity["reason"] = "source packet has no tokenizer/model path"
        return None, identity
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    except Exception as exc:  # pragma: no cover - exercised by missing runtime assets
        identity["reason"] = f"local tokenizer load failed: {type(exc).__name__}: {exc}"
        return None, identity
    identity.update({
        "status": "loaded",
        "name_or_path": str(getattr(tokenizer, "name_or_path", model_path)),
        "class": type(tokenizer).__name__,
    })
    tokenizer_json = Path(model_path) / "tokenizer.json"
    if tokenizer_json.is_file():
        identity["tokenizer_json"] = binding(tokenizer_json)
    return tokenizer, identity


def _span_lineage(
    record: Mapping[str, Any],
    pred_index: int,
    *,
    tokenizer: Any | None,
    tokenizer_ids: Sequence[int] | None,
    token_offsets: Sequence[Sequence[int]] | None,
    prompt_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Return exact literal and, when possible, generated-token span lineage."""
    parsed = record["parsed"]
    raw_text = parsed.get("raw_decode_text")
    pred = parsed.get("pred", [])[pred_index]
    raw_span = pred.get("raw_span_text")
    char_start = pred.get("char_start")
    char_end = pred.get("char_end")
    literal_ok = (
        isinstance(raw_text, str)
        and isinstance(raw_span, str)
        and type(char_start) is int
        and type(char_end) is int
        and 0 <= char_start <= char_end <= len(raw_text)
        and raw_text[char_start:char_end] == raw_span
    )
    result: dict[str, Any] = {
        "pred_index": int(pred_index),
        "generated_order": pred.get("generated_order"),
        "description": pred.get("description"),
        "raw_char_start": char_start,
        "raw_char_end": char_end,
        "raw_span_text": raw_span,
        "raw_span_sha256": text_hash(raw_span) if isinstance(raw_span, str) else None,
        "source_declared_raw_span_sha256": pred.get("raw_span_sha256"),
        "literal_span_status": "exact" if literal_ok else "gap",
        "token_span_status": "unavailable",
        "token_start": None,
        "token_end_exclusive": None,
        "token_count": None,
        "token_ids_sha256": None,
        "deployment_prefix": {
            "prompt_token_count": len(prompt_token_ids),
            "prompt_token_ids_sha256": digest(list(prompt_token_ids)),
            "generated_tokens_before_row": None,
            "combined_prefix_token_count": None,
            "prefix_generated_ids_sha256": None,
            "boundary_verified": False,
        },
    }
    if not literal_ok:
        result["token_span_status"] = "gap_literal_span_invalid"
        return result
    if tokenizer is None or tokenizer_ids is None or token_offsets is None:
        result["token_span_status"] = "gap_tokenizer_unavailable"
        return result
    if list(tokenizer_ids) != list(record["token_ids"]):
        result["token_span_status"] = "gap_tokenizer_trace_mismatch"
        return result
    covered = [
        index
        for index, offset in enumerate(token_offsets)
        if len(offset) == 2
        and type(offset[0]) is int
        and type(offset[1]) is int
        and offset[1] > offset[0]
        and offset[0] >= char_start
        and offset[1] <= char_end
    ]
    contiguous = bool(covered) and covered == list(range(covered[0], covered[-1] + 1))
    exact_offsets = (
        contiguous
        and token_offsets[covered[0]][0] == char_start
        and token_offsets[covered[-1]][1] == char_end
    )
    if not exact_offsets:
        result["token_span_status"] = "gap_token_offsets_do_not_cover_literal_span"
        return result
    token_start = covered[0]
    token_end = covered[-1] + 1
    token_slice = list(record["token_ids"])[token_start:token_end]
    prefix_slice = list(record["token_ids"])[:token_start]
    result.update({
        "token_span_status": "exact",
        "token_start": token_start,
        "token_end_exclusive": token_end,
        "token_count": token_end - token_start,
        "token_ids_sha256": digest(token_slice),
    })
    result["deployment_prefix"].update({
        "generated_tokens_before_row": token_start,
        "combined_prefix_token_count": len(prompt_token_ids) + token_start,
        "prefix_generated_ids_sha256": digest(prefix_slice),
        "boundary_verified": True,
    })
    return result


def write_lineage(census_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Persist strict-row literal/token-position lineage without model calls."""
    census_path = census_path.resolve()
    census = load_json(census_path)
    require(census.get("schema") == SCHEMA, "wrong census schema")
    source = Path(census["source"]["path"]).resolve()
    require(file_hash(source) == census["source"]["sha256"], "source packet changed since census")
    universe = census.get("frozen_universe")
    universe_path = Path(universe["path"]).resolve() if isinstance(universe, Mapping) else None
    if universe_path is not None:
        require(file_hash(universe_path) == universe["sha256"], "frozen universe changed since census")
    source_raw = load_json(source)
    raw_rows, _ = _source_rows(source_raw)
    raw_by_id = {_record_id(row): row for row in raw_rows}
    records, _ = _load_records(source, universe_path)
    record_by_id = {record["example_id"]: record for record in records}
    tokenizer, tokenizer_identity = _load_lineage_tokenizer(source_raw)
    rows: list[dict[str, Any]] = []
    gap_reasons = Counter()
    for census_record in census.get("records", []):
        example_id = str(census_record["example_id"])
        record = record_by_id[example_id]
        raw_record = raw_by_id[example_id]
        prompt_ids = raw_record.get("prompt_token_ids", [])
        if not isinstance(prompt_ids, list) or not all(type(value) is int and value >= 0 for value in prompt_ids):
            prompt_ids = []
        tokenizer_ids: Sequence[int] | None = None
        token_offsets: Sequence[Sequence[int]] | None = None
        if tokenizer is not None:
            raw_text = record["parsed"].get("raw_decode_text")
            if isinstance(raw_text, str):
                try:
                    encoded = tokenizer(raw_text, add_special_tokens=False, return_offsets_mapping=True)
                    tokenizer_ids = list(encoded["input_ids"])
                    token_offsets = list(encoded["offset_mapping"])
                except Exception:
                    tokenizer_ids = None
                    token_offsets = None
        for pair in census_record.get("strict_repeat_rows", []):
            seed = _span_lineage(
                record, int(pair["seed_pred_index"]), tokenizer=tokenizer,
                tokenizer_ids=tokenizer_ids, token_offsets=token_offsets,
                prompt_token_ids=prompt_ids,
            )
            later = _span_lineage(
                record, int(pair["later_pred_index"]), tokenizer=tokenizer,
                tokenizer_ids=tokenizer_ids, token_offsets=token_offsets,
                prompt_token_ids=prompt_ids,
            )
            for span in (seed, later):
                if span["token_span_status"] != "exact":
                    gap_reasons[span["token_span_status"]] += 1
            rows.append({
                "example_id": example_id,
                "image_id": record["image_id"],
                "source_record_digest": record["source_record_digest"],
                "token_trace": {
                    "source_path": record["source_path"],
                    "stable_token_ids_sha256": record["token_ids_sha256"],
                    "stable_token_count": len(record["token_ids"]),
                    "stop_reason": record["stop_reason"],
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_ids_sha256": digest(prompt_ids),
                },
                "strict_pair": {
                    "seed_pred_index": int(pair["seed_pred_index"]),
                    "later_pred_index": int(pair["later_pred_index"]),
                    "iou": float(pair["best_iou"]),
                    "seed": seed,
                    "later": later,
                },
            })
    exact_spans = sum(
        int(span["token_span_status"] == "exact")
        for row in rows
        for span in (row["strict_pair"]["seed"], row["strict_pair"]["later"])
    )
    total_spans = 2 * len(rows)
    result = {
        "schema": LINEAGE_SCHEMA,
        "status": "complete_exact_token_positions" if exact_spans == total_spans else "literal_spans_with_token_gaps",
        "census": binding(census_path),
        "source": census["source"],
        "frozen_universe": census.get("frozen_universe"),
        "tokenizer": tokenizer_identity,
        "prefix_definition": (
            "The deployment prefix is prompt_token_ids followed by stable generated token IDs before "
            "the row's token_start; token_end_exclusive identifies the row span. Prefix IDs are "
            "reconstructable from the bound source packet and the recorded positions/hashes."
        ),
        "counts": {
            "strict_rows": len(rows),
            "span_records": total_spans,
            "exact_token_position_spans": exact_spans,
            "literal_gap_spans": total_spans - exact_spans,
            "gap_reasons": dict(sorted(gap_reasons.items())),
        },
        "rows": rows,
        "claim_boundary": (
            "Lineage is source reconstruction only. It does not add rows, alter the frozen review "
            "selection, or authorize labels/training; raw source records remain the deployment evidence."
        ),
    }
    if output_path is None:
        output_path = census_path.parent / "lineage.json"
    publish(output_path.resolve(), result)
    return result


def project_raw128(
    source_path: Path,
    census_path: Path,
    *,
    review_manifest_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Census an A-owned raw128 subset without selecting or rendering new cards."""
    source_path = source_path.resolve()
    census_path = census_path.resolve()
    census = load_json(census_path)
    require(census.get("schema") == SCHEMA, "wrong source384 census schema")
    require(file_hash(Path(census["source"]["path"])) == census["source"]["sha256"],
            "source384 packet changed since census")
    universe = census.get("frozen_universe")
    require(isinstance(universe, Mapping), "source384 census has no frozen universe binding")
    universe_path = Path(universe["path"]).resolve()
    require(file_hash(universe_path) == universe["sha256"], "frozen universe changed since census")
    if review_manifest_path is None:
        review_manifest_path = census_path.parent / "review-manifest.json"
    review_manifest_path = review_manifest_path.resolve()
    review_manifest = load_json(review_manifest_path)
    require(review_manifest.get("schema") == "native_owner_scale.repeat.review_manifest.v1",
            "wrong frozen review manifest schema")
    require(review_manifest.get("selection") == census.get("review_selection"),
            "frozen review selection differs from source384 census")
    source_raw = load_json(source_path)
    selection_projection = isinstance(source_raw, Mapping) and source_raw.get("schema") == "native_owner_scale.selection.v2"
    selected_ids: list[str] | None = None
    if selection_projection:
        selected = source_raw.get("selected")
        require(isinstance(selected, list) and len(selected) == 128, "A selection-v2 source must contain 128 selected identities")
        selected_ids = [str(item.get("example_id")) for item in selected if isinstance(item, Mapping)]
        require(len(selected_ids) == len(selected) and len(set(selected_ids)) == len(selected_ids),
                "A selection-v2 source has missing or duplicate selected identities")
        source_records = {str(item["example_id"]): item for item in census.get("records", [])}
        missing = sorted(set(selected_ids) - set(source_records))
        require(not missing, f"A selection-v2 source expands beyond the computed census: {missing[:3]}")
        projected_records = [source_records[example_id] for example_id in selected_ids]
        aggregate = Counter()
        for record in projected_records:
            aggregate.update(record["counts"])
        strict_images = sorted({
            record["image_id"] for record in projected_records
            if record["counts"]["strict_repeat_rows"]
        })
        projected_counts = {
            "source_records": len(projected_records),
            "eligible": len(projected_records),
            "executed": len(projected_records),
            "valid": int(aggregate["valid_complete_rows"]),
            "admitted": int(aggregate["strict_repeat_rows"]),
            "held": int(aggregate["subthreshold_candidates"]),
            "strict_repeat_images": len(strict_images),
            "strict_repeat_rows": int(aggregate["strict_repeat_rows"]),
            "subthreshold_candidates": int(aggregate["subthreshold_candidates"]),
            "invalid": int(aggregate["invalid"]),
            "invalid_geometry_rows": int(aggregate["invalid_geometry_rows"]),
            "invalid_category_rows": int(aggregate["invalid_category_rows"]),
            "malformed_rows": int(aggregate["malformed_rows"]),
            "parser_drop_rows": int(aggregate["parser_drop_rows"]),
            "cap": int(aggregate["cap"]),
            "eos": int(aggregate["eos"]),
            "complete_token_length": int(aggregate["complete_token_length"]),
            "declared_invalid_predictions": int(aggregate["declared_invalid_predictions"]),
        }
        projected = {
            "source": binding(source_path),
            "frozen_universe": census["frozen_universe"],
            "predicate": census["predicate"],
            "counts": projected_counts,
            "strict_repeat_images": strict_images,
            "records": projected_records,
            "review_selection": [],
            "output_cap": census["output_cap"],
        }
    else:
        projected = run_census(source_path, universe_path=universe_path, review_limit=REVIEW_LIMIT_PER_STRATUM)
    count_keys = (
        "source_records", "eligible", "executed", "valid", "admitted", "held",
        "strict_repeat_images", "strict_repeat_rows", "subthreshold_candidates",
        "invalid", "invalid_geometry_rows", "invalid_category_rows",
        "malformed_rows", "parser_drop_rows", "cap", "eos",
        "complete_token_length", "declared_invalid_predictions",
    )
    output = {
        "schema": PROJECTION_SCHEMA,
        "status": "candidate_cpu_verified_no_new_review_selection",
        "source384_census": binding(census_path),
        "source": projected["source"],
        "source_schema": source_raw.get("schema") if isinstance(source_raw, Mapping) else "jsonl_records",
        "projection_kind": (
            "existing_source384_census_filtered_by_A_selection_v2"
            if selection_projection else "recomputed_from_A_source_packet"
        ),
        "a_selection_identity": {
            "selected_count": len(selected_ids) if selected_ids is not None else None,
            "selected_ids_sha256": digest(selected_ids) if selected_ids is not None else None,
            "manifest_counts": source_raw.get("counts") if selection_projection else None,
        },
        "frozen_universe": projected["frozen_universe"],
        "frozen_review_manifest": binding(review_manifest_path),
        "frozen_review_selection": {
            "units": len(census["review_selection"]),
            "selection_sha256": digest(census["review_selection"]),
            "reused_without_change": True,
            "new_cards_or_units_published": False,
        },
        "predicate": projected["predicate"],
        "output_cap": CAP,
        "counts": {key: projected["counts"].get(key) for key in count_keys},
        "strict_repeat_images": projected["strict_repeat_images"],
        "records": projected["records"],
        "discarded_internal_review_selection": {
            "units": len(projected["review_selection"]),
            "published": False,
            "computed": not selection_projection,
            "reason": "projection must not change B's frozen 15-card review selection",
        },
        "claim_boundary": (
            "This is an A-owned raw128 projection constrained to the frozen 384 "
            "universe. For selection-v2 input it is an identity projection of the "
            "already-computed source384 Stable50 census, not a new rollout. It does "
            "not expand samples, nominate cards, authorize labels, or test "
            "negative-learning efficacy."
        ),
    }
    if output_path is None:
        output_path = census_path.parent / "raw128-projection.json"
    publish(output_path.resolve(), output)
    return output


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("census", help="replay source rows and render review cards")
    c.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    c.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    c.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    c.add_argument("--review-limit", type=int, default=REVIEW_LIMIT_PER_STRATUM)
    c.add_argument("--no-render", action="store_true")
    v = sub.add_parser("consume", help="cold-replay census with real consumer checks")
    v.add_argument("--census", type=Path, required=True)
    v.add_argument("--output", type=Path)
    l = sub.add_parser("lineage", help="write strict-row literal/token-position lineage; no model calls")
    l.add_argument("--census", type=Path, required=True)
    l.add_argument("--output", type=Path)
    x = sub.add_parser("project", help="census an A-owned raw128 subset without changing frozen review cards")
    x.add_argument("--source", type=Path, required=True)
    x.add_argument("--census", type=Path, required=True)
    x.add_argument("--review-manifest", type=Path)
    x.add_argument("--output", type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "census":
        census = write_outputs(args.source.resolve(), args.output.resolve(), universe_path=args.universe.resolve(),
                               review_limit=args.review_limit, render=not args.no_render)
        print(json.dumps({"status": census["status"], "output": str(args.output.resolve()), "counts": census["counts"]}, sort_keys=True))
        return 0
    if args.command == "consume":
        result = consume(args.census, args.output)
        print(json.dumps({"status": result["status"], "counts": result["counts"], "threshold_boundary": result["threshold_boundary"]}, sort_keys=True))
        return 0
    if args.command == "project":
        projection = project_raw128(
            args.source,
            args.census,
            review_manifest_path=args.review_manifest,
            output_path=args.output,
        )
        print(json.dumps({"status": projection["status"], "counts": projection["counts"]}, sort_keys=True))
        return 0
    lineage = write_lineage(args.census, args.output)
    print(json.dumps({"status": lineage["status"], "counts": lineage["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
