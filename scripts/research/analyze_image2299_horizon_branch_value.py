#!/usr/bin/env python3
"""Offline horizon-four branch-value analysis for image 2299.

The replay runner writes one receipt per branch arm and one call bundle per
request.  This reader intentionally does not run inference or infer physical
ownership from a branch label.  The arm mapping is supplied explicitly with
``--arm RECEIPT:BRANCH_RANK:LABEL[:GREEDY]`` (or an equivalent manifest stored
in a receipt).  Only the first four complete generated rows are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.analyze_image2299_near_complete_relabel_successor_transition import (  # noqa: E402
    coord_bins_to_pixel_box,
    intersection_over_union,
)


SCHEMA_VERSION = "image2299_horizon_branch_value_analysis.v1"
HORIZON = 4
PARENT_PERSON_RANKS = frozenset({0, 1})
PARENT_GREEDY_PERSON_RANK = 14
PARENT_GREEDY_OWNER_LABEL = "owner0012"
IOU_FLOOR = 0.5
IOU_MARGIN_FLOOR = 0.05
BOOTSTRAP_SEED_ROOT = 2026071804000001
BOOTSTRAP_REPLICATES = 100_000
_TOKEN_RE = re.compile(r"^<\|coord_(\d+)\|>$")
_OWNER_RANKS = {"owner0003": 3, "owner0004": 2, "owner0006": 4, "owner0012": 14}
_NATURAL_STOP_REASONS = frozenset(
    {
        "eos",
        "eos_token",
        "end_of_sequence",
        "im_end",
        "natural",
        "stop",
        "terminated",
    }
)


@dataclass(frozen=True)
class ArmSpec:
    """Explicit mapping from a replay receipt to its physical branch arm."""

    receipt: Path
    branch_rank: int
    label: str
    parent_greedy_branch: bool = False

    def artifact(self) -> dict[str, Any]:
        return {
            "receipt": str(self.receipt),
            "branch_person_rank": int(self.branch_rank),
            "label": self.label,
            "parent_greedy_branch": bool(self.parent_greedy_branch),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_category(value: Any) -> str:
    """Normalize only case and whitespace; never apply semantic aliases."""

    return " ".join(str(value).strip().casefold().split())


def _coord_value(value: Any) -> int:
    text = str(value)
    match = _TOKEN_RE.fullmatch(text)
    if match is not None:
        number = int(match.group(1))
    else:
        number = int(value)
    if not 0 <= number <= 999:
        raise ValueError(f"coordinate outside 0..999: {value!r}")
    return number


def _object_box(source: Mapping[str, Any], *, width: int, height: int) -> list[float]:
    for key in ("pixel_box", "bbox", "bbox_xyxy", "source_canvas_box_xyxy"):
        value = source.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
            return [float(item) for item in value]
    coord_source = source.get("bbox_2d", source.get("coord_bins"))
    if not isinstance(coord_source, Sequence) or isinstance(coord_source, (str, bytes)) or len(coord_source) != 4:
        raise ValueError("relabel object lacks a four-coordinate box")
    bins = [_coord_value(item) for item in coord_source]
    return [float(item) for item in coord_bins_to_pixel_box(bins, width=width, height=height)]


def build_relabel_objects(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build zero-based per-category physical ranks from one val.coord record."""

    width = int(record["width"])
    height = int(record["height"])
    sources = record.get("objects")
    if not isinstance(sources, list):
        raise ValueError("source record lacks objects")
    counts: Counter[str] = Counter()
    objects: list[dict[str, Any]] = []
    for serialized_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"relabel object {serialized_index} is not an object")
        raw_category = source.get("desc", source.get("description", source.get("category_name")))
        if raw_category is None:
            raise ValueError(f"relabel object {serialized_index} lacks a category")
        category = normalize_category(raw_category)
        rank = int(counts[category])
        counts[category] += 1
        objects.append(
            {
                "serialized_index": serialized_index,
                "category": category,
                "description": str(raw_category),
                "category_rank": rank,
                "annotation_id": source.get("coco_ann_id", source.get("annotation_id")),
                "pixel_box": _object_box(source, width=width, height=height),
            }
        )
    return objects


def load_source_record(path: Path, image_id: str) -> tuple[dict[str, Any], str]:
    """Load the requested JSONL record and hash its exact source line."""

    target = str(image_id)
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if isinstance(value, Mapping) and str(value.get("image_id")) == target:
                return dict(value), _sha256_bytes(raw_line)
    raise ValueError(f"image {image_id} is absent from {path}")


def match_prediction(
    prediction: Mapping[str, Any],
    relabel_objects: Sequence[Mapping[str, Any]],
    *,
    minimum_iou: float = IOU_FLOOR,
    minimum_margin: float = IOU_MARGIN_FLOOR,
) -> dict[str, Any]:
    """Match a prediction only to the same normalized category and a unique box."""

    raw_category = prediction.get("description", prediction.get("category", prediction.get("label")))
    if raw_category is None:
        return {"matched": False, "reason": "missing_category"}
    category = normalize_category(raw_category)
    bbox = prediction.get("bbox", prediction.get("bbox_xyxy", prediction.get("pixel_box")))
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return {"matched": False, "reason": "malformed_box", "normalized_category": category}
    try:
        pixel_box = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return {"matched": False, "reason": "malformed_box", "normalized_category": category}
    candidates = [
        (intersection_over_union(pixel_box, obj["pixel_box"]), obj)
        for obj in relabel_objects
        if normalize_category(obj.get("category", obj.get("description"))) == category
    ]
    candidates.sort(key=lambda item: (-float(item[0]), int(item[1].get("serialized_index", 0))))
    if not candidates:
        return {
            "matched": False,
            "reason": "no_same_description_relabel_object",
            "normalized_category": category,
            "best_iou": 0.0,
            "second_iou": 0.0,
            "iou_margin": 0.0,
        }
    best_iou, best = candidates[0]
    second_iou = float(candidates[1][0]) if len(candidates) > 1 else 0.0
    margin = float(best_iou) - second_iou
    if float(best_iou) < float(minimum_iou):
        reason = "best_iou_below_threshold"
    elif margin < float(minimum_margin):
        reason = "top_match_margin_below_threshold"
    else:
        return {
            "matched": True,
            "normalized_category": category,
            "matched_category": normalize_category(best.get("category", best.get("description"))),
            "matched_category_rank": int(best["category_rank"]),
            "matched_annotation_id": best.get("annotation_id"),
            "best_iou": float(best_iou),
            "second_iou": second_iou,
            "iou_margin": margin,
        }
    return {
        "matched": False,
        "reason": reason,
        "normalized_category": category,
        "best_iou": float(best_iou),
        "second_iou": second_iou,
        "iou_margin": margin,
        "best_candidate_category_rank": int(best["category_rank"]),
    }


def _sort_index(item: Mapping[str, Any], fallback: int) -> tuple[int, int]:
    value = item.get("generated_order", item.get("row_index"))
    try:
        return (int(value), fallback)
    except (TypeError, ValueError):
        return (fallback, fallback)


def _parse_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    rank = value.get("branch_person_rank", value.get("emitted_branch_person_rank", value.get("emitted_person_rank")))
    label = value.get("label", value.get("arm_label", value.get("branch_label")))
    greedy = value.get("parent_greedy_branch", value.get("is_parent_greedy_branch", value.get("greedy_branch")))
    if rank is None or label is None or greedy is None:
        return None
    return {"branch_rank": int(rank), "label": str(label), "greedy": _parse_bool(greedy)}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y", "greedy"}:
        return True
    if text in {"0", "false", "no", "n", "sampled", "non-greedy", "nongreedy"}:
        return False
    raise ValueError(f"invalid greedy flag: {value!r}")


def arm_spec_from_receipt_manifest(receipt: Mapping[str, Any], receipt_path: Path) -> ArmSpec | None:
    """Read an explicit arm declaration if the receipt carries one."""

    candidates = [receipt.get(name) for name in ("arm", "arm_manifest", "branch_manifest", "manifest")]
    for candidate in candidates:
        parsed = _parse_mapping(candidate)
        if parsed is not None:
            return ArmSpec(receipt_path, parsed["branch_rank"], parsed["label"], parsed["greedy"])
    return None


def parse_arm_spec(text: str) -> ArmSpec:
    """Parse ``receipt:rank:label[:greedy]`` without guessing missing fields."""

    parts = str(text).split(":", 3)
    if len(parts) not in {3, 4} or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError("--arm must be RECEIPT:BRANCH_RANK:LABEL[:GREEDY]")
    try:
        rank = int(parts[1])
    except ValueError as exc:
        raise ValueError("--arm branch rank must be an integer") from exc
    if rank < 0:
        raise ValueError("--arm branch rank must be non-negative")
    greedy = _parse_bool(parts[3]) if len(parts) == 4 else False
    return ArmSpec(Path(parts[0]).expanduser().resolve(), rank, parts[2], greedy)


def resolve_arm_specs(
    receipts: Sequence[Path],
    arm_texts: Sequence[str] | None = None,
) -> list[ArmSpec]:
    """Resolve explicit CLI arms or explicit receipt manifests."""

    if arm_texts:
        if receipts:
            raise ValueError("use --receipt or --arm, not both")
        specs = [parse_arm_spec(text) for text in arm_texts]
    else:
        if not receipts:
            raise ValueError("at least one --receipt or --arm is required")
        specs = []
        for item in receipts:
            path = item.expanduser().resolve(strict=True)
            receipt = _read_json(path)
            spec = arm_spec_from_receipt_manifest(receipt, path)
            if spec is None:
                raise ValueError(
                    f"receipt lacks explicit branch rank/label/greedy arm manifest: {path}"
                )
            specs.append(spec)
    if len({str(spec.receipt) for spec in specs}) != len(specs):
        raise ValueError("duplicate receipt in arm panel")
    for spec in specs:
        if spec.branch_rank < 0:
            raise ValueError("branch rank must be non-negative")
        owner_match = re.search(r"owner\d{4}", spec.label.casefold())
        if owner_match:
            owner = owner_match.group(0)
            expected = _OWNER_RANKS.get(owner)
            if expected is not None and int(spec.branch_rank) != expected:
                raise ValueError(f"arm {spec.label} declares rank {spec.branch_rank}, expected {expected}")
        if spec.parent_greedy_branch and int(spec.branch_rank) != PARENT_GREEDY_PERSON_RANK:
            raise ValueError("the parent-greedy branch must emit person rank 14")
    greedy = [spec for spec in specs if spec.parent_greedy_branch]
    if len(greedy) != 1:
        raise ValueError(f"expected exactly one parent-greedy branch, found {len(greedy)}")
    return specs


def _bundle_path(call: Mapping[str, Any], receipt_path: Path) -> Path:
    value = call.get("bundle_path", call.get("call_bundle"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"receipt call lacks bundle_path: {receipt_path}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = receipt_path.parent / path
    return path.resolve(strict=True)


def _decode_mode(call: Mapping[str, Any], bundle: Mapping[str, Any]) -> str:
    value = call.get("decode_mode", bundle.get("runtime", {}).get("decode_mode"))
    if value is None:
        value = "sampled" if call.get("sampling_seed", bundle.get("sampling_seed")) is not None else "greedy"
    return str(value).strip().casefold()


def _sampling_seed(call: Mapping[str, Any], bundle: Mapping[str, Any]) -> int | None:
    value = call.get("sampling_seed", bundle.get("sampling_seed"))
    if value is None:
        return None
    return int(value)


def _horizon_events(bundle: Mapping[str, Any]) -> tuple[list[tuple[str, Mapping[str, Any]]], int, int]:
    """Return interleaved complete predictions and malformed drops at H=4."""

    parse = bundle.get("parse_result")
    if not isinstance(parse, Mapping):
        return [], 0, 1
    predictions = [dict(item) for item in parse.get("predictions", []) if isinstance(item, Mapping)]
    drops = [dict(item) for item in parse.get("dropped_predictions", []) if isinstance(item, Mapping)]
    predictions = [
        item for _, item in sorted(enumerate(predictions), key=lambda pair: _sort_index(pair[1], pair[0]))
    ]
    drops = [
        item for _, item in sorted(enumerate(drops), key=lambda pair: _sort_index(pair[1], pair[0]))
    ]
    combined: list[tuple[tuple[int, int], str, Mapping[str, Any]]] = []
    for index, item in enumerate(predictions):
        combined.append((_sort_index(item, index), "prediction", item))
    for index, item in enumerate(drops):
        combined.append((_sort_index(item, index), "malformed", item))
    combined.sort(key=lambda item: item[0])
    events: list[tuple[str, Mapping[str, Any]]] = []
    complete_seen = 0
    malformed_count = 0
    for _, kind, item in combined:
        if kind == "prediction":
            if complete_seen >= HORIZON:
                break
            events.append((kind, item))
            complete_seen += 1
        elif complete_seen < HORIZON:
            events.append((kind, item))
            malformed_count += 1
    reported_drops = int(parse.get("dropped_prediction_count", len(drops)) or 0)
    if not drops and reported_drops and complete_seen < HORIZON:
        missing = min(reported_drops, HORIZON - complete_seen)
        malformed_count += missing
        events.extend(("malformed", {"reason": "parser_dropped_prediction"}) for _ in range(missing))
    if not combined and reported_drops:
        malformed_count = min(reported_drops, HORIZON)
        events = [("malformed", {"reason": "parser_dropped_prediction"}) for _ in range(malformed_count)]
    return events, complete_seen, malformed_count


def _natural_terminal(bundle: Mapping[str, Any], *, complete_count: int, malformed_count: int) -> bool:
    if complete_count >= HORIZON:
        return False
    if malformed_count > 0:
        return False
    projection = bundle.get("horizon_projection")
    if isinstance(projection, Mapping):
        classification = str(projection.get("termination_classification", "")).strip().casefold()
        if classification == "natural_termination_before_horizon":
            return True
        if classification in {"token_limit_truncated", "invalid_or_malformed"}:
            return False
        stop_reason = projection.get("stop_reason")
    else:
        stop_reason = bundle.get("stop_reason")
        decode = bundle.get("decode_result")
        if isinstance(decode, Mapping):
            stop_reason = decode.get("stop_reason", stop_reason)
    reason = str(stop_reason or "").strip().casefold().replace("-", "_")
    return reason in _NATURAL_STOP_REASONS and malformed_count == 0


def classify_call(
    bundle: Mapping[str, Any],
    *,
    arm: ArmSpec,
    relabel_objects: Sequence[Mapping[str, Any]],
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Classify one replay call without assigning unresolved rows."""

    events, complete_count, malformed_count = _horizon_events(bundle)
    person_ranks: list[int] = []
    parent_repeats = 0
    branch_repeats = 0
    within_suffix_repeats = 0
    tie_count = 0
    unresolved_count = 0
    malformed_events = malformed_count
    seen_suffix: set[int] = set()
    row_outcomes: list[dict[str, Any]] = []
    for row_index, (kind, prediction) in enumerate(events):
        if kind == "malformed":
            row_outcomes.append({"row_index": row_index, "outcome": "malformed", "reason": prediction.get("reason")})
            continue
        match = match_prediction(prediction, relabel_objects)
        outcome: dict[str, Any] = {"row_index": row_index, "prediction": dict(prediction), "match": match}
        if not match.get("matched"):
            if str(match.get("reason")) in {"malformed_box", "missing_category"}:
                malformed_events += 1
                outcome["outcome"] = "malformed"
            else:
                unresolved_count += 1
                outcome["outcome"] = "unresolved"
        else:
            category = str(match.get("matched_category", match.get("normalized_category", "")))
            rank = int(match["matched_category_rank"])
            outcome["matched_category"] = category
            outcome["matched_category_rank"] = rank
            if category == "person":
                person_ranks.append(rank)
                if rank in PARENT_PERSON_RANKS:
                    parent_repeats += 1
                    outcome["outcome"] = "parent_repeat"
                elif rank == int(arm.branch_rank):
                    branch_repeats += 1
                    outcome["outcome"] = "branch_repeat"
                elif rank in seen_suffix:
                    within_suffix_repeats += 1
                    outcome["outcome"] = "within_suffix_repeat"
                else:
                    outcome["outcome"] = "new_person"
                seen_suffix.add(rank)
            elif category == "tie":
                tie_count += 1
                outcome["outcome"] = "tie"
            else:
                outcome["outcome"] = "matched_other_category"
        row_outcomes.append(outcome)
    terminal_count = int(_natural_terminal(bundle, complete_count=complete_count, malformed_count=malformed_count))
    # A parser can report a malformed status without exposing a dropped row.
    parse = bundle.get("parse_result")
    parser_status = str(bundle.get("parser_status", ""))
    if isinstance(parse, Mapping):
        parser_status = str(parse.get("parse_status", parser_status))
    if not malformed_events and parser_status in {"invalid", "invalid_or_malformed", "accepted_with_drops"}:
        malformed_events = 1
    safety_count = (
        parent_repeats
        + branch_repeats
        + within_suffix_repeats
        + tie_count
        + terminal_count
        + malformed_events
        + unresolved_count
    )
    seed = _sampling_seed({}, bundle)
    return {
        "bundle_path": str(bundle_path.resolve()) if bundle_path is not None else None,
        "sampling_seed": seed,
        "decode_mode": _decode_mode({}, bundle),
        "complete_row_count": int(complete_count),
        "analyzed_row_count": len(events),
        "matched_person_ranks": person_ranks,
        "unique_new_person_count": len({rank for rank in person_ranks if rank not in PARENT_PERSON_RANKS and rank != int(arm.branch_rank)}),
        "parent_repeat_count": parent_repeats,
        "branch_repeat_count": branch_repeats,
        "within_suffix_repeat_count": within_suffix_repeats,
        "tie_count": tie_count,
        "terminal_count": terminal_count,
        "malformed_count": malformed_events,
        "unresolved_count": unresolved_count,
        "safety_count": safety_count,
        "row_outcomes": row_outcomes,
    }


def _load_arm_calls(spec: ArmSpec, *, image_id: str, relabel_objects: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = spec.receipt.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path)
    if str(receipt.get("image_id")) != str(image_id):
        raise ValueError(f"receipt image id disagrees with --image-id: {receipt_path}")
    calls = receipt.get("calls")
    if not isinstance(calls, list):
        raise ValueError(f"receipt lacks calls: {receipt_path}")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for call in calls:
        if not isinstance(call, Mapping):
            raise ValueError(f"receipt call is not an object: {receipt_path}")
        bundle_path = _bundle_path(call, receipt_path)
        bundle = _read_json(bundle_path)
        if str(bundle.get("image_id")) != str(image_id):
            raise ValueError(f"call bundle image id disagrees with --image-id: {bundle_path}")
        mode = _decode_mode(call, bundle)
        seed = _sampling_seed(call, bundle)
        key = (mode, seed)
        if key in seen:
            raise ValueError(f"duplicate call mode/seed in arm {spec.label}: {key}")
        seen.add(key)
        record = classify_call(bundle, arm=spec, relabel_objects=relabel_objects, bundle_path=bundle_path)
        record["decode_mode"] = mode
        record["sampling_seed"] = seed
        records.append(record)
    records.sort(key=lambda item: (0 if item["decode_mode"] == "sampled" else 1, item["sampling_seed"] is None, item["sampling_seed"] or 0, str(item["bundle_path"])))
    return receipt, records


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def _aggregate_calls(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcome_counts: Counter[str] = Counter()
    for record in records:
        for key in ("parent_repeat_count", "branch_repeat_count", "within_suffix_repeat_count", "tie_count", "terminal_count", "malformed_count", "unresolved_count"):
            if int(record.get(key, 0)):
                outcome_counts[key] += int(record[key])
    sampled = [record for record in records if str(record.get("decode_mode")) == "sampled" and record.get("sampling_seed") is not None]
    greedy = [record for record in records if str(record.get("decode_mode")) == "greedy" or record.get("sampling_seed") is None]
    seed_metrics: dict[str, dict[str, float]] = {}
    for record in sampled:
        seed = str(int(record["sampling_seed"]))
        if seed in seed_metrics:
            raise ValueError(f"duplicate sampled seed {seed}")
        seed_metrics[seed] = {
            "unique_new_person_count": float(record["unique_new_person_count"]),
            "safety_count": float(record["safety_count"]),
        }
    return {
        "call_count": len(records),
        "sampled_call_count": len(sampled),
        "greedy_call_count": len(greedy),
        "sampled_seed_count": len(seed_metrics),
        "unique_new_person_total": sum(int(record["unique_new_person_count"]) for record in records),
        "unique_new_person_mean": _mean(record["unique_new_person_count"] for record in records),
        "safety_count_total": sum(int(record["safety_count"]) for record in records),
        "safety_count_mean": _mean(record["safety_count"] for record in records),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "sampled_seed_metrics": dict(sorted(seed_metrics.items(), key=lambda item: int(item[0]))),
    }


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot quantile an empty sequence")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_difference(
    sampled: Mapping[str, float],
    greedy: Mapping[str, float],
    *,
    seed_root: int = BOOTSTRAP_SEED_ROOT,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Return a deterministic paired sampled-minus-greedy interval."""

    seed_order = sorted(set(sampled) & set(greedy), key=lambda value: int(value) if str(value).lstrip("-").isdigit() else str(value))
    if set(sampled) != set(greedy):
        raise ValueError("paired bootstrap requires identical seed sets")
    if not seed_order:
        raise ValueError("paired bootstrap requires at least one shared sampling seed")
    contrasts = [float(sampled[seed]) - float(greedy[seed]) for seed in seed_order]
    rng = random.Random(int(seed_root))
    n = len(contrasts)
    draws: list[float] = []
    for _ in range(int(replicates)):
        draws.append(sum(contrasts[rng.randrange(n)] for _ in range(n)) / n)
    return {
        "point": sum(contrasts) / n,
        "lower_95": _quantile(draws, 0.025),
        "upper_95": _quantile(draws, 0.975),
        "paired_seed_count": n,
        "replicates": int(replicates),
        "bootstrap_seed_root": int(seed_root),
        "paired_unit": "sampling_seed",
        "direction": "sampled_minus_parent_greedy",
    }


def analyze_arms(
    arms: Sequence[ArmSpec],
    *,
    source_jsonl: Path,
    image_id: str = "2299",
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Analyze an explicit four-arm panel and return one JSON-compatible object."""

    if not arms:
        raise ValueError("arm panel is empty")
    source_record, source_sha256 = load_source_record(source_jsonl, str(image_id))
    relabel_objects = build_relabel_objects(source_record)
    arm_payloads: list[dict[str, Any]] = []
    arm_records: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        receipt, records = _load_arm_calls(arm, image_id=str(image_id), relabel_objects=relabel_objects)
        arm_records[arm.label] = records
        aggregate = _aggregate_calls(records)
        arm_payloads.append(
            {
                **arm.artifact(),
                "receipt_schema_version": receipt.get("schema_version"),
                "receipt_sha256": _sha256_file(arm.receipt),
                "sampling_seeds": sorted(int(item["sampling_seed"]) for item in records if item.get("sampling_seed") is not None),
                "aggregate": aggregate,
                "calls": records,
            }
        )
    greedy_arms = [arm for arm in arms if arm.parent_greedy_branch]
    if len(greedy_arms) != 1:
        raise ValueError("exactly one parent-greedy arm is required")
    greedy_label = greedy_arms[0].label
    greedy_metrics = _aggregate_calls(arm_records[greedy_label])["sampled_seed_metrics"]
    comparisons: dict[str, Any] = {}
    for arm in arms:
        if arm.parent_greedy_branch:
            continue
        sampled_metrics = _aggregate_calls(arm_records[arm.label])["sampled_seed_metrics"]
        sampled_unique = {seed: value["unique_new_person_count"] for seed, value in sampled_metrics.items()}
        greedy_unique = {seed: value["unique_new_person_count"] for seed, value in greedy_metrics.items()}
        sampled_safety = {seed: value["safety_count"] for seed, value in sampled_metrics.items()}
        greedy_safety = {seed: value["safety_count"] for seed, value in greedy_metrics.items()}
        unique_diff = paired_bootstrap_difference(sampled_unique, greedy_unique, replicates=bootstrap_replicates)
        safety_diff = paired_bootstrap_difference(sampled_safety, greedy_safety, seed_root=BOOTSTRAP_SEED_ROOT + 1, replicates=bootstrap_replicates)
        comparisons[arm.label] = {
            "sampled_arm_label": arm.label,
            "greedy_arm_label": greedy_label,
            "sampled_branch_person_rank": int(arm.branch_rank),
            "greedy_branch_person_rank": int(greedy_arms[0].branch_rank),
            "unique_new_person_difference": unique_diff,
            "safety_count_difference": safety_diff,
            # Keep a compact metrics mapping for consumers that iterate estimands.
            "differences": {
                "unique_new_person_count": unique_diff,
                "safety_count": safety_diff,
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": str(image_id),
        "horizon": HORIZON,
        "matching": {
            "normalized_category": "casefold_and_whitespace",
            "minimum_iou": IOU_FLOOR,
            "minimum_top_minus_second_margin": IOU_MARGIN_FLOOR,
            "unmatched_policy": "retain_unresolved",
        },
        "parent": {"covered_person_ranks": sorted(PARENT_PERSON_RANKS)},
        "source": {
            "path": str(source_jsonl.expanduser().resolve()),
            "record_sha256": source_sha256,
            "width": int(source_record["width"]),
            "height": int(source_record["height"]),
            "object_count": len(relabel_objects),
            "category_counts": dict(sorted(Counter(str(item["category"]) for item in relabel_objects).items())),
        },
        "arms": sorted(arm_payloads, key=lambda item: (not bool(item["parent_greedy_branch"]), str(item["label"]))),
        "greedy_arm_label": greedy_label,
        "comparisons": comparisons,
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed_root": BOOTSTRAP_SEED_ROOT,
            "interval_method": "paired_percentile",
            "direction": "sampled_minus_parent_greedy",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", type=Path, default=[])
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="RECEIPT:BRANCH_RANK:LABEL[:GREEDY]",
        help="Explicit arm mapping; repeat once per receipt.",
    )
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--image-id", default="2299")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.bootstrap_replicates) <= 0:
        raise SystemExit("--bootstrap-replicates must be positive")
    try:
        arms = resolve_arm_specs(args.receipt, args.arm)
        if len(arms) != 4:
            raise ValueError(f"expected exactly four receipt arms, found {len(arms)}")
        payload = analyze_arms(
            arms,
            source_jsonl=args.source_jsonl,
            image_id=str(args.image_id),
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
        _write_once(args.output, payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
