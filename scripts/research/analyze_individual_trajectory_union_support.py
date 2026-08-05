#!/usr/bin/env python3
"""Audit fixed-budget support in one greedy and sampled trajectories.

This is deliberately an experiment-local reader for
``current_seeded_sampled_rollouts.v1`` artifacts.  It does not run inference,
perform human review, or promote a shared matching API.  Parsed rows are kept
separate from entity and geometry judgements: an unmatched row enters the
review queue as unresolved unless a duplicate is mechanically evident.

The command line interface is intentionally strict for the frozen panel (one
greedy and sixteen sampled trajectories per image).  The pure functions accept
smaller panels when ``require_full_panel=False`` so synthetic matching tests
can exercise the fixed-budget semantics without fabricating the full cohort.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
import glob
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "individual_trajectory_union_support.v1"
ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
REVIEW_DECISIONS_SCHEMA_VERSION = "individual_trajectory_union_support_review_decisions.v1"
FIXED_BUDGETS = (4, 8, 16, 32)
IOU_THRESHOLD = 0.50
EXPECTED_IMAGE_IDS = (
    "1584",
    "2685",
    "4134",
    "5001",
    "6040",
    "7511",
    "10707",
    "13348",
    "13923",
    "14038",
    "14439",
    "16228",
)
EXPECTED_SAMPLED_SEEDS = tuple(range(21001, 21017))
EXPECTED_GREEDY_SEED = 21000

_REVIEW_ENTITY_STATUSES = frozenset(
    {
        "verified_owner",
        "duplicate_owner",
        "semantic_error",
        "unsupported_hallucination",
        "uncertain",
    }
)

_COORD_RE = re.compile(r"coord_(\d+)")
_ALLOWED_IDENTITY_KNOBS = frozenset(
    {
        "device",
        "decode_mode",
        "do_sample",
        "sampling_enabled",
        "temperature",
        "top_p",
        "seed",
        "seeds",
        "sampling_seed",
        "image_ids",
        "output",
        "output_path",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))


def _validate_review_decisions_document(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize the small, human-authored review overlay.

    The overlay is intentionally separate from automatic matching.  A
    prediction identifier is the only join key; image and trajectory fields
    are repeated and checked later so a copied decision cannot silently apply
    to a different rollout.
    """

    if document.get("schema_version") != REVIEW_DECISIONS_SCHEMA_VERSION:
        raise ValueError(
            "review decisions schema_version must be "
            f"{REVIEW_DECISIONS_SCHEMA_VERSION!r}"
        )
    decisions = document.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("review decisions document must contain a decisions list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = ("prediction_id", "image_id", "trajectory_id", "entity_status", "evidence")
    for index, raw in enumerate(decisions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"review decision {index + 1} is not an object")
        missing = [key for key in required if raw.get(key) is None or raw.get(key) == ""]
        if missing:
            raise ValueError(f"review decision {index + 1} is missing: {', '.join(missing)}")
        prediction_id = str(raw["prediction_id"])
        if prediction_id in seen:
            raise ValueError(f"duplicate review decision key: {prediction_id}")
        seen.add(prediction_id)
        entity_status = str(raw["entity_status"])
        if entity_status not in _REVIEW_ENTITY_STATUSES:
            raise ValueError(f"unsupported review entity_status: {entity_status!r}")
        evidence = raw["evidence"]
        if evidence is None or evidence == "" or evidence == [] or evidence == {}:
            raise ValueError(f"review decision {prediction_id} has empty evidence")
        normalized.append(
            {
                **dict(raw),
                "prediction_id": prediction_id,
                "image_id": str(raw["image_id"]),
                "trajectory_id": str(raw["trajectory_id"]),
                "entity_status": entity_status,
            }
        )
    return normalized


def load_review_decisions(path: str | Path) -> dict[str, Any]:
    """Load a strict human-review overlay and attach immutable provenance."""

    source = Path(path).expanduser().resolve(strict=True)
    document = _read_json(source)
    if not isinstance(document, Mapping):
        raise ValueError("review decisions document must be a JSON object")
    normalized = _validate_review_decisions_document(document)
    return {
        "schema_version": REVIEW_DECISIONS_SCHEMA_VERSION,
        "decisions": normalized,
        "_source_path": str(source),
        "_source_sha256": _sha256_file(source),
    }


def _normalize_category(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _coord_value(value: Any) -> float:
    if isinstance(value, str):
        match = _COORD_RE.search(value)
        if match is None:
            raise ValueError(f"unsupported coordinate token: {value!r}")
        return float(match.group(1))
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    raise ValueError(f"unsupported coordinate value: {value!r}")


def _box(value: Any, *, width: float | None = None, height: float | None = None, normalized: bool = False) -> tuple[float, float, float, float]:
    """Read pixel xyxy boxes and generation-7 ``coord_*`` boxes."""

    if isinstance(value, Mapping):
        for key in ("bbox", "bbox_xyxy", "bbox_2d", "box", "coord_bins"):
            if value.get(key) is not None:
                return _box(value[key], width=width, height=height, normalized=normalized or key in {"bbox_2d", "coord_bins"})
        raise ValueError("box mapping has no bbox/bbox_2d/coord_bins field")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"box must contain four coordinates: {value!r}")
    coords = tuple(_coord_value(item) for item in value)
    is_normalized = normalized or any(isinstance(item, str) for item in value)
    if is_normalized:
        if width is None or height is None:
            raise ValueError("normalized coordinates require image width and height")
        coords = (coords[0] * width / 1000.0, coords[1] * height / 1000.0, coords[2] * width / 1000.0, coords[3] * height / 1000.0)
    x1, y1, x2, y2 = (float(item) for item in coords)
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid xyxy box: {value!r}")
    return (x1, y1, x2, y2)


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    denominator = area_left + area_right - intersection
    return 0.0 if denominator <= 0.0 else intersection / denominator


def _expand_artifact_paths(paths: Iterable[str | Path]) -> list[Path]:
    expanded: set[Path] = set()
    for raw in paths:
        candidate = str(raw)
        matches = [Path(item) for item in glob.glob(candidate)]
        if not matches:
            matches = [Path(candidate)]
        for path in matches:
            path = path.expanduser()
            if path.is_dir():
                expanded.update(item for item in path.rglob("*.json") if item.is_file())
            elif path.is_file():
                expanded.add(path)
            else:
                raise FileNotFoundError(path)
    return sorted((path.resolve() for path in expanded), key=str)


def load_rollout_artifacts(paths: Iterable[str | Path], *, require_decode_mode: bool = False) -> list[dict[str, Any]]:
    """Load and validate one or more rollout JSON artifacts.

    Directories and shell-style globs are accepted because full runs are often
    split into one greedy file plus image shards.
    """

    artifacts: list[dict[str, Any]] = []
    for path in _expand_artifact_paths(paths):
        value = _read_json(path)
        if not isinstance(value, Mapping) or value.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
            continue
        config = value.get("config")
        rows = value.get("rollouts")
        if not isinstance(config, Mapping) or not isinstance(rows, list) or not rows:
            raise ValueError(f"invalid rollout artifact contract: {path}")
        config_mode = config.get("decode_mode")
        if require_decode_mode and config_mode not in {"greedy", "sampled"}:
            raise ValueError(f"rollout artifact lacks explicit decode_mode: {path}")
        artifact = dict(value)
        artifact["_source_path"] = str(path)
        artifact["_source_sha256"] = _sha256_file(path)
        artifact["_artifact_config"] = dict(config)
        normalized_rows: list[dict[str, Any]] = []
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"rollout row is not an object: {path}")
            row = dict(raw_row)
            row_mode = row.get("decode_mode", config_mode)
            if require_decode_mode and row_mode not in {"greedy", "sampled"}:
                raise ValueError(f"rollout row lacks explicit decode_mode: {path}")
            if require_decode_mode and row.get("stop_reason") not in {"im_end", "length"}:
                raise ValueError(f"rollout row lacks natural-stop/right-censor evidence: {path}")
            if config_mode is not None and row_mode is not None and row_mode != config_mode:
                raise ValueError(f"rollout decode_mode differs from artifact config: {path}")
            row["decode_mode"] = row_mode
            row["_artifact_config"] = dict(config)
            row["_source_path"] = str(path)
            normalized_rows.append(row)
        artifact["rollouts"] = normalized_rows
        artifacts.append(artifact)
    if not artifacts:
        raise ValueError("no current_seeded_sampled_rollouts.v1 artifacts found")
    return artifacts


def load_generation7_annotations(path: str | Path, *, image_ids: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load generation-7 JSONL annotations into physical-owner records."""

    requested = None if image_ids is None else {str(item) for item in image_ids}
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source = Path(path).expanduser().resolve(strict=True)
    text = source.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
        rows = decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"annotation row {row_index + 1} is not an object")
        image_id = str(row.get("image_id", row.get("id", "")))
        if not image_id or (requested is not None and image_id not in requested):
            continue
        width = float(row.get("width", 1))
        height = float(row.get("height", 1))
        objects = row.get("objects", row.get("annotations", []))
        if not isinstance(objects, list):
            raise ValueError(f"annotation objects are not a list for image {image_id}")
        for object_index, obj in enumerate(objects):
            if not isinstance(obj, Mapping):
                raise ValueError(f"annotation object is not an object for image {image_id}")
            owner_value = obj.get("owner_id", obj.get("physical_owner_id", obj.get("coco_ann_id", object_index)))
            owner_id = str(owner_value)
            if not owner_id.startswith(f"{image_id}:"):
                owner_id = f"{image_id}:{owner_id}"
            category = _normalize_category(obj.get("category_name", obj.get("desc", obj.get("category"))))
            if not category:
                raise ValueError(f"annotation owner {owner_id} lacks category")
            if obj.get("bbox_2d") is not None:
                box = _box(obj["bbox_2d"], width=width, height=height, normalized=True)
            else:
                box = _box(obj.get("bbox", obj.get("bbox_xyxy")), width=width, height=height, normalized=False)
            result[image_id].append(
                {
                    "owner_id": owner_id,
                    "category": category,
                    "bbox": box,
                    "category_id": obj.get("category_id"),
                    "annotation_index": object_index,
                    "image_id": image_id,
                }
            )
    return {key: value for key, value in result.items()}


def _identity_without_knobs(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _identity_without_knobs(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _ALLOWED_IDENTITY_KNOBS and not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_identity_without_knobs(item) for item in value]
    if isinstance(value, tuple):
        return [_identity_without_knobs(item) for item in value]
    return value


def _trajectory_mode(row: Mapping[str, Any]) -> str:
    mode = row.get("decode_mode")
    if mode in {"greedy", "sampled"}:
        return str(mode)
    config = row.get("_artifact_config")
    if isinstance(config, Mapping) and config.get("decode_mode") in {"greedy", "sampled"}:
        return str(config["decode_mode"])
    temperature = row.get("temperature", config.get("temperature") if isinstance(config, Mapping) else None)
    if temperature is not None:
        return "greedy" if float(temperature) == 0.0 else "sampled"
    return "sampled" if row.get("seed") is not None else "greedy"


def _trajectory_id(row: Mapping[str, Any]) -> str:
    explicit = row.get("trajectory_id") or row.get("request_id")
    if explicit:
        return str(explicit)
    if _trajectory_mode(row) == "greedy":
        return "greedy"
    if row.get("seed") is not None:
        return f"seed-{int(row['seed'])}"
    return f"sample-{row.get('example_id', row.get('image_id', 'unknown'))}"


def _prompt_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "prompt_token_ids_sha256",
            "executed_prompt_token_ids_sha256",
            "executed_media_sha256",
            "image_sha256",
            "image_width",
            "image_height",
        )
        if row.get(key) is not None
    }


def _validate_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rollout rows")
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("image_id"))].append(row)
    for image_id, image_rows in by_image.items():
        identities = [_prompt_identity(row) for row in image_rows]
        for field in {key for item in identities for key in item}:
            values = [item[field] for item in identities if field in item]
            if values and any(value != values[0] for value in values[1:]):
                raise ValueError(f"prompt/image identity mismatch for image {image_id}: {field}")
        signatures = []
        for row in image_rows:
            config = row.get("_artifact_config", {})
            signature = _identity_without_knobs(
                {
                    "config": config,
                    "model_identity": row.get("model_identity"),
                    "checkpoint": row.get("checkpoint"),
                }
            )
            signatures.append(signature)
        if any(item != signatures[0] for item in signatures[1:]):
            raise ValueError(f"checkpoint/config identity mismatch for image {image_id}")
    # The prompt/image identity must also agree across artifact shards for a
    # shared image, not merely within each one.


def _prediction_list(row: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    parser = row.get("predictions", {})
    if isinstance(parser, list):
        return [item for item in parser if isinstance(item, Mapping)], {"parse_status": "accepted"}
    if not isinstance(parser, Mapping):
        return [], {"parse_status": "malformed", "dropped_prediction_count": 1}
    values = parser.get("predictions", parser.get("rows", []))
    if values is None:
        values = []
    if not isinstance(values, list):
        raise ValueError("predictions.predictions must be a list")
    return [item for item in values if isinstance(item, Mapping)], parser


def _parsed_rows(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values, parser = _prediction_list(row)
    image_id = str(row.get("image_id"))
    trajectory_id = _trajectory_id(row)
    complete: list[dict[str, Any]] = []
    parse_status = str(parser.get("parse_status", "accepted"))
    if parse_status not in {"accepted", "accepted_with_drops", "empty", "malformed"}:
        parse_status = "malformed"
    for fallback_index, value in enumerate(values):
        if parse_status == "malformed":
            break
        generated_index = value.get("generated_order", value.get("row_index", fallback_index))
        try:
            generated_index = int(generated_index)
        except (TypeError, ValueError):
            generated_index = fallback_index
        category = _normalize_category(value.get("description", value.get("desc", value.get("category_name", value.get("category")))))
        try:
            if value.get("bbox") is not None or value.get("bbox_xyxy") is not None:
                box = _box(value.get("bbox", value.get("bbox_xyxy")), normalized=False)
            else:
                width = row.get("image_width", row.get("width"))
                height = row.get("image_height", row.get("height"))
                if width is None or height is None:
                    raise ValueError("coordinate-bin prediction lacks image dimensions")
                box = _box(
                    value.get("bbox_2d", value.get("coord_bins")),
                    normalized=True,
                    width=float(width),
                    height=float(height),
                )
        except (TypeError, ValueError):
            # A malformed object span is retained as parser evidence but does
            # not consume a fixed complete-row budget.
            continue
        if not category:
            continue
        prediction_id = str(value.get("object_span_id", value.get("prediction_id", f"{trajectory_id}:row-{generated_index}")))
        complete.append(
            {
                "image_id": image_id,
                "trajectory_id": trajectory_id,
                "decode_mode": _trajectory_mode(row),
                "seed": row.get("seed"),
                "generated_row_index": generated_index,
                "prediction_id": prediction_id,
                "category": category,
                "bbox": box,
                "raw": dict(value),
            }
        )
    complete.sort(key=lambda item: (int(item["generated_row_index"]), str(item["prediction_id"])))
    dropped = parser.get("dropped_prediction_count", 0)
    try:
        dropped_count = int(dropped)
    except (TypeError, ValueError):
        dropped_count = 0
    malformed_attempt_count = (1 if parse_status == "malformed" else 0) + dropped_count
    evidence = {
        "parse_status": parse_status,
        "valid_prediction_count": len(complete),
        "reported_valid_prediction_count": parser.get("valid_prediction_count"),
        "dropped_prediction_count": dropped_count,
        "dropped_predictions": parser.get("dropped_predictions", []),
        "malformed_attempt_count": malformed_attempt_count,
    }
    return complete, evidence


def _malformed_before_budget(
    parser: Mapping[str, Any],
    complete_rows: Sequence[Mapping[str, Any]],
    budget: int,
) -> int:
    """Count parser drops before the fixed complete-row cutoff.

    Parser drops carry generated-order and/or character-span chronology in the
    live artifact.  Character spans take precedence; generated order is used
    only as a same-coordinate fallback.  Missing chronology is conservatively
    placed before the cutoff.
    """

    dropped = [item for item in parser.get("dropped_predictions", []) if isinstance(item, Mapping)]
    complete = [item for item in complete_rows if isinstance(item, Mapping)]
    budget = int(budget)

    def sources(item: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        raw = item.get("raw")
        if isinstance(raw, Mapping):
            yield raw
        yield item

    def char_span(item: Mapping[str, Any]) -> tuple[int, int] | None:
        for source in sources(item):
            try:
                start = int(source["char_start"])
                end = int(source["char_end"])
            except (KeyError, TypeError, ValueError):
                continue
            if start >= 0 and end >= start:
                return start, end
        return None

    def generated_order(item: Mapping[str, Any]) -> int | None:
        for source in sources(item):
            for key in ("generated_order", "generated_row_index"):
                try:
                    if source.get(key) is not None:
                        return int(source[key])
                except (TypeError, ValueError):
                    continue
        return None

    try:
        reported = int(parser.get("dropped_prediction_count", len(dropped)) or 0)
    except (TypeError, ValueError):
        reported = len(dropped)
    missing = max(0, reported - len(dropped))
    if budget <= 0:
        return 0
    if len(complete) < budget:
        return len(dropped) + missing

    complete_with_spans = [(span, item) for item in complete if (span := char_span(item))]
    if len(complete_with_spans) == len(complete):
        complete_with_spans.sort(key=lambda value: value[0])
        boundary_span, boundary_row = complete_with_spans[budget - 1]
        boundary_end = boundary_span[1]
        boundary_order = generated_order(boundary_row)
        malformed_count = 0
        for item in dropped:
            span = char_span(item)
            if span is not None:
                malformed_count += int(span[0] < boundary_end)
                continue
            order = generated_order(item)
            malformed_count += int(
                order < boundary_order
                if order is not None and boundary_order is not None
                else True
            )
        return malformed_count + missing

    complete_with_order = [(order, item) for item in complete if (order := generated_order(item)) is not None]
    if len(complete_with_order) < len(complete):
        return len(dropped) + missing
    complete_with_order.sort(key=lambda value: value[0])
    boundary_order = complete_with_order[budget - 1][0]
    return (
        sum(
            int(order < boundary_order if (order := generated_order(item)) is not None else True)
            for item in dropped
        )
        + missing
    )


def _collect_rows(artifacts: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    trajectory_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    identity_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        model_identity = artifact.get("model_identity")
        for rollout in artifact.get("rollouts", []):
            row = dict(rollout)
            row["model_identity"] = model_identity
            row["_artifact_config"] = dict(artifact.get("_artifact_config", artifact.get("config", {})))
            row["image_id"] = str(row.get("image_id"))
            prompt_metadata = artifact.get("prompt_metadata")
            if isinstance(prompt_metadata, Mapping):
                metadata = prompt_metadata.get(str(row.get("example_id")), prompt_metadata.get(str(row.get("image_id"))))
                if isinstance(metadata, Mapping):
                    for source_key, target_key in (("width", "image_width"), ("height", "image_height"), ("image_sha256", "image_sha256")):
                        if row.get(target_key) is None and metadata.get(source_key) is not None:
                            row[target_key] = metadata[source_key]
            identity_rows.append(row)
            complete, parser_evidence = _parsed_rows(row)
            trajectory_id = _trajectory_id(row)
            evidence_key = (str(row.get("image_id")), trajectory_id)
            if evidence_key in trajectory_evidence:
                raise ValueError(f"duplicate trajectory identity: {evidence_key[0]}:{trajectory_id}")
            trajectory_evidence[evidence_key] = {
                "image_id": str(row.get("image_id")),
                "decode_mode": _trajectory_mode(row),
                "seed": row.get("seed"),
                "stop_reason": str(row.get("stop_reason", row.get("terminal_reason", "unknown"))),
                "natural_stop": str(row.get("stop_reason", row.get("terminal_reason", ""))) in {"im_end", "natural_stop", "eos"},
                "right_censored": str(row.get("stop_reason", row.get("terminal_reason", ""))) in {"length", "max_new_tokens", "right_censored"},
                "complete_row_count": len(complete),
                "parser": parser_evidence,
                "source_path": row.get("_source_path"),
            }
            all_rows.extend(complete)
    _validate_identity(identity_rows)
    return all_rows, trajectory_evidence


def _min_cost_max_cardinality_assignment(predictions: Sequence[Mapping[str, Any]], owners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Exact cardinality-first matching with deterministic min-cost flow."""

    ordered_predictions = sorted(predictions, key=lambda item: (int(item["generated_row_index"]), str(item["prediction_id"])))
    ordered_owners = sorted(owners, key=lambda item: str(item["owner_id"]))
    prediction_count = len(ordered_predictions)
    owner_count = len(ordered_owners)
    if prediction_count == 0 or owner_count == 0:
        return []
    source = 0
    prediction_base = 1
    owner_base = prediction_base + prediction_count
    sink = owner_base + owner_count
    node_count = sink + 1
    graph: list[list[list[float | int]]] = [[] for _ in range(node_count)]

    def add_edge(start: int, end: int, capacity: int, cost: float, metadata: tuple[int, int] | None = None) -> None:
        graph[start].append([end, len(graph[end]), capacity, cost, metadata])
        graph[end].append([start, len(graph[start]) - 1, 0, -cost, None])

    for prediction_index in range(prediction_count):
        add_edge(source, prediction_base + prediction_index, 1, 0.0)
    for owner_index in range(owner_count):
        add_edge(owner_base + owner_index, sink, 1, 0.0)
    for prediction_index, prediction in enumerate(ordered_predictions):
        for owner_index, owner in enumerate(ordered_owners):
            if prediction["category"] != owner["category"]:
                continue
            overlap = _iou(prediction["bbox"], owner["bbox"])
            if overlap < IOU_THRESHOLD:
                continue
            # Tie perturbation is strictly below any cardinality or IoU term;
            # graph insertion and sorted IDs remain the final deterministic tie.
            tie = 1e-10 / float(1 + prediction_index * (owner_count + 1) + owner_index)
            add_edge(prediction_base + prediction_index, owner_base + owner_index, 1, -overlap - tie, (prediction_index, owner_index))

    while True:
        distance = [math.inf] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distance[source] = 0.0
        queue: deque[int] = deque([source])
        in_queue = {source}
        while queue:
            node = queue.popleft()
            in_queue.discard(node)
            for edge_index, edge in enumerate(graph[node]):
                target, _, capacity, cost, _ = edge
                if int(capacity) <= 0 or distance[target] <= distance[node] + float(cost) + 1e-15:
                    continue
                distance[target] = distance[node] + float(cost)
                previous[target] = (node, edge_index)
                if target not in in_queue:
                    queue.append(target)
                    in_queue.add(target)
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            prior, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[prior][edge_index]
            edge[2] = int(edge[2]) - 1
            reverse_index = int(edge[1])
            graph[node][reverse_index][2] = int(graph[node][reverse_index][2]) + 1
            node = prior

    matches: list[dict[str, Any]] = []
    for prediction_index, prediction in enumerate(ordered_predictions):
        node = prediction_base + prediction_index
        for edge in graph[node]:
            target, _, capacity, _, metadata = edge
            if not (owner_base <= int(target) < sink) or metadata is None or int(capacity) != 0:
                continue
            owner_index = int(target) - owner_base
            owner = ordered_owners[owner_index]
            matches.append(
                {
                    "prediction_id": str(prediction["prediction_id"]),
                    "generated_row_index": int(prediction["generated_row_index"]),
                    "owner_id": str(owner["owner_id"]),
                    "owner_category": str(owner["category"]),
                    "owner_bbox": tuple(owner["bbox"]),
                    "category": str(prediction["category"]),
                    "intersection_over_union": float(_iou(prediction["bbox"], owner["bbox"])),
                }
            )
    return sorted(matches, key=lambda item: (int(item["generated_row_index"]), str(item["prediction_id"]), str(item["owner_id"])))


def match_prefix(predictions: Sequence[Mapping[str, Any]], owners: Sequence[Mapping[str, Any]], budget: int) -> dict[str, Any]:
    """Match exactly the first ``budget`` complete rows, recomputing globally."""

    prefix = sorted(predictions, key=lambda item: (int(item["generated_row_index"]), str(item["prediction_id"])))[: int(budget)]
    matches = _min_cost_max_cardinality_assignment(prefix, owners)
    matched_predictions = {str(item["prediction_id"]) for item in matches}
    matched_owners = {str(item["owner_id"]) for item in matches}
    # An exact-category match is not necessarily an unambiguous physical
    # owner.  Keep such rows out of committed coverage until a human review
    # selects the owner; otherwise duplicate same-category objects can make
    # the support count look better than the evidence warrants.
    ambiguous_matches: dict[str, dict[str, Any]] = {}
    for match in matches:
        prediction = next(
            item for item in prefix if str(item["prediction_id"]) == str(match["prediction_id"])
        )
        candidates = [
            (str(owner["owner_id"]), float(_iou(prediction["bbox"], owner["bbox"])))
            for owner in owners
            if str(owner["category"]) == str(prediction["category"])
            and _iou(prediction["bbox"], owner["bbox"]) >= IOU_THRESHOLD
        ]
        if len(candidates) > 1:
            candidates.sort(key=lambda item: (-item[1], item[0]))
            ambiguous_matches[str(match["prediction_id"])] = {
                "automatic_owner_id": str(match["owner_id"]),
                "automatic_owner_iou": float(match["intersection_over_union"]),
                "candidate_owner_ids": [item[0] for item in candidates],
                "candidate_owner_ious": {item[0]: float(item[1]) for item in candidates},
            }
    if ambiguous_matches:
        matched_predictions -= set(ambiguous_matches)
        matched_owners -= {
            str(item["automatic_owner_id"])
            for item in ambiguous_matches.values()
        }
    committed_matches = [
        item for item in matches if str(item["prediction_id"]) not in ambiguous_matches
    ]
    review_queue: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for prediction in prefix:
        prediction_id = str(prediction["prediction_id"])
        match = next((item for item in matches if item["prediction_id"] == prediction_id), None)
        if match is not None:
            ambiguity = ambiguous_matches.get(prediction_id)
            if ambiguity is not None:
                receipt = {
                    **dict(prediction),
                    "entity_status": "ambiguous_matched_review",
                    "geometry_status": "ambiguous",
                    "review_required": True,
                    "intersection_over_union": ambiguity["automatic_owner_iou"],
                    **ambiguity,
                }
                review_queue.append(receipt)
            else:
                entity_status = "verified_owner"
                geometry_status = "acceptable"
                receipt = {**dict(prediction), **match, "entity_status": entity_status, "geometry_status": geometry_status, "review_required": False}
        else:
            same_category = [
                (str(owner["owner_id"]), _iou(prediction["bbox"], owner["bbox"]))
                for owner in owners
                if str(owner["category"]) == str(prediction["category"])
            ]
            same_category.sort(key=lambda item: (-item[1], item[0]))
            other_category = max((_iou(prediction["bbox"], owner["bbox"]) for owner in owners if str(owner["category"]) != str(prediction["category"])), default=0.0)
            if same_category and same_category[0][1] >= IOU_THRESHOLD and same_category[0][0] in matched_owners:
                entity_status = "duplicate"
                geometry_status = "not_applicable"
            elif same_category and same_category[0][1] > 0.0:
                entity_status = "unresolved_pending_crop_review"
                geometry_status = "unresolved"
            elif other_category >= IOU_THRESHOLD:
                entity_status = "semantic_mismatch_unresolved"
                geometry_status = "unresolved"
            else:
                entity_status = "unresolved_pending_crop_review"
                geometry_status = "unresolved"
            receipt = {
                **dict(prediction),
                "entity_status": entity_status,
                "geometry_status": geometry_status,
                "review_required": True,
                "intersection_over_union": None,
                "candidate_owner_id": same_category[0][0] if same_category else None,
                "candidate_owner_iou": same_category[0][1] if same_category else (other_category if other_category > 0.0 else None),
            }
            review_queue.append(receipt)
        # Raw tuple boxes are useful in Python tests but not JSON-friendly in
        # the final receipt; the serializer below converts them recursively.
        receipts.append(receipt)
    return {
        "budget": int(budget),
        "matches": committed_matches,
        "matched_owner_ids": sorted(matched_owners),
        "matched_prediction_ids": sorted(matched_predictions),
        "row_assignment_receipts": receipts,
        "unmatched_review_queue": review_queue,
        "coverage": len(matched_owners),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _prepare_review_index(
    review_document: Mapping[str, Any] | None,
    grouped_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate review joins before any decision can change coverage."""

    if review_document is None:
        return {}, {}
    decisions = _validate_review_decisions_document(review_document)
    contexts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for image_id, trajectory_map in grouped_rows.items():
        for trajectory_id, rows in trajectory_map.items():
            for row in rows:
                contexts[str(row["prediction_id"])].append((str(image_id), str(trajectory_id)))
    index: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        prediction_id = str(decision["prediction_id"])
        candidates = contexts.get(prediction_id, [])
        if not candidates:
            raise ValueError(f"unknown review decision prediction_id: {prediction_id}")
        if len(candidates) != 1:
            raise ValueError(f"ambiguous prediction_id in rollout panel: {prediction_id}")
        image_id, trajectory_id = candidates[0]
        if str(decision["image_id"]) != image_id or str(decision["trajectory_id"]) != trajectory_id:
            raise ValueError(
                f"inconsistent review decision context for {prediction_id}: "
                f"expected {image_id}/{trajectory_id}, got "
                f"{decision['image_id']}/{decision['trajectory_id']}"
            )
        index[prediction_id] = decision
    return index, {
        "schema_version": REVIEW_DECISIONS_SCHEMA_VERSION,
        "source_path": review_document.get("_source_path"),
        "source_sha256": review_document.get("_source_sha256")
        or _sha256_json(
            {
                "schema_version": REVIEW_DECISIONS_SCHEMA_VERSION,
                "decisions": decisions,
            }
        ),
    }


def _apply_review_decisions(
    assignment: dict[str, Any],
    *,
    review_index: Mapping[str, Mapping[str, Any]],
    owners: Sequence[Mapping[str, Any]],
    image_id: str,
    trajectory_id: str,
    budget: int,
    applied: dict[str, dict[str, Any]],
) -> None:
    """Apply decisions only to rows still in the automatic review queue."""

    if not review_index:
        return
    owner_by_id = {str(owner["owner_id"]): owner for owner in owners}
    queue = list(assignment.get("unmatched_review_queue", []))
    queue_ids: set[str] = set()
    remaining: list[dict[str, Any]] = []
    for receipt in queue:
        prediction_id = str(receipt["prediction_id"])
        if prediction_id in queue_ids:
            raise ValueError(f"duplicate review queue prediction_id: {prediction_id}")
        queue_ids.add(prediction_id)
        decision = review_index.get(prediction_id)
        if decision is None:
            remaining.append(receipt)
            continue
        if str(decision["image_id"]) != str(image_id) or str(decision["trajectory_id"]) != str(trajectory_id):
            raise ValueError(f"inconsistent review decision context for {prediction_id}")
        status = str(decision["entity_status"])
        owner_id_value = decision.get("owner_id")
        owner_id = None if owner_id_value is None else str(owner_id_value)
        owner = owner_by_id.get(owner_id) if owner_id is not None else None
        if owner_id is not None and owner is None:
            raise ValueError(f"review decision {prediction_id} names unknown owner_id: {owner_id}")
        predicted_category = str(receipt["category"])
        if status in {"verified_owner", "duplicate_owner"} and owner is not None and str(owner["category"]) != predicted_category:
            raise ValueError(
                f"review decision {prediction_id} category mismatch: "
                f"prediction={predicted_category}, owner={owner['category']}"
            )
        if (
            status in {"verified_owner", "duplicate_owner"}
            and decision.get("category") is not None
            and _normalize_category(decision["category"]) != predicted_category
        ):
            raise ValueError(f"review decision {prediction_id} category does not match prediction")
        if status in {"verified_owner", "duplicate_owner"} and owner is None:
            raise ValueError(f"review decision {prediction_id} requires an existing owner_id")
        if status == "verified_owner":
            geometry_status = decision.get("geometry_status")
            if geometry_status is None or str(geometry_status).strip() == "":
                raise ValueError(f"verified_owner decision {prediction_id} requires geometry_status")
            if receipt.get("entity_status") == "ambiguous_matched_review":
                candidate_ids = {str(item) for item in receipt.get("candidate_owner_ids", [])}
                if owner_id not in candidate_ids:
                    raise ValueError(
                        f"verified_owner decision {prediction_id} selects an owner outside "
                        "the automatic ambiguity candidates"
                    )
            if owner_id in {str(item) for item in assignment.get("matched_owner_ids", [])}:
                raise ValueError(
                    f"verified_owner decision {prediction_id} would count owner {owner_id} "
                    f"twice in trajectory {trajectory_id} at budget {budget}; use duplicate_owner"
                )
            assert owner is not None  # guarded above
            match = {
                "prediction_id": prediction_id,
                "generated_row_index": int(receipt["generated_row_index"]),
                "owner_id": owner_id,
                "owner_category": str(owner["category"]),
                "owner_bbox": tuple(owner["bbox"]),
                "category": predicted_category,
                "intersection_over_union": float(_iou(receipt["bbox"], owner["bbox"])),
            }
            assignment["matches"].append(match)
            assignment["matched_owner_ids"] = sorted(
                {*map(str, assignment.get("matched_owner_ids", [])), owner_id}
            )
            assignment["matched_prediction_ids"] = sorted(
                {*map(str, assignment.get("matched_prediction_ids", [])), prediction_id}
            )
            assignment["coverage"] = len(assignment["matched_owner_ids"])
            receipt.update(
                {
                    **match,
                    "entity_status": "verified_owner",
                    "geometry_status": str(geometry_status),
                    "review_required": False,
                    "review_decision_evidence": decision["evidence"],
                }
            )
        elif status == "duplicate_owner":
            # A duplicate decision never adds owner coverage.  It is valid
            # whether the automatic assignment had already selected this
            # owner or the row was left unmatched by the IoU matcher.
            assert owner is not None  # guarded above
            receipt.update(
                {
                    "owner_id": owner_id,
                    "owner_category": str(owner["category"]),
                    "owner_bbox": tuple(owner["bbox"]),
                    "intersection_over_union": float(_iou(receipt["bbox"], owner["bbox"])),
                    "entity_status": "duplicate_owner",
                    "geometry_status": str(decision.get("geometry_status", "not_applicable")),
                    "review_required": False,
                    "review_decision_evidence": decision["evidence"],
                }
            )
        else:
            # Explicit negative decisions separate confirmed harmful rows from
            # unresolved/unknown evidence.  They do not create owner matches.
            receipt.pop("owner_id", None)
            receipt.pop("owner_category", None)
            receipt.pop("owner_bbox", None)
            receipt.update(
                {
                    "entity_status": status,
                    "geometry_status": str(decision.get("geometry_status", "not_applicable")),
                    "review_required": False,
                    "review_decision_evidence": decision["evidence"],
                }
            )
            if status == "uncertain":
                # An uncertain human decision is not a rejection.  Keep it in
                # the unresolved collection so coverage and harmful counts
                # remain conservative even though no further UI review is
                # required for this row.
                remaining.append(receipt)
        applied_record = applied.setdefault(
            prediction_id,
            {
                "prediction_id": prediction_id,
                "image_id": str(image_id),
                "trajectory_id": str(trajectory_id),
                "entity_status": status,
                "budgets": [],
            },
        )
        applied_record["budgets"].append(int(budget))
    assignment["matches"] = sorted(
        assignment.get("matches", []),
        key=lambda item: (int(item.get("generated_row_index", 0)), str(item.get("prediction_id", "")), str(item.get("owner_id", ""))),
    )
    assignment["unmatched_review_queue"] = remaining


def analyze_rollout_payloads(
    artifacts: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    require_full_panel: bool = False,
    budgets: Sequence[int] = FIXED_BUDGETS,
    review_decisions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return image-level fixed-budget coverage and review receipts."""

    if require_full_panel:
        for artifact in artifacts:
            config = artifact.get("_artifact_config", artifact.get("config", {}))
            if not isinstance(config, Mapping):
                raise ValueError("rollout artifact config is missing")
            if not isinstance(artifact.get("model_identity"), Mapping):
                raise ValueError("full-panel rollout artifact lacks checkpoint/model identity evidence")
            if int(config.get("max_new_tokens", -1)) != 512:
                raise ValueError("full-panel analysis requires max_new_tokens=512")
            if float(config.get("repetition_penalty", -1.0)) != 1.0:
                raise ValueError("full-panel analysis requires repetition_penalty=1.0")
            mode = config.get("decode_mode")
            if mode == "greedy":
                if float(config.get("temperature", -1.0)) != 0.0:
                    raise ValueError("greedy artifact temperature must be 0")
            elif mode == "sampled":
                if float(config.get("temperature", -1.0)) != 0.4 or float(config.get("top_p", -1.0)) != 0.95:
                    raise ValueError("sampled artifact must use temperature=0.4 and top_p=0.95")
        for artifact in artifacts:
            for rollout in artifact.get("rollouts", []):
                if not rollout.get("prompt_token_ids_sha256") or not rollout.get("executed_media_sha256"):
                    raise ValueError("full-panel rollout lacks prompt or image identity evidence")

    rows, trajectory_evidence = _collect_rows(artifacts)
    grouped_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped_rows[str(row["image_id"])][str(row["trajectory_id"])].append(row)
    grouped_trajectory: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (image_id, trajectory_id), evidence in trajectory_evidence.items():
        grouped_trajectory[str(image_id)][trajectory_id] = evidence
    if set(grouped_rows) - set(str(key) for key in annotations):
        missing = sorted(set(grouped_rows) - set(str(key) for key in annotations))
        raise ValueError(f"rollout images missing generation-7 annotations: {missing}")
    review_index, review_provenance = _prepare_review_index(review_decisions, grouped_rows)
    applied_review_decisions: dict[str, dict[str, Any]] = {}

    image_results: list[dict[str, Any]] = []
    for image_id in sorted(grouped_trajectory, key=lambda item: (int(item) if item.isdigit() else item)):
        trajectory_map = grouped_trajectory[image_id]
        greedy_ids = [tid for tid, info in trajectory_map.items() if info["decode_mode"] == "greedy"]
        sampled_ids = [tid for tid, info in trajectory_map.items() if info["decode_mode"] == "sampled"]
        if require_full_panel:
            if set(grouped_trajectory) != set(EXPECTED_IMAGE_IDS):
                raise ValueError(
                    "full panel image cohort differs from the frozen twelve-image cohort: "
                    f"{sorted(grouped_trajectory)}"
                )
            if len(greedy_ids) != 1 or len(sampled_ids) != 16:
                raise ValueError(f"image {image_id} requires exactly one greedy and sixteen sampled trajectories")
            seeds = sorted(int(trajectory_map[tid]["seed"]) for tid in sampled_ids if trajectory_map[tid].get("seed") is not None)
            if tuple(seeds) != EXPECTED_SAMPLED_SEEDS:
                raise ValueError(f"image {image_id} sampled seed panel is incomplete or changed: {seeds}")
            greedy_seed = trajectory_map[greedy_ids[0]].get("seed")
            if greedy_seed is not None and int(greedy_seed) != EXPECTED_GREEDY_SEED:
                raise ValueError(f"image {image_id} greedy seed is not {EXPECTED_GREEDY_SEED}")
        if len(greedy_ids) != 1:
            raise ValueError(f"image {image_id} must have exactly one greedy trajectory")
        if not sampled_ids:
            raise ValueError(f"image {image_id} has no sampled trajectories")
        greedy_id = greedy_ids[0]
        owners = [dict(owner) for owner in annotations[str(image_id)]]
        budget_results: list[dict[str, Any]] = []
        for budget in budgets:
            budget = int(budget)
            per_trajectory = {
                trajectory_id: match_prefix(grouped_rows[image_id].get(trajectory_id, []), owners, budget)
                for trajectory_id in sorted(trajectory_map)
            }
            for trajectory_id, assignment in per_trajectory.items():
                _apply_review_decisions(
                    assignment,
                    review_index=review_index,
                    owners=owners,
                    image_id=image_id,
                    trajectory_id=trajectory_id,
                    budget=budget,
                    applied=applied_review_decisions,
                )
                parser_evidence = trajectory_map[trajectory_id]["parser"]
                parsed_for_trajectory = grouped_rows[image_id].get(trajectory_id, [])
                malformed_count = _malformed_before_budget(parser_evidence, parsed_for_trajectory, budget)
                duplicate_count = sum(
                    1
                    for receipt in assignment["row_assignment_receipts"]
                    if receipt.get("entity_status") in {"duplicate", "duplicate_owner"}
                )
                explicit_harmful_count = sum(
                    1
                    for receipt in assignment["row_assignment_receipts"]
                    if receipt.get("entity_status") in {"semantic_error", "unsupported_hallucination"}
                )
                assignment["malformed_row_count"] = malformed_count
                assignment["harmful_row_count"] = malformed_count + duplicate_count + explicit_harmful_count
                assignment["row_counts"] = {
                    "duplicate": duplicate_count,
                    "malformed": malformed_count,
                    "unsupported_hallucination": sum(
                        1
                        for receipt in assignment["row_assignment_receipts"]
                        if receipt.get("entity_status") == "unsupported_hallucination"
                    ),
                    "semantic_error": sum(
                        1
                        for receipt in assignment["row_assignment_receipts"]
                        if receipt.get("entity_status") == "semantic_error"
                    ),
                    "unresolved": sum(
                        1
                        for receipt in assignment["row_assignment_receipts"]
                        if receipt.get("entity_status")
                        in {
                            "semantic_mismatch_unresolved",
                            "unresolved_pending_crop_review",
                            "ambiguous_matched_review",
                            "uncertain",
                        }
                    ),
                }
                potential_owner_statuses = {
                    "uncertain",
                    "unresolved_pending_crop_review",
                    "semantic_mismatch_unresolved",
                    "ambiguous_matched_review",
                }
                potential_new_owner_count = sum(
                    1
                    for receipt in assignment["row_assignment_receipts"]
                    if receipt.get("entity_status") in potential_owner_statuses
                )
                assignment["coverage_lower_bound"] = int(assignment["coverage"])
                assignment["coverage_upper_bound"] = min(
                    len(owners),
                    int(assignment["coverage"]) + potential_new_owner_count,
                )
                assignment["potential_new_owner_row_count"] = potential_new_owner_count
                assignment["harmful_row_lower_bound"] = int(assignment["harmful_row_count"])
                assignment["harmful_row_upper_bound"] = int(assignment["harmful_row_count"]) + potential_new_owner_count
            right_censored = [
                trajectory_id
                for trajectory_id, evidence in trajectory_map.items()
                if bool(evidence["right_censored"]) and int(evidence["complete_row_count"]) < budget
            ]
            owner_sets = {trajectory_id: set(per_trajectory[trajectory_id]["matched_owner_ids"]) for trajectory_id in per_trajectory}
            greedy_owners = owner_sets[greedy_id]
            sampled_coverage = {trajectory_id: len(owner_sets[trajectory_id]) for trajectory_id in sampled_ids}
            best_coverage = max(sampled_coverage.values(), default=0)
            best_ids = sorted(trajectory_id for trajectory_id, coverage in sampled_coverage.items() if coverage == best_coverage)
            union_owners = set().union(*(owner_sets[trajectory_id] for trajectory_id in sampled_ids))
            matrix_trajectory_ids = [greedy_id, *sorted(sampled_ids)]
            matrix_owner_ids = sorted((str(owner["owner_id"]) for owner in owners), key=str)
            matrix = [[1 if owner_id in owner_sets[trajectory_id] else 0 for trajectory_id in matrix_trajectory_ids] for owner_id in matrix_owner_ids]
            complete = not right_censored
            lower_bounds = {"C_g": len(greedy_owners), "C_best": best_coverage, "C_union": len(union_owners)}
            coverage = {
                "C_g": len(greedy_owners) if complete else None,
                "C_best": best_coverage if complete else None,
                "C_union": len(union_owners) if complete else None,
                "individual_trajectory_gain": best_coverage - len(greedy_owners) if complete else None,
                "union_only_gain": len(union_owners) - best_coverage if complete else None,
            }
            all_receipts = [receipt for trajectory_id in sorted(per_trajectory) for receipt in per_trajectory[trajectory_id]["row_assignment_receipts"]]
            all_review = [receipt for trajectory_id in sorted(per_trajectory) for receipt in per_trajectory[trajectory_id]["unmatched_review_queue"]]
            panel_harmful_row_count = sum(
                int(per_trajectory[trajectory_id].get("harmful_row_count", 0))
                for trajectory_id in per_trajectory
            )
            owner_conditioned_geometry: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for receipt in all_receipts:
                owner_id = receipt.get("owner_id")
                if owner_id is not None:
                    owner_conditioned_geometry[str(owner_id)].append(
                        {
                            "trajectory_id": receipt.get("trajectory_id"),
                            "prediction_id": receipt.get("prediction_id"),
                            "generated_row_index": receipt.get("generated_row_index"),
                            "intersection_over_union": receipt.get("intersection_over_union"),
                            "geometry_status": receipt.get("geometry_status"),
                        }
                    )
            malformed_rows = sum(int(per_trajectory[trajectory_id].get("malformed_row_count", 0)) for trajectory_id in per_trajectory)
            row_counts = {
                "duplicate": sum(
                    1
                    for receipt in all_receipts
                    if receipt.get("entity_status") in {"duplicate", "duplicate_owner"}
                ),
                "malformed": malformed_rows,
                "unsupported_hallucination": sum(1 for receipt in all_receipts if receipt.get("entity_status") in {"unsupported_hallucination", "hallucination"}),
                "semantic_error": sum(1 for receipt in all_receipts if receipt.get("entity_status") == "semantic_error"),
                "semantic_mismatch_unresolved": sum(1 for receipt in all_receipts if receipt.get("entity_status") == "semantic_mismatch_unresolved"),
                "unresolved_pending_crop_review": sum(1 for receipt in all_receipts if receipt.get("entity_status") == "unresolved_pending_crop_review"),
            }
            ambiguous_count = sum(
                1 for receipt in all_receipts if receipt.get("entity_status") == "ambiguous_matched_review"
            )
            if ambiguous_count:
                row_counts["ambiguous_matched_review"] = ambiguous_count
            greedy_assignment = per_trajectory[greedy_id]
            route_local_certificates: dict[str, dict[str, Any]] = {}
            for sampled_id in sorted(sampled_ids):
                sampled_assignment = per_trajectory[sampled_id]
                evidence_complete = sampled_id not in right_censored and greedy_id not in right_censored
                sampled_lower = int(sampled_assignment["coverage_lower_bound"])
                greedy_upper = int(greedy_assignment["coverage_upper_bound"])
                sampled_harmful_upper = int(sampled_assignment["harmful_row_upper_bound"])
                greedy_harmful_lower = int(greedy_assignment["harmful_row_lower_bound"])
                coverage_condition = sampled_lower > greedy_upper
                harmful_condition = sampled_harmful_upper <= greedy_harmful_lower
                route_local_certificates[sampled_id] = {
                    "sampled_trajectory_id": sampled_id,
                    "greedy_trajectory_id": greedy_id,
                    "evidence_status": "complete" if evidence_complete else "missing_right_censor",
                    "sampled_coverage_lower_bound": sampled_lower,
                    "greedy_coverage_upper_bound": greedy_upper,
                    "sampled_harmful_upper_bound": sampled_harmful_upper,
                    "greedy_harmful_lower_bound": greedy_harmful_lower,
                    "coverage_condition": coverage_condition,
                    "harmful_condition": harmful_condition,
                    "conservative_certificate": evidence_complete and coverage_condition and harmful_condition,
                    "conservative_gain_lower_bound": sampled_lower - greedy_upper,
                }
            budget_results.append(
                {
                    "budget": budget,
                    "branch_evidence_status": "complete" if complete else "missing_right_censor",
                    "automatic_matching_status": "provisional_pending_crop_review",
                    "right_censored_trajectories": sorted(right_censored),
                    "coverage": coverage,
                    "C_g": coverage["C_g"],
                    "C_best": coverage["C_best"],
                    "C_union": coverage["C_union"],
                    "individual_trajectory_gain": coverage["individual_trajectory_gain"],
                    "union_only_gain": coverage["union_only_gain"],
                    "lower_bounds": lower_bounds,
                    "best_trajectory_id": best_ids[0] if best_ids else None,
                    "best_trajectory_ids": best_ids,
                    "sampled_only_owner_ids": sorted(union_owners - greedy_owners),
                    "owner_sets": {trajectory_id: sorted(owner_ids) for trajectory_id, owner_ids in sorted(owner_sets.items())},
                    "owner_by_trajectory_matrix": {
                        "owner_ids": matrix_owner_ids,
                        "trajectory_ids": matrix_trajectory_ids,
                        "matrix": matrix,
                    },
                    "trajectory_assignments": {trajectory_id: per_trajectory[trajectory_id] for trajectory_id in sorted(per_trajectory)},
                    "row_assignment_receipts": all_receipts,
                    "unmatched_review_queue": all_review,
                    "panel_harmful_row_count": panel_harmful_row_count,
                    "row_counts": row_counts,
                    "owner_conditioned_geometry": {
                        owner_id: sorted(values, key=lambda item: (str(item.get("trajectory_id")), int(item.get("generated_row_index", 0))))
                        for owner_id, values in sorted(owner_conditioned_geometry.items())
                    },
                    "route_local_conservative_certificates": route_local_certificates,
                }
            )
        image_results.append(
            {
                "image_id": image_id,
                "greedy_trajectory_id": greedy_id,
                "sampled_trajectory_ids": sorted(sampled_ids),
                "trajectory_evidence": {trajectory_id: trajectory_map[trajectory_id] for trajectory_id in sorted(trajectory_map)},
                "parser_evidence": {
                    "parse_status_counts": {
                        status: sum(1 for info in trajectory_map.values() if info["parser"].get("parse_status") == status)
                        for status in ("accepted", "accepted_with_drops", "empty", "malformed")
                    },
                    "malformed_attempt_count": sum(int(info["parser"].get("malformed_attempt_count", 0)) for info in trajectory_map.values()),
                    "dropped_prediction_count": sum(int(info["parser"].get("dropped_prediction_count", 0)) for info in trajectory_map.values()),
                },
                "owners": owners,
                "budgets": budget_results,
            }
        )
    observed_panel = {
        "image_count": len(grouped_trajectory),
        "greedy_seeds": sorted(
            {
                info.get("seed")
                for trajectories in grouped_trajectory.values()
                for info in trajectories.values()
                if info.get("decode_mode") == "greedy" and info.get("seed") is not None
            },
            key=str,
        ),
        "sampled_seeds": sorted(
            {
                info.get("seed")
                for trajectories in grouped_trajectory.values()
                for info in trajectories.values()
                if info.get("decode_mode") == "sampled" and info.get("seed") is not None
            },
            key=str,
        ),
        "greedy_trajectories_per_image": sorted(
            {
                sum(info.get("decode_mode") == "greedy" for info in trajectories.values())
                for trajectories in grouped_trajectory.values()
            }
        ),
        "sampled_trajectories_per_image": sorted(
            {
                sum(info.get("decode_mode") == "sampled" for info in trajectories.values())
                for trajectories in grouped_trajectory.values()
            }
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "fixed_budgets": [int(item) for item in budgets],
        "iou_threshold": IOU_THRESHOLD,
        "require_full_panel": bool(require_full_panel),
        "automatic_matching_status": "provisional_pending_crop_review",
        "branch_recommendation": None,
        "panel_expected": (
            {
                "image_ids": list(EXPECTED_IMAGE_IDS),
                "greedy_per_image": 1,
                "greedy_seed": EXPECTED_GREEDY_SEED,
                "sampled_per_image": 16,
                "sampled_seeds": list(EXPECTED_SAMPLED_SEEDS),
            }
            if require_full_panel
            else None
        ),
        "observed_panel": observed_panel,
        "images": image_results,
        "image_results": image_results,
        "trajectory_count": len(trajectory_evidence),
    }
    if review_decisions is not None:
        unused = sorted(set(review_index) - set(applied_review_decisions))
        if unused:
            raise ValueError(
                "review decisions were not applied to automatically unresolved "
                f"review rows: {unused}"
            )
        result["review_provenance"] = {
            **review_provenance,
            "applied_decisions": sorted(
                (_jsonable(item) for item in applied_review_decisions.values()),
                key=lambda item: str(item["prediction_id"]),
            ),
        }
    return _jsonable(result)


def analyze_rollout_files(
    rollout_artifacts: Iterable[str | Path],
    annotations_path: str | Path,
    *,
    require_full_panel: bool = False,
    budgets: Sequence[int] = FIXED_BUDGETS,
    review_decisions_path: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = load_rollout_artifacts(rollout_artifacts, require_decode_mode=require_full_panel)
    image_ids = {str(row.get("image_id")) for artifact in artifacts for row in artifact.get("rollouts", [])}
    annotations = load_generation7_annotations(annotations_path, image_ids=image_ids)
    review_decisions = load_review_decisions(review_decisions_path) if review_decisions_path is not None else None
    result = analyze_rollout_payloads(
        artifacts,
        annotations,
        require_full_panel=require_full_panel,
        budgets=budgets,
        review_decisions=review_decisions,
    )
    analyzer_path = Path(__file__).resolve(strict=True)
    result["sources"] = {
        "rollout_artifacts": [artifact.get("_source_path") for artifact in artifacts],
        "rollout_artifact_sha256": [artifact.get("_source_sha256") for artifact in artifacts],
        "annotations_path": str(Path(annotations_path).expanduser().resolve()),
        "annotations_sha256": _sha256_file(Path(annotations_path).expanduser().resolve(strict=True)),
        "analyzer_path": str(analyzer_path),
        "analyzer_sha256": _sha256_file(analyzer_path),
        "analysis_policy": {
            "fixed_budgets": list(result["fixed_budgets"]),
            "iou_threshold": result["iou_threshold"],
            "require_full_panel": result["require_full_panel"],
            "review_decisions_used": review_decisions_path is not None,
            "observed_panel": result["observed_panel"],
        },
    }
    if review_decisions_path is not None:
        review_source = result.get("review_provenance", {})
        result["sources"]["review_decisions_path"] = review_source.get("source_path")
        result["sources"]["review_decisions_sha256"] = review_source.get("source_sha256")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-artifact", "--rollout-artifacts", "--rollouts", dest="rollout_artifact", action="append", required=True, help="JSON artifact, directory, or glob; repeat for shards")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--review-decisions",
        type=Path,
        help="Optional crop-review JSON overlay; applied only to unresolved review-queue rows",
    )
    parser.add_argument(
        "--budget",
        dest="budgets",
        action="append",
        type=int,
        choices=range(1, 513),
        help=(
            "Complete-row budget to analyze; repeat for multiple budgets. "
            "Defaults to the frozen 4, 8, 16, and 32 budgets."
        ),
    )
    parser.add_argument("--allow-incomplete-panel", action="store_true", help="Permit synthetic or partial panels; branch evidence remains explicit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze_rollout_files(
        args.rollout_artifact,
        args.annotations,
        require_full_panel=not args.allow_incomplete_panel,
        budgets=tuple(args.budgets) if args.budgets else FIXED_BUDGETS,
        review_decisions_path=args.review_decisions,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
