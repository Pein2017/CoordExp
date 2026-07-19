#!/usr/bin/env python3
"""Probe whether same-coverage prefixes are sensitive to row order.

This is an experiment-local runner.  A case names three canonical object rows
``A``, ``B`` and ``C``.  The executable arms are ``A -> B -> C``, ``B -> A ->
C`` and the activation control ``B -> C``.  Rows are appended to the exact
materialized image prompt as token ids; an existing token row is never decoded
and re-tokenized.

The pure case/arm/parse helpers intentionally live in this script so the
focused tests do not import torch, transformers, or a model checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


CASE_SCHEMA_VERSION = "same_covered_set_prefix_order.case.v1"
SCHEMA_VERSION = "same_covered_set_prefix_order.v1"
ARMS = (
    "a_then_b_then_c",
    "b_then_a_then_c",
    "b_then_c_coverage_control",
)
ARM_ENTITY_ORDER = {
    "a_then_b_then_c": ("a_entity_id", "b_entity_id", "c_entity_id"),
    "b_then_a_then_c": ("b_entity_id", "a_entity_id", "c_entity_id"),
    "b_then_c_coverage_control": ("b_entity_id", "c_entity_id"),
}
DEFAULT_INFER_CONFIG = Path(
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_description_first_geometry_sorted_pure_cross_entropy_type_gate_dora_step4887_same_covered_set_prefix_order.yaml"
)

from src.templates.renderer import (  # noqa: E402  (safe, model-free import)
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IM_END_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)


_COMPLETE_ROW_RE = re.compile(
    re.escape(OBJECT_REF_START_TOKEN)
    + r".*?"
    + re.escape(OBJECT_REF_END_TOKEN)
    + re.escape(BOX_START_TOKEN)
    + r"(?:<\|coord_[0-9]+\|>){4}"
    + re.escape(BOX_END_TOKEN),
    re.DOTALL,
)
_OBJECT_START_RE = re.compile(re.escape(OBJECT_REF_START_TOKEN))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def parse_seed_list(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse an ordered, non-empty, duplicate-free seed list."""

    pieces = [piece.strip() for piece in value.split(",") if piece.strip()] if isinstance(value, str) else list(value)
    try:
        seeds = tuple(int(piece) for piece in pieces)
    except (TypeError, ValueError) as exc:
        raise ValueError("seeds must be integers, for example 11,12,13") from exc
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique so each rollout is auditable")
    return seeds


def parse_case_ids(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Parse optional case selectors while preserving request order."""

    if value is None:
        return None
    if isinstance(value, str):
        values = [piece.strip() for piece in value.split(",") if piece.strip()]
    else:
        values = [str(piece).strip() for piece in value if str(piece).strip()]
    if not values:
        raise ValueError("case ids must contain at least one non-empty id")
    if len(set(values)) != len(values):
        raise ValueError("case ids must be unique")
    return tuple(values)


def _entity_bbox(entity: Mapping[str, Any]) -> tuple[float, float, float, float]:
    value = entity.get("bbox_norm1000", entity.get("bbox"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("entity bbox_norm1000 must contain four values")
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("entity bbox_norm1000 values must be numeric") from exc
    if any(not math.isfinite(item) for item in box) or any(item < 0 or item > 1000 for item in box):
        raise ValueError("entity bbox_norm1000 values must lie in [0,1000]")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("entity bbox_norm1000 must have positive area")
    return box


def _row_token_value(entity: Mapping[str, Any]) -> Sequence[Any] | None:
    for field in ("row_token_ids", "token_ids", "row_ids"):
        if field in entity:
            return entity[field]
    return None


def _validate_row_text(row_text: str, *, entity_id: str) -> None:
    if not row_text:
        raise ValueError(f"entity {entity_id} row_text must be non-empty")
    match = _COMPLETE_ROW_RE.search(row_text)
    if match is None or row_text[: match.start()].strip() or row_text[match.end() :].strip():
        raise ValueError(f"entity {entity_id} row_text must contain exactly one complete object row")


def validate_case_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small JSON case document before any model execution."""

    spec = _as_mapping(spec, "case specification")
    schema = str(spec.get("schema_version", ""))
    if schema not in {CASE_SCHEMA_VERSION, "same_covered_set_prefix_order.v1"}:
        raise ValueError(f"unsupported case schema_version {schema!r}")
    if spec.get("image_id") is None or not str(spec.get("image_id")).strip():
        raise ValueError("case specification requires a non-empty image_id")
    entities_value = spec.get("entities")
    if not isinstance(entities_value, list) or not entities_value:
        raise ValueError("case specification requires a non-empty entities list")
    entities: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    for raw in entities_value:
        entity = dict(_as_mapping(raw, "entity"))
        entity_id = str(entity.get("entity_id", "")).strip()
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        if entity_id in seen_entities:
            raise ValueError(f"entity ids must be unique; repeated {entity_id!r}")
        seen_entities.add(entity_id)
        _entity_bbox(entity)
        row_text = entity.get("row_text")
        row_ids = _row_token_value(entity)
        has_text = isinstance(row_text, str)
        has_ids = row_ids is not None
        if has_text == has_ids:
            raise ValueError(f"entity {entity_id} must provide exactly one of row_text or row_token_ids")
        if has_text:
            _validate_row_text(row_text, entity_id=entity_id)
        else:
            if not isinstance(row_ids, Sequence) or isinstance(row_ids, (str, bytes)) or not row_ids:
                raise ValueError(f"entity {entity_id} row_token_ids must be a non-empty list")
            for token_id in row_ids:
                if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                    raise ValueError(f"entity {entity_id} row_token_ids must contain non-negative integers")
        entity["entity_id"] = entity_id
        entity["bbox_norm1000"] = list(_entity_bbox(entity))
        entities.append(entity)

    case_values = spec.get("cases")
    if not isinstance(case_values, list) or not case_values:
        raise ValueError("case specification requires a non-empty cases list")
    case_ids: set[str] = set()
    entity_ids = {str(entity["entity_id"]) for entity in entities}
    cases: list[dict[str, Any]] = []
    for raw_case in case_values:
        case = dict(_as_mapping(raw_case, "case"))
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("case_id must be non-empty")
        if case_id in case_ids:
            raise ValueError(f"case ids must be unique; repeated {case_id!r}")
        case_ids.add(case_id)
        selected = tuple(str(case.get(field, "")).strip() for field in ("a_entity_id", "b_entity_id", "c_entity_id"))
        if any(not value for value in selected):
            raise ValueError(f"case {case_id} requires a_entity_id, b_entity_id, and c_entity_id")
        if len(set(selected)) != 3:
            raise ValueError(f"case {case_id} requires distinct A/B/C entity identities")
        missing = [value for value in selected if value not in entity_ids]
        if missing:
            raise ValueError(f"case {case_id} references unknown entity ids: {missing}")
        case["case_id"] = case_id
        for field, value in zip(("a_entity_id", "b_entity_id", "c_entity_id"), selected, strict=True):
            case[field] = value
        cases.append(case)
    result = dict(spec)
    result["schema_version"] = schema
    result["image_id"] = str(spec["image_id"])
    result["entities"] = entities
    result["cases"] = cases
    return result


def _normalise_token_ids(value: Sequence[Any], *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{label} must be a non-empty token-id sequence")
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label} must contain non-negative integer ids")
        ids.append(int(item))
    return ids


def materialize_entity_rows(spec: Mapping[str, Any], *, tokenizer: Any | None = None) -> dict[str, dict[str, Any]]:
    """Materialize each row exactly once, preserving supplied token rows."""

    checked = validate_case_spec(spec)
    rows: dict[str, dict[str, Any]] = {}
    for entity in checked["entities"]:
        entity_id = str(entity["entity_id"])
        supplied_ids = _row_token_value(entity)
        if supplied_ids is not None:
            token_ids = _normalise_token_ids(supplied_ids, label=f"entity {entity_id} row_token_ids")
            row_text = entity.get("row_text") if isinstance(entity.get("row_text"), str) else None
            source = "supplied_token_ids"
        else:
            if tokenizer is None:
                raise ValueError(f"entity {entity_id} row_text requires a tokenizer")
            row_text = str(entity["row_text"])
            encode = getattr(tokenizer, "encode", None)
            if not callable(encode):
                raise ValueError("tokenizer must expose encode(text, add_special_tokens=False)")
            token_ids = _normalise_token_ids(
                encode(row_text, add_special_tokens=False),
                label=f"entity {entity_id} tokenized row",
            )
            source = "tokenized_row_text_once"
        rows[entity_id] = {
            "entity_id": entity_id,
            "description": str(entity.get("description", "")),
            "bbox_norm1000": list(_entity_bbox(entity)),
            "row_text": row_text,
            "row_token_ids": token_ids,
            "row_token_ids_sha256": _sha256_json(token_ids),
            "row_text_sha256": None if row_text is None else _sha256_text(row_text),
            "row_source": source,
        }
    return rows


def build_prefix_arms(case: Mapping[str, Any], entity_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build the three exact-ID prefix arms and validate order invariants."""

    case_id = str(case.get("case_id", ""))
    if not case_id:
        raise ValueError("case requires case_id")
    selected = {field: str(case.get(field, "")).strip() for field in ("a_entity_id", "b_entity_id", "c_entity_id")}
    if len(set(selected.values())) != 3 or any(not value for value in selected.values()):
        raise ValueError(f"case {case_id} requires distinct A/B/C entity identities")
    if any(value not in entity_rows for value in selected.values()):
        raise ValueError(f"case {case_id} references an entity without materialized row ids")
    arms: dict[str, dict[str, Any]] = {}
    for arm_name, fields in ARM_ENTITY_ORDER.items():
        entity_ids = [selected[field] for field in fields]
        rows = [dict(entity_rows[entity_id]) for entity_id in entity_ids]
        row_ids = [list(map(int, row["row_token_ids"])) for row in rows]
        prefix = [token_id for row in row_ids for token_id in row]
        arms[arm_name] = {
            "arm_name": arm_name,
            "entity_ids": entity_ids,
            "covered_entity_ids": entity_ids,
            "row_count": len(row_ids),
            "row_token_ids": row_ids,
            "row_token_ids_sha256": [_sha256_json(row) for row in row_ids],
            "prefix_token_ids": prefix,
            "prefix_token_ids_sha256": _sha256_json(prefix),
            "final_row_token_ids": row_ids[-1],
            "final_row_token_ids_sha256": _sha256_json(row_ids[-1]),
            "final_row_text": rows[-1].get("row_text"),
            "final_row_text_sha256": rows[-1].get("row_text_sha256"),
        }
    validate_same_coverage_order_invariants(arms, entity_rows, case=case)
    return arms


def validate_same_coverage_order_invariants(
    arms: Mapping[str, Mapping[str, Any]],
    entity_rows: Mapping[str, Mapping[str, Any]],
    *,
    case: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate the scientific contrasts before model execution."""

    for arm_name in ARMS:
        if arm_name not in arms:
            raise ValueError(f"missing required arm {arm_name}")
        arm = arms[arm_name]
        entity_ids = [str(value) for value in arm.get("entity_ids", [])]
        if not entity_ids or any(value not in entity_rows for value in entity_ids):
            raise ValueError(f"arm {arm_name} has unknown or empty entity_ids")
        if len(entity_ids) != int(arm.get("row_count", -1)):
            raise ValueError(f"arm {arm_name} row_count does not match entity_ids")
        row_ids = [tuple(int(item) for item in row) for row in arm.get("row_token_ids", [])]
        expected = [tuple(int(item) for item in entity_rows[eid]["row_token_ids"]) for eid in entity_ids]
        if row_ids != expected:
            raise ValueError(f"arm {arm_name} row token ids differ from entity rows")
        if list(arm.get("prefix_token_ids", [])) != [item for row in expected for item in row]:
            raise ValueError(f"arm {arm_name} prefix token ids are not a direct row concatenation")
    left = arms["a_then_b_then_c"]
    right = arms["b_then_a_then_c"]
    left_rows = Counter(tuple(row) for row in left["row_token_ids"])
    right_rows = Counter(tuple(row) for row in right["row_token_ids"])
    if left_rows != right_rows:
        raise ValueError("A-B-C and B-A-C must contain the same complete row token multiset")
    if set(left["covered_entity_ids"]) != set(right["covered_entity_ids"]):
        raise ValueError("A-B-C and B-A-C must cover the same physical entity set")
    if int(left["row_count"]) != int(right["row_count"]):
        raise ValueError("A-B-C and B-A-C must have the same row count")
    if tuple(left["row_token_ids"][:2]) == tuple(right["row_token_ids"][:2]):
        raise ValueError("A/B canonical rows must differ so the order intervention is observable")
    if tuple(left["final_row_token_ids"]) != tuple(right["final_row_token_ids"]):
        raise ValueError("A-B-C and B-A-C must end with byte-identical C row token ids")
    if left.get("final_row_text") is not None and right.get("final_row_text") is not None:
        if str(left["final_row_text"]).encode("utf-8") != str(right["final_row_text"]).encode("utf-8"):
            raise ValueError("A-B-C and B-A-C must end with byte-identical C row text")
    control = arms["b_then_c_coverage_control"]
    expected_control = (
        (str(case["b_entity_id"]), str(case["c_entity_id"]))
        if case is not None
        else (str(right["entity_ids"][0]), str(right["entity_ids"][2]))
    )
    if tuple(control["entity_ids"]) != expected_control:
        raise ValueError("B-C coverage control must contain B then C")
    return {
        "same_row_token_multiset": True,
        "same_physical_covered_set": True,
        "same_row_count": True,
        "byte_identical_final_c_row": True,
        "order_only_difference": tuple(left["entity_ids"][:2]) != tuple(right["entity_ids"][:2]),
        "coverage_control_entity_ids": list(control["entity_ids"]),
    }


# Short aliases make the pure contract convenient for small external probes.
validate_prefix_order_invariants = validate_same_coverage_order_invariants


def extract_first_complete_row(text: str) -> dict[str, Any] | None:
    """Return the first complete object row span without retokenizing it."""

    if not isinstance(text, str):
        raise ValueError("row-stop extraction requires text")
    match = _COMPLETE_ROW_RE.search(text)
    if match is None:
        return None
    return {
        "row_text": match.group(0),
        "char_start": int(match.start()),
        "char_end": int(match.end()),
        "row_text_sha256": _sha256_text(match.group(0)),
    }


def extract_row_stop(text: str, *, malformed_limit: int = 2) -> dict[str, Any]:
    """Classify continuation stopping as complete row, terminal, malformed, or length."""

    if malformed_limit <= 0:
        raise ValueError("malformed_limit must be positive")
    complete = extract_first_complete_row(text)
    if complete is not None:
        return {"stop_reason": "complete_row", **complete}
    terminal_at = text.find(IM_END_TOKEN)
    if terminal_at >= 0:
        return {
            "stop_reason": "terminal",
            "char_start": terminal_at,
            "char_end": terminal_at + len(IM_END_TOKEN),
            "row_text": None,
            "row_text_sha256": None,
        }
    malformed_count = 0
    for start in _OBJECT_START_RE.finditer(text):
        suffix = text[start.start() :]
        if _COMPLETE_ROW_RE.search(suffix) is None:
            malformed_count += 1
    if malformed_count >= malformed_limit:
        return {
            "stop_reason": "malformed_limit",
            "malformed_count": malformed_count,
            "char_start": None,
            "char_end": len(text),
            "row_text": None,
            "row_text_sha256": None,
        }
    return {
        "stop_reason": "length",
        "malformed_count": malformed_count,
        "char_start": None,
        "char_end": len(text),
        "row_text": None,
        "row_text_sha256": None,
    }


def _xyxy_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = area_left + area_right - inter
    return 0.0 if union <= 0 else inter / union


def _is_person_description(description: str) -> bool:
    value = re.sub(r"[^a-z ]+", " ", description.lower()).strip()
    return value in {"person", "people", "man", "woman", "boy", "girl", "human"} or value.startswith("person ")


def match_predictions_to_entities(
    predictions: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
    iou_threshold: float = 0.5,
    ambiguity_margin: float = 0.05,
) -> list[dict[str, Any]]:
    """Match person boxes by IoU, preserving unmatched/ambiguous outcomes."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0 <= iou_threshold <= 1 or ambiguity_margin < 0:
        raise ValueError("invalid IoU threshold or ambiguity margin")
    entity_items = list(entities.items()) if isinstance(entities, Mapping) else [(str(item.get("entity_id")), item) for item in entities]
    result: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        description = str(prediction.get("description", ""))
        raw_box = prediction.get("bbox", prediction.get("bbox_xyxy"))
        if not isinstance(raw_box, Sequence) or len(raw_box) != 4:
            result.append({"prediction_index": index, "status": "unmatched", "reason": "missing_bbox", "candidates": []})
            continue
        pixel_box = [float(value) for value in raw_box]
        normalized_box = [
            pixel_box[0] / image_width * 1000,
            pixel_box[1] / image_height * 1000,
            pixel_box[2] / image_width * 1000,
            pixel_box[3] / image_height * 1000,
        ]
        if not _is_person_description(description):
            result.append({
                "prediction_index": index,
                "description": description,
                "status": "not_person",
                "predicted_bbox_norm1000": normalized_box,
                "candidates": [],
            })
            continue
        candidates = []
        for entity_id, entity in entity_items:
            try:
                entity_box = _entity_bbox(entity)
            except ValueError:
                continue
            candidates.append({"entity_id": str(entity_id), "iou": _xyxy_iou(normalized_box, entity_box)})
        candidates.sort(key=lambda item: (-float(item["iou"]), str(item["entity_id"])))
        if not candidates or float(candidates[0]["iou"]) < iou_threshold:
            status = "unmatched"
            matched_id = None
        elif len(candidates) > 1 and float(candidates[0]["iou"]) - float(candidates[1]["iou"]) <= ambiguity_margin:
            status = "ambiguous"
            matched_id = None
        else:
            status = "matched"
            matched_id = candidates[0]["entity_id"]
        result.append({
            "prediction_index": index,
            "description": description,
            "status": status,
            "matched_entity_id": matched_id,
            "predicted_bbox": pixel_box,
            "predicted_bbox_norm1000": normalized_box,
            "candidates": candidates,
        })
    return result


def validate_artifact_payload(payload: Mapping[str, Any]) -> None:
    """Validate the compact result shape without loading torch or a model."""

    payload = _as_mapping(payload, "artifact")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected same-covered-set artifact schema version")
    if not isinstance(payload.get("config"), Mapping):
        raise ValueError("artifact config metadata is required")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("artifact must contain at least one case")
    for case in cases:
        case = _as_mapping(case, "artifact case")
        for field in ("case_id", "image_id", "entity_ledger", "invariants", "arms"):
            if field not in case:
                raise ValueError(f"artifact case is missing {field}")
        if not isinstance(case["entity_ledger"], list) or not case["entity_ledger"]:
            raise ValueError("artifact case entity_ledger must be a non-empty list")
        if not isinstance(case["arms"], Mapping):
            raise ValueError("artifact case arms must be an object")
        for arm_name in ARMS:
            arm = case["arms"].get(arm_name)
            if not isinstance(arm, Mapping):
                raise ValueError(f"artifact case is missing arm {arm_name}")
            for field in ("entity_ids", "prefix_token_ids", "prefix_token_ids_sha256", "row_count", "runs"):
                if field not in arm:
                    raise ValueError(f"artifact arm {arm_name} is missing {field}")
            if not isinstance(arm["entity_ids"], list) or not arm["entity_ids"]:
                raise ValueError(f"artifact arm {arm_name}.entity_ids must be a list")
            if not isinstance(arm["prefix_token_ids"], list):
                raise ValueError(f"artifact arm {arm_name}.prefix_token_ids must be a list")
            if not isinstance(arm["runs"], list) or not arm["runs"]:
                raise ValueError(f"artifact arm {arm_name}.runs must be a non-empty list")
            for run in arm["runs"]:
                run = _as_mapping(run, "artifact run")
                for field in ("mode", "seed", "status", "raw_generated_token_ids", "raw_generated_text", "row_stop", "parse_evidence"):
                    if field not in run:
                        raise ValueError(f"artifact run is missing {field}")
                if run["status"] not in {"success", "failed"}:
                    raise ValueError("artifact run status must be success or failed")
                if not isinstance(run["raw_generated_token_ids"], list) or not isinstance(run["raw_generated_text"], str):
                    raise ValueError("artifact run raw generation fields have the wrong shape")
                if not isinstance(run["row_stop"], Mapping) or not isinstance(run["parse_evidence"], Mapping):
                    raise ValueError("artifact run must retain row_stop and parse_evidence")


def _processor_config(config: Any) -> Any:
    from src.config.models import ProcessorConfig

    return ProcessorConfig(
        do_resize=config.model.processor.do_resize,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )


def _template_config(config: Any) -> Any:
    from src.config.models import TemplateConfig, TemplatePromptConfig

    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(system=config.template.prompt.system, user=config.template.prompt.user),
    )


def _physical_image_id(example: Any) -> str:
    metadata = getattr(example, "metadata", {})
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    value = source.get("image_id") if isinstance(source, Mapping) else None
    return str(value if value is not None else example.example_id)


def _select_example(examples: Sequence[Any], image_id: str) -> Any:
    matches = [example for example in examples if str(example.example_id) == str(image_id) or _physical_image_id(example) == str(image_id)]
    if len(matches) != 1:
        raise ValueError(f"image_id {image_id!r} resolved to {len(matches)} raw examples")
    return matches[0]


def _build_request(config: Any, frontend: Any, example: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    from src.inference.backend import DecodeRequest, GenerationPolicy
    from src.inference.image_plan import plan_image_batch
    from src.inference.prompt import build_prompt_record

    plan = plan_image_batch([example], components=frontend.qwen, processor_config=_processor_config(config), row_indices=[0]).rows[0]
    record = build_prompt_record(example, _template_config(config), processor=frontend.qwen.processor, row_index=0, merged_visual_tokens=plan.merged_visual_tokens)
    request = DecodeRequest(
        request_id=f"same-covered-set:{example.example_id}",
        chat_text=record.chat_text,
        input_prompt_token_ids=tuple(record.input_prompt_token_ids),
        expected_executed_prompt_token_ids=tuple(record.expected_executed_prompt_token_ids),
        image_path=plan.image_path,
        declared_image_width=plan.declared_width,
        declared_image_height=plan.declared_height,
        decoded_image_width=plan.decoded_width,
        decoded_image_height=plan.decoded_height,
        image_sha256=plan.image_content_sha256,
        expected_image_grid_thw=tuple(plan.expected_image_grid_thw),
        logical_transform_id=plan.logical_transform_id,
        generation_policy=GenerationPolicy(max_new_tokens=1),
    )
    prompt_meta = {
        "prompt_token_ids": list(record.expected_executed_prompt_token_ids),
        "prompt_token_ids_sha256": _sha256_json(record.expected_executed_prompt_token_ids),
        "chat_text_sha256": _sha256_text(record.chat_text),
        "image_path": str(plan.image_path),
        "image_sha256": plan.image_content_sha256,
        "width": int(plan.decoded_width),
        "height": int(plan.decoded_height),
    }
    return request, plan, prompt_meta


def _seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _append_prefix(native_inputs: Mapping[str, Any], prefix_token_ids: Sequence[int]) -> tuple[dict[str, Any], int]:
    import torch

    input_ids = native_inputs["input_ids"]
    prefix = torch.tensor([list(map(int, prefix_token_ids))], dtype=input_ids.dtype, device=input_ids.device)
    model_inputs = dict(native_inputs)
    model_inputs["input_ids"] = torch.cat((input_ids, prefix), dim=1)
    if "attention_mask" in model_inputs:
        model_inputs["attention_mask"] = torch.cat((model_inputs["attention_mask"], torch.ones_like(prefix)), dim=1)
    return model_inputs, int(model_inputs["input_ids"].shape[1])


class _RowStoppingCriteria:
    """Transformers-compatible stopping hook for one complete generated row."""

    def __init__(self, *, prompt_width: int, tokenizer: Any, malformed_limit: int) -> None:
        self.prompt_width = int(prompt_width)
        self.tokenizer = tokenizer
        self.malformed_limit = int(malformed_limit)

    def __call__(self, input_ids: Any, scores: Any = None, **_: Any) -> Any:
        import torch

        generated_ids = input_ids[0, self.prompt_width:].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        reason = extract_row_stop(generated_text, malformed_limit=self.malformed_limit)["stop_reason"]
        should_stop = reason in {"complete_row", "terminal", "malformed_limit"}
        return torch.tensor([bool(should_stop)], dtype=torch.bool, device=input_ids.device)


def _generate_one(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    prefix_token_ids: Sequence[int],
    tokenizer: Any,
    image_width: int,
    image_height: int,
    mode: str,
    seed: int | None,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    malformed_limit: int,
) -> dict[str, Any]:
    import torch

    if mode not in {"greedy", "sample"}:
        raise ValueError(f"unsupported generation mode {mode!r}")
    if seed is not None:
        _seed_torch(seed)
    model_inputs, prompt_width = _append_prefix(native_inputs, prefix_token_ids)
    kwargs: dict[str, Any] = {
        **model_inputs,
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": mode == "sample",
        "eos_token_id": session._im_end_token_id(),
        "pad_token_id": session._pad_token_id(),
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    if mode == "sample":
        kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
    # The runtime already depends on transformers, but keep this import local
    # so all pure helpers/tests remain model-free.  If an older transformers
    # build lacks the stopping-criteria module, post-generation extraction
    # still preserves the same artifact evidence and stop classification.
    try:
        from transformers import StoppingCriteriaList

        kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_RowStoppingCriteria(prompt_width=prompt_width, tokenizer=tokenizer, malformed_limit=malformed_limit)]
        )
    except ImportError:
        pass
    with torch.inference_mode():
        output = session._model.generate(**kwargs)
    sequences = getattr(output, "sequences", None)
    if sequences is None or int(sequences.shape[0]) != 1:
        raise RuntimeError("generation did not return one sequence")
    raw_ids = [int(value) for value in sequences[0, prompt_width:].tolist()]
    raw_text = tokenizer.decode(raw_ids, skip_special_tokens=False)
    row_stop = extract_row_stop(raw_text, malformed_limit=malformed_limit)
    parser_text = row_stop.get("row_text") or raw_text
    from src.inference.parsing import parse_compact_object_box_closed

    parsed = parse_compact_object_box_closed(
        parser_text,
        row_id=f"same-covered-set:{mode}:{seed}",
        row_index=0,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    return {
        "mode": mode,
        "seed": seed,
        "status": "success",
        "prefix_token_ids": list(map(int, prefix_token_ids)),
        "prefix_token_ids_sha256": _sha256_json(prefix_token_ids),
        "raw_generated_token_ids": raw_ids,
        "raw_generated_token_ids_sha256": _sha256_json(raw_ids),
        "raw_generated_text": raw_text,
        "raw_generated_text_sha256": _sha256_text(raw_text),
        "row_stop": row_stop,
        "parse_evidence": parsed.to_artifact_dict(),
        "parsed_predictions": parsed.predictions,
    }


def _failed_run(*, mode: str, seed: int | None, prefix_token_ids: Sequence[int], error: BaseException) -> dict[str, Any]:
    return {
        "mode": mode,
        "seed": seed,
        "status": "failed",
        "prefix_token_ids": list(map(int, prefix_token_ids)),
        "prefix_token_ids_sha256": _sha256_json(prefix_token_ids),
        "raw_generated_token_ids": [],
        "raw_generated_text": "",
        "raw_generated_token_ids_sha256": _sha256_json([]),
        "raw_generated_text_sha256": _sha256_text(""),
        "row_stop": {"stop_reason": "failed", "row_text": None, "row_text_sha256": None},
        "parse_evidence": {"parse_status": "not_run", "predictions": [], "dropped_predictions": []},
        "failure": {"type": type(error).__name__, "message": str(error)},
    }


def _case_selection(spec: Mapping[str, Any], case_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    cases = [dict(case) for case in spec["cases"]]
    if case_ids is None:
        return cases
    selected = [case for requested in case_ids for case in cases if str(case["case_id"]) == requested]
    if len(selected) != len(case_ids):
        missing = [requested for requested in case_ids if not any(str(case["case_id"]) == requested for case in cases)]
        raise ValueError(f"unknown case ids: {missing}")
    return selected


def run_probe(
    *,
    cases_path: Path,
    output: Path,
    infer_config: Path = DEFAULT_INFER_CONFIG,
    case_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (11, 12, 13),
    temperature: float = 0.2,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 256,
    malformed_limit: int = 2,
    device: str = "cuda:0",
    include_greedy: bool = True,
    force: bool = False,
) -> Path:
    """Run selected cases with one loaded current HF backend session."""

    import torch

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.runtime import assemble_frontend

    if output.exists() and not force:
        raise ValueError(f"refusing to overwrite {output}; pass --force")
    seeds = parse_seed_list(seeds)
    if not 0 < temperature:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0,1]")
    if repetition_penalty <= 0 or max_new_tokens <= 0:
        raise ValueError("repetition_penalty and max_new_tokens must be positive")
    checked_spec = validate_case_spec(json.loads(cases_path.read_text(encoding="utf-8")))
    selected_cases = _case_selection(checked_spec, case_ids)
    resolved = load_infer_config(infer_config.expanduser().resolve(strict=True))
    config = resolved.config
    raw_examples = list(load_raw_examples(config.data.input_jsonl))
    example = _select_example(raw_examples, str(checked_spec["image_id"]))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    # Tokenize row text once with the loaded frontend tokenizer, before opening
    # the executable model.  Supplied token ids take the no-tokenizer path.
    entity_rows = materialize_entity_rows(checked_spec, tokenizer=frontend.qwen.tokenizer)
    arms_by_case = {str(case["case_id"]): build_prefix_arms(case, entity_rows) for case in selected_cases}
    request, plan, prompt_meta = _build_request(config, frontend, example)
    output.parent.mkdir(parents=True, exist_ok=True)
    case_artifacts: list[dict[str, Any]] = []
    with open_backend_session(frontend.launch) as session:
        native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
        if tuple(executed_ids[0]) != request.expected_executed_prompt_token_ids:
            raise RuntimeError("base image prompt token parity failed")
        image_width, image_height = int(plan.decoded_width), int(plan.decoded_height)
        for case in selected_cases:
            case_id = str(case["case_id"])
            case_arms = arms_by_case[case_id]
            artifact_arms: dict[str, Any] = {}
            # Matching uses the complete image ledger, not only A/B/C.  This
            # lets the artifact distinguish a prediction matching an uncovered
            # person from an unmatched prediction while the prefix intervention
            # itself remains restricted to the declared three rows.
            matched_entities = entity_rows
            for arm_name, arm in case_arms.items():
                runs: list[dict[str, Any]] = []
                modes: list[tuple[str, int | None]] = ([('greedy', None)] if include_greedy else []) + [("sample", int(seed)) for seed in seeds]
                for mode, seed in modes:
                    try:
                        run = _generate_one(
                            session=session,
                            native_inputs=native_inputs,
                            prefix_token_ids=arm["prefix_token_ids"],
                            tokenizer=session._tokenizer,
                            image_width=image_width,
                            image_height=image_height,
                            mode=mode,
                            seed=seed,
                            temperature=temperature,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            max_new_tokens=max_new_tokens,
                            malformed_limit=malformed_limit,
                        )
                        run["entity_matches"] = match_predictions_to_entities(run["parsed_predictions"], matched_entities, image_width=image_width, image_height=image_height)
                    except Exception as exc:  # preserve per-arm/seed failures and continue
                        run = _failed_run(mode=mode, seed=seed, prefix_token_ids=arm["prefix_token_ids"], error=exc)
                    runs.append(run)
                artifact_arms[arm_name] = {**arm, "runs": runs}
            case_artifacts.append({
                "case_id": case_id,
                "image_id": str(checked_spec["image_id"]),
                "entity_ledger": [dict(row) for row in entity_rows.values()],
                "invariants": validate_same_coverage_order_invariants(artifact_arms, entity_rows, case=case),
                "arms": artifact_arms,
            })
        receipt_artifact = session.receipt.to_artifact_dict()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "same_covered_set_prefix_order",
        "config": {
            "infer_config_path": str(infer_config.expanduser().resolve()),
            "resolved_fingerprint": resolved.fingerprint,
            "device": str(device),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "malformed_limit": int(malformed_limit),
            "seeds": list(seeds),
            "include_greedy": bool(include_greedy),
            "case_ids": [str(case["case_id"]) for case in selected_cases],
        },
        "case_spec": {
            "path": str(cases_path.expanduser().resolve()),
            "sha256": _sha256_text(cases_path.read_text(encoding="utf-8")),
            "schema_version": checked_spec["schema_version"],
        },
        "image": {
            "image_id": str(checked_spec["image_id"]),
            "example_id": str(example.example_id),
            "image_path": str(plan.image_path),
            "image_sha256": plan.image_content_sha256,
            "width": image_width,
            "height": image_height,
            "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
            "executed_media_sha256": media_sha[0],
        },
        "base_prompt": {**prompt_meta, "observed_prompt_token_ids_sha256": _sha256_json(executed_ids[0])},
        "model_identity": receipt_artifact,
        "cases": case_artifacts,
    }
    validate_artifact_payload(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON case document")
    parser.add_argument("--output", type=Path, help="Output JSON artifact; defaults to case-spec output_path")
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    parser.add_argument("--case-ids", action="append", help="Comma-separated case ids; may be repeated")
    parser.add_argument("--seeds", default="11,12,13")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--malformed-limit", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-greedy", action="store_true", help="Skip the greedy arm run")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        args.seeds = parse_seed_list(args.seeds)
        args.case_ids = parse_case_ids(args.case_ids)
        spec = validate_case_spec(json.loads(args.cases.expanduser().resolve(strict=True).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is None:
        output_path = spec.get("output_path")
        if not isinstance(output_path, str) or not output_path.strip():
            parser.error("--output is required unless the case document contains output_path")
        args.output = Path(output_path)
    return args


def main() -> int:
    args = _parse_args()
    run_probe(
        cases_path=args.cases.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
        infer_config=args.infer_config,
        case_ids=args.case_ids,
        seeds=args.seeds,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        malformed_limit=args.malformed_limit,
        device=args.device,
        include_greedy=not args.no_greedy,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
