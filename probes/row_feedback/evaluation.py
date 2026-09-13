"""Frozen independent endpoint and paired S/F evaluation consumer.

This module is deliberately CPU-only.  It selects identities without looking at
model outputs, binds original and native COCO records, validates natural-output
envelopes from the row-feedback runtime, and reuses the committed parser and
cardinality-first matcher through the existing dense-enumeration scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from probes.dora_owner_learning.candidate_opportunity import file_hash, require, score
from probes.dora_owner_learning.entrance_ce_eval import aggregate_scores
from probes.dora_owner_learning.reward_rows import _pred_objects
from src.artifacts import publish_json_exclusive
from src.data.geometry import iou_xyxy, parse_coord_token
from src.inference.parsing import parse_compact_object_box_closed
from src.templates.renderer import IM_END_TOKEN


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-13-row-feedback-pilot/evaluation"
SOURCE = Path("/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl")
NATIVE_SOURCE = Path(
    "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/val.coord.jsonl"
)
PRIOR_EVALUATION = BASE / "2026-09-12-native-owner-scale-and-state/evaluation/selection.json"
CURRENT_CONFIRMATION = (
    BASE
    / "2026-09-13-owner-successor-scale-throughput/evaluation/confirmation-selection.json"
)
FROZEN_4096 = BASE / "2026-09-13-owner-successor-scale-throughput/supply/pool.json"
TRAINING_INPUTS = (
    BASE
    / "2026-09-12-native-owner-scale-and-state/scale/training/preparation/inputs-v2.json"
)
SELECTION_V2 = (
    BASE / "2026-09-12-native-owner-scale-and-state/scale/preparation/selection-v2.json"
)
SELECTION_V2_REMAINDER = (
    BASE
    / "2026-09-12-native-owner-scale-and-state/scale/preparation/selection-v2-remainder.json"
)
SUPERVISION_BANK = BASE / "2026-09-13-row-feedback-pilot/data-v2/supervision-bank.json"

PANEL_SIZE = 32
REVIEW_SIZE = 8
MAX_VISIBLE_TOKENS = 3084
EOS_TOKEN_ID = 151645
PANEL_SALT = "row-feedback-independent-endpoint-2026-09-13:v1:"
REVIEW_SALT = "row-feedback-dense-review-2026-09-13:v1:"

DEFAULT_EXCLUSIONS = (
    ("prior-evaluation-and-ledger", PRIOR_EVALUATION),
    ("current-confirmation-and-ledger", CURRENT_CONFIRMATION),
    ("frozen4096-supply", FROZEN_4096),
    ("n16-training-and-reference-roles", TRAINING_INPUTS),
    ("admission-selection-v2", SELECTION_V2),
    ("admission-selection-v2-remainder", SELECTION_V2_REMAINDER),
    ("row-feedback-supervision-bank", SUPERVISION_BANK),
)

_ID_RE = re.compile(r"coco2017_(?:train|val)_(\d{1,12})(?=[^\d]|$)")


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def publish(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(path, value)


def _example_id(value: Any) -> int:
    if type(value) is int:
        return value
    text = str(value)
    match = _ID_RE.search(text)
    require(match is not None, f"unsupported image identity: {text!r}")
    return int(match.group(1))


def _explicit_image_id(record: Mapping[str, Any]) -> int:
    if "image_id" in record:
        return int(record["image_id"])
    if "example_id" in record:
        return _example_id(record["example_id"])
    image = record.get("image")
    require(isinstance(image, Mapping), "record lacks image identity")
    return _explicit_image_id(image)


def _ids_from_selection(value: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()
    for key in ("image_ids", "blind_review_ids", "excluded_image_ids"):
        result.update(int(item) for item in value.get(key, []))
    return result


def _ids_from_supply(value: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()
    for key in ("image_ids", "excluded_ids", "independent_panel_ids", "protected_reference_ids"):
        result.update(_example_id(item) if isinstance(item, str) else int(item) for item in value.get(key, []))
    for item in value.get("items", []):
        result.add(_explicit_image_id(item))
    return result


def _ids_from_training(value: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()
    for key in ("positive_records", "conditional_records"):
        result.update(_explicit_image_id(item) for item in value.get(key, []))
    result.update(_example_id(item) for item in value.get("normal_keys", []))
    return result


def _ids_from_admission(value: Mapping[str, Any]) -> set[int]:
    result = {_example_id(item) for item in value.get("source_universe_example_ids", [])}
    for key in ("nominations", "selected"):
        result.update(_explicit_image_id(item) for item in value.get(key, []))
    return result


def _ids_from_supervision_bank(value: Mapping[str, Any]) -> set[int]:
    result = {_example_id(item) for item in value.get("train_image_ids", [])}
    result.update(_explicit_image_id(item) for item in value.get("records", []))
    return result


_EXCLUSION_CONTRACTS = {
    "native_owner_scale_state.evaluation.selection.v1": (
        {"frozen_before_new_outputs"},
        _ids_from_selection,
    ),
    "native_owner_successor_scale_throughput.confirmation_selection.v1": (
        {"frozen_cpu_no_model_calls"},
        _ids_from_selection,
    ),
    "owner_successor_scale.supply_pool.v1": (
        {"frozen_before_new_rollouts"},
        _ids_from_supply,
    ),
    "parallel_owner_training.inputs.v1": (
        {"prepared_no_model_execution"},
        _ids_from_training,
    ),
    "native_owner_scale.selection.v2": (
        {"frozen_cpu_no_model_calls"},
        _ids_from_admission,
    ),
    "row_feedback.supervision_bank.v1": (
        {"candidate_awaiting_root_protocol_acceptance", "lead_accepted"},
        _ids_from_supervision_bank,
    ),
    "row_feedback.supervision_bank.v2": (
        {"candidate_awaiting_root_protocol_acceptance", "lead_accepted"},
        _ids_from_supervision_bank,
    ),
}


def exclusion_inventory(
    specs: Sequence[tuple[str, str | Path]] = DEFAULT_EXCLUSIONS,
) -> tuple[set[int], list[dict[str, Any]]]:
    """Project only declared identity fields from frozen role manifests."""

    excluded: set[int] = set()
    sources: list[dict[str, Any]] = []
    for role, path in specs:
        value = read(path)
        schema = value.get("schema")
        require(schema in _EXCLUSION_CONTRACTS, f"unsupported exclusion schema for {role}: {schema}")
        statuses, projector = _EXCLUSION_CONTRACTS[schema]
        require(value.get("status") in statuses, f"unfrozen exclusion source for {role}")
        ids = projector(value)
        require(ids, f"empty exclusion source for {role}")
        excluded.update(ids)
        sources.append(
            {
                "role": role,
                "schema": schema,
                "status": value["status"],
                "binding": binding(path),
                "image_count": len(ids),
                "image_ids_sha256": digest(sorted(ids)),
            }
        )
    return excluded, sources


def select_ids(universe: Iterable[int], exclusions: Iterable[int], count: int, salt: str) -> list[int]:
    values = [int(value) for value in universe]
    require(len(values) == len(set(values)), "duplicate source image identity")
    eligible = set(values) - {int(value) for value in exclusions}
    require(len(eligible) >= count, "insufficient independent images; no backfill")
    return sorted(
        eligible,
        key=lambda image_id: hashlib.sha256(f"{salt}{image_id}".encode()).hexdigest(),
    )[:count]


def _source_index(path: str | Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = int(row["image_id"])
        require(image_id not in result, f"duplicate image {image_id} in {path}")
        result[image_id] = {
            "row_number": row_number,
            "record": row,
            "record_sha256": hashlib.sha256(line.encode()).hexdigest(),
        }
    return result


def _image_path(source_path: Path, row: Mapping[str, Any]) -> Path:
    images = row.get("images")
    require(isinstance(images, list) and len(images) == 1, "source row must contain one image")
    path = (source_path.parent / str(images[0])).resolve()
    require(path.is_file(), f"missing endpoint image: {path}")
    return path


def _materialize_records(
    panel: Sequence[int], source: Path, native_source: Path
) -> list[dict[str, Any]]:
    original = _source_index(source)
    native = _source_index(native_source)
    require(set(original) == set(native), "original/native source identity mismatch")
    records: list[dict[str, Any]] = []
    for image_id in panel:
        left, right = original[image_id], native[image_id]
        source_row, native_row = left["record"], right["record"]
        require(
            sorted(digest(obj) for obj in source_row["objects"])
            == sorted(digest(obj) for obj in native_row["objects"]),
            f"object multiset changed for {image_id}",
        )
        require(
            (source_row["width"], source_row["height"])
            == (native_row["width"], native_row["height"]),
            f"geometry changed for {image_id}",
        )
        left_image = _image_path(source, source_row)
        right_image = _image_path(native_source, native_row)
        require(left_image == right_image, f"media path changed for {image_id}")
        records.append(
            {
                "image_id": image_id,
                "example_id": f"coco2017_val_{image_id:012d}",
                "width": int(source_row["width"]),
                "height": int(source_row["height"]),
                "annotated_object_count": len(source_row["objects"]),
                "source_row_number": left["row_number"],
                "source_record_sha256": left["record_sha256"],
                "native_row_number": right["row_number"],
                "native_record_sha256": right["record_sha256"],
                "image_path": str(left_image),
                "image_sha256": file_hash(left_image),
                "source_record": source_row,
                "native_record": native_row,
            }
        )
    return records


def validate_selection(value: Mapping[str, Any], *, verify_files: bool = True) -> None:
    require(
        value.get("schema")
        in {
            "row_feedback.independent_endpoint_selection.v1",
            "row_feedback.independent_endpoint_selection.v2",
        },
        "selection schema",
    )
    require(value.get("status") == "frozen_cpu_no_model_calls", "selection status")
    panel = [int(item) for item in value["image_ids"]]
    review = [int(item) for item in value["dense_review_ids"]]
    excluded = {int(item) for item in value["excluded_image_ids"]}
    require(len(panel) == len(set(panel)) == PANEL_SIZE, "endpoint32 denominator")
    require(len(review) == len(set(review)) == REVIEW_SIZE, "dense-review8 denominator")
    require(set(review) <= set(panel), "dense-review8 outside endpoint32")
    require(not set(panel) & excluded, "endpoint overlaps excluded role")
    require(value["image_ids_sha256"] == digest(panel), "endpoint32 digest")
    require(value["dense_review_ids_sha256"] == digest(review), "dense-review8 digest")
    require(value.get("model_calls") == 0, "selection must precede model calls")
    require(value.get("panel_salt") == PANEL_SALT, "panel salt")
    require(value.get("dense_review_salt") == REVIEW_SALT, "dense-review salt")
    records = value["records"]
    require([int(row["image_id"]) for row in records] == panel, "record order/coverage")
    by_id = {int(row["image_id"]): row for row in records}
    require(
        review
        == sorted(
            panel,
            key=lambda item: (
                -int(by_id[item]["annotated_object_count"]),
                hashlib.sha256(f"{REVIEW_SALT}{item}".encode()).hexdigest(),
            ),
        )[:REVIEW_SIZE],
        "dense-review8 selection changed",
    )
    if verify_files:
        for key in ("source", "native_source"):
            expected = value[key]
            actual = binding(expected["path"])
            require(actual == expected, f"{key} binding changed")
        specs = [
            (item["role"], item["binding"]["path"])
            for item in value["exclusion_sources"]
        ]
        recomputed_excluded, recomputed_sources = exclusion_inventory(specs)
        require(recomputed_sources == value["exclusion_sources"], "exclusion source projection changed")
        require(sorted(recomputed_excluded) == value["excluded_image_ids"], "exclusion union changed")
        original = _source_index(value["source"]["path"])
        native = _source_index(value["native_source"]["path"])
        require(
            select_ids(original, recomputed_excluded, PANEL_SIZE, PANEL_SALT) == panel,
            "deterministic endpoint32 selection changed",
        )
        for row in records:
            image_id = int(row["image_id"])
            require(original[image_id]["record_sha256"] == row["source_record_sha256"], "source row changed")
            require(native[image_id]["record_sha256"] == row["native_record_sha256"], "native row changed")
            require(original[image_id]["row_number"] == row["source_row_number"], "source row number changed")
            require(native[image_id]["row_number"] == row["native_row_number"], "native row number changed")
            require(original[image_id]["record"] == row["source_record"], "embedded source record changed")
            require(native[image_id]["record"] == row["native_record"], "embedded native record changed")
            require(file_hash(row["image_path"]) == row["image_sha256"], "image bytes changed")
        if "supersedes" in value:
            expected = value["supersedes"]
            require(binding(expected["path"]) == expected, "superseded selection binding changed")


def cost_contract(selection_sha256: str) -> dict[str, Any]:
    calls = PANEL_SIZE * 2
    visible = calls * MAX_VISIBLE_TOKENS
    return {
        "schema": "row_feedback.endpoint_cost_contract.v1",
        "status": "awaiting_real_runtime_measurement",
        "selection_sha256": selection_sha256,
        "arms": ["S", "F"],
        "images_per_arm": PANEL_SIZE,
        "natural_calls": calls,
        "max_visible_tokens_per_call": MAX_VISIBLE_TOKENS,
        "max_visible_tokens_total": visible,
        "max_internal_generated_slots_total": visible,
        "measurement_required": {
            "minimum_completed_calls_per_arm": 1,
            "fields": [
                "timing.wall_seconds",
                "visible_generated_tokens",
                "internal_slot_count",
                "model_forwards",
                "image_forwards",
            ],
            "estimate": "64 * measured paired mean wall seconds / 3600 allocated GPU-hours",
        },
        "boundary": "Token/slot maxima are hard endpoint bounds. GPU-hours remain unknown until a real production-shaped paired slice is measured.",
    }


def freeze_selection(
    *,
    output: str | Path = ROOT,
    source: str | Path = SOURCE,
    native_source: str | Path = NATIVE_SOURCE,
    exclusion_specs: Sequence[tuple[str, str | Path]] = DEFAULT_EXCLUSIONS,
    selection_name: str = "selection.json",
    cost_name: str = "cost-contract.json",
    supersedes: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze endpoint32 and dense-review8 without reading arm outputs."""

    output, source, native_source = Path(output), Path(source), Path(native_source)
    selection_path = output / selection_name
    require(not selection_path.exists(), "endpoint selection already frozen")
    original = _source_index(source)
    excluded, exclusion_sources = exclusion_inventory(exclusion_specs)
    panel = select_ids(original, excluded, PANEL_SIZE, PANEL_SALT)
    records = _materialize_records(panel, source, native_source)
    counts = {int(row["image_id"]): int(row["annotated_object_count"]) for row in records}
    review = sorted(
        panel,
        key=lambda image_id: (
            -counts[image_id],
            hashlib.sha256(f"{REVIEW_SALT}{image_id}".encode()).hexdigest(),
        ),
    )[:REVIEW_SIZE]
    value = {
        "schema": "row_feedback.independent_endpoint_selection.v2",
        "status": "frozen_cpu_no_model_calls",
        "model_calls": 0,
        "source": binding(source),
        "native_source": binding(native_source),
        "source_images": len(original),
        "excluded_image_ids": sorted(excluded),
        "excluded_source_images": len(set(original) & excluded),
        "eligible_source_images": len(set(original) - excluded),
        "image_ids": panel,
        "image_ids_sha256": digest(panel),
        "dense_review_ids": review,
        "dense_review_ids_sha256": digest(review),
        "panel_salt": PANEL_SALT,
        "dense_review_salt": REVIEW_SALT,
        "selection_rule": "First 32 by SHA256(panel_salt + decimal image_id) from original COCO val after the declared union exclusions; identity-only and no output-conditioned backfill.",
        "dense_review_rule": "Within the already-fixed endpoint32, highest original COCO annotation count first; salted identity tie-break; fixed before arm outputs.",
        "exclusion_sources": exclusion_sources,
        "records": records,
        "evaluation_contract": {
            "parser": "compact-object-box-closed-v1",
            "matching": "category-constrained cardinality-first global assignment",
            "iou_thresholds": [0.5, 0.6, 0.8],
            "strict_repeat": "any-class native-pixel IoU > 0.95, counted once per later valid row",
            "unknown_policy": "Unmatched predictions remain annotation-relative; they are not automatically hallucinations.",
            "arms": ["S", "F"],
            "retained_native_n16": "context only; not a third matched training arm",
        },
        "provenance_boundary": "Identity-disjoint from the seven bound role manifests, including frozen4096 supply and the materialized row-feedback bank. Binding a candidate bank here is conservative exclusion, not scientific admission. This does not claim pretraining/SFT disjointness or exhaustively scan undeclared raw traces.",
    }
    if supersedes is not None:
        previous_path = Path(supersedes)
        previous = read(previous_path)
        validate_selection(previous, verify_files=True)
        require(previous["image_ids"] == panel, "provenance correction changed endpoint32")
        require(previous["dense_review_ids"] == review, "provenance correction changed dense-review8")
        value["supersedes"] = binding(previous_path)
        value["provenance_correction"] = (
            "Only the supervision-bank binding changed from v1 to data-v2. Endpoint32, dense-review8, "
            "all image-role exclusions, and evaluation semantics are unchanged; the corrected bank "
            "attributes history/witness rollout production to the older producer and N16 only to the "
            "fit anchor/current native teacher."
        )
    validate_selection(value, verify_files=True)
    publish(selection_path, value)
    selection_sha = file_hash(selection_path)
    publish(output / cost_name, cost_contract(selection_sha))
    return value


def validate_runtime_result(value: Mapping[str, Any], *, expected_arm: str) -> None:
    require(expected_arm in ("S", "F") and value.get("arm") == expected_arm, "runtime arm")
    ids = value.get("visible_token_ids")
    require(isinstance(ids, list) and all(type(item) is int for item in ids), "visible token ids")
    require(isinstance(value.get("text"), str), "visible text")
    count = value.get("visible_generated_tokens")
    require(type(count) is int and count == len(ids) and 0 <= count <= MAX_VISIBLE_TOKENS, "visible token count")
    finish = value.get("finish_reason")
    require(finish in ("eos", "length"), "finish reason")
    require(type(value.get("eos")) is bool and value["eos"] == (finish == "eos"), "EOS flag")
    require(type(value.get("cap")) is bool and value["cap"] == (finish == "length"), "cap flag")
    if finish == "eos":
        require(bool(ids) and ids[-1] == EOS_TOKEN_ID, "terminal EOS token identity")
    else:
        require(EOS_TOKEN_ID not in ids, "length output contains EOS token")
    if finish == "length":
        require(count == MAX_VISIBLE_TOKENS, "length finish before visible cap")
    slots = value.get("slot_work")
    require(isinstance(slots, Mapping), "slot-work mapping")
    for key in ("prefill", "history", "generated", "total"):
        require(type(slots.get(key)) is int and slots[key] >= 0, f"slot-work {key}")
    require(slots["total"] == slots["prefill"] + slots["history"] + slots["generated"], "slot-work sum")
    require(value.get("internal_slot_count") == slots["total"], "internal slot count")
    for key in ("model_forwards", "image_forwards"):
        require(type(value.get(key)) is int and value[key] >= 0, key)
    timing = value.get("timing")
    require(isinstance(timing, Mapping) and timing, "timing receipt")
    require(
        all(type(item) in (int, float) and not isinstance(item, bool) and item >= 0 for item in timing.values()),
        "timing values",
    )
    require("wall_seconds" in timing, "timing wall_seconds")
    decode = value.get("decode_contract")
    require(isinstance(decode, Mapping), "decode contract")
    expected = {
        "max_visible_tokens": MAX_VISIBLE_TOKENS,
        "eos_token_id": EOS_TOKEN_ID,
        "do_sample": False,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1,
    }
    for key, item in expected.items():
        require(decode.get(key) == item, f"decode contract {key}")


def make_output_envelope(
    *,
    selection_sha256: str,
    record: Mapping[str, Any],
    arm: str,
    prompt_token_ids: Sequence[int],
    prepared_inputs_sha256: str,
    model_identity: Mapping[str, Any],
    adapter_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    validate_runtime_result(runtime, expected_arm=arm)
    return {
        "schema": "row_feedback.natural_output.v1",
        "status": "completed",
        "selection_sha256": selection_sha256,
        "image_id": int(record["image_id"]),
        "example_id": str(record["example_id"]),
        "source_record_sha256": record["source_record_sha256"],
        "native_record_sha256": record["native_record_sha256"],
        "prompt_token_ids_sha256": digest(list(prompt_token_ids)),
        "prepared_inputs_sha256": prepared_inputs_sha256,
        "visible_token_ids_sha256": digest(runtime["visible_token_ids"]),
        "visible_text_sha256": hashlib.sha256(runtime["text"].encode()).hexdigest(),
        "model_identity": dict(model_identity),
        "adapter_identity": dict(adapter_identity),
        "arm": arm,
        "runtime": dict(runtime),
    }


def consumer_contract(selection_path: str | Path) -> dict[str, Any]:
    """Bind the exact endpoint/runtime seam after CPU API integration."""

    from probes.row_feedback import runtime as feedback_runtime

    selection_path = Path(selection_path)
    selection = read(selection_path)
    validate_selection(selection, verify_files=True)
    signature = inspect.signature(feedback_runtime.generate_visible)
    expected_parameters = [
        "qwen",
        "inputs",
        "prompt_ids",
        "history_ids",
        "arm",
        "max_visible_tokens",
        "eos_token_id",
        "feedback_source_overrides",
        "capture_feedback_sources",
    ]
    require(list(signature.parameters) == expected_parameters, "runtime generate_visible signature changed")
    require(signature.parameters["max_visible_tokens"].default == MAX_VISIBLE_TOKENS, "runtime visible cap")
    require(signature.parameters["eos_token_id"].default == EOS_TOKEN_ID, "runtime EOS token")
    materializer_signature = inspect.signature(feedback_runtime.materialize_endpoint_record)
    require(
        list(materializer_signature.parameters)
        == ["qwen", "frontend", "config", "selection_record", "native_source"],
        "runtime endpoint materializer signature changed",
    )
    code_paths = [
        Path(__file__).resolve(),
        Path(feedback_runtime.__file__).resolve(),
        Path(inspect.getsourcefile(parse_compact_object_box_closed)).resolve(),
        Path(inspect.getsourcefile(score)).resolve(),
    ]
    return {
        "schema": "row_feedback.endpoint_consumer_contract.v1",
        "status": "cpu_verified_awaiting_real_outputs",
        "selection": binding(selection_path),
        "runtime_call": str(signature),
        "runtime_materializer": str(materializer_signature),
        "runtime_result_fields": [
            "arm",
            "visible_token_ids",
            "text",
            "finish_reason",
            "eos",
            "cap",
            "visible_generated_tokens",
            "internal_slot_count",
            "model_forwards",
            "image_forwards",
            "timing.wall_seconds",
            "slot_work.prefill",
            "slot_work.history",
            "slot_work.generated",
            "slot_work.total",
            "decode_contract",
        ],
        "caller_envelope_fields": [
            "selection_sha256",
            "image_id",
            "example_id",
            "source_record_sha256",
            "native_record_sha256",
            "prompt_token_ids_sha256",
            "prepared_inputs_sha256",
            "model_identity",
            "adapter_identity",
            "visible_token_ids_sha256",
            "visible_text_sha256",
        ],
        "terminal_policy": "EOS remains a counted visible token and must be final; only its decoded terminal marker is removed before parsing.",
        "score_contract": selection["evaluation_contract"],
        "code": [binding(path) for path in code_paths],
        "acceptance": {
            "command": "pytest -q probes/row_feedback/tests/test_evaluation.py probes/row_feedback/tests/test_runtime.py",
            "observed": "10 passed",
            "scope": "CPU contract and fake-native runtime; no model-quality evidence",
        },
    }


def _valid_envelope(
    row: Mapping[str, Any], *, arm: str, selection_sha256: str, record: Mapping[str, Any]
) -> None:
    require(row.get("schema") == "row_feedback.natural_output.v1", "output schema")
    require(row.get("status") == "completed", "incomplete output")
    require(row.get("arm") == arm, "output arm")
    require(row.get("selection_sha256") == selection_sha256, "selection binding")
    require(int(row.get("image_id")) == int(record["image_id"]), "output image identity")
    require(row.get("example_id") == record["example_id"], "output example identity")
    require(row.get("source_record_sha256") == record["source_record_sha256"], "source record binding")
    require(row.get("native_record_sha256") == record["native_record_sha256"], "native record binding")
    for key in ("prompt_token_ids_sha256", "prepared_inputs_sha256"):
        require(isinstance(row.get(key), str) and len(row[key]) == 64, key)
    require(row.get("visible_token_ids_sha256") == digest(row["runtime"]["visible_token_ids"]), "visible token digest")
    require(
        row.get("visible_text_sha256")
        == hashlib.sha256(row["runtime"]["text"].encode()).hexdigest(),
        "visible text digest",
    )
    require(isinstance(row.get("model_identity"), Mapping) and row["model_identity"], "model identity")
    require(isinstance(row.get("adapter_identity"), Mapping) and row["adapter_identity"], "adapter identity")
    validate_runtime_result(row["runtime"], expected_arm=arm)


def _gt_objects(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in record["source_record"]["objects"]:
        obj = dict(item)
        require("coco_ann_id" in obj, "GT owner identity")
        obj["object_id"] = str(obj["coco_ann_id"])
        obj["bbox_2d"] = [
            parse_coord_token(value, field=f"gt[{obj['object_id']}].bbox[{index}]")
            for index, value in enumerate(obj["bbox_2d"])
        ]
        result.append(obj)
    return result


def _revisit_candidates(score_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    predictions, _ = _pred_objects(dict(score_row))
    result = []
    for later, (category, box) in enumerate(predictions):
        for earlier, (old_category, old_box) in enumerate(predictions[:later]):
            overlap = iou_xyxy(box, old_box)
            if category == old_category and 0.5 <= overlap <= 0.95:
                result.append(
                    {
                        "earlier_valid_row": earlier,
                        "later_valid_row": later,
                        "category": category,
                        "iou": overlap,
                    }
                )
    return result


def _score_output(row: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    runtime = row["runtime"]
    parser_text = runtime["text"]
    if runtime["eos"]:
        require(parser_text.endswith(IM_END_TOKEN), "decoded text lacks terminal EOS")
        parser_text = parser_text[: -len(IM_END_TOKEN)]
    parsed = parse_compact_object_box_closed(
        parser_text,
        row_id=record["example_id"],
        row_index=int(record["native_row_number"]) - 1,
        image_width=int(record["width"]),
        image_height=int(record["height"]),
    ).to_artifact_dict()
    score_row = {
        "row_id": record["example_id"],
        "image_width": int(record["width"]),
        "image_height": int(record["height"]),
        "gt": _gt_objects(record),
        "pred": parsed["predictions"],
        "dropped_prediction_count": parsed["dropped_prediction_count"],
    }
    card = score(
        score_row,
        seed=None,
        length=int(runtime["visible_generated_tokens"]),
        stop="im_end" if runtime["finish_reason"] == "eos" else "length",
    )
    return {
        "parser": parsed,
        "score": card,
        "same_category_overlap_candidates_iou50_to_95": _revisit_candidates(score_row),
        "revisit_boundary": "Same-category geometric candidates only; semantic same-owner re-entry or alias drift requires physical review.",
    }


def _index_outputs(path: str | Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {int(row["image_id"]): row for row in rows}
    require(len(result) == len(rows), f"duplicate output image in {path}")
    return result


def _runtime_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    timing_keys = sorted(set().union(*(row["runtime"]["timing"].keys() for row in rows)))
    return {
        "visible_generated_tokens": sum(row["runtime"]["visible_generated_tokens"] for row in rows),
        "internal_slot_count": sum(row["runtime"]["internal_slot_count"] for row in rows),
        "model_forwards": sum(row["runtime"]["model_forwards"] for row in rows),
        "image_forwards": sum(row["runtime"]["image_forwards"] for row in rows),
        "eos": sum(row["runtime"]["eos"] for row in rows),
        "caps": sum(row["runtime"]["cap"] for row in rows),
        "finish_reasons": dict(Counter(row["runtime"]["finish_reason"] for row in rows)),
        "slot_work": {
            key: sum(row["runtime"]["slot_work"][key] for row in rows)
            for key in ("prefill", "history", "generated", "total")
        },
        "timing": {
            key: sum(float(row["runtime"]["timing"].get(key, 0)) for row in rows)
            for key in timing_keys
        },
    }


def estimate_endpoint_cost(
    *, s_rows_path: str | Path, f_rows_path: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    """Project the fixed 64-call endpoint from a real paired runtime slice."""

    rows: list[Mapping[str, Any]] = []
    observed: dict[str, Any] = {}
    for arm, path in (("S", s_rows_path), ("F", f_rows_path)):
        arm_rows = read_jsonl(path)
        require(arm_rows, f"missing measured {arm} slice")
        for row in arm_rows:
            require(row.get("arm") == arm and isinstance(row.get("runtime"), Mapping), f"measured {arm} envelope")
            validate_runtime_result(row["runtime"], expected_arm=arm)
        seconds = [float(row["runtime"]["timing"]["wall_seconds"]) for row in arm_rows]
        observed[arm] = {
            "binding": binding(path),
            "calls": len(arm_rows),
            "wall_seconds_total": sum(seconds),
            "wall_seconds_mean": sum(seconds) / len(seconds),
            "wall_seconds_max": max(seconds),
            "visible_generated_tokens": sum(row["runtime"]["visible_generated_tokens"] for row in arm_rows),
            "internal_slot_count": sum(row["runtime"]["internal_slot_count"] for row in arm_rows),
            "model_forwards": sum(row["runtime"]["model_forwards"] for row in arm_rows),
            "image_forwards": sum(row["runtime"]["image_forwards"] for row in arm_rows),
        }
        rows.extend(arm_rows)
    all_seconds = [float(row["runtime"]["timing"]["wall_seconds"]) for row in rows]
    mean_seconds = sum(all_seconds) / len(all_seconds)
    result = {
        "schema": "row_feedback.endpoint_cost_estimate.v1",
        "status": "measured_projection_not_execution",
        "planned_calls": PANEL_SIZE * 2,
        "observed": observed,
        "projected_allocated_gpu_hours_at_observed_mean": PANEL_SIZE * 2 * mean_seconds / 3600,
        "projected_allocated_gpu_hours_at_observed_max": PANEL_SIZE * 2 * max(all_seconds) / 3600,
        "assumptions": [
            "one GPU per natural call",
            "slice uses the production model, precision, prompt, media preparation and visible cap",
            "projection is cost-only and is frozen before viewing trained endpoint quality",
        ],
        "boundary": "The observed-max projection is a finite empirical planning bound, not a guarantee against later runtime variance.",
    }
    if output is not None:
        publish(output, result)
    return result


def consume_pair(
    *, selection_path: str | Path, s_rows_path: str | Path, f_rows_path: str | Path, output: str | Path
) -> dict[str, Any]:
    selection_path, output = Path(selection_path), Path(output)
    selection = read(selection_path)
    validate_selection(selection, verify_files=True)
    selection_sha = file_hash(selection_path)
    records = {int(row["image_id"]): row for row in selection["records"]}
    arms = {"S": _index_outputs(s_rows_path), "F": _index_outputs(f_rows_path)}
    expected = set(int(item) for item in selection["image_ids"])
    require(set(arms["S"]) == set(arms["F"]) == expected, "paired endpoint32 coverage")
    per_arm: dict[str, list[dict[str, Any]]] = {"S": [], "F": []}
    for image_id in selection["image_ids"]:
        record = records[int(image_id)]
        for arm in ("S", "F"):
            row = arms[arm][int(image_id)]
            _valid_envelope(row, arm=arm, selection_sha256=selection_sha, record=record)
            per_arm[arm].append({"image_id": int(image_id), **_score_output(row, record)})
        for key in (
            "selection_sha256",
            "source_record_sha256",
            "native_record_sha256",
            "prompt_token_ids_sha256",
            "prepared_inputs_sha256",
            "model_identity",
        ):
            require(arms["S"][int(image_id)][key] == arms["F"][int(image_id)][key], f"paired {key}")
    aggregates = {}
    for arm in ("S", "F"):
        cards = [item["score"] for item in per_arm[arm]]
        metrics = aggregate_scores(cards)
        for threshold in ("50", "60", "80"):
            denominator = metrics[threshold]["tp"] + metrics[threshold]["fn"]
            metrics[threshold]["recall"] = metrics[threshold]["tp"] / denominator if denominator else 0.0
            metrics[threshold]["annotated_owner_denominator"] = denominator
        metrics["same_category_overlap_candidates_iou50_to_95"] = sum(
            len(item["same_category_overlap_candidates_iou50_to_95"]) for item in per_arm[arm]
        )
        aggregates[arm] = {"metrics": metrics, "runtime": _runtime_totals(arms[arm].values())}
    changes: dict[str, dict[str, Any]] = {}
    for threshold in ("50", "60", "80"):
        per_image = []
        totals = Counter()
        for index, image_id in enumerate(selection["image_ids"]):
            before = set(per_arm["S"][index]["score"][threshold]["owners"])
            after = set(per_arm["F"][index]["score"][threshold]["owners"])
            row = {
                "image_id": int(image_id),
                "gained": sorted(after - before),
                "lost": sorted(before - after),
                "retained": sorted(before & after),
            }
            totals.update({key: len(row[key]) for key in ("gained", "lost", "retained")})
            per_image.append(row)
        changes[threshold] = {**dict(totals), "per_image": per_image}
    result = {
        "schema": "row_feedback.paired_endpoint_result.v1",
        "status": "completed",
        "selection": binding(selection_path),
        "inputs": {"S": binding(s_rows_path), "F": binding(f_rows_path)},
        "images_per_arm": PANEL_SIZE,
        "comparison": "F minus S at equal fixed endpoint32 identities; native N16 is context only",
        "aggregates": aggregates,
        "owner_changes_F_vs_S": changes,
        "per_arm": per_arm,
        "interpretation_boundary": "Owners are matched COCO annotations under the declared category-constrained assignment. Unmatched predictions are not automatically hallucinations; physical re-entry/alias drift is unresolved until the preselected dense8 review.",
    }
    publish(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--output", type=Path, default=ROOT)
    freeze.add_argument("--selection-name", default="selection.json")
    freeze.add_argument("--cost-name", default="cost-contract.json")
    freeze.add_argument("--supersedes", type=Path)
    verify = commands.add_parser("verify-selection")
    verify.add_argument("--selection", type=Path, default=ROOT / "selection.json")
    consume = commands.add_parser("consume")
    consume.add_argument("--selection", type=Path, default=ROOT / "selection.json")
    consume.add_argument("--s-rows", type=Path, required=True)
    consume.add_argument("--f-rows", type=Path, required=True)
    consume.add_argument("--output", type=Path, default=ROOT / "paired-result.json")
    estimate = commands.add_parser("estimate-cost")
    estimate.add_argument("--s-rows", type=Path, required=True)
    estimate.add_argument("--f-rows", type=Path, required=True)
    estimate.add_argument("--output", type=Path, default=ROOT / "measured-cost-estimate.json")
    contract = commands.add_parser("freeze-consumer-contract")
    contract.add_argument("--selection", type=Path, default=ROOT / "selection-v2.json")
    contract.add_argument("--output", type=Path, default=ROOT / "consumer-contract.json")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_selection(
            output=args.output,
            selection_name=args.selection_name,
            cost_name=args.cost_name,
            supersedes=args.supersedes,
        )
    elif args.command == "verify-selection":
        result = read(args.selection)
        validate_selection(result, verify_files=True)
    elif args.command == "consume":
        result = consume_pair(
            selection_path=args.selection,
            s_rows_path=args.s_rows,
            f_rows_path=args.f_rows,
            output=args.output,
        )
    elif args.command == "estimate-cost":
        result = estimate_endpoint_cost(
            s_rows_path=args.s_rows,
            f_rows_path=args.f_rows,
            output=args.output,
        )
    else:
        result = consumer_contract(args.selection)
        publish(args.output, result)
    print(json.dumps({"status": "ok", "schema": result["schema"]}, sort_keys=True))


if __name__ == "__main__":
    main()
