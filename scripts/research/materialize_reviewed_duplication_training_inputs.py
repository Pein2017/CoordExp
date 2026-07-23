#!/usr/bin/env python3
"""Materialize explicitly approved duplicate bursts for the local assembler.

Approval is intentionally not inferred from geometry, union receipts, or queue
membership.  A decision JSONL row with ``candidate_id``, ``decision=approve``,
``reviewer``, and a ``comment`` and/or ``evidence`` is the sole promotion
authority.  The resulting files are precisely the three
inputs consumed by ``assemble_duplicate_trajectory_state_banks.py``:

* ``rollout-trajectories.json``;
* ``reviewed-owner-ledger.jsonl``; and
* ``source-preservation-events.jsonl``.

All other rows are neutral.  The materializer is a narrow research-side seam;
it never creates an automatic training label from overlap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import glob
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.assemble_duplicate_trajectory_state_banks import (  # noqa: E402
    AssemblyError,
    exact_complete_rows,
)
from src.inference.backend import token_ids_sha256  # noqa: E402


SCHEMA_VERSION = "reviewed_duplication_training_input_materialization.v1"
QUEUE_SCHEMA_VERSION = "physical_owner_duplication_review_queue.v1"
UNION_SCHEMA_VERSION = "individual_trajectory_union_support.v1"
ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
IMAGE_PAD_TOKEN_ID = 151655
_DUPLICATE_STATUSES = frozenset({"duplicate", "duplicate_owner"})


class MaterializationError(ValueError):
    """Raised when explicit review authority cannot be replayed exactly."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _clone(value: Any) -> Any:
    return json.loads(_canonical(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise MaterializationError(f"JSON document must be an object: {source}")
    return dict(value)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializationError(f"invalid JSONL at {source}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise MaterializationError(f"JSONL row must be an object: {source}:{line_number}")
            rows.append(dict(value))
    return rows


def _string(value: Any, field: str) -> str:
    result = str(value if value is not None else "").strip()
    if not result:
        raise MaterializationError(f"{field} must be non-empty")
    return result


def _image_id(value: Any, field: str) -> str:
    if isinstance(value, bool):
        raise MaterializationError(f"{field} must be an image ID")
    return _string(value, field)


def _int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise MaterializationError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MaterializationError(f"{field} must be an integer") from exc
    if result < minimum:
        raise MaterializationError(f"{field} must be >= {minimum}")
    return result


def _sha(value: Any, field: str) -> str:
    result = _string(value, field)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result.lower()):
        raise MaterializationError(f"{field} must be a SHA-256 hex digest")
    return result.lower()


def _status(value: Mapping[str, Any]) -> str:
    return str(value.get("entity_status", "")).strip().lower()


def _owner_id(value: Mapping[str, Any]) -> str | None:
    for key in ("owner_id", "candidate_owner_id"):
        raw = value.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def _category(value: Mapping[str, Any]) -> str | None:
    raw = value.get("owner_category", value.get("category"))
    result = " ".join(str(raw if raw is not None else "").strip().lower().split())
    return result or None


def _trajectory_id(row: Mapping[str, Any], mode: str) -> str:
    explicit = row.get("trajectory_id", row.get("request_id"))
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    if mode == "greedy":
        return "greedy"
    if row.get("seed") is None:
        raise MaterializationError("sampled rollout lacks trajectory_id and seed")
    return f"seed-{_int(row['seed'], 'rollout.seed')}"


def _materialized_trajectory_id(image_id: str, local_trajectory_id: str) -> str:
    """Make the assembler-facing identity globally unique across images."""

    return f"{image_id}:{local_trajectory_id}"


def _expand_paths(values: Iterable[str | Path]) -> list[Path]:
    result: set[Path] = set()
    for raw in values:
        text = str(raw)
        matches = [Path(item) for item in glob.glob(text, recursive=True)] if any(char in text for char in "*?[") else []
        candidates = matches or [Path(text)]
        for candidate in candidates:
            candidate = candidate.expanduser()
            if candidate.is_dir():
                result.update(item.resolve() for item in candidate.rglob("*.json") if item.is_file())
            elif candidate.is_file():
                result.add(candidate.resolve())
            else:
                raise FileNotFoundError(candidate)
    if not result:
        raise MaterializationError("no rollout JSON files supplied")
    return sorted(result, key=str)


def _prediction_map(row: Mapping[str, Any], trajectory_id: str, *, field: str) -> dict[tuple[int, str], dict[str, Any]]:
    parser = row.get("predictions", {})
    values = parser if isinstance(parser, list) else (parser.get("predictions", parser.get("rows", [])) if isinstance(parser, Mapping) else None)
    if values is None or not isinstance(values, list):
        raise MaterializationError(f"{field}.predictions is malformed")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for ordinal, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise MaterializationError(f"{field}.predictions[{ordinal}] must be an object")
        row_index = _int(raw.get("generated_order", raw.get("row_index", ordinal)), f"{field}.predictions[{ordinal}].generated_order")
        prediction_id = raw.get("object_span_id", raw.get("prediction_id"))
        parsed_id = str(prediction_id).strip() if prediction_id is not None else f"{trajectory_id}:row-{row_index}"
        key = (row_index, parsed_id)
        if key in result:
            raise MaterializationError(f"duplicate exact prediction identity: {field}:{row_index}:{parsed_id}")
        result[key] = dict(raw)
    return result


def _load_rollouts(paths: Iterable[str | Path], *, expected_mode: str) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, str]]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for path in _expand_paths(paths):
        document = _read_json(path)
        if document.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
            raise MaterializationError(f"unexpected rollout schema: {path}")
        config = document.get("config")
        rows = document.get("rollouts")
        if not isinstance(config, Mapping) or not isinstance(rows, list):
            raise MaterializationError(f"invalid rollout artifact contract: {path}")
        if str(config.get("decode_mode", "")) != expected_mode:
            raise MaterializationError(f"{path} is not a {expected_mode} artifact")
        digest = _sha256_file(path)
        sources.append({"path": str(path), "sha256": digest, "decode_mode": expected_mode})
        for ordinal, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                raise MaterializationError(f"{path}.rollouts[{ordinal}] must be an object")
            mode = str(raw.get("decode_mode", config.get("decode_mode")))
            if mode != expected_mode:
                raise MaterializationError(f"{path}.rollouts[{ordinal}] has inconsistent decode_mode")
            image_id = _image_id(raw.get("image_id"), f"{path}.rollouts[{ordinal}].image_id")
            trajectory_id = _trajectory_id(raw, mode)
            key = (image_id, trajectory_id)
            if key in records:
                raise MaterializationError(f"duplicate trajectory identity: {image_id}:{trajectory_id}")
            prompt = raw.get("prompt_token_ids")
            generated = raw.get("generated_token_ids")
            if not isinstance(prompt, list) or not isinstance(generated, list):
                raise MaterializationError(f"{path}.rollouts[{ordinal}] lacks exact prompt/generated token IDs")
            try:
                prompt_ids = [int(value) for value in prompt]
                generated_ids = [int(value) for value in generated]
            except (TypeError, ValueError) as exc:
                raise MaterializationError(f"{path}.rollouts[{ordinal}] token IDs are malformed") from exc
            prompt_hash = _sha(raw.get("prompt_token_ids_sha256"), f"{path}.rollouts[{ordinal}].prompt_token_ids_sha256")
            generated_hash = _sha(raw.get("generated_token_ids_sha256"), f"{path}.rollouts[{ordinal}].generated_token_ids_sha256")
            if token_ids_sha256(prompt_ids) != prompt_hash or token_ids_sha256(generated_ids) != generated_hash:
                raise MaterializationError(f"{path}.rollouts[{ordinal}] token hash mismatch")
            metadata = document.get("prompt_metadata")
            metadata_row = metadata.get(str(raw.get("example_id")), metadata.get(image_id)) if isinstance(metadata, Mapping) else None
            records[key] = {
                "image_id": image_id,
                "trajectory_id": trajectory_id,
                "decode_mode": mode,
                "seed": raw.get("seed"),
                "row": dict(raw),
                "config": dict(config),
                "metadata": dict(metadata_row) if isinstance(metadata_row, Mapping) else {},
                "source_path": str(path),
                "source_sha256": digest,
                "model_identity_sha256": _sha256_json(document.get("model_identity")),
            }
    return records, sorted(sources, key=lambda item: (item["decode_mode"], item["path"]))


def _load_union(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    source = Path(path).expanduser().resolve(strict=True)
    document = _read_json(source)
    if document.get("schema_version") != UNION_SCHEMA_VERSION:
        raise MaterializationError(f"unexpected union-support schema: {source}")
    images = document.get("image_results", document.get("images"))
    if not isinstance(images, list):
        raise MaterializationError("union-support lacks image_results")
    result: dict[str, dict[str, Any]] = {}
    for ordinal, raw in enumerate(images):
        if not isinstance(raw, Mapping):
            raise MaterializationError(f"union image {ordinal} must be an object")
        image_id = _image_id(raw.get("image_id"), f"union image {ordinal}.image_id")
        if image_id in result:
            raise MaterializationError(f"duplicate union image identity: {image_id}")
        budgets = raw.get("budgets")
        if not isinstance(budgets, list):
            raise MaterializationError(f"union image {image_id} lacks budgets")
        selected = [item for item in budgets if isinstance(item, Mapping) and int(item.get("budget", -1)) == 16]
        if len(selected) != 1:
            raise MaterializationError(f"union image {image_id} requires exactly one budget-16 block")
        receipts = selected[0].get("row_assignment_receipts")
        if not isinstance(receipts, list):
            raise MaterializationError(f"union image {image_id} lacks row receipts")
        by_trajectory: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for receipt_index, raw_receipt in enumerate(receipts):
            if not isinstance(raw_receipt, Mapping):
                raise MaterializationError(f"union receipt {image_id}[{receipt_index}] must be an object")
            receipt = dict(raw_receipt)
            receipt_image = _image_id(receipt.get("image_id", image_id), f"union receipt {image_id}[{receipt_index}].image_id")
            if receipt_image != image_id:
                raise MaterializationError(f"union receipt image mismatch: {image_id}")
            trajectory_id = _string(receipt.get("trajectory_id"), f"union receipt {image_id}[{receipt_index}].trajectory_id")
            row_index = _int(receipt.get("generated_row_index"), f"union receipt {image_id}[{receipt_index}].generated_row_index")
            if row_index in by_trajectory[trajectory_id]:
                raise MaterializationError(f"duplicate union receipt row: {image_id}:{trajectory_id}:{row_index}")
            by_trajectory[trajectory_id][row_index] = receipt
        owners = raw.get("owners", [])
        if not isinstance(owners, list):
            raise MaterializationError(f"union image {image_id}.owners must be a list")
        result[image_id] = {"owners": _clone(owners), "receipts": dict(by_trajectory)}
    return result, {"path": str(source), "sha256": _sha256_file(source)}


def _load_queue(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    source = Path(path).expanduser().resolve(strict=True)
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_read_jsonl(source)):
        candidate_id = _string(raw.get("candidate_id"), f"candidate_queue[{index}].candidate_id")
        if candidate_id in result:
            raise MaterializationError(f"duplicate candidate_id in queue: {candidate_id}")
        if raw.get("schema_version") != QUEUE_SCHEMA_VERSION or raw.get("candidate_status") != "high_confidence_candidate_repeated_physical_owner" or raw.get("not_a_training_label") is not True:
            raise MaterializationError(f"candidate_queue[{index}] is not an eligible review-queue candidate")
        result[candidate_id] = raw
    return result, {"path": str(source), "sha256": _sha256_file(source)}


def _load_decisions(path: str | Path, candidates: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, str]], dict[str, str]]:
    source = Path(path).expanduser().resolve(strict=True)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_read_jsonl(source)):
        allowed = {"candidate_id", "decision", "reviewer", "comment", "evidence"}
        if not {"candidate_id", "decision", "reviewer"}.issubset(raw) or set(raw) - allowed:
            raise MaterializationError("decision rows require candidate_id, decision, reviewer, and optional comment/evidence only")
        candidate_id = _string(raw.get("candidate_id"), f"decision[{index}].candidate_id")
        decision_value = _string(raw.get("decision"), f"decision[{index}].decision").lower()
        if decision_value not in {"approve", "approved"}:
            raise MaterializationError("only explicit decision=approve/approved rows may appear in the approval JSONL")
        reviewer = _string(raw.get("reviewer"), f"decision[{index}].reviewer")
        comment_raw = raw.get("comment")
        evidence_raw = raw.get("evidence")
        if comment_raw is None and evidence_raw is None:
            raise MaterializationError("approved decision requires comment or evidence")
        comment = _string(comment_raw, f"decision[{index}].comment") if comment_raw is not None else ""
        evidence = _string(evidence_raw, f"decision[{index}].evidence") if evidence_raw is not None else ""
        note = comment if not evidence else (evidence if not comment else f"{comment}\nEvidence: {evidence}")
        if candidate_id not in candidates:
            raise MaterializationError(f"decision references unknown candidate_id: {candidate_id}")
        if candidate_id in seen:
            raise MaterializationError(f"duplicate approved candidate_id: {candidate_id}")
        seen.add(candidate_id)
        result.append({"candidate_id": candidate_id, "decision": "approve", "reviewer": reviewer, "comment": comment, "evidence": evidence, "note": note})
    if not result:
        raise MaterializationError("decision JSONL contains no explicit approvals")
    return sorted(result, key=lambda item: item["candidate_id"]), {"path": str(source), "sha256": _sha256_file(source)}


def _source_pair_event(rollout: Mapping[str, Any], review: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    event_id = _string(rollout.get("event_id"), f"{field}.rollout.event_id")
    if _string(review.get("event_id"), f"{field}.review.event_id") != event_id:
        raise MaterializationError(f"{field} rollout/review event IDs differ")
    image = rollout.get("image")
    if not isinstance(image, Mapping):
        raise MaterializationError(f"{field}.rollout.image must be an object")
    image_id = _image_id(image.get("image_id"), f"{field}.rollout.image.image_id")
    provenance = review.get("review_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("event_family") != "source_preservation":
        raise MaterializationError(f"{field} is not an immutable Source-preservation pair")
    return {
        "event_id": event_id,
        "image_id": image_id,
        "image": _clone(image),
        "physical_entities": _clone(review.get("physical_entities", [])),
        "source_kind": "state_bank_pair",
        "immutable_source_pair": {"rollout": _clone(rollout), "review": _clone(review)},
        "immutable_source_sha256": _sha256_json({"rollout": rollout, "review": review}),
        "checkpoint_id": _source_checkpoint_id(rollout, field=field),
    }


def _source_checkpoint_id(rollout: Mapping[str, Any], *, field: str) -> str:
    candidates = rollout.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise MaterializationError(f"{field}.rollout must contain one immutable Source candidate")
    generation = candidates[0].get("generation_provenance")
    if not isinstance(generation, Mapping):
        raise MaterializationError(f"{field}.rollout candidate lacks generation provenance")
    return _sha(generation.get("checkpoint_id"), f"{field}.rollout checkpoint_id")


def _load_source_pool(
    *,
    source_events_jsonl: Iterable[str | Path],
    source_rollout_jsonl: str | Path | None,
    source_review_jsonl: str | Path | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    direct_paths = list(source_events_jsonl)
    paired = source_rollout_jsonl is not None or source_review_jsonl is not None
    if bool(direct_paths) == paired:
        raise MaterializationError("provide either preassembled source-event JSONL or one rollout/review pair pool")
    events: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    if direct_paths:
        seen: set[str] = set()
        for raw_path in direct_paths:
            path = Path(raw_path).expanduser().resolve(strict=True)
            sources.append({"path": str(path), "sha256": _sha256_file(path), "kind": "preassembled_source_event_jsonl"})
            for index, raw in enumerate(_read_jsonl(path)):
                event_id = _string(raw.get("event_id"), f"source_event[{index}].event_id")
                image_id = _image_id(raw.get("image_id"), f"source_event[{index}].image_id")
                if event_id in seen:
                    raise MaterializationError(f"duplicate immutable Source event_id: {event_id}")
                seen.add(event_id)
                event = _clone(raw)
                event.update(
                    {
                        "event_id": event_id,
                        "image_id": image_id,
                        "source_kind": "preassembled_source_event",
                        "immutable_source_event": _clone(raw),
                        "immutable_source_sha256": _sha256_json(raw),
                    }
                )
                events.append(event)
    else:
        assert source_rollout_jsonl is not None and source_review_jsonl is not None
        rollout_path = Path(source_rollout_jsonl).expanduser().resolve(strict=True)
        review_path = Path(source_review_jsonl).expanduser().resolve(strict=True)
        rollouts = _read_jsonl(rollout_path)
        reviews = _read_jsonl(review_path)
        rollout_by_id = {_string(row.get("event_id"), f"source rollout[{index}].event_id"): row for index, row in enumerate(rollouts)}
        review_by_id = {_string(row.get("event_id"), f"source review[{index}].event_id"): row for index, row in enumerate(reviews)}
        if len(rollout_by_id) != len(rollouts) or len(review_by_id) != len(reviews) or set(rollout_by_id) != set(review_by_id):
            raise MaterializationError("Source rollout/review pair IDs must be unique and identical")
        sources.extend(
            [
                {"path": str(rollout_path), "sha256": _sha256_file(rollout_path), "kind": "source_rollout_jsonl"},
                {"path": str(review_path), "sha256": _sha256_file(review_path), "kind": "source_review_jsonl"},
            ]
        )
        source_ids = [
            event_id
            for event_id in sorted(rollout_by_id)
            if isinstance(review_by_id[event_id].get("review_provenance"), Mapping)
            and review_by_id[event_id]["review_provenance"].get("event_family") == "source_preservation"
        ]
        if not source_ids:
            raise MaterializationError("Source rollout/review pool contains no source_preservation pairs")
        events = [_source_pair_event(rollout_by_id[event_id], review_by_id[event_id], field=f"source pair {event_id}") for event_id in source_ids]
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_image[str(event["image_id"])].append(event)
    return {image_id: sorted(items, key=lambda item: str(item["event_id"])) for image_id, items in by_image.items()}, sorted(sources, key=lambda item: item["path"])


def _image_pad_interval(prompt: Sequence[int]) -> list[int]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, token in enumerate(prompt):
        if int(token) == IMAGE_PAD_TOKEN_ID and start is None:
            start = index
        elif int(token) != IMAGE_PAD_TOKEN_ID and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(prompt)))
    if len(runs) != 1:
        raise MaterializationError(f"expected one contiguous image-pad run, found {runs}")
    return list(runs[0])


def _review_row(
    *,
    trajectory_id: str,
    local_trajectory_id: str,
    row_index: int,
    role: str,
    owner_id: str | None,
    category: str | None,
    burst_id: str | None,
    source: str,
    comment: str,
    trusted: bool,
    reviewer: str = "materialize-reviewed-duplication-training-inputs",
) -> dict[str, Any]:
    dimensions = "trusted" if trusted else "unknown"
    return {
        "trajectory_id": trajectory_id,
        "row_index": row_index,
        "physical_owner_id": owner_id,
        "review_role": role,
        "usable_owner_row": role in {"accepted", "recovery"},
        "burst_id": burst_id,
        "category": category,
        "entity_review_status": dimensions,
        "category_review_status": dimensions,
        "geometry_review_status": dimensions,
        "binding_review_status": dimensions,
        "review_provenance": {
            "source": source,
            "reviewer": reviewer,
            "confidence": "approved" if trusted and role in {"duplicate", "recovery"} else ("verified" if trusted else "neutral"),
            "comment": comment,
            "local_trajectory_id": local_trajectory_id,
        },
    }


def _exact_image_extent(image: Mapping[str, Any], axis: str) -> int:
    value = image.get(axis)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MaterializationError(f"exact target image.{axis} must be a positive integer")
    return value


def _union_pixel_bbox_to_norm1000_bins(
    value: Any,
    *,
    image_width: int,
    image_height: int,
    field: str,
) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise MaterializationError(f"{field} must be a four-value absolute-pixel xyxy box")
    extents = (image_width, image_height, image_width, image_height)
    bins: list[int] = []
    for index, (raw, extent) in enumerate(zip(value, extents, strict=True)):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise MaterializationError(f"{field}[{index}] must be a finite numeric pixel coordinate")
        pixel = float(raw)
        if not math.isfinite(pixel):
            raise MaterializationError(f"{field}[{index}] must be a finite numeric pixel coordinate")
        # CoordExp's norm1000 conversion convention is nearest-bin rounding,
        # then a strict 0..999 clamp for the V1 coordinate-token surface.
        bins.append(max(0, min(999, int(round(pixel * 1000.0 / float(extent))))))
    x1, y1, x2, y2 = bins
    if x2 <= x1 or y2 <= y1:
        raise MaterializationError(
            f"{field} collapses after absolute-pixel to norm1000 conversion: {bins}"
        )
    return bins


def _physical_entities(
    union_image: Mapping[str, Any],
    source_event: Mapping[str, Any],
    *,
    target_image_id: str,
    target_image: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_entities = source_event.get("physical_entities")
    if str(source_event.get("image_id")) == target_image_id and isinstance(source_entities, list) and source_entities:
        return _clone(source_entities)
    image_width = _exact_image_extent(target_image, "width")
    image_height = _exact_image_extent(target_image, "height")
    result: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(union_image.get("owners", [])):
        if not isinstance(raw, Mapping):
            continue
        owner_id = raw.get("owner_id", raw.get("physical_owner_id"))
        category = _category(raw)
        bbox = raw.get("bbox", raw.get("bbox_xyxy"))
        if owner_id is None or category is None or not isinstance(bbox, list) or len(bbox) != 4:
            raise MaterializationError(f"union owner {ordinal} cannot be preserved as a physical entity")
        reference_bbox = _union_pixel_bbox_to_norm1000_bins(
            bbox,
            image_width=image_width,
            image_height=image_height,
            field=f"union owner {ordinal}.bbox",
        )
        result.append(
            {
                "entity_id": str(owner_id),
                "category": category,
                "entity_trusted": True,
                "geometry_trusted": True,
                "reference_bbox": reference_bbox,
                "review_source": "trajectory-union-support owners",
                "reviewer": "materialize-reviewed-duplication-training-inputs",
                "review_confidence": "verified",
                "comment": (
                    "Preserved union-support physical owner after absolute-pixel "
                    f"to norm1000 nearest-bin conversion using {image_width}x{image_height}; "
                    f"source_bbox={bbox}, reference_bbox={reference_bbox}."
                ),
            }
        )
    return result


def _assert_prediction_join(
    *,
    exact: Mapping[str, Any],
    row_index: int,
    prediction_id: str,
    expected_box: Any,
    field: str,
) -> None:
    prediction = _prediction_map(exact["row"], str(exact["trajectory_id"]), field=field).get((row_index, prediction_id))
    if prediction is None:
        raise MaterializationError(f"{field} lacks exact prediction {row_index}:{prediction_id}")
    if expected_box is not None and prediction.get("bbox") != expected_box:
        raise MaterializationError(f"{field} exact prediction geometry does not match reviewed queue")


def _materialize_candidate(
    *,
    candidate: Mapping[str, Any],
    decision_comment: str,
    decision_reviewer: str,
    union_images: Mapping[str, Mapping[str, Any]],
    exact_records: Mapping[tuple[str, str], Mapping[str, Any]],
    source_event: Mapping[str, Any],
    source_match_type: str,
    split: str,
    queue_source: Mapping[str, str],
    decision_source: Mapping[str, str],
    union_source: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    candidate_id = _string(candidate.get("candidate_id"), "candidate.candidate_id")
    image_id = _image_id(candidate.get("image_id"), f"candidate {candidate_id}.image_id")
    local_trajectory_id = _string(candidate.get("trajectory_id"), f"candidate {candidate_id}.trajectory_id")
    trajectory_id = _materialized_trajectory_id(image_id, local_trajectory_id)
    if source_match_type not in {"same_image", "global_fallback"}:
        raise MaterializationError(f"unsupported Source match type: {source_match_type}")
    union_image = union_images.get(image_id)
    exact = exact_records.get((image_id, local_trajectory_id))
    if union_image is None or exact is None:
        raise MaterializationError(f"candidate {candidate_id} cannot join union/exact trajectory")
    expected_rollout = candidate.get("exact_rollout")
    if not isinstance(expected_rollout, Mapping):
        raise MaterializationError(f"candidate {candidate_id} lacks exact rollout provenance")
    if expected_rollout.get("source_path") != exact["source_path"] or expected_rollout.get("source_sha256") != exact["source_sha256"]:
        raise MaterializationError(f"candidate {candidate_id} exact rollout path/hash mismatch")
    row = exact["row"]
    try:
        exact_rows = exact_complete_rows([int(value) for value in row["generated_token_ids"]])
    except AssemblyError as exc:
        raise MaterializationError(f"candidate {candidate_id} exact generated rows are not assembler-compatible") from exc
    if expected_rollout.get("generated_token_ids_sha256") != row.get("generated_token_ids_sha256"):
        raise MaterializationError(f"candidate {candidate_id} generated-token hash mismatch")
    duplicates = candidate.get("duplicate_rows")
    recovery = candidate.get("first_later_trusted_unseen_verified_owner_row")
    earlier = candidate.get("earlier_accepted_owner_row")
    if not isinstance(duplicates, list) or not duplicates or not isinstance(recovery, Mapping) or not isinstance(earlier, Mapping):
        raise MaterializationError(f"candidate {candidate_id} lacks an explicit duplicate/recovery/first-owner record")
    duplicate_indices = [_int(item.get("row_index"), f"candidate {candidate_id}.duplicate_rows[{index}].row_index") for index, item in enumerate(duplicates) if isinstance(item, Mapping)]
    if len(duplicate_indices) != len(duplicates) or duplicate_indices != list(range(duplicate_indices[0], duplicate_indices[0] + len(duplicate_indices))):
        raise MaterializationError(f"candidate {candidate_id} duplicate rows must be contiguous")
    recovery_index = _int(recovery.get("row_index"), f"candidate {candidate_id}.recovery.row_index")
    if recovery_index != duplicate_indices[-1] + 1:
        raise MaterializationError(f"candidate {candidate_id} recovery must immediately follow the selected duplicate burst")
    if recovery_index >= len(exact_rows):
        raise MaterializationError(f"candidate {candidate_id} recovery exceeds exact rollout rows")
    owner_id = _string(candidate.get("physical_owner_id"), f"candidate {candidate_id}.physical_owner_id")
    recovery_owner = _string(recovery.get("owner_id"), f"candidate {candidate_id}.recovery.owner_id")
    if recovery_owner == owner_id:
        raise MaterializationError(f"candidate {candidate_id} recovery retains the duplicate owner")
    receipts = union_image["receipts"].get(local_trajectory_id, {})
    first_index = _int(earlier.get("row_index"), f"candidate {candidate_id}.earlier.row_index")
    if first_index >= duplicate_indices[0]:
        raise MaterializationError(f"candidate {candidate_id} first owner is not earlier than its duplicate burst")
    first_receipt = receipts.get(first_index)
    if not isinstance(first_receipt, Mapping) or _status(first_receipt) != "verified_owner" or _owner_id(first_receipt) != owner_id:
        raise MaterializationError(f"candidate {candidate_id} first-owner union receipt is not verified_owner")
    _assert_prediction_join(exact=exact, row_index=first_index, prediction_id=_string(earlier.get("prediction_id"), "candidate earlier prediction_id"), expected_box=earlier.get("prediction_bbox"), field=f"candidate {candidate_id}.earlier")
    prefix: list[dict[str, Any]] = []
    for row_index in range(duplicate_indices[0]):
        receipt = receipts.get(row_index)
        if isinstance(receipt, Mapping) and _status(receipt) == "verified_owner":
            context_owner = _owner_id(receipt)
            context_category = _category(receipt)
            if context_owner is not None and context_category is not None:
                prefix.append(_review_row(trajectory_id=trajectory_id, local_trajectory_id=local_trajectory_id, row_index=row_index, role="accepted", owner_id=context_owner, category=context_category, burst_id=None, source="union-support verified_owner context", comment="Verified context retained; no automatic approval inferred.", trusted=True))
                continue
        # Context uncertainty is preserved as a suffix-stopping neutral row;
        # it is not converted into a negative or used to revoke an explicitly
        # approved burst whose earlier repeated owner is independently verified.
        status = _status(receipt) if isinstance(receipt, Mapping) else "missing_or_malformed"
        prefix.append(_review_row(trajectory_id=trajectory_id, local_trajectory_id=local_trajectory_id, row_index=row_index, role="neutral", owner_id=None, category=None, burst_id=None, source="union-support non-verified prefix context", comment=f"{status}: retained as neutral context; no owner label promoted.", trusted=False))
    ledger_by_index: dict[int, dict[str, Any]] = {item["row_index"]: item for item in prefix}
    for raw_duplicate, row_index in zip(duplicates, duplicate_indices, strict=True):
        assert isinstance(raw_duplicate, Mapping)
        receipt = receipts.get(row_index)
        if not isinstance(receipt, Mapping) or _status(receipt) not in _DUPLICATE_STATUSES or _owner_id(receipt) != owner_id:
            raise MaterializationError(f"candidate {candidate_id} duplicate row {row_index} no longer matches union comparator evidence")
        _assert_prediction_join(exact=exact, row_index=row_index, prediction_id=_string(raw_duplicate.get("prediction_id"), f"candidate {candidate_id}.duplicate prediction_id"), expected_box=raw_duplicate.get("prediction_bbox"), field=f"candidate {candidate_id}.duplicate")
        ledger_by_index[row_index] = _review_row(trajectory_id=trajectory_id, local_trajectory_id=local_trajectory_id, row_index=row_index, role="duplicate", owner_id=owner_id, category=_string(raw_duplicate.get("category"), f"candidate {candidate_id}.duplicate.category"), burst_id=candidate_id, source="explicit main-agent approval", comment=decision_comment, trusted=True, reviewer=decision_reviewer)
    recovery_receipt = receipts.get(recovery_index)
    if not isinstance(recovery_receipt, Mapping) or _status(recovery_receipt) != "verified_owner" or _owner_id(recovery_receipt) != recovery_owner:
        raise MaterializationError(f"candidate {candidate_id} explicit recovery no longer matches a verified_owner union receipt")
    _assert_prediction_join(exact=exact, row_index=recovery_index, prediction_id=_string(recovery.get("prediction_id"), f"candidate {candidate_id}.recovery.prediction_id"), expected_box=recovery.get("prediction_bbox"), field=f"candidate {candidate_id}.recovery")
    ledger_by_index[recovery_index] = _review_row(trajectory_id=trajectory_id, local_trajectory_id=local_trajectory_id, row_index=recovery_index, role="recovery", owner_id=recovery_owner, category=_string(recovery.get("category"), f"candidate {candidate_id}.recovery.category"), burst_id=None, source="explicit main-agent approved recovery", comment=decision_comment, trusted=True, reviewer=decision_reviewer)
    for row_index in range(len(exact_rows)):
        if row_index not in ledger_by_index:
            ledger_by_index[row_index] = _review_row(trajectory_id=trajectory_id, local_trajectory_id=local_trajectory_id, row_index=row_index, role="neutral", owner_id=None, category=None, burst_id=None, source="union-support non-promoted row", comment="Not explicitly approved for duplication training; neutral and suffix-stopping.", trusted=False)
    prompt_ids = [int(value) for value in row["prompt_token_ids"]]
    generated_ids = [int(value) for value in row["generated_token_ids"]]
    checkpoint_id = source_event.get("checkpoint_id")
    if checkpoint_id is None:
        checkpoint_id = _sha256_json({"source_event": source_event.get("immutable_source_sha256"), "model_identity": exact["model_identity_sha256"]})
    checkpoint_id = _sha(checkpoint_id, f"candidate {candidate_id}.source checkpoint")
    config = exact["config"]
    provenance = {
        "mode": str(exact["decode_mode"]),
        "seed": _int(exact.get("seed"), f"candidate {candidate_id}.seed"),
        "temperature": float(config.get("temperature")),
        "top_p": float(config.get("top_p")),
        "repetition_penalty": float(config.get("repetition_penalty")),
        "checkpoint_id": checkpoint_id,
        "prompt_token_ids_sha256": str(row["prompt_token_ids_sha256"]),
    }
    image = source_event.get("image") if str(source_event.get("image_id")) == image_id else None
    if not isinstance(image, Mapping):
        metadata = exact["metadata"]
        image = {"image_id": int(image_id) if image_id.isdigit() else image_id, "path": metadata.get("image_path"), "content_sha256": metadata.get("image_sha256"), "width": metadata.get("width"), "height": metadata.get("height")}
    if _image_id(image.get("image_id"), f"candidate {candidate_id}.source image") != image_id:
        raise MaterializationError(f"candidate {candidate_id} selected Source image mismatch")
    source_state_bank_pair: dict[str, Any] = {}
    if source_event.get("source_kind") == "state_bank_pair":
        immutable_pair = source_event.get("immutable_source_pair")
        if not isinstance(immutable_pair, Mapping):
            raise MaterializationError(
                f"candidate {candidate_id} immutable Source pair is malformed"
            )
        source_rollout = immutable_pair.get("rollout")
        source_review = immutable_pair.get("review")
        if not isinstance(source_rollout, Mapping) or not isinstance(source_review, Mapping):
            raise MaterializationError(
                f"candidate {candidate_id} immutable Source pair lacks rollout/review mappings"
            )
        # The duplicate-bank assembler consumes these canonical wrapper keys.
        # Keep them as exact immutable copies alongside the full pair payload.
        source_state_bank_pair = {
            "state_bank_rollout": _clone(source_rollout),
            "state_bank_review": _clone(source_review),
        }
    trajectory = {
        "trajectory_id": trajectory_id,
        "image_id": int(image_id) if image_id.isdigit() else image_id,
        "split": split,
        "image": _clone(image),
        "physical_entities": _physical_entities(
            union_image,
            source_event,
            target_image_id=image_id,
            target_image=image,
        ),
        "executed_prompt_token_ids": prompt_ids,
        "executed_prompt_token_ids_sha256": str(row["prompt_token_ids_sha256"]),
        "image_pad_interval": _image_pad_interval(prompt_ids),
        "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": str(row["generated_token_ids_sha256"]),
        "generation_provenance": provenance,
        "materialization_provenance": {
            "schema_version": SCHEMA_VERSION,
            "local_trajectory_id": local_trajectory_id,
            "emitted_trajectory_id": trajectory_id,
            "exact_rollout": {
                "path": exact["source_path"],
                "sha256": exact["source_sha256"],
                "model_identity_sha256": exact["model_identity_sha256"],
            },
        },
    }
    source_output = {
        "event_id": str(source_event["event_id"]),
        "image_id": image_id,
        "trajectory_id": trajectory_id,
        "burst_id": candidate_id,
        "source_burst_key": {"trajectory_id": trajectory_id, "burst_id": candidate_id},
        "immutable_source_event_id": str(source_event["event_id"]),
        "immutable_source_image_id": str(source_event["image_id"]),
        "source_match_type": source_match_type,
        "immutable_source_kind": str(source_event["source_kind"]),
        "immutable_source_sha256": str(source_event["immutable_source_sha256"]),
        "immutable_source_payload": _clone(source_event.get("immutable_source_pair", source_event.get("immutable_source_event"))),
        **source_state_bank_pair,
        "materialization_provenance": {
            "schema_version": SCHEMA_VERSION,
            "approved_candidate_id": candidate_id,
            "local_trajectory_id": local_trajectory_id,
            "emitted_trajectory_id": trajectory_id,
            "decision_comment": decision_comment,
            "decision_reviewer": decision_reviewer,
            "candidate_queue": dict(queue_source),
            "decision_jsonl": dict(decision_source),
            "union_support": dict(union_source),
            "exact_rollout": {"path": exact["source_path"], "sha256": exact["source_sha256"], "model_identity_sha256": exact["model_identity_sha256"]},
        },
    }
    return trajectory, [ledger_by_index[index] for index in sorted(ledger_by_index)], source_output


def build_reviewed_duplication_training_inputs(
    *,
    candidate_queue_jsonl: str | Path,
    approved_decisions_jsonl: str | Path,
    union_support_json: str | Path,
    greedy_json_paths: Iterable[str | Path],
    sampled_json_paths: Iterable[str | Path] = (),
    source_events_jsonl: Iterable[str | Path] = (),
    source_rollout_jsonl: str | Path | None = None,
    source_review_jsonl: str | Path | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Materialize only explicit approvals into strict assembler inputs."""

    if split not in {"train", "eval"}:
        raise MaterializationError("split must be train or eval")
    candidates, queue_source = _load_queue(candidate_queue_jsonl)
    decisions, decision_source = _load_decisions(approved_decisions_jsonl, candidates)
    union_images, union_source = _load_union(union_support_json)
    greedy, greedy_sources = _load_rollouts(greedy_json_paths, expected_mode="greedy")
    sampled_paths = list(sampled_json_paths)
    sampled, sampled_sources = (_load_rollouts(sampled_paths, expected_mode="sampled") if sampled_paths else ({}, []))
    exact_records = dict(greedy)
    for key, record in sampled.items():
        if key in exact_records:
            raise MaterializationError(f"duplicate trajectory identity across exact inputs: {key[0]}:{key[1]}")
        exact_records[key] = record
    source_by_image, source_sources = _load_source_pool(source_events_jsonl=source_events_jsonl, source_rollout_jsonl=source_rollout_jsonl, source_review_jsonl=source_review_jsonl)
    selected_by_trajectory: set[tuple[str, str]] = set()
    used_source_events: set[str] = set()
    global_source_events = sorted(
        [event for events in source_by_image.values() for event in events],
        key=lambda item: str(item["event_id"]),
    )
    trajectories: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    source_events: list[dict[str, Any]] = []
    for decision in decisions:
        candidate = candidates[decision["candidate_id"]]
        image_id = _image_id(candidate.get("image_id"), "candidate.image_id")
        trajectory_id = _string(candidate.get("trajectory_id"), "candidate.trajectory_id")
        key = (image_id, trajectory_id)
        if key in selected_by_trajectory:
            raise MaterializationError(f"at most one approved burst per trajectory is supported by this bounded materializer: {image_id}:{trajectory_id}")
        selected_by_trajectory.add(key)
        same_image_choices = [
            event for event in source_by_image.get(image_id, [])
            if str(event["event_id"]) not in used_source_events
        ]
        if same_image_choices:
            source_event = same_image_choices[0]
            source_match_type = "same_image"
        else:
            fallback_choices = [
                event for event in global_source_events
                if str(event["event_id"]) not in used_source_events
            ]
            if not fallback_choices:
                raise MaterializationError(f"no unused immutable Source event remains for approved candidate {decision['candidate_id']}")
            source_event = fallback_choices[0]
            source_match_type = "global_fallback"
        if str(source_event["event_id"]) in used_source_events:
            raise MaterializationError("immutable Source event would be replicated")
        used_source_events.add(str(source_event["event_id"]))
        trajectory, reviewed_rows, output_source_event = _materialize_candidate(candidate=candidate, decision_comment=decision["note"], decision_reviewer=decision["reviewer"], union_images=union_images, exact_records=exact_records, source_event=source_event, source_match_type=source_match_type, split=split, queue_source=queue_source, decision_source=decision_source, union_source=union_source)
        trajectories.append(trajectory)
        ledger.extend(reviewed_rows)
        source_events.append(output_source_event)
    trajectories.sort(key=lambda item: str(item["trajectory_id"]))
    ledger.sort(key=lambda item: (str(item["trajectory_id"]), int(item["row_index"])))
    source_events.sort(key=lambda item: str(item["event_id"]))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "approval_authority": "explicit approved-decisions JSONL only",
        "approved_candidate_count": len(decisions),
        "trajectory_count": len(trajectories),
        "reviewed_owner_ledger_row_count": len(ledger),
        "source_preservation_event_count": len(source_events),
        "source_events_unique": len(used_source_events) == len(source_events),
        "source_match_type_counts": {
            match_type: sum(event["source_match_type"] == match_type for event in source_events)
            for match_type in ("same_image", "global_fallback")
        },
        "inputs": {
            "candidate_queue": queue_source,
            "approved_decisions": decision_source,
            "union_support": union_source,
            "exact_rollouts": sorted(greedy_sources + sampled_sources, key=lambda item: (item["decode_mode"], item["path"])),
            "source_pool": source_sources,
        },
        "approved_candidates": [
            {"candidate_id": item["candidate_id"], "comment": item["comment"]}
            for item in decisions
        ],
        "admitted_bursts": [
            {"candidate_id": event["burst_id"], "trajectory_id": event["trajectory_id"], "image_id": event["image_id"], "source_event_id": event["immutable_source_event_id"], "source_match_type": event["source_match_type"]}
            for event in source_events
        ],
    }
    return {"rollout_trajectories": trajectories, "reviewed_owner_ledger": ledger, "source_preservation_events": source_events, "receipt": receipt}


def write_reviewed_duplication_training_inputs(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = {
        "rollout_trajectories": destination / "rollout-trajectories.json",
        "reviewed_owner_ledger": destination / "reviewed-owner-ledger.jsonl",
        "source_preservation_events": destination / "source-preservation-events.jsonl",
        "receipt": destination / "receipt.json",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite materialized input(s): " + ", ".join(str(path) for path in existing))
    targets["rollout_trajectories"].write_bytes(_canonical(result["rollout_trajectories"]) + b"\n")
    for key in ("reviewed_owner_ledger", "source_preservation_events"):
        with targets[key].open("wb") as handle:
            for row in result[key]:
                handle.write(_canonical(row) + b"\n")
    targets["receipt"].write_bytes(_canonical(result["receipt"]) + b"\n")
    return targets


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-queue", required=True)
    parser.add_argument("--approved-decisions", required=True)
    parser.add_argument("--union-support", required=True)
    parser.add_argument("--greedy-json", action="append", required=True)
    parser.add_argument("--sampled-json", action="append", default=[])
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-events-jsonl", action="append")
    source_group.add_argument("--source-rollout-jsonl")
    parser.add_argument("--source-review-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    args = parser.parse_args(argv)
    if bool(args.source_rollout_jsonl) != bool(args.source_review_jsonl):
        parser.error("--source-rollout-jsonl and --source-review-jsonl must be provided together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_reviewed_duplication_training_inputs(
        candidate_queue_jsonl=args.candidate_queue,
        approved_decisions_jsonl=args.approved_decisions,
        union_support_json=args.union_support,
        greedy_json_paths=args.greedy_json,
        sampled_json_paths=args.sampled_json,
        source_events_jsonl=args.source_events_jsonl or (),
        source_rollout_jsonl=args.source_rollout_jsonl,
        source_review_jsonl=args.source_review_jsonl,
        split=args.split,
    )
    paths = write_reviewed_duplication_training_inputs(result, args.output_dir)
    print(json.dumps({key: str(path) for key, path in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
