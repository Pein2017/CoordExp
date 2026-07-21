#!/usr/bin/env python3
"""Assemble a positive-only best-sampled-trajectory StateBank screen.

This command is intentionally an experiment-local *assembler*.  It consumes
the frozen 16-sample route analysis, exact-token rollout artifacts, the
generation-7 annotation JSONL, and a canonical StateBank manifest.  It never
decodes or retokenizes text and never runs inference.

The route analysis is the authority for owner matching and route quality.  A
sampled route is admitted only when it is naturally stopped, parser-accepted,
strictly improves the greedy verified-owner set, and does not add confirmed
duplicates or malformed rows.  Exact sampled rows up to the final newly added
owner are then screened for the conservative geometry/owner rule before being
joined to the canonical ``assemble_state_bank`` utility.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.errors import ArtifactContractError  # noqa: E402
from src.config.fingerprint import sha256_file, sha256_json  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402
from src.rollout_calibration import (  # noqa: E402
    assemble_state_bank,
    load_state_bank,
    load_state_bank_manifest_binding,
)

from scripts.research.run_greedy_prefix_forced_owner_path import (  # noqa: E402
    BOX_END,
    BOX_START,
    OBJECT_REF_END,
    OBJECT_REF_START,
    split_generated_rows,
)


SCHEMA_VERSION = "positive_path_imitation_state_bank_assembler.v1"
PRE_STATE_BANK_SCHEMA_VERSION = "positive_path_imitation_pre_state_bank.v1"
ROUTE_ANALYSIS_SCHEMA_VERSION = "individual_trajectory_union_support.v1"
ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END = 152670
IMAGE_PAD_TOKEN_ID = 151655
GEOMETRY_IOU_THRESHOLD = 0.75
DEFAULT_REQUIRED_EVENT_MULTIPLE = None


class AssemblyError(ValueError):
    """Raised when source evidence cannot support this assembly."""


@dataclass(frozen=True)
class RouteChoice:
    """One deterministic sampled-route choice and its screening receipt."""

    image_id: str
    route_id: str | None
    seed: int | None
    greedy_owner_ids: tuple[str, ...]
    selected_owner_ids: tuple[str, ...]
    added_owner_ids: tuple[str, ...]
    duplicate_delta: int | None
    malformed_delta: int | None
    unresolved_rows: int | None
    last_added_owner_row_index: int | None
    admissible: bool
    rejection_reason: str | None
    candidate_route_count: int
    receipt: dict[str, Any]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssemblyError(f"{name} must be a list")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssemblyError(f"{name} must be a non-empty string")
    return value


def _int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssemblyError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise AssemblyError(f"{name} must be >= {minimum}")
    return value


def _float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{name} must be numeric")
    result = float(value)
    if not (result == result and abs(result) != float("inf")):
        raise AssemblyError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise AssemblyError(f"{name} must be >= {minimum}")
    return result


def _read_json(path: str | Path) -> Any:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssemblyError(f"invalid JSON: {resolved}: {exc}") from exc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects while preserving line order."""

    resolved = Path(path).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssemblyError(f"invalid JSONL at {resolved}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AssemblyError(f"JSONL row at {resolved}:{line_number} must be an object")
            rows.append(value)
    return rows


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssemblyError(f"output already exists: {path}")
    path.write_bytes(_canonical(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssemblyError(f"output already exists: {path}")
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical(dict(row)) + b"\n")


def _normalize_category(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _image_id(value: Any, name: str = "image_id") -> str:
    if isinstance(value, bool):
        raise AssemblyError(f"{name} must be an image id")
    text = str(value)
    try:
        return str(int(text))
    except ValueError as exc:
        raise AssemblyError(f"{name} must be an image id") from exc


def _coord_bin(value: Any, name: str) -> int:
    if isinstance(value, str):
        match = value.strip()
        if match.startswith("<|coord_") and match.endswith("|>"):
            match = match[len("<|coord_") : -len("|>")]
        elif match.startswith("coord_"):
            match = match[len("coord_") :]
        try:
            value = int(match)
        except ValueError as exc:
            raise AssemblyError(f"{name} is not a coordinate token: {value!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
        raise AssemblyError(f"{name} must be an integer coordinate bin in [0, 999]")
    return int(value)


def _token_ids(value: Any, name: str) -> list[int]:
    values = _list(value, name)
    result: list[int] = []
    for index, item in enumerate(values):
        result.append(_int(item, f"{name}[{index}]", minimum=0))
    return result


def _norm_sha(value: Any, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64:
        raise AssemblyError(f"{name} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise AssemblyError(f"{name} must be a SHA-256 digest") from exc
    return text


def _route_row_counts(assignment: Mapping[str, Any]) -> dict[str, int]:
    value = assignment.get("row_counts")
    if not isinstance(value, Mapping):
        value = {}
    return {str(key): int(item or 0) for key, item in value.items() if isinstance(item, (int, float))}


def _route_seed(evidence: Mapping[str, Any], assignment: Mapping[str, Any]) -> int:
    value = evidence.get("seed", assignment.get("seed"))
    return _int(value, "route seed", minimum=0)


def _route_receipt(
    image_result: Mapping[str, Any],
    *,
    budget: int = 16,
) -> RouteChoice:
    """Choose an admissible route using the declared lexicographic order.

    The route analysis stores the complete route assignment, so this function
    deliberately does not recompute matching or infer an owner from geometry.
    """

    image_id = _image_id(image_result.get("image_id"))
    greedy_id = _string(image_result.get("greedy_trajectory_id"), "greedy_trajectory_id")
    evidence_by_id = _mapping(image_result.get("trajectory_evidence"), "trajectory_evidence")
    sampled_ids = [str(item) for item in _list(image_result.get("sampled_trajectory_ids"), "sampled_trajectory_ids")]
    budgets = _list(image_result.get("budgets"), "budgets")
    selected_budget: Mapping[str, Any] | None = None
    for raw in budgets:
        item = _mapping(raw, "budget")
        if _int(item.get("budget"), "budget", minimum=1) == int(budget):
            selected_budget = item
            break
    if selected_budget is None:
        raise AssemblyError(f"image {image_id} lacks budget {budget}")
    assignments = _mapping(selected_budget.get("trajectory_assignments"), "trajectory_assignments")
    owner_sets_raw = _mapping(selected_budget.get("owner_sets"), "owner_sets")
    greedy_assignment = _mapping(assignments.get(greedy_id), f"trajectory_assignments[{greedy_id}]")
    greedy_owner_ids = tuple(sorted(str(item) for item in _list(owner_sets_raw.get(greedy_id), "greedy owner set")))
    greedy_counts = _route_row_counts(greedy_assignment)
    greedy_dup = int(greedy_counts.get("duplicate", 0))
    greedy_malformed = int(greedy_counts.get("malformed", 0))

    candidate_rows: list[tuple[tuple[Any, ...], str, Mapping[str, Any], Mapping[str, Any], tuple[str, ...], tuple[str, ...]]] = []
    rejection_reasons: Counter[str] = Counter()
    route_receipts: dict[str, Any] = {}
    for trajectory_id in sorted(sampled_ids):
        evidence = _mapping(evidence_by_id.get(trajectory_id), f"trajectory_evidence[{trajectory_id}]")
        assignment = _mapping(assignments.get(trajectory_id), f"trajectory_assignments[{trajectory_id}]")
        owner_ids = tuple(sorted(str(item) for item in _list(owner_sets_raw.get(trajectory_id), f"owner_sets[{trajectory_id}]")))
        owner_set = set(owner_ids)
        greedy_set = set(greedy_owner_ids)
        added = tuple(sorted(owner_set - greedy_set))
        parser = _mapping(evidence.get("parser"), f"trajectory_evidence[{trajectory_id}].parser")
        parse_status = str(parser.get("parse_status"))
        stop_reason = str(evidence.get("stop_reason"))
        counts = _route_row_counts(assignment)
        duplicate = int(counts.get("duplicate", 0))
        malformed = int(counts.get("malformed", assignment.get("malformed_row_count", 0) or 0))
        unresolved = int(counts.get("unresolved", 0))
        seed = _route_seed(evidence, assignment)
        reasons: list[str] = []
        if parse_status != "accepted":
            reasons.append(f"parser_status:{parse_status}")
        if stop_reason != "im_end":
            reasons.append(f"stop_reason:{stop_reason}")
        if not owner_set > greedy_set:
            reasons.append("owner_set_not_strict_superset")
        if duplicate > greedy_dup:
            reasons.append("duplicate_count_worse_than_greedy")
        if malformed > greedy_malformed:
            reasons.append("malformed_count_worse_than_greedy")
        route_receipts[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "seed": seed,
            "source_path": str(evidence.get("source_path", "")),
            "parser_status": parse_status,
            "stop_reason": stop_reason,
            "greedy_owner_ids": list(greedy_owner_ids),
            "owner_ids": list(owner_ids),
            "added_owner_ids": list(added),
            "added_owner_count": len(added),
            "duplicate_count": duplicate,
            "malformed_count": malformed,
            "unresolved_rows": unresolved,
            "duplicate_delta": duplicate - greedy_dup,
            "malformed_delta": malformed - greedy_malformed,
            "admissible": not reasons,
            "rejection_reasons": reasons,
        }
        if reasons:
            rejection_reasons.update(reasons)
            continue
        key = (-len(added), duplicate - greedy_dup, malformed - greedy_malformed, unresolved, seed)
        candidate_rows.append((key, trajectory_id, evidence, assignment, owner_ids, added))

    if not candidate_rows:
        receipt = {
            "image_id": image_id,
            "budget": int(budget),
            "greedy_trajectory_id": greedy_id,
            "greedy_owner_ids": list(greedy_owner_ids),
            "greedy_duplicate_count": greedy_dup,
            "greedy_malformed_count": greedy_malformed,
            "candidate_route_count": 0,
            "selected_route_id": None,
            "selected_seed": None,
            "admissible": False,
            "rejection_reason": "no_admissible_sampled_route",
            "candidate_routes": route_receipts,
            "rejection_counts": dict(sorted(rejection_reasons.items())),
        }
        return RouteChoice(
            image_id=image_id,
            route_id=None,
            seed=None,
            greedy_owner_ids=greedy_owner_ids,
            selected_owner_ids=(),
            added_owner_ids=(),
            duplicate_delta=None,
            malformed_delta=None,
            unresolved_rows=None,
            last_added_owner_row_index=None,
            admissible=False,
            rejection_reason="no_admissible_sampled_route",
            candidate_route_count=0,
            receipt=receipt,
        )

    _, route_id, evidence, assignment, selected_owner_ids, added_owner_ids = sorted(candidate_rows, key=lambda item: (item[0], item[1]))[0]
    counts = _route_row_counts(assignment)
    selected_seed = _route_seed(evidence, assignment)
    receipts = _list(assignment.get("row_assignment_receipts"), f"trajectory_assignments[{route_id}].row_assignment_receipts")
    added_rows = [
        _int(item.get("generated_row_index"), "generated_row_index", minimum=0)
        for item in receipts
        if isinstance(item, Mapping) and str(item.get("owner_id")) in set(added_owner_ids)
    ]
    if not added_rows:
        raise AssemblyError(f"route {route_id} has added owners but no added-owner row receipt")
    last_added = max(added_rows)
    selected = route_receipts[route_id]
    selected.update({"last_added_owner_row_index": last_added})
    receipt = {
        "image_id": image_id,
        "budget": int(budget),
        "greedy_trajectory_id": greedy_id,
        "greedy_owner_ids": list(greedy_owner_ids),
        "greedy_duplicate_count": greedy_dup,
        "greedy_malformed_count": greedy_malformed,
        "candidate_route_count": len(candidate_rows),
        "selected_route_id": route_id,
        "selected_seed": selected_seed,
        "selected_route": selected,
        "candidate_routes": route_receipts,
        "last_added_owner_row_index": last_added,
        "admissible": True,
    }
    return RouteChoice(
        image_id=image_id,
        route_id=route_id,
        seed=selected_seed,
        greedy_owner_ids=greedy_owner_ids,
        selected_owner_ids=tuple(selected_owner_ids),
        added_owner_ids=tuple(added_owner_ids),
        duplicate_delta=int(counts.get("duplicate", 0)) - greedy_dup,
        malformed_delta=int(counts.get("malformed", assignment.get("malformed_row_count", 0) or 0)) - greedy_malformed,
        unresolved_rows=int(counts.get("unresolved", 0)),
        last_added_owner_row_index=last_added,
        admissible=True,
        rejection_reason=None,
        candidate_route_count=len(candidate_rows),
        receipt=receipt,
    )


def select_best_route(image_result: Mapping[str, Any], *, budget: int = 16) -> dict[str, Any]:
    """Public pure route-selection helper returning a JSON-safe receipt."""

    return _route_receipt(image_result, budget=budget).receipt


def _row_geometry_receipt(row: Mapping[str, Any], owners: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status = str(row.get("entity_status", ""))
    owner_id = None if row.get("owner_id") is None else str(row.get("owner_id"))
    category = _normalize_category(row.get("category", row.get("owner_category", "")))
    category_owners = [item for item in owners if _normalize_category(item.get("category")) == category]
    iou = 0.0
    if row.get("intersection_over_union") is not None:
        iou = _float(row.get("intersection_over_union"), "intersection_over_union", minimum=0.0)
    owner_known = owner_id is not None and any(str(item.get("owner_id")) == owner_id for item in owners)
    unique_category = len(category_owners) == 1
    # A unique same-category owner is sufficient to trust the owner identity
    # and retain schema/description imitation, but does not by itself trust
    # coordinates.  Coordinate sites stay present (the exact row is retained)
    # and the canonical trainer masks them while geometry_review_status is
    # unknown.  Multi-instance rows require the explicit 0.75 IoU gate; below
    # that threshold the whole row is excluded.
    eligible = status == "verified_owner" and owner_known and (unique_category or iou >= GEOMETRY_IOU_THRESHOLD)
    geometry_trusted = bool(eligible and iou >= GEOMETRY_IOU_THRESHOLD)
    if status != "verified_owner":
        reason = f"entity_status:{status or 'missing'}"
    elif not owner_known:
        reason = "owner_not_in_physical_ledger"
    elif unique_category and iou >= GEOMETRY_IOU_THRESHOLD:
        reason = "unique_gt_category_and_iou_at_least_0.75"
    elif unique_category:
        reason = "unique_gt_category_owner_trusted_geometry_unknown"
    elif iou >= GEOMETRY_IOU_THRESHOLD:
        reason = "multi_instance_iou_at_least_0.75"
    else:
        reason = "multi_instance_iou_below_0.75"
    return {
        "gradient_eligible": bool(eligible),
        "entity_status": status,
        "owner_id": owner_id,
        "category": category,
        "category_owner_count": len(category_owners),
        "intersection_over_union": iou,
        "geometry_trusted": geometry_trusted,
        "reason": reason,
    }


def row_gradient_eligibility(row: Mapping[str, Any], owners: Sequence[Mapping[str, Any]]) -> bool:
    """Return the declared geometry/owner trust decision for one row."""

    return bool(_row_geometry_receipt(row, owners)["gradient_eligible"])


def geometry_eligibility_receipt(row: Mapping[str, Any], owners: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Public JSON-safe geometry decision used by focused tests and receipts."""

    return _row_geometry_receipt(row, owners)


def exact_row_site_types(row_token_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Classify every exact row token without decoding or retokenizing."""

    tokens = _token_ids(list(row_token_ids), "row_token_ids")
    if len(tokens) < 9 or tokens[0] != OBJECT_REF_START or tokens[-1] != BOX_END:
        raise AssemblyError("candidate row is not a complete object row")
    try:
        description_end = tokens.index(OBJECT_REF_END)
        box_start = tokens.index(BOX_START, description_end + 1)
    except ValueError as exc:
        raise AssemblyError("candidate row lacks canonical description/box markers") from exc
    if description_end <= 1 or box_start < description_end + 1:
        raise AssemblyError("candidate row has an empty description")
    if tokens.count(OBJECT_REF_START) != 1 or tokens.count(OBJECT_REF_END) != 1 or tokens.count(BOX_START) != 1 or tokens.count(BOX_END) != 1:
        raise AssemblyError("candidate row contains duplicate grammar markers")
    coordinate_offsets = list(range(box_start + 1, len(tokens) - 1))
    if len(coordinate_offsets) != 4 or any(not COORDINATE_TOKEN_START <= tokens[index] < COORDINATE_TOKEN_END for index in coordinate_offsets):
        raise AssemblyError("candidate row must contain exactly four coordinate token IDs")
    sites: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if index in {0, description_end, box_start, len(tokens) - 1}:
            token_type = "schema"
        elif box_start < index < len(tokens) - 1:
            token_type = "coordinate"
        else:
            token_type = "desc_text"
        sites.append({"candidate_token_offset": index, "intended_token_type": token_type})
    return sites


def exact_row_slices(generated_token_ids: Sequence[int], row_index: int) -> tuple[list[int], list[int]]:
    """Return ``(prefix, full_row)`` using exact integer token positions."""

    tokens = _token_ids(list(generated_token_ids), "generated_token_ids")
    rows = split_generated_rows(tokens)
    index = _int(row_index, "row_index", minimum=0)
    if index >= len(rows):
        raise AssemblyError(f"row index {index} exceeds complete generated row count {len(rows)}")
    start = sum(len(row) for row in rows[:index])
    return tokens[:start], list(rows[index])


def _annotation_entities(annotation: Mapping[str, Any], *, source_path: Path) -> list[dict[str, Any]]:
    image_id = _image_id(annotation.get("image_id"), "annotation.image_id")
    objects = _list(annotation.get("objects"), f"annotation[{image_id}].objects")
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(objects):
        item = _mapping(raw, f"annotation[{image_id}].objects[{index}]")
        ann_id = item.get("coco_ann_id")
        if isinstance(ann_id, bool) or not isinstance(ann_id, int):
            raise AssemblyError(f"annotation[{image_id}].objects[{index}].coco_ann_id must be an integer")
        owner_id = f"{image_id}:{ann_id}"
        if owner_id in seen:
            raise AssemblyError(f"duplicate annotation owner: {owner_id}")
        seen.add(owner_id)
        bbox_tokens = _list(item.get("bbox_2d"), f"annotation[{image_id}].objects[{index}].bbox_2d")
        if len(bbox_tokens) != 4:
            raise AssemblyError(f"annotation[{image_id}].objects[{index}].bbox_2d must have four coordinates")
        category = str(item.get("category_name", item.get("desc", ""))).strip()
        if not category:
            raise AssemblyError(f"annotation[{image_id}].objects[{index}] has no category")
        entities.append(
            {
                "entity_id": owner_id,
                "category": category,
                "entity_trusted": True,
                "geometry_trusted": True,
                "reference_bbox": [_coord_bin(value, f"annotation[{image_id}].objects[{index}].bbox_2d[{j}]") for j, value in enumerate(bbox_tokens)],
                "review_source": str(source_path),
                "reviewer": "positive-path-imitation-annotation-ledger",
                "review_confidence": "high",
                "comment": "Canonical generation-7 physical owner annotation.",
            }
        )
    return entities


def _image_pad_interval(prompt_ids: Sequence[int]) -> tuple[int, int]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, token in enumerate(prompt_ids):
        if int(token) == IMAGE_PAD_TOKEN_ID and start is None:
            start = index
        elif int(token) != IMAGE_PAD_TOKEN_ID and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(prompt_ids)))
    if len(runs) != 1:
        raise AssemblyError(f"expected one contiguous image-pad run, found {runs}")
    return runs[0]


def _prefix_status(
    *,
    row_index: int,
    receipts_by_index: Mapping[int, Mapping[str, Any]],
    malformed_count: int,
) -> tuple[str, list[dict[str, Any]]]:
    if row_index == 0:
        return "empty", []
    prior = [receipts_by_index[index] for index in range(row_index) if index in receipts_by_index]
    if len(prior) != row_index or malformed_count > 0 or any(str(item.get("entity_status")) != "verified_owner" for item in prior):
        return "unresolved", []
    proofs = [
        {
            "prefix_object_row_index": index,
            "owner_id": str(item["owner_id"]),
            "review_provenance": {
                "source": "trajectory-union-support-budget16.json",
                "reviewer": "automatic-verified-owner-match",
                "confidence": "high",
                "comment": f"Exact sampled prefix row; IoU={float(item.get('intersection_over_union', 0.0)):.6f}.",
            },
        }
        for index, item in enumerate(prior)
    ]
    return "resolved", proofs


def _route_rollout_index(rows_by_key: Mapping[tuple[str, int], Mapping[str, Any]], image_id: str, seed: int) -> Mapping[str, Any]:
    try:
        return rows_by_key[(image_id, int(seed))]
    except KeyError as exc:
        raise AssemblyError(f"sampled rollout is missing image {image_id}, seed {seed}") from exc


def _build_event_pair(
    *,
    image_result: Mapping[str, Any],
    choice: RouteChoice,
    route_assignment: Mapping[str, Any],
    sampled_rollout: Mapping[str, Any],
    reference_record: Mapping[str, Any],
    physical_entities: Sequence[Mapping[str, Any]],
    checkpoint_id: str,
    analysis_name: str,
    split: str,
    event_weight: float,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    if choice.route_id is None or choice.seed is None or choice.last_added_owner_row_index is None:
        raise AssemblyError("cannot build events for an inadmissible route")
    image_id = choice.image_id
    generated_ids = _token_ids(sampled_rollout.get("generated_token_ids"), f"sampled rollout {image_id}.generated_token_ids")
    generated_hash = _norm_sha(sampled_rollout.get("generated_token_ids_sha256"), f"sampled rollout {image_id}.generated_token_ids_sha256")
    if generated_hash != token_ids_sha256(generated_ids):
        raise AssemblyError(f"sampled rollout {image_id}, seed {choice.seed} generated-token hash mismatch")
    prompt_ids = _token_ids(reference_record.get("executed_prompt_token_ids"), f"reference event {image_id}.executed_prompt_token_ids")
    prompt_hash = _norm_sha(reference_record.get("executed_prompt_token_ids_sha256"), f"reference event {image_id}.executed_prompt_token_ids_sha256")
    if prompt_hash != token_ids_sha256(prompt_ids):
        raise AssemblyError(f"reference event {image_id} prompt-token hash mismatch")
    if sampled_rollout.get("prompt_token_ids") != prompt_ids:
        raise AssemblyError(f"sampled rollout {image_id}, seed {choice.seed} prompt token IDs differ from reference")
    route_receipts = [
        _mapping(item, f"route {choice.route_id}.row_assignment_receipts[{index}]")
        for index, item in enumerate(_list(route_assignment.get("row_assignment_receipts"), f"route {choice.route_id}.row_assignment_receipts"))
    ]
    receipts_by_index = {int(item["generated_row_index"]): item for item in route_receipts}
    malformed_count = int(_route_row_counts(route_assignment).get("malformed", route_assignment.get("malformed_row_count", 0) or 0))
    cutoff = int(choice.last_added_owner_row_index)
    owners = [dict(item) for item in _list(image_result.get("owners"), f"image {image_id}.owners")]
    candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for row_index in range(cutoff + 1):
        receipt = receipts_by_index.get(row_index)
        if receipt is None:
            continue
        geometry = _row_geometry_receipt(receipt, owners)
        if not geometry["gradient_eligible"]:
            continue
        prefix_ids, candidate_ids = exact_row_slices(generated_ids, row_index)
        raw = receipt.get("raw")
        raw_mapping = _mapping(raw, f"route {choice.route_id} row {row_index}.raw")
        coord_bins = [_coord_bin(value, f"route row {row_index}.coord_bins[{j}]") for j, value in enumerate(_list(raw_mapping.get("coord_bins"), f"route row {row_index}.coord_bins"))]
        actual_bins = [int(token) - COORDINATE_TOKEN_START for token in candidate_ids if COORDINATE_TOKEN_START <= int(token) < COORDINATE_TOKEN_END]
        if actual_bins != coord_bins:
            raise AssemblyError(f"route {choice.route_id} row {row_index} coordinate tokens disagree with parser bins")
        prefix_status, prefix_proofs = _prefix_status(
            row_index=row_index,
            receipts_by_index=receipts_by_index,
            malformed_count=malformed_count,
        )
        prefix_unresolved_count = sum(
            1
            for prior_index in range(row_index)
            if prior_index in receipts_by_index
            and str(receipts_by_index[prior_index].get("entity_status")) != "verified_owner"
        )
        event_id = f"positive-path-image-{image_id}-seed-{choice.seed}-row-{row_index}"
        candidate_id = f"sampled-positive-image-{image_id}-seed-{choice.seed}-row-{row_index}"
        generation_provenance = {
            "mode": "sampled",
            "seed": int(choice.seed),
            "temperature": _float(sampled_rollout.get("_temperature", 0.4), "sampled temperature"),
            "top_p": _float(sampled_rollout.get("_top_p", 0.95), "sampled top_p"),
            "repetition_penalty": _float(sampled_rollout.get("_repetition_penalty", 1.0), "sampled repetition_penalty"),
            "checkpoint_id": checkpoint_id,
            "prompt_token_ids_sha256": prompt_hash,
            "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
        }
        candidate = {
            "candidate_id": candidate_id,
            "token_ids": candidate_ids,
            "token_ids_sha256": token_ids_sha256(candidate_ids),
            "generation_provenance": generation_provenance,
            "evidence_text": str(raw_mapping.get("raw_span_text", "")),
        }
        review_candidate = {
            "candidate_id": candidate_id,
            "role": "positive",
            "harmful_kind": None,
            "physical_owner_id": str(receipt.get("owner_id")),
            "coverage_status": "uncovered",
            "entity_review_status": "trusted",
            "geometry_review_status": "trusted" if geometry["geometry_trusted"] else "unknown",
            "entity_eligible": True,
            "geometry_eligible": False,
            "owner_resolution_interval": [0, len(candidate_ids)],
            "coordinate_decision": None,
            "selected_sites": exact_row_site_types(candidate_ids),
        }
        rollout = {
            "event_id": event_id,
            "image": dict(_mapping(reference_record.get("image"), f"reference event {image_id}.image")),
            "split": split,
            "split_group_id": f"image:{image_id}",
            "executed_prompt_token_ids": prompt_ids,
            "executed_prompt_token_ids_sha256": prompt_hash,
            "image_pad_interval": list(reference_record.get("image_pad_interval") or _image_pad_interval(prompt_ids)),
            "prefix_token_ids": prefix_ids,
            "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
            "candidates": [candidate],
        }
        review = {
            "event_id": event_id,
            "admission_status": "accepted",
            "rejection_reason": None,
            "physical_entities": [dict(item) for item in physical_entities],
            "prefix_object_row_count": int(row_index),
            "prefix_coverage_status": prefix_status,
            "prefix_covered_owner_proofs": prefix_proofs,
            "entity_transition_eligible": False,
            "coordinate_boundary_eligible": False,
            "positive_path_imitation_eligible": True,
            "image_balanced_event_weight": float(event_weight),
            "candidates": [review_candidate],
            "review_provenance": {
                "schema_version": SCHEMA_VERSION,
                "policy": "best-sampled-route-positive-row-imitation",
                "analysis_artifact": analysis_name,
                "route_id": choice.route_id,
                "seed": int(choice.seed),
                "generated_row_index": int(row_index),
                "last_added_owner_row_index": int(cutoff),
                "added_owner_ids": list(choice.added_owner_ids),
                "geometry_owner_rule": {
                    "entity_status": "verified_owner",
                    "unique_gt_category_or_iou_at_least": GEOMETRY_IOU_THRESHOLD,
                    "category_owner_count": geometry["category_owner_count"],
                    "intersection_over_union": geometry["intersection_over_union"],
                    "reason": geometry["reason"],
                },
                "prefix_context": {
                    "coverage_status": prefix_status,
                    "duplicate_count": int(_route_row_counts(route_assignment).get("duplicate", 0)),
                    "malformed_count": malformed_count,
                    "unresolved_rows": prefix_unresolved_count,
                    "route_unresolved_rows": int(_route_row_counts(route_assignment).get("unresolved", 0)),
                },
            },
        }
        event_receipt = {
            "event_id": event_id,
            "image_id": image_id,
            "route_id": choice.route_id,
            "seed": int(choice.seed),
            "generated_row_index": int(row_index),
            "owner_id": str(receipt.get("owner_id")),
            "geometry": geometry,
            "source_image_path": str(_mapping(reference_record.get("image"), f"reference event {image_id}.image").get("path", "")),
            "gt_bbox": list(receipt.get("owner_bbox", [])),
            "predicted_bbox": list(receipt.get("bbox", [])),
            "added_owner_count": len(choice.added_owner_ids),
            "unresolved_prefix_rows": prefix_unresolved_count,
            "prefix_coverage_status": prefix_status,
            "candidate_token_count": len(candidate_ids),
            "candidate_token_ids_sha256": token_ids_sha256(candidate_ids),
            "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
            "image_balanced_event_weight": float(event_weight),
        }
        candidates.append((rollout, review, event_receipt))
    return candidates


def _build_events_for_image(
    *,
    image_result: Mapping[str, Any],
    choice: RouteChoice,
    sampled_rollout: Mapping[str, Any],
    reference_record: Mapping[str, Any],
    physical_entities: Sequence[Mapping[str, Any]],
    checkpoint_id: str,
    analysis_name: str,
    split: str,
    event_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if choice.route_id is None:
        return [], [], []
    selected_budget = next(
        _mapping(raw, "budget")
        for raw in _list(image_result.get("budgets"), "budgets")
        if int(_mapping(raw, "budget").get("budget", -1)) == 16
    )
    assignment = _mapping(
        _mapping(selected_budget.get("trajectory_assignments"), "trajectory_assignments").get(choice.route_id),
        f"trajectory_assignments[{choice.route_id}]",
    )
    built = _build_event_pair(
        image_result=image_result,
        choice=choice,
        route_assignment=assignment,
        sampled_rollout=sampled_rollout,
        reference_record=reference_record,
        physical_entities=physical_entities,
        checkpoint_id=checkpoint_id,
        analysis_name=analysis_name,
        split=split,
        event_weight=event_weight,
    )
    return (
        [item[0] for item in built],
        [item[1] for item in built],
        [item[2] for item in built],
    )


def image_balanced_weights(event_counts: Mapping[str, int]) -> dict[str, float]:
    """Assign equal total image weight and mean event weight exactly one."""

    normalized = {str(image_id): _int(count, f"event_counts[{image_id}]", minimum=1) for image_id, count in event_counts.items()}
    if not normalized:
        return {}
    total_events = sum(normalized.values())
    image_count = len(normalized)
    return {image_id: float(total_events) / float(image_count * count) for image_id, count in sorted(normalized.items())}


def _exclusion_key(
    excluded: Sequence[str],
    by_id: Mapping[str, Mapping[str, int]],
) -> tuple[int, int, int, tuple[str, ...]]:
    image_ids = tuple(sorted(str(item) for item in excluded))
    lost_added = sum(int(by_id[item].get("added_owner_count", 0)) for item in image_ids)
    unresolved = sum(int(by_id[item].get("unresolved_rows", 0)) for item in image_ids)
    return (len(image_ids), lost_added, -unresolved, image_ids)


def deterministic_batch_fit_exclusion(
    image_stats: Sequence[Mapping[str, Any]],
    required_event_multiple: int,
) -> dict[str, Any]:
    """Choose the smallest deterministic image exclusion for batch fitting."""

    multiple = _int(required_event_multiple, "required_event_multiple", minimum=1)
    stats: dict[str, dict[str, int]] = {}
    for raw in image_stats:
        item = _mapping(raw, "image_stats")
        image_id = _image_id(item.get("image_id"), "image_stats.image_id")
        if image_id in stats:
            raise AssemblyError(f"duplicate image stats: {image_id}")
        stats[image_id] = {
            "event_count": _int(item.get("event_count"), f"image_stats[{image_id}].event_count", minimum=1),
            "added_owner_count": _int(item.get("added_owner_count", 0), f"image_stats[{image_id}].added_owner_count", minimum=0),
            "unresolved_rows": _int(item.get("unresolved_rows", 0), f"image_stats[{image_id}].unresolved_rows", minimum=0),
        }
    ordered_ids = tuple(sorted(stats))
    total = sum(item["event_count"] for item in stats.values())
    if total % multiple == 0:
        return {
            "required_event_multiple": multiple,
            "total_event_count_before": total,
            "total_event_count_after": total,
            "excluded_image_ids": [],
            "excluded_images": [],
            "status": "already_divisible",
        }
    chosen: tuple[str, ...] | None = None
    chosen_key: tuple[int, int, int, tuple[str, ...]] | None = None
    for count in range(1, len(ordered_ids) + 1):
        # In practice the panel remainder is resolved by one image.  The
        # bounded combination search keeps the tie-break semantics literal and
        # remains small for the declared 119-image screen.
        for combo in itertools.combinations(ordered_ids, count):
            if (total - sum(stats[item]["event_count"] for item in combo)) % multiple:
                continue
            key = _exclusion_key(combo, stats)
            if chosen_key is None or key < chosen_key:
                chosen, chosen_key = combo, key
        if chosen is not None:
            break
    if chosen is None:
        raise AssemblyError(f"cannot make {total} events divisible by {multiple}")
    excluded_images = [
        {
            "image_id": image_id,
            "reason": "required_event_multiple_batch_fit",
            **stats[image_id],
        }
        for image_id in chosen
    ]
    after = total - sum(stats[item]["event_count"] for item in chosen)
    return {
        "required_event_multiple": multiple,
        "total_event_count_before": total,
        "total_event_count_after": after,
        "excluded_image_ids": list(chosen),
        "excluded_images": excluded_images,
        "status": "excluded_images_for_batch_fit",
    }


def _load_rollout_rows(paths: Sequence[Path], *, sampled_only: bool) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    configs: dict[str, Any] = {}
    for path in paths:
        payload = _mapping(_read_json(path), f"rollout artifact {path}")
        if payload.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
            raise AssemblyError(f"unsupported rollout schema in {path}")
        config = _mapping(payload.get("config"), f"rollout artifact {path}.config")
        mode = str(config.get("decode_mode"))
        if sampled_only and mode != "sampled":
            raise AssemblyError(f"sampled artifact has non-sampled decode mode: {path}")
        if not sampled_only and mode != "greedy":
            raise AssemblyError(f"greedy artifact has non-greedy decode mode: {path}")
        configs[str(path.resolve())] = {
            "temperature": float(config.get("temperature", 0.0)),
            "top_p": float(config.get("top_p", 1.0)),
            "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
            "max_new_tokens": int(config.get("max_new_tokens", 0)),
        }
        for raw in _list(payload.get("rollouts"), f"rollout artifact {path}.rollouts"):
            item = dict(_mapping(raw, f"rollout artifact {path}.rollout"))
            image_id = _image_id(item.get("image_id"), f"{path}.rollout.image_id")
            seed = _int(item.get("seed"), f"{path}.rollout.seed", minimum=0)
            key = (image_id, seed)
            if key in rows:
                raise AssemblyError(f"duplicate rollout identity: image {image_id}, seed {seed}")
            item["_source_path"] = str(path.resolve())
            item["_temperature"] = configs[str(path.resolve())]["temperature"]
            item["_top_p"] = configs[str(path.resolve())]["top_p"]
            item["_repetition_penalty"] = configs[str(path.resolve())]["repetition_penalty"]
            rows[key] = item
    return rows, configs


def _source_artifacts(paths: Sequence[tuple[str, Path]]) -> list[dict[str, str]]:
    return [
        {"artifact_id": artifact_id, "sha256": sha256_file(path.resolve(strict=True))}
        for artifact_id, path in sorted(paths, key=lambda item: item[0])
    ]


def _resolve_split(reference_record: Mapping[str, Any]) -> str:
    split = str(reference_record.get("split", ""))
    if split not in {"train", "eval"}:
        raise AssemblyError(f"reference event has unsupported split: {split!r}")
    return split


def build_pre_state_bank(
    *,
    trajectory_analysis: Mapping[str, Any],
    greedy_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    sampled_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    reference_records: Mapping[str, Mapping[str, Any]],
    checkpoint_id: str,
    required_event_multiple: int | None = DEFAULT_REQUIRED_EVENT_MULTIPLE,
    analysis_name: str = "trajectory-union-support-budget16.json",
    annotation_source_path: str | Path = "train-256-clean-rollout.coord.jsonl",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build exact pre-StateBank rollout/review mappings and receipt."""

    if trajectory_analysis.get("schema_version") != ROUTE_ANALYSIS_SCHEMA_VERSION:
        raise AssemblyError("trajectory analysis schema_version is not the canonical union-support schema")
    image_results = _list(trajectory_analysis.get("image_results"), "trajectory_analysis.image_results")
    route_choices: dict[str, RouteChoice] = {}
    image_stats: list[dict[str, Any]] = []
    pre_events: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    excluded_images: list[dict[str, Any]] = []
    for raw_image in sorted(image_results, key=lambda item: int(str(_mapping(item, "image_result").get("image_id")))):
        image_result = _mapping(raw_image, "image_result")
        image_id = _image_id(image_result.get("image_id"))
        choice = _route_receipt(image_result)
        route_choices[image_id] = choice
        if not choice.admissible:
            excluded_images.append({"image_id": image_id, "reason": "no_admissible_sampled_route"})
            continue
        if choice.seed is None or choice.route_id is None:
            raise AssemblyError(f"admissible route for image {image_id} lacks route identity")
        sampled = dict(_route_rollout_index(sampled_rows, image_id, choice.seed))
        evidence = _mapping(_mapping(image_result.get("trajectory_evidence"), "trajectory_evidence").get(choice.route_id), "selected route evidence")
        expected_source = str(evidence.get("source_path", ""))
        if expected_source and Path(expected_source).resolve() != Path(str(sampled.get("_source_path"))).resolve():
            raise AssemblyError(f"selected route source path mismatch for image {image_id}: {expected_source} != {sampled.get('_source_path')}")
        if str(evidence.get("stop_reason")) != "im_end" or str(_mapping(evidence.get("parser"), "selected route parser").get("parse_status")) != "accepted":
            raise AssemblyError(f"selected route evidence is not accepted/naturally stopped for image {image_id}")
        ref = _mapping(reference_records.get(image_id), f"reference_records[{image_id}]")
        annotation = _mapping(annotations.get(image_id), f"annotations[{image_id}]")
        entities = _annotation_entities(annotation, source_path=Path(annotation_source_path).expanduser().resolve())
        owner_ids = {str(item["entity_id"]) for item in entities}
        if not set(choice.selected_owner_ids).issubset(owner_ids):
            raise AssemblyError(f"selected route owner set contains unknown annotation owner for image {image_id}")
        # Build once with a temporary unit weight; weights are replaced after
        # event counts are known, preserving exact event ordering.
        built = _build_events_for_image(
            image_result=image_result,
            choice=choice,
            sampled_rollout=sampled,
            reference_record=ref,
            physical_entities=entities,
            checkpoint_id=checkpoint_id,
            analysis_name=analysis_name,
            split=_resolve_split(ref),
            event_weight=1.0,
        )
        rollouts, reviews, event_receipts = built
        if not rollouts:
            excluded_images.append({
                "image_id": image_id,
                "reason": "no_gradient_eligible_rows",
                "route_id": choice.route_id,
                "seed": choice.seed,
                "last_added_owner_row_index": choice.last_added_owner_row_index,
            })
            continue
        image_stats.append({
            "image_id": image_id,
            "event_count": len(rollouts),
            "added_owner_count": len(choice.added_owner_ids),
            "unresolved_rows": int(choice.unresolved_rows or 0),
            "route_id": choice.route_id,
            "seed": choice.seed,
        })
        for rollout, review, receipt in zip(rollouts, reviews, event_receipts, strict=True):
            event_id = str(rollout["event_id"])
            if event_id in pre_events:
                raise AssemblyError(f"duplicate generated event id: {event_id}")
            pre_events[event_id] = (rollout, review, receipt)

    batch_fit: dict[str, Any] = {
        "required_event_multiple": None,
        "total_event_count_before": sum(int(item["event_count"]) for item in image_stats),
        "total_event_count_after": sum(int(item["event_count"]) for item in image_stats),
        "excluded_image_ids": [],
        "excluded_images": [],
        "status": "disabled",
    }
    excluded_for_fit: set[str] = set()
    if required_event_multiple is not None:
        batch_fit = deterministic_batch_fit_exclusion(image_stats, int(required_event_multiple))
        excluded_for_fit = set(str(item) for item in batch_fit["excluded_image_ids"])
        excluded_images.extend(dict(item) for item in batch_fit["excluded_images"])

    retained_stats = [item for item in image_stats if str(item["image_id"]) not in excluded_for_fit]
    weights = image_balanced_weights({str(item["image_id"]): int(item["event_count"]) for item in retained_stats})
    rollouts: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    event_receipts: list[dict[str, Any]] = []
    retained_ids = {str(item["image_id"]) for item in retained_stats}
    for event_id in sorted(pre_events):
        rollout, review, receipt = pre_events[event_id]
        image_id = str(_mapping(rollout["image"], f"event {event_id}.image").get("image_id"))
        if image_id not in retained_ids:
            continue
        weight = weights[image_id]
        review = dict(review)
        review["image_balanced_event_weight"] = float(weight)
        review_provenance = dict(_mapping(review["review_provenance"], f"event {event_id}.review_provenance"))
        review_provenance["image_balanced_event_weight"] = float(weight)
        review["review_provenance"] = review_provenance
        receipt = dict(receipt)
        receipt["image_balanced_event_weight"] = float(weight)
        receipt["source_image_path"] = str(_mapping(rollout["image"], f"event {event_id}.image").get("path", ""))
        rollouts.append(dict(rollout))
        reviews.append(review)
        event_receipts.append(receipt)

    total_weight = sum(float(item["image_balanced_event_weight"]) for item in event_receipts)
    mean_weight = total_weight / len(event_receipts) if event_receipts else 0.0
    def _best_event(key: Any, *, reverse: bool = True) -> dict[str, Any] | None:
        if not event_receipts:
            return None
        return dict(sorted(event_receipts, key=key, reverse=reverse)[0])

    audit_candidates = {
        "high_gain": _best_event(
            lambda item: (
                int(item.get("added_owner_count", 0)),
                -int(item.get("generated_row_index", 0)),
                str(item.get("image_id")),
            )
        ),
        "crowded": _best_event(
            lambda item: (
                int(item.get("geometry", {}).get("category_owner_count", 0)),
                int(item.get("added_owner_count", 0)),
                str(item.get("image_id")),
            )
        ),
        "near_threshold": _best_event(
            lambda item: (
                -abs(float(item.get("geometry", {}).get("intersection_over_union", 0.0)) - GEOMETRY_IOU_THRESHOLD),
                int(item.get("added_owner_count", 0)),
                str(item.get("image_id")),
            )
        ),
    }
    route_receipts = [route_choices[key].receipt for key in sorted(route_choices, key=lambda value: int(value))]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "pre_state_bank_schema_version": PRE_STATE_BANK_SCHEMA_VERSION,
        "status": "pre_state_bank",
        "source_checkpoint_id": checkpoint_id,
        "route_selection": route_receipts,
        "route_count": len(route_receipts),
        "admissible_route_count": sum(1 for item in route_choices.values() if item.admissible),
        "gradient_bearing_image_count_before_batch_fit": len(image_stats),
        "gradient_event_count_before_batch_fit": sum(int(item["event_count"]) for item in image_stats),
        "gradient_bearing_image_count": len(retained_stats),
        "gradient_event_count": len(rollouts),
        "image_event_counts": [dict(item) for item in retained_stats],
        "image_balanced_event_weight": {
            "mean_over_bank": mean_weight,
            "total_weight": total_weight,
            "image_total_weight": float(len(rollouts) / len(retained_stats)) if retained_stats else 0.0,
            "weights_by_image": weights,
        },
        "batch_fit": batch_fit,
        "excluded_images": sorted(excluded_images, key=lambda item: (int(str(item.get("image_id", "0"))), str(item.get("reason", "")))),
        "event_receipts": event_receipts,
        "audit_candidates": audit_candidates,
        "counts": {
            "rollout_rows": len(rollouts),
            "review_rows": len(reviews),
            "event_receipts": len(event_receipts),
        },
    }
    return rollouts, reviews, receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-analysis", type=Path, required=True)
    parser.add_argument("--greedy-rollout", type=Path, required=True)
    parser.add_argument("--sampled-rollout", type=Path, action="append", required=True, dest="sampled_rollouts")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-event-multiple", type=int, default=DEFAULT_REQUIRED_EVENT_MULTIPLE)
    return parser.parse_args(argv)


def _assemble_or_preserve_pre_state_bank(
    *,
    output_dir: Path,
    rollout_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    receipt: dict[str, Any],
    source_checkpoint: Any,
    prompt_identity_sha256: str,
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write pre-StateBank rows and assemble once optional fields are live."""

    pre_root = output_dir / "pre-state-bank"
    _write_jsonl(pre_root / "rollout_rows.jsonl", rollout_rows)
    _write_jsonl(pre_root / "review_rows.jsonl", review_rows)
    status = "pre_state_bank_optional_schema_pending"
    try:
        manifest = assemble_state_bank(
            output_dir=output_dir / "state-bank",
            rollout_rows=rollout_rows,
            review_rows=review_rows,
            source_checkpoint=source_checkpoint,
            prompt_identity_sha256=prompt_identity_sha256,
            source_artifacts=source_artifacts,
        )
    except ArtifactContractError as exc:
        if exc.code not in {"state_bank.keys", "state_bank.assembler_review_keys"} or "positive_path_imitation" not in str(exc) and "image_balanced_event_weight" not in str(exc):
            raise
        receipt = dict(receipt)
        receipt["assembly_error"] = {"code": exc.code, "message": str(exc)}
    else:
        status = "assembled"
        receipt = dict(receipt)
        receipt["state_bank_manifest"] = manifest.to_artifact_dict()
        receipt["state_bank_manifest_path"] = str((output_dir / "state-bank" / "manifest.json").resolve())
        receipt["status"] = status
        loaded = load_state_bank(
            output_dir / "state-bank" / "manifest.json",
            expected_source_checkpoint=source_checkpoint,
            expected_prompt_identity_sha256=prompt_identity_sha256,
        )
        receipt["state_bank_validation_receipt"] = loaded.validation_receipt.to_artifact_dict()
        return receipt
    receipt["status"] = status
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis_path = args.trajectory_analysis.expanduser().resolve(strict=True)
    greedy_path = args.greedy_rollout.expanduser().resolve(strict=True)
    sampled_paths = [Path(item).expanduser().resolve(strict=True) for item in args.sampled_rollouts]
    annotations_path = args.annotations.expanduser().resolve(strict=True)
    reference_manifest_path = args.reference_manifest.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if len(sampled_paths) != 8:
        raise SystemExit("exactly eight sampled-shard artifacts are required for the K=16 panel")
    analysis = _mapping(_read_json(analysis_path), "trajectory analysis")
    greedy_rows, greedy_configs = _load_rollout_rows([greedy_path], sampled_only=False)
    sampled_rows, sampled_configs = _load_rollout_rows(sampled_paths, sampled_only=True)
    annotation_rows = load_jsonl(annotations_path)
    annotations = {_image_id(row.get("image_id")): row for row in annotation_rows}
    if len(annotations) != len(annotation_rows):
        raise SystemExit("annotations contain duplicate image IDs")
    reference_binding = load_state_bank_manifest_binding(reference_manifest_path)
    reference_records_path = reference_manifest_path.parent / "records.jsonl"
    reference_records_rows = load_jsonl(reference_records_path)
    reference_records = {_image_id(row.get("image", {}).get("image_id")): row for row in reference_records_rows}
    if len(reference_records) != len(reference_records_rows):
        raise SystemExit("reference records contain duplicate image IDs")
    # Attach exact sampler settings to selected sampled rows.  The route
    # analysis remains the owner/quality authority; this only binds generation
    # provenance to the source rollout config.
    for key, row in sampled_rows.items():
        config = sampled_configs.get(str(Path(str(row.get("_source_path"))).resolve()), {})
        row["_temperature"] = config.get("temperature", 0.4)
        row["_top_p"] = config.get("top_p", 0.95)
        row["_repetition_penalty"] = config.get("repetition_penalty", 1.0)
    rollouts, reviews, receipt = build_pre_state_bank(
        trajectory_analysis=analysis,
        greedy_rows=greedy_rows,
        sampled_rows=sampled_rows,
        annotations=annotations,
        reference_records=reference_records,
        checkpoint_id=reference_binding.source_checkpoint_id,
        required_event_multiple=args.required_event_multiple,
        analysis_name=analysis_path.name,
        annotation_source_path=annotations_path,
    )
    source_paths: list[tuple[str, Path]] = [
        ("trajectory-analysis", analysis_path),
        ("greedy-rollout", greedy_path),
        ("annotations", annotations_path),
        ("reference-state-bank-manifest", reference_manifest_path),
        ("reference-state-bank-records", reference_records_path),
    ]
    source_paths.extend((f"sampled-rollout-{index:02d}", path) for index, path in enumerate(sorted(sampled_paths), start=0))
    artifacts = _source_artifacts(source_paths)
    receipt["source_artifacts"] = artifacts
    receipt["prompt_identity_sha256"] = reference_binding.prompt_identity_sha256
    receipt["source_checkpoint"] = reference_binding.source_checkpoint.to_artifact_dict()
    split_identity = [
        {"image_id": str(_mapping(row["image"], "event image").get("image_id")), "split": row["split"], "split_group_id": row["split_group_id"]}
        for row in rollouts
    ]
    receipt["split_identity"] = {"assignments": split_identity, "sha256": sha256_json(split_identity)}
    final_receipt = _assemble_or_preserve_pre_state_bank(
        output_dir=output_dir,
        rollout_rows=rollouts,
        review_rows=reviews,
        receipt=receipt,
        source_checkpoint=reference_binding.source_checkpoint,
        prompt_identity_sha256=reference_binding.prompt_identity_sha256,
        source_artifacts=artifacts,
    )
    _write_json(output_dir / "assembly-receipt.json", final_receipt)
    print(json.dumps({"status": final_receipt["status"], "event_count": len(rollouts), "image_count": len({str(row["image"]["image_id"]) for row in rollouts}), "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
