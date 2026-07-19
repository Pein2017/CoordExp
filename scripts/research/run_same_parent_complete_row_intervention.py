#!/usr/bin/env python3
"""Run the frozen same-parent complete-row intervention.

This is a deliberately narrow Stage 3 experiment.  The sampled and native
branch rows are copied from already validated artifacts; the model is called
only to replay the suffix after an exact complete-row prefix.  An exact native
row no-op parity check is a hard gate for every primary comparison.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import glob
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_local_branch_causal_value import (  # noqa: E402
    _annotate_owner_matches,
    _generate_row,
    _single_native_inputs,
    append_row_if_complete,
    build_positive_entity_ledger,
    extend_covered_set_if_unambiguous,
    git_execution_identity,
    hash_prefix_token_ids,
    sha256_file,
)
from scripts.research.run_sampled_history_prefix_sufficiency_ladder import (  # noqa: E402
    MAX_ROOT_ROWS,
    SCHEMA_VERSION as STAGE_TWO_SCHEMA_VERSION,
    StageTwoValidationError,
    _build_request,
    _image_sort,
    _load_json,
    _row_is_clean_complete,
    _select_example,
    _trajectory_for_target,
    load_stage_two_admission,
    summarize_continuation,
)
from scripts.research.run_sampled_history_target_reachability import (  # noqa: E402
    compare_execution_identity,
)
from scripts.research.run_local_branch_causal_value import validate_frozen_file_identity  # noqa: E402
from scripts.research.validate_sampled_history_prefix_sufficiency_ladder_union import (  # noqa: E402
    validate_union as validate_stage_two_union,
)


SCHEMA_VERSION = "sampled_history_target_reachability.stage_three.v1"
PHASE = "stage_three_same_parent_complete_row_intervention"
TOTAL_TOKEN_BUDGET = 512
REQUIRED_CANDIDATE_COUNT = 3
# These are the frozen Qwen3-VL coordinate-token identifiers for bins 0..999.
COORDINATE_TOKEN_ID_MIN = 151670
COORDINATE_TOKEN_ID_MAX = 152669


class StageThreeValidationError(ValueError):
    """Raised when frozen Stage 3 sources cannot support a primary claim."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _ids(value: Any, *, label: str, allow_empty: bool = True) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageThreeValidationError(f"{label} must be a token-id sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise StageThreeValidationError(f"{label} contains an invalid token id")
        result.append(int(item))
    if not allow_empty and not result:
        raise StageThreeValidationError(f"{label} must not be empty")
    return result


def _row_owner_ids(row: Mapping[str, Any]) -> set[str]:
    return {
        str(value)
        for value in row.get("strict_matched_owner_ids", [])
        if value is not None
    }


def _single_clean_owner(row: Mapping[str, Any], *, label: str) -> str:
    if not _row_is_clean_complete(row):
        raise StageThreeValidationError(f"{label} is not a clean complete row")
    owners = _row_owner_ids(row)
    if len(owners) != 1:
        raise StageThreeValidationError(f"{label} must have exactly one strict owner")
    return next(iter(owners))


def _coords(row: Mapping[str, Any], *, label: str) -> list[int]:
    predictions = row.get("parsed_predictions")
    if not isinstance(predictions, list) or len(predictions) != 1 or not isinstance(predictions[0], Mapping):
        raise StageThreeValidationError(f"{label} lacks one parsed prediction")
    bins = predictions[0].get("coord_bins")
    values = _ids(bins, label=f"{label} coordinate bins", allow_empty=False)
    if len(values) != 4 or any(value > 999 for value in values):
        raise StageThreeValidationError(f"{label} coordinates are not four bins in [0,999]")
    return values


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("boxes must contain four xyxy values")
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def compare_exact_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the raw token and status fields required by the no-op gate."""

    left_ids = _ids(left.get("raw_generated_token_ids", []), label="left row ids")
    right_ids = _ids(right.get("raw_generated_token_ids", []), label="right row ids")
    left_stop = left.get("row_stop") if isinstance(left.get("row_stop"), Mapping) else {}
    right_stop = right.get("row_stop") if isinstance(right.get("row_stop"), Mapping) else {}
    left_parse = left.get("parse_evidence") if isinstance(left.get("parse_evidence"), Mapping) else {}
    right_parse = right.get("parse_evidence") if isinstance(right.get("parse_evidence"), Mapping) else {}
    checks = {
        "raw_token_ids_equal": left_ids == right_ids,
        "status_equal": left.get("status") == right.get("status"),
        "stop_reason_equal": left_stop.get("stop_reason") == right_stop.get("stop_reason"),
        "parse_status_equal": left_parse.get("parse_status") == right_parse.get("parse_status"),
    }
    return {"passed": all(checks.values()), "checks": checks}


def compare_suffix_rows(
    direct_rows: Sequence[Mapping[str, Any]], replay_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare every direct suffix row with its exact replay counterpart."""

    checks: list[dict[str, Any]] = []
    for index in range(max(len(direct_rows), len(replay_rows))):
        direct = direct_rows[index] if index < len(direct_rows) else None
        replay = replay_rows[index] if index < len(replay_rows) else None
        if direct is None or replay is None:
            checks.append({"suffix_index": index, "passed": False, "reason": "row_count_mismatch"})
            continue
        comparison = compare_exact_rows(direct, replay)
        checks.append({"suffix_index": index, **comparison})
    return {
        "passed": bool(checks) and all(item.get("passed") is True for item in checks),
        "checks": checks,
        "direct_row_count": len(direct_rows),
        "replay_row_count": len(replay_rows),
    }


def compare_branch_structure(
    native_token_ids: Sequence[int], sampled_token_ids: Sequence[int]
) -> dict[str, Any]:
    """Require equal row syntax and descriptions with differences only in coordinates."""

    native = _ids(native_token_ids, label="native branch structure", allow_empty=False)
    sampled = _ids(sampled_token_ids, label="sampled branch structure", allow_empty=False)
    native_positions = [
        index
        for index, token_id in enumerate(native)
        if COORDINATE_TOKEN_ID_MIN <= token_id <= COORDINATE_TOKEN_ID_MAX
    ]
    sampled_positions = [
        index
        for index, token_id in enumerate(sampled)
        if COORDINATE_TOKEN_ID_MIN <= token_id <= COORDINATE_TOKEN_ID_MAX
    ]
    checks = {
        "equal_token_count": len(native) == len(sampled),
        "equal_coordinate_positions": native_positions == sampled_positions,
        "exactly_four_native_coordinates": len(native_positions) == 4,
        "exactly_four_sampled_coordinates": len(sampled_positions) == 4,
        "all_non_coordinate_tokens_equal": (
            len(native) == len(sampled)
            and all(
                native[index] == sampled[index]
                for index in range(len(native))
                if index not in native_positions and index < len(sampled)
            )
        ),
        "at_least_one_coordinate_differs": (
            native_positions == sampled_positions
            and any(native[index] != sampled[index] for index in native_positions)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "coordinate_positions": native_positions,
        "native_token_ids": native,
        "sampled_token_ids": sampled,
    }


def _row_metadata_parity(
    direct_row: Mapping[str, Any],
    replay_row: Mapping[str, Any],
    *,
    generated_count_offset: int,
) -> dict[str, Any]:
    """Compare one suffix row, including adjusted cumulative token count."""

    direct_stop = direct_row.get("row_stop") if isinstance(direct_row.get("row_stop"), Mapping) else {}
    replay_stop = replay_row.get("row_stop") if isinstance(replay_row.get("row_stop"), Mapping) else {}
    direct_parse = direct_row.get("parse_evidence") if isinstance(direct_row.get("parse_evidence"), Mapping) else {}
    replay_parse = replay_row.get("parse_evidence") if isinstance(replay_row.get("parse_evidence"), Mapping) else {}
    direct_count = direct_row.get("generated_token_count_after_row")
    replay_count = replay_row.get("generated_token_count_after_row")
    checks = {
        "raw_token_ids_equal": _ids(direct_row.get("raw_generated_token_ids", []), label="direct parity row ids")
        == _ids(replay_row.get("raw_generated_token_ids", []), label="replay parity row ids"),
        "status_equal": direct_row.get("status") == replay_row.get("status"),
        "stop_reason_equal": direct_stop.get("stop_reason") == replay_stop.get("stop_reason"),
        "parse_status_equal": direct_parse.get("parse_status") == replay_parse.get("parse_status"),
        "row_index_equal": direct_row.get("row_index") == replay_row.get("row_index"),
        "accepted_complete_row_equal": direct_row.get("accepted_complete_row") == replay_row.get("accepted_complete_row"),
        "generated_count_equal_after_offset": (
            isinstance(direct_count, int)
            and isinstance(replay_count, int)
            and int(direct_count) - int(generated_count_offset) == int(replay_count)
        ),
        "budget_remaining_equal": direct_row.get("total_budget_remaining_before_row")
        == replay_row.get("total_budget_remaining_before_row"),
    }
    return {"passed": all(checks.values()), "checks": checks}


def compare_continuation_parity(
    frozen_continuation: Mapping[str, Any],
    replay_continuation: Mapping[str, Any],
    *,
    frozen_includes_branch: bool,
    branch_token_count: int,
) -> dict[str, Any]:
    """Compare full continuation metadata with consistent branch-count adjustment."""

    frozen_rows = frozen_continuation.get("rows")
    replay_rows = replay_continuation.get("rows")
    if not isinstance(frozen_rows, list) or not isinstance(replay_rows, list):
        return {"passed": False, "reason": "continuation_rows_missing", "row_checks": []}
    expected_rows = frozen_rows[1:] if frozen_includes_branch else frozen_rows
    offset = int(branch_token_count) if frozen_includes_branch else 0
    row_checks: list[dict[str, Any]] = []
    for index in range(max(len(expected_rows), len(replay_rows))):
        if index >= len(expected_rows) or index >= len(replay_rows):
            row_checks.append({"index": index, "passed": False, "reason": "row_count_mismatch"})
            continue
        row_checks.append({"index": index, **_row_metadata_parity(expected_rows[index], replay_rows[index], generated_count_offset=offset)})
    frozen_generated = frozen_continuation.get("generated_token_count")
    replay_generated = replay_continuation.get("generated_token_count")
    frozen_budget = frozen_continuation.get("total_token_budget")
    replay_budget = replay_continuation.get("total_token_budget")
    expected_generated = int(frozen_generated) - offset if isinstance(frozen_generated, int) else None
    expected_budget = int(frozen_budget) - offset if frozen_includes_branch and isinstance(frozen_budget, int) else frozen_budget
    metadata_checks = {
        "continuation_row_count": len(expected_rows) == len(replay_rows),
        "final_prefix_token_ids_sha256": frozen_continuation.get("final_prefix_token_ids_sha256") == replay_continuation.get("final_prefix_token_ids_sha256"),
        "horizon_rows_requested": (
            isinstance(frozen_continuation.get("horizon_rows_requested"), int)
            and int(frozen_continuation["horizon_rows_requested"]) - (1 if frozen_includes_branch else 0)
            == replay_continuation.get("horizon_rows_requested")
        ),
        "horizon_rows_complete": frozen_continuation.get("horizon_rows_complete") == replay_continuation.get("horizon_rows_complete"),
        "generated_token_count": expected_generated == replay_generated,
        "budget_exhausted": (
            isinstance(expected_generated, int)
            and isinstance(expected_budget, int)
            and bool(expected_generated >= expected_budget) == bool(replay_continuation.get("budget_exhausted"))
        ),
        "total_budget": expected_budget == replay_budget,
    }
    return {
        "passed": bool(row_checks) and all(item.get("passed") is True for item in row_checks) and all(metadata_checks.values()),
        "row_checks": row_checks,
        "metadata_checks": metadata_checks,
        "frozen_includes_branch": bool(frozen_includes_branch),
        "generated_count_offset": offset,
        "expected_generated_token_count": expected_generated,
        "expected_total_token_budget": expected_budget,
    }


MODEL_IDENTITY_FIELDS = (
    "backend",
    "backend_mode",
    "backend_version",
    "generation_config_fingerprint",
    "model_identity",
    "processor_identity",
    "tokenizer_identity",
    "response_family",
)


def stable_model_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return model identity while ignoring run-local device/batch settings."""

    return {field: receipt.get(field) for field in MODEL_IDENTITY_FIELDS}


def compare_stable_model_identity(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        field: stable_model_identity(observed).get(field)
        == stable_model_identity(expected).get(field)
        for field in MODEL_IDENTITY_FIELDS
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": stable_model_identity(observed),
        "expected": stable_model_identity(expected),
    }


def _branch_diff(parent: Sequence[int], child: Sequence[int], *, label: str) -> list[int]:
    parent_ids = _ids(parent, label=f"{label} parent prefix")
    child_ids = _ids(child, label=f"{label} child prefix")
    if child_ids[: len(parent_ids)] != parent_ids:
        raise StageThreeValidationError(f"{label} child prefix does not extend parent exactly")
    return child_ids[len(parent_ids) :]


def _load_discovery_sampled_row(stage_two_image: Mapping[str, Any], *, row_index: int, seed: int, expected_ids: Sequence[int]) -> dict[str, Any]:
    stage_one_source = stage_two_image.get("stage_one_source")
    if not isinstance(stage_one_source, Mapping) or not stage_one_source.get("path"):
        raise StageThreeValidationError("Stage 2 image lacks Stage 1 source path")
    stage_one_path = Path(str(stage_one_source["path"])).expanduser().resolve(strict=True)
    stage_one = _load_json(stage_one_path)
    images = stage_one.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise StageThreeValidationError("Stage 1 source lacks one image")
    source = images[0].get("source_discovery_shard")
    if not isinstance(source, Mapping) or not source.get("path"):
        raise StageThreeValidationError("Stage 1 source lacks discovery shard path")
    discovery_path = Path(str(source["path"])).expanduser().resolve(strict=True)
    declared_sha = str(source.get("sha256", ""))
    if not declared_sha or sha256_file(discovery_path) != declared_sha:
        raise StageThreeValidationError("discovery shard SHA-256 disagrees with Stage 1 source")
    discovery = _load_json(discovery_path)
    source_images = discovery.get("images")
    if not isinstance(source_images, list) or len(source_images) != 1 or not isinstance(source_images[0], Mapping):
        raise StageThreeValidationError("discovery shard lacks one image")
    trajectory = _trajectory_for_target(source_images[0], stage_two_image["provisional_target"])
    rows = trajectory.get("rows")
    if not isinstance(rows, list):
        raise StageThreeValidationError("discovery trajectory lacks rows")
    matches = [row for row in rows if isinstance(row, Mapping) and int(row.get("row_index", -1)) == int(row_index)]
    if len(matches) != 1:
        raise StageThreeValidationError("discovery sampled branch row is not unique")
    row = dict(matches[0])
    ids = _ids(row.get("raw_generated_token_ids", []), label="sampled branch ids", allow_empty=False)
    if ids != list(map(int, expected_ids)):
        raise StageThreeValidationError("discovery sampled row differs from exact P_(k+1)-P_k")
    return row


def load_stage_three_sources(admission_path: Path) -> dict[str, Any]:
    """Rehash the Stage 3 admission, Stage 2 union, admission, and shards."""

    resolved_admission = admission_path.expanduser().resolve(strict=True)
    admission = _load_json(resolved_admission)
    if admission.get("schema_version") != 1 or admission.get("unit_id") != "2026-07-19-sampled-history-target-reachability-and-complete-row-value":
        raise StageThreeValidationError("Stage 3 admission schema or unit mismatch")
    union_path = Path(str(admission.get("source_stage_two_union", ""))).expanduser().resolve(strict=True)
    if sha256_file(union_path) != str(admission.get("source_stage_two_union_sha256", "")):
        raise StageThreeValidationError("Stage 2 union SHA-256 disagrees with Stage 3 admission")
    stage_two_admission_path = Path(str(admission.get("source_stage_two_admission", ""))).expanduser().resolve(strict=True)
    if sha256_file(stage_two_admission_path) != str(admission.get("source_stage_two_admission_sha256", "")):
        raise StageThreeValidationError("Stage 2 admission SHA-256 disagrees with Stage 3 admission")
    stage_two_admission = load_stage_two_admission(stage_two_admission_path)
    if stage_two_admission["path"] != str(stage_two_admission_path):
        raise StageThreeValidationError("resolved Stage 2 admission path mismatch")
    union = _load_json(union_path)
    if union.get("schema_version") != "sampled_history_target_reachability.stage_two.union.v1" or union.get("passed") is not True:
        raise StageThreeValidationError("Stage 2 union is not a passed Stage 2 union")
    union_images = union.get("images")
    if not isinstance(union_images, list) or not union_images:
        raise StageThreeValidationError("Stage 2 union lacks image records")
    shard_paths: list[Path] = []
    for record in union_images:
        if not isinstance(record, Mapping) or not record.get("path"):
            raise StageThreeValidationError("Stage 2 union contains malformed shard record")
        shard_path = Path(str(record["path"])).expanduser().resolve(strict=True)
        if sha256_file(shard_path) != str(record.get("sha256", "")):
            raise StageThreeValidationError(f"Stage 2 shard SHA-256 mismatch: {shard_path}")
        shard_paths.append(shard_path)
    validated_union = validate_stage_two_union(shard_paths=shard_paths, admission_path=stage_two_admission_path)
    if _canonical_hash(validated_union["images"]) != _canonical_hash(union["images"]):
        raise StageThreeValidationError("Stage 2 union image identity differs from revalidated shards")
    candidates = admission.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != REQUIRED_CANDIDATE_COUNT:
        raise StageThreeValidationError("Stage 3 admission must contain exactly three candidates")
    candidate_map: dict[str, dict[str, Any]] = {}
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise StageThreeValidationError("Stage 3 candidate is malformed")
        entry = dict(raw)
        image_id = str(entry.get("image_id", ""))
        if not image_id or image_id in candidate_map:
            raise StageThreeValidationError("Stage 3 candidate image identity is invalid")
        if entry.get("admission") != "primary":
            raise StageThreeValidationError(f"Stage 3 candidate is not primary: {image_id}")
        candidate_map[image_id] = entry
    shard_records = {str(item["image_id"]): item for item in union_images}
    if not set(candidate_map).issubset(shard_records):
        raise StageThreeValidationError("Stage 3 candidate is absent from Stage 2 union")
    stage_two_images: dict[str, dict[str, Any]] = {}
    stage_two_artifacts: dict[str, dict[str, Any]] = {}
    for image_id, record in shard_records.items():
        artifact = _load_json(Path(str(record["path"])))
        images = artifact.get("images")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
            raise StageThreeValidationError(f"Stage 2 shard lacks one image: {image_id}")
        stage_two_images[image_id] = dict(images[0])
        stage_two_artifacts[image_id] = artifact
    return {
        "admission_path": str(resolved_admission),
        "admission_sha256": sha256_file(resolved_admission),
        "admission": admission,
        "stage_two_union_path": str(union_path),
        "stage_two_union_sha256": sha256_file(union_path),
        "stage_two_union": union,
        "stage_two_admission_path": str(stage_two_admission_path),
        "stage_two_admission_sha256": sha256_file(stage_two_admission_path),
        "stage_two_admission": stage_two_admission,
        "stage_two_shards": {image_id: {"path": str(shard_records[image_id]["path"]), "sha256": str(shard_records[image_id]["sha256"]), "image": stage_two_images[image_id], "artifact": stage_two_artifacts[image_id]} for image_id in stage_two_images},
        "candidates": candidate_map,
    }


def prepare_candidate_source(sources: Mapping[str, Any], image_id: str) -> dict[str, Any]:
    """Extract and verify the exact parent, native branch, and sampled branch."""

    if image_id not in sources["candidates"]:
        raise StageThreeValidationError(f"image is not a Stage 3 candidate: {image_id}")
    candidate = dict(sources["candidates"][image_id])
    stage_image = sources["stage_two_shards"][image_id]["image"]
    k = int(candidate["parent_prefix_row_count"])
    ladder = stage_image.get("sampled_prefix_ladder")
    prefixes = ladder.get("prefixes") if isinstance(ladder, Mapping) else None
    if not isinstance(prefixes, list) or k < 0 or k + 1 >= len(prefixes):
        raise StageThreeValidationError(f"candidate {image_id} lacks adjacent sampled prefixes")
    parent = prefixes[k]
    child = prefixes[k + 1]
    if int(parent.get("k", -1)) != k or int(child.get("k", -1)) != k + 1:
        raise StageThreeValidationError(f"candidate {image_id} does not bind exact P_k and P_(k+1) entries")
    if list(candidate.get("stage_two_transition", [])) != ["clean_miss", "hit"]:
        raise StageThreeValidationError(f"candidate {image_id} Stage 2 transition is not [clean_miss, hit]")
    if child.get("reachability_state") not in {None, "hit"}:
        # The per-prefix reachability state is stored in the evaluation entry;
        # this protects older artifacts that do not duplicate it in the ladder.
        raise StageThreeValidationError(f"candidate {image_id} child prefix carries an invalid reachability state")
    parent_ids = _ids(parent.get("prefix_token_ids"), label="Stage 3 parent prefix")
    child_ids = _ids(child.get("prefix_token_ids"), label="Stage 3 child prefix")
    if hash_prefix_token_ids(parent_ids) != str(candidate["parent_prefix_sha256"]):
        raise StageThreeValidationError(f"candidate {image_id} parent prefix hash mismatch")
    sampled_ids = _branch_diff(parent_ids, child_ids, label="sampled branch")
    if hash_prefix_token_ids(sampled_ids) != str(candidate["sampled_branch_row_sha256"]):
        raise StageThreeValidationError(f"candidate {image_id} sampled branch hash mismatch")
    direct_entry = next((item for item in stage_image.get("prefix_evaluations", []) if isinstance(item, Mapping) and int(item.get("prefix", {}).get("k", -1)) == k), None)
    if not isinstance(direct_entry, Mapping):
        raise StageThreeValidationError(f"candidate {image_id} lacks direct P_k continuation")
    if direct_entry.get("reachability_state") != "clean_miss":
        raise StageThreeValidationError(f"candidate {image_id} direct P_k is not clean_miss")
    child_entry = next((item for item in stage_image.get("prefix_evaluations", []) if isinstance(item, Mapping) and int(item.get("prefix", {}).get("k", -1)) == k + 1), None)
    if not isinstance(child_entry, Mapping):
        raise StageThreeValidationError(f"candidate {image_id} lacks P_(k+1) continuation")
    if child_entry.get("reachability_state") != "hit":
        raise StageThreeValidationError(f"candidate {image_id} P_(k+1) is not a hit")
    target_first_hit = child_entry.get("target_first_hit_row_index")
    if not isinstance(target_first_hit, int) or target_first_hit <= k:
        raise StageThreeValidationError(f"candidate {image_id} target is not strictly after the branch row")
    target_owner = str(candidate["target_owner_id"])
    parent_owners = {str(value) for value in parent.get("prefix_owner_ids", [])}
    if target_owner in parent_owners:
        raise StageThreeValidationError(f"candidate {image_id} target is already present in parent owners")
    if str(candidate["branch_owner_id"]) == target_owner:
        raise StageThreeValidationError(f"candidate {image_id} branch owner equals target owner")
    direct_rows = direct_entry.get("continuation", {}).get("rows")
    if not isinstance(direct_rows, list) or not direct_rows:
        raise StageThreeValidationError(f"candidate {image_id} lacks direct native rows")
    native_branch = dict(direct_rows[0])
    native_owner = _single_clean_owner(native_branch, label="native branch")
    if native_owner != str(candidate["branch_owner_id"]):
        raise StageThreeValidationError(f"candidate {image_id} native branch owner mismatch")
    native_ids = _ids(native_branch.get("raw_generated_token_ids"), label="native branch ids", allow_empty=False)
    if hash_prefix_token_ids(native_ids) != str(candidate["native_branch_row_sha256"]):
        raise StageThreeValidationError(f"candidate {image_id} native branch hash mismatch")
    if len(native_ids) != int(candidate["branch_row_token_count"]):
        raise StageThreeValidationError(f"candidate {image_id} native branch token length mismatch")
    sampled_row = _load_discovery_sampled_row(stage_image, row_index=k, seed=int(stage_image["provisional_target"]["sampled_seed"]), expected_ids=sampled_ids)
    sampled_owner = _single_clean_owner(sampled_row, label="sampled branch")
    if sampled_owner != str(candidate["branch_owner_id"]):
        raise StageThreeValidationError(f"candidate {image_id} sampled branch owner mismatch")
    if len(sampled_ids) != len(native_ids) or len(sampled_ids) != int(candidate["branch_row_token_count"]):
        raise StageThreeValidationError(f"candidate {image_id} branch lengths are not equal")
    structural_gate = compare_branch_structure(native_ids, sampled_ids)
    if not structural_gate["passed"]:
        raise StageThreeValidationError(f"candidate {image_id} branch rows are not coordinate-only variants")
    if _coords(native_branch, label="native branch") != list(map(int, candidate["native_coordinates"] or [])):
        raise StageThreeValidationError(f"candidate {image_id} native coordinate bins disagree")
    if _coords(sampled_row, label="sampled branch") != list(map(int, candidate["sampled_coordinates"] or [])):
        raise StageThreeValidationError(f"candidate {image_id} sampled coordinate bins disagree")
    return {
        "candidate": candidate,
        "stage_two_image": stage_image,
        "parent_prefix": parent,
        "parent_token_ids": parent_ids,
        "prefix_owner_ids": list(parent.get("prefix_owner_ids", [])),
        "native_branch": native_branch,
        "sampled_branch": sampled_row,
        "native_branch_token_ids": native_ids,
        "sampled_branch_token_ids": sampled_ids,
        "direct_continuation_rows": [dict(row) for row in direct_rows],
        "frozen_native_continuation": dict(direct_entry["continuation"]),
        "frozen_sampled_continuation": dict(child_entry["continuation"]),
        "frozen_sampled_prefix_evaluation": dict(child_entry),
        "structural_gate": structural_gate,
        "branch_coordinate_delta_sampled_minus_native": [
            int(a) - int(b) for a, b in zip(candidate["sampled_coordinates"], candidate["native_coordinates"])
        ],
        "branch_box_iou": box_iou(candidate["sampled_coordinates"], candidate["native_coordinates"]),
    }


def _generate_prefix_continuation(*, session: Any, native_inputs: Mapping[str, Any], prefix_token_ids: Sequence[int], prefix_owner_ids: Sequence[str], tokenizer: Any, image_width: int, image_height: int, entity_ledger: Sequence[Mapping[str, Any]], start_row_index: int, horizon_rows: int, malformed_limit: int, temperature: float, total_token_budget: int) -> dict[str, Any]:
    """Small copy of the validated Stage 2 continuation seam."""

    current = _ids(prefix_token_ids, label="continuation prefix")
    rows: list[dict[str, Any]] = []
    generated_total = 0
    covered = {str(value) for value in prefix_owner_ids}
    while len(rows) < int(horizon_rows) and generated_total < int(total_token_budget):
        remaining = int(total_token_budget) - generated_total
        row_index = int(start_row_index) + len(rows)
        row = _generate_row(
            session=session,
            native_inputs=native_inputs,
            prefix_token_ids=current,
            tokenizer=tokenizer,
            image_width=int(image_width),
            image_height=int(image_height),
            mode="greedy",
            seed=None,
            temperature=float(temperature),
            top_p=0.95,
            repetition_penalty=1.0,
            max_new_tokens=remaining,
            malformed_limit=int(malformed_limit),
            row_index=row_index,
        )
        row["row_index"] = row_index
        row["input_prefix_token_ids"] = list(current)
        row["input_prefix_token_ids_sha256"] = hash_prefix_token_ids(current)
        generated = _ids(row.get("raw_generated_token_ids", []), label="generated continuation row ids")
        generated_total += len(generated)
        row["generated_token_count_after_row"] = generated_total
        row["total_budget_remaining_before_row"] = remaining
        _annotate_owner_matches(row, entity_ledger=entity_ledger, image_width=int(image_width), image_height=int(image_height), covered_entity_ids=sorted(covered))
        row["covered_owner_ids_before_row"] = sorted(covered)
        current_after, append_receipt = append_row_if_complete(current, row)
        row["append_receipt"] = append_receipt
        row["accepted_complete_row"] = bool(append_receipt.get("appended"))
        row["appended_to_prefix"] = bool(append_receipt.get("appended"))
        if row["appended_to_prefix"]:
            current = current_after
        covered_after, coverage_receipt = extend_covered_set_if_unambiguous(covered, row)
        row["coverage_receipt"] = coverage_receipt
        if coverage_receipt.get("coverage_updated"):
            covered = set(covered_after)
        row["covered_owner_ids_after_row"] = sorted(covered)
        row["cumulative_prefix_token_ids"] = list(current)
        row["cumulative_prefix_token_ids_sha256"] = hash_prefix_token_ids(current)
        rows.append(row)
        reason = str((row.get("row_stop") or {}).get("stop_reason", "unknown"))
        if reason in {"terminal", "malformed_limit", "contaminated_complete_row", "failed"} or row.get("status") == "failed" or not row["appended_to_prefix"]:
            break
    return {
        "mode": "greedy",
        "rows": rows,
        "initial_prefix_token_ids": list(map(int, prefix_token_ids)),
        "initial_prefix_token_ids_sha256": hash_prefix_token_ids(prefix_token_ids),
        "final_prefix_token_ids": list(current),
        "final_prefix_token_ids_sha256": hash_prefix_token_ids(current),
        "continuation_row_count": len(rows),
        "horizon_rows_requested": int(horizon_rows),
        "horizon_rows_complete": len(rows) == int(horizon_rows) and all(_row_is_clean_complete(row) for row in rows),
        "generated_token_count": generated_total,
        "total_token_budget": int(total_token_budget),
        "budget_exhausted": generated_total >= int(total_token_budget),
    }


def _owner_set(prefix_owner_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> set[str]:
    result = {str(value) for value in prefix_owner_ids}
    for row in rows:
        result.update(_row_owner_ids(row))
    return result


def summarize_branch_comparison(
    source: Mapping[str, Any],
    *,
    native_suffix: Mapping[str, Any],
    sampled_suffix: Mapping[str, Any],
    native_no_op_parity: Mapping[str, Any],
    sampled_source_replay_parity: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = source["candidate"]
    prefix_owners = source["prefix_owner_ids"]
    target = str(candidate["target_owner_id"])
    branch_owner = str(candidate["branch_owner_id"])
    native_rows = [source["native_branch"], *native_suffix.get("rows", [])]
    sampled_rows = [source["sampled_branch"], *sampled_suffix.get("rows", [])]
    native_summary = summarize_continuation(native_rows, prefix_owner_ids=prefix_owners, target_owner_id=target, target_start_row_index=int(candidate["parent_prefix_row_count"]) + 1, generated_token_count=sum(len(_ids(row.get("raw_generated_token_ids", []), label="native summary ids")) for row in native_rows), total_token_budget=TOTAL_TOKEN_BUDGET, horizon_rows_complete=bool(native_suffix.get("horizon_rows_complete")))
    sampled_summary = summarize_continuation(sampled_rows, prefix_owner_ids=prefix_owners, target_owner_id=target, target_start_row_index=int(candidate["parent_prefix_row_count"]) + 1, generated_token_count=sum(len(_ids(row.get("raw_generated_token_ids", []), label="sampled summary ids")) for row in sampled_rows), total_token_budget=TOTAL_TOKEN_BUDGET, horizon_rows_complete=bool(sampled_suffix.get("horizon_rows_complete")))
    native_owners = _owner_set(prefix_owners, native_rows)
    sampled_owners = _owner_set(prefix_owners, sampled_rows)
    other_native = native_owners - {branch_owner, target}
    other_sampled = sampled_owners - {branch_owner, target}
    structural_gate = dict(source.get("structural_gate", {}))
    target_access_interpretable = bool(
        structural_gate.get("passed")
        and source["candidate"].get("stage_two_transition") == ["clean_miss", "hit"]
        and target not in set(prefix_owners)
        and branch_owner != target
    )
    target_access_reproduced = bool(
        target in sampled_owners and target not in native_owners
    )
    safe_unique_owner_eligible = bool(
        target_access_interpretable
        and not native_summary["unresolved_row_indices"]
        and not sampled_summary["unresolved_row_indices"]
        and not native_summary["crop_review_warnings"]
        and not sampled_summary["crop_review_warnings"]
    )
    primary_allowed = bool(
        structural_gate.get("passed")
        and native_no_op_parity.get("passed")
        and sampled_source_replay_parity.get("passed")
        and target_access_interpretable
    )
    return {
        "structural_gate": structural_gate,
        "native_no_op_parity": dict(native_no_op_parity),
        "sampled_source_replay_parity": dict(sampled_source_replay_parity),
        "target_access_interpretable": target_access_interpretable,
        "target_access_reproduced": target_access_reproduced,
        "safe_unique_owner_eligible": safe_unique_owner_eligible,
        "primary_causal_claim_allowed": primary_allowed,
        "primary_causal_refusal_reason": None if primary_allowed else "stage3_primary_gate_failed",
        "native_summary": native_summary,
        "sampled_summary": sampled_summary,
        "native_unique_owner_ids": sorted(native_owners),
        "sampled_unique_owner_ids": sorted(sampled_owners),
        "unique_owner_difference": len(sampled_owners) - len(native_owners),
        "gained_owner_ids": sorted(sampled_owners - native_owners),
        "lost_owner_ids": sorted(native_owners - sampled_owners),
        "other_owner_gained_ids": sorted(other_sampled - other_native),
        "other_owner_lost_ids": sorted(other_native - other_sampled),
        "target_retrieved_native": target in native_owners,
        "target_retrieved_sampled": target in sampled_owners,
        "duplicate_owner_difference": sorted(set(sampled_summary["duplicate_owner_ids"]) - set(native_summary["duplicate_owner_ids"])),
        "unresolved_row_difference": sorted(set(sampled_summary["unresolved_row_indices"]) - set(native_summary["unresolved_row_indices"])),
        "warning_count_difference": len(sampled_summary["crop_review_warnings"]) - len(native_summary["crop_review_warnings"]),
    }


def run_candidate_replays(*, source: Mapping[str, Any], session: Any, native_inputs: Mapping[str, Any], tokenizer: Any, image_width: int, image_height: int, entity_ledger: Sequence[Mapping[str, Any]], malformed_limit: int, temperature: float = 0.4) -> dict[str, Any]:
    """Replay the native no-op and sampled-row suffix for one candidate."""

    candidate = source["candidate"]
    parent_ids = source["parent_token_ids"]
    prefix_owners = source["prefix_owner_ids"]
    branch_owners = sorted(set(prefix_owners) | {str(candidate["branch_owner_id"])})
    k = int(candidate["parent_prefix_row_count"])
    native_suffix = _generate_prefix_continuation(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=parent_ids + source["native_branch_token_ids"],
        prefix_owner_ids=branch_owners,
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        entity_ledger=entity_ledger,
        start_row_index=k + 1,
        horizon_rows=MAX_ROOT_ROWS - k - 1,
        malformed_limit=malformed_limit,
        temperature=temperature,
        total_token_budget=TOTAL_TOKEN_BUDGET - len(source["native_branch_token_ids"]),
    )
    sampled_suffix = _generate_prefix_continuation(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=parent_ids + source["sampled_branch_token_ids"],
        prefix_owner_ids=branch_owners,
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        entity_ledger=entity_ledger,
        start_row_index=k + 1,
        horizon_rows=MAX_ROOT_ROWS - k - 1,
        malformed_limit=malformed_limit,
        temperature=temperature,
        # The frozen sampled source is the P_(k+1) continuation, whose budget
        # starts after the sampled branch has already been committed.
        total_token_budget=TOTAL_TOKEN_BUDGET,
    )
    native_no_op_parity = compare_continuation_parity(
        source["frozen_native_continuation"],
        native_suffix,
        frozen_includes_branch=True,
        branch_token_count=len(source["native_branch_token_ids"]),
    )
    sampled_source_replay_parity = compare_continuation_parity(
        source["frozen_sampled_continuation"],
        sampled_suffix,
        frozen_includes_branch=False,
        branch_token_count=len(source["sampled_branch_token_ids"]),
    )
    return {
        "native_no_op_suffix": native_suffix,
        "sampled_row_intervention_suffix": sampled_suffix,
        "comparison": summarize_branch_comparison(
            source,
            native_suffix=native_suffix,
            sampled_suffix=sampled_suffix,
            native_no_op_parity=native_no_op_parity,
            sampled_source_replay_parity=sampled_source_replay_parity,
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-admission", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ids", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    try:
        sources = load_stage_three_sources(args.stage3_admission)
        selected = {value.strip() for value in str(args.image_ids).split(",") if value.strip()}
        candidate_ids = set(sources["candidates"])
        if selected - candidate_ids:
            raise StageThreeValidationError(f"--image-ids outside Stage 3 candidates: {sorted(selected - candidate_ids)}")
        image_ids = sorted(selected or candidate_ids, key=_image_sort)
        prepared = {image_id: prepare_candidate_source(sources, image_id) for image_id in image_ids}
    except (OSError, KeyError, TypeError, ValueError, StageTwoValidationError, StageThreeValidationError) as exc:
        raise SystemExit(f"Stage 3 source validation failed before runtime: {exc}") from exc
    try:
        import torch
        from src.config.fingerprint import sha256_json
        from src.config.inference import load_infer_config
        from src.data import load_raw_examples
        from src.inference.backend import open_backend_session
        from src.inference.runtime import assemble_frontend
    except Exception as exc:
        raise SystemExit(f"runtime import failed; no artifact was written: {type(exc).__name__}: {exc}") from exc
    resolved = load_infer_config(args.infer_config.expanduser().resolve(strict=True))
    config = resolved.config
    if str(config.model.dtype) != "fp32":
        raise SystemExit(f"Stage 3 requires model.dtype=fp32, observed {config.model.dtype!r}")
    first_image = prepared[image_ids[0]]["stage_two_image"]
    frozen_inputs = sources["stage_two_shards"][image_ids[0]].get("artifact", {}).get("frozen_inputs") or {}
    manifest_path = Path(str(frozen_inputs.get("manifest", ""))).expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path)
    source_jsonl = Path(str((manifest.get("inference_contract") or {}).get("source_jsonl", config.data.input_jsonl))).expanduser().resolve(strict=True)
    try:
        frozen_file_identity = validate_frozen_file_identity(
            manifest,
            infer_config_path=args.infer_config,
            source_jsonl_path=source_jsonl,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Stage 3 frozen file identity failed before runtime: {exc}") from exc
    frozen_config_fingerprints = {
        str((record.get("artifact", {}).get("config") or {}).get("resolved_config_fingerprint", ""))
        for record in sources["stage_two_shards"].values()
    }
    if frozen_config_fingerprints != {str(resolved.fingerprint)}:
        raise SystemExit(
            "Stage 3 resolved config fingerprint disagrees with Stage 2 shards: "
            f"current={resolved.fingerprint} frozen={sorted(frozen_config_fingerprints)}"
        )
    frozen_model_fingerprints = {
        _canonical_hash(stable_model_identity(record.get("artifact", {}).get("model_identity") or {}))
        for record in sources["stage_two_shards"].values()
    }
    if len(frozen_model_fingerprints) != 1:
        raise SystemExit("Stage 2 shards do not share one stable model identity")
    examples = list(load_raw_examples(source_jsonl))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    outputs: list[dict[str, Any]] = []
    with open_backend_session(frontend.launch) as session:
        model_receipt = session.receipt.to_artifact_dict()
        expected_model_receipt = sources["stage_two_shards"][image_ids[0]]["artifact"].get("model_identity") or {}
        model_identity_check = compare_stable_model_identity(model_receipt, expected_model_receipt)
        if not model_identity_check["passed"]:
            raise SystemExit(
                "Stage 3 model identity disagrees with Stage 2 frozen receipt: "
                f"{json.dumps(model_identity_check, sort_keys=True)}"
            )
        for image_id in image_ids:
            source = prepared[image_id]
            stage_image = source["stage_two_image"]
            target = stage_image["provisional_target"]
            example = _select_example(examples, image_id)
            request, plan, prompt_meta = _build_request(config, frontend, example)
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
            one_native = _single_native_inputs(native_inputs)
            execution_identity_check = compare_execution_identity(
                observed_prompt=prompt_meta,
                observed_runtime={
                    "executed_media_sha256": media_sha[0],
                    "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]),
                    "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
                },
                discovery_prompt=stage_image["prompt"],
                discovery_runtime=stage_image["runtime"],
            )
            if not execution_identity_check["passed"]:
                raise SystemExit(
                    f"Stage 3 execution identity disagrees with Stage 2 for {image_id}: "
                    f"{json.dumps(execution_identity_check, sort_keys=True)}"
                )
            ledger = [dict(row) for row in stage_image.get("entity_ledger", []) if isinstance(row, Mapping)] or build_positive_entity_ledger(example)
            parent_ids = source["parent_token_ids"]
            prefix_owners = list(source["prefix_owner_ids"])
            k = int(source["candidate"]["parent_prefix_row_count"])
            replay = run_candidate_replays(source=source, session=session, native_inputs=one_native, tokenizer=session._tokenizer, image_width=int(plan.decoded_width), image_height=int(plan.decoded_height), entity_ledger=ledger, malformed_limit=int((manifest.get("discovery_budget") or {}).get("malformed_row_limit", 2)), temperature=0.4)
            native_suffix = replay["native_no_op_suffix"]
            sampled_suffix = replay["sampled_row_intervention_suffix"]
            comparison = replay["comparison"]
            outputs.append({
                "image_id": image_id,
                "stage3_admission_entry": source["candidate"],
                "stage2_source": {"path": sources["stage_two_shards"][image_id]["path"], "sha256": sources["stage_two_shards"][image_id]["sha256"]},
                "parent": {"row_count": k, "token_ids": parent_ids, "token_ids_sha256": hash_prefix_token_ids(parent_ids), "owner_ids": prefix_owners},
                "native_branch": source["native_branch"],
                "sampled_branch": source["sampled_branch"],
                "branch_coordinate_delta_sampled_minus_native": source["branch_coordinate_delta_sampled_minus_native"],
                "branch_box_iou": source["branch_box_iou"],
                "direct_native_continuation_rows": source["direct_continuation_rows"],
                "frozen_native_continuation": source["frozen_native_continuation"],
                "frozen_sampled_continuation": source["frozen_sampled_continuation"],
                "frozen_sampled_prefix_evaluation": source["frozen_sampled_prefix_evaluation"],
                "structural_gate": source["structural_gate"],
                "native_no_op_suffix": native_suffix,
                "sampled_row_intervention_suffix": sampled_suffix,
                "comparison": comparison,
                "identity_check": execution_identity_check,
                "prompt": {"image_id": image_id, "prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]), "prompt": prompt_meta},
                "runtime": {"observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]), "executed_media_sha256": media_sha[0]},
                "target": target,
            })
    source_identity = git_execution_identity(Path(__file__).resolve().parents[2])
    source_identity.pop("runner_sha256", None)
    source_identity["stage_three_runner_sha256"] = sha256_file(Path(__file__).resolve())
    source_identity["stage_two_runner_sha256"] = sha256_file(Path(__file__).with_name("run_sampled_history_prefix_sufficiency_ladder.py"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "sampled_history_target_reachability",
        "phase": PHASE,
        "source_identity": source_identity,
        "frozen_inputs": {
            "stage_three_admission": str(args.stage3_admission.expanduser().resolve()),
            "stage_three_admission_sha256": sha256_file(args.stage3_admission.expanduser().resolve()),
            "stage_two_union": sources["stage_two_union_path"],
            "stage_two_union_sha256": sources["stage_two_union_sha256"],
            "stage_two_admission": sources["stage_two_admission_path"],
            "stage_two_admission_sha256": sources["stage_two_admission_sha256"],
            "stage_two_shard_sha256": {image_id: sources["stage_two_shards"][image_id]["sha256"] for image_id in sorted(sources["stage_two_shards"], key=_image_sort)},
        },
        "config": {"infer_config": str(args.infer_config.expanduser().resolve()), "resolved_config_fingerprint": resolved.fingerprint, "device": args.device, "physical_batch_size": 1, "model_dtype": "fp32", "repetition_penalty": 1.0, "absolute_horizon_complete_rows": MAX_ROOT_ROWS, "total_generated_token_budget": TOTAL_TOKEN_BUDGET},
        "model_identity": model_receipt,
        "model_identity_check": model_identity_check,
        "frozen_file_identity": frozen_file_identity,
        "images": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
