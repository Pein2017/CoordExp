#!/usr/bin/env python3
"""Run the experiment-local sampled-prefix sufficiency ladder.

For each frozen sampled target, this runner reconstructs the exact sampled
prefix after ``k`` complete rows (``k = 0..j``), then releases ordinary greedy
generation until the absolute eight-row horizon.  It is intentionally narrow:
the runner does not change the model, add a coverage mechanism, score logits,
or train anything.  It only asks whether an already observed sampled history
changes later greedy target reachability.

The pure reconstruction and result helpers are importable by tests without
loading PyTorch or a checkpoint.  Runtime execution reuses the tested native
HF path and exact token-prefix append used by the preceding research units.
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

from scripts.research.run_local_branch_causal_value import (
    _annotate_owner_matches,
    _generate_row,
    _single_native_inputs,
    append_row_if_complete,
    build_positive_entity_ledger,
    extend_covered_set_if_unambiguous,
    git_execution_identity,
    hash_prefix_token_ids,
    sha256_file,
    validate_frozen_file_identity,
)
from scripts.research.run_sampled_history_target_reachability import (
    MAX_ROOT_ROWS,
    StageOneValidationError,
    _find_root_trajectory,
    _image_sort,
    _load_json,
    compare_execution_identity,
    strict_owner_union,
    validate_frozen_inputs,
)
from scripts.research.run_same_covered_set_prefix_order_probe import (
    _build_request,
    _select_example,
)


SCHEMA_VERSION = "sampled_history_target_reachability.stage_two.v1"
PHASE = "stage_two_prefix_sufficiency_ladder"
TOTAL_TOKEN_BUDGET = 512
DEFAULT_ALLOWED_STAGE_ONE_LABELS = ("terminally_omitted", "delayed")


class StageTwoValidationError(ValueError):
    """Raised when a Stage 2 source or prefix is not scientifically usable."""


def _ids(value: Any, *, label: str, allow_empty: bool = True) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageTwoValidationError(f"{label} must be a token-id sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise StageTwoValidationError(f"{label} contains an invalid token id")
        result.append(int(item))
    if not allow_empty and not result:
        raise StageTwoValidationError(f"{label} must not be empty")
    return result


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _row_is_clean_complete(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("status") == "success"
        and row.get("accepted_complete_row") is True
        and isinstance(row.get("row_stop"), Mapping)
        and row["row_stop"].get("stop_reason") == "complete_row"
    )


def _trajectory_for_target(image: Mapping[str, Any], target: Mapping[str, Any]) -> Mapping[str, Any]:
    seed = target.get("sampled_seed")
    trajectories = image.get("trajectories")
    if not isinstance(trajectories, list):
        raise StageTwoValidationError("discovery image lacks trajectories")
    matches = [
        item
        for item in trajectories
        if isinstance(item, Mapping) and item.get("mode") == "sample" and item.get("seed") == seed
    ]
    if len(matches) != 1:
        raise StageTwoValidationError(
            f"target {target.get('image_id')}/{target.get('target_owner_id')} requires one sampled trajectory; found {len(matches)}"
        )
    return matches[0]


def reconstruct_sampled_prefix_ladder(
    image: Mapping[str, Any], target: Mapping[str, Any], *, horizon_rows: int = MAX_ROOT_ROWS
) -> dict[str, Any]:
    """Reconstruct exact ``P_k`` values from one frozen sampled trajectory.

    ``P_k`` contains the raw token ids of the first ``k`` complete sampled
    rows.  No decoded text is used.  The target's declared parent hash must
    equal ``P_j``; otherwise the case is refused before any model call.
    """

    if not 1 <= int(horizon_rows) <= 8:
        raise StageTwoValidationError("horizon_rows must be in [1,8]")
    row_target = int(target.get("first_sampled_row_index_zero_based", -1))
    if row_target <= 0:
        raise StageTwoValidationError("Stage 2 target must occur after sampled row zero")
    if row_target >= int(horizon_rows):
        raise StageTwoValidationError("sampled target row is outside the absolute horizon")
    owner = str(target.get("target_owner_id"))
    trajectory = _trajectory_for_target(image, target)
    rows = trajectory.get("rows")
    if not isinstance(rows, list) or len(rows) <= row_target:
        raise StageTwoValidationError("sampled trajectory ends before the frozen target row")
    ordered = sorted(
        (row for row in rows if isinstance(row, Mapping)),
        key=lambda row: int(row.get("row_index", -1)),
    )
    if [int(row.get("row_index", -1)) for row in ordered[: row_target + 1]] != list(range(row_target + 1)):
        raise StageTwoValidationError("sampled trajectory row indexes are not contiguous through target")
    target_row = ordered[row_target]
    target_owners = {str(value) for value in target_row.get("strict_matched_owner_ids", [])}
    if owner not in target_owners or not _row_is_clean_complete(target_row):
        raise StageTwoValidationError("frozen target row is not a clean complete row owned by the target")
    occurrence = target.get("source_occurrence")
    if isinstance(occurrence, Mapping):
        occurrence_ids = _ids(occurrence.get("raw_generated_token_ids", []), label="source occurrence raw ids", allow_empty=False)
        row_ids = _ids(target_row.get("raw_generated_token_ids", []), label="sampled target row raw ids", allow_empty=False)
        if occurrence_ids != row_ids:
            raise StageTwoValidationError("frozen source occurrence does not equal sampled target row raw ids")
    current: list[int] = []
    ladder: list[dict[str, Any]] = []
    for k in range(row_target + 1):
        prefix = list(current)
        expected_hash = hash_prefix_token_ids(prefix)
        ladder.append({
            "k": k,
            "target_sampled_row_index_zero_based": row_target,
            "target_owner_id": owner,
            "sampled_seed": int(target.get("sampled_seed")),
            "prefix_token_ids": prefix,
            "prefix_token_ids_sha256": expected_hash,
            "prefix_row_count": k,
            "prefix_owner_ids": sorted(strict_owner_union({"rows": ordered[:k]})),
            "source_row_token_ids_sha256": [
                hash_prefix_token_ids(_ids(row.get("raw_generated_token_ids", []), label="sampled source row ids", allow_empty=False))
                for row in ordered[:k]
            ],
        })
        if k < row_target:
            row = ordered[k]
            if not _row_is_clean_complete(row):
                raise StageTwoValidationError(f"sampled source row {k} is not a clean complete row")
            current.extend(_ids(row.get("raw_generated_token_ids", []), label=f"sampled source row {k} ids", allow_empty=False))
    declared_parent = str(target.get("target_parent_prefix_sha256", ""))
    if ladder[-1]["prefix_token_ids_sha256"] != declared_parent:
        raise StageTwoValidationError(
            f"reconstructed P_j hash disagrees with frozen parent hash: observed {ladder[-1]['prefix_token_ids_sha256']} expected {declared_parent}"
        )
    return {
        "sampled_seed": int(target.get("sampled_seed")),
        "target_owner_id": owner,
        "target_sampled_row_index_zero_based": row_target,
        "sampled_trajectory_horizon_rows": int(trajectory.get("horizon_rows_generated", len(ordered))),
        "sampled_trajectory_final_prefix_sha256": str(trajectory.get("final_prefix_token_ids_sha256", "")),
        "prefixes": ladder,
        "source_target_row_raw_token_ids_sha256": hash_prefix_token_ids(
            _ids(target_row.get("raw_generated_token_ids", []), label="sampled target row ids", allow_empty=False)
        ),
    }


def _row_owner_ids(row: Mapping[str, Any]) -> set[str]:
    return {
        str(match.get("matched_entity_id"))
        for match in row.get("entity_matches", [])
        if isinstance(match, Mapping)
        and match.get("status") == "matched"
        and match.get("matched_entity_id") is not None
    }


def summarize_continuation(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix_owner_ids: Sequence[str],
    target_owner_id: str,
    target_start_row_index: int,
    generated_token_count: int,
    total_token_budget: int,
    horizon_rows_complete: bool,
) -> dict[str, Any]:
    """Summarize strict target/duplicate/terminal evidence for one ladder arm."""

    prefix_owners = {str(value) for value in prefix_owner_ids}
    continuation_owners: set[str] = set()
    duplicate_owners: set[str] = set()
    seen_owners = set(prefix_owners)
    unresolved: list[int] = []
    malformed: list[int] = []
    target_hits: list[int] = []
    target_hit_quality: list[dict[str, Any]] = []
    target_relevant_ambiguity: list[dict[str, Any]] = []
    target_relevant_owner_issues: list[dict[str, Any]] = []
    crop_review_warnings: list[dict[str, Any]] = []
    terminal_rows: list[int] = []

    def collect_target_candidates(row_index: int, matches: Any) -> bool:
        """Record positive-overlap target candidates and return whether relevant."""

        relevant = False
        if not isinstance(matches, list):
            return relevant
        for match in matches:
            if not isinstance(match, Mapping) or match.get("status") not in {"unmatched", "ambiguous"}:
                continue
            for candidate in match.get("candidates", []):
                if not isinstance(candidate, Mapping) or str(candidate.get("entity_id")) != target_owner_id:
                    continue
                try:
                    iou = float(candidate.get("iou", 0.0))
                except (TypeError, ValueError):
                    iou = 0.0
                if iou > 0.0:
                    relevant = True
                    target_relevant_ambiguity.append({
                        "row_index": row_index,
                        "prediction_index": match.get("prediction_index"),
                        "match_status": match.get("status"),
                        "candidate_iou": iou,
                    })
        return relevant

    for row in rows:
        index = int(row.get("row_index", -1))
        owners = _row_owner_ids(row)
        continuation_owners.update(owners)
        duplicate_owners.update(owners & seen_owners)
        seen_owners.update(owners)
        matches = row.get("entity_matches")
        target_in_owners = target_owner_id in owners
        if not _row_is_clean_complete(row):
            stop = row.get("row_stop") if isinstance(row.get("row_stop"), Mapping) else {}
            reason = str(stop.get("stop_reason", "unknown"))
            if reason == "terminal":
                terminal_rows.append(index)
            elif row.get("status") == "failed" or reason in {"malformed_limit", "contaminated_complete_row", "failed"}:
                malformed.append(index)
            else:
                unresolved.append(index)
            relevant = collect_target_candidates(index, matches)
            if target_in_owners:
                relevant = True
                target_relevant_owner_issues.append({
                    "row_index": index,
                    "strict_owner_ids": sorted(owners),
                    "reason": reason,
                })
            if not relevant and reason not in {"terminal"}:
                crop_review_warnings.append({
                    "row_index": index,
                    "warning": "unrelated_incomplete_or_unmatched_row",
                    "stop_reason": reason,
                })
            continue
        owner_unresolved = isinstance(matches, list) and any(
            isinstance(match, Mapping) and match.get("status") in {"unmatched", "ambiguous"}
            for match in matches
        )
        if owner_unresolved:
            relevant = collect_target_candidates(index, matches) or target_in_owners
            if relevant:
                unresolved.append(index)
                if target_in_owners:
                    target_relevant_owner_issues.append({
                        "row_index": index,
                        "strict_owner_ids": sorted(owners),
                        "reason": "mixed_or_unresolved_target_ownership",
                    })
            else:
                crop_review_warnings.append({
                    "row_index": index,
                    "warning": "unrelated_unmatched_or_ambiguous_prediction",
                })
            continue
        target_present = target_owner_id in owners
        if target_present:
            quality = {
                "row_index": index,
                "strict_owner_ids": sorted(owners),
                "single_owner": owners == {target_owner_id},
                "all_predictions_matched": isinstance(matches, list) and all(
                    isinstance(match, Mapping) and match.get("status") == "matched" for match in matches
                ),
            }
            if not (owners == {target_owner_id} and quality["all_predictions_matched"]):
                unresolved.append(index)
                target_relevant_owner_issues.append({
                    "row_index": index,
                    "strict_owner_ids": sorted(owners),
                    "reason": "mixed_or_unresolved_target_ownership",
                })
                target_hit_quality.append({**quality, "valid_hit": False})
                continue
            target_hits.append(index)
            target_hit_quality.append({**quality, "valid_hit": True})
    first_hit = min(target_hits) if target_hits else None
    if first_hit is not None:
        reachability_state = "hit"
    elif not terminal_rows and not horizon_rows_complete and int(generated_token_count) >= int(total_token_budget):
        # The budget cap is an observation boundary, even if the final row was
        # incomplete or contained ambiguous evidence. Preserve that evidence
        # below rather than misclassifying the arm as a clean miss.
        reachability_state = "right_censored"
    elif unresolved or target_relevant_ambiguity or target_relevant_owner_issues:
        reachability_state = "unresolved"
    elif terminal_rows or horizon_rows_complete:
        reachability_state = "clean_miss"
    elif int(generated_token_count) >= int(total_token_budget):
        reachability_state = "right_censored"
    else:
        reachability_state = "unresolved"
    return {
        "reachability_state": reachability_state,
        "target_first_hit_row_index": first_hit,
        "target_hit_quality": target_hit_quality,
        "target_relevant_ambiguity": target_relevant_ambiguity,
        "target_relevant_owner_issues": target_relevant_owner_issues,
        "crop_review_warnings": crop_review_warnings,
        "prefix_owner_ids": sorted(prefix_owners),
        "continuation_owner_ids": sorted(continuation_owners),
        "all_strict_unique_owner_ids": sorted(prefix_owners | continuation_owners),
        "duplicate_owner_ids": sorted(duplicate_owners),
        "unresolved_row_indices": sorted(set(unresolved)),
        "malformed_row_indices": sorted(set(malformed)),
        "natural_terminal_row_index": min(terminal_rows) if terminal_rows else None,
        "generated_token_count": int(generated_token_count),
        "total_token_budget": int(total_token_budget),
        "budget_exhausted": int(generated_token_count) >= int(total_token_budget),
        "target_start_row_index": int(target_start_row_index),
        "horizon_rows_complete": bool(horizon_rows_complete),
    }


def _generate_prefix_continuation(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    prefix_token_ids: Sequence[int],
    prefix_owner_ids: Sequence[str],
    tokenizer: Any,
    image_width: int,
    image_height: int,
    entity_ledger: Sequence[Mapping[str, Any]],
    start_row_index: int,
    horizon_rows: int,
    malformed_limit: int,
    temperature: float,
    total_token_budget: int = TOTAL_TOKEN_BUDGET,
) -> dict[str, Any]:
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
        _annotate_owner_matches(
            row,
            entity_ledger=entity_ledger,
            image_width=int(image_width),
            image_height=int(image_height),
            covered_entity_ids=sorted(covered),
        )
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
        if reason in {"terminal", "malformed_limit", "contaminated_complete_row", "failed"} or row.get("status") == "failed":
            break
        if not row["appended_to_prefix"]:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provisional-targets", type=Path, required=True)
    parser.add_argument("--discovery-shard-glob", action="append", default=[])
    parser.add_argument("--stage-one-glob", action="append", required=True)
    parser.add_argument("--stage-two-admission", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ids", default="")
    parser.add_argument("--include-stage-one-label", action="append", dest="include_labels", default=list(DEFAULT_ALLOWED_STAGE_ONE_LABELS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_stage_one_outputs(paths: Sequence[Path], *, manifest_sha256: str, provisional_targets_sha256: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        artifact = _load_json(path)
        if artifact.get("schema_version") != "sampled_history_target_reachability.v1" or artifact.get("phase") != "stage_one_extended_root_greedy_screen":
            raise StageTwoValidationError(f"invalid Stage 1 artifact: {path}")
        frozen = artifact.get("frozen_inputs")
        images = artifact.get("images")
        if not isinstance(frozen, Mapping) or not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
            raise StageTwoValidationError(f"Stage 1 artifact lacks one-image identity receipt: {path}")
        if str(frozen.get("manifest_sha256")) != manifest_sha256 or str(frozen.get("provisional_targets_sha256")) != provisional_targets_sha256:
            raise StageTwoValidationError(f"Stage 1 frozen input digest mismatch: {path}")
        image_id = str(images[0].get("image_id"))
        if image_id in result:
            raise StageTwoValidationError(f"duplicate Stage 1 image {image_id}")
        source_discovery = images[0].get("source_discovery_shard")
        if not isinstance(source_discovery, Mapping) or not str(source_discovery.get("sha256", "")):
            raise StageTwoValidationError(f"Stage 1 artifact lacks discovery-shard identity: {path}")
        result[image_id] = {"path": str(path), "sha256": sha256_file(path), "artifact": artifact, "image": dict(images[0])}
    return result


def _stage_one_label_matches_policy(actual: str, declared: str) -> bool:
    """Check the frozen admission stratum against the actual Stage 1 label."""

    if actual == declared:
        return True
    # The admission document may refine a delayed result with final-set
    # censoring, while the Stage 1 artifact's canonical label remains
    # ``delayed``.
    return actual == "delayed" and declared.startswith("delayed")


def load_stage_two_admission(path: Path) -> dict[str, Any]:
    """Load and verify the frozen post-Stage-1 admission decision.

    The admission document is an explicit scientific input.  Its declared
    Stage 1 union is re-hashed, required to have passed, and each referenced
    per-image Stage 1 artifact is re-hashed before any target is admitted.
    """

    admission_path = path.expanduser().resolve(strict=True)
    document = _load_json(admission_path)
    if document.get("schema_version") != 1:
        raise StageTwoValidationError("Stage 2 admission schema version mismatch")
    if document.get("unit_id") != "2026-07-19-sampled-history-target-reachability-and-complete-row-value":
        raise StageTwoValidationError("Stage 2 admission unit mismatch")
    union_path_value = document.get("source_stage_one_union")
    declared_union_sha = str(document.get("source_stage_one_union_sha256", ""))
    if not union_path_value or not declared_union_sha:
        raise StageTwoValidationError("Stage 2 admission lacks declared Stage 1 union path/hash")
    union_path = Path(str(union_path_value)).expanduser().resolve(strict=True)
    observed_union_sha = sha256_file(union_path)
    if observed_union_sha != declared_union_sha:
        raise StageTwoValidationError("Stage 2 admission Stage 1 union SHA-256 mismatch")
    union = _load_json(union_path)
    if union.get("passed") is not True or union.get("schema_version") != "sampled_history_target_reachability.union.v1":
        raise StageTwoValidationError("declared Stage 1 union is not a passed Stage 1 union artifact")
    union_images = union.get("images")
    if not isinstance(union_images, list) or not union_images:
        raise StageTwoValidationError("declared Stage 1 union has no image records")
    stage_one_records: dict[str, dict[str, Any]] = {}
    for record in union_images:
        if not isinstance(record, Mapping):
            raise StageTwoValidationError("declared Stage 1 union contains malformed image record")
        image_id = str(record.get("image_id"))
        stage_path_value = record.get("path")
        declared_sha = str(record.get("sha256", ""))
        if not image_id or not stage_path_value or not declared_sha or image_id in stage_one_records:
            raise StageTwoValidationError("declared Stage 1 union image identity is invalid")
        stage_path = Path(str(stage_path_value)).expanduser().resolve(strict=True)
        observed_sha = sha256_file(stage_path)
        if observed_sha != declared_sha:
            raise StageTwoValidationError(f"declared Stage 1 artifact SHA-256 mismatch for image {image_id}")
        stage_artifact = _load_json(stage_path)
        images = stage_artifact.get("images")
        if not isinstance(images, list) or len(images) != 1 or str(images[0].get("image_id")) != image_id:
            raise StageTwoValidationError(f"Stage 1 union artifact image mismatch for {image_id}")
        stage_one_records[image_id] = {
            "path": str(stage_path),
            "sha256": observed_sha,
            "artifact": stage_artifact,
            "image": dict(images[0]),
            "label": str((images[0].get("target_classification") or {}).get("label", "")),
        }
    targets = document.get("targets")
    if not isinstance(targets, list) or not targets:
        raise StageTwoValidationError("Stage 2 admission has no target policy entries")
    entries: dict[str, dict[str, Any]] = {}
    for raw_entry in targets:
        if not isinstance(raw_entry, Mapping):
            raise StageTwoValidationError("Stage 2 admission target entry is malformed")
        entry = dict(raw_entry)
        image_id = str(entry.get("image_id"))
        if not image_id or image_id in entries or image_id not in stage_one_records:
            raise StageTwoValidationError(f"Stage 2 admission target image identity is invalid: {image_id}")
        actual_label = stage_one_records[image_id]["label"]
        declared_stratum = str(entry.get("final_recall_stratum", ""))
        if not actual_label or not declared_stratum or not _stage_one_label_matches_policy(actual_label, declared_stratum):
            raise StageTwoValidationError(
                f"Stage 2 admission policy disagrees with Stage 1 label for image {image_id}: actual={actual_label!r} declared={declared_stratum!r}"
            )
        if not isinstance(entry.get("run_prefix_ladder"), bool):
            raise StageTwoValidationError(f"Stage 2 admission run_prefix_ladder must be boolean for {image_id}")
        if not str(entry.get("allowed_claim", "")):
            raise StageTwoValidationError(f"Stage 2 admission allowed_claim is missing for {image_id}")
        entries[image_id] = entry
    selected = {image_id: entry for image_id, entry in entries.items() if entry["run_prefix_ladder"]}
    if not selected:
        raise StageTwoValidationError("Stage 2 admission selects no prefix-ladder targets")
    return {
        "path": str(admission_path),
        "sha256": sha256_file(admission_path),
        "document": document,
        "stage_one_union_path": str(union_path),
        "stage_one_union_sha256": observed_union_sha,
        "stage_one_union": union,
        "stage_one_records": stage_one_records,
        "entries": entries,
        "selected_entries": selected,
    }


def _validate_stage_two_sources(
    *,
    manifest_path: Path,
    provisional_targets_path: Path,
    discovery_paths: Sequence[Path],
    stage_one_paths: Sequence[Path],
    admission_path: Path,
    include_labels: Sequence[str],
) -> dict[str, Any]:
    admission = load_stage_two_admission(admission_path)
    frozen = validate_frozen_inputs(manifest_path=manifest_path, provisional_targets_path=provisional_targets_path, shard_paths=discovery_paths)
    stage_one = _load_stage_one_outputs(stage_one_paths, manifest_sha256=frozen["manifest_sha256"], provisional_targets_sha256=frozen["provisional_targets_sha256"])
    declared_stage_one = admission["stage_one_records"]
    if set(stage_one) != set(declared_stage_one):
        raise StageTwoValidationError(
            "Stage 1 artifacts do not exactly cover the admission union: "
            f"missing={sorted(set(declared_stage_one) - set(stage_one))} "
            f"extra={sorted(set(stage_one) - set(declared_stage_one))}"
        )
    for image_id, record in stage_one.items():
        declared = declared_stage_one[image_id]
        if record["path"] != declared["path"] or record["sha256"] != declared["sha256"]:
            raise StageTwoValidationError(f"Stage 1 artifact disagrees with admitted union for image {image_id}")
    for image_id, record in stage_one.items():
        expected = frozen["shards"].get(image_id, {}).get("sha256")
        observed = record["image"].get("source_discovery_shard", {}).get("sha256")
        if expected != observed:
            raise StageTwoValidationError(f"Stage 1 discovery-shard digest mismatch for image {image_id}")
    allowed_labels = {str(value) for value in include_labels}
    target_images: dict[str, dict[str, Any]] = {}
    for image_id, admission_entry in admission["selected_entries"].items():
        if image_id not in frozen["target_images"]:
            raise StageTwoValidationError(f"admitted target image is absent from provisional targets: {image_id}")
        if image_id not in stage_one:
            raise StageTwoValidationError(f"missing Stage 1 output for target image {image_id}")
        target = frozen["target_images"][image_id]
        if str(admission_entry.get("target_owner_id")) != str(target.get("target_owner_id")):
            raise StageTwoValidationError(f"admission target owner disagrees with provisional target for image {image_id}")
        stage_image = stage_one[image_id]["image"]
        classification = stage_image.get("target_classification")
        label = str(classification.get("label")) if isinstance(classification, Mapping) else "missing"
        if label not in allowed_labels:
            raise StageTwoValidationError(f"admission-selected target label is excluded by --include-stage-one-label for image {image_id}: {label}")
        reconstructed = reconstruct_sampled_prefix_ladder(frozen["shards"][image_id]["image"], target)
        target_images[image_id] = {
            "target": target,
            "stage_one": stage_one[image_id],
            "stage_one_label": label,
            "admission_entry": dict(admission_entry),
            "sampled_prefix_ladder": reconstructed,
        }
    if not target_images:
        raise StageTwoValidationError("no Stage 1 target remains after label filtering")
    return {
        **frozen,
        "stage_one": stage_one,
        "target_images": target_images,
        "include_stage_one_labels": [str(value) for value in include_labels],
        "stage2_admission": admission,
    }


def main() -> int:
    args = _parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    discovery_paths = [Path(item) for pattern in args.discovery_shard_glob for item in glob.glob(pattern)]
    stage_one_paths = [Path(item) for pattern in args.stage_one_glob for item in glob.glob(pattern)]
    try:
        frozen = _validate_stage_two_sources(
            manifest_path=args.manifest,
            provisional_targets_path=args.provisional_targets,
            discovery_paths=discovery_paths,
            stage_one_paths=stage_one_paths,
            admission_path=args.stage_two_admission,
            include_labels=args.include_labels,
        )
    except (OSError, StageOneValidationError, StageTwoValidationError) as exc:
        raise SystemExit(f"Stage 2 source validation failed before runtime: {exc}") from exc
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
        raise SystemExit(f"Stage 2 requires model.dtype=fp32, observed {config.model.dtype!r}")
    manifest = frozen["manifest"]
    contract = manifest.get("inference_contract") or {}
    budget = manifest.get("discovery_budget") or {}
    source_jsonl = Path(str(contract.get("source_jsonl", config.data.input_jsonl))).expanduser().resolve(strict=True)
    if Path(str(config.data.input_jsonl)).expanduser().resolve(strict=True) != source_jsonl:
        raise SystemExit("frozen source JSONL path disagrees with resolved inference config")
    frozen_file_identity = validate_frozen_file_identity(manifest, infer_config_path=args.infer_config, source_jsonl_path=source_jsonl)
    examples = list(load_raw_examples(source_jsonl))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    selected_ids = {value.strip() for value in str(args.image_ids).split(",") if value.strip()}
    all_ids = set(frozen["target_images"])
    if selected_ids - all_ids:
        raise SystemExit(f"--image-ids outside Stage 2 target set: {sorted(selected_ids - all_ids)}")
    image_ids = sorted(selected_ids or all_ids, key=_image_sort)
    outputs: list[dict[str, Any]] = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    with open_backend_session(frontend.launch) as session:
        for image_id in image_ids:
            source_image = frozen["shards"][image_id]["image"]
            target_data = frozen["target_images"][image_id]
            target = target_data["target"]
            ladder_source = target_data["sampled_prefix_ladder"]
            example = _select_example(examples, image_id)
            request, plan, prompt_meta = _build_request(config, frontend, example)
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
            one_native = _single_native_inputs(native_inputs)
            observed_runtime_identity = {
                "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
                "executed_media_sha256": media_sha[0],
                "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]),
            }
            identity_check = compare_execution_identity(observed_prompt=prompt_meta, observed_runtime=observed_runtime_identity, discovery_prompt=source_image.get("prompt", {}), discovery_runtime=source_image.get("runtime", {}))
            if not identity_check["passed"]:
                raise SystemExit(f"discovery/runtime identity mismatch for image {image_id}: {identity_check['checks']}")
            ledger = [dict(row) for row in source_image.get("entity_ledger", []) if isinstance(row, Mapping)] or build_positive_entity_ledger(example)
            entries: list[dict[str, Any]] = []
            for prefix in ladder_source["prefixes"]:
                k = int(prefix["k"])
                continuation = _generate_prefix_continuation(
                    session=session,
                    native_inputs=one_native,
                    prefix_token_ids=prefix["prefix_token_ids"],
                    prefix_owner_ids=prefix["prefix_owner_ids"],
                    tokenizer=session._tokenizer,
                    image_width=int(plan.decoded_width),
                    image_height=int(plan.decoded_height),
                    entity_ledger=ledger,
                    start_row_index=k,
                    horizon_rows=MAX_ROOT_ROWS - k,
                    malformed_limit=int(budget.get("malformed_row_limit", 2)),
                    temperature=float(contract.get("temperature", 0.4)),
                )
                summary = summarize_continuation(
                    continuation["rows"],
                    prefix_owner_ids=prefix["prefix_owner_ids"],
                    target_owner_id=str(target["target_owner_id"]),
                    target_start_row_index=k,
                    generated_token_count=int(continuation["generated_token_count"]),
                    total_token_budget=TOTAL_TOKEN_BUDGET,
                    horizon_rows_complete=bool(continuation["horizon_rows_complete"]),
                )
                entries.append({"prefix": prefix, "continuation": continuation, **summary})
            outputs.append({
                "image_id": image_id,
                "provisional_target": target,
                "stage2_admission_entry": target_data["admission_entry"],
                "stage_one_source": {"path": target_data["stage_one"]["path"], "sha256": target_data["stage_one"]["sha256"], "target_classification": target_data["stage_one"]["image"].get("target_classification")},
                "sampled_prefix_ladder": ladder_source,
                "prefix_evaluations": entries,
                "entity_ledger": ledger,
                "prompt": prompt_meta,
                "runtime": observed_runtime_identity,
                "identity_check": identity_check,
            })
        model_receipt = session.receipt.to_artifact_dict()
    source_identity = git_execution_identity(Path(__file__).resolve().parents[2])
    source_identity.pop("runner_sha256", None)
    source_identity["stage_two_runner_sha256"] = sha256_file(Path(__file__).resolve())
    source_identity["reused_local_branch_helper_sha256"] = sha256_file(Path(__file__).with_name("run_local_branch_causal_value.py"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "sampled_history_target_reachability",
        "phase": PHASE,
        "source_identity": source_identity,
        "frozen_inputs": {
            "manifest": frozen["manifest_path"],
            "manifest_sha256": frozen["manifest_sha256"],
            "provisional_targets": frozen["provisional_targets_path"],
            "provisional_targets_sha256": frozen["provisional_targets_sha256"],
            "stage_two_admission": frozen["stage2_admission"]["path"],
            "stage_two_admission_sha256": frozen["stage2_admission"]["sha256"],
            "stage_one_union": frozen["stage2_admission"]["stage_one_union_path"],
            "stage_one_union_sha256": frozen["stage2_admission"]["stage_one_union_sha256"],
            "discovery_shard_sha256": {image_id: frozen["shards"][image_id]["sha256"] for image_id in sorted(frozen["shards"], key=_image_sort)},
            "stage_one_artifact_sha256": {image_id: frozen["stage_one"][image_id]["sha256"] for image_id in sorted(frozen["stage_one"], key=_image_sort)},
            "stage2_admission_entries": {image_id: frozen["target_images"][image_id]["admission_entry"] for image_id in sorted(frozen["target_images"], key=_image_sort)},
            "checkpoint_config_source_identity": frozen_file_identity,
        },
        "config": {
            "infer_config": str(args.infer_config.expanduser().resolve()),
            "resolved_config_fingerprint": resolved.fingerprint,
            "device": args.device,
            "physical_batch_size": 1,
            "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "total_generated_token_budget_per_ladder": TOTAL_TOKEN_BUDGET,
            "absolute_horizon_complete_rows": MAX_ROOT_ROWS,
            "included_stage_one_labels": [str(value) for value in args.include_labels],
        },
        "model_identity": model_receipt,
        "images": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
