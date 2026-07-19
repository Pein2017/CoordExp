#!/usr/bin/env python3
"""Run the frozen Stage 1 extended greedy reachability screen.

This experiment-local runner deliberately does one thing: it replays the
frozen root-greedy trajectory with a total budget of 512 newly generated token
identifiers, while retaining every raw row and positive-only owner match.  It
does not run interventions, score logits, or train a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import glob
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


SCHEMA_VERSION = "sampled_history_target_reachability.v1"
MAX_ROOT_ROWS = 8
TOTAL_TOKEN_BUDGET = 512


class StageOneValidationError(ValueError):
    """Raised when a frozen Stage 1 input is not trustworthy."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageOneValidationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageOneValidationError(f"JSON artifact must be an object: {path}")
    return value


def _ids(value: Any, *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageOneValidationError(f"{label} must be a token-id sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise StageOneValidationError(f"{label} contains an invalid token id")
        result.append(int(item))
    return result


def _image_sort(value: str) -> tuple[int, str]:
    return (0, value) if value.isdigit() else (1, value)


def _target_key(target: Mapping[str, Any]) -> tuple[str, str]:
    return str(target.get("image_id")), str(target.get("target_owner_id"))


def _find_root_trajectory(image: Mapping[str, Any]) -> Mapping[str, Any]:
    trajectories = image.get("trajectories")
    if not isinstance(trajectories, list):
        raise StageOneValidationError("discovery image trajectories are missing")
    roots = [
        item for item in trajectories
        if isinstance(item, Mapping) and item.get("mode") == "greedy" and item.get("seed") is None
    ]
    if len(roots) != 1:
        raise StageOneValidationError("discovery image must contain exactly one greedy root trajectory")
    return roots[0]


def strict_owner_union(trajectory: Mapping[str, Any]) -> set[str]:
    """Recompute resolved physical owners from row match evidence."""

    owners: set[str] = set()
    rows = trajectory.get("rows")
    if not isinstance(rows, list):
        return owners
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        matches = row.get("entity_matches")
        if isinstance(matches, list):
            owners.update(
                str(match.get("matched_entity_id"))
                for match in matches
                if isinstance(match, Mapping)
                and match.get("status") == "matched"
                and match.get("matched_entity_id") is not None
            )
        else:
            owners.update(str(value) for value in row.get("strict_matched_owner_ids", []))
    return owners


def sampled_owner_union(image: Mapping[str, Any]) -> set[str]:
    """Union resolved owners from each sampled root trajectory exactly once."""

    trajectories = image.get("trajectories")
    if not isinstance(trajectories, list):
        return set()
    result: set[str] = set()
    for trajectory in trajectories:
        if isinstance(trajectory, Mapping) and trajectory.get("mode") == "sample":
            result.update(strict_owner_union(trajectory))
    return result


def compare_execution_identity(
    *,
    observed_prompt: Mapping[str, Any],
    observed_runtime: Mapping[str, Any],
    discovery_prompt: Mapping[str, Any],
    discovery_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the exact image/request identity before interpreting a replay."""

    fields = {
        "executed_media_sha256": (observed_runtime.get("executed_media_sha256"), discovery_runtime.get("executed_media_sha256")),
        "executed_prompt_token_ids_sha256": (observed_runtime.get("executed_prompt_token_ids_sha256"), discovery_runtime.get("executed_prompt_token_ids_sha256")),
        "observed_image_grid_thw": (observed_runtime.get("observed_image_grid_thw"), discovery_runtime.get("observed_image_grid_thw")),
        "image_sha256": (observed_prompt.get("image_sha256"), discovery_prompt.get("image_sha256")),
        "chat_text_sha256": (observed_prompt.get("chat_text_sha256"), discovery_prompt.get("chat_text_sha256")),
        "prompt_token_ids_sha256": (observed_prompt.get("prompt_token_ids_sha256"), discovery_prompt.get("prompt_token_ids_sha256")),
        "width": (observed_prompt.get("width"), discovery_prompt.get("width")),
        "height": (observed_prompt.get("height"), discovery_prompt.get("height")),
    }
    checks: dict[str, bool] = {}
    for name, (observed, expected) in fields.items():
        # These fields are present in the frozen discovery schema.  If a
        # future artifact omits an optional hash on both sides, equality is
        # still well-defined; one-sided omission is a refusal.
        checks[name] = observed == expected
    return {"passed": all(checks.values()), "checks": checks, "observed": {name: values[0] for name, values in fields.items()}, "expected": {name: values[1] for name, values in fields.items()}}


def _find_sample_target(
    image: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a frozen target is present in its exact natural sampled action."""

    expected_hash = str(target.get("target_parent_prefix_sha256", target.get("target_parent_prefix_token_ids_sha256", "")))
    # The provisional file uses the shorter field name; accept only that or
    # the explicit token-id spelling, never a silently recomputed alternate.
    if not expected_hash:
        raise StageOneValidationError(f"target {_target_key(target)} has no parent prefix hash")
    seed = target.get("sampled_seed")
    row_index = target.get("first_sampled_row_index_zero_based")
    owner = str(target.get("target_owner_id"))
    evaluations = image.get("prefix_evaluations")
    if not isinstance(evaluations, list):
        raise StageOneValidationError("discovery image prefix evaluations are missing")
    matches: list[dict[str, Any]] = []
    dirty_matches: list[dict[str, Any]] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            continue
        prefix = evaluation.get("prefix")
        if not isinstance(prefix, Mapping):
            continue
        prefix_hash = str(prefix.get("prefix_token_ids_sha256", ""))
        if prefix_hash != expected_hash:
            continue
        for action in prefix.get("natural_actions", []):
            if not isinstance(action, Mapping):
                continue
            if action.get("mode") != "sample" or action.get("seed") != seed or action.get("row_index") != row_index:
                continue
            owners = {str(value) for value in action.get("strict_matched_owner_ids", [])}
            occurrence = {
                "prefix_hash": prefix_hash,
                "seed": int(seed),
                "row_index": int(row_index),
                "target_owner_id": owner,
                "raw_generated_token_ids": _ids(action.get("raw_generated_token_ids", []), label="natural sampled action raw ids"),
                "raw_generated_token_ids_sha256": action.get("raw_generated_token_ids_sha256"),
                "action": dict(action),
            }
            stop_reason = (action.get("row_stop") or {}).get("stop_reason") if isinstance(action.get("row_stop"), Mapping) else None
            clean = action.get("status") == "success" and stop_reason == "complete_row" and owners == {owner}
            if clean:
                matches.append(occurrence)
            elif owner in owners:
                dirty_matches.append(occurrence)
    if dirty_matches and not matches:
        raise StageOneValidationError(
            f"target {_target_key(target)} resolves only to a non-clean sampled row"
        )
    if len(matches) != 1:
        raise StageOneValidationError(
            f"target {_target_key(target)} must resolve to exactly one natural sampled action; found {len(matches)}"
        )
    return matches[0]


def validate_frozen_inputs(
    *, manifest_path: Path, provisional_targets_path: Path, shard_paths: Sequence[Path]
) -> dict[str, Any]:
    """Validate all target occurrences and both negative controls before runtime."""

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    provisional_targets_path = provisional_targets_path.expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path)
    targets_doc = _load_json(provisional_targets_path)
    if targets_doc.get("schema_version") != 1:
        raise StageOneValidationError("provisional target schema version mismatch")
    if targets_doc.get("unit_id") != "2026-07-19-sampled-history-target-reachability-and-complete-row-value":
        raise StageOneValidationError("provisional target unit mismatch")
    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != 8:
        raise StageOneValidationError("frozen discovery manifest must contain eight images")
    image_map = {str(item.get("image_id")): item for item in images if isinstance(item, Mapping)}
    if set(image_map) != {str(item.get("image_id")) for item in targets_doc.get("provisional_targets", [])} | {str(v) for v in targets_doc.get("negative_discovery_controls", [])} | {str(targets_doc.get("root_decision_diagnostic", {}).get("image_id"))}:
        raise StageOneValidationError("manifest and provisional target image sets disagree")
    shard_by_image: dict[str, tuple[Path, dict[str, Any]]] = {}
    expected_hashes = {str(k): str(v) for k, v in (targets_doc.get("source_artifacts") or {}).items()}
    for path in shard_paths:
        resolved = path.expanduser().resolve(strict=True)
        artifact_hash = sha256_file(resolved)
        shard = _load_json(resolved)
        if shard.get("schema_version") != "local_branch_causal_value.v2" or shard.get("phase") != "discovery":
            raise StageOneValidationError(f"invalid discovery shard: {resolved}")
        shard_images = shard.get("images")
        if not isinstance(shard_images, list) or len(shard_images) != 1:
            raise StageOneValidationError(f"each discovery shard must contain one image: {resolved}")
        image_id = str(shard_images[0].get("image_id"))
        if image_id not in image_map or image_id in shard_by_image:
            raise StageOneValidationError(f"duplicate or unknown discovery shard image: {image_id}")
        if expected_hashes.get(image_id) != artifact_hash:
            raise StageOneValidationError(f"discovery shard hash mismatch for image {image_id}")
        shard_by_image[image_id] = (resolved, shard)
    if set(shard_by_image) != set(image_map):
        raise StageOneValidationError(f"discovery shards do not cover frozen pool: missing {sorted(set(image_map) - set(shard_by_image))}")
    provisional = targets_doc.get("provisional_targets")
    if not isinstance(provisional, list) or len(provisional) != 6:
        raise StageOneValidationError("Stage 1 requires exactly six frozen provisional targets")
    target_map = {_target_key(item): item for item in provisional if isinstance(item, Mapping)}
    if len(target_map) != 6:
        raise StageOneValidationError("provisional targets must be unique")
    verified_targets: list[dict[str, Any]] = []
    for target in provisional:
        image_id = str(target.get("image_id"))
        _, shard = shard_by_image[image_id]
        image = shard["images"][0]
        occurrence = _find_sample_target(image, target)
        root = _find_root_trajectory(image)
        root_first_eight_owners = strict_owner_union({
            "rows": list(root.get("rows", []))[:MAX_ROOT_ROWS],
        })
        target_owner = str(target.get("target_owner_id"))
        if target_owner in root_first_eight_owners:
            raise StageOneValidationError(
                f"frozen target {image_id}/{target_owner} is already in root-greedy first-eight owners"
            )
        expected_action_hash = hash_prefix_token_ids(occurrence["raw_generated_token_ids"])
        declared_action_hash = str(occurrence.get("raw_generated_token_ids_sha256") or "")
        if declared_action_hash and declared_action_hash != expected_action_hash:
            raise StageOneValidationError(
                f"sampled target row hash mismatch for {image_id}/{target_owner}"
            )
        verified_targets.append({**dict(target), "source_occurrence": occurrence})
    negative_controls = [str(value) for value in targets_doc.get("negative_discovery_controls", [])]
    for image_id in negative_controls:
        image = shard_by_image[image_id][1]["images"][0]
        root = _find_root_trajectory(image)
        greedy_owners = strict_owner_union(root)
        if not isinstance(image.get("trajectories"), list):
            raise StageOneValidationError(f"negative discovery control {image_id} has no trajectories")
        sampled_owners = sampled_owner_union(image)
        sampled_only = sorted(sampled_owners - greedy_owners)
        if sampled_only:
            raise StageOneValidationError(
                f"negative discovery control {image_id} has sampled-only physical owners: {sampled_only}"
            )
    return {
        "manifest": manifest,
        "targets_document": targets_doc,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "provisional_targets_path": str(provisional_targets_path),
        "provisional_targets_sha256": sha256_file(provisional_targets_path),
        "target_images": {str(item["image_id"]): item for item in verified_targets},
        "negative_control_image_ids": negative_controls,
        "negative_control_owner_unions": {
            image_id: {
                "greedy_owner_ids": sorted(strict_owner_union(_find_root_trajectory(shard_by_image[image_id][1]["images"][0]))),
                "sampled_union_owner_ids": sorted(sampled_owner_union(shard_by_image[image_id][1]["images"][0])),
            }
            for image_id in negative_controls
        },
        "shards": {
            image_id: {"path": str(path), "sha256": sha256_file(path), "image": shard["images"][0]}
            for image_id, (path, shard) in shard_by_image.items()
        },
    }


def first_eight_parity(
    fresh_rows: Sequence[Mapping[str, Any]], source_root: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exact raw-row parity before interpreting the extension."""

    source_rows = source_root.get("rows")
    checks: list[dict[str, Any]] = []
    if not isinstance(source_rows, list) or len(source_rows) < MAX_ROOT_ROWS or len(fresh_rows) < MAX_ROOT_ROWS:
        return {"passed": False, "reason": "source_or_fresh_root_has_fewer_than_eight_rows", "checks": checks}
    for index in range(MAX_ROOT_ROWS):
        expected = _ids(source_rows[index].get("raw_generated_token_ids", []), label="source root raw ids")
        observed = _ids(fresh_rows[index].get("raw_generated_token_ids", []), label="fresh root raw ids")
        checks.append({
            "row_index": index,
            "expected_sha256": hash_prefix_token_ids(expected),
            "observed_sha256": hash_prefix_token_ids(observed),
            "raw_token_ids_equal": expected == observed,
            "source_complete_row": (source_rows[index].get("row_stop") or {}).get("stop_reason") == "complete_row",
            "fresh_complete_row": (fresh_rows[index].get("row_stop") or {}).get("stop_reason") == "complete_row",
            "status_equal": source_rows[index].get("status") == fresh_rows[index].get("status") == "success",
        })
    return {
        "passed": all(
            bool(item["raw_token_ids_equal"])
            and bool(item["source_complete_row"])
            and bool(item["fresh_complete_row"])
            and bool(item["status_equal"])
            for item in checks
        ),
        "checks": checks,
    }


def plausible_target_ambiguity(row: Mapping[str, Any], *, target_owner_id: str) -> dict[str, Any] | None:
    """Return narrow candidate evidence that a row may contain the target.

    Only an unresolved or ambiguous match with a candidate entry for the
    target and strictly positive intersection-over-union is treated as
    plausible.  Zero-overlap candidates are not promoted to ambiguity.
    """

    target = str(target_owner_id)
    matches = row.get("entity_matches")
    if not isinstance(matches, list):
        return None
    for match in matches:
        if not isinstance(match, Mapping) or match.get("status") not in {"unmatched", "ambiguous"}:
            continue
        candidates = match.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or str(candidate.get("entity_id")) != target:
                continue
            try:
                iou = float(candidate.get("iou", 0.0))
            except (TypeError, ValueError):
                continue
            if iou > 0.0:
                return {
                    "row_index": int(row.get("row_index", -1)),
                    "prediction_index": match.get("prediction_index"),
                    "match_status": str(match.get("status")),
                    "target_owner_id": target,
                    "candidate_iou": iou,
                    "candidate_center_distance_norm": candidate.get("center_distance_norm"),
                }
    return None


def classify_target_from_extended_rows(
    rows: Sequence[Mapping[str, Any]], *, target_owner_id: str, total_token_budget: int = TOTAL_TOKEN_BUDGET
) -> dict[str, Any]:
    """Classify a target without treating malformed output as terminal."""

    owner = str(target_owner_id)
    generated_total = sum(len(_ids(row.get("raw_generated_token_ids", []), label="row raw ids")) for row in rows)
    target_rows: list[int] = []
    target_ambiguous = False
    target_ambiguity_evidence: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "success" or (row.get("row_stop") or {}).get("stop_reason") != "complete_row":
            continue
        evidence = plausible_target_ambiguity(row, target_owner_id=owner)
        if evidence is not None:
            target_ambiguity_evidence.append(evidence)
        owners = {str(value) for value in row.get("strict_matched_owner_ids", [])}
        if owner not in owners:
            continue
        matches = row.get("entity_matches")
        clean = (
            len(owners) == 1
            and isinstance(matches, list)
            and bool(matches)
            and all(isinstance(match, Mapping) and match.get("status") == "matched" for match in matches)
        )
        if clean:
            target_rows.append(int(row.get("row_index", -1)))
        else:
            target_ambiguous = True
    first_target_row = min(target_rows) if target_rows else None
    if first_target_row is not None:
        return {"label": "delayed" if first_target_row >= MAX_ROOT_ROWS else "unresolved", "target_first_row_index": first_target_row, "generated_token_count": generated_total, "terminal_row_index": None}
    for row in rows:
        stop_reason = str((row.get("row_stop") or {}).get("stop_reason", "unknown"))
        if stop_reason == "terminal":
            if target_ambiguity_evidence:
                return {"label": "unresolved", "target_first_row_index": None, "generated_token_count": generated_total, "terminal_row_index": None, "refusal_reason": "plausible_target_ambiguity_before_terminal", "target_ambiguity_evidence": target_ambiguity_evidence}
            return {"label": "terminally_omitted", "target_first_row_index": None, "generated_token_count": generated_total, "terminal_row_index": int(row.get("row_index", -1))}
        if stop_reason in {"malformed_limit", "contaminated_complete_row", "failed"} or row.get("status") == "failed":
            return {"label": "unresolved", "target_first_row_index": None, "generated_token_count": generated_total, "terminal_row_index": None, "refusal_reason": "malformed_or_failed_before_target"}
    if target_ambiguous:
        return {"label": "unresolved", "target_first_row_index": None, "generated_token_count": generated_total, "terminal_row_index": None, "refusal_reason": "target_owner_mixed_or_ambiguous_row", "target_ambiguity_evidence": target_ambiguity_evidence}
    if generated_total >= int(total_token_budget):
        if target_ambiguity_evidence:
            return {"label": "unresolved", "target_first_row_index": None, "generated_token_count": generated_total, "terminal_row_index": None, "refusal_reason": "plausible_target_ambiguity_at_budget_cap", "target_ambiguity_evidence": target_ambiguity_evidence}
        return {"label": "right_censored", "target_first_row_index": None, "generated_token_count": generated_total, "terminal_row_index": None}
    return {"label": "unresolved", "target_first_row_index": None, "generated_token_count": generated_total, "terminal_row_index": None, "refusal_reason": "trajectory_ended_without_terminal_or_budget_cap"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provisional-targets", type=Path, required=True)
    parser.add_argument("--shard-glob", action="append", default=[])
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ids", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _generate_extended_root(*, session: Any, native_inputs: Mapping[str, Any], tokenizer: Any, image_width: int, image_height: int, malformed_limit: int, temperature: float, total_token_budget: int = TOTAL_TOKEN_BUDGET) -> dict[str, Any]:
    current: list[int] = []
    rows: list[dict[str, Any]] = []
    generated_total = 0
    while generated_total < int(total_token_budget):
        remaining = int(total_token_budget) - generated_total
        row_index = len(rows)
        row = _generate_row(
            session=session,
            native_inputs=native_inputs,
            prefix_token_ids=current,
            tokenizer=tokenizer,
            image_width=image_width,
            image_height=image_height,
            mode="greedy",
            seed=None,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.0,
            max_new_tokens=remaining,
            malformed_limit=malformed_limit,
            row_index=row_index,
        )
        row["row_index"] = row_index
        row["input_prefix_token_ids"] = list(current)
        row["input_prefix_token_ids_sha256"] = hash_prefix_token_ids(current)
        raw = _ids(row.get("raw_generated_token_ids", []), label="fresh root raw ids")
        generated_total += len(raw)
        row["generated_token_count_after_row"] = generated_total
        row["total_budget_remaining_before_row"] = remaining
        current_after, append_receipt = append_row_if_complete(current, row)
        row["append_receipt"] = append_receipt
        row["accepted_complete_row"] = bool(append_receipt.get("appended"))
        row["appended_to_prefix"] = bool(append_receipt.get("appended"))
        if row["appended_to_prefix"]:
            current = current_after
        row["cumulative_prefix_token_ids"] = list(current)
        row["cumulative_prefix_token_ids_sha256"] = hash_prefix_token_ids(current)
        rows.append(row)
        stop_reason = str((row.get("row_stop") or {}).get("stop_reason", "unknown"))
        if stop_reason in {"terminal", "malformed_limit", "contaminated_complete_row"} or row.get("status") == "failed":
            break
        if not row["appended_to_prefix"]:
            break
    terminal = any((row.get("row_stop") or {}).get("stop_reason") == "terminal" for row in rows)
    return {
        "mode": "greedy",
        "rows": rows,
        "final_prefix_token_ids": current,
        "final_prefix_token_ids_sha256": hash_prefix_token_ids(current),
        "total_generated_token_count": generated_total,
        "total_token_budget": int(total_token_budget),
        "budget_exhausted": generated_total >= int(total_token_budget),
        "natural_terminal": terminal,
    }


def main() -> int:
    args = _parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    shard_paths = [Path(item) for pattern in args.shard_glob for item in glob.glob(pattern)]
    try:
        frozen = validate_frozen_inputs(
            manifest_path=args.manifest,
            provisional_targets_path=args.provisional_targets,
            shard_paths=shard_paths,
        )
    except (OSError, StageOneValidationError) as exc:
        raise SystemExit(f"frozen Stage 1 validation failed before runtime: {exc}") from exc
    # Runtime imports are intentionally delayed until every target occurrence,
    # shard hash, and negative control has passed validation.
    try:
        import torch
        from src.config.fingerprint import sha256_json
        from src.config.inference import load_infer_config
        from src.data import load_raw_examples
        from src.inference.backend import open_backend_session
        from src.inference.runtime import assemble_frontend
        from scripts.research.run_same_covered_set_prefix_order_probe import _build_request, _select_example
    except Exception as exc:
        raise SystemExit(f"runtime import failed; no artifact was written: {type(exc).__name__}: {exc}") from exc
    resolved = load_infer_config(args.infer_config.expanduser().resolve(strict=True))
    config = resolved.config
    if str(config.model.dtype) != "fp32":
        raise SystemExit(f"Stage 1 requires model.dtype=fp32, observed {config.model.dtype!r}")
    manifest = frozen["manifest"]
    contract = manifest.get("inference_contract") or {}
    budget = manifest.get("discovery_budget") or {}
    source_jsonl = Path(str(contract.get("source_jsonl", config.data.input_jsonl))).expanduser().resolve(strict=True)
    configured_source_jsonl = Path(str(config.data.input_jsonl)).expanduser().resolve(strict=True)
    if configured_source_jsonl != source_jsonl:
        raise SystemExit(
            "frozen source JSONL path disagrees with resolved inference config: "
            f"config={configured_source_jsonl}, frozen={source_jsonl}"
        )
    try:
        frozen_file_identity = validate_frozen_file_identity(
            manifest,
            infer_config_path=args.infer_config,
            source_jsonl_path=source_jsonl,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"frozen checkpoint/config/source identity check failed: {exc}") from exc
    examples = list(load_raw_examples(source_jsonl))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    selected_ids = {value.strip() for value in str(args.image_ids).split(",") if value.strip()}
    all_ids = set(frozen["shards"])
    if selected_ids - all_ids:
        raise SystemExit(f"--image-ids outside frozen pool: {sorted(selected_ids - all_ids)}")
    image_ids = sorted(selected_ids or all_ids, key=_image_sort)
    outputs: list[dict[str, Any]] = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    with open_backend_session(frontend.launch) as session:
        for image_id in image_ids:
            source_image = frozen["shards"][image_id]["image"]
            example = _select_example(examples, image_id)
            request, plan, prompt_meta = _build_request(config, frontend, example)
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
            one_native = _single_native_inputs(native_inputs)
            observed_runtime_identity = {
                "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
                "executed_media_sha256": media_sha[0],
                "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]),
            }
            identity_check = compare_execution_identity(
                observed_prompt=prompt_meta,
                observed_runtime=observed_runtime_identity,
                discovery_prompt=source_image.get("prompt", {}),
                discovery_runtime=source_image.get("runtime", {}),
            )
            if not identity_check["passed"]:
                raise SystemExit(
                    f"discovery/runtime identity mismatch before generation for image {image_id}: {identity_check['checks']}"
                )
            ledger = [dict(row) for row in source_image.get("entity_ledger", []) if isinstance(row, Mapping)] or build_positive_entity_ledger(example)
            fresh = _generate_extended_root(
                session=session,
                native_inputs=one_native,
                tokenizer=session._tokenizer,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                malformed_limit=int(budget.get("malformed_row_limit", 2)),
                temperature=float(contract.get("temperature", 0.4)),
            )
            covered: set[str] = set()
            for row in fresh["rows"]:
                covered_before = sorted(covered)
                _annotate_owner_matches(
                    row,
                    entity_ledger=ledger,
                    image_width=int(plan.decoded_width),
                    image_height=int(plan.decoded_height),
                    covered_entity_ids=covered_before,
                )
                row["covered_owner_ids_before_row"] = covered_before
                covered_after, coverage_receipt = extend_covered_set_if_unambiguous(covered, row)
                row["coverage_receipt"] = coverage_receipt
                if coverage_receipt.get("coverage_updated"):
                    covered = set(covered_after)
                row["covered_owner_ids_after_row"] = sorted(covered)
            fresh["final_covered_owner_ids"] = sorted(covered)
            root = _find_root_trajectory(source_image)
            parity = first_eight_parity(fresh["rows"], root)
            target = frozen["target_images"].get(image_id)
            classification = None
            if target is not None:
                classification = classify_target_from_extended_rows(fresh["rows"], target_owner_id=str(target["target_owner_id"])) if parity.get("passed") else {"label": "unresolved", "refusal_reason": "first_eight_root_token_mismatch"}
            outputs.append({
                "image_id": image_id,
                "prompt": prompt_meta,
                "source_discovery_shard": {
                    "path": frozen["shards"][image_id]["path"],
                    "sha256": frozen["shards"][image_id]["sha256"],
                },
                "provisional_target": target,
                "negative_discovery_control": image_id in frozen["negative_control_image_ids"],
                "first_eight_root_parity": parity,
                "extended_root_greedy": fresh,
                "target_classification": classification,
                "entity_ledger": ledger,
                "runtime": observed_runtime_identity,
                "identity_check": identity_check,
            })
        model_receipt = session.receipt.to_artifact_dict()
    source_identity = git_execution_identity(Path(__file__).resolve().parents[2])
    # ``git_execution_identity`` predates this runner and labels its own
    # helper file as ``runner_sha256``.  Remove that ambiguous field and keep
    # explicit identities for this runner and the reused helper.
    source_identity.pop("runner_sha256", None)
    source_identity["stage_one_runner_sha256"] = sha256_file(Path(__file__).resolve())
    source_identity["reused_local_branch_helper_sha256"] = sha256_file(Path(__file__).with_name("run_local_branch_causal_value.py"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "sampled_history_target_reachability",
        "phase": "stage_one_extended_root_greedy_screen",
        "source_identity": source_identity,
        "frozen_inputs": {
            "manifest": frozen["manifest_path"],
            "manifest_sha256": frozen["manifest_sha256"],
            "provisional_targets": frozen["provisional_targets_path"],
            "provisional_targets_sha256": frozen["provisional_targets_sha256"],
            "discovery_shard_sha256": {image_id: frozen["shards"][image_id]["sha256"] for image_id in sorted(frozen["shards"], key=_image_sort)},
            "checkpoint_config_source_identity": frozen_file_identity,
        },
        "config": {
            "infer_config": str(args.infer_config.expanduser().resolve()),
            "resolved_config_fingerprint": resolved.fingerprint,
            "device": args.device,
            "physical_batch_size": 1,
            "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "total_generated_token_budget": TOTAL_TOKEN_BUDGET,
            "maximum_complete_rows_for_parity": MAX_ROOT_ROWS,
        },
        "model_identity": model_receipt,
        "images": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
