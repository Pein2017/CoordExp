#!/usr/bin/env python3
"""Measure selected sampled-route owner transfer in clean greedy rollouts.

The selected-route additions are defined by the route analysis and StateBank
assembly receipt, not by the clean treatment outputs:

``added = selected_route_owner_ids - route_analysis_greedy_owner_ids``

Each clean ``gt_vs_pred.jsonl`` arm is then matched with the same-category,
cardinality-first, maximum-IoU matching used by
``compare_clean_rollout_owner_coverage.py``.  This is an annotated-owner
transfer probe only.  Unmatched predictions are counted by the underlying
matching utility but are never labelled hallucinations.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

if __package__ in {None, ""}:  # Allow ``python scripts/research/...py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.compare_clean_rollout_owner_coverage import (  # noqa: E402
    _global_matches,
    _gt_objects,
    _gt_signature,
    _pred_objects,
    _read_jsonl,
)


SCHEMA_VERSION = "selected_route_added_owner_transfer.v2"
IMAGE_ID_RE = re.compile(r"(?:^|/)(\d{12})\.(?:jpg|jpeg|png)$", re.IGNORECASE)


class AnalysisError(ValueError):
    """Raised when the named artifacts cannot support a paired comparison."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=True)


def _read_json(path: str | Path) -> Any:
    resolved = _resolved(path)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON: {resolved}") from exc


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{name} must be an object")
    return value


def _image_id_from_row(row: Mapping[str, Any]) -> str:
    image_path = str(row.get("image_path", ""))
    match = IMAGE_ID_RE.search(image_path)
    if match:
        return str(int(match.group(1)))
    row_id = str(row.get("row_id", row.get("example_id", "")))
    match = re.search(r"(\d{12})$", row_id)
    if match:
        return str(int(match.group(1)))
    raise AnalysisError(f"row {row_id!r} has no COCO image id")


def _owner_id(image_id: str, raw_gt: Mapping[str, Any], index: int) -> str:
    value = raw_gt.get("object_id", raw_gt.get("id"))
    if value is None or isinstance(value, bool):
        raise AnalysisError(f"image {image_id} GT object {index} has no object_id")
    return f"{image_id}:{value}"


def _matched_owner_ids(row: Mapping[str, Any], *, match_iou_threshold: float) -> set[str]:
    image_id = _image_id_from_row(row)
    row_id = str(row.get("row_id", row.get("example_id", image_id)))
    gt = _gt_objects(dict(row), row_id=row_id)
    pred, _ = _pred_objects(dict(row))
    return {
        _owner_id(image_id, _object(row["gt"][gt_index], f"row {row_id} gt[{gt_index}]"), gt_index)
        for gt_index, _, _ in _global_matches(gt, pred, match_iou_threshold)
    }


def _gt_owner_ids(row: Mapping[str, Any]) -> set[str]:
    image_id = _image_id_from_row(row)
    objects = row.get("gt", [])
    if not isinstance(objects, list):
        raise AnalysisError(f"row {row.get('row_id', row.get('example_id'))!r} has malformed GT list")
    return {_owner_id(image_id, _object(item, "GT object"), index) for index, item in enumerate(objects)}


def _read_state_bank_records(path: Path) -> list[dict[str, Any]]:
    """Read StateBank records, whose key is ``event_id`` rather than ``row_id``."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"invalid StateBank JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise AnalysisError(f"StateBank JSONL row at {path}:{line_number} must be an object")
            rows.append(value)
    return rows


def _load_arm(path: str | Path, *, match_iou_threshold: float) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    rows_by_key = _read_jsonl(_resolved(path))
    rows_by_image: dict[str, dict[str, Any]] = {}
    matched_by_image: dict[str, set[str]] = {}
    gt_by_image: dict[str, set[str]] = {}
    for row in rows_by_key.values():
        image_id = _image_id_from_row(row)
        if image_id in rows_by_image:
            raise AnalysisError(f"{path} contains duplicate image id {image_id}")
        rows_by_image[image_id] = row
        matched_by_image[image_id] = _matched_owner_ids(row, match_iou_threshold=match_iou_threshold)
        gt_by_image[image_id] = _gt_owner_ids(row)
    return rows_by_image, matched_by_image, gt_by_image


def _state_bank_image_ids(
    records_path: Path, manifest: Mapping[str, Any]
) -> tuple[set[str], int, set[str], dict[str, set[str]]]:
    """Return StateBank image scope and stable positive-event owner IDs.

    StateBank records are one positive-row event each.  The candidate's
    ``physical_owner_id`` is the stable owner identity; candidate labels or
    event IDs are intentionally not used for the owner partition.  Keep the
    all-candidate owner union for the existing scope receipt, and separately
    retain the per-image positive-event owner IDs used to split route-added
    owners into directly targeted and non-directly targeted groups.
    """

    rows = _read_state_bank_records(records_path)
    expected_count = manifest.get("record_count")
    if isinstance(expected_count, int) and len(rows) != expected_count:
        raise AnalysisError(f"StateBank record count mismatch: manifest={expected_count}, actual={len(rows)}")
    image_ids: set[str] = set()
    event_owner_ids: set[str] = set()
    positive_event_owner_ids_by_image: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        image = _object(row.get("image"), f"StateBank record {index}.image")
        raw_image_id = image.get("image_id")
        if raw_image_id is None:
            raise AnalysisError(f"StateBank record {index}.image lacks image_id")
        image_id = str(int(raw_image_id))
        image_ids.add(image_id)
        positive_event_owner_ids_by_image.setdefault(image_id, set())
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            raise AnalysisError(f"StateBank record {index}.candidates must be a list")
        for candidate in candidates:
            if isinstance(candidate, Mapping) and candidate.get("physical_owner_id") is not None:
                owner_id = str(candidate["physical_owner_id"])
                event_owner_ids.add(owner_id)
                # Older synthetic fixtures omit role; StateBank records are
                # positive-row records by construction.  If a role is
                # present, only an explicit positive candidate is a direct
                # positive-row event target.
                if candidate.get("role") in (None, "positive"):
                    positive_event_owner_ids_by_image[image_id].add(owner_id)
    return image_ids, len(rows), event_owner_ids, positive_event_owner_ids_by_image


def _route_scope(
    assembly: Mapping[str, Any], state_bank_image_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    raw_selection = assembly.get("route_selection")
    if not isinstance(raw_selection, list):
        raise AnalysisError("assembly receipt route_selection must be a list")
    by_image: dict[str, Mapping[str, Any]] = {}
    for raw in raw_selection:
        item = _object(raw, "route_selection item")
        image_id = str(int(item.get("image_id")))
        if image_id in by_image:
            raise AnalysisError(f"route_selection contains duplicate image id {image_id}")
        by_image[image_id] = item

    selected_routes: dict[str, dict[str, Any]] = {}
    added_by_image: dict[str, set[str]] = {}
    ordinary_by_image: dict[str, set[str]] = {}
    for image_id in sorted(state_bank_image_ids, key=int):
        item = by_image.get(image_id)
        if item is None or not bool(item.get("admissible")):
            raise AnalysisError(f"StateBank image {image_id} lacks an admissible selected route")
        route_id = item.get("selected_route_id")
        if not isinstance(route_id, str) or not route_id:
            raise AnalysisError(f"StateBank image {image_id} lacks selected_route_id")
        candidates = _object(item.get("candidate_routes"), f"route_selection[{image_id}].candidate_routes")
        route = _object(candidates.get(route_id), f"route_selection[{image_id}].candidate_routes[{route_id}]")
        selected_owner_ids = {str(value) for value in route.get("owner_ids", [])}
        greedy_owner_ids = {str(value) for value in route.get("greedy_owner_ids", [])}
        added_owner_ids = {str(value) for value in route.get("added_owner_ids", [])}
        if added_owner_ids != selected_owner_ids - greedy_owner_ids:
            raise AnalysisError(f"selected route {image_id}/{route_id} has inconsistent added_owner_ids")
        ordinary_owner_ids = selected_owner_ids - added_owner_ids
        selected_routes[image_id] = {
            "route_id": route_id,
            "seed": route.get("seed", item.get("selected_seed")),
            "selected_owner_ids": sorted(selected_owner_ids),
            "greedy_owner_ids": sorted(greedy_owner_ids),
            "added_owner_ids": sorted(added_owner_ids),
            "last_added_owner_row_index": route.get("last_added_owner_row_index"),
        }
        added_by_image[image_id] = added_owner_ids
        ordinary_by_image[image_id] = ordinary_owner_ids

    admissible = {
        image_id: item
        for image_id, item in by_image.items()
        if bool(item.get("admissible"))
    }
    all_admitted_added = {
        str(owner)
        for item in admissible.values()
        for owner in _object(item.get("candidate_routes"), "candidate_routes").get(str(item.get("selected_route_id")), {}).get("added_owner_ids", [])
    }
    scope_meta = {
        "route_analysis_image_count": len(by_image),
        "route_analysis_admissible_image_count": len(admissible),
        "state_bank_image_count": len(state_bank_image_ids),
        "admissible_images_without_state_bank_events": sorted(set(admissible) - state_bank_image_ids, key=int),
        "state_bank_images_not_in_admissible_routes": sorted(state_bank_image_ids - set(admissible), key=int),
        "route_analysis_admitted_added_owner_count": len(all_admitted_added),
        "state_bank_scope_added_owner_count": sum(len(value) for value in added_by_image.values()),
    }
    return selected_routes, added_by_image, ordinary_by_image, scope_meta


def _arm_summary(
    *,
    label: str,
    matched_by_image: Mapping[str, set[str]],
    owner_sets: Mapping[str, set[str]],
    source_matched: Mapping[str, set[str]] | None,
) -> dict[str, Any]:
    total = sum(len(values) for values in owner_sets.values())
    matched_by_scope = {image_id: owner_sets[image_id] & matched_by_image[image_id] for image_id in owner_sets}
    matched_count = sum(len(values) for values in matched_by_scope.values())
    image_count_with_match = sum(bool(values) for values in matched_by_scope.values())
    result: dict[str, Any] = {
        "owner_count": total,
        "matched_owner_count": matched_count,
        "matched_owner_rate": matched_count / total if total else 0.0,
        "image_count": len(owner_sets),
        "image_count_with_match": image_count_with_match,
        "matched_owner_ids": sorted(owner for values in matched_by_scope.values() for owner in values),
        "matched_owner_counts_by_image": {
            image_id: len(matched_by_scope[image_id]) for image_id in sorted(matched_by_scope, key=int)
        },
    }
    if source_matched is None:
        result["relative_to_source"] = None
    else:
        gained_by_image = {
            image_id: (owner_sets[image_id] & matched_by_image[image_id]) - source_matched[image_id]
            for image_id in owner_sets
        }
        lost_by_image = {
            image_id: source_matched[image_id] - (owner_sets[image_id] & matched_by_image[image_id])
            for image_id in owner_sets
        }
        gained = sorted(owner for values in gained_by_image.values() for owner in values)
        lost = sorted(owner for values in lost_by_image.values() for owner in values)
        delta_by_image = {
            image_id: len(gained_by_image[image_id]) - len(lost_by_image[image_id])
            for image_id in owner_sets
        }
        random_generator = random.Random(20260721)
        image_ids = sorted(delta_by_image, key=int)
        bootstrap_totals = sorted(
            sum(delta_by_image[image_ids[random_generator.randrange(len(image_ids))]] for _ in image_ids)
            for _ in range(20_000)
        )
        result["relative_to_source"] = {
            "convention": "treatment_minus_source_on_the_same_selected_owner_set",
            "gained_owner_count": len(gained),
            "lost_owner_count": len(lost),
            "net_owner_count": len(gained) - len(lost),
            "gained_owner_ids": gained,
            "lost_owner_ids": lost,
            "gained_owner_counts_by_image": {
                image_id: len(gained_by_image[image_id]) for image_id in sorted(gained_by_image, key=int)
            },
            "lost_owner_counts_by_image": {
                image_id: len(lost_by_image[image_id]) for image_id in sorted(lost_by_image, key=int)
            },
            "paired_image_cluster_bootstrap": {
                "method": "percentile bootstrap of the total matched-owner count change with image as the resampling cluster",
                "seed": 20260721,
                "replicate_count": 20_000,
                "confidence_level": 0.95,
                "lower_total_owner_change": bootstrap_totals[500],
                "upper_total_owner_change": bootstrap_totals[19_499],
                "positive_image_count": sum(value > 0 for value in delta_by_image.values()),
                "negative_image_count": sum(value < 0 for value in delta_by_image.values()),
                "zero_image_count": sum(value == 0 for value in delta_by_image.values()),
            },
        }
    return result


def _clean_owner_coverage(
    *,
    matched_by_image: Mapping[str, set[str]],
    gt_by_image: Mapping[str, set[str]],
    admitted_image_ids: set[str],
) -> dict[str, Any]:
    """Summarize all annotated-owner matches by StateBank admission scope."""

    all_image_ids = set(gt_by_image)
    partitions = {
        "admitted_state_bank_118": set(admitted_image_ids),
        "non_admitted_138": all_image_ids - set(admitted_image_ids),
    }
    result: dict[str, Any] = {}
    for name, image_ids in partitions.items():
        gt_count = sum(len(gt_by_image[image_id]) for image_id in image_ids)
        matched_count = sum(len(matched_by_image[image_id]) for image_id in image_ids)
        result[name] = {
            "image_count": len(image_ids),
            "gt_owner_count": gt_count,
            "matched_owner_count": matched_count,
            "owner_coverage": matched_count / gt_count if gt_count else 0.0,
        }
    return result


def _prediction_signature(value: Any) -> tuple[str, tuple[int, ...]] | None:
    if not isinstance(value, Mapping):
        return None
    category = str(value.get("description", value.get("desc", value.get("label", "")))).strip().lower()
    bins = value.get("coord_bins")
    if not category or not isinstance(bins, list) or len(bins) != 4:
        return None
    try:
        return category, tuple(int(item) for item in bins)
    except (TypeError, ValueError):
        return None


def _source_artifact_alignment(
    *,
    route_greedy_path: Path,
    clean_source_rows: Mapping[str, Mapping[str, Any]],
    clean_source_run_manifest_path: Path,
) -> dict[str, Any]:
    """Compare the route-analysis greedy artifact with the clean Source arm.

    The selected additions are defined against the former, while Source match
    counts are recomputed from the latter.  This receipt makes that scope
    boundary explicit instead of silently treating the two artifacts as one
    deterministic decode.
    """

    route_doc = _object(_read_json(route_greedy_path), "route-analysis greedy rollout")
    route_rollouts_raw = route_doc.get("rollouts")
    if not isinstance(route_rollouts_raw, list):
        raise AnalysisError("route-analysis greedy artifact rollouts must be a list")
    route_rollouts: dict[str, Mapping[str, Any]] = {}
    for raw in route_rollouts_raw:
        rollout = _object(raw, "route-analysis greedy rollout row")
        image_value = rollout.get("image_id")
        if image_value is None:
            raise AnalysisError("route-analysis greedy rollout row lacks image_id")
        image_id = str(int(image_value))
        if image_id in route_rollouts:
            raise AnalysisError(f"route-analysis greedy artifact duplicates image {image_id}")
        route_rollouts[image_id] = rollout
    clean_manifest = _object(_read_json(clean_source_run_manifest_path), "clean Source run manifest")
    route_config = _object(route_doc.get("config"), "route-analysis greedy config")
    route_identity = _object(route_doc.get("model_identity"), "route-analysis greedy model_identity")
    clean_generation = _object(clean_manifest.get("generation_policy"), "clean Source generation_policy")
    clean_backend_session = _object(clean_manifest.get("backend_session"), "clean Source backend_session")
    clean_effective = _object(clean_backend_session.get("effective_settings"), "clean Source effective settings")

    route_generation = {
        "decode_mode": route_config.get("decode_mode"),
        "do_sample": False,
        "max_new_tokens": route_config.get("max_new_tokens"),
        "temperature": route_config.get("temperature"),
        "top_p": route_config.get("top_p"),
        "repetition_penalty": route_config.get("repetition_penalty"),
    }
    clean_generation_normalized = {
        "decode_mode": "greedy" if clean_generation.get("do_sample") is False else "sample",
        "do_sample": clean_generation.get("do_sample"),
        "max_new_tokens": clean_generation.get("max_new_tokens"),
        "temperature": clean_generation.get("temperature"),
        "top_p": clean_generation.get("top_p"),
        "repetition_penalty": clean_generation.get("repetition_penalty"),
    }
    route_effective = _object(route_identity.get("effective_settings"), "route-analysis effective settings")
    route_model = _object(route_identity.get("model_identity"), "route-analysis nested model identity")
    clean_model = _object(clean_manifest.get("model_identity"), "clean Source model identity")

    def artifact_paths(model: Mapping[str, Any]) -> dict[str, Any]:
        adapter = _object(model.get("adapter"), "model adapter")
        base = _object(model.get("base"), "model base")
        embedding = _object(model.get("embedding_delta"), "model embedding_delta")
        embedding_identity = _object(embedding.get("identity"), "embedding identity")
        return {
            "base_model_path": base.get("path"),
            "adapter_path": adapter.get("adapter_path"),
            "embedding_delta_path": embedding_identity.get("delta_path"),
        }

    route_paths = artifact_paths(route_model)
    clean_paths = artifact_paths(clean_model)

    def stable_effective_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
        """Drop run-performance telemetry before comparing decode settings."""

        return {key: value for key, value in settings.items() if key != "performance"}

    stable_route_effective = stable_effective_settings(route_effective)
    stable_clean_effective = stable_effective_settings(clean_effective)
    exact_prediction_rows = 0
    category_sequence_rows = 0
    compared_rows = 0
    route_prompt_metadata = route_doc.get("prompt_metadata")
    route_prompt_metadata = route_prompt_metadata if isinstance(route_prompt_metadata, Mapping) else {}
    clean_prompt_trace = clean_manifest.get("prompt_trace")
    clean_prompt_trace_by_row = {
        str(item.get("row_id")): item
        for item in clean_prompt_trace
        if isinstance(item, Mapping) and item.get("row_id") is not None
    } if isinstance(clean_prompt_trace, list) else {}
    compared_prompt_rows = 0
    matching_prompt_token_sha_rows = 0
    matching_image_path_rows = 0
    for image_id, route_row in route_rollouts.items():
        clean_row = clean_source_rows.get(image_id)
        if clean_row is None:
            continue
        example_id = str(route_row.get("example_id", clean_row.get("row_id", "")))
        route_prompt = route_prompt_metadata.get(example_id)
        clean_prompt = clean_prompt_trace_by_row.get(example_id)
        if isinstance(route_prompt, Mapping) and isinstance(clean_prompt, Mapping):
            compared_prompt_rows += 1
            route_prompt_sha = route_prompt.get("prompt_token_ids_sha256")
            clean_prompt_shas = {
                clean_prompt.get("backend_executed_prompt_token_ids_sha256"),
                clean_prompt.get("expected_executed_prompt_token_ids_sha256"),
            }
            if route_prompt_sha is not None and route_prompt_sha in clean_prompt_shas:
                matching_prompt_token_sha_rows += 1
            if route_prompt.get("image_path") == clean_row.get("image_path"):
                matching_image_path_rows += 1
        route_prediction_doc = _object(route_row.get("predictions"), f"route rollout {image_id}.predictions")
        route_predictions = route_prediction_doc.get("predictions", [])
        clean_predictions = clean_row.get("pred", [])
        if not isinstance(route_predictions, list) or not isinstance(clean_predictions, list):
            continue
        route_signature = [_prediction_signature(item) for item in route_predictions]
        clean_signature = [_prediction_signature(item) for item in clean_predictions]
        compared_rows += 1
        if [item[0] if item is not None else None for item in route_signature] == [item[0] if item is not None else None for item in clean_signature]:
            category_sequence_rows += 1
        if route_signature == clean_signature:
            exact_prediction_rows += 1

    route_resolved_fingerprint = route_config.get("resolved_fingerprint")
    clean_resolved_fingerprint = _object(clean_manifest.get("resolved_config_fingerprints"), "clean resolved config fingerprints").get("infer_config")
    return {
        "route_analysis_greedy_artifact": {
            "schema_version": route_doc.get("schema_version"),
            "rollout_count": route_doc.get("rollout_count"),
            "config": route_generation,
            "generation_config_fingerprint": route_identity.get("generation_config_fingerprint"),
            "resolved_config_fingerprint": route_resolved_fingerprint,
            "effective_settings": route_effective,
            "model_artifact_paths": route_paths,
        },
        "clean_source_run_manifest": {
            "backend": clean_manifest.get("backend"),
            "backend_mode": clean_manifest.get("backend_mode"),
            "generation_policy": clean_generation_normalized,
            "generation_config_fingerprint": clean_manifest.get("generation_config_fingerprint"),
            "resolved_config_fingerprint": clean_resolved_fingerprint,
            "effective_settings": clean_effective,
            "model_artifact_paths": clean_paths,
        },
        "identity_comparison": {
            "generation_policy_equal": route_generation == clean_generation_normalized,
            "generation_config_fingerprint_equal": route_identity.get("generation_config_fingerprint") == clean_manifest.get("generation_config_fingerprint"),
            "effective_backend_settings_equal": route_effective == clean_effective,
            "stable_effective_backend_settings_equal": stable_route_effective == stable_clean_effective,
            "model_artifact_paths_equal": route_paths == clean_paths,
            "resolved_config_fingerprint_equal": route_resolved_fingerprint == clean_resolved_fingerprint,
        },
        "decoded_output_comparison": {
            "compared_image_count": compared_rows,
            "exact_parsed_prediction_rows": exact_prediction_rows,
            "different_parsed_prediction_rows": compared_rows - exact_prediction_rows,
            "same_category_sequence_rows": category_sequence_rows,
            "different_category_sequence_rows": compared_rows - category_sequence_rows,
        },
        "prompt_input_alignment": {
            "compared_image_count": compared_prompt_rows,
            "matching_executed_prompt_token_sha_rows": matching_prompt_token_sha_rows,
            "matching_image_path_rows": matching_image_path_rows,
        },
        "scope_caveat": (
            "The route-analysis additions are defined against route-analysis greedy owner IDs, "
            "whereas Source matches are recomputed from the clean Source gt_vs_pred artifact. "
            "The model artifact paths, generation policy, generation-config fingerprint, and "
            "stable backend decode settings match; where prompt metadata is present, the executed "
            "prompt-token SHA and image path also match. The clean manifest adds performance telemetry "
            "to effective_settings, so full settings-object equality is false; the resolved config "
            "fingerprints still differ and the parsed greedy outputs are not identical. Therefore "
            "Source matching 93 of these labeled additions is a cross-artifact alignment caveat, "
            "not evidence that the treatment gained those owners."
        ),
    }


def analyze_selected_route_added_owner_transfer(
    *,
    assembly_receipt_path: str | Path,
    state_bank_manifest_path: str | Path,
    state_bank_records_path: str | Path,
    source_panel_path: str | Path,
    route_analysis_greedy_rollout_path: str | Path,
    clean_rollout_paths: Mapping[str, str | Path],
    match_iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Return a hash-bound selected-route transfer receipt."""

    if not (0.0 <= match_iou_threshold <= 1.0):
        raise AnalysisError("match_iou_threshold must be between 0 and 1")
    required_arms = ("source", "step10", "step15", "step16")
    if tuple(clean_rollout_paths) != required_arms:
        missing = [name for name in required_arms if name not in clean_rollout_paths]
        extra = [name for name in clean_rollout_paths if name not in required_arms]
        raise AnalysisError(f"clean_rollout_paths must contain source, step10, step15, step16; missing={missing}, extra={extra}")

    assembly_path = _resolved(assembly_receipt_path)
    manifest_path = _resolved(state_bank_manifest_path)
    records_path = _resolved(state_bank_records_path)
    source_panel = _resolved(source_panel_path)
    route_greedy = _resolved(route_analysis_greedy_rollout_path)
    assembly = _object(_read_json(assembly_path), "assembly receipt")
    manifest = _object(_read_json(manifest_path), "StateBank manifest")
    records_hash = _sha256_file(records_path)
    declared_records_hash = manifest.get("records_sha256")
    if declared_records_hash and str(declared_records_hash) != records_hash:
        raise AnalysisError(f"StateBank records hash mismatch: manifest={declared_records_hash}, actual={records_hash}")
    (
        state_bank_image_ids,
        state_bank_record_count,
        state_bank_event_owner_ids,
        state_bank_positive_event_owner_ids_by_image,
    ) = _state_bank_image_ids(records_path, manifest)
    selected_routes, added_by_image, ordinary_by_image, scope_meta = _route_scope(assembly, state_bank_image_ids)

    direct_positive_event_target_added_by_image = {
        image_id: added_by_image[image_id]
        & state_bank_positive_event_owner_ids_by_image.get(image_id, set())
        for image_id in state_bank_image_ids
    }
    non_direct_positive_event_target_added_by_image = {
        image_id: added_by_image[image_id] - direct_positive_event_target_added_by_image[image_id]
        for image_id in state_bank_image_ids
    }

    source_artifacts = assembly.get("source_artifacts", [])
    declared_panel_hash = None
    if isinstance(source_artifacts, list):
        for item in source_artifacts:
            if isinstance(item, Mapping) and item.get("artifact_id") == "trajectory-analysis":
                declared_panel_hash = str(item.get("sha256"))
                break
    source_panel_hash = _sha256_file(source_panel)
    if declared_panel_hash is not None and declared_panel_hash != source_panel_hash:
        raise AnalysisError(f"source panel hash mismatch: assembly={declared_panel_hash}, actual={source_panel_hash}")

    arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
    arm_matches: dict[str, dict[str, set[str]]] = {}
    arm_gt: dict[str, dict[str, set[str]]] = {}
    for label in required_arms:
        rows, matches, gt = _load_arm(clean_rollout_paths[label], match_iou_threshold=match_iou_threshold)
        arm_rows[label] = rows
        arm_matches[label] = matches
        arm_gt[label] = gt
    scope_ids = set(state_bank_image_ids)
    for label in required_arms:
        if set(arm_rows[label]) != set(arm_rows["source"]):
            raise AnalysisError(f"{label} and source image row sets differ")
        for image_id in sorted(scope_ids, key=int):
            if image_id not in arm_rows[label]:
                raise AnalysisError(f"{label} is missing StateBank image {image_id}")
        if label != "source":
            for image_id in sorted(arm_rows[label], key=int):
                if _gt_signature(arm_rows[label][image_id], row_id=str(arm_rows[label][image_id].get("row_id"))) != _gt_signature(arm_rows["source"][image_id], row_id=str(arm_rows["source"][image_id].get("row_id"))):
                    raise AnalysisError(f"GT mismatch between source and {label} for image {image_id}")

    source_scope_matched_added = {
        image_id: added_by_image[image_id] & arm_matches["source"][image_id] for image_id in scope_ids
    }
    source_scope_matched_ordinary = {
        image_id: ordinary_by_image[image_id] & arm_matches["source"][image_id] for image_id in scope_ids
    }
    arms: dict[str, Any] = {}
    for label in required_arms:
        arms[label] = {
            "added": _arm_summary(
                label=label,
                matched_by_image=arm_matches[label],
                owner_sets=added_by_image,
                source_matched=None if label == "source" else source_scope_matched_added,
            ),
            "ordinary_non_added": _arm_summary(
                label=label,
                matched_by_image=arm_matches[label],
                owner_sets=ordinary_by_image,
                source_matched=None if label == "source" else source_scope_matched_ordinary,
            ),
            "added_direct_positive_event_target": _arm_summary(
                label=label,
                matched_by_image=arm_matches[label],
                owner_sets=direct_positive_event_target_added_by_image,
                source_matched=(
                    None
                    if label == "source"
                    else {
                        image_id: direct_positive_event_target_added_by_image[image_id]
                        & arm_matches["source"][image_id]
                        for image_id in state_bank_image_ids
                    }
                ),
            ),
            "added_non_direct_positive_event_target": _arm_summary(
                label=label,
                matched_by_image=arm_matches[label],
                owner_sets=non_direct_positive_event_target_added_by_image,
                source_matched=(
                    None
                    if label == "source"
                    else {
                        image_id: non_direct_positive_event_target_added_by_image[image_id]
                        & arm_matches["source"][image_id]
                        for image_id in state_bank_image_ids
                    }
                ),
            ),
        }

    per_image: dict[str, Any] = {}
    for image_id in sorted(scope_ids, key=int):
        per_image[image_id] = {
            **selected_routes[image_id],
            "added_owner_ids": sorted(added_by_image[image_id]),
            "added_direct_positive_event_target_owner_ids": sorted(
                direct_positive_event_target_added_by_image[image_id]
            ),
            "added_non_direct_positive_event_target_owner_ids": sorted(
                non_direct_positive_event_target_added_by_image[image_id]
            ),
            "ordinary_non_added_owner_ids": sorted(ordinary_by_image[image_id]),
            "arms": {
                label: {
                    "matched_added_owner_ids": sorted(added_by_image[image_id] & arm_matches[label][image_id]),
                    "matched_added_direct_positive_event_target_owner_ids": sorted(
                        direct_positive_event_target_added_by_image[image_id] & arm_matches[label][image_id]
                    ),
                    "matched_added_non_direct_positive_event_target_owner_ids": sorted(
                        non_direct_positive_event_target_added_by_image[image_id] & arm_matches[label][image_id]
                    ),
                    "matched_ordinary_owner_ids": sorted(ordinary_by_image[image_id] & arm_matches[label][image_id]),
                }
                for label in required_arms
            },
        }
        source_added = set(per_image[image_id]["arms"]["source"]["matched_added_owner_ids"])
        source_added_direct = set(
            per_image[image_id]["arms"]["source"]["matched_added_direct_positive_event_target_owner_ids"]
        )
        source_added_non_direct = set(
            per_image[image_id]["arms"]["source"]["matched_added_non_direct_positive_event_target_owner_ids"]
        )
        source_ordinary = set(per_image[image_id]["arms"]["source"]["matched_ordinary_owner_ids"])
        for label in ("step10", "step15", "step16"):
            treatment_added = set(per_image[image_id]["arms"][label]["matched_added_owner_ids"])
            treatment_added_direct = set(
                per_image[image_id]["arms"][label]["matched_added_direct_positive_event_target_owner_ids"]
            )
            treatment_added_non_direct = set(
                per_image[image_id]["arms"][label]["matched_added_non_direct_positive_event_target_owner_ids"]
            )
            per_image[image_id]["arms"][label]["gained_added_owner_ids"] = sorted(treatment_added - source_added)
            per_image[image_id]["arms"][label]["lost_added_owner_ids"] = sorted(source_added - treatment_added)
            per_image[image_id]["arms"][label]["gained_added_direct_positive_event_target_owner_ids"] = sorted(
                treatment_added_direct - source_added_direct
            )
            per_image[image_id]["arms"][label]["lost_added_direct_positive_event_target_owner_ids"] = sorted(
                source_added_direct - treatment_added_direct
            )
            per_image[image_id]["arms"][label]["gained_added_non_direct_positive_event_target_owner_ids"] = sorted(
                treatment_added_non_direct - source_added_non_direct
            )
            per_image[image_id]["arms"][label]["lost_added_non_direct_positive_event_target_owner_ids"] = sorted(
                source_added_non_direct - treatment_added_non_direct
            )
            per_image[image_id]["arms"][label]["gained_ordinary_owner_ids"] = sorted(
                set(per_image[image_id]["arms"][label]["matched_ordinary_owner_ids"]) - source_ordinary
            )
            per_image[image_id]["arms"][label]["lost_ordinary_owner_ids"] = sorted(
                source_ordinary - set(per_image[image_id]["arms"][label]["matched_ordinary_owner_ids"])
            )

    clean_source_run_manifest = _resolved(Path(clean_rollout_paths["source"]).parent / "run_manifest.json")
    input_paths: dict[str, Path] = {
        "assembly_receipt": assembly_path,
        "state_bank_manifest": manifest_path,
        "state_bank_records": records_path,
        "source_panel": source_panel,
        "route_analysis_greedy_rollout": route_greedy,
        "clean_source_run_manifest": clean_source_run_manifest,
    }
    input_paths.update({f"clean_rollout_{label}": _resolved(path) for label, path in clean_rollout_paths.items()})
    inputs = {
        key: {"path": str(path), "sha256": _sha256_file(path)} for key, path in sorted(input_paths.items())
    }
    inputs["source_panel"]["declared_sha256_in_assembly"] = declared_panel_hash
    inputs["state_bank_records"]["declared_sha256_in_manifest"] = declared_records_hash

    source_gt_scope = set().union(*(arm_gt["source"][image_id] for image_id in scope_ids))
    selected_owner_scope = set().union(*(added_by_image[image_id] | ordinary_by_image[image_id] for image_id in scope_ids))
    missing_gt_owners = sorted(selected_owner_scope - source_gt_scope)
    if missing_gt_owners:
        raise AnalysisError(f"selected route owner IDs missing from clean GT: {missing_gt_owners[:5]}")

    source_alignment = _source_artifact_alignment(
        route_greedy_path=route_greedy,
        clean_source_rows=arm_rows["source"],
        clean_source_run_manifest_path=clean_source_run_manifest,
    )

    selected_route_owner_union_by_image = {
        image_id: added_by_image[image_id] | ordinary_by_image[image_id] for image_id in scope_ids
    }
    selected_route_owner_union = set().union(*selected_route_owner_union_by_image.values())
    route_universe = {
        "selected_route_owner_union_count": len(selected_route_owner_union),
        "selected_route_added_owner_union_count": sum(len(values) for values in added_by_image.values()),
        "selected_route_added_direct_positive_event_target_owner_union_count": sum(
            len(values) for values in direct_positive_event_target_added_by_image.values()
        ),
        "selected_route_added_non_direct_positive_event_target_owner_union_count": sum(
            len(values) for values in non_direct_positive_event_target_added_by_image.values()
        ),
        "selected_route_ordinary_non_added_owner_union_count": sum(len(values) for values in ordinary_by_image.values()),
        "route_analysis_greedy_owner_union_count": len(set().union(*(set(values) for values in ordinary_by_image.values()))),
        "matched_selected_route_owner_union_count_by_arm": {
            label: sum(
                len(selected_route_owner_union_by_image[image_id] & arm_matches[label][image_id])
                for image_id in scope_ids
            )
            for label in required_arms
        },
        "matched_selected_route_owner_union_rate_by_arm": {
            label: (
                sum(
                    len(selected_route_owner_union_by_image[image_id] & arm_matches[label][image_id])
                    for image_id in scope_ids
                )
                / len(selected_route_owner_union)
                if selected_route_owner_union
                else 0.0
            )
            for label in required_arms
        },
        "matched_selected_route_added_direct_positive_event_target_owner_count_by_arm": {
            label: sum(
                len(direct_positive_event_target_added_by_image[image_id] & arm_matches[label][image_id])
                for image_id in state_bank_image_ids
            )
            for label in required_arms
        },
        "matched_selected_route_added_non_direct_positive_event_target_owner_count_by_arm": {
            label: sum(
                len(non_direct_positive_event_target_added_by_image[image_id] & arm_matches[label][image_id])
                for image_id in state_bank_image_ids
            )
            for label in required_arms
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "match_iou_threshold": float(match_iou_threshold),
            "matching": "same-category, cardinality-first, maximum-IoU global assignment from compare_clean_rollout_owner_coverage.py",
            "added_owner_definition": "selected_route_owner_ids minus route_analysis_greedy_owner_ids",
            "direct_positive_event_target_definition": (
                "selected route-added owner IDs whose stable physical_owner_id appears in a positive StateBank candidate"
            ),
            "non_direct_positive_event_target_definition": (
                "selected route-added owner IDs absent from the positive StateBank candidate physical_owner_id set"
            ),
            "ordinary_non_added_definition": "selected_route_owner_ids minus selected_route_added_owner_ids",
            "scope": "StateBank final image set; four admissible routes without StateBank events are excluded from the primary 118-image comparison",
            "unmatched_predictions_are_not_hallucinations": True,
            "relative_delta_convention": "treatment minus source on the same selected owner set",
        },
        "inputs": inputs,
        "scope": {
            **scope_meta,
            "state_bank_record_count": state_bank_record_count,
            "state_bank_event_owner_id_count": len(state_bank_event_owner_ids),
            "state_bank_positive_event_owner_id_count": sum(
                len(values) for values in state_bank_positive_event_owner_ids_by_image.values()
            ),
            "selected_route_added_owner_count": sum(len(values) for values in added_by_image.values()),
            "selected_route_added_direct_positive_event_target_owner_count": sum(
                len(values) for values in direct_positive_event_target_added_by_image.values()
            ),
            "selected_route_added_non_direct_positive_event_target_owner_count": sum(
                len(values) for values in non_direct_positive_event_target_added_by_image.values()
            ),
            "selected_route_ordinary_non_added_owner_count": sum(len(values) for values in ordinary_by_image.values()),
            "selected_route_added_owner_ids_missing_from_clean_gt": missing_gt_owners,
        },
        "arms": arms,
        "all_annotated_owner_coverage_by_admission_scope": {
            label: _clean_owner_coverage(
                matched_by_image=arm_matches[label],
                gt_by_image=arm_gt[label],
                admitted_image_ids=scope_ids,
            )
            for label in required_arms
        },
        "route_owner_universe": route_universe,
        "source_artifact_alignment": source_alignment,
        "per_image": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-receipt", type=Path, required=True)
    parser.add_argument("--state-bank-manifest", type=Path, required=True)
    parser.add_argument("--state-bank-records", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--route-analysis-greedy-rollout", type=Path, required=True)
    parser.add_argument("--source-clean-rollout", type=Path, required=True)
    parser.add_argument("--step10-clean-rollout", type=Path, required=True)
    parser.add_argument("--step15-clean-rollout", type=Path, required=True)
    parser.add_argument("--step16-clean-rollout", type=Path, required=True)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_selected_route_added_owner_transfer(
        assembly_receipt_path=args.assembly_receipt,
        state_bank_manifest_path=args.state_bank_manifest,
        state_bank_records_path=args.state_bank_records,
        source_panel_path=args.source_panel,
        route_analysis_greedy_rollout_path=args.route_analysis_greedy_rollout,
        clean_rollout_paths={
            "source": args.source_clean_rollout,
            "step10": args.step10_clean_rollout,
            "step15": args.step15_clean_rollout,
            "step16": args.step16_clean_rollout,
        },
        match_iou_threshold=args.match_iou_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
