#!/usr/bin/env python3
"""Assemble one image-balanced row-local owner-versus-STOP StateBank.

The positive rows are exact model-produced tokens from the receipt-bound
Source@B16 and sampled@B16 panels.  The harmful action is an explicitly
constructed one-token ``<|im_end|>`` at the same exact prefix.  It is admitted
only when prefix matching proves that the positive row cleanly adds one
previously uncovered trusted physical owner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.fingerprint import sha256_file  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402
from src.rollout_calibration import (  # noqa: E402
    CheckpointIdentity,
    assemble_state_bank,
)
from scripts.research.analyze_individual_trajectory_union_support import (  # noqa: E402
    _parsed_rows,
    load_generation7_annotations,
    match_prefix,
)
from scripts.research.assemble_constant_dose_breadth_state_banks import (  # noqa: E402
    _project_sampled_b16,
)
from scripts.research.assemble_positive_path_imitation_state_bank import (  # noqa: E402
    exact_row_site_types,
    exact_row_slices,
)


SCHEMA_VERSION = "row_local_owner_stop_state_bank_assembler.v1"
IM_END_TOKEN_ID = 151645
BOX_END_TOKEN_ID = 151649
IMAGE_PAD_TOKEN_ID = 151655


class AssemblyError(ValueError):
    """Raised when frozen evidence cannot support the requested bank."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssemblyError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AssemblyError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssemblyError(f"output already exists: {path}")
    path.write_bytes(_canonical(value) + b"\n")


def _batch_inventory(receipt: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise AssemblyError("census receipt omits inputs")
    binding = inputs.get(name)
    if not isinstance(binding, Mapping) or not isinstance(binding.get("batches"), list):
        raise AssemblyError(f"census receipt omits {name}.batches")
    return [dict(item) for item in binding["batches"] if isinstance(item, Mapping)]


def _load_bound_batch(item: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(item.get("path", ""))).expanduser().resolve(strict=True)
    expected = str(item.get("sha256", ""))
    actual = sha256_file(path)
    if actual != expected:
        raise AssemblyError(f"batch SHA-256 mismatch: {path}: {actual} != {expected}")
    return path, _read_json(path)


def _candidate_rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        image_id = str(int(row["image_id"]))
        if image_id in result:
            raise AssemblyError(f"duplicate candidate-pool image: {image_id}")
        result[image_id] = row
    return result


def _annotation_entities(
    annotation: Mapping[str, Any], *, source_path: Path
) -> list[dict[str, Any]]:
    image_id = str(int(annotation["image_id"]))
    entities: list[dict[str, Any]] = []
    for index, raw in enumerate(annotation.get("objects", [])):
        if not isinstance(raw, Mapping):
            raise AssemblyError(f"annotation object is not an object: {image_id}:{index}")
        ann_id = int(raw["coco_ann_id"])
        bbox = []
        for token in raw["bbox_2d"]:
            text = str(token)
            if text.startswith("<|coord_") and text.endswith("|>"):
                text = text[len("<|coord_") : -len("|>")]
            bbox.append(int(text))
        entities.append(
            {
                "entity_id": f"{image_id}:{ann_id}",
                "category": str(raw.get("category_name", raw.get("desc", ""))),
                "entity_trusted": True,
                "geometry_trusted": True,
                "reference_bbox": bbox,
                "review_source": str(source_path),
                "reviewer": "generation-7-annotation-ledger",
                "review_confidence": "high",
                "comment": "Canonical generation-7 physical-owner annotation.",
            }
        )
    return entities


def _image_pad_interval(prompt_ids: Sequence[int]) -> list[int]:
    indices = [index for index, token in enumerate(prompt_ids) if int(token) == IMAGE_PAD_TOKEN_ID]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise AssemblyError("executed prompt must contain one contiguous image-pad interval")
    return [indices[0], indices[-1] + 1]


def _source_projection(raw: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    source = raw.get("source_b16")
    if not isinstance(source, Mapping):
        raise AssemblyError("Source rollout omits source_b16 projection")
    status = str(source.get("status", ""))
    if status not in {"accepted_budget", "accepted_natural_end"}:
        raise AssemblyError(f"Source projection is not accepted: {status}")
    token_ids = [int(item) for item in source.get("projected_token_ids", [])]
    if token_ids_sha256(token_ids) != source.get("projected_token_ids_sha256"):
        raise AssemblyError("Source projection token hash mismatch")
    projected = dict(raw)
    projected["predictions"] = copy.deepcopy(source["projected_parser_evidence"])
    return token_ids, projected


def _sampled_projection(raw: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    token_ids, token_hash, receipt = _project_sampled_b16(
        raw,
        im_end_token_id=IM_END_TOKEN_ID,
        box_end_token_id=BOX_END_TOKEN_ID,
    )
    if token_ids_sha256(token_ids) != token_hash:
        raise AssemblyError("sampled projection token hash mismatch")
    if str(receipt.get("status", "")).startswith("failed"):
        raise AssemblyError(f"sampled B16 projection failed: {receipt.get('status')}")
    return [int(item) for item in token_ids], dict(raw)


def _current_receipt(
    assignment: Mapping[str, Any], prediction_id: str
) -> Mapping[str, Any] | None:
    for item in assignment.get("row_assignment_receipts", []):
        if isinstance(item, Mapping) and str(item.get("prediction_id")) == prediction_id:
            return item
    return None


def _prefix_proofs(pre: Mapping[str, Any], *, source_path: str) -> list[dict[str, Any]]:
    proofs: list[dict[str, Any]] = []
    receipts = sorted(
        (dict(item) for item in pre.get("row_assignment_receipts", []) if isinstance(item, Mapping)),
        key=lambda item: int(item["generated_row_index"]),
    )
    for index, item in enumerate(receipts):
        if int(item["generated_row_index"]) != index or item.get("entity_status") != "verified_owner":
            raise AssemblyError("prefix proof is not fully resolved in generated-row order")
        proofs.append(
            {
                "prefix_object_row_index": index,
                "owner_id": str(item["owner_id"]),
                "review_provenance": {
                    "source": source_path,
                    "reviewer": "row-local-global-owner-matcher",
                    "confidence": "high",
                    "comment": f"Exact prefix row; IoU={float(item['intersection_over_union']):.6f}.",
                },
            }
        )
    return proofs


def _generation_provenance(
    raw: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    checkpoint_id: str,
    prompt_hash: str,
    prefix_hash: str,
) -> dict[str, Any]:
    mode = str(raw.get("decode_mode", config.get("decode_mode", "sampled")))
    if mode == "source_b16":
        mode = "greedy"
    sampled = mode == "sampled"
    seed = raw.get("sample_index", raw.get("seed", 0))
    return {
        "mode": mode,
        "seed": int(0 if seed is None else seed),
        "temperature": float(config.get("temperature", 0.4 if sampled else 0.0)),
        "top_p": float(config.get("top_p", 0.95 if sampled else 1.0)),
        "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
        "checkpoint_id": checkpoint_id,
        "prompt_token_ids_sha256": prompt_hash,
        "prefix_token_ids_sha256": prefix_hash,
    }


def _positive_occurrences(
    *,
    receipt: Mapping[str, Any],
    cohort_ids: set[str],
    owners_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    checkpoint_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    occurrences: list[dict[str, Any]] = []
    counts = defaultdict(int)
    # The v1 StateBank contract identifies the sole producer-declared greedy
    # path as harmful.  Keep exact greedy Source rows for evaluation/control,
    # but do not mislabel them as sampled positives in this treatment bank.
    inventories = (("sampled_manifest_binding", True),)
    for binding_name, sampled in inventories:
        for batch_item in _batch_inventory(receipt, binding_name):
            batch_path, payload = _load_bound_batch(batch_item)
            config = payload.get("config", {})
            if not isinstance(config, Mapping):
                raise AssemblyError(f"batch config is not an object: {batch_path}")
            for raw_value in payload.get("rollouts", []):
                if not isinstance(raw_value, Mapping):
                    raise AssemblyError(f"rollout is not an object: {batch_path}")
                raw = dict(raw_value)
                image_id = str(int(raw["image_id"]))
                if image_id not in cohort_ids:
                    continue
                counts["trajectory_count"] += 1
                try:
                    projected_ids, parsed_input = (
                        _sampled_projection(raw) if sampled else _source_projection(raw)
                    )
                    parsed_rows, parser = _parsed_rows(parsed_input)
                    parsed_rows = parsed_rows[:16]
                    if parser.get("parse_status") not in {"accepted", "accepted_with_drops", "empty"}:
                        counts["parser_watch_count"] += 1
                        continue
                    owners = owners_by_image[image_id]
                    for row_index, parsed_row in enumerate(parsed_rows):
                        pre = match_prefix(parsed_rows, owners, budget=row_index)
                        pre_receipts = pre.get("row_assignment_receipts", [])
                        prefix_resolved = (
                            len(pre_receipts) == row_index
                            and all(
                                isinstance(item, Mapping)
                                and item.get("entity_status") == "verified_owner"
                                for item in pre_receipts
                            )
                        )
                        if not prefix_resolved:
                            counts["unresolved_prefix_watch_count"] += 1
                            continue
                        post = match_prefix(parsed_rows, owners, budget=row_index + 1)
                        current = _current_receipt(post, str(parsed_row["prediction_id"]))
                        if current is None or current.get("entity_status") != "verified_owner":
                            counts["nonpositive_watch_count"] += 1
                            continue
                        owner_id = str(current["owner_id"])
                        pre_owners = {str(item) for item in pre.get("matched_owner_ids", [])}
                        post_owners = {str(item) for item in post.get("matched_owner_ids", [])}
                        if owner_id in pre_owners or post_owners != pre_owners | {owner_id}:
                            counts["exchange_or_nonexpansion_watch_count"] += 1
                            continue
                        if float(current.get("intersection_over_union", 0.0)) < 0.75:
                            counts["owner_trusted_geometry_unknown_count"] += 1
                            continue
                        prefix_ids, row_ids = exact_row_slices(projected_ids, row_index)
                        prompt_ids = [int(item) for item in raw["prompt_token_ids"]]
                        prompt_hash = str(raw["prompt_token_ids_sha256"])
                        if token_ids_sha256(prompt_ids) != prompt_hash:
                            raise AssemblyError("executed prompt token hash mismatch")
                        prefix_hash = token_ids_sha256(prefix_ids)
                        row_hash = token_ids_sha256(row_ids)
                        occurrences.append(
                            {
                                "image_id": image_id,
                                "prefix_ids": prefix_ids,
                                "prefix_hash": prefix_hash,
                                "row_ids": row_ids,
                                "row_hash": row_hash,
                                "row_index": row_index,
                                "owner_id": owner_id,
                                "owner_iou": float(current["intersection_over_union"]),
                                "prediction_id": str(parsed_row["prediction_id"]),
                                "evidence_text": str(parsed_row.get("raw", {}).get("raw_span_text", "")),
                                "prompt_ids": prompt_ids,
                                "prompt_hash": prompt_hash,
                                "image_width": int(
                                    raw.get("image_width", candidate_rows[image_id]["width"])
                                ),
                                "image_height": int(
                                    raw.get("image_height", candidate_rows[image_id]["height"])
                                ),
                                "image_content_sha256": str(raw["source_image_file_sha256"]),
                                "source_path": str(batch_path),
                                "source_sha256": str(batch_item["sha256"]),
                                "route_id": str(raw.get("trajectory_id", "source-b16")),
                                "provenance": _generation_provenance(
                                    raw,
                                    config,
                                    checkpoint_id=checkpoint_id,
                                    prompt_hash=prompt_hash,
                                    prefix_hash=prefix_hash,
                                ),
                                "prefix_proofs": _prefix_proofs(pre, source_path=str(batch_path)),
                            }
                        )
                        counts["positive_occurrence_count"] += 1
                except (AssemblyError, KeyError, TypeError, ValueError, IndexError):
                    counts["trajectory_slice_failure_count"] += 1
                    continue
    return occurrences, dict(counts)


def _group_occurrences(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        key = (str(item["image_id"]), str(item["prefix_hash"]), str(item["owner_id"]))
        grouped[key].append(dict(item))
    results: list[dict[str, Any]] = []
    for key, aliases in grouped.items():
        by_row_hash: dict[str, dict[str, Any]] = {}
        for alias in sorted(aliases, key=lambda item: (str(item["row_hash"]), str(item["route_id"]))):
            by_row_hash.setdefault(str(alias["row_hash"]), alias)
        representative = next(iter(by_row_hash.values()))
        results.append(
            {
                "image_id": key[0],
                "prefix_hash": key[1],
                "owner_id": key[2],
                "row_index": int(representative["row_index"]),
                "aliases": list(by_row_hash.values()),
            }
        )
    return results


def _select_one_per_image(
    groups: Sequence[Mapping[str, Any]], *, max_events: int | None
) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in groups:
        # StateBank v1 represents an empty prefix with ``empty`` but admits
        # entity-transition supervision only for ``resolved`` prefixes.  Keep
        # that legacy container boundary explicit instead of widening its
        # schema in this experiment.
        if int(item["row_index"]) == 0:
            continue
        by_image[str(item["image_id"])].append(dict(item))
    selected: list[dict[str, Any]] = []
    for image_id, candidates in by_image.items():
        candidates.sort(
            key=lambda item: (
                int(item["row_index"]) == 0,
                -len(item["aliases"]),
                int(item["row_index"]),
                str(item["owner_id"]),
                str(item["prefix_hash"]),
            )
        )
        selected.append(candidates[0])
    selected.sort(
        key=lambda item: hashlib.sha256(
            f"row-local-owner-stop:{item['image_id']}".encode("utf-8")
        ).hexdigest()
    )
    return selected if max_events is None else selected[:max_events]


def _event_rows(
    groups: Sequence[Mapping[str, Any]],
    *,
    candidate_rows: Mapping[str, Mapping[str, Any]],
    candidate_pool_path: Path,
    checkpoint_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rollouts: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for group in groups:
        aliases = [dict(item) for item in group["aliases"]]
        representative = aliases[0]
        image_id = str(group["image_id"])
        annotation = candidate_rows[image_id]
        raw_image_path = Path(str(annotation["images"][0]))
        image_path = (
            raw_image_path
            if raw_image_path.is_absolute()
            else (candidate_pool_path.parent / raw_image_path).resolve(strict=True)
        )
        event_id = (
            f"row-local-owner-stop-image-{image_id}-depth-{group['row_index']}-"
            f"owner-{str(group['owner_id']).replace(':', '-')}-prefix-{str(group['prefix_hash'])[:12]}"
        )
        candidate_payloads: list[dict[str, Any]] = []
        candidate_reviews: list[dict[str, Any]] = []
        for alias_index, alias in enumerate(aliases):
            candidate_id = f"{event_id}-positive-{alias_index:02d}-{alias['row_hash'][:12]}"
            candidate_payloads.append(
                {
                    "candidate_id": candidate_id,
                    "token_ids": alias["row_ids"],
                    "token_ids_sha256": alias["row_hash"],
                    "generation_provenance": alias["provenance"],
                    "evidence_text": alias["evidence_text"],
                }
            )
            candidate_reviews.append(
                {
                    "candidate_id": candidate_id,
                    "role": "positive",
                    "harmful_kind": None,
                    "physical_owner_id": group["owner_id"],
                    "coverage_status": "uncovered",
                    "entity_review_status": "trusted",
                    "geometry_review_status": "trusted",
                    "entity_eligible": True,
                    "geometry_eligible": False,
                    "owner_resolution_interval": [0, len(alias["row_ids"])],
                    "coordinate_decision": None,
                    "selected_sites": exact_row_site_types(alias["row_ids"]),
                }
            )
        stop_id = f"{event_id}-constructed-stop"
        stop_provenance = {
            "mode": "greedy",
            "seed": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "checkpoint_id": checkpoint_id,
            "prompt_token_ids_sha256": representative["prompt_hash"],
            "prefix_token_ids_sha256": representative["prefix_hash"],
        }
        candidate_payloads.append(
            {
                "candidate_id": stop_id,
                "token_ids": [IM_END_TOKEN_ID],
                "token_ids_sha256": token_ids_sha256([IM_END_TOKEN_ID]),
                "generation_provenance": stop_provenance,
                "evidence_text": "Constructed exact-prefix <|im_end|> counterfactual; not a sampled action.",
            }
        )
        candidate_reviews.append(
            {
                "candidate_id": stop_id,
                "role": "harmful",
                "harmful_kind": "premature_terminal",
                "physical_owner_id": None,
                "coverage_status": "unknown",
                "entity_review_status": "trusted",
                "geometry_review_status": "unknown",
                "entity_eligible": True,
                "geometry_eligible": False,
                "owner_resolution_interval": None,
                "coordinate_decision": None,
                "selected_sites": [
                    {"candidate_token_offset": 0, "intended_token_type": "schema"}
                ],
            }
        )
        rollouts.append(
            {
                "event_id": event_id,
                "image": {
                    "image_id": int(image_id),
                    "path": str(image_path),
                    "width": int(representative["image_width"]),
                    "height": int(representative["image_height"]),
                    "content_sha256": representative["image_content_sha256"],
                },
                "split": "train",
                "split_group_id": f"image:{image_id}",
                "executed_prompt_token_ids": representative["prompt_ids"],
                "executed_prompt_token_ids_sha256": representative["prompt_hash"],
                "image_pad_interval": _image_pad_interval(representative["prompt_ids"]),
                "prefix_token_ids": representative["prefix_ids"],
                "prefix_token_ids_sha256": representative["prefix_hash"],
                "candidates": candidate_payloads,
            }
        )
        reviews.append(
            {
                "event_id": event_id,
                "admission_status": "accepted",
                "rejection_reason": None,
                "physical_entities": _annotation_entities(
                    annotation, source_path=candidate_pool_path
                ),
                "prefix_object_row_count": int(group["row_index"]),
                "prefix_coverage_status": "resolved",
                "prefix_covered_owner_proofs": representative["prefix_proofs"],
                "entity_transition_eligible": True,
                "coordinate_boundary_eligible": False,
                "candidates": candidate_reviews,
                "review_provenance": {
                    "schema_version": SCHEMA_VERSION,
                    "policy": "resolved-prefix-clean-owner-expansion-versus-constructed-stop",
                    "target_owner_id": group["owner_id"],
                    "target_owner_iou": representative["owner_iou"],
                    "target_row_index": int(group["row_index"]),
                    "positive_alias_count": len(aliases),
                    "constructed_stop": True,
                    "constructed_stop_is_not_sampled_or_on_policy": True,
                    "matched_remaining_budget_required": False,
                },
            }
        )
    return rollouts, reviews


def assemble(
    *,
    census_root: Path,
    candidate_pool: Path,
    reference_state_bank_manifest: Path,
    output_root: Path,
    max_events: int | None,
    max_aliases_per_event: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise AssemblyError(f"output root already exists: {output_root}")
    census_receipt_path = census_root / "receipt.json"
    image_census_path = census_root / "image-census.jsonl"
    receipt = _read_json(census_receipt_path)
    census_rows = _read_jsonl(image_census_path)
    cohort_ids = {str(int(item["image_id"])) for item in census_rows}
    if len(cohort_ids) != 2004:
        raise AssemblyError(f"expected 2,004 census images, found {len(cohort_ids)}")
    candidate_rows = _candidate_rows(candidate_pool)
    owners_by_image = load_generation7_annotations(candidate_pool, image_ids=cohort_ids)
    reference = _read_json(reference_state_bank_manifest)
    checkpoint = CheckpointIdentity.from_mapping(reference["source_checkpoint"])
    checkpoint_id = str(reference["source_checkpoint_id"])

    occurrences, replay_counts = _positive_occurrences(
        receipt=receipt,
        cohort_ids=cohort_ids,
        owners_by_image=owners_by_image,
        candidate_rows=candidate_rows,
        checkpoint_id=checkpoint_id,
    )
    groups = _group_occurrences(occurrences)
    selected = _select_one_per_image(groups, max_events=max_events)
    for item in selected:
        item["available_alias_count"] = len(item["aliases"])
        item["aliases"] = list(item["aliases"][:max_aliases_per_event])
    rollouts, reviews = _event_rows(
        selected,
        candidate_rows=candidate_rows,
        candidate_pool_path=candidate_pool,
        checkpoint_id=checkpoint_id,
    )
    output_root.mkdir(parents=True)
    state_bank_dir = output_root / "state-bank"
    manifest = assemble_state_bank(
        output_dir=state_bank_dir,
        rollout_rows=rollouts,
        review_rows=reviews,
        source_checkpoint=checkpoint,
        prompt_identity_sha256=str(reference["prompt_identity_sha256"]),
        source_artifacts=(
            {"artifact_id": "admission-census-receipt", "sha256": sha256_file(census_receipt_path)},
            {"artifact_id": "admission-census-image-records", "sha256": sha256_file(image_census_path)},
            {"artifact_id": "candidate-pool", "sha256": sha256_file(candidate_pool)},
        ),
    )
    selected_depths = defaultdict(int)
    for item in selected:
        selected_depths[str(item["row_index"])] += 1
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "inputs": {
            "census_root": str(census_root),
            "census_receipt_sha256": sha256_file(census_receipt_path),
            "image_census_sha256": sha256_file(image_census_path),
            "candidate_pool": str(candidate_pool),
            "candidate_pool_sha256": sha256_file(candidate_pool),
            "reference_state_bank_manifest": str(reference_state_bank_manifest),
            "reference_state_bank_manifest_sha256": sha256_file(reference_state_bank_manifest),
        },
        "replay_counts": replay_counts,
        "positive_group_count": len(groups),
        "positive_group_image_count": len({str(item["image_id"]) for item in groups}),
        "positive_nonzero_prefix_group_count": sum(
            int(item["row_index"]) > 0 for item in groups
        ),
        "positive_nonzero_prefix_image_count": len(
            {
                str(item["image_id"])
                for item in groups
                if int(item["row_index"]) > 0
            }
        ),
        "root_state_groups_excluded_by_state_bank_v1_count": sum(
            int(item["row_index"]) == 0 for item in groups
        ),
        "selected_event_count": len(selected),
        "selected_nonzero_prefix_event_count": sum(int(item["row_index"]) > 0 for item in selected),
        "selected_alias_count": sum(len(item["aliases"]) for item in selected),
        "max_aliases_per_event": max_aliases_per_event,
        "selected_available_alias_count": sum(
            int(item["available_alias_count"]) for item in selected
        ),
        "selected_depth_counts": dict(sorted(selected_depths.items(), key=lambda item: int(item[0]))),
        "candidate_semantics": {
            "positive": "exact model-produced clean owner-expansion row",
            "harmful": "constructed one-token exact-prefix premature STOP",
            "stop_is_natural_sampled_support": False,
            "matched_remaining_budget_required": False,
        },
        "state_bank_manifest": str((state_bank_dir / "manifest.json").resolve()),
        "state_bank_id": manifest.bank_id,
        "state_bank_record_count": manifest.record_count,
        "state_bank_records_sha256": manifest.records_sha256,
    }
    _write_json(output_root / "assembly-receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--reference-state-bank-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-aliases-per-event", type=int, default=1)
    args = parser.parse_args()
    if args.max_events is not None and args.max_events <= 0:
        parser.error("--max-events must be positive")
    if args.max_aliases_per_event <= 0:
        parser.error("--max-aliases-per-event must be positive")
    result = assemble(
        census_root=args.census_root.expanduser().resolve(strict=True),
        candidate_pool=args.candidate_pool.expanduser().resolve(strict=True),
        reference_state_bank_manifest=args.reference_state_bank_manifest.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
        max_events=args.max_events,
        max_aliases_per_event=args.max_aliases_per_event,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
