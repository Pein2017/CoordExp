#!/usr/bin/env python3
"""Materialize exact-prefix locality controls and owner-composition cases.

This script is receipt-bound to the frozen candidate pool, census, sampled
trajectory panel, Source@B16 panel, and row-local training bank.  It performs
no model execution and never retokenizes a persisted model-produced prefix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.analyze_individual_trajectory_union_support import (  # noqa: E402
    load_generation7_annotations,
)
from scripts.research.assemble_row_local_owner_stop_state_bank import (  # noqa: E402
    AssemblyError,
    BOX_END_TOKEN_ID,
    _batch_inventory,
    _candidate_rows,
    _group_occurrences,
    _load_bound_batch,
    _positive_occurrences,
    _read_json,
    _read_jsonl,
    _source_projection,
)
from src.config.fingerprint import sha256_file  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402


SCHEMA_VERSION = "continuation_locality_owner_compositionality.manifest.v1"
UNIT_ID = "2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality"


class MaterializationError(ValueError):
    """Raised when frozen evidence cannot support an exact manifest."""


def _band(object_count: int) -> str:
    if 1 <= object_count <= 3:
        return "sparse_1_to_3"
    if object_count <= 7:
        return "medium_4_to_7"
    if object_count <= 15:
        return "dense_8_to_15"
    return "very_dense_16_plus"


def _stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _owner_category(annotation: Mapping[str, Any], owner_id: str) -> str:
    suffix = str(owner_id).split(":", 1)[-1]
    for item in annotation.get("objects", []):
        if str(item.get("coco_ann_id")) == suffix:
            return str(item.get("category_name", item.get("desc", "")))
    raise MaterializationError(f"owner {owner_id} is absent from candidate annotation")


def _image_path(annotation: Mapping[str, Any], *, candidate_pool: Path) -> Path:
    values = annotation.get("images")
    if not isinstance(values, list) or len(values) != 1:
        raise MaterializationError("candidate annotation requires exactly one image")
    path = Path(str(values[0]))
    if not path.is_absolute():
        path = candidate_pool.parent / path
    return path.resolve(strict=True)


def _row_prefixes(token_ids: Sequence[int]) -> list[list[int]]:
    values = [int(item) for item in token_ids]
    ends = [index for index, value in enumerate(values) if value == BOX_END_TOKEN_ID]
    if not ends:
        raise MaterializationError("Source projection has no complete row")
    return [values[: end + 1] for end in ends]


def _source_rollouts(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for batch_item in _batch_inventory(receipt, "source_manifest_binding"):
        batch_path, payload = _load_bound_batch(batch_item)
        for value in payload.get("rollouts", []):
            if not isinstance(value, Mapping):
                raise MaterializationError(f"malformed rollout in {batch_path}")
            raw = dict(value)
            image_id = str(int(raw["image_id"]))
            try:
                projected_ids, _ = _source_projection(raw)
            except AssemblyError:
                # Invalid-before-budget Source rows are explicit census
                # censoring, not usable exact complete-row boundaries.
                continue
            prompt_ids = [int(item) for item in raw["prompt_token_ids"]]
            prompt_hash = str(raw["prompt_token_ids_sha256"])
            if token_ids_sha256(prompt_ids) != prompt_hash:
                raise MaterializationError(f"Source prompt hash mismatch for {image_id}")
            prefixes = _row_prefixes(projected_ids)
            natural_end = bool(raw.get("source_b16", {}).get("natural_end_before_budget"))
            row = {
                "image_id": image_id,
                "prompt_ids": prompt_ids,
                "prompt_hash": prompt_hash,
                "prefixes": prefixes,
                "complete_row_count": len(prefixes),
                "natural_end": natural_end,
                "source_path": str(batch_path),
                "source_sha256": str(batch_item["sha256"]),
                "image_content_sha256": str(raw["source_image_file_sha256"]),
            }
            if image_id in rows:
                raise MaterializationError(f"duplicate Source rollout for image {image_id}")
            rows[image_id] = row
    return rows


def _source_candidate(census_row: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        item
        for item in census_row.get("candidates", [])
        if isinstance(item, Mapping) and item.get("candidate_id") == "source-b16"
    ]
    if len(matches) != 1:
        raise MaterializationError("census row lacks one source-b16 candidate")
    return matches[0]


def _covered_source_owners(candidate: Mapping[str, Any], depth: int) -> list[str]:
    owners = {
        str(row["owner_id"])
        for row in candidate.get("geometry", {}).get("rows", [])
        if int(row.get("generated_row_index", -1)) < depth
        and row.get("entity_status") == "verified_owner"
        and row.get("owner_id") is not None
    }
    return sorted(owners)


def _source_boundary(
    *,
    image_id: str,
    depth: int,
    cohort: str,
    source_rows: Mapping[str, Mapping[str, Any]],
    census_by_image: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    candidate_pool: Path,
    matched_event_id: str | None = None,
) -> dict[str, Any] | None:
    source = source_rows.get(image_id)
    if source is None or depth <= 0 or depth > int(source["complete_row_count"]):
        return None
    annotation = candidate_rows[image_id]
    prefix = [int(item) for item in source["prefixes"][depth - 1]]
    candidate = _source_candidate(census_by_image[image_id])
    covered = _covered_source_owners(candidate, depth)
    all_owner_count = len(annotation.get("objects", []))
    if depth < int(source["complete_row_count"]):
        observed_next_action = "row_opener"
    elif bool(source["natural_end"]):
        observed_next_action = "terminal"
    else:
        observed_next_action = "budget_end"
    return {
        "boundary_id": f"{cohort}-image-{image_id}-depth-{depth}-{token_ids_sha256(prefix)[:12]}",
        "cohort": cohort,
        "image_id": image_id,
        "image_path": str(_image_path(annotation, candidate_pool=candidate_pool)),
        "image_content_sha256": str(source["image_content_sha256"]),
        "base_prompt_token_ids": list(source["prompt_ids"]),
        "base_prompt_token_ids_sha256": str(source["prompt_hash"]),
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": token_ids_sha256(prefix),
        "prefix_depth": depth,
        "observed_next_action": observed_next_action,
        "natural_end": bool(source["natural_end"]),
        "covered_owner_ids": covered,
        "remaining_annotation_owner_count": max(0, all_owner_count - len(covered)),
        "annotation_object_count": all_owner_count,
        "object_count_band": _band(all_owner_count),
        "matched_training_event_id": matched_event_id,
        "source_artifact": {
            "path": str(source["source_path"]),
            "sha256": str(source["source_sha256"]),
        },
    }


def _select_same_image_near(
    *,
    event_records: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    census_by_image: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    candidate_pool: Path,
    count: int,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for event in event_records:
        image_id = str(int(event["image"]["image_id"]))
        depth = int(event["prefix_object_row_count"])
        boundary = _source_boundary(
            image_id=image_id,
            depth=depth,
            cohort="same_image_near_continue",
            source_rows=source_rows,
            census_by_image=census_by_image,
            candidate_rows=candidate_rows,
            candidate_pool=candidate_pool,
            matched_event_id=str(event["event_id"]),
        )
        if boundary is None or boundary["observed_next_action"] != "row_opener":
            continue
        if boundary["prefix_token_ids_sha256"] == event["prefix_token_ids_sha256"]:
            continue
        eligible.append(boundary)
    eligible.sort(key=lambda item: _stable_key("same-image-near", item["boundary_id"]))
    if len(eligible) < count:
        raise MaterializationError(
            f"requested {count} same-image-near controls, found {len(eligible)}"
        )
    return eligible[:count]


def _select_untouched_continue(
    *,
    count: int,
    event_records: Sequence[Mapping[str, Any]],
    trained_image_ids: set[str],
    source_rows: Mapping[str, Mapping[str, Any]],
    census_by_image: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    candidate_pool: Path,
) -> list[dict[str, Any]]:
    targets = sorted(
        event_records,
        key=lambda item: _stable_key("untouched-target", item["event_id"]),
    )
    candidates: dict[tuple[int, str], list[str]] = defaultdict(list)
    for image_id in sorted((set(source_rows) & set(census_by_image)) - trained_image_ids):
        annotation = candidate_rows.get(image_id)
        if annotation is None:
            continue
        band = _band(len(annotation.get("objects", [])))
        total = int(source_rows[image_id]["complete_row_count"])
        for depth in range(1, total):
            candidates[(depth, band)].append(image_id)
    for key, values in candidates.items():
        values.sort(key=lambda image_id: _stable_key("untouched-candidate", key, image_id))
    used: set[str] = set()
    selected: list[dict[str, Any]] = []
    for target in targets:
        if len(selected) >= count:
            break
        depth = int(target["prefix_object_row_count"])
        image_id = str(int(target["image"]["image_id"]))
        band = _band(len(candidate_rows[image_id].get("objects", [])))
        available = candidates.get((depth, band), [])
        match = next((value for value in available if value not in used), None)
        if match is None:
            continue
        boundary = _source_boundary(
            image_id=match,
            depth=depth,
            cohort="untouched_matched_continue",
            source_rows=source_rows,
            census_by_image=census_by_image,
            candidate_rows=candidate_rows,
            candidate_pool=candidate_pool,
            matched_event_id=str(target["event_id"]),
        )
        if boundary is None or boundary["observed_next_action"] != "row_opener":
            continue
        used.add(match)
        selected.append(boundary)
    if len(selected) < count:
        raise MaterializationError(
            f"requested {count} untouched matched controls, found {len(selected)}"
        )
    return selected


def _select_untouched_terminal(
    *,
    count: int,
    trained_image_ids: set[str],
    source_rows: Mapping[str, Mapping[str, Any]],
    census_by_image: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    candidate_pool: Path,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for image_id in sorted((set(source_rows) & set(census_by_image)) - trained_image_ids):
        source = source_rows[image_id]
        if not bool(source["natural_end"]):
            continue
        candidate = _source_candidate(census_by_image[image_id])
        if not candidate.get("uncovered_trusted_owner_ids_at_stop"):
            continue
        boundary = _source_boundary(
            image_id=image_id,
            depth=int(source["complete_row_count"]),
            cohort="untouched_terminal_with_remaining_owner",
            source_rows=source_rows,
            census_by_image=census_by_image,
            candidate_rows=candidate_rows,
            candidate_pool=candidate_pool,
        )
        if boundary is not None:
            boundary["census_uncovered_owner_ids_at_stop"] = sorted(
                str(value) for value in candidate["uncovered_trusted_owner_ids_at_stop"]
            )
            eligible.append(boundary)
    eligible.sort(
        key=lambda item: (
            int(item["prefix_depth"]),
            _stable_key("untouched-terminal", item["boundary_id"]),
        )
    )
    if len(eligible) < count:
        raise MaterializationError(
            f"requested {count} untouched terminal controls, found {len(eligible)}"
        )
    return eligible[:count]


def _owner_case(
    *,
    group: Mapping[str, Any],
    state_groups: Sequence[Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    candidate_pool: Path,
) -> dict[str, Any]:
    representative = dict(group["aliases"][0])
    image_id = str(group["image_id"])
    annotation = candidate_rows[image_id]
    owner_id = str(group["owner_id"])
    category = _owner_category(annotation, owner_id)
    category_count = sum(
        str(item.get("category_name", item.get("desc", ""))) == category
        for item in annotation.get("objects", [])
    )
    covered = sorted(
        str(item["owner_id"]) for item in representative.get("prefix_proofs", [])
    )
    secondary = []
    for other in state_groups:
        other_owner = str(other["owner_id"])
        if other_owner == owner_id:
            continue
        alias = dict(other["aliases"][0])
        secondary.append(
            {
                "owner_id": other_owner,
                "category": _owner_category(annotation, other_owner),
                "row_token_ids": list(alias["row_ids"]),
                "row_token_ids_sha256": str(alias["row_hash"]),
                "evidence_text": str(alias["evidence_text"]),
            }
        )
    secondary.sort(key=lambda item: (item["owner_id"], item["row_token_ids_sha256"]))
    prefix = list(representative["prefix_ids"])
    row = list(representative["row_ids"])
    return {
        "case_id": (
            f"owner-case-image-{image_id}-depth-{group['row_index']}-"
            f"owner-{owner_id.replace(':', '-')}-prefix-{group['prefix_hash'][:12]}"
        ),
        "image_id": image_id,
        "image_path": str(_image_path(annotation, candidate_pool=candidate_pool)),
        "image_content_sha256": str(representative["image_content_sha256"]),
        "base_prompt_token_ids": list(representative["prompt_ids"]),
        "base_prompt_token_ids_sha256": str(representative["prompt_hash"]),
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": token_ids_sha256(prefix),
        "prefix_depth": int(group["row_index"]),
        "covered_owner_ids": covered,
        "annotation_object_count": len(annotation.get("objects", [])),
        "object_count_band": _band(len(annotation.get("objects", []))),
        "target": {
            "owner_id": owner_id,
            "category": category,
            "same_category_annotation_count": category_count,
            "row_token_ids": row,
            "row_token_ids_sha256": token_ids_sha256(row),
            "evidence_text": str(representative["evidence_text"]),
            "owner_iou": float(representative["owner_iou"]),
        },
        "secondary_targets": secondary,
        "sampled_artifact": {
            "path": str(representative["source_path"]),
            "sha256": str(representative["source_sha256"]),
            "trajectory_id": str(representative["route_id"]),
        },
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists() or output.parent.exists():
        raise MaterializationError(f"refusing to overwrite output root: {output.parent}")
    census_root = args.census_root.expanduser().resolve(strict=True)
    candidate_pool = args.candidate_pool.expanduser().resolve(strict=True)
    state_bank_records = args.state_bank_records.expanduser().resolve(strict=True)
    reference_manifest = args.reference_state_bank_manifest.expanduser().resolve(strict=True)
    receipt = _read_json(census_root / "receipt.json")
    census_rows = _read_jsonl(census_root / "image-census.jsonl")
    census_by_image = {str(int(item["image_id"])): item for item in census_rows}
    cohort_ids = set(census_by_image)
    candidate_rows = _candidate_rows(candidate_pool)
    owners = load_generation7_annotations(candidate_pool, image_ids=cohort_ids)
    reference = _read_json(reference_manifest)
    occurrences, replay_counts = _positive_occurrences(
        receipt=receipt,
        cohort_ids=cohort_ids,
        owners_by_image=owners,
        candidate_rows=candidate_rows,
        checkpoint_id=str(reference["source_checkpoint_id"]),
    )
    groups = [item for item in _group_occurrences(occurrences) if int(item["row_index"]) > 0]
    source_rows = _source_rollouts(receipt)
    event_records = _read_jsonl(state_bank_records)
    trained_image_ids = {str(int(item["image"]["image_id"])) for item in event_records}

    same_image_near = _select_same_image_near(
        event_records=event_records,
        source_rows=source_rows,
        census_by_image=census_by_image,
        candidate_rows=candidate_rows,
        candidate_pool=candidate_pool,
        count=int(args.same_image_near_count),
    )
    untouched = _select_untouched_continue(
        count=int(args.untouched_continue_count),
        event_records=event_records,
        trained_image_ids=trained_image_ids,
        source_rows=source_rows,
        census_by_image=census_by_image,
        candidate_rows=candidate_rows,
        candidate_pool=candidate_pool,
    )
    terminal = _select_untouched_terminal(
        count=int(args.untouched_terminal_count),
        trained_image_ids=trained_image_ids,
        source_rows=source_rows,
        census_by_image=census_by_image,
        candidate_rows=candidate_rows,
        candidate_pool=candidate_pool,
    )

    by_state: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_state[(str(group["image_id"]), str(group["prefix_hash"]))].append(group)
    multi_states = {
        key: values
        for key, values in by_state.items()
        if len({str(item["owner_id"]) for item in values}) >= 2
    }
    required_groups = [item for values in multi_states.values() for item in values]
    required_keys = {
        (str(item["image_id"]), str(item["prefix_hash"]), str(item["owner_id"]))
        for item in required_groups
    }
    remaining = [
        item
        for item in groups
        if (str(item["image_id"]), str(item["prefix_hash"]), str(item["owner_id"]))
        not in required_keys
    ]
    remaining.sort(
        key=lambda item: _stable_key(
            "owner-atlas", item["image_id"], item["prefix_hash"], item["owner_id"]
        )
    )
    atlas_count = int(args.owner_case_count)
    if len(required_groups) > atlas_count or len(required_groups) + len(remaining) < atlas_count:
        raise MaterializationError("owner-case request is incompatible with available exact groups")
    selected_groups = [*required_groups, *remaining[: atlas_count - len(required_groups)]]
    selected_groups.sort(
        key=lambda item: _stable_key(
            "owner-atlas-final", item["image_id"], item["prefix_hash"], item["owner_id"]
        )
    )
    owner_cases = [
        _owner_case(
            group=group,
            state_groups=by_state[(str(group["image_id"]), str(group["prefix_hash"]))],
            candidate_rows=candidate_rows,
            candidate_pool=candidate_pool,
        )
        for group in selected_groups
    ]

    locality = [*same_image_near, *untouched, *terminal]
    result = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "inputs": {
            "census_receipt": {
                "path": str(census_root / "receipt.json"),
                "sha256": sha256_file(census_root / "receipt.json"),
            },
            "image_census": {
                "path": str(census_root / "image-census.jsonl"),
                "sha256": sha256_file(census_root / "image-census.jsonl"),
            },
            "candidate_pool": {"path": str(candidate_pool), "sha256": sha256_file(candidate_pool)},
            "training_state_bank_records": {
                "path": str(state_bank_records),
                "sha256": sha256_file(state_bank_records),
            },
            "reference_state_bank_manifest": {
                "path": str(reference_manifest),
                "sha256": sha256_file(reference_manifest),
            },
        },
        "locality_boundaries": locality,
        "owner_cases": owner_cases,
        "selection_summary": {
            "replay_counts": replay_counts,
            "positive_occurrence_count": len(occurrences),
            "positive_nonzero_prefix_group_count": len(groups),
            "multi_owner_prefix_state_count": len(multi_states),
            "multi_owner_prefix_image_count": len({key[0] for key in multi_states}),
            "multi_owner_target_case_count": len(required_groups),
            "same_image_near_count": len(same_image_near),
            "untouched_continue_count": len(untouched),
            "untouched_terminal_count": len(terminal),
            "locality_cohort_counts": dict(Counter(item["cohort"] for item in locality)),
            "locality_depth_counts": dict(Counter(str(item["prefix_depth"]) for item in locality)),
            "owner_case_count": len(owner_cases),
            "owner_case_depth_counts": dict(Counter(str(item["prefix_depth"]) for item in owner_cases)),
            "owner_case_same_category_multi_instance_count": sum(
                int(item["target"]["same_category_annotation_count"]) > 1
                for item in owner_cases
            ),
            "owner_case_with_secondary_target_count": sum(
                bool(item["secondary_targets"]) for item in owner_cases
            ),
        },
        "claim_boundary": (
            "all records are fixed-prefix diagnostics; none is a free-rollout final-set outcome"
        ),
    }
    _write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--state-bank-records", type=Path, required=True)
    parser.add_argument("--reference-state-bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--same-image-near-count", type=int, default=480)
    parser.add_argument("--untouched-continue-count", type=int, default=480)
    parser.add_argument("--untouched-terminal-count", type=int, default=240)
    parser.add_argument("--owner-case-count", type=int, default=400)
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(json.dumps(result["selection_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
