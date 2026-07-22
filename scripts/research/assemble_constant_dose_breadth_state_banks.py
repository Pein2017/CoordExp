#!/usr/bin/env python3
"""Assemble the fixed-dose broad and concentrated StateBanks.

This is the narrow final assembly step for the 2,432-image constant-dose
panel.  It consumes only frozen artifacts: the candidate pool and split,
the old/new greedy and sampled rollout artifacts, the trajectory-union
analysis, and the upstream panel-union receipt.  In particular, it does not
borrow a reference StateBank or accept a manually edited review overlay.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import glob
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.fingerprint import sha256_file, sha256_json  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402

from scripts.research.assemble_positive_path_imitation_state_bank import (  # noqa: E402
    AssemblyError,
    _image_id,
    _image_pad_interval,
    _mapping,
    _norm_sha,
    _read_json,
    _token_ids,
)
from scripts.research.assemble_source_preservation_multi_route_state_banks import (  # noqa: E402
    ROUTE_ANALYSIS_SCHEMA_VERSION,
    _build_multi_candidates,
    _build_source_candidates,
    _load_rollout_rows,
    _source_artifacts,
    _write_arm,
    _write_json,
    derive_checkpoint_identity_from_rollout_artifact,
    materialize_constant_dose_breadth_arm,
    select_constant_dose_breadth_arms,
    validate_constant_dose_panel_execution_contract,
    validate_constant_dose_training_reservoir,
)
from scripts.research.validate_constant_dose_trajectory_panel_union import (  # noqa: E402
    GREEDY_SEED,
    SAMPLED_SEEDS,
    validate_panel_union,
)


SCHEMA_VERSION = "constant_dose_breadth_state_bank_assembly.v1"
UNION_SCHEMA_VERSION = "constant_dose_trajectory_panel_union.v1"
CANONICAL_ANALYSIS_IOU_THRESHOLD = 0.5
ANALYZER_PATH = Path(__file__).resolve().with_name(
    "analyze_individual_trajectory_union_support.py"
)


def _canonical_image_id(value: Any, *, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise AssemblyError(f"{field} has no usable image_id")
    if isinstance(value, (int, str)) and str(value).strip():
        return str(value).strip()
    raise AssemblyError(f"{field} has no usable image_id")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise AssemblyError(f"blank JSONL row {line_number}: {path}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AssemblyError(f"invalid JSONL row {line_number}: {path}") from exc
        if not isinstance(value, Mapping):
            raise AssemblyError(f"non-object JSONL row {line_number}: {path}")
        rows.append(dict(value))
    return rows


def _candidate_pool(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_read_jsonl(path), 1):
        image_id = _canonical_image_id(row.get("image_id"), field=f"candidate-pool row {index}")
        if image_id in result:
            raise AssemblyError(f"candidate pool duplicates image_id {image_id}")
        result[image_id] = row
    if len(result) != 2432:
        raise AssemblyError(f"candidate pool has {len(result)} images, expected 2432")
    return result


def _split_membership(*, candidate_pool: Path, split_receipt: Path) -> dict[str, set[str]]:
    receipt = _read_json(split_receipt)
    if not isinstance(receipt, Mapping):
        raise AssemblyError("split receipt must be an object")
    source = receipt.get("input")
    outputs = receipt.get("outputs")
    if not isinstance(source, Mapping) or source.get("sha256") != sha256_file(candidate_pool):
        raise AssemblyError("split receipt does not bind the supplied candidate pool")
    if not isinstance(outputs, Mapping) or set(outputs) != {"train_candidate", "development", "heldout"}:
        raise AssemblyError("split receipt must name exactly train_candidate/development/heldout")
    expected_counts = {"train_candidate": 2048, "development": 256, "heldout": 128}
    result: dict[str, set[str]] = {}
    for name, expected_count in expected_counts.items():
        entry = outputs[name]
        if not isinstance(entry, Mapping):
            raise AssemblyError(f"split receipt {name} entry is invalid")
        path = Path(str(entry.get("path", ""))).expanduser().resolve(strict=True)
        if entry.get("sha256") != sha256_file(path) or entry.get("count") != expected_count:
            raise AssemblyError(f"split receipt {name} does not bind its frozen JSONL")
        ids = {
            _canonical_image_id(row.get("image_id"), field=f"{name} split")
            for row in _read_jsonl(path)
        }
        if len(ids) != expected_count:
            raise AssemblyError(f"{name} split has {len(ids)} unique images, expected {expected_count}")
        result[name] = ids
    if any(left & right for index, left in enumerate(result.values()) for right in list(result.values())[index + 1:]):
        raise AssemblyError("frozen candidate-pool split memberships overlap")
    pool_ids = set(_candidate_pool(candidate_pool))
    if set().union(*result.values()) != pool_ids:
        raise AssemblyError("frozen split is not the exact candidate-pool union")
    return result


def _expand_paths(values: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        matches = [Path(item) for item in glob.glob(str(value))] or [value]
        for path in matches:
            paths.add(path.expanduser().resolve(strict=True))
    return sorted(paths)


def _load_rollouts(paths: Sequence[Path], *, sampled: bool) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    rows, configs = _load_rollout_rows(paths, sampled_only=sampled)
    for row in rows.values():
        config = configs[str(Path(str(row["_source_path"])).resolve())]
        row["_temperature"] = config["temperature"]
        row["_top_p"] = config["top_p"]
        row["_repetition_penalty"] = config["repetition_penalty"]
    return rows, configs


def _rollout_prompt_evidence(
    *, paths: Sequence[Path], rows: Mapping[tuple[str, int], Mapping[str, Any]]
) -> dict[tuple[str, int], dict[str, Any]]:
    """Read exact prompt/image evidence for every row without re-tokenizing."""

    metadata_by_path: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        artifact = _read_json(path)
        if not isinstance(artifact, Mapping):
            raise AssemblyError(f"rollout artifact must be an object: {path}")
        metadata = artifact.get("prompt_metadata")
        if not isinstance(metadata, Mapping):
            raise AssemblyError(f"rollout artifact lacks prompt_metadata: {path}")
        metadata_by_path[str(path.resolve())] = metadata
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for key, row in rows.items():
        source_path = str(Path(str(row.get("_source_path", ""))).resolve())
        metadata = metadata_by_path.get(source_path)
        if metadata is None:
            raise AssemblyError(f"rollout row has unknown source artifact: {source_path}")
        example_id = str(row.get("example_id", ""))
        image_id = key[0]
        raw = metadata.get(example_id, metadata.get(image_id))
        if not isinstance(raw, Mapping):
            raise AssemblyError(f"rollout {image_id}:{key[1]} lacks prompt metadata keyed by example_id")
        prompt_ids = _token_ids(row.get("prompt_token_ids"), f"rollout {image_id}:{key[1]}.prompt_token_ids")
        prompt_hash = _norm_sha(row.get("prompt_token_ids_sha256"), f"rollout {image_id}:{key[1]}.prompt_token_ids_sha256")
        if prompt_hash != token_ids_sha256(prompt_ids):
            raise AssemblyError(f"rollout {image_id}:{key[1]} prompt-token hash mismatch")
        metadata_ids = _token_ids(raw.get("prompt_token_ids"), f"prompt metadata {image_id}.prompt_token_ids")
        metadata_hash = _norm_sha(raw.get("prompt_token_ids_sha256"), f"prompt metadata {image_id}.prompt_token_ids_sha256")
        if metadata_hash != token_ids_sha256(metadata_ids) or metadata_ids != prompt_ids or metadata_hash != prompt_hash:
            raise AssemblyError(f"rollout {image_id}:{key[1]} prompt IDs differ from its exact prompt metadata")
        image_path = raw.get("image_path")
        image_sha = raw.get("image_sha256")
        width, height = raw.get("width"), raw.get("height")
        if not isinstance(image_path, str) or not image_path or not isinstance(image_sha, str):
            raise AssemblyError(f"prompt metadata lacks image path/hash for {image_id}:{key[1]}")
        if isinstance(width, bool) or not isinstance(width, int) or isinstance(height, bool) or not isinstance(height, int):
            raise AssemblyError(f"prompt metadata lacks integer image dimensions for {image_id}:{key[1]}")
        if row.get("executed_media_sha256") != image_sha:
            raise AssemblyError(f"rollout {image_id}:{key[1]} media hash differs from prompt metadata")
        result[key] = {
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": prompt_hash,
            "image_path": str(Path(image_path).expanduser().resolve()),
            "image_sha256": image_sha,
            "width": width,
            "height": height,
        }
    return result


def derive_reference_records(
    *,
    candidate_rows: Mapping[str, Mapping[str, Any]],
    candidate_pool_path: Path,
    greedy_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    sampled_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    prompt_evidence: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive per-image references from the frozen rollout and pool evidence."""

    references: dict[str, dict[str, Any]] = {}
    for image_id, candidate in candidate_rows.items():
        required_keys = [(image_id, GREEDY_SEED), *[(image_id, seed) for seed in sorted(SAMPLED_SEEDS)]]
        if any(key not in prompt_evidence for key in required_keys):
            raise AssemblyError(f"image {image_id} lacks greedy or all sixteen sampled prompt records")
        first = prompt_evidence[required_keys[0]]
        for key in required_keys[1:]:
            other = prompt_evidence[key]
            for field in ("prompt_token_ids", "prompt_token_ids_sha256", "image_path", "image_sha256", "width", "height"):
                if other[field] != first[field]:
                    raise AssemblyError(f"greedy/sampled exact prompt evidence differs for image {image_id}: {field}")
        images = candidate.get("images")
        metadata = candidate.get("metadata")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise AssemblyError(f"candidate-pool image {image_id} lacks one image path")
        if not isinstance(metadata, Mapping) or metadata.get("split") != "train":
            raise AssemblyError(f"candidate-pool image {image_id} lacks train split metadata")
        path = Path(images[0])
        candidate_path = (candidate_pool_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if candidate_path != Path(str(first["image_path"])).resolve():
            raise AssemblyError(f"candidate-pool image path differs from rollout evidence for image {image_id}")
        if candidate.get("width") != first["width"] or candidate.get("height") != first["height"]:
            raise AssemblyError(f"candidate-pool dimensions differ from rollout evidence for image {image_id}")
        references[image_id] = {
            "image": {
                "image_id": int(image_id),
                "path": str(candidate_path),
                "width": int(first["width"]),
                "height": int(first["height"]),
                "content_sha256": str(first["image_sha256"]),
            },
            "split": "train",
            "executed_prompt_token_ids": list(first["prompt_token_ids"]),
            "executed_prompt_token_ids_sha256": str(first["prompt_token_ids_sha256"]),
            "image_pad_interval": list(_image_pad_interval(first["prompt_token_ids"])),
        }
    if set(greedy_rows) != {(image, GREEDY_SEED) for image in candidate_rows}:
        raise AssemblyError("greedy rows are not exactly one canonical seed per candidate-pool image")
    expected_sampled = {(image, seed) for image in candidate_rows for seed in SAMPLED_SEEDS}
    if set(sampled_rows) != expected_sampled:
        raise AssemblyError("sampled rows are not exactly the canonical sixteen seeds per candidate-pool image")
    return references


def _panel_receipt(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != UNION_SCHEMA_VERSION or value.get("passed") is not True:
        raise AssemblyError("upstream trajectory-union receipt is not a passed canonical receipt")
    if value.get("image_count") != 2432:
        raise AssemblyError("upstream trajectory-union receipt does not cover 2,432 images")
    contract = value.get("execution_metadata")
    if not isinstance(contract, Mapping):
        raise AssemblyError("upstream trajectory-union receipt lacks request-scoped execution_metadata")
    result = dict(value)
    result["execution_metadata"] = validate_constant_dose_panel_execution_contract(contract)
    return result


def _validate_panel_receipt_match(
    saved: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    """Require the saved union receipt to describe the supplied raw artifacts.

    The assembler recomputes the panel union from the raw rollout artifacts.
    Comparing only the prompt fingerprint would leave the saved producer
    fingerprint and old/new panel counts unbound to those inputs.
    """

    fields = (
        "schema_version",
        "passed",
        "image_count",
        "old_image_count",
        "new_image_count",
        "sampled_seed_count",
        "prompt_policy_fingerprint",
        "execution_metadata",
    )
    mismatches = [field for field in fields if saved.get(field) != recomputed.get(field)]
    if mismatches:
        raise AssemblyError(
            "upstream trajectory-union receipt differs from the supplied raw artifacts: "
            + ", ".join(mismatches)
        )


def _validate_trajectory_analysis_provenance(
    analysis: Mapping[str, Any],
    *,
    candidate_pool: Path,
    rollout_paths: Sequence[Path],
    validated_union: Mapping[str, Any],
) -> None:
    """Bind trajectory analysis to the exact evidence selected for assembly."""

    sources = analysis.get("sources")
    if not isinstance(sources, Mapping):
        raise AssemblyError("trajectory analysis lacks source provenance")
    if sources.get("annotations_sha256") != sha256_file(candidate_pool):
        raise AssemblyError(
            "trajectory analysis annotations hash differs from the supplied candidate pool"
        )

    recorded_rollout_hashes = sources.get("rollout_artifact_sha256")
    if not isinstance(recorded_rollout_hashes, list) or not all(
        isinstance(value, str) for value in recorded_rollout_hashes
    ):
        raise AssemblyError("trajectory analysis lacks rollout artifact hash provenance")
    supplied_rollout_hashes = [sha256_file(path) for path in rollout_paths]
    if Counter(recorded_rollout_hashes) != Counter(supplied_rollout_hashes):
        raise AssemblyError(
            "trajectory analysis rollout artifact hashes differ from the supplied artifacts"
        )

    analyzer_path = ANALYZER_PATH.resolve(strict=True)
    if sources.get("analyzer_path") != str(analyzer_path):
        raise AssemblyError("trajectory analysis analyzer path is stale or swapped")
    if sources.get("analyzer_sha256") != sha256_file(analyzer_path):
        raise AssemblyError("trajectory analysis analyzer hash is stale or swapped")

    policy = sources.get("analysis_policy")
    if not isinstance(policy, Mapping):
        raise AssemblyError("trajectory analysis lacks analysis policy provenance")
    if policy.get("fixed_budgets") != [16] or analysis.get("fixed_budgets") != [16]:
        raise AssemblyError("trajectory analysis must use exactly fixed budget [16]")
    if (
        policy.get("iou_threshold") != CANONICAL_ANALYSIS_IOU_THRESHOLD
        or analysis.get("iou_threshold") != CANONICAL_ANALYSIS_IOU_THRESHOLD
    ):
        raise AssemblyError("trajectory analysis must use canonical IoU threshold 0.5")
    require_full_panel = policy.get("require_full_panel")
    if not isinstance(require_full_panel, bool) or analysis.get(
        "require_full_panel"
    ) is not require_full_panel:
        raise AssemblyError("trajectory analysis full/incomplete panel policy is missing or inconsistent")
    if policy.get("review_decisions_used") is not False:
        raise AssemblyError("trajectory analysis must not use a review overlay")
    if analysis.get("review_provenance") is not None or any(
        key in sources
        for key in ("review_decisions_path", "review_decisions_sha256")
    ):
        raise AssemblyError("trajectory analysis contains review overlay provenance")

    expected_observed_panel = {
        "image_count": validated_union.get("image_count"),
        "greedy_seeds": [GREEDY_SEED],
        "sampled_seeds": sorted(SAMPLED_SEEDS),
        "greedy_trajectories_per_image": [1],
        "sampled_trajectories_per_image": [len(SAMPLED_SEEDS)],
    }
    observed_panel = policy.get("observed_panel")
    if (
        observed_panel != expected_observed_panel
        or analysis.get("observed_panel") != expected_observed_panel
    ):
        raise AssemblyError(
            "trajectory analysis observed panel differs from the validated rollout union"
        )


def _selection_receipt(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Project selection to immutable JSON without serializing internal inputs twice."""

    arms: dict[str, Any] = {}
    for name in ("broad", "concentrated"):
        arm = _mapping(selection.get(name), f"selection.{name}")
        arms[name] = {
            key: copy.deepcopy(arm[key])
            for key in (
                "image_ids", "event_count", "sampled_event_count", "source_event_count",
                "mean_event_weight", "total_event_weight", "image_family_event_counts",
                "selection_distributions", "trajectory_panel_execution_metadata",
                "rank_matching_receipt",
            )
        }
        arms[name]["events"] = [
            {
                key: copy.deepcopy(event[key])
                for key in (
                    "event_id", "image_id", "event_family", "object_count_band", "selection_rank",
                    "route_id", "route_seed", "generated_row_index", "owner_id",
                    "prefix_token_ids_sha256", "candidate_token_ids_sha256", "route_count", "row_depth",
                    "complete_row_coordinate_token_supervision",
                    "image_balanced_event_weight",
                )
            }
            for event in arm["events"]
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_selection",
        "trajectory_panel_execution_metadata": copy.deepcopy(selection["trajectory_panel_execution_metadata"]),
        "eligible_unique_training_image_count": selection["eligible_unique_training_image_count"],
        "eligible_images_by_band": copy.deepcopy(selection["eligible_images_by_band"]),
        "broad_band_quota": copy.deepcopy(selection["broad_band_quota"]),
        "concentrated_band_quota": copy.deepcopy(selection["concentrated_band_quota"]),
        "pair_quota_by_object_count_band": copy.deepcopy(
            selection["pair_quota_by_object_count_band"]
        ),
        "rank_matching_receipt": copy.deepcopy(selection["rank_matching_receipt"]),
        "arms": arms,
    }


def assemble_constant_dose_breadth_state_banks(
    *,
    candidate_pool: Path,
    split_receipt: Path,
    trajectory_analysis: Path,
    panel_union_receipt: Path,
    old_greedy: Sequence[Path],
    new_greedy: Sequence[Path],
    old_sampled: Sequence[Path],
    new_sampled: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Validate frozen evidence and write two immutable matched StateBanks."""

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise AssemblyError(f"output path already exists and will not be overwritten: {output}")
    pool_path = candidate_pool.expanduser().resolve(strict=True)
    split_path = split_receipt.expanduser().resolve(strict=True)
    analysis_path = trajectory_analysis.expanduser().resolve(strict=True)
    union_path = panel_union_receipt.expanduser().resolve(strict=True)
    old_greedy_paths, new_greedy_paths = _expand_paths(old_greedy), _expand_paths(new_greedy)
    old_sampled_paths, new_sampled_paths = _expand_paths(old_sampled), _expand_paths(new_sampled)
    if not all((old_greedy_paths, new_greedy_paths, old_sampled_paths, new_sampled_paths)):
        raise AssemblyError("old/new greedy and sampled rollout artifacts are all required")

    panel_receipt = _panel_receipt(union_path)
    recomputed_union = validate_panel_union(
        candidate_pool=pool_path,
        split_receipt=split_path,
        old_greedy=old_greedy_paths,
        old_sampled=old_sampled_paths,
        new_greedy=new_greedy_paths,
        new_sampled=new_sampled_paths,
    )
    _validate_panel_receipt_match(panel_receipt, recomputed_union)
    analysis = _read_json(analysis_path)
    if not isinstance(analysis, Mapping) or analysis.get("schema_version") != ROUTE_ANALYSIS_SCHEMA_VERSION:
        raise AssemblyError("unsupported trajectory union analysis schema")
    _validate_trajectory_analysis_provenance(
        analysis,
        candidate_pool=pool_path,
        rollout_paths=[
            *old_greedy_paths,
            *new_greedy_paths,
            *old_sampled_paths,
            *new_sampled_paths,
        ],
        validated_union=recomputed_union,
    )
    split_ids = _split_membership(candidate_pool=pool_path, split_receipt=split_path)
    candidate_rows = _candidate_pool(pool_path)
    all_greedy_paths = [*old_greedy_paths, *new_greedy_paths]
    all_sampled_paths = [*old_sampled_paths, *new_sampled_paths]
    greedy_rows, _ = _load_rollouts(all_greedy_paths, sampled=False)
    sampled_rows, _ = _load_rollouts(all_sampled_paths, sampled=True)
    prompt_evidence = {
        **_rollout_prompt_evidence(paths=all_greedy_paths, rows=greedy_rows),
        **_rollout_prompt_evidence(paths=all_sampled_paths, rows=sampled_rows),
    }
    reference_records = derive_reference_records(
        candidate_rows=candidate_rows,
        candidate_pool_path=pool_path,
        greedy_rows=greedy_rows,
        sampled_rows=sampled_rows,
        prompt_evidence=prompt_evidence,
    )
    identities = [
        derive_checkpoint_identity_from_rollout_artifact(_mapping(_read_json(path), f"rollout {path}"), artifact_path=path)
        for path in [*all_greedy_paths, *all_sampled_paths]
    ]
    source_checkpoint = identities[0]
    if any(identity != source_checkpoint for identity in identities[1:]):
        raise AssemblyError("old/new rollout artifacts do not share one exact checkpoint identity")
    checkpoint_id = sha256_json(source_checkpoint.to_artifact_dict())

    raw_results = analysis.get("image_results")
    if not isinstance(raw_results, list):
        raise AssemblyError("trajectory analysis lacks image_results")
    image_results = {
        _image_id(_mapping(item, "trajectory analysis image result").get("image_id")): _mapping(item, "trajectory analysis image result")
        for item in raw_results
    }
    if len(image_results) != len(raw_results):
        raise AssemblyError("trajectory analysis contains duplicate image IDs")
    training_ids = split_ids["train_candidate"]
    missing_analysis = sorted(training_ids - set(image_results), key=int)
    if missing_analysis:
        raise AssemblyError(f"trajectory analysis lacks training images: {missing_analysis[:8]}")
    training_bands: dict[str, str] = {}
    annotations: dict[str, dict[str, Any]] = {}
    for image in training_ids:
        row = candidate_rows[image]
        objects = row.get("objects")
        if not isinstance(objects, list) or not objects:
            raise AssemblyError(f"candidate-pool training image {image} has no annotation objects")
        count = len(objects)
        training_bands[image] = (
            "sparse_1_to_3" if count <= 3 else "medium_4_to_7" if count <= 7
            else "dense_8_to_15" if count <= 15 else "very_dense_16_plus"
        )
        annotations[image] = row
    sampled_candidates, sampled_census = _build_multi_candidates(
        image_results=image_results,
        sampled_rows=sampled_rows,
        reference_records=reference_records,
        annotations=annotations,
        image_ids=sorted(training_ids, key=int),
        checkpoint_id=checkpoint_id,
    )
    source_candidates, source_census = _build_source_candidates(
        image_results=image_results,
        greedy_rows=greedy_rows,
        reference_records=reference_records,
        annotations=annotations,
        image_ids=sorted(training_ids, key=int),
        manual_review=None,
    )
    validate_constant_dose_training_reservoir(
        training_image_bands=training_bands,
        development_image_ids=sorted(split_ids["development"], key=int),
        heldout_image_ids=sorted(split_ids["heldout"], key=int),
        sampled_candidates=sampled_candidates,
        source_candidates=source_candidates,
    )
    selection = select_constant_dose_breadth_arms(
        sampled_candidates=sampled_candidates,
        source_candidates=source_candidates,
        training_image_bands=training_bands,
        trajectory_panel_execution_metadata=panel_receipt["execution_metadata"],
    )
    broad = materialize_constant_dose_breadth_arm(selection["broad"], checkpoint_id=checkpoint_id)
    concentrated = materialize_constant_dose_breadth_arm(selection["concentrated"], checkpoint_id=checkpoint_id)

    source_paths = [
        ("candidate-pool", pool_path), ("split-receipt", split_path),
        ("trajectory-analysis", analysis_path), ("trajectory-union-receipt", union_path),
        *[(f"old-greedy-{index:02d}", path) for index, path in enumerate(old_greedy_paths)],
        *[(f"new-greedy-{index:02d}", path) for index, path in enumerate(new_greedy_paths)],
        *[(f"old-sampled-{index:02d}", path) for index, path in enumerate(old_sampled_paths)],
        *[(f"new-sampled-{index:02d}", path) for index, path in enumerate(new_sampled_paths)],
    ]
    artifacts = _source_artifacts(source_paths)
    verified_identities = [
        {"artifact_path": str(path), "checkpoint_id": checkpoint_id, "checkpoint_identity": source_checkpoint.to_artifact_dict()}
        for path in [*all_greedy_paths, *all_sampled_paths]
    ]
    output.mkdir(parents=True)
    selection_document = _selection_receipt(selection)
    selection_document["source_artifacts"] = artifacts
    _write_json(output / "selection-receipt.json", selection_document)
    common_census = {
        "sampled_candidates": sampled_census,
        "source_candidates": source_census,
        "selection": selection_document,
    }
    arm_receipts: dict[str, Any] = {}
    for name, assembled in (("broad", broad), ("concentrated", concentrated)):
        root = output / f"{name}-plus-source-preservation"
        arm_receipts[name] = _write_arm(
            root=root,
            rollouts=assembled[0], reviews=assembled[1], receipts=assembled[2],
            census={**common_census, "arm": assembled[3]}, arm_receipt=assembled[3],
            source_checkpoint=source_checkpoint,
            prompt_identity_sha256=str(panel_receipt["prompt_policy_fingerprint"]),
            source_artifacts=artifacts,
            verified_rollout_checkpoint_identities=verified_identities,
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "assembled",
        "source_checkpoint": source_checkpoint.to_artifact_dict(),
        "source_checkpoint_id": checkpoint_id,
        "prompt_identity_sha256": panel_receipt["prompt_policy_fingerprint"],
        "execution_metadata": panel_receipt["execution_metadata"],
        "reference_derivation": {
            "policy": "exact_rollout_prompt_ids_plus_candidate_pool_image_and_split_metadata",
            "image_count": len(reference_records),
            "greedy_and_all_sampled_prompt_ids_cross_checked": True,
            "image_pad_interval": "canonical_single_contiguous_image_pad_run",
        },
        "selection_receipt": str((output / "selection-receipt.json").resolve()),
        "arms": {name: str((output / f"{name}-plus-source-preservation" / "assembly-receipt.json").resolve()) for name in arm_receipts},
        "source_artifacts": artifacts,
    }
    _write_json(output / "assembly-receipt.json", receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--split-receipt", type=Path, required=True)
    parser.add_argument("--trajectory-analysis", type=Path, required=True)
    parser.add_argument("--panel-union-receipt", type=Path, required=True)
    for name in ("old-greedy", "new-greedy", "old-sampled", "new-sampled"):
        parser.add_argument(f"--{name}", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = assemble_constant_dose_breadth_state_banks(
            candidate_pool=args.candidate_pool,
            split_receipt=args.split_receipt,
            trajectory_analysis=args.trajectory_analysis,
            panel_union_receipt=args.panel_union_receipt,
            old_greedy=args.old_greedy,
            new_greedy=args.new_greedy,
            old_sampled=args.old_sampled,
            new_sampled=args.new_sampled,
            output_dir=args.output_dir,
        )
    except (AssemblyError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"status": receipt["status"], "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
