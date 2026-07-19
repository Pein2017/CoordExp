#!/usr/bin/env python3
"""Validate a union of per-image Stage 2 prefix-ladder artifacts."""

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

from scripts.research.run_sampled_history_target_reachability import StageOneValidationError, _load_json
from scripts.research.run_sampled_history_prefix_sufficiency_ladder import (
    PHASE,
    SCHEMA_VERSION,
    load_stage_two_admission,
    sha256_file,
)


ALLOWED_STATES = {"hit", "clean_miss", "right_censored", "unresolved"}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _scientific_config(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("device", None)
    return result


def _validate_image(image: Mapping[str, Any], path: Path) -> str:
    image_id = str(image.get("image_id", ""))
    if not image_id:
        raise StageOneValidationError(f"Stage 2 image lacks image_id: {path}")
    entries = image.get("prefix_evaluations")
    if not isinstance(entries, list) or not entries:
        raise StageOneValidationError(f"Stage 2 image lacks prefix evaluations: {path}")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise StageOneValidationError(f"Stage 2 prefix evaluation is not an object: {path}")
        if entry.get("reachability_state") not in ALLOWED_STATES:
            raise StageOneValidationError(f"Stage 2 has invalid reachability state: {path}")
        prefix = entry.get("prefix")
        if not isinstance(prefix, Mapping) or not isinstance(prefix.get("prefix_token_ids_sha256"), str):
            raise StageOneValidationError(f"Stage 2 prefix identity is missing: {path}")
        continuation = entry.get("continuation")
        if not isinstance(continuation, Mapping) or not isinstance(continuation.get("rows"), list):
            raise StageOneValidationError(f"Stage 2 continuation is missing: {path}")
    admission_entry = image.get("stage2_admission_entry")
    if not isinstance(admission_entry, Mapping):
        raise StageOneValidationError(f"Stage 2 image lacks exact admission entry: {path}")
    return image_id


def validate_union(*, shard_paths: Sequence[Path], admission_path: Path) -> dict[str, Any]:
    """Validate a per-image union against the frozen Stage 2 admission cohort."""

    if not shard_paths:
        raise StageOneValidationError("Stage 2 union requires at least one shard")
    admission = load_stage_two_admission(admission_path)
    admitted_entries = admission["selected_entries"]
    admitted_ids = set(admitted_entries)
    seen: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for raw_path in shard_paths:
        path = raw_path.expanduser().resolve(strict=True)
        artifact = _load_json(path)
        if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("phase") != PHASE:
            raise StageOneValidationError(f"invalid Stage 2 shard: {path}")
        images = artifact.get("images")
        frozen = artifact.get("frozen_inputs")
        model_identity = artifact.get("model_identity")
        config = artifact.get("config")
        source_identity = artifact.get("source_identity")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
            raise StageOneValidationError(f"Stage 2 shard must contain one image: {path}")
        if not all(isinstance(value, Mapping) for value in (frozen, model_identity, config, source_identity)):
            raise StageOneValidationError(f"Stage 2 shard lacks identity receipt: {path}")
        image = images[0]
        image_id = _validate_image(image, path)
        if image_id in seen:
            raise StageOneValidationError(f"duplicate Stage 2 image: {image_id}")
        if image_id not in admitted_entries:
            raise StageOneValidationError(f"Stage 2 image is not admitted for the prefix ladder: {image_id}")
        image_admission = image["stage2_admission_entry"]
        if _canonical_hash(image_admission) != _canonical_hash(admitted_entries[image_id]):
            raise StageOneValidationError(f"Stage 2 image admission entry disagrees with frozen policy: {image_id}")
        required_frozen = {
            "stage_two_admission": admission["path"],
            "stage_two_admission_sha256": admission["sha256"],
            "stage_one_union": admission["stage_one_union_path"],
            "stage_one_union_sha256": admission["stage_one_union_sha256"],
        }
        for field, expected in required_frozen.items():
            if str(frozen.get(field, "")) != str(expected):
                raise StageOneValidationError(f"Stage 2 shard frozen input {field} disagrees with admission: {path}")
        frozen_entries = frozen.get("stage2_admission_entries")
        if not isinstance(frozen_entries, Mapping) or set(str(key) for key in frozen_entries) != admitted_ids:
            raise StageOneValidationError(f"Stage 2 shard frozen admission cohort is incomplete: {path}")
        for admitted_id, expected_entry in admitted_entries.items():
            if _canonical_hash(frozen_entries.get(admitted_id)) != _canonical_hash(expected_entry):
                raise StageOneValidationError(f"Stage 2 shard frozen admission entry disagrees for image {admitted_id}: {path}")
        source_fields = {
            "stage_two_runner_sha256": source_identity.get("stage_two_runner_sha256"),
            "reused_local_branch_helper_sha256": source_identity.get("reused_local_branch_helper_sha256"),
        }
        if any(not isinstance(value, str) or not value for value in source_fields.values()):
            raise StageOneValidationError(f"Stage 2 shard lacks runner/helper identity: {path}")
        normalized = {
            "frozen_inputs": frozen,
            "model_identity": model_identity,
            "config": _scientific_config(config),
            "source_identity": source_fields,
        }
        if reference is None:
            reference = normalized
        elif any(_canonical_hash(normalized[key]) != _canonical_hash(reference[key]) for key in normalized):
            raise StageOneValidationError(f"Stage 2 shard identity conflict: {path}")
        seen[image_id] = {"image_id": image_id, "path": str(path), "sha256": sha256_file(path)}
    if set(seen) != admitted_ids:
        raise StageOneValidationError(
            "Stage 2 union does not exactly cover the admitted cohort: "
            f"missing={sorted(admitted_ids - set(seen))} extra={sorted(set(seen) - admitted_ids)}"
        )
    return {
        "schema_version": "sampled_history_target_reachability.stage_two.union.v1",
        "experiment": "sampled_history_target_reachability",
        "phase": PHASE,
        "image_count": len(seen),
        "images": sorted(seen.values(), key=lambda item: item["image_id"]),
        "stage_two_admission": admission["path"],
        "stage_two_admission_sha256": admission["sha256"],
        "stage_one_union": admission["stage_one_union_path"],
        "stage_one_union_sha256": admission["stage_one_union_sha256"],
        "shared_frozen_inputs_sha256": _canonical_hash(reference["frozen_inputs"]) if reference else None,
        "shared_model_identity_sha256": _canonical_hash(reference["model_identity"]) if reference else None,
        "shared_config_sha256": _canonical_hash(reference["config"]) if reference else None,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-glob", action="append", default=[])
    parser.add_argument("--stage-two-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(item) for pattern in args.shard_glob for item in glob.glob(pattern)]
    try:
        result = validate_union(shard_paths=paths, admission_path=args.stage_two_admission)
    except (OSError, StageOneValidationError) as exc:
        raise SystemExit(f"Stage 2 union validation failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
