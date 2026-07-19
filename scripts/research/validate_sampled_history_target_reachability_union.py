#!/usr/bin/env python3
"""Validate a complete Stage 1 per-image shard union."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.run_sampled_history_target_reachability import StageOneValidationError, _load_json, sha256_file


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _scientific_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the config fields that must agree across per-device shards."""

    result = dict(value)
    # Shards may be run on different CUDA ordinals; this is operational, not
    # a scientific configuration change.
    result.pop("device", None)
    return result


def validate_union(*, manifest_path: Path, shard_paths: Sequence[Path]) -> dict[str, Any]:
    """Require eight non-conflicting one-image outputs with shared identities."""

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path)
    expected_ids = {str(item.get("image_id")) for item in manifest.get("images", []) if isinstance(item, Mapping)}
    if len(expected_ids) != 8:
        raise StageOneValidationError("frozen manifest must contain exactly eight images")
    if len(shard_paths) != 8:
        raise StageOneValidationError("Stage 1 union requires exactly eight shard artifacts")
    manifest_sha256 = sha256_file(manifest_path)
    records: list[dict[str, Any]] = []
    seen_images: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for raw_path in shard_paths:
        path = raw_path.expanduser().resolve(strict=True)
        artifact = _load_json(path)
        if artifact.get("schema_version") != "sampled_history_target_reachability.v1" or artifact.get("phase") != "stage_one_extended_root_greedy_screen":
            raise StageOneValidationError(f"invalid Stage 1 shard: {path}")
        images = artifact.get("images")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
            raise StageOneValidationError(f"Stage 1 shard must contain one image: {path}")
        image_id = str(images[0].get("image_id"))
        if image_id not in expected_ids or image_id in seen_images:
            raise StageOneValidationError(f"duplicate or unknown Stage 1 image: {image_id}")
        frozen_inputs = artifact.get("frozen_inputs")
        model_identity = artifact.get("model_identity")
        config = artifact.get("config")
        source_identity = artifact.get("source_identity")
        if not isinstance(frozen_inputs, Mapping) or not isinstance(model_identity, Mapping) or not isinstance(config, Mapping) or not isinstance(source_identity, Mapping):
            raise StageOneValidationError(f"Stage 1 shard lacks identity receipt: {path}")
        if str(frozen_inputs.get("manifest_sha256", "")) != manifest_sha256:
            raise StageOneValidationError(f"Stage 1 shard manifest digest mismatch: {path}")
        source_identity_fields = {
            "stage_one_runner_sha256": source_identity.get("stage_one_runner_sha256"),
            "reused_local_branch_helper_sha256": source_identity.get("reused_local_branch_helper_sha256"),
        }
        if any(not isinstance(value, str) or not value for value in source_identity_fields.values()):
            raise StageOneValidationError(f"Stage 1 shard lacks explicit runner/helper identity: {path}")
        normalized = {
            "frozen_inputs": frozen_inputs,
            "model_identity": model_identity,
            "config": _scientific_config(config),
            "source_identity": source_identity_fields,
        }
        if reference is None:
            reference = normalized
        elif any(_canonical_hash(normalized[key]) != _canonical_hash(reference[key]) for key in normalized):
            raise StageOneValidationError(f"Stage 1 shard identity conflict: {path}")
        seen_images[image_id] = {"path": str(path), "sha256": sha256_file(path)}
        records.append({"image_id": image_id, **seen_images[image_id]})
    if set(seen_images) != expected_ids:
        raise StageOneValidationError(f"Stage 1 shard coverage mismatch: missing {sorted(expected_ids - set(seen_images))}")
    return {
        "schema_version": "sampled_history_target_reachability.union.v1",
        "experiment": "sampled_history_target_reachability",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "image_count": len(records),
        "images": sorted(records, key=lambda item: (int(item["image_id"]) if item["image_id"].isdigit() else item["image_id"])),
        "shared_frozen_inputs_sha256": _canonical_hash(reference["frozen_inputs"]) if reference else None,
        "shared_model_identity_sha256": _canonical_hash(reference["model_identity"]) if reference else None,
        "shared_config_sha256": _canonical_hash(reference["config"]) if reference else None,
        "shared_source_identity_sha256": _canonical_hash(reference["source_identity"]) if reference else None,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-glob", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(item) for pattern in args.shard_glob for item in glob.glob(pattern)]
    try:
        result = validate_union(manifest_path=args.manifest, shard_paths=paths)
    except (OSError, StageOneValidationError) as exc:
        raise SystemExit(f"Stage 1 union validation failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
