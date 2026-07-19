#!/usr/bin/env python3
"""Validate the exact three-shard Stage 3 intervention union."""

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

from scripts.research.run_sampled_history_prefix_sufficiency_ladder import StageTwoValidationError, _load_json  # noqa: E402
from scripts.research.run_same_parent_complete_row_intervention import (  # noqa: E402
    PHASE,
    REQUIRED_CANDIDATE_COUNT,
    SCHEMA_VERSION,
    _canonical_hash,
    compare_continuation_parity,
    compare_execution_identity,
    compare_stable_model_identity,
    load_stage_three_sources,
    prepare_candidate_source,
    summarize_branch_comparison,
    sha256_file,
)


def _scientific_config(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("device", None)
    return result


def _validate_image(
    image: Mapping[str, Any],
    path: Path,
    candidate: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    prepared: Mapping[str, Any],
    expected_model_identity: Mapping[str, Any],
    observed_model_identity: Mapping[str, Any],
    recorded_model_check: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    image_id = str(image.get("image_id", ""))
    if image_id != str(candidate.get("image_id")):
        raise StageTwoValidationError(f"Stage 3 image/candidate identity mismatch: {path}")
    comparison = image.get("comparison")
    if not isinstance(comparison, Mapping):
        raise StageTwoValidationError(f"Stage 3 image lacks comparison: {path}")
    structural = comparison.get("structural_gate")
    native_parity = comparison.get("native_no_op_parity")
    sampled_parity = comparison.get("sampled_source_replay_parity")
    if not isinstance(structural, Mapping) or structural.get("passed") is not True:
        raise StageTwoValidationError(f"Stage 3 structural gate failed: {path}")
    if not isinstance(native_parity, Mapping) or not isinstance(sampled_parity, Mapping):
        raise StageTwoValidationError(f"Stage 3 replay parity gates are missing: {path}")
    frozen_native = image.get("frozen_native_continuation")
    frozen_sampled = image.get("frozen_sampled_continuation")
    native_suffix = image.get("native_no_op_suffix")
    sampled_suffix = image.get("sampled_row_intervention_suffix")
    if not all(isinstance(value, Mapping) for value in (frozen_native, frozen_sampled, native_suffix, sampled_suffix)):
        raise StageTwoValidationError(f"Stage 3 continuation artifacts are missing: {path}")
    if _canonical_hash(frozen_native) != _canonical_hash(prepared["frozen_native_continuation"]):
        raise StageTwoValidationError(f"Stage 3 frozen native continuation drift: {path}")
    if _canonical_hash(frozen_sampled) != _canonical_hash(prepared["frozen_sampled_continuation"]):
        raise StageTwoValidationError(f"Stage 3 frozen sampled continuation drift: {path}")
    frozen_sampled_eval = image.get("frozen_sampled_prefix_evaluation")
    if not isinstance(frozen_sampled_eval, Mapping) or _canonical_hash(frozen_sampled_eval) != _canonical_hash(prepared["frozen_sampled_prefix_evaluation"]):
        raise StageTwoValidationError(f"Stage 3 frozen sampled prefix evaluation drift: {path}")
    direct_rows = image.get("direct_native_continuation_rows")
    if not isinstance(direct_rows, list) or _canonical_hash(direct_rows) != _canonical_hash(prepared["direct_continuation_rows"]):
        raise StageTwoValidationError(f"Stage 3 direct native continuation drift: {path}")
    recomputed_native = compare_continuation_parity(
        prepared["frozen_native_continuation"],
        native_suffix,
        frozen_includes_branch=True,
        branch_token_count=len(prepared["native_branch_token_ids"]),
    )
    recomputed_sampled = compare_continuation_parity(
        prepared["frozen_sampled_continuation"],
        sampled_suffix,
        frozen_includes_branch=False,
        branch_token_count=len(prepared["sampled_branch_token_ids"]),
    )
    if _canonical_hash(dict(native_parity)) != _canonical_hash(recomputed_native):
        raise StageTwoValidationError(f"Stage 3 native parity receipt was not recomputed from raw content: {path}")
    if _canonical_hash(dict(sampled_parity)) != _canonical_hash(recomputed_sampled):
        raise StageTwoValidationError(f"Stage 3 sampled parity receipt was not recomputed from raw content: {path}")
    recomputed_comparison = summarize_branch_comparison(
        prepared,
        native_suffix=native_suffix,
        sampled_suffix=sampled_suffix,
        native_no_op_parity=recomputed_native,
        sampled_source_replay_parity=recomputed_sampled,
    )
    if _canonical_hash(dict(comparison)) != _canonical_hash(recomputed_comparison):
        raise StageTwoValidationError(f"Stage 3 comparison was not recomputed from raw content: {path}")
    if not recomputed_comparison.get("primary_causal_claim_allowed"):
        raise StageTwoValidationError(f"Stage 3 primary claim is refused: {path}")
    if image.get("structural_gate") is not None and _canonical_hash(image["structural_gate"]) != _canonical_hash(prepared["structural_gate"]):
        raise StageTwoValidationError(f"Stage 3 top-level structural gate drift: {path}")
    parent = image.get("parent")
    if not isinstance(parent, Mapping) or list(parent.get("token_ids", [])) != list(prepared["parent_token_ids"]):
        raise StageTwoValidationError(f"Stage 3 parent prefix drift: {path}")
    for field, prepared_key in (("native_branch", "native_branch"), ("sampled_branch", "sampled_branch")):
        branch = image.get(field)
        if not isinstance(branch, Mapping) or _canonical_hash(branch) != _canonical_hash(prepared[prepared_key]):
            raise StageTwoValidationError(f"Stage 3 {field} drift: {path}")
    identity = image.get("identity_check")
    prompt_container = image.get("prompt")
    runtime = image.get("runtime")
    if not isinstance(identity, Mapping) or identity.get("passed") is not True:
        raise StageTwoValidationError(f"Stage 3 execution identity receipt failed: {path}")
    if not isinstance(prompt_container, Mapping) or not isinstance(prompt_container.get("prompt"), Mapping) or not isinstance(runtime, Mapping):
        raise StageTwoValidationError(f"Stage 3 prompt/runtime receipt is missing: {path}")
    expected_stage_image = source["stage_two_shards"][image_id]["image"]
    recomputed_identity = compare_execution_identity(
        observed_prompt=prompt_container["prompt"],
        observed_runtime={
            **dict(runtime),
            "executed_prompt_token_ids_sha256": prompt_container.get("prompt_token_ids_sha256"),
        },
        discovery_prompt=expected_stage_image["prompt"],
        discovery_runtime=expected_stage_image["runtime"],
    )
    if _canonical_hash(dict(identity)) != _canonical_hash(recomputed_identity):
        raise StageTwoValidationError(f"Stage 3 execution identity was not recomputed from raw content: {path}")
    model_check = compare_stable_model_identity(observed_model_identity, expected_model_identity)
    if not isinstance(recorded_model_check, Mapping) or _canonical_hash(dict(recorded_model_check)) != _canonical_hash(model_check):
        raise StageTwoValidationError(f"Stage 3 model identity receipt drift: {path}")
    return image_id, recomputed_comparison


def validate_union(*, shard_paths: Sequence[Path], admission_path: Path) -> dict[str, Any]:
    """Require exactly the three frozen candidates and shared identities."""

    if len(shard_paths) != REQUIRED_CANDIDATE_COUNT:
        raise StageTwoValidationError("Stage 3 union requires exactly three shards")
    sources = load_stage_three_sources(admission_path)
    candidates = sources["candidates"]
    if len(candidates) != REQUIRED_CANDIDATE_COUNT:
        raise StageTwoValidationError("Stage 3 admission candidate count is not three")
    prepared_sources = {
        image_id: prepare_candidate_source(sources, image_id)
        for image_id in candidates
    }
    expected_model_identity = sources["stage_two_shards"][next(iter(candidates))]["artifact"].get("model_identity") or {}
    if any(
        _canonical_hash(compare_stable_model_identity(
            record["artifact"].get("model_identity") or {}, expected_model_identity
        )["observed"])
        != _canonical_hash(compare_stable_model_identity(
            record["artifact"].get("model_identity") or {}, expected_model_identity
        )["expected"])
        for record in sources["stage_two_shards"].values()
    ):
        raise StageTwoValidationError("Stage 2 shards do not share one stable model identity")
    seen: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for raw_path in shard_paths:
        path = raw_path.expanduser().resolve(strict=True)
        artifact = _load_json(path)
        if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("phase") != PHASE:
            raise StageTwoValidationError(f"invalid Stage 3 shard: {path}")
        images = artifact.get("images")
        frozen = artifact.get("frozen_inputs")
        model_identity = artifact.get("model_identity")
        config = artifact.get("config")
        source_identity = artifact.get("source_identity")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
            raise StageTwoValidationError(f"Stage 3 shard must contain one image: {path}")
        if not all(isinstance(value, Mapping) for value in (frozen, model_identity, config, source_identity)):
            raise StageTwoValidationError(f"Stage 3 shard lacks identity receipt: {path}")
        image_id = str(images[0].get("image_id", ""))
        if image_id not in candidates:
            raise StageTwoValidationError(f"Stage 3 image is not a frozen candidate: {image_id}")
        image_id, _recomputed = _validate_image(
            images[0],
            path,
            candidates[image_id],
            source=sources,
            prepared=prepared_sources[image_id],
            expected_model_identity=expected_model_identity,
            observed_model_identity=model_identity,
            recorded_model_check=artifact.get("model_identity_check"),
        )
        if image_id in seen:
            raise StageTwoValidationError(f"duplicate Stage 3 image: {image_id}")
        entry = images[0].get("stage3_admission_entry")
        if not isinstance(entry, Mapping) or _canonical_hash(entry) != _canonical_hash(candidates[image_id]):
            raise StageTwoValidationError(f"Stage 3 admission entry disagrees: {image_id}")
        required_frozen = {
            "stage_three_admission": sources["admission_path"],
            "stage_three_admission_sha256": sources["admission_sha256"],
            "stage_two_union": sources["stage_two_union_path"],
            "stage_two_union_sha256": sources["stage_two_union_sha256"],
            "stage_two_admission": sources["stage_two_admission_path"],
            "stage_two_admission_sha256": sources["stage_two_admission_sha256"],
        }
        for field, expected in required_frozen.items():
            if str(frozen.get(field, "")) != str(expected):
                raise StageTwoValidationError(f"Stage 3 frozen input {field} disagrees: {path}")
        shard_digest = frozen.get("stage_two_shard_sha256")
        if not isinstance(shard_digest, Mapping) or dict(shard_digest) != {key: value for key, value in ((key, sources["stage_two_shards"][key]["sha256"]) for key in sorted(sources["stage_two_shards"]))}:
            raise StageTwoValidationError(f"Stage 3 frozen Stage 2 shard digest map disagrees: {path}")
        source_fields = {
            "stage_three_runner_sha256": source_identity.get("stage_three_runner_sha256"),
            "stage_two_runner_sha256": source_identity.get("stage_two_runner_sha256"),
        }
        if any(not isinstance(value, str) or not value for value in source_fields.values()):
            raise StageTwoValidationError(f"Stage 3 shard lacks runner identity: {path}")
        normalized = {
            "frozen_inputs": frozen,
            "model_identity": model_identity,
            "config": _scientific_config(config),
            "source_identity": source_fields,
        }
        if reference is None:
            reference = normalized
        elif any(_canonical_hash(normalized[key]) != _canonical_hash(reference[key]) for key in normalized):
            raise StageTwoValidationError(f"Stage 3 shard identity conflict: {path}")
        stage3_model_check = images[0].get("model_identity_check")
        if not isinstance(stage3_model_check, Mapping) or stage3_model_check.get("passed") is not True:
            raise StageTwoValidationError(f"Stage 3 model identity gate is missing: {path}")
        seen[image_id] = {"image_id": image_id, "path": str(path), "sha256": sha256_file(path)}
    if set(seen) != set(candidates):
        raise StageTwoValidationError(f"Stage 3 union does not exactly cover candidates: missing={sorted(set(candidates)-set(seen))} extra={sorted(set(seen)-set(candidates))}")
    return {
        "schema_version": "sampled_history_target_reachability.stage_three.union.v1",
        "experiment": "sampled_history_target_reachability",
        "phase": PHASE,
        "image_count": len(seen),
        "images": sorted(seen.values(), key=lambda item: item["image_id"]),
        "stage_three_admission": sources["admission_path"],
        "stage_three_admission_sha256": sources["admission_sha256"],
        "stage_two_union": sources["stage_two_union_path"],
        "stage_two_union_sha256": sources["stage_two_union_sha256"],
        "stage_two_admission": sources["stage_two_admission_path"],
        "stage_two_admission_sha256": sources["stage_two_admission_sha256"],
        "shared_frozen_inputs_sha256": _canonical_hash(reference["frozen_inputs"]) if reference else None,
        "shared_model_identity_sha256": _canonical_hash(reference["model_identity"]) if reference else None,
        "shared_config_sha256": _canonical_hash(reference["config"]) if reference else None,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-admission", type=Path, required=True)
    parser.add_argument("--shard-glob", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(item) for pattern in args.shard_glob for item in glob.glob(pattern)]
    try:
        result = validate_union(shard_paths=paths, admission_path=args.stage3_admission)
    except (OSError, StageTwoValidationError, ValueError) as exc:
        raise SystemExit(f"Stage 3 union validation failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
