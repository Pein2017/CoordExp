#!/usr/bin/env python3
"""Extend the frozen image-7816 coordinate-only branch pair to termination."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_local_branch_causal_value import (  # noqa: E402
    _single_native_inputs,
    build_positive_entity_ledger,
    git_execution_identity,
    hash_prefix_token_ids,
    sha256_file,
    validate_frozen_file_identity,
)
from scripts.research.run_sampled_history_prefix_sufficiency_ladder import (  # noqa: E402
    _build_request,
    _load_json,
    _select_example,
    summarize_continuation,
)
from scripts.research.run_sampled_history_target_reachability import (  # noqa: E402
    compare_execution_identity,
)
from scripts.research.run_same_parent_complete_row_intervention import (  # noqa: E402
    _canonical_hash,
    _generate_prefix_continuation,
    _ids,
    _owner_set,
    compare_exact_rows,
    compare_stable_model_identity,
    load_stage_three_sources,
    prepare_candidate_source,
)


SCHEMA_VERSION = "sampled_history_target_reachability.stage_four.v1"
PHASE = "stage_four_same_parent_final_horizon"
UNIT_ID = "2026-07-19-sampled-history-target-reachability-and-complete-row-value"


class StageFourValidationError(ValueError):
    """Raised when frozen sources cannot support the terminal comparison."""


def compare_initial_rows(
    frozen_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require the replay to reproduce every frozen initial row exactly."""

    checks: list[dict[str, Any]] = []
    for index, frozen in enumerate(frozen_rows):
        if index >= len(replay_rows):
            checks.append({"index": index, "passed": False, "reason": "replay_too_short"})
            continue
        checks.append({"index": index, **compare_exact_rows(frozen, replay_rows[index])})
    return {
        "passed": bool(checks) and all(item.get("passed") is True for item in checks),
        "frozen_row_count": len(frozen_rows),
        "replay_row_count": len(replay_rows),
        "checks": checks,
    }


def compare_all_rows(
    frozen_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require exact row count in addition to exact row content."""

    result = compare_initial_rows(frozen_rows, replay_rows)
    result["equal_row_count"] = len(frozen_rows) == len(replay_rows)
    result["passed"] = bool(result["passed"] and result["equal_row_count"])
    return result


def classify_final_comparison(
    *,
    target_owner_id: str,
    native_summary: Mapping[str, Any],
    sampled_summary: Mapping[str, Any],
    native_owner_ids: Sequence[str],
    sampled_owner_ids: Sequence[str],
    all_identity_and_parity_gates_passed: bool,
) -> dict[str, Any]:
    """Classify final set value using only raw-row-derived summaries."""

    native = {str(value) for value in native_owner_ids}
    sampled = {str(value) for value in sampled_owner_ids}
    target = str(target_owner_id)
    gained = sorted(sampled - native)
    lost = sorted(native - sampled)
    both_terminal = (
        native_summary.get("natural_terminal_row_index") is not None
        and sampled_summary.get("natural_terminal_row_index") is not None
    )
    either_censored = bool(
        (
            native_summary.get("budget_exhausted")
            and native_summary.get("natural_terminal_row_index") is None
        )
        or (
            sampled_summary.get("budget_exhausted")
            and sampled_summary.get("natural_terminal_row_index") is None
        )
    )
    safety_fields = (
        "unresolved_row_indices",
        "malformed_row_indices",
        "crop_review_warnings",
    )
    safety_clean = all(
        not summary.get(field)
        for summary in (native_summary, sampled_summary)
        for field in safety_fields
    )
    sampled_duplicate_free = not sampled_summary.get("duplicate_owner_ids")

    if not all_identity_and_parity_gates_passed:
        classification = "unresolved"
    elif either_censored:
        classification = "right_censored"
    elif not both_terminal:
        classification = "unresolved"
    elif not safety_clean:
        classification = "unresolved"
    elif target in native or native == sampled:
        classification = "same_final_set_or_timing"
    elif target in sampled and lost:
        classification = "sampled_final_exchange"
    elif (
        target in sampled
        and not lost
        and target in gained
        and sampled_duplicate_free
    ):
        classification = "sampled_final_gain"
    else:
        classification = "unresolved"

    return {
        "classification": classification,
        "claim_allowed": bool(
            all_identity_and_parity_gates_passed
            and both_terminal
            and safety_clean
            and classification
            in {"sampled_final_gain", "same_final_set_or_timing", "sampled_final_exchange"}
        ),
        "both_arms_naturally_terminated": both_terminal,
        "safety_clean": safety_clean,
        "sampled_duplicate_free": sampled_duplicate_free,
        "native_unique_owner_ids": sorted(native),
        "sampled_unique_owner_ids": sorted(sampled),
        "gained_owner_ids": gained,
        "lost_owner_ids": lost,
        "target_retrieved_native": target in native,
        "target_retrieved_sampled": target in sampled,
        "unique_owner_difference": len(sampled) - len(native),
    }


def load_stage_four_source(admission_path: Path) -> dict[str, Any]:
    """Rehash and validate every frozen Stage 4 input before model loading."""

    resolved = admission_path.expanduser().resolve(strict=True)
    admission = _load_json(resolved)
    if admission.get("schema_version") != 1 or admission.get("unit_id") != UNIT_ID:
        raise StageFourValidationError("Stage 4 admission schema or unit mismatch")

    path_fields = {
        "stage_three_union": ("source_stage_three_union", "source_stage_three_union_sha256"),
        "stage_three_admission": ("source_stage_three_admission", "source_stage_three_admission_sha256"),
        "stage_three_shard": ("source_stage_three_shard", "source_stage_three_shard_sha256"),
        "stage_one_shard": ("source_stage_one_shard", "source_stage_one_shard_sha256"),
    }
    paths: dict[str, Path] = {}
    for label, (path_key, hash_key) in path_fields.items():
        path = Path(str(admission.get(path_key, ""))).expanduser().resolve(strict=True)
        if sha256_file(path) != str(admission.get(hash_key, "")):
            raise StageFourValidationError(f"{label} SHA-256 disagrees with Stage 4 admission")
        paths[label] = path

    contract = admission.get("execution_contract")
    if not isinstance(contract, Mapping) or str(contract.get("image_id")) != "7816":
        raise StageFourValidationError("Stage 4 admits only image 7816")
    branch_count = int(contract.get("branch_row_token_count", -1))
    total_budget = int(contract.get("total_trajectory_generated_token_budget", -1))
    suffix_budget = int(contract.get("post_branch_generated_token_budget", -1))
    if branch_count != 9 or total_budget != 512 or suffix_budget != total_budget - branch_count:
        raise StageFourValidationError("Stage 4 token-budget accounting is inconsistent")

    stage_three_union = _load_json(paths["stage_three_union"])
    if stage_three_union.get("passed") is not True or int(stage_three_union.get("image_count", -1)) != 3:
        raise StageFourValidationError("Stage 3 union is not the passed three-image union")
    union_record = next(
        (
            item
            for item in stage_three_union.get("images", [])
            if isinstance(item, Mapping) and str(item.get("image_id")) == "7816"
        ),
        None,
    )
    if not isinstance(union_record, Mapping):
        raise StageFourValidationError("Stage 3 union lacks image 7816")
    if (
        str(Path(str(union_record.get("path"))).expanduser().resolve(strict=True))
        != str(paths["stage_three_shard"])
        or str(union_record.get("sha256")) != sha256_file(paths["stage_three_shard"])
    ):
        raise StageFourValidationError("Stage 3 union image-7816 receipt disagrees")

    stage_three = _load_json(paths["stage_three_shard"])
    images = stage_three.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise StageFourValidationError("Stage 3 shard must contain one image")
    stage_three_image = dict(images[0])
    comparison = stage_three_image.get("comparison")
    if not isinstance(comparison, Mapping):
        raise StageFourValidationError("Stage 3 comparison is missing")
    required_stage_three_gates = (
        bool((comparison.get("structural_gate") or {}).get("passed")),
        bool((comparison.get("native_no_op_parity") or {}).get("passed")),
        bool((comparison.get("sampled_source_replay_parity") or {}).get("passed")),
        comparison.get("primary_causal_claim_allowed") is True,
        comparison.get("target_access_reproduced") is True,
        comparison.get("safe_unique_owner_eligible") is True,
        int(comparison.get("unique_owner_difference", 0)) == 1,
        list(comparison.get("gained_owner_ids", [])) == ["211764"],
        not comparison.get("lost_owner_ids"),
    )
    if not all(required_stage_three_gates):
        raise StageFourValidationError("Stage 3 image 7816 does not satisfy Stage 4 admission")

    stage_one = _load_json(paths["stage_one_shard"])
    stage_one_images = stage_one.get("images")
    if not isinstance(stage_one_images, list) or len(stage_one_images) != 1 or not isinstance(stage_one_images[0], Mapping):
        raise StageFourValidationError("Stage 1 shard must contain one image")
    stage_one_image = dict(stage_one_images[0])
    classification = stage_one_image.get("target_classification") or {}
    if classification.get("label") != "terminally_omitted" or int(classification.get("terminal_row_index", -1)) != 10:
        raise StageFourValidationError("Stage 1 native endpoint is not the frozen terminal omission")

    sources = load_stage_three_sources(paths["stage_three_admission"])
    prepared = prepare_candidate_source(sources, "7816")
    if _canonical_hash(prepared["native_branch"]) != _canonical_hash(stage_three_image.get("native_branch")):
        raise StageFourValidationError("native branch differs from Stage 3 shard")
    if _canonical_hash(prepared["sampled_branch"]) != _canonical_hash(stage_three_image.get("sampled_branch")):
        raise StageFourValidationError("sampled branch differs from Stage 3 shard")

    return {
        "admission_path": str(resolved),
        "admission_sha256": sha256_file(resolved),
        "admission": admission,
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": {key: sha256_file(value) for key, value in paths.items()},
        "stage_three_artifact": stage_three,
        "stage_three_image": stage_three_image,
        "stage_one_artifact": stage_one,
        "stage_one_image": stage_one_image,
        "stage_three_sources": sources,
        "prepared": prepared,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-admission", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    try:
        frozen = load_stage_four_source(args.stage4_admission)
    except (OSError, KeyError, TypeError, ValueError, StageFourValidationError) as exc:
        raise SystemExit(f"Stage 4 source validation failed before runtime: {exc}") from exc

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
    contract = frozen["admission"]["execution_contract"]
    if str(config.model.dtype) != "fp32":
        raise SystemExit(f"Stage 4 requires model.dtype=fp32, observed {config.model.dtype!r}")
    expected_config_fingerprint = str(frozen["stage_three_artifact"]["config"]["resolved_config_fingerprint"])
    if str(resolved.fingerprint) != expected_config_fingerprint:
        raise SystemExit("Stage 4 resolved config fingerprint disagrees with Stage 3")

    sources = frozen["stage_three_sources"]
    stage_two_artifact = sources["stage_two_shards"]["7816"]["artifact"]
    frozen_inputs = stage_two_artifact.get("frozen_inputs") or {}
    manifest_path = Path(str(frozen_inputs.get("manifest", ""))).expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path)
    source_jsonl = Path(
        str((manifest.get("inference_contract") or {}).get("source_jsonl", config.data.input_jsonl))
    ).expanduser().resolve(strict=True)
    frozen_file_identity = validate_frozen_file_identity(
        manifest,
        infer_config_path=args.infer_config,
        source_jsonl_path=source_jsonl,
    )
    examples = list(load_raw_examples(source_jsonl))
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    prepared = frozen["prepared"]
    stage_three_image = frozen["stage_three_image"]
    stage_one_image = frozen["stage_one_image"]
    example = _select_example(examples, "7816")
    request, plan, prompt_meta = _build_request(config, frontend, example)
    outputs: dict[str, Any]
    with open_backend_session(frontend.launch) as session:
        model_receipt = session.receipt.to_artifact_dict()
        model_identity_check = compare_stable_model_identity(
            model_receipt,
            frozen["stage_three_artifact"].get("model_identity") or {},
        )
        if not model_identity_check["passed"]:
            raise SystemExit("Stage 4 model identity disagrees with Stage 3")
        native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
        one_native = _single_native_inputs(native_inputs)
        execution_identity_check = compare_execution_identity(
            observed_prompt=prompt_meta,
            observed_runtime={
                "executed_media_sha256": media_sha[0],
                "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]),
                "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
            },
            discovery_prompt=stage_three_image["prompt"]["prompt"],
            discovery_runtime={
                **dict(stage_three_image["runtime"]),
                "executed_prompt_token_ids_sha256": stage_three_image["prompt"]["prompt_token_ids_sha256"],
            },
        )
        if not execution_identity_check["passed"]:
            raise SystemExit("Stage 4 prompt/media/grid identity disagrees with Stage 3")
        ledger = [
            dict(row)
            for row in sources["stage_two_shards"]["7816"]["image"].get("entity_ledger", [])
            if isinstance(row, Mapping)
        ] or build_positive_entity_ledger(example)
        branch_owner_ids = sorted(
            set(prepared["prefix_owner_ids"]) | {str(contract["branch_owner_id"])}
        )
        common = {
            "session": session,
            "native_inputs": one_native,
            "prefix_owner_ids": branch_owner_ids,
            "tokenizer": session._tokenizer,
            "image_width": int(plan.decoded_width),
            "image_height": int(plan.decoded_height),
            "entity_ledger": ledger,
            "start_row_index": 1,
            "horizon_rows": int(contract["continuation_row_ceiling"]),
            "malformed_limit": int((manifest.get("discovery_budget") or {}).get("malformed_row_limit", 2)),
            "temperature": 0.4,
            "total_token_budget": int(contract["post_branch_generated_token_budget"]),
        }
        native_suffix = _generate_prefix_continuation(
            **common,
            prefix_token_ids=prepared["parent_token_ids"] + prepared["native_branch_token_ids"],
        )
        sampled_suffix = _generate_prefix_continuation(
            **common,
            prefix_token_ids=prepared["parent_token_ids"] + prepared["sampled_branch_token_ids"],
        )

    native_stage_three_parity = compare_initial_rows(
        stage_three_image["native_no_op_suffix"]["rows"], native_suffix["rows"]
    )
    sampled_stage_three_parity = compare_initial_rows(
        stage_three_image["sampled_row_intervention_suffix"]["rows"], sampled_suffix["rows"]
    )
    native_full_rows = [prepared["native_branch"], *native_suffix["rows"]]
    sampled_full_rows = [prepared["sampled_branch"], *sampled_suffix["rows"]]
    native_stage_one_parity = compare_all_rows(
        stage_one_image["extended_root_greedy"]["rows"], native_full_rows
    )
    all_parity = bool(
        execution_identity_check["passed"]
        and model_identity_check["passed"]
        and native_stage_three_parity["passed"]
        and sampled_stage_three_parity["passed"]
        and native_stage_one_parity["passed"]
    )
    target = str(contract["target_owner_id"])
    total_budget = int(contract["total_trajectory_generated_token_budget"])
    native_summary = summarize_continuation(
        native_full_rows,
        prefix_owner_ids=prepared["prefix_owner_ids"],
        target_owner_id=target,
        target_start_row_index=1,
        generated_token_count=len(prepared["native_branch_token_ids"]) + int(native_suffix["generated_token_count"]),
        total_token_budget=total_budget,
        horizon_rows_complete=False,
    )
    sampled_summary = summarize_continuation(
        sampled_full_rows,
        prefix_owner_ids=prepared["prefix_owner_ids"],
        target_owner_id=target,
        target_start_row_index=1,
        generated_token_count=len(prepared["sampled_branch_token_ids"]) + int(sampled_suffix["generated_token_count"]),
        total_token_budget=total_budget,
        horizon_rows_complete=False,
    )
    final_comparison = classify_final_comparison(
        target_owner_id=target,
        native_summary=native_summary,
        sampled_summary=sampled_summary,
        native_owner_ids=sorted(_owner_set(prepared["prefix_owner_ids"], native_full_rows)),
        sampled_owner_ids=sorted(_owner_set(prepared["prefix_owner_ids"], sampled_full_rows)),
        all_identity_and_parity_gates_passed=all_parity,
    )

    source_identity = git_execution_identity(Path(__file__).resolve().parents[2])
    source_identity.pop("runner_sha256", None)
    source_identity["stage_four_runner_sha256"] = sha256_file(Path(__file__).resolve())
    outputs = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "sampled_history_target_reachability",
        "phase": PHASE,
        "source_identity": source_identity,
        "frozen_inputs": {
            "stage_four_admission": frozen["admission_path"],
            "stage_four_admission_sha256": frozen["admission_sha256"],
            **{
                key: {"path": frozen["paths"][key], "sha256": frozen["hashes"][key]}
                for key in frozen["paths"]
            },
        },
        "config": {
            "infer_config": str(args.infer_config.expanduser().resolve()),
            "resolved_config_fingerprint": resolved.fingerprint,
            "device": args.device,
            "physical_batch_size": 1,
            "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "total_trajectory_generated_token_budget": total_budget,
            "post_branch_generated_token_budget": int(contract["post_branch_generated_token_budget"]),
        },
        "model_identity": model_receipt,
        "model_identity_check": model_identity_check,
        "frozen_file_identity": frozen_file_identity,
        "execution_identity_check": execution_identity_check,
        "image": {
            "image_id": "7816",
            "target_owner_id": target,
            "branch_owner_id": str(contract["branch_owner_id"]),
            "native_branch": prepared["native_branch"],
            "sampled_branch": prepared["sampled_branch"],
            "native_arm": {
                "suffix": native_suffix,
                "summary": native_summary,
                "stage_three_initial_rows_parity": native_stage_three_parity,
                "stage_one_full_trajectory_parity": native_stage_one_parity,
            },
            "sampled_arm": {
                "suffix": sampled_suffix,
                "summary": sampled_summary,
                "stage_three_initial_rows_parity": sampled_stage_three_parity,
            },
            "all_identity_and_parity_gates_passed": all_parity,
            "final_comparison": final_comparison,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(outputs, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
