#!/usr/bin/env python3
"""Run the frozen image-7816 row-0 by row-4 coordinate-history crossover."""

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
    _row_is_clean_complete,
    _select_example,
    summarize_continuation,
)
from scripts.research.run_sampled_history_target_reachability import (  # noqa: E402
    compare_execution_identity,
)
from scripts.research.run_same_parent_complete_row_intervention import (  # noqa: E402
    _generate_prefix_continuation,
    _owner_set,
    compare_branch_structure,
    compare_stable_model_identity,
)
from scripts.research.run_same_parent_final_horizon import (  # noqa: E402
    UNIT_ID,
    load_stage_four_source,
)


SCHEMA_VERSION = "sampled_history_target_reachability.stage_five.v1"
PHASE = "stage_five_coordinate_history_crossover"


class StageFiveValidationError(ValueError):
    """Raised when the frozen crossover cannot support a causal comparison."""


def _strict_owner(row: Mapping[str, Any], *, label: str) -> str:
    if not _row_is_clean_complete(row):
        raise StageFiveValidationError(f"{label} is not a clean complete row")
    owners = {str(value) for value in row.get("strict_matched_owner_ids", [])}
    if len(owners) != 1:
        raise StageFiveValidationError(f"{label} does not have one strict owner")
    return next(iter(owners))


def _flatten_row_ids(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(token_id) for row in rows for token_id in row.get("raw_generated_token_ids", [])]


def _first_suffix_owner(continuation: Mapping[str, Any]) -> dict[str, Any]:
    rows = continuation.get("rows")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
        return {"interpretable": False, "owner_id": None, "reason": "missing_first_suffix_row"}
    row = rows[0]
    if not _row_is_clean_complete(row):
        return {"interpretable": False, "owner_id": None, "reason": "first_suffix_row_not_clean"}
    owners = sorted({str(value) for value in row.get("strict_matched_owner_ids", [])})
    matches = row.get("entity_matches")
    all_matched = isinstance(matches, list) and bool(matches) and all(
        isinstance(match, Mapping) and match.get("status") == "matched" for match in matches
    )
    if len(owners) != 1 or not all_matched:
        return {
            "interpretable": False,
            "owner_id": None,
            "reason": "first_suffix_owner_not_unique_and_matched",
            "strict_owner_ids": owners,
        }
    return {
        "interpretable": True,
        "owner_id": owners[0],
        "row_index": int(row.get("row_index", -1)),
        "raw_generated_token_ids_sha256": row.get("raw_generated_token_ids_sha256"),
    }


def classify_crossover(
    *,
    sampled_zero_native_four_owner: str | None,
    native_zero_sampled_four_owner: str | None,
    target_owner_id: str,
    duplicate_owner_id: str,
    gates_passed: bool,
) -> dict[str, Any]:
    """Classify the two new cells of the frozen two-by-two crossover."""

    left = None if sampled_zero_native_four_owner is None else str(sampled_zero_native_four_owner)
    right = None if native_zero_sampled_four_owner is None else str(native_zero_sampled_four_owner)
    target = str(target_owner_id)
    duplicate = str(duplicate_owner_id)
    if not gates_passed or left is None or right is None:
        classification = "other_or_unresolved"
    elif left == target and right == duplicate:
        classification = "persistent_row_zero_effect"
    elif left == duplicate and right == target:
        classification = "row_four_sufficient_or_mediating"
    elif left == target and right == target:
        classification = "either_coordinate_state_sufficient"
    elif left == duplicate and right == duplicate:
        classification = "joint_interaction"
    else:
        classification = "other_or_unresolved"
    return {
        "classification": classification,
        "claim_allowed": bool(gates_passed and classification != "other_or_unresolved"),
        "sampled_row_zero__native_row_four_owner_id": left,
        "native_row_zero__sampled_row_four_owner_id": right,
        "target_owner_id": target,
        "native_duplicate_owner_id": duplicate,
    }


def load_stage_five_source(admission_path: Path) -> dict[str, Any]:
    """Rehash the Stage 5 admission and construct all four exact prefixes."""

    resolved = admission_path.expanduser().resolve(strict=True)
    admission = _load_json(resolved)
    if admission.get("schema_version") != 1 or admission.get("unit_id") != UNIT_ID:
        raise StageFiveValidationError("Stage 5 admission schema or unit mismatch")
    paths: dict[str, Path] = {}
    for label, path_key, hash_key in (
        ("stage_three_shard", "source_stage_three_shard", "source_stage_three_shard_sha256"),
        ("stage_four_admission", "source_stage_four_admission", "source_stage_four_admission_sha256"),
        ("stage_four_artifact", "source_stage_four_artifact", "source_stage_four_artifact_sha256"),
    ):
        path = Path(str(admission.get(path_key, ""))).expanduser().resolve(strict=True)
        if sha256_file(path) != str(admission.get(hash_key, "")):
            raise StageFiveValidationError(f"{label} SHA-256 disagrees with Stage 5 admission")
        paths[label] = path

    contract = admission.get("execution_contract")
    if not isinstance(contract, Mapping) or str(contract.get("image_id")) != "7816":
        raise StageFiveValidationError("Stage 5 admits only image 7816")
    prefix_tokens = int(contract.get("forced_prefix_token_count", -1))
    total_budget = int(contract.get("total_trajectory_generated_token_budget", -1))
    suffix_budget = int(contract.get("post_prefix_generated_token_budget", -1))
    if prefix_tokens != 45 or total_budget != 512 or suffix_budget != total_budget - prefix_tokens:
        raise StageFiveValidationError("Stage 5 token-budget accounting is inconsistent")

    stage_four_frozen = load_stage_four_source(paths["stage_four_admission"])
    stage_four = _load_json(paths["stage_four_artifact"])
    stage_four_image = stage_four.get("image")
    if not isinstance(stage_four_image, Mapping):
        raise StageFiveValidationError("Stage 4 artifact lacks image evidence")
    final_comparison = stage_four_image.get("final_comparison") or {}
    if (
        final_comparison.get("classification") != "sampled_final_gain"
        or final_comparison.get("claim_allowed") is not True
        or list(final_comparison.get("gained_owner_ids", [])) != ["211764"]
        or final_comparison.get("lost_owner_ids")
    ):
        raise StageFiveValidationError("Stage 4 did not establish the frozen final gain")

    stage_three = _load_json(paths["stage_three_shard"])
    images = stage_three.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise StageFiveValidationError("Stage 3 shard must contain one image")
    image = dict(images[0])
    native_rows = [image["native_branch"], *image["native_no_op_suffix"]["rows"]]
    sampled_rows = [image["sampled_branch"], *image["sampled_row_intervention_suffix"]["rows"]]
    if len(native_rows) < 6 or len(sampled_rows) < 6:
        raise StageFiveValidationError("Stage 3 trajectories are too short for the crossover")
    if any(
        native_rows[index].get("raw_generated_token_ids")
        != sampled_rows[index].get("raw_generated_token_ids")
        for index in (1, 2, 3)
    ):
        raise StageFiveValidationError("rows 1 through 3 are not raw-token identical")
    row_zero_structure = compare_branch_structure(
        native_rows[0]["raw_generated_token_ids"], sampled_rows[0]["raw_generated_token_ids"]
    )
    row_four_structure = compare_branch_structure(
        native_rows[4]["raw_generated_token_ids"], sampled_rows[4]["raw_generated_token_ids"]
    )
    if not row_zero_structure["passed"] or not row_four_structure["passed"]:
        raise StageFiveValidationError("row 0 or row 4 is not a coordinate-only variant")
    frozen_rows = admission.get("frozen_rows") or {}
    for key, row in (
        ("native_row_zero", native_rows[0]),
        ("sampled_row_zero", sampled_rows[0]),
        ("native_row_four", native_rows[4]),
        ("sampled_row_four", sampled_rows[4]),
    ):
        expected = frozen_rows.get(key) or {}
        if hash_prefix_token_ids(row.get("raw_generated_token_ids", [])) != str(expected.get("raw_row_sha256", "")):
            raise StageFiveValidationError(f"{key} raw row hash disagrees")
        if _strict_owner(row, label=key) != str(expected.get("owner_id")):
            raise StageFiveValidationError(f"{key} owner disagrees")

    prefixes = {
        "native_row_zero__native_row_four": native_rows[:5],
        "sampled_row_zero__native_row_four": [sampled_rows[0], *native_rows[1:5]],
        "native_row_zero__sampled_row_four": [native_rows[0], *sampled_rows[1:5]],
        "sampled_row_zero__sampled_row_four": sampled_rows[:5],
    }
    admission_prefixes = {
        str(item["arm"]): item for item in admission.get("factorial_prefixes", []) if isinstance(item, Mapping)
    }
    target = str(contract["target_owner_id"])
    for arm, rows in prefixes.items():
        if len(rows) != 5 or any(len(row.get("raw_generated_token_ids", [])) != 9 for row in rows):
            raise StageFiveValidationError(f"{arm} is not five complete nine-token rows")
        if any(not _row_is_clean_complete(row) for row in rows):
            raise StageFiveValidationError(f"{arm} contains a non-clean row")
        owners = {str(value) for row in rows for value in row.get("strict_matched_owner_ids", [])}
        if target in owners:
            raise StageFiveValidationError(f"{arm} already contains the target")
        token_ids = _flatten_row_ids(rows)
        if len(token_ids) != prefix_tokens:
            raise StageFiveValidationError(f"{arm} prefix token count disagrees")
        if hash_prefix_token_ids(token_ids) != str((admission_prefixes.get(arm) or {}).get("prefix_sha256", "")):
            raise StageFiveValidationError(f"{arm} prefix hash disagrees")

    endpoint_native_owner = _strict_owner(native_rows[5], label="native endpoint row 5")
    endpoint_sampled_owner = _strict_owner(sampled_rows[5], label="sampled endpoint row 5")
    if endpoint_native_owner != str(contract["native_duplicate_owner_id"]) or endpoint_sampled_owner != target:
        raise StageFiveValidationError("frozen endpoint row-5 owners disagree")

    return {
        "admission_path": str(resolved),
        "admission_sha256": sha256_file(resolved),
        "admission": admission,
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": {key: sha256_file(value) for key, value in paths.items()},
        "stage_four_frozen": stage_four_frozen,
        "stage_four_artifact": stage_four,
        "stage_three_artifact": stage_three,
        "stage_three_image": image,
        "prefixes": prefixes,
        "row_zero_structure": row_zero_structure,
        "row_four_structure": row_four_structure,
        "endpoint_owners": {
            "native_row_zero__native_row_four": endpoint_native_owner,
            "sampled_row_zero__sampled_row_four": endpoint_sampled_owner,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage5-admission", type=Path, required=True)
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
        frozen = load_stage_five_source(args.stage5_admission)
    except (OSError, KeyError, TypeError, ValueError, StageFiveValidationError) as exc:
        raise SystemExit(f"Stage 5 source validation failed before runtime: {exc}") from exc

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
    stage_four = frozen["stage_four_artifact"]
    if str(config.model.dtype) != "fp32":
        raise SystemExit(f"Stage 5 requires model.dtype=fp32, observed {config.model.dtype!r}")
    if str(resolved.fingerprint) != str(stage_four["config"]["resolved_config_fingerprint"]):
        raise SystemExit("Stage 5 resolved config fingerprint disagrees with Stage 4")

    sources = frozen["stage_four_frozen"]["stage_three_sources"]
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

    example = _select_example(examples, "7816")
    request, plan, prompt_meta = _build_request(config, frontend, example)
    cross_arm_names = (
        "sampled_row_zero__native_row_four",
        "native_row_zero__sampled_row_four",
    )
    arm_results: dict[str, Any] = {}
    with open_backend_session(frontend.launch) as session:
        model_receipt = session.receipt.to_artifact_dict()
        model_identity_check = compare_stable_model_identity(
            model_receipt, stage_four.get("model_identity") or {}
        )
        if not model_identity_check["passed"]:
            raise SystemExit("Stage 5 model identity disagrees with Stage 4")
        native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
        one_native = _single_native_inputs(native_inputs)
        execution_identity_check = compare_execution_identity(
            observed_prompt=prompt_meta,
            observed_runtime={
                "executed_media_sha256": media_sha[0],
                "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]),
                "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
            },
            discovery_prompt=frozen["stage_three_image"]["prompt"]["prompt"],
            discovery_runtime={
                **dict(frozen["stage_three_image"]["runtime"]),
                "executed_prompt_token_ids_sha256": frozen["stage_three_image"]["prompt"]["prompt_token_ids_sha256"],
            },
        )
        if not execution_identity_check["passed"]:
            raise SystemExit("Stage 5 prompt/media/grid identity disagrees")
        ledger = [
            dict(row)
            for row in sources["stage_two_shards"]["7816"]["image"].get("entity_ledger", [])
            if isinstance(row, Mapping)
        ] or build_positive_entity_ledger(example)
        for arm in cross_arm_names:
            prefix_rows = frozen["prefixes"][arm]
            prefix_ids = _flatten_row_ids(prefix_rows)
            prefix_owners = sorted(
                {str(value) for row in prefix_rows for value in row.get("strict_matched_owner_ids", [])}
            )
            continuation = _generate_prefix_continuation(
                session=session,
                native_inputs=one_native,
                prefix_token_ids=prefix_ids,
                prefix_owner_ids=prefix_owners,
                tokenizer=session._tokenizer,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                entity_ledger=ledger,
                start_row_index=5,
                horizon_rows=int(contract["continuation_row_ceiling"]),
                malformed_limit=int((manifest.get("discovery_budget") or {}).get("malformed_row_limit", 2)),
                temperature=0.4,
                total_token_budget=int(contract["post_prefix_generated_token_budget"]),
            )
            full_rows = [*prefix_rows, *continuation["rows"]]
            summary = summarize_continuation(
                full_rows,
                prefix_owner_ids=[],
                target_owner_id=str(contract["target_owner_id"]),
                target_start_row_index=5,
                generated_token_count=len(prefix_ids) + int(continuation["generated_token_count"]),
                total_token_budget=int(contract["total_trajectory_generated_token_budget"]),
                horizon_rows_complete=False,
            )
            arm_results[arm] = {
                "prefix_token_ids": prefix_ids,
                "prefix_token_ids_sha256": hash_prefix_token_ids(prefix_ids),
                "prefix_owner_ids": prefix_owners,
                "continuation": continuation,
                "first_suffix_owner": _first_suffix_owner(continuation),
                "summary": summary,
                "final_unique_owner_ids": sorted(_owner_set([], full_rows)),
            }

    first_left = arm_results[cross_arm_names[0]]["first_suffix_owner"]
    first_right = arm_results[cross_arm_names[1]]["first_suffix_owner"]
    gates_passed = bool(
        model_identity_check["passed"]
        and execution_identity_check["passed"]
        and first_left.get("interpretable")
        and first_right.get("interpretable")
    )
    primary = classify_crossover(
        sampled_zero_native_four_owner=first_left.get("owner_id"),
        native_zero_sampled_four_owner=first_right.get("owner_id"),
        target_owner_id=str(contract["target_owner_id"]),
        duplicate_owner_id=str(contract["native_duplicate_owner_id"]),
        gates_passed=gates_passed,
    )
    source_identity = git_execution_identity(Path(__file__).resolve().parents[2])
    source_identity.pop("runner_sha256", None)
    source_identity["stage_five_runner_sha256"] = sha256_file(Path(__file__).resolve())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "sampled_history_target_reachability",
        "phase": PHASE,
        "source_identity": source_identity,
        "frozen_inputs": {
            "stage_five_admission": frozen["admission_path"],
            "stage_five_admission_sha256": frozen["admission_sha256"],
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
            "total_trajectory_generated_token_budget": int(contract["total_trajectory_generated_token_budget"]),
            "post_prefix_generated_token_budget": int(contract["post_prefix_generated_token_budget"]),
        },
        "model_identity": model_receipt,
        "model_identity_check": model_identity_check,
        "frozen_file_identity": frozen_file_identity,
        "execution_identity_check": execution_identity_check,
        "row_zero_structure": frozen["row_zero_structure"],
        "row_four_structure": frozen["row_four_structure"],
        "endpoint_first_suffix_owners": frozen["endpoint_owners"],
        "cross_arms": arm_results,
        "primary_crossover_result": primary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
