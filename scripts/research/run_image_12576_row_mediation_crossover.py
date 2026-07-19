#!/usr/bin/env python3
"""Run the frozen Stage 7 row-3/row-4 mediation crossover on image 12576."""

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
)
from scripts.research.run_sampled_history_prefix_sufficiency_ladder import (  # noqa: E402
    _build_request,
    _load_json,
    _row_is_clean_complete,
    _select_example,
)
from scripts.research.run_sampled_history_target_reachability import (  # noqa: E402
    compare_execution_identity,
)
from scripts.research.run_same_parent_complete_row_intervention import (  # noqa: E402
    _generate_prefix_continuation,
    compare_branch_structure,
    compare_stable_model_identity,
)
from scripts.research.run_same_parent_final_horizon import UNIT_ID  # noqa: E402


SCHEMA_VERSION = "sampled_history_target_reachability.stage_seven.v1"
PHASE = "stage_seven_image_12576_row_three_row_four_mediation_crossover"
EXPECTED_ARMS = {
    "native_row_three__native_row_four",
    "sampled_row_three__native_row_four",
    "native_row_three__sampled_row_four",
    "sampled_row_three__sampled_row_four",
}


class StageSevenValidationError(ValueError):
    """Raised when the frozen Stage 7 evidence cannot support execution."""


def _one_image(stage_three: Mapping[str, Any]) -> dict[str, Any]:
    images = stage_three.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], Mapping)
        or str(images[0].get("image_id")) != "12576"
    ):
        raise StageSevenValidationError("Stage 3 source must contain only image 12576")
    return dict(images[0])


def _row_ids(row: Mapping[str, Any], *, label: str) -> list[int]:
    values = row.get("raw_generated_token_ids")
    if not isinstance(values, list) or not values or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise StageSevenValidationError(f"{label} lacks integer raw token identifiers")
    return [int(value) for value in values]


def _strict_owner(row: Mapping[str, Any], *, label: str) -> str:
    owners = {str(value) for value in row.get("strict_matched_owner_ids", [])}
    if not _row_is_clean_complete(row) or len(owners) != 1:
        raise StageSevenValidationError(f"{label} is not a clean single-owner row")
    return next(iter(owners))


def _admitted_row(
    admission: Mapping[str, Any], name: str, source_row: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = admission.get("frozen_rows")
    if not isinstance(frozen, Mapping) or not isinstance(frozen.get(name), Mapping):
        raise StageSevenValidationError(f"admission lacks {name}")
    declared = dict(frozen[name])
    token_ids = declared.get("token_ids")
    if not isinstance(token_ids, list) or any(not isinstance(value, int) for value in token_ids):
        raise StageSevenValidationError(f"{name} token identifiers are invalid")
    if hash_prefix_token_ids(token_ids) != str(declared.get("token_ids_sha256")):
        raise StageSevenValidationError(f"{name} token hash mismatch")
    if token_ids != _row_ids(source_row, label=name):
        raise StageSevenValidationError(f"{name} disagrees with Stage 3 source")
    owner = _strict_owner(source_row, label=name)
    if owner != str(declared.get("physical_owner_id")):
        raise StageSevenValidationError(f"{name} physical owner mismatch")
    return {**dict(source_row), "raw_generated_token_ids": list(token_ids)}


def _candidate_row(
    admission: Mapping[str, Any], name: str, source_row: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = admission.get("frozen_row_seven_candidates")
    if not isinstance(candidates, Mapping) or not isinstance(candidates.get(name), Mapping):
        raise StageSevenValidationError(f"admission lacks row-7 candidate {name}")
    declared = dict(candidates[name])
    token_ids = declared.get("token_ids")
    if not isinstance(token_ids, list) or any(not isinstance(value, int) for value in token_ids):
        raise StageSevenValidationError(f"{name} token identifiers are invalid")
    if hash_prefix_token_ids(token_ids) != str(declared.get("token_ids_sha256")):
        raise StageSevenValidationError(f"{name} token hash mismatch")
    if token_ids != _row_ids(source_row, label=name):
        raise StageSevenValidationError(f"{name} disagrees with Stage 3 source")
    if _strict_owner(source_row, label=name) != str(declared.get("physical_owner_id")):
        raise StageSevenValidationError(f"{name} physical owner mismatch")
    return dict(source_row)


def load_stage_seven_source(admission_path: Path) -> dict[str, Any]:
    """Rehash the admission and reconstruct its four exact prefixes."""

    resolved = admission_path.expanduser().resolve(strict=True)
    admission = _load_json(resolved)
    if admission.get("schema_version") != 1 or admission.get("unit_id") != UNIT_ID:
        raise StageSevenValidationError("Stage 7 admission schema or unit mismatch")
    if len(admission.get("factorial_prefixes", [])) != 4:
        raise StageSevenValidationError("Stage 7 must contain exactly four prefixes")

    evidence = admission.get("source_evidence")
    if not isinstance(evidence, Mapping):
        raise StageSevenValidationError("Stage 7 admission lacks source evidence")

    def checked(key: str) -> Path:
        path = Path(str(evidence.get(key, ""))).expanduser().resolve(strict=True)
        if sha256_file(path) != str(evidence.get(f"{key}_sha256", "")):
            raise StageSevenValidationError(f"{key} SHA-256 disagrees with admission")
        return path

    stage_three_path = checked("stage_three_shard")
    checked("stage_three_union")
    checked("stage_three_admission")
    source_jsonl = checked("source_jsonl")
    checkpoint_json = checked("checkpoint_json")
    infer_config = checked("infer_config")

    checkpoint = _load_json(checkpoint_json)
    checkpoint_root = checkpoint_json.parents[2]
    adapter_relative = (
        ((checkpoint.get("adapter") or {}).get("identity") or {}).get("required_files") or {}
    ).get("adapter_model.safetensors")
    embedding_relative = (
        ((checkpoint.get("special_token_embeddings") or {}).get("identity") or {}).get(
            "tensor_path"
        )
    )
    if not adapter_relative or not embedding_relative:
        raise StageSevenValidationError("checkpoint lacks adapter or embedding payload paths")
    adapter_path = (checkpoint_root / str(adapter_relative)).resolve(strict=True)
    embedding_path = (checkpoint_root / str(embedding_relative)).resolve(strict=True)
    if sha256_file(adapter_path) != str(evidence.get("adapter_model_sha256")):
        raise StageSevenValidationError("adapter model SHA-256 disagrees with admission")
    if sha256_file(embedding_path) != str(evidence.get("special_token_embeddings_sha256")):
        raise StageSevenValidationError(
            "special-token embedding SHA-256 disagrees with admission"
        )

    stage_three = _load_json(stage_three_path)
    if stage_three.get("schema_version") != "sampled_history_target_reachability.stage_three.v1":
        raise StageSevenValidationError("Stage 3 source schema mismatch")
    image = _one_image(stage_three)
    native_suffix = (image.get("native_no_op_suffix") or {}).get("rows")
    sampled_suffix = (image.get("sampled_row_intervention_suffix") or {}).get("rows")
    if not isinstance(native_suffix, list) or not isinstance(sampled_suffix, list):
        raise StageSevenValidationError("Stage 3 source lacks endpoint suffix rows")
    if len(native_suffix) < 4 or len(sampled_suffix) < 4:
        raise StageSevenValidationError("Stage 3 endpoint suffix is too short")

    parent = admission.get("frozen_parent")
    if not isinstance(parent, Mapping) or int(parent.get("complete_row_count", -1)) != 3:
        raise StageSevenValidationError("Stage 7 parent contract mismatch")
    parent_ids = parent.get("token_ids")
    if not isinstance(parent_ids, list) or len(parent_ids) != 27:
        raise StageSevenValidationError("Stage 7 parent must contain 27 token identifiers")
    if hash_prefix_token_ids(parent_ids) != str(parent.get("token_ids_sha256")):
        raise StageSevenValidationError("Stage 7 parent hash mismatch")
    if parent_ids != list((image.get("parent") or {}).get("token_ids", [])):
        raise StageSevenValidationError("Stage 7 parent disagrees with Stage 3")

    rows = {
        "native_row_three": _admitted_row(admission, "native_row_three", image["native_branch"]),
        "sampled_row_three": _admitted_row(admission, "sampled_row_three", image["sampled_branch"]),
        "native_row_four": _admitted_row(admission, "native_row_four", native_suffix[0]),
        "sampled_row_four": _admitted_row(admission, "sampled_row_four", sampled_suffix[0]),
        "common_row_five": _admitted_row(admission, "common_row_five", native_suffix[1]),
        "common_row_six": _admitted_row(admission, "common_row_six", native_suffix[2]),
    }
    if _row_ids(native_suffix[1], label="native row 5") != _row_ids(sampled_suffix[1], label="sampled row 5"):
        raise StageSevenValidationError("row 5 is not common across endpoints")
    if _row_ids(native_suffix[2], label="native row 6") != _row_ids(sampled_suffix[2], label="sampled row 6"):
        raise StageSevenValidationError("row 6 is not common across endpoints")
    if not compare_branch_structure(
        rows["native_row_three"]["raw_generated_token_ids"],
        rows["sampled_row_three"]["raw_generated_token_ids"],
    )["passed"]:
        raise StageSevenValidationError("row-3 pair changes more than coordinates")
    if not compare_branch_structure(
        rows["native_row_four"]["raw_generated_token_ids"],
        rows["sampled_row_four"]["raw_generated_token_ids"],
    )["passed"]:
        raise StageSevenValidationError("row-4 pair changes more than coordinates")

    candidates = {
        "native_candidate": _candidate_row(admission, "native_candidate", native_suffix[3]),
        "sampled_target_candidate": _candidate_row(
            admission, "sampled_target_candidate", sampled_suffix[3]
        ),
    }
    component_map = {
        "native_row_three__native_row_four": ("native_row_three", "native_row_four"),
        "sampled_row_three__native_row_four": ("sampled_row_three", "native_row_four"),
        "native_row_three__sampled_row_four": ("native_row_three", "sampled_row_four"),
        "sampled_row_three__sampled_row_four": ("sampled_row_three", "sampled_row_four"),
    }
    prefixes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in admission["factorial_prefixes"]:
        if not isinstance(item, Mapping):
            raise StageSevenValidationError("Stage 7 prefix entry is not an object")
        arm = str(item.get("arm"))
        if arm not in component_map or arm in seen:
            raise StageSevenValidationError(f"unexpected or duplicate Stage 7 arm {arm!r}")
        seen.add(arm)
        row_three, row_four = component_map[arm]
        component_rows = [
            rows[row_three],
            rows[row_four],
            rows["common_row_five"],
            rows["common_row_six"],
        ]
        expected_ids = [int(value) for value in parent_ids]
        for row in component_rows:
            expected_ids.extend(_row_ids(row, label=f"{arm} component"))
        declared_ids = item.get("prefix_token_ids")
        if declared_ids != expected_ids or len(expected_ids) != 65:
            raise StageSevenValidationError(f"{arm} exact prefix composition mismatch")
        prefix_hash = hash_prefix_token_ids(expected_ids)
        if prefix_hash != str(item.get("prefix_token_ids_sha256")):
            raise StageSevenValidationError(f"{arm} exact prefix hash mismatch")
        prefixes.append({**dict(item), "prefix_token_ids": expected_ids, "rows": component_rows})
    if seen != EXPECTED_ARMS:
        raise StageSevenValidationError("Stage 7 arm set mismatch")

    contract = admission.get("execution_contract")
    if not isinstance(contract, Mapping):
        raise StageSevenValidationError("Stage 7 lacks execution contract")
    if (
        str(contract.get("image_id")) != "12576"
        or int(contract.get("forced_prefix_complete_row_count", -1)) != 7
        or int(contract.get("forced_prefix_token_count", -1)) != 65
        or int(contract.get("post_prefix_generated_token_budget", -1)) != 447
        or int(contract.get("retries", -1)) != 0
    ):
        raise StageSevenValidationError("Stage 7 execution contract mismatch")
    return {
        "admission": admission,
        "admission_path": str(resolved),
        "admission_sha256": sha256_file(resolved),
        "stage_three": stage_three,
        "stage_three_image": image,
        "rows": rows,
        "candidates": candidates,
        "prefixes": prefixes,
        "prefix_owner_ids": sorted(
            {
                *[str(value) for value in parent.get("verified_strict_owner_ids", [])],
                *[
                    str(rows[name].get("strict_matched_owner_ids", [None])[0])
                    for name in (
                        "native_row_three",
                        "native_row_four",
                        "common_row_five",
                        "common_row_six",
                    )
                ],
            }
            - {"None"}
        ),
        "paths": {
            "stage_three_shard": str(stage_three_path),
            "source_jsonl": str(source_jsonl),
            "infer_config": str(infer_config),
        },
    }


def classify_generated_row(
    row: Mapping[str, Any], *, native_owner_id: str, target_owner_id: str
) -> dict[str, Any]:
    owners = sorted({str(value) for value in row.get("strict_matched_owner_ids", [])})
    if not _row_is_clean_complete(row) or len(owners) != 1:
        return {"class": "unresolved", "owner_id": None, "owner_ids": owners}
    owner = owners[0]
    if owner == native_owner_id:
        label = "native_owner"
    elif owner == target_owner_id:
        label = "sampled_target_owner"
    else:
        label = "other_strict_owner"
    return {"class": label, "owner_id": owner, "owner_ids": owners}


def classify_crossover(cross_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    first = str(cross_results["sampled_row_three__native_row_four"]["class"])
    second = str(cross_results["native_row_three__sampled_row_four"]["class"])
    if first in {"unresolved", "other_strict_owner"} or second in {
        "unresolved",
        "other_strict_owner",
    }:
        conclusion = "unresolved_alternative"
    elif (first, second) == ("sampled_target_owner", "native_owner"):
        conclusion = "row_three_direct_retention"
    elif (first, second) == ("native_owner", "sampled_target_owner"):
        conclusion = "row_four_mediation_or_screening"
    elif (first, second) == ("sampled_target_owner", "sampled_target_owner"):
        conclusion = "either_changed_row_sufficient"
    elif (first, second) == ("native_owner", "native_owner"):
        conclusion = "joint_interaction_or_endpoint_dependency"
    else:
        conclusion = "unresolved_alternative"
    return {
        "sampled_row_three__native_row_four": first,
        "native_row_three__sampled_row_four": second,
        "conclusion": conclusion,
        "primary_mediation_claim_allowed": conclusion != "unresolved_alternative",
    }


def validate_endpoint_parity(
    arm_results: Mapping[str, Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    pairs = {
        "native": (
            "native_row_three__native_row_four",
            candidates["native_candidate"],
            "native_owner",
        ),
        "sampled": (
            "sampled_row_three__sampled_row_four",
            candidates["sampled_target_candidate"],
            "sampled_target_owner",
        ),
    }
    receipt: dict[str, Any] = {}
    for label, (arm, expected, expected_class) in pairs.items():
        row = arm_results[arm]["row"]
        raw_equal = row.get("raw_generated_token_ids") == expected.get("raw_generated_token_ids")
        class_equal = arm_results[arm]["classification"].get("class") == expected_class
        receipt[label] = {
            "arm": arm,
            "raw_token_ids_equal": raw_equal,
            "owner_class_equal": class_equal,
            "passed": raw_equal and class_equal,
        }
    receipt["passed"] = all(receipt[label]["passed"] for label in ("native", "sampled"))
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage7-admission", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force")
    try:
        frozen = load_stage_seven_source(args.stage7_admission)
    except (OSError, KeyError, TypeError, ValueError, StageSevenValidationError) as exc:
        raise SystemExit(f"Stage 7 source validation failed before runtime: {exc}") from exc

    expected_config = Path(frozen["paths"]["infer_config"]).resolve()
    observed_config = args.infer_config.expanduser().resolve(strict=True)
    if observed_config != expected_config:
        raise SystemExit("Stage 7 infer config path disagrees with frozen admission")
    try:
        import torch
        from src.config.fingerprint import sha256_json
        from src.config.inference import load_infer_config
        from src.data import load_raw_examples
        from src.inference.backend import open_backend_session
        from src.inference.runtime import assemble_frontend
    except Exception as exc:
        raise SystemExit(
            f"runtime import failed; no artifact was written: {type(exc).__name__}: {exc}"
        ) from exc

    resolved = load_infer_config(observed_config)
    config = resolved.config
    admission = frozen["admission"]
    evidence = admission["source_evidence"]
    contract = admission["execution_contract"]
    if str(config.model.dtype) != "fp32":
        raise SystemExit("Stage 7 requires model.dtype=fp32")
    if str(resolved.fingerprint) != str(evidence["resolved_config_fingerprint"]):
        raise SystemExit("Stage 7 resolved config fingerprint mismatch")
    source_jsonl = Path(frozen["paths"]["source_jsonl"]).resolve()
    if Path(str(config.data.input_jsonl)).expanduser().resolve() != source_jsonl:
        raise SystemExit("Stage 7 source JSONL disagrees with resolved config")

    examples = list(load_raw_examples(source_jsonl))
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    example = _select_example(examples, "12576")
    request, plan, prompt_meta = _build_request(config, frontend, example)
    ledger = build_positive_entity_ledger(example)
    stage_three = frozen["stage_three"]
    stage_image = frozen["stage_three_image"]
    arm_results: dict[str, Any] = {}

    with open_backend_session(frontend.launch) as session:
        model_receipt = session.receipt.to_artifact_dict()
        model_identity_check = compare_stable_model_identity(
            model_receipt, stage_three.get("model_identity") or {}
        )
        if not model_identity_check["passed"]:
            raise SystemExit("Stage 7 model identity disagrees with Stage 3")
        native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs(
            (request,)
        )
        execution_identity_check = compare_execution_identity(
            observed_prompt=prompt_meta,
            observed_runtime={
                "executed_media_sha256": media_sha[0],
                "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]),
                "observed_image_grid_thw": (
                    None if observed_grids[0] is None else list(observed_grids[0])
                ),
            },
            discovery_prompt=(stage_image.get("prompt") or {}).get("prompt") or {},
            discovery_runtime={
                **dict(stage_image.get("runtime") or {}),
                "executed_prompt_token_ids_sha256": (
                    stage_image.get("prompt") or {}
                ).get("prompt_token_ids_sha256"),
            },
        )
        if not execution_identity_check["passed"]:
            raise SystemExit("Stage 7 prompt, media, or image-grid identity mismatch")
        one_native = _single_native_inputs(native_inputs)
        for arm in frozen["prefixes"]:
            continuation = _generate_prefix_continuation(
                session=session,
                native_inputs=one_native,
                prefix_token_ids=arm["prefix_token_ids"],
                prefix_owner_ids=frozen["prefix_owner_ids"],
                tokenizer=session._tokenizer,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                entity_ledger=ledger,
                start_row_index=7,
                horizon_rows=1,
                malformed_limit=2,
                temperature=0.4,
                total_token_budget=int(contract["post_prefix_generated_token_budget"]),
            )
            generated_rows = continuation.get("rows") or []
            if len(generated_rows) != 1:
                raise SystemExit(f"Stage 7 arm {arm['arm']} did not generate exactly one row")
            row = generated_rows[0]
            classification = classify_generated_row(
                row,
                native_owner_id=str(
                    admission["frozen_row_seven_candidates"]["native_candidate"][
                        "physical_owner_id"
                    ]
                ),
                target_owner_id=str(
                    admission["frozen_row_seven_candidates"]["sampled_target_candidate"][
                        "physical_owner_id"
                    ]
                ),
            )
            arm_results[str(arm["arm"])] = {
                "role": arm.get("status"),
                "prefix_token_ids": arm["prefix_token_ids"],
                "prefix_token_ids_sha256": arm["prefix_token_ids_sha256"],
                "row": row,
                "classification": classification,
                "continuation_receipt": {
                    "continuation_row_count": continuation.get("continuation_row_count"),
                    "generated_token_count": continuation.get("generated_token_count"),
                    "horizon_rows_complete": continuation.get("horizon_rows_complete"),
                    "budget_exhausted": continuation.get("budget_exhausted"),
                },
            }

    endpoint_parity = validate_endpoint_parity(arm_results, frozen["candidates"])
    if not endpoint_parity["passed"]:
        raise SystemExit("Stage 7 exact endpoint row-7 parity failed")
    crossover = classify_crossover(
        {
            name: arm_results[name]["classification"]
            for name in (
                "sampled_row_three__native_row_four",
                "native_row_three__sampled_row_four",
            )
        }
    )
    source_identity = git_execution_identity(Path(__file__).resolve().parents[2])
    source_identity.pop("runner_sha256", None)
    source_identity["stage_seven_runner_sha256"] = sha256_file(Path(__file__).resolve())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "source_identity": source_identity,
        "frozen_inputs": {
            "stage7_admission": frozen["admission_path"],
            "stage7_admission_sha256": frozen["admission_sha256"],
            "stage_three_shard": frozen["paths"]["stage_three_shard"],
            "stage_three_shard_sha256": evidence["stage_three_shard_sha256"],
        },
        "config": {
            "infer_config": str(observed_config),
            "resolved_config_fingerprint": resolved.fingerprint,
            "device": args.device,
            "physical_batch_size": 1,
            "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "forced_prefix_token_count": 65,
            "post_prefix_generated_token_budget": 447,
            "generated_rows_per_arm": 1,
        },
        "model_identity": model_receipt,
        "model_identity_check": model_identity_check,
        "execution_identity_check": execution_identity_check,
        "endpoint_parity": endpoint_parity,
        "crossover_interpretation": crossover,
        "arms": arm_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
