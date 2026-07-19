#!/usr/bin/env python3
"""Run the frozen Stage 6 row-four coordinate factorial on image 7816."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_coordinate_history_crossover import (  # noqa: E402
    _flatten_row_ids,
    _first_suffix_owner,
    _strict_owner,
    load_stage_five_source,
)
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
    box_iou,
    compare_branch_structure,
    compare_stable_model_identity,
)
from scripts.research.run_complete_candidate_row_scoring import COORDINATE_TOKEN_START  # noqa: E402
from scripts.research.run_same_parent_final_horizon import (  # noqa: E402
    UNIT_ID,
    load_stage_four_source,
)


SCHEMA_VERSION = "sampled_history_target_reachability.stage_six.v1"
PHASE = "stage_six_row_four_coordinate_factorial"
SCORING_MANIFEST_SCHEMA_VERSION = "complete_candidate_row_scoring.manifest.v1"
EXPECTED_ARM_NAMES = {
    "factor_x1_native_y1_native_y2_native",
    "factor_x1_native_y1_native_y2_sampled",
    "factor_x1_native_y1_sampled_y2_native",
    "factor_x1_native_y1_sampled_y2_sampled",
    "factor_x1_sampled_y1_native_y2_native",
    "factor_x1_sampled_y1_native_y2_sampled",
    "factor_x1_sampled_y1_sampled_y2_native",
    "factor_x1_sampled_y1_sampled_y2_sampled",
    "reverse_natural_coordinate_direction",
    "orthogonal_x2_only_coordinate_control",
}


class StageSixValidationError(ValueError):
    """Raised when frozen Stage 6 sources cannot support execution."""


def _row_box(coords: Sequence[int]) -> list[float]:
    if len(coords) != 4:
        raise StageSixValidationError("row-four coordinates must have four values")
    x1, y1, x2, y2 = [int(v) for v in coords]
    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
        raise StageSixValidationError(f"invalid normalized box coordinates: {coords}")
    return [float(x1), float(y1), float(x2), float(y2)]


def _ledger_boxes(stage_three_image: Mapping[str, Any]) -> dict[str, list[float]]:
    source = stage_three_image.get("stage2_source")
    if not isinstance(source, Mapping):
        raise StageSixValidationError("Stage 3 image lacks Stage 2 source")
    path = Path(str(source.get("path", ""))).expanduser().resolve(strict=True)
    document = _load_json(path)
    images = document.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise StageSixValidationError("Stage 2 source must contain one image")
    ledger = images[0].get("entity_ledger")
    if not isinstance(ledger, list):
        raise StageSixValidationError("Stage 2 source lacks entity ledger")
    result: dict[str, list[float]] = {}
    for entity in ledger:
        if not isinstance(entity, Mapping) or not isinstance(entity.get("bbox_norm1000"), list):
            continue
        result[str(entity.get("entity_id"))] = [float(v) for v in entity["bbox_norm1000"]]
    return result


def load_stage_six_source(admission_path: Path) -> dict[str, Any]:
    """Rehash the Stage 6 admission and construct its ten exact prefixes."""
    resolved = admission_path.expanduser().resolve(strict=True)
    admission = _load_json(resolved)
    if admission.get("schema_version") != 1 or admission.get("unit_id") != UNIT_ID:
        raise StageSixValidationError("Stage 6 admission schema or unit mismatch")

    def checked(key: str, hash_key: str) -> Path:
        path = Path(str(admission.get(key, ""))).expanduser().resolve(strict=True)
        if sha256_file(path) != str(admission.get(hash_key, "")):
            raise StageSixValidationError(f"{key} SHA-256 disagrees with Stage 6 admission")
        return path

    stage5_admission = checked("source_stage_five_admission", "source_stage_five_admission_sha256")
    stage5_artifact = checked("source_stage_five_artifact", "source_stage_five_artifact_sha256")
    stage3_shard = checked("source_stage_three_shard", "source_stage_three_shard_sha256")
    stage5 = load_stage_five_source(stage5_admission)
    stage5_artifact_document = _load_json(stage5_artifact)
    if stage5_artifact_document.get("schema_version") != "sampled_history_target_reachability.stage_five.v1":
        raise StageSixValidationError("Stage 6 source Stage 5 artifact schema mismatch")
    if Path(stage5["paths"]["stage_three_shard"]).resolve() != stage3_shard:
        raise StageSixValidationError("Stage 6 Stage 3 shard disagrees with Stage 5 source")
    contract = admission.get("execution_contract")
    if not isinstance(contract, Mapping) or str(contract.get("image_id")) != "7816":
        raise StageSixValidationError("Stage 6 admits only image 7816")
    if int(contract.get("forced_prefix_token_count", -1)) != 45 or int(contract.get("post_prefix_generated_token_budget", -1)) != 467 or int(contract.get("total_trajectory_generated_token_budget", -1)) != 512:
        raise StageSixValidationError("Stage 6 token budget must be 45 + 467 = 512")

    stage_three = _load_json(stage3_shard)
    images = stage_three.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise StageSixValidationError("Stage 3 shard must contain one image")
    stage_three_image = dict(images[0])
    if str(stage_three_image.get("image_id")) != "7816":
        raise StageSixValidationError("Stage 3 shard image mismatch")
    native_prefix = stage5["prefixes"]["native_row_zero__native_row_four"]
    sampled_prefix = stage5["prefixes"]["sampled_row_zero__sampled_row_four"]
    common_rows = native_prefix[:4]
    native_row4 = native_prefix[4]
    if any(native_prefix[i].get("raw_generated_token_ids") != sampled_prefix[i].get("raw_generated_token_ids") for i in (1, 2, 3)):
        raise StageSixValidationError("rows 1 through 3 are not common")
    common_hash = hash_prefix_token_ids(_flatten_row_ids(common_rows))
    if common_hash != str(contract.get("common_rows_zero_through_three_token_ids_sha256")):
        raise StageSixValidationError("common rows 0 through 3 hash disagrees")
    if _strict_owner(native_row4, label="native row 4") != str(contract["row_four_owner_id"]):
        raise StageSixValidationError("native row 4 owner mismatch")
    ledger_boxes = _ledger_boxes(stage_three_image)
    owner_box = ledger_boxes.get(str(contract["row_four_owner_id"]))
    target_box = ledger_boxes.get(str(contract["target_owner_id"]))
    if owner_box is None or target_box is None:
        raise StageSixValidationError("Stage 6 owner or target absent from physical ledger")

    arms: list[dict[str, Any]] = []
    seen_prefixes: set[str] = set()
    for item in admission.get("arms", []):
        if not isinstance(item, Mapping):
            raise StageSixValidationError("Stage 6 arm is not an object")
        arm = dict(item)
        token_ids = arm.get("row_token_ids")
        coords = arm.get("coordinates")
        if not isinstance(token_ids, list) or len(token_ids) != 9 or any(not isinstance(v, int) for v in token_ids):
            raise StageSixValidationError(f"{arm.get('arm')} row must contain nine integer tokens")
        _row_box(coords)
        if hash_prefix_token_ids(token_ids) != str(arm.get("row_token_ids_sha256")):
            raise StageSixValidationError(f"{arm.get('arm')} row hash mismatch")
        declared_coords = [int(v) for v in coords]
        if token_ids[4:8] != [COORDINATE_TOKEN_START + value for value in declared_coords]:
            raise StageSixValidationError(f"{arm.get('arm')} coordinate tokens disagree with declared coordinates")
        structure = compare_branch_structure(native_row4["raw_generated_token_ids"], token_ids)
        if not structure.get("passed") and arm.get("role") != "factorial native endpoint and exact replay control":
            raise StageSixValidationError(f"{arm.get('arm')} changes more than coordinates")
        owner_iou = box_iou(_row_box(coords), owner_box)
        target_iou = box_iou(_row_box(coords), target_box)
        other_iou = max((box_iou(_row_box(coords), box) for entity_id, box in ledger_boxes.items() if entity_id not in {str(contract["row_four_owner_id"]), str(contract["target_owner_id"])}), default=0.0)
        if owner_iou < 0.5 or target_iou >= 0.5 or owner_iou - other_iou < 0.05:
            raise StageSixValidationError(f"{arm.get('arm')} fails strict owner/target geometry gate")
        rows = [*common_rows, {**dict(native_row4), "raw_generated_token_ids": token_ids, "strict_matched_owner_ids": [str(contract["row_four_owner_id"])]}]
        prefix_ids = _flatten_row_ids(rows)
        prefix_hash = hash_prefix_token_ids(prefix_ids)
        if prefix_hash != str(arm.get("prefix_token_ids_sha256")):
            raise StageSixValidationError(f"{arm.get('arm')} prefix hash mismatch")
        if prefix_hash in seen_prefixes:
            raise StageSixValidationError("duplicate Stage 6 prefix")
        seen_prefixes.add(prefix_hash)
        arms.append({**arm, "rows": rows, "prefix_token_ids": prefix_ids, "prefix_token_ids_sha256": prefix_hash})
    if len(arms) != 10:
        raise StageSixValidationError("Stage 6 requires exactly ten admitted arms")
    if {str(arm.get("arm")) for arm in arms} != EXPECTED_ARM_NAMES:
        raise StageSixValidationError("Stage 6 arm set is not the frozen full factorial plus two controls")

    candidate_rows = admission.get("candidate_rows_for_teacher_forced_scoring")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != 2:
        raise StageSixValidationError("Stage 6 requires exactly two frozen candidate rows")
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping):
            raise StageSixValidationError("Stage 6 candidate row is not an object")
        candidate_ids = candidate.get("row_token_ids")
        if (
            not isinstance(candidate_ids, list)
            or len(candidate_ids) != 9
            or any(not isinstance(value, int) for value in candidate_ids)
            or hash_prefix_token_ids(candidate_ids)
            != str(candidate.get("row_token_ids_sha256"))
        ):
            raise StageSixValidationError(
                f"candidate row {candidate.get('candidate_id')!r} token hash or shape disagrees"
            )
    endpoint_expected = {"factor_x1_native_y1_native_y2_native": "205108", "factor_x1_sampled_y1_sampled_y2_sampled": "211764"}
    for arm_name, owner in endpoint_expected.items():
        arm = next(item for item in arms if item["arm"] == arm_name)
        expected = (admission.get("arms") or [])
        declared = next(item for item in expected if item.get("arm") == arm_name).get("frozen_expected_first_suffix_owner_id")
        if str(declared or owner) != owner:
            raise StageSixValidationError("endpoint expected owner mismatch")
    stage4_image = stage5["stage_four_artifact"].get("image") or {}
    for arm_name, owner in endpoint_expected.items():
        stage4_arm = stage4_image.get("native_arm" if arm_name.endswith("native_y2_native") else "sampled_arm") or {}
        suffix_rows = (stage4_arm.get("suffix") or {}).get("rows") or []
        endpoint_rows = [row for row in suffix_rows if isinstance(row, Mapping) and int(row.get("row_index", -1)) == 5]
        if len(endpoint_rows) != 1 or _strict_owner(endpoint_rows[0], label=f"Stage 4 {arm_name} endpoint") != owner:
            raise StageSixValidationError(f"Stage 4 endpoint parity evidence disagrees for {arm_name}")
    cross_sampled = (stage5_artifact_document.get("cross_arms") or {}).get("native_row_zero__sampled_row_four") or {}
    cross_first = (cross_sampled.get("first_suffix_owner") or {}).get("owner_id")
    if str(cross_first) != str(contract["target_owner_id"]):
        raise StageSixValidationError("Stage 5 sampled-row-four endpoint evidence disagrees")
    return {"admission_path": str(resolved), "admission_sha256": sha256_file(resolved), "admission": admission, "stage5": stage5, "stage5_artifact": stage5_artifact_document, "stage_three_image": stage_three_image, "arms": arms, "paths": {"stage5_admission": str(stage5_admission), "stage5_artifact": str(stage5_artifact), "stage3_shard": str(stage3_shard)}}


def build_scoring_manifest(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Build a literal-token complete-row scorer manifest after all gates pass."""
    contract = frozen["admission"]["execution_contract"]
    stage_three_image = frozen["stage_three_image"]
    base = [int(v) for v in stage_three_image["prompt"]["prompt"]["prompt_token_ids"]]
    candidate_rows = frozen["admission"]["candidate_rows_for_teacher_forced_scoring"]
    candidates = []
    for item in candidate_rows:
        candidates.append({"candidate_id": item["candidate_id"], "owner": item.get("owner_id"), "category": "person", "role": item.get("role"), "row": {"token_ids": [int(v) for v in item["row_token_ids"]], "token_ids_sha256": item["row_token_ids_sha256"]}})
    boundaries = []
    for arm in frozen["arms"]:
        boundaries.append({"boundary_id": arm["arm"], "prefix_mode": "base_prompt_plus_generated", "prefix": {"token_ids": arm["prefix_token_ids"], "token_ids_sha256": arm["prefix_token_ids_sha256"]}, "candidates": candidates})
    return {"schema_version": SCORING_MANIFEST_SCHEMA_VERSION, "unit_id": UNIT_ID, "images": [{"image_id": str(contract["image_id"]), "base_prompt": {"token_ids": base, "token_ids_sha256": hash_prefix_token_ids(base)}, "boundaries": boundaries}]}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage6-admission", type=Path, required=True)
    p.add_argument("--infer-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scoring-manifest-output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _resolved_output_paths(output: Path, scoring_manifest_output: Path) -> tuple[Path, Path]:
    primary = output.expanduser().resolve()
    companion = scoring_manifest_output.expanduser().resolve()
    if primary == companion:
        raise StageSixValidationError(
            "Stage 6 primary output and scoring manifest must use different paths"
        )
    return primary, companion


def main() -> int:
    args = _parse_args()
    try:
        primary_output, scoring_manifest_output = _resolved_output_paths(
            args.output, args.scoring_manifest_output
        )
    except StageSixValidationError as exc:
        raise SystemExit(str(exc)) from exc
    if (primary_output.exists() or scoring_manifest_output.exists()) and not args.force:
        raise SystemExit("refusing to overwrite existing Stage 6 output; pass --force")
    try:
        frozen = load_stage_six_source(args.stage6_admission)
    except (OSError, KeyError, TypeError, ValueError, StageSixValidationError) as exc:
        raise SystemExit(f"Stage 6 source validation failed before runtime: {exc}") from exc
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
        raise SystemExit("Stage 6 requires model.dtype=fp32")
    expected_fp = (frozen["stage5"]["stage_four_artifact"].get("config") or {}).get("resolved_config_fingerprint")
    if expected_fp is not None and str(resolved.fingerprint) != str(expected_fp):
        raise SystemExit("Stage 6 resolved config fingerprint disagrees with frozen Stage 4")
    stage_two_sources = frozen["stage5"]["stage_four_frozen"]["stage_three_sources"]
    stage_two_artifact = stage_two_sources["stage_two_shards"]["7816"]["artifact"]
    frozen_inputs = stage_two_artifact.get("frozen_inputs") or {}
    manifest_path = Path(str(frozen_inputs.get("manifest", ""))).expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path)
    source_jsonl = Path(str((manifest.get("inference_contract") or {}).get("source_jsonl", config.data.input_jsonl))).expanduser().resolve(strict=True)
    frozen_file_identity = validate_frozen_file_identity(manifest, infer_config_path=args.infer_config, source_jsonl_path=source_jsonl)
    examples = list(load_raw_examples(source_jsonl))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    example = _select_example(examples, "7816")
    request, plan, prompt_meta = _build_request(config, frontend, example)
    ledger = build_positive_entity_ledger(example)
    arm_results: dict[str, Any] = {}
    with open_backend_session(frontend.launch) as session:
        model_receipt = session.receipt.to_artifact_dict()
        identity = compare_stable_model_identity(model_receipt, frozen["stage5_artifact"].get("model_identity") or {})
        if not identity["passed"]:
            raise SystemExit("Stage 6 model identity disagrees")
        native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
        execution = compare_execution_identity(observed_prompt=prompt_meta, observed_runtime={"executed_media_sha256": media_sha[0], "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]), "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0])}, discovery_prompt=frozen["stage_three_image"]["prompt"]["prompt"], discovery_runtime={**dict(frozen["stage_three_image"]["runtime"]), "executed_prompt_token_ids_sha256": frozen["stage_three_image"]["prompt"]["prompt_token_ids_sha256"]})
        if not execution["passed"]:
            raise SystemExit("Stage 6 prompt/media/grid identity disagrees")
        for arm in frozen["arms"]:
            prefix_owner_ids = sorted({str(value) for row in arm["rows"] for value in row.get("strict_matched_owner_ids", [])})
            continuation = _generate_prefix_continuation(session=session, native_inputs=_single_native_inputs(native_inputs), prefix_token_ids=arm["prefix_token_ids"], prefix_owner_ids=prefix_owner_ids, tokenizer=session._tokenizer, image_width=int(plan.decoded_width), image_height=int(plan.decoded_height), entity_ledger=ledger, start_row_index=5, horizon_rows=int(contract["continuation_row_ceiling"]), malformed_limit=2, temperature=0.4, total_token_budget=int(contract["post_prefix_generated_token_budget"]))
            full_rows = [*arm["rows"], *continuation["rows"]]
            summary = summarize_continuation(full_rows, prefix_owner_ids=[], target_owner_id=str(contract["target_owner_id"]), target_start_row_index=5, generated_token_count=len(arm["prefix_token_ids"]) + int(continuation["generated_token_count"]), total_token_budget=512, horizon_rows_complete=False)
            arm_results[arm["arm"]] = {"prefix_token_ids": arm["prefix_token_ids"], "prefix_token_ids_sha256": arm["prefix_token_ids_sha256"], "prefix_owner_ids": prefix_owner_ids, "continuation": continuation, "first_suffix_owner": _first_suffix_owner(continuation), "summary": summary, "final_unique_owner_ids": sorted(_owner_set([], full_rows))}
    endpoint_native = arm_results["factor_x1_native_y1_native_y2_native"]["first_suffix_owner"]
    endpoint_sampled = arm_results["factor_x1_sampled_y1_sampled_y2_sampled"]["first_suffix_owner"]
    stage4_image = frozen["stage5"]["stage_four_artifact"].get("image") or {}
    endpoint_parity: dict[str, Any] = {}
    stage5_sampled_endpoint = (
        (frozen["stage5_artifact"].get("cross_arms") or {})
        .get("native_row_zero__sampled_row_four", {})
        .get("continuation", {})
    )
    expected_endpoint_rows = {
        "native": [
            row
            for row in (((stage4_image.get("native_arm") or {}).get("suffix") or {}).get("rows", []))
            if isinstance(row, Mapping) and int(row.get("row_index", -1)) >= 5
        ],
        "sampled": [
            row
            for row in stage5_sampled_endpoint.get("rows", [])
            if isinstance(row, Mapping)
        ],
    }
    for label, arm_name in (
        ("native", "factor_x1_native_y1_native_y2_native"),
        ("sampled", "factor_x1_sampled_y1_sampled_y2_sampled"),
    ):
        expected_rows = expected_endpoint_rows[label]
        observed_rows = arm_results[arm_name]["continuation"].get("rows", [])
        exact = len(expected_rows) == len(observed_rows) and all(a.get("raw_generated_token_ids") == b.get("raw_generated_token_ids") for a, b in zip(expected_rows, observed_rows))
        endpoint_parity[label] = {
            "passed": exact,
            "expected_source": "Stage 4 native arm" if label == "native" else "Stage 5 native-row-zero plus sampled-row-four arm",
            "expected_row_count": len(expected_rows),
            "observed_row_count": len(observed_rows),
            "expected_first_owner": _strict_owner(expected_rows[0], label=f"frozen {label} endpoint") if expected_rows else None,
            "observed_first_owner": (endpoint_native if label == "native" else endpoint_sampled).get("owner_id"),
        }
    if not all(item["passed"] for item in endpoint_parity.values()):
        raise SystemExit("Stage 6 endpoint exact continuation parity failed")
    if endpoint_native.get("owner_id") != "205108" or endpoint_sampled.get("owner_id") != "211764":
        raise SystemExit("Stage 6 endpoint parity failed")
    manifest = build_scoring_manifest(frozen)
    source_identity = git_execution_identity(Path(__file__).resolve().parents[2])
    source_identity.pop("runner_sha256", None)
    source_identity["stage_six_runner_sha256"] = sha256_file(Path(__file__).resolve())
    payload = {"schema_version": SCHEMA_VERSION, "phase": PHASE, "source_identity": source_identity, "frozen_inputs": {"stage6_admission": frozen["admission_path"], "stage6_admission_sha256": frozen["admission_sha256"], "paths": frozen["paths"]}, "config": {"infer_config": str(args.infer_config.resolve()), "resolved_config_fingerprint": resolved.fingerprint, "device": args.device, "physical_batch_size": 1, "model_dtype": "fp32", "repetition_penalty": 1.0, "total_trajectory_generated_token_budget": 512, "post_prefix_generated_token_budget": 467}, "model_identity": model_receipt, "model_identity_check": identity, "execution_identity_check": execution, "frozen_file_identity": frozen_file_identity, "endpoint_parity": endpoint_parity, "scoring_manifest": {"path": str(scoring_manifest_output)}, "arms": arm_results}
    primary_output.parent.mkdir(parents=True, exist_ok=True)
    scoring_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    # Finalize the companion first.  On a fresh run, the canonical primary
    # result can therefore exist only after its required scoring input exists.
    scoring_manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    primary_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
