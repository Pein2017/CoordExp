#!/usr/bin/env python3
"""Screen fixed model-native transitions for batch-shape and precision effects.

This entrypoint is deliberately experiment-local.  For each selected image it
appends the first complete row from an immutable sampled rollout to the exact
source prompt, then compares one greedy recipient with four homogeneous copies.
It does not change the model forward pass, shared inference runtime, or decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_native_commit_redistribution import (  # noqa: E402
    BOX_END,
    BOX_START,
    COORDINATE_TOKEN_END_EXCLUSIVE,
    COORDINATE_TOKEN_START,
    OBJECT_REF_END,
    OBJECT_REF_START,
    _json_hash,
    _read_json,
    _validate_identity_continuity,
)
from scripts.research.run_sampled_rescue_transition import (  # noqa: E402
    _first_action,
    _sha256_file,
    _temporary_cwd,
    _write_bundle_once,
)


DEFAULT_CONFIG = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)
DEFAULT_SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)
SOURCE_CALL_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/executions/"
    "dense-union-51-primary-after-wave-local-tail-contract/artifacts/calls"
)
DEFAULT_SOURCE_BUNDLES: dict[str, Path] = {
    "7574": SOURCE_CALL_ROOT
    / "f56e79e626cee96fc8b7f1b7d98db0fb8d7e338bf243d372ec3d9259a3e91562"
    / "terminal-output-bundle.json",
    "8629": SOURCE_CALL_ROOT
    / "3c97ac12bcfce1787d56733b928ef25166c78178ca24c8fe7d99df0247495154"
    / "terminal-output-bundle.json",
    "9891": SOURCE_CALL_ROOT
    / "6bca67d00fceac633b20fa8773a0ac74c373d8817b3f013bdfb3b259b7e86683"
    / "terminal-output-bundle.json",
    "12576": SOURCE_CALL_ROOT
    / "b3006ff584a22ef984bf6922c9f3c91d6292eb405aae9b323fbba181f396b936"
    / "terminal-output-bundle.json",
    "13659": SOURCE_CALL_ROOT
    / "ac6c352af660dc3586381ed29229dbb22b2a29fbc87a8e167d97bfaf45ae8d8a"
    / "terminal-output-bundle.json",
    "17714": SOURCE_CALL_ROOT
    / "1409ebbeb330a82dfe895af91d47e3d1f9221d98c98058d9e66e85fc5f40b8a8"
    / "terminal-output-bundle.json",
}
DEFAULT_IMAGE_IDS = tuple(DEFAULT_SOURCE_BUNDLES)
MODEL_DTYPE_BY_PUBLIC_NAME = {
    "bfloat16": "bf16",
    "float32": "fp32",
}
TORCH_DTYPE_BY_PUBLIC_NAME = {
    "bfloat16": "torch.bfloat16",
    "float32": "torch.float32",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed model-native next actions under physical batch one "
            "and four homogeneous copies."
        )
    )
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--image-id",
        action="append",
        dest="image_ids",
        help="Selected Common Objects in Context image identifier; repeat as needed.",
    )
    parser.add_argument(
        "--model-dtype",
        choices=tuple(MODEL_DTYPE_BY_PUBLIC_NAME),
        default="bfloat16",
        help="Full-model execution dtype for this run.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def select_source_bundles(image_ids: Sequence[str] | None) -> list[tuple[str, Path]]:
    selected = list(DEFAULT_IMAGE_IDS if not image_ids else (str(value) for value in image_ids))
    if not selected:
        raise ValueError("at least one image identifier is required")
    if len(set(selected)) != len(selected):
        raise ValueError("image identifiers must be unique")
    unknown = [image_id for image_id in selected if image_id not in DEFAULT_SOURCE_BUNDLES]
    if unknown:
        raise ValueError(f"unknown selected image identifiers: {unknown}")
    return [(image_id, DEFAULT_SOURCE_BUNDLES[image_id]) for image_id in selected]


def _validate_description_tokens(tokens: Sequence[int]) -> None:
    if not tokens:
        raise ValueError("object row has an empty description")
    forbidden = {OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END}
    for token in tokens:
        if int(token) in forbidden or (
            COORDINATE_TOKEN_START <= int(token) < COORDINATE_TOKEN_END_EXCLUSIVE
        ):
            raise ValueError("object description contains a wrapper or coordinate token")


def extract_first_complete_row(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the first canonical row while allowing variable description length."""

    decode = source_bundle.get("decode_result")
    if not isinstance(decode, Mapping):
        raise ValueError("source bundle lacks decode_result")
    generated = decode.get("generated_token_ids")
    prompt = decode.get("prompt_token_ids")
    if not isinstance(generated, list) or not isinstance(prompt, list):
        raise ValueError("source bundle lacks prompt or generated token identifiers")
    values = [int(token) for token in generated]
    for start, token in enumerate(values):
        if token != OBJECT_REF_START:
            continue
        try:
            object_end = values.index(OBJECT_REF_END, start + 1)
            box_start = values.index(BOX_START, object_end + 1)
            box_end = values.index(BOX_END, box_start + 1)
        except ValueError:
            continue
        if box_start != object_end + 1 or box_end != box_start + 5:
            continue
        description = values[start + 1 : object_end]
        coordinates = values[box_start + 1 : box_end]
        try:
            _validate_description_tokens(description)
        except ValueError:
            continue
        if len(coordinates) != 4 or any(
            not (COORDINATE_TOKEN_START <= item < COORDINATE_TOKEN_END_EXCLUSIVE)
            for item in coordinates
        ):
            continue
        row = values[start : box_end + 1]
        return {
            "prompt_token_ids": [int(value) for value in prompt],
            "prompt_token_ids_sha256": _json_hash(prompt),
            "row_token_start": start,
            "row_token_end": box_end + 1,
            "row_token_ids": row,
            "row_token_ids_sha256": _json_hash(row),
            "description_token_ids": description,
            "coordinate_token_ids": coordinates,
        }
    raise ValueError("source generated tokens contain no complete canonical object row")


def validate_source_bundle(
    source_bundle: Mapping[str, Any], *, image_id: str
) -> dict[str, Any]:
    scheduled = source_bundle.get("scheduled_request")
    execution = source_bundle.get("execution_evidence")
    if not isinstance(scheduled, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("source bundle lacks scheduled or execution evidence")
    arm = scheduled.get("arm")
    if not isinstance(arm, Mapping) or arm.get("arm_code") != "FULL_BAG_K":
        raise ValueError("source bundle is not Full-Image K-Rollout Independent Bagging")
    if scheduled.get("cell_index") != 0:
        raise ValueError("source bundle is not canonical cell zero")
    if str(execution.get("image_id")) != str(image_id):
        raise ValueError("source bundle image identifier does not match the selected case")
    return {
        "arm_code": arm.get("arm_code"),
        "arm_full_name": arm.get("full_name"),
        "cell_index": scheduled.get("cell_index"),
        "sampling_seed": scheduled.get("sampling_seed"),
        "source_image_sha256": execution.get("source_image_sha256"),
    }


ACTION_COMPARISON_FIELDS = (
    "status",
    "generated_order",
    "description",
    "bbox_xyxy",
    "drop_reason",
    "char_start",
    "char_end",
    "token_start",
    "token_end",
    "token_ids",
)


def canonical_first_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """Drop request-owned identifiers while retaining complete action semantics."""

    return {field: action.get(field) for field in ACTION_COMPARISON_FIELDS}


def _canonical_action_hash(action: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_first_action(action),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def compare_action_records(
    single_record: Mapping[str, Any],
    homogeneous_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(homogeneous_records) != 4:
        raise ValueError("exactly four homogeneous records are required")
    homogeneous_action_hashes = [
        _canonical_action_hash(record["first_free_action"])
        for record in homogeneous_records
    ]
    homogeneous_generation_hashes = [
        _json_hash(record["generated_token_ids"]) for record in homogeneous_records
    ]
    homogeneous_actions_equal = len(set(homogeneous_action_hashes)) == 1
    homogeneous_generations_equal = len(set(homogeneous_generation_hashes)) == 1
    if not homogeneous_actions_equal:
        raise ValueError("homogeneous first actions disagree within one physical batch")
    single_action_hash = _canonical_action_hash(single_record["first_free_action"])
    single_generation_hash = _json_hash(single_record["generated_token_ids"])
    return {
        "homogeneous_first_actions_exactly_equal": homogeneous_actions_equal,
        "homogeneous_complete_generations_exactly_equal": homogeneous_generations_equal,
        "single_vs_homogeneous_first_action_exactly_equal": (
            single_action_hash == homogeneous_action_hashes[0]
        ),
        "single_vs_homogeneous_complete_generation_exactly_equal": (
            single_generation_hash == homogeneous_generation_hashes[0]
        ),
        "primary_first_action_divergence": (
            single_action_hash != homogeneous_action_hashes[0]
        ),
        "single_first_action_sha256": single_action_hash,
        "homogeneous_first_action_sha256": homogeneous_action_hashes[0],
        "single_complete_generation_sha256": single_generation_hash,
        "homogeneous_complete_generation_sha256": homogeneous_generation_hashes[0],
    }


def _compact_result(result: Any, *, raw: Any) -> dict[str, Any]:
    from src.inference.parsing import parse_compact_object_box_closed

    parsed = parse_compact_object_box_closed(
        result.parser_text,
        row_id=result.request_id,
        row_index=0,
        image_width=raw.image.width,
        image_height=raw.image.height,
    )
    action = _first_action(
        parsed,
        result.parser_text,
        generated_token_ids=result.generated_token_ids,
        stop_reason=result.stop_reason,
    )
    artifact = result.to_artifact_dict()
    return {
        "request_id": result.request_id,
        "generated_token_ids": [int(token) for token in result.generated_token_ids],
        "raw_generated_text": result.raw_generated_text,
        "parser_text": result.parser_text,
        "stop_reason": result.stop_reason,
        "first_free_action": action,
        "parse_result": parsed.to_artifact_dict(),
        "execution_receipt": artifact.get("execution_receipt"),
    }


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeGenerationPolicy, DecodeRequest, HFGenerateBackend
    from src.inference.image_plan import (
        materialize_image_plan_batch,
        verify_processor_model_vision_parity,
    )
    from src.inference.pipeline import (
        _processor_config,
        _template_config,
        _tokenizer_identity,
    )
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.runtime import assemble_runtime

    if int(args.max_new_tokens) <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    selected = select_source_bundles(args.image_ids)
    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    execution_config = resolved.config
    requested_config_dtype = MODEL_DTYPE_BY_PUBLIC_NAME[str(args.model_dtype)]
    if execution_config.model.dtype != requested_config_dtype:
        execution_config = execution_config.model_copy(
            update={
                "model": execution_config.model.model_copy(
                    update={"dtype": requested_config_dtype}
                )
            }
        )
    raw_rows = load_raw_examples(source_path)
    raw_by_image_id: dict[str, Any] = {}
    for row in raw_rows:
        source = row.metadata.get("source", {})
        image_id = source.get("image_id") if hasattr(source, "get") else None
        if image_id is not None:
            raw_by_image_id[str(int(image_id))] = row
    missing = [image_id for image_id, _ in selected if image_id not in raw_by_image_id]
    if missing:
        raise SystemExit(f"selected images are absent from the source JSONL: {missing}")

    runtime = assemble_runtime(execution_config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    model_dtype = str(next(iter(qwen.model.parameters())).dtype)
    expected_dtype = TORCH_DTYPE_BY_PUBLIC_NAME[str(args.model_dtype)]
    if model_dtype != expected_dtype:
        raise SystemExit(f"requested model dtype {expected_dtype}, received {model_dtype}")

    template_config = _template_config(resolved.config)
    case_inputs: list[dict[str, Any]] = []
    for image_id, source_bundle_path in selected:
        bundle_path = source_bundle_path.expanduser().resolve(strict=True)
        source_bundle = _read_json(bundle_path)
        source_evidence = validate_source_bundle(source_bundle, image_id=image_id)
        source_row = extract_first_complete_row(source_bundle)
        raw = raw_by_image_id[image_id]
        base_prompt = build_prompt_record(
            raw,
            template_config,
            processor=qwen.processor,
            row_index=0,
        )
        continuation = AssistantContinuation(
            text=qwen.tokenizer.decode(
                source_row["row_token_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        recipient_prompt = build_prompt_record(
            raw,
            template_config,
            processor=qwen.processor,
            row_index=0,
            assistant_continuation=continuation,
        )
        if list(base_prompt.prompt_token_ids) != source_row["prompt_token_ids"]:
            raise SystemExit(f"image {image_id} base prompt differs from its source bundle")
        expected_recipient = [
            *source_row["prompt_token_ids"],
            *source_row["row_token_ids"],
        ]
        if list(recipient_prompt.prompt_token_ids) != expected_recipient:
            raise SystemExit(
                f"image {image_id} recipient prompt is not exact source prompt plus first row"
            )
        case_inputs.append(
            {
                "image_id": image_id,
                "raw": raw,
                "source_bundle_path": bundle_path,
                "source_bundle": source_bundle,
                "source_evidence": source_evidence,
                "source_row": source_row,
                "recipient_prompt": recipient_prompt,
            }
        )

    image_plan = materialize_image_plan_batch(
        [case["raw"] for case in case_inputs],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=list(range(len(case_inputs))),
    )
    generation_fingerprint = sha256_json(
        resolved.config.generation.model_dump(mode="json")
    )
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    policy = DecodeGenerationPolicy.greedy(
        max_new_tokens=int(args.max_new_tokens), repetition_penalty=1.0
    )
    cases: list[dict[str, Any]] = []
    for case in case_inputs:
        image_id = case["image_id"]
        raw = case["raw"]
        prompt = case["recipient_prompt"]
        model_inputs = image_plan.model_inputs_by_row_id[raw.example_id]
        identity_checks = _validate_identity_continuity(
            source_bundle=case["source_bundle"],
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            source_image_sha256=_sha256_file(raw.image.path),
        )
        prompt_hash = _json_hash(prompt.prompt_token_ids)
        single_request = DecodeRequest(
            request_id=f"batch-precision-prevalence:{image_id}:single:{prompt_hash[:12]}",
            prompt_token_ids=list(prompt.prompt_token_ids),
            model_inputs=model_inputs,
            generation_policy=policy,
        )
        single_result = backend.generate_batch(
            [single_request],
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
        )[0]
        homogeneous_requests = [
            DecodeRequest(
                request_id=(
                    f"batch-precision-prevalence:{image_id}:homogeneous-"
                    f"position-{position}:{prompt_hash[:12]}"
                ),
                prompt_token_ids=list(prompt.prompt_token_ids),
                model_inputs=model_inputs,
                generation_policy=policy,
            )
            for position in range(4)
        ]
        homogeneous_results = backend.generate_batch(
            homogeneous_requests,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
        )
        single_record = _compact_result(single_result, raw=raw)
        homogeneous_records = [
            _compact_result(result, raw=raw) for result in homogeneous_results
        ]
        comparison = compare_action_records(single_record, homogeneous_records)
        cases.append(
            {
                "image_id": image_id,
                "source_bundle_path": str(case["source_bundle_path"]),
                "source_bundle_sha256": _sha256_file(case["source_bundle_path"]),
                "source_evidence": case["source_evidence"],
                "source_prompt_token_ids_sha256": case["source_row"][
                    "prompt_token_ids_sha256"
                ],
                "source_first_row": {
                    key: value
                    for key, value in case["source_row"].items()
                    if key != "prompt_token_ids"
                },
                "recipient_prompt_token_ids_sha256": prompt_hash,
                "recipient_prompt_token_count": len(prompt.prompt_token_ids),
                "identity_checks": identity_checks,
                "single_recipient": single_record,
                "homogeneous_four_copy_recipients": homogeneous_records,
                "comparison": comparison,
            }
        )

    divergent_ids = [
        case["image_id"]
        for case in cases
        if case["comparison"]["primary_first_action_divergence"]
    ]
    return {
        "schema_version": "selected_transition_batch_precision_prevalence.v1",
        "status": "complete",
        "requested_model_execution_dtype": str(args.model_dtype),
        "actual_model_execution_dtype": model_dtype,
        "infer_config_path": str(config_path),
        "infer_config_sha256": _sha256_file(config_path),
        "source_jsonl_path": str(source_path),
        "source_jsonl_sha256": _sha256_file(source_path),
        "decode_policy": {
            "mode": "greedy",
            "repetition_penalty": 1.0,
            "max_new_tokens": int(args.max_new_tokens),
            "interpretation": "first_free_action_only",
        },
        "selected_image_ids": [image_id for image_id, _ in selected],
        "case_count": len(cases),
        "primary_first_action_divergence_count": len(divergent_ids),
        "primary_first_action_divergent_image_ids": divergent_ids,
        "all_homogeneous_copy_checks_passed": True,
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_screen(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_root = args.output_root.expanduser().resolve()
    receipt_path = output_root / "receipt.json"
    _write_bundle_once(receipt_path, result)
    print(
        json.dumps(
            {
                "receipt_path": str(receipt_path),
                "model_dtype": result["actual_model_execution_dtype"],
                "case_count": result["case_count"],
                "divergent_image_ids": result[
                    "primary_first_action_divergent_image_ids"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
