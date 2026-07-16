#!/usr/bin/env python3
"""Run the frozen raw-pixel visual-support counterfactual commit panel.

This runner deliberately composes lossless in-memory RGB arrays and then sends
each condition through the ordinary no-resize image processor and Qwen3-VL
visual tower.  It does not replay or replace contextualized features.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.visual_support_counterfactual import (
    FeatureLayoutError,
    FreshFeatureCaptureController,
    PixelBounds,
    compose_rgb_patch,
    file_sha256,
    load_rgb_uint8,
    materialize_rgb_uint8,
    rgb_array_sha256,
)
from src.analysis.sampled_rescue_transition.artifacts import box_iou

from scripts.research.run_visual_support_counterfactual_commit import (
    _decode_dict,
    _evidence_image_id,
    _find_raw,
    _first_action_label,
    _hash_ids,
    _int_list,
    _result_record,
    _safe_name,
)
from scripts.research.run_sampled_rescue_transition import (
    _compose_complete_row_from_receipts,
    _temporary_cwd,
    _verify_donor_runtime_identity,
)


UNIT_ID = "2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit"
RECIPIENT_IMAGE_ID = "12576"
DONOR_IMAGE_ID = "17436"
TARGET_BOUNDS = PixelBounds(251, 360, 392, 624)
CONTROL_BOUNDS = PixelBounds(150, 20, 291, 284)
EXPECTED_SHAPE = (1152, 864, 3)
EXPECTED_GRID = (1, 72, 54)
EXPECTED_MERGED_TOKENS = 972
FROZEN_PREFIX_TOKEN_COUNT = 56
FROZEN_PROMPT_TOKEN_HASH = "11f6c79dcd2b570cf5794238994aa319f1bb23f8a580ffef850dafb6bc59b864"
FROZEN_COHERENT_ROW_TOKEN_HASH = "0e65d82cfbcc12af3c357266848c5f7bc4bf94eab5ba3ea806c953ec073d6355"
FROZEN_MAX_NEW_TOKENS = 512
FROZEN_SAMPLING_TEMPERATURE = 0.4
FROZEN_TARGET_OBJECT_ID = "678023"
FROZEN_RIGHT_CUP_OBJECT_ID = "678923"
FROZEN_TARGET_PIZZA_OBJECT_ID = "1571077"
SUPPORTED_VALID_ALTERNATIVE_LABELS = frozenset(
    {"left_cup_revisit", "target_pizza_fallback", "another_supported_object"}
)
FROZEN_SEEDS = (
    602711627830374173,
    5641984501295450920,
    8306462649179848189,
    8458627694586881429,
    4423663331540486457,
    1604446646505700360,
    6481035254874347622,
    6885534711486411115,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen pre-vision raw-pixel visual-support panel."
    )
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--recipient-bundle", type=Path, required=True)
    parser.add_argument("--recipient-prefix-token-count", type=int, required=True)
    parser.add_argument("--description-row-receipt", type=Path, required=True)
    parser.add_argument("--geometry-row-receipt", type=Path, required=True)
    parser.add_argument("--donor-review-note", required=True)
    parser.add_argument("--donor-image-id", default=DONOR_IMAGE_ID)
    parser.add_argument("--target-object-id", default="678023")
    parser.add_argument("--right-cup-object-id", default="678923")
    parser.add_argument("--target-pizza-object-id", default="1571077")
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampled-runtime-attestation", type=Path)
    parser.add_argument("--sampling-seed", action="append", type=int)
    parser.add_argument("--sampling-temperature", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--include-samples", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_panel(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def run_panel(args: argparse.Namespace) -> dict[str, object]:
    _assert_frozen_cli(args)
    seeds = tuple(int(x) for x in (args.sampling_seed or FROZEN_SEEDS))
    if args.include_samples and seeds != FROZEN_SEEDS:
        raise SystemExit(f"sampling seeds must exactly match frozen ordered vector: {FROZEN_SEEDS}")

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import (
        DecodeGenerationPolicy,
        HFGenerateBackend,
        load_and_rebind_sampled_runtime_attestation_aggregate,
    )
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.runtime import assemble_runtime
    from scripts.research.run_visual_support_counterfactual_commit import _decode_request

    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    rows = load_raw_examples(source_path)
    bundle = _read_json(args.recipient_bundle)
    decode = _decode_dict(bundle)
    prompt_ids = _int_list(decode.get("prompt_token_ids"))
    generated_ids = _int_list(decode.get("generated_token_ids"))
    recipient_count = int(args.recipient_prefix_token_count)
    if recipient_count > len(generated_ids):
        raise SystemExit("recipient prefix token count exceeds recipient generated token count")
    recipient_raw = _find_raw(rows, _evidence_image_id(bundle))
    donor_raw = _find_raw(rows, args.donor_image_id)
    if str(recipient_raw.metadata["source"]["image_id"]) != RECIPIENT_IMAGE_ID:
        raise SystemExit(f"recipient must be frozen image {RECIPIENT_IMAGE_ID}")
    if str(donor_raw.metadata["source"]["image_id"]) != DONOR_IMAGE_ID:
        raise SystemExit(f"donor must be frozen image {DONOR_IMAGE_ID}")

    recipient_rgb = load_rgb_uint8(Path(recipient_raw.image.path))
    donor_rgb = load_rgb_uint8(Path(donor_raw.image.path))
    if tuple(recipient_rgb.shape) != EXPECTED_SHAPE or tuple(donor_rgb.shape) != EXPECTED_SHAPE:
        raise SystemExit(f"frozen RGB shape mismatch: {recipient_rgb.shape} {donor_rgb.shape}")
    TARGET_BOUNDS.validate(width=recipient_rgb.shape[1], height=recipient_rgb.shape[0])
    CONTROL_BOUNDS.validate(width=recipient_rgb.shape[1], height=recipient_rgb.shape[0])

    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    generation_fingerprint = sha256_json(resolved.config.generation.model_dump(mode="json"))
    _verify_donor_runtime_identity(
        donor=bundle,
        active_image_id=str(recipient_raw.metadata["source"]["image_id"]),
        active_model_identity=model_identity,
        active_tokenizer_identity=tokenizer_identity,
        active_generation_config_fingerprint=generation_fingerprint,
    )

    composed_row = _compose_complete_row_from_receipts(args.description_row_receipt, args.geometry_row_receipt)
    for source_call_bundle in composed_row["source_call_bundles"]:
        if isinstance(source_call_bundle, Mapping):
            _verify_donor_runtime_identity(
                donor=dict(source_call_bundle),
                active_image_id=str(recipient_raw.metadata["source"]["image_id"]),
                active_model_identity=model_identity,
                active_tokenizer_identity=tokenizer_identity,
                active_generation_config_fingerprint=generation_fingerprint,
            )
    recipient_plan = materialize_image_plan_batch(
        [recipient_raw], components=qwen, processor_config=_processor_config(resolved.config), materialize=True, row_indices=[0]
    )
    recipient_plan_row = recipient_plan.rows[0]
    recipient_inputs = recipient_plan.model_inputs_by_row_id[recipient_raw.example_id]
    if tuple(recipient_plan_row.expected_image_grid_thw) != EXPECTED_GRID:
        raise SystemExit(f"frozen image grid mismatch: {recipient_plan_row.expected_image_grid_thw}")
    if int(recipient_plan_row.merged_visual_tokens) != EXPECTED_MERGED_TOKENS:
        raise SystemExit("frozen merged visual-token count mismatch")

    base_prompt = build_prompt_record(recipient_raw, _template_config(resolved.config), processor=qwen.processor, row_index=0)
    if list(base_prompt.prompt_token_ids) != prompt_ids:
        raise SystemExit("recipient bundle prompt does not match active prompt")
    composed_ids = [*generated_ids[:recipient_count], *[int(x) for x in composed_row["token_ids"]]]
    continuation_text = qwen.tokenizer.decode(composed_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    prompt = build_prompt_record(
        recipient_raw, _template_config(resolved.config), processor=qwen.processor, row_index=0,
        assistant_continuation=AssistantContinuation(text=continuation_text),
    )
    resulting_prompt_ids = list(prompt.prompt_token_ids)
    if resulting_prompt_ids != [*prompt_ids, *composed_ids]:
        raise SystemExit("active prompt does not equal exact frozen prefix plus coherent row")
    prompt_hash = _hash_ids(resulting_prompt_ids)
    row_hash = _hash_ids(composed_row["token_ids"])
    if prompt_hash != FROZEN_PROMPT_TOKEN_HASH:
        raise SystemExit(
            "frozen recipient prompt-token hash mismatch: "
            f"{prompt_hash} != {FROZEN_PROMPT_TOKEN_HASH}"
        )
    if row_hash != FROZEN_COHERENT_ROW_TOKEN_HASH:
        raise SystemExit(
            "frozen coherent-row token hash mismatch: "
            f"{row_hash} != {FROZEN_COHERENT_ROW_TOKEN_HASH}"
        )

    conditions: list[tuple[str, str, PixelBounds | None]] = [
        ("clean_pixel_replay", "clean", None),
        ("target_raw_bbox_donor_patch", "target", TARGET_BOUNDS),
        ("equal_shape_unrelated_region_donor_patch", "control", CONTROL_BOUNDS),
    ]
    target_control_disjoint = _pixel_bounds_disjoint(TARGET_BOUNDS, CONTROL_BOUNDS)
    if not target_control_disjoint:
        raise SystemExit("frozen target and control pixel bounds overlap")
    composed: dict[str, dict[str, Any]] = {}
    for condition, mode, bounds in conditions:
        pixels, receipt = compose_rgb_patch(recipient_rgb, donor_rgb, bounds=bounds, mode=mode)  # type: ignore[arg-type]
        inputs, processor_receipt = materialize_rgb_uint8(pixels, image_processor=qwen.processor.image_processor)
        grid = tuple(int(x) for x in inputs["image_grid_thw"].reshape(-1).tolist())
        if grid != EXPECTED_GRID:
            raise SystemExit(f"{condition} processor grid mismatch: {grid}")
        composed[condition] = {"pixels": pixels, "inputs": inputs, "pixel": receipt, "processor": processor_receipt}
    processor_hashes = {name: str(record["processor"]["processor_output_sha256"]) for name, record in composed.items()}
    if len(set(processor_hashes.values())) != 3:
        raise SystemExit(f"condition-specific processor hashes are not distinct: {processor_hashes}")
    clean_inputs = composed["clean_pixel_replay"]["inputs"]
    _require_tensor_equal(recipient_inputs["pixel_values"], clean_inputs["pixel_values"], "clean pixel processor tensors")
    _require_tensor_equal(recipient_inputs["image_grid_thw"], clean_inputs["image_grid_thw"], "clean pixel processor grid")

    backend = HFGenerateBackend(
        model=qwen.model, tokenizer=qwen.tokenizer, model_identity=model_identity,
        tokenizer_identity=tokenizer_identity, generation_config_fingerprint=generation_fingerprint,
    )
    greedy_policy = DecodeGenerationPolicy.greedy(max_new_tokens=int(args.max_new_tokens), repetition_penalty=1.0)
    standard_greedy = backend.generate_batch(
        [_decode_request(image_id=RECIPIENT_IMAGE_ID, prompt_ids=resulting_prompt_ids, model_inputs=recipient_inputs, policy=greedy_policy, condition="standard_clean", index=0)],
        model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_config_fingerprint=generation_fingerprint,
    )[0]
    condition_results: dict[str, dict[str, Any]] = {}
    for condition, mode, _bounds in conditions:
        request = _decode_request(
            image_id=RECIPIENT_IMAGE_ID, prompt_ids=resulting_prompt_ids, model_inputs=composed[condition]["inputs"],
            policy=greedy_policy, condition=condition, index=0,
        )
        result, visual_receipt = _run_fresh(
            backend=backend, model=qwen.model, expected_grid=EXPECTED_GRID, request=request,
            model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_fingerprint=generation_fingerprint,
        )
        record = _result_record(
            result, qwen.tokenizer, recipient_raw, args.bundle_root, condition,
            right_cup_object_id=str(args.right_cup_object_id), target_object_id=str(args.target_object_id), target_pizza_object_id=str(args.target_pizza_object_id),
        )
        condition_results[condition] = {"condition": condition, "mode": mode, "pixel": composed[condition]["pixel"], "processor": composed[condition]["processor"], "greedy": record, "greedy_result": result, "fresh_visual": visual_receipt}

    clean_greedy = condition_results["clean_pixel_replay"]["greedy_result"] if "greedy_result" in condition_results["clean_pixel_replay"] else None
    clean_result = _find_result(condition_results["clean_pixel_replay"])
    greedy_parity = _require_noop_parity_local(standard_greedy, clean_result, scope="greedy")

    sampled_noop: list[dict[str, object]] = []
    capability = None
    standard_sampled: list[Any] = []
    sampled_requests: list[Any] = []
    if args.include_samples:
        sampled_policy = DecodeGenerationPolicy.sampled(max_new_tokens=int(args.max_new_tokens), repetition_penalty=1.0, temperature=float(args.sampling_temperature), top_p=0.95)
        sampled_requests = [
            _decode_request(image_id=RECIPIENT_IMAGE_ID, prompt_ids=resulting_prompt_ids, model_inputs=recipient_inputs, policy=sampled_policy, condition="standard_sample", index=index, seed=seed)
            for index, seed in enumerate(seeds)
        ]
        capability = _load_capability(args, backend=backend, sampled_policy=sampled_policy)
        standard_sampled = backend.generate_batch_with_verified_runtime_attestation(
            sampled_requests, model_identity=model_identity, tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint, verified_runtime_attestation=capability,
        )
        clean_sampled, clean_sample_visual = _run_fresh_batch(
            backend=backend, model=qwen.model, expected_grid=EXPECTED_GRID,
            requests=[_decode_request(image_id=RECIPIENT_IMAGE_ID, prompt_ids=resulting_prompt_ids, model_inputs=clean_inputs, policy=sampled_policy, condition="clean_sample", index=index, seed=seed) for index, seed in enumerate(seeds)],
            capability=capability, model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_fingerprint=generation_fingerprint,
        )
        for index, (left, right) in enumerate(zip(standard_sampled, clean_sampled, strict=True)):
            sampled_noop.append(_require_noop_parity_local(left, right, scope=f"sample-{index}"))
        condition_results["clean_pixel_replay"]["samples"] = [_result_record(x, qwen.tokenizer, recipient_raw, args.bundle_root, "clean_pixel_replay", right_cup_object_id=str(args.right_cup_object_id), target_object_id=str(args.target_object_id), target_pizza_object_id=str(args.target_pizza_object_id)) for x in clean_sampled]
        condition_results["clean_pixel_replay"]["sample_fresh_visual"] = clean_sample_visual
        for condition, _mode, _bounds in conditions[1:]:
            requests = [_decode_request(image_id=RECIPIENT_IMAGE_ID, prompt_ids=resulting_prompt_ids, model_inputs=composed[condition]["inputs"], policy=sampled_policy, condition=condition, index=index, seed=seed) for index, seed in enumerate(seeds)]
            results, visual = _run_fresh_batch(
                backend=backend, model=qwen.model, expected_grid=EXPECTED_GRID, requests=requests,
                capability=capability, model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_fingerprint=generation_fingerprint,
            )
            condition_results[condition]["samples"] = [_result_record(x, qwen.tokenizer, recipient_raw, args.bundle_root, condition, right_cup_object_id=str(args.right_cup_object_id), target_object_id=str(args.target_object_id), target_pizza_object_id=str(args.target_pizza_object_id)) for x in results]
            condition_results[condition]["sample_fresh_visual"] = visual

    condition_records = []
    for condition, _mode, _bounds in conditions:
        data = condition_results[condition]
        record = {key: value for key, value in data.items() if key != "greedy_result"}
        condition_records.append(record)
    fresh_receipts = [
        record.get("fresh_visual")
        for record in condition_records
    ]
    fresh_receipts_valid = all(
        isinstance(receipt, Mapping)
        and bool(receipt.get("fresh_visual_recomputation"))
        and int(receipt.get("feature_call_count", 0)) == 1
        and bool(receipt.get("hook_restored"))
        and len(receipt.get("calls", ())) == 1
        and len(receipt["calls"][0].get("primary", ())) >= 1
        and len(receipt["calls"][0].get("deepstack", ())) == 3
        for receipt in fresh_receipts
    )
    pixel_receipts_valid = all(
        isinstance(record.get("pixel"), Mapping)
        and bool(record["pixel"].get("selected_slice_exact_equal_to_donor"))
        and bool(record["pixel"].get("complement_exact_equal_to_recipient"))
        for record in condition_records
    )
    processor_receipts_valid = all(
        isinstance(record.get("processor"), Mapping)
        and record["processor"].get("do_resize") is False
        and tuple(record["processor"].get("image_grid_thw", ())) == EXPECTED_GRID
        for record in condition_records
    )
    sampled_noop_valid = (
        not args.include_samples
        or bool(sampled_noop) and all(bool(item.get("verified")) for item in sampled_noop)
    )
    trust_inputs: dict[str, object] = {
        "frozen_cli_arguments": True,
        "frozen_prompt_and_row_hashes": True,
        "target_control_disjoint": target_control_disjoint,
        "pixel_selected_and_complement_exact": pixel_receipts_valid,
        "processor_no_resize_grid": processor_receipts_valid,
        "processor_hashes_distinct": len(set(processor_hashes.values())) == 3,
        "clean_processor_parity": bool(_processor_parity(recipient_inputs, clean_inputs)["verified"]),
        "greedy_noop_parity": bool(greedy_parity.get("verified")),
        "sampled_noop_parity": sampled_noop_valid,
        "fresh_primary_and_three_deepstack_recomputation": fresh_receipts_valid,
    }
    trust_gate = {
        "status": "passed" if all(bool(value) for value in trust_inputs.values()) else "failed",
        "input": trust_inputs,
    }
    if args.include_samples:
        sample_by_condition = {
            str(record["condition"]): [
                str(item.get("first_action_label", "unknown"))
                for item in record.get("samples", [])
                if isinstance(item, Mapping)
            ]
            for record in condition_records
        }
        paired_summary = summarize_paired_first_action_labels(
            sample_by_condition.get("clean_pixel_replay", []),
            sample_by_condition.get("target_raw_bbox_donor_patch", []),
            sample_by_condition.get("equal_shape_unrelated_region_donor_patch", []),
            seeds=seeds,
            trust_gate=trust_gate,
        )
    else:
        paired_summary = summarize_paired_first_action_labels(
            [], [], [], trust_gate=trust_gate,
        )
    summary = paired_summary
    return {
        "schema_version": "prevision_raw_pixel_visual_support_counterfactual_commit.v1",
        "unit_id": UNIT_ID,
        "image_id": RECIPIENT_IMAGE_ID,
        "donor_image_id": DONOR_IMAGE_ID,
        "donor_review_note": str(args.donor_review_note),
        "config_path": str(config_path),
        "source_jsonl": str(source_path),
        "source_paths": {"recipient": str(recipient_raw.image.path), "donor": str(donor_raw.image.path)},
        "source_file_sha256": {"recipient": file_sha256(Path(recipient_raw.image.path)), "donor": file_sha256(Path(donor_raw.image.path))},
        "recipient_rgb_sha256": rgb_array_sha256(recipient_rgb),
        "donor_rgb_sha256": rgb_array_sha256(donor_rgb),
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "generation_config_fingerprint": generation_fingerprint,
        "prompt_token_hash": prompt_hash,
        "prompt_token_count": len(resulting_prompt_ids),
        "coherent_row_token_hash": row_hash,
        "coherent_row_token_ids": [int(x) for x in composed_row["token_ids"]],
        "recipient_prefix_token_count": recipient_count,
        "expected_grid_thw": list(EXPECTED_GRID),
        "merged_visual_tokens": EXPECTED_MERGED_TOKENS,
        "do_resize": False,
        "target_bounds_xyxy_half_open": TARGET_BOUNDS.to_list(),
        "control_bounds_xyxy_half_open": CONTROL_BOUNDS.to_list(),
        "target_control_disjoint": target_control_disjoint,
        "processor_hashes": processor_hashes,
        "processor_hashes_distinct": True,
        "recipient_clean_processor_parity": _processor_parity(recipient_inputs, clean_inputs),
        "standard_clean_greedy": _result_record(standard_greedy, qwen.tokenizer, recipient_raw, args.bundle_root, "standard_clean", right_cup_object_id=str(args.right_cup_object_id), target_object_id=str(args.target_object_id), target_pizza_object_id=str(args.target_pizza_object_id)),
        "noop_greedy": greedy_parity,
        "noop_sampled": sampled_noop,
        "conditions": condition_records,
        "sampled_first_action_summary": summary,
        "paired_first_action_summary": paired_summary,
        "trust_gate": trust_gate,
        "sampling_seeds": list(seeds) if args.include_samples else [],
        "primary_estimand": {"definition": "fraction of eight paired requests whose first_action_label is right_cup", "target_control_difference": "R(equal_shape_unrelated_region_donor_patch) - R(target_raw_bbox_donor_patch)"},
    }


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SystemExit(f"JSON bundle must be an object: {path}")
    return value


def _assert_frozen_cli(args: argparse.Namespace) -> None:
    """Reject any override that would change the frozen scientific panel."""

    checks = {
        "recipient_prefix_token_count": (int(args.recipient_prefix_token_count), FROZEN_PREFIX_TOKEN_COUNT),
        "max_new_tokens": (int(args.max_new_tokens), FROZEN_MAX_NEW_TOKENS),
        "sampling_temperature": (float(args.sampling_temperature), FROZEN_SAMPLING_TEMPERATURE),
        "donor_image_id": (str(args.donor_image_id), DONOR_IMAGE_ID),
        "target_object_id": (str(args.target_object_id), FROZEN_TARGET_OBJECT_ID),
        "right_cup_object_id": (str(args.right_cup_object_id), FROZEN_RIGHT_CUP_OBJECT_ID),
        "target_pizza_object_id": (str(args.target_pizza_object_id), FROZEN_TARGET_PIZZA_OBJECT_ID),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise SystemExit(f"frozen pre-vision panel arguments mismatch: {mismatches}")


def _pixel_bounds_disjoint(left: PixelBounds, right: PixelBounds) -> bool:
    """Return whether two half-open rectangles have no shared pixel."""

    return bool(
        left.x2 <= right.x1
        or right.x2 <= left.x1
        or left.y2 <= right.y1
        or right.y2 <= left.y1
    )


def summarize_paired_first_action_labels(
    clean_labels: Sequence[str],
    target_labels: Sequence[str],
    control_labels: Sequence[str],
    *,
    seeds: Sequence[int] | None = None,
    trust_gate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Summarize exact paired labels using the frozen unit decision rules.

    This is intentionally independent of model outputs, parsing, or metrics.
    The three sequences are aligned by index and must all contain the eight
    frozen requests before a scientific verdict is evaluated.
    """

    clean = [str(value) for value in clean_labels]
    target = [str(value) for value in target_labels]
    control = [str(value) for value in control_labels]
    lengths = {"clean": len(clean), "target": len(target), "control": len(control)}
    if len({len(clean), len(target), len(control)}) != 1:
        raise ValueError(f"paired label lengths differ: {lengths}")
    if seeds is not None and len(seeds) != len(clean):
        raise ValueError("seed count must match paired label count")
    seed_values = [int(value) for value in (seeds or range(len(clean)))]
    per_seed = [
        {
            "index": index,
            "seed": seed_values[index],
            "clean": clean[index],
            "target": target[index],
            "control": control[index],
            "target_selective_loss": clean[index] == "right_cup" and control[index] == "right_cup" and target[index] != "right_cup",
            "reverse_selective_loss": clean[index] == "right_cup" and target[index] == "right_cup" and control[index] != "right_cup",
        }
        for index in range(len(clean))
    ]
    target_selective = [row for row in per_seed if row["target_selective_loss"]]
    reverse_selective = [row for row in per_seed if row["reverse_selective_loss"]]
    supported_alternatives = sum(
        target[index] in SUPPORTED_VALID_ALTERNATIVE_LABELS
        for index in range(len(target))
        if clean[index] == "right_cup" and control[index] == "right_cup" and target[index] != "right_cup"
    )
    right_cup_fractions = {
        "clean": _right_cup_fraction(clean),
        "target": _right_cup_fraction(target),
        "control": _right_cup_fraction(control),
    }
    gate = dict(trust_gate or {})
    gate_status = gate.get("status", "failed")
    gate_passed = gate_status in {True, "passed", "pass", "verified", "ok"}
    gate_record = {
        "status": "passed" if gate_passed else "failed",
        "input": dict(gate.get("input", gate.get("inputs", {}))) if isinstance(gate.get("input", gate.get("inputs", {})), Mapping) else gate.get("input", gate.get("inputs", {})),
    }
    if not gate_passed:
        verdict = "inconclusive"
    elif len(clean) != len(FROZEN_SEEDS):
        verdict = "not_evaluated"
    else:
        clean_retained = sum(value == "right_cup" for value in clean)
        target_retained = sum(value == "right_cup" for value in target)
        control_retained = sum(value == "right_cup" for value in control)
        target_loss_count = len(target) - target_retained
        asymmetric_invalidity_collapse = bool(gate.get("asymmetric_invalidity_collapse", False))
        positive = (
            clean_retained >= 7
            and control_retained >= 7
            and target_loss_count >= 6
            and len(target_selective) >= 6
            and len(reverse_selective) == 0
            and supported_alternatives >= (len(target_selective) // 2 + 1)
        )
        strong_null = (
            clean_retained >= 7
            and target_retained >= 7
            and control_retained >= 7
            and len(target_selective) <= 1
            and len(reverse_selective) <= 1
            and not asymmetric_invalidity_collapse
        )
        verdict = "positive" if positive else "strong_null" if strong_null else "inconclusive"
    return {
        "per_seed": per_seed,
        "target_selective_count": len(target_selective),
        "reverse_selective_count": len(reverse_selective),
        "supported_valid_alternative_count": int(supported_alternatives),
        "condition_right_cup_fractions": right_cup_fractions,
        "trust_gate": gate_record,
        "verdict": verdict,
        "request_count": len(clean),
        "label_lengths": lengths,
    }


def _right_cup_fraction(labels: Sequence[str]) -> float | None:
    return None if not labels else float(sum(value == "right_cup" for value in labels) / len(labels))


def _require_tensor_equal(left: torch.Tensor, right: torch.Tensor, scope: str) -> None:
    if not torch.equal(left, right):
        raise SystemExit(f"{scope} mismatch")


def _processor_parity(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, object]:
    _require_tensor_equal(left["pixel_values"], right["pixel_values"], "clean processor pixel_values")
    _require_tensor_equal(left["image_grid_thw"], right["image_grid_thw"], "clean processor image_grid_thw")
    from src.analysis.visual_support_counterfactual import tensor_sha256
    return {"verified": True, "pixel_values_sha256_equal": tensor_sha256(left["pixel_values"]) == tensor_sha256(right["pixel_values"]), "image_grid_thw_sha256_equal": tensor_sha256(left["image_grid_thw"]) == tensor_sha256(right["image_grid_thw"])}


def _run_fresh(*, backend: Any, model: Any, expected_grid: Sequence[int], request: Any, model_identity: Mapping[str, Any], tokenizer_identity: Mapping[str, Any], generation_fingerprint: str) -> tuple[Any, dict[str, object]]:
    with FreshFeatureCaptureController(model=model, expected_grid_thw=expected_grid) as controller:
        result = backend.generate_batch([request], model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_config_fingerprint=generation_fingerprint)[0]
    return result, controller.validate_completed(expected_feature_calls=1)


def _run_fresh_batch(*, backend: Any, model: Any, expected_grid: Sequence[int], requests: Sequence[Any], capability: Any, model_identity: Mapping[str, Any], tokenizer_identity: Mapping[str, Any], generation_fingerprint: str) -> tuple[list[Any], dict[str, object]]:
    with FreshFeatureCaptureController(model=model, expected_grid_thw=expected_grid) as controller:
        results = backend.generate_batch_with_verified_runtime_attestation(requests, model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_config_fingerprint=generation_fingerprint, verified_runtime_attestation=capability)
    return results, controller.validate_completed(expected_feature_calls=1)


def _find_result(record: Mapping[str, Any]) -> Any:
    result = record.get("greedy_result")
    if result is None:
        raise RuntimeError("internal result retention error")
    return result


def _require_noop_parity_local(left: Any, right: Any, *, scope: str) -> dict[str, object]:
    if [int(x) for x in left.generated_token_ids] != [int(x) for x in right.generated_token_ids]:
        raise SystemExit(f"no-op parity failed for {scope}: generated tokens differ")
    if left.execution_receipt is None or right.execution_receipt is None:
        raise SystemExit(f"no-op parity failed for {scope}: missing execution receipt")
    left_hash = str(left.execution_receipt.canonical_float32_score_trace_hash)
    right_hash = str(right.execution_receipt.canonical_float32_score_trace_hash)
    if left_hash != right_hash:
        raise SystemExit(f"no-op parity failed for {scope}: score traces differ")
    return {"verified": True, "scope": scope, "generated_token_ids_sha256": _hash_ids(left.generated_token_ids), "canonical_float32_score_trace_hash": left_hash}


def _load_capability(args: argparse.Namespace, *, backend: Any, sampled_policy: Any) -> Any:
    from src.inference.backend import load_and_rebind_sampled_runtime_attestation_aggregate
    path = args.sampled_runtime_attestation or Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/runtime-attestation-v2/request-scoped-sampling-three-policy-cuda.json")
    return load_and_rebind_sampled_runtime_attestation_aggregate(path.expanduser().resolve(strict=True), decode_generation_policy_fingerprint=sampled_policy.fingerprint, backend=backend)


if __name__ == "__main__":
    raise SystemExit(main())
