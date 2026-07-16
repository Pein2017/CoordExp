#!/usr/bin/env python3
"""Run the fixed-encoding visual-support counterfactual commit panel.

This is an experiment-local consumer of the existing exact composed-row
runtime.  It caches the recipient and donor Qwen3 Vision-Language (Qwen3-VL)
feature streams once, replays those streams through the normal Hugging Face
generation backend, and changes only same-position feature rows for the
target or control support.  No training or pixel re-encoding is performed.
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

from src.analysis.sampled_rescue_transition.artifacts import box_iou
from src.analysis.visual_support_counterfactual import (
    FeatureBundle,
    FeatureLayoutError,
    FeatureReplayController,
    build_merged_support_mask,
    capture_feature_bundle,
    choose_deterministic_control_mask,
    feature_bundle_fingerprint,
    prune_symmetric_translation_overlap,
    validate_equal_mask_shape_and_count,
    validate_feature_layout,
)


DEFAULT_TARGET_OBJECT_ID = "678023"
DEFAULT_RIGHT_CUP_OBJECT_ID = "678923"
DEFAULT_TARGET_PIZZA_OBJECT_ID = "1571077"

# This is intentionally an experiment-local receipt guard, not a reusable
# runtime contract.  The pilot was manually frozen to one 864x1152 image
# encoding: Qwen3 Vision-Language (Qwen3-VL) grid [1, 72, 54], merge size 2,
# and a 36x27 merged support grid.
_FROZEN_PILOT_GRID_THW = (1, 72, 54)
_FROZEN_PILOT_MERGE_SIZE = 2
_FROZEN_PILOT_MERGED_SHAPE = (36, 27)
_FROZEN_PILOT_PRIMARY_STREAM_COUNT = 1
_FROZEN_PILOT_DEEPSTACK_STREAM_COUNT = 3
_FROZEN_PILOT_BASE_MASK_COUNT = 88
_FROZEN_PILOT_FINAL_MASK_COUNT = 86
_FROZEN_PILOT_REMOVED_CONTROL_RELATIVE_POSITIONS = ((10, 6), (10, 7))


def _assert_frozen_pilot_geometry(
    *,
    layout: Any,
    recipient_features: FeatureBundle,
    donor_features: FeatureBundle,
    target_mask: torch.Tensor,
    control_mask: torch.Tensor,
    control_pruning: Mapping[str, object],
) -> dict[str, object]:
    """Fail fast if the manually reviewed pilot geometry has drifted."""

    checks: list[tuple[str, bool, object]] = [
        ("grid_thw", tuple(layout.grid_thw) == _FROZEN_PILOT_GRID_THW, list(layout.grid_thw)),
        ("merge_size", int(layout.merge_size) == _FROZEN_PILOT_MERGE_SIZE, int(layout.merge_size)),
        (
            "merged_shape",
            (int(layout.merged_height), int(layout.merged_width)) == _FROZEN_PILOT_MERGED_SHAPE,
            [int(layout.merged_height), int(layout.merged_width)],
        ),
        (
            "recipient_primary_stream_count",
            len(recipient_features.primary) == _FROZEN_PILOT_PRIMARY_STREAM_COUNT,
            len(recipient_features.primary),
        ),
        (
            "donor_primary_stream_count",
            len(donor_features.primary) == _FROZEN_PILOT_PRIMARY_STREAM_COUNT,
            len(donor_features.primary),
        ),
        (
            "recipient_deepstack_stream_count",
            len(recipient_features.deepstack) == _FROZEN_PILOT_DEEPSTACK_STREAM_COUNT,
            len(recipient_features.deepstack),
        ),
        (
            "donor_deepstack_stream_count",
            len(donor_features.deepstack) == _FROZEN_PILOT_DEEPSTACK_STREAM_COUNT,
            len(donor_features.deepstack),
        ),
        (
            "target_base_mask_count",
            len(control_pruning.get("target_base_indices", ())) == _FROZEN_PILOT_BASE_MASK_COUNT,
            len(control_pruning.get("target_base_indices", ())),
        ),
        (
            "control_base_mask_count",
            len(control_pruning.get("control_base_indices", ())) == _FROZEN_PILOT_BASE_MASK_COUNT,
            len(control_pruning.get("control_base_indices", ())),
        ),
        (
            "target_final_mask_count",
            int(target_mask.sum().item()) == _FROZEN_PILOT_FINAL_MASK_COUNT,
            int(target_mask.sum().item()),
        ),
        (
            "control_final_mask_count",
            int(control_mask.sum().item()) == _FROZEN_PILOT_FINAL_MASK_COUNT,
            int(control_mask.sum().item()),
        ),
        (
            "removed_control_relative_positions",
            tuple(tuple(int(v) for v in row) for row in control_pruning.get("removed_relative_positions", ()))
            == _FROZEN_PILOT_REMOVED_CONTROL_RELATIVE_POSITIONS,
            control_pruning.get("removed_relative_positions", []),
        ),
        (
            "final_masks_disjoint",
            not bool((target_mask.to(dtype=torch.bool) & control_mask.to(dtype=torch.bool)).any()),
            bool((target_mask.to(dtype=torch.bool) & control_mask.to(dtype=torch.bool)).any()),
        ),
        (
            "receipt_disjoint",
            control_pruning.get("disjoint") is True,
            control_pruning.get("disjoint"),
        ),
    ]
    failed = [name for name, passed, _observed in checks if not passed]
    if failed:
        observed = {name: value for name, _passed, value in checks}
        raise SystemExit(
            "frozen visual-support pilot geometry contract failed: "
            f"failed={failed}; observed={observed}"
        )
    return {
        "verified": True,
        "grid_thw": list(_FROZEN_PILOT_GRID_THW),
        "merge_size": _FROZEN_PILOT_MERGE_SIZE,
        "merged_shape": list(_FROZEN_PILOT_MERGED_SHAPE),
        "primary_stream_count": _FROZEN_PILOT_PRIMARY_STREAM_COUNT,
        "deepstack_stream_count": _FROZEN_PILOT_DEEPSTACK_STREAM_COUNT,
        "base_mask_count": _FROZEN_PILOT_BASE_MASK_COUNT,
        "final_mask_count": _FROZEN_PILOT_FINAL_MASK_COUNT,
        "removed_control_relative_positions": [
            list(row) for row in _FROZEN_PILOT_REMOVED_CONTROL_RELATIVE_POSITIONS
        ],
        "disjoint": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-encoding visual-support counterfactual commit panel."
    )
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--recipient-bundle", type=Path, required=True)
    parser.add_argument("--recipient-prefix-token-count", type=int, required=True)
    parser.add_argument("--description-row-receipt", type=Path, required=True)
    parser.add_argument("--geometry-row-receipt", type=Path, required=True)
    parser.add_argument("--donor-image-id", default="17436")
    parser.add_argument(
        "--donor-source-jsonl",
        type=Path,
        help="Optional same-grid donor source; defaults to --source-jsonl.",
    )
    parser.add_argument(
        "--donor-review-note",
        required=True,
        help="Manual visual-review record identifying the frozen donor selection.",
    )
    parser.add_argument("--target-object-id", default=DEFAULT_TARGET_OBJECT_ID)
    parser.add_argument("--right-cup-object-id", default=DEFAULT_RIGHT_CUP_OBJECT_ID)
    parser.add_argument("--target-pizza-object-id", default=DEFAULT_TARGET_PIZZA_OBJECT_ID)
    parser.add_argument(
        "--target-bbox",
        type=_parse_bbox,
        help="Optional explicit pixel-space target bbox; otherwise source annotation is used.",
    )
    parser.add_argument(
        "--control-bbox",
        type=_parse_bbox,
        help="Optional explicit pixel-space control bbox; otherwise deterministic mask translation is used.",
    )
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampled-runtime-attestation", type=Path)
    parser.add_argument("--sampling-seed", action="append", type=int)
    parser.add_argument("--sampling-temperature", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Run the one-greedy/eight-sampled primary panel after no-op parity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_panel(args)
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def run_panel(args: argparse.Namespace) -> dict[str, object]:
    """Execute one bounded panel; callers should not use this for population sweeps."""

    if int(args.recipient_prefix_token_count) < 0:
        raise SystemExit("recipient prefix token count must be non-negative")
    if int(args.max_new_tokens) <= 0:
        raise SystemExit("max-new-tokens must be positive")
    donor_source_path = (
        args.donor_source_jsonl if args.donor_source_jsonl is not None else args.source_jsonl
    )

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import (
        DecodeGenerationPolicy,
        DecodeRequest,
        HFGenerateBackend,
        load_and_rebind_sampled_runtime_attestation_aggregate,
    )
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.parsing import parse_compact_object_box_closed
    from src.inference.runtime import assemble_runtime
    from scripts.research.run_sampled_rescue_transition import (
        _compose_complete_row_from_receipts,
        _first_action,
        _read_json_bundle,
        _temporary_cwd,
        _verify_donor_runtime_identity,
    )

    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    donor_source_path = donor_source_path.expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    recipient_rows = load_raw_examples(source_path)
    donor_rows = recipient_rows if donor_source_path == source_path else load_raw_examples(donor_source_path)
    recipient_bundle = _read_json_bundle(args.recipient_bundle)
    recipient_decode = _decode_dict(recipient_bundle)
    recipient_prompt_ids = _int_list(recipient_decode.get("prompt_token_ids"))
    recipient_generated_ids = _int_list(recipient_decode.get("generated_token_ids"))
    recipient_count = int(args.recipient_prefix_token_count)
    if not 0 <= recipient_count <= len(recipient_generated_ids):
        raise SystemExit("recipient prefix token count is outside recipient generated tokens")
    requested_recipient_image = _evidence_image_id(recipient_bundle)
    recipient_raw = _find_raw(recipient_rows, requested_recipient_image)
    donor_raw = _find_raw(donor_rows, args.donor_image_id)
    if donor_raw.example_id == recipient_raw.example_id:
        raise SystemExit("donor image must differ from recipient image")

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
        donor=recipient_bundle,
        active_image_id=str(recipient_raw.metadata["source"]["image_id"]),
        active_model_identity=model_identity,
        active_tokenizer_identity=tokenizer_identity,
        active_generation_config_fingerprint=generation_fingerprint,
    )
    composed_row = _compose_complete_row_from_receipts(
        args.description_row_receipt,
        args.geometry_row_receipt,
    )
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
        [recipient_raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    donor_plan = materialize_image_plan_batch(
        [donor_raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    recipient_plan_row = recipient_plan.rows[0]
    donor_plan_row = donor_plan.rows[0]
    if recipient_plan_row.expected_image_grid_thw != donor_plan_row.expected_image_grid_thw:
        raise SystemExit(
            "donor and recipient visual grids differ: "
            f"{recipient_plan_row.expected_image_grid_thw} != {donor_plan_row.expected_image_grid_thw}"
        )
    recipient_inputs = recipient_plan.model_inputs_by_row_id[recipient_raw.example_id]
    donor_inputs = donor_plan.model_inputs_by_row_id[donor_raw.example_id]
    recipient_features = capture_feature_bundle(qwen.model, recipient_inputs)
    donor_features = capture_feature_bundle(qwen.model, donor_inputs)
    merge_size = int(qwen.processor_identity.merge_size)
    layout = validate_feature_layout(
        recipient_features,
        donor_features,
        grid_thw=recipient_plan_row.expected_image_grid_thw,
        merge_size=merge_size,
    )

    base_prompt = build_prompt_record(
        recipient_raw,
        _template_config(resolved.config),
        processor=qwen.processor,
        row_index=0,
    )
    if list(base_prompt.prompt_token_ids) != recipient_prompt_ids:
        raise SystemExit("recipient bundle prompt does not match active prompt")
    recipient_prefix_ids = recipient_generated_ids[:recipient_count]
    composed_ids = [*recipient_prefix_ids, *[int(x) for x in composed_row["token_ids"]]]
    continuation_text = qwen.tokenizer.decode(
        composed_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prompt = build_prompt_record(
        recipient_raw,
        _template_config(resolved.config),
        processor=qwen.processor,
        row_index=0,
        assistant_continuation=AssistantContinuation(text=continuation_text),
    )
    resulting_prompt_ids = list(prompt.prompt_token_ids)
    expected_prompt_ids = [*recipient_prompt_ids, *composed_ids]
    if resulting_prompt_ids != expected_prompt_ids:
        raise SystemExit("active prompt does not equal exact recipient prefix plus coherent row")
    prompt_hash = _hash_ids(resulting_prompt_ids)
    row_hash = _hash_ids(composed_row["token_ids"])

    object_by_id = {str(obj.object_id): obj for obj in recipient_raw.objects}
    target_obj = object_by_id.get(str(args.target_object_id))
    if target_obj is None:
        raise SystemExit(f"target object id not found in recipient row: {args.target_object_id}")
    target_bbox = (
        tuple(float(x) for x in args.target_bbox)
        if getattr(args, "target_bbox", None) is not None
        else _source_bbox_to_pixels(target_obj.bbox, recipient_raw.image.width, recipient_raw.image.height)
    )
    target_mask = build_merged_support_mask(
        bbox_xyxy=target_bbox,
        image_width=recipient_raw.image.width,
        image_height=recipient_raw.image.height,
        layout=layout,
        halo=1,
    )
    forbidden_mask = torch.zeros_like(target_mask)
    for object_id in (
        args.target_object_id,
        args.right_cup_object_id,
        args.target_pizza_object_id,
    ):
        candidate = object_by_id.get(str(object_id))
        if candidate is None:
            continue
        candidate_bbox = _source_bbox_to_pixels(
            candidate.bbox,
            recipient_raw.image.width,
            recipient_raw.image.height,
        )
        candidate_mask = build_merged_support_mask(
            bbox_xyxy=candidate_bbox,
            image_width=recipient_raw.image.width,
            image_height=recipient_raw.image.height,
            layout=layout,
            halo=1,
        )
        forbidden_mask |= candidate_mask
    if args.control_bbox is not None:
        base_control_mask = build_merged_support_mask(
            bbox_xyxy=args.control_bbox,
            image_width=recipient_raw.image.width,
            image_height=recipient_raw.image.height,
            layout=layout,
            halo=1,
        )
        validate_equal_mask_shape_and_count(target_mask, base_control_mask)
    else:
        base_control_mask = choose_deterministic_control_mask(
            target_mask,
            temporal=layout.temporal,
            merged_height=layout.merged_height,
            merged_width=layout.merged_width,
            forbidden_mask=None,
        )
    target_mask, control_mask, control_pruning = prune_symmetric_translation_overlap(
        target_mask,
        base_control_mask,
        temporal=layout.temporal,
        merged_height=layout.merged_height,
        merged_width=layout.merged_width,
    )
    if bool((control_mask & forbidden_mask).any()):
        raise SystemExit("control support overlaps a frozen active candidate after symmetric pruning")
    validate_equal_mask_shape_and_count(target_mask, control_mask)
    frozen_pilot_geometry = _assert_frozen_pilot_geometry(
        layout=layout,
        recipient_features=recipient_features,
        donor_features=donor_features,
        target_mask=target_mask,
        control_mask=control_mask,
        control_pruning=control_pruning,
    )

    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    greedy_policy = DecodeGenerationPolicy.greedy(
        max_new_tokens=int(args.max_new_tokens), repetition_penalty=1.0
    )
    greedy_request = _decode_request(
        image_id=str(recipient_raw.metadata["source"]["image_id"]),
        prompt_ids=resulting_prompt_ids,
        model_inputs=recipient_inputs,
        policy=greedy_policy,
        condition="greedy",
        index=0,
    )
    standard_greedy = backend.generate_batch(
        [greedy_request],
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )[0]
    replay_greedy, noop_greedy_receipt = _run_replayed(
        backend=backend,
        model=qwen.model,
        recipient=recipient_features,
        donor=donor_features,
        grid_thw=recipient_plan_row.expected_image_grid_thw,
        merge_size=merge_size,
        mask=None,
        mode="clean",
        request=greedy_request,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_fingerprint=generation_fingerprint,
    )
    greedy_noop_parity = _require_noop_parity(
        standard_greedy,
        replay_greedy,
        scope="greedy",
    )

    sampled_requests: list[DecodeRequest] = []
    standard_sampled: list[Any] = []
    replay_sampled: list[Any] = []
    noop_sampled_receipt: dict[str, object] | None = None
    sampled_noop_parity: list[dict[str, object]] = []
    seeds = tuple(args.sampling_seed or range(8))
    if args.include_samples:
        if len(seeds) != 8:
            raise SystemExit("exactly eight sampling seeds are required")
        sampled_policy = DecodeGenerationPolicy.sampled(
            max_new_tokens=int(args.max_new_tokens),
            repetition_penalty=1.0,
            temperature=float(args.sampling_temperature),
            top_p=0.95,
        )
        sampled_requests = [
            _decode_request(
                image_id=str(recipient_raw.metadata["source"]["image_id"]),
                prompt_ids=resulting_prompt_ids,
                model_inputs=recipient_inputs,
                policy=sampled_policy,
                condition="sample",
                index=index,
                seed=int(seed),
            )
            for index, seed in enumerate(seeds)
        ]
        capability = _load_sampling_capability(
            args,
            backend=backend,
            sampled_policy=sampled_policy,
        )
        standard_sampled = backend.generate_batch_with_verified_runtime_attestation(
            sampled_requests,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            verified_runtime_attestation=capability,
        )
        replay_sampled, noop_sampled_receipt = _run_replayed_batch(
            backend=backend,
            model=qwen.model,
            recipient=recipient_features,
            donor=donor_features,
            grid_thw=recipient_plan_row.expected_image_grid_thw,
            merge_size=merge_size,
            mask=None,
            mode="clean",
            requests=sampled_requests,
            capability=capability,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_fingerprint=generation_fingerprint,
        )
        for index, (left, right) in enumerate(zip(standard_sampled, replay_sampled, strict=True)):
            sampled_noop_parity.append(
                _require_noop_parity(left, right, scope=f"sample-{index}")
            )

    conditions: list[dict[str, object]] = []
    for condition_name, mode, selected_mask in (
        ("clean_feature_replay", "clean", None),
        ("target_support_donor_substitution", "target", target_mask),
        ("equal_area_unrelated_support_donor_substitution", "control", control_mask),
    ):
        greedy, controller_receipt = _run_replayed(
            backend=backend,
            model=qwen.model,
            recipient=recipient_features,
            donor=donor_features,
            grid_thw=recipient_plan_row.expected_image_grid_thw,
            merge_size=merge_size,
            mask=selected_mask,
            mode=mode,
            request=_decode_request(
                image_id=str(recipient_raw.metadata["source"]["image_id"]),
                prompt_ids=resulting_prompt_ids,
                model_inputs=recipient_inputs,
                policy=greedy_policy,
                condition=condition_name,
                index=0,
            ),
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_fingerprint=generation_fingerprint,
        )
        condition_record: dict[str, object] = {
            "condition": condition_name,
            "mode": mode,
            "greedy": _result_record(
                greedy,
                qwen.tokenizer,
                recipient_raw,
                args.bundle_root,
                condition_name,
                right_cup_object_id=str(args.right_cup_object_id),
                target_object_id=str(args.target_object_id),
                target_pizza_object_id=str(args.target_pizza_object_id),
            ),
            "controller": controller_receipt,
        }
        if args.include_samples:
            sampled, sample_controller = _run_replayed_batch(
                backend=backend,
                model=qwen.model,
                recipient=recipient_features,
                donor=donor_features,
                grid_thw=recipient_plan_row.expected_image_grid_thw,
                merge_size=merge_size,
                mask=selected_mask,
                mode=mode,
                requests=sampled_requests,
                capability=capability,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_fingerprint=generation_fingerprint,
            )
            condition_record["samples"] = [
                _result_record(
                    result,
                    qwen.tokenizer,
                    recipient_raw,
                    args.bundle_root,
                    condition_name,
                    right_cup_object_id=str(args.right_cup_object_id),
                    target_object_id=str(args.target_object_id),
                    target_pizza_object_id=str(args.target_pizza_object_id),
                )
                for result in sampled
            ]
            condition_record["sample_controller"] = sample_controller
        conditions.append(condition_record)

    sampled_summary: dict[str, dict[str, object]] = {}
    greedy_summary: dict[str, str] = {}
    for condition in conditions:
        greedy = condition.get("greedy")
        if isinstance(greedy, Mapping):
            greedy_summary[str(condition["condition"])] = str(
                greedy.get("first_action_label", "unknown")
            )
        samples = condition.get("samples")
        if not isinstance(samples, list):
            continue
        right_cup_count = sum(
            1 for sample in samples
            if isinstance(sample, Mapping) and sample.get("first_action_label") == "right_cup"
        )
        sampled_summary[str(condition["condition"])] = {
            "right_cup_count": int(right_cup_count),
            "request_count": len(samples),
            "right_cup_fraction": float(right_cup_count / len(samples)) if samples else None,
        }
    target_fraction = sampled_summary.get("target_support_donor_substitution", {}).get(
        "right_cup_fraction"
    )
    control_fraction = sampled_summary.get(
        "equal_area_unrelated_support_donor_substitution", {}
    ).get("right_cup_fraction")

    return {
        "schema_version": "visual_support_counterfactual_commit.v1",
        "unit_id": "2026-07-15-visual-support-counterfactual-commit",
        "image_id": str(recipient_raw.metadata["source"]["image_id"]),
        "donor_image_id": str(donor_raw.metadata["source"]["image_id"]),
        "donor_review_note": str(args.donor_review_note),
        "config_path": str(config_path),
        "source_jsonl": str(source_path),
        "donor_source_jsonl": str(donor_source_path),
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "generation_config_fingerprint": generation_fingerprint,
        "prompt_token_hash": prompt_hash,
        "prompt_token_count": len(resulting_prompt_ids),
        "coherent_row_token_hash": row_hash,
        "coherent_row_token_ids": [int(x) for x in composed_row["token_ids"]],
        "recipient_prefix_token_count": recipient_count,
        "grid": {
            "grid_thw": list(layout.grid_thw),
            "merge_size": layout.merge_size,
            "temporal": layout.temporal,
            "merged_height": layout.merged_height,
            "merged_width": layout.merged_width,
            "primary_token_count": layout.primary_token_count,
            "deepstack_token_counts": list(layout.deepstack_token_counts),
        },
        "target_bbox_pixel_xyxy": list(target_bbox),
        "target_mask_indices": [int(x) for x in torch.where(target_mask)[0].tolist()],
        "control_mask_indices": [int(x) for x in torch.where(control_mask)[0].tolist()],
        "control_pruning": control_pruning,
        "frozen_pilot_geometry": frozen_pilot_geometry,
        "target_control_mask_count": int(target_mask.sum().item()),
        "mask_equal_shape_and_count": True,
        "recipient_features": feature_bundle_fingerprint(recipient_features),
        "donor_features": feature_bundle_fingerprint(donor_features),
        "noop_greedy": noop_greedy_receipt,
        "noop_sampled": noop_sampled_receipt,
        "noop_parity": {
            "greedy": greedy_noop_parity,
            "samples": sampled_noop_parity,
        },
        "conditions": conditions,
        "sampled_first_action_summary": sampled_summary,
        "greedy_first_action_labels": greedy_summary,
        "sampled_visual_support_commit_effect": (
            float(control_fraction) - float(target_fraction)
            if target_fraction is not None and control_fraction is not None
            else None
        ),
        "primary_estimand": {
            "definition": "fraction of requests whose first valid free row is the right cup",
            "target_control_difference": "R(equal_area_unrelated_support_donor_substitution) - R(target_support_donor_substitution)",
            "right_cup_object_id": str(args.right_cup_object_id),
        },
        "sampling_seeds": list(seeds) if args.include_samples else [],
        "limitations": [
            "A null target effect does not prove a purely textual transducer because visual features are globally contextualized.",
            "This one-image panel does not establish population-level coverage or detector performance.",
        ],
    }


def _run_replayed(*, backend: Any, model: Any, recipient: FeatureBundle, donor: FeatureBundle, grid_thw: Sequence[int], merge_size: int, mask: Any, mode: str, request: Any, model_identity: Mapping[str, Any], tokenizer_identity: Mapping[str, Any], generation_fingerprint: str):
    with FeatureReplayController(
        model=model,
        recipient=recipient,
        donor=donor,
        grid_thw=grid_thw,
        merge_size=merge_size,
        selected_mask=mask,
        mode=mode,
    ) as controller:
        result = backend.generate_batch(
            [request],
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
        )[0]
    return result, controller.validate_completed(expected_feature_calls=1)


def _run_replayed_batch(*, backend: Any, model: Any, recipient: FeatureBundle, donor: FeatureBundle, grid_thw: Sequence[int], merge_size: int, mask: Any, mode: str, requests: Sequence[Any], capability: Any, model_identity: Mapping[str, Any], tokenizer_identity: Mapping[str, Any], generation_fingerprint: str):
    with FeatureReplayController(
        model=model,
        recipient=recipient,
        donor=donor,
        grid_thw=grid_thw,
        merge_size=merge_size,
        selected_mask=mask,
        mode=mode,
    ) as controller:
        results = backend.generate_batch_with_verified_runtime_attestation(
            requests,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            verified_runtime_attestation=capability,
        )
    return results, controller.validate_completed(expected_feature_calls=1)


def _load_sampling_capability(args: argparse.Namespace, *, backend: Any, sampled_policy: Any) -> Any:
    from src.inference.backend import load_and_rebind_sampled_runtime_attestation_aggregate

    path = args.sampled_runtime_attestation
    if path is None:
        path = Path(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            "2026-07-13-spatial-scope-history-disentanglement/runtime-attestation-v2/"
            "request-scoped-sampling-three-policy-cuda.json"
        )
    return load_and_rebind_sampled_runtime_attestation_aggregate(
        path.expanduser().resolve(strict=True),
        decode_generation_policy_fingerprint=sampled_policy.fingerprint,
        backend=backend,
    )


def _decode_request(*, image_id: str, prompt_ids: Sequence[int], model_inputs: Mapping[str, Any], policy: Any, condition: str, index: int, seed: int | None = None) -> Any:
    suffix = f"{condition}-index-{index}"
    if seed is not None:
        suffix += f"-seed-{seed}"
    return __import__("src.inference.backend", fromlist=["DecodeRequest"]).DecodeRequest(
        request_id=f"visual-support:{image_id}:{suffix}",
        prompt_token_ids=[int(x) for x in prompt_ids],
        model_inputs=model_inputs,
        generation_policy=policy,
        sampling_seed=seed,
    )


def _result_record(result: Any, tokenizer: Any, raw: Any, bundle_root: Path | None, condition: str, *, right_cup_object_id: str, target_object_id: str, target_pizza_object_id: str) -> dict[str, object]:
    from src.inference.parsing import parse_compact_object_box_closed

    # DecodeResult owns the canonical artifact serializer.  In particular,
    # TokenTrace is a dataclass without its own serializer, so do not attempt
    # to serialize each trace here.
    canonical_result = result.to_artifact_dict()
    suffix = _int_list(result.generated_token_ids)
    text = tokenizer.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    parsed = parse_compact_object_box_closed(
        text,
        row_id=result.request_id,
        row_index=0,
        image_width=raw.image.width,
        image_height=raw.image.height,
    )
    first = __import__("scripts.research.run_sampled_rescue_transition", fromlist=["_first_action"])._first_action(
        parsed,
        text,
        generated_token_ids=suffix,
        stop_reason=result.stop_reason,
    )
    record = {
        "request_id": canonical_result["request_id"],
        "stop_reason": canonical_result["stop_reason"],
        "generated_token_ids": canonical_result["generated_token_ids"],
        "raw_generated_text": text,
        "first_action": first,
        "first_action_label": _first_action_label(
            first,
            raw,
            right_cup_object_id=right_cup_object_id,
            target_object_id=target_object_id,
            target_pizza_object_id=target_pizza_object_id,
        ),
        "execution_receipt": canonical_result["execution_receipt"],
        "token_trace": canonical_result["token_trace"],
    }
    if bundle_root is not None:
        path = bundle_root.expanduser().resolve() / condition / f"{_safe_name(result.request_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        record["artifact_path"] = str(path)
    return record


def _require_noop_parity(left: Any, right: Any, *, scope: str) -> dict[str, object]:
    left_ids = [int(x) for x in left.generated_token_ids]
    right_ids = [int(x) for x in right.generated_token_ids]
    if left_ids != right_ids:
        raise SystemExit(f"no-op parity failed for {scope}: generated token identifiers differ")
    left_receipt = left.execution_receipt
    right_receipt = right.execution_receipt
    if left_receipt is None or right_receipt is None:
        raise SystemExit(f"no-op parity failed for {scope}: execution receipt is missing")
    if left_receipt.canonical_float32_score_trace_hash != right_receipt.canonical_float32_score_trace_hash:
        raise SystemExit(f"no-op parity failed for {scope}: canonical float32 score traces differ")
    left_ids_hash = _hash_ids(left_ids)
    right_ids_hash = _hash_ids(right_ids)
    left_score_hash = str(left_receipt.canonical_float32_score_trace_hash)
    right_score_hash = str(right_receipt.canonical_float32_score_trace_hash)
    return {
        "verified": True,
        "scope": scope,
        "generated_token_count": len(left_ids),
        "standard_generated_token_ids_sha256": left_ids_hash,
        "replay_generated_token_ids_sha256": right_ids_hash,
        "generated_token_ids_equal": left_ids_hash == right_ids_hash,
        "standard_canonical_float32_score_trace_hash": left_score_hash,
        "replay_canonical_float32_score_trace_hash": right_score_hash,
        "canonical_float32_score_trace_hash_equal": left_score_hash == right_score_hash,
    }


def _first_action_label(first: Mapping[str, object], raw: Any, *, right_cup_object_id: str = DEFAULT_RIGHT_CUP_OBJECT_ID, target_object_id: str = DEFAULT_TARGET_OBJECT_ID, target_pizza_object_id: str = DEFAULT_TARGET_PIZZA_OBJECT_ID) -> str:
    if first.get("status") != "valid_row":
        return str(first.get("status", "unknown"))
    description = str(first.get("description") or "").strip().lower()
    bbox = first.get("bbox_xyxy")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        return "valid_row_without_bbox"
    predicted_box = tuple(float(x) for x in bbox)
    object_matches: list[tuple[float, Any]] = []
    for obj in raw.objects:
        object_box = _source_bbox_to_pixels(obj.bbox, raw.image.width, raw.image.height)
        object_matches.append((box_iou(predicted_box, object_box), obj))
    best_iou, best_object = max(object_matches, key=lambda item: item[0], default=(0.0, None))
    if best_object is None or best_iou < 0.5:
        return "valid_row_unmatched_geometry"
    owner_description = str(best_object.description).strip().lower()
    if description and owner_description != description:
        return "chimera_phrase_geometry"
    object_id = str(best_object.object_id)
    if object_id == str(right_cup_object_id):
        return "right_cup"
    if object_id == str(target_object_id):
        return "left_cup_revisit"
    if object_id == str(target_pizza_object_id):
        return "target_pizza_fallback"
    return "another_supported_object"


def _find_raw(rows: Sequence[Any], image_id: object) -> Any:
    requested = str(image_id)
    for row in rows:
        source = row.metadata.get("source", {})
        if str(source.get("image_id")) == requested or str(row.example_id) == requested:
            return row
    raise SystemExit(f"image id not found in source JSONL: {requested}")


def _evidence_image_id(bundle: Mapping[str, object]) -> str:
    evidence = bundle.get("execution_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("image_id") is None:
        raise SystemExit("recipient bundle lacks execution_evidence.image_id")
    return str(evidence["image_id"])


def _decode_dict(bundle: Mapping[str, object]) -> Mapping[str, object]:
    decode = bundle.get("decode_result", bundle)
    if not isinstance(decode, Mapping):
        raise SystemExit("recipient bundle decode_result must be an object")
    return decode


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        raise SystemExit("bundle token ids must be a list")
    return [int(x) for x in value]


def _hash_ids(value: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in value], separators=(",", ":")).encode()).hexdigest()


def _source_bbox_to_pixels(bbox: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise SystemExit("source bbox must contain four values")
    return (
        float(bbox[0]) * float(width) / 999.0,
        float(bbox[1]) * float(height) / 999.0,
        float(bbox[2]) * float(width) / 999.0,
        float(bbox[3]) * float(height) / 999.0,
    )


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be x1,y1,x2,y2") from exc
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise argparse.ArgumentTypeError("bbox must be positive x1,y1,x2,y2")
    return values  # type: ignore[return-value]


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
