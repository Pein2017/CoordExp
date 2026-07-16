#!/usr/bin/env python3
"""Score the image-7574 first-coordinate state across physical batch layouts.

The script is intentionally research-local.  It reuses the frozen coherent-row
factorial context, naturally generates the common five-token row prefix through
the ordinary generation cache, and captures the raw sixth-step logits.  A
full-prefix direct forward is recorded as a secondary comparison.  It does not
modify the shared inference backend or run a long rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_native_commit_redistribution import (  # noqa: E402
    CONDITION_NAMES,
    COORDINATE_TOKEN_END_EXCLUSIVE,
    COORDINATE_TOKEN_START,
    DEFAULT_CONFIG,
    DEFAULT_SOURCE_BUNDLE,
    DEFAULT_SOURCE_JSONL,
    IMAGE_ID,
    NO_APPENDED_ROW,
    OTHER_DESCRIPTION_WITH_TARGET_GEOMETRY,
    OTHER_ROW_WITH_OTHER_GEOMETRY,
    TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY,
    TARGET_ROW_WITH_TARGET_GEOMETRY,
    _condition_prompt,
    _json_hash,
    _read_json,
    _temporary_cwd,
    _validate_identity_continuity,
    build_condition_rows,
    extract_source_rows,
)
from scripts.research.run_sampled_rescue_transition import (  # noqa: E402
    _sha256_file,
    _write_bundle_once,
)


COMMON_COORDINATE_RECIPIENT_SUFFIX = (151646, 65, 9605, 151647, 151648)
CACHED_REPLAY_MAXIMUM_NEW_TOKENS = 16
FROZEN_TARGET_PROMPT_SHA256 = (
    "d24725c7eca4b74b39043d2e6f52501decef760a0f95c270907745c7d741b351"
)
WHITE_BOWL_WINDOW = (149, 213)
ORANGE_BOWL_WINDOW = (406, 470)
EXPECTED_SINGLE_COORDINATE_BIN = 181
EXPECTED_BATCH_FOUR_COORDINATE_BIN = 438
MODEL_DTYPE_BY_PUBLIC_NAME = {
    "bfloat16": "bf16",
    "float32": "fp32",
}
TORCH_DTYPE_BY_PUBLIC_NAME = {
    "bfloat16": "torch.bfloat16",
    "float32": "torch.float32",
}

SINGLE_TARGET_LAYOUT = "Single Target"
PREDECESSOR_MIXED_LENGTH_ROTATION_LAYOUT = "Predecessor Mixed-Length Batch Rotation"
HOMOGENEOUS_TARGET_COPIES_LAYOUT = "Homogeneous Target Copies"
EQUAL_LENGTH_MIXED_ROTATION_LAYOUT = "Equal-Length Mixed Rotation"

PREDECESSOR_BATCH_CONDITIONS = (
    NO_APPENDED_ROW,
    TARGET_ROW_WITH_TARGET_GEOMETRY,
    OTHER_ROW_WITH_OTHER_GEOMETRY,
    TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY,
)

MIXED_COMPANION_CONDITIONS = (
    OTHER_ROW_WITH_OTHER_GEOMETRY,
    TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY,
    OTHER_DESCRIPTION_WITH_TARGET_GEOMETRY,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score the exact image-7574 first-coordinate state under single, "
            "homogeneous, and equal-length mixed physical batches."
        )
    )
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument(
        "--model-dtype",
        choices=tuple(MODEL_DTYPE_BY_PUBLIC_NAME),
        default="bfloat16",
        help=(
            "Full-model execution dtype: Brain Floating Point 16-bit (bfloat16) "
            "for source lineage or Institute of Electrical and Electronics "
            "Engineers 754 32-bit floating point (float32) for the declared "
            "precision diagnostic."
        ),
    )
    return parser


def build_layouts() -> list[dict[str, Any]]:
    """Return the frozen condition order for every physical batch layout."""

    layouts: list[dict[str, Any]] = [
        {
            "layout_name": SINGLE_TARGET_LAYOUT,
            "layout_instance": "single-target",
            "condition_names": [TARGET_ROW_WITH_TARGET_GEOMETRY],
            "target_positions": [0],
        },
    ]
    for left_rotation in range(4):
        names = [
            *PREDECESSOR_BATCH_CONDITIONS[left_rotation:],
            *PREDECESSOR_BATCH_CONDITIONS[:left_rotation],
        ]
        target_position = names.index(TARGET_ROW_WITH_TARGET_GEOMETRY)
        layouts.append(
            {
                "layout_name": PREDECESSOR_MIXED_LENGTH_ROTATION_LAYOUT,
                "layout_instance": (
                    "predecessor-mixed-length-target-position-"
                    f"{target_position}"
                ),
                "condition_names": names,
                "target_positions": [target_position],
            }
        )
    layouts.extend(
        [
        {
            "layout_name": HOMOGENEOUS_TARGET_COPIES_LAYOUT,
            "layout_instance": "homogeneous-target-copies",
            "condition_names": [TARGET_ROW_WITH_TARGET_GEOMETRY] * 4,
            "target_positions": [0, 1, 2, 3],
        },
        ]
    )
    for target_position in range(4):
        names = list(MIXED_COMPANION_CONDITIONS)
        names.insert(target_position, TARGET_ROW_WITH_TARGET_GEOMETRY)
        layouts.append(
            {
                "layout_name": EQUAL_LENGTH_MIXED_ROTATION_LAYOUT,
                "layout_instance": f"equal-length-mixed-target-position-{target_position}",
                "condition_names": names,
                "target_positions": [target_position],
            }
        )
    return layouts


def coordinate_token_ids() -> list[int]:
    identifiers = list(range(COORDINATE_TOKEN_START, COORDINATE_TOKEN_END_EXCLUSIVE))
    if len(identifiers) != 1000:
        raise SystemExit("coordinate token range must contain exactly 1,000 tokens")
    return identifiers


def _float_list(tensor: torch.Tensor) -> list[float]:
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    if not bool(torch.isfinite(flat).all()):
        raise SystemExit("non-finite float32 logit artifact")
    return [float(value) for value in flat.tolist()]


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def summarize_logits(full_vocabulary_logits: torch.Tensor) -> dict[str, Any]:
    """Serialize the conclusion-owning coordinate distribution in float32."""

    full = full_vocabulary_logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if full.ndim != 1:
        raise ValueError("one row of full-vocabulary logits is required")
    if full.numel() < COORDINATE_TOKEN_END_EXCLUSIVE:
        raise ValueError("full-vocabulary logits do not cover the coordinate token range")
    if not bool(torch.isfinite(full).all()):
        raise ValueError("full-vocabulary logits contain non-finite values")

    coordinate_logits = full[
        COORDINATE_TOKEN_START:COORDINATE_TOKEN_END_EXCLUSIVE
    ].clone()
    coordinate_log_probabilities = torch.log_softmax(coordinate_logits, dim=0)
    full_log_probabilities = torch.log_softmax(full, dim=0)[
        COORDINATE_TOKEN_START:COORDINATE_TOKEN_END_EXCLUSIVE
    ]
    top_values, top_indices = torch.topk(coordinate_logits, k=20)

    white_start, white_end = WHITE_BOWL_WINDOW
    orange_start, orange_end = ORANGE_BOWL_WINDOW
    white_slice = slice(white_start, white_end + 1)
    orange_slice = slice(orange_start, orange_end + 1)
    white_log_mass = torch.logsumexp(coordinate_log_probabilities[white_slice], dim=0)
    orange_log_mass = torch.logsumexp(coordinate_log_probabilities[orange_slice], dim=0)
    full_white_log_mass = torch.logsumexp(full_log_probabilities[white_slice], dim=0)
    full_orange_log_mass = torch.logsumexp(full_log_probabilities[orange_slice], dim=0)

    return {
        "coordinate_token_start": COORDINATE_TOKEN_START,
        "coordinate_token_end_exclusive": COORDINATE_TOKEN_END_EXCLUSIVE,
        "coordinate_raw_logits_float32": _float_list(coordinate_logits),
        "coordinate_conditional_log_probabilities_float32": _float_list(
            coordinate_log_probabilities
        ),
        "coordinate_full_vocabulary_log_probabilities_float32": _float_list(
            full_log_probabilities
        ),
        "coordinate_raw_logits_float32_sha256": _tensor_sha256(coordinate_logits),
        "top_coordinates": [
            {
                "rank": rank,
                "coordinate_bin": int(index),
                "token_id": int(COORDINATE_TOKEN_START + index),
                "raw_logit_float32": float(value),
                "conditional_log_probability_float32": float(
                    coordinate_log_probabilities[int(index)]
                ),
                "full_vocabulary_log_probability_float32": float(
                    full_log_probabilities[int(index)]
                ),
            }
            for rank, (value, index) in enumerate(
                zip(top_values.tolist(), top_indices.tolist()), start=1
            )
        ],
        "top_one_coordinate_bin": int(top_indices[0]),
        "top_one_coordinate_token_id": int(
            COORDINATE_TOKEN_START + int(top_indices[0])
        ),
        "top_one_minus_top_two_raw_logit_margin_float32": float(
            top_values[0] - top_values[1]
        ),
        "white_bowl_window": [white_start, white_end],
        "orange_bowl_window": [orange_start, orange_end],
        "white_bowl_conditional_probability_mass_float32": float(
            torch.exp(white_log_mass)
        ),
        "orange_bowl_conditional_probability_mass_float32": float(
            torch.exp(orange_log_mass)
        ),
        "white_bowl_full_vocabulary_probability_mass_float32": float(
            torch.exp(full_white_log_mass)
        ),
        "orange_bowl_full_vocabulary_probability_mass_float32": float(
            torch.exp(full_orange_log_mass)
        ),
        "white_minus_orange_log_probability_mass_margin_float32": float(
            white_log_mass - orange_log_mass
        ),
        "full_vocabulary_top_token_id": int(torch.argmax(full)),
    }


def compare_coordinate_summaries(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    """Return shift-sensitive and shift-invariant coordinate comparisons."""

    reference_raw = torch.tensor(
        reference["coordinate_raw_logits_float32"], dtype=torch.float32
    )
    candidate_raw = torch.tensor(
        candidate["coordinate_raw_logits_float32"], dtype=torch.float32
    )
    if reference_raw.shape != (1000,) or candidate_raw.shape != (1000,):
        raise ValueError("coordinate summaries must each contain 1,000 logits")

    reference_centered = reference_raw - reference_raw.mean()
    candidate_centered = candidate_raw - candidate_raw.mean()
    difference = candidate_centered - reference_centered
    reference_probabilities = torch.softmax(reference_raw, dim=0)
    candidate_probabilities = torch.softmax(candidate_raw, dim=0)
    midpoint = 0.5 * (reference_probabilities + candidate_probabilities)
    jensen_shannon = 0.5 * torch.sum(
        reference_probabilities
        * (torch.log(reference_probabilities) - torch.log(midpoint))
    ) + 0.5 * torch.sum(
        candidate_probabilities
        * (torch.log(candidate_probabilities) - torch.log(midpoint))
    )

    outside_mask = torch.ones(1000, dtype=torch.bool)
    outside_mask[WHITE_BOWL_WINDOW[0] : WHITE_BOWL_WINDOW[1] + 1] = False
    outside_mask[ORANGE_BOWL_WINDOW[0] : ORANGE_BOWL_WINDOW[1] + 1] = False
    margin_shift = float(
        candidate["white_minus_orange_log_probability_mass_margin_float32"]
    ) - float(reference["white_minus_orange_log_probability_mass_margin_float32"])
    return {
        "centered_maximum_absolute_difference_float32": float(
            torch.max(torch.abs(difference))
        ),
        "centered_root_mean_square_difference_float32": float(
            torch.sqrt(torch.mean(difference.square()))
        ),
        "outside_windows_centered_root_mean_square_difference_float32": float(
            torch.sqrt(torch.mean(difference[outside_mask].square()))
        ),
        "jensen_shannon_divergence_float32": max(float(jensen_shannon), 0.0),
        "white_minus_orange_margin_shift_float32": margin_shift,
    }


def _batch_tensor_receipt(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "input_ids_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask_row_sums": [int(value) for value in attention_mask.sum(dim=1)],
        "model_input_shapes": {
            key: list(value.shape)
            for key, value in sorted(model_inputs.items())
            if torch.is_tensor(value)
        },
    }


def _cached_natural_suffix_logits(
    *,
    backend: Any,
    requests: Sequence[Any],
    policy: Any,
    target_positions: Sequence[int],
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    """Run natural cached generation and return raw sixth-step logits."""

    from src.inference.backend import (
        _generation_config_from_arguments,
        effective_generation_arguments,
    )

    prompt_width = max(len(request.prompt_token_ids) for request in requests)
    target_device = backend._target_device(requests)
    input_ids, attention_mask = backend._padded_prompt_tensors(
        requests, prompt_width, device=target_device
    )
    generate_inputs = backend._collate_generate_inputs(requests, device=target_device)
    generate_inputs["input_ids"] = input_ids
    generate_inputs["attention_mask"] = attention_mask
    arguments = effective_generation_arguments(
        policy,
        eos_token_id=backend._im_end_token_id(),
        pad_token_id=backend._pad_token_id(),
        bos_token_id=backend._bos_token_id(),
        decoder_start_token_id=backend._decoder_start_token_id(),
    )
    generation_config = _generation_config_from_arguments(arguments, policy=policy)
    generation_config.max_new_tokens = CACHED_REPLAY_MAXIMUM_NEW_TOKENS
    generation_config.max_length = prompt_width + CACHED_REPLAY_MAXIMUM_NEW_TOKENS
    generation_config.output_logits = True
    generation_config.output_scores = True
    generation_config.return_dict_in_generate = True
    with torch.inference_mode():
        outputs = backend.model.generate(
            **generate_inputs,
            generation_config=generation_config,
            use_model_defaults=False,
        )
    raw_logits = getattr(outputs, "logits", None)
    scores = getattr(outputs, "scores", None)
    sequences = getattr(outputs, "sequences", None)
    if raw_logits is None or len(raw_logits) < 6:
        raise SystemExit("cached generation did not expose six raw logit steps")
    if scores is None or len(scores) < 6:
        raise SystemExit("cached generation did not expose six processed score steps")
    if sequences is None or sequences.shape[0] != len(requests):
        raise SystemExit("cached generation returned an unexpected batch shape")
    generated_prefixes = sequences[:, prompt_width : prompt_width + 5].detach().cpu().tolist()
    for target_position in target_positions:
        if generated_prefixes[int(target_position)] != list(
            COMMON_COORDINATE_RECIPIENT_SUFFIX
        ):
            raise SystemExit(
                "target recipient did not naturally generate the frozen five-token suffix"
            )
    sixth_tokens = [int(value) for value in sequences[:, prompt_width + 5].detach().cpu()]
    raw_sixth = raw_logits[5].detach().to(device="cpu", dtype=torch.float32).contiguous()
    processed_sixth = scores[5].detach().to(device="cpu", dtype=torch.float32).contiguous()
    if raw_sixth.shape != processed_sixth.shape:
        raise SystemExit("raw and processed sixth-step score shapes differ")
    maximum_raw_processed_difference = float(
        torch.max(torch.abs(raw_sixth - processed_sixth))
    )
    if maximum_raw_processed_difference > 1e-6:
        raise SystemExit(
            "sixth-step raw and processed logits differ despite neutral greedy policy"
        )
    raw_argmax_tokens = [int(value) for value in torch.argmax(raw_sixth, dim=-1)]
    if raw_argmax_tokens != sixth_tokens:
        raise SystemExit("cached sixth token does not equal the raw-logit greedy maximum")
    return raw_sixth, sixth_tokens, {
        "prompt_width": prompt_width,
        "tensor_receipt": _batch_tensor_receipt(
            input_ids, attention_mask, generate_inputs
        ),
        "maximum_new_tokens": CACHED_REPLAY_MAXIMUM_NEW_TOKENS,
        "natural_generated_prefix_token_ids": generated_prefixes,
        "required_target_suffix_token_ids": list(COMMON_COORDINATE_RECIPIENT_SUFFIX),
        "maximum_raw_processed_logit_difference_float32": maximum_raw_processed_difference,
    }


def _direct_full_prefix_logits(
    *,
    backend: Any,
    requests: Sequence[Any],
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    """Score the same recipient by recomputing the full prefix in one forward."""

    prompt_width = max(len(request.prompt_token_ids) for request in requests)
    target_device = backend._target_device(requests)
    input_ids, attention_mask = backend._padded_prompt_tensors(
        requests, prompt_width, device=target_device
    )
    model_inputs = backend._collate_generate_inputs(requests, device=target_device)
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = attention_mask
    with torch.inference_mode():
        outputs = backend.model(
            **model_inputs,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    logits = outputs.logits[:, -1, :].detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    selected_tokens = [int(value) for value in torch.argmax(logits, dim=-1)]
    return logits, selected_tokens, {
        "prompt_width": prompt_width,
        "tensor_receipt": _batch_tensor_receipt(
            input_ids, attention_mask, model_inputs
        ),
    }


def _request_id(*, layout_instance: str, repeat_index: int, row_index: int, path: str) -> str:
    return (
        "batch-coordinate-logit-invariance:"
        f"{layout_instance}:repeat-{repeat_index}:row-{row_index}:{path}"
    )


def _requests(
    *,
    layout: Mapping[str, Any],
    repeat_index: int,
    prompt_token_ids_by_condition: Mapping[str, Sequence[int]],
    model_inputs: Mapping[str, Any],
    policy: Any,
    direct: bool,
) -> list[Any]:
    from src.inference.backend import DecodeRequest

    requests = []
    for row_index, condition_name in enumerate(layout["condition_names"]):
        prompt = list(prompt_token_ids_by_condition[condition_name])
        if direct:
            prompt.extend(COMMON_COORDINATE_RECIPIENT_SUFFIX)
        requests.append(
            DecodeRequest(
                request_id=_request_id(
                    layout_instance=str(layout["layout_instance"]),
                    repeat_index=repeat_index,
                    row_index=row_index,
                    path="direct" if direct else "cached",
                ),
                prompt_token_ids=prompt,
                model_inputs=model_inputs,
                generation_policy=policy,
            )
        )
    return requests


def _serialize_rows(
    *,
    layout: Mapping[str, Any],
    logits: torch.Tensor,
    selected_tokens: Sequence[int],
    prompt_token_ids_by_condition: Mapping[str, Sequence[int]],
    direct: bool,
) -> list[dict[str, Any]]:
    rows = []
    target_positions = set(int(value) for value in layout["target_positions"])
    for row_index, condition_name in enumerate(layout["condition_names"]):
        summary = summarize_logits(logits[row_index])
        selected_token = int(selected_tokens[row_index])
        selected_coordinate_bin = (
            selected_token - COORDINATE_TOKEN_START
            if COORDINATE_TOKEN_START <= selected_token < COORDINATE_TOKEN_END_EXCLUSIVE
            else None
        )
        rows.append(
            {
                "batch_position": row_index,
                "condition_name": condition_name,
                "is_target_recipient": row_index in target_positions,
                "recipient_path": "direct_full_prefix" if direct else "cached_natural_suffix",
                "base_prompt_token_ids_sha256": _json_hash(
                    list(prompt_token_ids_by_condition[condition_name])
                ),
                "recipient_prompt_token_ids_sha256": _json_hash(
                    [
                        *prompt_token_ids_by_condition[condition_name],
                        *COMMON_COORDINATE_RECIPIENT_SUFFIX,
                    ]
                ),
                "selected_next_token_id": selected_token,
                "selected_coordinate_bin": selected_coordinate_bin,
                "logit_summary": summary,
            }
        )
    return rows


def _target_records(executions: Sequence[Mapping[str, Any]], path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for execution in executions:
        for row in execution[path]["rows"]:
            if row["is_target_recipient"]:
                records.append(
                    {
                        "layout_name": execution["layout_name"],
                        "layout_instance": execution["layout_instance"],
                        "repeat_index": execution["repeat_index"],
                        "batch_position": row["batch_position"],
                        "row": row,
                    }
                )
    return records


def build_cross_execution_summary(executions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build comparisons that can be recomputed from the serialized vectors."""

    output: dict[str, Any] = {}
    for path in ("cached_path", "direct_path"):
        records = _target_records(executions, path)
        reference = next(
            record
            for record in records
            if record["layout_name"] == SINGLE_TARGET_LAYOUT
            and record["repeat_index"] == 0
        )
        comparisons = []
        repeat_noise_by_key: dict[tuple[str, int], float] = {}
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(
                (record["layout_instance"], int(record["batch_position"])), []
            ).append(record)
        for key, group in grouped.items():
            if len(group) >= 2:
                ordered = sorted(group, key=lambda value: int(value["repeat_index"]))
                repeat_comparison = compare_coordinate_summaries(
                    ordered[0]["row"]["logit_summary"],
                    ordered[1]["row"]["logit_summary"],
                )
                repeat_noise_by_key[key] = max(
                    abs(repeat_comparison["white_minus_orange_margin_shift_float32"]),
                    repeat_comparison[
                        "centered_root_mean_square_difference_float32"
                    ],
                )
        reference_key = (
            reference["layout_instance"],
            int(reference["batch_position"]),
        )
        for record in records:
            comparison = compare_coordinate_summaries(
                reference["row"]["logit_summary"],
                record["row"]["logit_summary"],
            )
            key = (record["layout_instance"], int(record["batch_position"]))
            repeat_noise = max(
                repeat_noise_by_key.get(reference_key, 0.0),
                repeat_noise_by_key.get(key, 0.0),
                1e-12,
            )
            margin_shift = abs(
                comparison["white_minus_orange_margin_shift_float32"]
            )
            comparisons.append(
                {
                    "layout_name": record["layout_name"],
                    "layout_instance": record["layout_instance"],
                    "repeat_index": record["repeat_index"],
                    "batch_position": record["batch_position"],
                    **comparison,
                    "same_layout_repeat_noise_floor_float32": repeat_noise,
                    "absolute_margin_shift_divided_by_repeat_noise": float(
                        margin_shift / repeat_noise
                    ),
                }
            )
        output[path] = {
            "reference": {
                "layout_instance": reference["layout_instance"],
                "repeat_index": reference["repeat_index"],
                "batch_position": reference["batch_position"],
            },
            "comparisons": comparisons,
        }
    return output


def _trust_gate(
    executions: Sequence[Mapping[str, Any]], *, require_source_coordinate_modes: bool = True
) -> dict[str, Any]:
    cached_records = _target_records(executions, "cached_path")
    single_bins = [
        record["row"]["selected_coordinate_bin"]
        for record in cached_records
        if record["layout_name"] == SINGLE_TARGET_LAYOUT
    ]
    predecessor_bins = [
        record["row"]["selected_coordinate_bin"]
        for record in cached_records
        if record["layout_name"] == PREDECESSOR_MIXED_LENGTH_ROTATION_LAYOUT
        and record["layout_instance"]
        == "predecessor-mixed-length-target-position-1"
    ]
    source_phenotype_reproduced = bool(single_bins and predecessor_bins) and set(single_bins) == {
        EXPECTED_SINGLE_COORDINATE_BIN
    } and set(predecessor_bins) == {EXPECTED_BATCH_FOUR_COORDINATE_BIN}
    passed = source_phenotype_reproduced if require_source_coordinate_modes else True
    return {
        "passed": passed,
        "source_phenotype_required_for_interpretation": require_source_coordinate_modes,
        "source_phenotype_reproduced": source_phenotype_reproduced,
        "single_target_cached_coordinate_bins": single_bins,
        "expected_single_target_coordinate_bin": EXPECTED_SINGLE_COORDINATE_BIN,
        "predecessor_batch_target_cached_coordinate_bins": predecessor_bins,
        "expected_predecessor_batch_target_coordinate_bin": (
            EXPECTED_BATCH_FOUR_COORDINATE_BIN
        ),
        "failure_disposition": None if passed else "execution_state_mismatch_stop_interpretation",
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import (
        DecodeGenerationPolicy,
        HFGenerateBackend,
        _attention_implementation,
        _execution_device_identity,
        _runtime_identity,
    )
    from src.inference.image_plan import (
        materialize_image_plan_batch,
        verify_processor_model_vision_parity,
    )
    from src.inference.pipeline import (
        _processor_config,
        _template_config,
        _tokenizer_identity,
    )
    from src.inference.runtime import assemble_runtime

    if args.repeat_count != 2:
        raise SystemExit("this research unit requires exactly two same-process repeats")
    source_bundle_path = args.source_bundle.expanduser().resolve(strict=True)
    source_bundle = _read_json(source_bundle_path)
    source_rows = extract_source_rows(source_bundle)
    conditions = build_condition_rows(source_rows)
    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    raw_rows = load_raw_examples(source_path)
    raw = next(
        (
            row
            for row in raw_rows
            if str(row.metadata.get("source", {}).get("image_id")) == IMAGE_ID
        ),
        None,
    )
    if raw is None:
        raise SystemExit("image-7574 source row was not found")
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
    runtime = assemble_runtime(execution_config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    prompt_records = {
        name: _condition_prompt(
            raw,
            _template_config(resolved.config),
            qwen.processor,
            condition=condition,
            tokenizer=qwen.tokenizer,
        )
        for name, condition in conditions.items()
    }
    base_prompt_ids = source_rows["prompt_token_ids"]
    for name, prompt in prompt_records.items():
        row_tokens = conditions[name]["appended_row_token_ids"]
        expected = base_prompt_ids if row_tokens is None else [*base_prompt_ids, *row_tokens]
        if list(prompt.prompt_token_ids) != expected:
            raise SystemExit(f"{name} prompt does not equal frozen prompt plus exact row")
    target_prompt_hash = _json_hash(
        prompt_records[TARGET_ROW_WITH_TARGET_GEOMETRY].prompt_token_ids
    )
    if target_prompt_hash != FROZEN_TARGET_PROMPT_SHA256:
        raise SystemExit("coherent target prompt hash differs from the frozen recipient")
    equal_length_names = (TARGET_ROW_WITH_TARGET_GEOMETRY, *MIXED_COMPANION_CONDITIONS)
    equal_lengths = {
        len(prompt_records[name].prompt_token_ids) for name in equal_length_names
    }
    if len(equal_lengths) != 1:
        raise SystemExit("mixed factorial prompts must have identical token lengths")

    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_inputs = image_plan.model_inputs_by_row_id[
        prompt_records[NO_APPENDED_ROW].row_id
    ]
    generation_fingerprint = sha256_json(
        resolved.config.generation.model_dump(mode="json")
    )
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    identity_continuity = _validate_identity_continuity(
        source_bundle=source_bundle,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
        source_image_sha256=_sha256_file(raw.image.path),
    )
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    model_dtype = str(next(iter(qwen.model.parameters())).dtype)
    expected_model_dtype = TORCH_DTYPE_BY_PUBLIC_NAME[str(args.model_dtype)]
    if model_dtype != expected_model_dtype:
        raise SystemExit(
            f"requested {expected_model_dtype}, received {model_dtype}"
        )
    cached_policy = DecodeGenerationPolicy.greedy(
        max_new_tokens=CACHED_REPLAY_MAXIMUM_NEW_TOKENS, repetition_penalty=1.0
    )
    prompt_token_ids_by_condition = {
        name: list(prompt.prompt_token_ids) for name, prompt in prompt_records.items()
    }
    executions: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat_count):
        for layout in build_layouts():
            cached_requests = _requests(
                layout=layout,
                repeat_index=repeat_index,
                prompt_token_ids_by_condition=prompt_token_ids_by_condition,
                model_inputs=model_inputs,
                policy=cached_policy,
                direct=False,
            )
            cached_logits, cached_tokens, cached_receipt = _cached_natural_suffix_logits(
                backend=backend,
                requests=cached_requests,
                policy=cached_policy,
                target_positions=layout["target_positions"],
            )
            direct_requests = _requests(
                layout=layout,
                repeat_index=repeat_index,
                prompt_token_ids_by_condition=prompt_token_ids_by_condition,
                model_inputs=model_inputs,
                policy=cached_policy,
                direct=True,
            )
            direct_logits, direct_tokens, direct_receipt = _direct_full_prefix_logits(
                backend=backend,
                requests=direct_requests,
            )
            executions.append(
                {
                    **layout,
                    "repeat_index": repeat_index,
                    "cached_path": {
                        **cached_receipt,
                        "rows": _serialize_rows(
                            layout=layout,
                            logits=cached_logits,
                            selected_tokens=cached_tokens,
                            prompt_token_ids_by_condition=prompt_token_ids_by_condition,
                            direct=False,
                        ),
                    },
                    "direct_path": {
                        **direct_receipt,
                        "rows": _serialize_rows(
                            layout=layout,
                            logits=direct_logits,
                            selected_tokens=direct_tokens,
                            prompt_token_ids_by_condition=prompt_token_ids_by_condition,
                            direct=True,
                        ),
                    },
                }
            )

    trust_gate = _trust_gate(
        executions,
        require_source_coordinate_modes=(args.model_dtype == "bfloat16"),
    )
    result = {
        "schema_version": "batch_coordinate_logit_invariance.receipt.v1",
        "experiment_name": "Homogeneous and Mixed Batch Coordinate-Logit Invariance Probe",
        "image_id": IMAGE_ID,
        "source_bundle": {
            "path": str(source_bundle_path),
            "sha256": _sha256_file(source_bundle_path),
        },
        "source_image": {
            "path": str(raw.image.path),
            "sha256": _sha256_file(raw.image.path),
        },
        "infer_config_path": str(config_path),
        "requested_model_execution_dtype": str(args.model_dtype),
        "target_prompt_token_ids_sha256": target_prompt_hash,
        "common_coordinate_recipient_suffix_token_ids": list(
            COMMON_COORDINATE_RECIPIENT_SUFFIX
        ),
        "coordinate_token_ids": coordinate_token_ids(),
        "windows": {
            "white_bowl_coordinate_bins_inclusive": list(WHITE_BOWL_WINDOW),
            "orange_bowl_coordinate_bins_inclusive": list(ORANGE_BOWL_WINDOW),
        },
        "execution_identity": {
            "model_dtype": model_dtype,
            "model_evaluation_mode": not bool(qwen.model.training),
            "attention_implementation": _attention_implementation(qwen.model),
            "runtime_identity": _runtime_identity(),
            "device_identity": _execution_device_identity(backend._model_device()),
            "generation_config_fingerprint": generation_fingerprint,
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "identity_continuity": identity_continuity,
        },
        "layout_contract": build_layouts(),
        "executions": executions,
        "cross_execution_summary": build_cross_execution_summary(executions),
        "trust_gate": trust_gate,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_probe(args)
    output_root = args.output_root.expanduser().resolve()
    receipt_path = output_root / "receipt.json"
    _write_bundle_once(receipt_path, result)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "trust_gate": result["trust_gate"],
                "execution_count": len(result["executions"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["trust_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
