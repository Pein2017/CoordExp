#!/usr/bin/env python3
"""Score exact first-differing coordinate recipients with full logit evidence.

This experiment-local runner reconstructs three recipients from a frozen
selected-transition receipt.  It captures the natural cached logits at the
original first differing generated slot and directly scores the exact common
prefix.  It changes no shared inference or model-forward implementation.
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

from scripts.research.run_batch_coordinate_logit_invariance import (  # noqa: E402
    _batch_tensor_receipt,
    _direct_full_prefix_logits,
    _float_list,
    _tensor_sha256,
    coordinate_token_ids,
)
from scripts.research.run_batch_precision_selected_transition_prevalence import (  # noqa: E402
    COORDINATE_TOKEN_END_EXCLUSIVE,
    COORDINATE_TOKEN_START,
    DEFAULT_CONFIG,
    DEFAULT_SOURCE_JSONL,
    MODEL_DTYPE_BY_PUBLIC_NAME,
    TORCH_DTYPE_BY_PUBLIC_NAME,
    extract_first_complete_row,
    validate_source_bundle,
)
from scripts.research.run_native_commit_redistribution import (  # noqa: E402
    _json_hash,
    _read_json,
    _temporary_cwd,
    _validate_identity_continuity,
)
from scripts.research.run_sampled_rescue_transition import (  # noqa: E402
    _sha256_file,
    _write_bundle_once,
)


DEFAULT_PREVALENCE_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-selected-transition-batch-precision-prevalence-screen/"
    "source-bfloat16-selected-six-20260715a/receipt.json"
)
DEFAULT_IMAGE_IDS = ("8629", "13659", "17714")
SINGLE_RECIPIENT_LAYOUT = "Single Recipient"
HOMOGENEOUS_FOUR_COPY_LAYOUT = "Homogeneous Four-Copy Recipient"
LOCAL_WINDOW_RADIUS = 16
REPEAT_COUNT = 2

PREDECESSOR_BROAD_SHIFT_ANCHOR = {
    "centered_root_mean_square_difference_float32": 1.166623,
    "outside_window_centered_root_mean_square_difference_float32": 1.139275,
    "jensen_shannon_divergence_float32": 0.117722,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture all coordinate logits at three frozen first-differing "
            "generated slots under physical batch one and four."
        )
    )
    parser.add_argument(
        "--prevalence-receipt", type=Path, default=DEFAULT_PREVALENCE_RECEIPT
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
    )
    parser.add_argument("--repeat-count", type=int, default=REPEAT_COUNT)
    parser.add_argument("--local-window-radius", type=int, default=LOCAL_WINDOW_RADIUS)
    return parser


def first_differing_index(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    raise ValueError("the source single and homogeneous generations do not differ")


def derive_case_contracts(
    receipt: Mapping[str, Any], image_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    if receipt.get("requested_model_execution_dtype") != "bfloat16":
        raise ValueError("the source prevalence receipt must use bfloat16 execution")
    selected = list(DEFAULT_IMAGE_IDS if not image_ids else map(str, image_ids))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected image identifiers must be non-empty and unique")
    cases = {
        str(case.get("image_id")): case
        for case in receipt.get("cases", [])
        if isinstance(case, Mapping)
    }
    unknown = [image_id for image_id in selected if image_id not in cases]
    if unknown:
        raise ValueError(f"selected cases are absent from the receipt: {unknown}")

    contracts: list[dict[str, Any]] = []
    for image_id in selected:
        case = cases[image_id]
        if not bool(case.get("comparison", {}).get("primary_first_action_divergence")):
            raise ValueError(f"image {image_id} is not a source first-action divergence")
        single_tokens = [
            int(value) for value in case["single_recipient"]["generated_token_ids"]
        ]
        homogeneous_records = case.get("homogeneous_four_copy_recipients", [])
        if len(homogeneous_records) != 4:
            raise ValueError(f"image {image_id} lacks four homogeneous source records")
        homogeneous_tokens = [
            int(value) for value in homogeneous_records[0]["generated_token_ids"]
        ]
        if any(
            [int(value) for value in record["generated_token_ids"]]
            != homogeneous_tokens
            for record in homogeneous_records[1:]
        ):
            raise ValueError(f"image {image_id} source homogeneous generations disagree")
        difference_index = first_differing_index(single_tokens, homogeneous_tokens)
        if difference_index >= min(len(single_tokens), len(homogeneous_tokens)):
            raise ValueError(f"image {image_id} differs only by generation length")
        single_token = single_tokens[difference_index]
        homogeneous_token = homogeneous_tokens[difference_index]
        if not (
            COORDINATE_TOKEN_START
            <= single_token
            < COORDINATE_TOKEN_END_EXCLUSIVE
            and COORDINATE_TOKEN_START
            <= homogeneous_token
            < COORDINATE_TOKEN_END_EXCLUSIVE
        ):
            raise ValueError(f"image {image_id} first difference is not coordinate-valued")
        center = single_token - COORDINATE_TOKEN_START
        contracts.append(
            {
                "image_id": image_id,
                "source_bundle_path": str(case["source_bundle_path"]),
                "source_recipient_prompt_token_ids_sha256": case[
                    "recipient_prompt_token_ids_sha256"
                ],
                "source_first_row": case["source_first_row"],
                "first_differing_generated_token_index": difference_index,
                "common_generated_prefix_token_ids": single_tokens[:difference_index],
                "common_generated_prefix_token_ids_sha256": _json_hash(
                    single_tokens[:difference_index]
                ),
                "source_single_selected_token_id": single_token,
                "source_homogeneous_selected_token_id": homogeneous_token,
                "source_single_selected_coordinate_bin": center,
                "source_homogeneous_selected_coordinate_bin": (
                    homogeneous_token - COORDINATE_TOKEN_START
                ),
            }
        )
    return contracts


def local_window(center: int, radius: int) -> tuple[int, int]:
    if not (0 <= center < 1000):
        raise ValueError("local-window center must be a coordinate bin")
    if radius != LOCAL_WINDOW_RADIUS:
        raise ValueError(f"this unit freezes local-window radius {LOCAL_WINDOW_RADIUS}")
    return max(0, center - radius), min(999, center + radius)


def summarize_coordinate_logits(
    full_vocabulary_logits: torch.Tensor, *, window: tuple[int, int]
) -> dict[str, Any]:
    full = full_vocabulary_logits.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    if full.ndim != 1 or full.numel() < COORDINATE_TOKEN_END_EXCLUSIVE:
        raise ValueError("one complete full-vocabulary logit row is required")
    if not bool(torch.isfinite(full).all()):
        raise ValueError("full-vocabulary logits contain non-finite values")
    coordinate_logits = full[
        COORDINATE_TOKEN_START:COORDINATE_TOKEN_END_EXCLUSIVE
    ].clone()
    coordinate_log_probabilities = torch.log_softmax(coordinate_logits, dim=0)
    top_values, top_indices = torch.topk(coordinate_logits, k=20)
    start, end = window
    window_log_mass = torch.logsumexp(
        coordinate_log_probabilities[start : end + 1], dim=0
    )
    full_top_token = int(torch.argmax(full))
    selected_coordinate_bin = (
        full_top_token - COORDINATE_TOKEN_START
        if COORDINATE_TOKEN_START
        <= full_top_token
        < COORDINATE_TOKEN_END_EXCLUSIVE
        else None
    )
    return {
        "coordinate_token_start": COORDINATE_TOKEN_START,
        "coordinate_token_end_exclusive": COORDINATE_TOKEN_END_EXCLUSIVE,
        "coordinate_raw_logits_float32": _float_list(coordinate_logits),
        "coordinate_conditional_log_probabilities_float32": _float_list(
            coordinate_log_probabilities
        ),
        "coordinate_raw_logits_float32_sha256": _tensor_sha256(coordinate_logits),
        "full_vocabulary_top_token_id": full_top_token,
        "selected_coordinate_bin": selected_coordinate_bin,
        "top_one_minus_top_two_raw_logit_margin_float32": float(
            top_values[0] - top_values[1]
        ),
        "local_coordinate_window_inclusive": [start, end],
        "local_window_conditional_probability_mass_float32": float(
            torch.exp(window_log_mass)
        ),
        "local_window_conditional_log_probability_mass_float32": float(
            window_log_mass
        ),
        "top_coordinates": [
            {
                "rank": rank,
                "coordinate_bin": int(index),
                "token_id": COORDINATE_TOKEN_START + int(index),
                "raw_logit_float32": float(value),
                "conditional_log_probability_float32": float(
                    coordinate_log_probabilities[int(index)]
                ),
            }
            for rank, (value, index) in enumerate(
                zip(top_values.tolist(), top_indices.tolist()), start=1
            )
        ],
    }


def compare_coordinate_summaries(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reference_raw = torch.tensor(
        reference["coordinate_raw_logits_float32"], dtype=torch.float32
    )
    candidate_raw = torch.tensor(
        candidate["coordinate_raw_logits_float32"], dtype=torch.float32
    )
    if reference_raw.shape != (1000,) or candidate_raw.shape != (1000,):
        raise ValueError("coordinate summaries must each preserve 1,000 logits")
    if reference["local_coordinate_window_inclusive"] != candidate[
        "local_coordinate_window_inclusive"
    ]:
        raise ValueError("coordinate summaries use different local windows")
    start, end = map(int, reference["local_coordinate_window_inclusive"])
    reference_centered = reference_raw - reference_raw.mean()
    candidate_centered = candidate_raw - candidate_raw.mean()
    difference = candidate_centered - reference_centered
    outside = torch.ones(1000, dtype=torch.bool)
    outside[start : end + 1] = False
    reference_probabilities = torch.softmax(reference_raw, dim=0)
    candidate_probabilities = torch.softmax(candidate_raw, dim=0)
    midpoint = 0.5 * (reference_probabilities + candidate_probabilities)
    tiny = torch.finfo(torch.float32).tiny
    jensen_shannon = 0.5 * torch.sum(
        reference_probabilities
        * (
            torch.log(torch.clamp(reference_probabilities, min=tiny))
            - torch.log(torch.clamp(midpoint, min=tiny))
        )
    ) + 0.5 * torch.sum(
        candidate_probabilities
        * (
            torch.log(torch.clamp(candidate_probabilities, min=tiny))
            - torch.log(torch.clamp(midpoint, min=tiny))
        )
    )
    reference_bin = reference["selected_coordinate_bin"]
    candidate_bin = candidate["selected_coordinate_bin"]
    return {
        "centered_maximum_absolute_difference_float32": float(
            torch.max(torch.abs(difference))
        ),
        "centered_root_mean_square_difference_float32": float(
            torch.sqrt(torch.mean(difference.square()))
        ),
        "outside_window_centered_root_mean_square_difference_float32": float(
            torch.sqrt(torch.mean(difference[outside].square()))
        ),
        "jensen_shannon_divergence_float32": max(float(jensen_shannon), 0.0),
        "local_window_probability_mass_shift_float32": float(
            candidate["local_window_conditional_probability_mass_float32"]
        )
        - float(reference["local_window_conditional_probability_mass_float32"]),
        "local_window_log_probability_mass_shift_float32": float(
            candidate["local_window_conditional_log_probability_mass_float32"]
        )
        - float(reference["local_window_conditional_log_probability_mass_float32"]),
        "reference_selected_coordinate_bin": reference_bin,
        "candidate_selected_coordinate_bin": candidate_bin,
        "both_selected_bins_inside_local_window": bool(
            reference_bin is not None
            and candidate_bin is not None
            and start <= int(reference_bin) <= end
            and start <= int(candidate_bin) <= end
        ),
    }


def _natural_cached_logits(
    *, backend: Any, requests: Sequence[Any], policy: Any, common_prefix: Sequence[int]
) -> tuple[torch.Tensor | None, list[int] | None, dict[str, Any]]:
    from src.inference.backend import (
        _generation_config_from_arguments,
        effective_generation_arguments,
    )

    step_index = len(common_prefix)
    maximum_new_tokens = step_index + 1
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
    generation_config.max_new_tokens = maximum_new_tokens
    generation_config.max_length = prompt_width + maximum_new_tokens
    generation_config.output_logits = True
    generation_config.output_scores = True
    generation_config.return_dict_in_generate = True
    with torch.inference_mode():
        outputs = backend.model.generate(
            **generate_inputs,
            generation_config=generation_config,
            use_model_defaults=False,
        )
    sequences = outputs.sequences
    observed_prefixes = sequences[:, prompt_width : prompt_width + step_index]
    observed_prefix_lists = observed_prefixes.detach().cpu().tolist()
    recipient_reached = all(
        [int(value) for value in row] == [int(value) for value in common_prefix]
        for row in observed_prefix_lists
    )
    receipt = {
        "recipient_reached_by_every_request": recipient_reached,
        "prompt_width": prompt_width,
        "maximum_new_tokens": maximum_new_tokens,
        "common_prefix_token_ids": [int(value) for value in common_prefix],
        "observed_prefix_token_ids": observed_prefix_lists,
        "tensor_receipt": _batch_tensor_receipt(
            input_ids, attention_mask, generate_inputs
        ),
    }
    if not recipient_reached:
        return None, None, receipt
    raw_logits = outputs.logits[step_index].detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    processed_scores = outputs.scores[step_index].detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    maximum_difference = float(torch.max(torch.abs(raw_logits - processed_scores)))
    if maximum_difference > 1e-6:
        raise ValueError("raw and processed cached logits differ under neutral greedy decode")
    selected_tokens = [int(value) for value in torch.argmax(raw_logits, dim=-1)]
    generated_tokens = [
        int(value) for value in sequences[:, prompt_width + step_index].detach().cpu()
    ]
    if selected_tokens != generated_tokens:
        raise ValueError("cached generated token is not the raw-logit maximum")
    receipt["maximum_raw_processed_logit_difference_float32"] = maximum_difference
    return raw_logits, selected_tokens, receipt


def _request_id(
    *, image_id: str, layout: str, repeat_index: int, row_index: int, path: str
) -> str:
    normalized_layout = layout.lower().replace(" ", "-")
    return (
        "first-differing-slot-coordinate-logits:"
        f"{image_id}:{normalized_layout}:repeat-{repeat_index}:row-{row_index}:{path}"
    )


def _build_requests(
    *,
    image_id: str,
    layout: str,
    repeat_index: int,
    recipient_prompt_token_ids: Sequence[int],
    common_prefix: Sequence[int],
    model_inputs: Mapping[str, Any],
    policy: Any,
    direct: bool,
) -> list[Any]:
    from src.inference.backend import DecodeRequest

    count = 1 if layout == SINGLE_RECIPIENT_LAYOUT else 4
    prompt = [int(value) for value in recipient_prompt_token_ids]
    if direct:
        prompt.extend(int(value) for value in common_prefix)
    return [
        DecodeRequest(
            request_id=_request_id(
                image_id=image_id,
                layout=layout,
                repeat_index=repeat_index,
                row_index=row_index,
                path="direct" if direct else "cached",
            ),
            prompt_token_ids=list(prompt),
            model_inputs=model_inputs,
            generation_policy=policy,
        )
        for row_index in range(count)
    ]


def _serialize_rows(
    logits: torch.Tensor, selected_tokens: Sequence[int], *, window: tuple[int, int]
) -> list[dict[str, Any]]:
    rows = []
    for position, selected_token in enumerate(selected_tokens):
        summary = summarize_coordinate_logits(logits[position], window=window)
        if int(summary["full_vocabulary_top_token_id"]) != int(selected_token):
            raise ValueError("serialized top token differs from executed selected token")
        rows.append(
            {
                "batch_position": position,
                "selected_next_token_id": int(selected_token),
                "selected_coordinate_bin": summary["selected_coordinate_bin"],
                "logit_summary": summary,
            }
        )
    if len(rows) == 4:
        hashes = {
            row["logit_summary"]["coordinate_raw_logits_float32_sha256"]
            for row in rows
        }
        if len(hashes) != 1:
            raise ValueError("homogeneous copies have different coordinate-logit vectors")
    return rows


def build_case_comparisons(
    executions: Sequence[Mapping[str, Any]], *, path_name: str
) -> dict[str, Any]:
    by_layout_repeat: dict[tuple[str, int], Mapping[str, Any]] = {}
    for execution in executions:
        path = execution[path_name]
        if not path.get("recipient_reached_by_every_request", True):
            continue
        rows = path.get("rows")
        if not rows:
            continue
        by_layout_repeat[(execution["layout_name"], execution["repeat_index"])] = rows[0][
            "logit_summary"
        ]
    required = [
        (SINGLE_RECIPIENT_LAYOUT, repeat_index)
        for repeat_index in range(REPEAT_COUNT)
    ] + [
        (HOMOGENEOUS_FOUR_COPY_LAYOUT, repeat_index)
        for repeat_index in range(REPEAT_COUNT)
    ]
    if any(key not in by_layout_repeat for key in required):
        return {
            "eligible": False,
            "missing_layout_repeats": [list(key) for key in required if key not in by_layout_repeat],
        }

    repeat_comparisons = {
        layout: compare_coordinate_summaries(
            by_layout_repeat[(layout, 0)], by_layout_repeat[(layout, 1)]
        )
        for layout in (SINGLE_RECIPIENT_LAYOUT, HOMOGENEOUS_FOUR_COPY_LAYOUT)
    }
    batch_comparisons = [
        compare_coordinate_summaries(
            by_layout_repeat[(SINGLE_RECIPIENT_LAYOUT, repeat_index)],
            by_layout_repeat[(HOMOGENEOUS_FOUR_COPY_LAYOUT, repeat_index)],
        )
        for repeat_index in range(REPEAT_COUNT)
    ]
    metric_names = tuple(PREDECESSOR_BROAD_SHIFT_ANCHOR)
    maximum_repeat_noise = {
        metric: max(
            float(repeat_comparisons[layout][metric])
            for layout in repeat_comparisons
        )
        for metric in metric_names
    }
    enriched = []
    for repeat_index, comparison in enumerate(batch_comparisons):
        ratios = {
            metric: float(comparison[metric]) / float(anchor)
            for metric, anchor in PREDECESSOR_BROAD_SHIFT_ANCHOR.items()
        }
        repeat_stability_ratios = {
            metric: float(comparison[metric])
            / max(float(maximum_repeat_noise[metric]), 1e-12)
            for metric in metric_names
        }
        enriched.append(
            {
                "repeat_index": repeat_index,
                **comparison,
                "fraction_of_predecessor_broad_shift_anchor": ratios,
                "between_layout_divided_by_same_layout_repeat_noise": (
                    repeat_stability_ratios
                ),
            }
        )
    return {
        "eligible": True,
        "same_layout_repeat_comparisons": repeat_comparisons,
        "maximum_same_layout_repeat_noise": maximum_repeat_noise,
        "single_vs_homogeneous_comparisons": enriched,
    }


def classify_bfloat_case(
    comparison: Mapping[str, Any], *, source_split_reproduced: bool
) -> str:
    if not comparison.get("eligible") or not source_split_reproduced:
        return "inconclusive_recipient_or_source_split_failure"
    rows = comparison["single_vs_homogeneous_comparisons"]
    metric_names = tuple(PREDECESSOR_BROAD_SHIFT_ANCHOR)
    maximum_repeat_noise = comparison["maximum_same_layout_repeat_noise"]
    broad = []
    close = []
    for row in rows:
        fractions = row["fraction_of_predecessor_broad_shift_anchor"]
        repeat_ratios = row["between_layout_divided_by_same_layout_repeat_noise"]
        broad_metric_count = sum(float(fractions[name]) >= 0.25 for name in metric_names)
        broad.append(
            broad_metric_count >= 2
            and all(float(repeat_ratios[name]) >= 10.0 for name in metric_names)
            and (
                not bool(row["both_selected_bins_inside_local_window"])
                or abs(float(row["local_window_probability_mass_shift_float32"]))
                >= 0.05
            )
        )
        close.append(
            all(float(fractions[name]) <= 0.10 for name in metric_names)
            and bool(row["both_selected_bins_inside_local_window"])
            and abs(float(row["local_window_probability_mass_shift_float32"]))
            <= 0.02
            and all(
                float(maximum_repeat_noise[name])
                / float(PREDECESSOR_BROAD_SHIFT_ANCHOR[name])
                <= float(fractions[name]) + 1e-12
                for name in metric_names
            )
        )
    if all(broad):
        return "broad_repeat_stable_shift_candidate"
    if all(close):
        return "local_same_basin_numeric_jitter_candidate"
    return "bounded_intermediate_pattern"


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
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
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.runtime import assemble_runtime

    if int(args.repeat_count) != REPEAT_COUNT:
        raise SystemExit(f"this unit requires exactly {REPEAT_COUNT} repeats")
    if int(args.local_window_radius) != LOCAL_WINDOW_RADIUS:
        raise SystemExit(
            f"this unit requires local-window radius {LOCAL_WINDOW_RADIUS}"
        )
    prevalence_path = args.prevalence_receipt.expanduser().resolve(strict=True)
    prevalence_receipt = _read_json(prevalence_path)
    contracts = derive_case_contracts(prevalence_receipt, args.image_ids)
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
    raw_by_image_id = {
        str(int(row.metadata.get("source", {}).get("image_id"))): row
        for row in raw_rows
        if row.metadata.get("source", {}).get("image_id") is not None
    }
    missing = [
        contract["image_id"]
        for contract in contracts
        if contract["image_id"] not in raw_by_image_id
    ]
    if missing:
        raise SystemExit(f"selected images are absent from source JSON Lines: {missing}")

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
    prepared: list[dict[str, Any]] = []
    for contract in contracts:
        image_id = contract["image_id"]
        bundle_path = Path(contract["source_bundle_path"]).expanduser().resolve(strict=True)
        bundle = _read_json(bundle_path)
        source_evidence = validate_source_bundle(bundle, image_id=image_id)
        source_row = extract_first_complete_row(bundle)
        if source_row["row_token_ids_sha256"] != contract["source_first_row"][
            "row_token_ids_sha256"
        ]:
            raise SystemExit(f"image {image_id} source first row changed")
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
            raise SystemExit(f"image {image_id} base prompt differs from source")
        if _json_hash(recipient_prompt.prompt_token_ids) != contract[
            "source_recipient_prompt_token_ids_sha256"
        ]:
            raise SystemExit(f"image {image_id} recipient prompt hash changed")
        prepared.append(
            {
                **contract,
                "raw": raw,
                "source_bundle": bundle,
                "source_bundle_path": bundle_path,
                "source_evidence": source_evidence,
                "source_row": source_row,
                "recipient_prompt_token_ids": list(recipient_prompt.prompt_token_ids),
            }
        )

    image_plan = materialize_image_plan_batch(
        [case["raw"] for case in prepared],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=list(range(len(prepared))),
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
    policy = DecodeGenerationPolicy.greedy(max_new_tokens=16, repetition_penalty=1.0)
    case_results: list[dict[str, Any]] = []
    for case in prepared:
        image_id = case["image_id"]
        raw = case["raw"]
        model_inputs = image_plan.model_inputs_by_row_id[raw.example_id]
        identity_checks = _validate_identity_continuity(
            source_bundle=case["source_bundle"],
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            source_image_sha256=_sha256_file(raw.image.path),
        )
        window = local_window(
            int(case["source_single_selected_coordinate_bin"]),
            int(args.local_window_radius),
        )
        executions: list[dict[str, Any]] = []
        for repeat_index in range(REPEAT_COUNT):
            for layout in (SINGLE_RECIPIENT_LAYOUT, HOMOGENEOUS_FOUR_COPY_LAYOUT):
                cached_requests = _build_requests(
                    image_id=image_id,
                    layout=layout,
                    repeat_index=repeat_index,
                    recipient_prompt_token_ids=case["recipient_prompt_token_ids"],
                    common_prefix=case["common_generated_prefix_token_ids"],
                    model_inputs=model_inputs,
                    policy=policy,
                    direct=False,
                )
                cached_logits, cached_tokens, cached_receipt = _natural_cached_logits(
                    backend=backend,
                    requests=cached_requests,
                    policy=policy,
                    common_prefix=case["common_generated_prefix_token_ids"],
                )
                cached_path: dict[str, Any] = dict(cached_receipt)
                if cached_logits is not None and cached_tokens is not None:
                    cached_path["rows"] = _serialize_rows(
                        cached_logits, cached_tokens, window=window
                    )

                direct_requests = _build_requests(
                    image_id=image_id,
                    layout=layout,
                    repeat_index=repeat_index,
                    recipient_prompt_token_ids=case["recipient_prompt_token_ids"],
                    common_prefix=case["common_generated_prefix_token_ids"],
                    model_inputs=model_inputs,
                    policy=policy,
                    direct=True,
                )
                direct_logits, direct_tokens, direct_receipt = _direct_full_prefix_logits(
                    backend=backend, requests=direct_requests
                )
                executions.append(
                    {
                        "layout_name": layout,
                        "repeat_index": repeat_index,
                        "cached_natural_replay": cached_path,
                        "direct_full_prefix": {
                            **direct_receipt,
                            "recipient_reached_by_every_request": True,
                            "rows": _serialize_rows(
                                direct_logits, direct_tokens, window=window
                            ),
                        },
                    }
                )

        cached_comparison = build_case_comparisons(
            executions, path_name="cached_natural_replay"
        )
        direct_comparison = build_case_comparisons(
            executions, path_name="direct_full_prefix"
        )
        source_split_reproduced = False
        if args.model_dtype == "bfloat16" and cached_comparison.get("eligible"):
            expected_single = int(case["source_single_selected_coordinate_bin"])
            expected_homogeneous = int(case["source_homogeneous_selected_coordinate_bin"])
            observed_single = []
            observed_homogeneous = []
            for execution in executions:
                rows = execution["cached_natural_replay"].get("rows", [])
                if not rows:
                    continue
                selected_bin = rows[0]["selected_coordinate_bin"]
                if execution["layout_name"] == SINGLE_RECIPIENT_LAYOUT:
                    observed_single.append(selected_bin)
                else:
                    observed_homogeneous.append(selected_bin)
            source_split_reproduced = observed_single == [expected_single] * 2 and (
                observed_homogeneous == [expected_homogeneous] * 2
            )
        classification = (
            classify_bfloat_case(
                cached_comparison, source_split_reproduced=source_split_reproduced
            )
            if args.model_dtype == "bfloat16"
            else "float32_precision_control"
        )
        case_results.append(
            {
                "image_id": image_id,
                "source_bundle_path": str(case["source_bundle_path"]),
                "source_bundle_sha256": _sha256_file(case["source_bundle_path"]),
                "source_evidence": case["source_evidence"],
                "source_first_row": {
                    key: value
                    for key, value in case["source_row"].items()
                    if key != "prompt_token_ids"
                },
                "source_recipient_prompt_token_ids_sha256": case[
                    "source_recipient_prompt_token_ids_sha256"
                ],
                "first_differing_generated_token_index": case[
                    "first_differing_generated_token_index"
                ],
                "common_generated_prefix_token_ids": case[
                    "common_generated_prefix_token_ids"
                ],
                "common_generated_prefix_token_ids_sha256": case[
                    "common_generated_prefix_token_ids_sha256"
                ],
                "source_single_selected_coordinate_bin": case[
                    "source_single_selected_coordinate_bin"
                ],
                "source_homogeneous_selected_coordinate_bin": case[
                    "source_homogeneous_selected_coordinate_bin"
                ],
                "local_coordinate_window_inclusive": list(window),
                "identity_checks": identity_checks,
                "executions": executions,
                "cached_natural_replay_comparison": cached_comparison,
                "direct_full_prefix_comparison": direct_comparison,
                "source_cached_split_reproduced": source_split_reproduced,
                "bfloat16_case_classification": classification,
            }
        )

    bfloat_trust_passed = all(
        case["source_cached_split_reproduced"] for case in case_results
    ) if args.model_dtype == "bfloat16" else True
    return {
        "schema_version": "repeated_first_differing_slot_coordinate_logits.v1",
        "status": "complete",
        "requested_model_execution_dtype": str(args.model_dtype),
        "actual_model_execution_dtype": model_dtype,
        "prevalence_receipt_path": str(prevalence_path),
        "prevalence_receipt_sha256": _sha256_file(prevalence_path),
        "infer_config_path": str(config_path),
        "infer_config_sha256": _sha256_file(config_path),
        "source_jsonl_path": str(source_path),
        "source_jsonl_sha256": _sha256_file(source_path),
        "coordinate_token_ids": coordinate_token_ids(),
        "local_window_radius": LOCAL_WINDOW_RADIUS,
        "repeat_count": REPEAT_COUNT,
        "predecessor_broad_shift_anchor": PREDECESSOR_BROAD_SHIFT_ANCHOR,
        "execution_identity": {
            "model_dtype": model_dtype,
            "model_evaluation_mode": not bool(qwen.model.training),
            "attention_implementation": _attention_implementation(qwen.model),
            "runtime_identity": _runtime_identity(),
            "device_identity": _execution_device_identity(backend._model_device()),
            "generation_config_fingerprint": generation_fingerprint,
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
        },
        "bfloat16_source_split_trust_gate_passed": bfloat_trust_passed,
        "cases": case_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_panel(args)
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
                "source_split_trust_gate_passed": result[
                    "bfloat16_source_split_trust_gate_passed"
                ],
                "case_classifications": {
                    case["image_id"]: case["bfloat16_case_classification"]
                    for case in result["cases"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["bfloat16_source_split_trust_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
