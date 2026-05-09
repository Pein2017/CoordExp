"""Teacher-forced prefix-rollin diagnostics for compact-full detection.

This module is intentionally analysis-side: it does not call ``generate()``, does
not change decoding, and does not participate in training loss.  It consumes the
same prefix-rollin sidecars used by training and inspects teacher-forced logits
at local next-token positions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.common.detection_chat import build_detection_chat_messages
from src.common.detection_compact_rows import (
    OBJECT_REF_START_TOKEN,
    render_compact_row,
)
from src.common.detection_sequence import IM_END_TOKEN
from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig
from src.detection.data import (
    ObjectOrderingPlan,
    NormalizedDetectionSample,
    normalize_detection_row,
    parse_raw_detection_row,
)
from src.detection.objective import PreparedPrefixRollinExample, TokenTarget
from src.detection.objective import build_compact_prefix_rollin_example
from src.detection.runtime import resolve_latest_detection_prompts
from src.detection.template import CompactFullTemplate
from src.infer.checkpoints import validate_compact_coord_token_adapter_contract


@dataclass(frozen=True)
class GeneratedPrefixCase:
    """Generated compact prefix recovered from a free-decode artifact."""

    prefix_text: str
    pred_count: int
    prefix_text_source: str
    raw_ends_with_im_end: bool | None


def find_token_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
    *,
    start_hint: int = 0,
) -> int | None:
    """Return the start index for ``needle`` in ``haystack`` if present."""

    needle_tuple = tuple(int(value) for value in needle)
    if not needle_tuple:
        return None
    haystack_tuple = tuple(int(value) for value in haystack)
    max_start = len(haystack_tuple) - len(needle_tuple)
    for start in range(max(0, int(start_hint)), max_start + 1):
        if haystack_tuple[start : start + len(needle_tuple)] == needle_tuple:
            return start
    for start in range(0, max(0, int(start_hint))):
        if haystack_tuple[start : start + len(needle_tuple)] == needle_tuple:
            return start
    return None


def compute_prefix_rollin_position_delta(
    *,
    processor_input_ids: Sequence[int],
    example: PreparedPrefixRollinExample,
) -> int:
    """Map tokenized prefix-rollin positions onto processor/model positions."""

    full_start = find_token_subsequence(processor_input_ids, example.input_ids)
    if full_start is not None:
        return int(full_start)

    assistant_span = example.tokenized.assistant_token_span
    assistant_ids = example.input_ids[assistant_span.start : assistant_span.end]
    assistant_start = find_token_subsequence(
        processor_input_ids,
        assistant_ids,
        start_hint=assistant_span.start,
    )
    if assistant_start is None:
        raise ValueError("prefix_rollin_processor_alignment_failed")
    return int(assistant_start - assistant_span.start)


def score_prefix_rollin_logits(
    *,
    logits: torch.Tensor,
    example: PreparedPrefixRollinExample,
    tokenizer: object,
    processor_input_ids: Sequence[int] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Score EOS-vs-valid-next-object margins from teacher-forced logits."""

    batch_logits = _normalize_logits(logits)
    if int(batch_logits.shape[0]) != 1:
        raise ValueError("score_prefix_rollin_logits expects a single example")
    model_input_ids = tuple(
        int(token_id)
        for token_id in (
            tuple(example.input_ids)
            if processor_input_ids is None
            else tuple(processor_input_ids)
        )
    )
    position_delta = compute_prefix_rollin_position_delta(
        processor_input_ids=model_input_ids,
        example=example,
    )

    rows: list[dict[str, Any]] = []
    branch_target = _first_valid_next_object_target(example)
    if branch_target is not None:
        entry_row = _score_target_row(
            logits=batch_logits[0],
            example=example,
            tokenizer=tokenizer,
            target=branch_target,
            model_input_ids=model_input_ids,
            position_delta=position_delta,
            branch_kind="valid_next_object",
            valid_token_ids=_valid_token_ids_for_target(branch_target),
            top_k=top_k,
        )
        entry_row["prefix_mode"] = "gt_prefix_entry_after_separator"
        entry_row["boundary_kind"] = "object_ref_after_forced_separator"
        rows.append(entry_row)

    boundary_target = _first_continue_boundary_target(example)
    if (
        boundary_target is not None
        and branch_target is not None
        and int(boundary_target.position) != int(branch_target.position)
    ):
        boundary_row = _score_target_row(
            logits=batch_logits[0],
            example=example,
            tokenizer=tokenizer,
            target=boundary_target,
            model_input_ids=model_input_ids,
            position_delta=position_delta,
            branch_kind="valid_next_object",
            valid_token_ids=(int(boundary_target.teacher_token_id),),
            top_k=top_k,
        )
        boundary_row["prefix_mode"] = "gt_prefix_free_boundary"
        boundary_row["boundary_kind"] = "separator_before_next_object"
        rows.append(boundary_row)

    eos_target = _semantic_eos_target(example)
    rows.append(
        _score_target_row(
            logits=batch_logits[0],
            example=example,
            tokenizer=tokenizer,
            target=eos_target,
            model_input_ids=model_input_ids,
            position_delta=position_delta,
            branch_kind="semantic_eos",
            valid_token_ids=(),
            top_k=top_k,
        )
    )
    return rows


def summarize_prefix_rollin_probe_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate a tiny prefix-forced run without claiming evaluation metrics."""

    valid_rows = [row for row in rows if row.get("branch_kind") == "valid_next_object"]
    eos_rows = [row for row in rows if row.get("branch_kind") == "semantic_eos"]

    margins: list[float] = []
    valid_masses: list[float] = []
    margins_by_k: dict[int, list[float]] = {}
    margins_by_mode: dict[str, list[float]] = {}
    for row in valid_rows:
        margin = row.get("continue_minus_eos_margin")
        if not _finite(margin):
            margin = row.get("margin_valid_mass_minus_eos_logprob")
        if _finite(margin):
            value = float(margin)
            margins.append(value)
            mode = str(row.get("prefix_mode") or "unknown")
            margins_by_mode.setdefault(mode, []).append(value)
            if "prefix_k" in row or "rollin_k" in row:
                k = int(row.get("prefix_k", row.get("rollin_k")))
                margins_by_k.setdefault(k, []).append(value)

        valid_mass = row.get("valid_mass")
        if _finite(valid_mass):
            valid_masses.append(float(valid_mass))

    by_prefix_k = {
        str(k): {
            "row_count": len(values),
            "continue_margin_mean": mean(values),
            "continue_margin_min": min(values),
            "continue_margin_max": max(values),
            "continue_margin_le_zero_rate": sum(
                1 for value in values if value <= 0.0
            )
            / len(values),
        }
        for k, values in sorted(margins_by_k.items())
    }
    by_prefix_mode = {
        mode: {
            "row_count": len(values),
            "continue_margin_mean": mean(values),
            "continue_margin_min": min(values),
            "continue_margin_max": max(values),
            "continue_margin_le_zero_rate": sum(
                1 for value in values if value <= 0.0
            )
            / len(values),
        }
        for mode, values in sorted(margins_by_mode.items())
    }

    return {
        "diagnostic": "forced_prefix_continue_vs_eos_v0",
        "probe_family": "prefix_rollin_teacher_forced",
        "row_count": len(rows),
        "valid_next_object_row_count": len(valid_rows),
        "semantic_eos_row_count": len(eos_rows),
        "valid_margin_mean": mean(margins) if margins else None,
        "valid_margin_min": min(margins) if margins else None,
        "valid_margin_max": max(margins) if margins else None,
        "valid_margin_le_zero_rate": (
            sum(1 for value in margins if value <= 0.0) / len(margins)
            if margins
            else None
        ),
        "continue_margin_mean": mean(margins) if margins else None,
        "continue_margin_min": min(margins) if margins else None,
        "continue_margin_max": max(margins) if margins else None,
        "continue_margin_le_zero_rate": (
            sum(1 for value in margins if value <= 0.0) / len(margins)
            if margins
            else None
        ),
        "valid_mass_mean": mean(valid_masses) if valid_masses else None,
        "valid_mass_min": min(valid_masses) if valid_masses else None,
        "valid_mass_max": max(valid_masses) if valid_masses else None,
        "continue_margin_by_prefix_mode": by_prefix_mode,
        "continue_margin_by_prefix_k": by_prefix_k,
        "rollin_k_values": sorted(
            {
                int(row["rollin_k"])
                for row in rows
                if "rollin_k" in row and row["rollin_k"] is not None
            }
        ),
        "prefix_k_values": sorted(
            {
                int(row["prefix_k"])
                for row in rows
                if "prefix_k" in row and row["prefix_k"] is not None
            }
        ),
    }


def run_prefix_rollin_teacher_forced_probe(
    *,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    split: str,
    limit: int,
    k_values: Sequence[str],
    device: str,
    attn_implementation: str,
    top_k: int,
    prefix_modes: Sequence[str] = ("gt_prefix",),
    decode_artifact_path: Path | None = None,
    trace_artifact_path: Path | None = None,
) -> tuple[Path, Path]:
    """Run a local real-model probe and write ``per_case.jsonl`` + summary."""

    from PIL import Image
    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer

    training_config = ConfigLoader.load_materialized_training_config(str(config_path))
    if not isinstance(training_config, LatestDetectionTrainingConfig):
        raise TypeError("prefix rollin probe requires LatestDetectionTrainingConfig")
    processor_kwargs: dict[str, Any] = {"do_resize": False}
    mode_set = _normalize_prefix_modes(prefix_modes)

    scorer = TeacherForcedScorer(
        checkpoint_path=checkpoint_path,
        device=device,
        attn_implementation=attn_implementation,
        coord_mode="coord_tokens",
    )
    validate_compact_coord_token_adapter_contract(
        scorer.resolved_checkpoint,
        detection_sequence_format=training_config.detection_template.id,
    )

    dataset_jsonl = _resolve_data_path(
        training_config.data.val_jsonl if split == "val" else training_config.data.train_jsonl
    )
    image_root = _resolve_data_path(training_config.data.image_root)
    system_prompt, user_prompt = resolve_latest_detection_prompts(training_config)
    rows: list[dict[str, Any]] = []
    decode_rows = _load_artifact_rows_by_index(decode_artifact_path)
    trace_rows = _load_artifact_rows_by_index(trace_artifact_path)

    if "generated_prefix" in mode_set and decode_artifact_path is None:
        raise ValueError("prefix_mode=generated_prefix requires --decode-artifact")

    for record_idx, raw_payload in enumerate(_iter_jsonl(dataset_jsonl)):
        if record_idx >= int(limit):
            break
        raw = parse_raw_detection_row(raw_payload)
        normalized = normalize_detection_row(
            raw,
            object_ordering=_ordering_for_record(
                training_config=training_config,
                record_idx=record_idx,
            ),
        )
        if len(normalized.images) != 1:
            raise ValueError(
                "prefix_rollin_teacher_forced_probe currently supports single-image "
                f"records only; got {len(normalized.images)} images for "
                f"record_idx={record_idx}"
            )
        image_path = image_root / normalized.images[0]
        image = Image.open(image_path).convert("RGB")

        if "gt_prefix" in mode_set:
            resolved_k_values = _resolve_k_values(
                k_values, object_count=len(normalized.objects)
            )
            for k in resolved_k_values:
                example = build_compact_prefix_rollin_example(
                    objects=normalized.objects,
                    rollin_order=normalized.objects,
                    k=int(k),
                    tokenizer=scorer.tokenizer,
                    normalized_sample=normalized,
                    system_prompt=system_prompt,
                    user_content=f"<image>\n{user_prompt}",
                )
                processor_messages = build_detection_chat_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    images=[image],
                    assistant_text=example.rendered_assistant.text,
                )
                processor_chat_text = scorer.processor.apply_chat_template(
                    processor_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                model_inputs = scorer.processor(
                    text=[processor_chat_text],
                    images=[image],
                    return_tensors="pt",
                    padding=False,
                    **processor_kwargs,
                )
                model_inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in model_inputs.items()
                }
                with torch.inference_mode():
                    outputs = scorer.model(**model_inputs, use_cache=False)
                logits = getattr(outputs, "logits", None)
                if not isinstance(logits, torch.Tensor):
                    raise RuntimeError("teacher-forced scorer did not return logits")
                input_ids = model_inputs.get("input_ids")
                if not isinstance(input_ids, torch.Tensor):
                    raise RuntimeError("teacher-forced scorer missing input_ids")
                case_rows = score_prefix_rollin_logits(
                    logits=logits.detach().cpu(),
                    example=example,
                    tokenizer=scorer.tokenizer,
                    processor_input_ids=input_ids[0].detach().cpu().tolist(),
                    top_k=top_k,
                )
                for row in case_rows:
                    rows.append(
                        {
                            "config_path": str(config_path),
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_mode": scorer.resolved_checkpoint.checkpoint_mode,
                            "split": split,
                            "record_idx": record_idx,
                            "image_id": normalized.image_id,
                            "file_name": normalized.file_name,
                            "processor_kwargs": processor_kwargs,
                            "processor_input_ids_len": int(input_ids.shape[-1]),
                            "image_grid_thw": _tensor_to_list(
                                model_inputs.get("image_grid_thw")
                            ),
                            **row,
                        }
                    )

        if "generated_prefix" in mode_set:
            generated_case = _generated_prefix_case_for_record(
                record_idx=record_idx,
                decode_rows=decode_rows,
                trace_rows=trace_rows,
            )
            if generated_case is None:
                continue
            generated_rows = _score_generated_prefix_case(
                scorer=scorer,
                normalized=normalized,
                image=image,
                generated_case=generated_case,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                processor_kwargs=processor_kwargs,
                device=device,
                top_k=top_k,
            )
            for row in generated_rows:
                rows.append(
                    {
                        "config_path": str(config_path),
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_mode": scorer.resolved_checkpoint.checkpoint_mode,
                        "split": split,
                        "record_idx": record_idx,
                        "image_id": normalized.image_id,
                        "file_name": normalized.file_name,
                        "decode_artifact_path": (
                            str(decode_artifact_path) if decode_artifact_path else None
                        ),
                        "trace_artifact_path": (
                            str(trace_artifact_path) if trace_artifact_path else None
                        ),
                        **row,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_path = output_dir / "per_case.jsonl"
    with per_case_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "summary.json"
    summary = {
        **summarize_prefix_rollin_probe_rows(rows),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "limit": int(limit),
        "prefix_modes": sorted(mode_set),
        "per_case_jsonl": str(per_case_path),
        "dataset_jsonl": str(dataset_jsonl),
        "image_root": str(image_root),
        "decode_artifact_path": str(decode_artifact_path) if decode_artifact_path else None,
        "trace_artifact_path": str(trace_artifact_path) if trace_artifact_path else None,
        "processor_kwargs": processor_kwargs,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return per_case_path, summary_path


def _score_target_row(
    *,
    logits: torch.Tensor,
    example: PreparedPrefixRollinExample,
    tokenizer: object,
    target: TokenTarget,
    model_input_ids: Sequence[int],
    position_delta: int,
    branch_kind: str,
    valid_token_ids: Sequence[int],
    top_k: int,
) -> dict[str, Any]:
    processor_position = int(target.position + position_delta)
    if processor_position <= 0 or processor_position >= int(logits.shape[0]):
        raise ValueError(
            "prefix_rollin_target_position_out_of_bounds: "
            f"position={processor_position} seq_len={int(logits.shape[0])}"
        )
    observed = int(model_input_ids[processor_position])
    if observed != int(target.teacher_token_id):
        raise ValueError(
            "prefix_rollin_processor_target_identity_mismatch: "
            f"observed={observed} expected={int(target.teacher_token_id)}"
        )

    row_logits = logits[processor_position - 1].float()
    log_probs = torch.log_softmax(row_logits, dim=-1)
    eos_id = int(example.stop_contract.im_end_token_id)
    eos_logprob = float(log_probs[eos_id].detach().cpu().item())
    eos_logit = float(row_logits[eos_id].detach().cpu().item())
    unique_valid_ids = tuple(
        dict.fromkeys(int(token_id) for token_id in valid_token_ids)
    )

    continue_logsumexp: float | None = None
    valid_logprob_mass: float | None = None
    valid_mass: float | None = None
    valid_max_logit: float | None = None
    margin_logprob: float | None = None
    continue_minus_eos_margin: float | None = None
    margin_logit: float | None = None
    if unique_valid_ids:
        valid_tensor = torch.tensor(
            unique_valid_ids, dtype=torch.long, device=row_logits.device
        )
        valid_log_probs = log_probs.index_select(dim=-1, index=valid_tensor)
        valid_logits = row_logits.index_select(dim=-1, index=valid_tensor)
        valid_logprob_mass_tensor = torch.logsumexp(valid_log_probs, dim=-1)
        continue_logsumexp_tensor = torch.logsumexp(valid_logits, dim=-1)
        valid_logprob_mass = float(valid_logprob_mass_tensor.detach().cpu().item())
        valid_mass = float(valid_logprob_mass_tensor.exp().detach().cpu().item())
        continue_logsumexp = float(continue_logsumexp_tensor.detach().cpu().item())
        valid_max_logit = float(valid_logits.max().detach().cpu().item())
        margin_logprob = float(valid_logprob_mass - eos_logprob)
        continue_minus_eos_margin = float(continue_logsumexp - eos_logit)
        margin_logit = float(valid_max_logit - eos_logit)

    return {
        "diagnostic": "forced_prefix_continue_vs_eos_v0",
        "probe_family": "prefix_rollin_teacher_forced",
        "template_id": example.template_id,
        "objective_variant": example.mode,
        "prefix_mode": "gt_prefix",
        "prefix_k": int(example.rollin_state.k),
        "rollin_k": int(example.rollin_state.k),
        "gt_count": len(example.rollin_state.permutation),
        "object_count": len(example.rollin_state.permutation),
        "remaining_gt_count": len(example.rollin_state.remaining),
        "prefix_object_instance_ids": [
            str(item) for item in example.rollin_state.emitted
        ],
        "emitted_object_instance_ids": [
            str(item) for item in example.rollin_state.emitted
        ],
        "remaining_object_instance_ids": [
            str(item) for item in example.rollin_state.remaining
        ],
        "teacher_suffix_object_instance_ids": [
            str(item) for item in example.rollin_state.remaining
        ],
        "prefix_token_count": len(example.debug_spans["rollin_prefix"].token_positions),
        "supervised_suffix_token_count": len(
            example.debug_spans["supervised_suffix"].token_positions
        ),
        "semantic_eos_token_count": len(
            example.debug_spans["semantic_eos"].token_positions
        ),
        "branch_kind": branch_kind,
        "valid_next_object_branch_present": branch_kind == "valid_next_object",
        "target_kind": str(target.kind),
        "target_token_role": _enum_value(target.token_role),
        "target_semantic_role": _enum_value(target.semantic_role),
        "target_object_instance_id": target.object_instance_id,
        "target_position": int(target.position),
        "processor_position": processor_position,
        "teacher_token_id": int(target.teacher_token_id),
        "teacher_token_text": _decode_token(tokenizer, int(target.teacher_token_id)),
        "teacher_logit": float(row_logits[int(target.teacher_token_id)].detach().cpu().item()),
        "teacher_logprob": float(log_probs[int(target.teacher_token_id)].detach().cpu().item()),
        "valid_token_ids": list(unique_valid_ids),
        "valid_token_texts": [_decode_token(tokenizer, token_id) for token_id in unique_valid_ids],
        "eos_token_id": eos_id,
        "eos_token_text": IM_END_TOKEN,
        "continue_logsumexp": continue_logsumexp,
        "continue_logprob_mass": valid_logprob_mass,
        "valid_logprob_mass": valid_logprob_mass,
        "valid_mass": valid_mass,
        "eos_logprob": eos_logprob,
        "continue_minus_eos_margin": continue_minus_eos_margin,
        "margin_valid_mass_minus_eos_logprob": margin_logprob,
        "valid_max_logit": valid_max_logit,
        "eos_logit": eos_logit,
        "valid_max_minus_eos_margin": margin_logit,
        "margin_valid_max_logit_minus_eos_logit": margin_logit,
        "top_tokens": _top_tokens(
            logits=row_logits,
            log_probs=log_probs,
            tokenizer=tokenizer,
            top_k=top_k,
        ),
    }


def _score_generated_prefix_case(
    *,
    scorer: Any,
    normalized: NormalizedDetectionSample,
    image: Any,
    generated_case: GeneratedPrefixCase,
    system_prompt: str,
    user_prompt: str,
    processor_kwargs: Mapping[str, Any],
    device: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Score a self-generated compact prefix against continue-vs-EOS."""

    pred_count = int(generated_case.pred_count)
    gt_count = len(normalized.objects)
    if pred_count >= gt_count:
        return []

    template = CompactFullTemplate()
    prefix_text = generated_case.prefix_text
    boundary_prefix_text = prefix_text + ("\n" if prefix_text else "")
    next_object = normalized.objects[pred_count]
    continuation_text = template.render_entry(next_object)
    assistant_text = boundary_prefix_text + continuation_text

    raw_prefix_ids = _encode_no_special_tokens(scorer.tokenizer, prefix_text)
    separator_ids = _encode_no_special_tokens(scorer.tokenizer, "\n")
    assistant_ids = _encode_no_special_tokens(scorer.tokenizer, assistant_text)
    boundary_prefix_ids = _encode_no_special_tokens(
        scorer.tokenizer, boundary_prefix_text
    )
    object_ref_id = _require_token_id(scorer.tokenizer, OBJECT_REF_START_TOKEN)

    free_boundary_target_position_in_assistant = len(raw_prefix_ids)
    if prefix_text:
        free_boundary_teacher_token_id = int(separator_ids[0])
        free_boundary_kind = "separator_before_next_object"
    else:
        free_boundary_teacher_token_id = object_ref_id
        free_boundary_kind = "first_object_entry"
    if (
        int(assistant_ids[free_boundary_target_position_in_assistant])
        != free_boundary_teacher_token_id
    ):
        raise ValueError(
            "generated_prefix_free_boundary_identity_mismatch: "
            f"observed={assistant_ids[free_boundary_target_position_in_assistant]} "
            f"expected={free_boundary_teacher_token_id}"
        )

    entry_target_position_in_assistant = len(boundary_prefix_ids)
    if int(assistant_ids[entry_target_position_in_assistant]) != object_ref_id:
        raise ValueError(
            "generated_prefix_target_identity_mismatch: "
            f"observed={assistant_ids[entry_target_position_in_assistant]} "
            f"expected={object_ref_id}"
        )

    processor_messages = build_detection_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[image],
        assistant_text=assistant_text,
    )
    processor_chat_text = scorer.processor.apply_chat_template(
        processor_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    model_inputs = scorer.processor(
        text=[processor_chat_text],
        images=[image],
        return_tensors="pt",
        padding=False,
        **dict(processor_kwargs),
    )
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }
    with torch.inference_mode():
        outputs = scorer.model(**model_inputs, use_cache=False)
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("teacher-forced scorer did not return logits")
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("teacher-forced scorer missing input_ids")

    processor_ids = input_ids[0].detach().cpu().tolist()
    assistant_start = find_token_subsequence(processor_ids, assistant_ids)
    if assistant_start is None:
        raise ValueError("generated_prefix_processor_alignment_failed")

    common_metadata = {
        "diagnostic": "forced_prefix_continue_vs_eos_v0",
        "probe_family": "generated_prefix_teacher_forced",
        "template_id": "compact_full",
        "objective_variant": "prefix_rollin_et_rmp_ce",
        "prefix_mode": "generated_prefix_count_depth",
        "prefix_k": pred_count,
        "rollin_k": pred_count,
        "gt_count": gt_count,
        "object_count": gt_count,
        "generated_pred_count": pred_count,
        "remaining_gt_count": gt_count - pred_count,
        "prefix_text_source": generated_case.prefix_text_source,
        "raw_ends_with_im_end": generated_case.raw_ends_with_im_end,
        "prefix_object_instance_ids": [],
        "emitted_object_instance_ids": [],
        "remaining_object_instance_ids": [
            str(obj.object_instance_id) for obj in normalized.objects[pred_count:]
        ],
        "teacher_suffix_object_instance_ids": [
            str(obj.object_instance_id) for obj in normalized.objects[pred_count:]
        ],
        "target_object_instance_id": str(next_object.object_instance_id),
        "prefix_token_count": len(boundary_prefix_ids),
        "supervised_suffix_token_count": len(assistant_ids) - len(boundary_prefix_ids),
        "semantic_eos_token_count": 0,
        "processor_kwargs": dict(processor_kwargs),
        "processor_input_ids_len": int(input_ids.shape[-1]),
        "image_grid_thw": _tensor_to_list(model_inputs.get("image_grid_thw")),
    }

    free_boundary_metadata = {
        **common_metadata,
        "prefix_mode": "generated_prefix_free_boundary",
        "boundary_kind": free_boundary_kind,
    }
    free_boundary_row = _score_forced_prefix_boundary_logits(
        logits=logits.detach().cpu(),
        tokenizer=scorer.tokenizer,
        processor_input_ids=processor_ids,
        target_position=int(assistant_start + free_boundary_target_position_in_assistant),
        teacher_token_id=free_boundary_teacher_token_id,
        valid_token_ids=(free_boundary_teacher_token_id,),
        eos_token_id=_require_token_id(scorer.tokenizer, IM_END_TOKEN),
        branch_kind="valid_next_object",
        top_k=top_k,
        metadata=free_boundary_metadata,
    )

    entry_metadata = {
        **common_metadata,
        "prefix_mode": "generated_prefix_entry_after_separator",
        "boundary_kind": "object_ref_after_forced_separator",
    }
    entry_row = _score_forced_prefix_boundary_logits(
        logits=logits.detach().cpu(),
        tokenizer=scorer.tokenizer,
        processor_input_ids=processor_ids,
        target_position=int(assistant_start + entry_target_position_in_assistant),
        teacher_token_id=object_ref_id,
        valid_token_ids=(object_ref_id,),
        eos_token_id=_require_token_id(scorer.tokenizer, IM_END_TOKEN),
        branch_kind="valid_next_object",
        top_k=top_k,
        metadata=entry_metadata,
    )
    return [free_boundary_row, entry_row]


def _score_forced_prefix_boundary_logits(
    *,
    logits: torch.Tensor,
    tokenizer: object,
    processor_input_ids: Sequence[int],
    target_position: int,
    teacher_token_id: int,
    valid_token_ids: Sequence[int],
    eos_token_id: int,
    branch_kind: str,
    top_k: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one forced-prefix local next-token boundary."""

    batch_logits = _normalize_logits(logits)
    if int(batch_logits.shape[0]) != 1:
        raise ValueError("forced-prefix boundary scoring expects a single example")
    processor_position = int(target_position)
    if processor_position <= 0 or processor_position >= int(batch_logits.shape[1]):
        raise ValueError(
            "forced_prefix_target_position_out_of_bounds: "
            f"position={processor_position} seq_len={int(batch_logits.shape[1])}"
        )
    observed = int(processor_input_ids[processor_position])
    if observed != int(teacher_token_id):
        raise ValueError(
            "forced_prefix_processor_target_identity_mismatch: "
            f"observed={observed} expected={int(teacher_token_id)}"
        )

    row_logits = batch_logits[0, processor_position - 1].float()
    log_probs = torch.log_softmax(row_logits, dim=-1)
    eos_id = int(eos_token_id)
    eos_logprob = float(log_probs[eos_id].detach().cpu().item())
    eos_logit = float(row_logits[eos_id].detach().cpu().item())
    unique_valid_ids = tuple(dict.fromkeys(int(token_id) for token_id in valid_token_ids))

    continue_logsumexp: float | None = None
    valid_logprob_mass: float | None = None
    valid_mass: float | None = None
    valid_max_logit: float | None = None
    continue_minus_eos_margin: float | None = None
    valid_max_minus_eos_margin: float | None = None
    if unique_valid_ids:
        valid_tensor = torch.tensor(
            unique_valid_ids, dtype=torch.long, device=row_logits.device
        )
        valid_log_probs = log_probs.index_select(dim=-1, index=valid_tensor)
        valid_logits = row_logits.index_select(dim=-1, index=valid_tensor)
        valid_logprob_mass_tensor = torch.logsumexp(valid_log_probs, dim=-1)
        continue_logsumexp_tensor = torch.logsumexp(valid_logits, dim=-1)
        valid_logprob_mass = float(valid_logprob_mass_tensor.detach().cpu().item())
        valid_mass = float(valid_logprob_mass_tensor.exp().detach().cpu().item())
        continue_logsumexp = float(continue_logsumexp_tensor.detach().cpu().item())
        valid_max_logit = float(valid_logits.max().detach().cpu().item())
        continue_minus_eos_margin = float(continue_logsumexp - eos_logit)
        valid_max_minus_eos_margin = float(valid_max_logit - eos_logit)

    return {
        **dict(metadata),
        "branch_kind": branch_kind,
        "valid_next_object_branch_present": branch_kind == "valid_next_object",
        "target_kind": "generated_prefix_boundary",
        "target_token_role": "object_control",
        "target_semantic_role": "object_entry",
        "target_object_instance_id": metadata.get("target_object_instance_id"),
        "target_position": processor_position,
        "processor_position": processor_position,
        "teacher_token_id": int(teacher_token_id),
        "teacher_token_text": _decode_token(tokenizer, int(teacher_token_id)),
        "teacher_logit": float(row_logits[int(teacher_token_id)].detach().cpu().item()),
        "teacher_logprob": float(
            log_probs[int(teacher_token_id)].detach().cpu().item()
        ),
        "valid_token_ids": list(unique_valid_ids),
        "valid_token_texts": [_decode_token(tokenizer, token_id) for token_id in unique_valid_ids],
        "eos_token_id": eos_id,
        "eos_token_text": IM_END_TOKEN,
        "continue_logsumexp": continue_logsumexp,
        "continue_logprob_mass": valid_logprob_mass,
        "valid_logprob_mass": valid_logprob_mass,
        "valid_mass": valid_mass,
        "eos_logprob": eos_logprob,
        "continue_minus_eos_margin": continue_minus_eos_margin,
        "margin_valid_mass_minus_eos_logprob": (
            float(valid_logprob_mass - eos_logprob)
            if valid_logprob_mass is not None
            else None
        ),
        "valid_max_logit": valid_max_logit,
        "eos_logit": eos_logit,
        "valid_max_minus_eos_margin": valid_max_minus_eos_margin,
        "margin_valid_max_logit_minus_eos_logit": valid_max_minus_eos_margin,
        "top_tokens": _top_tokens(
            logits=row_logits,
            log_probs=log_probs,
            tokenizer=tokenizer,
            top_k=top_k,
        ),
    }


def _first_valid_next_object_target(
    example: PreparedPrefixRollinExample,
) -> TokenTarget | None:
    remaining = {str(item) for item in example.rollin_state.remaining}
    if not remaining:
        return None
    suffix_entry_positions = set(
        int(pos)
        for pos in example.debug_spans["supervised_suffix_entries"].token_positions
    )
    for target in example.recursive_detection_targets.token_targets:
        if (
            target.position in suffix_entry_positions
            and target.object_instance_id in remaining
        ):
            return target
    return None


def _first_continue_boundary_target(
    example: PreparedPrefixRollinExample,
) -> TokenTarget | None:
    if not example.rollin_state.remaining:
        return None
    suffix_positions = tuple(
        int(pos) for pos in example.debug_spans["supervised_suffix"].token_positions
    )
    if not suffix_positions:
        return None
    boundary_position = min(suffix_positions)
    for target in example.recursive_detection_targets.token_targets:
        if int(target.position) == boundary_position:
            return target
    return None


def _semantic_eos_target(example: PreparedPrefixRollinExample) -> TokenTarget:
    eos_positions = set(int(pos) for pos in example.debug_spans["semantic_eos"].token_positions)
    for target in example.recursive_detection_targets.token_targets:
        if int(target.position) in eos_positions:
            return target
    raise ValueError("prefix_rollin_semantic_eos_target_missing")


def _valid_token_ids_for_target(target: TokenTarget) -> tuple[int, ...]:
    if target.valid_token_ids:
        return tuple(int(token_id) for token_id in target.valid_token_ids)
    return (int(target.teacher_token_id),)


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits.unsqueeze(0)
    if logits.ndim == 3:
        return logits
    raise ValueError(f"logits must have shape [T,V] or [B,T,V], got {tuple(logits.shape)}")


def _decode_token(tokenizer: object, token_id: int) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(decode([int(token_id)]))
        except Exception:
            pass
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, Mapping):
            for token_text, vocab_id in vocab.items():
                if int(vocab_id) == int(token_id):
                    return str(token_text)
    return f"<token:{int(token_id)}>"


def _top_tokens(
    *,
    logits: torch.Tensor,
    log_probs: torch.Tensor,
    tokenizer: object,
    top_k: int,
) -> list[dict[str, Any]]:
    limit = max(0, min(int(top_k), int(logits.shape[-1])))
    if limit == 0:
        return []
    values, indices = torch.topk(logits, k=limit)
    rows: list[dict[str, Any]] = []
    for value, token_id in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        token_id_int = int(token_id)
        rows.append(
            {
                "token_id": token_id_int,
                "token_text": _decode_token(tokenizer, token_id_int),
                "logit": float(value),
                "logprob": float(log_probs[token_id_int].detach().cpu().item()),
            }
        )
    return rows


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _tensor_to_list(value: object) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _encode_no_special_tokens(tokenizer: object, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise ValueError("forced-prefix probe requires tokenizer.encode")
    return [int(token_id) for token_id in encode(text, add_special_tokens=False)]


def _require_token_id(tokenizer: object, token: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ValueError(f"forced-prefix probe requires token id for {token!r}")
    token_id = int(convert(token))
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and token_id == int(unk_token_id):
        raise ValueError(f"forced-prefix probe could not resolve {token!r}")
    return token_id


def _load_artifact_rows_by_index(path: Path | None) -> dict[int, Mapping[str, Any]]:
    if path is None:
        return {}
    resolved = _resolve_data_path(path)
    rows: dict[int, Mapping[str, Any]] = {}
    for fallback_idx, payload in enumerate(_iter_jsonl(resolved)):
        raw_idx = payload.get("line_idx", payload.get("record_idx", fallback_idx))
        rows[int(raw_idx)] = payload
    return rows


def _generated_prefix_case_for_record(
    *,
    record_idx: int,
    decode_rows: Mapping[int, Mapping[str, Any]],
    trace_rows: Mapping[int, Mapping[str, Any]],
) -> GeneratedPrefixCase | None:
    decode_row = decode_rows.get(int(record_idx))
    if decode_row is None:
        return None

    raw_output_json = decode_row.get("raw_output_json")
    raw_objects = []
    if isinstance(raw_output_json, Mapping):
        objects_raw = raw_output_json.get("objects")
        if isinstance(objects_raw, list):
            raw_objects = objects_raw
    if not raw_objects:
        pred_objects = decode_row.get("pred")
        if isinstance(pred_objects, list):
            raw_objects = pred_objects

    trace_row = trace_rows.get(int(record_idx))
    if trace_row is not None:
        trace_tokens = trace_row.get("generated_token_text")
        if isinstance(trace_tokens, list) and all(
            isinstance(token, str) for token in trace_tokens
        ):
            prefix_tokens: list[str] = []
            for token in trace_tokens:
                if token == IM_END_TOKEN:
                    break
                prefix_tokens.append(token)
            prefix_text = "".join(prefix_tokens)
            return GeneratedPrefixCase(
                prefix_text=prefix_text,
                pred_count=len(raw_objects),
                prefix_text_source="pred_token_trace.generated_token_text",
                raw_ends_with_im_end=_optional_bool(decode_row.get("raw_ends_with_im_end")),
            )

    if not raw_objects:
        return GeneratedPrefixCase(
            prefix_text="",
            pred_count=0,
            prefix_text_source="empty_prediction",
            raw_ends_with_im_end=_optional_bool(decode_row.get("raw_ends_with_im_end")),
        )

    return GeneratedPrefixCase(
        prefix_text=_render_compact_prefix_objects(raw_objects),
        pred_count=len(raw_objects),
        prefix_text_source="gt_vs_pred.raw_output_json.objects",
        raw_ends_with_im_end=_optional_bool(decode_row.get("raw_ends_with_im_end")),
    )


def _render_compact_prefix_objects(objects: Sequence[Any]) -> str:
    rows: list[str] = []
    for obj in objects:
        if not isinstance(obj, Mapping):
            raise ValueError("generated prefix object must be a mapping")
        desc = str(obj.get("desc") or "").strip()
        bbox = obj.get("bbox_2d")
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
            raise ValueError("generated prefix object requires bbox_2d sequence")
        coord_tokens = tuple(str(token) for token in bbox)
        rows.append(
            render_compact_row(
                desc,
                coord_tokens,
                include_object_ref_marker=True,
                include_bbox_start_marker=True,
            )
        )
    return "\n".join(rows)


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _ordering_for_record(
    *,
    training_config: LatestDetectionTrainingConfig,
    record_idx: int,
) -> ObjectOrderingPlan:
    if training_config.data.object_ordering == "random_permutation":
        return ObjectOrderingPlan.random_permutation(
            seed=17_003 + int(record_idx),
            seed_source="prefix_forced_eval",
        )
    return ObjectOrderingPlan.sorted(seed_source="prefix_forced_eval")


def _resolve_k_values(k_values: Sequence[str], *, object_count: int) -> list[int]:
    resolved: list[int] = []
    for raw in k_values:
        value = str(raw).strip().lower()
        if value in {"every", "all_prefixes", "all-prefixes", "range", "0..n", "0:n"}:
            for k in range(0, int(object_count) + 1):
                if k not in resolved:
                    resolved.append(k)
            continue
        if value in {"all", "n", "object_count"}:
            k = int(object_count)
        else:
            k = int(value)
        if k < 0 or k > int(object_count):
            raise ValueError(f"k must satisfy 0 <= k <= {object_count}, got {k}")
        if k not in resolved:
            resolved.append(k)
    return resolved


def _resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    coordexp_candidate = Path("/data/CoordExp") / candidate
    if coordexp_candidate.exists():
        return coordexp_candidate
    return candidate


def _normalize_prefix_modes(prefix_modes: Sequence[str]) -> set[str]:
    modes: set[str] = set()
    for raw in prefix_modes:
        value = str(raw).strip().lower().replace("-", "_")
        if not value:
            continue
        if value == "both":
            modes.update({"gt_prefix", "generated_prefix"})
            continue
        if value in {"gt", "clean", "clean_gt"}:
            value = "gt_prefix"
        if value in {"generated", "self", "self_prefix", "decode_prefix"}:
            value = "generated_prefix"
        if value not in {"gt_prefix", "generated_prefix"}:
            raise ValueError(
                "prefix mode must be one of gt_prefix, generated_prefix, or both; "
                f"got {raw!r}"
            )
        modes.add(value)
    return modes or {"gt_prefix"}


def _parse_prefix_modes(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or ["gt_prefix"]


def _parse_k_values(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or ["0", "1", "all"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a teacher-forced compact prefix-rollin EOS-vs-branch probe."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("temp/prefix_rollin_tf_probe"),
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--prefix-modes",
        default="gt_prefix",
        help=(
            "Comma-separated modes: gt_prefix, generated_prefix, or both. "
            "generated_prefix requires --decode-artifact."
        ),
    )
    parser.add_argument("--decode-artifact", type=Path, default=None)
    parser.add_argument("--trace-artifact", type=Path, default=None)
    parser.add_argument(
        "--k-values",
        default="0,1,all",
        help=(
            "Comma-separated forced prefix depths. Use 'all'/'n' for K=N, "
            "or 'every'/'0..n' to score the full K=0..N curve."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    per_case, summary = run_prefix_rollin_teacher_forced_probe(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
        k_values=_parse_k_values(args.k_values),
        device=args.device,
        attn_implementation=args.attn_implementation,
        top_k=args.top_k,
        prefix_modes=_parse_prefix_modes(args.prefix_modes),
        decode_artifact_path=args.decode_artifact,
        trace_artifact_path=args.trace_artifact,
    )
    print(f"wrote {per_case}")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compute_prefix_rollin_position_delta",
    "find_token_subsequence",
    "run_prefix_rollin_teacher_forced_probe",
    "score_prefix_rollin_logits",
    "summarize_prefix_rollin_probe_rows",
]
