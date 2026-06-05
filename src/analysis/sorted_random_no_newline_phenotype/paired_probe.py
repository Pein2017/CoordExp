from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from . import PHASE_ID, PROJECT_ID, RUN_ID, SCHEMA_VERSION
from .boundary_roles import summarize_boundary

if TYPE_CHECKING:
    from .config import A32Config


REAL_PREFIX_RUNTIME_KIND = "real_gpu_prefix_readout_v1"
READOUT_BEHAVIOR = "canonical sorted teacher-prefix readout"
PREFIX_SOURCE_POLICY = "canonical_sorted_teacher_prefix_readout"
PREFIX_ORDER_POLICY_ID = "canonical_sorted_yx_teacher_v1"
READOUT_PROMPT_ORDERING = "sorted"
PREFIX_READOUT_DECODE_POLICY = "forward_logprob_readout_no_decode"


def checkpoint_roles_from_config(config: A32Config) -> list[str]:
    return [str(role) for role in config.checkpoints]


def build_shard_manifest_row(
    config: A32Config,
    *,
    shard_id: int,
    prefix_state_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    roles = checkpoint_roles_from_config(config)
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "shard_id": int(shard_id),
        "checkpoint_roles": roles,
        "checkpoint_training_ordering": {
            role: str(config.checkpoints[role].training_ordering) for role in roles
        },
        "checkpoint_readout_prompt_ordering": {
            role: str(config.checkpoints[role].readout_prompt_ordering)
            for role in roles
        },
        "checkpoint_paths": {
            role: str(config.checkpoints[role].checkpoint_path) for role in roles
        },
        "checkpoint_objective_policy": {
            role: str(config.checkpoints[role].objective_policy) for role in roles
        },
        "checkpoint_comparison_group": {
            role: str(config.checkpoints[role].comparison_group) for role in roles
        },
        "checkpoint_template_contract_id": {
            role: str(config.checkpoints[role].template_contract_id)
            for role in roles
        },
        "prefix_source_policy": PREFIX_SOURCE_POLICY,
        "readout_prompt_ordering": READOUT_PROMPT_ORDERING,
        "teacher_prefix_ordering": PREFIX_ORDER_POLICY_ID,
        "readout_behavior": READOUT_BEHAVIOR,
        "prefix_state_ids": [
            str(row["prefix_state_id"])
            for row in prefix_state_rows
            if row.get("prefix_state_id") is not None
        ],
        "prefix_state_count": len(prefix_state_rows),
    }


def build_mocked_paired_readout_rows(
    config: A32Config,
    *,
    prefix_state_rows: Sequence[Mapping[str, Any]],
    boundary_summaries_by_role: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    roles = checkpoint_roles_from_config(config)
    rows: list[dict[str, Any]] = []
    for prefix_row in prefix_state_rows:
        prefix_state_id = str(prefix_row["prefix_state_id"])
        for role in roles:
            checkpoint = config.checkpoints[role]
            rows.append(
                {
                    "project_id": PROJECT_ID,
                    "phase_id": PHASE_ID,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": RUN_ID,
                    "prefix_state_id": prefix_state_id,
                    "shard_id": _json_safe(
                        prefix_row.get("shard_id"),
                        "prefix_state_row.shard_id",
                    ),
                    "image_id": _json_safe(
                        prefix_row.get("image_id"),
                        "prefix_state_row.image_id",
                    ),
                    "checkpoint_role": role,
                    "checkpoint_roles": list(roles),
                    "checkpoint_training_ordering": str(
                        checkpoint.training_ordering
                    ),
                    "checkpoint_readout_prompt_ordering": str(
                        checkpoint.readout_prompt_ordering
                    ),
                    "objective_policy": str(checkpoint.objective_policy),
                    "comparison_group": str(checkpoint.comparison_group),
                    "template_contract_id": str(checkpoint.template_contract_id),
                    "prefix_source_policy": PREFIX_SOURCE_POLICY,
                    "readout_prompt_ordering": READOUT_PROMPT_ORDERING,
                    "teacher_prefix_ordering": PREFIX_ORDER_POLICY_ID,
                    "readout_behavior": READOUT_BEHAVIOR,
                    "prefix_state_row": _json_safe(
                        prefix_row,
                        "prefix_state_row",
                    ),
                    "boundary_summary": _json_safe(
                        _boundary_summary_for_prefix(
                            boundary_summaries_by_role,
                            role=role,
                            prefix_state_id=prefix_state_id,
                        ),
                        "boundary_summary",
                    ),
                }
            )
    return rows


def build_real_paired_readout_rows(
    config: A32Config,
    *,
    prefix_state_rows: Sequence[Mapping[str, Any]],
    boundary_summaries_by_role: Mapping[str, Mapping[str, Mapping[str, Any]]],
    shard_id: int,
    gpu_id: str,
    checkpoint_fingerprints: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build JSON-safe real prefix-readout rows from externally scored evidence."""

    roles = checkpoint_roles_from_config(config)
    fingerprints = dict(checkpoint_fingerprints or {})
    rows: list[dict[str, Any]] = []
    for prefix_row in prefix_state_rows:
        prefix_state_id = str(prefix_row["prefix_state_id"])
        for role in roles:
            checkpoint = config.checkpoints[role]
            summary = _json_safe(
                _boundary_summary_for_prefix(
                    boundary_summaries_by_role,
                    role=role,
                    prefix_state_id=prefix_state_id,
                ),
                "boundary_summary",
            )
            rows.append(
                {
                    "project_id": PROJECT_ID,
                    "phase_id": PHASE_ID,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": RUN_ID,
                    "runtime_kind": REAL_PREFIX_RUNTIME_KIND,
                    "gpu_id": str(gpu_id),
                    "decode_policy": PREFIX_READOUT_DECODE_POLICY,
                    "constraint_policy": "none",
                    "prefix_state_id": prefix_state_id,
                    "shard_id": int(shard_id),
                    "image_id": _json_safe(
                        prefix_row.get("image_id"),
                        "prefix_state_row.image_id",
                    ),
                    "checkpoint_role": role,
                    "checkpoint_roles": list(roles),
                    "checkpoint_training_ordering": str(
                        checkpoint.training_ordering
                    ),
                    "checkpoint_readout_prompt_ordering": str(
                        checkpoint.readout_prompt_ordering
                    ),
                    "checkpoint_path": str(checkpoint.checkpoint_path),
                    "checkpoint_fingerprint": fingerprints.get(role),
                    "prefix_source_policy": PREFIX_SOURCE_POLICY,
                    "readout_prompt_ordering": READOUT_PROMPT_ORDERING,
                    "teacher_prefix_ordering": PREFIX_ORDER_POLICY_ID,
                    "readout_behavior": READOUT_BEHAVIOR,
                    "prefix_state_row": _json_safe(
                        prefix_row,
                        "prefix_state_row",
                    ),
                    "boundary_summary": summary,
                    "boundary_winner_class": summary.get("boundary_winner_class"),
                    "residual_vs_eos_margin": summary.get(
                        "residual_vs_eos_margin"
                    ),
                    "residual_vs_winner_margin": summary.get(
                        "residual_vs_winner_margin"
                    ),
                    "strict_r95_x1_hit_rate": float(
                        summary.get("strict_r95_x1_hit_rate", 0.0)
                    ),
                    "boundary_residual_favored_rate": float(
                        summary.get("boundary_residual_favored_rate", 0.0)
                    ),
                }
            )
    return rows


def build_real_shard_summary_row(
    config: A32Config,
    *,
    shard_id: int,
    gpu_id: str,
    prefix_state_count: int,
    readout_row_count: int,
) -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "runtime_kind": REAL_PREFIX_RUNTIME_KIND,
        "gpu_id": str(gpu_id),
        "decode_policy": PREFIX_READOUT_DECODE_POLICY,
        "constraint_policy": "none",
        "shard_id": int(shard_id),
        "checkpoint_roles": checkpoint_roles_from_config(config),
        "prefix_source_policy": PREFIX_SOURCE_POLICY,
        "readout_prompt_ordering": READOUT_PROMPT_ORDERING,
        "teacher_prefix_ordering": PREFIX_ORDER_POLICY_ID,
        "prefix_state_count": int(prefix_state_count),
        "readout_row_count": int(readout_row_count),
    }


def run_real_paired_checkpoint_probe(
    config: A32Config,
    *,
    shard_id: int,
    allow_overwrite: bool = False,
    gpu_id: str | None = None,
) -> dict[str, Any]:
    """Run real desc/EOS and desc-conditioned x1 readout for one shard."""

    gpu = _gpu_id(gpu_id)
    sampled_path = config.artifact_root / "prefix_state_sampled_rows.jsonl"
    sampled_rows = _read_jsonl(sampled_path)
    shard_rows = [
        row for row in sampled_rows if int(row.get("shard_id", -1)) == int(shard_id)
    ]
    shard_path = (
        config.artifact_root
        / "prefix_readout_shards"
        / f"shard_{int(shard_id)}.jsonl"
    )
    summary_path = (
        config.artifact_root
        / "prefix_state_shard_summaries"
        / f"shard_{int(shard_id)}.json"
    )
    _ensure_can_write(shard_path, allow_overwrite=allow_overwrite)
    _ensure_can_write(summary_path, allow_overwrite=allow_overwrite)

    boundary_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    checkpoint_fingerprints: dict[str, str] = {}
    for role, checkpoint in config.checkpoints.items():
        model_handle = _load_model_for_checkpoint(config, checkpoint_path=checkpoint.checkpoint_path)
        checkpoint_fingerprints[role] = _checkpoint_fingerprint(model_handle)
        role_summaries: dict[str, dict[str, Any]] = {}
        for state in shard_rows:
            role_summaries[str(state["prefix_state_id"])] = _score_prefix_state(
                config,
                model_handle=model_handle,
                state=state,
            )
        boundary_summaries[role] = role_summaries

    readout_rows = build_real_paired_readout_rows(
        config,
        prefix_state_rows=shard_rows,
        boundary_summaries_by_role=boundary_summaries,
        shard_id=int(shard_id),
        gpu_id=gpu,
        checkpoint_fingerprints=checkpoint_fingerprints,
    )
    manifest_row = {
        "row_type": "shard_manifest",
        "runtime_kind": REAL_PREFIX_RUNTIME_KIND,
        "gpu_id": gpu,
        **build_shard_manifest_row(
            config,
            shard_id=int(shard_id),
            prefix_state_rows=shard_rows,
        ),
    }
    summary_row = build_real_shard_summary_row(
        config,
        shard_id=int(shard_id),
        gpu_id=gpu,
        prefix_state_count=len(shard_rows),
        readout_row_count=len(readout_rows),
    )
    _write_jsonl(shard_path, [manifest_row, *readout_rows])
    _write_json(summary_path, summary_row)
    return {
        "stage": "paired_checkpoint_probe",
        "runtime_kind": REAL_PREFIX_RUNTIME_KIND,
        "shard_id": int(shard_id),
        "gpu_id": gpu,
        "prefix_state_count": len(shard_rows),
        "readout_row_count": len(readout_rows),
        "shard_path": str(shard_path),
        "summary_path": str(summary_path),
    }


def _score_prefix_state(
    config: A32Config,
    *,
    model_handle: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    from src.analysis.prefix_state_transition_tomography.prefix_rendering import (
        render_boundary_assistant_text,
        render_forced_desc_pre_x1_assistant_text,
    )

    gt_objects = list(state.get("gt_objects") or [])
    emitted_order = [int(idx) for idx in state.get("emitted_gt_indices") or []]
    residual_ids = {int(idx) for idx in state.get("residual_gt_indices") or []}
    objects_by_idx = {int(obj["gt_idx"]): obj for obj in gt_objects}
    emitted_rows = [objects_by_idx[idx] for idx in emitted_order if idx in objects_by_idx]
    residual_objects = [
        obj for obj in gt_objects if int(obj.get("gt_idx", -1)) in residual_ids
    ]
    boundary_text = render_boundary_assistant_text(emitted_rows)
    image_path = _resolve_image_path(config, state)
    candidates = _score_boundary_candidates(
        model_handle=model_handle,
        image_path=image_path,
        assistant_prefix=boundary_text,
        candidates_with_roles=_candidate_descs_with_roles(state),
    )
    eos = _score_chat_eos_at_boundary(
        model_handle=model_handle,
        image_path=image_path,
        assistant_prefix=boundary_text,
    )
    summary = summarize_boundary(candidates, eos["score"])
    summary["eos_score"] = eos["score"]
    summary["eos_token_id"] = eos["token_id"]
    summary["eos_token_text"] = eos["token_text"]
    x1_summary = _score_residual_x1_coverage(
        config,
        model_handle=model_handle,
        image_path=image_path,
        emitted_rows=emitted_rows,
        residual_objects=residual_objects,
        render_forced_desc_pre_x1_assistant_text=render_forced_desc_pre_x1_assistant_text,
    )
    summary.update(x1_summary)
    summary["boundary_residual_favored_rate"] = (
        1.0
        if str(summary.get("boundary_winner_class")) in {
            "residual_same_desc_favored",
            "residual_other_desc_favored",
        }
        else 0.0
    )
    return summary


def _score_boundary_candidates(
    *,
    model_handle: Any,
    image_path: Path,
    assistant_prefix: str,
    candidates_with_roles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates_with_roles:
        desc = str(candidate.get("desc") or "").strip()
        if not desc:
            continue
        suffix = _forced_desc_pre_x1_suffix(desc)
        score = _score_suffix_mean_logprob(
            model_handle=model_handle,
            image_path=image_path,
            assistant_prefix=assistant_prefix,
            suffix=suffix,
        )
        scored.append(
            {
                "desc": desc,
                "roles": [str(role) for role in candidate.get("roles", ())],
                "score": score,
            }
        )
    return scored


def _score_residual_x1_coverage(
    config: A32Config,
    *,
    model_handle: Any,
    image_path: Path,
    emitted_rows: Sequence[Mapping[str, Any]],
    residual_objects: Sequence[Mapping[str, Any]],
    render_forced_desc_pre_x1_assistant_text: Any,
) -> dict[str, Any]:
    if not residual_objects:
        return {
            "strict_r95_x1_hit_rate": 0.0,
            "strict_r95_x1_residual_count": 0,
            "strict_r95_x1_hit_count": 0,
            "x1_probe_details": [],
        }

    details: list[dict[str, Any]] = []
    hit_count = 0
    residual_count = 0
    residual_by_desc: dict[str, list[Mapping[str, Any]]] = {}
    for obj in residual_objects:
        residual_by_desc.setdefault(str(obj.get("desc", "")), []).append(obj)

    for desc in sorted(key for key in residual_by_desc if key):
        forced_text = render_forced_desc_pre_x1_assistant_text(emitted_rows, desc)
        logits = _last_logits(
            model_handle=model_handle,
            image_path=image_path,
            assistant_text=forced_text,
            expected_context_suffix="<|box_start|>",
        )
        metrics, peaks = _x1_distribution_and_peaks(
            config,
            model_handle=model_handle,
            logits=logits,
            residual_same=residual_by_desc[desc],
        )
        peak_values = [int(peak["x1"]) for peak in peaks.get("merged_peaks", ())]
        for obj in residual_by_desc[desc]:
            bbox = _bbox_xyxy(obj)
            x1 = int(bbox[0])
            width = max(0, int(bbox[2]) - int(bbox[0]))
            radius = _strict_r95(config, width)
            hit = any(abs(x1 - peak) <= radius for peak in peak_values)
            residual_count += 1
            hit_count += int(hit)
            details.append(
                {
                    "desc": desc,
                    "gt_idx": int(obj["gt_idx"]),
                    "gt_x1": x1,
                    "bbox_width": width,
                    "strict_r95_radius": radius,
                    "strict_r95_hit": hit,
                    "x1_top1_bin": int(metrics.top1_bin),
                    "x1_target_rank": int(metrics.rank_gt),
                    "p_gt_cond": float(metrics.p_gt_cond),
                    "coord_vocab_mass": float(metrics.coord_vocab_mass),
                    "merged_peak_count": int(peaks["merged_peak_count"]),
                    "merged_peaks": peaks["merged_peaks"],
                }
            )

    return {
        "strict_r95_x1_hit_rate": 0.0 if residual_count == 0 else hit_count / residual_count,
        "strict_r95_x1_residual_count": residual_count,
        "strict_r95_x1_hit_count": hit_count,
        "x1_probe_details": details,
    }


def _x1_distribution_and_peaks(
    config: A32Config,
    *,
    model_handle: Any,
    logits: Any,
    residual_same: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    import torch
    from src.analysis.candidate_field_cardinality_tomography.x1_candidate_field import (
        extract_x1_peaks,
    )
    from src.analysis.hard_ce_coord_logit_locality import (
        distribution_metrics_from_logits,
        resolve_coord_token_ids,
    )

    coord_vocab = resolve_coord_token_ids(model_handle.tokenizer)
    gt_values = [int(_bbox_xyxy(obj)[0]) for obj in residual_same]
    target = gt_values[0] if gt_values else 0
    metrics = distribution_metrics_from_logits(
        logits=logits,
        coord_token_ids=coord_vocab.coord_token_ids,
        gt_bin=target,
        top_k=int(config.peak.raw_topk_k),
    )
    coord_index = torch.tensor(
        [int(token_id) for token_id in coord_vocab.coord_token_ids],
        dtype=torch.long,
        device=logits.device,
    )
    coord_logits = logits.index_select(0, coord_index).float()
    cond = torch.softmax(coord_logits, dim=0).detach().cpu().tolist()
    probs = {index: float(value) for index, value in enumerate(cond)}
    peaks = extract_x1_peaks(
        probs,
        gt_x1_values=gt_values,
        merge_radius=int(config.peak.primary_merge_radius),
        absolute_mass_floor=float(config.peak.absolute_mass_floor),
        relative_floor=float(config.peak.relative_floor),
        gt_x1_neighborhood_radius=int(config.peak.gt_x1_neighborhood_radius),
    )
    return metrics, peaks


def _score_suffix_mean_logprob(
    *,
    model_handle: Any,
    image_path: Path,
    assistant_prefix: str,
    suffix: str,
) -> float:
    import torch

    prefix_ids = _input_ids(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=assistant_prefix,
    )
    full_ids, logits = _input_ids_and_logits(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=assistant_prefix + suffix,
    )
    suffix_start = int(prefix_ids.shape[-1])
    _assert_prefix_token_alignment(prefix_ids, full_ids, suffix_start=suffix_start)
    values: list[float] = []
    for pos in range(suffix_start, int(full_ids.shape[-1])):
        pred_pos = pos - 1
        if pred_pos < 0:
            continue
        token_id = int(full_ids[0, pos])
        value = torch.log_softmax(logits[0, pred_pos], dim=-1)[token_id]
        values.append(float(value.detach().cpu()))
    return 0.0 if not values else sum(values) / len(values)


def _score_chat_eos_at_boundary(
    *,
    model_handle: Any,
    image_path: Path,
    assistant_prefix: str,
) -> dict[str, Any]:
    import torch
    from src.common.detection_sequence import IM_END_TOKEN
    from src.common.qwen_generation import resolve_qwen_chat_generation_token_ids

    logits = _last_logits(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=assistant_prefix,
    )
    token_ids = resolve_qwen_chat_generation_token_ids(model_handle.tokenizer)
    probs = torch.log_softmax(logits, dim=-1)
    return {
        "score": float(probs[int(token_ids.eos_token_id)].detach().cpu()),
        "token_id": int(token_ids.eos_token_id),
        "token_text": IM_END_TOKEN,
    }


def _last_logits(
    *,
    model_handle: Any,
    image_path: Path,
    assistant_text: str,
    expected_context_suffix: str | None = None,
) -> Any:
    input_ids, logits = _input_ids_and_logits(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=assistant_text,
        expected_context_suffix=expected_context_suffix,
    )
    del input_ids
    return logits[0, -1]


def _input_ids(model_handle: Any, image_path: Path, assistant_text: str) -> Any:
    return _processor_inputs(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=assistant_text,
    )["input_ids"]


def _input_ids_and_logits(
    *,
    model_handle: Any,
    image_path: Path,
    assistant_text: str,
    expected_context_suffix: str | None = None,
) -> tuple[Any, Any]:
    import torch

    inputs = _processor_inputs(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=assistant_text,
        expected_context_suffix=expected_context_suffix,
    )
    device = next(model_handle.model.parameters()).device
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model_handle.model(**inputs, use_cache=False)
    return inputs["input_ids"], outputs.logits


def _processor_inputs(
    *,
    model_handle: Any,
    image_path: Path,
    assistant_text: str,
    expected_context_suffix: str | None = None,
) -> Mapping[str, Any]:
    from PIL import Image
    from src.common.detection_chat import build_detection_chat_messages
    from src.common.qwen_generation import call_processor_with_qwen_geometry
    from src.config.prompts import get_template_prompts

    system_prompt, user_prompt = get_template_prompts(
        ordering=READOUT_PROMPT_ORDERING,
        coord_mode="coord_tokens",
        prompt_variant="coco_80",
        object_field_order="desc_first",
        bbox_format="xyxy",
        detection_sequence_format="compact_full",
        row_separator="none",
    )
    messages = build_detection_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[str(image_path)],
        assistant_text=assistant_text,
    )
    full_text = _render_continuation_chat_text(model_handle.processor, messages)
    _assert_rendered_continuation_context(
        full_text=full_text,
        assistant_text=assistant_text,
        expected_context_suffix=expected_context_suffix,
    )
    image = Image.open(image_path).convert("RGB")
    return call_processor_with_qwen_geometry(
        model_handle.processor,
        text=[full_text],
        images=[image],
        return_tensors="pt",
        padding=False,
    )


def _render_continuation_chat_text(processor: Any, messages: Sequence[Mapping[str, Any]]) -> str:
    try:
        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "processor.apply_chat_template must support continue_final_message=True "
            "for A3.2 prefix-state readout"
        ) from exc
    if not isinstance(full_text, str):
        raise TypeError("processor.apply_chat_template(..., tokenize=False) must return str")
    return full_text


def _assert_rendered_continuation_context(
    *,
    full_text: str,
    assistant_text: str,
    expected_context_suffix: str | None,
) -> None:
    rendered_tail = full_text.rstrip()
    if assistant_text and not rendered_tail.endswith(assistant_text):
        raise RuntimeError("rendered chat template does not preserve assistant continuation tail")
    if expected_context_suffix is not None and not rendered_tail.endswith(expected_context_suffix):
        raise RuntimeError(
            "rendered chat template does not end at expected continuation suffix "
            f"{expected_context_suffix!r}"
        )


def _assert_prefix_token_alignment(prefix_ids: Any, full_ids: Any, *, suffix_start: int) -> None:
    import torch

    if int(prefix_ids.shape[-1]) != int(suffix_start):
        raise RuntimeError("suffix_start must equal prefix token length")
    if int(full_ids.shape[-1]) < int(prefix_ids.shape[-1]):
        raise RuntimeError("full context is shorter than prefix context")
    prefix_row = prefix_ids[0].detach().cpu()
    full_prefix_row = full_ids[0, : int(suffix_start)].detach().cpu()
    if not torch.equal(prefix_row, full_prefix_row):
        raise RuntimeError("prefix tokens changed after appending scored suffix")


def _candidate_descs_with_roles(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("candidate_descs_with_roles") or []
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            desc = str(entry.get("desc") or "").strip()
            if not desc:
                continue
            roles = entry.get("roles", [])
            rows.append(
                {
                    "desc": desc,
                    "roles": [str(role) for role in roles]
                    if isinstance(roles, Sequence) and not isinstance(roles, str)
                    else [str(roles)],
                }
            )
    if rows:
        return rows
    return [
        {"desc": str(desc), "roles": ["other_gt_desc"]}
        for desc in state.get("candidate_descs", ())
    ]


def _forced_desc_pre_x1_suffix(desc: str) -> str:
    from src.analysis.prefix_state_transition_tomography.prefix_rendering import (
        render_forced_desc_pre_x1_assistant_text,
    )

    return render_forced_desc_pre_x1_assistant_text([], desc)


def _resolve_image_path(config: A32Config, state: Mapping[str, Any]) -> Path:
    image_ref = state.get("image_path")
    if image_ref is None or str(image_ref).strip() == "":
        raise FileNotFoundError(f"prefix state has no image_path: {state.get('prefix_state_id')}")
    path = Path(str(image_ref))
    candidates = [path] if path.is_absolute() else [config.image_root / path, config.image_root / "images" / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"image for prefix state was not found: {rendered}")


def _load_model_for_checkpoint(config: A32Config, *, checkpoint_path: Path) -> Any:
    from src.analysis.hard_ce_coord_logit_locality import (
        StudyConfig,
        StudyExecutionConfig,
        StudyModelConfig,
        StudyPaths,
        load_model_handle,
    )

    study_config = StudyConfig(
        paths=StudyPaths(
            checkpoint=Path(checkpoint_path),
            resolved_config=config.artifact_root / "resolved_config.yaml",
            source_config=None,
            dataset_jsonl=config.val_jsonl,
            image_root=config.image_root,
            artifact_root=config.artifact_root,
            self_rollout_root=None,
            self_rollout_regen_config=None,
        ),
        model=StudyModelConfig(
            prompt_variant="coco_80",
            object_field_order="desc_first",
            bbox_format="xyxy",
            detection_sequence_format="compact_full",
            object_ordering="sorted",
            device="auto",
            attn_implementation="auto",
            torch_dtype="bfloat16",
        ),
        execution=StudyExecutionConfig(sample_limit=1, batch_size=1),
    )
    return load_model_handle(study_config)


def _checkpoint_fingerprint(model_handle: Any) -> str:
    resolved = getattr(model_handle, "resolved_checkpoint", None)
    payload = {
        "checkpoint_mode": getattr(resolved, "checkpoint_mode", None),
        "requested_model_checkpoint": getattr(resolved, "requested_model_checkpoint", None),
        "requested_adapter_checkpoint": getattr(resolved, "requested_adapter_checkpoint", None),
        "resolved_base_model_checkpoint": getattr(resolved, "resolved_base_model_checkpoint", None),
        "resolved_adapter_checkpoint": getattr(resolved, "resolved_adapter_checkpoint", None),
    }
    return "checkpoint:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _strict_r95(config: A32Config, axis_len: int) -> int:
    return math.floor(
        min(
            int(config.fn_probe.strict_r95_cap_bins),
            float(config.fn_probe.strict_r95_axis_fraction) * int(axis_len),
        )
    )


def _bbox_xyxy(obj: Mapping[str, Any]) -> list[int]:
    bbox = obj.get("bbox_xyxy")
    if not isinstance(bbox, Sequence) or isinstance(bbox, str) or len(bbox) != 4:
        raise ValueError(f"object is missing bbox_xyxy with four coordinates: {obj}")
    return [int(value) for value in bbox]


def _gpu_id(gpu_id: str | None) -> str:
    if gpu_id is not None and str(gpu_id).strip():
        return str(gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        return visible.strip()
    return "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row must be an object: {path}")
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps(_json_safe(row, f"rows[{index}]"), allow_nan=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(_json_safe(payload, "payload"), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _ensure_can_write(path: Path, *, allow_overwrite: bool) -> None:
    if Path(path).exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def _boundary_summary_for_prefix(
    boundary_summaries_by_role: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    role: str,
    prefix_state_id: str,
) -> Mapping[str, Any]:
    role_summaries = boundary_summaries_by_role.get(role)
    if role_summaries is None:
        raise ValueError(f"missing boundary summaries for checkpoint role: {role}")
    summary = role_summaries.get(prefix_state_id)
    if summary is None:
        raise ValueError(
            f"missing boundary summary for checkpoint role {role}: {prefix_state_id}"
        )
    return summary


def _json_safe(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, str | int):
        return value
    return str(value)


__all__ = [
    "READOUT_BEHAVIOR",
    "REAL_PREFIX_RUNTIME_KIND",
    "build_real_paired_readout_rows",
    "build_real_shard_summary_row",
    "build_mocked_paired_readout_rows",
    "build_shard_manifest_row",
    "checkpoint_roles_from_config",
    "run_real_paired_checkpoint_probe",
]
