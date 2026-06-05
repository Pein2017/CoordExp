from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .boundary_scoring import rank_desc_scores, summarize_boundary_alignment
from .config import CheckpointConfig, PeakConfig, PrefixStateTransitionConfig
from .jsonl import read_jsonl, write_jsonl
from .prefix_rendering import (
    render_boundary_assistant_text,
    render_forced_desc_pre_x1_assistant_text,
)
from .x1_readout import partition_x1_peaks, summarize_x1_partitions


LEGACY_CHECKPOINT_ROLES = ("et_rmp_ce", "pure_ce")
CHECKPOINT_ROLES = LEGACY_CHECKPOINT_ROLES
PRE_X1_CONTEXT_SUFFIX = "<|box_start|>"


def rows_for_shard(
    rows: Sequence[Mapping[str, Any]],
    shard_id: int,
    num_shards: int,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for idx, row in enumerate(rows):
        planned = row.get("shard_id", row.get("planned_shard_id"))
        if planned is None:
            planned = idx % num_shards
        if int(planned) == int(shard_id):
            selected.append(row)
    return selected


def run_paired_checkpoint_probe(
    *,
    config: PrefixStateTransitionConfig,
    shard_id: int,
) -> dict[str, Any]:
    summary_path = config.artifact_root / "prefix_state_index_summary.json"
    if not summary_path.exists():
        return {"stage": "paired_checkpoint_probe", "status": "blocked", "reason": "missing_prefix_state_index_summary"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("launch_eligible") is not True:
        return {
            "stage": "paired_checkpoint_probe",
            "status": "blocked",
            "reason": "prefix_state_index_not_launch_eligible",
            "failed_launch_gates": summary.get("failed_launch_gates"),
        }
    sampled_rows = read_jsonl(config.artifact_root / "prefix_state_sampled_rows.jsonl")
    shard_rows = rows_for_shard(sampled_rows, shard_id, config.sampling.num_shards)
    checkpoint_roles = _checkpoint_roles(config)
    shard_root = config.artifact_root / "shards" / f"shard_{shard_id:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    progress_path = shard_root / "paired_probe_progress.json"
    _write_json(
        progress_path,
        {
            "status": "running",
            "shard_id": shard_id,
            "planned_prefix_state_rows": len(shard_rows),
            "checkpoint_roles": checkpoint_roles,
            "started_at": started_at,
            "updated_at": started_at,
            "completed_prefix_state_role_pairs": 0,
            "boundary_score_rows": 0,
            "forced_x1_rows": 0,
        },
    )
    boundary_rows: list[dict[str, Any]] = []
    forced_rows: list[dict[str, Any]] = []
    completed_pairs = 0
    for role in checkpoint_roles:
        handle = _load_model_handle(
            checkpoint=config.checkpoints[role],
            config=config,
            role=role,
        )
        try:
            for row in shard_rows:
                b_rows, x_rows = _probe_prefix_state(
                    state=row,
                    checkpoint_role=role,
                    checkpoint=config.checkpoints[role],
                    model_handle=handle,
                    peak=config.peak,
                )
                boundary_rows.extend(b_rows)
                forced_rows.extend(x_rows)
                completed_pairs += 1
                if completed_pairs % 25 == 0:
                    _write_json(
                        progress_path,
                        {
                            "status": "running",
                            "shard_id": shard_id,
                            "planned_prefix_state_rows": len(shard_rows),
                            "checkpoint_roles": checkpoint_roles,
                            "started_at": started_at,
                            "updated_at": time.time(),
                            "completed_prefix_state_role_pairs": completed_pairs,
                            "boundary_score_rows": len(boundary_rows),
                            "forced_x1_rows": len(forced_rows),
                        },
                    )
        finally:
            _release_model_handle(handle)
    boundary_count = _write_jsonl_atomic(shard_root / "boundary_score_rows.jsonl", boundary_rows)
    forced_count = _write_jsonl_atomic(shard_root / "forced_x1_rows.jsonl", forced_rows)
    shard_summary = {
        "status": "ok",
        "shard_id": shard_id,
        "prefix_state_rows": len(shard_rows),
        "boundary_score_rows": boundary_count,
        "forced_x1_rows": forced_count,
        "checkpoint_roles": checkpoint_roles,
        "completed_prefix_state_role_pairs": completed_pairs,
        "started_at": started_at,
        "finished_at": time.time(),
    }
    _write_json(shard_root / "shard_summary.json", shard_summary)
    _write_json(shard_root / "shard_manifest.json", shard_summary)
    _write_json(progress_path, {**shard_summary, "status": "complete"})
    return shard_summary


def _checkpoint_roles(config: PrefixStateTransitionConfig) -> list[str]:
    return [str(role) for role in config.checkpoints]


def validate_sampled_image_paths(*, config: PrefixStateTransitionConfig) -> dict[str, Any]:
    sample_path = config.artifact_root / "prefix_state_sampled_rows.jsonl"
    if not sample_path.exists():
        return {"status": "missing_sampled_rows", "checked_rows": 0, "missing_rows": 0, "examples": []}
    rows = read_jsonl(sample_path)
    examples: list[dict[str, Any]] = []
    missing_rows = 0
    for row in rows:
        try:
            _resolve_image_path(row, image_root=config.image_root, must_exist=True)
        except FileNotFoundError as exc:
            missing_rows += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "prefix_state_id": row.get("prefix_state_id"),
                        "image_path": row.get("image_path"),
                        "error": str(exc),
                    }
                )
    return {
        "status": "ok" if missing_rows == 0 else "missing_images",
        "checked_rows": len(rows),
        "missing_rows": missing_rows,
        "image_root": None if config.image_root is None else str(config.image_root),
        "examples": examples,
    }


def _probe_prefix_state(
    *,
    state: Mapping[str, Any],
    checkpoint_role: str,
    model_handle: Any,
    peak: PeakConfig,
    checkpoint: CheckpointConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_objects = list(state.get("gt_objects") or [])
    emitted_order = [int(idx) for idx in state.get("emitted_gt_indices") or []]
    emitted_ids = set(emitted_order)
    residual_ids = {int(idx) for idx in state.get("residual_gt_indices") or []}
    by_gt_idx = {int(obj["gt_idx"]): obj for obj in gt_objects}
    emitted_rows = [by_gt_idx[idx] for idx in emitted_order if idx in by_gt_idx]
    row_separator = _checkpoint_row_separator(checkpoint)
    boundary_text = render_boundary_assistant_text(
        emitted_rows,
        row_separator=row_separator,
    )
    image_path = _resolve_image_path(state, must_exist=True)
    candidate_descs = list(dict.fromkeys(str(desc) for desc in state.get("all_descs") or []))
    boundary_scores = _score_boundary(
        model_handle=model_handle,
        image_path=image_path,
        state=state,
        boundary_assistant_text=boundary_text,
        candidate_descs=candidate_descs,
        row_separator=row_separator,
    )
    forced_rows: list[dict[str, Any]] = []
    for desc in _probe_descs(state):
        forced_text = render_forced_desc_pre_x1_assistant_text(
            emitted_rows,
            desc,
            row_separator=row_separator,
        )
        emitted_same = [
            obj for obj in gt_objects if int(obj["gt_idx"]) in emitted_ids and str(obj["desc"]) == desc
        ]
        residual_same = [
            obj for obj in gt_objects if int(obj["gt_idx"]) in residual_ids and str(obj["desc"]) == desc
        ]
        forced_rows.append(
            _score_forced_x1(
                model_handle=model_handle,
                image_path=image_path,
                state=state,
                checkpoint_role=checkpoint_role,
                checkpoint=checkpoint,
                desc=desc,
                forced_assistant_text=forced_text,
                emitted_same=emitted_same,
                residual_same=residual_same,
                peak=peak,
            )
        )
    boundary_rows = [
        {
            **_base_readout_row(state, checkpoint_role=checkpoint_role, probe_desc=row["desc"]),
            **_checkpoint_metadata(checkpoint),
            "readout_type": "boundary_full_desc_span",
            "probe_desc_role": _probe_desc_role(state, str(row["desc"])),
            "boundary_score_policy_id": "boundary_desc_span_mean_v1",
            "boundary_alignment": boundary_scores["summary"]["boundary_alignment"],
            "desc_span_score": row["score"],
            "desc_span_rank": row["rank"],
            "desc_role": row.get("role"),
            "eos_score": boundary_scores["summary"]["eos_score"],
            "best_desc": boundary_scores["summary"]["best_desc"],
            "best_role": boundary_scores["summary"]["best_role"],
            "margin_best_residual_vs_eos": boundary_scores["summary"]["margin_best_residual_vs_eos"],
        }
        for row in boundary_scores["ranked_desc_scores"]
    ]
    return boundary_rows, forced_rows


def _score_boundary(
    *,
    model_handle: Any,
    image_path: Path,
    state: Mapping[str, Any],
    boundary_assistant_text: str,
    candidate_descs: Sequence[str],
    row_separator: str = "none",
) -> dict[str, Any]:
    scores = []
    for desc in candidate_descs:
        suffix = render_forced_desc_pre_x1_assistant_text(
            [],
            desc,
            row_separator=row_separator,
        )
        if boundary_assistant_text:
            suffix = _row_separator_text(row_separator) + suffix
        score = _score_suffix_mean_logprob(
            model_handle=model_handle,
            image_path=image_path,
            assistant_prefix=boundary_assistant_text,
            suffix=suffix,
        )
        scores.append({"desc": desc, "role": _desc_role(state, desc), "score": score})
    eos_score = _score_eos_at_boundary(
        model_handle=model_handle,
        image_path=image_path,
        assistant_prefix=boundary_assistant_text,
    )
    summary = summarize_boundary_alignment(scores, eos_score=eos_score)
    return {"ranked_desc_scores": rank_desc_scores(scores), "summary": summary}


def _checkpoint_row_separator(checkpoint: CheckpointConfig | None) -> str:
    if checkpoint is None:
        return "none"
    if checkpoint.template_contract_id == "compact_full_newline_native_v1":
        return "newline"
    if checkpoint.template_contract_id in {"", "compact_full_no_newline_native_v1"}:
        return "none"
    raise ValueError(
        f"unsupported template_contract_id for prefix rendering: "
        f"{checkpoint.template_contract_id}"
    )


def _row_separator_text(row_separator: str) -> str:
    if row_separator == "none":
        return ""
    if row_separator == "newline":
        return "\n"
    raise ValueError(f"unsupported row_separator: {row_separator}")


def _score_forced_x1(
    *,
    model_handle: Any,
    image_path: Path,
    state: Mapping[str, Any],
    checkpoint_role: str,
    checkpoint: CheckpointConfig | None,
    desc: str,
    forced_assistant_text: str,
    emitted_same: Sequence[Mapping[str, Any]],
    residual_same: Sequence[Mapping[str, Any]],
    peak: PeakConfig,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from src.analysis.candidate_field_cardinality_tomography.x1_candidate_field import extract_x1_peaks
    from src.analysis.hard_ce_coord_logit_locality import (
        distribution_metrics_from_logits,
        resolve_coord_token_ids,
    )

    logits = _last_logits(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=forced_assistant_text,
        expected_context_suffix=PRE_X1_CONTEXT_SUFFIX,
    )
    coord_vocab = resolve_coord_token_ids(model_handle.tokenizer)
    target_object = residual_same[0] if residual_same else (emitted_same[0] if emitted_same else None)
    target_x1 = int(target_object["bbox_xyxy"][0]) if target_object else 0
    metrics = distribution_metrics_from_logits(
        logits=logits,
        coord_token_ids=coord_vocab.coord_token_ids,
        gt_bin=target_x1,
        radii=(16, 24, 32),
        top_k=peak.raw_topk_k,
    )
    probs = {idx: float(value) for idx, value in enumerate(np.asarray(metrics.conditional_probs))}
    gt_x1_values = [int(obj["bbox_xyxy"][0]) for obj in [*emitted_same, *residual_same]]
    peaks = extract_x1_peaks(
        probs,
        gt_x1_values=gt_x1_values,
        merge_radius=peak.primary_merge_radius,
        absolute_mass_floor=peak.absolute_mass_floor,
        relative_floor=peak.relative_floor,
        gt_x1_neighborhood_radius=peak.gt_x1_neighborhood_radius,
    )
    partitioned = partition_x1_peaks(
        peaks["merged_peaks"],
        emitted_same,
        residual_same,
        radius=peak.gt_x1_neighborhood_radius,
    )
    summary = summarize_x1_partitions(partitioned, residual_same)
    del torch
    return {
        **_base_readout_row(state, checkpoint_role=checkpoint_role, probe_desc=desc),
        **_checkpoint_metadata(checkpoint),
        "readout_type": "forced_desc_pre_x1",
        "probe_desc_role": _probe_desc_role(state, desc),
        "coord_vocab_mass": float(metrics.coord_vocab_mass),
        "target_x1_bin": target_x1,
        "x1_top1_bin": int(metrics.top1_bin),
        "x1_target_rank": int(metrics.rank_gt),
        "p_gt_cond": float(metrics.p_gt_cond),
        "top_bins": metrics.top_bins,
        "merged_peak_count": peaks["merged_peak_count"],
        "merged_peaks": peaks["merged_peaks"],
        "partitioned_peaks": partitioned,
        **summary,
    }


def _score_suffix_mean_logprob(
    *,
    model_handle: Any,
    image_path: Path,
    assistant_prefix: str,
    suffix: str,
) -> float:
    import torch

    prefix_ids = _input_ids(model_handle=model_handle, image_path=image_path, assistant_text=assistant_prefix)
    full_ids, logits = _input_ids_and_logits(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=assistant_prefix + suffix,
    )
    suffix_start = int(prefix_ids.shape[-1])
    _assert_prefix_token_alignment(prefix_ids, full_ids, suffix_start=suffix_start)
    logprobs = []
    for pos in range(suffix_start, int(full_ids.shape[-1])):
        token_id = int(full_ids[0, pos])
        pred_pos = pos - 1
        if pred_pos < 0:
            continue
        value = torch.log_softmax(logits[0, pred_pos], dim=-1)[token_id]
        logprobs.append(float(value.detach().cpu()))
    if not logprobs:
        return 0.0
    return sum(logprobs) / len(logprobs)


def _score_eos_at_boundary(*, model_handle: Any, image_path: Path, assistant_prefix: str) -> float:
    import torch

    logits = _last_logits(model_handle=model_handle, image_path=image_path, assistant_text=assistant_prefix)
    eos_ids = [model_handle.tokenizer.eos_token_id]
    eos_ids.extend(
        token_id
        for token_id in (getattr(model_handle.tokenizer, "pad_token_id", None),)
        if token_id is not None and token_id not in eos_ids
    )
    probs = torch.log_softmax(logits, dim=-1)
    return max(float(probs[int(token_id)].detach().cpu()) for token_id in eos_ids if token_id is not None)


def _last_logits(
    *,
    model_handle: Any,
    image_path: Path,
    assistant_text: str,
    expected_context_suffix: str | None = None,
) -> Any:
    ids, logits = _input_ids_and_logits(
        model_handle=model_handle,
        image_path=image_path,
        assistant_text=assistant_text,
        expected_context_suffix=expected_context_suffix,
    )
    del ids
    return logits[0, -1]


def _input_ids(model_handle: Any, image_path: Path, assistant_text: str) -> Any:
    inputs = _processor_inputs(model_handle=model_handle, image_path=image_path, assistant_text=assistant_text)
    return inputs["input_ids"]


def _input_ids_and_logits(
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
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model_handle.model(**inputs, use_cache=False)
    return inputs["input_ids"], outputs.logits


def _processor_inputs(
    model_handle: Any,
    image_path: Path,
    assistant_text: str,
    expected_context_suffix: str | None = None,
) -> Mapping[str, Any]:
    from PIL import Image
    from src.common.detection_chat import build_detection_chat_messages
    from src.config.prompts import get_template_prompts

    system_prompt, user_prompt = get_template_prompts(
        ordering="sorted",
        coord_mode="coord_tokens",
        prompt_variant="coco_80",
        object_field_order="desc_first",
        bbox_format="xyxy",
        detection_sequence_format="compact_full",
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
    return model_handle.processor(text=[full_text], images=[image], return_tensors="pt", padding=False)


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
            "processor.apply_chat_template must support continue_final_message=True for "
            "prefix-state continuation probes"
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
            f"rendered chat template does not end at expected continuation suffix {expected_context_suffix!r}"
        )


def _assert_prefix_token_alignment(prefix_ids: Any, full_ids: Any, *, suffix_start: int) -> None:
    import torch

    if int(full_ids.shape[-1]) < suffix_start:
        raise RuntimeError("full tokenization is shorter than prefix tokenization")
    prefix_slice = prefix_ids[:, :suffix_start].detach().cpu()
    full_slice = full_ids[:, :suffix_start].detach().cpu()
    if not torch.equal(prefix_slice, full_slice):
        raise RuntimeError("prefix/full tokenizations diverged before scored suffix")


def _load_model_handle(
    *,
    checkpoint: CheckpointConfig,
    config: PrefixStateTransitionConfig,
    role: str,
) -> Any:
    from src.analysis.hard_ce_coord_logit_locality import (
        StudyConfig,
        StudyExecutionConfig,
        StudyModelConfig,
        StudyPaths,
        load_model_handle,
    )

    study_config = StudyConfig(
        paths=StudyPaths(
            checkpoint=checkpoint.checkpoint_path,
            resolved_config=config.artifact_root / f"{role}_resolved_config.json",
            source_config=None,
            dataset_jsonl=config.val_jsonl,
            image_root=config.image_root or Path("/"),
            artifact_root=config.artifact_root / "runtime" / role,
            self_rollout_root=None,
            self_rollout_regen_config=None,
        ),
        model=StudyModelConfig(object_ordering="sorted"),
        execution=StudyExecutionConfig(sample_limit=1, batch_size=1, top_k=config.peak.raw_topk_k),
    )
    return load_model_handle(study_config)


def _release_model_handle(handle: Any) -> None:
    try:
        del handle
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    inprogress = path.with_name(path.name + ".inprogress")
    count = write_jsonl(inprogress, rows)
    inprogress.replace(path)
    return count


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _base_readout_row(
    state: Mapping[str, Any],
    *,
    checkpoint_role: str,
    probe_desc: str,
) -> dict[str, Any]:
    return {
        "schema_version": state.get("schema_version"),
        "project_id": state.get("project_id"),
        "phase_id": state.get("phase_id"),
        "run_id": state.get("run_id"),
        "checkpoint_id": state.get("checkpoint_id"),
        "checkpoint_role": checkpoint_role,
        "split": state.get("split"),
        "source_dataset_jsonl": state.get("source_dataset_jsonl"),
        "source_line_idx": state.get("source_line_idx"),
        "image_id": state.get("image_id"),
        "image_path": state.get("image_path"),
        "prefix_state_id": state.get("prefix_state_id"),
        "transition_type": state.get("transition_type"),
        "prefix_condition": state.get("prefix_condition"),
        "prefix_depth": state.get("prefix_depth"),
        "prefix_order_policy_id": state.get("prefix_order_policy_id"),
        "emitted_gt_indices": state.get("emitted_gt_indices"),
        "residual_gt_indices": state.get("residual_gt_indices"),
        "emitted_descs": state.get("emitted_descs"),
        "residual_descs": state.get("residual_descs"),
        "probe_desc": probe_desc,
        "shard_id": state.get("shard_id"),
    }


def _resolve_image_path(
    state: Mapping[str, Any],
    *,
    image_root: Path | None = None,
    must_exist: bool = False,
) -> Path:
    image_path = Path(str(state["image_path"]))
    candidates: list[Path] = []
    if image_path.is_absolute():
        candidates.append(image_path)
    else:
        if image_root is not None:
            candidates.append(Path(image_root) / image_path)
        candidates.append(Path(str(state["source_dataset_jsonl"])).parent / image_path)
    if not must_exist:
        return candidates[0]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not resolve image path; candidates="
        + ", ".join(str(candidate) for candidate in candidates)
    )

def _probe_descs(state: Mapping[str, Any]) -> list[str]:
    values = [state.get("target_residual_desc"), state.get("hard_competitor_desc")]
    return [str(value) for value in dict.fromkeys(values) if value]


def _checkpoint_metadata(checkpoint: CheckpointConfig | None) -> dict[str, str]:
    if checkpoint is None:
        return {
            "objective_policy": "",
            "training_ordering": "",
            "template_contract_id": "",
            "comparison_group": "",
        }
    return {
        "objective_policy": checkpoint.objective_policy,
        "training_ordering": checkpoint.training_ordering,
        "template_contract_id": checkpoint.template_contract_id,
        "comparison_group": checkpoint.comparison_group,
    }


def _probe_desc_role(state: Mapping[str, Any], desc: str) -> str:
    if desc == state.get("target_residual_desc"):
        return "target_residual_desc"
    if desc == state.get("hard_competitor_desc"):
        return "hard_competitor_desc"
    return _desc_role(state, desc)


def _desc_role(state: Mapping[str, Any], desc: str) -> str:
    residual = set(str(value) for value in state.get("residual_descs") or [])
    emitted = set(str(value) for value in state.get("emitted_descs") or [])
    if desc in residual:
        return "residual"
    if desc in emitted:
        return "emitted"
    return "other"


__all__ = [
    "CHECKPOINT_ROLES",
    "PRE_X1_CONTEXT_SUFFIX",
    "rows_for_shard",
    "run_paired_checkpoint_probe",
    "validate_sampled_image_paths",
]
