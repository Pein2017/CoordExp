from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_jsonl, write_json
from .prefixes import render_pre_x1_prefix


@dataclass(frozen=True)
class X1ProbeRuntime:
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    max_cases: int | None = None
    shard_id: int | None = None
    raw_topk_k: int = 32
    primary_merge_radius: int = 24
    absolute_mass_floor: float = 0.002
    relative_floor: float = 0.10
    gt_x1_neighborhood_radius: int = 24
    progress_interval: int = 25


def extract_x1_peaks(
    probs: Mapping[int, float],
    *,
    gt_x1_values: Sequence[int],
    merge_radius: int = 24,
    absolute_mass_floor: float = 0.002,
    relative_floor: float = 0.10,
    gt_x1_neighborhood_radius: int = 24,
) -> dict[str, Any]:
    if not probs:
        return {"merged_peak_count": 0, "gt_instance_coverage_count": 0, "merged_peaks": []}
    top_mass = max(float(value) for value in probs.values())
    threshold = max(absolute_mass_floor, top_mass * relative_floor)
    local_peaks = []
    for coord, mass in sorted((int(k), float(v)) for k, v in probs.items()):
        if mass < threshold:
            continue
        left = float(probs.get(coord - 1, -1.0))
        right = float(probs.get(coord + 1, -1.0))
        if mass >= left and mass >= right:
            local_peaks.append({"x1": coord, "mass": mass})
    merged = _merge_peaks(local_peaks, merge_radius=merge_radius)
    covered = 0
    for gt in gt_x1_values:
        if any(abs(int(gt) - int(peak["x1"])) <= gt_x1_neighborhood_radius for peak in merged):
            covered += 1
    return {
        "merged_peak_count": len(merged),
        "gt_instance_coverage_count": covered,
        "merged_peaks": merged,
        "threshold": threshold,
    }


def materialize_x1_candidate_field_rows(
    *,
    artifact_root: Path,
    checkpoint_path: Path,
    runtime: X1ProbeRuntime | None = None,
) -> dict[str, Any]:
    """Run desc-conditioned pre-x1 logits for sampled probe-plan cases.

    This is a smoke/full analysis stage, not training. It loads checkpoint-3664
    and writes JSON-safe posterior summaries, leaving dense probability arrays
    out of the artifact contract for now.
    """

    runtime = runtime or X1ProbeRuntime()
    root = Path(artifact_root)
    case_rows = read_jsonl(root / "case_index.jsonl")
    probe_plan = read_jsonl(root / "probe_plan.jsonl")
    sampled_plan = [row for row in probe_plan if row.get("probe_sampled") is True]
    if runtime.shard_id is not None:
        sampled_plan = [
            row for row in sampled_plan if int(row.get("planned_shard_id", -1)) == int(runtime.shard_id)
        ]
    if runtime.max_cases is not None:
        sampled_plan = sampled_plan[: int(runtime.max_cases)]

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_case[str(row["case_id"])].append(row)

    output_root = root if runtime.shard_id is None else root / "shards" / f"shard_{runtime.shard_id:03d}"
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "x1_candidate_field_rows.jsonl"
    inprogress_path = output_root / "x1_candidate_field_rows.jsonl.inprogress"
    progress_path = output_root / "x1_candidate_field_progress.json"
    inprogress_path.write_text("", encoding="utf-8")
    started_at = time.time()
    _write_progress(
        progress_path,
        shard_id=runtime.shard_id,
        planned_cases=len(sampled_plan),
        emitted_rows=0,
        valid_rows=0,
        checkpoint_path=Path(checkpoint_path),
        started_at=started_at,
        last_probe_plan_row_id=None,
        status="running",
    )

    model_handle = _load_model_handle(
        checkpoint_path=Path(checkpoint_path),
        device=runtime.device,
        torch_dtype=runtime.torch_dtype,
        attn_implementation=runtime.attn_implementation,
    )
    emitted_rows = 0
    valid_rows = 0
    for plan_row in sampled_plan:
        case_id = str(plan_row["case_id"])
        members = by_case.get(case_id, [])
        if not members:
            continue
        reference = members[0]
        try:
            row = _probe_case_pre_x1(
                reference=reference,
                members=members,
                plan_row=plan_row,
                model_handle=model_handle,
                checkpoint_path=Path(checkpoint_path),
                runtime=runtime,
            )
        except Exception as exc:
            row = _base_probe_row(reference=reference, plan_row=plan_row, checkpoint_path=Path(checkpoint_path))
            row.update({"probe_status": "error", "error": str(exc)})
        with inprogress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        emitted_rows += 1
        valid_rows += 1 if row.get("probe_status") == "ok" else 0
        if emitted_rows % max(int(runtime.progress_interval), 1) == 0:
            _write_progress(
                progress_path,
                shard_id=runtime.shard_id,
                planned_cases=len(sampled_plan),
                emitted_rows=emitted_rows,
                valid_rows=valid_rows,
                checkpoint_path=Path(checkpoint_path),
                started_at=started_at,
                last_probe_plan_row_id=str(plan_row.get("probe_plan_row_id")),
                status="running",
            )

    inprogress_path.replace(rows_path)
    summary = {
        "stage": "x1_candidate_field",
        "shard_id": runtime.shard_id,
        "planned_cases": len(sampled_plan),
        "emitted_rows": emitted_rows,
        "valid_rows": valid_rows,
        "checkpoint_path": str(checkpoint_path),
    }
    write_json(output_root / "x1_candidate_field_summary.json", summary)
    _write_progress(
        progress_path,
        shard_id=runtime.shard_id,
        planned_cases=len(sampled_plan),
        emitted_rows=emitted_rows,
        valid_rows=valid_rows,
        checkpoint_path=Path(checkpoint_path),
        started_at=started_at,
        last_probe_plan_row_id=None if emitted_rows == 0 else "final",
        status="complete",
    )
    if runtime.shard_id is not None:
        write_json(
            output_root / "shard_manifest.json",
            {
                "shard_id": runtime.shard_id,
                "stage_status": "ok",
                "case_count": len(sampled_plan),
                "output_row_counts": {"x1_candidate_field_rows.jsonl": emitted_rows},
                "input_case_ids": [str(row["case_id"]) for row in sampled_plan],
            },
        )
    return summary


def _write_progress(
    path: Path,
    *,
    shard_id: int | None,
    planned_cases: int,
    emitted_rows: int,
    valid_rows: int,
    checkpoint_path: Path,
    started_at: float,
    last_probe_plan_row_id: str | None,
    status: str,
) -> None:
    elapsed_seconds = max(time.time() - started_at, 0.0)
    write_json(
        path,
        {
            "stage": "x1_candidate_field",
            "status": status,
            "shard_id": shard_id,
            "planned_cases": planned_cases,
            "emitted_rows": emitted_rows,
            "valid_rows": valid_rows,
            "remaining_cases": max(planned_cases - emitted_rows, 0),
            "progress_fraction": emitted_rows / planned_cases if planned_cases else 1.0,
            "elapsed_seconds": elapsed_seconds,
            "rows_per_second": emitted_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "last_probe_plan_row_id": last_probe_plan_row_id,
            "checkpoint_path": str(checkpoint_path),
        },
    )


def detect_projection_collision(
    boxes: Sequence[Sequence[int]],
    *,
    primary_merge_radius: int = 24,
    y_separation_threshold: int = 48,
    center_distance_threshold: int = 64,
) -> bool:
    for idx, first in enumerate(boxes):
        for second in boxes[idx + 1 :]:
            if len(first) != 4 or len(second) != 4:
                continue
            if abs(int(first[0]) - int(second[0])) >= primary_merge_radius:
                continue
            y_sep = abs(int(first[1]) - int(second[1]))
            c1 = ((int(first[0]) + int(first[2])) / 2.0, (int(first[1]) + int(first[3])) / 2.0)
            c2 = ((int(second[0]) + int(second[2])) / 2.0, (int(second[1]) + int(second[3])) / 2.0)
            center_dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
            if y_sep >= y_separation_threshold or center_dist >= center_distance_threshold:
                return True
    return False


def _merge_peaks(peaks: Sequence[Mapping[str, float]], *, merge_radius: int) -> list[dict[str, float]]:
    merged: list[dict[str, float]] = []
    for peak in sorted(peaks, key=lambda item: (float(item["x1"]), -float(item["mass"]))):
        if not merged or abs(float(peak["x1"]) - float(merged[-1]["x1"])) >= merge_radius:
            merged.append({"x1": float(peak["x1"]), "mass": float(peak["mass"])})
            continue
        if float(peak["mass"]) > float(merged[-1]["mass"]):
            merged[-1] = {"x1": float(peak["x1"]), "mass": float(peak["mass"])}
    return merged


def _probe_case_pre_x1(
    *,
    reference: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    plan_row: Mapping[str, Any],
    model_handle: Any,
    checkpoint_path: Path,
    runtime: X1ProbeRuntime,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    from src.analysis.hard_ce_coord_logit_locality import distribution_metrics_from_logits
    from src.common.detection_chat import build_detection_chat_messages
    from src.config.prompts import get_template_prompts
    from src.tokens.coord.codec import get_coord_token_ids

    desc = str(reference["desc_text_canonical"])
    prefix = render_pre_x1_prefix(desc)
    system_prompt, user_prompt = get_template_prompts(
        ordering="sorted",
        coord_mode="coord_tokens",
        prompt_variant="coco_80",
        object_field_order="desc_first",
        bbox_format="xyxy",
        detection_sequence_format="compact_full",
    )
    image_path = _resolve_image_path(reference)
    messages = build_detection_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[str(image_path)],
        assistant_text=str(prefix["prompt_text"]),
    )
    processor = model_handle["processor"]
    try:
        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except TypeError:
        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=False)
    device = _model_device(model_handle["model"])
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model_handle["model"](**inputs, use_cache=False)
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("model forward did not return logits")
    coord_ids_raw = get_coord_token_ids(model_handle["tokenizer"], validate=True)
    coord_ids = (
        coord_ids_raw.coord_token_ids
        if hasattr(coord_ids_raw, "coord_token_ids")
        else list(coord_ids_raw)
    )
    gt_x1_values = [int(row["bbox_xyxy"][0]) for row in members if row.get("bbox_xyxy")]
    target_x1 = gt_x1_values[0] if gt_x1_values else 0
    metrics = distribution_metrics_from_logits(
        logits=logits[0, -1],
        coord_token_ids=coord_ids,
        gt_bin=target_x1,
        radii=(16, 24, 32),
        top_k=runtime.raw_topk_k,
    )
    probs = {idx: float(value) for idx, value in enumerate(np.asarray(metrics.conditional_probs))}
    peak = extract_x1_peaks(
        probs,
        gt_x1_values=gt_x1_values,
        merge_radius=runtime.primary_merge_radius,
        absolute_mass_floor=runtime.absolute_mass_floor,
        relative_floor=runtime.relative_floor,
        gt_x1_neighborhood_radius=runtime.gt_x1_neighborhood_radius,
    )
    row = _base_probe_row(reference=reference, plan_row=plan_row, checkpoint_path=checkpoint_path)
    row.update(
        {
            "probe_status": "ok",
            "posterior_snapshot_id": f"{plan_row['probe_plan_row_id']}:pre_x1",
            "prefix_condition": "teacher_set_empty_prefix",
            "prompt_instance_id": f"{plan_row['probe_plan_row_id']}:teacher_empty:{desc}",
            "prefix_row_count": prefix["prefix_row_count"],
            "prefix_text_sha256": prefix["prefix_text_sha256"],
            "prompt_text_sha256": prefix["prompt_text_sha256"],
            "prompt_template_id": "coco_80:compact_full:desc_first:xyxy",
            "object_field_order": "desc_first",
            "bbox_format": "xyxy",
            "coord_surface": "norm1000_coord_tokens",
            "normalization": "lower_strip_collapse_ws_v1",
            "coord_vocab_mass": float(metrics.coord_vocab_mass),
            "noncoord_vocab_mass": float(max(0.0, 1.0 - metrics.coord_vocab_mass)),
            "target_x1_bin": target_x1,
            "same_desc_gt_x1_values": gt_x1_values,
            "same_desc_gt_count_annotated": len(gt_x1_values),
            "x1_top1_bin": int(metrics.top1_bin),
            "x1_target_rank": int(metrics.rank_gt),
            "p_gt_cond": float(metrics.p_gt_cond),
            "p_gt_full": float(metrics.p_gt_full),
            "top_bins": metrics.top_bins,
            "peak_policy": {
                "absolute_mass_floor": runtime.absolute_mass_floor,
                "relative_floor": runtime.relative_floor,
                "primary_merge_radius": runtime.primary_merge_radius,
                "gt_x1_neighborhood_radius": runtime.gt_x1_neighborhood_radius,
                "raw_topk_k": runtime.raw_topk_k,
            },
            "merged_peak_count": peak["merged_peak_count"],
            "gt_instance_coverage_count": peak["gt_instance_coverage_count"],
            "merged_peaks": peak["merged_peaks"],
            "x1_projection_collision": detect_projection_collision(
                [row["bbox_xyxy"] for row in members if row.get("bbox_xyxy")]
            ),
        }
    )
    return row


def _base_probe_row(
    *,
    reference: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": reference["schema_version"],
        "project_id": reference["project_id"],
        "phase_id": reference["phase_id"],
        "run_id": reference["run_id"],
        "checkpoint_id": checkpoint_path.name,
        "case_id": reference["case_id"],
        "case_index_row_id": reference["case_index_row_id"],
        "probe_plan_row_id": plan_row["probe_plan_row_id"],
        "split": reference["split"],
        "pool_role": reference["pool_role"],
        "source_dataset_jsonl": reference["source_dataset_jsonl"],
        "dataset_manifest_id": reference["dataset_manifest_id"],
        "dataset_manifest_sha256": reference["dataset_manifest_sha256"],
        "fn_rescue_overlay_membership": reference["fn_rescue_overlay_membership"],
        "source_line_idx": reference["source_line_idx"],
        "image_id": reference["image_id"],
        "image_path": reference["image_path"],
        "desc_id": reference["desc_id"],
        "desc_text": reference["desc_text"],
        "desc_text_canonical": reference["desc_text_canonical"],
        "same_desc_cluster_id": reference["same_desc_cluster_id"],
        "target_gt_idx": reference["target_gt_idx"],
        "gt_idx": reference["gt_idx"],
        "sampling_policy_id": plan_row["sampling_policy_id"],
        "sampling_policy_sha256": plan_row["sampling_policy_sha256"],
        "strata_key": plan_row["strata_key"],
        "shard_id": plan_row["planned_shard_id"],
    }


def _load_model_handle(
    *,
    checkpoint_path: Path,
    device: str,
    torch_dtype: str,
    attn_implementation: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from src.infer.checkpoints import (
        resolve_inference_checkpoint,
        validate_compact_coord_token_adapter_contract,
    )
    from src.tokens.row_offsets import install_coord_offset_adapter, reattach_coord_offset_hooks

    resolved = resolve_inference_checkpoint(model_checkpoint=str(checkpoint_path))
    validate_compact_coord_token_adapter_contract(
        resolved,
        detection_sequence_format="compact_full",
    )
    processor_source = str(resolved.resolved_base_model_checkpoint)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        local_files_only=Path(processor_source).exists(),
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("AutoProcessor did not expose a tokenizer")
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or "<|endoftext|>"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(str(torch_dtype), torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(resolved.resolved_base_model_checkpoint),
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        local_files_only=Path(str(resolved.resolved_base_model_checkpoint)).exists(),
    )
    model.to(torch.device(device))
    adapter_checkpoint = str(resolved.resolved_adapter_checkpoint or "").strip()
    adapter_info = getattr(resolved, "adapter_info", None)
    coord_offset_spec = getattr(adapter_info, "coord_offset_spec", None) if adapter_info else None
    if adapter_checkpoint:
        if coord_offset_spec is not None:
            install_coord_offset_adapter(
                model,
                coord_ids=coord_offset_spec.coord_ids,
                tie_head=coord_offset_spec.tie_head,
            )
        from swift import Swift

        model = Swift.from_pretrained(model, model_id=adapter_checkpoint, inference_mode=True)
        if coord_offset_spec is not None:
            reattach_coord_offset_hooks(model)
    model.eval()
    return {"model": model, "processor": processor, "tokenizer": tokenizer, "resolved_checkpoint": resolved}


def _model_device(model: Any):
    return next(model.parameters()).device


def _resolve_image_path(row: Mapping[str, Any]) -> Path:
    image_path = Path(str(row["image_path"]))
    if image_path.is_absolute():
        return image_path
    dataset_root = Path(str(row["source_dataset_jsonl"])).parent
    return dataset_root / image_path
