from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import CHECKPOINT_ROLES, SCHEMA_VERSION
from .jsonl import read_jsonl, write_jsonl
from .posterior import axis_len_for_slot, classify_slot_posterior
from .prefix_modes import render_compact_prefix_rows


REAL_RUNTIME_KIND = "real_gpu_post_x1_slot_posterior_v1"
SLOTS_AFTER_X1 = ("y1", "x2", "y2")


def run_real_slot_posterior(
    *,
    artifact_root: Path,
    config: Mapping[str, Any],
    shard_id: int,
    gpu_id: str | None = None,
    limit: int | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    gpu = str(gpu_id if gpu_id is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    root = Path(artifact_root)
    prefix_rows = read_jsonl(root / "prefix_states.jsonl")
    case_by_id = {str(row["case_id"]): row for row in read_jsonl(root / "case_universe.jsonl")}
    num_shards = int(config.get("num_shards", 8))
    selected = [
        row
        for idx, row in enumerate(prefix_rows)
        if idx % num_shards == int(shard_id)
    ]
    if limit is not None:
        selected = selected[: int(limit)]
    shard_path = root / "slot_posterior_shards" / f"shard_{int(shard_id)}.jsonl"
    if shard_path.exists() and not allow_overwrite:
        raise FileExistsError(f"slot posterior shard already exists: {shard_path}")
    rows: list[dict[str, Any]] = []
    for role in CHECKPOINT_ROLES:
        checkpoint = config["checkpoints"][role]
        handle = _load_model_handle(checkpoint_path=Path(checkpoint["checkpoint_path"]), artifact_root=root, role=role)
        try:
            coord_token_ids = _coord_token_ids(handle)
            for prefix_row in selected:
                case = case_by_id[str(prefix_row["case_id"])]
                image_path = _resolve_image_path(case, image_root=Path(str(config["image_root"])))
                rows.extend(
                    _probe_prefix_row(
                        model_handle=handle,
                        coord_token_ids=coord_token_ids,
                        checkpoint_role=role,
                        checkpoint=config["checkpoints"][role],
                        prefix_row=prefix_row,
                        case=case,
                        image_path=image_path,
                    )
                )
        finally:
            _release_model_handle(handle)
    count = write_jsonl(shard_path, rows)
    summary_path = root / "slot_posterior_shard_summaries.jsonl"
    existing = read_jsonl(summary_path) if summary_path.exists() and allow_overwrite else []
    existing = [row for row in existing if int(row.get("shard_id", -1)) != int(shard_id)]
    existing.append(
        {
            "artifact_schema_version": SCHEMA_VERSION,
            "row_schema_version": "slot_posterior_shard_summary.v1",
            "runtime_kind": REAL_RUNTIME_KIND,
            "shard_id": int(shard_id),
            "gpu_id": gpu,
            "row_count": count,
            "prefix_state_rows": len(selected),
        }
    )
    write_jsonl(summary_path, sorted(existing, key=lambda row: int(row["shard_id"])))
    return {
        "status": "ok",
        "runtime_kind": REAL_RUNTIME_KIND,
        "shard_id": int(shard_id),
        "gpu_id": gpu,
        "prefix_state_rows": len(selected),
        "row_count": count,
        "artifact": str(shard_path),
    }


def _probe_prefix_row(
    *,
    model_handle: Any,
    coord_token_ids: Sequence[int],
    checkpoint_role: str,
    checkpoint: Mapping[str, Any],
    prefix_row: Mapping[str, Any],
    case: Mapping[str, Any],
    image_path: Path,
) -> list[dict[str, Any]]:
    target = _target_object(case)
    target_bbox = [int(value) for value in target["bbox_coord_token_xyxy"]]
    row_separator = str(checkpoint["template_contract"]["row_separator"])
    out: list[dict[str, Any]] = []
    for slot in SLOTS_AFTER_X1:
        assistant_text = _assistant_prefix_for_slot(
            prefix_objects=prefix_row.get("prefix_objects") or [],
            desc=str(prefix_row["desc"]),
            target_bbox=target_bbox,
            slot=slot,
            row_separator=row_separator,
        )
        logits = _last_logits(
            model_handle=model_handle,
            image_path=image_path,
            assistant_text=assistant_text,
            expected_context_suffix=_expected_suffix(target_bbox, slot),
        )
        target_value = _slot_value(target_bbox, slot)
        row = classify_slot_posterior(
            full_vocab_logits=logits,
            coord_token_ids=coord_token_ids,
            slot=slot,
            target_value=target_value,
            target_axis_len=axis_len_for_slot(slot, target_bbox),
            competitors=_slot_objects(case, desc=str(case["desc"]), slot=slot, exclude_gt_idx=int(case["target_gt_idx"])),
            other_desc_objects=_slot_objects(case, desc=str(case["desc"]), slot=slot, other_desc=True),
            low_margin_threshold=0.05,
            coord_mass_low_threshold=0.01,
        )
        out.append(
            {
                **row,
                "checkpoint_role": checkpoint_role,
                "comparison_role": checkpoint.get("comparison_role"),
                "controlled_comparison_group": checkpoint.get("controlled_comparison_group"),
                "template_contract": checkpoint.get("template_contract"),
                "runtime_kind": REAL_RUNTIME_KIND,
                "case_id": case["case_id"],
                "prefix_row_id": prefix_row.get("prefix_row_id"),
                "prefix_state_id": prefix_row.get("prefix_row_id"),
                "prefix_mode": prefix_row.get("prefix_mode"),
                "desc": case.get("desc"),
                "target_gt_idx": case.get("target_gt_idx"),
                "target_gt_box": target_bbox,
                "same_desc_competitor_gt_boxes": case.get("same_desc_competitor_bboxes", []),
                "primary_basin_label_source": case.get("primary_basin_label_source"),
                "image_id": case.get("image_id"),
                "image_path": str(image_path),
                "assistant_prefix_sha256": _sha256_text(assistant_text),
                "template_prompt_hash": _sha256_text(assistant_text + checkpoint_role),
                "forced_prompt_sha256": _sha256_text(assistant_text),
                "system_prompt_sha256": None,
                "user_prompt_sha256": None,
            }
        )
    return out


def _assistant_prefix_for_slot(
    *,
    prefix_objects: Sequence[Mapping[str, Any]],
    desc: str,
    target_bbox: Sequence[int],
    slot: str,
    row_separator: str,
) -> str:
    prefix = render_compact_prefix_rows(prefix_objects, row_separator=row_separator)
    coord_prefix_count = {"y1": 1, "x2": 2, "y2": 3}[slot]
    forced = (
        f"<|object_ref_start|>{desc.strip().lower()}<|box_start|>"
        + "".join(f"<|coord_{int(value)}|>" for value in target_bbox[:coord_prefix_count])
    )
    sep = "\n" if row_separator == "newline" else ""
    return forced if not prefix else prefix + sep + forced


def _expected_suffix(target_bbox: Sequence[int], slot: str) -> str:
    coord_prefix_count = {"y1": 1, "x2": 2, "y2": 3}[slot]
    return "".join(f"<|coord_{int(value)}|>" for value in target_bbox[:coord_prefix_count])


def _last_logits(*, model_handle: Any, image_path: Path, assistant_text: str, expected_context_suffix: str) -> Any:
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
    return outputs.logits[0, -1].detach().cpu()


def _processor_inputs(*, model_handle: Any, image_path: Path, assistant_text: str, expected_context_suffix: str) -> Mapping[str, Any]:
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
    full_text = model_handle.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    if not str(full_text).rstrip().endswith(expected_context_suffix):
        raise RuntimeError(f"rendered prompt does not end with expected suffix {expected_context_suffix!r}")
    image = Image.open(image_path).convert("RGB")
    return model_handle.processor(text=[full_text], images=[image], return_tensors="pt", padding=False)


def _load_model_handle(*, checkpoint_path: Path, artifact_root: Path, role: str) -> Any:
    from src.analysis.hard_ce_coord_logit_locality import (
        StudyConfig,
        StudyExecutionConfig,
        StudyModelConfig,
        StudyPaths,
        load_model_handle,
    )

    config = StudyConfig(
        paths=StudyPaths(
            checkpoint=checkpoint_path,
            resolved_config=artifact_root / "runtime" / role / "resolved_config.json",
            source_config=None,
            dataset_jsonl=artifact_root / "case_universe.jsonl",
            image_root=Path("/"),
            artifact_root=artifact_root / "runtime" / role,
            self_rollout_root=None,
            self_rollout_regen_config=None,
        ),
        model=StudyModelConfig(object_ordering="sorted", device="cuda:0", torch_dtype="bfloat16"),
        execution=StudyExecutionConfig(sample_limit=1, batch_size=1, top_k=8),
    )
    return load_model_handle(config)


def _coord_token_ids(handle: Any) -> tuple[int, ...]:
    from src.analysis.hard_ce_coord_logit_locality import resolve_coord_token_ids

    return resolve_coord_token_ids(handle.tokenizer).coord_token_ids


def _release_model_handle(handle: Any) -> None:
    del handle
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _target_object(case: Mapping[str, Any]) -> Mapping[str, Any]:
    target_idx = int(case["target_gt_idx"])
    for obj in case.get("objects") or []:
        if int(obj.get("gt_idx", -1)) == target_idx:
            return obj
    raise ValueError(f"target object missing for case {case.get('case_id')}")


def _slot_objects(
    case: Mapping[str, Any],
    *,
    desc: str,
    slot: str,
    exclude_gt_idx: int | None = None,
    other_desc: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for obj in case.get("objects") or []:
        obj_desc = str(obj.get("desc"))
        if other_desc:
            if obj_desc == desc:
                continue
        elif obj_desc != desc:
            continue
        if exclude_gt_idx is not None and int(obj.get("gt_idx", -1)) == int(exclude_gt_idx):
            continue
        bbox = [int(value) for value in obj["bbox_coord_token_xyxy"]]
        out.append(
            {
                "gt_idx": int(obj.get("gt_idx", -1)),
                "desc": obj_desc,
                "value": _slot_value(bbox, slot),
                "axis_len": axis_len_for_slot(slot, bbox),
            }
        )
    return out


def _slot_value(bbox: Sequence[int], slot: str) -> int:
    return int(bbox[{"x1": 0, "y1": 1, "x2": 2, "y2": 3}[slot]])


def _resolve_image_path(case: Mapping[str, Any], *, image_root: Path) -> Path:
    image_path = Path(str(case["image_path"]))
    if image_path.is_absolute() and image_path.exists():
        return image_path
    candidates = [
        image_root / image_path,
        image_root / "images" / image_path,
        image_root.parent / image_path,
        image_root.parent / "images" / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not resolve image path: " + ", ".join(str(path) for path in candidates))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["REAL_RUNTIME_KIND", "SLOTS_AFTER_X1", "run_real_slot_posterior"]
