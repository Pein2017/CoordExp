from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .fn_probe import (
    HINT_LEVELS,
    PREFIX_CONDITIONS,
    build_fn_candidate_score_rows,
    build_fn_probe_rows,
    build_fn_slot_evidence_rows,
)
from .status import (
    CONSTRAINT_POLICY,
    DECODE_POLICY,
    REAL_FN_HINT_RUNTIME_KIND,
)

if TYPE_CHECKING:
    from .config import A32Config


_COORD_RE = re.compile(r"<\|coord_(\d+)\|>")


def run_real_fn_hint_probe(
    config: A32Config,
    *,
    shard_id: int,
    allow_overwrite: bool = False,
    gpu_id: str | None = None,
) -> dict[str, Any]:
    cases = _read_jsonl(config.artifact_root / "fn_probe" / "fn_cases.jsonl")
    shard_cases = [
        case
        for index, case in enumerate(cases)
        if index % int(config.sampling.num_shards) == int(shard_id)
    ]
    shard_dir = config.artifact_root / "fn_probe" / "fn_hint_shards"
    paths = {
        "probe_rows": shard_dir / f"shard_{int(shard_id)}_probe_rows.jsonl",
        "candidate_scores": shard_dir
        / f"shard_{int(shard_id)}_candidate_scores.jsonl",
        "slot_evidence": shard_dir / f"shard_{int(shard_id)}_slot_evidence.jsonl",
        "decode_rows": shard_dir / f"shard_{int(shard_id)}_decode_rows.jsonl",
        "summary": shard_dir / f"shard_{int(shard_id)}_summary.json",
    }
    _ensure_paths_can_write(paths.values(), allow_overwrite=allow_overwrite)
    provenance = _runtime_provenance(shard_id=shard_id, gpu_id=_gpu_id(gpu_id))
    if not shard_cases:
        for key in ("probe_rows", "candidate_scores", "slot_evidence", "decode_rows"):
            _write_jsonl(paths[key], [])
        _write_json(paths["summary"], {**provenance, "fn_cases": 0, "probe_rows": 0})
        return {
            "stage": "fn_hint_probe",
            **provenance,
            "fn_cases": 0,
            "probe_rows": 0,
            "written": {key: str(path) for key, path in paths.items()},
        }

    rows_by_role: dict[str, list[Mapping[str, Any]]] = {}
    for case in shard_cases:
        rows_by_role.setdefault(str(case["checkpoint_role"]), []).append(case)

    all_probe_specs: list[dict[str, Any]] = []
    all_candidate_inputs: list[dict[str, Any]] = []
    all_slot_inputs: list[dict[str, Any]] = []
    all_decode_rows: list[dict[str, Any]] = []
    for role, role_cases in rows_by_role.items():
        model_handle = _load_model_for_role(config, role)
        for case in role_cases:
            for prefix_condition in PREFIX_CONDITIONS:
                prefix_objects = _prefix_objects_for_condition(case, prefix_condition)
                prefix_text = _render_prefix_objects(prefix_objects)
                prefix_candidate_inputs = _score_candidate_descs(
                    model_handle=model_handle,
                    case=case,
                    prefix_text=prefix_text,
                )
                for hint_level in HINT_LEVELS:
                    probe_id = (
                        f"{case['fn_case_id']}:{prefix_condition}:{hint_level}"
                    )
                    all_candidate_inputs.extend(
                        _candidate_scores_for_probe(
                            prefix_candidate_inputs,
                            probe_id=probe_id,
                        )
                    )
                    decode = _decode_hint_probe(
                        model_handle=model_handle,
                        case=case,
                        prefix_text=prefix_text,
                        hint_level=hint_level,
                    )
                    all_decode_rows.append({**provenance, **decode, "probe_id": probe_id})
                    all_probe_specs.append(
                        {
                            "probe_id": probe_id,
                            "fn_case_id": case["fn_case_id"],
                            "hint_level": hint_level,
                            "prefix_condition": prefix_condition,
                            "anchor_policy": f"{prefix_condition}_real_runtime_v1",
                            "valid_parse": bool(decode["valid_parse"]),
                            "generated_continuation": decode["generated_tail_text"],
                            "prefix_objects": prefix_objects,
                        }
                    )
                    all_slot_inputs.extend(
                        _slot_evidence_from_decode(
                            case,
                            probe_id=probe_id,
                            hint_level=hint_level,
                            generated_box=decode.get("generated_box"),
                        )
                    )

    candidate_rows = build_fn_candidate_score_rows(all_candidate_inputs)
    slot_rows = build_fn_slot_evidence_rows(
        all_slot_inputs,
        broad_x1_radius=config.fn_probe.broad_x1_radius,
    )
    probe_rows = build_fn_probe_rows(
        shard_cases,
        probe_specs=all_probe_specs,
        candidate_score_rows=candidate_rows,
        slot_evidence_rows=slot_rows,
        hint_policy_id=config.fn_probe.hint_policy_id,
    )
    candidate_rows = [_decorate(row, provenance) for row in candidate_rows]
    slot_rows = [_decorate(row, provenance) for row in slot_rows]
    probe_rows = [_decorate(row, provenance) for row in probe_rows]
    _write_jsonl(paths["probe_rows"], probe_rows)
    _write_jsonl(paths["candidate_scores"], candidate_rows)
    _write_jsonl(paths["slot_evidence"], slot_rows)
    _write_jsonl(paths["decode_rows"], all_decode_rows)
    _write_json(
        paths["summary"],
        {
            **provenance,
            "fn_cases": len(shard_cases),
            "probe_rows": len(probe_rows),
            "candidate_score_rows": len(candidate_rows),
            "slot_evidence_rows": len(slot_rows),
        },
    )
    return {
        "stage": "fn_hint_probe",
        **provenance,
        "fn_cases": len(shard_cases),
        "probe_rows": len(probe_rows),
        "written": {key: str(path) for key, path in paths.items()},
    }


def _decode_hint_probe(
    *,
    model_handle: Any,
    case: Mapping[str, Any],
    prefix_text: str,
    hint_level: str,
) -> dict[str, Any]:
    import torch
    from src.common.qwen_generation import resolve_qwen_chat_generation_token_ids
    from src.analysis.sorted_random_no_newline_phenotype.paired_probe import (
        _processor_inputs,
    )

    assistant_prefix = prefix_text + _hint_text(case, hint_level)
    inputs = _processor_inputs(
        model_handle=model_handle,
        image_path=_case_image_path(case),
        assistant_text=assistant_prefix,
    )
    device = next(model_handle.model.parameters()).device
    generate_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    prompt_len = int(generate_inputs["input_ids"].shape[-1])
    qwen_ids = resolve_qwen_chat_generation_token_ids(model_handle.tokenizer)
    max_new_tokens = {"none": 16, "desc": 4, "desc_x1": 3, "desc_x1_y1": 2}[
        hint_level
    ]
    with torch.inference_mode():
        generated = model_handle.model.generate(
            **generate_inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=qwen_ids.eos_token_id,
            pad_token_id=qwen_ids.pad_token_id,
        )
    generated_ids = _first_token_row(generated)
    tail_ids = generated_ids[prompt_len:]
    tail_text = model_handle.tokenizer.decode(
        tail_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    target_coord_box = _target_coord_box_for_case(case)
    parsed_box, parse_errors = _parse_generated_box(
        hint_level,
        tail_text,
        target_box=target_coord_box,
    )
    return {
        "fn_case_id": case["fn_case_id"],
        "checkpoint_role": case["checkpoint_role"],
        "hint_level": hint_level,
        "assistant_prefix_sha256": _sha256_text(assistant_prefix),
        "generated_tail_text": tail_text,
        "generated_tail_sha256": _sha256_text(tail_text),
        "generated_box": parsed_box,
        "target_box_coord_token": target_coord_box,
        "valid_parse": parsed_box is not None,
        "parse_errors": parse_errors,
        "target_iou": 0.0
        if parsed_box is None
        else _bbox_iou_xyxy(parsed_box, target_coord_box),
        "target_iou_surface": "coord_token",
        "decode_policy": DECODE_POLICY,
        "constraint_policy": CONSTRAINT_POLICY,
    }


def _score_candidate_descs(
    *,
    model_handle: Any,
    case: Mapping[str, Any],
    prefix_text: str,
) -> list[dict[str, Any]]:
    from src.analysis.prefix_state_transition_tomography.prefix_rendering import (
        render_forced_desc_pre_x1_assistant_text,
    )
    from src.analysis.sorted_random_no_newline_phenotype.paired_probe import (
        _score_suffix_mean_logprob,
    )

    descs = _candidate_descs(case)
    rows: list[dict[str, Any]] = []
    for index, desc in enumerate(descs):
        suffix = render_forced_desc_pre_x1_assistant_text([], desc)
        score = _score_suffix_mean_logprob(
            model_handle=model_handle,
            image_path=_case_image_path(case),
            assistant_prefix=prefix_text,
            suffix=suffix,
        )
        rows.append(
            {
                "fn_case_id": case["fn_case_id"],
                "candidate_id": f"candidate:{index}:{desc}",
                "candidate_gt_idx": case.get("fn_gt_idx") if desc == case.get("fn_desc") else None,
                "desc": desc,
                "role": "residual_same_desc" if desc == case.get("fn_desc") else "hard_competitor",
                "score": score,
            }
        )
    return rows


def _candidate_scores_for_probe(
    candidate_inputs: Sequence[Mapping[str, Any]],
    *,
    probe_id: str,
) -> list[dict[str, Any]]:
    return [{**dict(row), "probe_id": str(probe_id)} for row in candidate_inputs]


def _candidate_descs(case: Mapping[str, Any]) -> list[str]:
    descs = [str(case.get("fn_desc", ""))]
    for pred in case.get("pred_rows_ordered", ()) or ():
        if isinstance(pred, Mapping):
            desc = str(pred.get("desc", ""))
            if desc and desc not in descs:
                descs.append(desc)
    return [desc for desc in descs if desc]


def _slot_evidence_from_decode(
    case: Mapping[str, Any],
    *,
    probe_id: str,
    hint_level: str,
    generated_box: Any,
) -> list[dict[str, Any]]:
    if generated_box is None:
        return []
    target = _target_coord_box_for_case(case)
    slots_by_hint = {
        "none": ("x1", "y1", "x2", "y2"),
        "desc": ("x1", "y1", "x2", "y2"),
        "desc_x1": ("x1", "y1", "x2", "y2"),
        "desc_x1_y1": ("x1", "y1", "x2", "y2"),
    }
    hinted_slots_by_hint = {
        "none": set(),
        "desc": set(),
        "desc_x1": {"x1"},
        "desc_x1_y1": {"x1", "y1"},
    }
    slot_names = slots_by_hint[hint_level]
    hinted_slots = hinted_slots_by_hint[hint_level]
    width = max(0, target[2] - target[0])
    height = max(0, target[3] - target[1])
    axis_len = {"x1": width, "x2": width, "y1": height, "y2": height}
    slot_index = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
    return [
        {
            "probe_id": probe_id,
            "fn_case_id": case["fn_case_id"],
            "slot": slot,
            "axis_len": axis_len[slot],
            "gt_idx": case.get("fn_gt_idx"),
            "gt_value": target[slot_index[slot]],
            "peak_value": int(generated_box[slot_index[slot]]),
            "score": None if slot in hinted_slots else 1.0,
            "evidence_source": "hint_control" if slot in hinted_slots else "model_decode",
            "hinted_control": slot in hinted_slots,
            "model_predicted": slot not in hinted_slots,
        }
        for slot in slot_names
    ]


def _hint_text(case: Mapping[str, Any], hint_level: str) -> str:
    from src.common.detection_compact_rows import render_compact_row

    desc = str(case.get("fn_desc", "")).strip()
    bbox = _target_coord_box_for_case(case)
    if hint_level == "none":
        return ""
    if hint_level == "desc":
        coords: tuple[str, ...] = ()
    elif hint_level == "desc_x1":
        coords = (_coord_token(bbox[0]),)
    elif hint_level == "desc_x1_y1":
        coords = (_coord_token(bbox[0]), _coord_token(bbox[1]))
    else:
        raise ValueError(f"unsupported hint_level: {hint_level}")
    return render_compact_row(
        desc,
        coords,
        include_object_ref_marker=True,
        include_bbox_start_marker=True,
    )


def _parse_generated_box(
    hint_level: str,
    tail_text: str,
    *,
    target_box: Sequence[int],
) -> tuple[list[int] | None, list[str]]:
    required = {"none": 4, "desc": 4, "desc_x1": 3, "desc_x1_y1": 2}[hint_level]
    tokens = [int(match.group(1)) for match in _COORD_RE.finditer(str(tail_text))]
    if len(tokens) < required:
        return None, [f"expected_{required}_leading_coord_tokens"]
    if hint_level in {"none", "desc"}:
        box = tokens[:4]
    elif hint_level == "desc_x1":
        box = [int(target_box[0]), *tokens[:3]]
    else:
        box = [int(target_box[0]), int(target_box[1]), *tokens[:2]]
    if not _valid_xyxy(box):
        return None, ["invalid_xyxy_box"]
    return [int(value) for value in box], []


def _prefix_objects_for_condition(
    case: Mapping[str, Any],
    prefix_condition: str,
) -> list[dict[str, Any]]:
    if prefix_condition == "empty_prefix":
        return []
    if prefix_condition in {
        "rollout_prefix",
        "teacher_sorted_prefix",
        "teacher_oracle_remaining_prefix",
        "same_desc_removed_prefix",
        "same_desc_shuffled_prefix",
    }:
        rows = []
        for pred in case.get("pred_rows_ordered", ()) or ():
            if not isinstance(pred, Mapping):
                continue
            desc = str(pred.get("desc", ""))
            if prefix_condition == "same_desc_removed_prefix" and desc == case.get("fn_desc"):
                continue
            bbox = pred.get("bbox_xyxy") or pred.get("bbox") or pred.get("bbox_2d")
            if not bbox:
                continue
            bbox_coord = _coord_token_box_for_case_surface(case, bbox)
            rows.append(
                {
                    "source": "rollout_pred",
                    "gt_idx": None,
                    "pred_idx": pred.get("pred_idx"),
                    "desc": desc,
                    "bbox": bbox_coord,
                    "bbox_xyxy": bbox_coord,
                }
            )
        if prefix_condition == "same_desc_shuffled_prefix":
            rows = list(reversed(rows))
        return rows
    raise ValueError(f"unsupported prefix_condition: {prefix_condition}")


def _target_coord_box_for_case(case: Mapping[str, Any]) -> list[int]:
    return _coord_token_box_for_case_surface(case, case["fn_bbox"])


def _coord_token_box_for_case_surface(
    case: Mapping[str, Any],
    bbox: Sequence[Any],
) -> list[int]:
    raw = [int(value) for value in bbox]
    if _case_box_is_pixel_surface(case, raw):
        width = _positive_axis_len(case.get("width"))
        height = _positive_axis_len(case.get("height"))
        if width is None or height is None:
            return [_clip_coord_token(value) for value in raw]
        return [
            _scale_pixel_to_coord_token(raw[0], width),
            _scale_pixel_to_coord_token(raw[1], height),
            _scale_pixel_to_coord_token(raw[2], width),
            _scale_pixel_to_coord_token(raw[3], height),
        ]
    return [_clip_coord_token(value) for value in raw]


def _case_box_is_pixel_surface(case: Mapping[str, Any], bbox: Sequence[int]) -> bool:
    coord_mode = str(case.get("coord_mode", "")).lower()
    if coord_mode == "pixel":
        return True
    return any(value > 1000 or value < 0 for value in bbox)


def _positive_axis_len(value: Any) -> int | None:
    try:
        axis = int(value)
    except (TypeError, ValueError):
        return None
    return axis if axis > 0 else None


def _scale_pixel_to_coord_token(value: int, axis_len: int) -> int:
    return _clip_coord_token(int(round(float(value) * 1000.0 / float(axis_len))))


def _clip_coord_token(value: int) -> int:
    return max(0, min(1000, int(value)))


def _render_prefix_objects(prefix_objects: Sequence[Mapping[str, Any]]) -> str:
    from src.analysis.prefix_state_transition_tomography.prefix_rendering import (
        render_teacher_prefix,
    )

    return render_teacher_prefix(prefix_objects)


def _load_model_for_role(config: A32Config, role: str) -> Any:
    from src.analysis.sorted_random_no_newline_phenotype.paired_probe import (
        _load_model_for_checkpoint,
    )

    return _load_model_for_checkpoint(
        config,
        checkpoint_path=config.checkpoints[role].checkpoint_path,
    )


def _case_image_path(case: Mapping[str, Any]) -> Path:
    path = Path(str(case.get("image_path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"FN case image_path does not exist: {path}")
    return path


def _first_token_row(value: Any) -> list[int]:
    if hasattr(value, "sequences"):
        value = value.sequences
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        return [int(item) for item in value[0]]
    if isinstance(value, Sequence):
        return [int(item) for item in value]
    raise TypeError("generated token ids must be a tensor or sequence")


def _runtime_provenance(*, shard_id: int, gpu_id: str) -> dict[str, Any]:
    probe_runtime_id = f"a3_2_fn_hint_shard_{int(shard_id)}_{_sha256_text(gpu_id)[:8]}"
    return {
        "runtime_kind": REAL_FN_HINT_RUNTIME_KIND,
        "probe_runtime_id": probe_runtime_id,
        "shard_id": int(shard_id),
        "gpu_id": str(gpu_id),
        "decode_policy": DECODE_POLICY,
        "constraint_policy": CONSTRAINT_POLICY,
    }


def _decorate(row: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(row), **dict(provenance)}


def _coord_token(value: int) -> str:
    value = max(0, min(999, int(value)))
    return f"<|coord_{value}|>"


def _valid_xyxy(box: Sequence[int]) -> bool:
    return len(box) == 4 and int(box[2]) > int(box[0]) and int(box[3]) > int(box[1])


def _bbox_iou_xyxy(a: Sequence[Any], b: Sequence[Any]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _ensure_paths_can_write(paths: Sequence[Path], *, allow_overwrite: bool) -> None:
    for path in paths:
        if Path(path).exists() and not allow_overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


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
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _gpu_id(gpu_id: str | None) -> str:
    if gpu_id is not None and str(gpu_id).strip():
        return str(gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        return visible.strip()
    return "unknown"


__all__ = ["run_real_fn_hint_probe"]
