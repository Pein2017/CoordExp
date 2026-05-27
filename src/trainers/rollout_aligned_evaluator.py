from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch

from src.common.geometry import denorm_and_clamp, flatten_points
from src.common.object_field_order import build_object_payload
from src.common.prediction_parsing import extract_special_tokens, load_prediction_dict
from src.coord_tokens.codec import token_to_int
from src.eval.detection import EvalOptions, evaluate_and_save
from src.infer.artifacts import write_score_provenance_sidecar
from src.infer.prompt import DetectionPromptPolicy, prompt_policy_fingerprint
from src.infer.runtime import (
    build_decode_policy_fingerprint,
    build_decode_request_from_rollout_matching_config,
    build_model_identity_fingerprint,
)
from .rollout_matching.contracts import GTObject, ParsedPredObject
from .rollout_matching.parsing import coerce_int as _coerce_int


_IM_END = "<|im_end|>"


def _write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _stage2_eval_model_handle(owner: Any) -> str:
    model = getattr(owner, "model", None)
    config = getattr(model, "config", None)
    name_or_path = getattr(config, "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path.strip():
        return name_or_path.strip()
    return str(getattr(getattr(owner, "args", None), "output_dir", "stage2_live_model"))


def _write_stage2_eval_score_provenance(
    *,
    owner: Any,
    scored_path: Path,
    raw_path: Path,
    eval_prompt_variant: str | None,
    eval_rollout_backend: str,
    eval_vllm_mode: str,
    eval_detection_score_mode: str,
    eval_detection_cfg: Mapping[str, Any],
    eval_decode_override: Optional[Mapping[str, Any]] = None,
) -> None:
    backend = "vllm" if str(eval_rollout_backend).strip().lower() == "vllm" else "hf"
    backend_mode = str(eval_vllm_mode or ("server" if backend == "vllm" else "local"))
    rollout_matching_cfg = getattr(owner, "rollout_matching_cfg", {}) or {}
    if isinstance(rollout_matching_cfg, Mapping):
        decode_request_base = build_decode_request_from_rollout_matching_config(
            rollout_matching_cfg,
            decode_override=eval_decode_override,
        )
    else:
        decode_request_base = build_decode_request_from_rollout_matching_config({})
    decode_request = replace(
        decode_request_base,
        backend=backend,  # type: ignore[arg-type]
        backend_mode=backend_mode,
        trace_logprobs=str(eval_detection_score_mode) == "confidence_postop",
    )
    decode_request = replace(
        decode_request,
        decode_policy_fingerprint=build_decode_policy_fingerprint(decode_request),
    )
    prompt_fingerprint = prompt_policy_fingerprint(
        DetectionPromptPolicy(
            name="stage2_rollout_correction_eval",
            version="1",
            system_prompt="",
            user_prompt=json.dumps(
                {
                    "eval_prompt_variant": eval_prompt_variant,
                    "object_field_order": str(owner._object_field_order()),
                    "object_ordering": str(owner._object_ordering()),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            image_count=1,
            do_resize=False,
        )
    )
    model_handle = _stage2_eval_model_handle(owner)
    backend_sync_identity = None
    if backend == "vllm" and backend_mode == "server":
        backend_sync_identity = getattr(
            owner,
            "_vllm_server_last_backend_sync_identity",
            getattr(owner, "_vllm_server_last_sync_provenance", None),
        )
    score_mode = str(eval_detection_score_mode or "constant")
    pred_score_source = str(
        eval_detection_cfg.get("pred_score_source", "eval_rollout_constant")
    )
    pred_score_version = int(eval_detection_cfg.get("pred_score_version", 1) or 1)
    constant_score = float(eval_detection_cfg.get("constant_score", 1.0) or 1.0)
    write_score_provenance_sidecar(
        scored_path=scored_path,
        prompt_policy_fingerprint=prompt_fingerprint,
        decode_policy_fingerprint=build_decode_policy_fingerprint(decode_request),
        model_identity_fingerprint=build_model_identity_fingerprint(
            checkpoint_mode="training_live_model",
            requested_model_checkpoint=model_handle,
            requested_adapter_checkpoint=None,
            resolved_base_model_checkpoint=model_handle,
            resolved_adapter_checkpoint=None,
            backend=backend,
            backend_mode=backend_mode,
            backend_model=model_handle,
            backend_sync_identity=(
                backend_sync_identity
                if isinstance(backend_sync_identity, Mapping)
                else None
            ),
        ),
        policy_name="confidence_postop" if score_mode == "confidence_postop" else "constant_score",
        score_source=(
            "confidence_postop:v2"
            if score_mode == "confidence_postop"
            else f"{pred_score_source}:v{pred_score_version}"
        ),
        aggregation_rule=(
            "bbox_logprob_confidence_exp"
            if score_mode == "confidence_postop"
            else "constant_per_prediction"
        ),
        token_span_rule=(
            "generated_token_trace_bbox_and_desc_spans"
            if score_mode == "confidence_postop"
            else "none"
        ),
        constant_score_value=None if score_mode == "confidence_postop" else constant_score,
        source_raw_artifact_path=raw_path,
        parser_policy="stage2_rollout_correction_eval",
        extra={"eval_surface": "stage2_rollout_correction"},
    )


def _stage2_eval_output_dir(*, owner: Any, global_step: int) -> Path:
    output_root = Path(str(getattr(getattr(owner, "args", None), "output_dir", ".")))
    return output_root / "eval_detection" / f"step_{int(global_step):07d}"


def _resolve_stage2_eval_source_jsonl(owner: Any) -> Path | None:
    output_dir = Path(str(getattr(getattr(owner, "args", None), "output_dir", "") or ""))
    resolved_config_path = output_dir / "resolved_config.json"
    if not resolved_config_path.is_file():
        return None
    try:
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    resolved = payload.get("resolved")
    if not isinstance(resolved, Mapping):
        return None
    custom = resolved.get("custom")
    if not isinstance(custom, Mapping):
        return None
    val_jsonl_raw = custom.get("val_jsonl")
    if not isinstance(val_jsonl_raw, str) or not val_jsonl_raw.strip():
        return None

    val_jsonl = Path(val_jsonl_raw.strip())
    if val_jsonl.is_absolute():
        return val_jsonl if val_jsonl.is_file() else None
    cwd_candidate = (Path.cwd() / val_jsonl).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    config_path_raw = payload.get("config_path")
    if isinstance(config_path_raw, str) and config_path_raw.strip():
        config_path = Path(config_path_raw)
        if config_path.is_file():
            for parent in [config_path.parent, *config_path.parents]:
                candidate = (parent / val_jsonl).resolve()
                if candidate.is_file():
                    return candidate
    return None


def _load_stage2_eval_source_rows_by_base_idx(owner: Any) -> Dict[int, Dict[str, Any]]:
    source_jsonl = _resolve_stage2_eval_source_jsonl(owner)
    if source_jsonl is None:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    try:
        with source_jsonl.open("r", encoding="utf-8") as handle:
            for idx, raw_line in enumerate(handle):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, Mapping):
                    item = dict(row)
                    item["_source_jsonl"] = str(source_jsonl)
                    item["_source_jsonl_dir"] = str(source_jsonl.parent)
                    out[int(idx)] = item
    except OSError:
        return {}
    return out


def _coerce_positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _first_image_from_source_row(row: Mapping[str, Any]) -> str | None:
    images = row.get("images")
    if isinstance(images, list):
        for value in images:
            if isinstance(value, str) and value.strip():
                return value.strip()
    image = row.get("image")
    if isinstance(image, str) and image.strip():
        return image.strip()
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        return file_name.strip()
    return None


def _rescale_stage2_eval_record_geometry(
    record: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> Dict[str, Any]:
    out = dict(record)
    old_width = _coerce_positive_int(out.get("width")) or 1000
    old_height = _coerce_positive_int(out.get("height")) or 1000

    def _scale_points(points: Any) -> Any:
        if not isinstance(points, list) or len(points) % 2 != 0:
            return points
        scaled: List[int] = []
        for idx, value in enumerate(points):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return points
            if idx % 2 == 0:
                scaled.append(int(round(number * float(width) / float(old_width))))
            else:
                scaled.append(int(round(number * float(height) / float(old_height))))
        return denorm_and_clamp(scaled, float(width), float(height), coord_mode="pixel")

    for key in ("gt", "pred"):
        objects: List[Dict[str, Any]] = []
        for obj in out.get(key) or []:
            if not isinstance(obj, Mapping):
                continue
            copied = dict(obj)
            if isinstance(copied.get("points"), list):
                copied["points"] = _scale_points(copied.get("points"))
            if isinstance(copied.get("bbox_2d"), list):
                copied["bbox_2d"] = _scale_points(copied.get("bbox_2d"))
            objects.append(copied)
        out[key] = objects

    out["width"] = int(width)
    out["height"] = int(height)
    return out


def _enrich_stage2_eval_artifact_source_provenance(
    artifact: Mapping[str, Any],
    *,
    source_rows_by_base_idx: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    out = dict(artifact)
    try:
        base_idx = int(out.get("base_idx"))
    except (TypeError, ValueError):
        return out
    source = source_rows_by_base_idx.get(int(base_idx))
    if not isinstance(source, Mapping):
        return out

    width = _coerce_positive_int(source.get("width"))
    height = _coerce_positive_int(source.get("height"))
    image = _first_image_from_source_row(source)
    if width is None or height is None or image is None:
        return out

    provenance = {
        "source_jsonl": str(source.get("_source_jsonl") or ""),
        "source_jsonl_dir": str(source.get("_source_jsonl_dir") or ""),
        "base_idx": int(base_idx),
        "stage2_eval_source_enriched": True,
    }
    provenance = {k: v for k, v in provenance.items() if v not in ("", None)}

    for record_key in ("base_record", "scored_record"):
        record = out.get(record_key)
        if not isinstance(record, Mapping):
            continue
        enriched = _rescale_stage2_eval_record_geometry(
            record,
            width=int(width),
            height=int(height),
        )
        enriched["image"] = str(image)
        enriched["images"] = [str(image)]
        enriched["file_name"] = str(source.get("file_name") or image)
        if source.get("image_id") is not None:
            enriched["image_id"] = source.get("image_id")
        existing_provenance = dict(enriched.get("provenance") or {})
        existing_provenance.update(provenance)
        enriched["provenance"] = existing_provenance
        out[record_key] = enriched

    out["image"] = str(image)
    out["images"] = [str(image)]
    out["width"] = int(width)
    out["height"] = int(height)
    if source.get("image_id") is not None:
        out["image_id"] = source.get("image_id")
    metadata = dict(out.get("metadata") or {})
    source_metadata = source.get("metadata")
    if isinstance(source_metadata, Mapping):
        metadata.update(dict(source_metadata))
    metadata.setdefault("source_jsonl", str(source.get("_source_jsonl") or ""))
    metadata.setdefault("base_idx", int(base_idx))
    out["metadata"] = {k: v for k, v in metadata.items() if v not in ("", None)}
    return out


def build_eval_detection_record(
    *,
    sample: Mapping[str, Any],
    gts: Sequence[GTObject],
    preds: Sequence[GTObject],
    pred_meta: Sequence[ParsedPredObject],
    object_field_order: Literal["desc_first", "geometry_first"],
    record_index: int,
    pred_score_source: str,
    pred_score_version: int,
    score_mode: str,
    constant_score: float,
    raw_text: str | None,
    error_codes: Sequence[str],
    error_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    out = build_eval_detection_record_confidence_postop_input(
        sample=sample,
        gts=gts,
        preds=preds,
        pred_meta=pred_meta,
        object_field_order=object_field_order,
        record_index=record_index,
        raw_text=raw_text,
        error_codes=error_codes,
        error_entries=error_entries,
    )

    score_mode_norm = str(score_mode or "constant").strip().lower()
    score_const = float(constant_score)
    pred_payload: List[Dict[str, Any]] = []
    for payload in list(out.get("pred", [])):
        payload = dict(payload)
        if score_mode_norm == "constant":
            payload["score"] = float(score_const)
        pred_payload.append(payload)
    out["pred"] = pred_payload
    out["pred_score_source"] = str(pred_score_source)
    out["pred_score_version"] = int(pred_score_version)
    return out


def build_eval_detection_record_confidence_postop_input(
    *,
    sample: Mapping[str, Any],
    gts: Sequence[GTObject],
    preds: Sequence[GTObject],
    pred_meta: Sequence[ParsedPredObject],
    object_field_order: Literal["desc_first", "geometry_first"],
    record_index: int,
    raw_text: str | None,
    error_codes: Sequence[str],
    error_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build an eval-step record compatible with confidence_postop and offline infer."""

    from src.common.geometry import denorm_and_clamp

    images_raw = sample.get("images")
    images: List[str] = []
    if isinstance(images_raw, list):
        for v in images_raw:
            if isinstance(v, str) and v.strip():
                images = [str(v)]
                break
    if not images:
        image_one = sample.get("image")
        if isinstance(image_one, str) and image_one.strip():
            images = [str(image_one)]
    if not images:
        images = [f"image_{int(record_index)}.jpg"]

    width = sample.get("width")
    height = sample.get("height")
    try:
        width = int(width) if width is not None else None
    except (TypeError, ValueError):
        width = None
    try:
        height = int(height) if height is not None else None
    except (TypeError, ValueError):
        height = None
    if width is None:
        width = 1000
    if height is None:
        height = 1000

    def _normalize_geometry_key(value: Any) -> str:
        key = str(value or "").strip().lower()
        if key == "bbox":
            return "bbox_2d"
        return key

    gt_payload: List[Dict[str, Any]] = []
    for obj in gts:
        gtype = _normalize_geometry_key(getattr(obj, "geom_type", ""))
        pts_px = denorm_and_clamp(
            [int(x) for x in obj.points_norm1000],
            float(width),
            float(height),
            coord_mode="norm1000",
        )
        gt_payload.append(
            {
                "type": gtype,
                "points": pts_px,
                "desc": str(getattr(obj, "desc", "") or "").strip(),
                "score": 1.0,
            }
        )

    pred_payload: List[Dict[str, Any]] = []
    raw_objects: List[Dict[str, Any]] = []
    for idx, pobj in enumerate(preds):
        desc = ""
        if idx < len(pred_meta):
            desc = str(getattr(pred_meta[idx], "desc", "") or "").strip()
        gtype = _normalize_geometry_key(getattr(pobj, "geom_type", ""))
        pts_norm = [int(x) for x in pobj.points_norm1000]
        pts_px = denorm_and_clamp(
            pts_norm,
            float(width),
            float(height),
            coord_mode="norm1000",
        )
        pred_payload.append(
            {
                "type": gtype,
                "points": pts_px,
                "desc": desc,
            }
        )
        # raw_output_json must preserve coord bins (0..999), not pixel points.
        try:
            raw_objects.append(
                build_object_payload(
                    desc=desc,
                    geometry_key=gtype,
                    geometry_value=pts_norm,
                    object_field_order=object_field_order,
                )
            )
        except Exception:
            raw_objects.append({"type": gtype, "points": pts_norm, "desc": desc})

    raw_text_value = str(raw_text or "")
    raw_output_json = load_prediction_dict(raw_text_value)
    if raw_output_json is None:
        raw_output_json = {"objects": raw_objects}

    errors_payload = [str(code) for code in list(error_codes)]
    error_entries_payload: List[Dict[str, Any]] = []
    for entry in error_entries:
        if not isinstance(entry, Mapping):
            continue
        error_entries_payload.append(
            {
                "code": str(entry.get("code", "") or ""),
                "message": str(entry.get("message", "") or ""),
                "stage": str(entry.get("stage", "") or ""),
            }
        )

    image_value = images[0] if images else f"image_{int(record_index)}.jpg"
    out: Dict[str, Any] = {
        "index": int(record_index),
        "image": image_value,
        "mode": "text",
        "coord_mode": "pixel",
        "images": images,
        "gt": gt_payload,
        "pred": pred_payload,
        "width": int(width),
        "height": int(height),
        "raw_output_json": raw_output_json,
        "raw_special_tokens": extract_special_tokens(
            raw_text_value,
            preserve_duplicates=True,
        ),
        "raw_ends_with_im_end": raw_text_value.endswith(_IM_END),
        "errors": errors_payload,
        "error_entries": error_entries_payload,
    }
    if sample.get("image_id") is not None:
        out["image_id"] = sample.get("image_id")
    metadata = sample.get("metadata")
    if isinstance(metadata, Mapping):
        out["metadata"] = dict(metadata)
    return out


def extract_eval_gt_objects(sample: Mapping[str, Any]) -> List[GTObject]:
    payload = sample.get("assistant_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("rollout-matching requires assistant_payload in each sample")
    objects_raw = payload.get("objects")
    if not isinstance(objects_raw, list):
        raise ValueError("assistant_payload must contain top-level 'objects' list")

    objs: List[GTObject] = []
    for idx, entry in enumerate(objects_raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"assistant_payload.objects[{int(idx)}] must be a mapping")
        desc = entry.get("desc")
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].desc must be a non-empty string"
            )
        geom_keys = [
            k for k in ("bbox_2d", "poly") if k in entry and entry[k] is not None
        ]
        if len(geom_keys) != 1:
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}] must contain exactly one geometry key (bbox_2d|poly)"
            )
        geom_key = geom_keys[0]
        raw_pts = flatten_points(entry.get(geom_key))
        if raw_pts is None or len(raw_pts) % 2 != 0:
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].{geom_key} must be a flat even-length sequence"
            )
        pts: List[int] = []
        ok = True
        for v in raw_pts:
            if isinstance(v, str) and v.startswith("<|coord_"):
                try:
                    pts.append(int(token_to_int(v)))
                except (TypeError, ValueError):
                    ok = False
                    break
            else:
                vi = _coerce_int(v)
                if vi is None:
                    ok = False
                    break
                pts.append(int(vi))
        if not ok:
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].{geom_key} contains invalid coordinate values"
            )
        if geom_key == "bbox_2d" and len(pts) != 4:
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].bbox_2d must contain exactly 4 coordinates"
            )
        if geom_key == "poly" and (len(pts) < 6 or len(pts) % 2 != 0):
            raise ValueError(
                f"assistant_payload.objects[{int(idx)}].poly must contain >=6 coordinates and even arity"
            )
        objs.append(
            GTObject(
                index=int(idx),
                geom_type=geom_key,
                points_norm1000=pts,
                desc=desc.strip(),
            )
        )
    return objs


def _build_stage2_eval_infer_summary(
    *,
    owner: Any,
    eval_prompt_variant: str | None,
    eval_rollout_backend: str,
    eval_vllm_mode: str,
    eval_detection_score_mode: str,
    eval_detection_cfg: Mapping[str, Any],
    eval_decode_override: Optional[Mapping[str, Any]],
    sample_count: int,
    trace_count: int,
) -> Dict[str, Any]:
    generation = {
        "decode_mode": str(owner._cfg("decode_mode", "greedy")),
        "max_new_tokens": int(owner._cfg("max_new_tokens", 0) or 0),
    }
    if isinstance(eval_decode_override, Mapping):
        generation["decode_mode"] = str(
            eval_decode_override.get("decode_mode", generation["decode_mode"])
        )
        if "temperature" in eval_decode_override:
            generation["temperature"] = float(eval_decode_override["temperature"])
        if "top_p" in eval_decode_override:
            generation["top_p"] = float(eval_decode_override["top_p"])
        if "top_k" in eval_decode_override:
            generation["top_k"] = int(eval_decode_override["top_k"])
    return {
        "mode": "stage2_eval_rollout",
        "backend": {
            "type": str(eval_rollout_backend),
            "vllm_mode": str(eval_vllm_mode),
        },
        "generation": generation,
        "infer": {
            "prompt_variant": str(eval_prompt_variant or ""),
            "object_field_order": str(owner._object_field_order()),
            "object_ordering": str(owner._object_ordering()),
            "limit": int(sample_count),
        },
        "eval_detection": {
            "score_mode": str(eval_detection_score_mode),
            "metrics": str(eval_detection_cfg.get("metrics", "coco") or "coco"),
            "trace_count": int(trace_count),
        },
        "counters": {
            "records": int(sample_count),
            "trace_records": int(trace_count),
        },
    }


def _build_eval_options(eval_cfg: Mapping[str, Any], *, output_dir: Path) -> EvalOptions:
    def _coerce_optional_float_list(raw: Any) -> list[float] | None:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            out: list[float] = []
            for value in raw:
                out.append(float(value))
            return out
        return [float(raw)]

    iou_thrs = _coerce_optional_float_list(eval_cfg.get("iou_thrs", None))
    f1ish_iou_thrs = _coerce_optional_float_list(
        eval_cfg.get("f1ish_iou_thrs", [0.3, 0.5])
    )
    if f1ish_iou_thrs is None:
        f1ish_iou_thrs = [0.3, 0.5]
    return EvalOptions(
        metrics=str(eval_cfg.get("metrics", "coco") or "coco"),
        strict_parse=bool(eval_cfg.get("strict_parse", True)),
        use_segm=bool(eval_cfg.get("use_segm", False)),
        iou_thrs=iou_thrs,
        f1ish_iou_thrs=[float(x) for x in f1ish_iou_thrs],
        f1ish_pred_scope=str(eval_cfg.get("f1ish_pred_scope", "annotated") or "annotated"),
        output_dir=output_dir,
        overlay=False,
        overlay_k=0,
        num_workers=0,
        semantic_model=str(
            eval_cfg.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
            or "sentence-transformers/all-MiniLM-L6-v2"
        ),
        semantic_threshold=float(eval_cfg.get("semantic_threshold", 0.6) or 0.6),
        semantic_device=str(eval_cfg.get("semantic_device", "auto") or "auto"),
        semantic_batch_size=int(eval_cfg.get("semantic_batch_size", 64) or 64),
        lvis_max_dets=int(eval_cfg.get("lvis_max_dets", 300) or 300),
    )


def _materialize_stage2_eval_artifacts(
    *,
    owner: Any,
    global_step: int,
    eval_rollout_artifacts_all: List[Dict[str, Any]],
    eval_prompt_variant: str | None,
    eval_rollout_backend: str,
    eval_vllm_mode: str,
    eval_detection_score_mode: str,
    eval_detection_cfg: Mapping[str, Any],
    eval_decode_override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    eval_dir = _stage2_eval_output_dir(owner=owner, global_step=global_step)
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_rows: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    source_rows_by_base_idx = _load_stage2_eval_source_rows_by_base_idx(owner)
    for record_idx, artifact in enumerate(eval_rollout_artifacts_all):
        artifact = _enrich_stage2_eval_artifact_source_provenance(
            artifact,
            source_rows_by_base_idx=source_rows_by_base_idx,
        )
        base_record = dict(artifact.get("base_record", {}))
        scored_record = dict(artifact.get("scored_record", {}))
        base_record["index"] = int(record_idx)
        scored_record["index"] = int(record_idx)
        base_rows.append(base_record)
        scored_rows.append(scored_record)

        rollout = artifact.get("rollout", {})
        if isinstance(rollout, Mapping):
            generated_token_text = rollout.get("generated_token_text")
            token_logprobs = rollout.get("token_logprobs")
            if isinstance(generated_token_text, list) and isinstance(token_logprobs, list):
                trace_rows.append(
                    {
                        "line_idx": int(record_idx),
                        "generated_token_text": list(generated_token_text),
                        "token_logprobs": [float(x) for x in token_logprobs],
                    }
                )

        artifact_row = dict(artifact)
        artifact_row["index"] = int(record_idx)
        artifact_row["base_record"] = base_record
        artifact_row["scored_record"] = scored_record
        raw_rows.append(artifact_row)

    _write_jsonl(eval_dir / "gt_vs_pred.jsonl", base_rows)
    _write_jsonl(eval_dir / "gt_vs_pred_scored.jsonl", scored_rows)
    _write_jsonl(eval_dir / "raw_rollouts.jsonl", raw_rows)
    if trace_rows:
        _write_jsonl(eval_dir / "pred_token_trace.jsonl", trace_rows)

    resolved_config_path = (
        Path(str(getattr(getattr(owner, "args", None), "output_dir", "")))
        / "resolved_config.json"
    )
    if resolved_config_path.is_file():
        (eval_dir / "resolved_config.path").write_text(
            str(resolved_config_path.resolve()),
            encoding="utf-8",
        )

    infer_summary = _build_stage2_eval_infer_summary(
        owner=owner,
        eval_prompt_variant=eval_prompt_variant,
        eval_rollout_backend=eval_rollout_backend,
        eval_vllm_mode=eval_vllm_mode,
        eval_detection_score_mode=eval_detection_score_mode,
        eval_detection_cfg=eval_detection_cfg,
        eval_decode_override=eval_decode_override,
        sample_count=len(base_rows),
        trace_count=len(trace_rows),
    )
    (eval_dir / "infer_summary.json").write_text(
        json.dumps(infer_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_stage2_eval_score_provenance(
        owner=owner,
        scored_path=eval_dir / "gt_vs_pred_scored.jsonl",
        raw_path=eval_dir / "gt_vs_pred.jsonl",
        eval_prompt_variant=eval_prompt_variant,
        eval_rollout_backend=eval_rollout_backend,
        eval_vllm_mode=eval_vllm_mode,
        eval_detection_score_mode=eval_detection_score_mode,
        eval_detection_cfg=eval_detection_cfg,
        eval_decode_override=eval_decode_override,
    )

    options = _build_eval_options(eval_detection_cfg, output_dir=eval_dir)
    return evaluate_and_save(eval_dir / "gt_vs_pred_scored.jsonl", options=options)


def finalize_rollout_aligned_evaluation(
    *,
    owner: Any,
    logger: Any,
    metric_key_prefix: str,
    eval_prompt_variant: str | None,
    eval_detection_enabled: bool,
    eval_detection_use_confidence_postop: bool,
    eval_detection_score_mode: str,
    eval_detection_score_version: int,
    eval_detection_score_source: str,
    eval_detection_cfg: Mapping[str, Any],
    desc_enabled: bool,
    n_samples: float,
    gt_total: float,
    pred_total: float,
    matched_total: float,
    fp_total: float,
    fn_total: float,
    gating_rejections_total: float,
    dropped_invalid_total: float,
    dropped_ambiguous_total: float,
    trunc_samples: float,
    matched_iou_sum: float,
    matched_iou_count: float,
    n_samples_valid_pred: float,
    n_samples_any_match: float,
    n_steps: float,
    desc_pairs_total: float,
    desc_exact_ok_total: float,
    desc_sem_ok_total: float,
    desc_sem_sim_sum_total: float,
    desc_sem_sim_count_total: float,
    sem_loaded_local: float,
    vllm_decode_error_count_local: float,
    runtime_local_s: float,
    eval_detection_records_local: List[Dict[str, Any]],
    eval_rollout_artifacts_local: List[Dict[str, Any]],
    do_dump: bool,
    dump_fail_samples: List[Dict[str, Any]],
    dump_other_samples: List[Dict[str, Any]],
    dump_max_samples: int,
    gs: int,
    eval_rollout_backend: str,
    eval_vllm_mode: str,
    top_k: int,
    gate_thr: float,
    mask_res: int,
    fp_cost: float,
    fn_cost: float,
    was_training: bool,
    metric_name_matches_key_fn: Any,
    stage2_eval_metric_key_fn: Any,
) -> Dict[str, float]:
    try:
        import torch.distributed as dist
    except (TypeError, ValueError):
        dist = None  # type: ignore[assignment]

    world_size = 1
    rank = 0
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = int(dist.get_world_size())
        rank = int(dist.get_rank())

    sums_t = torch.tensor(
        [
            n_samples,
            gt_total,
            pred_total,
            matched_total,
            fp_total,
            fn_total,
            gating_rejections_total,
            dropped_invalid_total,
            dropped_ambiguous_total,
            trunc_samples,
            matched_iou_sum,
            matched_iou_count,
            n_samples_valid_pred,
            n_samples_any_match,
            n_steps,
            desc_pairs_total,
            desc_exact_ok_total,
            desc_sem_ok_total,
            desc_sem_sim_sum_total,
            desc_sem_sim_count_total,
            sem_loaded_local,
            vllm_decode_error_count_local,
        ],
        device=owner.model.device,
        dtype=torch.float64,
    )
    rt_t = torch.tensor(
        [float(runtime_local_s)], device=owner.model.device, dtype=torch.float64
    )
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(rt_t, op=dist.ReduceOp.MAX)

    (
        n_samples,
        gt_total,
        pred_total,
        matched_total,
        fp_total,
        fn_total,
        gating_rejections_total,
        dropped_invalid_total,
        dropped_ambiguous_total,
        trunc_samples,
        matched_iou_sum,
        matched_iou_count,
        n_samples_valid_pred,
        n_samples_any_match,
        n_steps,
        desc_pairs_total,
        desc_exact_ok_total,
        desc_sem_ok_total,
        desc_sem_sim_sum_total,
        desc_sem_sim_count_total,
        sem_loaded_sum,
        vllm_decode_error_count_local,
    ) = [float(x.item()) for x in sums_t]
    runtime = float(rt_t.item())

    precision = (matched_total / pred_total) if pred_total > 0 else 0.0
    recall = (matched_total / gt_total) if gt_total > 0 else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    def _k(suffix: str) -> str:
        return stage2_eval_metric_key_fn(
            metric_key_prefix=str(metric_key_prefix), suffix=str(suffix)
        )

    metrics: Dict[str, float] = {}
    metrics[_k("time/runtime_s")] = float(runtime)
    if runtime > 0:
        metrics[_k("time/samples_per_s")] = float(n_samples / runtime)
        metrics[_k("time/steps_per_s")] = float(n_steps / runtime)

    metrics[_k("rollout/precision")] = float(precision)
    metrics[_k("rollout/recall")] = float(recall)
    metrics[_k("rollout/f1")] = float(f1)

    metrics[_k("rollout/pred_objects")] = float(pred_total)
    metrics[_k("rollout/gt_objects_total")] = float(gt_total)
    metrics[_k("rollout/matched")] = float(matched_total)
    metrics[_k("rollout/fp_total")] = float(fp_total)
    metrics[_k("rollout/fn_total")] = float(fn_total)
    metrics[_k("rollout/gating_rejections")] = float(gating_rejections_total)

    metrics[_k("rollout/parse_dropped_invalid")] = float(dropped_invalid_total)
    metrics[_k("rollout/parse_dropped_ambiguous")] = float(dropped_ambiguous_total)
    metrics[_k("rollout/parse_truncated_rate")] = (
        float(trunc_samples / n_samples) if n_samples > 0 else 0.0
    )

    metrics[_k("rollout/sample_valid_pred_rate")] = (
        float(n_samples_valid_pred / n_samples) if n_samples > 0 else 0.0
    )
    metrics[_k("rollout/sample_any_match_rate")] = (
        float(n_samples_any_match / n_samples) if n_samples > 0 else 0.0
    )

    metrics[_k("rollout/matched_maskiou_mean")] = (
        float(matched_iou_sum / matched_iou_count) if matched_iou_count > 0 else 0.0
    )
    metrics[_k("rollout/vllm_decode_error_count")] = float(
        vllm_decode_error_count_local
    )

    if desc_enabled:
        metrics[_k("rollout/desc_pairs_total")] = float(desc_pairs_total)
        exact_acc = (
            float(desc_exact_ok_total / desc_pairs_total)
            if desc_pairs_total > 0
            else 1.0
        )
        metrics[_k("rollout/desc_exact_acc_on_matched")] = float(exact_acc)

        sem_enabled = bool(sem_loaded_sum >= float(world_size) - 0.5)
        metrics[_k("rollout/desc_sem_enabled")] = float(1.0 if sem_enabled else 0.0)
        if sem_enabled:
            sem_acc = (
                float(desc_sem_ok_total / desc_pairs_total)
                if desc_pairs_total > 0
                else 1.0
            )
            metrics[_k("rollout/desc_sem_acc_on_matched")] = float(sem_acc)
            if desc_sem_sim_count_total > 0:
                metrics[_k("rollout/desc_sem_sim_mean")] = float(
                    desc_sem_sim_sum_total / desc_sem_sim_count_total
                )
                metrics[_k("rollout/desc_sem_sim_count")] = float(
                    desc_sem_sim_count_total
                )

    if eval_prompt_variant is not None:
        metrics[_k("rollout/prompt_variant_is_coco_80")] = (
            1.0 if str(eval_prompt_variant).strip().lower() == "coco_80" else 0.0
        )

    if eval_detection_enabled:
        cfg_mode = str(eval_detection_score_mode or "constant").strip().lower()
        metrics[_k("rollout/config_score_mode_is_constant")] = float(
            1.0 if cfg_mode == "constant" else 0.0
        )
        metrics[_k("rollout/config_score_mode_is_confidence_postop")] = float(
            1.0 if cfg_mode in {"confidence_postop", "confidence"} else 0.0
        )

        effective_confidence_postop = bool(eval_detection_use_confidence_postop)
        eff_mode = "confidence_postop" if effective_confidence_postop else "constant"
        metrics[_k("rollout/effective_score_mode_is_constant")] = float(
            1.0 if eff_mode == "constant" else 0.0
        )
        metrics[_k("rollout/effective_score_mode_is_confidence_postop")] = float(
            1.0 if eff_mode == "confidence_postop" else 0.0
        )

        metrics[_k("rollout/config_pred_score_version")] = float(
            int(eval_detection_score_version)
        )
        eff_version = (
            2 if effective_confidence_postop else int(eval_detection_score_version)
        )
        metrics[_k("rollout/effective_pred_score_version")] = float(int(eff_version))

        cfg_source = str(eval_detection_score_source or "").strip()
        eff_source = "confidence_postop" if effective_confidence_postop else cfg_source
        metrics[_k("rollout/config_pred_score_source_is_eval_rollout_constant")] = float(
            1.0 if cfg_source == "eval_rollout_constant" else 0.0
        )
        metrics[_k("rollout/config_pred_score_source_is_confidence_postop")] = float(
            1.0 if cfg_source == "confidence_postop" else 0.0
        )
        metrics[
            _k("rollout/effective_pred_score_source_is_eval_rollout_constant")
        ] = float(1.0 if eff_source == "eval_rollout_constant" else 0.0)
        metrics[_k("rollout/effective_pred_score_source_is_confidence_postop")] = float(
            1.0 if eff_source == "confidence_postop" else 0.0
        )

    if eval_detection_enabled:
        eval_records_all: List[Dict[str, Any]] = [
            dict(record) for record in eval_detection_records_local
        ]
        eval_rollout_artifacts_all: List[Dict[str, Any]] = [
            dict(record) for record in eval_rollout_artifacts_local
        ]
        if dist is not None and dist.is_available() and dist.is_initialized():
            eval_records_all = []
            eval_rollout_artifacts_all = []
            gather_object = getattr(dist, "gather_object", None)
            if callable(gather_object):
                gathered_records = (
                    [None for _ in range(int(world_size))] if int(rank) == 0 else None
                )
                gathered_rollout_artifacts = (
                    [None for _ in range(int(world_size))] if int(rank) == 0 else None
                )
                try:
                    gather_object(
                        list(eval_detection_records_local),
                        object_gather_list=gathered_records,
                        dst=0,
                    )
                except TypeError:
                    gather_object(
                        list(eval_detection_records_local),
                        gathered_records,
                        0,
                    )
                try:
                    gather_object(
                        list(eval_rollout_artifacts_local),
                        object_gather_list=gathered_rollout_artifacts,
                        dst=0,
                    )
                except TypeError:
                    gather_object(
                        list(eval_rollout_artifacts_local),
                        gathered_rollout_artifacts,
                        0,
                    )
                if int(rank) == 0 and isinstance(gathered_records, list):
                    for src_rank, part in enumerate(gathered_records):
                        if not isinstance(part, list):
                            raise TypeError(
                                "eval_detection gather_object returned non-list part: "
                                f"src_rank={int(src_rank)} type={type(part).__name__}"
                            )
                        for rec in part:
                            if isinstance(rec, Mapping):
                                eval_records_all.append(dict(rec))
                if int(rank) == 0 and isinstance(gathered_rollout_artifacts, list):
                    for src_rank, part in enumerate(gathered_rollout_artifacts):
                        if not isinstance(part, list):
                            raise TypeError(
                                "eval_rollout_artifacts gather_object returned non-list part: "
                                f"src_rank={int(src_rank)} type={type(part).__name__}"
                            )
                        for rec in part:
                            if isinstance(rec, Mapping):
                                eval_rollout_artifacts_all.append(dict(rec))
            else:
                gathered_records = [None for _ in range(int(world_size))]
                gathered_rollout_artifacts = [None for _ in range(int(world_size))]
                dist.all_gather_object(
                    gathered_records, list(eval_detection_records_local)
                )
                dist.all_gather_object(
                    gathered_rollout_artifacts,
                    list(eval_rollout_artifacts_local),
                )
                for src_rank, part in enumerate(gathered_records):
                    if not isinstance(part, list):
                        raise TypeError(
                            "eval_detection all_gather_object returned non-list part: "
                            f"src_rank={int(src_rank)} type={type(part).__name__}"
                        )
                    for rec in part:
                        if isinstance(rec, Mapping):
                            eval_records_all.append(dict(rec))
                for src_rank, part in enumerate(gathered_rollout_artifacts):
                    if not isinstance(part, list):
                        raise TypeError(
                            "eval_rollout_artifacts all_gather_object returned non-list part: "
                            f"src_rank={int(src_rank)} type={type(part).__name__}"
                        )
                    for rec in part:
                        if isinstance(rec, Mapping):
                            eval_rollout_artifacts_all.append(dict(rec))

        for record_idx, record in enumerate(eval_records_all):
            record["index"] = int(record_idx)

        eval_det_payload: Dict[str, Any] = {
            "ok": 0.0,
            "runtime_s": 0.0,
            "metrics": {},
            "counters": {},
            "error": "",
        }
        eval_det_exc: Exception | None = None
        if int(rank) == 0:
            t_coco0 = time.perf_counter()
            try:
                output_dir_raw = str(
                    getattr(getattr(owner, "args", None), "output_dir", "") or ""
                ).strip()
                should_materialize_artifacts = bool(
                    eval_detection_cfg.get("materialize_artifacts", True)
                )
                eval_decode_override = None
                eval_decode_override_fn = getattr(owner, "_eval_decode_override", None)
                if callable(eval_decode_override_fn):
                    eval_decode_override = eval_decode_override_fn(
                        has_token_trace=bool(eval_detection_use_confidence_postop)
                    )
                if output_dir_raw and should_materialize_artifacts:
                    eval_summary = _materialize_stage2_eval_artifacts(
                        owner=owner,
                        global_step=int(gs),
                        eval_rollout_artifacts_all=eval_rollout_artifacts_all,
                        eval_prompt_variant=eval_prompt_variant,
                        eval_rollout_backend=eval_rollout_backend,
                        eval_vllm_mode=eval_vllm_mode,
                        eval_detection_score_mode=eval_detection_score_mode,
                        eval_detection_cfg=eval_detection_cfg,
                        eval_decode_override=eval_decode_override,
                    )
                    coco_metrics = eval_summary.get("metrics", {})
                    coco_counters = eval_summary.get("counters", {})
                else:
                    raise ValueError(
                        "Stage-2 official eval requires "
                        "rollout_matching.eval_detection.materialize_artifacts=true "
                        "and training.output_dir so scored artifacts can be "
                        "provenance-checked before metric computation."
                    )
                eval_det_payload["ok"] = 1.0
                eval_det_payload["metrics"] = {
                    str(k): float(v) for k, v in coco_metrics.items()
                }
                eval_det_payload["counters"] = {
                    str(k): int(v)
                    for k, v in coco_counters.items()
                    if isinstance(v, (int, float))
                }
            except Exception as exc:
                eval_det_exc = exc
                eval_det_payload["error"] = repr(exc)
                logger.exception("Eval-step COCO/mAP failed")
            finally:
                eval_det_payload["runtime_s"] = float(time.perf_counter() - t_coco0)

        if dist is not None and dist.is_available() and dist.is_initialized():
            payload_list: List[Any] = [eval_det_payload]

            backend = None
            try:
                backend = str(dist.get_backend())
            except Exception:
                backend = None

            broadcast_device = torch.device("cpu")
            if backend == "nccl":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "torch.distributed backend is NCCL but CUDA is not available"
                    )
                broadcast_device = torch.device(
                    "cuda", int(torch.cuda.current_device())
                )

            try:
                dist.broadcast_object_list(
                    payload_list, src=0, device=broadcast_device
                )
            except TypeError as exc:
                if backend == "nccl":
                    raise RuntimeError(
                        "broadcast_object_list(..., device=...) is required for NCCL; "
                        "upgrade PyTorch to a version that supports the 'device' argument."
                    ) from exc
                dist.broadcast_object_list(payload_list, src=0)

            if not isinstance(payload_list[0], Mapping):
                raise TypeError(
                    "eval_detection broadcast payload is not a Mapping: "
                    f"type={type(payload_list[0]).__name__}"
                )
            recv_payload = dict(payload_list[0])
        else:
            recv_payload = eval_det_payload

        try:
            metrics[_k("time/coco_eval_runtime_s")] = float(
                recv_payload.get("runtime_s", 0.0) or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            metrics[_k("time/coco_eval_runtime_s")] = 0.0
        try:
            metrics[_k("rollout/coco_eval_ok")] = float(
                recv_payload.get("ok", 0.0) or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            metrics[_k("rollout/coco_eval_ok")] = 0.0

        metrics[_k("rollout/mAP")] = 0.0
        coco_metrics_recv = recv_payload.get("metrics", {})
        if isinstance(coco_metrics_recv, Mapping) and "bbox_AP" in coco_metrics_recv:
            try:
                metrics[_k("rollout/mAP")] = float(coco_metrics_recv["bbox_AP"])
            except (TypeError, ValueError):
                metrics[_k("rollout/mAP")] = 0.0

        metric_for_best_model = str(
            getattr(getattr(owner, "args", None), "metric_for_best_model", "") or ""
        ).strip()
        coco_ok = float(metrics.get(_k("rollout/coco_eval_ok"), 0.0) or 0.0)
        if coco_ok <= 0.0:
            coco_err = str(recv_payload.get("error", "") or "").strip()
            if metric_name_matches_key_fn(
                metric_for_best_model,
                stage2_eval_metric_key_fn("eval", "rollout/mAP"),
            ):
                msg = (
                    "Eval-step COCO/mAP failed while metric_for_best_model targets eval/detection/mAP; "
                    "aborting to avoid invalid best-checkpoint selection. "
                    f"error={coco_err or 'unknown'}"
                )
            else:
                msg = (
                    "Eval-step COCO/mAP failed; aborting (fail-fast) to avoid silent metric corruption. "
                    f"error={coco_err or 'unknown'}"
                )
            if int(rank) == 0 and eval_det_exc is not None:
                raise RuntimeError(msg) from eval_det_exc
            raise RuntimeError(msg)

        coco_counters_recv = recv_payload.get("counters", {})
        if isinstance(coco_counters_recv, Mapping):
            for name in (
                "empty_pred",
                "invalid_geometry",
                "invalid_coord",
                "missing_size",
                "size_mismatch",
                "degenerate",
                "unknown_dropped",
                "semantic_mapped",
                "semantic_unmapped",
            ):
                if name not in coco_counters_recv:
                    continue
                try:
                    metrics[_k(f"rollout/coco_counter_{name}")] = float(
                        coco_counters_recv[name]
                    )
                except (TypeError, ValueError):
                    continue

    if do_dump:
        try:
            samples_out: List[Dict[str, Any]] = list(dump_fail_samples)
            if len(samples_out) < dump_max_samples:
                need = int(dump_max_samples) - int(len(samples_out))
                samples_out.extend(list(dump_other_samples)[:need])

            payload = {
                "kind": "eval_monitor_dump",
                "global_step": int(gs),
                "epoch": float(
                    getattr(getattr(owner, "state", None), "epoch", 0.0) or 0.0
                ),
                "time": float(time.time()),
                "meta": {
                    "phase": "eval",
                    "metric_key_prefix": str(metric_key_prefix),
                    "rollout_backend": str(eval_rollout_backend),
                    "vllm_mode": str(eval_vllm_mode),
                    "decode_mode": str(owner._cfg("decode_mode", "greedy")),
                    "max_new_tokens": int(owner._cfg("max_new_tokens", 0) or 0),
                    "candidate_top_k": int(top_k),
                    "maskiou_gate": float(gate_thr),
                    "maskiou_resolution": int(mask_res),
                    "fp_cost": float(fp_cost),
                    "fn_cost": float(fn_cost),
                },
                "metrics": metrics,
                "samples": samples_out,
            }
            owner._write_monitor_dump(global_step=int(gs), payload=payload)
            owner._eval_monitor_dump_count += 1
        except Exception as exc:
            logger.warning(
                "Failed to write eval monitor dump at global_step=%s: %r",
                int(gs),
                exc,
            )

    owner.log(metrics)
    owner.control = owner.callback_handler.on_evaluate(
        owner.args, owner.state, owner.control, metrics
    )

    if was_training:
        owner.model.train()

    return metrics
