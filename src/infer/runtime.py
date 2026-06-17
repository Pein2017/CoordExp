from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, cast

from src.infer.backend import DetectionDecodeResult
from src.infer.constraints import (
    STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN,
    STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT,
    STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN,
    STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE,
    STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY,
    STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY,
    STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY,
    STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY,
    STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY,
    STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_OPEN,
)
from src.infer.parsing import (
    DecodedDetectionResult,
    diagnostic_parser_result,
    require_metric_bearing,
    strict_parser_result,
)
from src.tokens.coord.codec import sequence_has_coord_tokens

PromptTokenParity = Literal["verified", "unverified", "unverifiable"]
AllowedBBoxFormat = Literal["xyxy", "cxcy_logw_logh", "cxcywh"]
ObjectFieldOrder = Literal["desc_first", "geometry_first"]
ObjectOrdering = Literal["sorted", "random"]

COORDJSON_FORMAT = "coordjson"
DEFAULT_PROMPT_VARIANT = "default"
DEFAULT_BBOX_FORMAT: AllowedBBoxFormat = "xyxy"
_DISTRIBUTED_SOURCE_INDEX_KEY = "_coordexp_source_index"
_DISTRIBUTED_MANIFEST_TIMEOUT_S = 1800.0

# Test and integration seams for heavy/offline-only dependencies. These names
# intentionally default to None so importing this module stays lightweight.
AutoProcessor = None
Qwen3VLForConditionalGeneration = None
install_token_embeddings_adapter = None
reattach_token_embeddings_adapter_hooks = None
tqdm = None


def _semantic_template_id_from_sequence_format(detection_sequence_format: str) -> str:
    normalized = (
        str(detection_sequence_format).strip().lower().replace("-", "_").replace(" ", "_")
    )
    if normalized == COORDJSON_FORMAT:
        return "stage1_json_pretty"
    if normalized in {"compact", "compact_full"}:
        return "compact"
    raise ValueError(
        "inference checkpoint validation requires a semantic detection template; "
        f"unsupported detection_sequence_format={detection_sequence_format!r}"
    )


def _resolve_detection_template_contract(template_id: str) -> Any:
    from src.detection.template_contracts import resolve_detection_template_contract

    return resolve_detection_template_contract(template_id)


def _resolve_runtime_detection_template_id(cfg: Any) -> str:
    raw_template_id = getattr(cfg, "detection_template_id", None)
    if raw_template_id is not None:
        return _resolve_detection_template_contract(str(raw_template_id)).template_id
    return _resolve_detection_template_contract(
        _semantic_template_id_from_sequence_format(
            getattr(cfg, "detection_sequence_format", COORDJSON_FORMAT)
        )
    ).template_id


def _detection_sequence_format_for_template_id(template_id: str) -> str:
    contract = _resolve_detection_template_contract(template_id)
    return "compact" if contract.is_compact else COORDJSON_FORMAT


def _parser_mode_for_template_id(template_id: str) -> str:
    contract = _resolve_detection_template_contract(template_id)
    if not contract.is_compact:
        return "strict_expected"
    if contract.template_id == "compact":
        return "marker_delimited_strict"
    return "strict_expected"


# Map fine-grained error tags to canonical counter buckets.
ERROR_CANONICAL = {
    "geometry_keys": "invalid_geometry",
    "geometry_points": "invalid_geometry",
    "geometry_kind": "invalid_geometry",
    "bbox_points": "invalid_geometry",
    "poly_points": "invalid_geometry",
    "degenerate": "invalid_geometry",
    "coord_parse": "invalid_coord",
    "coord_range": "invalid_coord",
    "mode_gt_mismatch": "mode_gt_mismatch",
    "size_mismatch": "size_mismatch",
    "empty_pred": "empty_pred",
    "generation_failed": "generation_failed",
    "image_load_failed": "image_load_failed",
    "multi_image_not_supported": "multi_image_not_supported",
}


def _owner_instance_override(owner: Any, name: str) -> Any:
    try:
        raw = getattr(owner, "__dict__", {}).get(name)
    except (AttributeError, TypeError):
        return None
    return raw if callable(raw) else None


def rollout_owner_cfg(owner: Any, key: str, default: Any) -> Any:
    cfg_fn = getattr(owner, "_cfg", None)
    if callable(cfg_fn):
        return cfg_fn(key, default)
    cfg = getattr(owner, "rollout_matching_cfg", None)
    if not isinstance(cfg, Mapping):
        return default
    return cfg.get(str(key), default)


@dataclass(frozen=True)
class RolloutDecodeFacts:
    """Resolved Stage-2 rollout config facts consumed by decode policy mapping."""

    rollout_matching_cfg: Mapping[str, Any]


@dataclass(frozen=True)
class RolloutRuntimeFacts:
    """Resolved rollout runtime facts consumed by shared dispatch/decode helpers."""

    rollout_matching_cfg: Mapping[str, Any]
    context: Literal["train", "eval"]
    effective_backend: Literal["hf", "vllm"]
    decode_batch_size: int
    vllm_mode: Literal["colocate", "server"]


def resolve_rollout_decode_facts_from_owner(owner: Any) -> RolloutDecodeFacts:
    """Translate an owner-like Stage-2 trainer into decode-policy facts."""

    rollout_matching_cfg = getattr(owner, "rollout_matching_cfg", {}) or {}
    if not isinstance(rollout_matching_cfg, Mapping):
        raise TypeError("owner.rollout_matching_cfg must be a mapping")
    return RolloutDecodeFacts(rollout_matching_cfg=rollout_matching_cfg)


def resolve_rollout_runtime_facts_from_owner(
    owner: Any,
    *,
    context: Optional[Literal["train", "eval"]] = None,
    rollout_backend: Optional[Literal["hf", "vllm"]] = None,
) -> RolloutRuntimeFacts:
    """Translate an owner-like Stage-2 trainer into shared rollout facts."""

    rollout_context = (
        context if context is not None else current_rollout_context_from_owner(owner)
    )
    backend = (
        rollout_backend
        if rollout_backend is not None
        else effective_rollout_backend_from_owner(owner, context=rollout_context)
    )
    decode_facts = resolve_rollout_decode_facts_from_owner(owner)
    vllm_mode: Literal["colocate", "server"] = (
        vllm_mode_from_rollout_owner(owner) if backend == "vllm" else "colocate"
    )
    return RolloutRuntimeFacts(
        rollout_matching_cfg=decode_facts.rollout_matching_cfg,
        context=rollout_context,
        effective_backend=backend,
        decode_batch_size=rollout_decode_batch_size_from_owner(
            owner,
            context=rollout_context,
        ),
        vllm_mode=vllm_mode,
    )


def normalize_rollout_backend_value(
    raw: Any,
    *,
    key_path: str,
    allow_none: bool,
) -> Optional[Literal["hf", "vllm"]]:
    if raw is None:
        if allow_none:
            return None
        raise ValueError(f"{key_path} must be one of {{hf,vllm}}")
    value = str(raw).strip().lower()
    if allow_none and value in {"", "null", "none"}:
        return None
    if value not in {"hf", "vllm"}:
        allowed = "{null,hf,vllm}" if allow_none else "{hf,vllm}"
        raise ValueError(f"{key_path} must be one of {allowed}, got {raw!r}")
    return cast(Literal["hf", "vllm"], value)


def effective_rollout_backend_from_owner(
    owner: Any,
    *,
    context: Literal["train", "eval"] = "train",
) -> Literal["hf", "vllm"]:
    override = _owner_instance_override(owner, "_effective_rollout_backend")
    if override is not None:
        return cast(Literal["hf", "vllm"], override(context=context))

    train_backend = normalize_rollout_backend_value(
        rollout_owner_cfg(owner, "rollout_backend", "hf"),
        key_path="rollout_matching.rollout_backend",
        allow_none=False,
    )
    assert train_backend is not None

    eval_backend = normalize_rollout_backend_value(
        rollout_owner_cfg(owner, "eval_rollout_backend", "vllm"),
        key_path="rollout_matching.eval_rollout_backend",
        allow_none=False,
    )
    assert eval_backend is not None
    return train_backend if context == "train" else eval_backend


def current_rollout_context_from_owner(owner: Any) -> Literal["train", "eval"]:
    override = _owner_instance_override(owner, "_current_rollout_context")
    if override is not None:
        return cast(Literal["train", "eval"], override())

    model_obj = getattr(owner, "model", None)
    if model_obj is None:
        return "train"
    training_attr = getattr(model_obj, "training", True)
    return "train" if bool(training_attr) else "eval"


def vllm_mode_from_rollout_owner(owner: Any) -> Literal["colocate", "server"]:
    override = _owner_instance_override(owner, "_vllm_mode")
    if override is not None:
        return cast(Literal["colocate", "server"], override())

    vcfg_raw = rollout_owner_cfg(owner, "vllm", {}) or {}
    if not isinstance(vcfg_raw, Mapping):
        raise ValueError("rollout_matching.vllm must be a mapping")
    mode = str(vcfg_raw.get("mode", "colocate") or "colocate").strip().lower()
    if mode not in {"colocate", "server"}:
        raise ValueError(
            "rollout_matching.vllm.mode must be 'colocate' or 'server'; "
            f"got {mode!r}"
        )
    return cast(Literal["colocate", "server"], mode)


def rollout_decode_batch_size_from_owner(
    owner: Any,
    *,
    context: Literal["train", "eval"] = "train",
) -> int:
    override = _owner_instance_override(owner, "_decode_batch_size")
    if override is not None:
        return int(override(context=context))

    context_norm = str(context).strip().lower()
    if context_norm not in {"train", "eval"}:
        raise ValueError("decode batch-size context must be one of {'train', 'eval'}")

    if context_norm == "train":
        raw = rollout_owner_cfg(owner, "rollout_decode_batch_size", None)
        missing_msg = "rollout_matching.rollout_decode_batch_size must be provided explicitly"
        type_msg = "rollout_matching.rollout_decode_batch_size must be an int"
        positive_msg = "rollout_matching.rollout_decode_batch_size must be > 0"
    else:
        raw = rollout_owner_cfg(owner, "eval_decode_batch_size", None)
        missing_msg = "rollout_matching.eval_decode_batch_size must be provided explicitly"
        type_msg = "rollout_matching.eval_decode_batch_size must be an int"
        positive_msg = "rollout_matching.eval_decode_batch_size must be > 0"

    if raw is None:
        raise ValueError(missing_msg)

    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(type_msg) from exc

    if value <= 0:
        raise ValueError(positive_msg)

    return int(value)


def _flatten_gt_points(value: Any) -> list[Any] | None:
    """Flatten GT point containers without importing dataset geometry modules."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        flat: list[Any] = []
        for pair in value:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                return None
            flat.extend(pair)
        return flat
    return list(value)


def resolve_offline_image_path(owner: Any, jsonl_path: Path, image_rel: str) -> Path:
    """Resolve one offline inference image path with the canonical strict policy."""

    from src.common.paths import resolve_image_path_strict

    root_image_dir: Path | None = None
    root_raw = str(owner.cfg.root_image_dir or "").strip()
    if root_raw:
        root_image_dir = Path(root_raw).resolve()

    resolved = resolve_image_path_strict(
        str(image_rel),
        jsonl_dir=jsonl_path.parent,
        root_image_dir=root_image_dir,
    )
    if resolved is None:
        raise FileNotFoundError(
            "Image path does not exist after strict resolution: "
            f"image={image_rel!r} jsonl_dir={str(jsonl_path.parent)!r} "
            f"root_image_dir={str(root_image_dir) if root_image_dir is not None else None!r}"
        )
    return resolved


def prepare_offline_image(
    owner: Any,
    jsonl_path: Path,
    record: Mapping[str, Any],
    *,
    strict_decode: bool = True,
):
    """Load the single RGB image used by offline inference/eval rows."""

    from PIL import Image

    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(
            "infer input record must contain exactly one image in `images`: "
            f"got images={images!r}"
        )
    image_field = images[0]
    if not isinstance(image_field, str) or not image_field.strip():
        raise ValueError(
            "infer input record has invalid image field in `images[0]`: "
            f"got {image_field!r}"
        )

    img_path = resolve_offline_image_path(owner, jsonl_path, image_field)
    try:
        image = Image.open(img_path).convert("RGB")
    except (OSError, ValueError) as exc:
        if strict_decode:
            raise ValueError(
                f"Failed to open image at {img_path}: {exc.__class__.__name__}: {exc}"
            ) from exc
        return img_path, None
    width_raw = record.get("width")
    height_raw = record.get("height")
    if width_raw is not None and height_raw is not None:
        try:
            expected_size = (int(width_raw), int(height_raw))
        except (TypeError, ValueError):
            expected_size = None
        if expected_size is not None and image.size != expected_size:
            raise ValueError(
                "Image size does not match record width/height: "
                f"image={str(img_path)!r} actual={image.size!r} expected={expected_size!r}"
            )
    return img_path, image


def process_offline_gt(
    owner: Any,
    record: Mapping[str, Any],
    *,
    width: int,
    height: int,
    errors: List[str],
) -> List[Dict[str, Any]]:
    """Convert offline GT objects through the owner coordinate standardizer."""

    return owner.coord.process_record_gt(record, width=width, height=height, errors=errors)


def _strip_generation_terminal_preserving_template_text(
    text: str,
) -> tuple[str, str | None]:
    from src.common.detection_sequence import END_OF_TEXT_TOKEN, IM_END_TOKEN

    im_end_pos = text.find(IM_END_TOKEN)
    if im_end_pos >= 0:
        return text[:im_end_pos], IM_END_TOKEN
    effective = str(text)
    terminal_token: str | None = None
    while effective.endswith(END_OF_TEXT_TOKEN):
        effective = effective[: -len(END_OF_TEXT_TOKEN)]
        terminal_token = END_OF_TEXT_TOKEN
    return effective, terminal_token


def parse_detection_template_output_artifact(
    text: str,
    *,
    detection_template_id: str,
    object_field_order: str = "desc_first",
) -> Dict[str, Any]:
    contract = _resolve_detection_template_contract(detection_template_id)
    if not contract.is_compact:
        raise ValueError(
            "parse_detection_template_output_artifact only handles compact templates"
        )

    from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
    from src.detection.evaluation import parse_detection_output_strict_expected

    effective_text, terminal_token = _strip_generation_terminal_preserving_template_text(
        str(text)
    )
    try:
        raw_output_json = parse_detection_output_strict_expected(
            effective_text,
            expected_template=contract.template_id,
            parser_mode="strict_expected",
            object_field_order=object_field_order,
        )
        parse_error_code = None
    except ValueError:
        raw_output_json = None
        parse_error_code = "strict_template_mismatch"
    return {
        "raw_output_json": raw_output_json,
        "parse_mode": "strict_expected",
        "serialization_policy": contract.template_id,
        "object_field_order": object_field_order,
        "object_separator": (
            "\n"
            if contract.canonical_final_separator == "\n"
            else (
                BOX_START_TOKEN
                if str(object_field_order) == "geometry_first"
                else OBJECT_REF_START_TOKEN
            )
        ),
        "terminal_token": terminal_token,
        "parse_error_code": parse_error_code,
        "parse_error_offset": None,
        "detection_template_id": contract.template_id,
    }


def process_offline_pred(
    owner: Any,
    raw_text: str,
    *,
    width: int,
    height: int,
    errors: List[str],
    compact_parse_artifact: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Parse generated text into pixel-space prediction objects."""

    if getattr(owner, "detection_template_contract", None) is not None:
        contract = owner.detection_template_contract
    else:
        contract = _resolve_detection_template_contract(
            getattr(owner, "detection_template_id", "stage1_json_pretty")
        )
    if contract.is_compact:
        artifact = compact_parse_artifact or parse_detection_template_output_artifact(
            raw_text,
            detection_template_id=contract.template_id,
            object_field_order=str(getattr(owner, "object_field_order", "desc_first")),
        )
        payload = artifact.get("raw_output_json")
        if not isinstance(payload, Mapping):
            errors.append("empty_pred")
            return []
        objects = payload.get("objects")
        if not isinstance(objects, list) or not objects:
            errors.append("empty_pred")
            return []
        preds = owner.coord.process_objects(
            objects,
            width=width,
            height=height,
            is_gt=False,
            errors=errors,
        )
        if not preds and "empty_pred" not in errors:
            errors.append("empty_pred")
        return preds
    return owner.coord.process_prediction_text(
        raw_text, width=width, height=height, errors=errors
    )


def decode_offline_detection_result(
    owner: Any,
    raw_text: str,
    *,
    width: int,
    height: int,
    compact_parse_artifact: Optional[Dict[str, Any]] = None,
) -> DecodedDetectionResult:
    """Decode one generated detection output into the canonical parse object."""

    pred_errors: List[str] = []
    pred = process_offline_pred(
        owner,
        raw_text,
        width=width,
        height=height,
        errors=pred_errors,
        compact_parse_artifact=compact_parse_artifact,
    )
    predictions = tuple(compact_gt_vs_pred_objects(pred))
    parser_id = str(getattr(owner, "detection_template_id", "stage1_json_pretty"))
    diagnostics: Dict[str, Any] = {
        "invalid_count": len(pred_errors),
        "dropped_invalid": len(pred_errors),
    }
    if compact_parse_artifact is not None:
        parse_error_code = compact_parse_artifact.get("parse_error_code")
        diagnostics["parse_error_code"] = parse_error_code
        diagnostics["parse_mode"] = compact_parse_artifact.get("parse_mode")
        if parse_error_code:
            pred_errors.append(str(parse_error_code))

    if pred_errors:
        return diagnostic_parser_result(
            predictions=predictions,
            parser_id=parser_id,
            errors=tuple(pred_errors),
            diagnostics=diagnostics,
            salvage_recovered=False,
        )
    return strict_parser_result(
        predictions=predictions,
        parser_id=parser_id,
        diagnostics=diagnostics,
    )


def compact_gt_vs_pred_objects(objs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Strip internal fields to the unified gt_vs_pred.jsonl object schema."""

    compact: List[Dict[str, Any]] = []
    for obj in objs:
        if not isinstance(obj, Mapping):
            continue
        kind = obj.get("type")
        points = obj.get("points")
        if kind not in {"bbox_2d", "poly"}:
            continue
        if not isinstance(points, list):
            continue
        desc = str(obj.get("desc", "") or "").strip()
        compact.append(
            {
                "type": kind,
                "points": points,
                "desc": desc,
            }
        )
    return compact


def materialize_offline_gt_vs_pred_record(
    *,
    image: str,
    width: int,
    height: int,
    mode: str,
    gt: Sequence[Mapping[str, Any]],
    decoded_result: DecodedDetectionResult,
    raw_output_json: Any,
    raw_special_tokens: Sequence[str],
    raw_ends_with_im_end: bool,
    errors: Sequence[str] | None = None,
    error_entries: Sequence[Mapping[str, Any]] | None = None,
    allow_diagnostic: bool = False,
) -> Dict[str, Any]:
    """Project a decoded detection result into the stable gt_vs_pred row schema."""

    if not allow_diagnostic:
        decoded_result = require_metric_bearing(
            decoded_result,
            consumer="official_gt_vs_pred_materialization",
        )
    parser_metadata = decoded_result.to_artifact_metadata()
    output = {
        "image": image,
        "width": int(width),
        "height": int(height),
        "mode": str(mode),
        "coord_mode": "pixel",
        "gt": [dict(obj) for obj in gt],
        "pred": [dict(obj) for obj in decoded_result.predictions],
        "raw_output_json": raw_output_json,
        "raw_special_tokens": list(raw_special_tokens),
        "raw_ends_with_im_end": bool(raw_ends_with_im_end),
        "errors": [str(code) for code in (errors or decoded_result.errors)],
        "error_entries": [dict(entry) for entry in (error_entries or ())],
        "parser_id": parser_metadata["parser_id"],
        "parser_policy": parser_metadata["parser_policy"],
        "metric_bearing": parser_metadata["metric_bearing"],
        "salvage_recovered": parser_metadata["salvage_recovered"],
        "parser_error_count": parser_metadata["parser_error_count"],
    }
    if allow_diagnostic:
        output.update(decoded_result.to_artifact_metadata())
    return output


def detect_mode_from_gt(
    gt_jsonl: str,
    *,
    sample_size: int = 128,
) -> Tuple[Literal["coord", "text"], str]:
    """Deterministically resolve coord vs text from GT JSONL."""

    checked = 0
    path = Path(gt_jsonl)

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            if checked >= sample_size:
                break
            line = raw_line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                snippet = line if len(line) <= 200 else (line[:200] + "...")
                raise ValueError(
                    f"Malformed JSONL at {path}:{line_no}: {snippet}"
                ) from exc
            if not isinstance(rec, dict):
                raise ValueError(
                    f"Non-object JSONL record at {path}:{line_no}: {line[:200]}"
                )

            if "width" not in rec or "height" not in rec:
                raise ValueError(f"Missing width/height at {path}:{line_no}")

            width = rec.get("width")
            height = rec.get("height")
            try:
                width_i = int(width)
                height_i = int(height)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid width/height at {path}:{line_no}: "
                    f"width={width!r} height={height!r}"
                ) from exc
            if width_i <= 0 or height_i <= 0:
                raise ValueError(
                    f"Invalid width/height at {path}:{line_no}: "
                    f"width={width_i} height={height_i}"
                )

            objs = rec.get("objects") or rec.get("gt") or []
            if objs is None:
                objs = []
            if not isinstance(objs, list):
                raise ValueError(
                    f"GT record objects/gt must be a list at {path}:{line_no}; "
                    f"got {type(objs).__name__}"
                )
            if len(objs) == 0:
                continue

            max_dim = max(width_i, height_i)
            for obj in objs:
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"GT objects must be mappings at {path}:{line_no}; "
                        f"got {type(obj).__name__}"
                    )

                pts_raw = _flatten_gt_points(
                    obj.get("bbox_2d") or obj.get("poly") or obj.get("points") or []
                )
                if not pts_raw:
                    continue

                if sequence_has_coord_tokens(pts_raw):
                    return "coord", "coord_tokens_found"

                numeric = [v for v in pts_raw if isinstance(v, (int, float))]
                if numeric and max(numeric) > max_dim:
                    return "coord", "points_exceed_image"

            checked += 1

    if checked == 0:
        return "text", "no_valid_records"

    return "text", "within_image_bounds"


@dataclass
class GenerationConfig:
    temperature: float = 0.01
    top_p: float = 0.95
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.05
    # Number of samples to decode per forward pass (HF) / per client micro-batch (vLLM).
    # Keep at 1 by default to preserve memory headroom.
    batch_size: int = 1
    seed: Optional[int] = None
    stop_pressure_mode: Optional[str] = None
    stop_pressure_min_new_tokens: int = 0
    stop_pressure_trigger_rule: Optional[str] = None
    stop_pressure_logit_bias: float = 0.0
    trace_logprobs: bool = False

    @property
    def stop_pressure_active(self) -> bool:
        return (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_OPEN
            and int(self.stop_pressure_min_new_tokens) > 0
        ) or (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY
        ) or (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY
        ) or (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY
        ) or (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY
            and float(self.stop_pressure_logit_bias) > 0.0
        ) or (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY
            and float(self.stop_pressure_logit_bias) > 0.0
        ) or (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY
            and float(self.stop_pressure_logit_bias) > 0.0
        ) or (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY
            and float(self.stop_pressure_logit_bias) > 0.0
        )

    def apply_hf_stop_pressure(self, gen_kwargs: dict[str, Any]) -> None:
        if (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN
            and self.stop_pressure_trigger_rule
            == STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_OPEN
            and int(self.stop_pressure_min_new_tokens) > 0
        ):
            gen_kwargs["min_new_tokens"] = int(self.stop_pressure_min_new_tokens)

    def build_hf_stop_pressure_logits_processor(
        self,
        *,
        tokenizer: object,
        prompt_lengths: Sequence[int],
    ) -> object | None:
        if self.stop_pressure_trigger_rule != STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY:
            return None
        suppress_structural_close_tokens: bool
        if (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY
        ):
            suppress_structural_close_tokens = True
            suppress_special_terminators = True
            fresh_boundary_only = False
        elif (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY
        ):
            suppress_structural_close_tokens = False
            suppress_special_terminators = True
            fresh_boundary_only = False
        elif (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY
        ):
            suppress_structural_close_tokens = True
            suppress_special_terminators = False
            fresh_boundary_only = True
        elif (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY
        ):
            from src.infer.constraints import (
                build_array_branch_continuation_steering_logits_processor,
            )

            return build_array_branch_continuation_steering_logits_processor(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(self.stop_pressure_logit_bias),
            )
        elif (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT
        ):
            from src.infer.constraints import (
                build_bbox_tail_closure_steering_logits_processor,
            )

            return build_bbox_tail_closure_steering_logits_processor(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(self.stop_pressure_logit_bias),
            )
        elif (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN
        ):
            from src.infer.constraints import (
                build_bbox_tail_then_object_open_steering_logits_processor,
            )

            return build_bbox_tail_then_object_open_steering_logits_processor(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(self.stop_pressure_logit_bias),
            )
        elif (
            self.stop_pressure_mode
            == STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE
        ):
            from src.infer.constraints import (
                build_bbox_tail_then_object_open_once_steering_logits_processor,
            )

            return build_bbox_tail_then_object_open_once_steering_logits_processor(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(self.stop_pressure_logit_bias),
            )
        else:
            return None
        from src.infer.constraints import (
            build_terminating_token_suppression_logits_processor,
        )

        return build_terminating_token_suppression_logits_processor(
            tokenizer=tokenizer,
            prompt_lengths=prompt_lengths,
            suppress_structural_close_tokens=suppress_structural_close_tokens,
            suppress_special_terminators=suppress_special_terminators,
            fresh_boundary_only=fresh_boundary_only,
        )


@dataclass
class GenerationResult:
    text: str = ""
    generated_token_ids: Optional[list[int]] = None
    generated_token_text: Optional[list[str]] = None
    token_logprobs: Optional[list[float]] = None
    prompt_token_ids: Optional[list[int]] = None
    stop_reason: Optional[str] = None
    error: Optional[Exception] = None


@dataclass
class InferenceConfig:
    gt_jsonl: str
    model_checkpoint: str
    mode: Literal["coord", "text", "auto"]
    requested_mode: Optional[Literal["coord", "text", "auto"]] = None
    mode_resolution_reason: Optional[str] = None
    prompt_variant: str = DEFAULT_PROMPT_VARIANT
    bbox_format: AllowedBBoxFormat = DEFAULT_BBOX_FORMAT
    detection_template_id: str = "stage1_json_pretty"
    detection_sequence_format: str = COORDJSON_FORMAT
    object_field_order: ObjectFieldOrder = "desc_first"
    object_ordering: ObjectOrdering = "sorted"
    row_separator: str = "newline"
    parser_mode: str = "strict_expected"
    compact_full_parse_mode: str = "marker_delimited_strict"
    allow_diagnostic_gt_vs_pred: bool = False
    pred_coord_mode: Literal["auto", "norm1000", "pixel"] = "auto"
    adapter_checkpoint: Optional[str] = None
    checkpoint_mode: str = "full_model"
    requested_model_checkpoint: Optional[str] = None
    requested_adapter_checkpoint: Optional[str] = None
    resolved_base_model_checkpoint: Optional[str] = None
    resolved_adapter_checkpoint: Optional[str] = None

    # Canonical unified artifact names (can be overridden by pipeline runner).
    out_path: str = "gt_vs_pred.jsonl"
    pred_token_trace_path: Optional[str] = None
    summary_path: Optional[str] = None

    # Optional pipeline-resolved root image dir used by infer/eval/vis for a
    # single deterministic image-resolution decision.
    root_image_dir: Optional[str] = None

    device: str = "cuda:0"
    limit: int = 0
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed_enabled: bool = False

    backend_type: Literal["hf", "vllm"] = "hf"
    backend: dict[str, Any] = field(default_factory=dict)
    prompt_policy_fingerprint: Optional[str] = None
    decode_policy_fingerprint: Optional[str] = None
    model_identity_fingerprint: Optional[str] = None

    # When mode=auto, how many GT records to scan (see OpenSpec for rules).
    detect_samples: int = 128


class RunCounters:
    """Aggregated counters for run-level summary."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.error_codes: set[str] = set()
        self.total_read: int = 0
        self.total_emitted: int = 0

    def add(self, code: str) -> None:
        self.counts[code] = self.counts.get(code, 0) + 1
        self.error_codes.add(code)

    def merge_summary(self, summary: Mapping[str, Any]) -> None:
        errors_by_code = summary.get("errors_by_code")
        if not isinstance(errors_by_code, Mapping):
            errors_by_code = summary.get("counters")
        if isinstance(errors_by_code, Mapping):
            for raw_code, raw_count in errors_by_code.items():
                try:
                    count_i = int(raw_count)
                except (TypeError, ValueError):
                    continue
                if count_i <= 0:
                    continue
                code = str(raw_code)
                self.counts[code] = self.counts.get(code, 0) + count_i
                self.error_codes.add(code)

        for raw_code in summary.get("error_codes", []):
            if raw_code is None:
                continue
            self.error_codes.add(str(raw_code))

        try:
            self.total_read += int(summary.get("total_read", 0) or 0)
        except (TypeError, ValueError):
            pass
        try:
            self.total_emitted += int(summary.get("total_emitted", 0) or 0)
        except (TypeError, ValueError):
            pass

    def to_summary(self) -> dict[str, Any]:
        errors_by_code = dict(self.counts)
        errors_total = int(sum(int(v) for v in errors_by_code.values()))
        return {
            "errors_total": errors_total,
            "errors_by_code": errors_by_code,
            # Back-compat: historical name used by older tooling.
            "counters": errors_by_code,
            "error_codes": sorted(self.error_codes),
            "total_read": self.total_read,
            "total_emitted": self.total_emitted,
        }


@dataclass(frozen=True)
class PromptBundle:
    messages: list[dict[str, Any]]
    prompt_text: str
    prompt_policy_fingerprint: str
    visual_metadata: dict[str, Any] = field(default_factory=dict)
    prompt_token_ids: Optional[list[int]] = None
    prompt_token_ids_source: Literal["chat_template", "unavailable"] = "unavailable"
    prompt_token_parity: PromptTokenParity = "unverifiable"


@dataclass(frozen=True)
class PromptParityResult:
    prompt_token_parity: PromptTokenParity
    verified: bool
    reason: str


@dataclass(frozen=True)
class DetectionDecodeRequest:
    backend: Literal["hf", "vllm"]
    backend_mode: str
    decode_mode: Literal["greedy", "sampling", "beam"]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_beams: int = 1
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop_tokens: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    trace_logprobs: bool = False
    trace_prompt_logprobs: bool = False
    generation_constraints: tuple[tuple[str, Any], ...] = ()
    decode_policy_fingerprint: str = ""


def build_decode_policy_fingerprint(request: DetectionDecodeRequest) -> str:
    """Build the canonical decode-policy fingerprint for provenance carriers."""

    payload = {
        "schema": "coordexp.decode_policy.v1",
        "backend": request.backend,
        "backend_mode": request.backend_mode,
        "decode_mode": request.decode_mode,
        "max_new_tokens": int(request.max_new_tokens),
        "temperature": float(request.temperature),
        "top_p": request.top_p,
        "top_k": request.top_k,
        "num_beams": int(request.num_beams),
        "repetition_penalty": request.repetition_penalty,
        "seed": request.seed,
        "stop_tokens": list(request.stop_tokens),
        "stop_strings": list(request.stop_strings),
        "trace_logprobs": bool(request.trace_logprobs),
        "trace_prompt_logprobs": bool(request.trace_prompt_logprobs),
        "generation_constraints": _canonicalize_for_json(
            dict(request.generation_constraints)
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "decode:" + hashlib.sha256(encoded).hexdigest()


def build_model_identity_fingerprint(
    *,
    checkpoint_mode: str,
    requested_model_checkpoint: Optional[str],
    requested_adapter_checkpoint: Optional[str],
    resolved_base_model_checkpoint: Optional[str],
    resolved_adapter_checkpoint: Optional[str],
    backend: str,
    backend_mode: str,
    backend_model: Optional[str] = None,
    tokenizer_source: Optional[str] = None,
    processor_source: Optional[str] = None,
    backend_sync_identity: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build the lightweight resolved model-identity fingerprint.

    This is an identity/provenance fingerprint over already-resolved handles,
    not an expensive tensor digest. Backend sync details can be added by the
    later vLLM adapter-sync slice without changing artifact readers.
    """

    payload = {
        "schema": "coordexp.model_identity.v1",
        "checkpoint_mode": str(checkpoint_mode),
        "requested_model_checkpoint": requested_model_checkpoint,
        "requested_adapter_checkpoint": requested_adapter_checkpoint,
        "resolved_base_model_checkpoint": resolved_base_model_checkpoint,
        "resolved_adapter_checkpoint": resolved_adapter_checkpoint,
        "backend": str(backend),
        "backend_mode": str(backend_mode),
        "backend_model": backend_model or resolved_base_model_checkpoint,
        "tokenizer_source": tokenizer_source or resolved_base_model_checkpoint,
        "processor_source": processor_source or resolved_base_model_checkpoint,
        "backend_sync_identity": _canonical_backend_sync_identity(
            backend_sync_identity or {}
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "model:" + hashlib.sha256(encoded).hexdigest()


def _canonicalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_for_json(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_json(item) for item in value]
    return value


def _canonical_backend_sync_identity(mapping: Mapping[str, Any]) -> dict[str, Any]:
    operational = {
        "base_url",
        "server_url",
        "timeout_s",
        "client_concurrency",
        "rank",
        "local_rank",
        "world_size",
        "device",
    }
    return _canonicalize_for_json(
        {
            str(key): value
            for key, value in mapping.items()
            if str(key) not in operational
        }
    )


def _constraint_items(mapping: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (str(key), _canonicalize_for_json(mapping[key]))
        for key in sorted(mapping.keys(), key=lambda item: str(item))
    )


def _infer_generation_constraints(
    infer_cfg: Mapping[str, Any],
    generation_cfg: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    _ = infer_cfg
    constraints: dict[str, Any] = {}
    stop_pressure_cfg = _get_nested_mapping(
        generation_cfg,
        "stop_pressure",
        path="infer.generation.stop_pressure",
    )
    stop_pressure_mode = stop_pressure_cfg.get("mode")
    if stop_pressure_mode is not None:
        constraints["stop_pressure"] = {
            "mode": str(stop_pressure_mode),
            "min_new_tokens": _get_int(
                stop_pressure_cfg,
                "min_new_tokens",
                0,
                path_prefix="infer.generation.stop_pressure",
            ),
            "trigger_rule": stop_pressure_cfg.get("trigger_rule"),
            "logit_bias": _get_float(
                stop_pressure_cfg,
                "logit_bias",
                0.0,
                path_prefix="infer.generation.stop_pressure",
            ),
        }
    return _constraint_items(constraints)


def legacy_generation_kwargs_from_decode_request(
    request: DetectionDecodeRequest,
    *,
    batch_size: int,
    stop_pressure_mode: Optional[str] = None,
    stop_pressure_min_new_tokens: int = 0,
    stop_pressure_trigger_rule: Optional[str] = None,
    stop_pressure_logit_bias: float = 0.0,
) -> dict[str, Any]:
    """Project the shared decode request into the legacy engine config shape.

    This keeps legacy config call sites fed from the canonical decode contract
    while callers migrate to runtime-owned decode requests.
    """

    return {
        "temperature": float(request.temperature),
        "top_p": float(1.0 if request.top_p is None else request.top_p),
        "max_new_tokens": int(request.max_new_tokens),
        "repetition_penalty": request.repetition_penalty,
        "batch_size": int(batch_size),
        "seed": request.seed,
        "stop_pressure_mode": stop_pressure_mode,
        "stop_pressure_min_new_tokens": int(stop_pressure_min_new_tokens),
        "stop_pressure_trigger_rule": stop_pressure_trigger_rule,
        "stop_pressure_logit_bias": float(stop_pressure_logit_bias),
        "trace_logprobs": bool(request.trace_logprobs),
    }


class OfflineInferenceEngine:
    """Runtime-owned offline inference owner.

    This class carries the remaining offline model lifecycle state needed by
    the artifact runner without requiring callers to import the legacy
    Heavy backend dependencies stay inside methods so importing
    `src.infer.runtime` remains cheap.
    """

    def __init__(
        self,
        cfg: InferenceConfig,
        gen_cfg: GenerationConfig,
        *,
        logger: Any | None = None,
    ) -> None:
        from src.common.geometry.bbox_parameterization import normalize_bbox_format
        from src.common.coord_standardizer import CoordinateStandardizer
        from src.common.object_field_order import (
            normalize_object_field_order,
            normalize_object_ordering,
        )
        from src.config.prompts import (
            coord_mode_from_coord_tokens_enabled,
            get_template_prompt_hash,
            get_template_prompts,
            resolve_dense_prompt_variant_key,
        )
        from src.infer.checkpoints import resolve_inference_checkpoint
        from src.utils import get_logger

        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.logger = logger or get_logger(__name__)

        self.cfg.rank = int(getattr(cfg, "rank", 0) or 0)
        self.cfg.local_rank = int(getattr(cfg, "local_rank", self.cfg.rank) or self.cfg.rank)
        self.cfg.world_size = max(int(getattr(cfg, "world_size", 1) or 1), 1)
        self.cfg.distributed_enabled = bool(
            getattr(cfg, "distributed_enabled", False) or self.cfg.world_size > 1
        )
        device = str(cfg.device or "cuda:0").strip() or "cuda:0"
        if self.cfg.distributed_enabled and device.startswith("cuda"):
            device = f"cuda:{self.cfg.local_rank}"
        self.cfg.device = device

        self.resolved_checkpoint = resolve_inference_checkpoint(
            model_checkpoint=cfg.model_checkpoint,
            adapter_checkpoint=cfg.adapter_checkpoint,
        )
        self.cfg.checkpoint_mode = self.resolved_checkpoint.checkpoint_mode
        self.cfg.requested_model_checkpoint = (
            self.resolved_checkpoint.requested_model_checkpoint
        )
        self.cfg.requested_adapter_checkpoint = (
            self.resolved_checkpoint.requested_adapter_checkpoint
        )
        self.cfg.resolved_base_model_checkpoint = (
            self.resolved_checkpoint.resolved_base_model_checkpoint
        )
        self.cfg.resolved_adapter_checkpoint = (
            self.resolved_checkpoint.resolved_adapter_checkpoint
        )

        self.prompt_variant = resolve_dense_prompt_variant_key(cfg.prompt_variant)
        self.cfg.prompt_variant = self.prompt_variant
        self.bbox_format = normalize_bbox_format(
            cfg.bbox_format, path="infer.bbox_format"
        )
        self.cfg.bbox_format = self.bbox_format
        self.detection_template_id = _resolve_runtime_detection_template_id(cfg)
        self.detection_template_contract = _resolve_detection_template_contract(
            self.detection_template_id
        )
        self.cfg.detection_template_id = self.detection_template_id
        self.detection_sequence_format = _detection_sequence_format_for_template_id(
            self.detection_template_id
        )
        self.cfg.detection_sequence_format = self.detection_sequence_format
        self.object_field_order = normalize_object_field_order(
            cfg.object_field_order,
            path="infer.object_field_order",
        )
        self.cfg.object_field_order = self.object_field_order
        self.object_ordering = normalize_object_ordering(
            cfg.object_ordering,
            path="infer.object_ordering",
        )
        self.cfg.object_ordering = self.object_ordering
        self.row_separator = self.detection_template_contract.row_separator
        self.cfg.row_separator = self.row_separator
        self.parser_mode = _parser_mode_for_template_id(self.detection_template_id)
        self.cfg.parser_mode = self.parser_mode
        self.cfg.compact_full_parse_mode = self.parser_mode

        self.requested_mode = cfg.requested_mode or cfg.mode
        self.resolved_mode = cfg.mode
        self.mode_reason: str | None = None
        if cfg.mode == "auto":
            self.resolved_mode, self.mode_reason = detect_mode_from_gt(
                cfg.gt_jsonl, sample_size=int(cfg.detect_samples or 128)
            )
        else:
            self.mode_reason = cfg.mode_resolution_reason
        coord_mode = coord_mode_from_coord_tokens_enabled(
            self.resolved_mode == "coord"
        )

        self.system_prompt, self.user_prompt = get_template_prompts(
            ordering=self.object_ordering,
            coord_mode=coord_mode,
            prompt_variant=self.prompt_variant,
            object_field_order=self.object_field_order,
            bbox_format=self.bbox_format,
            detection_template_id=self.detection_template_id,
        )
        self.prompt_template_hash = get_template_prompt_hash(
            ordering=self.object_ordering,
            coord_mode=coord_mode,
            prompt_variant=self.prompt_variant,
            object_field_order=self.object_field_order,
            bbox_format=self.bbox_format,
            detection_template_id=self.detection_template_id,
        )

        self.coord = CoordinateStandardizer(
            self.resolved_mode,
            pred_coord_mode=cfg.pred_coord_mode,
            bbox_format=self.bbox_format,
        )

        self.processor: Any | None = None
        self.model: Any | None = None
        self.vllm_llm: Any | None = None
        self.tokenizer: Any | None = None
        self.qwen_generation_token_ids: Any | None = None
        self.attn_implementation_requested: str | None = None
        self.attn_implementation_selected: str | None = None

    def _vllm_mode(self) -> str:
        from src.infer.backend import vllm_backend_mode

        return vllm_backend_mode(self)

    def _seed(self) -> None:
        if self.gen_cfg.seed is None:
            return

        import torch

        seed = int(self.gen_cfg.seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Best-effort determinism for HF generation. vLLM is best-effort only.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False

    def load_model(self) -> None:
        from src.infer.backend import (
            load_vllm_local_backend,
            validate_vllm_server_backend,
        )
        from src.infer.checkpoints import (
            VLLM_ADAPTER_UNSUPPORTED_MESSAGE,
            validate_compact_coord_token_adapter_contract,
        )

        backend = str(self.cfg.backend_type).lower().strip()
        resolved_base_model_checkpoint = str(
            self.cfg.resolved_base_model_checkpoint or self.cfg.model_checkpoint
        ).strip()
        resolved_adapter_checkpoint = str(
            self.cfg.resolved_adapter_checkpoint or ""
        ).strip()
        token_embeddings_adapter_spec = None
        if self.resolved_checkpoint.adapter_info is not None:
            token_embeddings_adapter_spec = self.resolved_checkpoint.adapter_info.token_embeddings_adapter_spec
        validate_compact_coord_token_adapter_contract(
            self.resolved_checkpoint,
            detection_template_id=self.detection_template_id,
        )

        if backend == "vllm":
            if resolved_adapter_checkpoint:
                raise RuntimeError(VLLM_ADAPTER_UNSUPPORTED_MESSAGE)
            self._seed()
            if self._vllm_mode() == "local":
                load_vllm_local_backend(self)
                return
            validate_vllm_server_backend(self)
            return

        import torch

        from src.common.qwen_generation import resolve_qwen_chat_generation_token_ids

        auto_processor_cls = AutoProcessor
        qwen_model_cls = Qwen3VLForConditionalGeneration
        if auto_processor_cls is None or qwen_model_cls is None:
            from transformers import (
                AutoProcessor as _AutoProcessor,
                Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
            )

            if auto_processor_cls is None:
                auto_processor_cls = _AutoProcessor
            if qwen_model_cls is None:
                qwen_model_cls = _Qwen3VLForConditionalGeneration

        install_token_embeddings_adapter_fn = install_token_embeddings_adapter
        reattach_token_embeddings_adapter_hooks_fn = reattach_token_embeddings_adapter_hooks
        if install_token_embeddings_adapter_fn is None or reattach_token_embeddings_adapter_hooks_fn is None:
            from src.coord_tokens.offset_adapter import (
                install_token_embeddings_adapter as _install_token_embeddings_adapter,
                reattach_token_embeddings_adapter_hooks as _reattach_token_embeddings_adapter_hooks,
            )

            if install_token_embeddings_adapter_fn is None:
                install_token_embeddings_adapter_fn = _install_token_embeddings_adapter
            if reattach_token_embeddings_adapter_hooks_fn is None:
                reattach_token_embeddings_adapter_hooks_fn = _reattach_token_embeddings_adapter_hooks

        self._seed()
        if self.model is None:
            attn_requested_raw = (self.cfg.backend or {}).get("attn_implementation")
            attn_requested = str(attn_requested_raw or "").strip()
            if not attn_requested or attn_requested.lower() == "auto":
                device = str(self.cfg.device or "").lower()
                if "cuda" in device and torch.cuda.is_available():
                    attn_requested = "flash_attention_2"
                else:
                    attn_requested = "sdpa"

            attn_requested = attn_requested.lower()
            self.attn_implementation_requested = attn_requested

            candidates: list[str] = []
            for cand in [attn_requested, "flash_attention_2", "sdpa", "eager"]:
                c = str(cand).strip().lower()
                if c and c not in candidates:
                    candidates.append(c)

            last_exc: Exception | None = None
            errors: list[str] = []
            for idx, cand in enumerate(candidates):
                try:
                    base_model = qwen_model_cls.from_pretrained(
                        resolved_base_model_checkpoint,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=cand,
                    )
                    model = base_model.to(self.cfg.device)
                    if resolved_adapter_checkpoint:
                        if token_embeddings_adapter_spec is not None:
                            install_token_embeddings_adapter_fn(
                                model,
                                token_ids=token_embeddings_adapter_spec.token_ids,
                                tie_head=token_embeddings_adapter_spec.tie_head,
                            )
                        try:
                            from swift import Swift
                        except ImportError as exc:
                            raise RuntimeError(
                                "HF adapter shorthand inference requires the 'swift' "
                                "package in the active environment."
                            ) from exc
                        try:
                            model = Swift.from_pretrained(
                                model,
                                model_id=resolved_adapter_checkpoint,
                                inference_mode=True,
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                "Failed to load Swift adapter checkpoint "
                                f"{resolved_adapter_checkpoint!r} onto base model "
                                f"{resolved_base_model_checkpoint!r}."
                            ) from exc
                        if token_embeddings_adapter_spec is not None:
                            reattached = reattach_token_embeddings_adapter_hooks_fn(model)
                            if reattached is None:
                                raise RuntimeError(
                                    "token_embeddings_adapter was declared in the adapter "
                                    "checkpoint, but its runtime hooks could not be "
                                    "reattached after Swift loading."
                                )
                    self.model = model
                    self.model.eval()
                    self.attn_implementation_selected = cand
                    break
                except (OSError, RuntimeError, ValueError, ImportError) as exc:
                    last_exc = exc
                    errors.append(f"{cand}: {type(exc).__name__}: {exc}")
                    if idx == 0 and len(candidates) > 1:
                        self.logger.warning(
                            "HF attention backend '%s' unavailable; falling back. Error: %s",
                            cand,
                            exc,
                        )

                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if self.model is None:
                raise RuntimeError(
                    "Failed to load HF model with any attention backend. "
                    f"candidates={candidates} errors={errors[:3]}"
                ) from last_exc

            if self.attn_implementation_selected != self.attn_implementation_requested:
                self.logger.warning(
                    "HF attention backend fallback: requested=%s selected=%s",
                    self.attn_implementation_requested,
                    self.attn_implementation_selected,
                )

        if self.processor is None:
            self.processor = auto_processor_cls.from_pretrained(
                resolved_base_model_checkpoint, trust_remote_code=True
            )

        tokenizer = getattr(self.processor, "tokenizer", None)
        self.tokenizer = tokenizer
        if tokenizer is not None:
            try:
                setattr(tokenizer, "padding_side", "left")
                if getattr(tokenizer, "pad_token_id", None) is None:
                    qwen_generation_ids = resolve_qwen_chat_generation_token_ids(
                        tokenizer
                    )
                    setattr(tokenizer, "pad_token_id", qwen_generation_ids.pad_token_id)
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Failed to configure tokenizer left-padding for inference."
                ) from exc

    def _generate_batch(self, images: Sequence[Any]) -> list[GenerationResult]:
        decode_results = InferenceRuntime(owner=self).generate_many(images=list(images))
        return [
            GenerationResult(
                text=result.text,
                generated_token_ids=result.generated_token_ids,
                generated_token_text=result.generated_tokens,
                token_logprobs=result.generated_logprobs,
                prompt_token_ids=result.prompt_token_ids,
                stop_reason=result.stop_reason,
                error=(
                    RuntimeError(str(result.backend_metadata["error"]))
                    if "error" in result.backend_metadata
                    else None
                ),
            )
            for result in decode_results
        ]

    def _generate(self, image: Any) -> str:
        results = self._generate_batch([image])
        if not results:
            return ""
        result = results[0]
        if result.error is not None:
            raise result.error
        return str(result.text or "")

    def infer(self) -> tuple[Path, Path]:
        return run_offline_artifact_inference(self)


def run_offline_inference(
    *,
    inference_kwargs: Mapping[str, Any],
    generation_kwargs: Mapping[str, Any],
    model: Any | None = None,
    processor: Any | None = None,
    logger: Any | None = None,
) -> OfflineInferenceRunResult:
    """Run offline inference through the runtime-owned compatibility seam.

    Callers should depend on this seam rather than constructing private
    backend owners directly.
    """

    inf_cfg = InferenceConfig(**dict(inference_kwargs))
    gen_cfg = GenerationConfig(**dict(generation_kwargs))
    engine = OfflineInferenceEngine(inf_cfg, gen_cfg, logger=logger)
    if model is not None:
        engine.model = model
    if processor is not None:
        engine.processor = processor
    base_jsonl_path, summary_path = run_offline_artifact_inference(engine)
    return OfflineInferenceRunResult(
        base_jsonl_path=base_jsonl_path,
        summary_path=summary_path,
        processor=getattr(engine, "processor", None),
    )


def make_offline_generation_config(**generation_kwargs: Any) -> Any:
    """Create the runtime-owned offline generation config."""

    return GenerationConfig(**dict(generation_kwargs))


def make_offline_inference_config(**inference_kwargs: Any) -> Any:
    """Create the runtime-owned offline inference config."""

    return InferenceConfig(**dict(inference_kwargs))


def make_offline_generation_result(**result_kwargs: Any) -> Any:
    """Create the runtime-owned offline generation result."""

    return GenerationResult(**dict(result_kwargs))


def make_offline_run_counters() -> Any:
    """Create the runtime-owned offline run counters."""

    return RunCounters()


def run_offline_artifact_inference(owner: Any) -> Tuple[Path, Path]:
    """Run the offline inference artifact loop through the shared runtime seam.

    Row emission, token-trace sidecars, summaries, and distributed merge
    orchestration live here so backend owners only need to provide model
    lifecycle and batch decode methods.
    """

    from contextlib import nullcontext
    from tqdm import tqdm as default_tqdm

    from src.common.prediction_parsing import extract_special_tokens, load_prediction_dict
    from src.infer.artifacts import (
        build_infer_resolved_meta,
        build_infer_resolved_meta_from_facts,
        build_infer_summary_payload,
        build_infer_summary_payload_from_facts,
        ensure_infer_artifact_dirs,
        resolve_infer_artifact_facts_from_owner,
        resolve_infer_artifact_paths,
        write_infer_summary,
    )

    self = owner
    if not hasattr(self, "detection_template_id"):
        self.detection_template_id = _resolve_runtime_detection_template_id(self.cfg)
    if not hasattr(self, "detection_template_contract"):
        self.detection_template_contract = _resolve_detection_template_contract(
            self.detection_template_id
        )
    if not hasattr(self, "parser_mode"):
        self.parser_mode = _parser_mode_for_template_id(self.detection_template_id)
    jsonl_path = Path(self.cfg.gt_jsonl)
    backend = str(self.cfg.backend_type).strip().lower()
    out_path, summary_path, trace_path = resolve_infer_artifact_paths(
        cfg=self.cfg,
        backend=backend,
    )
    worker_out_path, worker_summary_path, worker_trace_path, manifest_path = (
        offline_inference_distributed_paths(
            owner=self,
            out_path=out_path,
            summary_path=summary_path,
            trace_path=trace_path,
        )
    )

    determinism = "strict" if backend == "hf" else "best_effort"

    try:
        batch_size = int(getattr(self.gen_cfg, "batch_size", 1) or 1)
    except (TypeError, ValueError):
        batch_size = 1
    batch_size = max(1, int(batch_size))

    # Fail fast on operator-controlled input violations before loading the
    # model or emitting any partial artifacts.
    preflight_offline_inference_inputs(owner=self, jsonl_path=jsonl_path)

    counters = RunCounters()
    self.load_model()

    ensure_infer_artifact_dirs(
        out_path=worker_out_path,
        summary_path=worker_summary_path,
        trace_path=worker_trace_path,
    )
    artifact_facts = resolve_infer_artifact_facts_from_owner(
        owner=self,
        backend=backend,
        batch_size=batch_size,
    )
    resolved_meta = build_infer_resolved_meta_from_facts(
        facts=artifact_facts,
        out_path=worker_out_path,
        summary_path=worker_summary_path,
        trace_path=worker_trace_path,
    )

    self.logger.info("Inference resolved config: %s", json.dumps(resolved_meta))

    stage_by_code: Dict[str, str] = {
        "empty_pred": "infer.parse_pred",
        "invalid_coord": "infer.validate_pred",
        "invalid_geometry": "infer.validate_pred",
    }
    message_by_code: Dict[str, str] = {
        "empty_pred": "Prediction parsing produced no valid objects.",
        "invalid_coord": "Prediction contains invalid coordinate values.",
        "invalid_geometry": "Prediction contains invalid geometry.",
    }

    def _canonical(code: str) -> str:
        return ERROR_CANONICAL.get(str(code), str(code))

    def _error_entry(code: str) -> Dict[str, str]:
        c = _canonical(code)
        return {
            "code": c,
            "message": message_by_code.get(c, c),
            "stage": stage_by_code.get(c, "infer"),
        }

    def _emit(output: Dict[str, Any], error_codes: List[str]) -> None:
        fout.write(json.dumps(output, ensure_ascii=False) + "\n")
        for code in error_codes:
            counters.add(str(code))
        counters.total_emitted += 1

    def _flush_pending(pending: List[Dict[str, Any]]) -> None:
        if not pending:
            return

        if self.cfg.limit and self.cfg.limit > 0:
            remaining = int(self.cfg.limit) - int(counters.total_emitted)
            if remaining <= 0:
                return
            if len(pending) > remaining:
                pending = pending[:remaining]

        images = [p["image_obj"] for p in pending]
        results = self._generate_batch(images)  # noqa: SLF001
        if len(results) != len(pending):
            raise RuntimeError(
                f"generation returned {len(results)} outputs for {len(pending)} inputs"
            )

        for p, res in zip(pending, results):
            if res.error is not None:
                raise RuntimeError(
                    f"Generation failed for sample image={p['image']}"
                ) from res.error

        for p, res in zip(pending, results):
            raw_text = res.text
            raw_special_tokens = extract_special_tokens(
                raw_text, preserve_duplicates=True
            )
            compact_parse_artifact: Optional[Dict[str, Any]] = None
            if self.detection_template_contract.is_compact:
                compact_parse_artifact = parse_detection_template_output_artifact(
                    raw_text,
                    detection_template_id=self.detection_template_id,
                    object_field_order=self.object_field_order,
                )
                raw_output_json = compact_parse_artifact["raw_output_json"]
                raw_ends_with_im_end = (
                    compact_parse_artifact.get("terminal_token") == "<|im_end|>"
                )
            else:
                raw_ends_with_im_end = raw_text.endswith("<|im_end|>")
                raw_output_json = load_prediction_dict(raw_text)

            decoded_result = decode_offline_detection_result(
                self,
                raw_text,
                width=int(p["width"]),
                height=int(p["height"]),
                compact_parse_artifact=compact_parse_artifact,
            )

            error_codes = [_canonical(c) for c in decoded_result.errors]
            error_entries = [_error_entry(c) for c in decoded_result.errors]
            line_idx = int(counters.total_emitted)

            output = materialize_offline_gt_vs_pred_record(
                image=p["image"],
                width=int(p["width"]),
                height=int(p["height"]),
                mode=self.resolved_mode,
                gt=p["gt"],
                decoded_result=decoded_result,
                raw_output_json=raw_output_json,
                raw_special_tokens=raw_special_tokens,
                raw_ends_with_im_end=raw_ends_with_im_end,
                errors=error_codes,
                error_entries=error_entries,
                allow_diagnostic=bool(
                    getattr(self.cfg, "allow_diagnostic_gt_vs_pred", False)
                ),
            )
            if compact_parse_artifact is not None:
                output.update(
                    {
                        "parse_mode": compact_parse_artifact["parse_mode"],
                        "serialization_policy": compact_parse_artifact[
                            "serialization_policy"
                        ],
                        "object_separator": compact_parse_artifact["object_separator"],
                        "terminal_token": compact_parse_artifact["terminal_token"],
                        "parse_error_code": compact_parse_artifact["parse_error_code"],
                        "parse_error_offset": compact_parse_artifact[
                            "parse_error_offset"
                        ],
                        "object_field_order": compact_parse_artifact[
                            "object_field_order"
                        ],
                    }
                )
            output["detection_template_id"] = self.detection_template_id
            output["object_field_order"] = self.object_field_order
            if p.get("image_id") is not None:
                output["image_id"] = p.get("image_id")
            if isinstance(p.get("metadata"), Mapping):
                output["metadata"] = dict(p["metadata"])
            if self.cfg.distributed_enabled:
                output[_DISTRIBUTED_SOURCE_INDEX_KEY] = int(p["source_index"])
            if (
                ftrace is not None
                and res.generated_token_text is not None
                and res.token_logprobs is not None
            ):
                token_trace_payload = {
                    "generated_token_text": list(res.generated_token_text),
                    "token_logprobs": list(res.token_logprobs),
                }
                trace_record = {
                    "line_idx": line_idx,
                    **token_trace_payload,
                    "raw_output_sha256": hashlib.sha256(
                        raw_text.encode("utf-8")
                    ).hexdigest(),
                    "token_trace_sha256": hashlib.sha256(
                        json.dumps(
                            token_trace_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                if self.cfg.distributed_enabled:
                    trace_record[_DISTRIBUTED_SOURCE_INDEX_KEY] = int(
                        p["source_index"]
                    )
                ftrace.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
            _emit(output, error_codes)

    pbar_enabled = (not self.cfg.distributed_enabled) or int(self.cfg.rank) == 0
    pbar_total: Optional[int]
    if self.cfg.limit and self.cfg.limit > 0:
        pbar_total = int(self.cfg.limit)
    else:
        pbar_total = None
    pending: List[Dict[str, Any]] = []
    selected_index = 0

    tqdm_factory = tqdm or default_tqdm

    trace_cm = (
        worker_trace_path.open("w", encoding="utf-8")
        if worker_trace_path is not None
        else nullcontext(None)
    )
    with (
        jsonl_path.open("r", encoding="utf-8") as fin,
        worker_out_path.open("w", encoding="utf-8") as fout,
        trace_cm as ftrace,
        tqdm_factory(
            total=pbar_total,
            desc="Infer",
            unit="samples",
            dynamic_ncols=True,
            smoothing=0.1,
            mininterval=1.0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            disable=not pbar_enabled,
        ) as pbar,
    ):
        for line_no, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if not self.cfg.distributed_enabled:
                pbar.update(1)
                counters.total_read += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counters.add("invalid_json")
                continue

            if not isinstance(record, dict):
                raise ValueError(f"Non-object JSONL record at {jsonl_path}:{line_no}")

            width_raw = record.get("width")
            height_raw = record.get("height")
            try:
                width = int(width_raw)
                height = int(height_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid width/height at {jsonl_path}:{line_no}: "
                    f"width={width_raw!r} height={height_raw!r}"
                ) from exc

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid width/height at {jsonl_path}:{line_no}: "
                    f"width={width} height={height}"
                )

            images = record.get("images")
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError(
                    "Input record must contain exactly one image in `images` at "
                    f"{jsonl_path}:{line_no}: images={images!r}"
                )

            image_key = images[0]
            if not isinstance(image_key, str) or not image_key.strip():
                raise ValueError(
                    f"Invalid image field in `images[0]` at {jsonl_path}:{line_no}: "
                    f"{image_key!r}"
                )

            source_index = int(selected_index)
            if self.cfg.limit and source_index >= int(self.cfg.limit):
                break
            selected_index += 1

            if self.cfg.distributed_enabled and int(self.cfg.rank) == 0:
                pbar.update(1)

            if self.cfg.distributed_enabled and (
                source_index % int(self.cfg.world_size)
            ) != int(self.cfg.rank):
                continue

            if self.cfg.distributed_enabled:
                counters.total_read += 1

            gt_errors: List[str] = []
            gt = process_offline_gt(
                self, record, width=width, height=height, errors=gt_errors
            )
            if gt_errors:
                raise ValueError(
                    f"Invalid GT geometry at {jsonl_path}:{line_no} "
                    f"(mode={self.resolved_mode}): {gt_errors}"
                )
            gt = compact_gt_vs_pred_objects(gt)

            _img_path, image_obj = prepare_offline_image(self, jsonl_path, record)
            if image_obj is None:
                raise ValueError(
                    f"Failed to load image for inference at {_img_path} "
                    f"(from {jsonl_path}:{line_no})"
                )

            pending.append(
                {
                    "image": image_key,
                    "width": width,
                    "height": height,
                    "gt": gt,
                    "image_obj": image_obj,
                    "image_id": record.get("image_id"),
                    "metadata": (
                        dict(record["metadata"])
                        if isinstance(record.get("metadata"), Mapping)
                        else None
                    ),
                    "source_index": source_index,
                }
            )

            target = batch_size
            if self.cfg.limit and self.cfg.limit > 0:
                remaining = int(self.cfg.limit) - int(counters.total_emitted)
                target = max(1, min(int(target), int(remaining)))

            if len(pending) >= target:
                _flush_pending(pending)
                pending = []

        _flush_pending(pending)

    summary_payload = build_infer_summary_payload_from_facts(
        facts=artifact_facts,
        counters=counters,
        determinism=determinism,
    )
    write_infer_summary(
        summary_path=worker_summary_path,
        summary_payload=summary_payload,
    )

    if manifest_path is not None:
        write_offline_inference_distributed_manifest(
            owner=self,
            manifest_path=manifest_path,
            out_path=worker_out_path,
            summary_path=worker_summary_path,
            trace_path=worker_trace_path,
        )
        if int(self.cfg.rank) == 0:
            manifest_paths = wait_for_offline_inference_distributed_manifests(
                owner=self,
                base_out_path=out_path,
            )
            merged_counters = merge_offline_inference_distributed_outputs(
                manifest_paths=manifest_paths,
                final_out_path=out_path,
                final_summary_path=summary_path,
                final_trace_path=trace_path,
            )
            final_summary_payload = build_infer_summary_payload_from_facts(
                facts=artifact_facts,
                counters=merged_counters,
                determinism=determinism,
            )
            write_infer_summary(
                summary_path=summary_path,
                summary_payload=final_summary_payload,
            )
            self.logger.info(
                "Distributed inference finished: %s samples emitted, summary=%s",
                merged_counters.total_emitted,
                summary_path,
            )
        else:
            self.logger.info(
                "Distributed inference shard finished: rank=%s emitted=%s summary=%s",
                self.cfg.rank,
                counters.total_emitted,
                worker_summary_path,
            )
    else:
        self.logger.info(
            "Inference finished: %s samples emitted, summary=%s",
            counters.total_emitted,
            summary_path,
        )
    return out_path, summary_path


def preflight_offline_inference_inputs(*, owner: Any, jsonl_path: Path) -> None:
    """Validate operator-controlled offline inference inputs before side effects."""

    from PIL import Image

    self = owner
    limit = int(self.cfg.limit or 0)
    max_errors = 5
    errors: List[str] = []

    checked = 0
    with jsonl_path.open("r", encoding="utf-8") as fin:
        for line_no, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue

            checked += 1
            if limit and checked > limit:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                snippet = line if len(line) <= 200 else (line[:200] + "...")
                errors.append(f"Malformed JSONL at {jsonl_path}:{line_no}: {snippet}")
                if len(errors) >= max_errors:
                    break
                continue

            if not isinstance(record, dict):
                errors.append(f"Non-object JSONL record at {jsonl_path}:{line_no}")
                if len(errors) >= max_errors:
                    break
                continue

            width_raw = record.get("width")
            height_raw = record.get("height")
            try:
                width = int(width_raw)
                height = int(height_raw)
            except (TypeError, ValueError) as exc:
                errors.append(
                    f"Invalid width/height at {jsonl_path}:{line_no}: "
                    f"width={width_raw!r} height={height_raw!r} ({exc.__class__.__name__})"
                )
                if len(errors) >= max_errors:
                    break
                continue

            if width <= 0 or height <= 0:
                errors.append(
                    f"Invalid width/height at {jsonl_path}:{line_no}: "
                    f"width={width} height={height}"
                )
                if len(errors) >= max_errors:
                    break
                continue

            images = record.get("images")
            if not isinstance(images, list) or len(images) != 1:
                errors.append(
                    "Input record must contain exactly one image in `images` at "
                    f"{jsonl_path}:{line_no}: images={images!r}"
                )
                if len(errors) >= max_errors:
                    break
                continue

            image_field = images[0]
            if not isinstance(image_field, str) or not image_field.strip():
                errors.append(
                    f"Invalid image field in `images[0]` at {jsonl_path}:{line_no}: "
                    f"{image_field!r}"
                )
                if len(errors) >= max_errors:
                    break
                continue

            try:
                img_path = resolve_offline_image_path(self, jsonl_path, image_field)
            except FileNotFoundError as exc:
                errors.append(str(exc))
                if len(errors) >= max_errors:
                    break
                continue

            try:
                with Image.open(img_path) as im:
                    actual_size = im.size
                    im.convert("RGB")
            except (OSError, ValueError) as exc:
                errors.append(
                    f"Failed to open image at {img_path} (from {jsonl_path}:{line_no}): "
                    f"{exc.__class__.__name__}: {exc}"
                )
                if len(errors) >= max_errors:
                    break
                continue
            if actual_size != (width, height):
                errors.append(
                    f"Image size does not match record width/height at {jsonl_path}:{line_no}: "
                    f"image={str(img_path)!r} actual={actual_size!r} expected={(width, height)!r}"
                )
                if len(errors) >= max_errors:
                    break
                continue

            objs_raw = record.get("objects")
            gt_raw = record.get("gt")
            if objs_raw is not None and not isinstance(objs_raw, list):
                errors.append(
                    f"GT record 'objects' must be a list at {jsonl_path}:{line_no}; "
                    f"got {type(objs_raw).__name__}"
                )
                if len(errors) >= max_errors:
                    break
                continue
            if objs_raw is None and gt_raw is not None and not isinstance(gt_raw, list):
                errors.append(
                    f"GT record 'gt' must be a list at {jsonl_path}:{line_no}; "
                    f"got {type(gt_raw).__name__}"
                )
                if len(errors) >= max_errors:
                    break
                continue

            gt_errors: List[str] = []
            _ = process_offline_gt(
                self, record, width=width, height=height, errors=gt_errors
            )
            if gt_errors:
                errors.append(
                    f"Invalid GT geometry at {jsonl_path}:{line_no} "
                    f"(mode={self.resolved_mode}): {gt_errors}"
                )
                if len(errors) >= max_errors:
                    break

    if errors:
        msg = (
            "Inference preflight failed (operator-controlled input violations):\n"
            + "\n".join(f"- {e}" for e in errors)
        )
        raise ValueError(msg)


def offline_inference_distributed_paths(
    *,
    owner: Any,
    out_path: Path,
    summary_path: Path,
    trace_path: Optional[Path],
) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
    if not owner.cfg.distributed_enabled:
        return out_path, summary_path, trace_path, None

    shard_dir = out_path.parent / "shards" / f"rank_{int(owner.cfg.rank):05d}"
    worker_out_path = shard_dir / out_path.name
    worker_summary_path = shard_dir / summary_path.name
    worker_trace_path = shard_dir / trace_path.name if trace_path is not None else None
    manifest_path = shard_dir / "manifest.json"
    return worker_out_path, worker_summary_path, worker_trace_path, manifest_path


def write_offline_inference_distributed_manifest(
    *,
    owner: Any,
    manifest_path: Path,
    out_path: Path,
    summary_path: Path,
    trace_path: Optional[Path],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "rank": int(owner.cfg.rank),
        "local_rank": int(owner.cfg.local_rank),
        "world_size": int(owner.cfg.world_size),
        "artifacts": {
            "gt_vs_pred_jsonl": str(out_path),
            "summary_json": str(summary_path),
            "pred_token_trace_jsonl": str(trace_path) if trace_path is not None else None,
        },
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def wait_for_offline_inference_distributed_manifests(
    *,
    owner: Any,
    base_out_path: Path,
) -> list[Path]:
    import time

    shard_root = base_out_path.parent / "shards"
    manifest_paths = [
        shard_root / f"rank_{rank:05d}" / "manifest.json"
        for rank in range(int(owner.cfg.world_size))
    ]
    deadline = time.monotonic() + _DISTRIBUTED_MANIFEST_TIMEOUT_S
    while True:
        missing = [path for path in manifest_paths if not path.exists()]
        if not missing:
            return manifest_paths
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Timed out waiting for distributed inference shards: "
                + ", ".join(str(path) for path in missing[:3])
            )
        time.sleep(1.0)


def merge_offline_inference_distributed_outputs(
    *,
    manifest_paths: List[Path],
    final_out_path: Path,
    final_summary_path: Path,
    final_trace_path: Optional[Path],
) -> RunCounters:
    merged_counters = RunCounters()
    merged_rows: list[tuple[int, Dict[str, Any]]] = []
    merged_trace_rows: list[tuple[int, Dict[str, Any]]] = []
    seen_row_indices: set[int] = set()
    seen_trace_indices: set[int] = set()

    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("status") or "") != "complete":
            raise RuntimeError(
                f"Distributed inference manifest is incomplete: {manifest_path}"
            )
        artifacts = manifest.get("artifacts") or {}
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(
                f"Distributed inference manifest is malformed: {manifest_path}"
            )

        shard_out_path = Path(str(artifacts.get("gt_vs_pred_jsonl") or "").strip())
        shard_summary_path = Path(str(artifacts.get("summary_json") or "").strip())
        trace_raw = artifacts.get("pred_token_trace_jsonl")
        shard_trace_path = (
            Path(str(trace_raw).strip())
            if isinstance(trace_raw, str) and str(trace_raw).strip()
            else None
        )

        shard_summary = json.loads(shard_summary_path.read_text(encoding="utf-8"))
        merged_counters.merge_summary(shard_summary)

        with shard_out_path.open("r", encoding="utf-8") as fin:
            for line_no, raw_line in enumerate(fin, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"Distributed shard row must be an object: {shard_out_path}:{line_no}"
                    )
                source_index_raw = record.pop(_DISTRIBUTED_SOURCE_INDEX_KEY, None)
                try:
                    source_index = int(source_index_raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Distributed shard row is missing {_DISTRIBUTED_SOURCE_INDEX_KEY}: "
                        f"{shard_out_path}:{line_no}"
                    ) from exc
                if source_index in seen_row_indices:
                    raise RuntimeError(
                        f"Duplicate distributed row index {source_index} in {shard_out_path}"
                    )
                seen_row_indices.add(source_index)
                merged_rows.append((source_index, record))

        if shard_trace_path is not None and shard_trace_path.exists():
            with shard_trace_path.open("r", encoding="utf-8") as fin:
                for line_no, raw_line in enumerate(fin, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise RuntimeError(
                            "Distributed trace row must be an object: "
                            f"{shard_trace_path}:{line_no}"
                        )
                    source_index_raw = record.pop(_DISTRIBUTED_SOURCE_INDEX_KEY, None)
                    try:
                        source_index = int(source_index_raw)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"Distributed trace row is missing {_DISTRIBUTED_SOURCE_INDEX_KEY}: "
                            f"{shard_trace_path}:{line_no}"
                        ) from exc
                    if source_index in seen_trace_indices:
                        raise RuntimeError(
                            f"Duplicate distributed trace index {source_index} in {shard_trace_path}"
                        )
                    seen_trace_indices.add(source_index)
                    merged_trace_rows.append((source_index, record))

    merged_rows.sort(key=lambda item: item[0])
    if len(merged_rows) != int(merged_counters.total_emitted):
        raise RuntimeError(
            "Distributed inference merge row-count mismatch: "
            f"rows={len(merged_rows)} total_emitted={merged_counters.total_emitted}"
        )

    final_out_path.parent.mkdir(parents=True, exist_ok=True)
    with final_out_path.open("w", encoding="utf-8") as fout:
        for _, record in merged_rows:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    if final_trace_path is not None:
        final_trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_by_source = {
            source_index: record for source_index, record in merged_trace_rows
        }
        with final_trace_path.open("w", encoding="utf-8") as fout:
            for final_idx, (source_index, record) in enumerate(merged_rows):
                trace_record = trace_by_source.get(source_index)
                if trace_record is None:
                    continue
                trace_record = dict(trace_record)
                trace_record["line_idx"] = final_idx
                fout.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

    return merged_counters


def create_offline_engine(
    *,
    inference_kwargs: Mapping[str, Any],
    generation_kwargs: Mapping[str, Any],
    logger: Any | None = None,
    load_model: bool = False,
) -> Any:
    """Build the runtime-owned offline inference owner."""

    engine = OfflineInferenceEngine(
        make_offline_inference_config(**dict(inference_kwargs)),
        make_offline_generation_config(**dict(generation_kwargs)),
        logger=logger,
    )
    if load_model:
        engine.load_model()
    return engine


def create_offline_engine_from_configs(
    inference_config: Any,
    generation_config: Any,
    *,
    logger: Any | None = None,
) -> Any:
    """Build the runtime-owned offline inference owner from config objects."""

    return OfflineInferenceEngine(inference_config, generation_config, logger=logger)


class InferenceRuntime:
    """Shared inference runtime facade over concrete backend adapters."""

    def __init__(
        self,
        *,
        owner: Any,
        backend_generate: Optional[Callable[..., Sequence[Any]]] = None,
    ) -> None:
        self.owner = owner
        self._backend_generate = backend_generate

    def _backend_name(self) -> str:
        cfg = getattr(self.owner, "cfg", None)
        return str(getattr(cfg, "backend_type", "unknown") or "unknown").strip().lower()

    def _generate_legacy_results(self, images: Sequence[Any]) -> Sequence[Any]:
        backend_generate = self._backend_generate
        if backend_generate is None:
            from src.infer.backend import generate_batch

            backend_generate = generate_batch
        return backend_generate(
            owner=self.owner,
            images=list(images),
            result_factory=_RuntimeLegacyGenerationResult,
        )

    def generate_many(self, *, images: Sequence[Any]) -> list[DetectionDecodeResult]:
        """Generate and normalize backend responses into shared decode results."""

        if not images:
            return []
        backend = self._backend_name()
        legacy_results = self._generate_legacy_results(images)
        return [
            _legacy_result_to_decode_result(result, backend=backend, images=list(images))
            for result in legacy_results
        ]


@dataclass
class _RuntimeLegacyGenerationResult:
    text: str = ""
    generated_token_ids: Optional[list[int]] = None
    generated_token_text: Optional[list[str]] = None
    token_logprobs: Optional[list[float]] = None
    prompt_token_ids: Optional[list[int]] = None
    stop_reason: Optional[str] = None
    error: Optional[Exception] = None


@dataclass(frozen=True)
class OfflineInferenceRunResult:
    base_jsonl_path: Any
    summary_path: Any
    processor: Any | None = None


@dataclass(frozen=True)
class OfflineDebugGeneration:
    image_path: Any
    text: str
    error: Exception | None = None


def run_offline_debug_generations(
    *,
    inference_kwargs: Mapping[str, Any],
    generation_kwargs: Mapping[str, Any],
    jsonl_path: Any,
    records: Sequence[Mapping[str, Any]],
    batch: bool = False,
    logger: Any | None = None,
) -> list[OfflineDebugGeneration]:
    """Generate raw debug text for ad hoc scripts through the runtime seam."""

    inf_cfg = InferenceConfig(**dict(inference_kwargs))
    gen_cfg = GenerationConfig(**dict(generation_kwargs))
    engine = OfflineInferenceEngine(inf_cfg, gen_cfg, logger=logger)
    engine.load_model()

    prepared: list[tuple[Any, Any]] = []
    for record in records:
        img_path, image = prepare_offline_image(
            engine, jsonl_path, dict(record), strict_decode=False
        )
        prepared.append((img_path, image))

    if batch:
        images = [image for _img_path, image in prepared]
        if any(image is None for image in images):
            return [
                OfflineDebugGeneration(
                    image_path=img_path,
                    text="",
                    error=RuntimeError(
                        "image_load_failed"
                        if image is None
                        else "skipped_due_to_image_load_failure"
                    ),
                )
                for img_path, image in prepared
            ]
        results = engine._generate_batch(list(images))  # noqa: SLF001
        if len(results) != len(prepared):
            raise RuntimeError(
                "Offline debug batch generation returned a row-count mismatch: "
                f"expected {len(prepared)} got {len(results)}"
            )
        return [
            OfflineDebugGeneration(
                image_path=img_path,
                text=str(getattr(result, "text", "") or ""),
                error=getattr(result, "error", None),
            )
            for (img_path, _image), result in zip(prepared, results)
        ]

    out: list[OfflineDebugGeneration] = []
    for img_path, image in prepared:
        if image is None:
            out.append(
                OfflineDebugGeneration(
                    image_path=img_path,
                    text="",
                    error=RuntimeError("image_load_failed"),
                )
            )
            continue
        try:
            raw = engine._generate(image)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            out.append(OfflineDebugGeneration(image_path=img_path, text="", error=exc))
        else:
            out.append(OfflineDebugGeneration(image_path=img_path, text=str(raw or "")))
    return out


def _legacy_result_to_decode_result(
    result: Any,
    *,
    backend: str,
    images: list[Any],
) -> DetectionDecodeResult:
    error = getattr(result, "error", None)
    backend_metadata: dict[str, Any] = {
        "response_family": "legacy_generation_result",
        "input_images": images,
    }
    if error is not None:
        backend_metadata["error"] = str(error)
    return DetectionDecodeResult(
        text=str(getattr(result, "text", "") or ""),
        generated_token_ids=(
            [int(value) for value in getattr(result, "generated_token_ids")]
            if getattr(result, "generated_token_ids", None) is not None
            else None
        ),
        generated_tokens=(
            list(getattr(result, "generated_token_text"))
            if getattr(result, "generated_token_text", None) is not None
            else None
        ),
        generated_logprobs=(
            [float(value) for value in getattr(result, "token_logprobs")]
            if getattr(result, "token_logprobs", None) is not None
            else None
        ),
        stop_reason=(
            str(getattr(result, "stop_reason"))
            if getattr(result, "stop_reason", None) is not None
            else None
        ),
        backend=backend,
        backend_metadata=backend_metadata,
        prompt_token_ids=(
            [int(value) for value in getattr(result, "prompt_token_ids")]
            if getattr(result, "prompt_token_ids", None) is not None
            else None
        ),
    )


def _get_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"infer.{key} must be a mapping")
    return value


def _get_nested_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _get_float(
    mapping: Mapping[str, Any],
    key: str,
    default: float,
    *,
    path_prefix: str = "infer.generation",
) -> float:
    value = mapping.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path_prefix}.{key} must be a float") from exc


def _get_int(
    mapping: Mapping[str, Any],
    key: str,
    default: int,
    *,
    path_prefix: str = "infer.generation",
) -> int:
    value = mapping.get(key, default)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path_prefix}.{key} must be an int") from exc


def _get_optional_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path_prefix: str = "infer.generation",
) -> Optional[int]:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path_prefix}.{key} must be an int") from exc


def _get_optional_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path_prefix: str = "infer.generation",
) -> Optional[float]:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path_prefix}.{key} must be a float") from exc


def _get_bool(
    mapping: Mapping[str, Any],
    key: str,
    default: bool,
    *,
    path_prefix: str = "infer.generation",
) -> bool:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"{path_prefix}.{key} must be a bool")


def _normalize_decode_mode(
    raw_mode: Any,
    *,
    temperature: float,
    path: str = "infer.generation.decode_mode",
    allow_sample_alias: bool = True,
) -> Literal["greedy", "sampling", "beam"]:
    if raw_mode is None:
        return "greedy" if float(temperature) <= 0.0 else "sampling"
    if not isinstance(raw_mode, str):
        raise ValueError(f"{path} must be a string")
    mode = raw_mode.strip().lower().replace("-", "_")
    if allow_sample_alias and mode == "sample":
        mode = "sampling"
    if mode not in {"greedy", "sampling", "beam"}:
        raise ValueError(
            f"{path} must be one of "
            "{'greedy', 'sampling', 'beam'}"
        )
    return cast(Literal["greedy", "sampling", "beam"], mode)


def _validate_temperature_matches_decode_mode(
    *,
    decode_mode: Literal["greedy", "sampling", "beam"],
    temperature: float,
    path_prefix: str,
) -> None:
    if decode_mode == "greedy" and temperature > 0.0:
        raise ValueError(
            f"{path_prefix}.decode_mode=greedy requires temperature <= 0 "
            "until the shared-runtime bridge supports explicit sampling controls"
        )
    if decode_mode == "sampling" and temperature <= 0.0:
        raise ValueError(
            f"{path_prefix}.decode_mode=sampling requires temperature > 0 "
            "until the shared-runtime bridge supports explicit sampling controls"
        )


def build_decode_request_from_infer_config(
    infer_cfg: Mapping[str, Any],
) -> DetectionDecodeRequest:
    """Map authored offline `infer.*` config into the shared decode request."""

    backend_cfg = _get_mapping(infer_cfg, "backend")
    backend_raw = str(backend_cfg.get("type") or "hf").strip().lower()
    if backend_raw not in {"hf", "vllm"}:
        raise ValueError("infer.backend.type must be one of {'hf', 'vllm'}")
    backend = cast(Literal["hf", "vllm"], backend_raw)
    backend_mode_raw = str(
        backend_cfg.get("mode") or ("local" if backend == "hf" else "server")
    ).strip().lower()
    if backend == "hf":
        if backend_mode_raw != "local":
            raise ValueError("infer.backend.mode must be local for infer.backend.type=hf")
        backend_mode = "local"
    else:
        if backend_mode_raw not in {"local", "server"}:
            raise ValueError(
                "infer.backend.mode must be one of {'local', 'server'} "
                "for infer.backend.type=vllm"
            )
        backend_mode = backend_mode_raw

    gen_cfg = _get_mapping(infer_cfg, "generation")
    if not gen_cfg:
        raise ValueError("infer.generation section is required")

    temperature = _get_float(gen_cfg, "temperature", 0.01)
    num_beams = _get_int(gen_cfg, "num_beams", 1)
    decode_mode = _normalize_decode_mode(
        gen_cfg.get("decode_mode"),
        temperature=temperature,
        path="infer.generation.decode_mode",
        allow_sample_alias=True,
    )
    if decode_mode == "beam" and num_beams <= 1:
        raise ValueError("infer.generation.decode_mode=beam requires num_beams > 1")
    if decode_mode != "beam" and num_beams != 1:
        raise ValueError(
            "infer.generation.num_beams > 1 requires decode_mode=beam"
        )
    if decode_mode == "beam":
        raise ValueError(
            "infer.generation.decode_mode=beam is not supported by the current "
            "offline shared-runtime bridge"
        )
    _validate_temperature_matches_decode_mode(
        decode_mode=decode_mode,
        temperature=temperature,
        path_prefix="infer.generation",
    )
    if gen_cfg.get("top_k") is not None:
        raise ValueError(
            "infer.generation.top_k is not supported by the current offline "
            "shared-runtime bridge"
        )
    seed_value = gen_cfg.get("seed")
    seed = int(seed_value) if seed_value is not None else None

    request = DetectionDecodeRequest(
        backend=backend,
        backend_mode=backend_mode,
        decode_mode=decode_mode,
        max_new_tokens=_get_int(gen_cfg, "max_new_tokens", 1024),
        temperature=temperature,
        top_p=_get_optional_float(gen_cfg, "top_p")
        if gen_cfg.get("top_p") is not None
        else 0.95,
        top_k=_get_optional_int(gen_cfg, "top_k"),
        num_beams=num_beams,
        repetition_penalty=(
            _get_optional_float(gen_cfg, "repetition_penalty")
            if gen_cfg.get("repetition_penalty") is not None
            else 1.05
        ),
        seed=seed,
        stop_strings=("<|im_end|>",),
        trace_logprobs=_get_bool(gen_cfg, "trace_logprobs", False),
        trace_prompt_logprobs=_get_bool(gen_cfg, "trace_prompt_logprobs", False),
        generation_constraints=_infer_generation_constraints(infer_cfg, gen_cfg),
    )
    return replace(
        request,
        decode_policy_fingerprint=build_decode_policy_fingerprint(request),
    )


def build_decode_request_from_rollout_matching_config(
    rollout_matching_cfg: Mapping[str, Any],
    *,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> DetectionDecodeRequest:
    """Map Stage-2 `rollout_matching.*` config into the shared decode request."""

    cfg = rollout_matching_cfg
    backend_raw = str(cfg.get("rollout_backend", "hf") or "hf").strip().lower()
    if backend_raw not in {"hf", "vllm"}:
        raise ValueError(
            "rollout_matching.rollout_backend must be one of {'hf', 'vllm'}"
        )
    backend = cast(Literal["hf", "vllm"], backend_raw)
    vllm_cfg = _get_nested_mapping(cfg, "vllm", path="rollout_matching.vllm")
    if backend == "vllm":
        backend_mode = (
            str(vllm_cfg.get("mode", "colocate") or "colocate").strip().lower()
        )
        if backend_mode not in {"colocate", "server"}:
            raise ValueError(
                "rollout_matching.vllm.mode must be one of {'colocate', 'server'}"
            )
    else:
        backend_mode = "local"
    decoding_cfg = _get_nested_mapping(
        cfg,
        "decoding",
        path="rollout_matching.decoding",
    )

    decode_mode_declared = "decode_mode" in cfg and cfg.get("decode_mode") is not None
    decode_mode_raw = cfg.get("decode_mode") if decode_mode_declared else None
    max_new_tokens = _get_int(
        cfg,
        "max_new_tokens",
        512,
        path_prefix="rollout_matching",
    )
    num_beams = _get_int(
        cfg,
        "num_beams",
        1,
        path_prefix="rollout_matching",
    )
    repetition_penalty_raw = cfg.get("repetition_penalty", 1.0)
    repetition_penalty = float(
        1.0 if repetition_penalty_raw is None else repetition_penalty_raw
    )
    if repetition_penalty <= 0:
        raise ValueError("rollout_matching.repetition_penalty must be > 0")
    temperature = _get_float(
        decoding_cfg,
        "temperature",
        0.0,
        path_prefix="rollout_matching.decoding",
    )
    top_p = _get_float(
        decoding_cfg,
        "top_p",
        1.0,
        path_prefix="rollout_matching.decoding",
    )
    top_k = _get_int(
        decoding_cfg,
        "top_k",
        -1,
        path_prefix="rollout_matching.decoding",
    )
    if top_k != -1 and top_k < 1:
        raise ValueError(
            "rollout_matching.decoding.top_k must be -1 (disabled) or >= 1"
        )

    decode_mode = _normalize_decode_mode(
        decode_mode_raw
        if decode_mode_declared
        else ("sampling" if float(temperature) > 0.0 else "greedy"),
        temperature=temperature,
        path="rollout_matching.decode_mode",
        allow_sample_alias=False,
    )

    if decode_override is not None:
        if not isinstance(decode_override, Mapping):
            raise TypeError("rollout decode override must be a mapping when provided")
        override_declares_decode_mode = "decode_mode" in decode_override
        allowed = {"decode_mode", "temperature", "top_p", "top_k"}
        unknown = [
            str(key)
            for key in sorted(decode_override.keys(), key=lambda item: str(item))
            if str(key) not in allowed
        ]
        if unknown:
            raise ValueError(f"Unknown rollout decode override keys: {unknown}")
        decode_mode = _normalize_decode_mode(
            decode_override.get("decode_mode", decode_mode),
            temperature=temperature,
            path="rollout decode override.decode_mode",
            allow_sample_alias=False,
        )
        temperature = _get_float(
            decode_override,
            "temperature",
            temperature,
            path_prefix="rollout decode override",
        )
        top_p = _get_float(
            decode_override,
            "top_p",
            top_p,
            path_prefix="rollout decode override",
        )
        top_k = _get_int(
            decode_override,
            "top_k",
            top_k,
            path_prefix="rollout decode override",
        )
        if top_k != -1 and top_k < 1:
            raise ValueError(
                "rollout decode override.top_k must be -1 (disabled) or >= 1"
            )
        if (
            not override_declares_decode_mode
            and decode_mode == "greedy"
            and temperature > 0.0
        ):
            decode_mode = "sampling"

    if temperature < 0.0:
        raise ValueError("rollout_matching.decoding.temperature must be >= 0")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("rollout_matching.decoding.top_p must be in (0, 1]")
    _validate_temperature_matches_decode_mode(
        decode_mode=decode_mode,
        temperature=temperature,
        path_prefix="rollout_matching",
    )

    request = DetectionDecodeRequest(
        backend=backend,
        backend_mode=backend_mode,
        decode_mode=decode_mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=max(1, num_beams),
        repetition_penalty=repetition_penalty,
        stop_strings=("<|im_end|>",),
    )
    return replace(
        request,
        decode_policy_fingerprint=build_decode_policy_fingerprint(request),
    )


def build_decode_request_from_rollout_facts(
    facts: RolloutDecodeFacts | RolloutRuntimeFacts,
    *,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> DetectionDecodeRequest:
    """Map resolved Stage-2 rollout facts into a shared decode request."""

    return build_decode_request_from_rollout_matching_config(
        facts.rollout_matching_cfg,
        decode_override=decode_override,
    )


def build_decode_request_from_rollout_owner(
    owner: Any,
    *,
    decode_override: Optional[Mapping[str, Any]] = None,
) -> DetectionDecodeRequest:
    """Resolve a Stage-2 rollout decode request from an owner-like object.

    Backend adapters use this helper instead of calling trainer-private methods,
    keeping decode-policy parsing under the shared inference runtime.
    """

    return build_decode_request_from_rollout_facts(
        resolve_rollout_decode_facts_from_owner(owner),
        decode_override=decode_override,
    )
