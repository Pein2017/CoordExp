"""Centralized inference engine for CoordExp.

Features
--------
- Single entrypoint that works for coord-token (normalized) and pure-text
  checkpoints via an explicit ``mode`` switch.
- Standardized ``gt_vs_pred.jsonl`` with pixel-space geometries for both GT and
  predictions; polygons are preserved.
- Per-sample error reporting plus run-level counters and summary JSON.
- Deterministic generation when ``--seed`` is provided (torch + CUDA seeding).

Schema (per line of ``gt_vs_pred.jsonl``)
-----------------------------------
```
{
  "image": str,              # relative or absolute path
  "width": int,
  "height": int,
  "mode": "coord" | "text",
  "coord_mode": "norm1000" | "pixel" | null,  # optional trace/debug
  "gt": [ {"type","points","desc"} ],
  "pred": [ {"type","points","desc"} ],
  "raw_output_json": dict | null,  # parsed prediction dict (best-effort)
  "raw_special_tokens": [str],     # e.g. <|im_end|>, <|coord_123|>, ...
  "raw_ends_with_im_end": bool,
  "errors": ["..."]         # empty when none
}
```

Summary (``summary.json``)
-------------------------------
```
{
  "mode": "coord" | "text",
  "total_read": int,
  "total_emitted": int,
  "counters": {"invalid_json": 0, ...},
  "error_codes": ["invalid_coord", ...]
}
```

Notes
-----
- ``gt`` and ``pred`` points are always pixel-space when emitted.
- ``points_text`` mirrors the pixel coords as a text string to unify downstream
  consumers that expect pure-text geometry.
- Polygons are kept verbatim (single ring) for COCO segmentation.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.common.geometry.bbox_parameterization import (
    AllowedBBoxFormat,
    DEFAULT_BBOX_FORMAT,
    normalize_bbox_format,
)
from src.common.coord_standardizer import CoordinateStandardizer
from src.common.geometry import flatten_points, has_coord_tokens
from src.infer.artifacts import (
    build_infer_resolved_meta,
    build_infer_summary_payload,
    ensure_infer_artifact_dirs,
    resolve_infer_artifact_paths,
    write_infer_summary,
)
from src.infer.backends import generate_batch, generate_hf_batch, generate_vllm_batch
from src.common.object_field_order import (
    ObjectFieldOrder,
    ObjectOrdering,
    normalize_object_field_order,
    normalize_object_ordering,
)
from src.common.detection_sequence import (
    COORDJSON_FORMAT,
    normalize_detection_sequence_format,
)
from src.coord_tokens.offset_adapter import (
    install_coord_offset_adapter,
    reattach_coord_offset_hooks,
)
from src.config.prompts import (
    DEFAULT_PROMPT_VARIANT,
    coord_mode_from_coord_tokens_enabled,
    get_template_prompt_hash,
    get_template_prompts,
    resolve_dense_prompt_variant_key,
)
from src.common.prediction_parsing import extract_special_tokens, load_prediction_dict
from src.common.paths import resolve_image_path_strict
from src.infer.checkpoints import (
    ResolvedInferenceCheckpoint,
    VLLM_ADAPTER_UNSUPPORTED_MESSAGE,
    resolve_inference_checkpoint,
)
from src.utils import get_logger

_DISTRIBUTED_SOURCE_INDEX_KEY = "_coordexp_source_index"
_DISTRIBUTED_MANIFEST_TIMEOUT_S = 1800.0

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

STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN = (
    "min_new_tokens_after_object_open"
)
STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_OPEN = "raw_text_object_open"
STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY = (
    "suppress_terminating_tokens_after_object_boundary"
)
STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY = (
    "suppress_special_terminating_tokens_after_object_boundary"
)
STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY = (
    "suppress_first_structural_closure_after_object_boundary"
)
STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY = (
    "steer_first_array_branch_to_next_object_after_object_boundary"
)
STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT = (
    "steer_bbox_tail_closure_to_next_object"
)
STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN = (
    "steer_bbox_tail_then_object_open"
)
STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE = (
    "steer_bbox_tail_then_object_open_once"
)
STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY = "raw_text_object_boundary"


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

    def apply_hf_stop_pressure(self, gen_kwargs: Dict[str, Any]) -> None:
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
            from src.infer.stop_pressure import (
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
            from src.infer.stop_pressure import (
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
            from src.infer.stop_pressure import (
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
            from src.infer.stop_pressure import (
                build_bbox_tail_then_object_open_once_steering_logits_processor,
            )

            return build_bbox_tail_then_object_open_once_steering_logits_processor(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(self.stop_pressure_logit_bias),
            )
        else:
            return None
        from src.infer.stop_pressure import (
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
    generated_token_text: Optional[List[str]] = None
    token_logprobs: Optional[List[float]] = None
    error: Optional[Exception] = None


@dataclass
class InferenceConfig:
    gt_jsonl: str
    model_checkpoint: str
    mode: Literal["coord", "text", "auto"]
    prompt_variant: str = DEFAULT_PROMPT_VARIANT
    bbox_format: AllowedBBoxFormat = DEFAULT_BBOX_FORMAT
    detection_sequence_format: str = COORDJSON_FORMAT
    object_field_order: ObjectFieldOrder = "desc_first"
    object_ordering: ObjectOrdering = "sorted"
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
    backend: Dict[str, Any] = field(default_factory=dict)

    # When mode=auto, how many GT records to scan (see OpenSpec for rules).
    detect_samples: int = 128


def detect_mode_from_gt(
    gt_jsonl: str,
    *,
    sample_size: int = 128,
) -> Tuple[Literal["coord", "text"], str]:
    """Deterministically resolve coord vs text from GT JSONL (OpenSpec).

    Scan up to the first N non-empty-object records.

    Strictness:
    - Blank lines and records with empty GT objects are skipped.
    - Operator-controlled input violations (malformed JSON, missing/invalid size,
      wrong schema) fail fast with actionable diagnostics (path + line).

    Resolution:
    - coord if any coord tokens are found in any geometry
    - coord if any numeric coordinate exceeds max(width, height)
    - else text
    """

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
                    f"Invalid width/height at {path}:{line_no}: width={width!r} height={height!r}"
                ) from exc
            if width_i <= 0 or height_i <= 0:
                raise ValueError(
                    f"Invalid width/height at {path}:{line_no}: width={width_i} height={height_i}"
                )

            objs = rec.get("objects") or rec.get("gt") or []
            if objs is None:
                objs = []
            if not isinstance(objs, list):
                raise ValueError(
                    f"GT record objects/gt must be a list at {path}:{line_no}; got {type(objs).__name__}"
                )
            if len(objs) == 0:
                continue

            max_dim = max(width_i, height_i)
            for obj in objs:
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"GT objects must be mappings at {path}:{line_no}; got {type(obj).__name__}"
                    )

                pts_raw = flatten_points(
                    obj.get("bbox_2d") or obj.get("poly") or obj.get("points") or []
                )
                if not pts_raw:
                    continue

                if has_coord_tokens(pts_raw):
                    return "coord", "coord_tokens_found"

                numeric = [v for v in pts_raw if isinstance(v, (int, float))]
                if numeric and max(numeric) > max_dim:
                    return "coord", "points_exceed_image"

            checked += 1

    if checked == 0:
        return "text", "no_valid_records"

    return "text", "within_image_bounds"


class RunCounters:
    """Aggregated counters for run-level summary."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
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

    def to_summary(self) -> Dict[str, Any]:
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


class InferenceEngine:
    def __init__(
        self,
        cfg: InferenceConfig,
        gen_cfg: GenerationConfig,
        *,
        logger=None,
    ) -> None:
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

        self.resolved_checkpoint: ResolvedInferenceCheckpoint = (
            resolve_inference_checkpoint(
                model_checkpoint=cfg.model_checkpoint,
                adapter_checkpoint=cfg.adapter_checkpoint,
            )
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
        self.detection_sequence_format = normalize_detection_sequence_format(
            cfg.detection_sequence_format
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
        self.bbox_format = normalize_bbox_format(
            cfg.bbox_format, path="infer.bbox_format"
        )
        self.cfg.bbox_format = self.bbox_format

        self.requested_mode = cfg.mode
        self.resolved_mode = cfg.mode
        self.mode_reason: Optional[str] = None
        if cfg.mode == "auto":
            self.resolved_mode, self.mode_reason = detect_mode_from_gt(
                cfg.gt_jsonl, sample_size=int(cfg.detect_samples or 128)
            )
        coord_mode = coord_mode_from_coord_tokens_enabled(
            self.resolved_mode == "coord"
        )

        self.system_prompt, self.user_prompt = get_template_prompts(
            ordering=self.object_ordering,
            coord_mode=coord_mode,
            prompt_variant=self.prompt_variant,
            object_field_order=self.object_field_order,
            bbox_format=self.bbox_format,
            detection_sequence_format=self.detection_sequence_format,
        )
        self.prompt_template_hash = get_template_prompt_hash(
            ordering=self.object_ordering,
            coord_mode=coord_mode,
            prompt_variant=self.prompt_variant,
            object_field_order=self.object_field_order,
            bbox_format=self.bbox_format,
            detection_sequence_format=self.detection_sequence_format,
        )

        # Shared parser/standardizer: always emit pixel-space points.
        self.coord = CoordinateStandardizer(
            self.resolved_mode,
            pred_coord_mode=cfg.pred_coord_mode,
            bbox_format=self.bbox_format,
        )

        self.processor: AutoProcessor | None = None
        self.model: Any | None = None
        self.vllm_llm: Any | None = None
        self.attn_implementation_requested: Optional[str] = None
        self.attn_implementation_selected: Optional[str] = None

    def _vllm_mode(self) -> str:
        mode_raw = (self.cfg.backend or {}).get("mode", "server")
        mode = str(mode_raw or "server").strip().lower()
        return "local" if mode == "local" else "server"

    def _load_vllm_local(self) -> None:
        if self.vllm_llm is not None:
            return

        # vLLM's CLI defaults to "spawn" for safety, but the library API does not.
        # Without this, using vLLM inside a process that already touched CUDA can fail with:
        #   "Cannot re-initialize CUDA in forked subprocess"
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM local backend requires the 'vllm' package. Install it in the ms env, or set infer.backend.type=hf."
            ) from exc

        model = str(self.cfg.backend.get("model") or self.cfg.model_checkpoint).strip()
        if not model:
            raise RuntimeError(
                "infer.backend.model (or infer.model_checkpoint) is required for vLLM local mode"
            )

        # Reuse server_options-style knobs when present for reproducibility.
        server_opts = self.cfg.backend.get("server_options") or {}
        allowed_local_media_path = str(
            self.cfg.root_image_dir
            or os.environ.get("ROOT_IMAGE_DIR")
            or Path(self.cfg.gt_jsonl).parent.resolve()
        )

        kwargs: Dict[str, Any] = {}

        tp = server_opts.get("vllm_tensor_parallel_size", None)
        if tp is not None:
            try:
                kwargs["tensor_parallel_size"] = int(tp)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "infer.backend.server_options.vllm_tensor_parallel_size must be int-compatible"
                ) from exc

        util = server_opts.get("vllm_gpu_memory_utilization", None)
        if util is not None:
            try:
                kwargs["gpu_memory_utilization"] = float(util)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "infer.backend.server_options.vllm_gpu_memory_utilization must be float-compatible"
                ) from exc

        # `max_model_len` is a vLLM kwarg (mirrors --max-model-len on the server).
        mml = server_opts.get("vllm_max_model_len", None)
        if mml is not None:
            try:
                kwargs["max_model_len"] = int(mml)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "infer.backend.server_options.vllm_max_model_len must be int-compatible"
                ) from exc

        # Note: local vLLM does not guarantee single OS process (it may manage workers
        # internally), but avoids running a separate HTTP server.
        self.vllm_llm = LLM(
            model=model,
            trust_remote_code=True,
            allowed_local_media_path=str(allowed_local_media_path or ""),
            seed=int(self.gen_cfg.seed) if self.gen_cfg.seed is not None else None,
            **kwargs,
        )

    def _vllm_sampling_params(self):
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM backend requires the 'vllm' package. Install it in the ms env, or set infer.backend.type=hf."
            ) from exc

        return SamplingParams(
            temperature=float(self.gen_cfg.temperature),
            top_p=float(self.gen_cfg.top_p),
            max_tokens=int(self.gen_cfg.max_new_tokens),
            repetition_penalty=float(self.gen_cfg.repetition_penalty or 1.0),
            seed=int(self.gen_cfg.seed) if self.gen_cfg.seed is not None else None,
        )

    def _seed(self) -> None:
        if self.gen_cfg.seed is None:
            return

        seed = int(self.gen_cfg.seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Best-effort determinism for HF generation.
        # (vLLM backend does not guarantee byte-identical outputs.)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False

    def _validate_vllm_backend(self) -> None:
        """Fail fast on global vLLM backend misconfiguration/unavailability."""
        if self._vllm_mode() == "local":
            # Local mode bypasses the HTTP connectivity checks.
            return

        base_url = str(
            self.cfg.backend.get("base_url") or os.environ.get("VLLM_BASE_URL") or ""
        ).strip()
        if not base_url:
            raise RuntimeError(
                "infer.backend.type=vllm requires infer.backend.base_url (or env VLLM_BASE_URL) when backend.mode=server. "
                "To run without a server, set infer.backend.mode=local. To disable vLLM, set infer.backend.type=hf."
            )

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "vLLM backend requires the 'requests' package. Install it in the ms env, or set infer.backend.type=hf."
            ) from exc

        # Best-effort preflight connectivity check to avoid per-sample generation_failed.
        # We keep it lightweight and do not require any specific response schema.
        timeout_s = float(self.cfg.backend.get("timeout_s", 3.0))
        root = base_url.rstrip("/")
        models_url = (
            (root + "/models") if root.endswith("/v1") else (root + "/v1/models")
        )
        try:
            resp = requests.get(models_url, timeout=timeout_s)
        except (requests.exceptions.RequestException, OSError, ValueError) as exc:
            raise RuntimeError(
                "Failed to reach vLLM server for infer.backend.type=vllm. "
                f"Tried GET {models_url}. To disable vLLM, set infer.backend.type=hf."
            ) from exc
        if int(getattr(resp, "status_code", 0) or 0) >= 400:
            raise RuntimeError(
                "vLLM server preflight check failed for infer.backend.type=vllm. "
                f"GET {models_url} returned status={resp.status_code}. To disable vLLM, set infer.backend.type=hf."
            )

    def load_model(self) -> None:
        backend = str(self.cfg.backend_type).lower().strip()
        resolved_base_model_checkpoint = str(
            self.cfg.resolved_base_model_checkpoint or self.cfg.model_checkpoint
        ).strip()
        resolved_adapter_checkpoint = str(
            self.cfg.resolved_adapter_checkpoint or ""
        ).strip()
        coord_offset_spec = None
        if self.resolved_checkpoint.adapter_info is not None:
            coord_offset_spec = self.resolved_checkpoint.adapter_info.coord_offset_spec

        # HF backend loads model+processor. For vLLM we support two modes:
        # - server: OpenAI-compatible HTTP server (default)
        # - local: in-process vLLM Python API (no HTTP server)
        if backend == "vllm":
            if resolved_adapter_checkpoint:
                raise RuntimeError(VLLM_ADAPTER_UNSUPPORTED_MESSAGE)
            self._seed()
            if self._vllm_mode() == "local":
                self._load_vllm_local()
                return
            self._validate_vllm_backend()
            return

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

            candidates: List[str] = []
            for cand in [attn_requested, "flash_attention_2", "sdpa", "eager"]:
                c = str(cand).strip().lower()
                if c and c not in candidates:
                    candidates.append(c)

            last_exc: Exception | None = None
            errors: List[str] = []
            for idx, cand in enumerate(candidates):
                try:
                    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        resolved_base_model_checkpoint,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=cand,
                    )
                    model = base_model.to(self.cfg.device)
                    if resolved_adapter_checkpoint:
                        if coord_offset_spec is not None:
                            install_coord_offset_adapter(
                                model,
                                coord_ids=coord_offset_spec.coord_ids,
                                tie_head=coord_offset_spec.tie_head,
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
                        if coord_offset_spec is not None:
                            reattached = reattach_coord_offset_hooks(model)
                            if reattached is None:
                                raise RuntimeError(
                                    "coord_offset_adapter was declared in the adapter "
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

                    # Best-effort cleanup between attempts.
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
            self.processor = AutoProcessor.from_pretrained(
                resolved_base_model_checkpoint, trust_remote_code=True
            )

        # Decoder-only models require left padding for correct generation.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            try:
                setattr(tokenizer, "padding_side", "left")
                if getattr(tokenizer, "pad_token_id", None) is None:
                    eos_token_id = getattr(tokenizer, "eos_token_id", None)
                    if eos_token_id is None:
                        raise ValueError(
                            "tokenizer.eos_token_id is required when tokenizer.pad_token_id is unset"
                        )
                    setattr(tokenizer, "pad_token_id", eos_token_id)
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Failed to configure tokenizer left-padding for inference."
                ) from exc

    def _resolve_image_path(self, jsonl_path: Path, image_rel: str) -> Path:
        root_image_dir: Path | None = None
        root_raw = str(self.cfg.root_image_dir or "").strip()
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
                f"image={image_rel!r} jsonl_dir={str(jsonl_path.parent)!r} root_image_dir={str(root_image_dir) if root_image_dir is not None else None!r}"
            )
        return resolved

    def _build_messages(self, image: Image.Image) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

    def _generate(self, image: Image.Image) -> str:
        backend = str(self.cfg.backend_type).strip().lower()
        if backend == "hf":
            return self._generate_hf(image)
        if backend == "vllm":
            return self._generate_vllm(image)
        raise ValueError(f"infer.backend.type must be hf|vllm, got {backend!r}")

    def _generate_batch(self, images: List[Image.Image]) -> List[GenerationResult]:
        return generate_batch(
            owner=self, images=images, result_factory=GenerationResult
        )

    def _generate_hf_batch(self, images: List[Image.Image]) -> List[GenerationResult]:
        return generate_hf_batch(
            owner=self,
            images=images,
            result_factory=GenerationResult,
        )

    def _generate_vllm_batch(self, images: List[Image.Image]) -> List[GenerationResult]:
        return generate_vllm_batch(
            owner=self,
            images=images,
            result_factory=GenerationResult,
        )

    def _generate_hf(self, image: Image.Image) -> str:
        assert self.model is not None and self.processor is not None

        messages = self._build_messages(image)
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        model_inputs = self.processor(
            text=prompt_text, images=[image], return_tensors="pt"
        )
        model_inputs = {k: v.to(self.cfg.device) for k, v in model_inputs.items()}
        attention_mask = model_inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            prompt_lengths = [
                int(value)
                for value in attention_mask.sum(dim=1).detach().cpu().tolist()
            ]
        else:
            prompt_lengths = [int(model_inputs["input_ids"].shape[1])]

        gen_kwargs = dict(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=self.gen_cfg.temperature > 0,
            temperature=max(1e-4, self.gen_cfg.temperature),
            top_p=self.gen_cfg.top_p,
            use_cache=True,
        )
        if self.gen_cfg.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.gen_cfg.repetition_penalty
        logits_processor = self.gen_cfg.build_hf_stop_pressure_logits_processor(
            tokenizer=self.processor.tokenizer,
            prompt_lengths=prompt_lengths,
        )
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        self.gen_cfg.apply_hf_stop_pressure(gen_kwargs)

        # NOTE: Do not pass `generator=` into `model.generate()`.
        #
        # Some upstream / remote-code model implementations (incl. some Qwen3-VL
        # checkpoints) treat unknown kwargs as `model_kwargs` and raise:
        #   "The following `model_kwargs` are not used by the model: ['generator']"
        #
        # We seed torch/CUDA globally in `_seed()` for deterministic sampling.
        with torch.inference_mode():
            gen_ids = self.model.generate(**model_inputs, **gen_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        gen_only = gen_ids[:, prompt_len:]
        raw_text = self.processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return raw_text

    def _generate_vllm(self, image: Image.Image) -> str:
        """Generate via vLLM.

        Modes:
        - server (default): OpenAI-compatible HTTP server
        - local: in-process vLLM Python API (no HTTP server)
        """

        if self._vllm_mode() == "local":
            # Use the same OpenAI-style message structure as the server, but route
            # through the in-process vLLM API.
            out = self._generate_vllm_local_batch([image])
            if not out:
                return ""
            if out[0].error is not None:
                raise out[0].error
            return out[0].text

        # Server mode (OpenAI-compatible HTTP requests).
        return self._generate_vllm_server(image)

    def _generate_vllm_server(self, image: Image.Image) -> str:
        """Generate via an OpenAI-compatible vLLM server (best-effort).

        Config:
        - infer.backend.base_url: e.g. http://127.0.0.1:8000 or http://127.0.0.1:8000/v1
        - infer.backend.model: optional; defaults to infer.model_checkpoint
        - infer.backend.timeout_s: optional
        """

        try:
            import base64
            import io

            import requests
        except ImportError as exc:
            raise RuntimeError(
                "vLLM backend requires the 'requests' package. Install it in the ms env, or set infer.backend.type=hf."
            ) from exc

        base_url = str(
            self.cfg.backend.get("base_url") or os.environ.get("VLLM_BASE_URL") or ""
        ).strip()
        if not base_url:
            raise RuntimeError(
                "infer.backend.type=vllm requires infer.backend.base_url (or env VLLM_BASE_URL). "
                "To run without a server, set infer.backend.mode=local. To disable vLLM, set infer.backend.type=hf."
            )

        model = str(self.cfg.backend.get("model") or self.cfg.model_checkpoint).strip()
        timeout_s = float(self.cfg.backend.get("timeout_s", 180.0))

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ]

        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            url = base_url + "/chat/completions"
        else:
            url = base_url + "/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(self.gen_cfg.temperature),
            "top_p": float(self.gen_cfg.top_p),
            "max_tokens": int(self.gen_cfg.max_new_tokens),
            "stream": False,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        if self.gen_cfg.repetition_penalty is not None:
            payload["repetition_penalty"] = float(self.gen_cfg.repetition_penalty)
        if self.gen_cfg.seed is not None:
            payload["seed"] = int(self.gen_cfg.seed)

        headers = {"Content-Type": "application/json"}

        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(
                f"vLLM server error status={resp.status_code}: {resp.text[:2000]}"
            )

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM server returned no choices: {data}")

        c0 = choices[0] if isinstance(choices, list) else choices
        if isinstance(c0, dict) and isinstance(c0.get("message"), dict):
            content = c0["message"].get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(str(p.get("text", "")))
                return "".join(parts)

        if isinstance(c0, dict) and "text" in c0:
            return str(c0.get("text") or "")

        raise RuntimeError(f"Unrecognized vLLM response schema: {data}")

    def _generate_vllm_local_batch(
        self, images: List[Image.Image]
    ) -> List[GenerationResult]:
        """Generate via the in-process vLLM Python API (no HTTP server)."""

        if not images:
            return []

        self._load_vllm_local()
        assert self.vllm_llm is not None

        try:
            import base64
            import io
        except ImportError as exc:
            return [GenerationResult(text="", error=exc) for _ in images]

        # Build OpenAI-style messages; vLLM supports a batch of message lists.
        msg_batch = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            msg_batch.append(
                [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    },
                ]
            )

        sp = self._vllm_sampling_params()
        outs = None
        chat_exc: Optional[Exception] = None
        try:
            outs = self.vllm_llm.chat(msg_batch, sampling_params=sp, use_tqdm=False)
        except Exception as exc:  # noqa: BLE001
            chat_exc = exc
            outs = None

        if outs is None:
            if chat_exc is None:
                chat_exc = RuntimeError("vLLM local chat() produced no outputs")
            return [GenerationResult(text="", error=chat_exc) for _ in images]

        results: List[GenerationResult] = []
        for o in outs:
            seqs = getattr(o, "outputs", None) or []
            text = seqs[0].text if seqs else ""
            results.append(GenerationResult(text=text, error=None))

        # vLLM should return one output per request; enforce alignment.
        if len(results) != len(images):
            raise RuntimeError(
                f"vLLM returned {len(results)} outputs for {len(images)} requests"
            )

        return results

    def _process_gt(
        self,
        record: Dict[str, Any],
        *,
        width: int,
        height: int,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        return self.coord.process_record_gt(
            record, width=width, height=height, errors=errors
        )

    def _process_pred(
        self,
        raw_text: str,
        *,
        width: int,
        height: int,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        return self.coord.process_prediction_text(
            raw_text, width=width, height=height, errors=errors
        )

    @staticmethod
    def _compact_objects(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip internal fields to the unified gt_vs_pred.jsonl schema."""
        compact: List[Dict[str, Any]] = []
        for obj in objs:
            if not isinstance(obj, dict):
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

    def _prepare_image(
        self, jsonl_path: Path, record: Dict[str, Any]
    ) -> Tuple[Path, Image.Image]:
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

        img_path = self._resolve_image_path(jsonl_path, image_field)
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, ValueError):
            return img_path, None
        return img_path, image

    def _distributed_paths(
        self,
        *,
        out_path: Path,
        summary_path: Path,
        trace_path: Optional[Path],
    ) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
        if not self.cfg.distributed_enabled:
            return out_path, summary_path, trace_path, None

        shard_dir = out_path.parent / "shards" / f"rank_{int(self.cfg.rank):05d}"
        worker_out_path = shard_dir / out_path.name
        worker_summary_path = shard_dir / summary_path.name
        worker_trace_path = shard_dir / trace_path.name if trace_path is not None else None
        manifest_path = shard_dir / "manifest.json"
        return worker_out_path, worker_summary_path, worker_trace_path, manifest_path

    def _write_distributed_manifest(
        self,
        *,
        manifest_path: Path,
        out_path: Path,
        summary_path: Path,
        trace_path: Optional[Path],
    ) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "complete",
            "rank": int(self.cfg.rank),
            "local_rank": int(self.cfg.local_rank),
            "world_size": int(self.cfg.world_size),
            "artifacts": {
                "gt_vs_pred_jsonl": str(out_path),
                "summary_json": str(summary_path),
                "pred_token_trace_jsonl": (
                    str(trace_path) if trace_path is not None else None
                ),
            },
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _wait_for_distributed_manifests(self, *, base_out_path: Path) -> list[Path]:
        shard_root = base_out_path.parent / "shards"
        manifest_paths = [
            shard_root / f"rank_{rank:05d}" / "manifest.json"
            for rank in range(int(self.cfg.world_size))
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

    def _merge_distributed_outputs(
        self,
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
                            f"Distributed shard row is missing {_DISTRIBUTED_SOURCE_INDEX_KEY}: {shard_out_path}:{line_no}"
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
                                f"Distributed trace row is missing {_DISTRIBUTED_SOURCE_INDEX_KEY}: {shard_trace_path}:{line_no}"
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
            trace_by_source = {source_index: record for source_index, record in merged_trace_rows}
            with final_trace_path.open("w", encoding="utf-8") as fout:
                for final_idx, (source_index, record) in enumerate(merged_rows):
                    trace_record = trace_by_source.get(source_index)
                    if trace_record is None:
                        continue
                    trace_record = dict(trace_record)
                    trace_record["line_idx"] = final_idx
                    fout.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

        return merged_counters

    def _preflight_inputs(self, jsonl_path: Path) -> None:
        """Validate operator-controlled inputs before any generation/eval work.

        This enforces strict fail-fast behavior for resolvable errors:
        - JSONL must be well-formed objects
        - width/height must be positive ints
        - images must resolve strictly and be readable
        - GT geometry must be valid for the resolved mode
        """

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
                    errors.append(
                        f"Malformed JSONL at {jsonl_path}:{line_no}: {snippet}"
                    )
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
                        f"Invalid width/height at {jsonl_path}:{line_no}: width={width_raw!r} height={height_raw!r} ({exc.__class__.__name__})"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue

                if width <= 0 or height <= 0:
                    errors.append(
                        f"Invalid width/height at {jsonl_path}:{line_no}: width={width} height={height}"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue

                images = record.get("images")
                if not isinstance(images, list) or len(images) != 1:
                    errors.append(
                        f"Input record must contain exactly one image in `images` at {jsonl_path}:{line_no}: images={images!r}"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue

                image_field = images[0]
                if not isinstance(image_field, str) or not image_field.strip():
                    errors.append(
                        f"Invalid image field in `images[0]` at {jsonl_path}:{line_no}: {image_field!r}"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue

                try:
                    img_path = self._resolve_image_path(jsonl_path, image_field)
                except FileNotFoundError as exc:
                    errors.append(str(exc))
                    if len(errors) >= max_errors:
                        break
                    continue

                try:
                    with Image.open(img_path) as im:
                        im.convert("RGB")
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"Failed to open image at {img_path} (from {jsonl_path}:{line_no}): {exc.__class__.__name__}: {exc}"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue

                objs_raw = record.get("objects")
                gt_raw = record.get("gt")
                if objs_raw is not None and not isinstance(objs_raw, list):
                    errors.append(
                        f"GT record 'objects' must be a list at {jsonl_path}:{line_no}; got {type(objs_raw).__name__}"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue
                if (
                    objs_raw is None
                    and gt_raw is not None
                    and not isinstance(gt_raw, list)
                ):
                    errors.append(
                        f"GT record 'gt' must be a list at {jsonl_path}:{line_no}; got {type(gt_raw).__name__}"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue

                gt_errors: List[str] = []
                _ = self._process_gt(
                    record, width=width, height=height, errors=gt_errors
                )
                if gt_errors:
                    errors.append(
                        f"Invalid GT geometry at {jsonl_path}:{line_no} (mode={self.resolved_mode}): {gt_errors}"
                    )
                    if len(errors) >= max_errors:
                        break

        if errors:
            msg = (
                "Inference preflight failed (operator-controlled input violations):\n"
                + "\n".join(f"- {e}" for e in errors)
            )
            raise ValueError(msg)

    def infer(self) -> Tuple[Path, Path]:
        jsonl_path = Path(self.cfg.gt_jsonl)
        backend = str(self.cfg.backend_type).strip().lower()
        out_path, summary_path, trace_path = resolve_infer_artifact_paths(
            cfg=self.cfg,
            backend=backend,
        )
        worker_out_path, worker_summary_path, worker_trace_path, manifest_path = (
            self._distributed_paths(
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

        # Fail fast on operator-controlled input violations before loading the model or
        # emitting any partial artifacts.
        self._preflight_inputs(jsonl_path)

        counters = RunCounters()
        self.load_model()

        ensure_infer_artifact_dirs(
            out_path=worker_out_path,
            summary_path=worker_summary_path,
            trace_path=worker_trace_path,
        )
        resolved_meta = build_infer_resolved_meta(
            owner=self,
            backend=backend,
            batch_size=batch_size,
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
            results = self._generate_batch(images)
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
                raw_ends_with_im_end = raw_text.endswith("<|im_end|>")
                raw_output_json = load_prediction_dict(raw_text)

                pred_errors: List[str] = []
                pred = self._process_pred(
                    raw_text,
                    width=int(p["width"]),
                    height=int(p["height"]),
                    errors=pred_errors,
                )
                pred = self._compact_objects(pred)

                error_codes = [_canonical(c) for c in pred_errors]
                error_entries = [_error_entry(c) for c in pred_errors]
                line_idx = int(counters.total_emitted)

                output = {
                    "image": p["image"],
                    "width": p["width"],
                    "height": p["height"],
                    "mode": self.resolved_mode,
                    "coord_mode": "pixel",
                    "gt": p["gt"],
                    "pred": pred,
                    "raw_output_json": raw_output_json,
                    "raw_special_tokens": raw_special_tokens,
                    "raw_ends_with_im_end": raw_ends_with_im_end,
                    "errors": error_codes,
                    "error_entries": error_entries,
                }
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
                    trace_record = {
                        "line_idx": line_idx,
                        "generated_token_text": list(res.generated_token_text),
                        "token_logprobs": list(res.token_logprobs),
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

        trace_cm = (
            worker_trace_path.open("w", encoding="utf-8")
            if worker_trace_path is not None
            else nullcontext(None)
        )
        with (
            jsonl_path.open("r", encoding="utf-8") as fin,
            worker_out_path.open("w", encoding="utf-8") as fout,
            trace_cm as ftrace,
            tqdm(
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
                    raise ValueError(
                        f"Non-object JSONL record at {jsonl_path}:{line_no}"
                    )

                width_raw = record.get("width")
                height_raw = record.get("height")
                try:
                    width = int(width_raw)
                    height = int(height_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid width/height at {jsonl_path}:{line_no}: width={width_raw!r} height={height_raw!r}"
                    ) from exc

                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"Invalid width/height at {jsonl_path}:{line_no}: width={width} height={height}"
                    )

                images = record.get("images")
                if not isinstance(images, list) or len(images) != 1:
                    raise ValueError(
                        f"Input record must contain exactly one image in `images` at {jsonl_path}:{line_no}: images={images!r}"
                    )

                image_key = images[0]
                if not isinstance(image_key, str) or not image_key.strip():
                    raise ValueError(
                        f"Invalid image field in `images[0]` at {jsonl_path}:{line_no}: {image_key!r}"
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
                gt = self._process_gt(
                    record, width=width, height=height, errors=gt_errors
                )
                if gt_errors:
                    raise ValueError(
                        f"Invalid GT geometry at {jsonl_path}:{line_no} (mode={self.resolved_mode}): {gt_errors}"
                    )
                gt = self._compact_objects(gt)

                _img_path, image_obj = self._prepare_image(jsonl_path, record)

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

        summary_payload = build_infer_summary_payload(
            owner=self,
            counters=counters,
            backend=backend,
            determinism=determinism,
            batch_size=batch_size,
        )
        write_infer_summary(
            summary_path=worker_summary_path,
            summary_payload=summary_payload,
        )

        if manifest_path is not None:
            self._write_distributed_manifest(
                manifest_path=manifest_path,
                out_path=worker_out_path,
                summary_path=worker_summary_path,
                trace_path=worker_trace_path,
            )
            if int(self.cfg.rank) == 0:
                manifest_paths = self._wait_for_distributed_manifests(
                    base_out_path=out_path,
                )
                merged_counters = self._merge_distributed_outputs(
                    manifest_paths=manifest_paths,
                    final_out_path=out_path,
                    final_summary_path=summary_path,
                    final_trace_path=trace_path,
                )
                final_summary_payload = build_infer_summary_payload(
                    owner=self,
                    counters=merged_counters,
                    backend=backend,
                    determinism=determinism,
                    batch_size=batch_size,
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


__all__ = ["GenerationConfig", "InferenceConfig", "InferenceEngine"]
