from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from PIL import Image
import torch
import yaml

from src.common.object_field_order import build_object_payload
from src.common.paths import resolve_image_path_strict
from src.utils.assistant_json import dumps_coordjson

_BENCHMARK_NAME = "val200"
_HYDRATION_VERSION = "raw_text_decode_bias_v1"
_REQUIRED_COORD_MODE = "norm1000_text"
_REQUIRED_HISTORY_SCOPE = "full_model_history"
_APPROVED_BASE_MODEL_PATH = (
    "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
)
_APPROVED_ADAPTER_PATH = (
    "/data/CoordExp/output/stage1_2b/"
    "coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/"
    "epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/"
    "v1-20260417-084341/checkpoint-552"
)
_DEFAULT_REPETITION_PENALTIES = (1.0, 1.02, 1.05, 1.10)
_BRANCHPOINT_TOP_K = 10
_SPECIAL_TERMINATOR_TEXTS = ("<|im_end|>", "<|endoftext|>")
_ALLOWED_STAGES = (
    "hydrate",
    "counterfactual_eos",
    "counterfactual_branchpoint_census",
    "counterfactual_repeat_penalty",
    "decode_val200_repeat_penalty",
    "decode_val200_stop_pressure",
    "report",
)


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: tuple[str, ...]


@dataclass(frozen=True)
class StudyControls:
    history_scope: str
    val200_source_jsonl: str
    val200_source_indices: tuple[int, ...]


@dataclass(frozen=True)
class DecodeConfig:
    dataset_variant: str
    pipeline_config: str
    val200_source_jsonl: str
    device: str
    semantic_device: str
    top_p: float
    max_new_tokens: int
    batch_size: int
    seed: int
    detect_samples: int
    stop_pressure_mode: str
    stop_pressure_min_new_tokens: int
    stop_pressure_logit_bias: float
    views: tuple[str, ...]
    semantic_model: str
    semantic_threshold: float
    semantic_batch_size: int
    num_workers: int
    metrics: str
    use_segm: bool
    strict_parse: bool
    lvis_max_dets: int
    f1ish_iou_thrs: tuple[float, ...]
    f1ish_pred_scope: str


@dataclass(frozen=True)
class ScoringConfig:
    device: str
    attn_implementation: str
    repetition_penalties: tuple[float, ...]


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    base_path: str
    adapter_path: str | None
    prompt_variant: str
    object_field_order: str
    coord_mode: str


@dataclass(frozen=True)
class StudyModels:
    base_only: ModelConfig
    base_plus_adapter: ModelConfig | None = None


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    study: StudyControls
    decode: DecodeConfig
    scoring: ScoringConfig
    models: StudyModels


def _resolved_config_payload(cfg: StudyConfig) -> dict[str, object]:
    return {
        "run": {
            "name": cfg.run.name,
            "output_dir": cfg.run.output_dir,
            "stages": list(cfg.run.stages),
        },
        "study": {
            "history_scope": cfg.study.history_scope,
            "val200_source_jsonl": cfg.study.val200_source_jsonl,
            "val200_source_indices": list(cfg.study.val200_source_indices),
        },
        "decode": {
            "dataset_variant": cfg.decode.dataset_variant,
            "pipeline_config": cfg.decode.pipeline_config,
            "val200_source_jsonl": cfg.decode.val200_source_jsonl,
            "device": cfg.decode.device,
            "semantic_device": cfg.decode.semantic_device,
            "top_p": cfg.decode.top_p,
            "max_new_tokens": cfg.decode.max_new_tokens,
            "batch_size": cfg.decode.batch_size,
            "seed": cfg.decode.seed,
            "detect_samples": cfg.decode.detect_samples,
            "stop_pressure_mode": cfg.decode.stop_pressure_mode,
            "stop_pressure_min_new_tokens": cfg.decode.stop_pressure_min_new_tokens,
            "stop_pressure_logit_bias": cfg.decode.stop_pressure_logit_bias,
            "views": list(cfg.decode.views),
            "semantic_model": cfg.decode.semantic_model,
            "semantic_threshold": cfg.decode.semantic_threshold,
            "semantic_batch_size": cfg.decode.semantic_batch_size,
            "num_workers": cfg.decode.num_workers,
            "metrics": cfg.decode.metrics,
            "use_segm": cfg.decode.use_segm,
            "strict_parse": cfg.decode.strict_parse,
            "lvis_max_dets": cfg.decode.lvis_max_dets,
            "f1ish_iou_thrs": list(cfg.decode.f1ish_iou_thrs),
            "f1ish_pred_scope": cfg.decode.f1ish_pred_scope,
        },
        "scoring": {
            "device": cfg.scoring.device,
            "attn_implementation": cfg.scoring.attn_implementation,
            "repetition_penalties": list(cfg.scoring.repetition_penalties),
        },
        "models": {
            model_cfg.alias: {
                "alias": model_cfg.alias,
                "base_path": model_cfg.base_path,
                "adapter_path": model_cfg.adapter_path,
                "prompt_variant": model_cfg.prompt_variant,
                "object_field_order": model_cfg.object_field_order,
                "coord_mode": model_cfg.coord_mode,
            }
            for model_cfg in _iter_model_cfgs(cfg.models)
        },
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("study config must be a mapping")
    return payload


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise TypeError(f"expected object JSONL row in {path}")
            rows.append(parsed)
    return rows


def _shared_repo_root(anchor: Path) -> Path:
    try:
        resolved_anchor = anchor.resolve()
        cwd = resolved_anchor if resolved_anchor.is_dir() else resolved_anchor.parent
        completed = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        common_dir = Path(completed.stdout.strip())
        return common_dir.parent if common_dir.name == ".git" else common_dir
    except (OSError, subprocess.CalledProcessError, ValueError):
        resolved_anchor = anchor.resolve()
        return resolved_anchor if resolved_anchor.is_dir() else resolved_anchor.parent


def _resolve_repo_path(raw_path: str, *, anchor: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    local_candidate = anchor / path
    if local_candidate.exists():
        return local_candidate
    return _shared_repo_root(anchor) / path


def _parse_model_config(*, raw_model: Mapping[str, object]) -> ModelConfig:
    return ModelConfig(
        alias=str(raw_model["alias"]),
        base_path=str(raw_model["base_path"]),
        adapter_path=(
            None
            if raw_model.get("adapter_path") is None
            else str(raw_model["adapter_path"])
        ),
        prompt_variant=str(raw_model["prompt_variant"]),
        object_field_order=str(raw_model["object_field_order"]),
        coord_mode=str(raw_model["coord_mode"]),
    )


def _iter_model_cfgs(models: StudyModels) -> tuple[ModelConfig, ...]:
    return tuple(
        model_cfg
        for model_cfg in (models.base_only, models.base_plus_adapter)
        if model_cfg is not None
    )


def _validate_model_surface(models: StudyModels) -> None:
    if models.base_only.alias != "base_only":
        raise ValueError("models.base_only.alias must be 'base_only'")
    if models.base_only.base_path != _APPROVED_BASE_MODEL_PATH:
        raise ValueError("models.base_only.base_path must use the approved raw-text base")
    if models.base_only.adapter_path is not None:
        raise ValueError("models.base_only.adapter_path must be null")
    if models.base_plus_adapter is not None:
        if models.base_plus_adapter.alias != "base_plus_adapter":
            raise ValueError("models.base_plus_adapter.alias must be 'base_plus_adapter'")
        if models.base_plus_adapter.base_path != _APPROVED_BASE_MODEL_PATH:
            raise ValueError(
                "models.base_plus_adapter.base_path must use the approved raw-text base"
            )
        if models.base_plus_adapter.adapter_path != _APPROVED_ADAPTER_PATH:
            raise ValueError(
                "models.base_plus_adapter.adapter_path must use the approved raw-text adapter"
            )
    for model_cfg in _iter_model_cfgs(models):
        if model_cfg.coord_mode != _REQUIRED_COORD_MODE:
            raise ValueError(
                f"{model_cfg.alias} must use coord_mode={_REQUIRED_COORD_MODE!r}"
            )


def _validate_run_stages(stages: Sequence[str]) -> None:
    if not stages:
        raise ValueError("run.stages must not be empty")
    invalid = [stage for stage in stages if stage not in _ALLOWED_STAGES]
    if invalid:
        raise ValueError(f"unsupported decode-bias stage(s): {', '.join(invalid)}")


def _validate_scoring_surface(scoring: ScoringConfig) -> None:
    penalties = tuple(float(value) for value in scoring.repetition_penalties)
    if penalties != _DEFAULT_REPETITION_PENALTIES:
        raise ValueError(
            "scoring.repetition_penalties must be "
            f"{list(_DEFAULT_REPETITION_PENALTIES)!r}"
        )


def _validate_decode_surface(decode: DecodeConfig) -> None:
    if not decode.dataset_variant.strip():
        raise ValueError("decode.dataset_variant must not be empty")
    if not decode.pipeline_config.strip():
        raise ValueError("decode.pipeline_config must not be empty")
    if not decode.val200_source_jsonl.strip():
        raise ValueError("decode.val200_source_jsonl must not be empty")
    if not decode.views:
        raise ValueError("decode.views must not be empty")
    if decode.stop_pressure_mode not in {
        "steer_bbox_tail_closure_to_next_object",
        "steer_bbox_tail_then_object_open",
        "steer_bbox_tail_then_object_open_once",
        "steer_first_array_branch_to_next_object_after_object_boundary",
        "suppress_first_structural_closure_after_object_boundary",
        "suppress_terminating_tokens_after_object_boundary",
        "suppress_special_terminating_tokens_after_object_boundary",
    }:
        raise ValueError(
            "decode.stop_pressure_mode must be "
            "'steer_bbox_tail_closure_to_next_object' or "
            "'steer_bbox_tail_then_object_open' or "
            "'steer_bbox_tail_then_object_open_once' or "
            "'steer_first_array_branch_to_next_object_after_object_boundary' or "
            "'suppress_first_structural_closure_after_object_boundary' or "
            "'suppress_terminating_tokens_after_object_boundary' or "
            "'suppress_special_terminating_tokens_after_object_boundary'"
        )
    if int(decode.stop_pressure_min_new_tokens) < 0:
        raise ValueError("decode.stop_pressure_min_new_tokens must be >= 0")
    if (
        decode.stop_pressure_mode
        in {
            "steer_bbox_tail_closure_to_next_object",
            "steer_bbox_tail_then_object_open",
            "steer_bbox_tail_then_object_open_once",
            "steer_first_array_branch_to_next_object_after_object_boundary",
        }
        and float(decode.stop_pressure_logit_bias) <= 0.0
    ):
        raise ValueError(
            "decode.stop_pressure_logit_bias must be > 0 for continuation steering"
        )
    if (
        decode.stop_pressure_mode
        not in {
            "steer_bbox_tail_closure_to_next_object",
            "steer_bbox_tail_then_object_open",
            "steer_bbox_tail_then_object_open_once",
            "steer_first_array_branch_to_next_object_after_object_boundary",
        }
        and float(decode.stop_pressure_logit_bias) != 0.0
    ):
        raise ValueError(
            "decode.stop_pressure_logit_bias must be 0 unless continuation steering is enabled"
        )


def load_study_config(config_path: Path) -> StudyConfig:
    raw = _load_yaml(config_path)
    run_raw = raw["run"]
    study_raw = raw["study"]
    decode_raw = raw.get("decode") or {}
    models_raw = raw["models"]
    scoring_raw = raw.get("scoring") or {}
    if not isinstance(decode_raw, dict):
        raise TypeError("decode config must be a mapping when provided")
    if not isinstance(scoring_raw, dict):
        raise TypeError("scoring config must be a mapping when provided")
    cfg = StudyConfig(
        run=RunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
            stages=tuple(str(value) for value in run_raw["stages"]),
        ),
        study=StudyControls(
            history_scope=str(study_raw["history_scope"]),
            val200_source_jsonl=str(study_raw["val200_source_jsonl"]),
            val200_source_indices=tuple(
                int(value) for value in study_raw["val200_source_indices"]
            ),
        ),
        decode=DecodeConfig(
            dataset_variant=str(decode_raw.get("dataset_variant") or "standard").strip(),
            pipeline_config=str(
                decode_raw.get("pipeline_config") or "configs/infer/pipeline.yaml"
            ).strip(),
            val200_source_jsonl=str(
                decode_raw.get("val200_source_jsonl") or study_raw["val200_source_jsonl"]
            ).strip(),
            device=str(decode_raw.get("device") or "cuda:0").strip(),
            semantic_device=str(
                decode_raw.get("semantic_device") or decode_raw.get("device") or "cuda:0"
            ).strip(),
            top_p=float(decode_raw.get("top_p", 0.9)),
            max_new_tokens=int(decode_raw.get("max_new_tokens", 3084)),
            batch_size=int(decode_raw.get("batch_size", 4)),
            seed=int(decode_raw.get("seed", 42)),
            detect_samples=int(decode_raw.get("detect_samples", 128)),
            stop_pressure_mode=str(
                decode_raw.get("stop_pressure_mode")
                or "suppress_terminating_tokens_after_object_boundary"
            ).strip(),
            stop_pressure_min_new_tokens=int(
                decode_raw.get("stop_pressure_min_new_tokens", 0)
            ),
            stop_pressure_logit_bias=float(
                decode_raw.get("stop_pressure_logit_bias", 0.0)
            ),
            views=tuple(
                str(value)
                for value in decode_raw.get(
                    "views",
                    ("coco_real", "coco_real_strict", "coco_real_strict_plausible"),
                )
            ),
            semantic_model=str(
                decode_raw.get("semantic_model") or "model_cache/all-MiniLM-L6-v2-local"
            ).strip(),
            semantic_threshold=float(decode_raw.get("semantic_threshold", 0.5)),
            semantic_batch_size=int(decode_raw.get("semantic_batch_size", 64)),
            num_workers=int(decode_raw.get("num_workers", 8)),
            metrics=str(decode_raw.get("metrics") or "both").strip(),
            use_segm=bool(decode_raw.get("use_segm", False)),
            strict_parse=bool(decode_raw.get("strict_parse", False)),
            lvis_max_dets=int(decode_raw.get("lvis_max_dets", 300)),
            f1ish_iou_thrs=tuple(
                float(value) for value in decode_raw.get("f1ish_iou_thrs", (0.3, 0.5))
            ),
            f1ish_pred_scope=str(
                decode_raw.get("f1ish_pred_scope") or "annotated"
            ).strip(),
        ),
        scoring=ScoringConfig(
            device=str(scoring_raw.get("device") or "cuda:0").strip(),
            attn_implementation=str(
                scoring_raw.get("attn_implementation") or "auto"
            ).strip(),
            repetition_penalties=tuple(
                float(value)
                for value in scoring_raw.get(
                    "repetition_penalties",
                    list(_DEFAULT_REPETITION_PENALTIES),
                )
            ),
        ),
        models=StudyModels(
            base_only=_parse_model_config(raw_model=models_raw["base_only"]),
            base_plus_adapter=(
                None
                if "base_plus_adapter" not in models_raw
                else _parse_model_config(raw_model=models_raw["base_plus_adapter"])
            ),
        ),
    )
    if cfg.study.history_scope != _REQUIRED_HISTORY_SCOPE:
        raise ValueError(
            f"study.history_scope must be {_REQUIRED_HISTORY_SCOPE!r} for decode-bias"
        )
    if not cfg.study.val200_source_indices:
        raise ValueError("study.val200_source_indices must not be empty")
    if min(cfg.study.val200_source_indices) < 0:
        raise ValueError("study.val200_source_indices must be non-negative")
    _validate_run_stages(cfg.run.stages)
    _validate_decode_surface(cfg.decode)
    _validate_scoring_surface(cfg.scoring)
    _validate_model_surface(cfg.models)
    return cfg


def _extract_bbox_xyxy(obj: Mapping[str, object]) -> list[int]:
    raw_bbox = obj.get("bbox_2d")
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError("raw-text decode-bias hydration requires bbox_2d xyxy lists")
    return [int(value) for value in raw_bbox]


def _bbox_to_norm1000(
    bbox_xyxy: Sequence[int],
    *,
    width: int,
    height: int,
) -> list[int]:
    denom_w = max(float(width), 1.0)
    denom_h = max(float(height), 1.0)
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    return [
        int(max(0, min(999, round((float(x1) / denom_w) * 1000.0)))),
        int(max(0, min(999, round((float(y1) / denom_h) * 1000.0)))),
        int(max(0, min(999, round((float(x2) / denom_w) * 1000.0)))),
        int(max(0, min(999, round((float(y2) / denom_h) * 1000.0)))),
    ]


def _normalize_source_objects(source_row: Mapping[str, object]) -> list[dict[str, object]]:
    width = int(source_row["width"])
    height = int(source_row["height"])
    raw_objects = source_row.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("source_row.objects must contain at least one object")
    normalized: list[dict[str, object]] = []
    for raw_obj in raw_objects:
        if not isinstance(raw_obj, Mapping):
            raise ValueError("source_row.objects entries must be mappings")
        normalized.append(
            {
                "desc": str(raw_obj.get("desc") or ""),
                "bbox_2d": _bbox_to_norm1000(
                    _extract_bbox_xyxy(raw_obj),
                    width=width,
                    height=height,
                ),
            }
        )
    return normalized


def _render_assistant_text(
    *,
    objects: Sequence[Mapping[str, object]],
    object_field_order: str,
) -> str:
    rendered = [
        build_object_payload(
            desc=str(obj.get("desc") or ""),
            geometry_key="bbox_2d",
            geometry_value=list(obj["bbox_2d"]),
            object_field_order=object_field_order,
        )
        for obj in objects
    ]
    return dumps_coordjson({"objects": rendered})


def _render_open_prefix_assistant_text(
    *,
    objects: Sequence[Mapping[str, object]],
    object_field_order: str,
) -> str:
    closed = _render_assistant_text(
        objects=objects,
        object_field_order=object_field_order,
    )
    if not closed.endswith("]}"):
        raise ValueError("unexpected raw-text assistant surface while opening prefix")
    return closed[:-2]


def hydrate_case_rows(*, case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    hydrated: list[dict[str, object]] = []
    for row in case_rows:
        source_row = row.get("source_row")
        if not isinstance(source_row, Mapping):
            raise ValueError("case row missing source_row for hydration")
        object_field_order = str(row.get("object_field_order") or "desc_first")
        normalized_objects = _normalize_source_objects(source_row)
        prefix_objects = [normalized_objects[0]]
        continue_object = (
            normalized_objects[1]
            if len(normalized_objects) > 1
            else dict(normalized_objects[0])
        )
        baseline_assistant_text = _render_open_prefix_assistant_text(
            objects=prefix_objects,
            object_field_order=object_field_order,
        )
        stop_now_candidate_text = _render_assistant_text(
            objects=prefix_objects,
            object_field_order=object_field_order,
        )
        continue_with_gt_candidate_text = _render_assistant_text(
            objects=[*prefix_objects, continue_object],
            object_field_order=object_field_order,
        )
        exact_duplicate_candidate_text = _render_assistant_text(
            objects=[*prefix_objects, dict(prefix_objects[-1])],
            object_field_order=object_field_order,
        )
        hydrated_row = {key: value for key, value in row.items() if key != "source_row"}
        hydrated_row.update(
            {
                "benchmark_name": _BENCHMARK_NAME,
                "baseline_assistant_text": baseline_assistant_text,
                "stop_now_candidate_text": stop_now_candidate_text,
                "continue_with_gt_candidate_text": continue_with_gt_candidate_text,
                "exact_duplicate_candidate_text": exact_duplicate_candidate_text,
                "hydration_version": _HYDRATION_VERSION,
                "source_object_count": len(normalized_objects),
                "prefix_object_count": len(prefix_objects),
            }
        )
        hydrated.append(hydrated_row)
    return hydrated


def _build_case_rows(
    *,
    cfg: StudyConfig,
    source_rows: Sequence[Mapping[str, object]],
    source_jsonl_path: Path,
) -> list[dict[str, object]]:
    case_rows: list[dict[str, object]] = []
    model_cfgs = _iter_model_cfgs(cfg.models)
    for source_index in cfg.study.val200_source_indices:
        if source_index >= len(source_rows):
            raise IndexError(
                f"val200 source index {source_index} out of range for {source_jsonl_path}"
            )
        source_row = dict(source_rows[source_index])
        for model_cfg in model_cfgs:
            case_rows.append(
                {
                    "case_uid": f"{model_cfg.alias}:{_BENCHMARK_NAME}:{source_index}",
                    "model_alias": model_cfg.alias,
                    "base_path": model_cfg.base_path,
                    "adapter_path": model_cfg.adapter_path,
                    "coord_mode": model_cfg.coord_mode,
                    "prompt_variant": model_cfg.prompt_variant,
                    "object_field_order": model_cfg.object_field_order,
                    "source_jsonl": str(source_jsonl_path),
                    "source_index": int(source_index),
                    "image_id": source_row.get("image_id"),
                    "source_row": source_row,
                }
            )
    return case_rows


def _image_field_from_source_row(source_row: Mapping[str, object]) -> str:
    images = source_row.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    file_name = source_row.get("file_name")
    if file_name:
        return str(file_name)
    image_field = source_row.get("image")
    if image_field:
        return str(image_field)
    raise ValueError("source_row missing image field")


def _load_source_image(
    *,
    source_row: Mapping[str, object],
    source_jsonl_path: Path,
) -> Image.Image:
    image_rel = _image_field_from_source_row(source_row)
    resolved = resolve_image_path_strict(
        image_rel,
        jsonl_dir=source_jsonl_path.parent,
        root_image_dir=source_jsonl_path.parent,
    )
    if resolved is None:
        raise FileNotFoundError(f"Unable to resolve image path {image_rel!r}")
    with Image.open(resolved) as handle:
        return handle.convert("RGB")


def _make_teacher_forced_scorer(
    *,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
) -> object:
    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer

    checkpoint = Path(model_cfg.adapter_path or model_cfg.base_path)
    return TeacherForcedScorer(
        checkpoint_path=checkpoint,
        device=scoring_cfg.device,
        attn_implementation=scoring_cfg.attn_implementation,
        coord_mode=model_cfg.coord_mode,
    )


def _get_or_create_scorer(
    *,
    scorer_cache: dict[str, object] | None,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
) -> object:
    if scorer_cache is None:
        return _make_teacher_forced_scorer(
            model_cfg=model_cfg,
            scoring_cfg=scoring_cfg,
        )
    cached = scorer_cache.get(model_cfg.alias)
    if cached is not None:
        return cached
    scorer = _make_teacher_forced_scorer(
        model_cfg=model_cfg,
        scoring_cfg=scoring_cfg,
    )
    scorer_cache[model_cfg.alias] = scorer
    return scorer


def _score_candidate_token_rows(
    *,
    scorer: object,
    image: Image.Image,
    baseline_assistant_text: str,
    candidate_assistant_text: str,
    prompt_variant: str,
    object_field_order: str,
    repetition_penalty: float = 1.0,
) -> dict[str, object]:
    from src.analysis.raw_text_coordinate_continuation_scoring import (
        build_candidate_continuation_span,
    )

    candidate = build_candidate_continuation_span(
        tokenizer=getattr(scorer, "tokenizer"),
        baseline_assistant_text=baseline_assistant_text,
        candidate_assistant_text=candidate_assistant_text,
    )
    prepared = scorer.prepare_example(
        image=image,
        assistant_text=candidate_assistant_text,
        desc_positions_rel=[],
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
    )
    assistant_start = int(getattr(prepared, "assistant_start"))
    absolute_positions = [
        assistant_start + int(pos) for pos in candidate["assistant_relative_positions"]
    ]
    token_rows = scorer.score_prepared_span_token_rows(
        prepared=prepared,
        image=image,
        positions=absolute_positions,
        repetition_penalty=float(repetition_penalty),
    )
    return {
        "candidate_assistant_text": candidate_assistant_text,
        "assistant_relative_positions": list(candidate["assistant_relative_positions"]),
        "absolute_positions": absolute_positions,
        "assistant_start": assistant_start,
        "prepared": prepared,
        "token_rows": token_rows,
    }


def _aggregate_token_rows(
    token_rows: Sequence[Mapping[str, object]],
    *,
    score_key: str,
) -> dict[str, object]:
    values = [float(row[score_key]) for row in token_rows]
    if not values:
        return {
            "token_count": 0,
            "sum_logprob": 0.0,
            "mean_logprob": None,
        }
    total = float(sum(values))
    return {
        "token_count": len(values),
        "sum_logprob": total,
        "mean_logprob": float(total / len(values)),
    }


def _summarize_group_token_rows(
    grouped_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for group_name, rows in grouped_rows.items():
        raw_summary = _aggregate_token_rows(rows, score_key="raw_logprob")
        processed_summary = _aggregate_token_rows(rows, score_key="processed_logprob")
        summary[group_name] = {
            "token_count": int(raw_summary["token_count"]),
            "raw_sum_logprob": float(raw_summary["sum_logprob"]),
            "raw_mean_logprob": raw_summary["mean_logprob"],
            "processed_sum_logprob": float(processed_summary["sum_logprob"]),
            "processed_mean_logprob": processed_summary["mean_logprob"],
            "processed_minus_raw_sum_logprob": float(
                float(processed_summary["sum_logprob"])
                - float(raw_summary["sum_logprob"])
            ),
        }
    return summary


def _model_cfg_by_alias(models: StudyModels) -> dict[str, ModelConfig]:
    return {model_cfg.alias: model_cfg for model_cfg in _iter_model_cfgs(models)}


def _materialize_decode_val200_subset(
    *,
    cfg: StudyConfig,
    run_dir: Path,
    repo_root: Path,
) -> dict[str, object]:
    from src.analysis.coord_family_text_subset import materialize_text_pixel_subset

    source_jsonl_path = _resolve_repo_path(cfg.decode.val200_source_jsonl, anchor=repo_root)
    source_rows = _read_jsonl(source_jsonl_path)
    selected_rows: list[dict[str, object]] = []
    for source_index in cfg.study.val200_source_indices:
        if source_index >= len(source_rows):
            raise IndexError(
                f"decode source index {source_index} out of range for {source_jsonl_path}"
            )
        selected_rows.append(dict(source_rows[source_index]))

    subset_dir = run_dir / "decode_val200_inputs" / "subset"
    sampled_norm_path = subset_dir / "sampled.norm.jsonl"
    sampled_text_pixel_path = subset_dir / "sampled.text_pixel.jsonl"
    sampled_meta_path = subset_dir / "sampled.norm.jsonl.meta.json"
    sampled_images = [_image_field_from_source_row(row) for row in selected_rows]
    root_image_dir = source_jsonl_path.parent.resolve()
    _write_jsonl(sampled_norm_path, selected_rows)
    _write_json(
        sampled_meta_path,
        {
            "input_path": str(source_jsonl_path),
            "output_path": str(sampled_norm_path),
            "num_samples": len(selected_rows),
            "seed": None,
            "total_nonempty_lines_seen": len(source_rows),
            "sampled_source_line_indices": list(cfg.study.val200_source_indices),
            "sampled_images": sampled_images,
            "root_image_dir": str(root_image_dir),
            "input_mtime_ns": int(source_jsonl_path.stat().st_mtime_ns),
            "dataset_variant": cfg.decode.dataset_variant,
        },
    )
    text_pixel_summary = materialize_text_pixel_subset(
        sampled_norm_path,
        sampled_text_pixel_path,
    )
    subset_manifest_path = subset_dir / "subset_manifest.json"
    _write_json(
        subset_manifest_path,
        {
            "stage": "decode_val200_subset",
            "benchmark_scope": _BENCHMARK_NAME,
            "dataset_variant": cfg.decode.dataset_variant,
            "subset_path": str(sampled_norm_path),
            "text_pixel_subset_path": str(sampled_text_pixel_path),
            "num_subset_records": len(selected_rows),
            "root_image_dir": str(root_image_dir),
            "subset_meta_path": str(sampled_meta_path),
            "text_pixel_meta_path": str(
                sampled_text_pixel_path.with_suffix(".jsonl.meta.json")
            ),
            "source_line_indices": list(cfg.study.val200_source_indices),
            "coord_tokens_used_for_generation": False,
        },
    )
    return {
        "sampled_norm_jsonl": str(sampled_norm_path),
        "sampled_text_pixel_jsonl": str(sampled_text_pixel_path),
        "sampled_meta_path": str(sampled_meta_path),
        "subset_manifest_path": str(subset_manifest_path),
        "root_image_dir": str(root_image_dir),
        "row_count": len(selected_rows),
        "source_line_indices": list(cfg.study.val200_source_indices),
        "dataset_variant": cfg.decode.dataset_variant,
        "text_pixel_summary": text_pixel_summary,
    }


def _build_decode_infer_overrides(
    *,
    cfg: StudyConfig,
    model_cfg: ModelConfig,
    subset_artifacts: Mapping[str, object],
    run_output_dir: Path,
    run_name: str,
    repetition_penalty: float,
    stop_pressure_active: bool,
) -> dict[str, object]:
    checkpoint_path = str(model_cfg.adapter_path or model_cfg.base_path)
    overrides: dict[str, object] = {
        "run.output_dir": str(run_output_dir),
        "run.name": run_name,
        "run.root_image_dir": str(subset_artifacts["root_image_dir"]),
        "stages.infer": True,
        "stages.eval": False,
        "stages.vis": False,
        "infer.gt_jsonl": str(subset_artifacts["sampled_text_pixel_jsonl"]),
        "infer.model_checkpoint": checkpoint_path,
        "infer.prompt_variant": model_cfg.prompt_variant,
        "infer.object_field_order": model_cfg.object_field_order,
        "infer.object_ordering": "sorted",
        "infer.mode": "text",
        "infer.bbox_format": "xyxy",
        "infer.pred_coord_mode": "norm1000",
        "infer.backend.type": "hf",
        "infer.generation.temperature": 0.0,
        "infer.generation.top_p": cfg.decode.top_p,
        "infer.generation.max_new_tokens": cfg.decode.max_new_tokens,
        "infer.generation.repetition_penalty": float(repetition_penalty),
        "infer.generation.batch_size": cfg.decode.batch_size,
        "infer.generation.seed": cfg.decode.seed,
        "infer.device": cfg.decode.device,
        "infer.limit": 0,
        "infer.detect_samples": cfg.decode.detect_samples,
    }
    if stop_pressure_active:
        overrides.update(
            {
                "infer.generation.stop_pressure.mode": cfg.decode.stop_pressure_mode,
                "infer.generation.stop_pressure.min_new_tokens": (
                    cfg.decode.stop_pressure_min_new_tokens
                ),
                "infer.generation.stop_pressure.logit_bias": (
                    cfg.decode.stop_pressure_logit_bias
                ),
                "infer.generation.stop_pressure.trigger_rule": (
                    "raw_text_object_boundary"
                ),
            }
        )
    return overrides


def _build_decode_postop_config(*, run_dir: Path) -> dict[str, object]:
    return {
        "artifacts": {
            "run_dir": str(run_dir),
        },
        "confidence": {
            "fusion": {
                "w_geom": 0.7,
                "w_desc": 0.3,
            },
            "desc_span_policy": "best_effort",
            "empty_desc_policy": "geom_only",
        },
    }


def _build_decode_proxy_eval_config(
    *,
    cfg: StudyConfig,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "artifacts": {
            "scored_jsonl": str(run_dir / "gt_vs_pred_scored.jsonl"),
            "proxy_views_dir": str(run_dir / "proxy_eval_views"),
            "output_root": str(run_dir),
            "summary_json": str(run_dir / "proxy_eval_bundle_summary.json"),
        },
        "views": list(cfg.decode.views),
        "metadata_namespace": "coordexp_proxy_supervision",
        "eval": {
            "metrics": cfg.decode.metrics,
            "use_segm": cfg.decode.use_segm,
            "strict_parse": cfg.decode.strict_parse,
            "lvis_max_dets": cfg.decode.lvis_max_dets,
            "overlay": False,
            "overlay_k": 12,
            "num_workers": cfg.decode.num_workers,
            "semantic_model": cfg.decode.semantic_model,
            "semantic_threshold": cfg.decode.semantic_threshold,
            "semantic_device": cfg.decode.semantic_device,
            "semantic_batch_size": cfg.decode.semantic_batch_size,
            "f1ish_iou_thrs": list(cfg.decode.f1ish_iou_thrs),
            "f1ish_pred_scope": cfg.decode.f1ish_pred_scope,
        },
    }


def _summarize_decode_health(gt_vs_pred_jsonl: Path) -> dict[str, object]:
    rows = _read_jsonl(gt_vs_pred_jsonl)
    row_count = len(rows)
    parse_valid_count = 0
    nonempty_count = 0
    total_prediction_count = 0
    duplicate_like_row_count = 0
    repeated_desc_row_count = 0
    for row in rows:
        errors = row.get("errors")
        error_entries = row.get("error_entries")
        if not errors and not error_entries:
            parse_valid_count += 1
        predictions = row.get("pred")
        if isinstance(predictions, list) and predictions:
            nonempty_count += 1
            total_prediction_count += len(predictions)
            keys: dict[tuple[str, tuple[int, ...]], int] = {}
            desc_counts: dict[str, int] = {}
            for pred in predictions:
                if not isinstance(pred, Mapping):
                    continue
                desc = str(pred.get("desc") or "").strip().lower()
                points_raw = pred.get("points")
                points = (
                    tuple(int(value) for value in points_raw)
                    if isinstance(points_raw, list)
                    else tuple()
                )
                keys[(desc, points)] = keys.get((desc, points), 0) + 1
                desc_counts[desc] = desc_counts.get(desc, 0) + 1
            if any(count > 1 for count in keys.values()):
                duplicate_like_row_count += 1
            if any(desc and count > 1 for desc, count in desc_counts.items()):
                repeated_desc_row_count += 1
    denom = float(row_count) if row_count else 1.0
    return {
        "row_count": row_count,
        "parse_valid_count": parse_valid_count,
        "parse_valid_rate": float(parse_valid_count / denom),
        "nonempty_count": nonempty_count,
        "nonempty_rate": float(nonempty_count / denom),
        "total_prediction_count": total_prediction_count,
        "duplicate_like_row_count": duplicate_like_row_count,
        "duplicate_like_rate": float(duplicate_like_row_count / denom),
        "repeated_desc_row_count": repeated_desc_row_count,
        "repeated_desc_rate": float(repeated_desc_row_count / denom),
        "finish_reason_counts": {},
        "finish_reason_available": False,
    }


def _extract_bundle_metric(
    bundle_summary: Mapping[str, object],
    *,
    view: str,
    metric_key: str,
) -> object:
    views = bundle_summary.get("views")
    if not isinstance(views, Mapping):
        return None
    view_payload = views.get(view)
    if not isinstance(view_payload, Mapping):
        return None
    metrics = view_payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return metrics.get(metric_key)


def _run_decode_eval_workflow(
    *,
    cfg: StudyConfig,
    repo_root: Path,
    stage_dir: Path,
    stage_name: str,
    model_cfg: ModelConfig,
    subset_artifacts: Mapping[str, object],
    run_name: str,
    repetition_penalty: float,
    stop_pressure_active: bool,
) -> dict[str, object]:
    from src.eval.confidence_postop import run_confidence_postop_from_config
    from src.eval.proxy_eval_bundle import (
        _resolve_artifacts as _resolve_proxy_eval_artifacts,
        options_from_config as proxy_eval_options_from_config,
        run_proxy_eval_bundle,
    )
    from src.infer.pipeline import run_pipeline

    pipeline_config_path = _resolve_repo_path(cfg.decode.pipeline_config, anchor=repo_root)
    run_output_dir = stage_dir / "runs" / model_cfg.alias
    infer_overrides = _build_decode_infer_overrides(
        cfg=cfg,
        model_cfg=model_cfg,
        subset_artifacts=subset_artifacts,
        run_output_dir=run_output_dir,
        run_name=run_name,
        repetition_penalty=repetition_penalty,
        stop_pressure_active=stop_pressure_active,
    )
    config_dir = stage_dir / "configs" / model_cfg.alias / run_name
    infer_run_spec_path = config_dir / "infer_run_spec.yaml"
    _write_yaml(
        infer_run_spec_path,
        {
            "pipeline_config": str(pipeline_config_path),
            "overrides": infer_overrides,
        },
    )
    artifacts = run_pipeline(config_path=pipeline_config_path, overrides=infer_overrides)
    postop_config = _build_decode_postop_config(run_dir=artifacts.run_dir)
    postop_config_path = config_dir / "confidence_postop.yaml"
    _write_yaml(postop_config_path, postop_config)
    confidence_summary = run_confidence_postop_from_config(postop_config)
    proxy_eval_config = _build_decode_proxy_eval_config(cfg=cfg, run_dir=artifacts.run_dir)
    proxy_eval_config_path = config_dir / "proxy_eval_bundle.yaml"
    _write_yaml(proxy_eval_config_path, proxy_eval_config)
    bundle_summary = run_proxy_eval_bundle(
        _resolve_proxy_eval_artifacts(proxy_eval_config),
        options=proxy_eval_options_from_config(proxy_eval_config),
    )
    decode_health = _summarize_decode_health(artifacts.gt_vs_pred_jsonl)
    row: dict[str, object] = {
        "stage_name": stage_name,
        "benchmark_scope": _BENCHMARK_NAME,
        "dataset_variant": cfg.decode.dataset_variant,
        "model_alias": model_cfg.alias,
        "coord_mode": model_cfg.coord_mode,
        "run_name": run_name,
        "run_dir": str(artifacts.run_dir),
        "infer_run_spec_path": str(infer_run_spec_path),
        "confidence_config_path": str(postop_config_path),
        "proxy_eval_config_path": str(proxy_eval_config_path),
        "gt_vs_pred_jsonl": str(artifacts.gt_vs_pred_jsonl),
        "pred_token_trace_jsonl": str(artifacts.pred_token_trace_jsonl),
        "infer_summary_path": str(artifacts.summary_json),
        "scored_jsonl": str(artifacts.run_dir / "gt_vs_pred_scored.jsonl"),
        "confidence_postop_summary_path": str(
            artifacts.run_dir / "confidence_postop_summary.json"
        ),
        "proxy_eval_bundle_summary_path": str(
            artifacts.run_dir / "proxy_eval_bundle_summary.json"
        ),
        "repetition_penalty": float(repetition_penalty),
        "stop_pressure_active": bool(stop_pressure_active),
        "stop_pressure_mode": (
            cfg.decode.stop_pressure_mode if stop_pressure_active else None
        ),
        "stop_pressure_min_new_tokens": (
            int(cfg.decode.stop_pressure_min_new_tokens) if stop_pressure_active else 0
        ),
        "stop_pressure_logit_bias": (
            float(cfg.decode.stop_pressure_logit_bias) if stop_pressure_active else 0.0
        ),
        "stop_pressure_trigger_rule": (
            "raw_text_object_boundary" if stop_pressure_active else None
        ),
        "confidence_method": confidence_summary.get("confidence_method"),
    }
    row.update(decode_health)
    for view in cfg.decode.views:
        for metric_key in ("bbox_AP", "bbox_AP50", "bbox_AP75", "f1ish@0.50_full_micro"):
            row[f"{view}_{metric_key}"] = _extract_bundle_metric(
                bundle_summary,
                view=view,
                metric_key=metric_key,
            )
    return row


def _attach_decode_metric_deltas(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline_key_fn,
    baseline_predicate,
) -> list[dict[str, object]]:
    baseline_lookup: dict[tuple[str, ...], Mapping[str, object]] = {}
    for row in rows:
        if baseline_predicate(row):
            baseline_lookup[baseline_key_fn(row)] = row
    enriched: list[dict[str, object]] = []
    for row in rows:
        baseline = baseline_lookup.get(baseline_key_fn(row))
        payload = dict(row)
        if baseline is not None:
            payload["coco_real_bbox_AP_delta_from_baseline"] = (
                None
                if row.get("coco_real_bbox_AP") is None
                or baseline.get("coco_real_bbox_AP") is None
                else float(row["coco_real_bbox_AP"]) - float(baseline["coco_real_bbox_AP"])
            )
            payload["parse_valid_rate_delta_from_baseline"] = float(
                float(row["parse_valid_rate"]) - float(baseline["parse_valid_rate"])
            )
            payload["nonempty_rate_delta_from_baseline"] = float(
                float(row["nonempty_rate"]) - float(baseline["nonempty_rate"])
            )
            payload["duplicate_like_rate_delta_from_baseline"] = float(
                float(row["duplicate_like_rate"])
                - float(baseline["duplicate_like_rate"])
            )
        enriched.append(payload)
    return enriched


def _build_eos_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_alias"]), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for model_alias, model_rows in sorted(grouped.items()):
        sum_margin = sum(float(row["continue_minus_eos_sum_logprob"]) for row in model_rows)
        mean_margin_values = [
            float(row["continue_minus_eos_mean_logprob"])
            for row in model_rows
            if row.get("continue_minus_eos_mean_logprob") is not None
        ]
        summary_rows.append(
            {
                "model_alias": model_alias,
                "num_cases": len(model_rows),
                "stop_pressure_signature_count": sum(
                    1 for row in model_rows if bool(row["stop_pressure_signature"])
                ),
                "avg_continue_minus_eos_sum_logprob": float(sum_margin / len(model_rows)),
                "avg_continue_minus_eos_mean_logprob": (
                    float(sum(mean_margin_values) / len(mean_margin_values))
                    if mean_margin_values
                    else None
                ),
            }
        )
    return summary_rows


def _strip_trailing_special_terminators(text: str) -> str:
    stripped = str(text)
    changed = True
    while changed:
        changed = False
        trimmed = stripped.rstrip()
        for special_text in _SPECIAL_TERMINATOR_TEXTS:
            if trimmed.endswith(special_text):
                stripped = trimmed[: -len(special_text)]
                changed = True
                break
    return stripped


def _looks_like_array_close_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    return bool(stripped) and stripped.startswith("]") and set(stripped).issubset(
        {"]", "}"}
    )


def _looks_like_final_close_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    return bool(stripped) and stripped.startswith("}") and set(stripped).issubset(
        {"}"}
    )


def _looks_like_comma_continuation_token(text: str) -> bool:
    return _strip_trailing_special_terminators(text).lstrip().startswith(",")


def _looks_like_array_close_then_continue_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    if not stripped.startswith("]") or "," not in stripped:
        return False
    structural_part = stripped.split(",", 1)[0]
    return bool(structural_part) and set(structural_part).issubset({"]", "}"})


def _iter_token_ids(*, tokenizer: object) -> list[int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict):
            return sorted(
                {
                    int(token_id)
                    for token_id in vocab.values()
                    if isinstance(token_id, int)
                }
            )
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        return []
    try:
        return list(range(int(vocab_size)))
    except (TypeError, ValueError):
        return []


def _build_branchpoint_group_token_ids(
    *,
    tokenizer: object,
    branch_kind: str,
) -> dict[str, list[int]]:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer.decode is required for branchpoint token census")
    grouped = {
        "close_now": [],
        "next_object": [],
        "wrong_schema": [],
    }
    for token_id in _iter_token_ids(tokenizer=tokenizer):
        text = str(
            decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        if branch_kind == "array_close":
            if _looks_like_array_close_token(text):
                grouped["close_now"].append(int(token_id))
            elif _looks_like_array_close_then_continue_token(text):
                grouped["wrong_schema"].append(int(token_id))
            elif _looks_like_comma_continuation_token(text):
                grouped["next_object"].append(int(token_id))
        elif branch_kind == "final_close":
            if _looks_like_final_close_token(text):
                grouped["close_now"].append(int(token_id))
            elif _looks_like_comma_continuation_token(text):
                grouped["wrong_schema"].append(int(token_id))
        else:
            raise ValueError(f"unsupported branchpoint kind: {branch_kind}")
    return grouped


def _inspect_prepared_position(
    *,
    scorer: object,
    prepared: object,
    image: Image.Image,
    absolute_position: int,
    focus_token_ids: Mapping[str, int],
    group_token_ids: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    from src.analysis.raw_text_coordinate_decode_bias_scoring import (
        inspect_processed_position,
    )

    processor = getattr(scorer, "processor")
    model = getattr(scorer, "model")
    device = getattr(scorer, "device")
    model_inputs = processor(
        text=[getattr(prepared, "full_text")],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=False)
    logits = getattr(outputs, "logits", None)
    input_ids = model_inputs.get("input_ids")
    if not isinstance(logits, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("teacher-forced scorer did not return logits/input_ids")
    padded_len = int(input_ids.shape[1])
    full_input_ids = list(getattr(prepared, "full_input_ids"))
    pad_offset = int(padded_len - len(full_input_ids))
    observed_ids = input_ids[0, pad_offset:].detach().cpu().tolist()
    if [int(value) for value in observed_ids] != [int(value) for value in full_input_ids]:
        raise RuntimeError("assistant_span_build_failed")
    inspected = inspect_processed_position(
        logits=logits,
        input_ids=input_ids,
        batch_idx=0,
        position=pad_offset + int(absolute_position),
        tokenizer=getattr(scorer, "tokenizer", None),
        history_start=pad_offset,
        top_k=_BRANCHPOINT_TOP_K,
        focus_token_ids=focus_token_ids,
        group_token_ids=group_token_ids,
    )
    unpadded_position = int(inspected["position"]) - pad_offset
    return {
        **inspected,
        "position": unpadded_position,
        "assistant_relative_position": unpadded_position
        - int(getattr(prepared, "assistant_start")),
    }


def _resolve_final_close_branch_token_row(
    token_rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object] | None, str]:
    if not token_rows:
        return None, "missing_stop_path"
    for row in reversed(token_rows):
        if _looks_like_final_close_token(str(row.get("token_text") or "")):
            return row, "ok"
    if any(_looks_like_array_close_token(str(row.get("token_text") or "")) for row in token_rows):
        return None, "fused_with_array_close"
    return None, "missing_final_close_token"


def _build_branchpoint_summary_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_alias"]), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for model_alias, model_rows in sorted(grouped.items()):
        array_margins = [
            float(row["array_close_branch"]["stop_minus_continue_raw_logprob"])
            for row in model_rows
            if row.get("array_close_branch", {}).get("stop_minus_continue_raw_logprob")
            is not None
        ]
        final_margins = [
            float(
                row["final_close_branch"][
                    "final_close_minus_wrong_schema_top_raw_logprob"
                ]
            )
            for row in model_rows
            if row.get("final_close_branch", {}).get(
                "final_close_minus_wrong_schema_top_raw_logprob"
            )
            is not None
        ]
        summary_rows.append(
            {
                "model_alias": model_alias,
                "num_cases": len(model_rows),
                "array_branch_close_prefers_stop_count": sum(
                    1 for value in array_margins if value > 0.0
                ),
                "avg_array_branch_stop_minus_continue_raw_logprob": (
                    float(sum(array_margins) / len(array_margins))
                    if array_margins
                    else None
                ),
                "final_close_available_count": sum(
                    1
                    for row in model_rows
                    if str(row.get("final_close_branch", {}).get("status") or "")
                    == "ok"
                ),
                "final_close_fused_count": sum(
                    1
                    for row in model_rows
                    if str(row.get("final_close_branch", {}).get("status") or "")
                    == "fused_with_array_close"
                ),
                "final_close_prefers_close_over_wrong_schema_count": sum(
                    1 for value in final_margins if value > 0.0
                ),
                "avg_final_close_minus_wrong_schema_top_raw_logprob": (
                    float(sum(final_margins) / len(final_margins))
                    if final_margins
                    else None
                ),
            }
        )
    return summary_rows


def _materialize_counterfactual_eos_stage(
    *,
    cfg: StudyConfig,
    run_dir: Path,
    hydrated_rows: Sequence[Mapping[str, object]],
    source_jsonl_path: Path,
    source_rows: Sequence[Mapping[str, object]],
    scorer_cache: dict[str, object] | None = None,
) -> dict[str, object]:
    eos_dir = run_dir / "counterfactual_eos"
    model_cfg_map = _model_cfg_by_alias(cfg.models)
    case_rows: list[dict[str, object]] = []
    for hydrated_row in hydrated_rows:
        model_alias = str(hydrated_row["model_alias"])
        model_cfg = model_cfg_map[model_alias]
        scorer = _get_or_create_scorer(
            scorer_cache=scorer_cache,
            model_cfg=model_cfg,
            scoring_cfg=cfg.scoring,
        )
        source_index = int(hydrated_row["source_index"])
        source_row = source_rows[source_index]
        image = _load_source_image(
            source_row=source_row,
            source_jsonl_path=source_jsonl_path,
        )
        try:
            stop_bundle = _score_candidate_token_rows(
                scorer=scorer,
                image=image,
                baseline_assistant_text=str(hydrated_row["baseline_assistant_text"]),
                candidate_assistant_text=str(hydrated_row["stop_now_candidate_text"]),
                prompt_variant=str(hydrated_row["prompt_variant"]),
                object_field_order=str(hydrated_row["object_field_order"]),
                repetition_penalty=1.0,
            )
            continue_bundle = _score_candidate_token_rows(
                scorer=scorer,
                image=image,
                baseline_assistant_text=str(hydrated_row["baseline_assistant_text"]),
                candidate_assistant_text=str(
                    hydrated_row["continue_with_gt_candidate_text"]
                ),
                prompt_variant=str(hydrated_row["prompt_variant"]),
                object_field_order=str(hydrated_row["object_field_order"]),
                repetition_penalty=1.0,
            )
        finally:
            image.close()
        stop_summary = _aggregate_token_rows(
            stop_bundle["token_rows"],
            score_key="raw_logprob",
        )
        continue_summary = _aggregate_token_rows(
            continue_bundle["token_rows"],
            score_key="raw_logprob",
        )
        eos_branch_logprob = float(stop_bundle["token_rows"][0]["raw_logprob"])
        first_continue_branch_logprob = float(
            continue_bundle["token_rows"][0]["raw_logprob"]
        )
        continue_minus_eos_sum_logprob = float(
            float(continue_summary["sum_logprob"]) - float(stop_summary["sum_logprob"])
        )
        continue_mean = continue_summary["mean_logprob"]
        stop_mean = stop_summary["mean_logprob"]
        continue_minus_eos_mean_logprob = (
            float(float(continue_mean) - float(stop_mean))
            if continue_mean is not None and stop_mean is not None
            else None
        )
        case_rows.append(
            {
                "case_uid": str(hydrated_row["case_uid"]),
                "benchmark_name": _BENCHMARK_NAME,
                "model_alias": model_alias,
                "image_id": hydrated_row.get("image_id"),
                "coord_mode": str(hydrated_row["coord_mode"]),
                "baseline_assistant_text": str(hydrated_row["baseline_assistant_text"]),
                "stop_now_candidate_text": str(hydrated_row["stop_now_candidate_text"]),
                "continue_with_gt_candidate_text": str(
                    hydrated_row["continue_with_gt_candidate_text"]
                ),
                "eos_branch_logprob": eos_branch_logprob,
                "first_continue_branch_logprob": first_continue_branch_logprob,
                "eos_now_token_count": int(stop_summary["token_count"]),
                "continue_with_gt_token_count": int(continue_summary["token_count"]),
                "matched_token_count": int(
                    min(
                        int(stop_summary["token_count"]),
                        int(continue_summary["token_count"]),
                    )
                ),
                "eos_now_sum_logprob": float(stop_summary["sum_logprob"]),
                "eos_now_mean_logprob": stop_summary["mean_logprob"],
                "continue_with_gt_sum_logprob": float(continue_summary["sum_logprob"]),
                "continue_with_gt_mean_logprob": continue_summary["mean_logprob"],
                "continue_minus_eos_sum_logprob": continue_minus_eos_sum_logprob,
                "continue_minus_eos_mean_logprob": continue_minus_eos_mean_logprob,
                "stop_pressure_signature": bool(
                    eos_branch_logprob > first_continue_branch_logprob
                    and continue_minus_eos_sum_logprob < 0.0
                    and continue_minus_eos_mean_logprob is not None
                    and continue_minus_eos_mean_logprob >= 0.0
                ),
            }
        )
    summary_rows = _build_eos_summary_rows(case_rows)
    _write_jsonl(eos_dir / "case_rows.jsonl", case_rows)
    _write_jsonl(eos_dir / "summary_rows.jsonl", summary_rows)
    _write_json(
        eos_dir / "summary.json",
        {
            "benchmark_scope": _BENCHMARK_NAME,
            "coord_mode": _REQUIRED_COORD_MODE,
            "history_scope": cfg.study.history_scope,
            "case_row_count": len(case_rows),
            "summary_row_count": len(summary_rows),
        },
    )
    return {
        "case_row_count": len(case_rows),
        "summary_row_count": len(summary_rows),
        "case_rows_path": str(eos_dir / "case_rows.jsonl"),
        "summary_rows_path": str(eos_dir / "summary_rows.jsonl"),
        "summary_path": str(eos_dir / "summary.json"),
    }


def _materialize_counterfactual_branchpoint_census_stage(
    *,
    cfg: StudyConfig,
    run_dir: Path,
    hydrated_rows: Sequence[Mapping[str, object]],
    source_jsonl_path: Path,
    source_rows: Sequence[Mapping[str, object]],
    scorer_cache: dict[str, object] | None = None,
) -> dict[str, object]:
    branch_dir = run_dir / "counterfactual_branchpoint_census"
    model_cfg_map = _model_cfg_by_alias(cfg.models)
    group_token_cache: dict[tuple[str, str], dict[str, list[int]]] = {}
    case_rows: list[dict[str, object]] = []
    for hydrated_row in hydrated_rows:
        model_alias = str(hydrated_row["model_alias"])
        model_cfg = model_cfg_map[model_alias]
        scorer = _get_or_create_scorer(
            scorer_cache=scorer_cache,
            model_cfg=model_cfg,
            scoring_cfg=cfg.scoring,
        )
        source_index = int(hydrated_row["source_index"])
        source_row = source_rows[source_index]
        image = _load_source_image(
            source_row=source_row,
            source_jsonl_path=source_jsonl_path,
        )
        try:
            stop_bundle = _score_candidate_token_rows(
                scorer=scorer,
                image=image,
                baseline_assistant_text=str(hydrated_row["baseline_assistant_text"]),
                candidate_assistant_text=str(hydrated_row["stop_now_candidate_text"]),
                prompt_variant=str(hydrated_row["prompt_variant"]),
                object_field_order=str(hydrated_row["object_field_order"]),
                repetition_penalty=1.0,
            )
            continue_bundle = _score_candidate_token_rows(
                scorer=scorer,
                image=image,
                baseline_assistant_text=str(hydrated_row["baseline_assistant_text"]),
                candidate_assistant_text=str(
                    hydrated_row["continue_with_gt_candidate_text"]
                ),
                prompt_variant=str(hydrated_row["prompt_variant"]),
                object_field_order=str(hydrated_row["object_field_order"]),
                repetition_penalty=1.0,
            )
            array_groups = group_token_cache.setdefault(
                (model_alias, "array_close"),
                _build_branchpoint_group_token_ids(
                    tokenizer=getattr(scorer, "tokenizer"),
                    branch_kind="array_close",
                ),
            )
            array_branch = _inspect_prepared_position(
                scorer=scorer,
                prepared=stop_bundle["prepared"],
                image=image,
                absolute_position=int(stop_bundle["absolute_positions"][0]),
                focus_token_ids={
                    "stop_now_first_token": int(stop_bundle["token_rows"][0]["token_id"]),
                    "continue_first_token": int(
                        continue_bundle["token_rows"][0]["token_id"]
                    ),
                },
                group_token_ids=array_groups,
            )
            stop_focus = array_branch["focus_tokens"].get("stop_now_first_token")
            continue_focus = array_branch["focus_tokens"].get("continue_first_token")
            array_branch["stop_minus_continue_raw_logprob"] = (
                None
                if stop_focus is None or continue_focus is None
                else float(stop_focus["raw_logprob"]) - float(continue_focus["raw_logprob"])
            )

            final_row, final_status = _resolve_final_close_branch_token_row(
                stop_bundle["token_rows"]
            )
            if final_row is None:
                final_branch: dict[str, object] = {"status": final_status}
            else:
                final_groups = group_token_cache.setdefault(
                    (model_alias, "final_close"),
                    _build_branchpoint_group_token_ids(
                        tokenizer=getattr(scorer, "tokenizer"),
                        branch_kind="final_close",
                    ),
                )
                final_branch = _inspect_prepared_position(
                    scorer=scorer,
                    prepared=stop_bundle["prepared"],
                    image=image,
                    absolute_position=int(final_row["position"]),
                    focus_token_ids={
                        "final_close_token": int(final_row["token_id"]),
                    },
                    group_token_ids=final_groups,
                )
                close_focus = final_branch["focus_tokens"].get("final_close_token")
                wrong_schema_top = (
                    final_branch.get("group_summaries", {})
                    .get("wrong_schema", {})
                    .get("raw_top_token_logprob")
                )
                final_branch["status"] = "ok"
                final_branch["final_close_minus_wrong_schema_top_raw_logprob"] = (
                    None
                    if close_focus is None or wrong_schema_top is None
                    else float(close_focus["raw_logprob"]) - float(wrong_schema_top)
                )
        finally:
            image.close()
        case_rows.append(
            {
                "case_uid": str(hydrated_row["case_uid"]),
                "benchmark_name": _BENCHMARK_NAME,
                "model_alias": model_alias,
                "image_id": hydrated_row.get("image_id"),
                "source_index": int(hydrated_row["source_index"]),
                "coord_mode": str(hydrated_row["coord_mode"]),
                "branchpoint_top_k": _BRANCHPOINT_TOP_K,
                "array_close_branch": array_branch,
                "final_close_branch": final_branch,
            }
        )
    summary_rows = _build_branchpoint_summary_rows(case_rows)
    _write_jsonl(branch_dir / "case_rows.jsonl", case_rows)
    _write_jsonl(branch_dir / "summary_rows.jsonl", summary_rows)
    _write_json(
        branch_dir / "summary.json",
        {
            "benchmark_scope": _BENCHMARK_NAME,
            "coord_mode": _REQUIRED_COORD_MODE,
            "history_scope": cfg.study.history_scope,
            "branchpoint_top_k": _BRANCHPOINT_TOP_K,
            "case_row_count": len(case_rows),
            "summary_row_count": len(summary_rows),
        },
    )
    return {
        "case_row_count": len(case_rows),
        "summary_row_count": len(summary_rows),
        "case_rows_path": str(branch_dir / "case_rows.jsonl"),
        "summary_rows_path": str(branch_dir / "summary_rows.jsonl"),
        "summary_path": str(branch_dir / "summary.json"),
    }


def _build_repeat_candidate_specs(
    hydrated_row: Mapping[str, object],
) -> list[tuple[str, str]]:
    return [
        ("continue_with_gt", str(hydrated_row["continue_with_gt_candidate_text"])),
        ("exact_duplicate", str(hydrated_row["exact_duplicate_candidate_text"])),
    ]


def _attach_repeat_baseline_deltas(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline_lookup: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in rows:
        if float(row["repetition_penalty"]) == 1.0:
            baseline_lookup[
                (
                    str(row["case_uid"]),
                    str(row["model_alias"]),
                    str(row["candidate_kind"]),
                )
            ] = row
    enriched: list[dict[str, object]] = []
    for row in rows:
        key = (
            str(row["case_uid"]),
            str(row["model_alias"]),
            str(row["candidate_kind"]),
        )
        baseline = baseline_lookup[key]
        baseline_processed = float(baseline["processed_sum_logprob"])
        token_group_deltas = {
            group_name: float(group_summary["processed_sum_logprob"])
            - float(
                baseline["token_group_summaries"][group_name]["processed_sum_logprob"]
            )
            for group_name, group_summary in row["token_group_summaries"].items()
        }
        enriched.append(
            {
                **dict(row),
                "sum_logprob_delta_from_1_00": float(
                    float(row["processed_sum_logprob"]) - baseline_processed
                ),
                "token_group_deltas": token_group_deltas,
            }
        )
    return enriched


def _build_repeat_summary_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["model_alias"]),
                str(row["candidate_kind"]),
                float(row["repetition_penalty"]),
            ),
            [],
        ).append(row)
    summary_rows: list[dict[str, object]] = []
    for (model_alias, candidate_kind, repetition_penalty), group_rows in sorted(
        grouped.items()
    ):
        summary_rows.append(
            {
                "model_alias": model_alias,
                "candidate_kind": candidate_kind,
                "repetition_penalty": repetition_penalty,
                "num_rows": len(group_rows),
                "avg_processed_sum_logprob": float(
                    sum(float(row["processed_sum_logprob"]) for row in group_rows)
                    / len(group_rows)
                ),
                "avg_sum_logprob_delta_from_1_00": float(
                    sum(float(row["sum_logprob_delta_from_1_00"]) for row in group_rows)
                    / len(group_rows)
                ),
            }
        )
    return summary_rows


def _materialize_counterfactual_repeat_penalty_stage(
    *,
    cfg: StudyConfig,
    run_dir: Path,
    hydrated_rows: Sequence[Mapping[str, object]],
    source_jsonl_path: Path,
    source_rows: Sequence[Mapping[str, object]],
    scorer_cache: dict[str, object] | None = None,
) -> dict[str, object]:
    from src.analysis.raw_text_coordinate_decode_bias_scoring import (
        group_raw_text_token_rows,
    )

    repeat_dir = run_dir / "counterfactual_repeat_penalty"
    model_cfg_map = _model_cfg_by_alias(cfg.models)
    sweep_rows: list[dict[str, object]] = []
    for hydrated_row in hydrated_rows:
        model_alias = str(hydrated_row["model_alias"])
        model_cfg = model_cfg_map[model_alias]
        scorer = _get_or_create_scorer(
            scorer_cache=scorer_cache,
            model_cfg=model_cfg,
            scoring_cfg=cfg.scoring,
        )
        source_index = int(hydrated_row["source_index"])
        source_row = source_rows[source_index]
        image = _load_source_image(
            source_row=source_row,
            source_jsonl_path=source_jsonl_path,
        )
        try:
            for candidate_kind, candidate_text in _build_repeat_candidate_specs(hydrated_row):
                for penalty in cfg.scoring.repetition_penalties:
                    bundle = _score_candidate_token_rows(
                        scorer=scorer,
                        image=image,
                        baseline_assistant_text=str(
                            hydrated_row["baseline_assistant_text"]
                        ),
                        candidate_assistant_text=candidate_text,
                        prompt_variant=str(hydrated_row["prompt_variant"]),
                        object_field_order=str(hydrated_row["object_field_order"]),
                        repetition_penalty=float(penalty),
                    )
                    raw_summary = _aggregate_token_rows(
                        bundle["token_rows"],
                        score_key="raw_logprob",
                    )
                    processed_summary = _aggregate_token_rows(
                        bundle["token_rows"],
                        score_key="processed_logprob",
                    )
                    token_group_summaries = _summarize_group_token_rows(
                        group_raw_text_token_rows(
                            tokenizer=getattr(scorer, "tokenizer"),
                            candidate_assistant_text=candidate_text,
                            token_rows=bundle["token_rows"],
                        )
                    )
                    sweep_rows.append(
                        {
                            "case_uid": str(hydrated_row["case_uid"]),
                            "benchmark_name": _BENCHMARK_NAME,
                            "model_alias": model_alias,
                            "image_id": hydrated_row.get("image_id"),
                            "coord_mode": str(hydrated_row["coord_mode"]),
                            "candidate_kind": candidate_kind,
                            "candidate_assistant_text": candidate_text,
                            "repetition_penalty": float(penalty),
                            "token_count": int(raw_summary["token_count"]),
                            "raw_sum_logprob": float(raw_summary["sum_logprob"]),
                            "raw_mean_logprob": raw_summary["mean_logprob"],
                            "processed_sum_logprob": float(
                                processed_summary["sum_logprob"]
                            ),
                            "processed_mean_logprob": processed_summary["mean_logprob"],
                            "token_group_summaries": token_group_summaries,
                        }
                    )
        finally:
            image.close()
    enriched_rows = _attach_repeat_baseline_deltas(sweep_rows)
    summary_rows = _build_repeat_summary_rows(enriched_rows)
    _write_jsonl(repeat_dir / "sweep_rows.jsonl", enriched_rows)
    _write_jsonl(repeat_dir / "summary_rows.jsonl", summary_rows)
    _write_json(
        repeat_dir / "summary.json",
        {
            "benchmark_scope": _BENCHMARK_NAME,
            "coord_mode": _REQUIRED_COORD_MODE,
            "history_scope": cfg.study.history_scope,
            "repetition_penalties": list(cfg.scoring.repetition_penalties),
            "row_count": len(enriched_rows),
            "summary_row_count": len(summary_rows),
        },
    )
    return {
        "row_count": len(enriched_rows),
        "summary_row_count": len(summary_rows),
        "sweep_rows_path": str(repeat_dir / "sweep_rows.jsonl"),
        "summary_rows_path": str(repeat_dir / "summary_rows.jsonl"),
        "summary_path": str(repeat_dir / "summary.json"),
    }


def _format_penalty_slug(value: float) -> str:
    return str(f"{float(value):.2f}").replace(".", "_")


def _materialize_decode_val200_repeat_penalty_stage(
    *,
    cfg: StudyConfig,
    repo_root: Path,
    run_dir: Path,
    subset_artifacts: Mapping[str, object],
) -> dict[str, object]:
    stage_dir = run_dir / "decode_val200_repeat_penalty"
    rows: list[dict[str, object]] = []
    for model_cfg in _iter_model_cfgs(cfg.models):
        for penalty in cfg.scoring.repetition_penalties:
            rows.append(
                _run_decode_eval_workflow(
                    cfg=cfg,
                    repo_root=repo_root,
                    stage_dir=stage_dir,
                    stage_name="decode_val200_repeat_penalty",
                    model_cfg=model_cfg,
                    subset_artifacts=subset_artifacts,
                    run_name=f"rp_{_format_penalty_slug(penalty)}",
                    repetition_penalty=float(penalty),
                    stop_pressure_active=False,
                )
            )
    summary_rows = _attach_decode_metric_deltas(
        rows,
        baseline_key_fn=lambda row: (str(row["model_alias"]),),
        baseline_predicate=lambda row: float(row["repetition_penalty"]) == 1.0,
    )
    _write_jsonl(stage_dir / "summary_rows.jsonl", summary_rows)
    _write_json(
        stage_dir / "summary.json",
        {
            "benchmark_scope": _BENCHMARK_NAME,
            "dataset_variant": cfg.decode.dataset_variant,
            "coord_mode": _REQUIRED_COORD_MODE,
            "row_count": len(summary_rows),
            "repetition_penalties": list(cfg.scoring.repetition_penalties),
            "subset_manifest_path": str(subset_artifacts["subset_manifest_path"]),
        },
    )
    return {
        "row_count": len(summary_rows),
        "summary_row_count": len(summary_rows),
        "summary_rows_path": str(stage_dir / "summary_rows.jsonl"),
        "summary_path": str(stage_dir / "summary.json"),
        "dataset_variant": cfg.decode.dataset_variant,
    }


def _materialize_decode_val200_stop_pressure_stage(
    *,
    cfg: StudyConfig,
    repo_root: Path,
    run_dir: Path,
    subset_artifacts: Mapping[str, object],
) -> dict[str, object]:
    stage_dir = run_dir / "decode_val200_stop_pressure"
    rows: list[dict[str, object]] = []
    for model_cfg in _iter_model_cfgs(cfg.models):
        for stop_pressure_active in (False, True):
            rows.append(
                _run_decode_eval_workflow(
                    cfg=cfg,
                    repo_root=repo_root,
                    stage_dir=stage_dir,
                    stage_name="decode_val200_stop_pressure",
                    model_cfg=model_cfg,
                    subset_artifacts=subset_artifacts,
                    run_name=(
                        "stop_pressure_on" if stop_pressure_active else "stop_pressure_off"
                    ),
                    repetition_penalty=1.05,
                    stop_pressure_active=stop_pressure_active,
                )
            )
    summary_rows = _attach_decode_metric_deltas(
        rows,
        baseline_key_fn=lambda row: (str(row["model_alias"]),),
        baseline_predicate=lambda row: not bool(row["stop_pressure_active"]),
    )
    _write_jsonl(stage_dir / "summary_rows.jsonl", summary_rows)
    _write_json(
        stage_dir / "summary.json",
        {
            "benchmark_scope": _BENCHMARK_NAME,
            "dataset_variant": cfg.decode.dataset_variant,
            "coord_mode": _REQUIRED_COORD_MODE,
            "row_count": len(summary_rows),
            "stop_pressure_min_new_tokens": cfg.decode.stop_pressure_min_new_tokens,
            "subset_manifest_path": str(subset_artifacts["subset_manifest_path"]),
        },
    )
    return {
        "row_count": len(summary_rows),
        "summary_row_count": len(summary_rows),
        "summary_rows_path": str(stage_dir / "summary_rows.jsonl"),
        "summary_path": str(stage_dir / "summary.json"),
        "dataset_variant": cfg.decode.dataset_variant,
    }


def _materialize_report_stage(
    *,
    cfg: StudyConfig,
    run_dir: Path,
    stage_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    report_dir = run_dir / "report"
    summary = {
        "benchmark_scope": _BENCHMARK_NAME,
        "coord_mode": _REQUIRED_COORD_MODE,
        "history_scope": cfg.study.history_scope,
        "requested_stages": list(cfg.run.stages),
        "lanes": [
            stage_name
            for stage_name in (
                "counterfactual_eos",
                "counterfactual_branchpoint_census",
                "counterfactual_repeat_penalty",
                "decode_val200_repeat_penalty",
                "decode_val200_stop_pressure",
            )
            if stage_name in stage_results
        ],
        "repetition_penalties": list(cfg.scoring.repetition_penalties),
        "decode_dataset_variant": cfg.decode.dataset_variant,
        "stages": {key: dict(value) for key, value in stage_results.items()},
    }
    summary_rows: list[dict[str, object]] = [
        {
            "section": stage_name,
            "artifact_path": result.get(
                "summary_rows_path",
                result.get("case_rows_path", result.get("sweep_rows_path")),
            ),
            "row_count": int(
                result.get(
                    "summary_row_count",
                    result.get("case_row_count", result.get("row_count", 0)),
                )
            ),
        }
        for stage_name, result in stage_results.items()
    ]
    _write_json(report_dir / "summary.json", summary)
    _write_jsonl(report_dir / "summary_rows.jsonl", summary_rows)
    return {
        "summary_path": str(report_dir / "summary.json"),
        "summary_rows_path": str(report_dir / "summary_rows.jsonl"),
        "summary_row_count": len(summary_rows),
    }


def run_study(config_path: Path) -> dict[str, object]:
    cfg = load_study_config(config_path)
    repo_root = _shared_repo_root(config_path)
    output_root = _resolve_repo_path(cfg.run.output_dir, anchor=repo_root)
    run_dir = output_root / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_dir / "resolved_config.json"
    _write_json(resolved_config_path, _resolved_config_payload(cfg))

    source_jsonl_path = _resolve_repo_path(cfg.study.val200_source_jsonl, anchor=repo_root)
    source_rows = _read_jsonl(source_jsonl_path)
    case_rows = _build_case_rows(
        cfg=cfg,
        source_rows=source_rows,
        source_jsonl_path=source_jsonl_path,
    )
    hydrated_rows = hydrate_case_rows(case_rows=case_rows)
    hydrated_cases_path = run_dir / "counterfactual_inputs" / "hydrated_cases.jsonl"
    _write_jsonl(hydrated_cases_path, hydrated_rows)

    materialized_stages: list[str] = ["hydrate"]
    stage_results: dict[str, dict[str, object]] = {}
    scorer_cache: dict[str, object] = {}
    decode_subset_artifacts: dict[str, object] | None = None
    if any(
        stage in cfg.run.stages
        for stage in ("decode_val200_repeat_penalty", "decode_val200_stop_pressure")
    ):
        decode_subset_artifacts = _materialize_decode_val200_subset(
            cfg=cfg,
            run_dir=run_dir,
            repo_root=repo_root,
        )
    if "counterfactual_eos" in cfg.run.stages:
        stage_results["counterfactual_eos"] = _materialize_counterfactual_eos_stage(
            cfg=cfg,
            run_dir=run_dir,
            hydrated_rows=hydrated_rows,
            source_jsonl_path=source_jsonl_path,
            source_rows=source_rows,
            scorer_cache=scorer_cache,
        )
        materialized_stages.append("counterfactual_eos")
    if "counterfactual_branchpoint_census" in cfg.run.stages:
        stage_results[
            "counterfactual_branchpoint_census"
        ] = _materialize_counterfactual_branchpoint_census_stage(
            cfg=cfg,
            run_dir=run_dir,
            hydrated_rows=hydrated_rows,
            source_jsonl_path=source_jsonl_path,
            source_rows=source_rows,
            scorer_cache=scorer_cache,
        )
        materialized_stages.append("counterfactual_branchpoint_census")
    if "counterfactual_repeat_penalty" in cfg.run.stages:
        stage_results[
            "counterfactual_repeat_penalty"
        ] = _materialize_counterfactual_repeat_penalty_stage(
            cfg=cfg,
            run_dir=run_dir,
            hydrated_rows=hydrated_rows,
            source_jsonl_path=source_jsonl_path,
            source_rows=source_rows,
            scorer_cache=scorer_cache,
        )
        materialized_stages.append("counterfactual_repeat_penalty")
    if "decode_val200_repeat_penalty" in cfg.run.stages:
        if decode_subset_artifacts is None:
            raise RuntimeError("decode subset artifacts must exist before decode stages")
        stage_results[
            "decode_val200_repeat_penalty"
        ] = _materialize_decode_val200_repeat_penalty_stage(
            cfg=cfg,
            repo_root=repo_root,
            run_dir=run_dir,
            subset_artifacts=decode_subset_artifacts,
        )
        materialized_stages.append("decode_val200_repeat_penalty")
    if "decode_val200_stop_pressure" in cfg.run.stages:
        if decode_subset_artifacts is None:
            raise RuntimeError("decode subset artifacts must exist before decode stages")
        stage_results[
            "decode_val200_stop_pressure"
        ] = _materialize_decode_val200_stop_pressure_stage(
            cfg=cfg,
            repo_root=repo_root,
            run_dir=run_dir,
            subset_artifacts=decode_subset_artifacts,
        )
        materialized_stages.append("decode_val200_stop_pressure")
    if "report" in cfg.run.stages:
        stage_results["report"] = _materialize_report_stage(
            cfg=cfg,
            run_dir=run_dir,
            stage_results=stage_results,
        )
        materialized_stages.append("report")

    stage_manifest = {
        "run_name": cfg.run.name,
        "requested_stages": list(cfg.run.stages),
        "materialized_stages": materialized_stages,
        "models": [
            {
                "alias": model_cfg.alias,
                "base_path": model_cfg.base_path,
                "adapter_path": model_cfg.adapter_path,
                "coord_mode": model_cfg.coord_mode,
            }
            for model_cfg in _iter_model_cfgs(cfg.models)
        ],
        "benchmark": {
            "name": _BENCHMARK_NAME,
            "source_jsonl": str(source_jsonl_path),
            "source_indices": list(cfg.study.val200_source_indices),
        },
        "counterfactual": {
            "history_scope": cfg.study.history_scope,
            "coord_mode": _REQUIRED_COORD_MODE,
            "repetition_penalties": list(cfg.scoring.repetition_penalties),
        },
        "decode": {
            "dataset_variant": cfg.decode.dataset_variant,
            "pipeline_config": str(
                _resolve_repo_path(cfg.decode.pipeline_config, anchor=repo_root)
            ),
            "source_jsonl": str(
                _resolve_repo_path(cfg.decode.val200_source_jsonl, anchor=repo_root)
            ),
            "device": cfg.decode.device,
            "semantic_device": cfg.decode.semantic_device,
            "stop_pressure_min_new_tokens": cfg.decode.stop_pressure_min_new_tokens,
            "views": list(cfg.decode.views),
            "subset_manifest_path": (
                None
                if decode_subset_artifacts is None
                else str(decode_subset_artifacts["subset_manifest_path"])
            ),
            "sampled_norm_jsonl": (
                None
                if decode_subset_artifacts is None
                else str(decode_subset_artifacts["sampled_norm_jsonl"])
            ),
            "sampled_text_pixel_jsonl": (
                None
                if decode_subset_artifacts is None
                else str(decode_subset_artifacts["sampled_text_pixel_jsonl"])
            ),
        },
        "scoring": {
            "device": cfg.scoring.device,
            "attn_implementation": cfg.scoring.attn_implementation,
        },
        "hydration": {
            "version": _HYDRATION_VERSION,
            "row_count": len(hydrated_rows),
            "artifact_path": str(hydrated_cases_path),
        },
        "stages": stage_results,
    }
    _write_json(run_dir / "stage_manifest.json", stage_manifest)
    result: dict[str, object] = {
        "run_dir": str(run_dir),
        "stage_manifest_path": str(run_dir / "stage_manifest.json"),
        "hydrated_cases_path": str(hydrated_cases_path),
        "resolved_config_path": str(resolved_config_path),
    }
    result.update({f"{name}_result": payload for name, payload in stage_results.items()})
    return result
