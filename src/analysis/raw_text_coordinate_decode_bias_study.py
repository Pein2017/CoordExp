from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import yaml

from src.common.object_field_order import build_object_payload
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
    base_plus_adapter: ModelConfig


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    study: StudyControls
    models: StudyModels


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("study config must be a mapping")
    return payload


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
            rows.append(json.loads(stripped))
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


def _parse_model_config(*, model_key: str, raw_model: Mapping[str, object]) -> ModelConfig:
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


def _validate_model_surface(models: StudyModels) -> None:
    if models.base_only.alias != "base_only":
        raise ValueError("models.base_only.alias must be 'base_only'")
    if models.base_plus_adapter.alias != "base_plus_adapter":
        raise ValueError("models.base_plus_adapter.alias must be 'base_plus_adapter'")
    if models.base_only.base_path != _APPROVED_BASE_MODEL_PATH:
        raise ValueError("models.base_only.base_path must use the approved raw-text base")
    if models.base_only.adapter_path is not None:
        raise ValueError("models.base_only.adapter_path must be null")
    if models.base_plus_adapter.base_path != _APPROVED_BASE_MODEL_PATH:
        raise ValueError(
            "models.base_plus_adapter.base_path must use the approved raw-text base"
        )
    if models.base_plus_adapter.adapter_path != _APPROVED_ADAPTER_PATH:
        raise ValueError(
            "models.base_plus_adapter.adapter_path must use the approved raw-text adapter"
        )
    for model_cfg in (models.base_only, models.base_plus_adapter):
        if model_cfg.coord_mode != _REQUIRED_COORD_MODE:
            raise ValueError(
                f"{model_cfg.alias} must use coord_mode={_REQUIRED_COORD_MODE!r}"
            )


def load_study_config(config_path: Path) -> StudyConfig:
    raw = _load_yaml(config_path)
    run_raw = raw["run"]
    study_raw = raw["study"]
    models_raw = raw["models"]
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
        models=StudyModels(
            base_only=_parse_model_config(
                model_key="base_only",
                raw_model=models_raw["base_only"],
            ),
            base_plus_adapter=_parse_model_config(
                model_key="base_plus_adapter",
                raw_model=models_raw["base_plus_adapter"],
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
        hydrated_row = {
            key: value
            for key, value in row.items()
            if key != "source_row"
        }
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
    model_cfgs = (cfg.models.base_only, cfg.models.base_plus_adapter)
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


def run_study(config_path: Path) -> dict[str, object]:
    cfg = load_study_config(config_path)
    repo_root = _shared_repo_root(config_path)
    output_root = _resolve_repo_path(cfg.run.output_dir, anchor=repo_root)
    run_dir = output_root / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

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

    stage_manifest = {
        "run_name": cfg.run.name,
        "requested_stages": list(cfg.run.stages),
        "materialized_stages": ["hydrate"],
        "models": [
            {
                "alias": cfg.models.base_only.alias,
                "base_path": cfg.models.base_only.base_path,
                "adapter_path": cfg.models.base_only.adapter_path,
                "coord_mode": cfg.models.base_only.coord_mode,
            },
            {
                "alias": cfg.models.base_plus_adapter.alias,
                "base_path": cfg.models.base_plus_adapter.base_path,
                "adapter_path": cfg.models.base_plus_adapter.adapter_path,
                "coord_mode": cfg.models.base_plus_adapter.coord_mode,
            },
        ],
        "benchmark": {
            "name": _BENCHMARK_NAME,
            "source_jsonl": str(source_jsonl_path),
            "source_indices": list(cfg.study.val200_source_indices),
        },
        "counterfactual": {
            "history_scope": cfg.study.history_scope,
            "coord_mode": _REQUIRED_COORD_MODE,
        },
        "hydration": {
            "version": _HYDRATION_VERSION,
            "row_count": len(hydrated_rows),
            "artifact_path": str(hydrated_cases_path),
        },
    }
    _write_json(run_dir / "stage_manifest.json", stage_manifest)
    return {
        "run_dir": str(run_dir),
        "stage_manifest_path": str(run_dir / "stage_manifest.json"),
        "hydrated_cases_path": str(hydrated_cases_path),
    }
