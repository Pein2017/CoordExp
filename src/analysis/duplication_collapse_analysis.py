from __future__ import annotations

import gc
import inspect
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image
from transformers import (
    AutoProcessor,
    LogitsProcessorList,
    Qwen3VLForConditionalGeneration,
    RepetitionPenaltyLogitsProcessor,
)

from src.analysis.rollout_fn_factor_study import (
    HFStudyRunner,
    ResolvedCheckpoint,
    _close_prefix_rollout_text,
)
from src.analysis.unmatched_proposal_verifier import (
    TeacherForcedScorer,
    _find_subsequence,
    resolve_checkpoint_path,
    resolve_prompt_controls_for_checkpoint,
)
from src.common.geometry.bbox_parameterization import (
    DEFAULT_BBOX_FORMAT,
    AllowedBBoxFormat,
    normalize_bbox_format,
)
from src.common.object_field_order import (
    build_object_payload,
    normalize_object_field_order,
)
from src.common.paths import resolve_image_path_strict
from src.common.semantic_desc import normalize_desc
from src.config.prompts import resolve_dense_prompt_variant_key
from src.coord_tokens.codec import get_coord_token_ids, int_to_token
from src.eval.artifacts import (
    CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_SOURCE,
    CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_VERSION,
    CXCY_LOGW_LOGH_CONSTANT_SCORE,
    with_constant_scores,
    write_jsonl_records,
)
from src.eval.confidence_postop import ConfidencePostOpOptions, ConfidencePostOpPaths, run_confidence_postop
from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine
from src.utils.assistant_json import dumps_coordjson

REPO_ROOT = Path(__file__).resolve().parents[2]

_AUTHORITATIVE_TEMPERATURE = 0.0
_AUTHORITATIVE_TOP_P = 0.9
_AUTHORITATIVE_REPETITION_PENALTY = 1.05
_AUTHORITATIVE_MAX_NEW_TOKENS = 3084
_AUTHORITATIVE_SEED = 42

_DEFAULT_STAGES = (
    "inventory",
    "select_cases",
    "reproduce",
    "probe",
    "compare",
    "report",
)
_VALID_STAGES = set(_DEFAULT_STAGES)
_COORD_NEIGHBOR_RADIUS = 2
_COORD_TOKEN_COUNT = 1000
_EXPECTED_ARTIFACT_CELLS = (
    "gt_vs_pred_jsonl",
    "pred_token_trace_jsonl",
    "pred_confidence_jsonl",
    "resolved_config_json",
    "eval_metrics_json",
    "eval_coco_real_metrics_json",
    "eval_coco_real_strict_metrics_json",
    "eval_coco_real_strict_plausible_metrics_json",
)


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: Tuple[str, ...]
    checkpoint_name_filter: str = "merged"


@dataclass(frozen=True)
class WorkspaceConfig:
    root_dir: Optional[str] = None


@dataclass(frozen=True)
class CheckpointSpec:
    alias: str
    path: str
    historical_artifact_dirs: Tuple[str, ...] = ()
    prompt_variant: Optional[str] = None
    bbox_format: AllowedBBoxFormat = DEFAULT_BBOX_FORMAT
    object_field_order: Optional[str] = None
    stage: str = "unknown"
    objective_family: str = "unknown"
    geometry_regime: str = "unknown"
    rollout_training_regime: str = "unknown"
    interpretation_confounds: Tuple[str, ...] = ()
    coord_soft_ce_w1_state: str = "unknown"
    parent_checkpoint: Optional[str] = None
    family_comparison_role: str = "unknown"


@dataclass(frozen=True)
class SubsetConfig:
    max_cases_per_checkpoint: int = 4
    max_cases_total: int = 8
    min_pred_objects: int = 6
    min_duplicate_pairs: int = 2
    duplicate_iou_threshold: float = 0.7
    local_center_radius_scale: float = 0.8
    size_ratio_min: float = 0.75
    pinned_line_indices: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    bootstrap_case_manifest_jsonl: str | None = None
    replay_case_aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionConfig:
    device: str = "cuda:0"
    cuda_visible_devices: Optional[str] = None
    reproduce_batch_size: int = 1
    reproduce_attn_implementation: str = "auto"
    probe_attn_implementation: str = "eager"
    teacher_forced_attn_implementation: str = "eager"


@dataclass(frozen=True)
class DecodeConfig:
    temperature: float = _AUTHORITATIVE_TEMPERATURE
    top_p: float = _AUTHORITATIVE_TOP_P
    repetition_penalty: float = _AUTHORITATIVE_REPETITION_PENALTY
    max_new_tokens: int = _AUTHORITATIVE_MAX_NEW_TOKENS
    seed: int = _AUTHORITATIVE_SEED
    secondary_top_k: Tuple[int, ...] = ()
    secondary_top_p: Tuple[float, ...] = ()


@dataclass(frozen=True)
class ProbeConfig:
    max_cases: int = 4
    step_window_before: int = 1
    step_window_after: int = 1
    token_top_k: int = 10
    capture_native_vision: bool = True
    enable_interventions: bool = True
    intervention_max_cases: int = 1
    late_layer_count: int = 2
    visual_bias_scale: float = 1.15
    history_attenuation_scale: float = 0.5
    intervention_coord_only: bool = True
    intervention_greedy_steps: int = 8


@dataclass(frozen=True)
class ControlConfig:
    max_cases: int = 4
    include_close_candidate: bool = True
    coord_neighbor_radius: int = _COORD_NEIGHBOR_RADIUS
    same_desc_iou_threshold: float = 0.5


@dataclass(frozen=True)
class ReportConfig:
    write_markdown: bool = True


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    workspace: WorkspaceConfig
    checkpoints: Tuple[CheckpointSpec, ...]
    subset: SubsetConfig
    execution: ExecutionConfig
    decode: DecodeConfig
    probe: ProbeConfig
    controls: ControlConfig
    report: ReportConfig


@dataclass(frozen=True)
class HistoricalArtifactBundle:
    root: Path
    paths: Dict[str, Optional[Path]]


@dataclass(frozen=True)
class ResolvedStudyCheckpoint:
    spec: CheckpointSpec
    resolved: ResolvedCheckpoint
    bundles: Tuple[HistoricalArtifactBundle, ...]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Study config must be a mapping: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_path(path_value: str | None) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        raise ValueError("Expected a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _ensure_tuple_str(value: Any, *, default: Sequence[str], path: str) -> Tuple[str, ...]:
    if value is None:
        return tuple(str(v) for v in default)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list of strings")
    out = tuple(str(item).strip() for item in value if str(item).strip())
    if not out:
        raise ValueError(f"{path} must contain at least one non-empty entry")
    return out


def _ensure_tuple_int(value: Any, *, default: Sequence[int], path: str) -> Tuple[int, ...]:
    if value is None:
        return tuple(int(v) for v in default)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list of integers")
    try:
        out = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain integers") from exc
    return out


def _ensure_tuple_float(value: Any, *, default: Sequence[float], path: str) -> Tuple[float, ...]:
    if value is None:
        return tuple(float(v) for v in default)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list of floats")
    try:
        out = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain floats") from exc
    return out


def _parse_pinned_line_indices(raw: Any) -> Dict[str, Tuple[int, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("subset.pinned_line_indices must be a mapping alias -> [line_idx]")
    out: Dict[str, Tuple[int, ...]] = {}
    for key, value in raw.items():
        alias = str(key).strip()
        if not alias:
            raise ValueError("subset.pinned_line_indices keys must be non-empty")
        out[alias] = _ensure_tuple_int(
            value,
            default=(),
            path=f"subset.pinned_line_indices.{alias}",
        )
    return out


def _validate_decode_contract(decode: DecodeConfig) -> None:
    exact_checks = {
        "temperature": (decode.temperature, _AUTHORITATIVE_TEMPERATURE),
        "top_p": (decode.top_p, _AUTHORITATIVE_TOP_P),
        "repetition_penalty": (
            decode.repetition_penalty,
            _AUTHORITATIVE_REPETITION_PENALTY,
        ),
        "max_new_tokens": (decode.max_new_tokens, _AUTHORITATIVE_MAX_NEW_TOKENS),
        "seed": (decode.seed, _AUTHORITATIVE_SEED),
    }
    for key, (observed, expected) in exact_checks.items():
        if observed != expected:
            raise ValueError(
                f"decode.{key} must match the authoritative baseline {expected!r}; got {observed!r}"
            )


def _is_executable_checkpoint_surface(path: Path) -> bool:
    if not path.is_dir():
        return False
    required = (
        path / "config.json",
        path / "tokenizer_config.json",
        path / "coord_tokens.json",
    )
    if any(not item.is_file() for item in required):
        return False
    if (path / "model.safetensors.index.json").is_file():
        return True
    if any(path.glob("model*.safetensors")):
        return True
    return False


def _validate_checkpoint_surface(path: Path, *, alias: str, name_filter: str) -> None:
    name_filter = str(name_filter or "merged").strip()
    if name_filter and name_filter not in path.name:
        raise ValueError(
            f"Checkpoint {alias} must resolve to a {name_filter!r}-named checkpoint surface; got {path}"
        )
    if not _is_executable_checkpoint_surface(path):
        raise ValueError(
            "Duplication-collapse analysis requires an executable merged checkpoint surface "
            f"with model/tokenizer assets, not a training-only or metadata-only root: {path}"
        )


def _best_probe_surface(checkpoint: ResolvedStudyCheckpoint) -> str:
    if checkpoint.bundles:
        return str(checkpoint.bundles[0].root)
    return str(checkpoint.resolved.path)


def _has_infer_artifact(checkpoint: ResolvedStudyCheckpoint) -> bool:
    return bool(
        any(
            bundle.paths.get("gt_vs_pred_jsonl") is not None
            for bundle in checkpoint.bundles
        )
    )


def _probe_readiness(checkpoint: ResolvedStudyCheckpoint) -> str:
    for bundle in checkpoint.bundles:
        if (
            bundle.paths.get("gt_vs_pred_jsonl") is not None
            and bundle.paths.get("pred_token_trace_jsonl") is not None
        ):
            return "ready_to_probe"
    return "fresh_inference_needed"


def _family_comparison_summary(
    inventory_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_roles = (
        "pure_ce_reference",
        "ce_proxy_disabled_continuation",
        "soft_coordinate_supervised",
    )
    role_to_aliases: Dict[str, List[str]] = defaultdict(list)
    for row in inventory_rows:
        role = str(row.get("family_comparison_role") or "unknown")
        role_to_aliases[role].append(str(row.get("alias") or "unknown"))
    return {
        "roles": {key: sorted(value) for key, value in sorted(role_to_aliases.items())},
        "missing_expected_roles": [
            role for role in expected_roles if not role_to_aliases.get(role)
        ],
        "ce_reference_mode": (
            "clean_pure_ce"
            if role_to_aliases.get("pure_ce_reference")
            else (
                "proxy_continuation"
                if role_to_aliases.get("ce_proxy_disabled_continuation")
                else "missing_ce_reference"
            )
        ),
    }


def load_study_config(path: Path) -> StudyConfig:
    raw = _load_yaml(path)
    run_raw = raw.get("run") or {}
    workspace_raw = raw.get("workspace") or {}
    subset_raw = raw.get("subset") or {}
    execution_raw = raw.get("execution") or {}
    decode_raw = raw.get("decode") or {}
    probe_raw = raw.get("probe") or {}
    controls_raw = raw.get("controls") or {}
    report_raw = raw.get("report") or {}
    checkpoints_raw = raw.get("checkpoints") or []

    if not isinstance(checkpoints_raw, list) or not checkpoints_raw:
        raise ValueError("checkpoints must define at least one checkpoint entry")

    stages = _ensure_tuple_str(run_raw.get("stages"), default=_DEFAULT_STAGES, path="run.stages")
    unknown = sorted(set(stages) - _VALID_STAGES)
    if unknown:
        raise ValueError(f"run.stages contains unsupported values: {unknown}")

    checkpoints: List[CheckpointSpec] = []
    for idx, item in enumerate(checkpoints_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"checkpoints[{idx}] must be a mapping")
        artifact_dirs = _ensure_tuple_str(
            item.get("historical_artifact_dirs"),
            default=(),
            path=f"checkpoints[{idx}].historical_artifact_dirs",
        ) if item.get("historical_artifact_dirs") is not None else ()
        checkpoints.append(
            CheckpointSpec(
                alias=str(item.get("alias") or "").strip(),
                path=str(item.get("path") or "").strip(),
                historical_artifact_dirs=artifact_dirs,
                prompt_variant=(
                    str(item.get("prompt_variant")).strip()
                    if item.get("prompt_variant") is not None
                    else None
                ),
                bbox_format=normalize_bbox_format(
                    item.get("bbox_format", DEFAULT_BBOX_FORMAT),
                    path=f"checkpoints[{idx}].bbox_format",
                ),
                object_field_order=(
                    normalize_object_field_order(
                        str(item.get("object_field_order")),
                        path=f"checkpoints[{idx}].object_field_order",
                    )
                    if item.get("object_field_order") is not None
                    else None
                ),
                stage=str(item.get("stage") or "unknown").strip(),
                objective_family=str(item.get("objective_family") or "unknown").strip(),
                geometry_regime=str(item.get("geometry_regime") or "unknown").strip(),
                rollout_training_regime=str(
                    item.get("rollout_training_regime") or "unknown"
                ).strip(),
                interpretation_confounds=_ensure_tuple_str(
                    item.get("interpretation_confounds"),
                    default=(),
                    path=f"checkpoints[{idx}].interpretation_confounds",
                ) if item.get("interpretation_confounds") is not None else (),
                coord_soft_ce_w1_state=str(
                    item.get("coord_soft_ce_w1_state") or "unknown"
                ).strip(),
                parent_checkpoint=(
                    str(item.get("parent_checkpoint")).strip()
                    if item.get("parent_checkpoint") is not None
                    else None
                ),
                family_comparison_role=str(
                    item.get("family_comparison_role") or "unknown"
                ).strip(),
            )
        )

    for idx, item in enumerate(checkpoints):
        if not item.alias:
            raise ValueError(f"checkpoints[{idx}].alias is required")
        if not item.path:
            raise ValueError(f"checkpoints[{idx}].path is required")

    cfg = StudyConfig(
        run=RunConfig(
            name=str(run_raw.get("name") or "duplication-collapse-analysis").strip(),
            output_dir=str(run_raw.get("output_dir") or "output/analysis").strip(),
            stages=stages,
            checkpoint_name_filter=str(
                run_raw.get("checkpoint_name_filter") or "merged"
            ).strip(),
        ),
        workspace=WorkspaceConfig(
            root_dir=(
                str(workspace_raw.get("root_dir")).strip()
                if workspace_raw.get("root_dir") is not None
                else None
            )
        ),
        checkpoints=tuple(checkpoints),
        subset=SubsetConfig(
            max_cases_per_checkpoint=int(subset_raw.get("max_cases_per_checkpoint", 4)),
            max_cases_total=int(subset_raw.get("max_cases_total", 8)),
            min_pred_objects=int(subset_raw.get("min_pred_objects", 6)),
            min_duplicate_pairs=int(subset_raw.get("min_duplicate_pairs", 2)),
            duplicate_iou_threshold=float(subset_raw.get("duplicate_iou_threshold", 0.7)),
            local_center_radius_scale=float(
                subset_raw.get("local_center_radius_scale", 0.8)
            ),
            size_ratio_min=float(subset_raw.get("size_ratio_min", 0.75)),
            pinned_line_indices=_parse_pinned_line_indices(
                subset_raw.get("pinned_line_indices")
            ),
            bootstrap_case_manifest_jsonl=(
                str(subset_raw.get("bootstrap_case_manifest_jsonl")).strip()
                if subset_raw.get("bootstrap_case_manifest_jsonl") is not None
                else None
            ),
            replay_case_aliases=_ensure_tuple_str(
                subset_raw.get("replay_case_aliases"),
                default=(),
                path="subset.replay_case_aliases",
            ) if subset_raw.get("replay_case_aliases") is not None else (),
        ),
        execution=ExecutionConfig(
            device=str(execution_raw.get("device") or "cuda:0").strip(),
            cuda_visible_devices=(
                str(execution_raw.get("cuda_visible_devices")).strip()
                if execution_raw.get("cuda_visible_devices") is not None
                else None
            ),
            reproduce_batch_size=int(execution_raw.get("reproduce_batch_size", 1)),
            reproduce_attn_implementation=str(
                execution_raw.get("reproduce_attn_implementation") or "auto"
            ).strip(),
            probe_attn_implementation=str(
                execution_raw.get("probe_attn_implementation") or "eager"
            ).strip(),
            teacher_forced_attn_implementation=str(
                execution_raw.get("teacher_forced_attn_implementation") or "eager"
            ).strip(),
        ),
        decode=DecodeConfig(
            temperature=float(decode_raw.get("temperature", _AUTHORITATIVE_TEMPERATURE)),
            top_p=float(decode_raw.get("top_p", _AUTHORITATIVE_TOP_P)),
            repetition_penalty=float(
                decode_raw.get("repetition_penalty", _AUTHORITATIVE_REPETITION_PENALTY)
            ),
            max_new_tokens=int(
                decode_raw.get("max_new_tokens", _AUTHORITATIVE_MAX_NEW_TOKENS)
            ),
            seed=int(decode_raw.get("seed", _AUTHORITATIVE_SEED)),
            secondary_top_k=_ensure_tuple_int(
                decode_raw.get("secondary_top_k"),
                default=(),
                path="decode.secondary_top_k",
            ) if decode_raw.get("secondary_top_k") is not None else (),
            secondary_top_p=_ensure_tuple_float(
                decode_raw.get("secondary_top_p"),
                default=(),
                path="decode.secondary_top_p",
            ) if decode_raw.get("secondary_top_p") is not None else (),
        ),
        probe=ProbeConfig(
            max_cases=int(probe_raw.get("max_cases", 4)),
            step_window_before=int(probe_raw.get("step_window_before", 1)),
            step_window_after=int(probe_raw.get("step_window_after", 1)),
            token_top_k=int(probe_raw.get("token_top_k", 10)),
            capture_native_vision=bool(probe_raw.get("capture_native_vision", True)),
            enable_interventions=bool(probe_raw.get("enable_interventions", True)),
            intervention_max_cases=int(probe_raw.get("intervention_max_cases", 1)),
            late_layer_count=int(probe_raw.get("late_layer_count", 2)),
            visual_bias_scale=float(probe_raw.get("visual_bias_scale", 1.15)),
            history_attenuation_scale=float(
                probe_raw.get("history_attenuation_scale", 0.5)
            ),
            intervention_coord_only=bool(
                probe_raw.get("intervention_coord_only", True)
            ),
            intervention_greedy_steps=int(
                probe_raw.get("intervention_greedy_steps", 8)
            ),
        ),
        controls=ControlConfig(
            max_cases=int(controls_raw.get("max_cases", 4)),
            include_close_candidate=bool(
                controls_raw.get("include_close_candidate", True)
            ),
            coord_neighbor_radius=int(
                controls_raw.get("coord_neighbor_radius", _COORD_NEIGHBOR_RADIUS)
            ),
            same_desc_iou_threshold=float(
                controls_raw.get("same_desc_iou_threshold", 0.5)
            ),
        ),
        report=ReportConfig(
            write_markdown=bool(report_raw.get("write_markdown", True))
        ),
    )
    _validate_decode_contract(cfg.decode)
    return cfg


def _set_cuda_visible_devices(value: Optional[str]) -> None:
    if value is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(value)


def _resolve_run_dir(cfg: StudyConfig) -> Tuple[Path, str]:
    if cfg.workspace.root_dir:
        root = _resolve_path(cfg.workspace.root_dir)
        source = "workspace"
    else:
        root = _resolve_path(cfg.run.output_dir)
        source = "analysis_output"
    return root / cfg.run.name, source


def _audit_artifact_dir(root: Path) -> HistoricalArtifactBundle:
    paths: Dict[str, Optional[Path]] = {
        "gt_vs_pred_jsonl": root / "gt_vs_pred.jsonl",
        "pred_token_trace_jsonl": root / "pred_token_trace.jsonl",
        "pred_confidence_jsonl": root / "pred_confidence.jsonl",
        "resolved_config_json": root / "resolved_config.json",
        "eval_metrics_json": root / "eval" / "metrics.json",
        "eval_coco_real_metrics_json": root / "eval_coco_real" / "metrics.json",
        "eval_coco_real_strict_metrics_json": root / "eval_coco_real_strict" / "metrics.json",
        "eval_coco_real_strict_plausible_metrics_json": root
        / "eval_coco_real_strict_plausible"
        / "metrics.json",
    }
    return HistoricalArtifactBundle(
        root=root,
        paths={key: (path if path.exists() else None) for key, path in paths.items()},
    )


def _resolve_checkpoint(
    spec: CheckpointSpec,
    *,
    checkpoint_name_filter: str,
) -> ResolvedStudyCheckpoint:
    checkpoint_path, resolve_source = resolve_checkpoint_path(spec.path)
    _validate_checkpoint_surface(
        checkpoint_path,
        alias=spec.alias,
        name_filter=checkpoint_name_filter,
    )
    prompt_variant, object_field_order, prompt_source = resolve_prompt_controls_for_checkpoint(
        checkpoint_path,
        default_prompt_variant="coco_80",
        default_object_field_order="desc_first",
        override_prompt_variant=spec.prompt_variant,
        override_object_field_order=spec.object_field_order,
    )
    resolved = ResolvedCheckpoint(
        alias=spec.alias,
        path=checkpoint_path,
        resolve_source=str(resolve_source),
        artifact_kind="executable_checkpoint",
        fingerprint=str(checkpoint_path.name),
        prompt_variant=resolve_dense_prompt_variant_key(prompt_variant),
        object_field_order=normalize_object_field_order(
            object_field_order,
            path=f"resolved_checkpoint.{spec.alias}.object_field_order",
        ),
        prompt_control_source=str(prompt_source),
        provenance_sidecars={},
    )
    bundles: List[HistoricalArtifactBundle] = []
    for root_raw in spec.historical_artifact_dirs:
        bundle_root = _resolve_path(root_raw)
        if not bundle_root.exists():
            raise FileNotFoundError(
                f"Historical artifact dir does not exist for {spec.alias}: {bundle_root}"
            )
        if not bundle_root.is_dir():
            raise NotADirectoryError(
                f"Historical artifact root must be a directory for {spec.alias}: {bundle_root}"
            )
        bundles.append(_audit_artifact_dir(bundle_root))
    return ResolvedStudyCheckpoint(spec=spec, resolved=resolved, bundles=tuple(bundles))


def _inventory_row(checkpoint: ResolvedStudyCheckpoint) -> Dict[str, Any]:
    cells: Dict[str, Dict[str, Any]] = {}
    for bundle_idx, bundle in enumerate(checkpoint.bundles):
        for cell_name in _EXPECTED_ARTIFACT_CELLS:
            path = bundle.paths.get(cell_name)
            cells[f"bundle_{bundle_idx}:{cell_name}"] = {
                "available": bool(path is not None),
                "path": str(path) if path is not None else None,
            }
    if not checkpoint.bundles:
        for cell_name in _EXPECTED_ARTIFACT_CELLS:
            cells[f"bundle_0:{cell_name}"] = {
                "available": False,
                "path": None,
            }
    has_infer_artifact = _has_infer_artifact(checkpoint)
    probe_readiness = _probe_readiness(checkpoint)
    return {
        "alias": checkpoint.spec.alias,
        "checkpoint_path": str(checkpoint.resolved.path),
        "checkpoint_resolve_source": checkpoint.resolved.resolve_source,
        "prompt_variant": checkpoint.resolved.prompt_variant,
        "bbox_format": checkpoint.spec.bbox_format,
        "object_field_order": checkpoint.resolved.object_field_order,
        "prompt_control_source": checkpoint.resolved.prompt_control_source,
        "stage": checkpoint.spec.stage,
        "objective_family": checkpoint.spec.objective_family,
        "geometry_regime": checkpoint.spec.geometry_regime,
        "rollout_training_regime": checkpoint.spec.rollout_training_regime,
        "interpretation_confounds": list(checkpoint.spec.interpretation_confounds),
        "coord_soft_ce_w1_state": checkpoint.spec.coord_soft_ce_w1_state,
        "parent_checkpoint": checkpoint.spec.parent_checkpoint,
        "family_comparison_role": checkpoint.spec.family_comparison_role,
        "has_infer_artifact": bool(has_infer_artifact),
        "best_probe_surface": _best_probe_surface(checkpoint),
        "probe_readiness": probe_readiness,
        "artifact_bundles": [
            {
                "root": str(bundle.root),
                "paths": {
                    key: str(path) if path is not None else None
                    for key, path in bundle.paths.items()
                },
            }
            for bundle in checkpoint.bundles
        ],
        "comparison_cells": cells,
    }


def _bbox_xyxy(obj: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = obj.get("bbox_2d")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(bbox[i]) for i in range(4)]
            return (x1, y1, x2, y2)
        except (TypeError, ValueError):
            return None
    points = obj.get("points")
    if isinstance(points, list) and len(points) >= 4:
        try:
            if len(points) == 4:
                x1, y1, x2, y2 = [float(points[i]) for i in range(4)]
                return (x1, y1, x2, y2)
            xs = [float(points[i]) for i in range(0, len(points), 2)]
            ys = [float(points[i]) for i in range(1, len(points), 2)]
            return (min(xs), min(ys), max(xs), max(ys))
        except (TypeError, ValueError):
            return None
    return None


def _bbox_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_diag(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))


def _bbox_iou(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> float:
    inter_x1 = max(left[0], right[0])
    inter_y1 = max(left[1], right[1])
    inter_x2 = min(left[2], right[2])
    inter_y2 = min(left[3], right[3])
    inter = _bbox_area((inter_x1, inter_y1, inter_x2, inter_y2))
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(left) + _bbox_area(right) - inter
    return float(inter / max(union, 1e-6))


def _size_ratio(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> float:
    left_area = _bbox_area(left)
    right_area = _bbox_area(right)
    denom = max(left_area, right_area, 1e-6)
    return float(min(left_area, right_area) / denom)


def _center_distance(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> float:
    cx1, cy1 = _bbox_center(left)
    cx2, cy2 = _bbox_center(right)
    return math.hypot(cx1 - cx2, cy1 - cy2)


def _pair_duplicate_metrics(
    left_obj: Mapping[str, Any],
    right_obj: Mapping[str, Any],
    *,
    cfg: StudyConfig,
) -> Optional[Dict[str, Any]]:
    left_desc = normalize_desc(str(left_obj.get("desc") or ""))
    right_desc = normalize_desc(str(right_obj.get("desc") or ""))
    if not left_desc or left_desc != right_desc:
        return None
    left_box = _bbox_xyxy(left_obj)
    right_box = _bbox_xyxy(right_obj)
    if left_box is None or right_box is None:
        return None
    iou = _bbox_iou(left_box, right_box)
    size_ratio = _size_ratio(left_box, right_box)
    center_distance = _center_distance(left_box, right_box)
    local_radius = max(_bbox_diag(left_box), _bbox_diag(right_box), 1.0) * float(
        cfg.subset.local_center_radius_scale
    )
    duplicate_like = bool(
        iou >= float(cfg.subset.duplicate_iou_threshold)
        or (
            size_ratio >= float(cfg.subset.size_ratio_min)
            and center_distance <= float(local_radius)
        )
    )
    return {
        "desc": left_desc,
        "iou": float(iou),
        "size_ratio": float(size_ratio),
        "center_distance": float(center_distance),
        "local_radius": float(local_radius),
        "duplicate_like": duplicate_like,
    }


def _load_confidence_index(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        try:
            line_idx = int(row.get("line_idx"))
        except (TypeError, ValueError):
            continue
        out[line_idx] = row
    return out


def _load_trace_index(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        try:
            line_idx = int(row.get("line_idx"))
        except (TypeError, ValueError):
            continue
        out[line_idx] = row
    return out


def _find_primary_bundle(checkpoint: ResolvedStudyCheckpoint) -> Optional[HistoricalArtifactBundle]:
    for bundle in checkpoint.bundles:
        if bundle.paths.get("gt_vs_pred_jsonl") is not None:
            return bundle
    return checkpoint.bundles[0] if checkpoint.bundles else None


def _detect_onset(
    preds: Sequence[Mapping[str, Any]],
    *,
    confidence_record: Optional[Mapping[str, Any]],
    cfg: StudyConfig,
) -> Optional[Dict[str, Any]]:
    confidence_objects = list(confidence_record.get("objects") or []) if isinstance(
        confidence_record, Mapping
    ) else []
    duplicate_pairs = 0
    earliest: Optional[Dict[str, Any]] = None
    for object_idx in range(1, len(preds)):
        current = preds[object_idx]
        best_match: Optional[Dict[str, Any]] = None
        best_source_idx: Optional[int] = None
        for source_idx in range(object_idx):
            metrics = _pair_duplicate_metrics(preds[source_idx], current, cfg=cfg)
            if metrics is None or not metrics["duplicate_like"]:
                continue
            duplicate_pairs += 1
            score = (
                float(metrics["iou"]),
                float(metrics["size_ratio"]),
                -float(metrics["center_distance"]),
                -int(object_idx - source_idx),
            )
            if best_match is None or score > best_match["_score"]:
                best_match = {**metrics, "_score": score}
                best_source_idx = source_idx
        if best_match is None:
            continue
        current_conf = (
            confidence_objects[object_idx]
            if object_idx < len(confidence_objects)
            else {}
        )
        source_conf = (
            confidence_objects[best_source_idx]
            if best_source_idx is not None and best_source_idx < len(confidence_objects)
            else {}
        )
        onset_desc_positions = list(
            ((current_conf.get("confidence_details") or {}).get("desc_span_token_indices") or [])
        )
        onset_coord_positions = list(
            ((current_conf.get("confidence_details") or {}).get("matched_token_indices") or [])
        )
        source_desc_positions = list(
            ((source_conf.get("confidence_details") or {}).get("desc_span_token_indices") or [])
        )
        source_coord_positions = list(
            ((source_conf.get("confidence_details") or {}).get("matched_token_indices") or [])
        )
        earliest = {
            "object_idx": int(object_idx),
            "source_object_idx": int(best_source_idx) if best_source_idx is not None else None,
            "desc": str(current.get("desc") or ""),
            "source_desc": str(preds[best_source_idx].get("desc") or "")
            if best_source_idx is not None
            else None,
            "anchor_source": "detected_duplicate",
            "pair_metrics": {
                key: value
                for key, value in best_match.items()
                if not key.startswith("_")
            },
            "desc_span_token_indices": [int(v) for v in onset_desc_positions],
            "matched_token_indices": [int(v) for v in onset_coord_positions],
            "source_desc_span_token_indices": [int(v) for v in source_desc_positions],
            "source_matched_token_indices": [int(v) for v in source_coord_positions],
            "onset_token_span": [int(v) for v in onset_desc_positions],
            "onset_generated_token_idx": (
                int(onset_desc_positions[0])
                if onset_desc_positions
                else (int(onset_coord_positions[0]) if onset_coord_positions else None)
            ),
            "onset_field_phase": (
                "desc"
                if onset_desc_positions
                else (
                    "coord_x1"
                    if len(onset_coord_positions) >= 1
                    else None
                )
            ),
            "duplicate_pair_count": int(duplicate_pairs),
        }
        break
    return earliest


def _build_anchor_from_object_pair(
    preds: Sequence[Mapping[str, Any]],
    *,
    confidence_record: Optional[Mapping[str, Any]],
    cfg: StudyConfig,
    object_idx: int,
    source_object_idx: int,
    anchor_source: str,
) -> Optional[Dict[str, Any]]:
    if not (0 <= int(source_object_idx) < int(object_idx) < len(preds)):
        return None
    current = preds[int(object_idx)]
    source = preds[int(source_object_idx)]
    pair_metrics = _pair_duplicate_metrics(source, current, cfg=cfg)
    if pair_metrics is None:
        return None
    confidence_objects = list(confidence_record.get("objects") or []) if isinstance(
        confidence_record, Mapping
    ) else []
    current_conf = (
        confidence_objects[int(object_idx)]
        if int(object_idx) < len(confidence_objects)
        else {}
    )
    source_conf = (
        confidence_objects[int(source_object_idx)]
        if int(source_object_idx) < len(confidence_objects)
        else {}
    )
    onset_desc_positions = list(
        ((current_conf.get("confidence_details") or {}).get("desc_span_token_indices") or [])
    )
    onset_coord_positions = list(
        ((current_conf.get("confidence_details") or {}).get("matched_token_indices") or [])
    )
    source_desc_positions = list(
        ((source_conf.get("confidence_details") or {}).get("desc_span_token_indices") or [])
    )
    source_coord_positions = list(
        ((source_conf.get("confidence_details") or {}).get("matched_token_indices") or [])
    )
    return {
        "object_idx": int(object_idx),
        "source_object_idx": int(source_object_idx),
        "desc": str(current.get("desc") or ""),
        "source_desc": str(source.get("desc") or ""),
        "anchor_source": str(anchor_source),
        "pair_metrics": pair_metrics,
        "desc_span_token_indices": [int(v) for v in onset_desc_positions],
        "matched_token_indices": [int(v) for v in onset_coord_positions],
        "source_desc_span_token_indices": [int(v) for v in source_desc_positions],
        "source_matched_token_indices": [int(v) for v in source_coord_positions],
        "onset_token_span": [int(v) for v in onset_desc_positions],
        "onset_generated_token_idx": (
            int(onset_desc_positions[0])
            if onset_desc_positions
            else (int(onset_coord_positions[0]) if onset_coord_positions else None)
        ),
        "onset_field_phase": (
            "desc"
            if onset_desc_positions
            else ("coord_x1" if len(onset_coord_positions) >= 1 else None)
        ),
        "duplicate_pair_count": int(1 if pair_metrics.get("duplicate_like") else 0),
    }


def _resolve_probe_anchor(
    preds: Sequence[Mapping[str, Any]],
    *,
    confidence_record: Optional[Mapping[str, Any]],
    case_row: Mapping[str, Any],
    cfg: StudyConfig,
) -> Optional[Dict[str, Any]]:
    detected = _detect_onset(
        preds,
        confidence_record=confidence_record,
        cfg=cfg,
    )
    if detected is not None:
        return detected
    hint = case_row.get("analysis_anchor") or case_row.get("onset")
    if not isinstance(hint, Mapping):
        return None
    object_idx = hint.get("object_idx")
    source_object_idx = hint.get("source_object_idx")
    if object_idx is None or source_object_idx is None:
        return None
    return _build_anchor_from_object_pair(
        preds,
        confidence_record=confidence_record,
        cfg=cfg,
        object_idx=int(object_idx),
        source_object_idx=int(source_object_idx),
        anchor_source=str(hint.get("anchor_source") or "case_manifest_pair"),
    )


def _selection_priority(row: Mapping[str, Any]) -> Tuple[int, int, int, int, int]:
    onset = 1 if row.get("onset") else 0
    duplicate_pairs = int(row.get("same_desc_duplicate_pair_count") or 0)
    max_desc_count = int(row.get("max_desc_count") or 0)
    pred_count = int(row.get("pred_count") or 0)
    overflow = int(row.get("pred_count") or 0) - int(row.get("gt_count") or 0)
    return (onset, duplicate_pairs, max_desc_count, pred_count, overflow)


def _selection_reason(row: Mapping[str, Any]) -> str:
    onset = row.get("onset") or {}
    desc = str((onset.get("desc") or row.get("top_desc") or "")).strip()
    return (
        f"same-desc duplicate family desc={desc or 'unknown'} "
        f"pred_count={int(row.get('pred_count') or 0)} "
        f"max_desc_count={int(row.get('max_desc_count') or 0)} "
        f"duplicate_pairs={int(row.get('same_desc_duplicate_pair_count') or 0)} "
        f"line_idx={int(row.get('line_idx') or -1)}"
    )


def _mine_cases_for_checkpoint(
    checkpoint: ResolvedStudyCheckpoint,
    *,
    cfg: StudyConfig,
) -> List[Dict[str, Any]]:
    bundle = _find_primary_bundle(checkpoint)
    if bundle is None:
        return []
    gt_vs_pred_path = bundle.paths.get("gt_vs_pred_jsonl")
    if gt_vs_pred_path is None:
        return []
    gt_rows = _read_jsonl(gt_vs_pred_path)
    confidence_index = _load_confidence_index(bundle.paths.get("pred_confidence_jsonl"))
    rows: List[Dict[str, Any]] = []
    pinned = set(cfg.subset.pinned_line_indices.get(checkpoint.spec.alias, ()))
    for line_idx, gt_row in enumerate(gt_rows):
        preds = list(gt_row.get("pred") or [])
        gts = list(gt_row.get("gt") or [])
        if len(preds) < int(cfg.subset.min_pred_objects):
            continue
        duplicate_pair_count = 0
        desc_counter: Counter[str] = Counter()
        for pred in preds:
            desc = normalize_desc(str(pred.get("desc") or ""))
            if desc:
                desc_counter[desc] += 1
        for right_idx in range(1, len(preds)):
            for left_idx in range(right_idx):
                metrics = _pair_duplicate_metrics(preds[left_idx], preds[right_idx], cfg=cfg)
                if metrics is not None and metrics["duplicate_like"]:
                    duplicate_pair_count += 1
        onset = _detect_onset(
            preds,
            confidence_record=confidence_index.get(line_idx),
            cfg=cfg,
        )
        should_keep = (
            line_idx in pinned
            or (
                duplicate_pair_count >= int(cfg.subset.min_duplicate_pairs)
                and onset is not None
            )
        )
        if not should_keep:
            continue
        max_desc_count = max(desc_counter.values()) if desc_counter else 0
        top_desc = max(desc_counter, key=desc_counter.get) if desc_counter else None
        row = {
            "checkpoint_alias": checkpoint.spec.alias,
            "source_artifact_root": str(bundle.root),
            "source_gt_vs_pred_jsonl": str(gt_vs_pred_path),
            "source_pred_token_trace_jsonl": (
                str(bundle.paths["pred_token_trace_jsonl"])
                if bundle.paths.get("pred_token_trace_jsonl") is not None
                else None
            ),
            "source_pred_confidence_jsonl": (
                str(bundle.paths["pred_confidence_jsonl"])
                if bundle.paths.get("pred_confidence_jsonl") is not None
                else None
            ),
            "line_idx": int(line_idx),
            "image": str(gt_row.get("image") or ""),
            "image_id": gt_row.get("image_id"),
            "width": int(gt_row.get("width") or 0),
            "height": int(gt_row.get("height") or 0),
            "gt_count": int(len(gts)),
            "pred_count": int(len(preds)),
            "max_desc_count": int(max_desc_count),
            "top_desc": top_desc,
            "same_desc_duplicate_pair_count": int(duplicate_pair_count),
            "historical_row": gt_row,
            "onset": onset,
        }
        row["selection_reason"] = _selection_reason(row)
        rows.append(row)
    rows.sort(
        key=lambda item: (
            item["checkpoint_alias"],
            _selection_priority(item),
            -int(item["line_idx"]),
        ),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    for row in rows:
        alias_count = sum(
            1 for item in kept if item["checkpoint_alias"] == row["checkpoint_alias"]
        )
        if alias_count >= int(cfg.subset.max_cases_per_checkpoint):
            continue
        if len(kept) >= int(cfg.subset.max_cases_total):
            break
        kept.append(row)
    return kept


def _bootstrap_case_rows_for_checkpoint(
    checkpoint: ResolvedStudyCheckpoint,
    *,
    cfg: StudyConfig,
) -> List[Dict[str, Any]]:
    manifest_path_raw = cfg.subset.bootstrap_case_manifest_jsonl
    if manifest_path_raw is None:
        return []
    manifest_path = _resolve_path(manifest_path_raw)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"bootstrap case manifest does not exist: {manifest_path}"
        )

    source_cache: Dict[str, List[Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    for manifest_row in _read_jsonl(manifest_path):
        source_gt_vs_pred_raw = str(manifest_row.get("source_gt_vs_pred_jsonl") or "").strip()
        if not source_gt_vs_pred_raw:
            raise ValueError(
                "bootstrap case manifest rows must define source_gt_vs_pred_jsonl"
            )
        line_idx = int(manifest_row.get("line_idx") or 0)
        source_gt_vs_pred_path = _resolve_path(source_gt_vs_pred_raw)
        cache_key = str(source_gt_vs_pred_path)
        if cache_key not in source_cache:
            source_cache[cache_key] = _read_jsonl(source_gt_vs_pred_path)
        source_rows = source_cache[cache_key]
        if line_idx < 0 or line_idx >= len(source_rows):
            raise IndexError(
                "bootstrap case manifest line_idx is out of range for "
                f"{source_gt_vs_pred_path}: {line_idx}"
            )
        historical_row = dict(source_rows[line_idx])
        bootstrap_row = {
            "checkpoint_alias": checkpoint.spec.alias,
            "source_artifact_root": str(
                _resolve_path(
                    str(
                        manifest_row.get("source_artifact_root")
                        or source_gt_vs_pred_path.parent
                    )
                )
            ),
            "source_gt_vs_pred_jsonl": str(source_gt_vs_pred_path),
            "source_pred_token_trace_jsonl": (
                str(
                    _resolve_path(
                        str(manifest_row.get("source_pred_token_trace_jsonl"))
                    )
                )
                if manifest_row.get("source_pred_token_trace_jsonl") is not None
                else None
            ),
            "source_pred_confidence_jsonl": (
                str(
                    _resolve_path(
                        str(manifest_row.get("source_pred_confidence_jsonl"))
                    )
                )
                if manifest_row.get("source_pred_confidence_jsonl") is not None
                else None
            ),
            "line_idx": int(line_idx),
            "image": str(historical_row.get("image") or manifest_row.get("image") or ""),
            "image_id": historical_row.get("image_id", manifest_row.get("image_id")),
            "width": int(historical_row.get("width") or manifest_row.get("width") or 0),
            "height": int(historical_row.get("height") or manifest_row.get("height") or 0),
            "gt_count": int(
                manifest_row.get("gt_count")
                or len(historical_row.get("gt") or [])
            ),
            "pred_count": int(manifest_row.get("pred_count") or 0),
            "max_desc_count": int(manifest_row.get("max_desc_count") or 0),
            "top_desc": manifest_row.get("top_desc"),
            "same_desc_duplicate_pair_count": int(
                manifest_row.get("same_desc_duplicate_pair_count") or 0
            ),
            "historical_row": historical_row,
            "onset": manifest_row.get("onset"),
            "bootstrap_reference_case_id": manifest_row.get("case_id"),
        }
        selection_reason = str(manifest_row.get("selection_reason") or "").strip()
        if selection_reason:
            bootstrap_row["selection_reason"] = (
                f"bootstrap-manifest from {manifest_row.get('checkpoint_alias')}: "
                f"{selection_reason}"
            )
        else:
            bootstrap_row["selection_reason"] = (
                "bootstrap-manifest "
                f"source={manifest_row.get('checkpoint_alias') or 'unknown'} "
                f"line_idx={line_idx}"
            )
        rows.append(bootstrap_row)
        if len(rows) >= int(cfg.subset.max_cases_per_checkpoint):
            break
    return rows


def _case_id(
    alias: str,
    line_idx: int,
    *,
    source_alias: str | None = None,
    source_case_id: str | None = None,
) -> str:
    if source_case_id:
        return f"{alias}-from_{source_case_id}"
    suffix = f"-from_{source_alias}" if source_alias else ""
    return f"{alias}{suffix}-line_{int(line_idx):05d}"


def _expand_replay_cases(
    case_rows: Sequence[Dict[str, Any]],
    *,
    cfg: StudyConfig,
    resolved_checkpoints: Sequence[ResolvedStudyCheckpoint],
) -> List[Dict[str, Any]]:
    replay_aliases = tuple(str(alias).strip() for alias in cfg.subset.replay_case_aliases if str(alias).strip())
    if not replay_aliases or not case_rows:
        return list(case_rows)
    known_aliases = {item.spec.alias for item in resolved_checkpoints}
    expanded: List[Dict[str, Any]] = list(case_rows)
    for row in case_rows:
        source_alias = str(row["checkpoint_alias"])
        for replay_alias in replay_aliases:
            if replay_alias == source_alias or replay_alias not in known_aliases:
                continue
            replay_row = dict(row)
            replay_row["checkpoint_alias"] = replay_alias
            replay_row["replay_source_checkpoint_alias"] = source_alias
            replay_row["replay_source_case_id"] = str(row["case_id"])
            replay_row["selection_reason"] = (
                f"{row['selection_reason']} | replay_target={replay_alias} "
                f"replay_source={source_alias}"
            )
            replay_row["case_id"] = _case_id(
                replay_alias,
                int(row["line_idx"]),
                source_case_id=str(row["case_id"]),
            )
            expanded.append(replay_row)
    return expanded


def _build_inventory(
    resolved_checkpoints: Sequence[ResolvedStudyCheckpoint],
    *,
    run_dir: Path,
) -> Dict[str, Any]:
    inventory_rows = [_inventory_row(item) for item in resolved_checkpoints]
    inventory_dir = run_dir / "inventory"
    ready_rows = [
        row for row in inventory_rows if row.get("probe_readiness") == "ready_to_probe"
    ]
    fresh_rows = [
        row
        for row in inventory_rows
        if row.get("probe_readiness") != "ready_to_probe"
    ]
    inventory_payload = {
        "checkpoints": inventory_rows,
        "ready_to_probe": ready_rows,
        "fresh_inference_needed": fresh_rows,
        "family_comparison": _family_comparison_summary(inventory_rows),
    }
    _write_json(inventory_dir / "checkpoint_inventory.json", inventory_payload)
    return {
        "inventory_dir": str(inventory_dir),
        "checkpoints": inventory_rows,
        "ready_to_probe": ready_rows,
        "fresh_inference_needed": fresh_rows,
        "family_comparison": inventory_payload["family_comparison"],
    }


def _build_case_selection(
    resolved_checkpoints: Sequence[ResolvedStudyCheckpoint],
    *,
    cfg: StudyConfig,
    run_dir: Path,
) -> Dict[str, Any]:
    selected: List[Dict[str, Any]] = []
    for checkpoint in resolved_checkpoints:
        mined = _mine_cases_for_checkpoint(checkpoint, cfg=cfg)
        if not mined and cfg.subset.bootstrap_case_manifest_jsonl is not None:
            mined = _bootstrap_case_rows_for_checkpoint(checkpoint, cfg=cfg)
        selected.extend(mined)
    selected = selected[: int(cfg.subset.max_cases_total)]
    case_rows = []
    selected_cases = []
    for row in selected:
        bootstrap_reference_case_id = str(
            row.get("bootstrap_reference_case_id") or ""
        ).strip()
        if bootstrap_reference_case_id.startswith(f"{row['checkpoint_alias']}-"):
            case_id = bootstrap_reference_case_id
        elif bootstrap_reference_case_id:
            case_id = _case_id(
                row["checkpoint_alias"],
                int(row["line_idx"]),
                source_case_id=bootstrap_reference_case_id,
            )
        else:
            case_id = _case_id(row["checkpoint_alias"], int(row["line_idx"]))
        full_row = {**row, "case_id": case_id}
        selected_cases.append(full_row)
    source_case_count = len(selected_cases)
    selected_cases = _expand_replay_cases(
        selected_cases,
        cfg=cfg,
        resolved_checkpoints=resolved_checkpoints,
    )
    for full_row in selected_cases:
        case_rows.append({k: v for k, v in full_row.items() if k != "historical_row"})
    case_dir = run_dir / "cases"
    _write_jsonl(case_dir / "selected_cases.jsonl", case_rows)
    summary = {
        "num_selected_cases": int(len(case_rows)),
        "num_selected_source_cases": int(source_case_count),
        "num_replay_cases": int(len(case_rows) - source_case_count),
        "selected_case_ids": [row["case_id"] for row in case_rows],
    }
    _write_json(case_dir / "summary.json", summary)
    return {
        "case_dir": str(case_dir),
        "selected_cases": selected_cases,
        "summary": summary,
    }


def _assign_local_line_indices(
    selected_cases: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    local_idx_by_alias: Dict[str, int] = defaultdict(int)
    assigned: List[Dict[str, Any]] = []
    for row in selected_cases:
        alias = str(row["checkpoint_alias"])
        updated = dict(row)
        updated["local_line_idx"] = int(local_idx_by_alias[alias])
        local_idx_by_alias[alias] += 1
        assigned.append(updated)
    return assigned


def _subset_record_for_case(row: Mapping[str, Any]) -> Dict[str, Any]:
    gt_row = dict(row["historical_row"])
    source_jsonl = Path(str(row["source_gt_vs_pred_jsonl"]))
    source_artifact_root = Path(str(row.get("source_artifact_root") or source_jsonl.parent))
    root_image_dir: Path | None = REPO_ROOT
    resolved_config_path = source_artifact_root / "resolved_config.json"
    if resolved_config_path.exists():
        try:
            resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
            root_raw = resolved_config.get("root_image_dir")
            if isinstance(root_raw, str) and root_raw.strip():
                root_image_dir = Path(root_raw.strip())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            root_image_dir = REPO_ROOT
    resolved_image = resolve_image_path_strict(
        str(gt_row.get("image") or ""),
        jsonl_dir=source_jsonl.parent,
        root_image_dir=root_image_dir,
    )
    if resolved_image is None:
        raise FileNotFoundError(
            "Could not resolve historical image path for duplication case "
            f"{row['case_id']}: image={gt_row.get('image')!r} source_jsonl={source_jsonl}"
        )
    image_str = str(resolved_image.resolve())
    return {
        "images": [image_str],
        "image": image_str,
        "width": int(gt_row.get("width") or 0),
        "height": int(gt_row.get("height") or 0),
        "gt": list(gt_row.get("gt") or []),
        "objects": list(gt_row.get("gt") or []),
        "image_id": gt_row.get("image_id"),
        "metadata": {
            "duplication_case_id": str(row["case_id"]),
            "source_gt_vs_pred_jsonl": str(row["source_gt_vs_pred_jsonl"]),
            "source_line_idx": int(row["line_idx"]),
            "selection_reason": str(row["selection_reason"]),
        },
    }


def _load_image_for_case(image_field: str) -> Image.Image:
    resolved = resolve_image_path_strict(
        image_field,
        jsonl_dir=REPO_ROOT,
        root_image_dir=REPO_ROOT,
    )
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve image path from case bundle: {image_field}")
    return Image.open(resolved).convert("RGB")


def _run_reproduction_for_checkpoint(
    checkpoint: ResolvedStudyCheckpoint,
    *,
    cfg: StudyConfig,
    run_dir: Path,
    case_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    reproduce_root = run_dir / "reproduce" / checkpoint.spec.alias
    subset_rows = [_subset_record_for_case(row) for row in case_rows]
    subset_path = reproduce_root / "subset.jsonl"
    _write_jsonl(subset_path, subset_rows)

    infer_cfg = InferenceConfig(
        gt_jsonl=str(subset_path),
        model_checkpoint=str(checkpoint.resolved.path),
        mode="coord",
        prompt_variant=checkpoint.resolved.prompt_variant,
        bbox_format=checkpoint.spec.bbox_format,
        object_field_order=checkpoint.resolved.object_field_order,
        out_path=str(reproduce_root / "gt_vs_pred.jsonl"),
        pred_token_trace_path=str(reproduce_root / "pred_token_trace.jsonl"),
        summary_path=str(reproduce_root / "summary.json"),
        root_image_dir=str(REPO_ROOT),
        device=cfg.execution.device,
        backend_type="hf",
        backend={"attn_implementation": cfg.execution.reproduce_attn_implementation},
    )
    gen_cfg = GenerationConfig(
        temperature=float(cfg.decode.temperature),
        top_p=float(cfg.decode.top_p),
        max_new_tokens=int(cfg.decode.max_new_tokens),
        repetition_penalty=float(cfg.decode.repetition_penalty),
        batch_size=int(cfg.execution.reproduce_batch_size),
        seed=int(cfg.decode.seed),
    )
    engine = InferenceEngine(infer_cfg, gen_cfg)
    out_path, summary_path = engine.infer()
    pred_confidence_path = reproduce_root / "pred_confidence.jsonl"
    scored_path = reproduce_root / "gt_vs_pred_scored.jsonl"
    confidence_summary_path = reproduce_root / "confidence_postop_summary.json"
    if checkpoint.spec.bbox_format == "cxcy_logw_logh":
        rows = _read_jsonl(out_path)
        write_jsonl_records(
            scored_path,
            with_constant_scores(
                records=rows,
                pred_score_source=CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_SOURCE,
                pred_score_version=CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_VERSION,
                constant_score=CXCY_LOGW_LOGH_CONSTANT_SCORE,
            ),
        )
        confidence_summary = {
            "total_samples": int(len(rows)),
            "total_pred_objects": int(
                sum(
                    len(record.get("pred") or [])
                    for record in rows
                    if isinstance(record.get("pred"), list)
                )
            ),
            "kept_pred_objects": None,
            "dropped_pred_objects": None,
            "kept_fraction": None,
            "pred_score_source": CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_SOURCE,
            "pred_score_version": CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_VERSION,
            "confidence_method": "constant_score_materialization",
        }
        confidence_summary_path.write_text(
            json.dumps(confidence_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if pred_confidence_path.exists():
            pred_confidence_path.unlink()
    else:
        confidence_summary = run_confidence_postop(
            ConfidencePostOpPaths(
                gt_vs_pred_jsonl=out_path,
                pred_token_trace_jsonl=reproduce_root / "pred_token_trace.jsonl",
                pred_confidence_jsonl=pred_confidence_path,
                gt_vs_pred_scored_jsonl=scored_path,
                confidence_postop_summary_json=confidence_summary_path,
            ),
            options=ConfidencePostOpOptions(),
        )
    return {
        "subset_jsonl": str(subset_path),
        "gt_vs_pred_jsonl": str(out_path),
        "pred_token_trace_jsonl": str(reproduce_root / "pred_token_trace.jsonl"),
        "pred_confidence_jsonl": str(pred_confidence_path),
        "gt_vs_pred_scored_jsonl": str(scored_path),
        "summary_json": str(summary_path),
        "confidence_summary": confidence_summary,
    }


def _reproduce_cases(
    resolved_checkpoints: Sequence[ResolvedStudyCheckpoint],
    *,
    cfg: StudyConfig,
    run_dir: Path,
    selected_cases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not selected_cases:
        return {"reproductions": []}
    by_alias: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_cases:
        by_alias[str(row["checkpoint_alias"])].append(row)
    alias_to_checkpoint = {item.spec.alias: item for item in resolved_checkpoints}
    out_rows: List[Dict[str, Any]] = []
    for alias, rows in sorted(by_alias.items()):
        checkpoint = alias_to_checkpoint[alias]
        bundle = _run_reproduction_for_checkpoint(
            checkpoint,
            cfg=cfg,
            run_dir=run_dir,
            case_rows=rows,
        )
        out_rows.append({"checkpoint_alias": alias, **bundle})
    reproduce_dir = run_dir / "reproduce"
    _write_json(reproduce_dir / "summary.json", {"reproductions": out_rows})
    return {"reproductions": out_rows}


def _maybe_mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(mean(clean)) if clean else None


def _layer_group_mass_summary(
    attention_summary: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    layer_rows: List[Dict[str, Any]] = []
    visual_series: List[Tuple[int, float]] = []
    history_series: List[Tuple[int, float]] = []
    prior_coord_series: List[Tuple[int, float]] = []
    for layer in attention_summary:
        heads = list(layer.get("heads") or [])
        if not heads:
            continue
        group_names = sorted(
            {
                str(name)
                for head in heads
                for name in (head.get("group_masses") or {}).keys()
            }
        )
        mean_group_masses = {
            name: _maybe_mean(
                [
                    float((head.get("group_masses") or {}).get(name, 0.0))
                    for head in heads
                ]
            )
            for name in group_names
        }
        layer_idx = int(layer.get("layer_idx", len(layer_rows)))
        layer_rows.append(
            {
                "layer_idx": layer_idx,
                "mean_group_masses": mean_group_masses,
            }
        )
        visual_series.append((layer_idx, float(mean_group_masses.get("visual_tokens") or 0.0)))
        history_series.append(
            (layer_idx, float(mean_group_masses.get("generated_history") or 0.0))
        )
        prior_coord_series.append(
            (layer_idx, float(mean_group_masses.get("source_coord_tokens") or 0.0))
        )
    if not layer_rows:
        return {"layers": [], "overwrite_summary": {}}
    peak_visual_layer, peak_visual_mass = max(visual_series, key=lambda item: item[1])
    final_row = layer_rows[-1]
    final_visual_mass = float(final_row["mean_group_masses"].get("visual_tokens") or 0.0)
    final_history_mass = float(
        final_row["mean_group_masses"].get("generated_history") or 0.0
    )
    final_prior_coord_mass = float(
        final_row["mean_group_masses"].get("source_coord_tokens") or 0.0
    )
    peak_layer_row = next(
        row for row in layer_rows if int(row["layer_idx"]) == int(peak_visual_layer)
    )
    peak_layer_history = float(
        peak_layer_row["mean_group_masses"].get("generated_history") or 0.0
    )
    peak_layer_prior_coord = float(
        peak_layer_row["mean_group_masses"].get("source_coord_tokens") or 0.0
    )
    overwrite_summary = {
        "peak_visual_layer_idx": int(peak_visual_layer),
        "peak_visual_mass": float(peak_visual_mass),
        "peak_layer_history_mass": float(peak_layer_history),
        "peak_layer_prior_coord_mass": float(peak_layer_prior_coord),
        "final_visual_mass": float(final_visual_mass),
        "final_generated_history_mass": float(final_history_mass),
        "final_prior_coord_mass": float(final_prior_coord_mass),
        "visual_drop_from_peak_to_final": float(peak_visual_mass - final_visual_mass),
        "final_history_minus_visual": float(final_history_mass - final_visual_mass),
        "final_prior_coord_minus_visual": float(final_prior_coord_mass - final_visual_mass),
        "history_overwrite_detected": bool(
            peak_visual_mass > 0.0
            and peak_visual_layer < int(final_row["layer_idx"])
            and final_history_mass > final_visual_mass
        ),
        "prior_coord_overwrite_detected": bool(
            peak_visual_mass > 0.0
            and peak_visual_layer < int(final_row["layer_idx"])
            and final_prior_coord_mass > final_visual_mass
        ),
    }
    return {
        "layers": layer_rows,
        "overwrite_summary": overwrite_summary,
    }


def _tensor_summary(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, torch.Tensor):
        values = obj.detach().float()
        return {
            "shape": list(values.shape),
            "mean_abs": float(values.abs().mean().item()),
            "l2_norm": float(values.norm().item()),
        }
    if isinstance(obj, (list, tuple)):
        for item in obj:
            summary = _tensor_summary(item)
            if summary:
                return summary
    if isinstance(obj, Mapping):
        for item in obj.values():
            summary = _tensor_summary(item)
            if summary:
                return summary
    return {}


class Qwen3VLSurgeryProber:
    def __init__(
        self,
        *,
        checkpoint: ResolvedCheckpoint,
        bbox_format: AllowedBBoxFormat = DEFAULT_BBOX_FORMAT,
        device: str,
        attn_implementation: str,
    ) -> None:
        self.checkpoint = checkpoint
        self.bbox_format = normalize_bbox_format(
            bbox_format,
            path="duplication_collapse.prober.bbox_format",
        )
        self.device = device
        self.attn_implementation = attn_implementation
        self.model = self._load_model()
        self.processor = AutoProcessor.from_pretrained(
            str(self.checkpoint.path),
            trust_remote_code=True,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Probe processor did not expose tokenizer")
        self.tokenizer = tokenizer
        self.coord_token_ids = get_coord_token_ids(self.tokenizer, validate=True)
        self.coord_token_ids_tensor = torch.tensor(
            self.coord_token_ids,
            dtype=torch.long,
            device=self.device,
        )

    def _load_model(self) -> Qwen3VLForConditionalGeneration:
        candidates: List[str] = []
        requested = str(self.attn_implementation or "eager").strip().lower()
        for cand in (requested, "eager", "sdpa", "flash_attention_2"):
            if cand and cand not in candidates:
                candidates.append(cand)
        last_exc: Exception | None = None
        for cand in candidates:
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.checkpoint.path),
                    torch_dtype=torch.bfloat16,
                    attn_implementation=cand,
                )
                model = model.to(self.device)
                model.eval()
                self.attn_implementation = cand
                return model
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                last_exc = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise RuntimeError(
            f"Failed to load surgery prober model for {self.checkpoint.path}"
        ) from last_exc

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prompt_inputs(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        temp_cfg = InferenceConfig(
            gt_jsonl="unused.jsonl",
            model_checkpoint=str(self.checkpoint.path),
            mode="coord",
            prompt_variant=self.checkpoint.prompt_variant,
            bbox_format=self.bbox_format,
            object_field_order=self.checkpoint.object_field_order,
            out_path="unused.jsonl",
            device=self.device,
            backend_type="hf",
            root_image_dir=str(REPO_ROOT),
            backend={"attn_implementation": self.attn_implementation},
        )
        temp_engine = InferenceEngine(
            temp_cfg,
            GenerationConfig(
                temperature=_AUTHORITATIVE_TEMPERATURE,
                top_p=_AUTHORITATIVE_TOP_P,
                repetition_penalty=_AUTHORITATIVE_REPETITION_PENALTY,
                max_new_tokens=16,
                seed=_AUTHORITATIVE_SEED,
            ),
        )
        temp_engine.processor = self.processor
        temp_engine.model = self.model
        messages = temp_engine._build_messages(image)
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }

    def _dependency_roots(self) -> Dict[str, Any]:
        roots = {
            "model_class_file": inspect.getfile(self.model.__class__),
            "processor_class_file": inspect.getfile(self.processor.__class__),
        }
        visual = getattr(self.model, "visual", None)
        roots["has_visual_module"] = bool(visual is not None)
        if visual is not None:
            roots["visual_class_file"] = inspect.getfile(visual.__class__)
        return roots

    def _visual_token_positions(self, prompt_input_ids: torch.Tensor) -> List[int]:
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            return []
        positions = (
            (prompt_input_ids[0] == int(image_token_id)).nonzero(as_tuple=False).view(-1)
        )
        return [int(pos.item()) for pos in positions]

    def _apply_decode_processors(
        self,
        *,
        raw_logits: torch.Tensor,
        history_ids: Sequence[int],
        cfg: StudyConfig,
    ) -> torch.Tensor:
        processors = LogitsProcessorList()
        if float(cfg.decode.repetition_penalty) != 1.0:
            processors.append(
                RepetitionPenaltyLogitsProcessor(float(cfg.decode.repetition_penalty))
            )
        if not processors:
            return raw_logits
        history = torch.tensor([list(history_ids)], dtype=torch.long, device=self.device)
        return processors(history, raw_logits.unsqueeze(0))[0]

    def _top_tokens(self, scores: torch.Tensor, *, k: int) -> List[Dict[str, Any]]:
        limit = min(int(k), int(scores.shape[-1]))
        vals, idxs = torch.topk(scores, k=limit)
        probs = torch.softmax(scores.float(), dim=-1)
        out: List[Dict[str, Any]] = []
        for value, index in zip(vals.detach().cpu().tolist(), idxs.detach().cpu().tolist()):
            token_id = int(index)
            out.append(
                {
                    "token_id": token_id,
                    "token_text": self.tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "score": float(value),
                    "prob": float(probs[token_id].detach().cpu().item()),
                }
            )
        return out

    def _coord_distribution_summary(
        self,
        *,
        logits: torch.Tensor,
        target_token_id: Optional[int],
        previous_token_id: Optional[int],
        neighbor_radius: int,
    ) -> Dict[str, Any]:
        coord_logits = logits.index_select(dim=0, index=self.coord_token_ids_tensor)
        coord_probs = torch.softmax(coord_logits.float(), dim=-1)
        entropy = -torch.sum(coord_probs * torch.log(coord_probs.clamp_min(1e-12))).item()
        top1_idx = int(torch.argmax(coord_probs).detach().cpu().item())
        top1_prob = float(coord_probs[top1_idx].detach().cpu().item())
        expected_bin = float(
            torch.sum(
                coord_probs
                * torch.arange(
                    _COORD_TOKEN_COUNT,
                    device=coord_probs.device,
                    dtype=coord_probs.dtype,
                )
            ).detach().cpu().item()
        )
        target_prob = None
        if target_token_id is not None and int(target_token_id) in self.coord_token_ids:
            target_idx = self.coord_token_ids.index(int(target_token_id))
            target_prob = float(coord_probs[target_idx].detach().cpu().item())
        previous_prob = None
        previous_neighborhood_mass = None
        previous_bin = None
        if previous_token_id is not None and int(previous_token_id) in self.coord_token_ids:
            previous_bin = self.coord_token_ids.index(int(previous_token_id))
            previous_prob = float(coord_probs[previous_bin].detach().cpu().item())
            lo = max(0, previous_bin - int(neighbor_radius))
            hi = min(_COORD_TOKEN_COUNT, previous_bin + int(neighbor_radius) + 1)
            previous_neighborhood_mass = float(
                coord_probs[lo:hi].sum().detach().cpu().item()
            )
        return {
            "entropy": float(entropy),
            "top1_prob": float(top1_prob),
            "argmax_bin": int(top1_idx),
            "expected_bin": float(expected_bin),
            "target_bin_prob": target_prob,
            "previous_box_prob": previous_prob,
            "previous_box_neighborhood_mass": previous_neighborhood_mass,
            "previous_box_bin": previous_bin,
        }

    def _hidden_summary(
        self,
        hidden_states: Sequence[torch.Tensor] | None,
        *,
        previous_final: Optional[torch.Tensor],
    ) -> Tuple[List[Dict[str, Any]], Optional[torch.Tensor], Dict[str, Any]]:
        if not hidden_states:
            return [], previous_final, {}
        rows: List[Dict[str, Any]] = []
        prev_layer_vec: Optional[torch.Tensor] = None
        final_vec: Optional[torch.Tensor] = None
        for layer_idx, layer in enumerate(hidden_states):
            if not isinstance(layer, torch.Tensor):
                continue
            vec = layer[0, -1].detach().float().cpu()
            final_vec = vec
            layer_summary = {
                "layer_idx": int(layer_idx),
                "l2_norm": float(vec.norm().item()),
                "mean_abs": float(vec.abs().mean().item()),
            }
            if prev_layer_vec is not None:
                delta = vec - prev_layer_vec
                layer_summary["delta_l2_from_prev_layer"] = float(delta.norm().item())
                denom = float(vec.norm().item() * prev_layer_vec.norm().item())
                if denom > 0.0:
                    layer_summary["cosine_to_prev_layer"] = float(
                        torch.dot(vec, prev_layer_vec).item() / denom
                    )
            rows.append(layer_summary)
            prev_layer_vec = vec
        cross_step: Dict[str, Any] = {}
        if final_vec is not None and previous_final is not None:
            delta = final_vec - previous_final
            denom = float(final_vec.norm().item() * previous_final.norm().item())
            cross_step["final_hidden_delta_l2"] = float(delta.norm().item())
            if denom > 0.0:
                cross_step["final_hidden_cosine_to_prev_step"] = float(
                    torch.dot(final_vec, previous_final).item() / denom
                )
        return rows, final_vec, cross_step

    def _layer_logit_lens_summary(
        self,
        hidden_states: Sequence[torch.Tensor] | None,
        *,
        target_token_id: Optional[int],
        previous_token_id: Optional[int],
        phase: Optional[str],
    ) -> Dict[str, Any]:
        if not hidden_states or target_token_id is None:
            return {}
        if not str(phase or "").startswith("coord_"):
            return {}
        language_model = getattr(self.model, "language_model", None)
        norm = getattr(language_model, "norm", None)
        lm_head = getattr(self.model, "lm_head", None)
        if norm is None or lm_head is None:
            return {}

        rows: List[Dict[str, Any]] = []
        first_prev_favored_layer: Optional[int] = None
        first_target_favored_layer: Optional[int] = None
        final_margin: Optional[float] = None
        peak_prev_margin: Optional[float] = None
        peak_target_margin: Optional[float] = None

        for layer_idx, layer in enumerate(hidden_states):
            if not isinstance(layer, torch.Tensor):
                continue
            token_vec = layer[:, -1, :]
            normed_vec = norm(token_vec)
            logits = lm_head(normed_vec)[0].detach().float()
            probs = torch.softmax(logits, dim=-1)
            target_prob = float(probs[int(target_token_id)].detach().cpu().item())
            previous_prob = (
                float(probs[int(previous_token_id)].detach().cpu().item())
                if previous_token_id is not None
                else None
            )
            target_minus_previous = (
                target_prob - previous_prob
                if previous_prob is not None
                else None
            )
            rows.append(
                {
                    "layer_idx": int(layer_idx),
                    "target_prob": target_prob,
                    "previous_prob": previous_prob,
                    "target_minus_previous": target_minus_previous,
                }
            )
            if target_minus_previous is None:
                continue
            if first_prev_favored_layer is None and target_minus_previous < 0.0:
                first_prev_favored_layer = int(layer_idx)
            if first_target_favored_layer is None and target_minus_previous > 0.0:
                first_target_favored_layer = int(layer_idx)
            peak_prev_margin = (
                target_minus_previous
                if peak_prev_margin is None
                else min(peak_prev_margin, target_minus_previous)
            )
            peak_target_margin = (
                target_minus_previous
                if peak_target_margin is None
                else max(peak_target_margin, target_minus_previous)
            )
            final_margin = target_minus_previous

        return {
            "rows": rows,
            "num_layers": len(rows),
            "first_prev_favored_layer": first_prev_favored_layer,
            "first_target_favored_layer": first_target_favored_layer,
            "final_target_minus_previous": final_margin,
            "peak_prev_favored_margin": peak_prev_margin,
            "peak_target_favored_margin": peak_target_margin,
            "prev_favored_detected": first_prev_favored_layer is not None,
            "target_recovery_detected": (
                first_prev_favored_layer is not None
                and first_target_favored_layer is not None
                and first_target_favored_layer > first_prev_favored_layer
            ),
        }

    def _attention_summary(
        self,
        attentions: Sequence[torch.Tensor] | None,
        *,
        groups: Mapping[str, Sequence[int]],
    ) -> List[Dict[str, Any]]:
        if not attentions:
            return []
        out: List[Dict[str, Any]] = []
        for layer_idx, layer_attn in enumerate(attentions):
            if not isinstance(layer_attn, torch.Tensor):
                continue
            layer_values = layer_attn[0, :, -1, :].detach().float().cpu()
            head_rows: List[Dict[str, Any]] = []
            for head_idx in range(int(layer_values.shape[0])):
                head = layer_values[head_idx]
                seq_len = int(head.shape[0])
                group_masses = {
                    name: float(head[[idx for idx in indices if 0 <= int(idx) < seq_len]].sum().item())
                    if indices
                    else 0.0
                    for name, indices in groups.items()
                }
                top_vals, top_idxs = torch.topk(head, k=min(3, int(head.shape[0])))
                head_rows.append(
                    {
                        "head_idx": int(head_idx),
                        "group_masses": group_masses,
                        "top_positions": [int(v) for v in top_idxs.tolist()],
                        "top_attention_values": [float(v) for v in top_vals.tolist()],
                    }
                )
            out.append({"layer_idx": int(layer_idx), "heads": head_rows})
        return out

    def _decoder_layers(self) -> Sequence[torch.nn.Module]:
        language_model = getattr(self.model, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if isinstance(layers, (list, torch.nn.ModuleList)):
            return list(layers)
        return []

    def _intervention_specs(self, cfg: StudyConfig) -> List[Dict[str, Any]]:
        if not cfg.probe.enable_interventions:
            return []
        return [
            {
                "intervention_id": "late_layer_visual_bias",
                "mode": "visual_bias",
                "scale": float(cfg.probe.visual_bias_scale),
            },
            {
                "intervention_id": "late_layer_prior_history_attenuation",
                "mode": "history_attenuation",
                "scale": float(cfg.probe.history_attenuation_scale),
            },
        ]

    def _register_intervention_hooks(
        self,
        *,
        groups: Mapping[str, Sequence[int]],
        phase: Optional[str],
        cfg: StudyConfig,
        intervention: Mapping[str, Any],
    ) -> List[Any]:
        if cfg.probe.intervention_coord_only and (
            phase is None or not str(phase).startswith("coord_")
        ):
            return []
        positions: List[int] = []
        mode = str(intervention.get("mode") or "")
        if mode == "visual_bias":
            positions = [int(v) for v in groups.get("visual_tokens") or []]
        elif mode == "history_attenuation":
            for key in ("source_desc_tokens", "source_coord_tokens", "generated_history"):
                positions.extend(int(v) for v in (groups.get(key) or []))
        valid_positions = sorted({int(v) for v in positions if int(v) >= 0})
        if not valid_positions:
            return []
        decoder_layers = self._decoder_layers()
        if not decoder_layers:
            return []
        start_idx = max(0, len(decoder_layers) - int(cfg.probe.late_layer_count))
        hooks: List[Any] = []
        scale = float(intervention.get("scale") or 1.0)

        def _pre_hook(
            _module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Tuple[Tuple[Any, ...], Mapping[str, Any]] | None:
            if not args:
                return None
            hidden_states = args[0]
            if not isinstance(hidden_states, torch.Tensor):
                return None
            seq_len = int(hidden_states.shape[1])
            active = [idx for idx in valid_positions if idx < seq_len]
            if not active:
                return None
            modified = hidden_states.clone()
            modified[:, active, :] = modified[:, active, :] * scale
            new_args = (modified, *args[1:])
            return new_args, dict(kwargs)

        for layer in decoder_layers[start_idx:]:
            hooks.append(layer.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        return hooks

    def _forward_step(
        self,
        *,
        prompt_inputs: Mapping[str, torch.Tensor],
        generated_ids: Sequence[int],
        step_idx: int,
        groups: Mapping[str, Sequence[int]],
        phase: Optional[str],
        cfg: StudyConfig,
        intervention: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        hooks: List[Any] = []
        if intervention is not None:
            hooks = self._register_intervention_hooks(
                groups=groups,
                phase=phase,
                cfg=cfg,
                intervention=intervention,
            )
        try:
            prompt_ids = prompt_inputs["input_ids"]
            prompt_attention = prompt_inputs.get("attention_mask")
            generated_ids = [int(v) for v in generated_ids]
            if step_idx == 0:
                return self.model(
                    **prompt_inputs,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            prefill_ids = generated_ids[: step_idx - 1]
            prefill_inputs = dict(prompt_inputs)
            if prefill_ids:
                prefill_tensor = torch.tensor(
                    [prefill_ids],
                    dtype=prompt_ids.dtype,
                    device=self.device,
                )
                prefill_inputs["input_ids"] = torch.cat([prompt_ids, prefill_tensor], dim=1)
                if isinstance(prompt_attention, torch.Tensor):
                    prefill_mask = torch.ones(
                        (1, len(prefill_ids)),
                        dtype=prompt_attention.dtype,
                        device=self.device,
                    )
                    prefill_inputs["attention_mask"] = torch.cat(
                        [prompt_attention, prefill_mask],
                        dim=1,
                    )
            with torch.inference_mode():
                prefill_outputs = self.model(
                    **prefill_inputs,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            full_prefix_ids = torch.tensor(
                [generated_ids[:step_idx]],
                dtype=prompt_ids.dtype,
                device=self.device,
            )
            full_input_ids = torch.cat([prompt_ids, full_prefix_ids], dim=1)
            full_attention_mask = prompt_attention
            if isinstance(prompt_attention, torch.Tensor):
                prefix_mask = torch.ones(
                    (1, step_idx),
                    dtype=prompt_attention.dtype,
                    device=self.device,
                )
                full_attention_mask = torch.cat([prompt_attention, prefix_mask], dim=1)
            cache_position = torch.tensor(
                [int(prefill_inputs["input_ids"].shape[1])],
                dtype=torch.long,
                device=self.device,
            )
            step_inputs = self.model.prepare_inputs_for_generation(
                input_ids=full_input_ids,
                past_key_values=prefill_outputs.past_key_values,
                attention_mask=full_attention_mask,
                cache_position=cache_position,
                use_cache=True,
                pixel_values=prompt_inputs.get("pixel_values"),
                pixel_values_videos=prompt_inputs.get("pixel_values_videos"),
                image_grid_thw=prompt_inputs.get("image_grid_thw"),
                video_grid_thw=prompt_inputs.get("video_grid_thw"),
            )
            return self.model(
                **step_inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            for hook in hooks:
                hook.remove()

    def _intervention_behavior_label(
        self,
        *,
        phase: Optional[str],
        baseline_top_token_id: int,
        intervention_top_token_id: int,
        target_token_id: Optional[int],
        previous_token_id: Optional[int],
    ) -> str:
        if int(intervention_top_token_id) == int(baseline_top_token_id):
            return "unchanged"
        if target_token_id is not None and int(intervention_top_token_id) == int(target_token_id):
            return "shifted_to_target"
        if previous_token_id is not None and int(intervention_top_token_id) == int(previous_token_id):
            return "shifted_to_previous_copy"
        token_text = self.tokenizer.decode(
            [int(intervention_top_token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if phase in {"continue_or_open", "close"} or token_text.strip() in {"]", "}", "]}", ","}:
            return "suppression_like"
        return "shifted_other"

    def _probe_interventions_for_step(
        self,
        *,
        prompt_inputs: Mapping[str, torch.Tensor],
        generated_ids: Sequence[int],
        history_ids: Sequence[int],
        step_idx: int,
        groups: Mapping[str, Sequence[int]],
        phase: Optional[str],
        cfg: StudyConfig,
        baseline_processed_scores: torch.Tensor,
        baseline_group_summary: Mapping[str, Any],
        target_token_id: int,
        previous_token_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        if cfg.probe.intervention_coord_only and (
            phase is None or not str(phase).startswith("coord_")
        ):
            return []
        rows: List[Dict[str, Any]] = []
        baseline_top_token_id = int(torch.argmax(baseline_processed_scores).detach().cpu().item())
        for intervention in self._intervention_specs(cfg):
            current_outputs = self._forward_step(
                prompt_inputs=prompt_inputs,
                generated_ids=generated_ids,
                step_idx=step_idx,
                groups=groups,
                phase=phase,
                cfg=cfg,
                intervention=intervention,
            )
            raw_logits = current_outputs.logits[0, -1].detach().float()
            processed_scores = self._apply_decode_processors(
                raw_logits=raw_logits,
                history_ids=history_ids,
                cfg=cfg,
            )
            attn_summary = self._attention_summary(
                getattr(current_outputs, "attentions", None),
                groups=groups,
            )
            layer_group_summary = _layer_group_mass_summary(attn_summary)
            intervention_top_token_id = int(
                torch.argmax(processed_scores).detach().cpu().item()
            )
            coord_summary = {}
            if phase in {"coord_x1", "coord_y1", "coord_x2", "coord_y2"}:
                coord_summary = self._coord_distribution_summary(
                    logits=processed_scores,
                    target_token_id=target_token_id,
                    previous_token_id=previous_token_id,
                    neighbor_radius=int(cfg.controls.coord_neighbor_radius),
                )
            baseline_overwrite = (
                baseline_group_summary.get("overwrite_summary") or {}
            )
            intervention_overwrite = (
                layer_group_summary.get("overwrite_summary") or {}
            )
            target_prob = coord_summary.get("target_bin_prob")
            previous_prob = coord_summary.get("previous_box_prob")
            rows.append(
                {
                    "intervention_id": str(intervention.get("intervention_id")),
                    "mode": str(intervention.get("mode")),
                    "scale": float(intervention.get("scale") or 1.0),
                    "phase_scope": (
                        "coord_only" if cfg.probe.intervention_coord_only else "all"
                    ),
                    "processed_top_tokens": self._top_tokens(
                        processed_scores,
                        k=int(cfg.probe.token_top_k),
                    ),
                    "layer_group_mass_summary": layer_group_summary,
                    "coord_summary": coord_summary,
                    "signal_deltas": {
                        "final_history_minus_visual_delta": (
                            float(intervention_overwrite.get("final_history_minus_visual") or 0.0)
                            - float(baseline_overwrite.get("final_history_minus_visual") or 0.0)
                        ),
                        "final_prior_coord_minus_visual_delta": (
                            float(intervention_overwrite.get("final_prior_coord_minus_visual") or 0.0)
                            - float(baseline_overwrite.get("final_prior_coord_minus_visual") or 0.0)
                        ),
                        "visual_drop_from_peak_to_final_delta": (
                            float(intervention_overwrite.get("visual_drop_from_peak_to_final") or 0.0)
                            - float(baseline_overwrite.get("visual_drop_from_peak_to_final") or 0.0)
                        ),
                        "target_minus_previous_prob": (
                            None
                            if target_prob is None or previous_prob is None
                            else float(target_prob - previous_prob)
                        ),
                    },
                    "behavioral_outcome": self._intervention_behavior_label(
                        phase=phase,
                        baseline_top_token_id=baseline_top_token_id,
                        intervention_top_token_id=intervention_top_token_id,
                        target_token_id=target_token_id,
                        previous_token_id=previous_token_id,
                    ),
                    "intervention_top_token_id": intervention_top_token_id,
                    "intervention_top_token_text": self.tokenizer.decode(
                        [intervention_top_token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )
        return rows

    def probe_case(
        self,
        *,
        image_path: str,
        generation_ids: Sequence[int],
        generation_token_text: Sequence[str],
        onset: Mapping[str, Any],
        cfg: StudyConfig,
    ) -> Dict[str, Any]:
        with _load_image_for_case(image_path) as image:
            prompt_inputs = self._prompt_inputs(image)
            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            visual_positions = self._visual_token_positions(prompt_inputs["input_ids"])
            onset_desc = [int(v) for v in (onset.get("desc_span_token_indices") or [])]
            onset_coord = [int(v) for v in (onset.get("matched_token_indices") or [])]
            source_desc = [int(v) for v in (onset.get("source_desc_span_token_indices") or [])]
            source_coord = [int(v) for v in (onset.get("source_matched_token_indices") or [])]
            interesting = []
            if onset_desc:
                interesting.append(max(0, onset_desc[0] - 1))
                interesting.extend(onset_desc)
            interesting.extend(onset_coord)
            if onset_coord:
                interesting.append(min(len(generation_ids) - 1, onset_coord[-1] + 1))
            interesting = sorted({int(v) for v in interesting if 0 <= int(v) < len(generation_ids)})
            if not interesting:
                return {
                    "probe_surface_status": {
                        "raw_logits": "unavailable",
                        "processed_scores": "unavailable",
                        "llm_attentions": "unavailable",
                        "llm_hidden_states": "unavailable",
                        "llm_to_visual_attention": "unavailable",
                        "native_vision": "unavailable",
                    },
                    "dependency_roots": self._dependency_roots(),
                    "step_rows": [],
                }
            start_step = max(0, min(interesting) - int(cfg.probe.step_window_before))
            end_step = min(
                len(generation_ids) - 1,
                max(interesting) + int(cfg.probe.step_window_after),
            )
            surface_status = {
                "raw_logits": "available",
                "processed_scores": "available",
                "llm_attentions": "available",
                "llm_hidden_states": "available",
                "llm_to_visual_attention": "available" if visual_positions else "missing_visual_tokens",
                "native_vision": "unavailable",
            }
            vision_capture: Dict[str, Dict[str, Any]] = {}
            hooks: List[Any] = []
            visual_module = getattr(self.model, "visual", None)
            if cfg.probe.capture_native_vision and visual_module is not None:
                def _make_hook(tag: str):
                    def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                        vision_capture[tag] = _tensor_summary(output)
                    return _hook
                hooks.append(visual_module.register_forward_hook(_make_hook("visual_root")))
                blocks = getattr(visual_module, "blocks", None)
                if isinstance(blocks, (list, torch.nn.ModuleList)) and len(blocks) > 0:
                    hooks.append(blocks[0].register_forward_hook(_make_hook("visual_block_0")))
                    hooks.append(
                        blocks[-1].register_forward_hook(
                            _make_hook(f"visual_block_{len(blocks) - 1}")
                        )
                    )
            try:
                generated_ids = [int(v) for v in generation_ids]
                previous_final: Optional[torch.Tensor] = None
                step_rows: List[Dict[str, Any]] = []
                intervention_count = 0

                step_to_phase: Dict[int, str] = {}
                if onset_desc:
                    step_to_phase[max(0, onset_desc[0] - 1)] = "continue_or_open"
                    for idx, pos in enumerate(onset_desc):
                        step_to_phase[int(pos)] = f"desc[{idx}]"
                for label, pos in zip(("coord_x1", "coord_y1", "coord_x2", "coord_y2"), onset_coord):
                    step_to_phase[int(pos)] = label
                if onset_coord:
                    step_to_phase[min(len(generated_ids) - 1, onset_coord[-1] + 1)] = "close"

                for step_idx in range(start_step, end_step + 1):
                    prompt_ids = prompt_inputs["input_ids"]
                    prefix_ids = generated_ids[:step_idx]
                    history_ids = list(prompt_ids[0].detach().cpu().tolist())
                    groups = {
                        "visual_tokens": list(visual_positions),
                        "source_desc_tokens": [prompt_len + int(v) for v in source_desc],
                        "source_coord_tokens": [prompt_len + int(v) for v in source_coord],
                        "generated_history": list(range(prompt_len, prompt_len + step_idx)),
                    }
                    current_phase = step_to_phase.get(step_idx)
                    current_outputs = self._forward_step(
                        prompt_inputs=prompt_inputs,
                        generated_ids=generated_ids,
                        step_idx=step_idx,
                        groups=groups,
                        phase=current_phase,
                        cfg=cfg,
                    )
                    if step_idx > 0:
                        history_ids.extend(prefix_ids)
                    raw_logits = current_outputs.logits[0, -1].detach().float()
                    processed_scores = self._apply_decode_processors(
                        raw_logits=raw_logits,
                        history_ids=history_ids,
                        cfg=cfg,
                    )
                    actual_token_id = int(generated_ids[step_idx])
                    actual_token_text = (
                        generation_token_text[step_idx]
                        if step_idx < len(generation_token_text)
                        else self.tokenizer.decode(
                            [actual_token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                    attn_summary = self._attention_summary(
                        getattr(current_outputs, "attentions", None),
                        groups=groups,
                    )
                    layer_group_summary = _layer_group_mass_summary(attn_summary)
                    hidden_summary, previous_final, cross_step = self._hidden_summary(
                        getattr(current_outputs, "hidden_states", None),
                        previous_final=previous_final,
                    )
                    coord_summary = {}
                    previous_token_id = None
                    if current_phase in {"coord_x1", "coord_y1", "coord_x2", "coord_y2"}:
                        coord_slot = ("coord_x1", "coord_y1", "coord_x2", "coord_y2").index(current_phase)
                        previous_coord_positions = [int(v) for v in source_coord]
                        if coord_slot < len(previous_coord_positions):
                            source_pos = previous_coord_positions[coord_slot]
                            if 0 <= source_pos < len(generated_ids):
                                previous_token_id = int(generated_ids[source_pos])
                        coord_summary = self._coord_distribution_summary(
                            logits=processed_scores,
                            target_token_id=actual_token_id,
                            previous_token_id=previous_token_id,
                            neighbor_radius=int(cfg.controls.coord_neighbor_radius),
                        )
                    layer_logit_lens_summary = self._layer_logit_lens_summary(
                        getattr(current_outputs, "hidden_states", None),
                        target_token_id=actual_token_id,
                        previous_token_id=previous_token_id,
                        phase=current_phase,
                    )
                    intervention_rows: List[Dict[str, Any]] = []
                    if intervention_count < int(cfg.probe.intervention_max_cases):
                        intervention_rows = self._probe_interventions_for_step(
                            prompt_inputs=prompt_inputs,
                            generated_ids=generated_ids,
                            history_ids=history_ids,
                            step_idx=step_idx,
                            groups=groups,
                            phase=current_phase,
                            cfg=cfg,
                            baseline_processed_scores=processed_scores,
                            baseline_group_summary=layer_group_summary,
                            target_token_id=actual_token_id,
                            previous_token_id=previous_token_id,
                        )
                    if intervention_rows:
                        intervention_count += 1
                    step_rows.append(
                        {
                            "generated_step_idx": int(step_idx),
                            "phase": current_phase,
                            "actual_token_id": int(actual_token_id),
                            "actual_token_text": str(actual_token_text),
                            "actual_raw_logprob": float(
                                torch.log_softmax(raw_logits, dim=-1)[actual_token_id]
                                .detach()
                                .cpu()
                                .item()
                            ),
                            "actual_processed_logprob": float(
                                torch.log_softmax(processed_scores, dim=-1)[actual_token_id]
                                .detach()
                                .cpu()
                                .item()
                            ),
                            "raw_top_tokens": self._top_tokens(
                                raw_logits,
                                k=int(cfg.probe.token_top_k),
                            ),
                            "processed_top_tokens": self._top_tokens(
                                processed_scores,
                                k=int(cfg.probe.token_top_k),
                            ),
                            "coord_summary": coord_summary,
                            "attention_summary": attn_summary,
                            "layer_group_mass_summary": layer_group_summary,
                            "hidden_summary": hidden_summary,
                            "layer_logit_lens_summary": layer_logit_lens_summary,
                            "cross_step_summary": cross_step,
                            "interventions": intervention_rows,
                        }
                    )
                if vision_capture:
                    surface_status["native_vision"] = "available"
                return {
                    "probe_surface_status": surface_status,
                    "dependency_roots": self._dependency_roots(),
                    "visual_hook_summary": vision_capture,
                    "visual_token_positions": visual_positions,
                    "step_rows": step_rows,
                }
            except RuntimeError as exc:
                return {
                    "probe_surface_status": {
                        **surface_status,
                        "llm_attentions": f"failed:{type(exc).__name__}",
                        "llm_hidden_states": f"failed:{type(exc).__name__}",
                        "llm_to_visual_attention": f"failed:{type(exc).__name__}",
                        "native_vision": (
                            "available" if vision_capture else f"failed:{type(exc).__name__}"
                        ),
                    },
                    "dependency_roots": self._dependency_roots(),
                    "visual_hook_summary": vision_capture,
                    "step_rows": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                for hook in hooks:
                    hook.remove()


def _pixel_to_norm1000(
    bbox_xyxy: Tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> List[int]:
    x1, y1, x2, y2 = bbox_xyxy
    denom_w = max(float(width), 1.0)
    denom_h = max(float(height), 1.0)
    return [
        int(max(0, min(999, round((float(x1) / denom_w) * 1000.0)))),
        int(max(0, min(999, round((float(y1) / denom_h) * 1000.0)))),
        int(max(0, min(999, round((float(x2) / denom_w) * 1000.0)))),
        int(max(0, min(999, round((float(y2) / denom_h) * 1000.0)))),
    ]


def _object_entry_text(
    obj: Mapping[str, Any],
    *,
    width: int,
    height: int,
    object_field_order: str,
) -> str:
    bbox = _bbox_xyxy(obj)
    if bbox is None:
        raise ValueError("Candidate object is missing bbox geometry")
    bins = _pixel_to_norm1000(bbox, width=width, height=height)
    payload = build_object_payload(
        desc=str(obj.get("desc") or ""),
        geometry_key="bbox_2d",
        geometry_value=[int_to_token(int(v)) for v in bins],
        object_field_order=object_field_order,
    )
    return dumps_coordjson(payload)


def _append_ready_prefix(
    prefix_objects: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    object_field_order: str,
) -> str:
    prefix_text = dumps_coordjson({"objects": []})
    if not prefix_text.endswith("]}"):
        raise ValueError("Unexpected canonical CoordJSON rendering")
    prefix_text = prefix_text[:-2]
    for obj in prefix_objects:
        if not prefix_text.endswith("["):
            prefix_text += ", "
        prefix_text += _object_entry_text(
            obj,
            width=width,
            height=height,
            object_field_order=object_field_order,
        )
    if prefix_objects and not prefix_text.endswith(", "):
        prefix_text += ", "
    return prefix_text


def _build_candidate_assistant_text(
    prefix_objects: Sequence[Mapping[str, Any]],
    candidate_object: Optional[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    object_field_order: str,
) -> str:
    prefix_text = _append_ready_prefix(
        prefix_objects,
        width=width,
        height=height,
        object_field_order=object_field_order,
    )
    continuation_text = (
        _object_entry_text(
            candidate_object,
            width=width,
            height=height,
            object_field_order=object_field_order,
        )
        if candidate_object is not None
        else ""
    )
    return _close_prefix_rollout_text(
        prefix_text,
        continuation_text,
        object_field_order=object_field_order,
    )


def _assistant_relative_positions(
    *,
    tokenizer: Any,
    assistant_text: str,
    desc: str,
    bbox_norm1000: Optional[Sequence[int]],
) -> Dict[str, List[int]]:
    assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    desc_ids = tokenizer.encode(str(desc), add_special_tokens=False)
    desc_start = _find_subsequence(assistant_ids, desc_ids, start_hint=0)
    if desc_start is None:
        raise ValueError("Could not locate desc span inside candidate assistant text")
    desc_positions = [int(desc_start + idx) for idx in range(len(desc_ids))]
    coord_positions: List[int] = []
    if bbox_norm1000 is not None:
        search_start = int(desc_start + len(desc_ids))
        coord_token_ids = tokenizer.convert_tokens_to_ids(
            [int_to_token(int(v)) for v in bbox_norm1000]
        )
        for coord_token_id in coord_token_ids:
            found = None
            for pos in range(search_start, len(assistant_ids)):
                if int(assistant_ids[pos]) == int(coord_token_id):
                    found = pos
                    break
            if found is None:
                raise ValueError("Could not locate candidate coord token inside assistant text")
            coord_positions.append(int(found))
            search_start = int(found + 1)
    return {
        "assistant_ids": [int(v) for v in assistant_ids],
        "desc_positions_rel": desc_positions,
        "coord_positions_rel": coord_positions,
    }


def _teacher_forced_candidate_summary(
    *,
    scorer: TeacherForcedScorer,
    image: Image.Image,
    assistant_text: str,
    desc: str,
    bbox_norm1000: Optional[Sequence[int]],
    previous_bbox_norm1000: Optional[Sequence[int]],
    prompt_variant: str,
    object_field_order: str,
    neighbor_radius: int,
) -> Dict[str, Any]:
    rel_positions = _assistant_relative_positions(
        tokenizer=scorer.tokenizer,
        assistant_text=assistant_text,
        desc=desc,
        bbox_norm1000=bbox_norm1000,
    )
    prepared = scorer.prepare_example(
        image=image,
        assistant_text=assistant_text,
        desc_positions_rel=rel_positions["desc_positions_rel"],
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
    )
    assistant_start = _find_subsequence(
        prepared.full_input_ids,
        rel_positions["assistant_ids"],
        start_hint=max(0, len(prepared.full_input_ids) - len(rel_positions["assistant_ids"]) - 32),
    )
    if assistant_start is None:
        raise ValueError("assistant_span_build_failed")
    coord_positions = [
        int(assistant_start + pos) for pos in rel_positions["coord_positions_rel"]
    ]
    model_inputs = scorer.processor(
        text=[prepared.full_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {
        key: value.to(scorer.device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }
    with torch.inference_mode():
        outputs = scorer.model(**model_inputs, use_cache=False)
    logits = getattr(outputs, "logits", None)
    input_ids = model_inputs.get("input_ids")
    if not isinstance(logits, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("teacher-forced candidate summary requires logits + input_ids")
    if logits.shape[:2] != input_ids.shape[:2]:
        raise RuntimeError("teacher-forced candidate summary expects unsliced logits")
    desc_logprobs = []
    for pos in prepared.desc_positions:
        prev_logits = logits[0, int(pos) - 1].detach().float()
        target_id = int(input_ids[0, int(pos)].item())
        desc_logprobs.append(
            float(torch.log_softmax(prev_logits, dim=-1)[target_id].detach().cpu().item())
        )
    coord_rows = []
    for slot_idx, pos in enumerate(coord_positions):
        prev_logits = logits[0, int(pos) - 1].detach().float()
        target_token_id = int(input_ids[0, int(pos)].item())
        coord_probs = prev_logits.index_select(
            dim=0,
            index=torch.tensor(
                get_coord_token_ids(scorer.tokenizer, validate=True),
                dtype=torch.long,
                device=prev_logits.device,
            ),
        )
        coord_probs = torch.softmax(coord_probs.float(), dim=-1)
        top1_idx = int(torch.argmax(coord_probs).detach().cpu().item())
        expected_bin = float(
            torch.sum(
                coord_probs
                * torch.arange(
                    _COORD_TOKEN_COUNT,
                    dtype=coord_probs.dtype,
                    device=coord_probs.device,
                )
            )
            .detach()
            .cpu()
            .item()
        )
        target_bin = bbox_norm1000[slot_idx] if bbox_norm1000 is not None else None
        prev_bin = (
            previous_bbox_norm1000[slot_idx]
            if previous_bbox_norm1000 is not None and slot_idx < len(previous_bbox_norm1000)
            else None
        )
        row = {
            "slot": ("x1", "y1", "x2", "y2")[slot_idx],
            "entropy": float(
                (-torch.sum(coord_probs * torch.log(coord_probs.clamp_min(1e-12))))
                .detach()
                .cpu()
                .item()
            ),
            "top1_prob": float(coord_probs[top1_idx].detach().cpu().item()),
            "argmax_bin": int(top1_idx),
            "expected_bin": float(expected_bin),
            "target_bin_prob": (
                float(coord_probs[int(target_bin)].detach().cpu().item())
                if target_bin is not None
                else None
            ),
            "previous_box_prob": (
                float(coord_probs[int(prev_bin)].detach().cpu().item())
                if prev_bin is not None
                else None
            ),
            "previous_box_neighborhood_mass": (
                float(
                    coord_probs[
                        max(0, int(prev_bin) - int(neighbor_radius)) : min(
                            _COORD_TOKEN_COUNT,
                            int(prev_bin) + int(neighbor_radius) + 1,
                        )
                    ]
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
                if prev_bin is not None
                else None
            ),
            "actual_token_id": int(target_token_id),
        }
        coord_rows.append(row)
    return {
        "full_score": float(mean(desc_logprobs + [
            float(torch.log_softmax(logits[0, int(pos) - 1].detach().float(), dim=-1)[
                int(input_ids[0, int(pos)].item())
            ].detach().cpu().item())
            for pos in coord_positions
        ])) if (desc_logprobs or coord_positions) else float("nan"),
        "desc_score": float(mean(desc_logprobs)) if desc_logprobs else None,
        "coord_slot_rows": coord_rows,
        "coord_entropy_mean": _maybe_mean([row["entropy"] for row in coord_rows]),
        "coord_top1_prob_mean": _maybe_mean([row["top1_prob"] for row in coord_rows]),
        "coord_target_bin_prob_mean": _maybe_mean([
            row["target_bin_prob"] for row in coord_rows if row["target_bin_prob"] is not None
        ]),
        "coord_previous_box_prob_mean": _maybe_mean([
            row["previous_box_prob"] for row in coord_rows if row["previous_box_prob"] is not None
        ]),
        "coord_previous_box_neighborhood_mass_mean": _maybe_mean([
            row["previous_box_neighborhood_mass"]
            for row in coord_rows
            if row["previous_box_neighborhood_mass"] is not None
        ]),
    }


def _greedy_prefix_gt_matches(
    prefix_objects: Sequence[Mapping[str, Any]],
    gt_objects: Sequence[Mapping[str, Any]],
    *,
    same_desc_iou_threshold: float,
) -> Dict[int, int]:
    used_gt: set[int] = set()
    mapping: Dict[int, int] = {}
    for pred_idx, pred in enumerate(prefix_objects):
        pred_box = _bbox_xyxy(pred)
        pred_desc = normalize_desc(str(pred.get("desc") or ""))
        if pred_box is None or not pred_desc:
            continue
        best_gt_idx = None
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_objects):
            if gt_idx in used_gt:
                continue
            if normalize_desc(str(gt.get("desc") or "")) != pred_desc:
                continue
            gt_box = _bbox_xyxy(gt)
            if gt_box is None:
                continue
            iou = _bbox_iou(pred_box, gt_box)
            if iou >= float(same_desc_iou_threshold) and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx is not None:
            mapping[pred_idx] = int(best_gt_idx)
            used_gt.add(int(best_gt_idx))
    return mapping


def _clone_object_with_bbox(
    obj: Mapping[str, Any],
    bbox_xyxy: Sequence[float],
) -> Dict[str, Any]:
    clone = dict(obj)
    clone["points"] = [int(round(v)) for v in bbox_xyxy]
    clone["type"] = str(clone.get("type") or "bbox_2d")
    return clone


def _coord_split_candidate(
    *,
    base_object: Mapping[str, Any],
    alt_object: Mapping[str, Any],
    alt_slots: Sequence[int],
) -> Optional[Dict[str, Any]]:
    base_box = _bbox_xyxy(base_object)
    alt_box = _bbox_xyxy(alt_object)
    if base_box is None or alt_box is None:
        return None
    merged = list(base_box)
    for slot_idx in alt_slots:
        idx = int(slot_idx)
        if 0 <= idx < 4:
            merged[idx] = alt_box[idx]
    candidate = _clone_object_with_bbox(base_object, merged)
    if alt_object.get("desc"):
        candidate["desc"] = str(alt_object.get("desc"))
    return candidate


def _interpolate_object_bbox(
    *,
    source_object: Mapping[str, Any],
    target_object: Mapping[str, Any],
    alpha: float,
) -> Optional[Dict[str, Any]]:
    source_box = _bbox_xyxy(source_object)
    target_box = _bbox_xyxy(target_object)
    if source_box is None or target_box is None:
        return None
    mixed = [
        (1.0 - float(alpha)) * float(src) + float(alpha) * float(dst)
        for src, dst in zip(source_box, target_box)
    ]
    candidate = _clone_object_with_bbox(source_object, mixed)
    if target_object.get("desc"):
        candidate["desc"] = str(target_object.get("desc"))
    return candidate


def _choose_gt_next_candidate(
    *,
    prefix_objects: Sequence[Mapping[str, Any]],
    gt_objects: Sequence[Mapping[str, Any]],
    source_object: Mapping[str, Any],
    cfg: StudyConfig,
) -> Optional[Dict[str, Any]]:
    source_desc = normalize_desc(str(source_object.get("desc") or ""))
    source_box = _bbox_xyxy(source_object)
    if not source_desc or source_box is None:
        return None
    prefix_gt_matches = _greedy_prefix_gt_matches(
        prefix_objects,
        gt_objects,
        same_desc_iou_threshold=float(cfg.controls.same_desc_iou_threshold),
    )
    used_gt = set(prefix_gt_matches.values())
    candidates = []
    for gt_idx, gt_obj in enumerate(gt_objects):
        if gt_idx in used_gt:
            continue
        if normalize_desc(str(gt_obj.get("desc") or "")) != source_desc:
            continue
        gt_box = _bbox_xyxy(gt_obj)
        if gt_box is None:
            continue
        candidates.append(
            (
                _bbox_iou(source_box, gt_box),
                -_center_distance(source_box, gt_box),
                -gt_idx,
                dict(gt_obj),
            )
        )
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return dict(candidates[0][-1])


def _run_control_comparisons(
    resolved_checkpoints: Sequence[ResolvedStudyCheckpoint],
    *,
    cfg: StudyConfig,
    run_dir: Path,
    selected_cases: Sequence[Mapping[str, Any]],
    reproduction: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not selected_cases:
        return {"case_rows": []}
    checkpoint_lookup = {item.spec.alias: item for item in resolved_checkpoints}
    reproduce_lookup = {row["checkpoint_alias"]: row for row in reproduction.get("reproductions", [])}
    probe_lookup = {row["case_id"]: row for row in probe_rows}
    out_rows: List[Dict[str, Any]] = []
    for case in selected_cases[: int(cfg.controls.max_cases)]:
        case_id = str(case["case_id"])
        checkpoint_alias = str(case["checkpoint_alias"])
        reproduce_row = reproduce_lookup.get(checkpoint_alias)
        probe_row = probe_lookup.get(case_id)
        if reproduce_row is None or probe_row is None:
            continue
        gt_rows = _read_jsonl(Path(str(reproduce_row["gt_vs_pred_jsonl"])))
        conf_rows = _load_confidence_index(Path(str(reproduce_row["pred_confidence_jsonl"])))
        if int(case.get("local_line_idx", 0)) >= len(gt_rows):
            continue
        local_idx = int(case["local_line_idx"])
        reproduce_gt = gt_rows[local_idx]
        reproduce_conf = conf_rows.get(local_idx)
        onset = (
            probe_row.get("active_onset")
            or probe_row.get("reproduced_onset")
            or probe_row.get("historical_onset")
            or {}
        )
        object_idx = onset.get("object_idx")
        source_idx = onset.get("source_object_idx")
        if object_idx is None or source_idx is None:
            continue
        preds = list(reproduce_gt.get("pred") or [])
        gts = list(reproduce_gt.get("gt") or [])
        if not (0 <= int(object_idx) <= len(preds)) or not (0 <= int(source_idx) < len(preds)):
            continue
        prefix_objects = preds[: int(object_idx)]
        source_object = preds[int(source_idx)]
        predicted_object = (
            dict(preds[int(object_idx)])
            if 0 <= int(object_idx) < len(preds)
            else None
        )
        gt_next = _choose_gt_next_candidate(
            prefix_objects=prefix_objects,
            gt_objects=gts,
            source_object=source_object,
            cfg=cfg,
        )
        exact_duplicate = dict(source_object)
        checkpoint = checkpoint_lookup[checkpoint_alias]
        scorer = TeacherForcedScorer(
            checkpoint_path=checkpoint.resolved.path,
            device=cfg.execution.device,
            attn_implementation=cfg.execution.teacher_forced_attn_implementation,
        )
        image = _load_image_for_case(str(reproduce_gt.get("image") or ""))
        try:
            width = int(reproduce_gt.get("width") or 0)
            height = int(reproduce_gt.get("height") or 0)
            previous_bbox_norm1000 = None
            source_box = _bbox_xyxy(source_object)
            if source_box is not None:
                previous_bbox_norm1000 = _pixel_to_norm1000(
                    source_box,
                    width=width,
                    height=height,
                )
            case_row: Dict[str, Any] = {
                "case_id": case_id,
                "checkpoint_alias": checkpoint_alias,
                "fallback_control": None,
            }

            def _candidate_summary(
                candidate_object: Optional[Mapping[str, Any]],
            ) -> Optional[Dict[str, Any]]:
                if candidate_object is None:
                    return None
                candidate_box = _bbox_xyxy(candidate_object)
                candidate_bins = (
                    _pixel_to_norm1000(candidate_box, width=width, height=height)
                    if candidate_box is not None
                    else None
                )
                candidate_text = _build_candidate_assistant_text(
                    prefix_objects,
                    candidate_object,
                    width=width,
                    height=height,
                    object_field_order=checkpoint.resolved.object_field_order,
                )
                return _teacher_forced_candidate_summary(
                    scorer=scorer,
                    image=image,
                    assistant_text=candidate_text,
                    desc=str(candidate_object.get("desc") or ""),
                    bbox_norm1000=candidate_bins,
                    previous_bbox_norm1000=previous_bbox_norm1000,
                    prompt_variant=checkpoint.resolved.prompt_variant,
                    object_field_order=checkpoint.resolved.object_field_order,
                    neighbor_radius=int(cfg.controls.coord_neighbor_radius),
                )

            if gt_next is not None:
                case_row["gt_next"] = _candidate_summary(gt_next)
            else:
                case_row["fallback_control"] = "missing_same_desc_gt_next"
                case_row["gt_next"] = None
            case_row["exact_duplicate"] = _candidate_summary(exact_duplicate)
            case_row["predicted_object"] = _candidate_summary(predicted_object)
            if gt_next is not None:
                oracle_x1y1 = _coord_split_candidate(
                    base_object=exact_duplicate,
                    alt_object=gt_next,
                    alt_slots=(0, 1),
                )
                oracle_x2y2 = _coord_split_candidate(
                    base_object=exact_duplicate,
                    alt_object=gt_next,
                    alt_slots=(2, 3),
                )
                oracle_interp = _interpolate_object_bbox(
                    source_object=exact_duplicate,
                    target_object=gt_next,
                    alpha=0.5,
                )
                case_row["oracle_x1y1_from_gt_next"] = _candidate_summary(oracle_x1y1)
                case_row["oracle_x2y2_from_gt_next"] = _candidate_summary(oracle_x2y2)
                case_row["oracle_interp_from_gt_next"] = _candidate_summary(oracle_interp)
            else:
                case_row["oracle_x1y1_from_gt_next"] = None
                case_row["oracle_x2y2_from_gt_next"] = None
                case_row["oracle_interp_from_gt_next"] = None
            if cfg.controls.include_close_candidate:
                close_text = _build_candidate_assistant_text(
                    prefix_objects,
                    None,
                    width=width,
                    height=height,
                    object_field_order=checkpoint.resolved.object_field_order,
                )
                case_row["close"] = _teacher_forced_candidate_summary(
                    scorer=scorer,
                    image=image,
                    assistant_text=close_text,
                    desc=str(source_object.get("desc") or ""),
                    bbox_norm1000=None,
                    previous_bbox_norm1000=previous_bbox_norm1000,
                    prompt_variant=checkpoint.resolved.prompt_variant,
                    object_field_order=checkpoint.resolved.object_field_order,
                    neighbor_radius=int(cfg.controls.coord_neighbor_radius),
                )
            else:
                case_row["close"] = None
            gt_score = (
                float(case_row["gt_next"]["full_score"])
                if case_row.get("gt_next") is not None
                else None
            )
            dup_score = float(case_row["exact_duplicate"]["full_score"])
            close_score = (
                float(case_row["close"]["full_score"])
                if case_row.get("close") is not None
                else None
            )
            predicted_score = (
                float(case_row["predicted_object"]["full_score"])
                if case_row.get("predicted_object") is not None
                else None
            )
            oracle_x1y1_score = (
                float(case_row["oracle_x1y1_from_gt_next"]["full_score"])
                if case_row.get("oracle_x1y1_from_gt_next") is not None
                else None
            )
            oracle_x2y2_score = (
                float(case_row["oracle_x2y2_from_gt_next"]["full_score"])
                if case_row.get("oracle_x2y2_from_gt_next") is not None
                else None
            )
            oracle_interp_score = (
                float(case_row["oracle_interp_from_gt_next"]["full_score"])
                if case_row.get("oracle_interp_from_gt_next") is not None
                else None
            )
            case_row["margins"] = {
                "gt_next_minus_duplicate": (
                    None if gt_score is None else float(gt_score - dup_score)
                ),
                "close_minus_duplicate": (
                    None if close_score is None else float(close_score - dup_score)
                ),
                "predicted_minus_duplicate": (
                    None
                    if predicted_score is None
                    else float(predicted_score - dup_score)
                ),
                "oracle_x1y1_minus_duplicate": (
                    None
                    if oracle_x1y1_score is None
                    else float(oracle_x1y1_score - dup_score)
                ),
                "oracle_x2y2_minus_duplicate": (
                    None
                    if oracle_x2y2_score is None
                    else float(oracle_x2y2_score - dup_score)
                ),
                "oracle_interp_minus_duplicate": (
                    None
                    if oracle_interp_score is None
                    else float(oracle_interp_score - dup_score)
                ),
            }
            out_rows.append(case_row)
        finally:
            image.close()
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    compare_dir = run_dir / "compare"
    _write_jsonl(compare_dir / "case_rows.jsonl", out_rows)
    _write_json(compare_dir / "summary.json", {"num_cases": len(out_rows)})
    return {"case_rows": out_rows}


def _classify_case(
    *,
    probe_row: Mapping[str, Any],
    compare_row: Optional[Mapping[str, Any]],
) -> str:
    if not probe_row.get("probe") or not compare_row:
        return "insufficient-evidence"
    margins = compare_row.get("margins") or {}
    gt_margin = margins.get("gt_next_minus_duplicate")
    close_margin = margins.get("close_minus_duplicate")
    onset_source = str(
        ((probe_row.get("reproduced_onset") or {}).get("anchor_source")) or ""
    )
    if onset_source and onset_source != "detected_duplicate":
        if isinstance(gt_margin, (int, float)) and float(gt_margin) > 0.0:
            return "control-stable"
        return "control-risky"
    duplicate_summary = compare_row.get("exact_duplicate") or {}
    gt_summary = compare_row.get("gt_next") or {}
    coord_copy = duplicate_summary.get("coord_previous_box_neighborhood_mass_mean")
    coord_copy_prob = duplicate_summary.get("coord_previous_box_prob_mean")
    gt_target_prob = gt_summary.get("coord_target_bin_prob_mean")
    coord_signal = False
    if isinstance(gt_margin, (int, float)) and float(gt_margin) < 0.0:
        if isinstance(coord_copy, (int, float)) and float(coord_copy) >= 0.3:
            coord_signal = True
        if isinstance(coord_copy_prob, (int, float)) and isinstance(gt_target_prob, (int, float)):
            if float(coord_copy_prob) >= max(0.05, float(gt_target_prob) + 0.05):
                coord_signal = True
    if isinstance(close_margin, (int, float)) and float(close_margin) < 0.0:
        return "mixed" if coord_signal else "continuation-dominant"
    step_rows = list((probe_row.get("probe") or {}).get("step_rows") or [])
    if step_rows:
        last_hidden = [
            (row.get("cross_step_summary") or {}).get("final_hidden_delta_l2")
            for row in step_rows
        ]
        deltas = [float(v) for v in last_hidden if isinstance(v, (int, float))]
        if deltas and max(deltas) > 15.0:
            return "mixed" if coord_signal else "internal-state-dominant"
    if coord_signal:
        return "coordinate-dominant"
    return "mixed"


def _distribution_summary(values: Sequence[Any]) -> Dict[str, Any]:
    clean = sorted(
        float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)
    )
    if not clean:
        return {}
    mid = len(clean) // 2
    return {
        "n": len(clean),
        "mean": float(sum(clean) / len(clean)),
        "min": float(clean[0]),
        "median": float(clean[mid]),
        "max": float(clean[-1]),
        "p25": float(clean[max(0, int(round(0.25 * (len(clean) - 1))))]),
        "p75": float(clean[min(len(clean) - 1, int(round(0.75 * (len(clean) - 1))))]),
    }


def _case_mechanism_signals(
    step_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    coord_rows = [
        row
        for row in step_rows
        if str(row.get("phase") or "").startswith("coord_")
    ]
    anchor_row = coord_rows[0] if coord_rows else (step_rows[0] if step_rows else {})
    overwrite = ((anchor_row.get("layer_group_mass_summary") or {}).get("overwrite_summary")) or {}
    coord_summary = anchor_row.get("coord_summary") or {}
    logit_lens = anchor_row.get("layer_logit_lens_summary") or {}
    return {
        "anchor_phase": anchor_row.get("phase"),
        "overwrite_summary": overwrite,
        "coord_summary": coord_summary,
        "layer_logit_lens_summary": logit_lens,
        "history_overwrite_detected": bool(overwrite.get("history_overwrite_detected")),
        "prior_coord_overwrite_detected": bool(
            overwrite.get("prior_coord_overwrite_detected")
        ),
        "final_history_minus_visual": overwrite.get("final_history_minus_visual"),
        "final_prior_coord_minus_visual": overwrite.get(
            "final_prior_coord_minus_visual"
        ),
        "visual_drop_from_peak_to_final": overwrite.get(
            "visual_drop_from_peak_to_final"
        ),
        "prev_favored_detected": bool(logit_lens.get("prev_favored_detected")),
        "first_prev_favored_layer": logit_lens.get("first_prev_favored_layer"),
        "final_target_minus_previous": logit_lens.get(
            "final_target_minus_previous"
        ),
        "target_recovery_detected": bool(logit_lens.get("target_recovery_detected")),
    }


def _case_intervention_summary(
    step_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    intervention_rows = [
        row
        for step in step_rows
        for row in (step.get("interventions") or [])
    ]
    if not intervention_rows:
        return {"available": False, "by_intervention": {}}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in intervention_rows:
        grouped[str(row.get("intervention_id") or "unknown")].append(row)
    by_intervention: Dict[str, Any] = {}
    for intervention_id, rows in sorted(grouped.items()):
        outcomes = Counter(str(row.get("behavioral_outcome") or "unknown") for row in rows)
        target_minus_previous = [
            ((row.get("signal_deltas") or {}).get("target_minus_previous_prob"))
            for row in rows
        ]
        history_delta = [
            ((row.get("signal_deltas") or {}).get("final_history_minus_visual_delta"))
            for row in rows
        ]
        by_intervention[intervention_id] = {
            "num_records": len(rows),
            "behavioral_outcomes": dict(outcomes),
            "target_minus_previous_prob": _distribution_summary(target_minus_previous),
            "final_history_minus_visual_delta": _distribution_summary(history_delta),
            "suppression_like_present": bool(outcomes.get("suppression_like")),
            "target_shift_present": bool(outcomes.get("shifted_to_target")),
        }
    return {
        "available": True,
        "num_records": len(intervention_rows),
        "by_intervention": by_intervention,
    }


def _build_cohort_summary(
    report_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_checkpoint: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in report_rows:
        by_checkpoint[str(row.get("checkpoint_alias") or "unknown")].append(row)
    checkpoint_rows: Dict[str, Any] = {}
    for alias, rows in sorted(by_checkpoint.items()):
        classifications = Counter(str(row.get("classification") or "unknown") for row in rows)
        checkpoint_rows[alias] = {
            "num_cases": len(rows),
            "classification_counts": dict(classifications),
            "final_history_minus_visual": _distribution_summary(
                [
                    ((row.get("mechanism_signals") or {}).get("final_history_minus_visual"))
                    for row in rows
                ]
            ),
            "final_prior_coord_minus_visual": _distribution_summary(
                [
                    ((row.get("mechanism_signals") or {}).get("final_prior_coord_minus_visual"))
                    for row in rows
                ]
            ),
            "gt_next_minus_duplicate": _distribution_summary(
                [
                    ((row.get("control_margins") or {}).get("gt_next_minus_duplicate"))
                    for row in rows
                ]
            ),
        }
    return {
        "num_checkpoints": len(checkpoint_rows),
        "checkpoint_rows": checkpoint_rows,
    }


def _build_family_comparison_report(
    *,
    inventory_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    case_rows_by_alias: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in report_rows:
        case_rows_by_alias[str(row.get("checkpoint_alias") or "unknown")].append(row)
    out_rows: List[Dict[str, Any]] = []
    for inventory_row in inventory_rows:
        alias = str(inventory_row.get("alias") or "unknown")
        family_role = str(inventory_row.get("family_comparison_role") or "unknown")
        case_rows = case_rows_by_alias.get(alias, [])
        gt_margins = [
            ((row.get("control_margins") or {}).get("gt_next_minus_duplicate"))
            for row in case_rows
        ]
        overwrite_flags = [
            bool((row.get("mechanism_signals") or {}).get("history_overwrite_detected"))
            for row in case_rows
        ]
        coord_prev_mass = [
            ((row.get("mechanism_signals") or {}).get("coord_summary") or {}).get(
                "previous_box_neighborhood_mass"
            )
            for row in case_rows
        ]
        diagnostic_note = "insufficient-case-evidence"
        if family_role == "pure_ce_reference":
            diagnostic_note = "clean-pure-ce-reference"
        elif family_role == "ce_proxy_disabled_continuation":
            diagnostic_note = "ce-proxy-continuation"
        elif family_role == "soft_coordinate_supervised":
            diagnostic_note = "soft-coordinate-supervised"
        if not case_rows:
            if inventory_row.get("probe_readiness") != "ready_to_probe":
                mechanism_readout = (
                    "missing_probe_ready_artifact"
                )
            else:
                mechanism_readout = "no-duplication-case-selected"
        else:
            mean_margin = _distribution_summary(gt_margins).get("mean")
            overwrite_present = any(overwrite_flags)
            if family_role == "soft_coordinate_supervised" and (
                isinstance(mean_margin, (int, float)) and mean_margin < 0.0
            ):
                mechanism_readout = (
                    "soft-supervision-local-copy-basin-with-history-dominant-final-coord"
                    if overwrite_present
                    else "soft-supervision-local-copy-basin"
                )
            elif family_role in {"pure_ce_reference", "ce_proxy_disabled_continuation"}:
                if isinstance(mean_margin, (int, float)) and mean_margin >= 0.0:
                    mechanism_readout = "sharper-coord-discrimination-blocks-copy-basin"
                else:
                    mechanism_readout = "ce-side-family-still-shows-copy-basin-or-mixed-failure"
            else:
                mechanism_readout = "mixed-or-unknown-family-behavior"
        out_rows.append(
            {
                "alias": alias,
                "family_comparison_role": family_role,
                "coord_soft_ce_w1_state": inventory_row.get("coord_soft_ce_w1_state"),
                "parent_checkpoint": inventory_row.get("parent_checkpoint"),
                "probe_readiness": inventory_row.get("probe_readiness"),
                "has_infer_artifact": inventory_row.get("has_infer_artifact"),
                "best_probe_surface": inventory_row.get("best_probe_surface"),
                "diagnostic_note": diagnostic_note,
                "mechanism_readout": mechanism_readout,
                "gt_next_minus_duplicate": _distribution_summary(gt_margins),
                "previous_box_neighborhood_mass": _distribution_summary(coord_prev_mass),
            }
        )
    family_summary = _family_comparison_summary(inventory_rows)
    family_summary["rows"] = out_rows
    return family_summary


def _build_report(
    *,
    inventory: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
    reproduction: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    compare: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    compare_lookup = {row["case_id"]: row for row in compare.get("case_rows", [])}
    selected_lookup = {str(row["case_id"]): row for row in selected_cases}
    reproduction_aliases = {
        str(row["checkpoint_alias"])
        for row in reproduction.get("reproductions", [])
        if row.get("checkpoint_alias") is not None
    }
    report_rows = []
    counts: Counter[str] = Counter()
    for probe_row in probe_rows:
        case_id = str(probe_row["case_id"])
        selected_row = selected_lookup.get(case_id, {})
        compare_row = compare_lookup.get(case_id)
        classification = _classify_case(
            probe_row=probe_row,
            compare_row=compare_row,
        )
        counts[classification] += 1
        probe_payload = probe_row.get("probe") or {}
        step_rows = list(probe_payload.get("step_rows") or [])
        mechanism_signals = _case_mechanism_signals(step_rows)
        intervention_summary = _case_intervention_summary(step_rows)
        evidence_layers = {
            "artifact_audit": bool(selected_row),
            "rollout_reproduction": str(probe_row.get("checkpoint_alias")) in reproduction_aliases,
            "deterministic_reforward": bool(probe_payload),
            "controlled_compare": bool(compare_row),
            "deep_onset_probe": bool(step_rows),
        }
        precursor_signals = {
            "pred_count": selected_row.get("pred_count"),
            "gt_count": selected_row.get("gt_count"),
            "max_desc_count": selected_row.get("max_desc_count"),
            "same_desc_duplicate_pair_count": selected_row.get("same_desc_duplicate_pair_count"),
            "top_desc": selected_row.get("top_desc"),
            "selection_reason": selected_row.get("selection_reason"),
            "historical_onset_field_phase": ((selected_row.get("onset") or {}).get("onset_field_phase")),
            "reproduced_onset_field_phase": ((probe_row.get("reproduced_onset") or {}).get("onset_field_phase")),
            "reproduced_onset_anchor_source": ((probe_row.get("reproduced_onset") or {}).get("anchor_source")),
        }
        report_rows.append(
            {
                "case_id": case_id,
                "checkpoint_alias": probe_row["checkpoint_alias"],
                "classification": classification,
                "evidence_layers": evidence_layers,
                "precursor_signals": precursor_signals,
                "historical_onset": probe_row.get("historical_onset"),
                "reproduced_onset": probe_row.get("reproduced_onset"),
                "probe_surface_status": probe_payload.get("probe_surface_status"),
                "mechanism_signals": mechanism_signals,
                "intervention_summary": intervention_summary,
                "control_margins": ((compare_row or {}).get("margins")),
            }
        )
    cohort_summary = _build_cohort_summary(report_rows)
    family_comparison = _build_family_comparison_report(
        inventory_rows=list(inventory.get("checkpoints") or []),
        report_rows=report_rows,
    )
    summary = {
        "inventory": inventory,
        "readiness_split": {
            "ready_to_probe": list(inventory.get("ready_to_probe") or []),
            "fresh_inference_needed": list(inventory.get("fresh_inference_needed") or []),
        },
        "num_selected_cases": len(selected_cases),
        "num_reproduced_checkpoints": len(reproduction.get("reproductions", [])),
        "classification_counts": dict(counts),
        "case_rows": report_rows,
        "cohort_summary": cohort_summary,
        "family_comparison": family_comparison,
        "scope_note": (
            "FN injection remains out of primary scope unless direct probe evidence contradicts that assumption."
        ),
        "non_goal_note": (
            "Heuristic duplicate suppression and decode-policy mitigation are excluded as the primary outcome."
        ),
    }
    report_dir = run_dir / "report"
    _write_json(report_dir / "summary.json", summary)
    return summary


def _write_report_markdown(summary: Mapping[str, Any], *, run_dir: Path) -> None:
    report_dir = run_dir / "report"
    lines = [
        "# Duplication Collapse Analysis",
        "",
        "## Summary",
        "",
        f"- Selected cases: {int(summary.get('num_selected_cases') or 0)}",
        f"- Reproduced checkpoints: {int(summary.get('num_reproduced_checkpoints') or 0)}",
        "",
        "## Findings",
        "",
    ]
    counts = summary.get("classification_counts") or {}
    for key in sorted(counts):
        lines.append(f"- {key}: {int(counts[key])}")
    readiness_split = summary.get("readiness_split") or {}
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"- Ready to probe: {len(list(readiness_split.get('ready_to_probe') or []))}",
            f"- Fresh inference needed: {len(list(readiness_split.get('fresh_inference_needed') or []))}",
            "",
            "## Cohort",
            "",
        ]
    )
    cohort_summary = summary.get("cohort_summary") or {}
    for alias, row in sorted((cohort_summary.get("checkpoint_rows") or {}).items()):
        lines.append(
            f"- {alias}: {json.dumps(row, ensure_ascii=False)}"
        )
    lines.extend(["", "## Family Comparison", ""])
    family_comparison = summary.get("family_comparison") or {}
    missing_roles = list(family_comparison.get("missing_expected_roles") or [])
    if missing_roles:
        lines.append(f"- Missing expected roles: {json.dumps(missing_roles, ensure_ascii=False)}")
    for row in family_comparison.get("rows") or []:
        lines.append(f"- {json.dumps(row, ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- {summary.get('scope_note')}",
            f"- {summary.get('non_goal_note')}",
            "",
        ]
    )
    for row in summary.get("case_rows") or []:
        lines.append(f"### {row['case_id']}")
        lines.append("")
        lines.append(f"- Checkpoint: {row['checkpoint_alias']}")
        lines.append(f"- Classification: {row['classification']}")
        evidence_layers = row.get("evidence_layers") or {}
        if evidence_layers:
            lines.append(f"- Evidence layers: {json.dumps(evidence_layers, ensure_ascii=False)}")
        precursor_signals = row.get("precursor_signals") or {}
        if precursor_signals:
            lines.append(f"- Precursors: {json.dumps(precursor_signals, ensure_ascii=False)}")
        surface_status = row.get("probe_surface_status") or {}
        if surface_status:
            lines.append(f"- Probe surfaces: {json.dumps(surface_status, ensure_ascii=False)}")
        mechanism_signals = row.get("mechanism_signals") or {}
        if mechanism_signals:
            lines.append(f"- Mechanism signals: {json.dumps(mechanism_signals, ensure_ascii=False)}")
        intervention_summary = row.get("intervention_summary") or {}
        if intervention_summary:
            lines.append(f"- Intervention summary: {json.dumps(intervention_summary, ensure_ascii=False)}")
        control_margins = row.get("control_margins") or {}
        if control_margins:
            lines.append(f"- Control margins: {json.dumps(control_margins, ensure_ascii=False)}")
        lines.append("")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_study(config_path: Path) -> Dict[str, Any]:
    cfg = load_study_config(config_path)
    _set_cuda_visible_devices(cfg.execution.cuda_visible_devices)
    run_dir, run_root_source = _resolve_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "run_root_source": run_root_source,
    }
    resolved_checkpoints = tuple(
        _resolve_checkpoint(
            item,
            checkpoint_name_filter=cfg.run.checkpoint_name_filter,
        )
        for item in cfg.checkpoints
    )
    manifest = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "run_root_source": run_root_source,
        "decode_contract": {
            "temperature": cfg.decode.temperature,
            "top_p": cfg.decode.top_p,
            "repetition_penalty": cfg.decode.repetition_penalty,
            "max_new_tokens": cfg.decode.max_new_tokens,
            "seed": cfg.decode.seed,
            "secondary_top_k": list(cfg.decode.secondary_top_k),
            "secondary_top_p": list(cfg.decode.secondary_top_p),
        },
        "checkpoints": [
            _inventory_row(item) for item in resolved_checkpoints
        ],
    }
    _write_json(run_dir / "resolved_manifest.json", manifest)

    inventory = _build_inventory(resolved_checkpoints, run_dir=run_dir)
    result["inventory"] = inventory

    case_selection = _build_case_selection(
        resolved_checkpoints,
        cfg=cfg,
        run_dir=run_dir,
    )
    result["case_selection"] = case_selection["summary"]
    selected_cases = list(case_selection["selected_cases"])

    selected_cases = _assign_local_line_indices(selected_cases)

    reproduction = _reproduce_cases(
        resolved_checkpoints,
        cfg=cfg,
        run_dir=run_dir,
        selected_cases=selected_cases if "reproduce" in cfg.run.stages else (),
    ) if "reproduce" in cfg.run.stages else {"reproductions": []}
    result["reproduction"] = {
        "num_reproductions": len(reproduction.get("reproductions", []))
    }

    probe_rows: List[Dict[str, Any]] = []
    if "probe" in cfg.run.stages and selected_cases:
        reproduce_lookup = {
            row["checkpoint_alias"]: row for row in reproduction.get("reproductions", [])
        }
        checkpoint_lookup = {item.spec.alias: item for item in resolved_checkpoints}
        for case in selected_cases[: int(cfg.probe.max_cases)]:
            reproduce_row = reproduce_lookup.get(str(case["checkpoint_alias"]))
            if reproduce_row is None:
                continue
            gt_rows = _read_jsonl(Path(str(reproduce_row["gt_vs_pred_jsonl"])))
            conf_rows = _load_confidence_index(Path(str(reproduce_row["pred_confidence_jsonl"])))
            trace_rows = _load_trace_index(Path(str(reproduce_row["pred_token_trace_jsonl"])))
            local_idx = int(case["local_line_idx"])
            if local_idx >= len(gt_rows):
                continue
            reproduced_gt = gt_rows[local_idx]
            reproduced_conf = conf_rows.get(local_idx)
            reproduced_onset = _resolve_probe_anchor(
                list(reproduced_gt.get("pred") or []),
                case_row=case,
                confidence_record=reproduced_conf,
                cfg=cfg,
            )
            checkpoint = checkpoint_lookup[str(case["checkpoint_alias"])]
            image = _load_image_for_case(str(reproduced_gt.get("image") or ""))
            try:
                runner = HFStudyRunner(
                    checkpoint=checkpoint.resolved,
                    device=cfg.execution.device,
                    image_root=REPO_ROOT,
                )
                generation = runner.generate_image_only_batch(
                    images=[image],
                    gen_cfg=GenerationConfig(
                        temperature=cfg.decode.temperature,
                        top_p=cfg.decode.top_p,
                        repetition_penalty=cfg.decode.repetition_penalty,
                        max_new_tokens=cfg.decode.max_new_tokens,
                        seed=cfg.decode.seed,
                        batch_size=1,
                    ),
                )[0]
                trace_row = trace_rows.get(local_idx) or {}
                trace_match = bool(
                    generation.generated_token_text
                    and list(generation.generated_token_text)
                    == list(trace_row.get("generated_token_text") or [])
                )
            finally:
                image.close()
                del runner
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            prober = Qwen3VLSurgeryProber(
                checkpoint=checkpoint.resolved,
                bbox_format=checkpoint.spec.bbox_format,
                device=cfg.execution.device,
                attn_implementation=cfg.execution.probe_attn_implementation,
            )
            try:
                probe = prober.probe_case(
                    image_path=str(reproduced_gt.get("image") or ""),
                    generation_ids=generation.generated_token_ids,
                    generation_token_text=generation.generated_token_text,
                    onset=reproduced_onset or {},
                    cfg=cfg,
                )
            finally:
                prober.close()
            probe_row = {
                "case_id": str(case["case_id"]),
                "checkpoint_alias": str(case["checkpoint_alias"]),
                "historical_onset": case.get("onset"),
                "reproduced_onset": reproduced_onset,
                "trace_text_match": trace_match,
                "probe": probe,
            }
            probe_rows.append(probe_row)
        probe_dir = run_dir / "probe"
        _write_jsonl(probe_dir / "case_rows.jsonl", probe_rows)
        _write_json(probe_dir / "summary.json", {"num_cases": len(probe_rows)})
    result["probe"] = {"num_cases": len(probe_rows)}

    compare = _run_control_comparisons(
        resolved_checkpoints,
        cfg=cfg,
        run_dir=run_dir,
        selected_cases=selected_cases if "compare" in cfg.run.stages else (),
        reproduction=reproduction,
        probe_rows=probe_rows,
    ) if "compare" in cfg.run.stages else {"case_rows": []}
    result["compare"] = {"num_cases": len(compare.get("case_rows", []))}

    report = _build_report(
        inventory=inventory,
        selected_cases=selected_cases,
        reproduction=reproduction,
        probe_rows=probe_rows,
        compare=compare,
        run_dir=run_dir,
    )
    if cfg.report.write_markdown:
        _write_report_markdown(report, run_dir=run_dir)
    result["report"] = {
        "classification_counts": report.get("classification_counts", {}),
    }
    _write_json(run_dir / "summary.json", result)
    return result


def run_duplication_collapse_analysis_study(config_path: Path) -> Dict[str, Any]:
    return run_study(config_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the study YAML config.")
    args = parser.parse_args()
    result = run_study(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


__all__ = [
    "CheckpointSpec",
    "ControlConfig",
    "DecodeConfig",
    "ExecutionConfig",
    "HistoricalArtifactBundle",
    "ProbeConfig",
    "Qwen3VLSurgeryProber",
    "ReportConfig",
    "ResolvedStudyCheckpoint",
    "RunConfig",
    "StudyConfig",
    "SubsetConfig",
    "WorkspaceConfig",
    "_detect_onset",
    "_inventory_row",
    "_pair_duplicate_metrics",
    "load_study_config",
    "run_duplication_collapse_analysis_study",
    "run_study",
]
