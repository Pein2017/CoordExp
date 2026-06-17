"""Unified inference pipeline runner (infer -> eval and/or vis).

This module is intentionally YAML-first and does NOT use the training config
loader (no extends/inherit, no interpolation), per OpenSpec.

Primary entrypoint is `scripts/run_infer.py --config <yaml>`.
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, cast

from src.common.geometry.bbox_parameterization import normalize_bbox_format
from src.common.object_field_order import (
    ObjectFieldOrder,
    ObjectOrdering,
    normalize_object_field_order,
    normalize_object_ordering,
)
from src.config.prompts import (
    coord_mode_from_coord_tokens_enabled,
    get_template_prompt_hash,
    get_template_prompts,
    resolve_dense_prompt_variant_key,
)
from src.detection.template_contracts import (
    DetectionTemplateId,
    resolve_detection_template_contract,
)
from src.eval.artifacts import (
    CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_SOURCE,
    CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_VERSION,
    CXCY_LOGW_LOGH_CONSTANT_SCORE,
    CXCYWH_CONSTANT_PRED_SCORE_SOURCE,
    CXCYWH_CONSTANT_PRED_SCORE_VERSION,
    CXCYWH_CONSTANT_SCORE,
    with_constant_scores,
    write_jsonl_records,
)
from src.infer.artifacts import (
    build_eval_artifact_paths,
    build_score_policy_fingerprint,
    load_comparable_artifact,
)
from src.infer.checkpoints import (
    VLLM_ADAPTER_UNSUPPORTED_MESSAGE,
    resolve_inference_checkpoint,
    validate_compact_coord_token_adapter_contract,
)
from src.infer.prompt import DetectionPromptPolicy, prompt_policy_fingerprint
from src.infer.runtime import (
    build_decode_request_from_infer_config,
    build_model_identity_fingerprint,
    legacy_generation_kwargs_from_decode_request,
    run_offline_inference,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _offline_prompt_policy_fingerprint(
    *,
    coord_mode: str,
    prompt_variant: str,
    object_field_order: ObjectFieldOrder,
    bbox_format: str,
    detection_template_id: str,
    object_ordering: ObjectOrdering,
) -> str:
    system_prompt, user_prompt = get_template_prompts(
        ordering=object_ordering,
        coord_mode=coord_mode,
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
        bbox_format=bbox_format,
        detection_template_id=detection_template_id,
    )
    return prompt_policy_fingerprint(
        DetectionPromptPolicy(
            name="coordexp_offline_detection_prompt",
            version="1",
            system_prompt=system_prompt or "",
            user_prompt=user_prompt,
            image_count=1,
            do_resize=False,
        )
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML config requires PyYAML (import yaml). Install it in the ms env."
        ) from exc

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError("pipeline config must be a YAML mapping at top-level")
    return data


def _get_map(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    val = cfg.get(key, {})
    if val is None:
        return {}
    if not isinstance(val, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return val


def _get_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    val = cfg.get(key, default)
    if isinstance(val, bool):
        return val
    if val in (0, 1):
        return bool(val)
    raise ValueError(f"{key} must be a bool")


def _get_str(
    cfg: Mapping[str, Any], key: str, default: Optional[str] = None
) -> Optional[str]:
    if key not in cfg:
        return default
    val = cfg.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ValueError(f"{key} must be a string")
    return val


def _require_str(cfg: Mapping[str, Any], key: str) -> str:
    val = _get_str(cfg, key, None)
    if val is None or not str(val).strip():
        raise ValueError(f"{key} is required and must be a non-empty string")
    return val


def _require_choice(
    cfg: Mapping[str, Any], key: str, allowed: set[str], default: Optional[str] = None
) -> str:
    val = _get_str(cfg, key, default)
    if val is None:
        raise ValueError(f"{key} is required and must be one of {sorted(allowed)}")
    v = str(val).strip().lower()
    if v not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}, got {val!r}")
    return v


def _reject_retired_infer_template_knobs(infer_cfg: Mapping[str, Any]) -> None:
    for key in (
        "detection_sequence_format",
        "row_separator",
        "compact_full_parse_mode",
    ):
        if key in infer_cfg:
            raise ValueError(
                f"infer.{key} is retired; use top-level detection_template.id"
            )
    parsing_cfg = _get_map(infer_cfg, "parsing")
    if "compact_full" in parsing_cfg:
        raise ValueError(
            "infer.parsing.compact_full is retired; use top-level detection_template.id"
        )


def _resolve_detection_template_id(cfg: Mapping[str, Any]) -> DetectionTemplateId:
    template_cfg = _get_map(cfg, "detection_template")
    raw_id = template_cfg.get("id", "stage1_json_pretty")
    if not isinstance(raw_id, str):
        raise ValueError("detection_template.id must be a string")
    return resolve_detection_template_contract(raw_id).template_id


def _detection_sequence_format_for_template_id(template_id: str) -> str:
    contract = resolve_detection_template_contract(template_id)
    return "compact" if contract.is_compact else "coordjson"


def _parser_mode_for_template_id(template_id: str) -> str:
    contract = resolve_detection_template_contract(template_id)
    if not contract.is_compact:
        return "strict_expected"
    if contract.template_id == "compact":
        return "marker_delimited_strict"
    return "strict_expected"


def _get_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    val = cfg.get(key, default)
    try:
        return int(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an int") from exc


def _get_limit(cfg: Mapping[str, Any], key: str, default: int = 0) -> int:
    if key not in cfg:
        return int(default)
    val = cfg.get(key)
    if val is None:
        return int(default)
    try:
        limit = int(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an int or null") from exc
    if limit < 0:
        raise ValueError(f"{key} must be >= 0")
    return limit


def _resolve_eval_duplicate_control_enabled(eval_cfg: Mapping[str, Any]) -> bool:
    raw = eval_cfg.get("duplicate_control", {})
    if raw is None:
        return False
    if not isinstance(raw, Mapping):
        raise ValueError("eval.duplicate_control must be a mapping")
    unknown_keys = sorted(str(key) for key in raw.keys() if str(key) != "enabled")
    if unknown_keys:
        rendered = ", ".join(f"eval.duplicate_control.{key}" for key in unknown_keys)
        raise ValueError(
            f"Unknown duplicate-control keys are unsupported: {rendered}. "
            "Only eval.duplicate_control.enabled is allowed."
        )
    return _get_bool(raw, "enabled", False)


def _detect_infer_distributed_env() -> Tuple[int, int, int, bool]:
    rank = 0
    local_rank = 0
    world_size = 1

    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
            world_size = max(int(torch.distributed.get_world_size()), 1)
            local_rank = int(os.environ.get("LOCAL_RANK", rank) or rank)
            return rank, local_rank, world_size, world_size > 1
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass

    rank_raw = os.environ.get("RANK") or os.environ.get("SLURM_PROCID")
    local_rank_raw = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
    world_size_raw = os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS")

    try:
        if rank_raw is not None:
            rank = int(rank_raw)
    except (TypeError, ValueError):
        rank = 0
    try:
        if local_rank_raw is not None:
            local_rank = int(local_rank_raw)
        else:
            local_rank = rank
    except (TypeError, ValueError):
        local_rank = rank
    try:
        if world_size_raw is not None:
            world_size = max(int(world_size_raw), 1)
    except (TypeError, ValueError):
        world_size = 1

    return rank, local_rank, world_size, world_size > 1


def _resolve_infer_prompt_controls(
    infer_cfg: Mapping[str, Any],
) -> Tuple[str, str, ObjectFieldOrder, ObjectOrdering]:
    prompt_variant_raw = infer_cfg.get("prompt_variant", None)
    if prompt_variant_raw is not None and not isinstance(prompt_variant_raw, str):
        raise ValueError("infer.prompt_variant must be a string when provided")
    prompt_variant = resolve_dense_prompt_variant_key(prompt_variant_raw)
    bbox_format = normalize_bbox_format(
        infer_cfg.get("bbox_format", "xyxy"),
        path="infer.bbox_format",
    )
    object_ordering = normalize_object_ordering(
        infer_cfg.get("object_ordering", "sorted"),
        path="infer.object_ordering",
    )

    object_field_order_raw = infer_cfg.get("object_field_order", "desc_first")
    object_field_order = normalize_object_field_order(
        object_field_order_raw,
        path="infer.object_field_order",
    )

    return (
        prompt_variant,
        bbox_format,
        object_field_order,
        object_ordering,
    )


def _resolve_infer_coord_mode(
    infer_cfg: Mapping[str, Any],
) -> str:
    requested_mode = str(infer_cfg.get("mode", "auto") or "auto").strip().lower()
    if requested_mode == "coord":
        return coord_mode_from_coord_tokens_enabled(True)
    if requested_mode == "text":
        return coord_mode_from_coord_tokens_enabled(False)
    if requested_mode == "auto":
        gt_jsonl = str(infer_cfg.get("gt_jsonl", "") or "").strip()
        if not gt_jsonl:
            return coord_mode_from_coord_tokens_enabled(True)
        if not Path(gt_jsonl).is_file():
            return coord_mode_from_coord_tokens_enabled(True)
        detect_samples_raw = infer_cfg.get("detect_samples", 128)
        try:
            detect_samples = int(detect_samples_raw)
        except (TypeError, ValueError):
            detect_samples = 128
        from src.infer.runtime import detect_mode_from_gt

        resolved_mode, _reason = detect_mode_from_gt(
            gt_jsonl, sample_size=max(detect_samples, 1)
        )
        return coord_mode_from_coord_tokens_enabled(resolved_mode == "coord")
    raise ValueError("infer.mode must be one of {'coord', 'text', 'auto'}")


def _resolve_infer_runtime_mode(
    infer_cfg: Mapping[str, Any],
) -> tuple[Literal["coord", "text"], Optional[str]]:
    requested_mode = str(infer_cfg.get("mode", "auto") or "auto").strip().lower()
    if requested_mode == "coord":
        return "coord", None
    if requested_mode == "text":
        return "text", None
    if requested_mode != "auto":
        raise ValueError("infer.mode must be one of {'coord', 'text', 'auto'}")
    gt_jsonl = str(infer_cfg.get("gt_jsonl", "") or "").strip()
    if not gt_jsonl or not Path(gt_jsonl).is_file():
        return "coord", "default_coord_when_gt_unavailable"
    detect_samples_raw = infer_cfg.get("detect_samples", 128)
    try:
        detect_samples = int(detect_samples_raw)
    except (TypeError, ValueError):
        detect_samples = 128
    from src.infer.runtime import detect_mode_from_gt

    resolved_mode, reason = detect_mode_from_gt(
        gt_jsonl,
        sample_size=max(detect_samples, 1),
    )
    return resolved_mode, reason


def _derive_run_dir(cfg: Mapping[str, Any]) -> Path:
    run_cfg = _get_map(cfg, "run")
    art_cfg = _get_map(cfg, "artifacts")

    # Precedence: artifacts.run_dir > run.output_dir+run.name > parent(gt_vs_pred_jsonl)
    run_dir = _get_str(art_cfg, "run_dir")
    if run_dir:
        return Path(run_dir)

    out_dir = _get_str(run_cfg, "output_dir")
    run_name = _get_str(run_cfg, "name")
    if out_dir and run_name:
        return Path(out_dir) / run_name

    gt_vs_pred = _get_str(art_cfg, "gt_vs_pred_jsonl")
    if gt_vs_pred:
        return Path(gt_vs_pred).parent

    raise ValueError(
        "YAML must specify either artifacts.run_dir, or (run.output_dir + run.name), "
        "or artifacts.gt_vs_pred_jsonl"
    )


@dataclass(frozen=True)
class ResolvedArtifacts:
    run_dir: Path
    gt_vs_pred_jsonl: Path
    pred_token_trace_jsonl: Path
    gt_vs_pred_scored_jsonl: Path | None
    summary_json: Path
    eval_dir: Path
    vis_dir: Path
    gt_vs_pred_guarded_jsonl: Path | None = None
    gt_vs_pred_scored_guarded_jsonl: Path | None = None
    metrics_guarded_json: Path | None = None
    duplicate_guard_report_json: Path | None = None


@dataclass(frozen=True)
class ResolvedStages:
    infer: bool
    eval: bool
    vis: bool


def resolve_artifacts(
    cfg: Mapping[str, Any],
) -> Tuple[ResolvedArtifacts, ResolvedStages]:
    # Stages: if `stages` is provided, it must specify all three toggles.
    if "stages" in cfg:
        raw = cfg.get("stages")
        if raw is None:
            raise ValueError(
                "stages must be a mapping with infer/eval/vis (or omit stages)"
            )
        if not isinstance(raw, Mapping):
            raise ValueError("stages must be a mapping")
        for k in ("infer", "eval", "vis"):
            if k not in raw:
                raise ValueError("stages must include infer, eval, vis")
        stages_cfg = raw
        stages = ResolvedStages(
            infer=_get_bool(stages_cfg, "infer", True),
            eval=_get_bool(stages_cfg, "eval", False),
            vis=_get_bool(stages_cfg, "vis", False),
        )
    else:
        stages = ResolvedStages(infer=True, eval=False, vis=False)

    run_dir = _derive_run_dir(cfg)
    art_cfg = _get_map(cfg, "artifacts")

    gt_vs_pred = _get_str(art_cfg, "gt_vs_pred_jsonl")
    if gt_vs_pred:
        gt_vs_pred_jsonl = Path(gt_vs_pred)
    else:
        gt_vs_pred_jsonl = run_dir / "gt_vs_pred.jsonl"

    pred_token_trace = _get_str(art_cfg, "pred_token_trace_jsonl")
    pred_token_trace_jsonl = (
        Path(pred_token_trace)
        if pred_token_trace
        else run_dir / "pred_token_trace.jsonl"
    )

    gt_vs_pred_scored = _get_str(art_cfg, "gt_vs_pred_scored_jsonl")
    gt_vs_pred_scored_jsonl = (
        Path(gt_vs_pred_scored)
        if gt_vs_pred_scored
        else run_dir / "gt_vs_pred_scored.jsonl"
    )

    summary_json = Path(_get_str(art_cfg, "summary_json") or (run_dir / "summary.json"))

    eval_cfg = _get_map(cfg, "eval")
    vis_cfg = _get_map(cfg, "vis")

    eval_dir = Path(_get_str(eval_cfg, "output_dir") or (run_dir / "eval"))
    vis_dir = Path(_get_str(vis_cfg, "output_dir") or (run_dir / "vis"))
    eval_artifact_paths = build_eval_artifact_paths(run_dir=run_dir, eval_dir=eval_dir)

    return (
        ResolvedArtifacts(
            run_dir=run_dir,
            gt_vs_pred_jsonl=gt_vs_pred_jsonl,
            pred_token_trace_jsonl=pred_token_trace_jsonl,
            gt_vs_pred_scored_jsonl=gt_vs_pred_scored_jsonl,
            gt_vs_pred_guarded_jsonl=eval_artifact_paths["gt_vs_pred_guarded_jsonl"],
            gt_vs_pred_scored_guarded_jsonl=eval_artifact_paths[
                "gt_vs_pred_scored_guarded_jsonl"
            ],
            summary_json=summary_json,
            metrics_guarded_json=eval_artifact_paths["metrics_guarded_json"],
            duplicate_guard_report_json=eval_artifact_paths[
                "duplicate_guard_report_json"
            ],
            eval_dir=eval_dir,
            vis_dir=vis_dir,
        ),
        stages,
    )


def _load_or_raise_artifact(path: Path) -> Path:
    if path.exists():
        return path

    # Transition alias: allow pred.jsonl as a fallback for consumers.
    if path.name == "gt_vs_pred.jsonl":
        legacy = path.with_name("pred.jsonl")
        if legacy.exists():
            logger.warning("Using legacy artifact alias: %s", legacy)
            return legacy

    raise FileNotFoundError(f"Required artifact not found: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_inference_provenance_payloads(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    candidates: list[Mapping[str, Any]] = [payload]
    for key in ("provenance", "inference_provenance"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    return tuple(candidates)


def _extract_generation_provenance(payload: Mapping[str, Any]) -> Dict[str, str]:
    for candidate in _candidate_inference_provenance_payloads(payload):
        values = {
            "prompt_policy_fingerprint": candidate.get("prompt_policy_fingerprint"),
            "decode_policy_fingerprint": candidate.get("decode_policy_fingerprint"),
            "model_identity_fingerprint": candidate.get("model_identity_fingerprint"),
        }
        if all(isinstance(value, str) and value.strip() for value in values.values()):
            return {key: str(value) for key, value in values.items()}
    raise ValueError("missing_provenance: raw artifact lacks generation fingerprints")


def _parser_policy_for_score_provenance(cfg: Mapping[str, Any]) -> str:
    detection_template_id = _resolve_detection_template_id(cfg)
    return f"{detection_template_id}:{_parser_mode_for_template_id(detection_template_id)}"


def _write_scored_artifact_provenance(
    *,
    cfg: Mapping[str, Any],
    raw_path: Path,
    scored_path: Path,
    policy_name: str,
    score_source: str,
    aggregation_rule: str,
    token_span_rule: str,
    constant_score_value: Any,
) -> bool:
    """Write a score-bearing sidecar when exact raw provenance is available."""

    try:
        raw_loaded = load_comparable_artifact(raw_path)
        generation_provenance = _extract_generation_provenance(
            cast(Mapping[str, Any], raw_loaded["provenance"])
        )
    except ValueError as exc:
        logger.warning(
            "Could not stamp score provenance for %s because raw artifact %s "
            "is not comparable: %s",
            scored_path,
            raw_path,
            exc,
        )
        return False

    raw_identity = {
        "path": str(raw_path),
        "sha256": _sha256_file(raw_path),
    }
    parser_policy = _parser_policy_for_score_provenance(cfg)
    detection_template_id = _resolve_detection_template_id(cfg)
    score_policy_fingerprint = build_score_policy_fingerprint(
        policy_name=policy_name,
        score_source=score_source,
        aggregation_rule=aggregation_rule,
        token_span_rule=token_span_rule,
        constant_score_value=constant_score_value,
        source_raw_artifact_identity=raw_identity,
        parser_policy=parser_policy,
        metric_bearing=True,
    )
    sidecar = {
        **generation_provenance,
        "score_policy_fingerprint": score_policy_fingerprint,
        "metric_bearing": True,
        "artifact_path": str(scored_path),
        "detection_template": {
            "id": detection_template_id,
        },
        "detection_template_id": detection_template_id,
        "source_raw_artifact": str(raw_path),
        "source_raw_artifact_identity": raw_identity,
        "parser_policy": parser_policy,
        "score_policy": {
            "policy_name": policy_name,
            "score_source": score_source,
            "aggregation_rule": aggregation_rule,
            "token_span_rule": token_span_rule,
            "constant_score_value": constant_score_value,
        },
    }
    sidecar_path = scored_path.with_suffix(scored_path.suffix + ".provenance.json")
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


_SENSITIVE_KEYS = {
    "access_token",
    "hf_token",
    "huggingface_token",
    "token",
    "password",
    "secret",
}


def redact_config(obj: Any) -> Any:
    """Redact common secret fields before persisting resolved config artifacts."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k).lower()
            if (
                ks in _SENSITIVE_KEYS
                or ks.endswith("_token")
                or ks.endswith("_password")
                or ks.endswith("_secret")
            ):
                out[k] = "<REDACTED>"
            else:
                out[k] = redact_config(v)
        return out
    if isinstance(obj, list):
        return [redact_config(x) for x in obj]
    return obj


RESOLVED_CONFIG_SCHEMA_VERSION = 1


def load_resolved_config(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("resolved_config.json must be a JSON object")

    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("resolved_config.json schema_version must be an int")
    if schema_version != RESOLVED_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported resolved_config.json schema_version={schema_version}; "
            f"supported={RESOLVED_CONFIG_SCHEMA_VERSION}"
        )

    stages = raw.get("stages")
    if not isinstance(stages, dict) or not {"infer", "eval", "vis"}.issubset(stages):
        raise ValueError(
            "resolved_config.json is missing required stages.infer/eval/vis"
        )

    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, dict) or "run_dir" not in artifacts:
        raise ValueError("resolved_config.json is missing required artifacts")

    root_source = raw.get("root_image_dir_source")
    if root_source not in {"env", "config", "gt_parent", "none"}:
        raise ValueError(
            "resolved_config.json root_image_dir_source must be one of env|config|gt_parent|none"
        )

    return raw


def _candidate_resolved_config_paths_for_jsonl(jsonl_path: Path) -> List[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    pointer_path = jsonl_path.parent / "resolved_config.path"
    if pointer_path.exists():
        try:
            pointer_raw = str(pointer_path.read_text(encoding="utf-8") or "").strip()
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read manifest pointer at {pointer_path}."
            ) from exc
        if pointer_raw:
            pointed = Path(pointer_raw).expanduser()
            if not pointed.is_absolute():
                try:
                    pointed = (pointer_path.parent / pointed).resolve()
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to resolve manifest pointer target {pointer_raw!r} from {pointer_path}."
                    ) from exc
            _push(pointed)

    _push(jsonl_path.parent / "resolved_config.json")

    for parent in list(jsonl_path.parents)[:4]:
        _push(parent / "resolved_config.json")

    return candidates



def _find_resolved_config_for_jsonl(jsonl_path: Path) -> Optional[Dict[str, Any]]:
    jsonl_resolved = jsonl_path.resolve()
    fallback: Optional[Dict[str, Any]] = None

    for candidate in _candidate_resolved_config_paths_for_jsonl(jsonl_path):
        if not candidate.exists():
            continue
        try:
            resolved = load_resolved_config(candidate)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping invalid resolved config candidate %s: %s", candidate, exc)
            continue

        if fallback is None:
            fallback = resolved

        artifacts = resolved.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue

        gt_vs_pred_jsonl = artifacts.get("gt_vs_pred_jsonl")
        if isinstance(gt_vs_pred_jsonl, str) and gt_vs_pred_jsonl.strip():
            try:
                gt_vs_pred_path = Path(gt_vs_pred_jsonl).resolve()
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid artifact path gt_vs_pred_jsonl={gt_vs_pred_jsonl!r} in {candidate}."
                ) from exc
            if gt_vs_pred_path == jsonl_resolved:
                return resolved

        run_dir = artifacts.get("run_dir")
        if isinstance(run_dir, str) and run_dir.strip():
            try:
                run_dir_resolved = Path(run_dir).resolve()
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid artifact path run_dir={run_dir!r} in {candidate}."
                ) from exc
            if run_dir_resolved in jsonl_resolved.parents:
                return resolved

    return fallback


def _resolve_root_image_dir_common(
    *,
    run_root_image_dir: Optional[str] = None,
    gt_jsonl: Optional[str] = None,
    resolved_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[Path], str]:
    root_env = str(os.environ.get("ROOT_IMAGE_DIR") or "").strip()
    if root_env:
        return Path(root_env).resolve(), "env"

    if run_root_image_dir is not None and str(run_root_image_dir).strip():
        return Path(str(run_root_image_dir)).resolve(), "config"

    if resolved_cfg is not None:
        root_cfg = resolved_cfg.get("root_image_dir")
        root_source = resolved_cfg.get("root_image_dir_source")
        if isinstance(root_cfg, str) and root_cfg.strip():
            source = str(root_source).strip() if root_source is not None else "config"
            return Path(root_cfg).resolve(), source or "config"

    if gt_jsonl is not None and str(gt_jsonl).strip():
        return Path(str(gt_jsonl)).parent.resolve(), "gt_parent"

    return None, "none"

def resolve_root_image_dir_for_jsonl(jsonl_path: Path) -> Tuple[Optional[Path], str]:
    resolved = _find_resolved_config_for_jsonl(jsonl_path)
    return _resolve_root_image_dir_common(resolved_cfg=resolved)


def _resolve_root_image_dir(cfg: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    run_cfg = _get_map(cfg, "run")
    infer_cfg = _get_map(cfg, "infer")

    run_root_image_dir = _get_str(run_cfg, "root_image_dir")
    gt_jsonl = _get_str(infer_cfg, "gt_jsonl")

    root_path, source = _resolve_root_image_dir_common(
        run_root_image_dir=run_root_image_dir,
        gt_jsonl=gt_jsonl,
    )
    if root_path is None:
        return None, source
    return str(root_path), source


def run_pipeline(
    *,
    config_path: Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ResolvedArtifacts:
    """Run stages from a single YAML config.

    `overrides` is a flat mapping of dotted keys (e.g. `infer.limit`) used to
    implement legacy CLI overrides.
    """

    cfg = _load_yaml(config_path)

    # No inheritance / interpolation: treat as one file.
    if not isinstance(cfg, dict):
        raise ValueError("pipeline config must be a mapping")

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    infer_cfg = _get_map(cfg, "infer")
    _reject_retired_infer_template_knobs(infer_cfg)
    resolved_detection_template_id = _resolve_detection_template_id(cfg)
    requested_model_checkpoint = _get_str(infer_cfg, "model_checkpoint")
    requested_adapter_checkpoint = _get_str(infer_cfg, "adapter_checkpoint")
    resolved_checkpoint = None
    if requested_model_checkpoint:
        resolved_checkpoint = resolve_inference_checkpoint(
            model_checkpoint=requested_model_checkpoint,
            adapter_checkpoint=requested_adapter_checkpoint,
        )
        backend_cfg = _get_map(infer_cfg, "backend")
        backend_type = str(backend_cfg.get("type") or "").strip().lower()
        if (
            resolved_checkpoint.resolved_adapter_checkpoint is not None
            and backend_type == "vllm"
        ):
            raise ValueError(VLLM_ADAPTER_UNSUPPORTED_MESSAGE)
    (
        resolved_prompt_variant,
        resolved_bbox_format,
        resolved_object_field_order,
        resolved_object_ordering,
    ) = _resolve_infer_prompt_controls(infer_cfg)
    resolved_parser_mode = _parser_mode_for_template_id(resolved_detection_template_id)
    resolved_runtime_mode, resolved_mode_reason = _resolve_infer_runtime_mode(infer_cfg)
    resolved_coord_mode = coord_mode_from_coord_tokens_enabled(
        resolved_runtime_mode == "coord"
    )
    if resolved_checkpoint is not None:
        validate_compact_coord_token_adapter_contract(
            resolved_checkpoint,
            detection_template_id=resolved_detection_template_id,
        )
    resolved_prompt_hash = get_template_prompt_hash(
        ordering=resolved_object_ordering,
        coord_mode=resolved_coord_mode,
        prompt_variant=resolved_prompt_variant,
        object_field_order=resolved_object_field_order,
        bbox_format=resolved_bbox_format,
        detection_template_id=resolved_detection_template_id,
    )
    resolved_prompt_policy_fingerprint = _offline_prompt_policy_fingerprint(
        coord_mode=resolved_coord_mode,
        prompt_variant=resolved_prompt_variant,
        object_field_order=resolved_object_field_order,
        bbox_format=resolved_bbox_format,
        detection_template_id=resolved_detection_template_id,
        object_ordering=resolved_object_ordering,
    )
    resolved_decode_request = None
    if _get_map(infer_cfg, "generation"):
        resolved_decode_request = build_decode_request_from_infer_config(infer_cfg)
    resolved_model_identity_fingerprint = None
    if resolved_checkpoint is not None and resolved_decode_request is not None:
        resolved_model_identity_fingerprint = build_model_identity_fingerprint(
            checkpoint_mode=resolved_checkpoint.checkpoint_mode,
            requested_model_checkpoint=resolved_checkpoint.requested_model_checkpoint,
            requested_adapter_checkpoint=(
                resolved_checkpoint.requested_adapter_checkpoint
            ),
            resolved_base_model_checkpoint=(
                resolved_checkpoint.resolved_base_model_checkpoint
            ),
            resolved_adapter_checkpoint=(
                resolved_checkpoint.resolved_adapter_checkpoint
            ),
            backend=resolved_decode_request.backend,
            backend_mode=resolved_decode_request.backend_mode,
            backend_model=_get_str(backend_cfg, "model"),
        )
    artifacts, stages = resolve_artifacts(cfg)

    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts.eval_dir.mkdir(parents=True, exist_ok=True)
    artifacts.vis_dir.mkdir(parents=True, exist_ok=True)

    root_image_dir, root_image_dir_source = _resolve_root_image_dir(cfg)
    eval_cfg = _get_map(cfg, "eval")
    duplicate_control_enabled = _resolve_eval_duplicate_control_enabled(eval_cfg)

    # Log resolved config (stdout logger + artifact).
    cfg_redacted = redact_config(cfg)

    resolved_dump = {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "config_path": str(config_path),
        "metadata": dict(_get_map(cfg, "metadata")),
        "root_image_dir": root_image_dir,
        "root_image_dir_source": root_image_dir_source,
        "detection_template": {
            "id": resolved_detection_template_id,
        },
        "stages": {
            "infer": stages.infer,
            "eval": stages.eval,
            "vis": stages.vis,
        },
        "artifacts": {
            "run_dir": str(artifacts.run_dir),
            "gt_vs_pred_jsonl": str(artifacts.gt_vs_pred_jsonl),
            "pred_token_trace_jsonl": str(artifacts.pred_token_trace_jsonl),
            "gt_vs_pred_scored_jsonl": (
                str(artifacts.gt_vs_pred_scored_jsonl)
                if artifacts.gt_vs_pred_scored_jsonl is not None
                else None
            ),
            "gt_vs_pred_guarded_jsonl": (
                str(artifacts.gt_vs_pred_guarded_jsonl)
                if artifacts.gt_vs_pred_guarded_jsonl is not None
                else None
            ),
            "gt_vs_pred_scored_guarded_jsonl": (
                str(artifacts.gt_vs_pred_scored_guarded_jsonl)
                if artifacts.gt_vs_pred_scored_guarded_jsonl is not None
                else None
            ),
            "summary_json": str(artifacts.summary_json),
            "eval_dir": str(artifacts.eval_dir),
            "metrics_guarded_json": (
                str(artifacts.metrics_guarded_json)
                if artifacts.metrics_guarded_json is not None
                else None
            ),
            "duplicate_guard_report_json": (
                str(artifacts.duplicate_guard_report_json)
                if artifacts.duplicate_guard_report_json is not None
                else None
            ),
            "vis_dir": str(artifacts.vis_dir),
        },
        "infer": {
            "prompt_variant": resolved_prompt_variant,
            "coord_mode": resolved_coord_mode,
            "bbox_format": resolved_bbox_format,
            "detection_template_id": resolved_detection_template_id,
            "object_field_order": resolved_object_field_order,
            "object_ordering": resolved_object_ordering,
            "parsing": {
                "mode": resolved_parser_mode,
            },
            "generation": {},
            "prompt_template_hash": resolved_prompt_hash,
            "runtime_mode": resolved_runtime_mode,
            "mode_resolution_reason": resolved_mode_reason,
            "checkpoint_mode": (
                resolved_checkpoint.checkpoint_mode
                if resolved_checkpoint is not None
                else None
            ),
            "requested_model_checkpoint": (
                resolved_checkpoint.requested_model_checkpoint
                if resolved_checkpoint is not None
                else requested_model_checkpoint
            ),
            "requested_adapter_checkpoint": (
                resolved_checkpoint.requested_adapter_checkpoint
                if resolved_checkpoint is not None
                else requested_adapter_checkpoint
            ),
            "resolved_base_model_checkpoint": (
                resolved_checkpoint.resolved_base_model_checkpoint
                if resolved_checkpoint is not None
                else requested_model_checkpoint
            ),
            "resolved_adapter_checkpoint": (
                resolved_checkpoint.resolved_adapter_checkpoint
                if resolved_checkpoint is not None
                else requested_adapter_checkpoint
            ),
        },
        "inference_provenance": (
            {
                "comparable": resolved_model_identity_fingerprint is not None,
                "missing_provenance_fields": [],
                "invalid_provenance_fields": [],
                "score_policy": "none",
                "prompt_policy_fingerprint": resolved_prompt_policy_fingerprint,
                "decode_policy_fingerprint": (
                    resolved_decode_request.decode_policy_fingerprint
                    if resolved_decode_request is not None
                    else None
                ),
                "model_identity_fingerprint": resolved_model_identity_fingerprint,
            }
            if resolved_decode_request is not None
            else {
                "comparable": False,
                "missing_provenance_fields": [
                    "prompt_policy_fingerprint",
                    "decode_policy_fingerprint",
                    "model_identity_fingerprint",
                ],
                "invalid_provenance_fields": [],
                "score_policy": "none",
            }
        ),
        "eval": {
            "duplicate_control": {
                "enabled": duplicate_control_enabled,
            }
        },
        # Persist a redacted view of the config (avoid leaking secrets into artifacts).
        "cfg": cfg_redacted,
    }
    logger.info("Resolved pipeline config: %s", json.dumps(resolved_dump, indent=2))
    resolved_config_path = artifacts.run_dir / "resolved_config.json"
    resolved_config_path.write_text(
        json.dumps(resolved_dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Persist a manifest pointer next to the unified JSONL artifact so eval/vis can
    # recover the canonical run_dir manifest even when artifacts are laid out outside run_dir.
    artifacts.gt_vs_pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    (artifacts.gt_vs_pred_jsonl.parent / "resolved_config.path").write_text(
        str(resolved_config_path.resolve()),
        encoding="utf-8",
    )

    if resolved_bbox_format in {"cxcy_logw_logh", "cxcywh"}:
        confidence_cfg = _get_map(cfg, "confidence")
        if confidence_cfg:
            raise ValueError(
                f"infer.bbox_format={resolved_bbox_format} does not support confidence post-op in V1; remove the confidence section."
            )

    if stages.infer:
        _run_infer_stage(cfg, artifacts, root_image_dir=root_image_dir)
    else:
        _load_or_raise_artifact(artifacts.gt_vs_pred_jsonl)

    rank, _local_rank, _world_size, distributed_enabled = _detect_infer_distributed_env()
    if distributed_enabled and int(rank) != 0:
        return artifacts

    _maybe_run_confidence_postop(cfg, artifacts)

    if stages.eval:
        _run_eval_stage(cfg, artifacts)

    if stages.vis:
        _run_vis_stage(cfg, artifacts)

    return artifacts


def apply_overrides(
    cfg: Mapping[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply dotted-path overrides into a nested dict (copy-on-write)."""

    def _set(root: Dict[str, Any], dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur: Dict[str, Any] = root
        for p in parts[:-1]:
            nxt = cur.get(p)
            if nxt is None:
                nxt = {}
                cur[p] = nxt
            if not isinstance(nxt, dict):
                raise ValueError(f"cannot override {dotted}: {p} is not a mapping")
            cur = nxt
        cur[parts[-1]] = value

    out: Dict[str, Any] = json.loads(json.dumps(cfg))  # simple deep copy
    for k, v in overrides.items():
        _set(out, k, v)
    return out


def _run_infer_stage(
    cfg: Mapping[str, Any],
    artifacts: ResolvedArtifacts,
    *,
    root_image_dir: Optional[str],
) -> None:
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
    infer_cfg = _get_map(cfg, "infer")
    if not infer_cfg:
        raise ValueError("infer section is required when stages.infer=true")
    _reject_retired_infer_template_knobs(infer_cfg)
    detection_template_id = _resolve_detection_template_id(cfg)
    detection_sequence_format = _detection_sequence_format_for_template_id(
        detection_template_id
    )

    gt_jsonl = _require_str(infer_cfg, "gt_jsonl")
    model_checkpoint = _require_str(infer_cfg, "model_checkpoint")
    adapter_checkpoint = _get_str(infer_cfg, "adapter_checkpoint")
    resolved_checkpoint = resolve_inference_checkpoint(
        model_checkpoint=model_checkpoint,
        adapter_checkpoint=adapter_checkpoint,
    )
    mode_raw = _require_choice(infer_cfg, "mode", {"coord", "text", "auto"})
    requested_mode = cast(Literal["coord", "text", "auto"], mode_raw)
    runtime_mode, mode_resolution_reason = _resolve_infer_runtime_mode(infer_cfg)

    (
        prompt_variant,
        bbox_format,
        object_field_order,
        object_ordering,
    ) = _resolve_infer_prompt_controls(infer_cfg)

    pred_coord_mode_raw = _require_choice(
        infer_cfg, "pred_coord_mode", {"auto", "norm1000", "pixel"}
    )
    pred_coord_mode = cast(
        Literal["auto", "norm1000", "pixel"],
        pred_coord_mode_raw,
    )
    backend_cfg = _get_map(infer_cfg, "backend")
    backend_type_raw = _require_choice(backend_cfg, "type", {"hf", "vllm"})
    backend_type = cast(Literal["hf", "vllm"], backend_type_raw)
    if resolved_checkpoint.resolved_adapter_checkpoint is not None and backend_type != "hf":
        raise ValueError(VLLM_ADAPTER_UNSUPPORTED_MESSAGE)
    validate_compact_coord_token_adapter_contract(
        resolved_checkpoint,
        detection_template_id=detection_template_id,
    )

    gen_cfg_map = _get_map(infer_cfg, "generation")
    if not gen_cfg_map:
        raise ValueError("infer.generation section is required when stages.infer=true")
    decode_request = build_decode_request_from_infer_config(infer_cfg)

    def _i(key: str, default: int) -> int:
        val = gen_cfg_map.get(key, default)
        if val is None:
            return int(default)
        try:
            return int(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"infer.generation.{key} must be an int") from exc

    stop_pressure_cfg = _get_map(gen_cfg_map, "stop_pressure")
    stop_pressure_min_new_tokens_raw = stop_pressure_cfg.get("min_new_tokens", 0)
    if stop_pressure_min_new_tokens_raw is None:
        stop_pressure_min_new_tokens = 0
    else:
        try:
            stop_pressure_min_new_tokens = int(stop_pressure_min_new_tokens_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "infer.generation.stop_pressure.min_new_tokens must be an int"
            ) from exc
    stop_pressure_mode = _get_str(stop_pressure_cfg, "mode")
    stop_pressure_trigger_rule = _get_str(stop_pressure_cfg, "trigger_rule")
    stop_pressure_logit_bias_raw = stop_pressure_cfg.get("logit_bias", 0.0)
    if stop_pressure_logit_bias_raw is None:
        stop_pressure_logit_bias = 0.0
    else:
        try:
            stop_pressure_logit_bias = float(stop_pressure_logit_bias_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "infer.generation.stop_pressure.logit_bias must be a float"
            ) from exc

    generation_kwargs = legacy_generation_kwargs_from_decode_request(
        decode_request,
        batch_size=_i("batch_size", 1),
        stop_pressure_mode=stop_pressure_mode,
        stop_pressure_min_new_tokens=stop_pressure_min_new_tokens,
        stop_pressure_trigger_rule=stop_pressure_trigger_rule,
        stop_pressure_logit_bias=stop_pressure_logit_bias,
    )
    if stop_pressure_mode not in (
        None,
        STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN,
        STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT,
        STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN,
        STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE,
        STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY,
        STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY,
        STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY,
        STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY,
    ):
        raise ValueError(
            "infer.generation.stop_pressure.mode must be "
            "'min_new_tokens_after_object_open' or "
            "'steer_bbox_tail_closure_to_next_object' or "
            "'steer_bbox_tail_then_object_open' or "
            "'steer_bbox_tail_then_object_open_once' or "
            "'steer_first_array_branch_to_next_object_after_object_boundary' or "
            "'suppress_first_structural_closure_after_object_boundary' or "
            "'suppress_special_terminating_tokens_after_object_boundary' or "
            "'suppress_terminating_tokens_after_object_boundary'"
        )
    if stop_pressure_trigger_rule not in (
        None,
        STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_OPEN,
        STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY,
    ):
        raise ValueError(
            "infer.generation.stop_pressure.trigger_rule must be "
            "'raw_text_object_open' or 'raw_text_object_boundary'"
        )
    if stop_pressure_mode is None:
        if (
            int(stop_pressure_min_new_tokens) != 0
            or stop_pressure_trigger_rule is not None
            or float(stop_pressure_logit_bias) != 0.0
        ):
            raise ValueError(
                "infer.generation.stop_pressure.mode is required when "
                "stop-pressure fields are set"
            )
    else:
        if backend_type != "hf":
            raise ValueError(
                "infer.generation.stop_pressure is only supported for "
                "infer.backend.type=hf"
            )
        if (
            stop_pressure_mode == STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN
            and int(stop_pressure_min_new_tokens) <= 0
        ):
            raise ValueError(
                "infer.generation.stop_pressure.min_new_tokens must be > 0 "
                "when stop pressure is enabled"
            )
        if (
            stop_pressure_mode == STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN
            and stop_pressure_trigger_rule != STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_OPEN
        ):
            raise ValueError(
                "infer.generation.stop_pressure.trigger_rule must be "
                "'raw_text_object_open' when stop pressure is enabled"
            )
        if (
            stop_pressure_mode
            in (
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT,
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN,
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE,
                STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY,
                STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY,
                STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY,
                STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY,
            )
            and int(stop_pressure_min_new_tokens) != 0
        ):
            raise ValueError(
                "infer.generation.stop_pressure.min_new_tokens must be 0 "
                "for raw-text boundary stop pressure"
            )
        if (
            stop_pressure_mode
            in (
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT,
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN,
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE,
                STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY,
                STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY,
                STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY,
                STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY,
            )
            and stop_pressure_trigger_rule
            != STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY
        ):
            raise ValueError(
                "infer.generation.stop_pressure.trigger_rule must be "
                "'raw_text_object_boundary' for raw-text boundary stop pressure"
            )
        if (
            stop_pressure_mode
            in (
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT,
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN,
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE,
                STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY,
            )
            and float(stop_pressure_logit_bias) <= 0.0
        ):
            raise ValueError(
                "infer.generation.stop_pressure.logit_bias must be > 0 "
                "for continuation steering stop pressure"
            )
        if (
            stop_pressure_mode
            not in (
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT,
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN,
                STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE,
                STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY,
            )
            and float(stop_pressure_logit_bias) != 0.0
        ):
            raise ValueError(
                "infer.generation.stop_pressure.logit_bias must be 0 "
                "unless continuation steering stop pressure is enabled"
            )

    rank, local_rank, world_size, distributed_enabled = _detect_infer_distributed_env()
    device = str(_get_str(infer_cfg, "device", "cuda:0") or "cuda:0").strip() or "cuda:0"
    if bool(distributed_enabled) and device.startswith("cuda"):
        device = f"cuda:{int(local_rank)}"

    inference_kwargs = {
        "gt_jsonl": gt_jsonl,
        "model_checkpoint": model_checkpoint,
        "adapter_checkpoint": adapter_checkpoint,
        "checkpoint_mode": resolved_checkpoint.checkpoint_mode,
        "requested_model_checkpoint": resolved_checkpoint.requested_model_checkpoint,
        "requested_adapter_checkpoint": resolved_checkpoint.requested_adapter_checkpoint,
        "resolved_base_model_checkpoint": resolved_checkpoint.resolved_base_model_checkpoint,
        "resolved_adapter_checkpoint": resolved_checkpoint.resolved_adapter_checkpoint,
        "mode": runtime_mode,
        "requested_mode": requested_mode,
        "mode_resolution_reason": mode_resolution_reason,
        "prompt_variant": prompt_variant,
        "bbox_format": bbox_format,
        "detection_template_id": detection_template_id,
        "detection_sequence_format": detection_sequence_format,
        "object_field_order": object_field_order,
        "object_ordering": object_ordering,
        "allow_diagnostic_gt_vs_pred": _get_bool(
            infer_cfg,
            "allow_diagnostic_gt_vs_pred",
            False,
        ),
        "pred_coord_mode": pred_coord_mode,
        "out_path": str(artifacts.gt_vs_pred_jsonl),
        "pred_token_trace_path": str(artifacts.pred_token_trace_jsonl),
        "summary_path": str(artifacts.summary_json),
        "root_image_dir": str(root_image_dir) if root_image_dir else None,
        "device": device,
        "limit": _get_limit(infer_cfg, "limit", 0),
        "backend_type": backend_type,
        "backend": dict(backend_cfg) if backend_cfg else {},
        "prompt_policy_fingerprint": _offline_prompt_policy_fingerprint(
            coord_mode=coord_mode_from_coord_tokens_enabled(runtime_mode == "coord"),
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
            bbox_format=bbox_format,
            detection_template_id=detection_template_id,
            object_ordering=object_ordering,
        ),
        "decode_policy_fingerprint": decode_request.decode_policy_fingerprint,
        "model_identity_fingerprint": build_model_identity_fingerprint(
            checkpoint_mode=resolved_checkpoint.checkpoint_mode,
            requested_model_checkpoint=resolved_checkpoint.requested_model_checkpoint,
            requested_adapter_checkpoint=(
                resolved_checkpoint.requested_adapter_checkpoint
            ),
            resolved_base_model_checkpoint=(
                resolved_checkpoint.resolved_base_model_checkpoint
            ),
            resolved_adapter_checkpoint=(
                resolved_checkpoint.resolved_adapter_checkpoint
            ),
            backend=decode_request.backend,
            backend_mode=decode_request.backend_mode,
            backend_model=_get_str(backend_cfg, "model"),
        ),
        "detect_samples": _get_int(infer_cfg, "detect_samples", 128),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "distributed_enabled": distributed_enabled,
    }

    run_offline_inference(
        inference_kwargs=inference_kwargs,
        generation_kwargs=generation_kwargs,
        logger=logger,
    )


def _maybe_run_confidence_postop(
    cfg: Mapping[str, Any],
    artifacts: ResolvedArtifacts,
) -> None:
    from src.eval.confidence_postop import (
        options_from_config,
        paths_from_config,
        run_confidence_postop,
    )

    infer_cfg = _get_map(cfg, "infer")
    bbox_format = normalize_bbox_format(
        infer_cfg.get("bbox_format", "xyxy"),
        path="infer.bbox_format",
    )
    confidence_cfg = _get_map(cfg, "confidence")
    eval_cfg = _get_map(cfg, "eval")
    stages_cfg = _get_map(cfg, "stages")
    eval_enabled = _get_bool(stages_cfg, "eval", False) if stages_cfg else False
    metrics_mode = str(eval_cfg.get("metrics", "both")).strip().lower()
    want_scored = bool(confidence_cfg) or (
        eval_enabled and metrics_mode in {"coco", "lvis", "both"}
    )
    # Non-xyxy prepared bbox formats use a deterministic constant-score scored artifact for
    # official evaluation, so infer-only runs still need scored materialization.
    if not want_scored and bbox_format in {"cxcy_logw_logh", "cxcywh"}:
        want_scored = True
    if not want_scored:
        return
    base_path = artifacts.gt_vs_pred_jsonl
    trace_path = artifacts.pred_token_trace_jsonl
    scored_path = artifacts.gt_vs_pred_scored_jsonl
    if scored_path is None:
        return
    if bbox_format in {"cxcy_logw_logh", "cxcywh"}:
        if not base_path.is_file():
            return
        pred_score_source = (
            CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_SOURCE
            if bbox_format == "cxcy_logw_logh"
            else CXCYWH_CONSTANT_PRED_SCORE_SOURCE
        )
        pred_score_version = (
            CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_VERSION
            if bbox_format == "cxcy_logw_logh"
            else CXCYWH_CONSTANT_PRED_SCORE_VERSION
        )
        constant_score = (
            CXCY_LOGW_LOGH_CONSTANT_SCORE
            if bbox_format == "cxcy_logw_logh"
            else CXCYWH_CONSTANT_SCORE
        )
        newest_input_mtime = base_path.stat().st_mtime
        if scored_path.is_file() and scored_path.stat().st_mtime >= newest_input_mtime:
            try:
                load_comparable_artifact(scored_path, require_score=True)
                return
            except ValueError:
                logger.info(
                    "Recomputing constant-score artifact because %s lacks valid "
                    "score provenance for the current pipeline contract.",
                    scored_path,
                )
        rows: List[Dict[str, Any]] = []
        with base_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        write_jsonl_records(
            scored_path,
            with_constant_scores(
                records=rows,
                pred_score_source=pred_score_source,
                pred_score_version=pred_score_version,
                constant_score=constant_score,
            ),
        )
        _write_scored_artifact_provenance(
            cfg=cfg,
            raw_path=base_path,
            scored_path=scored_path,
            policy_name="constant_score",
            score_source=f"{pred_score_source}:v{pred_score_version}",
            aggregation_rule="constant_per_prediction",
            token_span_rule="none",
            constant_score_value=constant_score,
        )
        return
    if not base_path.is_file():
        return
    if not trace_path.is_file():
        logger.info(
            "Skipping confidence post-op: token trace artifact is missing at %s",
            trace_path,
        )
        return

    pred_confidence_jsonl = artifacts.run_dir / "pred_confidence.jsonl"
    confidence_postop_summary_json = artifacts.run_dir / "confidence_postop_summary.json"
    newest_input_mtime = max(base_path.stat().st_mtime, trace_path.stat().st_mtime)
    if (
        scored_path.is_file()
        and pred_confidence_jsonl.is_file()
        and confidence_postop_summary_json.is_file()
        and min(
            scored_path.stat().st_mtime,
            pred_confidence_jsonl.stat().st_mtime,
            confidence_postop_summary_json.stat().st_mtime,
        )
        >= newest_input_mtime
    ):
        try:
            load_comparable_artifact(scored_path, require_score=True)
            return
        except ValueError:
            logger.info(
                "Recomputing confidence scored artifact because %s lacks valid "
                "score provenance for the current pipeline contract.",
                scored_path,
            )

    postop_cfg = {
        "confidence": dict(confidence_cfg) if confidence_cfg else {},
        "artifacts": {
            "run_dir": str(artifacts.run_dir),
            "gt_vs_pred_jsonl": str(base_path),
            "pred_token_trace_jsonl": str(trace_path),
            "pred_confidence_jsonl": str(pred_confidence_jsonl),
            "gt_vs_pred_scored_jsonl": str(scored_path),
            "confidence_postop_summary_json": str(confidence_postop_summary_json),
        },
    }
    logger.info(
        "Running confidence post-op to materialize scored detections at %s",
        scored_path,
    )
    summary = run_confidence_postop(
        paths_from_config(postop_cfg),
        options=options_from_config(postop_cfg),
    )
    _write_scored_artifact_provenance(
        cfg=cfg,
        raw_path=base_path,
        scored_path=scored_path,
        policy_name=str(summary.get("confidence_method") or "confidence_postop"),
        score_source=(
            f"{summary.get('pred_score_source', 'confidence_postop')}:"
            f"v{summary.get('pred_score_version', 2)}"
        ),
        aggregation_rule="bbox_logprob_confidence_exp",
        token_span_rule="generated_token_trace_bbox_and_desc_spans",
        constant_score_value=None,
    )


def _run_eval_stage(cfg: Mapping[str, Any], artifacts: ResolvedArtifacts) -> None:
    from src.eval.detection import EvalOptions, evaluate_and_save

    eval_cfg = _get_map(cfg, "eval")
    duplicate_control_enabled = _resolve_eval_duplicate_control_enabled(eval_cfg)

    deprecated_keys = [
        k for k in ("unknown_policy", "semantic_fallback") if k in eval_cfg
    ]
    if deprecated_keys:
        rendered = ", ".join(f"eval.{k}" for k in deprecated_keys)
        raise ValueError(
            f"Deprecated evaluation keys are unsupported: {rendered}. "
            "Remove these keys to continue."
        )

    if "use_pred_score" in eval_cfg:
        raise ValueError(
            "eval.use_pred_score is unsupported. Fixed-score evaluation has been removed; "
            "remove eval.use_pred_score and run score-aware evaluation."
        )

    metrics_mode = str(eval_cfg.get("metrics", "both")).strip().lower()
    want_official = metrics_mode in {"coco", "lvis", "both"}

    # Unified pipeline contract:
    # - COCO metrics require scored artifacts.
    # - cxcy_logw_logh uses a constant-score scored artifact for official eval.
    # - f1ish-only runs can use the base artifact.
    if want_official:
        scored_path = artifacts.gt_vs_pred_scored_jsonl
        if scored_path is None:
            raise ValueError(
                "Official detection evaluation requires artifacts.gt_vs_pred_scored_jsonl. "
                "Run the confidence post-op first and configure this path."
            )
        if not scored_path.is_file():
            raise ValueError(
                "Official detection evaluation requires a scored artifact, but "
                f"{scored_path} does not exist. Auto confidence post-op requires "
                "artifacts.pred_token_trace_jsonl to be present; otherwise provide "
                "artifacts.gt_vs_pred_scored_jsonl explicitly."
            )
        load_comparable_artifact(scored_path, require_score=True)
        pred_path = scored_path
        guarded_pred_path = artifacts.gt_vs_pred_scored_guarded_jsonl
        active_input_family = "scored"
    else:
        pred_path = _load_or_raise_artifact(artifacts.gt_vs_pred_jsonl)
        guarded_pred_path = artifacts.gt_vs_pred_guarded_jsonl
        active_input_family = "raw"

    if duplicate_control_enabled and guarded_pred_path is None:
        raise ValueError(
            "Duplicate-control evaluation requires a resolved guarded prediction artifact path."
        )

    logger.info(
        "Resolved eval duplicate-control: enabled=%s input_family=%s guarded_pred_path=%s metrics_guarded_json=%s duplicate_guard_report_json=%s",
        duplicate_control_enabled,
        active_input_family,
        str(guarded_pred_path) if guarded_pred_path is not None else None,
        str(artifacts.metrics_guarded_json)
        if artifacts.metrics_guarded_json is not None
        else None,
        str(artifacts.duplicate_guard_report_json)
        if artifacts.duplicate_guard_report_json is not None
        else None,
    )

    options = EvalOptions(
        metrics=str(eval_cfg.get("metrics", "both")),
        strict_parse=bool(eval_cfg.get("strict_parse", False)),
        use_segm=bool(eval_cfg.get("use_segm", True)),
        iou_thrs=eval_cfg.get("iou_thrs", None),
        f1ish_iou_thrs=[
            float(x) for x in (eval_cfg.get("f1ish_iou_thrs", [0.3, 0.5]) or [])
        ],
        f1ish_pred_scope=str(eval_cfg.get("f1ish_pred_scope", "annotated")),
        output_dir=artifacts.eval_dir,
        overlay=bool(eval_cfg.get("overlay", False)),
        overlay_k=int(eval_cfg.get("overlay_k", 12)),
        num_workers=int(eval_cfg.get("num_workers", 0)),
        semantic_model=str(
            eval_cfg.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        semantic_threshold=float(eval_cfg.get("semantic_threshold", 0.6)),
        semantic_device=str(eval_cfg.get("semantic_device", "auto")),
        semantic_batch_size=int(eval_cfg.get("semantic_batch_size", 64)),
        lvis_max_dets=int(eval_cfg.get("lvis_max_dets", 300)),
        duplicate_control_enabled=duplicate_control_enabled,
        guarded_pred_path=guarded_pred_path,
        duplicate_guard_report_path=artifacts.duplicate_guard_report_json,
    )

    # evaluate_and_save() owns eval/* outputs (including metrics.json with counters).
    evaluate_and_save(pred_path, options=options)


def _run_vis_stage(cfg: Mapping[str, Any], artifacts: ResolvedArtifacts) -> None:
    from src.infer.vis import render_vis_from_jsonl

    vis_cfg = _get_map(cfg, "vis")
    pred_path = _load_or_raise_artifact(artifacts.gt_vs_pred_jsonl)

    root_image_dir_str, root_source = _resolve_root_image_dir(cfg)
    root_image_dir = Path(root_image_dir_str) if root_image_dir_str else None

    limit = int(vis_cfg.get("limit", 20))
    render_vis_from_jsonl(
        pred_path,
        out_dir=artifacts.vis_dir,
        limit=limit,
        root_image_dir=root_image_dir,
        root_source=root_source,
    )
