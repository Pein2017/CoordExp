"""Unified inference pipeline runner (infer -> eval and/or vis).

This module is intentionally YAML-first and does NOT use the training config
loader (no extends/inherit, no interpolation), per OpenSpec.

Primary entrypoint is `scripts/run_infer.py --config <yaml>`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, cast

from src.utils import get_logger

logger = get_logger(__name__)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
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


def _get_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    val = cfg.get(key, default)
    try:
        return int(val)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{key} must be an int") from exc


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
    summary_json: Path
    eval_dir: Path
    vis_dir: Path


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

    summary_json = Path(_get_str(art_cfg, "summary_json") or (run_dir / "summary.json"))

    eval_cfg = _get_map(cfg, "eval")
    vis_cfg = _get_map(cfg, "vis")

    eval_dir = Path(_get_str(eval_cfg, "output_dir") or (run_dir / "eval"))
    vis_dir = Path(_get_str(vis_cfg, "output_dir") or (run_dir / "vis"))

    return (
        ResolvedArtifacts(
            run_dir=run_dir,
            gt_vs_pred_jsonl=gt_vs_pred_jsonl,
            summary_json=summary_json,
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
                pointed = (pointer_path.parent / pointed).resolve()
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
            if Path(gt_vs_pred_jsonl).resolve() == jsonl_resolved:
                return resolved

        run_dir = artifacts.get("run_dir")
        if isinstance(run_dir, str) and run_dir.strip():
            run_dir_resolved = Path(run_dir).resolve()
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

    artifacts, stages = resolve_artifacts(cfg)

    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts.eval_dir.mkdir(parents=True, exist_ok=True)
    artifacts.vis_dir.mkdir(parents=True, exist_ok=True)

    root_image_dir, root_image_dir_source = _resolve_root_image_dir(cfg)

    # Log resolved config (stdout logger + artifact).
    cfg_redacted = redact_config(cfg)

    resolved_dump = {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "config_path": str(config_path),
        "root_image_dir": root_image_dir,
        "root_image_dir_source": root_image_dir_source,
        "stages": {
            "infer": stages.infer,
            "eval": stages.eval,
            "vis": stages.vis,
        },
        "artifacts": {
            "run_dir": str(artifacts.run_dir),
            "gt_vs_pred_jsonl": str(artifacts.gt_vs_pred_jsonl),
            "summary_json": str(artifacts.summary_json),
            "eval_dir": str(artifacts.eval_dir),
            "vis_dir": str(artifacts.vis_dir),
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

    if stages.infer:
        _run_infer_stage(cfg, artifacts, root_image_dir=root_image_dir)
    else:
        _load_or_raise_artifact(artifacts.gt_vs_pred_jsonl)

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
    from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine

    infer_cfg = _get_map(cfg, "infer")
    if not infer_cfg:
        raise ValueError("infer section is required when stages.infer=true")

    gt_jsonl = _require_str(infer_cfg, "gt_jsonl")
    model_checkpoint = _require_str(infer_cfg, "model_checkpoint")
    mode_raw = _require_choice(infer_cfg, "mode", {"coord", "text", "auto"})
    mode = cast(Literal["coord", "text", "auto"], mode_raw)

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

    gen_cfg_map = _get_map(infer_cfg, "generation")
    if not gen_cfg_map:
        raise ValueError("infer.generation section is required when stages.infer=true")

    def _f(key: str, default: float) -> float:
        val = gen_cfg_map.get(key, default)
        if val is None:
            return float(default)
        try:
            return float(val)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"infer.generation.{key} must be a float") from exc

    def _i(key: str, default: int) -> int:
        val = gen_cfg_map.get(key, default)
        if val is None:
            return int(default)
        try:
            return int(val)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"infer.generation.{key} must be an int") from exc

    seed_val = gen_cfg_map.get("seed", None)
    seed = int(seed_val) if seed_val is not None else None

    gen_cfg = GenerationConfig(
        temperature=_f("temperature", 0.01),
        top_p=_f("top_p", 0.95),
        max_new_tokens=_i("max_new_tokens", 1024),
        repetition_penalty=_f("repetition_penalty", 1.05),
        batch_size=_i("batch_size", 1),
        seed=seed,
    )

    inf_cfg = InferenceConfig(
        gt_jsonl=gt_jsonl,
        model_checkpoint=model_checkpoint,
        mode=mode,
        pred_coord_mode=pred_coord_mode,
        out_path=str(artifacts.gt_vs_pred_jsonl),
        summary_path=str(artifacts.summary_json),
        root_image_dir=str(root_image_dir) if root_image_dir else None,
        device=str(_get_str(infer_cfg, "device", "cuda:0") or "cuda:0"),
        limit=_get_int(infer_cfg, "limit", 0),
        backend_type=backend_type,
        backend=dict(backend_cfg) if backend_cfg else {},
        detect_samples=_get_int(infer_cfg, "detect_samples", 128),
    )

    engine = InferenceEngine(inf_cfg, gen_cfg)
    engine.infer()


def _run_eval_stage(cfg: Mapping[str, Any], artifacts: ResolvedArtifacts) -> None:
    from src.eval.detection import EvalOptions, evaluate_and_save

    eval_cfg = _get_map(cfg, "eval")

    deprecated_keys = [
        k for k in ("unknown_policy", "semantic_fallback") if k in eval_cfg
    ]
    if deprecated_keys:
        rendered = ", ".join(f"eval.{k}" for k in deprecated_keys)
        raise ValueError(
            f"Deprecated evaluation keys are unsupported: {rendered}. "
            "Remove these keys to continue."
        )

    # Unified pipeline contract: evaluator consumes artifact with embedded GT.
    pred_path = _load_or_raise_artifact(artifacts.gt_vs_pred_jsonl)

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
