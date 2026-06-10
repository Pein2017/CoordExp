from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.training_runtime import resolve_training_runtime_profile


def build_pipeline_manifest(
    cfg: Mapping[str, Any] | None,
    *,
    default_objective: list[str],
    default_diagnostics: list[str],
    trainer_variant: str,
    config_path: str,
    run_name: str,
    seed: int,
    coord_soft_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        cfg = {}
    if not isinstance(coord_soft_cfg, Mapping):
        coord_soft_cfg = {}

    runtime_profile = resolve_training_runtime_profile(trainer_variant)

    pipeline_raw = cfg.get("pipeline", None)
    if not isinstance(pipeline_raw, Mapping):
        if runtime_profile.explicit_pipeline_required:
            required_namespace = (
                runtime_profile.required_pipeline_namespace or "pipeline"
            )
            raise ValueError(
                f"{runtime_profile.variant} requires an explicit pipeline config; missing {required_namespace}."
            )
        pipeline_raw = {}

    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _finite_float(value: Any, default: float) -> float:
        out = _coerce_float(value, default)
        if not math.isfinite(out):
            raise ValueError("pipeline manifest contains non-finite float (NaN/Inf)")
        if out == 0.0:
            return 0.0
        return float(out)

    def _normalize_json_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for k in sorted(value.keys(), key=lambda x: str(x)):
                out[str(k)] = _normalize_json_value(value[k])
            return out
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_normalize_json_value(v) for v in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return _finite_float(value, 0.0)
        try:
            f = float(value)
        except (TypeError, ValueError):
            return value
        return _finite_float(f, 0.0)

    def _default_module_config(name: str) -> dict[str, Any]:
        return {}

    def _resolve(path: str, defaults: list[str]) -> list[dict[str, Any]]:
        raw = pipeline_raw.get(path, None)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            if runtime_profile.explicit_pipeline_required:
                raise TypeError(f"pipeline.{path} must be a list of module specs")
            raw = None

        if raw is None:
            return []

        out: list[dict[str, Any]] = []
        for idx, spec in enumerate(raw):
            if not isinstance(spec, Mapping):
                if runtime_profile.explicit_pipeline_required:
                    raise TypeError(
                        f"pipeline.{path}[{idx}] must be a mapping module spec"
                    )
                continue
            name = str(spec.get("name", "") or "").strip()
            if not name:
                if runtime_profile.explicit_pipeline_required:
                    raise ValueError(
                        f"pipeline.{path}[{idx}].name must be non-empty"
                    )
                continue
            authored_cfg_raw = spec.get("config", {})
            authored_cfg = (
                dict(authored_cfg_raw) if isinstance(authored_cfg_raw, Mapping) else {}
            )
            authored_app_raw = spec.get("application", {})
            authored_app = (
                dict(authored_app_raw) if isinstance(authored_app_raw, Mapping) else {}
            )
            merged_cfg = dict(_default_module_config(name))
            merged_cfg.update(authored_cfg)

            entry = {
                "name": name,
                "enabled": bool(spec.get("enabled", True)),
                "weight": max(0.0, _finite_float(spec.get("weight", 1.0), 1.0)),
                "application": authored_app,
                "config": merged_cfg,
            }
            out.append(entry)

        return out

    objective = _resolve("objective", default_objective)
    diagnostics = _resolve("diagnostics", default_diagnostics)

    extra: dict[str, Any] = {"variant": runtime_profile.variant}
    if coord_soft_cfg:
        extra["coord_soft_ce"] = dict(coord_soft_cfg)

    payload = _normalize_json_value(
        {"objective": objective, "diagnostics": diagnostics, "extra": extra}
    )
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_context = {
        "config": str(config_path),
        "run_name": str(run_name or ""),
        "seed": int(seed or 0),
    }
    return {
        "payload": payload,
        "objective": payload.get("objective", []),
        "diagnostics": payload.get("diagnostics", []),
        "extra": payload.get("extra", {}),
        "checksum": checksum,
        "run_context": run_context,
    }
