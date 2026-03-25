from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


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

    pipeline_raw = cfg.get("pipeline", None)
    if not isinstance(pipeline_raw, Mapping):
        variant = str(trainer_variant or "")
        if variant in {"stage2_two_channel", "stage2_rollout_aligned"}:
            raise ValueError(
                f"{variant} requires an explicit pipeline config; missing {variant}.*.pipeline."
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

    def _normalize_channels(channels_raw: Any) -> list[str]:
        found: set[str] = set()
        if isinstance(channels_raw, Sequence) and not isinstance(
            channels_raw, (str, bytes)
        ):
            for ch in channels_raw:
                ch_s = str(ch).strip().upper()
                if ch_s in {"A", "B"}:
                    found.add(ch_s)
        if not found:
            return ["A", "B"]
        return [ch for ch in ("A", "B") if ch in found]

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

    def _coord_soft_defaults() -> dict[str, Any]:
        enabled = bool(coord_soft_cfg.get("enabled", False))
        soft_default = 1.0 if enabled else 0.0
        w1_default = 1.0 if enabled else 0.0
        gate_default = 1.0 if enabled else 0.0
        return {
            "coord_ce_weight": _finite_float(coord_soft_cfg.get("ce_weight", 0.0), 0.0),
            "soft_ce_weight": _finite_float(
                coord_soft_cfg.get("soft_ce_weight", soft_default),
                soft_default,
            ),
            "w1_weight": _finite_float(
                coord_soft_cfg.get("w1_weight", w1_default),
                w1_default,
            ),
            "coord_gate_weight": _finite_float(
                coord_soft_cfg.get("gate_weight", gate_default),
                gate_default,
            ),
            "temperature": _finite_float(coord_soft_cfg.get("temperature", 1.0), 1.0),
            "target_sigma": _finite_float(coord_soft_cfg.get("target_sigma", 2.0), 2.0),
            "target_truncate": coord_soft_cfg.get("target_truncate", None),
        }

    def _default_module_config(name: str) -> dict[str, Any]:
        variant = str(trainer_variant or "")
        coord_soft_defaults = _coord_soft_defaults()

        if variant == "stage2_two_channel":
            desc_w = _finite_float(cfg.get("desc_ce_weight", 1.0), 1.0)

            if name == "token_ce":
                return {
                    "desc_ce_weight": desc_w,
                    "rollout_fn_desc_weight": desc_w,
                    "rollout_global_prefix_struct_ce_weight": 1.0,
                }

            if name == "loss_duplicate_burst_unlikelihood":
                return {}

            if name == "bbox_geo":
                return {
                    "smoothl1_weight": _finite_float(
                        cfg.get("bbox_smoothl1_weight", 1.0),
                        1.0,
                    ),
                    "ciou_weight": _finite_float(
                        cfg.get("bbox_ciou_weight", 1.0),
                        1.0,
                    ),
                }

            if name == "bbox_size_aux":
                return {}

            if name == "coord_reg":
                return {
                    "coord_ce_weight": _finite_float(
                        cfg.get(
                            "coord_ce_weight",
                            coord_soft_defaults.get("coord_ce_weight", 0.0),
                        ),
                        0.0,
                    ),
                    "coord_gate_weight": _finite_float(
                        cfg.get(
                            "coord_gate_weight",
                            coord_soft_defaults.get("coord_gate_weight", 0.0),
                        ),
                        0.0,
                    ),
                    "text_gate_weight": _finite_float(cfg.get("text_gate_weight", 0.0), 0.0),
                    "soft_ce_weight": _finite_float(
                        coord_soft_defaults.get("soft_ce_weight", 0.0),
                        0.0,
                    ),
                    "w1_weight": _finite_float(
                        coord_soft_defaults.get("w1_weight", 0.0),
                        0.0,
                    ),
                    "temperature": _finite_float(
                        coord_soft_defaults.get("temperature", 1.0),
                        1.0,
                    ),
                    "target_sigma": _finite_float(
                        coord_soft_defaults.get("target_sigma", 2.0),
                        2.0,
                    ),
                    "target_truncate": coord_soft_defaults.get("target_truncate", None),
                }

        if variant == "stage2_rollout_aligned":
            if name == "token_ce":
                return {
                    "rollout_fn_desc_weight": 1.0,
                    "rollout_global_prefix_struct_ce_weight": 1.0,
                }

            if name == "bbox_geo":
                return {
                    "smoothl1_weight": _finite_float(
                        cfg.get("bbox_smoothl1_weight", 1.0),
                        1.0,
                    ),
                    "ciou_weight": _finite_float(
                        cfg.get("bbox_ciou_weight", 1.0),
                        1.0,
                    ),
                }

            if name == "bbox_size_aux":
                return {}

            if name == "coord_reg":
                return {
                    "coord_ce_weight": _finite_float(
                        coord_soft_defaults.get("coord_ce_weight", 0.0),
                        0.0,
                    ),
                    "soft_ce_weight": _finite_float(
                        coord_soft_defaults.get("soft_ce_weight", 0.0),
                        0.0,
                    ),
                    "w1_weight": _finite_float(
                        coord_soft_defaults.get("w1_weight", 0.0),
                        0.0,
                    ),
                    "coord_gate_weight": _finite_float(
                        coord_soft_defaults.get("coord_gate_weight", 0.0),
                        0.0,
                    ),
                    "text_gate_weight": 0.0,
                    "temperature": _finite_float(
                        coord_soft_defaults.get("temperature", 1.0),
                        1.0,
                    ),
                    "target_sigma": _finite_float(
                        coord_soft_defaults.get("target_sigma", 2.0),
                        2.0,
                    ),
                    "target_truncate": coord_soft_defaults.get("target_truncate", None),
                }

        return {}

    def _resolve(path: str, defaults: list[str]) -> list[dict[str, Any]]:
        raw = pipeline_raw.get(path, None)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            variant = str(trainer_variant or "")
            if variant in {"stage2_two_channel", "stage2_rollout_aligned"}:
                raise TypeError(f"pipeline.{path} must be a list of module specs")
            raw = None

        if raw is None:
            return []

        out: list[dict[str, Any]] = []
        for spec in raw:
            if not isinstance(spec, Mapping):
                continue
            name = str(spec.get("name", "") or "").strip()
            if not name:
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

            out.append(
                {
                    "name": name,
                    "enabled": bool(spec.get("enabled", True)),
                    "weight": max(0.0, _finite_float(spec.get("weight", 1.0), 1.0)),
                    "channels": _normalize_channels(spec.get("channels", ["A", "B"])),
                    "application": authored_app,
                    "config": merged_cfg,
                }
            )

        return out

    objective = _resolve("objective", default_objective)
    diagnostics = _resolve("diagnostics", default_diagnostics)

    extra: dict[str, Any] = {"variant": str(trainer_variant or "")}

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
