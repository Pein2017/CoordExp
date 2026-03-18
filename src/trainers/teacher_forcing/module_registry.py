"""Unified teacher-forcing module registry (names + config allowlists).

This module is intentionally dependency-light so it can be imported from config
schema validation without pulling in torch/trainer runtime code.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

ALLOWED_OBJECTIVE_MODULES: Final[set[str]] = {
    "token_ce",
    "loss_dead_anchor_suppression",
    "bbox_geo",
    "bbox_size_aux",
    "coord_reg",
}
ALLOWED_DIAGNOSTIC_MODULES: Final[set[str]] = {"coord_diag"}

OBJECTIVE_CONFIG_ALLOWLIST: Final[dict[str, set[str]]] = {
    "token_ce": {
        "desc_ce_weight",
        "struct_ce_weight",
        "rollout_fn_desc_weight",
        "rollout_matched_prefix_struct_weight",
        "stop_signal_damping",
    },
    "loss_dead_anchor_suppression": set(),
    "bbox_geo": {
        "smoothl1_weight",
        "ciou_weight",
    },
    "bbox_size_aux": {
        "log_wh_weight",
        "oversize_penalty_weight",
        "oversize_area_frac_threshold",
        "oversize_log_w_threshold",
        "oversize_log_h_threshold",
        "eps",
    },
    "coord_reg": {
        "coord_ce_weight",
        "coord_el1_weight",
        "coord_ehuber_weight",
        "coord_huber_delta",
        "coord_entropy_weight",
        "coord_gate_weight",
        "text_gate_weight",
        "soft_ce_weight",
        "w1_weight",
        "temperature",
        "target_sigma",
        "target_truncate",
    },
}

OBJECTIVE_OPTIONAL_CONFIG_KEYS: Final[dict[str, set[str]]] = {
    "token_ce": {"stop_signal_damping"},
}

OBJECTIVE_APPLICATION_PRESET_ALLOWLIST: Final[dict[str, set[str]]] = {
    "token_ce": {
        "anchor_text_plus_final_struct",
        "anchor_text_only",
        "rollout_text_only",
    },
    "loss_dead_anchor_suppression": {"rollout_only"},
    "bbox_geo": {
        "anchor_if_single_iter_else_final",
        "anchor_only",
        "final_only",
        "anchor_and_final",
    },
    "bbox_size_aux": {
        "anchor_if_single_iter_else_final",
        "anchor_only",
        "final_only",
        "anchor_and_final",
    },
    "coord_reg": {
        "anchor_if_single_iter_else_final",
        "anchor_only",
        "final_only",
        "anchor_and_final",
    },
}

DIAGNOSTIC_CONFIG_ALLOWLIST: Final[dict[str, set[str]]] = {
    "coord_diag": set(),
}

TOKEN_CE_STOP_SIGNAL_DAMPING_ALLOWED_KEYS: Final[set[str]] = {
    "enabled",
    "min_weight",
    "max_weight",
    "branch_temperature",
    "curve_gamma",
    "detach_gate",
}

TOKEN_CE_STOP_SIGNAL_DAMPING_DEFAULTS: Final[dict[str, Any]] = {
    "enabled": False,
    "min_weight": 0.2,
    "max_weight": 1.0,
    "branch_temperature": 1.0,
    "curve_gamma": 2.0,
    "detach_gate": True,
}


def normalize_token_ce_stop_signal_damping_config(
    value: Any,
    *,
    path: str,
) -> dict[str, Any]:
    if value is None:
        return dict(TOKEN_CE_STOP_SIGNAL_DAMPING_DEFAULTS)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping when provided")

    payload = dict(value)
    unknown = set(payload.keys()) - set(TOKEN_CE_STOP_SIGNAL_DAMPING_ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown {path} keys: {sorted(str(k) for k in unknown)}; "
            f"allowed={sorted(TOKEN_CE_STOP_SIGNAL_DAMPING_ALLOWED_KEYS)}"
        )

    out = dict(TOKEN_CE_STOP_SIGNAL_DAMPING_DEFAULTS)
    out.update(payload)

    out["enabled"] = bool(out.get("enabled", False))
    out["detach_gate"] = bool(out.get("detach_gate", True))

    def _coerce_float(key: str) -> float:
        raw = out.get(key)
        try:
            value_f = float(raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{path}.{key} must be a float/int") from exc
        return float(value_f)

    min_weight = _coerce_float("min_weight")
    max_weight = _coerce_float("max_weight")
    branch_temperature = _coerce_float("branch_temperature")
    curve_gamma = _coerce_float("curve_gamma")

    if min_weight < 0.0:
        raise ValueError(f"{path}.min_weight must be >= 0")
    if max_weight < 0.0:
        raise ValueError(f"{path}.max_weight must be >= 0")
    if min_weight > max_weight:
        raise ValueError(f"{path}.min_weight must be <= {path}.max_weight")
    if branch_temperature <= 0.0:
        raise ValueError(f"{path}.branch_temperature must be > 0")
    if curve_gamma <= 0.0:
        raise ValueError(f"{path}.curve_gamma must be > 0")

    out["min_weight"] = float(min_weight)
    out["max_weight"] = float(max_weight)
    out["branch_temperature"] = float(branch_temperature)
    out["curve_gamma"] = float(curve_gamma)
    return out
