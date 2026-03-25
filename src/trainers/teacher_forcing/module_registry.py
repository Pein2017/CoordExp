"""Unified teacher-forcing module registry (names + config allowlists).

This module is intentionally dependency-light so it can be imported from config
schema validation without pulling in torch/trainer runtime code.
"""

from __future__ import annotations

from typing import Any, Final

ALLOWED_OBJECTIVE_MODULES: Final[set[str]] = {
    "token_ce",
    "loss_duplicate_burst_unlikelihood",
    "bbox_geo",
    "bbox_size_aux",
    "coord_reg",
}
ALLOWED_DIAGNOSTIC_MODULES: Final[set[str]] = {"coord_diag"}

OBJECTIVE_CONFIG_ALLOWLIST: Final[dict[str, set[str]]] = {
    "token_ce": {
        "desc_ce_weight",
        "rollout_fn_desc_weight",
        "rollout_global_prefix_struct_ce_weight",
    },
    "loss_duplicate_burst_unlikelihood": set(),
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
    "token_ce": {
        "rollout_global_prefix_struct_ce_weight",
    }
}

OBJECTIVE_APPLICATION_PRESET_ALLOWLIST: Final[dict[str, set[str]]] = {
    "token_ce": {
        "anchor_text_only",
        "rollout_text_only",
    },
    "loss_duplicate_burst_unlikelihood": {"rollout_only"},
    "bbox_geo": {
        "anchor_only",
    },
    "bbox_size_aux": {
        "anchor_only",
    },
    "coord_reg": {
        "anchor_only",
    },
}

DIAGNOSTIC_CONFIG_ALLOWLIST: Final[dict[str, set[str]]] = {
    "coord_diag": set(),
}

def normalize_token_ce_stop_signal_damping_config(
    value: Any,
    *,
    path: str,
) -> dict[str, Any]:
    raise ValueError(
        f"{path} is deprecated and unsupported. Remove token_ce.config.stop_signal_damping; "
        "the adaptive stop-signal-damping experiment was dropped because it is toxic for "
        "rollout and causes duplicate-heavy dense-scene proposals."
    )
