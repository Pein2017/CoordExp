from __future__ import annotations

"""Unified teacher-forcing module registry (names + config allowlists).

This module is intentionally dependency-light so it can be imported from config
schema validation without pulling in torch/trainer runtime code.
"""

from typing import Final

ALLOWED_OBJECTIVE_MODULES: Final[set[str]] = {"token_ce", "bbox_geo", "coord_reg"}
ALLOWED_DIAGNOSTIC_MODULES: Final[set[str]] = {"coord_diag"}

OBJECTIVE_CONFIG_ALLOWLIST: Final[dict[str, set[str]]] = {
    "token_ce": {
        "desc_ce_weight",
        "self_context_struct_ce_weight",
        "rollout_fn_desc_weight",
        "rollout_matched_prefix_struct_weight",
        "rollout_drop_invalid_struct_ce_multiplier",
    },
    "bbox_geo": {
        "smoothl1_weight",
        "ciou_weight",
        "a1_smoothl1_weight",
        "a1_ciou_weight",
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
        "self_context_soft_ce_weight",
        "w1_weight",
        "a1_soft_ce_weight",
        "a1_w1_weight",
        "temperature",
        "target_sigma",
        "target_truncate",
    },
}

DIAGNOSTIC_CONFIG_ALLOWLIST: Final[dict[str, set[str]]] = {
    "coord_diag": set(),
}

