from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, Mapping

LossEmissionGroup = Literal["text", "coord"]


@dataclass(frozen=True)
class ObjectiveLossAtomDefinition:
    atom_name: str
    state_key: str
    required_state: bool = True


@dataclass(frozen=True)
class ObjectiveModuleDefinition:
    family: str
    semantic_role: str
    config_keys: frozenset[str]
    application_presets: frozenset[str]
    projected_atoms: tuple[ObjectiveLossAtomDefinition, ...]
    optional_config_keys: frozenset[str] = frozenset()
    emission_group: LossEmissionGroup | None = None


@dataclass(frozen=True)
class DiagnosticModuleDefinition:
    family: str
    semantic_role: str
    config_keys: frozenset[str]


OBJECTIVE_MODULE_CATALOG: Final[dict[str, ObjectiveModuleDefinition]] = {
    "token_ce": ObjectiveModuleDefinition(
        family="text",
        semantic_role="token_ce",
        config_keys=frozenset(
            {
                "desc_ce_weight",
                "rollout_fn_desc_weight",
                "rollout_global_prefix_struct_ce_weight",
            }
        ),
        optional_config_keys=frozenset({"rollout_global_prefix_struct_ce_weight"}),
        application_presets=frozenset({"anchor_text_only", "rollout_text_only"}),
        projected_atoms=(
            ObjectiveLossAtomDefinition(
                atom_name="struct_ce",
                state_key="token_ce_struct_contrib",
            ),
            ObjectiveLossAtomDefinition(
                atom_name="desc_ce",
                state_key="token_ce_desc_contrib",
            ),
        ),
        emission_group="text",
    ),
    "loss_duplicate_burst_unlikelihood": ObjectiveModuleDefinition(
        family="rollout",
        semantic_role="duplicate_burst_unlikelihood",
        config_keys=frozenset(),
        application_presets=frozenset({"rollout_only"}),
        projected_atoms=(
            ObjectiveLossAtomDefinition(
                atom_name="loss_duplicate_burst_unlikelihood",
                state_key="loss_duplicate_burst_unlikelihood_contrib",
            ),
        ),
        emission_group="text",
    ),
    "bbox_geo": ObjectiveModuleDefinition(
        family="bbox",
        semantic_role="geometry",
        config_keys=frozenset(
            {
                "smoothl1_weight",
                "ciou_weight",
                "parameterization",
                "center_weight",
                "size_weight",
            }
        ),
        optional_config_keys=frozenset(
            {"parameterization", "center_weight", "size_weight"}
        ),
        application_presets=frozenset({"anchor_only"}),
        projected_atoms=(
            ObjectiveLossAtomDefinition(
                atom_name="bbox_smoothl1",
                state_key="bbox_smoothl1_contrib",
            ),
            ObjectiveLossAtomDefinition(
                atom_name="bbox_ciou",
                state_key="bbox_ciou_contrib",
            ),
        ),
        emission_group="coord",
    ),
    "bbox_size_aux": ObjectiveModuleDefinition(
        family="bbox",
        semantic_role="size_aux",
        config_keys=frozenset(
            {
                "log_wh_weight",
                "oversize_penalty_weight",
                "oversize_area_frac_threshold",
                "oversize_log_w_threshold",
                "oversize_log_h_threshold",
                "eps",
            }
        ),
        application_presets=frozenset({"anchor_only"}),
        projected_atoms=(
            ObjectiveLossAtomDefinition(
                atom_name="bbox_log_wh",
                state_key="bbox_log_wh_contrib",
            ),
            ObjectiveLossAtomDefinition(
                atom_name="bbox_oversize",
                state_key="bbox_oversize_contrib",
            ),
        ),
        emission_group="coord",
    ),
    "coord_reg": ObjectiveModuleDefinition(
        family="coord",
        semantic_role="regularizer",
        config_keys=frozenset(
            {
                "coord_ce_weight",
                "coord_gate_weight",
                "text_gate_weight",
                "soft_ce_weight",
                "w1_weight",
                "temperature",
                "target_sigma",
                "target_truncate",
                "adjacent_repulsion_weight",
                "adjacent_repulsion_filter_mode",
                "adjacent_repulsion_margin_ratio",
                "adjacent_repulsion_copy_margin",
            }
        ),
        application_presets=frozenset({"anchor_only"}),
        projected_atoms=(
            ObjectiveLossAtomDefinition(
                atom_name="coord_token_ce",
                state_key="coord_token_ce_contrib",
            ),
            ObjectiveLossAtomDefinition(
                atom_name="coord_soft_ce",
                state_key="coord_soft_ce_contrib",
            ),
            ObjectiveLossAtomDefinition(
                atom_name="coord_w1",
                state_key="coord_w1_contrib",
            ),
            ObjectiveLossAtomDefinition(
                atom_name="adjacent_repulsion",
                state_key="adjacent_repulsion_contrib",
                required_state=False,
            ),
            ObjectiveLossAtomDefinition(
                atom_name="coord_el1",
                state_key="coord_el1_contrib",
                required_state=False,
            ),
            ObjectiveLossAtomDefinition(
                atom_name="coord_ehuber",
                state_key="coord_ehuber_contrib",
                required_state=False,
            ),
            ObjectiveLossAtomDefinition(
                atom_name="coord_entropy",
                state_key="coord_entropy_contrib",
                required_state=False,
            ),
            ObjectiveLossAtomDefinition(
                atom_name="coord_gate",
                state_key="coord_gate_contrib",
            ),
            ObjectiveLossAtomDefinition(
                atom_name="text_gate",
                state_key="text_gate_contrib",
            ),
        ),
        optional_config_keys=frozenset(
            {
                "adjacent_repulsion_weight",
                "adjacent_repulsion_filter_mode",
                "adjacent_repulsion_margin_ratio",
                "adjacent_repulsion_copy_margin",
            }
        ),
        emission_group="coord",
    ),
}

DIAGNOSTIC_MODULE_CATALOG: Final[dict[str, DiagnosticModuleDefinition]] = {
    "coord_diag": DiagnosticModuleDefinition(
        family="coord",
        semantic_role="diagnostic",
        config_keys=frozenset(),
    ),
}


ALLOWED_OBJECTIVE_MODULES: Final[set[str]] = set(OBJECTIVE_MODULE_CATALOG)
ALLOWED_DIAGNOSTIC_MODULES: Final[set[str]] = set(DIAGNOSTIC_MODULE_CATALOG)

OBJECTIVE_CONFIG_ALLOWLIST: Final[dict[str, set[str]]] = {
    name: set(definition.config_keys)
    for name, definition in OBJECTIVE_MODULE_CATALOG.items()
}

OBJECTIVE_OPTIONAL_CONFIG_KEYS: Final[dict[str, set[str]]] = {
    name: set(definition.optional_config_keys)
    for name, definition in OBJECTIVE_MODULE_CATALOG.items()
    if definition.optional_config_keys
}

OBJECTIVE_APPLICATION_PRESET_ALLOWLIST: Final[dict[str, set[str]]] = {
    name: set(definition.application_presets)
    for name, definition in OBJECTIVE_MODULE_CATALOG.items()
}

DIAGNOSTIC_CONFIG_ALLOWLIST: Final[dict[str, set[str]]] = {
    name: set(definition.config_keys)
    for name, definition in DIAGNOSTIC_MODULE_CATALOG.items()
}


def objective_modules_for_family(family: str) -> tuple[str, ...]:
    return tuple(
        name
        for name, definition in OBJECTIVE_MODULE_CATALOG.items()
        if definition.family == family
    )


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


def validate_bbox_geo_config_values(
    config: Mapping[str, Any],
    *,
    path: str,
) -> None:
    parameterization = str(config.get("parameterization", "xyxy") or "xyxy").strip().lower()
    if parameterization not in {"xyxy", "center_size"}:
        raise ValueError(
            f"{path}.parameterization must be one of ['center_size', 'xyxy']"
        )

    center_weight = 1.0
    if "center_weight" in config:
        try:
            center_weight = float(config.get("center_weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}.center_weight must be numeric") from exc
        if center_weight < 0.0:
            raise ValueError(f"{path}.center_weight must be >= 0")

    size_weight = 1.0
    if "size_weight" in config:
        try:
            size_weight = float(config.get("size_weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}.size_weight must be numeric") from exc
        if size_weight < 0.0:
            raise ValueError(f"{path}.size_weight must be >= 0")

    if parameterization == "center_size" and center_weight == 0.0 and size_weight == 0.0:
        raise ValueError(
            f"{path}.parameterization=center_size requires center_weight > 0 or size_weight > 0"
        )


def validate_adjacent_repulsion_config_values(
    config: Mapping[str, Any],
    *,
    path: str,
) -> None:
    if "adjacent_repulsion_weight" in config:
        try:
            weight = float(config.get("adjacent_repulsion_weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}.adjacent_repulsion_weight must be numeric") from exc
        if weight < 0.0:
            raise ValueError(f"{path}.adjacent_repulsion_weight must be >= 0")

    if "adjacent_repulsion_margin_ratio" in config:
        try:
            margin_ratio = float(config.get("adjacent_repulsion_margin_ratio"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}.adjacent_repulsion_margin_ratio must be numeric"
            ) from exc
        if margin_ratio < 0.0:
            raise ValueError(f"{path}.adjacent_repulsion_margin_ratio must be >= 0")

    if "adjacent_repulsion_copy_margin" in config:
        try:
            copy_margin = float(config.get("adjacent_repulsion_copy_margin"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}.adjacent_repulsion_copy_margin must be numeric"
            ) from exc
        if copy_margin < 0.0 or copy_margin > 1.0:
            raise ValueError(
                f"{path}.adjacent_repulsion_copy_margin must be within [0, 1]"
            )

    if "adjacent_repulsion_filter_mode" in config:
        mode = str(config.get("adjacent_repulsion_filter_mode") or "").strip().lower()
        if mode not in {"same_desc", "global"}:
            raise ValueError(
                f"{path}.adjacent_repulsion_filter_mode must be one of ['global', 'same_desc']"
            )
