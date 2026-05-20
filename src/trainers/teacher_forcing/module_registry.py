from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

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
    "hard_sft": ObjectiveModuleDefinition(
        family="text",
        semantic_role="hard_sft",
        config_keys=frozenset(
            {
                "desc_ce_weight",
                "rollout_fn_desc_weight",
                "rollout_global_prefix_struct_ce_weight",
            }
        ),
        optional_config_keys=frozenset({"rollout_global_prefix_struct_ce_weight"}),
        application_presets=frozenset({"hard_sft"}),
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
    "stage2_trie_ce": ObjectiveModuleDefinition(
        family="text",
        semantic_role="stage2_trie_ce",
        config_keys=frozenset(
            {
                "support_weight",
                "balance_weight",
                "struct_weight",
                "desc_weight",
                "coord_hard_ce_weight",
                "eos_weight",
                "normalization",
            }
        ),
        application_presets=frozenset({"rollout_trie_hard_ce"}),
        projected_atoms=(
            ObjectiveLossAtomDefinition(
                atom_name="trie_ce",
                state_key="stage2_trie_ce_contrib",
            ),
        ),
        emission_group="text",
    ),
}

DIAGNOSTIC_MODULE_CATALOG: Final[dict[str, DiagnosticModuleDefinition]] = {}


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
