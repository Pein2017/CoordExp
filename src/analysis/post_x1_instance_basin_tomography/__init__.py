"""Post-x1 instance-basin tomography analysis package."""

from __future__ import annotations

PROJECT_ID = "post_x1_instance_basin_tomography"
PHASE_ID = "phase_a3_3"
SCHEMA_VERSION = "a3.3.v1"
ARTIFACT_SCHEMA_VERSION = SCHEMA_VERSION
SMOKE_RUN_ID = "three_ckpt_phase_a3_3_smoke"
FULL_RUN_ID = "three_ckpt_phase_a3_3"
FIVE_CKPT_SMOKE_RUN_ID = "fullobj_5ckpt_ckpt3668_phase_a3_3_smoke"
FIVE_CKPT_FULL_RUN_ID = "fullobj_5ckpt_ckpt3668_phase_a3_3"
RUN_IDS = (
    SMOKE_RUN_ID,
    FULL_RUN_ID,
    FIVE_CKPT_SMOKE_RUN_ID,
    FIVE_CKPT_FULL_RUN_ID,
)
FULL_RUN_IDS = (FULL_RUN_ID, FIVE_CKPT_FULL_RUN_ID)

LEGACY_CHECKPOINT_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
    "et_rmp_ce_ckpt3664",
)
CHECKPOINT_ROLES = LEGACY_CHECKPOINT_ROLES

PURE_CE_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
)
ET_RMP_ROLE = "et_rmp_ce_ckpt3664"


def checkpoint_roles_from_mapping(mapping: object) -> tuple[str, ...]:
    """Return ordered checkpoint roles from a config/artifact mapping."""

    if not isinstance(mapping, dict):
        return CHECKPOINT_ROLES
    return tuple(str(role) for role in mapping)
