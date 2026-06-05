"""Post-x1 instance-basin tomography analysis package."""

from __future__ import annotations

PROJECT_ID = "post_x1_instance_basin_tomography"
PHASE_ID = "phase_a3_3"
SCHEMA_VERSION = "a3.3.v1"
ARTIFACT_SCHEMA_VERSION = SCHEMA_VERSION
SMOKE_RUN_ID = "three_ckpt_phase_a3_3_smoke"
FULL_RUN_ID = "three_ckpt_phase_a3_3"

CHECKPOINT_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
    "et_rmp_ce_ckpt3664",
)

PURE_CE_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
)
ET_RMP_ROLE = "et_rmp_ce_ckpt3664"
