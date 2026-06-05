from __future__ import annotations

from pathlib import Path

from src.analysis.post_x1_instance_basin_tomography.config import (
    load_config as load_a33_config,
)
from src.analysis.prefix_state_transition_tomography.config import (
    load_config as load_prefix_state_config,
)
from src.analysis.sorted_random_no_newline_phenotype.config import (
    load_config as load_a32_config,
)


def test_a32_policy_objective_4ckpt_config_loads() -> None:
    config = load_a32_config(
        Path(
            "/data/CoordExp/configs/analysis/sorted_random_no_newline_phenotype/fullobj_policy_objective_4ckpt_ckpt3668_phase_a3_2_smoke.yaml"
        )
    )

    assert len(config.checkpoints) == 4
    assert config.checkpoints["fullobj_sorted_et_rmp_ce_ckpt3668"].objective_policy == "et_rmp_ce"


def test_a33_policy_objective_5ckpt_config_loads() -> None:
    config = load_a33_config(
        Path(
            "/data/CoordExp/configs/analysis/post_x1_instance_basin_tomography/five_ckpt_policy_objective_ckpt3668_phase_a3_3_smoke.yaml"
        )
    )

    assert len(config.checkpoints) == 5
    assert (
        config.checkpoints["et_rmp_ce_ckpt3664_reference"].template_contract.row_separator
        == "newline"
    )
    assert (
        config.checkpoints["fullobj_random_et_rmp_ce_ckpt3668"].controlled_comparison_group
        == "fullobj_2x2_20260601"
    )


def test_prefix_state_policy_objective_5ckpt_config_loads() -> None:
    config = load_prefix_state_config(
        Path(
            "/data/CoordExp/configs/analysis/prefix_state_transition_tomography/fullobj_policy_objective_5ckpt_ckpt3668_phase_a3_smoke.yaml"
        )
    )

    assert len(config.checkpoints) == 5
    assert config.checkpoints["fullobj_random_pure_ce_ckpt3668"].training_ordering == "random_permutation"
