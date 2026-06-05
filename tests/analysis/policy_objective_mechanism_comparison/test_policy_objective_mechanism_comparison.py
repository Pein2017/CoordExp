from __future__ import annotations

import json
from pathlib import Path

from src.analysis.policy_objective_mechanism_comparison.config import load_config
from src.analysis.policy_objective_mechanism_comparison.runner import run
from src.analysis.post_x1_instance_basin_tomography.config import load_config as load_a33
from src.analysis.prefix_state_transition_tomography.config import load_config as load_prefix
from src.analysis.sorted_random_no_newline_phenotype.config import load_config as load_a32


ROOT = Path("/data/CoordExp")
A32_SMOKE = ROOT / "configs/analysis/sorted_random_no_newline_phenotype/fullobj_policy_objective_4ckpt_ckpt3668_phase_a3_2_smoke.yaml"
A33_SMOKE = ROOT / "configs/analysis/post_x1_instance_basin_tomography/five_ckpt_policy_objective_ckpt3668_phase_a3_3_smoke.yaml"
PREFIX_SMOKE = ROOT / "configs/analysis/prefix_state_transition_tomography/fullobj_policy_objective_5ckpt_ckpt3668_phase_a3_smoke.yaml"
AGGREGATE = ROOT / "configs/analysis/policy_objective_mechanism_comparison/fullobj_5ckpt_ckpt3668.yaml"


def test_new_policy_objective_configs_load_ordered_roles() -> None:
    a32 = load_a32(A32_SMOKE)
    assert tuple(a32.checkpoints) == (
        "fullobj_random_pure_ce_ckpt3668",
        "fullobj_sorted_pure_ce_ckpt3668",
        "fullobj_random_et_rmp_ce_ckpt3668",
        "fullobj_sorted_et_rmp_ce_ckpt3668",
    )
    assert a32.checkpoints["fullobj_random_et_rmp_ce_ckpt3668"].objective_policy == "et_rmp_ce"

    a33 = load_a33(A33_SMOKE, validate_paths=False)
    assert tuple(a33.checkpoints)[-1] == "et_rmp_ce_ckpt3664_reference"
    assert a33.checkpoints["et_rmp_ce_ckpt3664_reference"].comparison_role == "reference_anchor"
    assert a33.checkpoints["fullobj_sorted_et_rmp_ce_ckpt3668"].objective_policy == "et_rmp_ce"

    prefix = load_prefix(PREFIX_SMOKE)
    assert len(prefix.checkpoints) == 5
    assert prefix.checkpoints["fullobj_sorted_pure_ce_ckpt3668"].training_ordering == "sorted"


def test_comparison_aggregator_materializes_summary_report_and_plot(tmp_path: Path) -> None:
    config = load_config(AGGREGATE)
    config = type(config)(
        project_id=config.project_id,
        schema_version=config.schema_version,
        run_id=config.run_id,
        artifact_root=tmp_path / "aggregate",
        checkpoint_roles=config.checkpoint_roles,
        artifact_sources=config.artifact_sources,
    )
    result = run(config)
    assert result["status"] == "ok"
    summary = json.loads((tmp_path / "aggregate" / "summary.json").read_text())
    assert len(summary["checkpoint_roles"]) == 5
    assert set(summary["legacy_reference_roles"]) == {"et_rmp_ce_ckpt3664_reference"}
    assert (tmp_path / "aggregate" / "comparison_report.md").is_file()
    assert (tmp_path / "aggregate" / "plots" / "role_row_counts.tsv").is_file()
    assert (tmp_path / "aggregate" / "plots" / "role_row_counts.svg").is_file()
