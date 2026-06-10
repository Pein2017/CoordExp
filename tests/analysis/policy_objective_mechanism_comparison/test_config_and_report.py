from __future__ import annotations

import json
from pathlib import Path

from src.analysis.policy_objective_mechanism_comparison.config import load_config
from src.analysis.policy_objective_mechanism_comparison.report import write_outputs


CONFIG_PATH = Path(
    "/data/CoordExp/configs/analysis/policy_objective_mechanism_comparison/fullobj_5ckpt_ckpt3668.yaml"
)


def test_policy_objective_comparison_config_loads_five_roles() -> None:
    config = load_config(CONFIG_PATH)

    assert len(config.checkpoint_roles) == 5
    assert (
        config.checkpoint_roles["fullobj_random_et_rmp_ce_ckpt3668"].objective_policy
        == "et_rmp_ce"
    )
    assert (
        config.checkpoint_roles["et_rmp_ce_ckpt3664_reference"].comparison_group
        == "legacy_reference_anchor"
    )


def test_policy_objective_comparison_report_separates_anchor(tmp_path: Path) -> None:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        raw.replace(
            "/data/CoordExp/outputs/analysis/autoreg_object_rollout/policy_objective_mechanism_comparison/fullobj_5ckpt_ckpt3668",
            str(tmp_path / "artifacts"),
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    result = write_outputs(config)

    summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert summary["checkpoint_role_count"] == 5
    assert len(summary["clean_2x2_roles"]) == 4
    assert summary["legacy_reference_anchor_roles"] == [
        "et_rmp_ce_ckpt3664_reference"
    ]
    report = Path(result["comparison_report_md"]).read_text(encoding="utf-8")
    assert "Clean 2x2 Cohort" in report
    assert "Legacy Reference Anchor" in report
