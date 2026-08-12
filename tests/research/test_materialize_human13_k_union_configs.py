from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import scripts.research.materialize_human13_k_union_configs as materializer
from scripts.research.run_human13_k_union_overfit import (
    Human13A6DonorBinding,
    Human13A6DonorRecord,
    Human13ManifestIdentity,
    _a6_donor_artifact_sha256,
)


CONFIG_ROOT = Path("configs/coordexp_swift/research/human13_k_union")
BASE_ARMS = (
    "frozen_source",
    "full_gt_capacity",
    "A0",
    "A1",
    "A3",
    "A4",
    "A7",
    "A8-prime",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_census(path: Path, *, applicable: bool = True) -> Path:
    payload = {
        "schema_version": "human13_a8_census_binding.v1",
        "manifest_identity": {
            "schema_version": "human13_k_union_manifest.v1",
            "unit_id": materializer.UNIT_ID,
            "panel_sha256": materializer.PANEL_SHA256,
            "manifest_sha256": "b" * 64,
        },
        "frozen_targets_sha256": "c" * 64,
        "census": {
            "schema_version": "human13_k_union_no_update_census.v1",
            "frozen_targets": {
                "byte_identical": True,
                "sha256_before": "c" * 64,
                "sha256_after": "c" * 64,
            },
            "aligned_surface": {
                "all_finite": True,
                "maximum_absolute_margin_drift": 0.1249 if applicable else 0.7499,
                "site_count": 4,
            },
            "a8_prime": {
                "applicable": applicable,
                "blocked": not applicable,
                "block_reason": (None if applicable else "required_margin_exceeds_0_5"),
                "required_margin": 0.125 if applicable else 0.75,
                "violating_site_count": 4 if applicable else 0,
            },
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_a6_binding(path: Path) -> Human13A6DonorBinding:
    binding = Human13A6DonorBinding(
        schema_version="human13_a6_donor_binding.v1",
        manifest_identity=Human13ManifestIdentity(
            schema_version="human13_k_union_manifest.v1",
            unit_id=materializer.UNIT_ID,
            panel_sha256=materializer.PANEL_SHA256,
            manifest_sha256="b" * 64,
        ),
        frozen_targets_sha256="c" * 64,
        artifact_sha256="0" * 64,
        applicable=True,
        donors=(
            Human13A6DonorRecord(
                image_id=2299,
                owner_id="2299:17",
                target_row_id="2299:sampled:21001:3",
                donor_trajectory_id="2299:sampled:21001",
                donor_prefix_token_ids=(1, 2, 3),
                donor_prior_row_ids=("2299:sampled:21001:0",),
                h_mid_eligible=True,
            ),
        ),
    )
    binding = replace(binding, artifact_sha256=_a6_donor_artifact_sha256(binding))
    path.write_text(
        json.dumps(materializer.a6_binding_to_dict(binding), sort_keys=True),
        encoding="utf-8",
    )
    return binding


def test_static_configs_freeze_exact_approved_arm_contracts() -> None:
    paths = sorted(CONFIG_ROOT.glob("*.yaml"))
    configs = [materializer.load_arm_config(path) for path in paths]

    assert tuple(config.arm_id for config in configs) == BASE_ARMS + ("A6",)
    assert {config.source for config in configs} == {materializer.FROZEN_SOURCE}
    assert {config.milestones for config in configs} == {(0, 1, 2, 4, 8, 16)}
    assert {config.global_max_length for config in configs} == {12_000}

    by_arm = {config.arm_id: config for config in configs}
    assert by_arm["A0"].arm_name == "A0 shared no-H background"
    assert by_arm["frozen_source"].updates is False
    assert by_arm["frozen_source"].optimizer is None
    for arm_id in set(by_arm) - {"frozen_source"}:
        config = by_arm[arm_id]
        assert config.updates is True
        assert config.trainable_surface == materializer.LANGUAGE_DORA_ONLY
        assert config.optimizer == materializer.FROZEN_ADAMW
        assert config.scheduler == materializer.FROZEN_SCHEDULER
        assert config.max_grad_norm == 1.0
    assert by_arm["A0"].coefficients == (0.0, 1.0, 1.0)
    assert by_arm["A7"].coefficients == (1.0, 0.0, 1.0)
    assert by_arm["full_gt_capacity"].coefficients == (1.0, 0.0, 0.0)
    for arm_id in ("A1", "A3", "A4", "A6", "A8-prime"):
        assert by_arm[arm_id].coefficients == (1.0, 1.0, 1.0)
    assert all(config.renormalize_active_families is False for config in configs)


def test_strict_config_rejects_unknown_or_drifted_fields(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIG_ROOT / "04_a3.yaml").read_text(encoding="utf-8"))
    raw["optimizer"]["learning_rate"] = 2.0e-5
    raw["surprise"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(materializer.MaterializationError, match="unknown fields"):
        materializer.load_arm_config(path)
    raw.pop("surprise")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="learning_rate"):
        materializer.load_arm_config(path)

    raw = yaml.safe_load((CONFIG_ROOT / "02_a0.yaml").read_text(encoding="utf-8"))
    raw["arm_name"] = "replay-only control"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="arm_name"):
        materializer.load_arm_config(path)


def test_materializer_omits_a6_without_sealed_eligible_donor_and_a8_without_binding(
    tmp_path: Path,
) -> None:
    receipt = materializer.materialize_plans(
        output_root=tmp_path / "runs",
        run_id="screen-001",
        config_root=CONFIG_ROOT,
    )

    assert receipt["mode"] == "dry_run"
    assert receipt["actions"] == materializer.ZERO_MODEL_ACTIONS
    assert [plan["arm_id"] for plan in receipt["plans"]] == [
        arm for arm in BASE_ARMS if arm != "A8-prime"
    ]
    assert receipt["omitted_arms"] == [
        {"arm_id": "A6", "reason": "sealed_eligible_h_mid_donor_unavailable"},
        {"arm_id": "A8-prime", "reason": "sealed_a8_census_binding_unavailable"},
    ]


def test_materializer_binds_applicable_a6_and_a8_and_isolates_every_arm(
    tmp_path: Path,
) -> None:
    census = _write_census(tmp_path / "census.json")
    expected_a6 = _write_a6_binding(tmp_path / "a6.json")

    receipt = materializer.materialize_plans(
        output_root=tmp_path / "runs",
        run_id="screen-002",
        config_root=CONFIG_ROOT,
        census_path=census,
        a6_binding_path=tmp_path / "a6.json",
    )

    assert [plan["arm_id"] for plan in receipt["plans"]] == [
        *BASE_ARMS,
        "A6",
    ]
    assert receipt["omitted_arms"] == []
    roots = [plan["output_root"] for plan in receipt["plans"]]
    state_roots = [plan["optimizer_state_root"] for plan in receipt["plans"]]
    assert len(set(roots)) == len(roots)
    assert len(set(state_roots)) == len(state_roots)
    assert all(root is None for root in state_roots[:1])
    assert all(root is not None for root in state_roots[1:])
    assert len({plan["source"]["adapter_sha256"] for plan in receipt["plans"]}) == 1
    assert len({plan["fresh_state_id"] for plan in receipt["plans"][1:]}) == 8

    a6 = next(plan for plan in receipt["plans"] if plan["arm_id"] == "A6")
    assert materializer.a6_binding_from_dict(a6["a6_donor_binding"]) == expected_a6
    a8 = next(plan for plan in receipt["plans"] if plan["arm_id"] == "A8-prime")
    assert a8["a8_census_binding"]["required_margin"] == 0.125


def test_a8_binding_is_fail_closed_on_blocked_or_unbound_census(tmp_path: Path) -> None:
    blocked = _write_census(tmp_path / "blocked.json", applicable=False)
    receipt = materializer.materialize_plans(
        output_root=tmp_path / "runs",
        run_id="screen-003",
        config_root=CONFIG_ROOT,
        census_path=blocked,
    )
    assert {item["arm_id"]: item["reason"] for item in receipt["omitted_arms"]}[
        "A8-prime"
    ] == "required_margin_exceeds_0_5"

    raw = json.loads(blocked.read_text(encoding="utf-8"))
    raw["census"]["a8_prime"]["applicable"] = True
    raw["census"]["a8_prime"]["blocked"] = False
    raw["census"]["a8_prime"]["required_margin"] = 0.1
    blocked.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="artifact_sha256"):
        materializer.materialize_plans(
            output_root=tmp_path / "runs",
            run_id="screen-004",
            config_root=CONFIG_ROOT,
            census_path=blocked,
        )

    raw.pop("artifact_sha256")
    raw["artifact_sha256"] = _canonical_sha256(raw)
    blocked.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError, match="derived.*drift"):
        materializer.materialize_plans(
            output_root=tmp_path / "runs",
            run_id="screen-005",
            config_root=CONFIG_ROOT,
            census_path=blocked,
        )


def test_dry_run_cli_performs_zero_runtime_action_and_writes_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = materializer.main(
        [
            "--config-root",
            str(CONFIG_ROOT),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "cli-proof",
        ]
    )

    assert code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["actions"] == materializer.ZERO_MODEL_ACTIONS
    assert not (tmp_path / "runs").exists()


def test_default_import_and_dry_run_do_not_import_model_runtime() -> None:
    code = """
import json
import sys
from pathlib import Path
import scripts.research.materialize_human13_k_union_configs as materializer
materializer.materialize_plans(
    output_root=Path('/tmp/human13-zero-action-proof'),
    run_id='import-proof',
    config_root=Path('configs/coordexp_swift/research/human13_k_union'),
)
forbidden = sorted(
    name for name in sys.modules
    if name == 'torch'
    or name.startswith('torch.')
    or name == 'transformers'
    or name.startswith('transformers.')
    or name == 'accelerate'
    or name.startswith('accelerate.')
    or name == 'src.qwen'
    or name.startswith('src.qwen.')
)
print(json.dumps(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
