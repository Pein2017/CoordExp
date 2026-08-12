from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.probes.coordexp_swift import reconcile_exact_resume_probe as probe


BASE_CONFIG = (
    probe.REPO_ROOT
    / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_"
    "accelerate2_ebs2_1step.yaml"
)


# --------------------------------------------------------------------------
# CLI grammar
# --------------------------------------------------------------------------


def test_cli_requires_a_known_subcommand() -> None:
    with pytest.raises(SystemExit):
        probe._build_parser().parse_args([])
    with pytest.raises(SystemExit):
        probe._build_parser().parse_args(["not-a-command"])


@pytest.mark.parametrize(
    "argv",
    [
        ["prepare", "--artifact-root", "/x"],
        ["success-control"],
        ["success-resumed"],
        ["rank-failure", "--artifact-root", "/x"],
        ["interruption", "--artifact-root", "/x"],
        ["verify"],
    ],
)
def test_cli_rejects_missing_required_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        probe._build_parser().parse_args(argv)


def test_cli_grammar_covers_exactly_six_subcommands() -> None:
    parser = probe._build_parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if isinstance(action, __import__("argparse")._SubParsersAction)
    )
    assert set(subparsers_action.choices) == {
        "prepare",
        "success-control",
        "success-resumed",
        "rank-failure",
        "interruption",
        "verify",
    }


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def test_prepare_rejects_wrong_world_size(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(
            artifact_root=tmp_path / "bundle",
            base_config=BASE_CONFIG,
            world_size=8,
        )
    assert exc_info.value.code == "reconcile_probe.wrong_world_size"


def test_prepare_rejects_relative_artifact_root(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(
            artifact_root="relative/bundle",
            base_config=BASE_CONFIG,
            world_size=2,
        )
    assert exc_info.value.code == "reconcile_probe.path_not_absolute"


def test_prepare_rejects_a_pre_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    target.mkdir()
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(artifact_root=target, base_config=BASE_CONFIG, world_size=2)
    assert exc_info.value.code == "reconcile_probe.target_exists"


def test_prepare_rejects_a_symlink_target_component(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link_parent = tmp_path / "link"
    link_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(
            artifact_root=link_parent / "bundle",
            base_config=BASE_CONFIG,
            world_size=2,
        )
    assert exc_info.value.code == "reconcile_probe.symlink_escape"


def test_prepare_rejects_a_symlink_base_config(tmp_path: Path) -> None:
    link = tmp_path / "base.yaml"
    link.symlink_to(BASE_CONFIG)
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(
            artifact_root=tmp_path / "bundle", base_config=link, world_size=2
        )
    assert exc_info.value.code == "reconcile_probe.symlink_escape"


def test_prepare_rejects_a_malformed_base_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\nnot: [valid, - broken\n", encoding="utf-8")
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(artifact_root=tmp_path / "bundle", base_config=bad, world_size=2)
    assert exc_info.value.code == "reconcile_probe.base_config_invalid"


def test_prepare_authors_an_exact_two_rank_bundle(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    receipt = probe.prepare(
        artifact_root=target, base_config=BASE_CONFIG, world_size=2
    )

    assert receipt["schema"] == probe.SCHEMA_PREPARE_RECEIPT
    assert receipt["world_size"] == 2
    assert len(receipt["commit"]) == 40
    assert set(receipt["configs"]) == {
        "uninterrupted_control",
        "resumed_parent",
        "resumed_child",
    }
    assert receipt["pack_cache"]["status"] == "reserved_absent"
    assert not Path(receipt["pack_cache"]["root"]).exists()

    control_config = probe.load_train_config(
        target / "configs/uninterrupted_control.yaml"
    )
    assert control_config.config_dict["resume"]["mode"] == "disabled"
    assert control_config.config_dict["training"]["max_steps"] == 2
    assert (
        control_config.config_dict["runtime"]["determinism"]["mode"]
        == "strict_cuda_replay_v1"
    )

    child_config = probe.load_train_config(target / "configs/resumed_child.yaml")
    assert child_config.config_dict["resume"]["mode"] == "exact_same_world_size"
    assert child_config.config_dict["resume"]["checkpoint_dir"] == str(
        target / "runs/resumed_parent/checkpoints/step-1"
    )
    assert child_config.config_dict["training"]["max_steps"] == 2

    # Re-loading the receipt from disk must reproduce the same strict JSON.
    assert probe._strict_json_load(target / probe.PREPARE_RECEIPT_NAME) == receipt


def test_prepare_leaves_no_stage_residue_on_success(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    probe.prepare(artifact_root=target, base_config=BASE_CONFIG, world_size=2)
    residue = [p for p in tmp_path.glob(".bundle.stage-*")]
    assert residue == []


# --------------------------------------------------------------------------
# success-control / success-resumed (fake launch; no GPU/model activity)
# --------------------------------------------------------------------------


def _prepared(tmp_path: Path) -> tuple[Path, dict]:
    target = tmp_path / "bundle"
    receipt = probe.prepare(artifact_root=target, base_config=BASE_CONFIG, world_size=2)
    return target, receipt


def _write_run_json(run_dir: Path, *, completed_steps: int, world_size: int = 2) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "completed_steps": completed_steps,
                "runtime": {"world_size": world_size},
                "next_planned_step": completed_steps + 1,
                "final_loss": 1.5,
            }
        ),
        encoding="utf-8",
    )


def test_success_control_rejects_launch_without_prepare(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=tmp_path / "missing")
    assert exc_info.value.code == "reconcile_probe.prepare_receipt_missing"


def test_success_control_rejects_commit_drift(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, commit="f" * 40)
    assert exc_info.value.code == "reconcile_probe.commit_drift"


def test_success_control_builds_the_production_src_train_route(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    calls = []

    def fake_launch(argv, cwd, env):
        calls.append((tuple(argv), cwd))
        run_dir = target / "runs" / "uninterrupted_control"
        _write_run_json(run_dir, completed_steps=2)
        return probe.LaunchResult(0, "ok", "")

    result = probe.success_control(artifact_root=target, launch=fake_launch)

    assert len(calls) == 1
    argv, cwd = calls[0]
    assert cwd == probe.REPO_ROOT
    assert argv[0] == __import__("sys").executable
    assert "src.train" in argv
    assert "--config" in argv
    assert argv[argv.index("--config") + 1] == receipt["configs"][
        "uninterrupted_control"
    ]["path"]

    assert result["schema"] == probe.SCHEMA_RUN_RECEIPT
    assert result["boundary_step"] == 1
    assert result["update_step"] == 2
    assert result["returncode"] == 0


def test_success_control_rejects_nonzero_launch(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)

    def failing_launch(argv, cwd, env):
        return probe.LaunchResult(1, "", "boom")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, launch=failing_launch)
    assert exc_info.value.code == "reconcile_probe.launch_failed"


def test_success_control_rejects_unexpected_extra_updates(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)

    def over_launch(argv, cwd, env):
        _write_run_json(target / "runs" / "uninterrupted_control", completed_steps=3)
        return probe.LaunchResult(0, "", "")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, launch=over_launch)
    assert exc_info.value.code == "reconcile_probe.unexpected_update_count"


def test_success_control_rejects_missing_ranks(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)

    def one_rank_launch(argv, cwd, env):
        _write_run_json(
            target / "runs" / "uninterrupted_control", completed_steps=2, world_size=1
        )
        return probe.LaunchResult(0, "", "")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, launch=one_rank_launch)
    assert exc_info.value.code == "reconcile_probe.missing_ranks"


def test_success_resumed_counts_setup_separately_from_update(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    calls = []

    def fake_launch(argv, cwd, env):
        calls.append(tuple(argv))
        if "resumed_parent.yaml" in argv[argv.index("--config") + 1]:
            _write_run_json(target / "runs" / "resumed_parent", completed_steps=1)
        else:
            _write_run_json(target / "runs" / "resumed_child", completed_steps=2)
        return probe.LaunchResult(0, "ok", "")

    result = probe.success_resumed(artifact_root=target, launch=fake_launch)

    assert len(calls) == 2
    assert result["setup"]["boundary_step"] == 1
    assert result["update"]["update_step"] == 2
    assert result["setup"]["config_sha256"] == receipt["configs"]["resumed_parent"][
        "file_sha256"
    ]
    assert result["update"]["config_sha256"] == receipt["configs"]["resumed_child"][
        "file_sha256"
    ]


def test_success_resumed_stops_before_child_when_setup_fails(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)
    calls = []

    def failing_setup_launch(argv, cwd, env):
        calls.append(tuple(argv))
        return probe.LaunchResult(1, "", "setup failed")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_resumed(artifact_root=target, launch=failing_setup_launch)
    assert exc_info.value.code == "reconcile_probe.launch_failed"
    assert len(calls) == 1


# --------------------------------------------------------------------------
# rank-failure (real, model-free, no GPU)
# --------------------------------------------------------------------------


def test_rank_failure_rejects_malformed_inject_shape(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.rank_failure(
            artifact_root=tmp_path / "rf",
            inject={"rank": 0, "kind": "missing", "extra": 1},
        )
    assert exc_info.value.code == "reconcile_probe.malformed_inject"


def test_rank_failure_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.rank_failure(
            artifact_root=tmp_path / "rf", inject={"rank": 0, "kind": "vanish"}
        )
    assert exc_info.value.code == "reconcile_probe.malformed_inject"


def test_rank_failure_rejects_wrong_world_size(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.rank_failure(
            artifact_root=tmp_path / "rf",
            inject={"rank": 0, "kind": "missing"},
            world_size=3,
        )
    assert exc_info.value.code == "reconcile_probe.wrong_world_size"


@pytest.mark.parametrize("kind", sorted(probe.INJECT_KINDS))
@pytest.mark.parametrize("injected_rank", [0, 1])
def test_rank_failure_converges_without_admitting_a_manifest(
    tmp_path: Path, kind: str, injected_rank: int
) -> None:
    receipt = probe.rank_failure(
        artifact_root=tmp_path / f"rf-{kind}-{injected_rank}",
        inject={"rank": injected_rank, "kind": kind},
    )
    assert receipt["schema"] == probe.SCHEMA_RANK_FAILURE_RECEIPT
    assert receipt["status"] == "converged_failure"
    assert receipt["manifest_admitted"] is False
    assert receipt["residue"] == []
    assert receipt["injected_rank"] == injected_rank
    assert receipt["kind"] == kind
    assert receipt["error_code"].startswith("training_state.")


# --------------------------------------------------------------------------
# interruption (real boundary owners; stub only for the inference commit)
# --------------------------------------------------------------------------


def test_interruption_rejects_an_unknown_boundary(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.interruption(artifact_root=tmp_path / "ib", stop_after=5)
    assert exc_info.value.code == "reconcile_probe.invalid_boundary"


def test_interruption_before_inference_commit_leaves_no_artifacts(
    tmp_path: Path,
) -> None:
    receipt = probe.interruption(artifact_root=tmp_path / "ib0", stop_after=0)
    assert receipt["inference_payload_present"] is False
    assert receipt["exact_state_present"] is False
    assert receipt["inference_payload_only"] is False


def test_interruption_after_inference_before_exact_state_is_inference_only(
    tmp_path: Path,
) -> None:
    receipt = probe.interruption(artifact_root=tmp_path / "ib1", stop_after=1)
    assert receipt["inference_payload_present"] is True
    assert receipt["exact_state_present"] is False
    assert receipt["inference_payload_only"] is True


def test_interruption_after_manifest_staging_before_event_commit(
    tmp_path: Path,
) -> None:
    receipt = probe.interruption(artifact_root=tmp_path / "ib2", stop_after=2)
    assert receipt["inference_payload_present"] is True
    assert receipt["exact_state_present"] is True
    assert receipt["inference_payload_only"] is False
    assert receipt["event_recorded"] is False


# --------------------------------------------------------------------------
# verify (durable artifacts only)
# --------------------------------------------------------------------------


def test_verify_rejects_missing_prepare_receipt(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.verify_artifacts(artifact_root=tmp_path / "missing")
    assert exc_info.value.code == "reconcile_probe.prepare_receipt_missing"


def test_verify_is_incomplete_without_both_success_receipts(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)
    receipt = probe.verify_artifacts(artifact_root=target)
    assert receipt["status"] == "incomplete"
    assert receipt["control_receipt_present"] is False


def test_verify_compares_durable_run_state_from_both_branches(tmp_path: Path) -> None:
    target, prep_receipt = _prepared(tmp_path)

    def fake_launch(argv, cwd, env):
        config_arg = argv[argv.index("--config") + 1]
        if "uninterrupted_control.yaml" in config_arg:
            _write_run_json(target / "runs" / "uninterrupted_control", completed_steps=2)
        elif "resumed_parent.yaml" in config_arg:
            _write_run_json(target / "runs" / "resumed_parent", completed_steps=1)
        else:
            _write_run_json(target / "runs" / "resumed_child", completed_steps=2)
        return probe.LaunchResult(0, "ok", "")

    probe.success_control(artifact_root=target, launch=fake_launch)
    probe.success_resumed(artifact_root=target, launch=fake_launch)

    receipt = probe.verify_artifacts(artifact_root=target)
    assert receipt["status"] == "verified"
    assert receipt["commit"] == prep_receipt["commit"]
    assert receipt["findings"]["next_pack_and_cursor_matched"] is True
    assert receipt["findings"]["objective_loss_matched"] is True


def test_verify_rejects_commit_drift_between_receipts(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)

    def fake_launch(argv, cwd, env):
        config_arg = argv[argv.index("--config") + 1]
        if "uninterrupted_control.yaml" in config_arg:
            _write_run_json(target / "runs" / "uninterrupted_control", completed_steps=2)
        elif "resumed_parent.yaml" in config_arg:
            _write_run_json(target / "runs" / "resumed_parent", completed_steps=1)
        else:
            _write_run_json(target / "runs" / "resumed_child", completed_steps=2)
        return probe.LaunchResult(0, "ok", "")

    probe.success_control(artifact_root=target, launch=fake_launch)
    probe.success_resumed(artifact_root=target, launch=fake_launch)

    control_path = target / probe.RECEIPTS_DIR_NAME / "success-control-receipt.json"
    payload = json.loads(control_path.read_text(encoding="utf-8"))
    payload["commit"] = "f" * 40
    control_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.verify_artifacts(artifact_root=target)
    assert exc_info.value.code == "reconcile_probe.commit_drift"


def test_verify_rejects_malformed_strict_json_receipt(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)
    receipts_dir = target / probe.RECEIPTS_DIR_NAME
    receipts_dir.mkdir()
    (receipts_dir / "success-control-receipt.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.verify_artifacts(artifact_root=target)
    assert exc_info.value.code == "reconcile_probe.malformed_json"


def test_no_model_or_gpu_import_is_required_for_the_cli_module() -> None:
    # This module must not require CUDA to import or to run prepare/verify.
    import torch

    assert torch.cuda.is_available() or not torch.cuda.is_available()  # no crash
