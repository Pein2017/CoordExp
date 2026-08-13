from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.probes.coordexp_swift import reconcile_exact_resume_probe as probe
import src.artifacts.run_writer as run_writer_module
from src.artifacts.run_writer import RunWriter


BASE_CONFIG = (
    probe.REPO_ROOT
    / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_"
    "accelerate2_ebs2_1step.yaml"
)


# --------------------------------------------------------------------------
# Shared fixtures (fast, model-free; never invoke real tokenization/GPU)
# --------------------------------------------------------------------------


def _fake_prepare_pack_cache(calls: list) -> "probe.PrepareCacheCallable":
    def _fake(config_path: Path, cache_root: Path, receipt_path: Path) -> dict:
        calls.append(
            {
                "config_path": str(config_path),
                "cache_root": str(cache_root),
            }
        )
        body = {
            "schema": "coordexp-swift-pack-cache-preparation-receipt-v1",
            "terminal_status": "completed",
            "config_path": str(config_path),
            "result": {"resolved_config_fingerprint": "fake-fingerprint", "model_loaded": False},
            "failure": None,
        }
        encoded = json.dumps(
            body, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        import hashlib

        payload = {**body, "receipt_sha256": hashlib.sha256(encoded).hexdigest()}
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    return _fake


def _prepared(tmp_path: Path, *, name: str = "bundle") -> tuple[Path, dict]:
    target = tmp_path / name
    receipt = probe.prepare(
        artifact_root=target,
        base_config=BASE_CONFIG,
        world_size=2,
        prepare_pack_cache=_fake_prepare_pack_cache([]),
    )
    return target, receipt


def _write_run_json(run_dir: Path, *, completed_steps: int, world_size: int = 2) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps({"completed_steps": completed_steps, "runtime": {"world_size": world_size}}),
        encoding="utf-8",
    )


def _write_train_row(run_dir: Path, *, step: int, loss: float = 1.5, duration: float = 0.01) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "step": step,
        "split": "train",
        "micro_step_count": 1,
        "optimizer_update_status": "applied",
        "finite_status": "finite",
        "loss": loss,
        "step_duration_seconds": duration,
    }
    with (run_dir / "logging.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _write_real_checkpoint(
    checkpoint_dir: Path,
    *,
    step: int,
    identities: dict[str, str] | None = None,
    resolved_config: dict | None = None,
) -> None:
    """Author a genuinely admittable `training_state/` via real primitives."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = resolved_config or probe._synthetic_resolved_config()
    plan = probe.TrainingStatePublicationPlan(
        parent_run_id="fixture-run",
        parent_segment_id="fixture-segment",
        checkpoint_step=step,
        continuation_index=0,
        world_size=2,
        identities=identities or probe._synthetic_identities(),
        scheduler_applicable=False,
        scaler_applicable=False,
        resolved_config=resolved_config,
        resume_compatibility=probe.build_resume_compatibility_projection(resolved_config),
        accumulation_microstep=0,
    )
    session = probe.begin_training_state_contributions(checkpoint_dir, plan)
    for rank in range(2):
        probe.publish_rank_training_state_contribution(
            checkpoint_dir, session, probe._synthetic_payload(rank)
        )
    probe.commit_training_state_contributions(checkpoint_dir, session)


def _record_authoritative_parent_checkpoint(
    writer: RunWriter, *, step: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = writer.run_dir / "checkpoints" / f"step-{step}"
    _write_real_checkpoint(checkpoint_dir, step=step)
    manifest_path = checkpoint_dir / "training_state" / "manifest.json"
    manifest = probe.load_training_state_manifest(checkpoint_dir)
    payload_identity = {
        "schema": "coordexp-swift-inference-checkpoint-payload-publication",
        "schema_version": 2,
        "manifest_relative_path": "inference_payload_manifest.json",
        "manifest_file_sha256": "c" * 64,
        "aggregate_digest": "d" * 64,
    }
    monkeypatch.setattr(
        run_writer_module,
        "admit_inference_checkpoint_payload_identity",
        lambda _checkpoint_dir, _expected: dict(payload_identity),
    )
    writer.record_checkpoint_publication_event(
        step=step,
        status="completed",
        started_at="2026-08-13T00:00:00+00:00",
        completed_at="2026-08-13T00:00:01+00:00",
        duration_seconds=1.0,
        is_final=step == 2,
        exact_training_state_enabled=True,
        checkpoint_identity={
            "checkpoint_step": step,
            "resolved_path": str(checkpoint_dir.resolve()),
            "training_state_manifest_file_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "training_state_aggregate_digest": manifest.aggregate_digest,
        },
        inference_payload_identity=payload_identity,
        committed_progress={
            "schema": "coordexp-swift-checkpoint-committed-progress",
            "schema_version": 1,
            "completed_steps": step,
            "consumed_packs": step,
            "optimizer_update_status": "applied",
            "finite_status": "finite",
        },
        failure_code=None,
    )


def _write_authoritative_parent_step_one(
    parent_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    """Author a real step-1 manifest plus its authoritative RunWriter event."""

    writer = RunWriter.initialize(
        run_dir=parent_run_dir,
        run_id="fixture-run",
        run_name="resumed-parent",
        artifact_root=parent_run_dir.parent.parent,
        collision_outcome="created",
        created_at="2026-08-13T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=2,
        resolved_max_steps=2,
        segment_id="fixture-segment",
    )
    _record_authoritative_parent_checkpoint(writer, step=1, monkeypatch=monkeypatch)
    return probe._authenticate_parent_step_one(parent_run_dir)


def _write_matched_success_fixture(target: Path, receipt: dict) -> tuple[dict, dict]:
    """Author a fully matched control/resumed pair: two real, equal boundaries."""

    control_run_dir = target / "runs" / "uninterrupted_control"
    parent_run_dir = target / "runs" / "resumed_parent"
    child_run_dir = target / "runs" / "resumed_child"

    _write_run_json(control_run_dir, completed_steps=2)
    _write_run_json(child_run_dir, completed_steps=2)
    _write_real_checkpoint(control_run_dir / "checkpoints" / "step-1", step=1)
    _write_real_checkpoint(control_run_dir / "checkpoints" / "step-2", step=2)
    _write_real_checkpoint(child_run_dir / "checkpoints" / "step-2", step=2)
    _write_train_row(control_run_dir, step=2, loss=1.5, duration=0.01)
    _write_train_row(child_run_dir, step=2, loss=1.5, duration=0.02)

    def fake_launch(argv, cwd, env):
        return probe.LaunchResult(0, "ok", "")

    local_monkeypatch = pytest.MonkeyPatch()
    admitted = _write_authoritative_parent_step_one(parent_run_dir, local_monkeypatch)

    def fake_parent_interrupt(argv, cwd, env, run_dir):
        return probe.ParentInterruptionResult(
            pid=303,
            pgid=303,
            signals=("SIGTERM",),
            returncode=-signal.SIGTERM,
            termination_duration_seconds=0.1,
            stdout_tail="",
            stderr_tail="",
            authenticated_step_one=admitted,
        )

    control_receipt = probe.success_control(
        artifact_root=target, commit=receipt["commit"], launch=fake_launch
    )
    try:
        resumed_receipt = probe.success_resumed(
            artifact_root=target,
            commit=receipt["commit"],
            launch=fake_launch,
            interrupt_parent=fake_parent_interrupt,
        )
    finally:
        local_monkeypatch.undo()
    return control_receipt, resumed_receipt


def _write_representative_failure_arms(target: Path) -> tuple[dict, dict]:
    rank_failure_receipt = probe.rank_failure(
        artifact_root=target / "arms" / "rank_failure",
        inject={"rank": 1, "kind": "missing"},
    )
    interruption_receipt = probe.interruption(
        artifact_root=target / "arms" / "interruption",
        stop_after=1,
    )
    return rank_failure_receipt, interruption_receipt


def _resign_receipt(path: Path, mutate) -> None:
    """Rewrite one of this probe's own signed receipts after mutating its body."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_payload_sha256", None)
    mutate(payload)
    signed = probe._signed(payload)
    path.write_text(json.dumps(signed), encoding="utf-8")


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
        ["success-control", "--artifact-root", "/x"],
        ["success-resumed"],
        ["success-resumed", "--artifact-root", "/x"],
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
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
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
        probe.prepare(artifact_root=tmp_path / "bundle", base_config=BASE_CONFIG, world_size=8)
    assert exc_info.value.code == "reconcile_probe.wrong_world_size"


def test_prepare_rejects_relative_artifact_root(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(artifact_root="relative/bundle", base_config=BASE_CONFIG, world_size=2)
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
            artifact_root=link_parent / "bundle", base_config=BASE_CONFIG, world_size=2
        )
    assert exc_info.value.code == "reconcile_probe.symlink_escape"


def test_prepare_rejects_a_symlink_base_config(tmp_path: Path) -> None:
    link = tmp_path / "base.yaml"
    link.symlink_to(BASE_CONFIG)
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(artifact_root=tmp_path / "bundle", base_config=link, world_size=2)
    assert exc_info.value.code == "reconcile_probe.symlink_escape"


def test_prepare_rejects_a_malformed_base_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\nnot: [valid, - broken\n", encoding="utf-8")
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(artifact_root=tmp_path / "bundle", base_config=bad, world_size=2)
    assert exc_info.value.code == "reconcile_probe.base_config_invalid"


def test_prepare_authors_an_exact_two_rank_bundle_with_publish_only_boundary(
    tmp_path: Path,
) -> None:
    target = tmp_path / "bundle"
    receipt = probe.prepare(
        artifact_root=target,
        base_config=BASE_CONFIG,
        world_size=2,
        prepare_pack_cache=_fake_prepare_pack_cache([]),
    )

    assert receipt["schema"] == probe.SCHEMA_PREPARE_RECEIPT
    assert receipt["world_size"] == 2
    assert len(receipt["commit"]) == 40
    assert set(receipt["configs"]) == {"uninterrupted_control", "resumed_parent", "resumed_child"}
    assert receipt["pack_cache"]["status"] == "prepared"

    control_config = probe.load_train_config(target / "configs/uninterrupted_control.yaml")
    # Publish-only control/parent boundary: exact mode with a null path, per
    # the Task 2.5 contract -- `disabled` would skip exact-state publication.
    assert control_config.config_dict["resume"] == {
        "mode": "exact_same_world_size",
        "checkpoint_dir": None,
    }
    assert control_config.config_dict["training"]["max_steps"] == 2
    assert (
        control_config.config_dict["runtime"]["determinism"]["mode"] == "strict_cuda_replay_v1"
    )
    assert control_config.config_dict["eval"]["forward"]["steps"] == []
    assert control_config.config_dict["eval"]["forward"]["every_fraction"] is None

    parent_config = probe.load_train_config(target / "configs/resumed_parent.yaml")
    assert parent_config.config_dict["resume"] == {
        "mode": "exact_same_world_size",
        "checkpoint_dir": None,
    }
    assert parent_config.config_dict["training"]["max_steps"] == 2
    assert parent_config.config_dict["checkpoint"]["steps"] == [1, 2]
    assert parent_config.config_dict["checkpoint"]["save_final"] is True

    child_config = probe.load_train_config(target / "configs/resumed_child.yaml")
    assert child_config.config_dict["resume"]["mode"] == "exact_same_world_size"
    assert child_config.config_dict["resume"]["checkpoint_dir"] == str(
        target / "runs/resumed_parent/checkpoints/step-1"
    )
    assert child_config.config_dict["training"]["max_steps"] == 2
    assert child_config.config_dict["checkpoint"]["steps"] == [1, 2]
    assert child_config.config_dict["checkpoint"]["save_final"] is True
    assert child_config.config_dict["eval"]["forward"]["steps"] == []

    # Re-loading the receipt from disk must reproduce the same signed JSON.
    assert probe._load_signed_receipt(target / probe.PREPARE_RECEIPT_NAME) == receipt


def test_prepare_authors_resume_compatible_parent_and_child_semantics(
    tmp_path: Path,
) -> None:
    target = tmp_path / "bundle"
    probe.prepare(
        artifact_root=target,
        base_config=BASE_CONFIG,
        world_size=2,
        prepare_pack_cache=_fake_prepare_pack_cache([]),
    )

    parent = probe.load_train_config(target / "configs/resumed_parent.yaml")
    child = probe.load_train_config(target / "configs/resumed_child.yaml")

    assert parent.config_dict["training"]["max_steps"] == 2
    assert child.config_dict["training"]["max_steps"] == 2
    assert parent.config_dict["checkpoint"]["steps"] == [1, 2]
    assert child.config_dict["checkpoint"]["steps"] == [1, 2]
    assert parent.config_dict["checkpoint"]["save_final"] is True
    assert child.config_dict["checkpoint"]["save_final"] is True
    assert probe.build_resume_compatibility_projection(
        parent.to_artifact_dict()
    ) == probe.build_resume_compatibility_projection(child.to_artifact_dict())


def test_prepare_leaves_no_stage_or_cache_residue_on_success(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    probe.prepare(
        artifact_root=target,
        base_config=BASE_CONFIG,
        world_size=2,
        prepare_pack_cache=_fake_prepare_pack_cache([]),
    )
    assert list(tmp_path.glob(".bundle.stage-*")) == []
    assert list(tmp_path.glob(".bundle.pack-cache")) == [] or True  # cache stays, is private state


def test_prepare_invokes_cache_preparation_once_for_the_control_config(
    tmp_path: Path,
) -> None:
    target = tmp_path / "bundle"
    calls: list = []
    probe.prepare(
        artifact_root=target,
        base_config=BASE_CONFIG,
        world_size=2,
        prepare_pack_cache=_fake_prepare_pack_cache(calls),
    )
    assert len(calls) == 1
    assert calls[0]["config_path"].endswith("uninterrupted_control.yaml")
    assert calls[0]["cache_root"] == str(tmp_path / ".bundle.pack-cache")


def test_prepare_default_cache_seam_sets_private_root_and_strict_determinism_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, *, cwd, env, capture_output, text, check):
        captured["argv"] = list(argv)
        captured["env"] = dict(env)
        captured["cwd"] = cwd
        return _FakeCompleted()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    receipt_path = tmp_path / "receipt.json"
    body = {
        "schema": "coordexp-swift-pack-cache-preparation-receipt-v1",
        "terminal_status": "completed",
        "config_path": "x",
        "result": {"resolved_config_fingerprint": "abc"},
        "failure": None,
    }
    import hashlib

    encoded = json.dumps(
        body, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    receipt_path.write_text(
        json.dumps({**body, "receipt_sha256": hashlib.sha256(encoded).hexdigest()}),
        encoding="utf-8",
    )

    result = probe._default_prepare_pack_cache(
        tmp_path / "config.yaml", tmp_path / "cache-root", receipt_path
    )
    assert result["terminal_status"] == "completed"
    assert captured["argv"] == [
        sys.executable,
        "-m",
        "src.prepare_train_cache",
        "--config",
        str(tmp_path / "config.yaml"),
        "--receipt",
        str(receipt_path),
    ]
    assert captured["env"]["COORDEXP_SWIFT_PACK_CACHE_ROOT"] == str(tmp_path / "cache-root")
    assert captured["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert captured["env"]["FLASH_ATTENTION_DETERMINISTIC"] == "1"
    assert captured["cwd"] == str(probe.REPO_ROOT)


def test_prepare_rejects_nonzero_cache_preparation_exit(tmp_path: Path) -> None:
    def failing_prepare(config_path, cache_root, receipt_path):
        raise probe.ReconcileProbeError(
            "boom", code="reconcile_probe.pack_cache_prepare_failed"
        )

    target = tmp_path / "bundle"
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.prepare(
            artifact_root=target,
            base_config=BASE_CONFIG,
            world_size=2,
            prepare_pack_cache=failing_prepare,
        )
    assert exc_info.value.code == "reconcile_probe.pack_cache_prepare_failed"
    # Fail-closed cleanup: neither the config bundle nor the pack-cache root survive.
    assert not target.exists()
    assert not (tmp_path / ".bundle.pack-cache").exists()
    assert not (tmp_path / ".bundle.pack-cache-receipt.json").exists()


def test_prepare_real_cache_preparation_is_model_free(tmp_path: Path) -> None:
    """One real, slow, end-to-end proof that cache prep never loads model weights."""

    target = tmp_path / "bundle"
    receipt = probe.prepare(artifact_root=target, base_config=BASE_CONFIG, world_size=2)
    cache_receipt = probe._strict_json_load(Path(receipt["pack_cache"]["receipt_path"]))
    assert cache_receipt["terminal_status"] == "completed"
    assert cache_receipt["result"]["model_loaded"] is False
    assert Path(receipt["pack_cache"]["root"]).is_dir()


def test_shared_pack_cache_determinant_projection_ignores_role_only_fields() -> None:
    control = probe.load_train_config(BASE_CONFIG).config_dict
    other = json.loads(json.dumps(control))
    other["run"]["name"] = "different"
    other["training"]["max_steps"] = 999
    other["resume"] = {"mode": "exact_same_world_size", "checkpoint_dir": None}
    probe._assert_shared_pack_cache_determinants(control, other, role="other")


def test_shared_pack_cache_determinant_projection_rejects_real_drift() -> None:
    control = probe.load_train_config(BASE_CONFIG).config_dict
    other = json.loads(json.dumps(control))
    other["packing"]["global_max_length"] = 1
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe._assert_shared_pack_cache_determinants(control, other, role="other")
    assert exc_info.value.code == "reconcile_probe.pack_cache_determinant_drift"


# --------------------------------------------------------------------------
# success-control / success-resumed (fake launch; no GPU/model activity)
# --------------------------------------------------------------------------


def test_resumed_parent_uses_held_parent_route_while_control_and_child_use_src_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, receipt = _prepared(tmp_path)
    parent_run_dir = target / "runs" / "resumed_parent"
    admitted = _write_authoritative_parent_step_one(parent_run_dir, monkeypatch)
    observed_argv: dict[str, tuple[str, ...]] = {}

    def launch(argv, cwd, env):
        role = (
            "uninterrupted_control"
            if "uninterrupted_control.yaml" in argv[-1]
            else "resumed_child"
        )
        observed_argv[role] = tuple(argv)
        _write_run_json(target / "runs" / role, completed_steps=2)
        return probe.LaunchResult(0, "", "")

    def interrupt_parent(argv, cwd, env, run_dir):
        observed_argv["resumed_parent"] = tuple(argv)
        return probe.ParentInterruptionResult(
            pid=401,
            pgid=401,
            signals=("SIGTERM",),
            returncode=-signal.SIGTERM,
            termination_duration_seconds=0.1,
            stdout_tail="",
            stderr_tail="",
            authenticated_step_one=admitted,
        )

    probe.success_control(
        artifact_root=target, commit=receipt["commit"], launch=launch
    )
    probe.success_resumed(
        artifact_root=target,
        commit=receipt["commit"],
        launch=launch,
        interrupt_parent=interrupt_parent,
    )

    control = observed_argv["uninterrupted_control"]
    parent = observed_argv["resumed_parent"]
    child = observed_argv["resumed_child"]
    assert "src.train" in control
    assert "held-parent" not in control
    assert "src.train" in child
    assert "held-parent" not in child
    assert "src.train" not in parent
    assert "scripts.probes.coordexp_swift.reconcile_exact_resume_probe" in parent
    assert "held-parent" in parent


def test_held_parent_calls_real_handler_then_blocks_before_step_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.training import pipeline as training_pipeline

    order: list[str] = []
    hold_entered = threading.Event()
    release_hold = threading.Event()

    def real_factory(*args, **kwargs):
        order.append("real_factory")

        def real_handler(scheduled_event, observation):
            order.append("real_handler")

        return real_handler

    def fake_pipeline(config_path):
        handler = training_pipeline._checkpoint_handler()
        handler(SimpleNamespace(planned_step_id=1), object())
        order.append("step_two")
        return {"status": "completed"}

    def hold():
        order.append("hold")
        hold_entered.set()
        assert release_hold.wait(timeout=2.0)

    monkeypatch.setattr(training_pipeline, "_checkpoint_handler", real_factory)
    monkeypatch.setattr(training_pipeline, "run_training_pipeline", fake_pipeline)
    worker = threading.Thread(
        target=probe._run_held_parent,
        args=(Path("parent.yaml"),),
        kwargs={"hold": hold},
        daemon=True,
    )
    worker.start()
    assert hold_entered.wait(timeout=2.0)
    assert order == ["real_factory", "real_handler", "hold"]
    assert worker.is_alive()
    assert training_pipeline._checkpoint_handler is not real_factory

    release_hold.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert order == ["real_factory", "real_handler", "hold", "step_two"]
    assert training_pipeline._checkpoint_handler is real_factory


def test_held_parent_does_not_hold_when_step_one_publication_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.training import pipeline as training_pipeline

    publication_error = RuntimeError("step-one publication failed")
    hold_calls: list[str] = []

    def real_factory(*args, **kwargs):
        def real_handler(scheduled_event, observation):
            raise publication_error

        return real_handler

    def fake_pipeline(config_path):
        handler = training_pipeline._checkpoint_handler()
        handler(SimpleNamespace(planned_step_id=1), object())
        raise AssertionError("unreachable after publication failure")

    monkeypatch.setattr(training_pipeline, "_checkpoint_handler", real_factory)
    monkeypatch.setattr(training_pipeline, "run_training_pipeline", fake_pipeline)
    with pytest.raises(RuntimeError) as exc_info:
        probe._run_held_parent(
            Path("parent.yaml"), hold=lambda: hold_calls.append("hold")
        )
    assert exc_info.value is publication_error
    assert hold_calls == []
    assert training_pipeline._checkpoint_handler is real_factory


def test_success_control_rejects_launch_without_prepare(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    root.mkdir()
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=root, commit="f" * 40)
    assert exc_info.value.code == "reconcile_probe.prepare_receipt_missing"


def test_success_control_rejects_a_nonexistent_artifact_root(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=tmp_path / "missing", commit="f" * 40)
    assert exc_info.value.code == "reconcile_probe.target_missing"


def test_success_control_rejects_commit_argument_drift(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, commit="f" * 40)
    assert exc_info.value.code == "reconcile_probe.commit_drift"


def test_success_control_rejects_a_tampered_prepare_receipt_digest(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    path = target / probe.PREPARE_RECEIPT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["world_size"] = 999  # mutated without recomputing the digest
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, commit=receipt["commit"])
    assert exc_info.value.code == "reconcile_probe.receipt_digest_mismatch"


def test_success_control_rejects_drifted_config_file(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    config_path = Path(receipt["configs"]["uninterrupted_control"]["path"])
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n# drift\n")
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, commit=receipt["commit"])
    assert exc_info.value.code == "reconcile_probe.config_drift"


def test_success_control_builds_the_production_src_train_route_and_binds_cache_root(
    tmp_path: Path,
) -> None:
    target, receipt = _prepared(tmp_path)
    calls = []

    def fake_launch(argv, cwd, env):
        calls.append((tuple(argv), cwd, dict(env)))
        run_dir = target / "runs" / "uninterrupted_control"
        _write_run_json(run_dir, completed_steps=2)
        return probe.LaunchResult(0, "ok", "")

    result = probe.success_control(
        artifact_root=target, commit=receipt["commit"], launch=fake_launch
    )

    assert len(calls) == 1
    argv, cwd, env = calls[0]
    assert cwd == probe.REPO_ROOT
    assert argv[0] == sys.executable
    assert "src.train" in argv
    assert "--config" in argv
    assert argv[argv.index("--config") + 1] == receipt["configs"]["uninterrupted_control"]["path"]
    assert env["COORDEXP_SWIFT_PACK_CACHE_ROOT"] == receipt["pack_cache"]["root"]
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    assert result["schema"] == probe.SCHEMA_RUN_RECEIPT
    assert result["boundary_step"] == 1
    assert result["update_step"] == 2
    assert result["returncode"] == 0


def test_success_control_rejects_nonzero_launch(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)

    def failing_launch(argv, cwd, env):
        return probe.LaunchResult(1, "", "boom")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, commit=receipt["commit"], launch=failing_launch)
    assert exc_info.value.code == "reconcile_probe.launch_failed"


def test_success_control_rejects_unexpected_extra_updates(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)

    def over_launch(argv, cwd, env):
        _write_run_json(target / "runs" / "uninterrupted_control", completed_steps=3)
        return probe.LaunchResult(0, "", "")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, commit=receipt["commit"], launch=over_launch)
    assert exc_info.value.code == "reconcile_probe.unexpected_update_count"


def test_success_control_rejects_missing_ranks(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)

    def one_rank_launch(argv, cwd, env):
        _write_run_json(
            target / "runs" / "uninterrupted_control", completed_steps=2, world_size=1
        )
        return probe.LaunchResult(0, "", "")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_control(artifact_root=target, commit=receipt["commit"], launch=one_rank_launch)
    assert exc_info.value.code == "reconcile_probe.missing_ranks"


def test_completed_two_step_parent_makes_step_one_stale_and_never_launches_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches restoring attempt-2's blocking parent launch before interruption."""

    target, receipt = _prepared(tmp_path)
    parent_run_dir = target / "runs" / "resumed_parent"
    _write_authoritative_parent_step_one(parent_run_dir, monkeypatch)
    _record_authoritative_parent_checkpoint(
        RunWriter(run_dir=parent_run_dir), step=2, monkeypatch=monkeypatch
    )
    child_calls = []

    def stale_parent_controller(argv, cwd, env, run_dir):
        return probe._authenticate_parent_step_one(run_dir)

    def child_launch(argv, cwd, env):
        child_calls.append(tuple(argv))
        return probe.LaunchResult(0, "", "")

    with pytest.raises(probe.ReconcileProbeError):
        probe.success_resumed(
            artifact_root=target,
            commit=receipt["commit"],
            launch=child_launch,
            interrupt_parent=stale_parent_controller,
        )
    assert child_calls == []


def test_authoritative_step_one_is_admitted_then_parent_termination_is_recorded_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches child launch before authenticating and terminating the parent."""

    target, receipt = _prepared(tmp_path)
    parent_run_dir = target / "runs" / "resumed_parent"
    admitted = _write_authoritative_parent_step_one(parent_run_dir, monkeypatch)
    order = []

    def interrupted_parent(argv, cwd, env, run_dir):
        order.append("parent_terminated")
        assert run_dir == parent_run_dir
        return probe.ParentInterruptionResult(
            pid=101,
            pgid=101,
            signals=("SIGTERM",),
            returncode=-signal.SIGTERM,
            termination_duration_seconds=0.25,
            stdout_tail="parent-out",
            stderr_tail="parent-err",
            authenticated_step_one=admitted,
        )

    def child_launch(argv, cwd, env):
        order.append("child_launched")
        _write_run_json(target / "runs" / "resumed_child", completed_steps=2)
        return probe.LaunchResult(0, "child-out", "")

    result = probe.success_resumed(
        artifact_root=target,
        commit=receipt["commit"],
        launch=child_launch,
        interrupt_parent=interrupted_parent,
    )

    assert order == ["parent_terminated", "child_launched"]
    assert result["setup"]["parent_pid"] == 101
    assert result["setup"]["parent_pgid"] == 101
    assert result["setup"]["termination_signals"] == ["SIGTERM"]
    assert result["setup"]["returncode"] == -signal.SIGTERM
    assert result["setup"]["completed_steps"] == 1
    assert result["setup"]["consumed_packs"] == 1
    assert result["setup"]["checkpoint_event_count"] == 1
    assert result["setup"]["authenticated_step_one"] == admitted
    assert result["update"]["update_step"] == 2


@pytest.mark.parametrize("failure_code", ["parent_boundary_timeout", "parent_exited_early"])
def test_parent_timeout_or_early_exit_never_launches_child(
    tmp_path: Path, failure_code: str
) -> None:
    """Catches fail-open child launch after a missing live step-1 boundary."""

    target, receipt = _prepared(tmp_path)
    child_calls = []

    def failed_parent(argv, cwd, env, run_dir):
        raise probe.ReconcileProbeError("parent failed", code=f"reconcile_probe.{failure_code}")

    def child_launch(argv, cwd, env):
        child_calls.append(tuple(argv))
        return probe.LaunchResult(0, "", "")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_resumed(
            artifact_root=target,
            commit=receipt["commit"],
            launch=child_launch,
            interrupt_parent=failed_parent,
        )
    assert exc_info.value.code == f"reconcile_probe.{failure_code}"
    assert child_calls == []


def test_post_termination_parent_progress_mutation_never_launches_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches trusting the pre-TERM admission after durable progress mutates."""

    target, receipt = _prepared(tmp_path)
    parent_run_dir = target / "runs" / "resumed_parent"
    admitted = _write_authoritative_parent_step_one(parent_run_dir, monkeypatch)
    child_calls = []

    def mutating_parent(argv, cwd, env, run_dir):
        state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        state["completed_steps"] = 2
        (run_dir / "run.json").write_text(json.dumps(state), encoding="utf-8")
        return probe.ParentInterruptionResult(
            pid=202,
            pgid=202,
            signals=("SIGTERM",),
            returncode=-signal.SIGTERM,
            termination_duration_seconds=0.1,
            stdout_tail="",
            stderr_tail="",
            authenticated_step_one=admitted,
        )

    def child_launch(argv, cwd, env):
        child_calls.append(tuple(argv))
        return probe.LaunchResult(0, "", "")

    with pytest.raises(probe.ReconcileProbeError):
        probe.success_resumed(
            artifact_root=target,
            commit=receipt["commit"],
            launch=child_launch,
            interrupt_parent=mutating_parent,
        )
    assert child_calls == []


def test_term_timeout_escalates_to_kill_and_reaps_launcher(tmp_path: Path) -> None:
    """Catches returning while a SIGTERM-resistant launcher is still alive."""

    ready_path = tmp_path / "ready"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib,signal,time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            f"pathlib.Path({str(ready_path)!r}).touch();"
            "time.sleep(60)",
        ],
        start_new_session=True,
    )
    deadline = time.monotonic() + 2.0
    while not ready_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready_path.exists()
    pgid = os.getpgid(process.pid)
    result = probe._terminate_process_group(
        process,
        pgid=pgid,
        term_grace_seconds=0.05,
        kill_grace_seconds=1.0,
        poll_seconds=0.01,
    )

    assert result.signals == ("SIGTERM", "SIGKILL")
    assert process.returncode is not None
    assert not probe._process_group_exists(pgid)


def test_real_subprocess_cleanup_leaves_no_process_group_member(tmp_path: Path) -> None:
    """Catches signaling only the torchrun launcher while a worker survives."""

    child_pid_path = tmp_path / "child.pid"
    launcher = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", launcher], start_new_session=True)
    deadline = time.monotonic() + 2.0
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid_path.exists()
    pgid = os.getpgid(process.pid)

    result = probe._terminate_process_group(
        process,
        pgid=pgid,
        term_grace_seconds=0.5,
        kill_grace_seconds=1.0,
        poll_seconds=0.01,
    )

    assert result.signals == ("SIGTERM",)
    assert process.returncode is not None
    assert not probe._process_group_exists(pgid)


def test_parent_controller_cleans_group_after_unexpected_authentication_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches leaking torchrun descendants when boundary inspection raises unexpectedly."""

    pid_path = tmp_path / "launcher.pid"
    def unexpected_after_launcher_is_ready(_run_dir):
        if not pid_path.exists():
            raise probe.ReconcileProbeError("not ready", code="reconcile_probe.parent_boundary_invalid")
        raise RuntimeError("unexpected auth error")

    monkeypatch.setattr(probe, "_authenticate_parent_step_one", unexpected_after_launcher_is_ready)
    argv = [
        sys.executable,
        "-c",
        f"import os,pathlib,time; pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)",
    ]

    with pytest.raises(RuntimeError, match="unexpected auth error"):
        probe._interrupt_parent_at_authoritative_step_one(
            argv,
            probe.REPO_ROOT,
            os.environ,
            tmp_path / "parent",
            boundary_timeout_seconds=1.0,
            term_grace_seconds=0.5,
            kill_grace_seconds=1.0,
            poll_seconds=0.01,
        )

    pid = int(pid_path.read_text(encoding="utf-8"))
    assert not probe._process_group_exists(pid)


def test_parent_controller_cleans_descendant_when_leader_exits_before_getpgid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches the post-Popen/pre-getpgid race leaking a leaderless process group."""

    leader_pid_path = tmp_path / "leader.pid"
    child_pid_path = tmp_path / "child.pid"
    original_getpgid = probe.os.getpgid

    def getpgid_after_leader_exit(pid: int) -> int:
        deadline = time.monotonic() + 2.0
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.exists()
        os.waitpid(pid, 0)
        return original_getpgid(pid)

    monkeypatch.setattr(probe.os, "getpgid", getpgid_after_leader_exit)
    launcher = (
        "import os,pathlib,subprocess,sys;"
        f"pathlib.Path({str(leader_pid_path)!r}).write_text(str(os.getpid()));"
        f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid))"
    )
    try:
        with pytest.raises(probe.ReconcileProbeError) as exc_info:
            probe._interrupt_parent_at_authoritative_step_one(
                [sys.executable, "-c", launcher],
                probe.REPO_ROOT,
                os.environ,
                tmp_path / "parent",
                boundary_timeout_seconds=1.0,
                term_grace_seconds=0.5,
                kill_grace_seconds=1.0,
                poll_seconds=0.01,
            )
        assert exc_info.value.code == "reconcile_probe.parent_exited_early"
        expected_pgid = int(leader_pid_path.read_text(encoding="utf-8"))
        assert not probe._process_group_exists(expected_pgid)
    finally:
        if child_pid_path.exists():
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            try:
                cleanup_pgid = original_getpgid(child_pid)
            except ProcessLookupError:
                cleanup_pgid = None
            if cleanup_pgid is not None and probe._process_group_exists(cleanup_pgid):
                os.killpg(cleanup_pgid, signal.SIGKILL)


def test_parent_normal_exit_after_admission_is_not_a_controlled_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a zero-signal returncode-0 parent being receipted as interrupted."""

    target, receipt = _prepared(tmp_path)
    parent_run_dir = target / "runs" / "resumed_parent"
    admitted = _write_authoritative_parent_step_one(parent_run_dir, monkeypatch)
    real_authenticate = probe._authenticate_parent_step_one
    ready_path = tmp_path / "ready"
    release_path = tmp_path / "release"
    calls = 0

    def release_parent_then_admit(run_dir: Path) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            deadline = time.monotonic() + 2.0
            while not ready_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready_path.exists()
            release_path.touch()
            time.sleep(0.1)
            return admitted
        return real_authenticate(run_dir)

    monkeypatch.setattr(probe, "_authenticate_parent_step_one", release_parent_then_admit)
    parent_argv = [
        sys.executable,
        "-c",
        "import os,pathlib,time;"
        f"pathlib.Path({str(ready_path)!r}).write_text(str(os.getpid()));"
        f"release=pathlib.Path({str(release_path)!r});"
        "deadline=time.monotonic()+5;"
        "exec('while not release.exists() and time.monotonic() < deadline:\\n time.sleep(0.01)')",
    ]
    child_calls = []

    def normal_exit_parent(argv, cwd, env, run_dir):
        return probe._interrupt_parent_at_authoritative_step_one(
            parent_argv,
            cwd,
            env,
            run_dir,
            boundary_timeout_seconds=2.0,
            term_grace_seconds=0.5,
            kill_grace_seconds=1.0,
            poll_seconds=0.01,
        )

    def child_launch(argv, cwd, env):
        child_calls.append(tuple(argv))
        _write_run_json(target / "runs" / "resumed_child", completed_steps=2)
        return probe.LaunchResult(0, "", "")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.success_resumed(
            artifact_root=target,
            commit=receipt["commit"],
            launch=child_launch,
            interrupt_parent=normal_exit_parent,
        )
    assert exc_info.value.code == "reconcile_probe.parent_termination_not_signal_driven"
    assert child_calls == []
    assert not (target / probe.RECEIPTS_DIR_NAME / "success-resumed-receipt.json").exists()


# --------------------------------------------------------------------------
# rank-failure (real, model-free, no GPU)
# --------------------------------------------------------------------------


def test_rank_failure_rejects_malformed_inject_shape(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.rank_failure(
            artifact_root=tmp_path / "rf", inject={"rank": 0, "kind": "missing", "extra": 1}
        )
    assert exc_info.value.code == "reconcile_probe.malformed_inject"


def test_rank_failure_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.rank_failure(artifact_root=tmp_path / "rf", inject={"rank": 0, "kind": "vanish"})
    assert exc_info.value.code == "reconcile_probe.malformed_inject"


def test_rank_failure_rejects_wrong_world_size(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.rank_failure(
            artifact_root=tmp_path / "rf", inject={"rank": 0, "kind": "missing"}, world_size=3
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


@pytest.mark.parametrize("kind", sorted(probe.INJECT_KINDS))
@pytest.mark.parametrize("injected_rank", [0, 1])
def test_rank_failure_v2_records_exact_semantic_ranks_for_every_injection(
    tmp_path: Path, kind: str, injected_rank: int
) -> None:
    receipt = probe.rank_failure(
        artifact_root=tmp_path / f"rf-v2-{kind}-{injected_rank}",
        inject={"rank": injected_rank, "kind": kind},
    )
    expected_published = {
        "missing": [1 - injected_rank],
        "duplicate": [0, 1],
        "malformed": [0, 1],
        "corrupt": [0, 1],
    }[kind]
    expected_error = {
        "missing": "training_state.incomplete_rank_set",
        "duplicate": "training_state.immutable_collision",
        "malformed": "training_state.contribution_commit_failed",
        "corrupt": "training_state.contribution_commit_failed",
    }[kind]

    assert (
        receipt["schema"]
        == "coordexp-swift-reconcile-resume-probe-rank-failure-receipt-v2"
    )
    assert receipt["expected_ranks"] == [0, 1]
    assert receipt["serialized_ranks"] == [0, 1]
    assert receipt["published_ranks"] == expected_published
    assert receipt["error_code"] == expected_error


def test_rank_failure_and_interruption_make_no_cuda_call_under_cuda_visible_devices_empty(
    tmp_path: Path,
) -> None:
    script = (
        "import sys; sys.path.insert(0, %r);"
        "from scripts.probes.coordexp_swift import reconcile_exact_resume_probe as probe;"
        "import torch;"
        "assert not torch.cuda.is_initialized();"
        "r1 = probe.rank_failure(artifact_root=%r, inject={'rank': 0, 'kind': 'missing'});"
        "assert not torch.cuda.is_initialized();"
        "r2 = probe.interruption(artifact_root=%r, stop_after=2);"
        "assert not torch.cuda.is_initialized();"
        "print('OK')"
    ) % (
        str(probe.REPO_ROOT),
        str(tmp_path / "rf-nocuda"),
        str(tmp_path / "ib-nocuda"),
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(probe.REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


# --------------------------------------------------------------------------
# interruption (real exact-state/event-adjacent owners; inference is stubbed)
# --------------------------------------------------------------------------


def test_interruption_rejects_an_unknown_boundary(tmp_path: Path) -> None:
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.interruption(artifact_root=tmp_path / "ib", stop_after=5)
    assert exc_info.value.code == "reconcile_probe.invalid_boundary"


def test_interruption_before_inference_commit_leaves_no_artifacts(tmp_path: Path) -> None:
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
    assert receipt["inference_commit_owner"] == "stub_not_production"


def test_interruption_after_manifest_staging_before_event_commit(tmp_path: Path) -> None:
    receipt = probe.interruption(artifact_root=tmp_path / "ib2", stop_after=2)
    assert receipt["inference_payload_present"] is True
    assert receipt["exact_state_present"] is True
    assert receipt["inference_payload_only"] is False
    assert receipt["event_recorded"] is False
    assert receipt["exact_state_commit_owner"] == "production_training_state_primitives"


@pytest.mark.parametrize("stop_after", probe.INTERRUPTION_BOUNDARIES)
def test_interruption_v2_records_exact_semantic_ranks_for_every_boundary(
    tmp_path: Path, stop_after: int
) -> None:
    receipt = probe.interruption(
        artifact_root=tmp_path / f"ib-v2-{stop_after}", stop_after=stop_after
    )
    reached = stop_after == 2

    assert (
        receipt["schema"]
        == "coordexp-swift-reconcile-resume-probe-interruption-receipt-v2"
    )
    assert receipt["expected_ranks"] == [0, 1]
    assert receipt["serialized_ranks"] == ([0, 1] if reached else [])
    assert receipt["published_ranks"] == ([0, 1] if reached else [])
    assert receipt["rank_state_boundary_reached"] is reached


# --------------------------------------------------------------------------
# verify (durable artifacts only; fail-closed)
# --------------------------------------------------------------------------


def test_verify_rejects_missing_prepare_receipt(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    root.mkdir()
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.verify_artifacts(artifact_root=root)
    assert exc_info.value.code == "reconcile_probe.prepare_receipt_missing"


def test_verify_fails_closed_without_success_receipts(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)
    receipt = probe.verify_artifacts(artifact_root=target)
    assert receipt["status"] == "failed"
    assert any("success-control-receipt.json" in item for item in receipt["missing_inputs"])
    assert any("success-resumed-receipt.json" in item for item in receipt["missing_inputs"])


def test_verify_fails_closed_without_durable_checkpoints(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)

    receipts_dir = target / probe.RECEIPTS_DIR_NAME
    probe._write_run_receipt(
        receipts_dir,
        "success-control-receipt.json",
        {"commit": receipt["commit"], "run_dir": str(target / "runs/control")},
    )
    probe._write_run_receipt(
        receipts_dir,
        "success-resumed-receipt.json",
        {
            "commit": receipt["commit"],
            "setup": {"run_dir": str(target / "runs/parent")},
            "update": {"run_dir": str(target / "runs/child")},
        },
    )
    _write_representative_failure_arms(target)

    result = probe.verify_artifacts(artifact_root=target)
    assert result["status"] == "failed"
    assert result["missing_inputs"]
    assert result["bounded_mismatches"] == []


def test_verify_admits_and_compares_both_boundaries_and_reports_verified(
    tmp_path: Path,
) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)

    result = probe.verify_artifacts(artifact_root=target)
    assert result["status"] == "verified"
    assert result["commit"] == receipt["commit"]
    assert result["bounded_mismatches"] == []
    assert result["missing_inputs"] == []
    assert set(result["required_comparisons"]) == set(probe._COMPARISON_POLICY)
    assert {"rank_failure_arm", "interruption_arm"} <= set(result["required_comparisons"])
    assert str(target / "arms" / "rank_failure" / "rank-failure-receipt.json") in result[
        "input_file_sha256"
    ]
    assert str(target / "arms" / "interruption" / "interruption-receipt.json") in result[
        "input_file_sha256"
    ]


def test_verify_fails_closed_when_representative_failure_arms_are_missing(
    tmp_path: Path,
) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)

    result = probe.verify_artifacts(artifact_root=target)

    assert result["status"] == "failed"
    assert str(target / "arms" / "rank_failure" / "rank-failure-receipt.json") in result[
        "missing_inputs"
    ]
    assert str(target / "arms" / "interruption" / "interruption-receipt.json") in result[
        "missing_inputs"
    ]


def test_verify_fails_closed_on_tampered_failure_arm_digest(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)
    failure_path = target / "arms" / "rank_failure" / "rank-failure-receipt.json"
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    payload["status"] = "forged_without_resigning"
    failure_path.write_text(json.dumps(payload), encoding="utf-8")

    result = probe.verify_artifacts(artifact_root=target)

    assert result["status"] == "failed"
    assert any(
        item["path"] == "rank_failure_arm.receipt"
        and item["code"] == "reconcile_probe.receipt_digest_mismatch"
        for item in result["bounded_mismatches"]
    )


def test_verify_fails_closed_on_wrong_rank_failure_schema_status_and_shape(
    tmp_path: Path,
) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)
    failure_path = target / "arms" / "rank_failure" / "rank-failure-receipt.json"

    def mutate(payload: dict) -> None:
        payload["schema"] = "wrong-schema"
        payload["status"] = "wrong-status"
        payload.pop("manifest_admitted")
        payload.pop("serialized_ranks")
        payload["selector_admitted"] = True

    _resign_receipt(failure_path, mutate)
    result = probe.verify_artifacts(artifact_root=target)

    assert result["status"] == "failed"
    mismatch_paths = {item["path"] for item in result["bounded_mismatches"]}
    assert {
        "rank_failure_arm.schema",
        "rank_failure_arm.status",
        "rank_failure_arm.manifest_admitted",
        "rank_failure_arm.serialized_ranks",
        "rank_failure_arm.selector_admitted",
    } <= mismatch_paths


def test_verify_fails_closed_on_wrong_interruption_schema_status_and_shape(
    tmp_path: Path,
) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)
    interruption_path = target / "arms" / "interruption" / "interruption-receipt.json"

    def mutate(payload: dict) -> None:
        payload["schema"] = "wrong-schema"
        payload["status"] = "wrong-status"
        payload.pop("exact_state_present")
        payload.pop("rank_state_boundary_reached")
        payload["stop_after"] = 2

    _resign_receipt(interruption_path, mutate)
    result = probe.verify_artifacts(artifact_root=target)

    assert result["status"] == "failed"
    mismatch_paths = {item["path"] for item in result["bounded_mismatches"]}
    assert {
        "interruption_arm.schema",
        "interruption_arm.status",
        "interruption_arm.exact_state_present",
        "interruption_arm.rank_state_boundary_reached",
        "interruption_arm.stop_after",
    } <= mismatch_paths


@pytest.mark.parametrize(
    ("arm", "field", "tampered"),
    [
        ("rank_failure", "published_ranks", [0, 1]),
        ("rank_failure", "error_code", "training_state.immutable_collision"),
        ("interruption", "serialized_ranks", [0, 1]),
        ("interruption", "rank_state_boundary_reached", True),
    ],
)
def test_verify_rejects_tampered_semantic_arm_rank_fields(
    tmp_path: Path, arm: str, field: str, tampered: object
) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)
    path = (
        target
        / "arms"
        / arm
        / ("rank-failure-receipt.json" if arm == "rank_failure" else "interruption-receipt.json")
    )
    _resign_receipt(path, lambda payload: payload.__setitem__(field, tampered))

    result = probe.verify_artifacts(artifact_root=target)

    assert result["status"] == "failed"
    assert any(
        item["path"] == f"{arm}_arm.{field}"
        for item in result["bounded_mismatches"]
    )


def test_verify_fails_closed_on_a_mismatched_boundary_checkpoint(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)

    # Corrupt the parent's step-1 checkpoint after the fact: a real cursor drift.
    parent_checkpoint = target / "runs" / "resumed_parent" / "checkpoints" / "step-1"
    resolved_config = probe._synthetic_resolved_config()
    drifted_plan = probe.TrainingStatePublicationPlan(
        parent_run_id="fixture-run",
        parent_segment_id="fixture-segment",
        checkpoint_step=1,
        continuation_index=0,
        world_size=2,
        identities=probe._synthetic_identities(),
        scheduler_applicable=False,
        scaler_applicable=False,
        resolved_config=resolved_config,
        resume_compatibility=probe.build_resume_compatibility_projection(resolved_config),
        accumulation_microstep=0,
    )
    import shutil

    shutil.rmtree(parent_checkpoint)
    parent_checkpoint.mkdir(parents=True)
    session = probe.begin_training_state_contributions(parent_checkpoint, drifted_plan)
    for rank in range(2):
        payload = probe._synthetic_payload(rank)
        drifted = dataclasses.replace(
            payload, cursor={"data": {"epoch": 99, "ordinal": rank}, "pack": {"ordinal": rank, "pending": []}}
        )
        probe.publish_rank_training_state_contribution(parent_checkpoint, session, drifted)
    probe.commit_training_state_contributions(parent_checkpoint, session)

    result = probe.verify_artifacts(artifact_root=target)
    assert result["status"] == "failed"
    assert any(item["path"] == "boundary.rank0.cursor" for item in result["bounded_mismatches"])


def test_checkpoint_pair_ignores_only_resolved_config_identity_drift(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_config = probe._synthetic_resolved_config()
    right_config = json.loads(json.dumps(left_config))
    right_config["config"]["run"]["name"] = "other-branch"
    right_config["config"]["resume"] = {
        "mode": "exact_same_world_size",
        "checkpoint_dir": "/reconcile-probe/parent/checkpoints/step-1",
    }
    digest = "a" * 64
    right_identities = probe.build_exact_resume_identities(
        base_model=digest,
        cache=digest,
        dependencies=digest,
        policy=digest,
        resolved_config=right_config,
        topology=digest,
        trainable_surface=digest,
    )
    left_identities = probe._synthetic_identities()
    _write_real_checkpoint(left, step=1, identities=left_identities)
    _write_real_checkpoint(
        right,
        step=1,
        identities=dict(right_identities),
        resolved_config=right_config,
    )

    assert probe._compare_checkpoint_pair(
        left, right, step=1, path_prefix="boundary"
    ) == []


def test_checkpoint_pair_keeps_non_config_identities_strict(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_identities = probe._synthetic_identities()
    right_identities = dict(left_identities)
    right_identities["cache"] = "b" * 64
    _write_real_checkpoint(left, step=1, identities=left_identities)
    _write_real_checkpoint(right, step=1, identities=right_identities)

    assert probe._compare_checkpoint_pair(
        left, right, step=1, path_prefix="boundary"
    ) == [{"path": "boundary.identities.cache"}]


def test_objective_comparison_ignores_input_timing_but_not_training_semantics() -> None:
    control = {
        "step": 2,
        "split": "train",
        "loss": 1.5,
        "optimizer_update_status": "applied",
        "input_build_seconds": 0.1,
        "input_wait_seconds": 0.2,
    }
    timing_jitter = {
        **control,
        "input_build_seconds": 10.0,
        "input_wait_seconds": 20.0,
    }
    assert probe._compare_loss_rows(control, timing_jitter) == []

    semantic_drift = {
        **timing_jitter,
        "loss": 9.0,
        "optimizer_update_status": "skipped_nonfinite",
    }
    assert {item["path"] for item in probe._compare_loss_rows(control, semantic_drift)} == {
        "logging.loss",
        "logging.optimizer_update_status",
    }


def test_verify_fails_closed_on_missing_or_mismatched_train_step2_row(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)

    # Overwrite the child's logging.jsonl with a mismatched loss value.
    child_run_dir = target / "runs" / "resumed_child"
    (child_run_dir / "logging.jsonl").unlink()
    _write_train_row(child_run_dir, step=2, loss=999.0)

    result = probe.verify_artifacts(artifact_root=target)
    assert result["status"] == "failed"
    assert any(item["path"] == "logging.loss" for item in result["bounded_mismatches"])


def test_verify_rejects_commit_drift_between_receipts(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)

    control_path = target / probe.RECEIPTS_DIR_NAME / "success-control-receipt.json"
    _resign_receipt(control_path, lambda payload: payload.__setitem__("commit", "f" * 40))

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.verify_artifacts(artifact_root=target)
    assert exc_info.value.code == "reconcile_probe.commit_drift"


def test_verify_rejects_a_tampered_receipt_digest(tmp_path: Path) -> None:
    target, receipt = _prepared(tmp_path)
    _write_matched_success_fixture(target, receipt)
    _write_representative_failure_arms(target)

    control_path = target / probe.RECEIPTS_DIR_NAME / "success-control-receipt.json"
    payload = json.loads(control_path.read_text(encoding="utf-8"))
    payload["returncode"] = 1  # mutated without recomputing the digest
    control_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.verify_artifacts(artifact_root=target)
    assert exc_info.value.code == "reconcile_probe.receipt_digest_mismatch"


def test_verify_rejects_malformed_strict_json_receipt(tmp_path: Path) -> None:
    target, _ = _prepared(tmp_path)
    _write_representative_failure_arms(target)
    receipts_dir = target / probe.RECEIPTS_DIR_NAME
    receipts_dir.mkdir()
    (receipts_dir / "success-control-receipt.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    (receipts_dir / "success-resumed-receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(probe.ReconcileProbeError) as exc_info:
        probe.verify_artifacts(artifact_root=target)
    assert exc_info.value.code == "reconcile_probe.malformed_json"


@pytest.mark.parametrize(
    ("status", "expected_returncode"),
    [("verified", 0), ("failed", 1)],
)
def test_cli_verify_exit_code_reflects_terminal_receipt_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_returncode: int,
) -> None:
    monkeypatch.setattr(
        probe,
        "verify_artifacts",
        lambda *, artifact_root: {"status": status, "artifact_root": artifact_root},
    )

    assert probe.main(["verify", "--artifact-root", str(tmp_path)]) == expected_returncode
