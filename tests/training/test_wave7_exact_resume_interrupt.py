from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any

import pytest

from src.artifacts.training_state import (
    REQUIRED_IDENTITY_KINDS,
    REQUIRED_RNG_KINDS,
    TRAINING_STATE_ARTIFACT_TYPE,
    TRAINING_STATE_COMMIT_STATUS,
    TRAINING_STATE_RESOLVED_CONFIG,
    TRAINING_STATE_RESUME_COMPATIBILITY,
    TRAINING_STATE_SAVE_BOUNDARY,
    TRAINING_STATE_SCHEMA,
    TRAINING_STATE_SCHEMA_VERSION,
    TrainingStateFile,
    TrainingStateManifest,
    TrainingStateRank,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_interrupt.py"
BASE_TRAIN_CONFIG = (
    REPO_ROOT / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
    "accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
)


@pytest.fixture
def controller() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "wave7_exact_resume_interrupt", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload(*, malformed: bool = False) -> bytes:
    if malformed:
        return b'{"schema":'
    identities = {
        name: f"{index + 1:x}" * 64
        for index, name in enumerate(REQUIRED_IDENTITY_KINDS)
    }
    resolved = TrainingStateFile(
        path=TRAINING_STATE_RESOLVED_CONFIG,
        role="resolved_config",
        size=1,
        sha256=identities["resolved_config"],
    )
    compatibility = TrainingStateFile(
        path=TRAINING_STATE_RESUME_COMPATIBILITY,
        role="resume_compatibility",
        size=1,
        sha256=identities["resume_compatibility"],
    )
    roles = {
        "cursor": "cursor.json",
        "optimizer": "optimizer.bin",
        "trainable_model": "trainable-model.bin",
        **{f"rng:{kind}": f"rng-{kind}.bin" for kind in REQUIRED_RNG_KINDS},
    }
    ranks = tuple(
        TrainingStateRank(
            rank=rank,
            rng_kinds=REQUIRED_RNG_KINDS,
            next_rank_local_micro_step=3,
            runtime_signature="f" * 64,
            files=tuple(
                sorted(
                    (
                        TrainingStateFile(
                            path=f"rank-{rank:05d}/{name}",
                            role=role,
                            size=1,
                            sha256="e" * 64,
                        )
                        for role, name in roles.items()
                    ),
                    key=lambda item: item.path,
                )
            ),
        )
        for rank in range(8)
    )
    manifest = TrainingStateManifest(
        schema=TRAINING_STATE_SCHEMA,
        schema_version=TRAINING_STATE_SCHEMA_VERSION,
        artifact_type=TRAINING_STATE_ARTIFACT_TYPE,
        commit_status=TRAINING_STATE_COMMIT_STATUS,
        parent_run_id="parent-run",
        parent_segment_id="parent-segment",
        checkpoint_step=3,
        continuation_index=0,
        world_size=8,
        save_boundary=TRAINING_STATE_SAVE_BOUNDARY,
        identities=identities,
        optimizer_applicable=True,
        scheduler_applicable=False,
        scaler_applicable=False,
        resolved_config=resolved,
        resume_compatibility=compatibility,
        ranks=ranks,
        aggregate_digest="0" * 64,
    )
    manifest = replace(manifest, aggregate_digest=manifest.computed_aggregate_digest())
    return (
        json.dumps(
            manifest.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_run_fixture(
    tmp_path: Path, *, malformed_manifest: bool = False
) -> tuple[Path, Path]:
    run_dir = tmp_path / "parent-run"
    manifest_template = tmp_path / "manifest-template.json"
    manifest_template.write_bytes(_manifest_payload(malformed=malformed_manifest))
    return run_dir, manifest_template


def _populate_existing_run(run_dir: Path, manifest_template: Path) -> None:
    state = run_dir / "checkpoints/step-3/training_state"
    state.mkdir(parents=True)
    manifest = json.loads(manifest_template.read_text(encoding="utf-8"))
    checkpoint_dir = run_dir / "checkpoints/step-3"
    event = {
        "checkpoint_identity": {
            "checkpoint_step": 3,
            "resolved_path": str(checkpoint_dir.resolve()),
            "training_state_aggregate_digest": manifest["aggregate_digest"],
            "training_state_manifest_file_sha256": hashlib.sha256(
                manifest_template.read_bytes()
            ).hexdigest(),
        },
        "checkpoint_path": "checkpoints/step-3",
        "completed_at": "2026-08-11T00:00:01+00:00",
        "duration_clock": "monotonic",
        "duration_seconds": 0.1,
        "exact_training_state_enabled": True,
        "failure_code": None,
        "is_final": False,
        "started_at": "2026-08-11T00:00:00+00:00",
        "status": "completed",
        "step": 3,
    }
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "measurement": {"checkpoint_publication_events": [event]},
                "status": "running",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "logging.jsonl").write_text(
        json.dumps({"split": "train", "step": 3}) + "\n", encoding="utf-8"
    )
    (state / "manifest.json").write_bytes(manifest_template.read_bytes())


def _write_target_config(tmp_path: Path, target_run_dir: Path) -> Path:
    path = tmp_path / "target-config.yaml"
    path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"extends: {BASE_TRAIN_CONFIG}",
                "run:",
                f"  name: {target_run_dir.name}",
                f"  artifact_root: {target_run_dir.parent}",
                "  collision_policy: fail",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _write_nested_launcher(tmp_path: Path) -> Path:
    helper = tmp_path / "nested_launcher.py"
    helper.write_text(
        """
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

role, run_text, template_text, marker_text = sys.argv[1:5]
run_dir = Path(run_text)
template = Path(template_text)
marker = Path(marker_text)
python = sys.executable

if role == "conda":
    child = subprocess.Popen(
        [python, __file__, "launcher", run_text, template_text, marker_text],
        start_new_session=True,
    )
    raise SystemExit(child.wait())
if role == "launcher":
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "run.json").exists():
        (run_dir / "run.json").write_text(
            '{"measurement":{"checkpoint_publication_events":[]},'
            '"status":"running"}\\n', encoding="utf-8"
        )
    if not (run_dir / "logging.jsonl").exists():
        (run_dir / "logging.jsonl").write_text(
            '{"split":"train","step":3}\\n', encoding="utf-8"
        )
    child = subprocess.Popen(
        [python, __file__, "manager", run_text, template_text, marker_text],
        start_new_session=True,
    )
    ready = run_dir / "workers.ready"
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    state = run_dir / "checkpoints/step-3/training_state"
    state.mkdir(parents=True, exist_ok=True)
    temporary = state / ".manifest.tmp"
    temporary.write_bytes(template.read_bytes())
    os.replace(temporary, state / "manifest.json")
    if template.name.startswith("delay-event-"):
        time.sleep(0.5)
    if not template.name.startswith("missing-event-"):
        manifest_bytes = (state / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        checkpoint_dir = run_dir / "checkpoints/step-3"
        failed = template.name.startswith("failed-event-")
        aggregate = manifest["aggregate_digest"]
        if template.name.startswith("mismatch-event-"):
            aggregate = "9" * 64
        identity = None if failed else {
            "checkpoint_step": 3,
            "resolved_path": str(checkpoint_dir.resolve()),
            "training_state_aggregate_digest": aggregate,
            "training_state_manifest_file_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
        }
        event = {
            "checkpoint_identity": identity,
            "checkpoint_path": "checkpoints/step-3",
            "completed_at": "2026-08-11T00:00:01+00:00",
            "duration_clock": "monotonic",
            "duration_seconds": 0.1,
            "exact_training_state_enabled": True,
            "failure_code": "injected.failure" if failed else None,
            "is_final": False,
            "started_at": "2026-08-11T00:00:00+00:00",
            "status": "failed" if failed else "completed",
            "step": 3,
        }
        run_path = run_dir / "run.json"
        run_payload = json.loads(run_path.read_text(encoding="utf-8"))
        run_payload["measurement"]["checkpoint_publication_events"].append(event)
        temporary_run = run_dir / ".run.json.tmp"
        temporary_run.write_text(
            json.dumps(run_payload, sort_keys=True) + "\\n", encoding="utf-8"
        )
        os.replace(temporary_run, run_path)
        (run_dir / "checkpoint-event.ready").write_text("ready", encoding="ascii")
    raise SystemExit(child.wait())
if role == "manager":
    children = [
        subprocess.Popen(
            [python, __file__, "worker", run_text, template_text, marker_text],
            start_new_session=True,
        )
        for _ in range(8)
    ]
    (run_dir / "workers.ready").write_text(
        " ".join(str(child.pid) for child in children), encoding="ascii"
    )
    for child in children:
        child.wait()
    raise SystemExit(0)
if role == "worker":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if template.name.startswith("spawn-late-"):
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.002)
        subprocess.Popen(
            [python, __file__, "late-worker", run_text, template_text, marker_text],
            start_new_session=True,
        )
    time.sleep(0.8)
    late = run_dir / "checkpoints/step-5"
    late.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints/final.json").write_text("late", encoding="ascii")
    while True:
        time.sleep(1)
if role == "late-worker":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (run_dir / "late.pid").write_text(str(os.getpid()), encoding="ascii")
    time.sleep(0.8)
    late = run_dir / "checkpoints/step-5"
    late.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints/final.json").write_text("late", encoding="ascii")
    while True:
        time.sleep(1)
if role == "exiting-launcher":
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [python, __file__, "orphan-worker", run_text, template_text, marker_text],
        start_new_session=True,
    )
    orphan = run_dir / "orphan.pid"
    deadline = time.monotonic() + 5
    while not orphan.exists() and time.monotonic() < deadline:
        time.sleep(0.002)
    raise SystemExit(0)
if role == "orphan-worker":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (run_dir / "orphan.pid").write_text(str(os.getpid()), encoding="ascii")
    time.sleep(0.02)
    state = run_dir / "checkpoints/step-3/training_state"
    state.mkdir(parents=True, exist_ok=True)
    temporary = state / ".manifest.tmp"
    temporary.write_bytes(template.read_bytes())
    os.replace(temporary, state / "manifest.json")
    time.sleep(0.8)
    late = run_dir / "checkpoints/step-5"
    late.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints/final.json").write_text("orphan-late", encoding="ascii")
    while True:
        time.sleep(1)
raise SystemExit(91)
""".lstrip(),
        encoding="utf-8",
    )
    return helper


def _write_fake_nvidia_smi(tmp_path: Path) -> Path:
    binary = tmp_path / "nvidia-smi"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _command(
    tmp_path: Path,
    *,
    run_dir: Path,
    manifest_template: Path,
    helper: Path,
    marker: Path,
    receipt: Path,
    config_run_dir: Path | None = None,
) -> tuple[list[str], Path]:
    target_config = _write_target_config(tmp_path, config_run_dir or run_dir)
    launcher = [
        sys.executable,
        str(helper),
        "conda",
        str(run_dir),
        str(manifest_template),
        str(marker),
        "--config",
        str(target_config),
    ]
    launcher_json = tmp_path / "launcher.json"
    launcher_json.write_text(json.dumps(launcher) + "\n", encoding="utf-8")
    return (
        [
            sys.executable,
            str(SCRIPT),
            "--launcher-json",
            str(launcher_json),
            "--parent-run-dir",
            str(run_dir),
            "--expected-checkpoint-step",
            "3",
            "--marker",
            str(marker),
            "--receipt",
            str(receipt),
            "--timeout-seconds",
            "4",
            "--term-grace-seconds",
            "0.08",
            "--kill-grace-seconds",
            "1",
            "--stability-seconds",
            "0.15",
            "--poll-seconds",
            "0.01",
            "--source",
            str(helper),
            "--config",
            str(target_config),
        ],
        launcher_json,
    )


def _run_cli(tmp_path: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    _write_fake_nvidia_smi(tmp_path)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )


def _popen_cli(tmp_path: Path, command: list[str]) -> subprocess.Popen[str]:
    _write_fake_nvidia_smi(tmp_path)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _proc_absent(pid: int) -> bool:
    return not Path(f"/proc/{pid}").exists()


def _mutate_step_three_event(run_dir: Path) -> None:
    run_path = run_dir / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    event = payload["measurement"]["checkpoint_publication_events"][0]
    event["checkpoint_identity"]["training_state_aggregate_digest"] = "9" * 64
    run_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_cli_captures_before_signal_kills_reparented_tree_and_prevents_late_write(
    tmp_path: Path,
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "attempt.json"
    receipt = tmp_path / "receipt.json"
    command, launcher_json = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )

    result = _run_cli(tmp_path, command)

    assert result.returncode == 0, result.stderr
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    graph = marker_payload["captured_process_graph"]
    assert len(graph) == 11
    assert {row["depth"] for row in graph} == {0, 1, 2, 3}
    assert len({row["session_id"] for row in graph}) == 11
    assert marker_payload["launcher"]["argv"] == json.loads(
        launcher_json.read_text(encoding="utf-8")
    )
    assert marker_payload["checkpoint"]["checkpoint_step"] == 3
    assert marker_payload["checkpoint"]["world_size"] == 8
    assert marker_payload["source_hashes"][0]["path"] == str(helper.resolve())
    assert receipt_payload["status"] == "passed"
    assert receipt_payload["termination"]["capture_completed_monotonic"] <= min(
        event["monotonic"]
        for event in receipt_payload["events"]
        if event["kind"].startswith("signal_")
    )
    assert receipt_payload["termination"]["remaining_pids"] == []
    assert receipt_payload["termination"]["remaining_pgids"] == []
    assert receipt_payload["postconditions"]["nvidia_compute_apps"] == []
    assert receipt_payload["postconditions"]["max_logged_train_step"] == 3
    assert receipt_payload["postconditions"]["late_write_detected"] is False
    assert not (run_dir / "checkpoints/step-5").exists()
    assert not (run_dir / "checkpoints/final.json").exists()
    assert all(_proc_absent(row["pid"]) for row in graph)


def test_manifest_does_not_trigger_marker_until_publication_event_is_durable(
    tmp_path: Path,
) -> None:
    run_dir, original_template = _write_run_fixture(tmp_path)
    manifest_template = tmp_path / "delay-event-manifest-template.json"
    manifest_template.write_bytes(original_template.read_bytes())
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "attempt.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    process = _popen_cli(tmp_path, command)
    manifest_path = run_dir / "checkpoints/step-3/training_state/manifest.json"
    deadline = time.monotonic() + 8
    while not manifest_path.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    marker_before_event = marker.exists()
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    events_before = run_payload["measurement"]["checkpoint_publication_events"]
    worker_pids = [
        int(value)
        for value in (run_dir / "workers.ready").read_text(encoding="ascii").split()
    ]
    workers_alive_before_event = all(not _proc_absent(pid) for pid in worker_pids)
    stdout, stderr = process.communicate(timeout=12)

    assert events_before == []
    assert marker_before_event is False
    assert workers_alive_before_event is True
    assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert (
        marker.stat().st_mtime_ns
        >= (run_dir / "checkpoint-event.ready").stat().st_mtime_ns
    )
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_payload["checkpoint_publication_event"]["step"] == 3


@pytest.mark.parametrize(
    ("template_name", "expected_code"),
    (
        ("missing-event-manifest-template.json", "wave7.checkpoint_event_timeout"),
        ("mismatch-event-manifest-template.json", "wave7.checkpoint_event_mismatch"),
        ("failed-event-manifest-template.json", "wave7.checkpoint_event_failed"),
    ),
)
def test_invalid_publication_event_fails_without_marker(
    tmp_path: Path, template_name: str, expected_code: str
) -> None:
    run_dir, original_template = _write_run_fixture(tmp_path)
    manifest_template = tmp_path / template_name
    manifest_template.write_bytes(original_template.read_bytes())
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "attempt.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    if template_name.startswith("missing-event-"):
        timeout_index = command.index("--timeout-seconds") + 1
        command[timeout_index] = "0.4"

    result = _run_cli(tmp_path, command)

    assert result.returncode == 1
    assert not marker.exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert expected_code in {error["code"] for error in payload["errors"]}


def test_cli_allows_launcher_to_create_fresh_parent_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "fresh" / "runs" / "interrupted-parent"
    manifest_template = tmp_path / "manifest-template.json"
    manifest_template.write_bytes(_manifest_payload())
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "attempt.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )

    result = _run_cli(tmp_path, command)

    assert result.returncode == 0, result.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["request"]["parent_run_dir"] == str(run_dir.resolve())


def test_existing_parent_with_valid_step_three_fails_before_launch(
    tmp_path: Path,
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    _populate_existing_run(run_dir, manifest_template)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "attempt.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    original_manifest = (
        run_dir / "checkpoints/step-3/training_state/manifest.json"
    ).read_bytes()

    result = _run_cli(tmp_path, command)

    assert result.returncode == 1
    assert not marker.exists()
    assert (
        run_dir / "checkpoints/step-3/training_state/manifest.json"
    ).read_bytes() == original_manifest
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["launch"]["attempted"] is False
    assert "wave7.parent_run_exists" in {error["code"] for error in payload["errors"]}


def test_launcher_config_target_mismatch_fails_before_launch(
    tmp_path: Path,
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "attempt.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
        config_run_dir=tmp_path / "other-runs" / "wrong-parent",
    )

    result = _run_cli(tmp_path, command)

    assert result.returncode == 1
    assert not marker.exists()
    assert not run_dir.exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["launch"]["attempted"] is False
    assert "wave7.parent_target_mismatch" in {
        error["code"] for error in payload["errors"]
    }


def test_remainder_launcher_argv_is_preserved_exactly(
    controller, tmp_path: Path
) -> None:
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    request = controller.parse_request(
        [
            "--parent-run-dir",
            str(tmp_path / "run"),
            "--marker",
            str(marker),
            "--receipt",
            str(receipt),
            "--timeout-seconds",
            "1",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ]
    )

    assert request.launcher_argv == (
        sys.executable,
        "-c",
        "raise SystemExit(0)",
    )


def test_surviving_captured_identity_publishes_failed_receipt(
    controller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, launcher_json = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    request = controller.parse_request(command[2:])
    request = replace(request, kill_grace_seconds=0.05, stability_seconds=0.01)
    real_remaining = controller._remaining_captured

    def retain_one(graph: tuple[Any, ...]) -> tuple[list[dict[str, Any]], list[int]]:
        pids, pgids = real_remaining(graph)
        if not pids:
            pids = [{"pid": graph[-1].pid, "state": "S", "identity_match": True}]
        return pids, pgids

    monkeypatch.setattr(controller, "_remaining_captured", retain_one)
    result = controller.execute(request, gpu_sampler=lambda: [])

    assert result == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "wave7.termination_survivor" in {
        error["code"] for error in payload["errors"]
    }
    assert payload["termination"]["remaining_pids"]
    assert json.loads(launcher_json.read_text()) == list(request.launcher_argv)


def test_late_write_after_termination_publishes_failed_receipt(
    controller, tmp_path: Path
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    request = replace(
        controller.parse_request(command[2:]),
        stability_seconds=0.3,
    )

    def append_late() -> None:
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.16)
        with (run_dir / "logging.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"split": "train", "step": 4}) + "\n")

    writer = threading.Thread(target=append_late)
    writer.start()
    result = controller.execute(request, gpu_sampler=lambda: [])
    writer.join(timeout=2)

    assert result == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["postconditions"]["late_write_detected"] is True
    assert "wave7.late_write" in {error["code"] for error in payload["errors"]}


def test_marker_mutation_after_publication_is_terminal_failure(
    controller, tmp_path: Path
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    request = controller.parse_request(command[2:])

    def mutate_marker() -> None:
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.002)
        marker.write_text('{"mutated":true}\n', encoding="utf-8")

    writer = threading.Thread(target=mutate_marker)
    writer.start()
    result = controller.execute(request, gpu_sampler=lambda: [])
    writer.join(timeout=2)

    assert result == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["marker"]["unchanged"] is False
    assert "wave7.marker_changed" in {error["code"] for error in payload["errors"]}


def test_publication_event_mutation_during_capture_fails_before_marker(
    controller, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    request = controller.parse_request(command[2:])
    real_capture = controller._capture_process_graph

    def mutate_after_capture(root_pid: int) -> tuple[Any, ...]:
        graph = real_capture(root_pid)
        _mutate_step_three_event(run_dir)
        return graph

    monkeypatch.setattr(controller, "_capture_process_graph", mutate_after_capture)

    result = controller.execute(request, gpu_sampler=lambda: [])

    assert result == 1
    assert not marker.exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert "wave7.checkpoint_event_changed" in {
        error["code"] for error in payload["errors"]
    }


def test_publication_event_mutation_after_marker_is_terminal_failure(
    controller, tmp_path: Path
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    request = controller.parse_request(command[2:])

    def mutate_event() -> None:
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.002)
        _mutate_step_three_event(run_dir)

    writer = threading.Thread(target=mutate_event)
    writer.start()
    result = controller.execute(request, gpu_sampler=lambda: [])
    writer.join(timeout=2)

    assert result == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["postconditions"]["checkpoint_publication_event_unchanged"] is False
    assert "wave7.checkpoint_event_changed" in {
        error["code"] for error in payload["errors"]
    }


def test_marker_mutation_during_stability_window_is_terminal_failure(
    controller, tmp_path: Path
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    request = replace(controller.parse_request(command[2:]), stability_seconds=0.4)

    def mutate_marker() -> None:
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.002)
        time.sleep(0.2)
        marker.write_text('{"mutated":"late"}\n', encoding="utf-8")

    writer = threading.Thread(target=mutate_marker)
    writer.start()
    result = controller.execute(request, gpu_sampler=lambda: [])
    writer.join(timeout=2)

    assert result == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["marker"]["unchanged"] is False
    assert "wave7.marker_changed" in {error["code"] for error in payload["errors"]}


def test_persistent_prelaunch_gpu_process_is_not_a_launcher_survivor(
    controller, tmp_path: Path
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    request = controller.parse_request(command[2:])
    unrelated = {"gpu_uuid": "GPU-existing", "pid": 4242, "process_name": "other"}

    result = controller.execute(request, gpu_sampler=lambda: [unrelated])

    assert result == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["postconditions"]["nvidia_compute_apps_baseline"] == [unrelated]
    assert payload["postconditions"]["nvidia_compute_apps_added"] == []


def test_new_postlaunch_gpu_process_is_terminal_failure(
    controller, tmp_path: Path
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    request = controller.parse_request(command[2:])
    unrelated = {"gpu_uuid": "GPU-existing", "pid": 4242, "process_name": "other"}
    added = {"gpu_uuid": "GPU-launch", "pid": 4343, "process_name": "python"}
    samples = iter(([unrelated], [unrelated, added]))

    result = controller.execute(request, gpu_sampler=lambda: next(samples))

    assert result == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["postconditions"]["nvidia_compute_apps_added"] == [added]
    assert "wave7.gpu_process_survivor" in {
        error["code"] for error in payload["errors"]
    }


def test_malformed_manifest_terminates_launch_and_publishes_failed_receipt(
    tmp_path: Path,
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path, malformed_manifest=True)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )

    result = _run_cli(tmp_path, command)

    assert result.returncode == 1
    assert not marker.exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "wave7.manifest_malformed" in {error["code"] for error in payload["errors"]}
    assert payload["termination"]["remaining_pids"] == []


def test_marker_collision_fails_before_launch_and_preserves_existing_marker(
    tmp_path: Path,
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    marker.write_text('{"owner":"someone-else"}\n', encoding="utf-8")
    original = marker.read_bytes()
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )

    result = _run_cli(tmp_path, command)

    assert result.returncode == 1
    assert marker.read_bytes() == original
    assert not (run_dir / "workers.ready").exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["launch"]["attempted"] is False
    assert "wave7.marker_collision" in {error["code"] for error in payload["errors"]}


def test_post_capture_reparented_session_is_discovered_killed_and_reaped(
    tmp_path: Path,
) -> None:
    run_dir, original_template = _write_run_fixture(tmp_path)
    manifest_template = tmp_path / "spawn-late-manifest-template.json"
    manifest_template.write_bytes(original_template.read_bytes())
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )

    result = _run_cli(tmp_path, command)
    late_pid = int((run_dir / "late.pid").read_text(encoding="ascii"))
    try:
        assert result.returncode == 0, result.stderr
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        initial = set(payload["termination"]["captured_pids"])
        final = {row["pid"] for row in payload["termination"]["captured_process_graph"]}
        assert late_pid not in initial
        assert late_pid in final
        assert late_pid in payload["termination"]["post_marker_discovered_pids"]
        assert _proc_absent(late_pid)
        assert not (run_dir / "checkpoints/step-5").exists()
        assert not (run_dir / "checkpoints/final.json").exists()
    finally:
        if not _proc_absent(late_pid):
            try:
                os.killpg(late_pid, 9)
            except ProcessLookupError:
                pass


def test_signal_order_is_deepest_pid_then_identity_proven_group(
    controller, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = controller.ProcessRecord(101, 1, 101, 101, "S", 11, 0)
    child = controller.ProcessRecord(102, 101, 101, 101, "S", 12, 1)
    graph = (parent, child)
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        controller, "_read_process", lambda pid: {101: parent, 102: child}[pid]
    )
    monkeypatch.setattr(
        controller,
        "_process_snapshot",
        lambda: {101: parent, 102: child},
    )
    monkeypatch.setattr(controller.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(
        controller.os, "kill", lambda pid, _signum: calls.append(("pid", pid))
    )
    monkeypatch.setattr(
        controller.os, "killpg", lambda pgid, _signum: calls.append(("pgid", pgid))
    )

    controller._signal_graph(graph, controller.signal.SIGTERM, events=[])

    assert calls == [("pid", 102), ("pid", 101), ("pgid", 101)]


def test_reused_pid_or_group_identity_is_never_signalled(
    controller, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = controller.ProcessRecord(201, 1, 201, 201, "S", 21, 0)
    replacement = controller.ProcessRecord(201, 1, 201, 201, "S", 22, 0)
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(controller, "_read_process", lambda _pid: replacement)
    monkeypatch.setattr(controller, "_process_snapshot", lambda: {201: replacement})
    monkeypatch.setattr(controller.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(
        controller.os, "kill", lambda pid, _signum: calls.append(("pid", pid))
    )
    monkeypatch.setattr(
        controller.os, "killpg", lambda pgid, _signum: calls.append(("pgid", pgid))
    )

    controller._signal_graph((captured,), controller.signal.SIGKILL, events=[])

    assert calls == []


def test_non_step_three_request_fails_before_launch_with_terminal_receipt(
    tmp_path: Path,
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, _ = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    step_index = command.index("--expected-checkpoint-step") + 1
    command[step_index] = "4"

    result = _run_cli(tmp_path, command)

    assert result.returncode == 1
    assert not marker.exists()
    assert not (run_dir / "workers.ready").exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["launch"]["attempted"] is False
    assert "wave7.expected_step" in {error["code"] for error in payload["errors"]}


def test_reused_parent_identity_cannot_adopt_or_signal_replacement_child(
    controller, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = controller.ProcessRecord(301, 1, 301, 301, "S", 31, 0)
    replacement = controller.ProcessRecord(301, 1, 301, 301, "S", 32, 0)
    unrelated_child = controller.ProcessRecord(302, 301, 302, 302, "S", 33, 0)
    monkeypatch.setattr(
        controller,
        "_process_snapshot",
        lambda: {301: replacement, 302: unrelated_child},
    )

    combined, discovered = controller._expand_related_processes(
        (captured,), baseline_controller_children={}
    )

    assert combined == (captured,)
    assert discovered == ()


def test_post_marker_process_expansion_rejects_more_than_graph_bound(
    controller, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = {
        pid: controller.ProcessRecord(
            pid,
            os.getpid(),
            pid,
            pid,
            "S",
            pid,
            0,
        )
        for pid in range(10_000, 10_000 + controller.MAX_GRAPH_PROCESSES + 1)
    }
    monkeypatch.setattr(controller, "_process_snapshot", lambda: snapshot)

    with pytest.raises(controller.Wave7InterruptError) as caught:
        controller._expand_related_processes((), baseline_controller_children={})

    assert caught.value.code == "wave7.process_inventory"
    assert caught.value.context == {
        "maximum": controller.MAX_GRAPH_PROCESSES,
        "observed": controller.MAX_GRAPH_PROCESSES + 1,
    }


def test_process_expansion_overflow_still_terminates_and_reaps_bounded_graph(
    controller, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = controller.ProcessRecord(9_000, 1, 9_000, 9_000, "S", 9_000, 0)
    snapshot = {captured.pid: captured}
    snapshot.update(
        {
            pid: controller.ProcessRecord(
                pid,
                os.getpid(),
                pid,
                pid,
                "S",
                pid,
                0,
            )
            for pid in range(10_000, 10_000 + controller.MAX_GRAPH_PROCESSES)
        }
    )
    signals: list[tuple[int, tuple[int, ...]]] = []
    monkeypatch.setattr(controller, "_process_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        controller,
        "_signal_graph",
        lambda graph, signum, events: signals.append(
            (signum, tuple(item.pid for item in graph))
        ),
    )
    monkeypatch.setattr(
        controller,
        "_remaining_captured",
        lambda graph: (
            ([{"pid": captured.pid, "state": "S", "identity_match": True}], [])
            if len(signals) < 3
            else ([], [])
        ),
    )
    monkeypatch.setattr(
        controller,
        "_reap_children",
        lambda process: [{"pid": captured.pid, "wait_status": 9}],
    )

    class Process:
        returncode = 0

        @staticmethod
        def poll() -> int:
            return 0

    request = controller.ControllerRequest(
        launcher_argv=("launcher",),
        parent_run_dir=tmp_path / "run",
        expected_checkpoint_step=3,
        marker=tmp_path / "marker.json",
        receipt=tmp_path / "receipt.json",
        timeout_seconds=1,
        term_grace_seconds=0,
        kill_grace_seconds=0,
        stability_seconds=0.01,
        poll_seconds=0.001,
        sources=(),
        configs=(),
    )

    termination = controller._terminate_captured(
        Process(),
        (captured,),
        request=request,
        capture_completed_monotonic=1.0,
        events=[],
        baseline_controller_children={},
    )

    assert any(
        signum == controller.signal.SIGKILL and captured.pid in pids
        for signum, pids in signals
    )
    assert termination["remaining_pids"] == []
    assert termination["remaining_pgids"] == []
    assert captured.pid in {row["pid"] for row in termination["reaped"]}
    assert "wave7.process_inventory" in {
        error["code"] for error in termination["cleanup_errors"]
    }


def test_exited_launcher_cleanup_is_seeded_from_subreaper_adoptee(
    tmp_path: Path,
) -> None:
    run_dir, manifest_template = _write_run_fixture(tmp_path)
    helper = _write_nested_launcher(tmp_path)
    marker = tmp_path / "marker.json"
    receipt = tmp_path / "receipt.json"
    command, launcher_json = _command(
        tmp_path,
        run_dir=run_dir,
        manifest_template=manifest_template,
        helper=helper,
        marker=marker,
        receipt=receipt,
    )
    argv = json.loads(launcher_json.read_text(encoding="utf-8"))
    argv[2] = "exiting-launcher"
    launcher_json.write_text(json.dumps(argv) + "\n", encoding="utf-8")

    result = _run_cli(tmp_path, command)
    orphan_pid = int((run_dir / "orphan.pid").read_text(encoding="ascii"))
    try:
        assert result.returncode == 1
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        accounted = {
            row["pid"] for row in payload["termination"]["captured_process_graph"]
        }
        assert orphan_pid in accounted
        assert payload["termination"]["remaining_pids"] == []
        assert payload["termination"]["remaining_pgids"] == []
        assert _proc_absent(orphan_pid)
        assert not (run_dir / "checkpoints/step-5").exists()
        assert not (run_dir / "checkpoints/final.json").exists()
    finally:
        if not _proc_absent(orphan_pid):
            try:
                os.killpg(orphan_pid, 9)
            except ProcessLookupError:
                pass
