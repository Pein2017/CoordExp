from __future__ import annotations

from pathlib import Path
import json
import subprocess
import time

import pytest

from probes.training_set_completion import coco227_trial as trial


def test_qualification_plan_is_exactly_six_two_update_runs_and_132_exposures():
    assert len(trial.QUALIFICATION_CONFIGS) == 6
    assert [
        (item["arm"], item["microbatch_size"], item["activation_checkpointing"])
        for item in trial.QUALIFICATION_CONFIGS
    ] == [
        ("S", 1, True),
        ("S", 2, True),
        ("S", 3, True),
        ("T", 1, True),
        ("T", 2, True),
        ("T", 3, True),
    ]
    assert 6 * 2 * 11 == 132
    assert trial.PARITY_TOLERANCES == {
        "loss_max_abs": 5e-5,
        "gradient_max_abs": 5e-4,
        "gradient_relative_l2": 1e-4,
        "parameter_max_abs": 5e-5,
        "parameter_relative_l2": 1e-5,
    }


def test_distributed_command_uses_current_python_module_entry():
    command = trial.distributed_training_command(
        manifest_path=Path("/tmp/manifest.json"), output=Path("/tmp/output")
    )
    assert command[:7] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "probes.training_set_completion.coco227_training",
    ]
    assert command[-4:] == [
        "--manifest",
        "/tmp/manifest.json",
        "--output",
        "/tmp/output",
    ]


def test_readback_jobs_are_interleaved_by_checkpoint_across_arms():
    jobs = [(arm, step) for step in trial.CHECKPOINT_STEPS for arm in trial.ARMS]
    assert jobs[:4] == [("S", 8), ("T", 8), ("S", 16), ("T", 16)]
    assert len(jobs) == 12


def test_readback_worker_command_carries_trial_and_qualified_batch(tmp_path: Path):
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({"selection": {"batch_size": 3}}))
    command = trial._readback_command(
        trial={"readback": {"worker_module": "example.worker"}},
        trial_path=tmp_path / "trial.json",
        training_manifest=tmp_path / "manifest.json",
        training_terminal=tmp_path / "terminal.json",
        adapter=tmp_path / "adapter",
        arm="S",
        step=8,
        output=tmp_path / "readback",
        gpu=4,
        qualification_result=qualification,
        attempt="attempt-001",
    )
    assert command[:4] == ["python", "-m", "example.worker", "endpoint-worker"]
    assert command[command.index("--batch-size") + 1] == "3"
    assert command[command.index("--qualification-result") + 1] == str(qualification)
    assert command[command.index("--trial") + 1] == str(tmp_path / "trial.json")


def test_completed_endpoint_collection_receives_trial_and_teacher_bindings(
    tmp_path: Path,
):
    endpoint = tmp_path / "readback/S/step-00008/endpoint.json"
    endpoint.parent.mkdir(parents=True)
    endpoint.write_text("{}")
    calls = []

    class Readback:
        @staticmethod
        def collect_endpoint(**kwargs):
            calls.append(kwargs)
            return {"endpoint": {"status": "completed_unscored"}, "admission": {}}

    trial_path = tmp_path / "trial.json"
    teacher = tmp_path / "teacher.json"
    qualification = tmp_path / "qualification.json"
    result = trial._collect_endpoint_if_complete(
        readback_module=Readback,
        trial={
            "arms": {"S": {"training_manifest": {"path": "/m/S.json"}}},
            "teacher": {"path": str(teacher)},
            "readback": {"qualification_result": {"path": str(qualification)}},
        },
        trial_path=trial_path,
        output=tmp_path,
        arm="S",
        step=8,
        adapter=tmp_path / "adapter",
    )
    assert result is not None
    assert calls[0]["trial_path"] == trial_path
    assert calls[0]["teacher_bank_path"] == teacher
    assert calls[0]["qualification_result"] == qualification


def test_controller_timeout_reaps_all_active_workers_and_preserves_failure_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / "trial-output"
    trial_path = tmp_path / "trial.json"
    release_path = tmp_path / "release.json"
    qualification = tmp_path / "qualification.json"
    trial_value = {
        "arms": {
            arm: {"training_manifest": {"path": str(tmp_path / f"{arm}.json")}}
            for arm in trial.ARMS
        },
        "readback": {"qualification_result": {"path": str(qualification)}},
    }
    trial_path.write_text(json.dumps(trial_value))
    release_path.write_text(
        json.dumps(
            {
                "status": "released",
                "trial_sha256": trial.training.file_hash(trial_path),
            }
        )
    )
    for arm in trial.ARMS:
        terminal = output / arm / "training" / "terminal.json"
        terminal.parent.mkdir(parents=True)
        terminal.write_text(
            json.dumps(
                {
                    "checkpoints": [
                        {"step": step, "adapter": {"root": str(tmp_path / f"{arm}-{step}")}}
                        for step in trial.CHECKPOINT_STEPS
                    ]
                }
            )
        )

    monkeypatch.setattr(trial, "validate_trial", lambda value: value)
    monkeypatch.setattr(trial, "validate_training_terminal", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trial.subprocess,
        "check_output",
        lambda *args, **kwargs: trial.TRIAL_TMUX_SESSION,
    )
    monkeypatch.setattr(trial, "_collect_endpoint_if_complete", lambda **kwargs: None)
    monkeypatch.setattr(trial, "_readback_command", lambda **kwargs: ["fake-readback"])
    monkeypatch.setattr(trial, "start_process_waiter", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trial,
        "next_process_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("global readback phase timeout")
        ),
    )

    class Process:
        next_pid = 7000

        def __init__(self):
            self.pid = Process.next_pid
            Process.next_pid += 1
            self.killed = False

        def poll(self):
            return -15 if self.killed else None

    class Stream:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    processes = []
    streams = []

    def spawn(*args, **kwargs):
        process, stream = Process(), Stream()
        processes.append(process)
        streams.append(stream)
        return process, stream, time.monotonic()

    def kill(process):
        process.killed = True

    monkeypatch.setattr(trial, "_spawn", spawn)
    monkeypatch.setattr(trial, "terminate_owned_process", kill)

    with pytest.raises(TimeoutError, match="global readback phase timeout"):
        trial.controller(trial_path=trial_path, output=output, release_path=release_path)

    assert len(processes) == len(streams) == 8
    assert all(process.killed for process in processes)
    assert all(stream.closed for stream in streams)
    failure = json.loads((output / "controller-failures/attempt-001.json").read_text())
    assert failure["status"] == "failed"
    assert failure["error"] == "TimeoutError: global readback phase timeout"
    assert failure["cleanup_errors"] == []
    assert not (output / "controller-terminal.json").exists()


def test_training_qualification_timeout_reaps_owned_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / "qualification"
    output.mkdir()
    plan_path = output / "plan.json"
    plan_path.write_text("{}")
    plan = {
        "configurations": [
            {
                "id": "S-mb1-checkpointed",
                "command": ["fake-training"],
                "output": str(tmp_path / "worker-output"),
                "manifest": {"path": str(tmp_path / "manifest.json")},
            }
        ]
    }
    monkeypatch.setattr(trial, "validate_qualification_plan", lambda value: plan)
    monkeypatch.setattr(
        trial.subprocess,
        "check_output",
        lambda *args, **kwargs: trial.TRAINING_QUALIFICATION_TMUX_SESSION,
    )

    class Process:
        pid = 8123
        killed = False

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(["fake-training"], timeout)

    process = Process()
    monkeypatch.setattr(trial.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        trial,
        "terminate_owned_process",
        lambda owned: setattr(owned, "killed", True),
    )

    with pytest.raises(RuntimeError, match="qualification failed"):
        trial.qualification_controller(plan_path=plan_path, output=output)

    assert process.killed is True
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "failed"
    assert "qualification failed" in terminal["error"]
