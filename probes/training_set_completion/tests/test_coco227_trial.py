from __future__ import annotations

from pathlib import Path
import json

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
