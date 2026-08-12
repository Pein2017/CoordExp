from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.research.launch_human13_k_union_matrix as launcher


def _plans(tmp_path: Path, count: int = 3) -> dict[str, object]:
    return {
        "schema_version": "human13_materialized_plans.v1",
        "mode": "dry_run",
        "plans": [
            {
                "arm_id": f"A{index}",
                "updates": True,
                "output_root": str(tmp_path / f"arm-{index}"),
                "optimizer_state_root": str(tmp_path / f"arm-{index}" / "state"),
                "resolved_plan_path": str(tmp_path / f"plan-{index}.json"),
            }
            for index in range(count)
        ],
    }


def test_default_launcher_is_dry_run_world_size_one_and_one_gpu_per_job(
    tmp_path: Path,
) -> None:
    receipt = launcher.plan_launches(_plans(tmp_path), gpu_ids=(3, 5, 7))

    assert receipt["mode"] == "dry_run"
    assert receipt["actions"] == launcher.ZERO_LAUNCH_ACTIONS
    assert [job["gpu_id"] for job in receipt["jobs"]] == [3, 5, 7]
    assert all(job["world_size"] == 1 for job in receipt["jobs"])
    assert all(
        job["command"][:4] == ["conda", "run", "-n", "ms"] for job in receipt["jobs"]
    )
    assert all("--num_processes" in job["command"] for job in receipt["jobs"])
    assert all(
        job["command"][job["command"].index("--num_processes") + 1] == "1"
        for job in receipt["jobs"]
    )


def test_launcher_rejects_more_than_eight_jobs_duplicate_gpu_or_shared_state(
    tmp_path: Path,
) -> None:
    with pytest.raises(launcher.LaunchContractError, match="at most eight"):
        launcher.plan_launches(_plans(tmp_path, count=9), gpu_ids=tuple(range(9)))
    with pytest.raises(launcher.LaunchContractError, match="GPU IDs.*unique"):
        launcher.plan_launches(_plans(tmp_path), gpu_ids=(0, 0, 1))

    plans = _plans(tmp_path)
    plans["plans"][1]["optimizer_state_root"] = plans["plans"][0][
        "optimizer_state_root"
    ]
    with pytest.raises(launcher.LaunchContractError, match="optimizer state roots"):
        launcher.plan_launches(plans, gpu_ids=(0, 1, 2))


def test_frozen_source_is_not_an_optimizer_job_and_eight_updated_arms_fit(
    tmp_path: Path,
) -> None:
    plans = _plans(tmp_path, count=8)
    plans["plans"].insert(
        0,
        {
            "arm_id": "frozen_source",
            "updates": False,
            "output_root": str(tmp_path / "frozen"),
            "optimizer_state_root": None,
            "resolved_plan_path": str(tmp_path / "frozen-plan.json"),
        },
    )

    receipt = launcher.plan_launches(plans, gpu_ids=tuple(range(8)))

    assert len(receipt["jobs"]) == 8
    assert all(job["arm_id"] != "frozen_source" for job in receipt["jobs"])
    assert receipt["omitted_no_update_arms"] == ["frozen_source"]


def test_execute_requires_both_explicit_flag_and_documented_authority(
    tmp_path: Path,
) -> None:
    jobs = launcher.plan_launches(_plans(tmp_path), gpu_ids=(0, 1, 2))
    with pytest.raises(launcher.LaunchContractError, match="separate user model/GPU"):
        launcher.execute_launches(jobs, execution_authorized=False)


def test_execute_fails_closed_without_a_real_runner_entry_contract(
    tmp_path: Path,
) -> None:
    jobs = launcher.plan_launches(_plans(tmp_path), gpu_ids=(0, 1, 2))

    with pytest.raises(launcher.LaunchContractError, match="runner entry contract"):
        launcher.execute_launches(jobs, execution_authorized=True)


def test_execute_assigns_isolated_cuda_visibility_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _plans(tmp_path, count=2)
    for plan in payload["plans"]:
        Path(plan["resolved_plan_path"]).write_text("{}", encoding="utf-8")
    runner_entry = tmp_path / "runner.py"
    runner_entry.write_text("raise SystemExit(0)\n", encoding="utf-8")
    jobs = launcher.plan_launches(
        payload,
        gpu_ids=(2, 4),
        runner_entry=runner_entry,
        runner_entry_contract="human13_runner_cli.v1",
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    class Process:
        def __init__(self, returncode: int = 0) -> None:
            self.returncode = returncode

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command: list[str], *, env: dict[str, str]) -> Process:
        calls.append((command, env))
        return Process()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    receipt = launcher.execute_launches(jobs, execution_authorized=True)

    assert receipt["process_start_count"] == 2
    assert receipt["retry_count"] == 0
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in calls] == ["2", "4"]
    assert [env["WORLD_SIZE"] for _, env in calls] == ["1", "1"]


def test_cli_default_does_not_start_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "plans.json"
    path.write_text(json.dumps(_plans(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("dry-run started a process"),
    )

    assert launcher.main(["--plans", str(path), "--gpus", "0,1,2"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "dry_run"
