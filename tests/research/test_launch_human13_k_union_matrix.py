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


def _inputs(tmp_path: Path, count: int = 3):
    payload = _plans(tmp_path, count=count)
    plans_path = tmp_path / "plans.json"
    plans_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    return payload, plans_path, manifest_path


def test_default_launcher_is_dry_run_world_size_one_and_one_gpu_per_job(
    tmp_path: Path,
) -> None:
    _payload, plans_path, manifest_path = _inputs(tmp_path)
    receipt = launcher.plan_launches(
        plans_path,
        gpu_ids=(3, 5, 7),
        manifest_path=manifest_path,
    )

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
    for job in receipt["jobs"]:
        assert job["runner_entry_contract"] == "human13_runner_cli.v1"
        assert job["execution_ready"] is True
        assert job["command"][job["command"].index("--plans-receipt") + 1] == str(
            plans_path.resolve()
        )
        assert job["command"][job["command"].index("--arm-id") + 1] == job["arm_id"]
        assert job["command"][job["command"].index("--manifest") + 1] == str(
            manifest_path.resolve()
        )
        assert "--execute" in job["command"]
        assert "--user-model-gpu-authority" in job["command"]


def test_launcher_rejects_more_than_eight_jobs_duplicate_gpu_or_shared_state(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(launcher.LaunchContractError, match="at most eight"):
        launcher.plan_launches(
            _plans(tmp_path, count=9),
            gpu_ids=tuple(range(9)),
            manifest_path=manifest,
            plans_receipt_path=tmp_path / "plans-nine.json",
        )
    with pytest.raises(launcher.LaunchContractError, match="GPU IDs.*unique"):
        launcher.plan_launches(
            _plans(tmp_path),
            gpu_ids=(0, 0, 1),
            manifest_path=manifest,
            plans_receipt_path=tmp_path / "plans-three.json",
        )

    plans = _plans(tmp_path)
    plans["plans"][1]["optimizer_state_root"] = plans["plans"][0][
        "optimizer_state_root"
    ]
    with pytest.raises(launcher.LaunchContractError, match="optimizer state roots"):
        launcher.plan_launches(
            plans,
            gpu_ids=(0, 1, 2),
            manifest_path=manifest,
            plans_receipt_path=tmp_path / "plans-shared.json",
        )


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

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    plans_path = tmp_path / "plans.json"
    plans_path.write_text(json.dumps(plans), encoding="utf-8")
    receipt = launcher.plan_launches(
        plans,
        gpu_ids=tuple(range(8)),
        manifest_path=manifest,
        plans_receipt_path=plans_path,
    )

    assert len(receipt["jobs"]) == 8
    assert all(job["arm_id"] != "frozen_source" for job in receipt["jobs"])
    assert receipt["omitted_no_update_arms"] == ["frozen_source"]


def test_execute_requires_both_explicit_flag_and_documented_authority(
    tmp_path: Path,
) -> None:
    _payload, plans_path, manifest_path = _inputs(tmp_path)
    jobs = launcher.plan_launches(
        plans_path, gpu_ids=(0, 1, 2), manifest_path=manifest_path
    )
    with pytest.raises(launcher.LaunchContractError, match="separate user model/GPU"):
        launcher.execute_launches(jobs, execution_authorized=False)


def test_explicit_arm_selection_allows_omitting_a4_and_rejects_absent_or_frozen(
    tmp_path: Path,
) -> None:
    payload, plans_path, manifest_path = _inputs(tmp_path, count=5)
    payload["plans"][0]["arm_id"] = "frozen_source"
    payload["plans"][0]["updates"] = False
    payload["plans"][1]["arm_id"] = "A1"
    payload["plans"][2]["arm_id"] = "A3"
    payload["plans"][3]["arm_id"] = "A4"
    payload["plans"][4]["arm_id"] = "A7"
    plans_path.write_text(json.dumps(payload), encoding="utf-8")

    receipt = launcher.plan_launches(
        plans_path,
        gpu_ids=(6, 7),
        manifest_path=manifest_path,
        selected_arm_ids=("A1", "A7"),
    )
    assert [job["arm_id"] for job in receipt["jobs"]] == ["A1", "A7"]
    assert receipt["unselected_updated_arms"] == ["A3", "A4"]

    with pytest.raises(launcher.LaunchContractError, match="absent"):
        launcher.plan_launches(
            plans_path,
            gpu_ids=(0,),
            manifest_path=manifest_path,
            selected_arm_ids=("A8-prime",),
        )
    with pytest.raises(launcher.LaunchContractError, match="does not update"):
        launcher.plan_launches(
            plans_path,
            gpu_ids=(0,),
            manifest_path=manifest_path,
            selected_arm_ids=("frozen_source",),
        )


def test_execute_starts_each_job_once_without_retry_and_binds_gpu(
    tmp_path: Path,
) -> None:
    _payload, plans_path, manifest_path = _inputs(tmp_path, count=2)
    launch_plan = launcher.plan_launches(
        plans_path,
        gpu_ids=(2, 4),
        manifest_path=manifest_path,
    )
    calls = []

    class Process:
        def __init__(self, command, *, env, cwd):
            calls.append((tuple(command), dict(env), cwd))
            self.returncode = None

        def wait(self):
            self.returncode = 0
            return 0

    receipt = launcher.execute_launches(
        launch_plan,
        execution_authorized=True,
        process_factory=Process,
    )

    assert receipt["mode"] == "execute"
    assert [item["returncode"] for item in receipt["jobs"]] == [0, 0]
    assert [item[1]["CUDA_VISIBLE_DEVICES"] for item in calls] == ["2", "4"]
    assert len(calls) == 2


def test_cli_default_does_not_start_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _payload, path, manifest_path = _inputs(tmp_path)
    monkeypatch.setattr(
        launcher,
        "execute_launches",
        lambda *_args, **_kwargs: pytest.fail("dry-run entered execute mode"),
    )

    assert (
        launcher.main(
            [
                "--plans",
                str(path),
                "--manifest",
                str(manifest_path),
                "--gpus",
                "0,1,2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["mode"] == "dry_run"
