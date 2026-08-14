from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import scripts.research.launch_human13_k_trajectory_rp_crossover as launcher
import scripts.research.train_human13_k_trajectory_rp_crossover as runner
from scripts.research.human13_rp_crossover_matrix_contracts import (
    DRY_RUN_COUNTER_KEYS,
    PHASE_MATRIX,
    PHASE_QUALIFICATION,
)


# ---------------------------------------------------------------------------
# Leaf config loading
# ---------------------------------------------------------------------------


def test_load_leaf_configs_returns_six_configs_covering_two_rp_by_three_arms() -> None:
    configs = launcher.load_leaf_configs()

    assert len(configs) == 6
    pairs = {(config.training_rp, config.arm_id) for config in configs}
    assert pairs == {
        (1.0, "A"),
        (1.0, "B"),
        (1.0, "C"),
        (1.10, "A"),
        (1.10, "B"),
        (1.10, "C"),
    }
    for config in configs:
        assert config.evaluation_rps == (1.0, 1.10)
        assert config.max_updates == 1
        assert config.retry_policy == "none"
        assert config.optimizer["learning_rate"] == pytest.approx(3.0e-6)


def test_load_leaf_configs_objective_components_and_compiler_coefficient_are_nested() -> (
    None
):
    by_arm = {config.arm_id: config for config in launcher.load_leaf_configs()}

    assert by_arm["A"].objective_components == ("trajectory",)
    assert by_arm["A"].compiler_coefficient is None
    assert by_arm["B"].objective_components == ("trajectory", "compiler")
    assert by_arm["B"].compiler_coefficient == pytest.approx(1.0)
    assert by_arm["C"].objective_components == (
        "trajectory",
        "compiler",
        "preservation",
    )
    assert by_arm["C"].compiler_coefficient == pytest.approx(1.0)


def test_load_leaf_configs_rejects_wrong_count(tmp_path: Path) -> None:
    root = tmp_path / "only_one"
    root.mkdir()
    real_configs = sorted(launcher.CONFIG_ROOT.glob("*.yaml"))
    (root / real_configs[0].name).write_text(
        real_configs[0].read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(launcher.LeafConfigError, match="exactly six"):
        launcher.load_leaf_configs(root)


def test_load_leaf_config_rejects_drifted_optimizer(tmp_path: Path) -> None:
    root = tmp_path / "drifted"
    root.mkdir()
    for source in sorted(launcher.CONFIG_ROOT.glob("*.yaml")):
        text = source.read_text(encoding="utf-8")
        if source.name.startswith("01_"):
            text = text.replace("learning_rate: 3.0e-6", "learning_rate: 1.0e-5")
        (root / source.name).write_text(text, encoding="utf-8")

    with pytest.raises(launcher.LeafConfigError, match="optimizer"):
        launcher.load_leaf_configs(root)


def test_load_leaf_config_rejects_unknown_field(tmp_path: Path) -> None:
    root = tmp_path / "extra_field"
    root.mkdir()
    for source in sorted(launcher.CONFIG_ROOT.glob("*.yaml")):
        text = source.read_text(encoding="utf-8")
        if source.name.startswith("01_"):
            text += "unexpected_field: true\n"
        (root / source.name).write_text(text, encoding="utf-8")

    with pytest.raises(launcher.LeafConfigError, match="unknown"):
        launcher.load_leaf_configs(root)


# ---------------------------------------------------------------------------
# Static DAG plan
# ---------------------------------------------------------------------------


def _dag_plan(tmp_path: Path, run_id: str = "smoke"):
    configs = launcher.load_leaf_configs()
    return launcher.build_dag_plan(
        configs, run_id=run_id, output_root=tmp_path / "artifacts"
    )


def test_build_dag_plan_has_six_matrix_acquisitions_and_two_qualification_acquisitions(
    tmp_path: Path,
) -> None:
    plan = _dag_plan(tmp_path)

    assert plan["schema_version"] == launcher.DAG_PLAN_SCHEMA
    assert plan["mode"] == "dry_run"
    assert len(plan["acquisitions"]) == 8
    matrix = [a for a in plan["acquisitions"] if a["phase"] == PHASE_MATRIX]
    qualification = [
        a for a in plan["acquisitions"] if a["phase"] == PHASE_QUALIFICATION
    ]
    assert len(matrix) == 6
    assert len(qualification) == 2
    assert sum(len(a["cells"]) for a in matrix) == 18
    assert sum(len(a["cells"]) for a in qualification) == 2
    for acquisition in qualification:
        assert [cell["cell_key"]["arm_id"] for cell in acquisition["cells"]] == ["C"]


def test_build_dag_plan_qualification_cells_excluded_from_matrix_disposition(
    tmp_path: Path,
) -> None:
    plan = _dag_plan(tmp_path)

    assert len(plan["matrix_cell_keys"]) == 18
    assert len(set(plan["matrix_cell_keys"])) == 18
    assert len(plan["qualification_cell_keys"]) == 2
    assert len(set(plan["qualification_cell_keys"])) == 2
    assert not set(plan["matrix_cell_keys"]) & set(plan["qualification_cell_keys"])


def test_build_dag_plan_source_baselines_are_prerequisites_not_cells(
    tmp_path: Path,
) -> None:
    plan = _dag_plan(tmp_path)

    assert {item["evaluation_rp"] for item in plan["source_baselines"]} == {1.0, 1.10}
    for baseline in plan["source_baselines"]:
        assert "gpu_id" not in baseline
        assert "output_root" not in baseline
    all_cell_keys = set(plan["matrix_cell_keys"]) | set(plan["qualification_cell_keys"])
    baseline_edges = [
        edge
        for edge in plan["dependency_edges"]
        if edge[0].startswith("source_baseline:")
    ]
    assert len(baseline_edges) == 2 * len(all_cell_keys)
    assert {edge[1] for edge in baseline_edges} == all_cell_keys


def test_build_dag_plan_same_group_arms_share_evidence_and_have_unique_roots(
    tmp_path: Path,
) -> None:
    plan = _dag_plan(tmp_path)

    output_roots = []
    proposal_ids = []
    for acquisition in plan["acquisitions"]:
        evidence_tags = {
            cell["shared_evidence_group_sha256"] for cell in acquisition["cells"]
        }
        assert evidence_tags == {acquisition["shared_evidence_group_sha256"]}
        for cell in acquisition["cells"]:
            output_roots.append(cell["output_root"])
            proposal_ids.append(cell["proposal_id"])

    all_evidence_tags = {
        a["shared_evidence_group_sha256"] for a in plan["acquisitions"]
    }
    assert len(all_evidence_tags) == 8
    assert len(output_roots) == len(set(output_roots)) == 20
    assert len(proposal_ids) == len(set(proposal_ids)) == 20


def test_build_dag_plan_cells_are_fresh_world_one_max_updates_one_no_retry(
    tmp_path: Path,
) -> None:
    plan = _dag_plan(tmp_path)

    for acquisition in plan["acquisitions"]:
        for cell in acquisition["cells"]:
            assert cell["source"] == "fresh"
            assert cell["optimizer"] == "fresh_adamw"
            assert cell["world_size"] == 1
            assert cell["max_updates"] == 1
            assert cell["retry_policy"] == "none"
            assert cell["evaluation_rps"] == [1.0, 1.10]


def test_build_dag_plan_actions_are_zero_and_writes_nothing(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"
    plan = _dag_plan(tmp_path)

    assert plan["actions"] == dict.fromkeys(DRY_RUN_COUNTER_KEYS, 0)
    assert not output_root.exists()


def test_build_dag_plan_rejects_wrong_leaf_config_coverage(tmp_path: Path) -> None:
    configs = launcher.load_leaf_configs()
    with pytest.raises(launcher.LaunchContractError, match="exactly six"):
        launcher.build_dag_plan(
            configs[:5], run_id="smoke", output_root=tmp_path / "artifacts"
        )


# ---------------------------------------------------------------------------
# GPU launch planning
# ---------------------------------------------------------------------------


def _dag_plan_path(tmp_path: Path, run_id: str = "smoke") -> Path:
    plan = _dag_plan(tmp_path, run_id=run_id)
    path = tmp_path / f"dag-plan-{run_id}.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _fake_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "fake_runner.py"
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    return runner


def test_plan_launches_rejects_in_memory_dag_plan_without_bound_path(
    tmp_path: Path,
) -> None:
    plan = _dag_plan(tmp_path)
    runner = _fake_runner(tmp_path)

    with pytest.raises(launcher.LaunchContractError, match="explicit dag_plan_path"):
        launcher.plan_launches(plan, gpu_ids=tuple(range(8)), runner_entry=runner)


def test_plan_launches_default_dry_run_assigns_one_gpu_per_live_node(
    tmp_path: Path,
) -> None:
    plan_path = _dag_plan_path(tmp_path)
    runner = _fake_runner(tmp_path)

    launch_plan = launcher.plan_launches(
        plan_path, gpu_ids=tuple(range(8)), runner_entry=runner
    )

    assert launch_plan["mode"] == "dry_run"
    assert launch_plan["actions"] == dict.fromkeys(DRY_RUN_COUNTER_KEYS, 0)
    assert len(launch_plan["jobs"]) == 8
    assert [job["gpu_id"] for job in launch_plan["jobs"]] == list(range(8))
    for job in launch_plan["jobs"]:
        assert job["world_size"] == 1
        assert job["retry_policy"] == "none"
        assert job["execution_ready"] is False
        assert job["command"][:4] == ["conda", "run", "-n", "ms"]
        assert str(runner.resolve()) in job["command"]
        assert "--execute" in job["command"]
        assert "--user-model-gpu-authority" in job["command"]
    receipt_paths = [job["receipt_path"] for job in launch_plan["jobs"]]
    assert len(receipt_paths) == len(set(receipt_paths)) == 8


def test_plan_launches_rejects_gpu_count_mismatch_or_duplicate(tmp_path: Path) -> None:
    plan = _dag_plan(tmp_path)
    runner = _fake_runner(tmp_path)

    with pytest.raises(launcher.LaunchContractError, match="one unique GPU"):
        launcher.plan_launches(plan, gpu_ids=tuple(range(7)), runner_entry=runner)
    with pytest.raises(launcher.LaunchContractError, match="unique"):
        launcher.plan_launches(
            plan, gpu_ids=(0, 0, 1, 2, 3, 4, 5, 6), runner_entry=runner
        )


def test_plan_launches_supports_node_subset_selection(tmp_path: Path) -> None:
    plan_path = _dag_plan_path(tmp_path)
    runner = _fake_runner(tmp_path)
    node_ids = ("rp100:matrix_a", "rp110:qualification")

    launch_plan = launcher.plan_launches(
        plan_path, gpu_ids=(3, 5), node_ids=node_ids, runner_entry=runner
    )

    assert [job["node_id"] for job in launch_plan["jobs"]] == list(node_ids)
    assert [job["gpu_id"] for job in launch_plan["jobs"]] == [3, 5]


def test_plan_launches_can_forward_an_explicit_runtime_factory(tmp_path: Path) -> None:
    plan_path = _dag_plan_path(tmp_path)

    launch_plan = launcher.plan_launches(
        plan_path,
        gpu_ids=(3,),
        node_ids=("rp100:matrix_a",),
        runtime_factory="project.runtime:create_node_runtime",
    )

    command = launch_plan["jobs"][0]["command"]
    assert command[command.index("--runtime-factory") + 1] == (
        "project.runtime:create_node_runtime"
    )
    assert launch_plan["jobs"][0]["execution_ready"] is True


def test_plan_launches_rejects_unknown_node_id(tmp_path: Path) -> None:
    plan = _dag_plan(tmp_path)
    runner = _fake_runner(tmp_path)

    with pytest.raises(launcher.LaunchContractError, match="node"):
        launcher.plan_launches(
            plan, gpu_ids=(0,), node_ids=("rp100:matrix_z",), runner_entry=runner
        )


def test_plan_launches_fails_closed_on_missing_runner_entry(tmp_path: Path) -> None:
    plan = _dag_plan(tmp_path)
    missing = tmp_path / "does_not_exist.py"

    with pytest.raises((launcher.LaunchContractError, FileNotFoundError, OSError)):
        launcher.plan_launches(plan, gpu_ids=tuple(range(8)), runner_entry=missing)


@pytest.mark.parametrize(
    ("node_id", "expected_arms"),
    [
        ("rp100:matrix_a", ["A", "B", "C"]),
        ("rp110:qualification", ["C"]),
    ],
)
def test_default_launcher_job_is_accepted_by_real_node_runner_dry_run(
    tmp_path: Path, node_id: str, expected_arms: list[str]
) -> None:
    plan_path = _dag_plan_path(tmp_path)
    launch_plan = launcher.plan_launches(plan_path, gpu_ids=(0,), node_ids=(node_id,))
    command = launch_plan["jobs"][0]["command"]
    runner_index = command.index(str(launcher.DEFAULT_RUNNER_ENTRY.resolve()))
    dry_run_argv = [
        item
        for item in command[runner_index + 1 :]
        if item not in {"--execute", "--user-model-gpu-authority"}
    ]
    output = io.StringIO()

    exit_code = runner.run_cli(
        dry_run_argv,
        node_runtime_factory=lambda _: pytest.fail("dry-run opened node runtime"),
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["node_id"] == node_id
    assert [cell["arm_id"] for cell in payload["cells"]] == expected_arms
    assert payload["actions"] == dict.fromkeys(DRY_RUN_COUNTER_KEYS, 0)
    assert not Path(launch_plan["jobs"][0]["receipt_path"]).exists()


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def test_execute_launches_requires_explicit_authority(tmp_path: Path) -> None:
    plan_path = _dag_plan_path(tmp_path)
    runner = _fake_runner(tmp_path)
    launch_plan = launcher.plan_launches(
        plan_path,
        gpu_ids=(0, 1),
        node_ids=("rp100:matrix_a", "rp110:matrix_a"),
        runner_entry=runner,
    )

    with pytest.raises(launcher.LaunchContractError, match="separate user model/GPU"):
        launcher.execute_launches(launch_plan, execution_authorized=False)


def test_execute_launches_starts_each_job_once_without_retry(tmp_path: Path) -> None:
    plan_path = _dag_plan_path(tmp_path)
    runner = _fake_runner(tmp_path)
    launch_plan = launcher.plan_launches(
        plan_path,
        gpu_ids=(2, 4),
        node_ids=("rp100:matrix_a", "rp110:matrix_a"),
        runner_entry=runner,
        runtime_factory="project.runtime:create_node_runtime",
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
        launch_plan, execution_authorized=True, process_factory=Process
    )

    assert receipt["mode"] == "execute"
    assert [item["returncode"] for item in receipt["jobs"]] == [0, 0]
    assert [item[1]["CUDA_VISIBLE_DEVICES"] for item in calls] == ["2", "4"]
    assert len(calls) == 2
    assert receipt["actions"]["gpu_allocations"] == 2
    assert receipt["actions"]["subprocess_launches"] == 2
    assert receipt["actions"]["model_loads"] == 0
    assert receipt["actions"]["output_roots_created"] == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_materialize_then_launch_dry_run_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _fake_runner(tmp_path)
    dag_plan_path = tmp_path / "dag-plan.json"
    output_root = tmp_path / "artifacts"

    assert (
        launcher.main(
            [
                "materialize",
                "--run-id",
                "smoke",
                "--output-root",
                str(output_root),
                "--write-dag-plan",
                str(dag_plan_path),
            ]
        )
        == 0
    )
    materialize_out = json.loads(capsys.readouterr().out)
    assert materialize_out["mode"] == "dry_run"
    assert dag_plan_path.is_file()
    assert not output_root.exists()

    monkeypatch.setattr(
        launcher,
        "execute_launches",
        lambda *_a, **_k: pytest.fail("dry-run entered execute mode"),
    )
    assert (
        launcher.main(
            [
                "launch",
                "--dag-plan",
                str(dag_plan_path),
                "--gpus",
                ",".join(str(i) for i in range(8)),
                "--runner-entry",
                str(runner),
            ]
        )
        == 0
    )
    launch_out = json.loads(capsys.readouterr().out)
    assert launch_out["mode"] == "dry_run"
    assert len(launch_out["jobs"]) == 8
