from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.research.execute_spatial_scope_history as execution_cli


def _inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / name
        for name in (
            "schedule.json",
            "infer.yaml",
            "cohort.jsonl",
            "source.jsonl",
            "selection.json",
            "runtime.json",
            "attestation.json",
        )
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    paths["attempts.jsonl"] = tmp_path / "attempts.jsonl"
    paths["artifacts"] = tmp_path / "artifacts"
    return paths


def _patch_inputs(monkeypatch: pytest.MonkeyPatch, *, waves: tuple[object, ...]):
    identity = SimpleNamespace(
        run_id="run-id",
        execution_identity=SimpleNamespace(),
    )
    schedule = SimpleNamespace(identity=identity, fingerprint="a" * 64)
    monkeypatch.setattr(
        execution_cli.PrimaryScheduleArtifact,
        "from_artifact_dict",
        lambda value: SimpleNamespace(schedule=schedule),
    )
    monkeypatch.setattr(
        execution_cli.AttemptLedger,
        "from_jsonl_bytes",
        lambda payload, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        execution_cli,
        "build_coordinator_plan",
        lambda schedule, **kwargs: SimpleNamespace(
            workers=("worker-zero",), waves=waves
        ),
    )


class _FakePool:
    instances: list[_FakePool] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.executed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute_plan(self, plan, *, artifact_root):
        self.executed = True
        self.plan = plan
        self.artifact_root = artifact_root
        return SimpleNamespace(batch_receipts=("batch",), worker_startups=("worker",))


def test_execution_cli_reuses_pool_contract_and_executes_resume_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, waves=("wave",))
    _FakePool.instances.clear()

    receipt = execution_cli.execute_research_schedule(
        schedule_path=inputs["schedule.json"],
        infer_config_path=inputs["infer.yaml"],
        cohort_ledger_path=inputs["cohort.jsonl"],
        source_jsonl_path=inputs["source.jsonl"],
        calibration_selection_receipt_path=inputs["selection.json"],
        source_runtime_identity_receipt_path=inputs["runtime.json"],
        sampled_runtime_attestation_path=inputs["attestation.json"],
        attempt_ledger_path=inputs["attempts.jsonl"],
        artifact_root=inputs["artifacts"],
        physical_gpu_tokens=("0", "1"),
        pool_factory=_FakePool,
        executor_factory=lambda: "executor",
    )

    assert receipt is not None
    pool = _FakePool.instances[0]
    assert pool.executed is True
    assert pool.kwargs["workers"] == ("worker-zero",)
    assert pool.kwargs["factory_config"]["attempt_ledger_path"].endswith(
        "attempts.jsonl"
    )


def test_execution_cli_empty_resume_frontier_does_not_start_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, waves=())
    _FakePool.instances.clear()

    receipt = execution_cli.execute_research_schedule(
        schedule_path=inputs["schedule.json"],
        infer_config_path=inputs["infer.yaml"],
        cohort_ledger_path=inputs["cohort.jsonl"],
        source_jsonl_path=inputs["source.jsonl"],
        calibration_selection_receipt_path=inputs["selection.json"],
        source_runtime_identity_receipt_path=inputs["runtime.json"],
        sampled_runtime_attestation_path=inputs["attestation.json"],
        attempt_ledger_path=inputs["attempts.jsonl"],
        artifact_root=inputs["artifacts"],
        physical_gpu_tokens=("0",),
        pool_factory=_FakePool,
        executor_factory=lambda: "executor",
    )

    assert receipt is None
    assert _FakePool.instances == []


def test_execution_cli_propagates_worker_failure_without_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, waves=("wave",))

    class FailingPool(_FakePool):
        def execute_plan(self, plan, *, artifact_root):
            raise RuntimeError("worker failed closed")

    with pytest.raises(RuntimeError, match="worker failed closed"):
        execution_cli.execute_research_schedule(
            schedule_path=inputs["schedule.json"],
            infer_config_path=inputs["infer.yaml"],
            cohort_ledger_path=inputs["cohort.jsonl"],
            source_jsonl_path=inputs["source.jsonl"],
            calibration_selection_receipt_path=inputs["selection.json"],
            source_runtime_identity_receipt_path=inputs["runtime.json"],
            sampled_runtime_attestation_path=inputs["attestation.json"],
            attempt_ledger_path=inputs["attempts.jsonl"],
            artifact_root=inputs["artifacts"],
            physical_gpu_tokens=("0",),
            pool_factory=FailingPool,
            executor_factory=lambda: "executor",
        )
