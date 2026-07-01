from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts import (
    MetricStreamEvent,
    RunArtifactManager,
    metric_stream_events_from_runtime_payload,
)
from src.common.errors import ArtifactContractError
from src.config.loader import load_train_config
from src.config.models import RunDirectory
from src.training.schedule import resolve_planned_step_schedule


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_artifact_manager_writes_manifest_config_schedule_and_receipts(
    tmp_path: Path,
) -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    schedule = resolve_planned_step_schedule(
        resolved.config,
        packs_per_epoch=2,
        world_size=1,
        source_config_path=str(FIXTURE_CONFIG),
    )
    manager = RunArtifactManager.initialize(
        run_directory=RunDirectory(
            run_name=resolved.config.run.name,
            artifact_root=tmp_path,
            run_dir=tmp_path / "run-a",
            collision_policy="fail",
        ),
        run_id="run-a",
        created_at="2026-06-30T00:00:00Z",
        runtime_identity={"python": "test"},
        backend_status={"single": ["active"], "accelerate": [], "deepspeed": []},
    )

    config_artifacts = manager.write_resolved_config(resolved)
    schedule_path = manager.write_schedule(schedule)
    receipt_path = manager.write_receipt(
        "runtime_setup",
        {"backend": "single", "world_size": 1},
        category="runtime",
    )
    manifest = manager.read_manifest()

    assert manager.manifest_path == tmp_path / "run-a" / "run_manifest.json"
    assert set(manifest) >= {
        "run_id",
        "run_name",
        "run_dir",
        "status",
        "created_at",
        "updated_at",
        "configs",
        "resolution",
        "runtime_identity",
        "schedule",
        "receipts",
        "metrics",
        "checkpoints",
        "eval",
        "runtime",
        "backend_status",
        "warnings",
    }
    assert manifest["run_id"] == "run-a"
    assert manifest["run_dir"] == str((tmp_path / "run-a").resolve())
    assert manifest["configs"]["resolved_yaml"] == "configs/resolved.yaml"
    assert manifest["configs"]["resolved_json"] == "configs/resolved.json"
    assert manifest["resolution"]["fingerprint"] == resolved.fingerprint
    assert manifest["schedule"]["resolved_step_schedule"] == "resolved_step_schedule.json"
    assert manifest["receipts"]["runtime"]["runtime_setup"] == "receipts/runtime/runtime_setup.json"
    assert manifest["backend_status"]["single"] == ["active"]
    assert config_artifacts.yaml_path.exists()
    assert schedule_path == tmp_path / "run-a" / "resolved_step_schedule.json"
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == {
        "backend": "single",
        "world_size": 1,
    }


def test_artifact_manager_writes_reports_and_finalizes_manifest(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    report_path = manager.write_report(
        "token_type_vocab",
        {"vocab_size": 152670, "coordinate_count": 1000},
    )
    manager.finalize(status="completed", completed_at="2026-06-30T00:00:05Z")

    manifest = manager.read_manifest()
    assert report_path == tmp_path / "run-a" / "reports" / "token_type_vocab.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "coordinate_count": 1000,
        "vocab_size": 152670,
    }
    assert manifest["reports"]["token_type_vocab"] == "reports/token_type_vocab.json"
    assert manifest["status"] == "completed"
    assert manifest["completed_at"] == "2026-06-30T00:00:05Z"
    assert manifest["updated_at"] == "2026-06-30T00:00:05Z"


def test_metric_stream_event_jsonl_shape_and_selector_eligibility(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    event = MetricStreamEvent(
        event_type="metric",
        planned_step_id=2,
        split="train",
        name="acc_top1",
        value=0.75,
        trigger_reasons=("training.logging",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
        reduction="single_rank",
        rank=0,
        world_size=1,
    )

    metric_path = manager.append_metric_event(event)
    record = json.loads(metric_path.read_text(encoding="utf-8").splitlines()[0])
    manifest = manager.read_manifest()

    assert metric_path == tmp_path / "run-a" / "metrics" / "train.jsonl"
    assert record == {
        "event_type": "metric",
        "planned_step_id": 2,
        "split": "train",
        "name": "acc_top1",
        "value": 0.75,
        "trigger_reasons": ["training.logging"],
        "optimizer_update_status": "applied",
        "finite_status": "finite",
        "warning_status": "none",
        "reduction": "single_rank",
        "rank": 0,
        "world_size": 1,
        "selector_eligible": True,
        "metadata": {},
    }
    assert manifest["metrics"]["streams"]["train"] == "metrics/train.jsonl"


def test_warning_only_eval_metric_remains_selector_eligible(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    event = MetricStreamEvent(
        event_type="metric",
        planned_step_id=4,
        split="eval.forward",
        name="acc_top1",
        value=0.95,
        trigger_reasons=("milestone_40pct",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="warned",
    )

    metric_path = manager.append_metric_event(event)
    record = json.loads(metric_path.read_text(encoding="utf-8").splitlines()[0])

    assert event.selector_eligible is True
    assert record["selector_eligible"] is True
    assert record["warning_status"] == "warned"


def test_rank_local_runtime_metrics_remain_not_selector_eligible(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    runtime_payload = {
        "planned_step_id": 3,
        "split": "train",
        "rank": 1,
        "world_size": 2,
        "reduction": "rank_local",
        "metrics": {"acc_top1": 0.5, "loss/total": 2.0},
    }
    events = metric_stream_events_from_runtime_payload(
        runtime_payload,
        event_type="metric",
        trigger_reasons=("training.logging",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )

    assert [event.name for event in events] == ["acc_top1", "loss/total"]
    assert all(event.reduction == "rank_local" for event in events)
    assert all(event.selector_eligible is False for event in events)

    for event in events:
        manager.append_metric_event(event)
    records = [
        json.loads(line)
        for line in (tmp_path / "run-a" / "metrics" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert {record["name"] for record in records} == {"acc_top1", "loss/total"}
    assert all(record["reduction"] == "rank_local" for record in records)
    assert all(record["selector_eligible"] is False for record in records)
    assert all(record["rank"] == 1 and record["world_size"] == 2 for record in records)


def test_runtime_metric_payload_preserves_explicit_unavailable_values(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    events = metric_stream_events_from_runtime_payload(
        {
            "planned_step_id": 4,
            "split": "eval.forward",
            "rank": 0,
            "world_size": 1,
            "reduction": "single_rank",
            "metrics": {"acc_top5": None},
        },
        event_type="metric",
        trigger_reasons=("explicit_step",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )

    assert events[0].value is None
    metric_path = manager.append_metric_event(events[0])
    record = json.loads(metric_path.read_text(encoding="utf-8"))

    assert record["split"] == "eval.forward"
    assert record["name"] == "acc_top5"
    assert record["value"] is None
    assert record["selector_eligible"] is False


def test_artifact_manager_rejects_nan_metric_values(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(ArtifactContractError) as exc_info:
        manager.append_metric_event(
            MetricStreamEvent(
                event_type="metric",
                planned_step_id=1,
                split="train",
                name="loss/total",
                value=float("nan"),
                trigger_reasons=(),
                optimizer_update_status="applied",
                finite_status="finite",
                warning_status="none",
            )
        )

    assert exc_info.value.code == "metric.value_non_finite"


def test_artifact_manager_refuses_non_empty_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "leftover.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        RunArtifactManager.initialize(
            run_directory=RunDirectory(
                run_name="run-a",
                artifact_root=tmp_path,
                run_dir=run_dir,
                collision_policy="fail",
            ),
            run_id="run-a",
            created_at="2026-06-30T00:00:00Z",
            runtime_identity={},
            backend_status={"single": ["active"], "accelerate": [], "deepspeed": []},
        )

    assert exc_info.value.code == "artifact.run_dir_not_empty"


def test_artifact_manager_refuses_receipt_overwrite(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.write_receipt("runtime_setup", {"backend": "single"}, category="runtime")

    with pytest.raises(ArtifactContractError) as exc_info:
        manager.write_receipt("runtime_setup", {"backend": "changed"}, category="runtime")

    assert exc_info.value.code == "artifact.receipt_exists"


def test_artifact_manager_repairs_manifest_after_config_registration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    manager = _manager(tmp_path)
    _fail_next_manifest_write(monkeypatch)

    with pytest.raises(ArtifactContractError) as exc_info:
        manager.write_resolved_config(resolved)

    assert exc_info.value.code == "test.manifest_write_failed"
    assert (tmp_path / "run-a" / "configs" / "resolved.json").exists()

    manager.write_resolved_config(resolved)
    manifest = manager.read_manifest()

    assert manifest["configs"]["resolved_json"] == "configs/resolved.json"
    assert manifest["resolution"]["fingerprint"] == resolved.fingerprint


def test_artifact_manager_rejects_stale_resolved_yaml_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    manager = _manager(tmp_path)
    _fail_next_manifest_write(monkeypatch)

    with pytest.raises(ArtifactContractError) as exc_info:
        manager.write_resolved_config(resolved)

    assert exc_info.value.code == "test.manifest_write_failed"
    (tmp_path / "run-a" / "configs" / "resolved.yaml").write_text(
        "stale: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactContractError) as retry_exc:
        manager.write_resolved_config(resolved)

    assert retry_exc.value.code == "artifact.resolved_config_exists"


def test_artifact_manager_repairs_manifest_after_schedule_registration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    schedule = resolve_planned_step_schedule(
        resolved.config,
        packs_per_epoch=2,
        world_size=1,
    )
    manager = _manager(tmp_path)
    _fail_next_manifest_write(monkeypatch)

    with pytest.raises(ArtifactContractError) as exc_info:
        manager.write_schedule(schedule)

    assert exc_info.value.code == "test.manifest_write_failed"
    assert (tmp_path / "run-a" / "resolved_step_schedule.json").exists()

    manager.write_schedule(schedule)
    manifest = manager.read_manifest()

    assert manifest["schedule"]["resolved_step_schedule"] == "resolved_step_schedule.json"


def test_artifact_manager_repairs_manifest_after_receipt_registration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    payload = {"backend": "single"}
    _fail_next_manifest_write(monkeypatch)

    with pytest.raises(ArtifactContractError) as exc_info:
        manager.write_receipt("runtime_setup", payload, category="runtime")

    assert exc_info.value.code == "test.manifest_write_failed"
    assert json.loads(
        (tmp_path / "run-a" / "receipts" / "runtime" / "runtime_setup.json").read_text(
            encoding="utf-8"
        )
    ) == payload

    manager.write_receipt("runtime_setup", payload, category="runtime")
    manifest = manager.read_manifest()

    assert manifest["receipts"]["runtime"]["runtime_setup"] == (
        "receipts/runtime/runtime_setup.json"
    )


def test_artifact_manager_does_not_duplicate_metric_after_manifest_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    event = MetricStreamEvent(
        event_type="metric",
        planned_step_id=1,
        split="train",
        name="acc_top1",
        value=0.1,
        trigger_reasons=(),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )
    _fail_next_manifest_write(monkeypatch)

    with pytest.raises(ArtifactContractError) as exc_info:
        manager.append_metric_event(event)

    assert exc_info.value.code == "test.manifest_write_failed"

    manager.append_metric_event(event)
    records = (tmp_path / "run-a" / "metrics" / "train.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(records) == 1
    assert manager.read_manifest()["metrics"]["streams"]["train"] == "metrics/train.jsonl"


def _manager(tmp_path: Path) -> RunArtifactManager:
    return RunArtifactManager.initialize(
        run_directory=RunDirectory(
            run_name="run-a",
            artifact_root=tmp_path,
            run_dir=tmp_path / "run-a",
            collision_policy="fail",
        ),
        run_id="run-a",
        created_at="2026-06-30T00:00:00Z",
        runtime_identity={},
        backend_status={"single": ["active"], "accelerate": [], "deepspeed": []},
    )


def _fail_next_manifest_write(monkeypatch: pytest.MonkeyPatch) -> None:
    original = RunArtifactManager._write_manifest
    remaining_failures = 1

    def fail_once(self: RunArtifactManager, payload: dict[str, object]) -> None:
        nonlocal remaining_failures
        if remaining_failures:
            remaining_failures -= 1
            raise ArtifactContractError(
                "synthetic manifest write failure",
                code="test.manifest_write_failed",
            )
        original(self, payload)

    monkeypatch.setattr(RunArtifactManager, "_write_manifest", fail_once)
