from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts.run_writer import RunWriter
from src.common.errors import ArtifactContractError


def test_initialize_writes_only_fixed_run_files_with_no_selector_state(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    assert writer.file_inventory() == ("logging.jsonl", "resolved_config.json", "run.json")
    state = writer.read_run()
    assert state["resolved_config_path"] == "resolved_config.json"
    assert state["runtime"] == {"world_size": 2}
    assert not set(state).intersection({"final", "best", "selector", "selector_value"})
    assert json.loads(writer.resolved_config_path.read_text()) == {"training": {"seed": 7}}
    assert not any((writer.run_dir / name).exists() for name in ("metrics", "receipts", "reports", "eval"))


def test_logging_appends_one_self_contained_train_and_eval_row(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.append_logging_row(
        {
            "step": 1,
            "split": "train",
            "loss/total": 1.25,
            "acc_top1": 0.5,
            "acc_top5": 0.75,
            "learning_rates": {"adapter": 1e-5},
            "optimizer_update_status": "applied",
            "finite_status": "finite",
        }
    )
    writer.append_logging_row(
        {"step": 1, "split": "eval", "example_count": 3, "pack_count": 2, "acc_top1": 0.6}
    )

    rows = [json.loads(line) for line in writer.logging_path.read_text().splitlines()]
    assert [row["split"] for row in rows] == ["train", "eval"]
    assert rows[0]["learning_rates"] == {"adapter": 1e-5}
    assert rows[1]["example_count"] == 3
    assert all("non_finite_fields" in row for row in rows)


def test_logging_normalizes_nested_nonfinite_values_to_null(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.append_logging_row(
        {"step": 2, "split": "train", "loss": float("nan"), "metrics": {"x": float("inf")}}
    )
    row = json.loads(writer.logging_path.read_text())
    assert row["loss"] is None and row["metrics"]["x"] is None
    assert row["non_finite_fields"] == ["loss", "metrics.x"]
    assert "NaN" not in writer.logging_path.read_text() and "Infinity" not in writer.logging_path.read_text()


def test_logging_rejects_non_json_values_without_appending_partial_row(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.append_logging_row({"step": 1, "split": "train", "bad": {1, 2}})
    assert exc_info.value.code == "run_writer.not_json_serializable"
    assert writer.logging_path.read_bytes() == b""


def test_warning_counts_are_grouped_and_bounded_without_contexts(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.record_warning("bad_example", count=2)
    writer.record_warning("bad_example", count=3)
    for index in range(40):
        writer.record_warning(f"code_{index}")
    counts = writer.read_run()["warning_counts"]
    assert counts["bad_example"] == 5
    assert len(counts) <= 32
    assert counts["other"] == 10
    assert all(isinstance(value, int) for value in counts.values())


def test_materialization_binding_is_one_time_and_survives_cache_deletion(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    writer.bind_materialization(
        "train", cache_format_version=4, semantic_fingerprint="fp", determinant_digest="digest"
    )
    cache.rmdir()
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_materialization(
            "train", cache_format_version=4, semantic_fingerprint="changed", determinant_digest="changed"
        )
    assert exc_info.value.code == "run_writer.materialization_already_bound"
    assert writer.read_run()["materializations"]["train"] == {
        "cache_format_version": 4,
        "semantic_fingerprint": "fp",
        "determinant_digest": "digest",
    }


def test_schedule_can_be_bound_once_after_early_initialization(tmp_path: Path) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run", run_id="run", run_name="experiment",
        artifact_root=tmp_path, collision_outcome="created", created_at="now",
        config_fingerprint="fp", resolved_config={}, world_size=1,
    )
    assert writer.read_run()["resolved_max_steps"] is None
    writer.bind_schedule(resolved_max_steps=5)
    assert writer.read_run()["resolved_max_steps"] == 5
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_schedule(resolved_max_steps=6)
    assert exc_info.value.code == "run_writer.schedule_already_bound"


def test_initialize_failure_leaves_no_partial_run_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_touch = Path.touch

    def fail_logging_touch(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "logging.jsonl":
            raise OSError("injected logging bootstrap failure")
        real_touch(path, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fail_logging_touch)
    run_dir = tmp_path / "run"
    with pytest.raises(OSError, match="injected logging"):
        RunWriter.initialize(
            run_dir=run_dir, run_id="run", run_name="experiment",
            artifact_root=tmp_path, collision_outcome="created", created_at="now",
            config_fingerprint="fp", resolved_config={}, world_size=1,
        )
    assert not run_dir.exists()
    assert not list(tmp_path.glob(".run.*.init"))


def test_initialize_can_atomically_replace_precreated_empty_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    writer = RunWriter.initialize(
        run_dir=run_dir, run_id="run", run_name="experiment",
        artifact_root=tmp_path, collision_outcome="created", created_at="now",
        config_fingerprint="fp", resolved_config={}, world_size=1,
    )
    assert writer.file_inventory() == ("logging.jsonl", "resolved_config.json", "run.json")


def test_final_and_best_aliases_are_canonical_selector_owners(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_final(step=5)
    assert writer.write_best(
        step=4, selector="acc_top1", value=0.8, optimizer_update_status="applied", finite_status="finite", checkpoint_committed=True
    )
    assert json.loads((writer.checkpoints_dir / "final.json").read_text()) == {
        "checkpoint_path": "checkpoints/step-5", "step": 5
    }
    assert json.loads((writer.checkpoints_dir / "best.json").read_text()) == {
        "checkpoint_path": "checkpoints/step-4", "selector": "acc_top1", "step": 4, "value": 0.8
    }
    state = writer.read_run()
    assert not set(state).intersection({"final", "best", "selector", "selector_value"})


@pytest.mark.parametrize(
    ("update", "finite", "committed", "value"),
    [
        ("skipped", "finite", True, 0.9),
        ("applied", "unsafe", True, 0.9),
        ("applied", "finite", False, 0.9),
        ("applied", "finite", True, float("nan")),
        ("applied", "finite", True, float("inf")),
    ],
)
def test_ineligible_step_cannot_advance_best(
    tmp_path: Path, update: str, finite: str, committed: bool, value: float
) -> None:
    writer = _writer(tmp_path)
    assert writer.write_best(
        step=1, selector="acc_top1", value=0.5, optimizer_update_status="applied", finite_status="finite", checkpoint_committed=True
    )
    assert not writer.write_best(
        step=2, selector="acc_top1", value=value, optimizer_update_status=update, finite_status=finite, checkpoint_committed=committed
    )
    assert json.loads((writer.checkpoints_dir / "best.json").read_text())["step"] == 1


def test_finalize_is_atomic_compact_and_bounds_terminal_error(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.finalize(
        status="failed", updated_at="2026-07-11T00:01:00Z", completed_steps=2, consumed_packs=8,
        checkpoint_event_count=1, optimizer_update_status="skipped", finite_status="unsafe", terminal_error="x" * 5000,
    )
    state = writer.read_run()
    assert state["status"] == "failed" and state["completed_steps"] == 2
    assert len(state["terminal_error"]) == 1024
    assert not list(writer.run_dir.glob(".run.json.*"))


def _writer(tmp_path: Path) -> RunWriter:
    return RunWriter.initialize(
        run_dir=tmp_path / "run-a", run_id="run-a", run_name="experiment",
        artifact_root=tmp_path, collision_outcome="created", created_at="2026-07-11T00:00:00Z",
        config_fingerprint="config-fp", resolved_config={"training": {"seed": 7}}, world_size=2,
        resolved_max_steps=5,
    )
