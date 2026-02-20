from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerControl, TrainerState, TrainingArguments

from src.callbacks.train_heartbeat import (
    HeartbeatDataCollator,
    TrainHeartbeatCallback,
    TrainHeartbeatWriter,
)


def _read_events(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def test_heartbeat_collator_emits_first_batch_once(tmp_path: Path) -> None:
    out = tmp_path / "heartbeat.jsonl"
    writer = TrainHeartbeatWriter(out)

    def _base_collator(features: list[dict]) -> dict:
        return {"n": len(features)}

    collator = HeartbeatDataCollator(_base_collator, writer=writer)
    assert collator([{"x": 1}, {"x": 2}]) == {"n": 2}
    assert collator([{"x": 3}]) == {"n": 1}

    events = _read_events(out)
    names = [str(e.get("event")) for e in events]
    assert names.count("first_batch_collate") == 1
    first = next(e for e in events if e.get("event") == "first_batch_collate")
    assert first.get("feature_count") == 2


def test_heartbeat_callback_emits_lifecycle_markers(tmp_path: Path) -> None:
    out = tmp_path / "heartbeat.jsonl"
    writer = TrainHeartbeatWriter(out)
    callback = TrainHeartbeatCallback(writer)

    args = TrainingArguments(output_dir=str(tmp_path), report_to=[])
    state = TrainerState()
    control = TrainerControl()

    callback.on_train_begin(args, state, control)
    callback.on_substep_end(args, state, control)
    callback.on_step_begin(args, state, control)

    state.global_step = 1
    callback.on_step_end(args, state, control)
    callback.on_log(args, state, control, logs={"loss": 1.0})
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})
    callback.on_save(args, state, control)
    callback.on_train_end(args, state, control)

    events = _read_events(out)
    names = [str(e.get("event")) for e in events]
    assert "train_begin" in names
    assert "first_substep_end" in names
    assert "first_step_begin" in names
    assert "first_step_end" in names
    assert "log" in names
    assert "evaluate" in names
    assert "save" in names
    assert "train_end" in names
