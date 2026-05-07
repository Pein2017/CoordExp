# Stage-2 Runtime State Schema Refactor Follow-Up Plan

Status: planned follow-up only. Task 10 created this plan as a decision-gate output and did not implement runtime-state code.

## Goal

Introduce a dedicated typed object for the Stage-2 checkpoint runtime-state payload owned by `Stage2ABTrainingTrainer._coordexp_checkpoint_runtime_state()` and `Stage2ABTrainingTrainer._coordexp_restore_checkpoint_runtime_state()`, while preserving all existing serialized checkpoint keys and avoiding metric/logging-key reshaping.

First concrete runtime state object selected by Task 10:

```text
Stage2CheckpointRuntimeState
```

## Boundary

Modify these files only:

```text
src/trainers/stage2_two_channel/runtime_state.py
src/trainers/stage2_two_channel.py
src/trainers/stage2_two_channel/__init__.py
tests/test_stage2_ab_training.py
```

Do not modify these contracts:

```text
src/trainers/teacher_forcing/contracts.py::ModuleResult
src/trainers/teacher_forcing/contracts.py::PipelineResult
src/trainers/teacher_forcing/objective_pipeline.py::run_teacher_forcing_pipeline
src/trainers/stage2_two_channel/types.py::Stage2PreparedSegment
src/trainers/stage2_two_channel/types.py::Stage2BatchMetrics
```

Do not rename these serialized checkpoint keys:

```text
stage2_pending_train_logs
stage2_metric_snapshots
stage2_post_rollout_segments
stage2_b_step_gs
stage2_b_step_micro
stage2_b_step_raw
stage2_a_step_gs
stage2_a_step_micro
stage2_a_step_raw
stage2_ab_realized_last_gs
stage2_ab_realized_recent
stage2_train_monitor_pending_gs
stage2_train_monitor_candidates
stage2_train_monitor_b_step_count
stage2_train_monitor_dump_last_step
stage2_train_monitor_dump_count
stage2_train_monitor_dump_written_step
```

## Step 1: Add Typed Runtime-State Tests

Append these tests to `tests/test_stage2_ab_training.py`.

```python
def test_stage2_checkpoint_runtime_state_roundtrips_serialized_payload():
    from src.trainers.stage2_two_channel.runtime_state import (
        Stage2CheckpointRuntimeState,
    )

    payload = {
        "stage2_pending_train_logs": {
            7: {
                "n_micro": 2,
                "weight_sum": 4.0,
                "gradmon_weight_sum": 3.0,
                "sums": {
                    "loss/token_ce": 1.25,
                    "stage2/raw_rollouts": 2.0,
                },
            }
        },
        "stage2_metric_snapshots": {
            "loss/token_ce": 0.5,
            "stage2_ab/b_ratio_realized": 1.0,
        },
        "stage2_post_rollout_segments": {
            "A": [({"input_ids": [1, 2]}, {"channel": "A"}, 2)],
            "B": [({"input_ids": [3]}, {"channel": "B"}, 1)],
        },
        "stage2_b_step_gs": 8,
        "stage2_b_step_micro": 1,
        "stage2_b_step_raw": [{"messages": [{"role": "user", "content": "b"}]}],
        "stage2_a_step_gs": 9,
        "stage2_a_step_micro": 2,
        "stage2_a_step_raw": [{"messages": [{"role": "user", "content": "a"}]}],
        "stage2_ab_realized_last_gs": 10,
        "stage2_ab_realized_recent": [0, 1, 1],
        "stage2_train_monitor_pending_gs": 11,
        "stage2_train_monitor_candidates": [
            {"global_step": 11, "channel": "B", "image_id": "sample-1"}
        ],
        "stage2_train_monitor_b_step_count": 3,
        "stage2_train_monitor_dump_last_step": 12,
        "stage2_train_monitor_dump_count": 4,
        "stage2_train_monitor_dump_written_step": 13,
    }

    state = Stage2CheckpointRuntimeState.from_mapping(payload)

    assert state.to_mapping() == payload


def test_stage2_checkpoint_runtime_state_rejects_malformed_pending_log():
    from src.trainers.stage2_two_channel.runtime_state import (
        Stage2CheckpointRuntimeState,
    )

    payload = {
        "stage2_pending_train_logs": {
            7: {
                "n_micro": 2,
                "weight_sum": 4.0,
                "gradmon_weight_sum": 3.0,
            }
        }
    }

    with pytest.raises(TypeError, match="stage2_pending_train_logs"):
        Stage2CheckpointRuntimeState.from_mapping(payload)


def test_stage2_checkpoint_runtime_state_hooks_roundtrip(monkeypatch):
    from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer
    from src.trainers.stage2_two_channel.runtime_state import (
        Stage2CheckpointRuntimeState,
    )

    restored_base_payloads = []

    monkeypatch.setattr(
        RolloutMatchingSFTTrainer,
        "_coordexp_checkpoint_runtime_state",
        lambda self: {"base_runtime_state": "preserved"},
    )

    def _restore_base_runtime_state(self, payload):
        restored_base_payloads.append(dict(payload))

    monkeypatch.setattr(
        RolloutMatchingSFTTrainer,
        "_coordexp_restore_checkpoint_runtime_state",
        _restore_base_runtime_state,
    )

    source = _make_min_trainer()
    source._stage2_pending_train_logs = {
        7: _PendingStage2Log(
            n_micro=2,
            weight_sum=4.0,
            gradmon_weight_sum=3.0,
            sums={"loss/token_ce": 1.25, "stage2/raw_rollouts": 2.0},
        )
    }
    source._stage2_metric_snapshots = {
        "loss/token_ce": 0.5,
        "stage2_ab/b_ratio_realized": 1.0,
    }
    source._stage2_post_rollout_segments = {
        "A": [({"input_ids": [1, 2]}, {"channel": "A"}, 2)],
        "B": [({"input_ids": [3]}, {"channel": "B"}, 1)],
    }
    source._stage2_b_step_gs = 8
    source._stage2_b_step_micro = 1
    source._stage2_b_step_raw = [{"messages": [{"role": "user", "content": "b"}]}]
    source._stage2_a_step_gs = 9
    source._stage2_a_step_micro = 2
    source._stage2_a_step_raw = [{"messages": [{"role": "user", "content": "a"}]}]
    source._stage2_ab_realized_last_gs = 10
    source._stage2_ab_realized_recent = [0, 1, 1]
    source._stage2_train_monitor_pending_gs = 11
    source._stage2_train_monitor_candidates = [
        {"global_step": 11, "channel": "B", "image_id": "sample-1"}
    ]
    source._stage2_train_monitor_b_step_count = 3
    source._stage2_train_monitor_dump_last_step = 12
    source._stage2_train_monitor_dump_count = 4
    source._stage2_train_monitor_dump_written_step = 13

    payload = source._coordexp_checkpoint_runtime_state()
    state_mapping = Stage2CheckpointRuntimeState.from_mapping(payload).to_mapping()

    assert payload["base_runtime_state"] == "preserved"
    assert state_mapping == {key: payload[key] for key in state_mapping}

    target = _make_min_trainer()
    target._coordexp_restore_checkpoint_runtime_state(payload)

    assert restored_base_payloads == [payload]
    assert target._stage2_pending_train_logs[7].n_micro == 2
    assert target._stage2_pending_train_logs[7].sums[
        "loss/token_ce"
    ] == pytest.approx(1.25)
    assert target._stage2_metric_snapshots == source._stage2_metric_snapshots
    assert target._stage2_post_rollout_segments == source._stage2_post_rollout_segments
    assert target._stage2_b_step_gs == 8
    assert target._stage2_b_step_micro == 1
    assert target._stage2_b_step_raw == source._stage2_b_step_raw
    assert target._stage2_a_step_gs == 9
    assert target._stage2_a_step_micro == 2
    assert target._stage2_a_step_raw == source._stage2_a_step_raw
    assert target._stage2_ab_realized_last_gs == 10
    assert list(target._stage2_ab_realized_recent) == [0, 1, 1]
    assert target._stage2_train_monitor_pending_gs == 11
    assert (
        target._stage2_train_monitor_candidates
        == source._stage2_train_monitor_candidates
    )
    assert target._stage2_train_monitor_b_step_count == 3
    assert target._stage2_train_monitor_dump_last_step == 12
    assert target._stage2_train_monitor_dump_count == 4
    assert target._stage2_train_monitor_dump_written_step == 13
```

Run the focused red check before implementation:

```bash
rtk conda run -n ms python -m pytest tests/test_stage2_ab_training.py -q -k "stage2_checkpoint_runtime_state"
```

Expected before implementation: all selected runtime-state tests fail because `src.trainers.stage2_two_channel.runtime_state` does not exist.

## Step 2: Add The Runtime-State Module

Create `src/trainers/stage2_two_channel/runtime_state.py` with these symbols:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

Stage2SerializedSegment = tuple[Mapping[str, Any], Mapping[str, Any], int]


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True)
class Stage2PendingTrainLogState:
    n_micro: int = 0
    weight_sum: float = 0.0
    gradmon_weight_sum: float = 0.0
    sums: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Stage2PendingTrainLogState":
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_pending_train_logs entries must be mappings")
        if "sums" not in payload:
            raise TypeError("stage2_pending_train_logs entries require sums")
        sums_raw = payload.get("sums")
        if not isinstance(sums_raw, Mapping):
            raise TypeError("stage2_pending_train_logs entries require mapping sums")
        return cls(
            n_micro=int(payload.get("n_micro", 0) or 0),
            weight_sum=float(payload.get("weight_sum", 0.0) or 0.0),
            gradmon_weight_sum=float(payload.get("gradmon_weight_sum", 0.0) or 0.0),
            sums={str(key): float(value) for key, value in sums_raw.items()},
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "n_micro": int(self.n_micro),
            "weight_sum": float(self.weight_sum),
            "gradmon_weight_sum": float(self.gradmon_weight_sum),
            "sums": dict(self.sums),
        }


@dataclass(frozen=True)
class Stage2StepBudgetState:
    global_step: int | None = None
    micro_step: int = 0
    raw_samples: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    @classmethod
    def from_payload(
        cls,
        *,
        global_step: Any,
        micro_step: Any,
        raw_samples: Any,
    ) -> "Stage2StepBudgetState":
        return cls(
            global_step=_as_int_or_none(global_step),
            micro_step=int(micro_step or 0),
            raw_samples=tuple(
                dict(sample) for sample in list(raw_samples or []) if isinstance(sample, Mapping)
            ),
        )


@dataclass(frozen=True)
class Stage2TrainMonitorState:
    pending_global_step: int | None = None
    candidates: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    b_step_count: int = 0
    dump_last_step: int | None = None
    dump_count: int = 0
    dump_written_step: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Stage2TrainMonitorState":
        return cls(
            pending_global_step=_as_int_or_none(payload.get("stage2_train_monitor_pending_gs")),
            candidates=tuple(
                dict(candidate)
                for candidate in list(payload.get("stage2_train_monitor_candidates") or [])
                if isinstance(candidate, Mapping)
            ),
            b_step_count=int(payload.get("stage2_train_monitor_b_step_count", 0) or 0),
            dump_last_step=_as_int_or_none(payload.get("stage2_train_monitor_dump_last_step")),
            dump_count=int(payload.get("stage2_train_monitor_dump_count", 0) or 0),
            dump_written_step=_as_int_or_none(
                payload.get("stage2_train_monitor_dump_written_step")
            ),
        )


@dataclass(frozen=True)
class Stage2CheckpointRuntimeState:
    pending_train_logs: Mapping[int, Stage2PendingTrainLogState] = field(default_factory=dict)
    metric_snapshots: Mapping[str, float] = field(default_factory=dict)
    post_rollout_segments: Mapping[str, Sequence[Stage2SerializedSegment]] = field(
        default_factory=lambda: {"A": (), "B": ()}
    )
    b_step: Stage2StepBudgetState = field(default_factory=Stage2StepBudgetState)
    a_step: Stage2StepBudgetState = field(default_factory=Stage2StepBudgetState)
    ab_realized_last_gs: int | None = None
    ab_realized_recent: Sequence[int] = field(default_factory=tuple)
    train_monitor: Stage2TrainMonitorState = field(default_factory=Stage2TrainMonitorState)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Stage2CheckpointRuntimeState":
        if not isinstance(payload, Mapping):
            raise TypeError("Stage2 checkpoint runtime state must be a Mapping")
        pending_raw = payload.get("stage2_pending_train_logs", {})
        if not isinstance(pending_raw, Mapping):
            raise TypeError("stage2_pending_train_logs must be a mapping")
        segments_raw = payload.get("stage2_post_rollout_segments", {})
        if not isinstance(segments_raw, Mapping):
            raise TypeError("stage2_post_rollout_segments must be a mapping")
        snapshots_raw = payload.get("stage2_metric_snapshots", {})
        if not isinstance(snapshots_raw, Mapping):
            raise TypeError("stage2_metric_snapshots must be a mapping")
        return cls(
            pending_train_logs={
                int(step): Stage2PendingTrainLogState.from_mapping(log_payload)
                for step, log_payload in pending_raw.items()
            },
            metric_snapshots={
                str(key): float(value) for key, value in snapshots_raw.items()
            },
            post_rollout_segments={
                "A": tuple(segments_raw.get("A") or ()),
                "B": tuple(segments_raw.get("B") or ()),
            },
            b_step=Stage2StepBudgetState.from_payload(
                global_step=payload.get("stage2_b_step_gs"),
                micro_step=payload.get("stage2_b_step_micro", 0),
                raw_samples=payload.get("stage2_b_step_raw"),
            ),
            a_step=Stage2StepBudgetState.from_payload(
                global_step=payload.get("stage2_a_step_gs"),
                micro_step=payload.get("stage2_a_step_micro", 0),
                raw_samples=payload.get("stage2_a_step_raw"),
            ),
            ab_realized_last_gs=_as_int_or_none(payload.get("stage2_ab_realized_last_gs")),
            ab_realized_recent=tuple(
                int(item) for item in list(payload.get("stage2_ab_realized_recent") or [])
            ),
            train_monitor=Stage2TrainMonitorState.from_mapping(payload),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage2_pending_train_logs": {
                int(step): pending.to_mapping()
                for step, pending in self.pending_train_logs.items()
            },
            "stage2_metric_snapshots": dict(self.metric_snapshots),
            "stage2_post_rollout_segments": {
                "A": list(self.post_rollout_segments.get("A") or ()),
                "B": list(self.post_rollout_segments.get("B") or ()),
            },
            "stage2_b_step_gs": self.b_step.global_step,
            "stage2_b_step_micro": int(self.b_step.micro_step),
            "stage2_b_step_raw": list(self.b_step.raw_samples),
            "stage2_a_step_gs": self.a_step.global_step,
            "stage2_a_step_micro": int(self.a_step.micro_step),
            "stage2_a_step_raw": list(self.a_step.raw_samples),
            "stage2_ab_realized_last_gs": self.ab_realized_last_gs,
            "stage2_ab_realized_recent": list(self.ab_realized_recent),
            "stage2_train_monitor_pending_gs": self.train_monitor.pending_global_step,
            "stage2_train_monitor_candidates": list(self.train_monitor.candidates),
            "stage2_train_monitor_b_step_count": int(self.train_monitor.b_step_count),
            "stage2_train_monitor_dump_last_step": self.train_monitor.dump_last_step,
            "stage2_train_monitor_dump_count": int(self.train_monitor.dump_count),
            "stage2_train_monitor_dump_written_step": self.train_monitor.dump_written_step,
        }
```

## Step 3: Route Checkpoint Methods Through The Type

In `src/trainers/stage2_two_channel.py`, add this import near the current Stage-2 helper imports:

```python
from src.trainers.stage2_two_channel.runtime_state import Stage2CheckpointRuntimeState
```

Replace the Stage-2-only body of `_coordexp_checkpoint_runtime_state()` with this call pattern. It intentionally avoids `asdict` because current `src/trainers/stage2_two_channel.py` imports `dataclass` and `field` from `dataclasses`, but not `asdict`.

```python
        stage2_runtime_state = Stage2CheckpointRuntimeState.from_mapping(
            {
                "stage2_pending_train_logs": {
                    int(step): {
                        "n_micro": int(pending.n_micro),
                        "weight_sum": float(pending.weight_sum),
                        "gradmon_weight_sum": float(pending.gradmon_weight_sum),
                        "sums": dict(pending.sums),
                    }
                    for step, pending in self._stage2_pending_train_logs.items()
                },
                "stage2_metric_snapshots": dict(self._stage2_metric_snapshots),
                "stage2_post_rollout_segments": {
                    str(channel): list(segments)
                    for channel, segments in self._stage2_post_rollout_segments.items()
                },
                "stage2_b_step_gs": self._stage2_b_step_gs,
                "stage2_b_step_micro": int(self._stage2_b_step_micro),
                "stage2_b_step_raw": list(self._stage2_b_step_raw),
                "stage2_a_step_gs": self._stage2_a_step_gs,
                "stage2_a_step_micro": int(self._stage2_a_step_micro),
                "stage2_a_step_raw": list(self._stage2_a_step_raw),
                "stage2_ab_realized_last_gs": self._stage2_ab_realized_last_gs,
                "stage2_ab_realized_recent": list(self._stage2_ab_realized_recent),
                "stage2_train_monitor_pending_gs": self._stage2_train_monitor_pending_gs,
                "stage2_train_monitor_candidates": list(
                    self._stage2_train_monitor_candidates
                ),
                "stage2_train_monitor_b_step_count": int(
                    self._stage2_train_monitor_b_step_count
                ),
                "stage2_train_monitor_dump_last_step": self._stage2_train_monitor_dump_last_step,
                "stage2_train_monitor_dump_count": int(
                    self._stage2_train_monitor_dump_count
                ),
                "stage2_train_monitor_dump_written_step": self._stage2_train_monitor_dump_written_step,
            }
        )
        payload.update(stage2_runtime_state.to_mapping())
```

Replace the Stage-2 payload parsing portion of `_coordexp_restore_checkpoint_runtime_state()` with this call pattern:

```python
        stage2_runtime_state = Stage2CheckpointRuntimeState.from_mapping(payload)

        self._stage2_pending_train_logs = {
            int(step): _PendingStage2Log(**pending.to_mapping())
            for step, pending in stage2_runtime_state.pending_train_logs.items()
        }
        self._stage2_metric_snapshots = dict(stage2_runtime_state.metric_snapshots)
        self._stage2_post_rollout_segments = {
            str(channel): list(segments)
            for channel, segments in stage2_runtime_state.post_rollout_segments.items()
        }
        for channel in ("A", "B"):
            self._stage2_post_rollout_segments.setdefault(channel, [])

        self._stage2_b_step_gs = stage2_runtime_state.b_step.global_step
        self._stage2_b_step_micro = int(stage2_runtime_state.b_step.micro_step)
        self._stage2_b_step_raw = list(stage2_runtime_state.b_step.raw_samples)
        self._stage2_a_step_gs = stage2_runtime_state.a_step.global_step
        self._stage2_a_step_micro = int(stage2_runtime_state.a_step.micro_step)
        self._stage2_a_step_raw = list(stage2_runtime_state.a_step.raw_samples)
        self._stage2_ab_realized_last_gs = stage2_runtime_state.ab_realized_last_gs
        self._stage2_ab_realized_recent = deque(
            list(stage2_runtime_state.ab_realized_recent),
            maxlen=200,
        )
        self._stage2_train_monitor_pending_gs = (
            stage2_runtime_state.train_monitor.pending_global_step
        )
        self._stage2_train_monitor_candidates = list(
            stage2_runtime_state.train_monitor.candidates
        )
        self._stage2_train_monitor_b_step_count = int(
            stage2_runtime_state.train_monitor.b_step_count
        )
        self._stage2_train_monitor_dump_last_step = (
            stage2_runtime_state.train_monitor.dump_last_step
        )
        self._stage2_train_monitor_dump_count = int(
            stage2_runtime_state.train_monitor.dump_count
        )
        self._stage2_train_monitor_dump_written_step = (
            stage2_runtime_state.train_monitor.dump_written_step
        )
```

In `src/trainers/stage2_two_channel/__init__.py`, export the new type:

```python
from .runtime_state import Stage2CheckpointRuntimeState
```

Add this string to `__all__`:

```python
"Stage2CheckpointRuntimeState",
```

## Step 4: Verification

Run these commands:

```bash
rtk conda run -n ms python -m pytest tests/test_stage2_ab_training.py -q -k "stage2_checkpoint_runtime_state or checkpoint_runtime_state or restore_checkpoint_runtime_state"
rtk conda run -n ms python -m pytest tests/test_stage2_ab_training.py tests/test_stage2_two_channel_training.py tests/test_batch_extras_contract.py tests/test_teacher_forcing_token_ce.py tests/test_stage1_set_continuation_branch_runtime.py -q
git diff --check
```

Expected after implementation:

```text
All selected tests pass.
git diff --check exits 0.
```

## Acceptance Criteria

- `ModuleResult` and `PipelineResult` remain the canonical teacher-forcing result containers.
- `Stage2PreparedSegment` remains unchanged in this follow-up.
- The Stage-2 checkpoint payload keeps all existing key names.
- `test_stage2_checkpoint_runtime_state_hooks_roundtrip()` calls both `Stage2ABTrainingTrainer._coordexp_checkpoint_runtime_state()` and `_coordexp_restore_checkpoint_runtime_state()`.
- Dynamic metric maps remain mappings and no logging key is renamed.
- Malformed pending-log payloads fail fast through `Stage2CheckpointRuntimeState.from_mapping()`.
