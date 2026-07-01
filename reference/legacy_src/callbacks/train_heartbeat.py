"""Lightweight training heartbeat instrumentation for smoke/triage runs.

The heartbeat is intentionally best-effort and should never fail training. It
writes JSONL events under the run output directory so we can distinguish
"stalled before first batch" from "very slow first optimizer step".
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..utils import get_logger

logger = get_logger(__name__)


class TrainHeartbeatWriter:
    """Append heartbeat events to a JSONL file (best-effort I/O)."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self._warned_io_error = False
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            **payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            if not self._warned_io_error:
                logger.warning("Train heartbeat write failed (%s): %r", self.path, exc)
                self._warned_io_error = True


class HeartbeatDataCollator:
    """Wrap a collator and emit an event when the first batch is collated."""

    def __init__(
        self,
        base_collator: Callable[[Any], Any],
        *,
        writer: TrainHeartbeatWriter,
    ) -> None:
        self.base_collator = base_collator
        self.writer = writer
        self._first_collate_emitted = False

    def __call__(self, features: Any) -> Any:
        if not self._first_collate_emitted:
            feature_count = None
            try:
                feature_count = len(features)
            except Exception:
                feature_count = None
            self.writer.emit("first_batch_collate", feature_count=feature_count)
            self._first_collate_emitted = True
        return self.base_collator(features)


class TrainHeartbeatCallback(TrainerCallback):
    """Emit callback-level heartbeat markers during training lifecycle."""

    def __init__(self, writer: TrainHeartbeatWriter, *, interval_s: float = 30.0) -> None:
        self.writer = writer
        self.interval_s = max(1.0, float(interval_s))
        self._last_periodic_ts = time.monotonic()
        self._seen_first_substep = False
        self._seen_first_step_begin = False
        self._seen_first_step_end = False

    def _state_payload(self, state: TrainerState) -> dict[str, Any]:
        return {
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "epoch": (
                float(state.epoch)
                if getattr(state, "epoch", None) is not None
                else None
            ),
        }

    def _maybe_periodic(self, state: TrainerState) -> None:
        now = time.monotonic()
        if now - self._last_periodic_ts < self.interval_s:
            return
        self._last_periodic_ts = now
        self.writer.emit("periodic", **self._state_payload(state))

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.writer.emit("train_begin", **self._state_payload(state))

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._seen_first_substep:
            self.writer.emit("first_substep_end", **self._state_payload(state))
            self._seen_first_substep = True
        self._maybe_periodic(state)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._seen_first_step_begin:
            self.writer.emit("first_step_begin", **self._state_payload(state))
            self._seen_first_step_begin = True
        self._maybe_periodic(state)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._seen_first_step_end:
            self.writer.emit("first_step_end", **self._state_payload(state))
            self._seen_first_step_end = True
        self._maybe_periodic(state)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        keys = sorted(str(k) for k in (logs or {}).keys())
        self.writer.emit("log", log_keys=keys, **self._state_payload(state))

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        keys = sorted(str(k) for k in (metrics or {}).keys())
        self.writer.emit("evaluate", metric_keys=keys, **self._state_payload(state))

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.writer.emit("save", **self._state_payload(state))

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.writer.emit("train_end", **self._state_payload(state))
