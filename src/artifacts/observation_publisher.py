"""JSONL-first publication of a canonical observation and its derived sinks.

``add-coordexp-swift-training-observability`` (requirement *Rank-Zero
Presentation Sinks*) gives this module one job: take the strict canonical row
that ``src/training/reporting.py`` built, publish it to ``logging.jsonl``
through the existing all-rank :class:`~src.artifacts.run_writer.RunWriter`
append/status handshake, and only then let rank zero PRESENT it.

Ownership boundaries this module deliberately keeps:

* typed metric reduction belongs to ``src/runtime/metrics.py``;
* canonical row construction belongs to ``src/training/reporting.py``;
* composition (who publishes what, with which cadence and run directory)
  belongs to ``src/training/session.py``.

:class:`ObservationPublisher` therefore exposes exactly one publication call
plus a terminal ``close``. It is a direct call chain, not a dispatcher: there
are no event names, no subscription table, no dynamic sink registry, and no
alternate scalar authority. ``logging.jsonl`` remains the sole durable record;
console and TensorBoard are strictly derived from a row that is already on
disk, so a derived-sink failure can never remove, rewrite, or invalidate a
published observation.

The TensorBoard sink is lazily created (a non-main rank therefore never even
constructs an event file), bounded, and governed by ONE one-way disabled latch
covering import, initialization, ``add_scalar``, ``flush``, and ``close``. The
first failure records at most one bounded run warning plus one best-effort
stderr line, best-effort discards the writer, and latches the sink off; a
failure during that cleanup cannot recurse or warn again.

The approximate ETA is segment-local: it is derived from monotonic elapsed time
and completed planned-step progress inside THIS process segment, is labeled
approximate, never enters a canonical row, a TensorBoard tag, or exact-resume
state, and is discarded when the process exits.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import ArtifactContractError, RuntimeContractError

try:
    from accelerate.utils import broadcast_object_list
except ImportError:  # pragma: no cover - exercised only in stripped environments.
    broadcast_object_list = None  # type: ignore[assignment]


#: The single bounded run-warning code for the whole derived TensorBoard sink.
TENSORBOARD_DISABLED_WARNING = "observation_publisher.tensorboard_disabled"

#: Run-local directory for the derived event files; never a per-rank tree.
TENSORBOARD_DIRECTORY_NAME = "tensorboard"

#: Row keys that ARE the observation identity rather than a scalar to mirror.
_IDENTITY_FIELDS = frozenset({"step", "split"})

#: Bounded queue for the derived event writer: presentation must not grow
#: without bound behind a slow or wedged filesystem.
_MAX_QUEUED_EVENTS = 128

#: Bounded in-memory presentation history (diagnostics only, never persisted).
_MAX_RETAINED_UPDATES = 64


def _append_logging_row_shared(
    *, writer: Any, row: Mapping[str, Any], runtime: Any
) -> None:
    """Append on rank zero and make its bounded outcome common to every rank.

    Moved verbatim from ``src.training.reporting`` by
    ``add-coordexp-swift-training-observability`` Wave 4: JSONL publication and
    its distributed success handshake are owned here, so ``reporting`` can stay
    the pure canonical row builder. Behavior, including both bounded error
    codes, is unchanged.
    """

    accelerator = getattr(runtime, "accelerator", runtime)
    is_main = bool(
        getattr(
            runtime, "is_main_process", getattr(accelerator, "is_main_process", True)
        )
    )
    status: dict[str, Any] = {"ok": True}
    if is_main:
        try:
            if writer is None:
                raise RuntimeError("rank zero has no run writer")
            writer.append_logging_row(row)
        except BaseException as exc:
            status = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:1024],
            }
    values: list[Any] = [status]
    if (
        int(getattr(runtime, "world_size", getattr(accelerator, "num_processes", 1)))
        > 1
    ):
        broadcast = getattr(accelerator, "broadcast_object_list", None)
        if callable(broadcast):
            result = broadcast(values, from_process=0)
            if result is not None:
                values = result
        elif broadcast_object_list is not None:
            broadcast_object_list(values, from_process=0)
        else:
            raise RuntimeContractError(
                "logging outcome broadcast requires accelerate",
                code="runtime.logging_broadcast_unavailable",
            )
    shared = values[0]
    if not isinstance(shared, Mapping) or not bool(shared.get("ok")):
        error = (
            shared.get("error", "invalid status")
            if isinstance(shared, Mapping)
            else shared
        )
        raise RuntimeContractError(
            f"rank zero logging append failed: {error}",
            code="runtime.logging_append_failed",
        )


@dataclass(frozen=True)
class PresentationUpdate:
    """One derived rank-zero presentation payload.

    ``step`` and ``total_steps`` are REQUIRED: every presentation renders
    current planned step over the resolved total. ``eta_seconds`` is the
    approximate segment-local estimate and exists only in memory.
    """

    split: str
    step: int
    total_steps: int
    terminal: bool
    scalars: Mapping[str, float]
    status: Mapping[str, str]
    eta_seconds: float | None


def _default_tensorboard_writer(log_dir: Path) -> Any:
    """Import and construct the event writer lazily, inside the failure latch."""

    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(
        log_dir=str(log_dir), max_queue=_MAX_QUEUED_EVENTS, flush_secs=120
    )


def _finite_scalar(value: Any) -> float | None:
    """Return a finite float for a numeric row value, else ``None``.

    Booleans, strings, ``None``, nested diagnostics, and non-finite values are
    never synthesized into a scalar tag or a console number.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _format_seconds(seconds: float) -> str:
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _format_bytes(value: float) -> str:
    gib = value / (1024.0**3)
    if gib >= 0.1:
        return f"{gib:.2f}GiB"
    return f"{value / (1024.0 ** 2):.0f}MiB"


class ObservationPublisher:
    """Publish one canonical row, then present it on rank zero.

    ``publish`` is the single train/eval publication interface. Presentation is
    enabled only when ``presentation_steps`` is given, which is exactly where
    ``src/training/session.py`` composes the resolved ``observability.steps``
    cadence and the resolved total planned steps.
    """

    def __init__(
        self,
        *,
        writer: Any,
        runtime: Any,
        run_dir: Path | str | None = None,
        presentation_steps: int | None = None,
        total_planned_steps: int | None = None,
        console: Any = None,
        tensorboard_factory: Callable[[Path], Any] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._writer = writer
        self._runtime = runtime
        self._run_dir = None if run_dir is None else Path(run_dir)
        self._console = console
        self._console_disabled = False
        self._tensorboard_factory = (
            _default_tensorboard_writer
            if tensorboard_factory is None
            else tensorboard_factory
        )
        self._monotonic = time.monotonic if monotonic is None else monotonic
        self._presentation_steps: int | None = None
        self._total_planned_steps: int | None = None
        if presentation_steps is not None:
            interval = int(presentation_steps)
            if interval <= 0:
                raise ArtifactContractError(
                    "presentation interval must be a positive planned-step count",
                    code="observation_publisher.presentation_interval_invalid",
                    context={"presentation_steps": interval},
                )
            if total_planned_steps is None or int(total_planned_steps) <= 0:
                raise ArtifactContractError(
                    "presentation requires the resolved total planned step count",
                    code="observation_publisher.presentation_total_missing",
                )
            if self._run_dir is None:
                raise ArtifactContractError(
                    "presentation requires the shared run directory",
                    code="observation_publisher.presentation_run_dir_missing",
                )
            self._presentation_steps = interval
            self._total_planned_steps = int(total_planned_steps)
        self._tensorboard: Any = None
        self._tensorboard_disabled = False
        self._tensorboard_warned = False
        # Segment-local ETA state: started fresh in this process, never
        # restored from a checkpoint and never written anywhere.
        self._segment_started_monotonic: float | None = None
        self._segment_first_step: int | None = None
        self._presented: list[PresentationUpdate] = []

    # -- publication ------------------------------------------------------

    def publish(self, row: Mapping[str, Any], *, terminal: bool = False) -> None:
        """Publish one canonical row, then present it if rank zero owes one.

        The JSONL append and its all-rank status handshake complete FIRST. A
        failed publication raises exactly as before and presents nothing: a
        derived sink may never describe an observation that has no canonical
        row.
        """

        # Identity is validated on EVERY rank, never only on rank zero: a
        # rank-local raise in front of the shared append/status handshake would
        # desynchronize the collective instead of failing the step closed.
        identity: tuple[str, int] | None = (
            None if self._presentation_steps is None else self._identity(row)
        )
        _append_logging_row_shared(writer=self._writer, row=row, runtime=self._runtime)
        if identity is None or not self._is_main:
            return
        split, step = identity
        if not self._presentation_due(split=split, step=step, terminal=terminal):
            return
        update = self._build_update(row, split=split, step=step, terminal=terminal)
        self._presented.append(update)
        del self._presented[:-_MAX_RETAINED_UPDATES]
        self._render_console(update)
        self._mirror_tensorboard(update)

    def close(self) -> None:
        """Flush and close the derived event writer exactly once, best effort."""

        if self._tensorboard_disabled:
            return
        board = self._tensorboard
        # Latch BEFORE the cleanup call so a close failure cannot recurse.
        self._tensorboard_disabled = True
        self._tensorboard = None
        if board is None:
            return
        try:
            board.flush()
            board.close()
        except BaseException as exc:
            self._warn_tensorboard_once("close", exc)

    # -- identity and cadence ---------------------------------------------

    @property
    def _is_main(self) -> bool:
        runtime = self._runtime
        accelerator = getattr(runtime, "accelerator", runtime)
        return bool(
            getattr(
                runtime,
                "is_main_process",
                getattr(accelerator, "is_main_process", True),
            )
        )

    def _identity(self, row: Mapping[str, Any]) -> tuple[str, int]:
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ArtifactContractError(
                "a presented observation requires its positive planned step",
                code="observation_publisher.presentation_step_missing",
            )
        return str(row.get("split")), int(step)

    def _presentation_due(self, *, split: str, step: int, terminal: bool) -> bool:
        interval = self._presentation_steps
        if interval is None:
            return False
        if split != "train":
            # Every completed eval invocation is presented when it runs.
            return True
        if terminal:
            # A successfully published terminal optimizer-boundary row is
            # presented even though its planned step is off cadence.
            return True
        if step == self._total_planned_steps:
            return True
        return step % interval == 0

    def _build_update(
        self, row: Mapping[str, Any], *, split: str, step: int, terminal: bool
    ) -> PresentationUpdate:
        scalars: dict[str, float] = {}
        status: dict[str, str] = {}
        for key, value in row.items():
            if key in _IDENTITY_FIELDS:
                continue
            number = _finite_scalar(value)
            if number is not None:
                scalars[str(key)] = number
            elif isinstance(value, str):
                status[str(key)] = value
        eta = self._segment_eta(split=split, step=step)
        return PresentationUpdate(
            split=split,
            step=step,
            total_steps=int(self._total_planned_steps or 0),
            terminal=terminal,
            scalars=scalars,
            status=status,
            eta_seconds=eta,
        )

    def _segment_eta(self, *, split: str, step: int) -> float | None:
        """Approximate remaining time from THIS segment's observed rate."""

        if split != "train" or self._total_planned_steps is None:
            return None
        now = float(self._monotonic())
        if self._segment_started_monotonic is None or self._segment_first_step is None:
            self._segment_started_monotonic = now
            self._segment_first_step = step
            return None
        completed = step - int(self._segment_first_step)
        elapsed = now - float(self._segment_started_monotonic)
        if completed <= 0 or elapsed <= 0.0:
            return None
        remaining = int(self._total_planned_steps) - step
        if remaining <= 0:
            return 0.0
        return (elapsed / float(completed)) * float(remaining)

    # -- console ----------------------------------------------------------

    def _render_console(self, update: PresentationUpdate) -> None:
        if self._console_disabled:
            return
        stream = sys.stderr if self._console is None else self._console
        try:
            stream.write(self._console_line(update) + "\n")
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()
        except BaseException:
            # Presentation is derived: a broken console never fails training.
            self._console_disabled = True

    def _console_line(self, update: PresentationUpdate) -> str:
        parts = [f"{update.split} {update.step}/{update.total_steps}"]
        scalars = update.scalars
        loss = scalars.get("loss/total")
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        learning_rates = sorted(
            key for key in scalars if key.startswith("lr/group_")
        )
        if learning_rates:
            parts.append(f"lr {scalars[learning_rates[0]]:.2e}")
        norm = scalars.get("grad_norm/pre_clip_rank_max")
        if norm is not None:
            parts.append(f"grad_norm {norm:.3f}")
        tokens = scalars.get("throughput/physical_tokens_per_second")
        if tokens is not None:
            parts.append(f"tok/s {tokens:,.0f}")
        memory = scalars.get("resource/gpu_current_memory_allocated_bytes")
        if memory is None:
            memory = scalars.get("resource/gpu_max_memory_allocated_bytes")
        if memory is not None:
            parts.append(f"mem {_format_bytes(memory)}")
        duration = scalars.get("step_duration_seconds")
        if duration is not None:
            parts.append(f"{duration:.3f}s/step")
        eval_duration = scalars.get("eval_duration_seconds")
        if eval_duration is not None:
            parts.append(f"eval {eval_duration:.2f}s")
        statuses = [
            update.status[key]
            for key in ("optimizer_update_status", "finite_status")
            if key in update.status
        ]
        if statuses:
            parts.append("/".join(statuses))
        if update.terminal:
            reason = update.status.get("optimizer_terminal_reason")
            parts.append(f"TERMINAL {reason}" if reason else "TERMINAL")
        if update.eta_seconds is not None:
            parts.append(f"eta ~{_format_seconds(update.eta_seconds)} (approx)")
        return " | ".join(parts)

    # -- tensorboard ------------------------------------------------------

    def _mirror_tensorboard(self, update: PresentationUpdate) -> None:
        if self._tensorboard_disabled:
            return
        board = self._tensorboard
        if board is None:
            assert self._run_dir is not None  # guarded at construction
            try:
                board = self._tensorboard_factory(
                    self._run_dir / TENSORBOARD_DIRECTORY_NAME
                )
            except BaseException as exc:
                self._disable_tensorboard("initialization", exc)
                return
            if board is None:
                self._disable_tensorboard(
                    "initialization", RuntimeError("no event writer was created")
                )
                return
            self._tensorboard = board
        try:
            for key in sorted(update.scalars):
                board.add_scalar(
                    f"{update.split}/{key}", update.scalars[key], update.step
                )
            board.flush()
        except BaseException as exc:
            self._disable_tensorboard("write", exc)

    def _disable_tensorboard(self, stage: str, exc: BaseException) -> None:
        if self._tensorboard_disabled:
            return
        board = self._tensorboard
        # LATCH FIRST: the best-effort cleanup below cannot re-enter this path.
        self._tensorboard_disabled = True
        self._tensorboard = None
        self._warn_tensorboard_once(stage, exc)
        if board is None:
            return
        try:
            board.close()
        except BaseException:
            return

    def _warn_tensorboard_once(self, stage: str, exc: BaseException) -> None:
        if self._tensorboard_warned:
            return
        self._tensorboard_warned = True
        writer = self._writer
        if writer is not None:
            record_warning = getattr(writer, "record_warning", None)
            if callable(record_warning):
                try:
                    record_warning(TENSORBOARD_DISABLED_WARNING)
                except BaseException:
                    pass
        try:
            # Bounded: the exception TYPE only, never arbitrary error text.
            sys.stderr.write(
                "coordexp-swift: derived tensorboard sink disabled after a "
                f"{stage} failure ({type(exc).__name__}); the canonical "
                "logging.jsonl stream is unaffected\n"
            )
        except BaseException:
            pass
