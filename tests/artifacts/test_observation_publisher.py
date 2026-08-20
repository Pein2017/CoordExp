"""Wave-4 contract for ``src/artifacts/observation_publisher.py``.

``add-coordexp-swift-training-observability`` (Rank-Zero Presentation Sinks)
makes console and TensorBoard DERIVED rank-zero presentations of the canonical
``logging.jsonl`` observation, never independent metric authorities:

* the canonical row is appended and its bounded outcome shared with every rank
  BEFORE either derived sink is consulted;
* a train row is presented at positive multiples of ``observability.steps``, at
  the resolved terminal planned step, and for a successfully published terminal
  optimizer-boundary row; every completed eval invocation is presented;
* every presentation payload carries the planned ``step`` and the resolved total
  planned steps, and the console renders current step over total;
* the approximate ETA is segment-local, in memory only, and never enters a row,
  a TensorBoard tag, or any artifact;
* TensorBoard import/initialization/``add_scalar``/``flush``/``close`` failures
  share ONE one-way disabled latch, emit at most one bounded run warning plus
  one best-effort stderr warning, cannot recurse through cleanup, and never
  invalidate the already published canonical row.

The publisher is one direct publication interface. There is no dispatcher, no
subscription table, no dynamic sink registry, and no alternate scalar authority.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.artifacts import observation_publisher
from src.artifacts.observation_publisher import ObservationPublisher
from src.artifacts.run_writer import RunWriter
from src.common.errors import ArtifactContractError, RuntimeContractError


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _Accelerator:
    def __init__(self, *, is_main_process: bool = True, num_processes: int = 1) -> None:
        self.is_main_process = is_main_process
        self.num_processes = num_processes


class _Runtime:
    """Single-rank runtime double: no broadcast is required at world size one."""

    def __init__(self, *, is_main_process: bool = True) -> None:
        self.is_main_process = is_main_process
        self.world_size = 1
        self.accelerator = _Accelerator(is_main_process=is_main_process)


class _RecordingTensorBoard:
    """A SummaryWriter-shaped double that records what the sink asked of it."""

    def __init__(self, log_dir: Path, *, witness: Any = None) -> None:
        self.log_dir = Path(log_dir)
        self.scalars: list[tuple[str, float, int]] = []
        self.flushes = 0
        self.closes = 0
        self._witness = witness
        self.witnessed_rows: list[int] = []

    def add_scalar(self, tag: str, value: Any, global_step: Any = None) -> None:
        if self._witness is not None:
            self.witnessed_rows.append(self._witness())
        self.scalars.append((tag, value, global_step))

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closes += 1

    @property
    def tags(self) -> list[str]:
        return [tag for tag, _, _ in self.scalars]


class _WitnessStream(io.StringIO):
    """A console stream that records the canonical row count at write time."""

    def __init__(self, witness: Any) -> None:
        super().__init__()
        self._witness = witness
        self.witnessed_rows: list[int] = []

    def write(self, text: str) -> int:  # type: ignore[override]
        if text.strip():
            self.witnessed_rows.append(self._witness())
        return super().write(text)


def _writer(tmp_path: Path, *, resolved_max_steps: int = 10) -> RunWriter:
    return RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="fp",
        resolved_config={},
        world_size=1,
        resolved_max_steps=resolved_max_steps,
    )


def _train_row(step: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": step,
        "split": "train",
        "micro_step_count": 2,
        "optimizer_update_status": "applied",
        "finite_status": "finite",
        "loss/total": 1.25,
        "lr/group_0": 1e-05,
        "grad_norm/pre_clip_rank_max": 0.5,
        "step_duration_seconds": 0.25,
        "throughput/physical_tokens_per_second": 4096.0,
        "resource/gpu_current_memory_allocated_bytes": 1073741824.0,
    }
    row.update(overrides)
    return row


def _eval_row(step: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": step,
        "split": "eval",
        "trigger_reasons": ["interval"],
        "example_count": 8,
        "pack_count": 2,
        "loss/total": 0.75,
        "eval_duration_seconds": 3.0,
    }
    row.update(overrides)
    return row


def _rows(writer: RunWriter) -> list[dict[str, Any]]:
    path = writer.logging_path
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _line_counter(writer: RunWriter) -> Any:
    def count() -> int:
        return len(_rows(writer))

    return count


def _publisher(
    tmp_path: Path,
    *,
    writer: RunWriter | None = None,
    steps: int = 2,
    total: int | None = 10,
    console: Any = None,
    tensorboard_factory: Any = None,
    runtime: Any = None,
    monotonic: Any = None,
) -> ObservationPublisher:
    resolved_writer = _writer(tmp_path) if writer is None else writer
    return ObservationPublisher(
        writer=resolved_writer,
        runtime=_Runtime() if runtime is None else runtime,
        run_dir=resolved_writer.run_dir,
        presentation_steps=steps,
        total_planned_steps=total,
        console=console,
        tensorboard_factory=tensorboard_factory,
        monotonic=monotonic,
    )


# ---------------------------------------------------------------------------
# One publication interface, not a dispatcher
# ---------------------------------------------------------------------------


def test_publisher_exposes_exactly_one_publication_interface() -> None:
    public = {
        name for name in dir(ObservationPublisher) if not name.startswith("_")
    }

    assert public == {"publish", "close"}


def test_publisher_module_owns_no_event_dispatcher_surface() -> None:
    names = {name.lower() for name in dir(observation_publisher)}
    forbidden = {
        "subscribe",
        "unsubscribe",
        "emit",
        "dispatch",
        "register_sink",
        "eventemitter",
        "sinks",
        "handlers",
        "listeners",
    }

    assert forbidden.isdisjoint(names)
    source = Path(observation_publisher.__file__).read_text(encoding="utf-8")
    for token in ("wandb", "sqlite", "scalars.json", "metrics.jsonl"):
        assert token not in source


def test_presentation_requires_a_resolved_total_planned_step_count(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)

    with pytest.raises(ArtifactContractError) as excinfo:
        ObservationPublisher(
            writer=writer,
            runtime=_Runtime(),
            run_dir=writer.run_dir,
            presentation_steps=2,
            total_planned_steps=None,
        )

    assert excinfo.value.code == "observation_publisher.presentation_total_missing"


@pytest.mark.parametrize("steps", [0, -1])
def test_presentation_interval_must_be_positive(tmp_path: Path, steps: int) -> None:
    writer = _writer(tmp_path)

    with pytest.raises(ArtifactContractError) as excinfo:
        ObservationPublisher(
            writer=writer,
            runtime=_Runtime(),
            run_dir=writer.run_dir,
            presentation_steps=steps,
            total_planned_steps=10,
        )

    assert excinfo.value.code == "observation_publisher.presentation_interval_invalid"


# ---------------------------------------------------------------------------
# Required `step` and the resolved total in every presentation payload
# ---------------------------------------------------------------------------


def test_every_presentation_update_carries_step_and_resolved_total(
    tmp_path: Path,
) -> None:
    console = io.StringIO()
    boards: list[_RecordingTensorBoard] = []
    publisher = _publisher(
        tmp_path,
        steps=1,
        total=10,
        console=console,
        tensorboard_factory=lambda log_dir: boards.append(
            _RecordingTensorBoard(log_dir)
        )
        or boards[-1],
    )

    publisher.publish(_train_row(3))
    publisher.publish(_eval_row(3))

    updates = publisher._presented  # type: ignore[attr-defined]
    assert [update.step for update in updates] == [3, 3]
    assert [update.total_steps for update in updates] == [10, 10]
    assert [update.split for update in updates] == ["train", "eval"]


def test_a_row_without_a_planned_step_is_rejected_before_presentation(
    tmp_path: Path,
) -> None:
    console = io.StringIO()
    board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = _publisher(
        tmp_path, steps=1, console=console, tensorboard_factory=lambda _dir: board
    )
    row = _train_row(1)
    row.pop("step")

    with pytest.raises(ArtifactContractError) as excinfo:
        publisher.publish(row)

    assert excinfo.value.code == "observation_publisher.presentation_step_missing"
    assert console.getvalue() == ""
    assert board.scalars == []


# ---------------------------------------------------------------------------
# JSONL-first ordering
# ---------------------------------------------------------------------------


def test_the_canonical_row_is_readable_before_any_presentation_call(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    witness = _line_counter(writer)
    console = _WitnessStream(witness)
    board = _RecordingTensorBoard(tmp_path / "tb", witness=witness)
    publisher = _publisher(
        tmp_path,
        writer=writer,
        steps=1,
        console=console,
        tensorboard_factory=lambda _dir: board,
    )

    publisher.publish(_train_row(1))

    # Both derived sinks observed the canonical row already on disk.
    assert console.witnessed_rows == [1]
    assert set(board.witnessed_rows) == {1}
    assert _rows(writer)[0]["step"] == 1


def test_failed_canonical_publication_presents_nothing(tmp_path: Path) -> None:
    failing_writer = SimpleNamespace(
        run_dir=tmp_path / "run",
        append_logging_row=lambda row: (_ for _ in ()).throw(OSError("disk full")),
    )
    console = io.StringIO()
    board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = ObservationPublisher(
        writer=failing_writer,
        runtime=_Runtime(),
        run_dir=tmp_path / "run",
        presentation_steps=1,
        total_planned_steps=10,
        console=console,
        tensorboard_factory=lambda _dir: board,
    )

    with pytest.raises(RuntimeContractError) as excinfo:
        publisher.publish(_train_row(1))

    assert excinfo.value.code == "runtime.logging_append_failed"
    assert console.getvalue() == ""
    assert board.scalars == []


def test_publication_never_mutates_or_extends_the_canonical_row(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    console = io.StringIO()
    board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = _publisher(
        tmp_path,
        writer=writer,
        steps=1,
        console=console,
        tensorboard_factory=lambda _dir: board,
    )
    row = _train_row(1)
    before = json.dumps(row, sort_keys=True)

    publisher.publish(row)

    assert json.dumps(row, sort_keys=True) == before
    published = _rows(writer)[0]
    # The writer's own canonical `non_finite_fields` binding is the only key the
    # published row may gain; the publisher itself adds nothing.
    assert set(published) - set(row) == {"non_finite_fields"}
    assert published["non_finite_fields"] == []
    assert {key: published[key] for key in row} == row
    # No presentation state (ETA, cadence, sink health) enters the artifact.
    for forbidden in ("eta", "console", "tensorboard", "presentation"):
        assert not any(forbidden in key for key in published)


def test_presentation_time_is_not_added_to_the_step_duration(
    tmp_path: Path,
) -> None:
    """The publisher observes a completed row; it never writes a timing field."""

    writer = _writer(tmp_path)
    slow_board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = _publisher(
        tmp_path,
        writer=writer,
        steps=1,
        console=io.StringIO(),
        tensorboard_factory=lambda _dir: slow_board,
    )
    row = _train_row(1, step_duration_seconds=0.25)

    publisher.publish(row)

    assert _rows(writer)[0]["step_duration_seconds"] == 0.25


# ---------------------------------------------------------------------------
# Train interval, terminal, and unconditional eval dispatch
# ---------------------------------------------------------------------------


def test_train_rows_are_presented_only_on_the_configured_interval(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    console = io.StringIO()
    board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = _publisher(
        tmp_path,
        writer=writer,
        steps=2,
        total=10,
        console=console,
        tensorboard_factory=lambda _dir: board,
    )

    for step in (1, 2, 3, 4):
        publisher.publish(_train_row(step))

    assert [update.step for update in publisher._presented] == [2, 4]  # type: ignore[attr-defined]
    # Every canonical row is appended regardless of the presentation cadence.
    assert [row["step"] for row in _rows(writer)] == [1, 2, 3, 4]
    assert {step for _, _, step in board.scalars} == {2, 4}


def test_the_resolved_terminal_step_is_presented_off_cadence(
    tmp_path: Path,
) -> None:
    publisher = _publisher(
        tmp_path, steps=4, total=5, console=io.StringIO(),
        tensorboard_factory=lambda log_dir: _RecordingTensorBoard(log_dir),
    )

    for step in (3, 4, 5):
        publisher.publish(_train_row(step))

    assert [update.step for update in publisher._presented] == [4, 5]  # type: ignore[attr-defined]


def test_a_terminal_boundary_row_is_presented_off_cadence(tmp_path: Path) -> None:
    console = io.StringIO()
    publisher = _publisher(tmp_path, steps=4, total=10, console=console)
    row = _train_row(
        7,
        optimizer_update_status="not_attempted",
        finite_status="unavailable",
        optimizer_boundary_terminal=True,
        optimizer_terminal_reason="pre_wrapper_mixed_scaler_overflow",
    )

    publisher.publish(row, terminal=True)

    assert [update.step for update in publisher._presented] == [7]  # type: ignore[attr-defined]
    assert "7/10" in console.getvalue()


def test_every_eval_row_is_mirrored_regardless_of_cadence(tmp_path: Path) -> None:
    board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = _publisher(
        tmp_path,
        steps=100,
        total=1000,
        console=io.StringIO(),
        tensorboard_factory=lambda _dir: board,
    )

    publisher.publish(_eval_row(3))
    publisher.publish(_eval_row(7))

    assert [update.step for update in publisher._presented] == [3, 7]  # type: ignore[attr-defined]
    assert [
        (tag, step) for tag, _, step in board.scalars if tag == "eval/loss/total"
    ] == [("eval/loss/total", 3), ("eval/loss/total", 7)]
    assert all(tag.startswith("eval/") for tag in board.tags)
    assert "eval/step" not in board.tags
    assert "eval/split" not in board.tags


# ---------------------------------------------------------------------------
# Rank-zero ownership
# ---------------------------------------------------------------------------


def test_a_non_main_rank_prints_nothing_and_creates_no_event_file(
    tmp_path: Path,
) -> None:
    console = io.StringIO()
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    publisher = ObservationPublisher(
        writer=None,
        runtime=_Runtime(is_main_process=False),
        run_dir=run_dir,
        presentation_steps=1,
        total_planned_steps=10,
        console=console,
    )

    publisher.publish(_train_row(1))
    publisher.publish(_eval_row(1))
    publisher.close()

    assert console.getvalue() == ""
    assert publisher._presented == []  # type: ignore[attr-defined]
    assert not (run_dir / "tensorboard").exists()


# ---------------------------------------------------------------------------
# Console presentation and the approximate segment-local ETA
# ---------------------------------------------------------------------------


def test_console_renders_current_step_over_total_with_compact_fields(
    tmp_path: Path,
) -> None:
    console = io.StringIO()
    publisher = _publisher(tmp_path, steps=1, total=100, console=console)

    publisher.publish(_train_row(7))

    (line,) = [item for item in console.getvalue().splitlines() if item.strip()]
    assert "train" in line
    assert "7/100" in line
    assert "1.25" in line  # loss/total
    assert "1.0" in line or "1e-05" in line  # lr/group_0
    assert "applied" in line
    assert "finite" in line


def test_eta_is_segment_local_approximate_and_never_persisted(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    console = io.StringIO()
    board = _RecordingTensorBoard(tmp_path / "tb")
    clock = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    publisher = _publisher(
        tmp_path,
        writer=writer,
        steps=1,
        total=10,
        console=console,
        tensorboard_factory=lambda _dir: board,
        monotonic=lambda: next(clock),
    )

    publisher.publish(_train_row(1))
    publisher.publish(_train_row(2))

    updates = publisher._presented  # type: ignore[attr-defined]
    # The first presented step of the segment has no rate yet.
    assert updates[0].eta_seconds is None
    # One segment-local second per completed planned step, eight steps left.
    assert updates[1].eta_seconds == pytest.approx(8.0, rel=0.5)
    assert "~" in console.getvalue()
    for row in _rows(writer):
        assert not any("eta" in key for key in row)
    assert not any("eta" in tag for tag in board.tags)


def test_eta_is_discarded_rather_than_carried_across_a_resume(
    tmp_path: Path,
) -> None:
    """A fresh publisher starts a fresh segment: no ETA state is restored."""

    clock = iter([100.0, 100.5])
    publisher = _publisher(
        tmp_path,
        steps=1,
        total=1000,
        console=io.StringIO(),
        monotonic=lambda: next(clock),
    )

    publisher.publish(_train_row(500))

    assert publisher._presented[0].eta_seconds is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TensorBoard projection and the real event reader
# ---------------------------------------------------------------------------


def test_tensorboard_tags_mirror_finite_numeric_row_fields_only(
    tmp_path: Path,
) -> None:
    board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = _publisher(
        tmp_path,
        steps=1,
        console=io.StringIO(),
        tensorboard_factory=lambda _dir: board,
    )
    row = _train_row(
        2,
        **{
            "accuracy_stats": {"top1_correct": 1},
            "unavailable_fields": ["lr/group_1"],
            "optimizer_update_applied": True,
            "lr/group_1": None,
            "loss/aux": float("inf"),
        },
    )

    publisher.publish(row)

    tags = set(board.tags)
    assert "train/loss/total" in tags
    assert "train/grad_norm/pre_clip_rank_max" in tags
    # Strings, nulls, nested diagnostics, booleans, and non-finite values are
    # never synthesized into a scalar tag.
    assert "train/optimizer_update_status" not in tags
    assert "train/finite_status" not in tags
    assert "train/accuracy_stats" not in tags
    assert "train/unavailable_fields" not in tags
    assert "train/optimizer_update_applied" not in tags
    assert "train/lr/group_1" not in tags
    assert "train/loss/aux" not in tags
    assert all(math.isfinite(value) for _, value, _ in board.scalars)
    assert {step for _, _, step in board.scalars} == {2}


def test_tensorboard_writer_is_created_lazily_under_the_run_directory(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    created: list[Path] = []

    def factory(log_dir: Path) -> _RecordingTensorBoard:
        created.append(Path(log_dir))
        return _RecordingTensorBoard(log_dir)

    publisher = _publisher(
        tmp_path,
        writer=writer,
        steps=2,
        console=io.StringIO(),
        tensorboard_factory=factory,
    )

    publisher.publish(_train_row(1))
    assert created == []

    publisher.publish(_train_row(2))
    assert created == [writer.run_dir / "tensorboard"]

    publisher.publish(_train_row(4))
    assert created == [writer.run_dir / "tensorboard"]


def test_terminal_close_flushes_and_closes_the_event_writer(
    tmp_path: Path,
) -> None:
    board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = _publisher(
        tmp_path, steps=1, console=io.StringIO(), tensorboard_factory=lambda _d: board
    )

    publisher.publish(_train_row(1))
    publisher.close()
    publisher.close()

    assert board.closes == 1


def test_a_real_event_reader_recovers_tags_values_and_global_steps(
    tmp_path: Path,
) -> None:
    """Task 4.5: a standard TensorBoard reader over a real run directory."""

    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    writer = _writer(tmp_path)
    publisher = _publisher(
        tmp_path, writer=writer, steps=2, total=4, console=io.StringIO()
    )

    publisher.publish(_train_row(2, **{"loss/total": 1.5}))
    publisher.publish(_eval_row(2, **{"loss/total": 0.5}))
    publisher.publish(_train_row(4, **{"loss/total": 1.0}))
    publisher.close()

    log_dir = writer.run_dir / "tensorboard"
    assert log_dir.is_dir()
    accumulator = event_accumulator.EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])

    assert "train/loss/total" in tags
    assert "eval/loss/total" in tags
    train_loss = accumulator.Scalars("train/loss/total")
    assert [event.step for event in train_loss] == [2, 4]
    assert [round(event.value, 4) for event in train_loss] == [1.5, 1.0]
    assert all(math.isfinite(event.value) for event in train_loss)
    (eval_loss,) = accumulator.Scalars("eval/loss/total")
    assert eval_loss.step == 2
    assert round(eval_loss.value, 4) == 0.5


# ---------------------------------------------------------------------------
# One-way TensorBoard failure latch (task 4.6)
# ---------------------------------------------------------------------------


class _CountingWriter:
    """A run writer double that counts bounded warnings and appended rows."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logging_path = run_dir / "logging.jsonl"
        self.warnings: list[str] = []
        self.rows: list[dict[str, Any]] = []

    def append_logging_row(self, row: Any) -> Path:
        self.rows.append(dict(row))
        with self.logging_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        return self.logging_path

    def record_warning(self, code: str, *, count: int = 1) -> None:
        self.warnings.append(code)


def _failure_publisher(
    tmp_path: Path, factory: Any, *, console: Any = None
) -> tuple[ObservationPublisher, _CountingWriter]:
    writer = _CountingWriter(tmp_path / "run")
    publisher = ObservationPublisher(
        writer=writer,
        runtime=_Runtime(),
        run_dir=writer.run_dir,
        presentation_steps=1,
        total_planned_steps=10,
        console=io.StringIO() if console is None else console,
        tensorboard_factory=factory,
    )
    return publisher, writer


def _boom(exc: BaseException) -> Any:
    def raise_it(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return raise_it


@pytest.mark.parametrize(
    ("arm", "error"),
    [
        ("import", ImportError("no module named tensorboard")),
        ("initialization", RuntimeError("log dir is not writable")),
    ],
)
def test_a_failed_tensorboard_start_keeps_the_canonical_row(
    tmp_path: Path, arm: str, error: BaseException, capsys: pytest.CaptureFixture[str]
) -> None:
    factory_calls: list[Path] = []

    def factory(log_dir: Path) -> Any:
        factory_calls.append(Path(log_dir))
        raise error

    publisher, writer = _failure_publisher(tmp_path, factory)

    publisher.publish(_train_row(1))
    publisher.publish(_train_row(2))
    publisher.publish(_eval_row(2))
    publisher.close()

    # The canonical stream is untouched by the derived sink's failure.
    assert [row["step"] for row in writer.rows] == [1, 2, 2]
    # One bounded run warning with a stable code for the whole run.
    assert writer.warnings == [observation_publisher.TENSORBOARD_DISABLED_WARNING]
    # The sink latched off: no later method call, including a second start.
    assert len(factory_calls) == 1
    captured = capsys.readouterr()
    assert captured.err.count("tensorboard") == 1


def test_an_add_scalar_failure_latches_the_sink_and_keeps_later_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Board(_RecordingTensorBoard):
        def add_scalar(self, tag: str, value: Any, global_step: Any = None) -> None:
            self.scalars.append((tag, value, global_step))
            raise RuntimeError("event file writer is gone")

    board = _Board(tmp_path / "tb")
    publisher, writer = _failure_publisher(tmp_path, lambda _dir: board)

    publisher.publish(_train_row(1))
    publisher.publish(_train_row(2))
    publisher.publish(_eval_row(2))
    publisher.close()

    assert [row["step"] for row in writer.rows] == [1, 2, 2]
    assert writer.warnings == [observation_publisher.TENSORBOARD_DISABLED_WARNING]
    # Exactly one add attempt: the latch stops every later sink call.
    assert len(board.scalars) == 1
    assert board.flushes == 0
    assert board.closes == 1
    assert capsys.readouterr().err.count("tensorboard") == 1


def test_a_flush_failure_after_a_successful_add_keeps_the_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Board(_RecordingTensorBoard):
        def flush(self) -> None:
            self.flushes += 1
            raise OSError("no space left on device")

    board = _Board(tmp_path / "tb")
    publisher, writer = _failure_publisher(tmp_path, lambda _dir: board)

    publisher.publish(_train_row(1))
    written = len(board.scalars)
    publisher.publish(_train_row(2))
    publisher.close()

    assert [row["step"] for row in writer.rows] == [1, 2]
    assert writer.warnings == [observation_publisher.TENSORBOARD_DISABLED_WARNING]
    # The adds for step 1 succeeded; the flush failure latched the sink, so the
    # step-2 presentation adds nothing and the terminal close is best effort.
    assert written > 0
    assert len(board.scalars) == written
    assert board.flushes == 1
    assert board.closes == 1
    assert capsys.readouterr().err.count("tensorboard") == 1


def test_a_close_failure_cannot_recurse_or_emit_a_second_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Board(_RecordingTensorBoard):
        def close(self) -> None:
            self.closes += 1
            raise RuntimeError("close failed")

    board = _Board(tmp_path / "tb")
    publisher, writer = _failure_publisher(tmp_path, lambda _dir: board)

    publisher.publish(_train_row(1))
    publisher.close()
    publisher.close()

    assert [row["step"] for row in writer.rows] == [1]
    # A terminal close failure disables the sink; it warns exactly once and it
    # cannot recurse into another close.
    assert writer.warnings == [observation_publisher.TENSORBOARD_DISABLED_WARNING]
    assert board.closes == 1
    assert capsys.readouterr().err.count("tensorboard") == 1


def test_a_cleanup_failure_after_a_write_failure_warns_exactly_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Board(_RecordingTensorBoard):
        def add_scalar(self, tag: str, value: Any, global_step: Any = None) -> None:
            raise RuntimeError("event file writer is gone")

        def close(self) -> None:
            self.closes += 1
            raise RuntimeError("close failed too")

    board = _Board(tmp_path / "tb")
    publisher, writer = _failure_publisher(tmp_path, lambda _dir: board)

    publisher.publish(_train_row(1))
    publisher.publish(_train_row(2))
    publisher.close()

    assert [row["step"] for row in writer.rows] == [1, 2]
    assert writer.warnings == [observation_publisher.TENSORBOARD_DISABLED_WARNING]
    assert board.closes == 1
    assert capsys.readouterr().err.count("tensorboard") == 1


def test_a_failing_run_warning_never_breaks_canonical_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class _HostileWarningWriter(_CountingWriter):
        def record_warning(self, code: str, *, count: int = 1) -> None:
            raise OSError("run.json is read-only")

    writer = _HostileWarningWriter(tmp_path / "run")
    publisher = ObservationPublisher(
        writer=writer,
        runtime=_Runtime(),
        run_dir=writer.run_dir,
        presentation_steps=1,
        total_planned_steps=10,
        console=io.StringIO(),
        tensorboard_factory=_boom(RuntimeError("initialization failed")),
    )

    publisher.publish(_train_row(1))
    publisher.publish(_train_row(2))

    assert [row["step"] for row in writer.rows] == [1, 2]


# ---------------------------------------------------------------------------
# Bounded diagnostic name lists (task 4.6)
# ---------------------------------------------------------------------------


def test_unavailable_fields_are_sorted_unique_and_bounded(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    names = [f"metric/{index:04d}" for index in range(300)]
    publisher = _publisher(tmp_path, writer=writer, steps=1, console=io.StringIO())

    publisher.publish(
        _train_row(1, unavailable_fields=list(reversed(names)) + [names[0]])
    )

    row = _rows(writer)[0]
    assert row["unavailable_fields"] == sorted(names)[:256]
    assert len(row["unavailable_fields"]) == 256
    assert row["unavailable_fields_truncated_count"] == 44
    assert len(set(row["unavailable_fields"])) == 256


def test_non_finite_fields_are_sorted_unique_and_bounded(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    names = [f"loss/{index:04d}" for index in range(300)]
    publisher = _publisher(tmp_path, writer=writer, steps=1, console=io.StringIO())

    publisher.publish(_train_row(1, non_finite_fields=list(reversed(names))))

    row = _rows(writer)[0]
    assert row["non_finite_fields"] == sorted(names)[:256]
    assert row["non_finite_fields_truncated_count"] == 44


def test_a_normal_row_carries_no_truncation_counter(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    publisher = _publisher(tmp_path, writer=writer, steps=1, console=io.StringIO())

    publisher.publish(_train_row(1, unavailable_fields=["b", "a", "a"]))

    row = _rows(writer)[0]
    assert row["unavailable_fields"] == ["a", "b"]
    assert "unavailable_fields_truncated_count" not in row
    assert "non_finite_fields_truncated_count" not in row


def test_a_diagnostic_name_over_the_byte_bound_is_rejected_without_error_text(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)

    with pytest.raises(ArtifactContractError) as excinfo:
        writer.append_logging_row(_train_row(1, unavailable_fields=["x" * 257]))

    assert excinfo.value.code == "run_writer.invalid_unavailable_fields"
    assert excinfo.value.context["max_name_bytes"] == 256
    assert _rows(writer) == []


def test_a_row_rejected_by_the_canonical_writer_is_never_presented(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    console = io.StringIO()
    board = _RecordingTensorBoard(tmp_path / "tb")
    publisher = _publisher(
        tmp_path,
        writer=writer,
        steps=1,
        console=console,
        tensorboard_factory=lambda _dir: board,
    )

    with pytest.raises(RuntimeContractError) as excinfo:
        publisher.publish(_train_row(1, unavailable_fields=["x" * 257]))

    # The bounded rank-zero append failure stays the published outcome.
    assert excinfo.value.code == "runtime.logging_append_failed"
    assert "run_writer.invalid_unavailable_fields" in str(excinfo.value)
    assert _rows(writer) == []
    assert console.getvalue() == ""
    assert board.scalars == []


def test_publish_returns_none_and_needs_no_sink_configuration(
    tmp_path: Path,
) -> None:
    """A publication-only publisher is the frozen-characterization fallback."""

    writer = _writer(tmp_path)
    publisher = ObservationPublisher(writer=writer, runtime=_Runtime())

    assert publisher.publish(_train_row(1)) is None
    assert [row["step"] for row in _rows(writer)] == [1]
    assert not (writer.run_dir / "tensorboard").exists()


def test_a_missing_step_fails_closed_on_every_rank_before_the_handshake(
    tmp_path: Path,
) -> None:
    """A rank-local raise in front of the shared handshake would desync it."""

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    peer = ObservationPublisher(
        writer=None,
        runtime=_Runtime(is_main_process=False),
        run_dir=run_dir,
        presentation_steps=1,
        total_planned_steps=10,
        console=io.StringIO(),
    )
    row = _train_row(1)
    row.pop("step")

    with pytest.raises(ArtifactContractError) as excinfo:
        peer.publish(row)

    assert excinfo.value.code == "observation_publisher.presentation_step_missing"
