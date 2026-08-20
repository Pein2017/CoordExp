"""Wave-3 tasks 3.1/3.2: the boundary consensus over a REAL two-rank group.

The single-process tests use a gatherer double, which can only prove that the
reducer is a pure function of the reports it is handed. These tests run two
real `gloo` processes over the production
`src.training.control_plane._build_rank_report_gatherer`, so the properties
under test are the distributed ones:

* both ranks select the SAME closed action from genuinely asymmetric rank-local
  fp16 state, before either enters the wrapper;
* a mixed pre-wrapper candidacy terminates on BOTH ranks (the rank whose own
  gradients are perfectly finite must not proceed alone);
* a post-wrapper mixed outcome terminates on BOTH ranks with the same code, and
  neither rank raises from its own skip flag before the consensus completes.

fp16 itself is duck-typed on CPU (the semantics under test are order and
consensus, not CUDA arithmetic); genuine CUDA fp16 arms are task 3.8's probes.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import multiprocessing as mp
from queue import Empty
import socket
import traceback
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime.optimizer_boundary import OptimizerBoundaryTerminal
from src.runtime.train_runtime import TrainRuntime

_WORLD_SIZE = 2
_PROCESS_GROUP_TIMEOUT_SECONDS = 30
_JOIN_TIMEOUT_SECONDS = 240
_LEARNING_RATE = 0.1


class _Fp16Accelerator:
    """The bounded fp16 surface `TrainRuntime` reads, over a real rank."""

    def __init__(self, *, rank: int, found_inf: bool, step_was_skipped: bool) -> None:
        self.process_index = rank
        self.num_processes = _WORLD_SIZE
        self.device = torch.device("cpu")
        self.is_main_process = rank == 0
        self.distributed_type = SimpleNamespace(name="MULTI_GPU")
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "fp16"
        self.scaler = _Fp16Scaler(found_inf=found_inf)
        self.unscale_calls: list[Any] = []
        self.optimizer_step_was_skipped = bool(step_was_skipped)

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        return tuple(objects)

    @contextmanager
    def no_sync(self, model: Any) -> Any:
        yield

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def unscale_gradients(self, optimizer: Any = None) -> None:
        self.unscale_calls.append(optimizer)

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> None:
        raise AssertionError("the Accelerate clipping helper is prohibited")


class _Fp16Scaler:
    def __init__(self, *, found_inf: bool) -> None:
        self.found_inf = found_inf

    def _found_inf_per_device(self, optimizer: Any = None) -> dict[str, torch.Tensor]:
        return {"cpu": torch.tensor(1.0 if self.found_inf else 0.0)}


class _SuppressibleOptimizer(torch.optim.SGD):
    def __init__(self, parameters: Any, *, suppress: bool) -> None:
        super().__init__(parameters, lr=_LEARNING_RATE)
        self.suppress = suppress
        self.step_calls = 0

    def step(self, closure: Any | None = None) -> Any:
        self.step_calls += 1
        if self.suppress:
            return None
        return super().step(closure)


def _init_process_group(rank: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
    )


def _build_runtime(
    *,
    rank: int,
    gatherer: Any,
    found_inf: bool,
    grad_value: float,
    suppress_update: bool,
    step_was_skipped: bool,
) -> tuple[TrainRuntime, _Fp16Accelerator]:
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1, bias=False)
    accelerator = _Fp16Accelerator(
        rank=rank, found_inf=found_inf, step_was_skipped=step_was_skipped
    )
    optimizer = _SuppressibleOptimizer(model.parameters(), suppress=suppress_update)
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig.model_validate(
            {"seed": 17, "determinism": {"mode": "legacy"}}
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=_WORLD_SIZE,
            effective_batch_size=_WORLD_SIZE,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        expected_mixed_precision="fp16",
        max_grad_norm=1.0,
        accelerator=accelerator,
        rank_report_gatherer=gatherer,
    )
    for parameter in runtime.model.parameters():
        parameter.grad = torch.full_like(parameter, grad_value)
    return runtime, accelerator


def _with_gatherer(worker: Any, rank: int, port: int, output: mp.Queue) -> None:
    from src.training.control_plane import _build_rank_report_gatherer

    try:
        _init_process_group(rank, port)
        gatherer = _build_rank_report_gatherer(_WORLD_SIZE)
        assert gatherer is not None
        try:
            worker(rank, gatherer)
        finally:
            close = getattr(gatherer, "close", None)
            if callable(close):
                close()
        output.put((rank, "ok"))
    except BaseException:  # pragma: no cover - reported through the queue
        output.put((rank, traceback.format_exc()))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _uniform_overflow(rank: int, gatherer: Any) -> None:
    runtime, accelerator = _build_runtime(
        rank=rank,
        gatherer=gatherer,
        found_inf=True,
        grad_value=float("inf"),
        suppress_update=True,
        step_was_skipped=False,
    )
    before = [parameter.detach().clone() for parameter in runtime.model.parameters()]
    decision = runtime.post_backward(planned_step_id=1)
    assert decision.optimizer_boundary_action == "scaler_skip", decision
    assert decision.terminal_reason is None
    assert len(accelerator.unscale_calls) == 1
    accelerator.optimizer_step_was_skipped = True
    receipt = runtime.execute_optimizer_boundary(decision, planned_step_id=1)
    assert receipt.post_wrapper_outcome == "all_skipped"
    assert receipt.terminal is False
    assert receipt.applied is False
    assert receipt.step_was_skipped is True
    assert receipt.group_learning_rates == (None,)
    assert runtime.optimizer_step_count == 1
    assert runtime.scheduler_step_count == 0
    assert runtime.optimizer.step_calls == 1
    for parameter, original in zip(runtime.model.parameters(), before, strict=True):
        assert torch.equal(parameter.detach(), original)


def _mixed_candidacy(rank: int, gatherer: Any) -> None:
    # Rank 0 overflowed; rank 1's gradients are perfectly finite. Neither rank
    # may act on its own local truth.
    overflowed = rank == 0
    runtime, accelerator = _build_runtime(
        rank=rank,
        gatherer=gatherer,
        found_inf=overflowed,
        grad_value=float("inf") if overflowed else 0.5,
        suppress_update=False,
        step_was_skipped=False,
    )
    decision = runtime.post_backward(planned_step_id=2)
    assert decision.optimizer_boundary_action is None, decision
    assert decision.terminal_reason == "pre_wrapper_mixed_scaler_overflow"
    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        runtime.execute_optimizer_boundary(decision, planned_step_id=2)
    receipt = excinfo.value.receipt
    assert receipt.attempted is False
    assert receipt.applied is False
    assert receipt.step_was_skipped is False
    assert receipt.mutation_state == "divergent_or_unknown"
    assert runtime.optimizer.step_calls == 0
    assert runtime.optimizer_step_count == 0
    assert runtime.scheduler_step_count == 0
    assert len(accelerator.unscale_calls) == 1


def _post_wrapper_mixed(rank: int, gatherer: Any) -> None:
    runtime, accelerator = _build_runtime(
        rank=rank,
        gatherer=gatherer,
        found_inf=False,
        grad_value=0.5,
        suppress_update=rank == 0,
        step_was_skipped=False,
    )
    decision = runtime.post_backward(planned_step_id=3)
    assert decision.optimizer_boundary_action == "apply", decision
    # Only rank 0's wrapper reports a skip. Rank 1 must not conclude success
    # and rank 0 must not raise before the consensus.
    accelerator.optimizer_step_was_skipped = rank == 0
    with pytest.raises(OptimizerBoundaryTerminal) as excinfo:
        runtime.execute_optimizer_boundary(decision, planned_step_id=3)
    receipt = excinfo.value.receipt
    assert receipt.terminal_reason == "post_wrapper_mixed"
    assert receipt.attempted is True
    assert receipt.applied is None
    assert receipt.step_was_skipped is None
    assert receipt.group_learning_rates == (None,)
    assert receipt.mutation_state == "divergent_or_unknown"
    # The wrapper DID run on every rank, so the completed-invocation counter
    # moved even though the boundary is terminal.
    assert runtime.optimizer_step_count == 1
    assert runtime.scheduler_step_count == 0


def _uniform_overflow_worker(rank: int, port: int, output: mp.Queue) -> None:
    _with_gatherer(_uniform_overflow, rank, port, output)


def _mixed_candidacy_worker(rank: int, port: int, output: mp.Queue) -> None:
    _with_gatherer(_mixed_candidacy, rank, port, output)


def _post_wrapper_mixed_worker(rank: int, port: int, output: mp.Queue) -> None:
    _with_gatherer(_post_wrapper_mixed, rank, port, output)


def _have_gloo() -> bool:
    if not dist.is_available():
        return False
    available = getattr(dist, "is_gloo_available", None)
    return True if available is None else bool(available())


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _drive_two_ranks(target: Any) -> None:
    context = mp.get_context("spawn")
    port = _find_free_tcp_port()
    output: mp.Queue = context.Queue()
    processes = [
        context.Process(target=target, args=(rank, port, output), daemon=False)
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=_JOIN_TIMEOUT_SECONDS)
        alive = [process for process in processes if process.is_alive()]
        if alive:
            for process in alive:
                process.terminate()
            pytest.fail(
                f"two-rank worker hung; alive_pids={[p.pid for p in alive]}"
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    messages: dict[int, str] = {}
    while True:
        try:
            rank, message = output.get_nowait()
        except Empty:
            break
        messages[int(rank)] = str(message)
    for rank in range(_WORLD_SIZE):
        assert messages.get(rank) == "ok", messages.get(rank, "<no report>")
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_uniform_fp16_overflow_completes_one_scaler_skip() -> None:
    """Task 3.2: unanimous candidacy -> `scaler_skip` + `all_skipped`."""

    _drive_two_ranks(_uniform_overflow_worker)


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_mixed_fp16_candidacy_terminates_on_every_rank() -> None:
    """Task 3.2: the finite rank must not proceed into the wrapper alone."""

    _drive_two_ranks(_mixed_candidacy_worker)


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_post_wrapper_mixed_converges_before_any_rank_raises() -> None:
    """Task 3.2: no rank-local raise from its own post-call skip flag."""

    _drive_two_ranks(_post_wrapper_mixed_worker)
