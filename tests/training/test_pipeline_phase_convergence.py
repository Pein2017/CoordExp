from __future__ import annotations

from datetime import timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import socket
from types import SimpleNamespace
from typing import Any

import pytest
import torch.distributed as dist

import src.training.control_plane as control_plane
import src.training.pipeline as pipeline
from src.artifacts.run_writer import RunWriter
from src.common.errors import RuntimeContractError
from src.training.control_plane import (
    _build_rank_report_gatherer,
    _run_rank_converged_phase,
)
from src.training.pipeline import (
    _begin_run_phase,
    _fail_active_run_phase,
)


_WORLD_SIZE = 2
_PROCESS_GROUP_TIMEOUT_SECONDS = 10
_JOIN_TIMEOUT_SECONDS = 30


def _rank_resource_snapshot(rank: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "cpu": {
            "scope": "current_process",
            "max_rss_bytes": 100 + rank * 50,
            "io_read_bytes": 10 + rank * 20,
            "io_write_bytes": 20 + rank * 5,
        },
        "gpu": {
            "scope": "current_process_current_device",
            "initialized": False,
            "unavailable_reason": "cuda_not_initialized",
        },
    }


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


def _phase_failure_worker(
    rank: int,
    port: int,
    run_dir: str,
    output: mp.Queue,
) -> None:
    gatherer: Any | None = None
    writer = RunWriter(Path(run_dir)) if rank == 0 else None
    lifecycle: dict[str, Any] = {
        "active_phase": None,
        "phase_started_monotonic": None,
        "completed_steps": 0,
        "consumed_packs": 0,
        "checkpoint_event_count": 0,
        "optimizer_update_status": None,
        "finite_status": None,
    }
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        gatherer = _build_rank_report_gatherer(_WORLD_SIZE)
        assert gatherer is not None

        def local_model_surface() -> str:
            _begin_run_phase(writer, lifecycle, "model_loading")
            if rank == 1:
                raise ValueError("peer-only secret-shaped failure detail")
            return "rank-zero-local-success"

        with pytest.raises(RuntimeContractError) as exc_info:
            _run_rank_converged_phase(
                "model_loading",
                rank=rank,
                world_size=_WORLD_SIZE,
                rank_report_gatherer=gatherer,
                body=local_model_surface,
                receipt_sink=pipeline._phase_receipt_sink(lifecycle, "model_loading"),
                resource_collector=lambda: _rank_resource_snapshot(rank),
            )
        assert exc_info.value.code == "runtime.distributed_phase_failed"
        _fail_active_run_phase(writer, lifecycle)
        if writer is not None:
            writer.finalize(
                status="failed",
                updated_at="terminal",
                completed_steps=0,
                consumed_packs=0,
                checkpoint_event_count=0,
                optimizer_update_status=None,
                finite_status=None,
                terminal_error=f"{type(exc_info.value).__name__}: {exc_info.value}",
            )
        output.put(
            (
                rank,
                exc_info.value.code,
                str(exc_info.value),
                tuple(exc_info.value.context["failed_ranks"]),
                None,
            )
        )
    except BaseException as exc:
        output.put((rank, "worker_error", f"{type(exc).__name__}: {exc}", (), None))
        raise
    finally:
        close = getattr(gatherer, "close", None)
        if callable(close):
            close()
        closed = bool(getattr(gatherer, "closed", gatherer is None))
        output.put((rank, "close_status", "", (), closed))
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _eval_hydration_failure_worker(
    rank: int,
    port: int,
    output: mp.Queue,
) -> None:
    gatherer: Any | None = None
    loader_identity: tuple[int, int] | None = None
    metric_started = False
    try:
        os.environ.pop("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", None)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        gatherer = _build_rank_report_gatherer(_WORLD_SIZE)
        assert gatherer is not None

        def load_rank_eval(
            *args: object,
            rank: int,
            world_size: int,
            **kwargs: object,
        ) -> object:
            nonlocal loader_identity
            loader_identity = (rank, world_size)
            if rank == 1:
                raise pipeline.PackingCacheInvalidError(
                    "corrupt payload assigned only to rank one"
                )
            return SimpleNamespace(
                micro_steps=(SimpleNamespace(),),
                canonical_ordinals=(0,),
                total_ordinal_count=2,
            )

        pipeline.load_rank_eval_micro_steps_from_cache = load_rank_eval
        try:
            pipeline._hydrate_eval_micro_steps_from_cache(
                {
                    "cache_dir": Path("/synthetic/eval-cache"),
                    "fingerprint": "eval-fp",
                    "micro_step_count": 2,
                },
                cache_root=Path("/synthetic/cache-root"),
                rank=rank,
                world_size=_WORLD_SIZE,
                rank_report_gatherer=gatherer,
            )
            metric_started = True
            raise AssertionError("corrupt eval hydration unexpectedly succeeded")
        except RuntimeContractError as exc:
            assert exc.code == "runtime.distributed_phase_failed"
            output.put(
                (
                    rank,
                    exc.code,
                    tuple(exc.context["failed_ranks"]),
                    metric_started,
                    loader_identity,
                )
            )
    except BaseException as exc:
        output.put(
            (
                rank,
                "worker_error",
                (),
                metric_started,
                f"{type(exc).__name__}: {exc}",
            )
        )
        raise
    finally:
        close = getattr(gatherer, "close", None)
        if callable(close):
            close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_peer_model_failure_converges_and_closes_bounded_gatherer(
    tmp_path: Path,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="created",
        config_fingerprint="fp",
        resolved_config={},
        world_size=_WORLD_SIZE,
    )
    context = mp.get_context("spawn")
    output: mp.Queue = context.Queue()
    port = _find_free_tcp_port()
    processes = [
        context.Process(
            target=_phase_failure_worker,
            args=(rank, port, str(writer.run_dir), output),
            daemon=False,
        )
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
                "two-rank phase convergence hung; "
                f"alive_pids={[process.pid for process in alive]}"
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    messages: list[tuple[int, str, str, tuple[int, ...], bool | None]] = []
    while True:
        try:
            messages.append(output.get_nowait())
        except Empty:
            break
    assert [process.exitcode for process in processes] == [0, 0]
    failures = sorted(item for item in messages if item[1] != "close_status")
    assert [(item[0], item[1], item[3]) for item in failures] == [
        (0, "runtime.distributed_phase_failed", (1,)),
        (1, "runtime.distributed_phase_failed", (1,)),
    ]
    assert failures[0][2] == failures[1][2]
    assert "peer-only secret-shaped" not in failures[0][2]
    closes = sorted(item for item in messages if item[1] == "close_status")
    assert [(item[0], item[4]) for item in closes] == [(0, True), (1, True)]

    state = json.loads(writer.run_path.read_text())
    assert state["status"] == "failed"
    assert state["measurement"]["terminal_phase"] == "model_loading"
    assert state["measurement"]["phases"]["model_loading"]["status"] == "failed"
    rank_resources = state["measurement"]["phases"]["model_loading"]["rank_resources"]
    assert rank_resources["per_rank"]["0"]["max_rss_bytes"] == 100
    assert rank_resources["per_rank"]["1"]["max_rss_bytes"] == 150
    assert rank_resources["global_maxima"] == {
        "scope": "all_rank_deterministic_maximum",
        "max_rss_bytes": 150,
        "io_read_bytes": 30,
        "io_write_bytes": 25,
    }
    assert state["measurement"]["steady_state_eligible"] is False
    assert "peer-only secret-shaped" not in state["terminal_error"]


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_corrupt_eval_shard_converges_before_global_metric() -> None:
    context = mp.get_context("spawn")
    output: mp.Queue = context.Queue()
    port = _find_free_tcp_port()
    processes = [
        context.Process(
            target=_eval_hydration_failure_worker,
            args=(rank, port, output),
            daemon=False,
        )
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
                "two-rank eval hydration convergence hung; "
                f"alive_pids={[process.pid for process in alive]}"
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    messages: list[tuple[int, str, tuple[int, ...], bool, object]] = []
    while True:
        try:
            messages.append(output.get_nowait())
        except Empty:
            break
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(messages) == [
        (0, "runtime.distributed_phase_failed", (1,), False, (0, 2)),
        (1, "runtime.distributed_phase_failed", (1,), False, (1, 2)),
    ]


def test_direct_phase_rejects_unbounded_rank_details_without_exposing_value() -> None:
    secret = "secret-shaped-" + "x" * 20_000

    with pytest.raises(RuntimeContractError) as exc_info:
        _run_rank_converged_phase(
            "model_loading",
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
            body=lambda: "ok",
            local_details=lambda: {"unbounded": secret},
            resource_collector=lambda: _rank_resource_snapshot(0),
        )

    assert exc_info.value.code == "runtime.phase_status_invalid"
    assert "secret-shaped" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Wave-0 pre-move characterization for
# `decompose-coordexp-swift-training-orchestration`.
#
# These additions freeze the single-rank companion to the two-rank ordered trace
# owned by `tests/training/test_orchestration_compatibility.py`.  They use the
# same seam as the tests above: the real `_run_rank_converged_phase` boundary and
# the real gatherer factories, with no transport substitution.
# ---------------------------------------------------------------------------


_WAVE0_SINGLE_RANK_CONVERGED_RECEIPT = {
    "rank_details": {"0": {"characterization": True}},
    "rank_resources": {
        "global_maxima": {
            "io_read_bytes": 10,
            "io_write_bytes": 20,
            "max_rss_bytes": 100,
            "scope": "all_rank_deterministic_maximum",
        },
        "per_rank": {
            "0": {
                "io_read_bytes": 10,
                "io_write_bytes": 20,
                "max_rss_bytes": 100,
                "scope": "current_process",
            }
        },
        "schema_version": 1,
        "scope": "current_process_lifetime_high_water_at_phase_observation",
        "world_size": 1,
    },
}


def test_wave0_single_rank_convergence_builds_no_collective_transport() -> None:
    assert _build_rank_report_gatherer(1) is None
    assert control_plane._build_model_free_preflight_gatherer(1) is None


def test_wave0_single_rank_phase_returns_body_result_and_exact_receipt() -> None:
    receipts: list[dict[str, object]] = []

    result = _run_rank_converged_phase(
        "model_loading",
        rank=0,
        world_size=1,
        rank_report_gatherer=None,
        body=lambda: "characterized-body-result",
        local_details=lambda: {"characterization": True},
        receipt_sink=receipts.append,
        resource_collector=lambda: _rank_resource_snapshot(0),
    )

    assert result == "characterized-body-result"
    assert receipts == [_WAVE0_SINGLE_RANK_CONVERGED_RECEIPT]


def test_wave0_single_rank_phase_raises_the_local_error_itself() -> None:
    original = RuntimeContractError(
        "characterized local failure",
        code="training.characterized_failure",
    )
    receipts: list[dict[str, object]] = []

    def failing_body() -> None:
        raise original

    with pytest.raises(RuntimeContractError) as exc_info:
        _run_rank_converged_phase(
            "model_loading",
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
            body=failing_body,
            receipt_sink=receipts.append,
            resource_collector=lambda: _rank_resource_snapshot(0),
        )

    assert exc_info.value is original
    assert receipts == [
        {"rank_resources": _WAVE0_SINGLE_RANK_CONVERGED_RECEIPT["rank_resources"]}
    ]


def test_wave0_multi_rank_phase_without_gatherer_fails_before_any_collective() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _run_rank_converged_phase(
            "model_loading",
            rank=0,
            world_size=2,
            rank_report_gatherer=None,
            body=lambda: "unreachable",
            resource_collector=lambda: _rank_resource_snapshot(0),
        )

    assert exc_info.value.code == "runtime.report_gather_unavailable"
    assert exc_info.value.context == {
        "rank": 0,
        "world_size": 2,
        "phase": "model_loading",
    }
