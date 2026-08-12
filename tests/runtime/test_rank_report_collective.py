from __future__ import annotations

from datetime import timedelta
import multiprocessing as mp
import pickle
from queue import Empty
import socket
import traceback
from typing import Any

import pytest
import torch.distributed as dist

from src.common.errors import RuntimeContractError
from src.runtime import RankGradientFiniteReport, RankScalarFiniteReport
from src.training.exact_resume import (
    DistributedExactResumeRestoreStatus,
    DistributedExactResumeStatus,
)
import src.training.pipeline as training_pipeline
from src.training.pipeline import (
    _RANK_REPORT_FRAME_BYTES,
    _RANK_REPORT_MAX_PAYLOAD_BYTES,
    _build_rank_report_gatherer,
)


_WORLD_SIZE = 8
_PROCESS_GROUP_TIMEOUT_SECONDS = 15
_JOIN_TIMEOUT_SECONDS = 60


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


def _destroy_process_group_best_effort() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def test_non_gloo_runtime_uses_and_closes_dedicated_gloo_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_group = object()
    created_groups: list[dict[str, Any]] = []
    destroyed_groups: list[Any] = []

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "is_gloo_available", lambda: True)

    def new_group(
        *,
        ranks: list[int],
        backend: str,
        timeout: timedelta,
    ) -> object:
        created_groups.append({"ranks": ranks, "backend": backend, "timeout": timeout})
        return control_group

    def destroy_process_group(group: Any) -> None:
        destroyed_groups.append(group)

    def gather_peer_frame(
        distributed: Any,
        payload: bytes,
        *,
        width: int,
        world_size: int,
        group: Any | None,
    ) -> tuple[bytes, ...]:
        assert distributed is dist
        assert width == _RANK_REPORT_FRAME_BYTES
        assert world_size == 2
        assert group is control_group
        local_header = training_pipeline._unpack_rank_report_header(
            payload[: training_pipeline._RANK_REPORT_HEADER.size]
        )
        local_payload_size = int(local_header["payload_size"])
        local_report = pickle.loads(
            payload[
                training_pipeline._RANK_REPORT_HEADER.size : training_pipeline._RANK_REPORT_HEADER.size
                + local_payload_size
            ]
        )
        peer_report = {**local_report, "rank": 1}
        peer_payload = pickle.dumps(peer_report, protocol=pickle.HIGHEST_PROTOCOL)
        peer_header = training_pipeline._rank_report_header(
            peer_report,
            sequence=int(local_header["sequence"]),
            payload=peer_payload,
            serialization_status=0,
        )

        def fixed_width(frame: bytes) -> bytes:
            return frame + bytes(width - len(frame))

        return fixed_width(payload), fixed_width(peer_header + peer_payload)

    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(dist, "destroy_process_group", destroy_process_group)
    monkeypatch.setattr(
        training_pipeline,
        "_all_gather_cpu_bytes",
        gather_peer_frame,
    )

    gather = _build_rank_report_gatherer(2)
    assert gather is not None
    report = {
        "kind": "metrics",
        "planned_step_id": 3,
        "split": "eval.forward",
        "rank": 0,
        "world_size": 2,
    }
    assert [item["rank"] for item in gather(report)] == [0, 1]
    assert [item["rank"] for item in gather(report)] == [0, 1]
    assert created_groups == [
        {
            "ranks": [0, 1],
            "backend": "gloo",
            "timeout": timedelta(
                seconds=training_pipeline._RANK_REPORT_CONTROL_TIMEOUT_SECONDS
            ),
        }
    ]

    gather.close()  # type: ignore[attr-defined]
    assert destroyed_groups == [control_group]


def _rank_report_worker(rank: int, port: int, output: mp.Queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    original_all_gather_object = dist.all_gather_object
    original_all_gather = dist.all_gather
    original_rank_report_header = training_pipeline._rank_report_header
    all_gather_calls = 0
    gather: Any | None = None

    def forbid_object_collective(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unbounded object collective must not be used")

    def checked_fixed_frame_gather(
        tensor_list: list[Any],
        tensor: Any,
        group: Any | None = None,
        async_op: bool = False,
    ) -> Any:
        nonlocal all_gather_calls
        all_gather_calls += 1
        assert int(tensor.numel()) == _RANK_REPORT_FRAME_BYTES
        assert all(
            int(item.numel()) == _RANK_REPORT_FRAME_BYTES for item in tensor_list
        )
        return original_all_gather(
            tensor_list,
            tensor,
            group=group,
            async_op=async_op,
        )

    dist.all_gather_object = forbid_object_collective  # type: ignore[assignment]
    dist.all_gather = checked_fixed_frame_gather  # type: ignore[assignment]
    try:
        gather = _build_rank_report_gatherer(_WORLD_SIZE)
        assert gather is not None

        scalar_reports = gather(
            RankScalarFiniteReport(
                planned_step_id=11,
                rank=rank,
                world_size=_WORLD_SIZE,
                total_loss_finite=True,
                term_finite={"base_ce": True},
                term_weighted_losses={"base_ce": float(rank)},
                term_raw_losses={"base_ce": float(rank)},
                term_selected_counts={"base_ce": rank + 1},
                term_eligible_segment_counts={"base_ce": rank + 1},
            )
        )
        assert [report.rank for report in scalar_reports] == list(range(_WORLD_SIZE))

        gradient_reports = gather(
            RankGradientFiniteReport(
                planned_step_id=11,
                rank=rank,
                world_size=_WORLD_SIZE,
                gradients_finite=True,
                backend_overflow=False,
                grad_norm=float(rank + 1),
            )
        )
        assert [report.grad_norm for report in gradient_reports] == [
            float(peer_rank + 1) for peer_rank in range(_WORLD_SIZE)
        ]

        publication_statuses = gather(
            DistributedExactResumeStatus(
                phase="commit",
                rank=rank,
                world_size=_WORLD_SIZE,
                ok=True,
            )
        )
        assert [status.world_size for status in publication_statuses] == [
            _WORLD_SIZE
        ] * _WORLD_SIZE

        restore_statuses = gather(
            DistributedExactResumeRestoreStatus(
                phase="restore_apply",
                rank=rank,
                world_size=_WORLD_SIZE,
                ok=False,
                error_code="training_state.test_failure",
                error_type="TestFailure",
            )
        )
        assert [status.world_size for status in restore_statuses] == [
            _WORLD_SIZE
        ] * _WORLD_SIZE

        metric_reports = gather(
            {
                "kind": "metrics",
                "planned_step_id": 11,
                "split": "eval.forward",
                "rank": rank,
                "world_size": _WORLD_SIZE,
                "metrics": {"loss/total": float(rank)},
            }
        )
        assert [report["rank"] for report in metric_reports] == list(range(_WORLD_SIZE))

        denominator_reports = gather(
            {
                "kind": "loss_denominators",
                "planned_step_id": 11,
                "rank": rank,
                "world_size": _WORLD_SIZE,
                "denominators": {"base_ce": {"eligible_segment_count": rank + 1}},
            }
        )
        assert [report["rank"] for report in denominator_reports] == list(
            range(_WORLD_SIZE)
        )

        mismatched_kind = "wrong-kind" if rank == _WORLD_SIZE - 1 else "metrics"
        with pytest.raises(RuntimeContractError) as exc_info:
            gather(
                {
                    "kind": mismatched_kind,
                    "planned_step_id": 12,
                    "split": "eval.forward",
                    "rank": rank,
                    "world_size": _WORLD_SIZE,
                    "metrics": {"loss/total": float(rank)},
                }
            )
        assert exc_info.value.code == "runtime.report_gather_identity"

        with pytest.raises(RuntimeContractError) as exc_info:
            gather(
                {
                    "kind": "oversized_probe",
                    "planned_step_id": 13,
                    "rank": rank,
                    "world_size": _WORLD_SIZE,
                    "payload": "x"
                    * (
                        _RANK_REPORT_MAX_PAYLOAD_BYTES + 1024
                        if rank == _WORLD_SIZE - 1
                        else 1
                    ),
                }
            )
        assert exc_info.value.code == "runtime.report_gather_size"

        with pytest.raises(RuntimeContractError) as exc_info:
            gather(
                {
                    "kind": "serialization_probe",
                    "planned_step_id": 14,
                    "rank": rank,
                    "world_size": _WORLD_SIZE,
                    "payload": (lambda: None) if rank == _WORLD_SIZE - 1 else None,
                }
            )
        assert exc_info.value.code == "runtime.report_serialize_failed"

        def out_of_sequence_header(
            report: Any,
            *,
            sequence: int,
            payload: bytes,
            serialization_status: int,
        ) -> bytes:
            return original_rank_report_header(
                report,
                sequence=sequence + 1,
                payload=payload,
                serialization_status=serialization_status,
            )

        if rank == _WORLD_SIZE - 1:
            training_pipeline._rank_report_header = out_of_sequence_header
        try:
            with pytest.raises(RuntimeContractError) as exc_info:
                gather(
                    {
                        "kind": "sequence_probe",
                        "planned_step_id": 15,
                        "rank": rank,
                        "world_size": _WORLD_SIZE,
                    }
                )
            assert exc_info.value.code == "runtime.report_gather_sequence"
        finally:
            training_pipeline._rank_report_header = original_rank_report_header

        assert all_gather_calls == 10
        output.put((rank, "ok:10"))
    except BaseException:
        output.put((rank, traceback.format_exc()))
        raise
    finally:
        close = getattr(gather, "close", None)
        if callable(close):
            close()
        dist.all_gather_object = original_all_gather_object  # type: ignore[assignment]
        dist.all_gather = original_all_gather  # type: ignore[assignment]
        training_pipeline._rank_report_header = original_rank_report_header
        _destroy_process_group_best_effort()


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_eight_rank_reports_use_bounded_typed_collectives_and_fail_closed() -> None:
    context = mp.get_context("spawn")
    port = _find_free_tcp_port()
    output: mp.Queue = context.Queue()
    processes = [
        context.Process(
            target=_rank_report_worker,
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
                "rank report collective hung; "
                f"alive_pids={[process.pid for process in alive]}"
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

    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    assert messages == {rank: "ok:10" for rank in range(_WORLD_SIZE)}
