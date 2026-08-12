from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.artifacts.resources import (
    collect_resource_snapshot,
    converge_rank_cpu_resources,
    merge_rank_cpu_resource_receipts,
    merge_resource_high_water,
    rank_cpu_resources_from_metric_rows,
    validate_rank_cpu_resource_receipt,
)
from src.common.errors import ArtifactContractError


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


class _UninitializedCuda:
    def is_initialized(self) -> bool:
        return False

    def current_device(self) -> int:
        raise AssertionError("current_device must not be called")

    def max_memory_allocated(self, _device: int) -> int:
        raise AssertionError("max_memory_allocated must not be called")

    def max_memory_reserved(self, _device: int) -> int:
        raise AssertionError("max_memory_reserved must not be called")


class _InitializedCuda:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def is_initialized(self) -> bool:
        self.calls.append(("is_initialized", None))
        return True

    def current_device(self) -> int:
        self.calls.append(("current_device", None))
        return 3

    def max_memory_allocated(self, device: int) -> int:
        self.calls.append(("max_memory_allocated", device))
        return 4096

    def max_memory_reserved(self, device: int) -> int:
        self.calls.append(("max_memory_reserved", device))
        return 8192


def _snapshot(
    *,
    max_rss_bytes: object,
    io_read_bytes: object,
    io_write_bytes: object,
    gpu: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "cpu": {
            "scope": "current_process",
            "max_rss_bytes": max_rss_bytes,
            "io_read_bytes": io_read_bytes,
            "io_write_bytes": io_write_bytes,
        },
        "gpu": gpu
        or {
            "scope": "current_process_current_device",
            "initialized": False,
            "unavailable_reason": "cuda_not_initialized",
        },
    }


def test_collects_available_current_process_cpu_resources() -> None:
    receipt = collect_resource_snapshot(
        getrusage_reader=lambda: SimpleNamespace(ru_maxrss=123),
        maxrss_unit_bytes=1024,
        proc_io_reader=lambda: "rchar: 99\nread_bytes: 456\nwrite_bytes: 789\n",
        cuda_api=_UninitializedCuda(),
    )

    assert receipt["cpu"] == {
        "scope": "current_process",
        "max_rss_bytes": 123 * 1024,
        "io_read_bytes": 456,
        "io_write_bytes": 789,
    }


def test_missing_proc_io_is_explicitly_unavailable() -> None:
    def missing_proc_io() -> str:
        raise FileNotFoundError("injected missing /proc")

    receipt = collect_resource_snapshot(
        getrusage_reader=lambda: SimpleNamespace(ru_maxrss=4),
        maxrss_unit_bytes=1024,
        proc_io_reader=missing_proc_io,
        cuda_api=_UninitializedCuda(),
    )

    assert receipt["cpu"] == {
        "scope": "current_process",
        "max_rss_bytes": 4096,
        "io_read_bytes": _unavailable("proc_self_io_unavailable"),
        "io_write_bytes": _unavailable("proc_self_io_unavailable"),
    }


def test_uninitialized_cuda_does_not_call_device_or_memory_apis() -> None:
    receipt = collect_resource_snapshot(
        getrusage_reader=lambda: SimpleNamespace(ru_maxrss=1),
        maxrss_unit_bytes=1024,
        proc_io_reader=lambda: "read_bytes: 2\nwrite_bytes: 3\n",
        cuda_api=_UninitializedCuda(),
    )

    assert receipt["gpu"] == {
        "scope": "current_process_current_device",
        "initialized": False,
        "unavailable_reason": "cuda_not_initialized",
    }


def test_initialized_cuda_reads_only_current_device_peak_counters() -> None:
    cuda = _InitializedCuda()

    receipt = collect_resource_snapshot(
        getrusage_reader=lambda: SimpleNamespace(ru_maxrss=1),
        maxrss_unit_bytes=1024,
        proc_io_reader=lambda: "read_bytes: 2\nwrite_bytes: 3\n",
        cuda_api=cuda,
    )

    assert receipt["gpu"] == {
        "scope": "current_process_current_device",
        "initialized": True,
        "device_index": 3,
        "max_memory_allocated_bytes": 4096,
        "max_memory_reserved_bytes": 8192,
    }
    assert cuda.calls == [
        ("is_initialized", None),
        ("current_device", None),
        ("max_memory_allocated", 3),
        ("max_memory_reserved", 3),
    ]


def test_merge_preserves_available_values_across_partial_unavailability() -> None:
    current = _snapshot(
        max_rss_bytes=100,
        io_read_bytes=_unavailable("proc_self_io_unavailable"),
        io_write_bytes=300,
    )
    sample = _snapshot(
        max_rss_bytes=_unavailable("getrusage_unavailable"),
        io_read_bytes=200,
        io_write_bytes=_unavailable("proc_self_io_unavailable"),
    )

    merged = merge_resource_high_water(current, sample)

    assert merged["cpu"] == {
        "scope": "current_process",
        "max_rss_bytes": 100,
        "io_read_bytes": 200,
        "io_write_bytes": 300,
    }


def test_merge_takes_monotonic_cpu_and_gpu_maxima() -> None:
    current = _snapshot(
        max_rss_bytes=100,
        io_read_bytes=200,
        io_write_bytes=300,
        gpu={
            "scope": "current_process_current_device",
            "initialized": True,
            "device_index": 2,
            "max_memory_allocated_bytes": 400,
            "max_memory_reserved_bytes": 500,
        },
    )
    sample = _snapshot(
        max_rss_bytes=150,
        io_read_bytes=190,
        io_write_bytes=350,
        gpu={
            "scope": "current_process_current_device",
            "initialized": True,
            "device_index": 2,
            "max_memory_allocated_bytes": 450,
            "max_memory_reserved_bytes": 490,
        },
    )

    merged = merge_resource_high_water(current, sample)

    assert merged == _snapshot(
        max_rss_bytes=150,
        io_read_bytes=200,
        io_write_bytes=350,
        gpu={
            "scope": "current_process_current_device",
            "initialized": True,
            "device_index": 2,
            "max_memory_allocated_bytes": 450,
            "max_memory_reserved_bytes": 500,
        },
    )


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        ({}, "resources.malformed_snapshot"),
        (
            _snapshot(max_rss_bytes=-1, io_read_bytes=2, io_write_bytes=3),
            "resources.malformed_snapshot",
        ),
        (
            {
                **_snapshot(max_rss_bytes=1, io_read_bytes=2, io_write_bytes=3),
                "schema_version": 2,
            },
            "resources.incompatible_schema_version",
        ),
    ],
)
def test_merge_rejects_malformed_or_incompatible_snapshots(
    value: dict[str, object], expected_code: str
) -> None:
    valid = _snapshot(max_rss_bytes=1, io_read_bytes=2, io_write_bytes=3)

    with pytest.raises(ArtifactContractError) as exc_info:
        merge_resource_high_water(valid, value)

    assert exc_info.value.code == expected_code


def test_collectors_are_strict_json_and_do_not_expose_failure_text() -> None:
    secret = "SECRET_RESOURCE_FAILURE_MUST_NOT_ESCAPE"

    def fail_rusage() -> object:
        raise RuntimeError(secret)

    def fail_proc_io() -> str:
        raise RuntimeError(secret)

    class FailingCuda:
        def is_initialized(self) -> bool:
            raise RuntimeError(secret)

    receipt = collect_resource_snapshot(
        getrusage_reader=fail_rusage,
        proc_io_reader=fail_proc_io,
        cuda_api=FailingCuda(),
    )
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)

    assert secret not in encoded
    assert set(receipt) == {"schema_version", "cpu", "gpu"}
    assert receipt["schema_version"] == 1
    assert receipt["cpu"]["max_rss_bytes"] == _unavailable("getrusage_unavailable")
    assert receipt["gpu"] == {
        "scope": "current_process_current_device",
        "initialized": False,
        "unavailable_reason": "cuda_initialization_state_unavailable",
    }


def test_rank_cpu_resources_preserve_samples_and_deterministic_global_maxima() -> None:
    receipt = converge_rank_cpu_resources(
        (
            (
                1,
                _snapshot(
                    max_rss_bytes=300,
                    io_read_bytes=_unavailable("proc_self_io_unavailable"),
                    io_write_bytes=20,
                ),
            ),
            (
                0,
                _snapshot(
                    max_rss_bytes=200,
                    io_read_bytes=40,
                    io_write_bytes=_unavailable("proc_self_io_unavailable"),
                ),
            ),
        ),
        world_size=2,
    )

    assert list(receipt["per_rank"]) == ["0", "1"]
    assert receipt["per_rank"]["0"]["max_rss_bytes"] == 200
    assert receipt["per_rank"]["1"]["max_rss_bytes"] == 300
    assert receipt["global_maxima"] == {
        "scope": "all_rank_deterministic_maximum",
        "max_rss_bytes": 300,
        "io_read_bytes": 40,
        "io_write_bytes": 20,
    }


def test_rank_metric_rows_make_missing_values_explicit_and_merge_observations() -> None:
    first = rank_cpu_resources_from_metric_rows(
        {
            "0": {
                "resource/cpu_max_rss_bytes": 100.0,
                "resource/cpu_io_read_bytes": 10.0,
            },
            "1": {"resource/cpu_io_write_bytes": 20.0},
        },
        world_size=2,
    )
    second = rank_cpu_resources_from_metric_rows(
        {
            "0": {"resource/cpu_io_write_bytes": 30.0},
            "1": {"resource/cpu_max_rss_bytes": 200.0},
        },
        world_size=2,
    )

    merged = merge_rank_cpu_resource_receipts(first, second)
    assert merged["per_rank"]["0"] == {
        "scope": "current_process",
        "max_rss_bytes": 100,
        "io_read_bytes": 10,
        "io_write_bytes": 30,
    }
    assert merged["per_rank"]["1"] == {
        "scope": "current_process",
        "max_rss_bytes": 200,
        "io_read_bytes": _unavailable("metric_not_reported_by_rank"),
        "io_write_bytes": 20,
    }
    assert merged["global_maxima"]["max_rss_bytes"] == 200


def test_rank_receipt_rejects_unbounded_unavailable_reason() -> None:
    receipt = converge_rank_cpu_resources(
        ((0, _snapshot(max_rss_bytes=1, io_read_bytes=2, io_write_bytes=3)),),
        world_size=1,
    )
    receipt["per_rank"]["0"]["io_read_bytes"] = _unavailable("x" * 1024)

    with pytest.raises(ArtifactContractError) as exc_info:
        validate_rank_cpu_resource_receipt(receipt)

    assert exc_info.value.code == "resources.malformed_snapshot"
