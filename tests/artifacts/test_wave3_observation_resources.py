"""Wave-3 (tasks 3.6/3.7) CUDA allocator observation and bounded row lists.

The allocator sample is a SEPARATE collector from ``collect_resource_snapshot``:
that snapshot's schema is embedded verbatim in durable phase receipts, so this
change adds a new bounded reader beside it instead of retyping it.

Everything here is fake-backed on purpose. Real CUDA evidence belongs to task
3.8; what these tests own is that an unmeasurable counter is reported as
explicitly unavailable and NEVER as a fabricated zero.
"""

from __future__ import annotations

import pytest

from src.artifacts.resources import (
    collect_cuda_allocator_sample,
    cuda_allocator_counter_deltas,
)
from src.artifacts.run_schema import _normalize_logging_row
from src.common.errors import ArtifactContractError


class _FakeCudaApi:
    def __init__(
        self,
        *,
        initialized: bool = True,
        allocated: object = 2048,
        reserved: object = 4096,
        stats: object = None,
    ) -> None:
        self._initialized = initialized
        self._allocated = allocated
        self._reserved = reserved
        self._stats = (
            {"num_alloc_retries": 3, "num_ooms": 1} if stats is None else stats
        )
        self.reset_calls = 0

    def is_initialized(self) -> bool:
        return self._initialized

    def current_device(self) -> int:
        return 0

    def memory_allocated(self, device: int) -> object:
        del device
        if isinstance(self._allocated, Exception):
            raise self._allocated
        return self._allocated

    def memory_reserved(self, device: int) -> object:
        del device
        if isinstance(self._reserved, Exception):
            raise self._reserved
        return self._reserved

    def memory_stats(self, device: int) -> object:
        del device
        if isinstance(self._stats, Exception):
            raise self._stats
        return self._stats

    def reset_peak_memory_stats(self, device: int = 0) -> None:
        self.reset_calls += 1


def _unavailable(value: object) -> bool:
    return isinstance(value, dict) and value.get("status") == "unavailable"


# ---------------------------------------------------------------------------
# 3.6 - allocator sample
# ---------------------------------------------------------------------------


def test_cuda_allocator_sample_reads_current_bytes_and_lifetime_counters() -> None:
    sample = collect_cuda_allocator_sample(cuda_api=_FakeCudaApi())

    assert sample["available"] is True
    assert sample["current_allocated_bytes"] == 2048
    assert sample["current_reserved_bytes"] == 4096
    assert sample["num_alloc_retries"] == 3
    assert sample["num_ooms"] == 1


def test_cuda_allocator_sample_never_resets_peak_statistics() -> None:
    api = _FakeCudaApi()

    collect_cuda_allocator_sample(cuda_api=api)

    # Resetting peaks would mutate measurement state shared with the phase
    # resource receipts (design decision 4).
    assert api.reset_calls == 0


def test_cuda_allocator_sample_is_unavailable_without_cuda_and_fabricates_no_zero() -> None:
    sample = collect_cuda_allocator_sample(cuda_api=None)

    assert sample["available"] is False
    assert sample["unavailable_reason"] == "cuda_api_unavailable"
    for field in (
        "current_allocated_bytes",
        "current_reserved_bytes",
        "num_alloc_retries",
        "num_ooms",
    ):
        assert _unavailable(sample[field]), field
        assert sample[field] != 0


def test_cuda_allocator_sample_is_unavailable_when_cuda_is_not_initialized() -> None:
    sample = collect_cuda_allocator_sample(cuda_api=_FakeCudaApi(initialized=False))

    assert sample["available"] is False
    assert sample["unavailable_reason"] == "cuda_not_initialized"


def test_cuda_allocator_sample_marks_one_failed_read_unavailable() -> None:
    sample = collect_cuda_allocator_sample(
        cuda_api=_FakeCudaApi(stats=RuntimeError("no stats"))
    )

    assert sample["available"] is False
    assert sample["current_allocated_bytes"] == 2048
    assert _unavailable(sample["num_alloc_retries"])
    assert _unavailable(sample["num_ooms"])


def test_cuda_allocator_sample_rejects_a_non_integer_counter() -> None:
    sample = collect_cuda_allocator_sample(
        cuda_api=_FakeCudaApi(stats={"num_alloc_retries": "3", "num_ooms": 1})
    )

    assert sample["available"] is False
    assert _unavailable(sample["num_alloc_retries"])
    assert sample["num_ooms"] == 1


# ---------------------------------------------------------------------------
# 3.6 - per-step counter deltas
# ---------------------------------------------------------------------------


def test_cuda_allocator_counter_deltas_are_per_step_differences() -> None:
    previous = collect_cuda_allocator_sample(
        cuda_api=_FakeCudaApi(stats={"num_alloc_retries": 3, "num_ooms": 1})
    )
    current = collect_cuda_allocator_sample(
        cuda_api=_FakeCudaApi(stats={"num_alloc_retries": 7, "num_ooms": 1})
    )

    assert cuda_allocator_counter_deltas(previous, current) == {
        "num_alloc_retries": 4,
        "num_ooms": 0,
    }


def test_first_observation_has_no_prior_snapshot_and_reports_no_delta() -> None:
    current = collect_cuda_allocator_sample(cuda_api=_FakeCudaApi())

    assert cuda_allocator_counter_deltas(None, current) == {
        "num_alloc_retries": None,
        "num_ooms": None,
    }


def test_cuda_allocator_counter_deltas_reject_a_non_monotonic_counter() -> None:
    previous = collect_cuda_allocator_sample(
        cuda_api=_FakeCudaApi(stats={"num_alloc_retries": 7, "num_ooms": 2})
    )
    current = collect_cuda_allocator_sample(
        cuda_api=_FakeCudaApi(stats={"num_alloc_retries": 3, "num_ooms": 2})
    )

    # A process-lifetime counter that went backwards is not a negative delta.
    assert cuda_allocator_counter_deltas(previous, current) == {
        "num_alloc_retries": None,
        "num_ooms": 0,
    }


def test_cuda_allocator_counter_deltas_are_none_when_either_side_is_unavailable() -> None:
    available = collect_cuda_allocator_sample(cuda_api=_FakeCudaApi())
    missing = collect_cuda_allocator_sample(cuda_api=None)

    assert cuda_allocator_counter_deltas(missing, available) == {
        "num_alloc_retries": None,
        "num_ooms": None,
    }
    assert cuda_allocator_counter_deltas(available, missing) == {
        "num_alloc_retries": None,
        "num_ooms": None,
    }


# ---------------------------------------------------------------------------
# 3.7 - bounded, sorted, unique diagnostic name lists
# ---------------------------------------------------------------------------


def test_unavailable_fields_are_sorted_and_unique() -> None:
    row = _normalize_logging_row(
        {
            "step": 1,
            "split": "train",
            "unavailable_fields": ["b", "a", "b"],
        }
    )

    assert row["unavailable_fields"] == ["a", "b"]
    assert "unavailable_fields_truncated_count" not in row


def test_bounded_lists_add_no_truncation_count_when_they_fit() -> None:
    row = _normalize_logging_row({"step": 1, "split": "train"})

    assert row["non_finite_fields"] == []
    assert "non_finite_fields_truncated_count" not in row
    assert "unavailable_fields" not in row


def test_unavailable_fields_are_bounded_with_a_truncation_count() -> None:
    names = [f"metric_{index:04d}" for index in range(300)]
    row = _normalize_logging_row(
        {"step": 1, "split": "train", "unavailable_fields": list(reversed(names))}
    )

    assert row["unavailable_fields"] == sorted(names)[:256]
    assert row["unavailable_fields_truncated_count"] == 44


def test_non_finite_fields_are_bounded_with_a_truncation_count() -> None:
    row = _normalize_logging_row(
        {
            "step": 1,
            "split": "train",
            **{f"metric_{index:04d}": float("nan") for index in range(300)},
        }
    )

    assert len(row["non_finite_fields"]) == 256
    assert row["non_finite_fields"] == sorted(row["non_finite_fields"])
    assert row["non_finite_fields_truncated_count"] == 44


def test_a_field_name_longer_than_its_byte_bound_is_rejected() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        _normalize_logging_row(
            {"step": 1, "split": "train", "unavailable_fields": ["x" * 257]}
        )

    assert exc_info.value.code == "run_writer.invalid_unavailable_fields"


def test_unavailable_fields_reject_non_string_entries() -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        _normalize_logging_row(
            {"step": 1, "split": "train", "unavailable_fields": [1]}
        )

    assert exc_info.value.code == "run_writer.invalid_unavailable_fields"
