"""Bounded, non-allocating process resource receipts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.common.errors import ArtifactContractError


_SCHEMA_VERSION = 1
_CPU_SCOPE = "current_process"
_GPU_SCOPE = "current_process_current_device"
_AUTO_CUDA = object()
_CPU_COUNTERS = ("max_rss_bytes", "io_read_bytes", "io_write_bytes")
_GPU_COUNTERS = (
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
)
_RANK_CPU_SCOPE = "all_rank_deterministic_maximum"
_RANK_RECEIPT_SCOPE = "current_process_lifetime_high_water_at_phase_observation"
_RESOURCE_REASON_MAX_LENGTH = 128


def collect_resource_snapshot(
    *,
    getrusage_reader: Callable[[], object] | None = None,
    proc_io_reader: Callable[[], str] | None = None,
    cuda_api: object = _AUTO_CUDA,
    maxrss_unit_bytes: int | None = None,
) -> dict[str, object]:
    """Collect current-process high-water counters without initializing CUDA.

    The injected readers make the collector testable without depending on host
    ``/proc`` or CUDA state. The automatic CUDA path inspects only an already
    imported ``torch`` module; it never imports Torch, invokes ``nvidia-smi``,
    resets peak counters, or allocates device memory.
    """

    return {
        "schema_version": _SCHEMA_VERSION,
        "cpu": _collect_cpu(
            getrusage_reader=getrusage_reader,
            proc_io_reader=proc_io_reader,
            maxrss_unit_bytes=maxrss_unit_bytes,
        ),
        "gpu": _collect_gpu(cuda_api),
    }


def merge_resource_high_water(
    current: dict[str, object], sample: dict[str, object]
) -> dict[str, object]:
    """Return the deterministic, monotonic high-water merge of two snapshots."""

    current_snapshot = _validate_snapshot(current)
    sample_snapshot = _validate_snapshot(sample)

    current_cpu = current_snapshot["cpu"]
    sample_cpu = sample_snapshot["cpu"]
    merged_cpu: dict[str, object] = {"scope": _CPU_SCOPE}
    for field in _CPU_COUNTERS:
        merged_cpu[field] = _merge_measurement(current_cpu[field], sample_cpu[field])

    return {
        "schema_version": _SCHEMA_VERSION,
        "cpu": merged_cpu,
        "gpu": _merge_gpu(current_snapshot["gpu"], sample_snapshot["gpu"]),
    }


def converge_rank_cpu_resources(
    samples: Sequence[tuple[int, Mapping[str, object]]],
    *,
    world_size: int,
) -> dict[str, object]:
    """Validate one bounded process snapshot per rank and derive global maxima."""

    if not _is_positive_int(world_size):
        _raise_malformed("rank resource world size must be positive")
    by_rank: dict[int, dict[str, object]] = {}
    for rank, snapshot in samples:
        if not _is_nonnegative_int(rank) or rank >= world_size or rank in by_rank:
            _raise_malformed("rank resource identities are invalid")
        by_rank[rank] = _validate_snapshot(dict(snapshot))["cpu"]
    if tuple(sorted(by_rank)) != tuple(range(world_size)):
        _raise_malformed("rank resource samples must cover every world rank")

    maxima: dict[str, object] = {"scope": _RANK_CPU_SCOPE}
    for field in _CPU_COUNTERS:
        value: object = _unavailable("metric_unavailable_on_all_ranks")
        for rank in range(world_size):
            value = _merge_measurement(value, by_rank[rank][field])
        maxima[field] = value
    receipt = {
        "schema_version": _SCHEMA_VERSION,
        "scope": _RANK_RECEIPT_SCOPE,
        "world_size": world_size,
        "per_rank": {str(rank): by_rank[rank] for rank in range(world_size)},
        "global_maxima": maxima,
    }
    return validate_rank_cpu_resource_receipt(receipt)


def rank_cpu_resources_from_metric_rows(
    per_rank_metrics: Mapping[str, Any] | None,
    *,
    world_size: int,
) -> dict[str, object]:
    """Project gathered scalar metric rows into the canonical rank receipt."""

    snapshots: list[tuple[int, Mapping[str, object]]] = []
    rows = per_rank_metrics if isinstance(per_rank_metrics, Mapping) else {}
    for rank in range(world_size):
        raw = rows.get(str(rank), rows.get(rank, {}))
        metrics = raw if isinstance(raw, Mapping) else {}
        cpu: dict[str, object] = {"scope": _CPU_SCOPE}
        for field in _CPU_COUNTERS:
            value = metrics.get(f"resource/cpu_{field}")
            cpu[field] = (
                int(value)
                if _is_nonnegative_number_as_int(value)
                else _unavailable("metric_not_reported_by_rank")
            )
        snapshots.append(
            (
                rank,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "cpu": cpu,
                    "gpu": _uninitialized_gpu("not_collected_for_rank_cpu_receipt"),
                },
            )
        )
    return converge_rank_cpu_resources(snapshots, world_size=world_size)


def merge_rank_cpu_resource_receipts(
    current: Mapping[str, object] | None,
    sample: Mapping[str, object],
) -> dict[str, object]:
    """Merge repeated phase observations without losing per-rank availability."""

    sample_receipt = validate_rank_cpu_resource_receipt(sample)
    if current is None:
        return sample_receipt
    current_receipt = validate_rank_cpu_resource_receipt(current)
    if current_receipt["world_size"] != sample_receipt["world_size"]:
        _raise_malformed("rank resource receipts use different world sizes")
    world_size = int(sample_receipt["world_size"])
    per_rank: dict[str, dict[str, object]] = {}
    for rank in range(world_size):
        key = str(rank)
        current_cpu = current_receipt["per_rank"][key]
        sample_cpu = sample_receipt["per_rank"][key]
        per_rank[key] = {
            "scope": _CPU_SCOPE,
            **{
                field: _merge_measurement(current_cpu[field], sample_cpu[field])
                for field in _CPU_COUNTERS
            },
        }
    snapshots = [
        (
            rank,
            {
                "schema_version": _SCHEMA_VERSION,
                "cpu": per_rank[str(rank)],
                "gpu": _uninitialized_gpu("not_collected_for_rank_cpu_receipt"),
            },
        )
        for rank in range(world_size)
    ]
    return converge_rank_cpu_resources(snapshots, world_size=world_size)


def validate_rank_cpu_resource_receipt(value: object) -> dict[str, Any]:
    """Validate and copy the bounded per-rank CPU phase receipt."""

    expected = {
        "schema_version",
        "scope",
        "world_size",
        "per_rank",
        "global_maxima",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        _raise_malformed("rank CPU resource receipt fields are invalid")
    if value["schema_version"] != _SCHEMA_VERSION:
        raise ArtifactContractError(
            "rank CPU resource receipt schema version is incompatible",
            code="resources.incompatible_schema_version",
            context={"expected": _SCHEMA_VERSION, "actual": value["schema_version"]},
        )
    if value["scope"] != _RANK_RECEIPT_SCOPE:
        _raise_malformed("rank CPU resource receipt scope is invalid")
    world_size = value["world_size"]
    if not _is_positive_int(world_size):
        _raise_malformed("rank CPU resource receipt world size is invalid")
    per_rank_value = value["per_rank"]
    if not isinstance(per_rank_value, Mapping) or set(per_rank_value) != {
        str(rank) for rank in range(world_size)
    }:
        _raise_malformed("rank CPU resource receipt rank coverage is invalid")
    per_rank = {
        str(rank): _validate_cpu(per_rank_value[str(rank)])
        for rank in range(world_size)
    }
    maxima_value = value["global_maxima"]
    if not isinstance(maxima_value, Mapping) or set(maxima_value) != {
        "scope",
        *_CPU_COUNTERS,
    }:
        _raise_malformed("rank CPU resource global maxima fields are invalid")
    if maxima_value["scope"] != _RANK_CPU_SCOPE:
        _raise_malformed("rank CPU resource global maxima scope is invalid")
    maxima = {"scope": _RANK_CPU_SCOPE}
    for field in _CPU_COUNTERS:
        maxima[field] = _validate_measurement(maxima_value[field], field)
        expected_maximum: object = _unavailable("metric_unavailable_on_all_ranks")
        for rank in range(world_size):
            expected_maximum = _merge_measurement(
                expected_maximum, per_rank[str(rank)][field]
            )
        if maxima[field] != expected_maximum:
            _raise_malformed("rank CPU resource global maxima are inconsistent")
    return {
        "schema_version": _SCHEMA_VERSION,
        "scope": _RANK_RECEIPT_SCOPE,
        "world_size": world_size,
        "per_rank": per_rank,
        "global_maxima": maxima,
    }


def _collect_cpu(
    *,
    getrusage_reader: Callable[[], object] | None,
    proc_io_reader: Callable[[], str] | None,
    maxrss_unit_bytes: int | None,
) -> dict[str, object]:
    rss_reader = getrusage_reader or _read_current_rusage
    io_reader = proc_io_reader or _read_proc_self_io

    max_rss = _collect_max_rss(rss_reader, maxrss_unit_bytes)
    io_read, io_write = _collect_proc_io(io_reader)
    return {
        "scope": _CPU_SCOPE,
        "max_rss_bytes": max_rss,
        "io_read_bytes": io_read,
        "io_write_bytes": io_write,
    }


def _collect_max_rss(
    reader: Callable[[], object], maxrss_unit_bytes: int | None
) -> object:
    try:
        usage = reader()
        raw_value = getattr(usage, "ru_maxrss")
        unit_bytes = (
            _native_maxrss_unit_bytes()
            if maxrss_unit_bytes is None
            else maxrss_unit_bytes
        )
        if not _is_nonnegative_int(raw_value) or not _is_positive_int(unit_bytes):
            return _unavailable("invalid_getrusage_max_rss")
        return raw_value * unit_bytes
    except Exception:
        return _unavailable("getrusage_unavailable")


def _collect_proc_io(reader: Callable[[], str]) -> tuple[object, object]:
    try:
        text = reader()
    except Exception:
        unavailable = _unavailable("proc_self_io_unavailable")
        return unavailable, dict(unavailable)
    if not isinstance(text, str):
        unavailable = _unavailable("proc_self_io_invalid")
        return unavailable, dict(unavailable)

    values: dict[str, int] = {}
    invalid_fields: set[str] = set()
    for line in text.splitlines():
        name, separator, raw_value = line.partition(":")
        if separator != ":" or name not in {"read_bytes", "write_bytes"}:
            continue
        if name in values or name in invalid_fields:
            invalid_fields.add(name)
            values.pop(name, None)
            continue
        try:
            value = int(raw_value.strip(), 10)
        except ValueError:
            invalid_fields.add(name)
            continue
        if value < 0:
            invalid_fields.add(name)
            continue
        values[name] = value

    def result(name: str) -> object:
        if name in invalid_fields:
            return _unavailable(f"proc_self_io_invalid_{name}")
        if name not in values:
            return _unavailable(f"proc_self_io_missing_{name}")
        return values[name]

    return result("read_bytes"), result("write_bytes")


def _collect_gpu(cuda_api: object) -> dict[str, object]:
    unavailable_reason: str | None = None
    if cuda_api is _AUTO_CUDA:
        torch_module = sys.modules.get("torch")
        if torch_module is None:
            unavailable_reason = "torch_not_imported"
            cuda_api = None
        else:
            try:
                cuda_api = getattr(torch_module, "cuda")
            except Exception:
                unavailable_reason = "torch_cuda_api_unavailable"
                cuda_api = None
    elif cuda_api is None:
        unavailable_reason = "cuda_api_unavailable"

    if cuda_api is None:
        return _uninitialized_gpu(unavailable_reason or "cuda_api_unavailable")

    try:
        initialized = cuda_api.is_initialized()  # type: ignore[attr-defined]
    except Exception:
        return _uninitialized_gpu("cuda_initialization_state_unavailable")
    if not isinstance(initialized, bool):
        return _uninitialized_gpu("cuda_initialization_state_invalid")
    if not initialized:
        return _uninitialized_gpu("cuda_not_initialized")

    device_index = _read_cuda_counter(
        lambda: cuda_api.current_device(),  # type: ignore[attr-defined]
        "cuda_current_device_unavailable",
    )
    if not isinstance(device_index, int):
        allocated: object = _unavailable("cuda_current_device_unavailable")
        reserved: object = _unavailable("cuda_current_device_unavailable")
    else:
        allocated = _read_cuda_counter(
            lambda: cuda_api.max_memory_allocated(device_index),  # type: ignore[attr-defined]
            "cuda_max_memory_allocated_unavailable",
        )
        reserved = _read_cuda_counter(
            lambda: cuda_api.max_memory_reserved(device_index),  # type: ignore[attr-defined]
            "cuda_max_memory_reserved_unavailable",
        )
    return {
        "scope": _GPU_SCOPE,
        "initialized": True,
        "device_index": device_index,
        "max_memory_allocated_bytes": allocated,
        "max_memory_reserved_bytes": reserved,
    }


def _read_cuda_counter(reader: Callable[[], object], reason: str) -> object:
    try:
        value = reader()
    except Exception:
        return _unavailable(reason)
    if not _is_nonnegative_int(value):
        return _unavailable(reason)
    return value


def _uninitialized_gpu(reason: str) -> dict[str, object]:
    return {
        "scope": _GPU_SCOPE,
        "initialized": False,
        "unavailable_reason": reason,
    }


def _read_current_rusage() -> object:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF)


def _read_proc_self_io() -> str:
    return Path("/proc/self/io").read_text(encoding="utf-8")


def _native_maxrss_unit_bytes() -> int:
    # macOS reports bytes; Linux and the supported training hosts report KiB.
    return 1 if sys.platform == "darwin" else 1024


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_number_as_int(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value).is_integer()
        and 0 <= float(value) <= sys.maxsize
    )


def _validate_snapshot(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        _raise_malformed("resource snapshot must be a dictionary")
    if "schema_version" not in value:
        _raise_malformed("resource snapshot schema version is missing")
    version = value["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        _raise_malformed("resource snapshot schema version must be an integer")
    if version != _SCHEMA_VERSION:
        raise ArtifactContractError(
            "resource snapshot schema version is incompatible",
            code="resources.incompatible_schema_version",
            context={"expected": _SCHEMA_VERSION, "actual": version},
        )
    if set(value) != {"schema_version", "cpu", "gpu"}:
        _raise_malformed("resource snapshot fields are invalid")
    cpu = _validate_cpu(value["cpu"])
    gpu = _validate_gpu(value["gpu"])
    return {"schema_version": _SCHEMA_VERSION, "cpu": cpu, "gpu": gpu}


def _validate_cpu(value: object) -> dict[str, object]:
    expected = {"scope", *_CPU_COUNTERS}
    if not isinstance(value, dict) or set(value) != expected:
        _raise_malformed("CPU resource fields are invalid")
    if value["scope"] != _CPU_SCOPE:
        _raise_malformed("CPU resource scope is invalid")
    result: dict[str, object] = {"scope": _CPU_SCOPE}
    for field in _CPU_COUNTERS:
        result[field] = _validate_measurement(value[field], field)
    return result


def _validate_gpu(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        _raise_malformed("GPU resource value must be a dictionary")
    if value.get("scope") != _GPU_SCOPE:
        _raise_malformed("GPU resource scope is invalid")
    initialized = value.get("initialized")
    if not isinstance(initialized, bool):
        _raise_malformed("GPU initialized state must be boolean")

    if not initialized:
        if set(value) != {"scope", "initialized", "unavailable_reason"}:
            _raise_malformed("uninitialized GPU resource fields are invalid")
        reason = value["unavailable_reason"]
        if not isinstance(reason, str) or not reason:
            _raise_malformed("GPU unavailable reason must be nonempty")
        return {
            "scope": _GPU_SCOPE,
            "initialized": False,
            "unavailable_reason": reason,
        }

    expected = {"scope", "initialized", "device_index", *_GPU_COUNTERS}
    if set(value) != expected:
        _raise_malformed("initialized GPU resource fields are invalid")
    result = {
        "scope": _GPU_SCOPE,
        "initialized": True,
        "device_index": _validate_measurement(value["device_index"], "device_index"),
    }
    for field in _GPU_COUNTERS:
        result[field] = _validate_measurement(value[field], field)
    return result


def _validate_measurement(value: object, field: str) -> object:
    if _is_nonnegative_int(value):
        return value
    if (
        isinstance(value, dict)
        and set(value) == {"status", "reason"}
        and value.get("status") == "unavailable"
        and isinstance(value.get("reason"), str)
        and bool(value["reason"])
        and len(value["reason"]) <= _RESOURCE_REASON_MAX_LENGTH
        and all(
            character.isascii()
            and (character.islower() or character.isdigit() or character == "_")
            for character in value["reason"]
        )
    ):
        return _unavailable(value["reason"])
    _raise_malformed(f"resource measurement {field} is invalid")


def _merge_measurement(current: object, sample: object) -> object:
    if isinstance(current, int) and isinstance(sample, int):
        return max(current, sample)
    if isinstance(current, int):
        return current
    if isinstance(sample, int):
        return sample
    current_reason = current["reason"]  # type: ignore[index]
    sample_reason = sample["reason"]  # type: ignore[index]
    return _unavailable(min(current_reason, sample_reason))


def _merge_gpu(
    current: dict[str, object], sample: dict[str, object]
) -> dict[str, object]:
    current_initialized = current["initialized"]
    sample_initialized = sample["initialized"]
    if not current_initialized and not sample_initialized:
        return _uninitialized_gpu(
            min(
                current["unavailable_reason"],  # type: ignore[arg-type]
                sample["unavailable_reason"],  # type: ignore[arg-type]
            )
        )
    if current_initialized and not sample_initialized:
        return _copy_initialized_gpu(current)
    if sample_initialized and not current_initialized:
        return _copy_initialized_gpu(sample)

    current_device = current["device_index"]
    sample_device = sample["device_index"]
    if (
        isinstance(current_device, int)
        and isinstance(sample_device, int)
        and current_device != sample_device
    ):
        raise ArtifactContractError(
            "resource snapshots refer to different CUDA devices",
            code="resources.incompatible_gpu_device",
            context={"current_device": current_device, "sample_device": sample_device},
        )
    return {
        "scope": _GPU_SCOPE,
        "initialized": True,
        "device_index": _merge_measurement(current_device, sample_device),
        "max_memory_allocated_bytes": _merge_measurement(
            current["max_memory_allocated_bytes"],
            sample["max_memory_allocated_bytes"],
        ),
        "max_memory_reserved_bytes": _merge_measurement(
            current["max_memory_reserved_bytes"],
            sample["max_memory_reserved_bytes"],
        ),
    }


def _copy_initialized_gpu(value: dict[str, object]) -> dict[str, object]:
    return {
        "scope": _GPU_SCOPE,
        "initialized": True,
        "device_index": _copy_measurement(value["device_index"]),
        "max_memory_allocated_bytes": _copy_measurement(
            value["max_memory_allocated_bytes"]
        ),
        "max_memory_reserved_bytes": _copy_measurement(
            value["max_memory_reserved_bytes"]
        ),
    }


def _copy_measurement(value: object) -> object:
    if isinstance(value, int):
        return value
    return _unavailable(value["reason"])  # type: ignore[index]


def _raise_malformed(message: str) -> None:
    raise ArtifactContractError(message, code="resources.malformed_snapshot")
