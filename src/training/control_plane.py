"""Bounded rank control plane for training phase convergence.

Design decision 3 of ``decompose-coordexp-swift-training-orchestration``: this
module owns the fixed-frame CPU rank-report transport, rank-report validation
and normalization, phase convergence, resource convergence, and gatherer
cleanup.  The helper bodies below are moved verbatim from
``src/training/pipeline.py`` -- phase names, report frames, timeout and size
bounds, error codes, rank ordering, receipt sinks, and collective order are
protected surfaces and must not change with the move.

The module never imports the facade (``src.training.pipeline``) or the training
session; the import direction is one-way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
import hashlib
import json
import math
import pickle
import re
import shlex
import struct
from typing import Any, TypeVar
import zlib

import torch

from src.artifacts.resources import (
    collect_resource_snapshot,
    converge_rank_cpu_resources,
    merge_resource_high_water,
)
from src.common.errors import RuntimeContractError


T = TypeVar("T")


_STRICT_CACHE_PREPARATION_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "FLASH_ATTENTION_DETERMINISTIC": "1",
}

_RANK_REPORT_MAGIC = b"CRG1"
_RANK_REPORT_HEADER = struct.Struct("!4sQQIIIIQQI")
_RANK_REPORT_MAX_PAYLOAD_BYTES = 64 * 1024
_RANK_REPORT_FRAME_BYTES = _RANK_REPORT_HEADER.size + _RANK_REPORT_MAX_PAYLOAD_BYTES
_RANK_REPORT_CONTROL_TIMEOUT_SECONDS = 120
_CACHE_PREFLIGHT_STRING_LIMIT = 4096
_CACHE_PREFLIGHT_IDENTITY_LIMIT = 128
_CACHE_PREFLIGHT_CODES = frozenset(
    {
        "training.pack_cache_not_prepared",
        "training.pack_cache_immutable_collision",
    }
)
_CACHE_PREFLIGHT_VALIDATION_CATEGORIES = frozenset(
    {
        "expected_target_missing",
        "publication_manifest_missing",
        "publication_manifest_malformed",
        "retired_or_unknown_version",
        "publication_incomplete",
        "semantic_fingerprint_mismatch",
        "required_payload_digest_mismatch",
        "required_payload_invalid",
        "chunk_plan_or_payload_invalid",
        "current_publication_invalid",
    }
)


def _rank_report_value(report: Any, name: str, default: Any) -> Any:
    if isinstance(report, Mapping):
        return report.get(name, default)
    return getattr(report, name, default)


def _rank_report_kind(report: Any) -> str:
    if isinstance(report, Mapping):
        return f"mapping:{report.get('kind', 'unspecified')}"
    report_type = type(report)
    return f"{report_type.__module__}:{report_type.__qualname__}"


def _rank_report_digest(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def _rank_report_header(
    report: Any,
    *,
    sequence: int,
    payload: bytes,
    serialization_status: int,
) -> bytes:
    return _RANK_REPORT_HEADER.pack(
        _RANK_REPORT_MAGIC,
        int(sequence),
        int(_rank_report_value(report, "planned_step_id", 0)),
        int(_rank_report_value(report, "rank", 0)),
        int(_rank_report_value(report, "world_size", 1)),
        int(serialization_status),
        len(payload),
        _rank_report_digest(_rank_report_kind(report)),
        _rank_report_digest(str(_rank_report_value(report, "split", ""))),
        zlib.crc32(payload),
    )


def _unpack_rank_report_header(header: bytes) -> dict[str, int | bytes]:
    (
        magic,
        sequence,
        planned_step_id,
        rank,
        world_size,
        serialization_status,
        payload_size,
        kind_digest,
        split_digest,
        payload_crc32,
    ) = _RANK_REPORT_HEADER.unpack(header)
    return {
        "magic": magic,
        "sequence": sequence,
        "planned_step_id": planned_step_id,
        "rank": rank,
        "world_size": world_size,
        "serialization_status": serialization_status,
        "payload_size": payload_size,
        "kind_digest": kind_digest,
        "split_digest": split_digest,
        "payload_crc32": payload_crc32,
    }


def _all_gather_cpu_bytes(
    distributed: Any,
    payload: bytes,
    *,
    width: int,
    world_size: int,
    group: Any | None,
) -> tuple[bytes, ...]:
    local = torch.zeros(width, dtype=torch.uint8, device="cpu")
    if payload:
        if len(payload) > width:
            raise ValueError("payload exceeds fixed collective width")
        local[: len(payload)] = torch.tensor(tuple(payload), dtype=torch.uint8)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    distributed.all_gather(gathered, local, group=group)
    return tuple(bytes(item.tolist()) for item in gathered)


def _validate_rank_report_headers(
    headers: Sequence[bytes],
    *,
    sequence: int,
    world_size: int,
) -> tuple[dict[str, int | bytes], ...]:
    unpacked = tuple(_unpack_rank_report_header(header) for header in headers)
    if any(header["magic"] != _RANK_REPORT_MAGIC for header in unpacked):
        raise RuntimeContractError(
            "rank report control headers have invalid framing",
            code="runtime.report_gather_framing",
            context={"sequence": sequence},
        )
    observed_sequences = sorted({int(header["sequence"]) for header in unpacked})
    if observed_sequences != [sequence]:
        raise RuntimeContractError(
            "rank report collectives entered in different sequence order",
            code="runtime.report_gather_sequence",
            context={
                "expected_sequence": sequence,
                "observed_sequences": observed_sequences,
            },
        )
    observed_ranks = sorted(int(header["rank"]) for header in unpacked)
    observed_world_sizes = sorted({int(header["world_size"]) for header in unpacked})
    if observed_ranks != list(range(world_size)) or observed_world_sizes != [
        world_size
    ]:
        raise RuntimeContractError(
            "rank report control headers disagree on distributed identity",
            code="runtime.report_gather_ranks",
            context={
                "expected_world_size": world_size,
                "observed_ranks": observed_ranks,
                "observed_world_sizes": observed_world_sizes,
            },
        )
    identity_fields = ("planned_step_id", "kind_digest", "split_digest")
    disagreements = {
        field: sorted({int(header[field]) for header in unpacked})
        for field in identity_fields
        if len({int(header[field]) for header in unpacked}) != 1
    }
    if disagreements:
        raise RuntimeContractError(
            "rank report control headers disagree on report identity",
            code="runtime.report_gather_identity",
            context={"sequence": sequence, "disagreements": disagreements},
        )
    failed_serialization_ranks = [
        int(header["rank"])
        for header in unpacked
        if int(header["serialization_status"]) != 0
    ]
    if failed_serialization_ranks:
        raise RuntimeContractError(
            "one or more ranks could not serialize a rank report",
            code="runtime.report_serialize_failed",
            context={
                "sequence": sequence,
                "failed_ranks": failed_serialization_ranks,
            },
        )
    oversized_ranks = [
        int(header["rank"])
        for header in unpacked
        if int(header["payload_size"]) > _RANK_REPORT_MAX_PAYLOAD_BYTES
    ]
    if oversized_ranks:
        raise RuntimeContractError(
            "rank report payload exceeds the bounded control-plane limit",
            code="runtime.report_gather_size",
            context={
                "sequence": sequence,
                "max_payload_bytes": _RANK_REPORT_MAX_PAYLOAD_BYTES,
                "oversized_ranks": oversized_ranks,
            },
        )
    return unpacked


def _build_rank_report_gatherer(world_size: int) -> Any | None:
    if world_size <= 1:
        return None

    distributed = torch.distributed
    control_group: Any | None = None
    control_group_ready = False
    sequence = 0
    closed = False

    def ensure_control_group() -> Any | None:
        nonlocal control_group, control_group_ready
        if control_group_ready:
            return control_group
        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeContractError(
                "multi-rank finite gates require initialized torch.distributed",
                code="runtime.distributed_gather_uninitialized",
                context={"world_size": world_size},
            )
        observed_world_size = int(distributed.get_world_size())
        if observed_world_size != world_size:
            raise RuntimeContractError(
                "rank report gatherer world size disagrees with torch.distributed",
                code="runtime.report_gather_ranks",
                context={
                    "expected_world_size": world_size,
                    "observed_world_size": observed_world_size,
                },
            )
        backend = str(distributed.get_backend()).lower()
        if "gloo" not in backend:
            is_gloo_available = getattr(distributed, "is_gloo_available", None)
            if callable(is_gloo_available) and not bool(is_gloo_available()):
                raise RuntimeContractError(
                    "bounded rank report gathering requires the gloo backend",
                    code="runtime.report_gather_backend",
                    context={"default_backend": backend},
                )
            control_group = distributed.new_group(
                ranks=list(range(world_size)),
                backend="gloo",
                timeout=timedelta(seconds=_RANK_REPORT_CONTROL_TIMEOUT_SECONDS),
            )
        control_group_ready = True
        return control_group

    def gather(local_report: Any) -> tuple[Any, ...]:
        nonlocal sequence
        if closed:
            raise RuntimeContractError(
                "rank report gatherer is closed",
                code="runtime.report_gather_closed",
                context={"world_size": world_size},
            )
        group = ensure_control_group()
        sequence += 1
        serialization_status = 0
        try:
            payload = pickle.dumps(local_report, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException:
            payload = b""
            serialization_status = 1
        local_header = _rank_report_header(
            local_report,
            sequence=sequence,
            payload=payload,
            serialization_status=serialization_status,
        )
        local_frame = local_header + payload[:_RANK_REPORT_MAX_PAYLOAD_BYTES]
        gathered_frames = _all_gather_cpu_bytes(
            distributed,
            local_frame,
            width=_RANK_REPORT_FRAME_BYTES,
            world_size=world_size,
            group=group,
        )
        headers = _validate_rank_report_headers(
            tuple(frame[: _RANK_REPORT_HEADER.size] for frame in gathered_frames),
            sequence=sequence,
            world_size=world_size,
        )
        reports: list[Any] = []
        for header, gathered_frame in zip(headers, gathered_frames, strict=True):
            payload_size = int(header["payload_size"])
            framed_payload = gathered_frame[
                _RANK_REPORT_HEADER.size : _RANK_REPORT_HEADER.size + payload_size
            ]
            if zlib.crc32(framed_payload) != int(header["payload_crc32"]):
                raise RuntimeContractError(
                    "rank report payload checksum does not match its control header",
                    code="runtime.report_gather_framing",
                    context={
                        "sequence": sequence,
                        "rank": int(header["rank"]),
                    },
                )
            try:
                reports.append(pickle.loads(framed_payload))
            except BaseException as exc:
                raise RuntimeContractError(
                    "rank report payload could not be decoded",
                    code="runtime.report_gather_framing",
                    context={
                        "sequence": sequence,
                        "rank": int(header["rank"]),
                    },
                ) from exc
        return tuple(reports)

    def close() -> None:
        nonlocal closed, control_group, control_group_ready
        if (
            control_group_ready
            and control_group is not None
            and distributed.is_available()
            and distributed.is_initialized()
        ):
            try:
                distributed.destroy_process_group(control_group)
            except BaseException:
                pass
        control_group = None
        control_group_ready = False
        closed = True
        gather.closed = True  # type: ignore[attr-defined]

    gather.close = close  # type: ignore[attr-defined]
    gather.closed = False  # type: ignore[attr-defined]
    return gather


def _build_model_free_preflight_gatherer(world_size: int) -> Any | None:
    """Create a temporary CPU-only convergence group before Accelerate setup."""

    if world_size <= 1:
        return None
    distributed = torch.distributed
    if not distributed.is_available():
        raise RuntimeContractError(
            "distributed cache preflight requires torch.distributed",
            code="runtime.preflight_distributed_unavailable",
            context={"world_size": world_size},
        )
    owns_default_group = not distributed.is_initialized()
    try:
        if owns_default_group:
            is_gloo_available = getattr(distributed, "is_gloo_available", None)
            if callable(is_gloo_available) and not bool(is_gloo_available()):
                raise RuntimeContractError(
                    "distributed cache preflight requires the CPU gloo backend",
                    code="runtime.preflight_distributed_unavailable",
                    context={"world_size": world_size},
                )
            distributed.init_process_group(
                backend="gloo",
                timeout=timedelta(seconds=_RANK_REPORT_CONTROL_TIMEOUT_SECONDS),
            )
        gatherer = _build_rank_report_gatherer(world_size)
        if gatherer is None:
            raise RuntimeContractError(
                "distributed cache preflight gatherer was not constructed",
                code="runtime.preflight_gatherer_unavailable",
                context={"world_size": world_size},
            )
    except BaseException:
        if (
            owns_default_group
            and distributed.is_available()
            and distributed.is_initialized()
        ):
            distributed.destroy_process_group()
        raise
    close_gatherer = getattr(gatherer, "close", None)
    closed = False

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        wrapped.closed = True  # type: ignore[attr-defined]
        try:
            if callable(close_gatherer):
                close_gatherer()
        finally:
            if (
                owns_default_group
                and distributed.is_available()
                and distributed.is_initialized()
            ):
                distributed.destroy_process_group()

    def wrapped(report: Any) -> tuple[Any, ...]:
        return tuple(gatherer(report))

    wrapped.close = close  # type: ignore[attr-defined]
    wrapped.closed = False  # type: ignore[attr-defined]
    return wrapped


def _build_model_free_control_plane(
    *,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
) -> Any:
    """Expose the small accelerator-like surface needed by artifact ownership."""

    def broadcast_object_values(values: list[Any], from_process: int = 0) -> list[Any]:
        if len(values) != 1:
            raise RuntimeContractError(
                "model-free control broadcast accepts one bounded object",
                code="runtime.preflight_broadcast_invalid",
                context={"object_count": len(values)},
            )
        if world_size <= 1:
            return values
        if rank_report_gatherer is None:
            raise RuntimeContractError(
                "model-free control broadcast requires the preflight gatherer",
                code="runtime.preflight_broadcast_unavailable",
                context={"rank": rank, "world_size": world_size},
            )
        reports = tuple(
            rank_report_gatherer(
                {
                    "kind": "model_free_control_broadcast",
                    "planned_step_id": 0,
                    "split": "model_free_control_broadcast",
                    "rank": rank,
                    "world_size": world_size,
                    "payload": values[0] if rank == from_process else None,
                }
            )
        )
        if len(reports) != world_size:
            raise RuntimeContractError(
                "model-free control broadcast did not receive every rank",
                code="runtime.preflight_broadcast_invalid",
                context={
                    "expected_report_count": world_size,
                    "observed_report_count": len(reports),
                },
            )
        source = reports[from_process]
        if (
            not isinstance(source, Mapping)
            or int(source.get("rank", -1)) != from_process
        ):
            raise RuntimeContractError(
                "model-free control broadcast source rank is invalid",
                code="runtime.preflight_broadcast_invalid",
                context={"from_process": from_process},
            )
        values[0] = source.get("payload")
        return values

    return type(
        "ModelFreeControlPlane",
        (),
        {
            "process_index": rank,
            "num_processes": world_size,
            "is_main_process": rank == 0,
            "broadcast_object_list": staticmethod(broadcast_object_values),
        },
    )()


def _bounded_cache_preflight_string(
    value: Any,
    *,
    field: str,
    max_length: int,
) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"invalid bounded cache preflight field: {field}")
    return value


def _normalize_cache_preflight_rank_diagnostic(value: Any) -> dict[str, Any]:
    """Validate the only cache diagnostics permitted on the rank control plane."""

    if not isinstance(value, Mapping):
        raise RuntimeContractError(
            "cache preflight rank diagnostic must be a typed mapping",
            code="runtime.phase_status_invalid",
        )
    try:
        code = _bounded_cache_preflight_string(
            value.get("code"),
            field="code",
            max_length=_CACHE_PREFLIGHT_IDENTITY_LIMIT,
        )
        if code not in _CACHE_PREFLIGHT_CODES:
            raise ValueError("unsupported cache preflight code")
        common_keys = {
            "code",
            "split",
            "cache_root",
            "expected_cache_target",
            "cache_version",
            "fingerprint",
            "validation_category",
            "automatic_recovery",
        }
        expected_keys = (
            common_keys | {"preparation_argv", "preparation_env"}
            if code == "training.pack_cache_not_prepared"
            else common_keys
        )
        if set(value) != expected_keys:
            raise ValueError("cache preflight diagnostic has unknown fields")
        normalized: dict[str, Any] = {
            "code": code,
            "split": _bounded_cache_preflight_string(
                value.get("split"),
                field="split",
                max_length=_CACHE_PREFLIGHT_IDENTITY_LIMIT,
            ),
            "cache_root": _bounded_cache_preflight_string(
                value.get("cache_root"),
                field="cache_root",
                max_length=_CACHE_PREFLIGHT_STRING_LIMIT,
            ),
            "expected_cache_target": _bounded_cache_preflight_string(
                value.get("expected_cache_target"),
                field="expected_cache_target",
                max_length=_CACHE_PREFLIGHT_STRING_LIMIT,
            ),
            "cache_version": _bounded_cache_preflight_string(
                value.get("cache_version"),
                field="cache_version",
                max_length=_CACHE_PREFLIGHT_IDENTITY_LIMIT,
            ),
            "fingerprint": _bounded_cache_preflight_string(
                value.get("fingerprint"),
                field="fingerprint",
                max_length=64,
            ),
            "validation_category": _bounded_cache_preflight_string(
                value.get("validation_category"),
                field="validation_category",
                max_length=_CACHE_PREFLIGHT_IDENTITY_LIMIT,
            ),
            "automatic_recovery": _bounded_cache_preflight_string(
                value.get("automatic_recovery"),
                field="automatic_recovery",
                max_length=_CACHE_PREFLIGHT_IDENTITY_LIMIT,
            ),
        }
        if re.fullmatch(r"[0-9a-f]{64}", normalized["fingerprint"]) is None:
            raise ValueError("cache preflight fingerprint is not canonical")
        if normalized["validation_category"] not in (
            _CACHE_PREFLIGHT_VALIDATION_CATEGORIES
        ):
            raise ValueError("unsupported cache preflight validation category")
        expected_recovery = (
            "single_process_preparation_required"
            if code == "training.pack_cache_not_prepared"
            else "unavailable"
        )
        if normalized["automatic_recovery"] != expected_recovery:
            raise ValueError("cache preflight recovery category disagrees with code")
        if code == "training.pack_cache_not_prepared":
            argv = value.get("preparation_argv")
            if (
                not isinstance(argv, Sequence)
                or isinstance(argv, (str, bytes))
                or len(argv) != 5
            ):
                raise ValueError("cache preparation argv has invalid shape")
            normalized_argv = [
                _bounded_cache_preflight_string(
                    item,
                    field=f"preparation_argv[{index}]",
                    max_length=_CACHE_PREFLIGHT_STRING_LIMIT,
                )
                for index, item in enumerate(argv)
            ]
            if normalized_argv[:4] != [
                "python",
                "-m",
                "src.prepare_train_cache",
                "--config",
            ]:
                raise ValueError("cache preparation argv is not allowlisted")
            environment = value.get("preparation_env")
            if not isinstance(environment, Mapping) or set(environment) != set(
                _STRICT_CACHE_PREPARATION_ENVIRONMENT
            ):
                raise ValueError("cache preparation environment has invalid shape")
            normalized_environment = {
                name: _bounded_cache_preflight_string(
                    environment.get(name),
                    field=f"preparation_env[{name}]",
                    max_length=_CACHE_PREFLIGHT_IDENTITY_LIMIT,
                )
                for name in sorted(_STRICT_CACHE_PREPARATION_ENVIRONMENT)
            }
            if normalized_environment != _STRICT_CACHE_PREPARATION_ENVIRONMENT:
                raise ValueError("cache preparation environment is not allowlisted")
            normalized["preparation_argv"] = normalized_argv
            normalized["preparation_env"] = normalized_environment
        return normalized
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "cache preflight rank diagnostic violates the bounded allowlist",
            code="runtime.phase_status_invalid",
        ) from exc


def _cache_preflight_rank_diagnostic(
    error: RuntimeContractError,
) -> dict[str, Any] | None:
    """Project a local error into a bounded typed diagnostic, never raw context."""

    if error.code not in _CACHE_PREFLIGHT_CODES:
        return None
    context = error.context
    projected: dict[str, Any] = {
        "code": error.code,
        "split": context.get("split"),
        "cache_root": context.get("cache_root"),
        "expected_cache_target": context.get("expected_cache_target"),
        "cache_version": context.get("cache_version"),
        "fingerprint": context.get("fingerprint"),
        "validation_category": context.get("validation_category"),
        "automatic_recovery": context.get("automatic_recovery"),
    }
    if error.code == "training.pack_cache_not_prepared":
        projected["preparation_argv"] = context.get("preparation_argv")
        projected["preparation_env"] = context.get("preparation_env")
    try:
        return _normalize_cache_preflight_rank_diagnostic(projected)
    except RuntimeContractError:
        return None


def _cache_preflight_error_from_rank_diagnostic(
    diagnostic: Mapping[str, Any],
) -> RuntimeContractError:
    normalized = _normalize_cache_preflight_rank_diagnostic(diagnostic)
    code = str(normalized.pop("code"))
    if code == "training.pack_cache_not_prepared":
        preparation_argv = list(normalized.pop("preparation_argv"))
        preparation_env = dict(normalized.pop("preparation_env"))
        preparation_command = shlex.join(
            [
                *(f"{name}={value}" for name, value in preparation_env.items()),
                *preparation_argv,
            ]
        )
        context = {
            **normalized,
            "preparation_argv": preparation_argv,
            "preparation_env": preparation_env,
            "preparation_command": preparation_command,
        }
        return RuntimeContractError(
            "required packing cache is not prepared; "
            f"expected cache target: {context['expected_cache_target']}; "
            f"prepare with: {preparation_command}",
            code=code,
            context=context,
        )
    return RuntimeContractError(
        "required v3 packing cache target is occupied by an invalid immutable "
        f"publication: {normalized['expected_cache_target']}",
        code=code,
        context=normalized,
    )


def _normalize_bounded_phase_details(
    value: Any,
    *,
    phase: str,
    rank: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeContractError(
            "phase rank details must be a bounded mapping",
            code="runtime.phase_status_invalid",
            context={"phase": phase, "rank": rank},
        )
    payload = dict(value)

    def validate(item: Any, *, depth: int) -> None:
        if depth > 6:
            raise ValueError("detail nesting depth")
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if math.isfinite(item):
                return
        elif isinstance(item, str):
            if item.isascii() and len(item) <= 256:
                return
        elif isinstance(item, Mapping):
            if len(item) <= 128 and all(
                isinstance(key, str) and 0 < len(key) <= 64 and key.isascii()
                for key in item
            ):
                for nested in item.values():
                    validate(nested, depth=depth + 1)
                return
        elif isinstance(item, (list, tuple)) and len(item) <= 128:
            for nested in item:
                validate(nested, depth=depth + 1)
            return
        raise ValueError("unbounded detail value")

    try:
        validate(payload, depth=0)
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > 16 * 1024:
            raise ValueError("detail payload bytes")
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "phase rank details exceed the bounded control-plane contract",
            code="runtime.phase_status_invalid",
            context={"phase": phase, "rank": rank},
        ) from exc
    return payload


def _validate_phase_status_reports(
    reports: Sequence[Any],
    *,
    phase: str,
    world_size: int,
) -> tuple[Mapping[str, Any], ...]:
    if len(reports) != world_size:
        raise RuntimeContractError(
            "phase status gather must return exactly one report per world rank",
            code="runtime.phase_status_count",
            context={
                "phase": phase,
                "expected_report_count": world_size,
                "observed_report_count": len(reports),
            },
        )
    by_rank: dict[int, Mapping[str, Any]] = {}
    for index, item in enumerate(reports):
        if not isinstance(item, Mapping):
            raise RuntimeContractError(
                "phase status reports must be mappings",
                code="runtime.phase_status_invalid",
                context={"phase": phase, "report_index": index},
            )
        try:
            item_rank = int(item["rank"])
            item_world_size = int(item["world_size"])
            item_phase = str(item["phase"])
            item_status = str(item["status"])
            item_planned_step_id = int(item["planned_step_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeContractError(
                "phase status report is missing required identity fields",
                code="runtime.phase_status_invalid",
                context={"phase": phase, "report_index": index},
            ) from exc
        if (
            item_rank < 0
            or item_rank >= world_size
            or item_rank in by_rank
            or item_world_size != world_size
            or item_phase != phase
            or item.get("kind") != "phase_status"
            or item.get("split") != phase
            or item_planned_step_id != 0
            or item_status not in {"completed", "failed"}
        ):
            raise RuntimeContractError(
                "phase status reports disagree on rank, world, phase, or status",
                code="runtime.phase_status_identity",
                context={"phase": phase, "report_index": index, "rank": item_rank},
            )
        error_type = item.get("error_type")
        error_code = item.get("error_code")
        if item_status == "failed":
            if (
                not isinstance(error_type, str)
                or not error_type
                or len(error_type) > 128
                or not isinstance(error_code, str)
                or not error_code
                or len(error_code) > 128
            ):
                raise RuntimeContractError(
                    "failed phase status report has invalid bounded error identity",
                    code="runtime.phase_status_invalid",
                    context={"phase": phase, "rank": item_rank},
                )
        elif error_type is not None or error_code is not None:
            raise RuntimeContractError(
                "successful phase status report cannot contain an error identity",
                code="runtime.phase_status_invalid",
                context={"phase": phase, "rank": item_rank},
            )
        normalized: dict[str, Any] = {
            "kind": "phase_status",
            "planned_step_id": 0,
            "split": phase,
            "phase": phase,
            "rank": item_rank,
            "world_size": world_size,
            "status": item_status,
            "error_type": error_type,
            "error_code": error_code,
            "resource_snapshot": merge_resource_high_water(
                dict(item.get("resource_snapshot", {})),
                dict(item.get("resource_snapshot", {})),
            ),
        }
        rank_details = item.get("rank_details")
        if rank_details is not None:
            normalized["rank_details"] = _normalize_bounded_phase_details(
                rank_details,
                phase=phase,
                rank=item_rank,
            )
        diagnostic = item.get("cache_preflight_error")
        if diagnostic is not None:
            if phase != "cache_preflight" or item_status != "failed":
                raise RuntimeContractError(
                    "cache preflight diagnostic appeared on an invalid phase report",
                    code="runtime.phase_status_invalid",
                    context={"phase": phase, "rank": item_rank},
                )
            normalized["cache_preflight_error"] = (
                _normalize_cache_preflight_rank_diagnostic(diagnostic)
            )
        by_rank[item_rank] = normalized
    if tuple(sorted(by_rank)) != tuple(range(world_size)):
        raise RuntimeContractError(
            "phase status gather must contain every world rank exactly once",
            code="runtime.phase_status_identity",
            context={"phase": phase, "observed_ranks": sorted(by_rank)},
        )
    return tuple(by_rank[rank] for rank in range(world_size))


def _run_rank_converged_phase(
    phase: str,
    *,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
    body: Callable[[], Any],
    local_details: Callable[[], Mapping[str, Any] | None] | None = None,
    receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
    resource_collector: Callable[[], Mapping[str, Any]] = collect_resource_snapshot,
) -> Any:
    """Converge caught rank-local Python failures at one common phase boundary.

    This cannot recover a killed rank or a failure inside a mismatched/blocking
    collective; every live rank must still reach this common status gather.
    """

    if rank < 0 or rank >= world_size or world_size <= 0:
        raise RuntimeContractError(
            "rank-converged phase has invalid distributed identity",
            code="runtime.phase_status_identity",
            context={"rank": rank, "world_size": world_size, "phase": phase},
        )
    result: Any = None
    local_error: Exception | None = None
    report: dict[str, Any] = {
        "kind": "phase_status",
        "planned_step_id": 0,
        "split": phase,
        "phase": phase,
        "rank": rank,
        "world_size": world_size,
        "status": "completed",
        "error_type": None,
        "error_code": None,
    }
    try:
        result = body()
    except Exception as exc:
        local_error = exc
        stable_error_code = getattr(exc, "code", "python_exception")
        if not isinstance(stable_error_code, str) or not stable_error_code:
            stable_error_code = "python_exception"
        report.update(
            status="failed",
            error_type=type(exc).__name__[:128],
            error_code=stable_error_code[:128],
        )
        if phase == "cache_preflight" and isinstance(exc, RuntimeContractError):
            diagnostic = _cache_preflight_rank_diagnostic(exc)
            if diagnostic is not None:
                report["cache_preflight_error"] = diagnostic

    report["resource_snapshot"] = dict(resource_collector())
    if local_details is not None:
        detail_payload = local_details()
        if detail_payload is not None:
            report["rank_details"] = dict(detail_payload)

    if world_size == 1:
        reports = (report,)
    else:
        if rank_report_gatherer is None:
            raise RuntimeContractError(
                "multi-rank phase convergence requires the bounded rank report gatherer",
                code="runtime.report_gather_unavailable",
                context={"rank": rank, "world_size": world_size, "phase": phase},
            )
        reports = tuple(rank_report_gatherer(report))
    validated = _validate_phase_status_reports(
        reports,
        phase=phase,
        world_size=world_size,
    )
    rank_resources = converge_rank_cpu_resources(
        [(int(item["rank"]), item["resource_snapshot"]) for item in validated],
        world_size=world_size,
    )
    converged_receipt: dict[str, Any] = {"rank_resources": rank_resources}
    if any("rank_details" in item for item in validated):
        if not all("rank_details" in item for item in validated):
            raise RuntimeContractError(
                "phase rank details must be present on every rank or none",
                code="runtime.phase_status_invalid",
                context={"phase": phase},
            )
        converged_receipt["rank_details"] = {
            str(item["rank"]): item["rank_details"] for item in validated
        }
    failures = [item for item in validated if item["status"] == "failed"]
    primary_error: Exception | None = None
    if failures:
        if world_size == 1 and local_error is not None:
            primary_error = local_error
        elif phase == "cache_preflight":
            canonical = failures[0].get("cache_preflight_error")
            if isinstance(canonical, Mapping):
                primary_error = _cache_preflight_error_from_rank_diagnostic(canonical)
        if primary_error is None:
            failed_ranks = [int(item["rank"]) for item in failures]
            failure_kinds = sorted(
                {f"{item['error_type']}:{item['error_code']}" for item in failures}
            )
            primary_error = RuntimeContractError(
                f"distributed phase {phase} failed on ranks {failed_ranks} "
                f"with {failure_kinds}",
                code="runtime.distributed_phase_failed",
                context={
                    "phase": phase,
                    "failed_ranks": failed_ranks,
                    "failure_kinds": failure_kinds,
                },
            )
    if receipt_sink is not None:
        try:
            receipt_sink(converged_receipt)
        except Exception as sink_error:
            if primary_error is None:
                raise
            sink_code = getattr(sink_error, "code", "python_exception")
            if not isinstance(sink_code, str) or not sink_code:
                sink_code = "python_exception"
            primary_error.add_note(
                "secondary receipt_sink failure: "
                f"{type(sink_error).__name__}:{sink_code[:128]}"
            )
            raise primary_error
    if primary_error is not None:
        raise primary_error
    return result


class RankControlPlane:
    """Own the bounded rank transport for one model-free training entry.

    ``open`` establishes only the existing model-free preflight gatherer.
    ``bind_accelerator`` replaces the transport at the current post-Accelerator
    boundary; it never validates identity eagerly, because the characterized
    contract converges an Accelerate identity mismatch as a phase failure on
    every live rank rather than raising locally on the mismatched rank.
    ``converge`` delegates to the moved phase-convergence boundary unchanged,
    and ``close`` closes the current transport and is idempotent.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        gatherer: Callable[[Any], Sequence[Any]] | None,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.gatherer = gatherer
        self.accelerator: Any | None = None

    @classmethod
    def open(cls, *, rank: int, world_size: int) -> "RankControlPlane":
        return cls(
            rank=rank,
            world_size=world_size,
            gatherer=_build_model_free_preflight_gatherer(world_size),
        )

    def converge(
        self,
        phase: str,
        body: Callable[[], T],
        *,
        local_details: Callable[[], Mapping[str, Any] | None] | None = None,
        receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> T:
        return _run_rank_converged_phase(
            phase,
            rank=self.rank,
            world_size=self.world_size,
            rank_report_gatherer=self.gatherer,
            body=body,
            local_details=local_details,
            receipt_sink=receipt_sink,
        )

    def bind_accelerator(self, accelerator: Any) -> None:
        self.accelerator = accelerator
        self.gatherer = _build_rank_report_gatherer(self.world_size)

    def close(self) -> None:
        close_gatherer = getattr(self.gatherer, "close", None)
        if callable(close_gatherer):
            close_gatherer()
