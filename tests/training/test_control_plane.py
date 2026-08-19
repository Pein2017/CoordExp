"""Contract for the bounded ``RankControlPlane`` rank convergence owner.

Wave 2 of ``decompose-coordexp-swift-training-orchestration`` (tasks 3.3-3.4,
design decision 3).  ``src/training/control_plane.py`` owns the fixed-frame CPU
rank-report transport, report validation/normalization, phase convergence,
resource convergence, and gatherer cleanup that ``src/training/pipeline.py``
held at Wave 0.

Every value asserted here is a *moved* surface: phase names, report schemas,
frame constants, timeout/size bounds, rank ordering, exception selection,
receipt sinks, and collective order must replay exactly as characterized before
the move.  A repoint is never grounds to change an expected value.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
import struct
from types import SimpleNamespace
from typing import Any

import pytest

from src.common.errors import RuntimeContractError
import src.training.control_plane as control_plane
from src.training.control_plane import RankControlPlane


# ---------------------------------------------------------------------------
# Characterized helpers
# ---------------------------------------------------------------------------


def _rank_resource_snapshot(rank: int) -> dict[str, Any]:
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


class ScriptedRankTransport:
    """One in-process stand-in for the bounded rank-report gatherer.

    It records the ordered collective and close events the real transport would
    perform and answers each gather with one report per rank, so the moved
    convergence body is exercised without a live process group.
    """

    def __init__(
        self,
        *,
        world_size: int,
        peer_overrides: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> None:
        self.world_size = world_size
        self.peer_overrides = dict(peer_overrides or {})
        self.events: list[tuple[str, Any]] = []
        self.close_calls = 0
        self.closed = False

    def __call__(self, local_report: Any) -> tuple[Any, ...]:
        if self.closed:
            raise AssertionError("gather after close")
        assert isinstance(local_report, Mapping)
        self.events.append(
            ("all_gather", (local_report.get("kind"), local_report.get("split")))
        )
        reports: list[Mapping[str, Any]] = []
        for rank in range(self.world_size):
            report = {
                **dict(local_report),
                "rank": rank,
                "resource_snapshot": _rank_resource_snapshot(rank),
            }
            report.update(self.peer_overrides.get(rank, {}))
            reports.append(report)
        return tuple(reports)

    def close(self) -> None:
        self.close_calls += 1
        if self.closed:
            return
        self.closed = True
        self.events.append(("close", None))


def _open_plane(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int,
    world_size: int,
    transport: Any,
) -> RankControlPlane:
    monkeypatch.setattr(
        control_plane,
        "_build_model_free_preflight_gatherer",
        lambda observed_world_size: transport,
    )
    plane = RankControlPlane.open(rank=rank, world_size=world_size)
    assert plane.rank == rank
    assert plane.world_size == world_size
    return plane


# ---------------------------------------------------------------------------
# Frozen frame and timeout bounds
# ---------------------------------------------------------------------------


def test_moved_rank_report_frame_constants_are_frozen() -> None:
    assert control_plane._RANK_REPORT_MAGIC == b"CRG1"
    assert control_plane._RANK_REPORT_HEADER.format == "!4sQQIIIIQQI"
    assert control_plane._RANK_REPORT_HEADER.size == struct.calcsize("!4sQQIIIIQQI")
    assert control_plane._RANK_REPORT_MAX_PAYLOAD_BYTES == 64 * 1024
    assert control_plane._RANK_REPORT_FRAME_BYTES == (
        control_plane._RANK_REPORT_HEADER.size
        + control_plane._RANK_REPORT_MAX_PAYLOAD_BYTES
    )
    assert control_plane._RANK_REPORT_CONTROL_TIMEOUT_SECONDS == 120


def test_moved_header_round_trips_identity_and_checksum() -> None:
    report = {
        "kind": "metrics",
        "planned_step_id": 7,
        "split": "eval.forward",
        "rank": 1,
        "world_size": 4,
    }
    payload = b"characterized-payload"

    header = control_plane._rank_report_header(
        report, sequence=3, payload=payload, serialization_status=0
    )
    unpacked = control_plane._unpack_rank_report_header(header)

    assert len(header) == control_plane._RANK_REPORT_HEADER.size
    assert unpacked["magic"] == control_plane._RANK_REPORT_MAGIC
    assert unpacked["sequence"] == 3
    assert unpacked["planned_step_id"] == 7
    assert unpacked["rank"] == 1
    assert unpacked["world_size"] == 4
    assert unpacked["serialization_status"] == 0
    assert unpacked["payload_size"] == len(payload)


def test_moved_header_validation_rejects_an_oversized_payload() -> None:
    headers = tuple(
        control_plane._rank_report_header(
            {"kind": "metrics", "planned_step_id": 1, "rank": rank, "world_size": 2},
            sequence=1,
            payload=b"x" * (control_plane._RANK_REPORT_MAX_PAYLOAD_BYTES + 1),
            serialization_status=0,
        )
        for rank in range(2)
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        control_plane._validate_rank_report_headers(headers, sequence=1, world_size=2)

    assert exc_info.value.code == "runtime.report_gather_size"
    assert exc_info.value.context["max_payload_bytes"] == (
        control_plane._RANK_REPORT_MAX_PAYLOAD_BYTES
    )


def test_moved_header_validation_rejects_a_sequence_disagreement() -> None:
    headers = (
        control_plane._rank_report_header(
            {"kind": "metrics", "planned_step_id": 1, "rank": 0, "world_size": 2},
            sequence=1,
            payload=b"",
            serialization_status=0,
        ),
        control_plane._rank_report_header(
            {"kind": "metrics", "planned_step_id": 1, "rank": 1, "world_size": 2},
            sequence=2,
            payload=b"",
            serialization_status=0,
        ),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        control_plane._validate_rank_report_headers(headers, sequence=1, world_size=2)

    assert exc_info.value.code == "runtime.report_gather_sequence"


# ---------------------------------------------------------------------------
# Single-rank convergence
# ---------------------------------------------------------------------------


def test_single_rank_open_builds_no_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[int] = []
    monkeypatch.setattr(
        control_plane,
        "_build_rank_report_gatherer",
        lambda world_size: built.append(world_size),
    )

    plane = RankControlPlane.open(rank=0, world_size=1)

    assert plane.gatherer is None
    assert built == []
    plane.close()


def test_single_rank_converge_returns_the_body_result_and_exact_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts: list[Mapping[str, Any]] = []
    monkeypatch.setattr(
        control_plane, "collect_resource_snapshot", lambda: _rank_resource_snapshot(0)
    )
    plane = RankControlPlane.open(rank=0, world_size=1)

    result = plane.converge(
        "model_loading",
        lambda: "characterized-body-result",
        local_details=lambda: {"characterization": True},
        receipt_sink=receipts.append,
    )

    assert result == "characterized-body-result"
    assert len(receipts) == 1
    assert receipts[0]["rank_details"] == {"0": {"characterization": True}}
    assert receipts[0]["rank_resources"]["world_size"] == 1


def test_single_rank_failure_reraises_the_local_exception_object() -> None:
    plane = RankControlPlane.open(rank=0, world_size=1)
    primary = RuntimeContractError(
        "primary cache admission failure",
        code="training.primary_cache_failure",
    )

    def fail_body() -> None:
        raise primary

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge("cache_preflight", fail_body)

    assert exc_info.value is primary


def test_receipt_sink_failure_never_replaces_the_primary_error() -> None:
    plane = RankControlPlane.open(rank=0, world_size=1)
    primary = RuntimeContractError(
        "primary cache admission failure",
        code="training.primary_cache_failure",
    )

    def fail_body() -> None:
        raise primary

    def fail_receipt_sink(receipt: Mapping[str, Any]) -> None:
        assert receipt
        raise RuntimeContractError(
            "secondary receipt persistence failure",
            code="run_writer.secondary_receipt_failure",
        )

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge("cache_preflight", fail_body, receipt_sink=fail_receipt_sink)

    assert exc_info.value is primary
    assert any(
        "secondary receipt_sink failure" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )


def test_converge_rejects_an_invalid_distributed_identity() -> None:
    plane = RankControlPlane(rank=2, world_size=2, gatherer=None)

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge("cache_preflight", lambda: None)

    assert exc_info.value.code == "runtime.phase_status_identity"


# ---------------------------------------------------------------------------
# Multi-rank convergence, ordering, and error selection
# ---------------------------------------------------------------------------


def test_multi_rank_success_preserves_rank_order_and_converges_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = ScriptedRankTransport(world_size=4)
    receipts: list[Mapping[str, Any]] = []
    monkeypatch.setattr(
        control_plane, "collect_resource_snapshot", lambda: _rank_resource_snapshot(0)
    )
    plane = _open_plane(monkeypatch, rank=0, world_size=4, transport=transport)

    result = plane.converge(
        "cache_preflight",
        lambda: "ok",
        local_details=lambda: {"phase_trace": {"step": 1}},
        receipt_sink=receipts.append,
    )

    assert result == "ok"
    assert transport.events == [("all_gather", ("phase_status", "cache_preflight"))]
    assert list(receipts[0]["rank_details"]) == ["0", "1", "2", "3"]
    assert receipts[0]["rank_resources"]["world_size"] == 4
    assert receipts[0]["rank_resources"]["per_rank"] == {
        str(rank): _rank_resource_snapshot(rank)["cpu"] for rank in range(4)
    }
    assert receipts[0]["rank_resources"]["global_maxima"] == {
        "scope": "all_rank_deterministic_maximum",
        "max_rss_bytes": 250,
        "io_read_bytes": 70,
        "io_write_bytes": 35,
    }


def test_multi_rank_convergence_requires_a_transport() -> None:
    plane = RankControlPlane(rank=0, world_size=2, gatherer=None)

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge("cache_preflight", lambda: None)

    assert exc_info.value.code == "runtime.report_gather_unavailable"


def test_multi_rank_failure_converges_to_the_declared_distributed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = ScriptedRankTransport(
        world_size=2,
        peer_overrides={
            1: {
                "status": "failed",
                "error_type": "RuntimeContractError",
                "error_code": "runtime.preflight_accelerate_identity_mismatch",
            }
        },
    )
    plane = _open_plane(monkeypatch, rank=0, world_size=2, transport=transport)

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge("accelerator_runtime_preflight", lambda: None)

    assert exc_info.value.code == "runtime.distributed_phase_failed"
    assert exc_info.value.context == {
        "phase": "accelerator_runtime_preflight",
        "failed_ranks": [1],
        "failure_kinds": [
            "RuntimeContractError:runtime.preflight_accelerate_identity_mismatch"
        ],
    }


def test_cache_preflight_failure_selects_the_canonical_rank_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = {
        "code": "training.pack_cache_not_prepared",
        "split": "train",
        "cache_root": "/cache-root",
        "expected_cache_target": "/cache-root/v3/abc",
        "cache_version": "v3",
        "fingerprint": "a" * 64,
        "validation_category": "expected_target_missing",
        "automatic_recovery": "single_process_preparation_required",
        "preparation_argv": [
            "python",
            "-m",
            "src.prepare_train_cache",
            "--config",
            "/config.yaml",
        ],
        "preparation_env": dict(control_plane._STRICT_CACHE_PREPARATION_ENVIRONMENT),
    }
    failed = {
        "status": "failed",
        "error_type": "RuntimeContractError",
        "error_code": "training.pack_cache_not_prepared",
        "cache_preflight_error": diagnostic,
    }
    transport = ScriptedRankTransport(
        world_size=2, peer_overrides={0: failed, 1: failed}
    )
    plane = _open_plane(monkeypatch, rank=0, world_size=2, transport=transport)

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge("cache_preflight", lambda: None)

    assert exc_info.value.code == "training.pack_cache_not_prepared"
    assert exc_info.value.context["expected_cache_target"] == "/cache-root/v3/abc"
    assert exc_info.value.context["preparation_env"] == (
        control_plane._STRICT_CACHE_PREPARATION_ENVIRONMENT
    )
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in (
        exc_info.value.context["preparation_command"]
    )


def test_phase_details_must_be_present_on_every_rank_or_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = ScriptedRankTransport(
        world_size=2, peer_overrides={1: {"rank_details": None}}
    )
    plane = _open_plane(monkeypatch, rank=0, world_size=2, transport=transport)

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge(
            "cache_preflight", lambda: None, local_details=lambda: {"trace": 1}
        )

    assert exc_info.value.code == "runtime.phase_status_invalid"


def test_phase_details_beyond_the_bounded_contract_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = ScriptedRankTransport(world_size=2)
    plane = _open_plane(monkeypatch, rank=0, world_size=2, transport=transport)

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge(
            "cache_preflight",
            lambda: None,
            local_details=lambda: {"blob": "x" * 512},
        )

    assert exc_info.value.code == "runtime.phase_status_invalid"


def test_phase_status_reports_must_cover_every_rank_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = ScriptedRankTransport(world_size=2, peer_overrides={1: {"rank": 0}})
    plane = _open_plane(monkeypatch, rank=0, world_size=2, transport=transport)

    with pytest.raises(RuntimeContractError) as exc_info:
        plane.converge("cache_preflight", lambda: None)

    assert exc_info.value.code == "runtime.phase_status_identity"


# ---------------------------------------------------------------------------
# Accelerator binding and close
# ---------------------------------------------------------------------------


def test_bind_accelerator_replaces_the_transport_with_the_admitted_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = ScriptedRankTransport(world_size=2)
    post = ScriptedRankTransport(world_size=2)
    built: list[int] = []

    def build_rank_report_gatherer(world_size: int) -> Any:
        built.append(world_size)
        return post

    monkeypatch.setattr(
        control_plane, "_build_rank_report_gatherer", build_rank_report_gatherer
    )
    plane = _open_plane(monkeypatch, rank=0, world_size=2, transport=preflight)
    assert plane.gatherer is preflight

    accelerator = SimpleNamespace(process_index=0, num_processes=2)
    plane.bind_accelerator(accelerator)

    assert built == [2]
    assert plane.gatherer is post
    assert plane.accelerator is accelerator
    assert preflight.events == []


def test_bind_accelerator_never_raises_on_a_mismatched_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mismatch must converge as a phase failure, not raise locally.

    The characterized two-rank contract requires every live rank to observe
    ``runtime.distributed_phase_failed`` with the mismatching rank named, which
    is only reachable if binding itself stays silent.
    """

    post = ScriptedRankTransport(world_size=2)
    monkeypatch.setattr(
        control_plane, "_build_rank_report_gatherer", lambda world_size: post
    )
    plane = _open_plane(
        monkeypatch, rank=0, world_size=2, transport=ScriptedRankTransport(world_size=2)
    )

    plane.bind_accelerator(SimpleNamespace(process_index=1, num_processes=8))

    assert plane.rank == 0
    assert plane.world_size == 2
    assert plane.gatherer is post


def test_ordered_collectives_replay_open_converge_bind_converge_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = ScriptedRankTransport(world_size=2)
    post = ScriptedRankTransport(world_size=2)
    monkeypatch.setattr(
        control_plane, "_build_rank_report_gatherer", lambda world_size: post
    )
    plane = _open_plane(monkeypatch, rank=0, world_size=2, transport=preflight)

    plane.converge("config_provenance_resolution", lambda: None)
    plane.converge("cache_preflight", lambda: None)
    plane.close()
    plane.bind_accelerator(SimpleNamespace(process_index=0, num_processes=2))
    plane.converge("accelerator_runtime_preflight", lambda: None)
    plane.close()

    assert preflight.events == [
        ("all_gather", ("phase_status", "config_provenance_resolution")),
        ("all_gather", ("phase_status", "cache_preflight")),
        ("close", None),
    ]
    assert post.events == [
        ("all_gather", ("phase_status", "accelerator_runtime_preflight")),
        ("close", None),
    ]


def test_close_is_idempotent_over_the_real_model_free_cleanup_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Distributed:
        def __init__(self) -> None:
            self.initialized = False
            self.init_calls = 0
            self.destroy_calls = 0

        def is_available(self) -> bool:
            return True

        def is_initialized(self) -> bool:
            return self.initialized

        def is_gloo_available(self) -> bool:
            return True

        def init_process_group(self, **kwargs: Any) -> None:
            assert kwargs["backend"] == "gloo"
            assert kwargs["timeout"] == timedelta(
                seconds=control_plane._RANK_REPORT_CONTROL_TIMEOUT_SECONDS
            )
            self.init_calls += 1
            self.initialized = True

        def destroy_process_group(self) -> None:
            self.destroy_calls += 1
            self.initialized = False

    distributed = _Distributed()
    inner = ScriptedRankTransport(world_size=2)
    monkeypatch.setattr(control_plane.torch, "distributed", distributed)
    monkeypatch.setattr(
        control_plane, "_build_rank_report_gatherer", lambda world_size: inner
    )

    plane = RankControlPlane.open(rank=0, world_size=2)
    assert distributed.init_calls == 1

    plane.close()
    plane.close()
    plane.close()

    assert distributed.destroy_calls == 1
    assert inner.close_calls == 1
    assert inner.events == [("close", None)]


def test_close_on_an_absent_transport_is_a_no_op() -> None:
    plane = RankControlPlane(rank=0, world_size=1, gatherer=None)

    plane.close()
    plane.close()

    assert plane.gatherer is None


def test_open_destroys_an_owned_group_when_the_gatherer_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Distributed:
        def __init__(self) -> None:
            self.initialized = False
            self.destroy_calls = 0

        def is_available(self) -> bool:
            return True

        def is_initialized(self) -> bool:
            return self.initialized

        def is_gloo_available(self) -> bool:
            return True

        def init_process_group(self, **kwargs: Any) -> None:
            self.initialized = True

        def destroy_process_group(self) -> None:
            self.destroy_calls += 1
            self.initialized = False

    distributed = _Distributed()
    monkeypatch.setattr(control_plane.torch, "distributed", distributed)
    monkeypatch.setattr(
        control_plane, "_build_rank_report_gatherer", lambda world_size: None
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        RankControlPlane.open(rank=0, world_size=2)

    assert exc_info.value.code == "runtime.preflight_gatherer_unavailable"
    assert distributed.destroy_calls == 1
    assert distributed.initialized is False


# ---------------------------------------------------------------------------
# Model-free control surface
# ---------------------------------------------------------------------------


def test_model_free_control_plane_exposes_the_bounded_accelerator_surface() -> None:
    transport = ScriptedRankTransport(world_size=2)
    surface = control_plane._build_model_free_control_plane(
        rank=0, world_size=2, rank_report_gatherer=transport
    )

    assert surface.process_index == 0
    assert surface.num_processes == 2
    assert surface.is_main_process is True
    assert surface.broadcast_object_list([{"payload": 1}]) == [{"payload": 1}]
    assert transport.events == [
        (
            "all_gather",
            ("model_free_control_broadcast", "model_free_control_broadcast"),
        )
    ]


def test_model_free_control_plane_rejects_an_unbounded_broadcast() -> None:
    surface = control_plane._build_model_free_control_plane(
        rank=0, world_size=2, rank_report_gatherer=None
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        surface.broadcast_object_list([1, 2])

    assert exc_info.value.code == "runtime.preflight_broadcast_invalid"


def test_control_plane_owner_never_imports_the_facade_or_session() -> None:
    import ast
    from pathlib import Path

    path = Path(control_plane.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    assert "src.training.pipeline" not in modules
    assert "src.training.session" not in modules


def test_converge_signature_is_the_declared_bounded_interface() -> None:
    import inspect

    signature = inspect.signature(RankControlPlane.converge)

    assert list(signature.parameters) == [
        "self",
        "phase",
        "body",
        "local_details",
        "receipt_sink",
    ]
    assert signature.parameters["local_details"].default is None
    assert signature.parameters["receipt_sink"].default is None
    assert signature.parameters["local_details"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["receipt_sink"].kind is inspect.Parameter.KEYWORD_ONLY
    open_signature = inspect.signature(RankControlPlane.open)
    assert list(open_signature.parameters) == ["rank", "world_size"]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in open_signature.parameters.values()
    )


def test_moved_helpers_are_reachable_only_from_the_new_owner() -> None:
    import src.training.pipeline as pipeline

    for name in (
        "_run_rank_converged_phase",
        "_build_rank_report_gatherer",
        "_build_model_free_preflight_gatherer",
        "_build_model_free_control_plane",
        "_validate_phase_status_reports",
        "_normalize_bounded_phase_details",
        "_all_gather_cpu_bytes",
        "_rank_report_header",
        "_unpack_rank_report_header",
        "_validate_rank_report_headers",
    ):
        assert hasattr(control_plane, name), name
        assert not hasattr(pipeline, name), name
