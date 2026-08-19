from __future__ import annotations

import copy
import json
import multiprocessing as mp
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue
import socket
import time
from types import SimpleNamespace
from typing import Protocol, Sequence

import pytest
import torch.distributed as dist

from src.common.errors import RuntimeContractError
from src.training import control_plane, execution_plan, pack_cache, pipeline
from src.training.pack_cache import (
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    PACKING_CACHE_VERSION,
    cache_dir_for_fingerprint,
    write_micro_step_cache,
)
from src.training.supervised_trainer import SupervisedMicroStep


MATERIALIZATION = {
    "strategy": PACKING_CACHE_MATERIALIZATION_STRATEGY,
    "workers": 1,
}
AUGMENTATION = {
    "split": "train",
    "mode": "disabled",
    "policy": "geometry_flips",
    "enabled": False,
    "seed": 7,
    "input_example_count": 1,
    "output_example_count": 1,
    "presentation_count": 1,
    "object_ordering": "source_order",
}


def _synthetic_registry_determinants(purpose: str) -> dict[str, object]:
    semantic: dict[str, object] = {
        "version": PACKING_CACHE_VERSION,
        "split": "train",
        "dataset": {"purpose": purpose},
        "template": {"purpose": purpose},
        "packing": {"purpose": purpose},
        "processor": {"purpose": purpose},
        "ordering": {"purpose": purpose},
        "augmentation": dict(AUGMENTATION),
        "qwen": {
            "processor_identity": {"purpose": purpose},
            "token_identity": {"purpose": purpose},
            "encoding_identity": {"purpose": purpose},
            "model_config_assets": {"purpose": purpose},
            "processor_assets": {"purpose": purpose},
            "tokenizer_assets": {"purpose": purpose},
        },
        "realized_vocab_groups": {"purpose": purpose},
        "micro_step_runtime_config": {
            "fa2_model_dtype": "no",
            "capture_fa2_branch": False,
            "require_fa2_branch_proof": False,
        },
        "micro_step_schema": pack_cache._supervised_micro_step_schema_identity(),
    }
    entries = pack_cache._build_determinant_entries(semantic)
    return {
        **semantic,
        "registry_schema_version": (
            pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION
        ),
        "determinants": entries,
        "aggregate_fingerprint": pack_cache._registry_entries_fingerprint(entries),
        "code_identity": pack_cache._registry_code_identity(entries),
    }


DETERMINANTS = _synthetic_registry_determinants("model-free-preflight")
FINGERPRINT = str(DETERMINANTS["aggregate_fingerprint"])
_WORLD_SIZE = 2
_DISTRIBUTED_TEST_DEADLINE_SECONDS = 150.0
_DISTRIBUTED_TEST_POLL_SECONDS = 0.1
_DISTRIBUTED_TEST_TERMINATE_GRACE_SECONDS = 5.0


class _RankProcess(Protocol):
    @property
    def pid(self) -> int | None: ...

    @property
    def exitcode(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...


class _ResultQueue(Protocol):
    def get(self, block: bool = True, timeout: float | None = None) -> object: ...

    def get_nowait(self) -> object: ...


def _runtime_baseline_receipt() -> dict[str, object]:
    return {
        "schema_version": 3,
        "baseline_sha256": "a" * 64,
        "attention_backend": "flash_attention_2",
        "admitted": True,
        "mismatches": [],
        "reference_only": {},
    }


def _have_gloo() -> bool:
    if not dist.is_available():
        return False
    available = getattr(dist, "is_gloo_available", None)
    return True if available is None else bool(available())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _reap_spawned_rank_processes(
    ranked_processes: Sequence[tuple[int, _RankProcess]],
    output: _ResultQueue,
    *,
    description: str,
    expected_results: int,
    deadline_seconds: float = _DISTRIBUTED_TEST_DEADLINE_SECONDS,
    poll_seconds: float = _DISTRIBUTED_TEST_POLL_SECONDS,
) -> list[dict[str, object]]:
    deadline = time.monotonic() + deadline_seconds
    messages: list[dict[str, object]] = []
    timed_out = False
    try:
        while True:
            while len(messages) < expected_results:
                try:
                    message = output.get_nowait()
                except Empty:
                    break
                if not isinstance(message, dict):
                    raise AssertionError(
                        f"{description} returned a non-mapping result: {message!r}"
                    )
                messages.append(message)

            alive: list[tuple[int, _RankProcess]] = []
            for rank, process in ranked_processes:
                process.join(timeout=0)
                if process.is_alive():
                    alive.append((rank, process))
            if not alive and len(messages) >= expected_results:
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                message = output.get(timeout=min(poll_seconds, remaining))
            except Empty:
                continue
            if not isinstance(message, dict):
                raise AssertionError(
                    f"{description} returned a non-mapping result: {message!r}"
                )
            messages.append(message)
    except BaseException:
        for _, process in ranked_processes:
            if process.is_alive():
                process.terminate()
        for _, process in ranked_processes:
            process.join(timeout=_DISTRIBUTED_TEST_TERMINATE_GRACE_SECONDS)
        raise

    if timed_out:
        before_cleanup = [
            {
                "rank": rank,
                "pid": process.pid,
                "exitcode": process.exitcode,
                "alive": process.is_alive(),
            }
            for rank, process in ranked_processes
        ]
        for _, process in ranked_processes:
            if process.is_alive():
                process.terminate()
        for _, process in ranked_processes:
            process.join(timeout=_DISTRIBUTED_TEST_TERMINATE_GRACE_SECONDS)
        after_cleanup = [
            {
                "rank": rank,
                "pid": process.pid,
                "exitcode": process.exitcode,
                "alive": process.is_alive(),
            }
            for rank, process in ranked_processes
        ]
        pytest.fail(
            f"{description} timed out after {deadline_seconds:.1f}s; "
            f"results={len(messages)}/{expected_results}; "
            f"before_cleanup={before_cleanup}; after_cleanup={after_cleanup}"
        )
    return messages


def test_rank_process_reaper_polls_past_legacy_one_pass_join_window() -> None:
    class _ScriptedDelayedProcess:
        def __init__(self, *, rank: int, extra_polls_before_exit: int) -> None:
            self.pid = 10_000 + rank
            self.exitcode: int | None = None
            self._alive = True
            self._extra_polls_before_exit = extra_polls_before_exit
            self.join_calls: list[float | None] = []
            self.terminated = False

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)
            if not self._alive:
                return
            if self._extra_polls_before_exit > 0:
                self._extra_polls_before_exit -= 1
                return
            self._alive = False
            self.exitcode = 0

        def terminate(self) -> None:
            self.terminated = True
            self._alive = False
            self.exitcode = -15

    output: Queue[object] = Queue()
    output.put({"rank": 0, "status": "done"})
    output.put({"rank": 1, "status": "done"})
    immediate = _ScriptedDelayedProcess(rank=0, extra_polls_before_exit=0)
    delayed = _ScriptedDelayedProcess(rank=1, extra_polls_before_exit=1)

    messages = _reap_spawned_rank_processes(
        [(0, immediate), (1, delayed)],
        output,
        description="scripted delayed worker",
        expected_results=2,
        deadline_seconds=0.1,
        poll_seconds=0.001,
    )

    assert messages == [
        {"rank": 0, "status": "done"},
        {"rank": 1, "status": "done"},
    ]
    assert len(delayed.join_calls) >= 2
    assert immediate.terminated is False
    assert delayed.terminated is False


def _micro_step() -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack="pack",
        encoded_examples=("example",),
        position_inputs="positions",
        token_sequence="tokens",
        vocab_groups="vocab",
        metadata={"pack_id": 0, "augmentation_receipt": dict(AUGMENTATION)},
    )


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            name="preflight-run",
            artifact_root=str(tmp_path / "artifacts"),
            output_dir="run",
            collision_policy="fail",
        ),
        runtime=SimpleNamespace(
            seed=17,
            determinism=SimpleNamespace(mode="legacy"),
        ),
        training=SimpleNamespace(precision="no"),
        model=SimpleNamespace(attn_implementation="flash_attention_2"),
        data=SimpleNamespace(train=object(), eval=None),
    )


def _peer_rank_report(report: object, *, rank: int) -> dict[str, object]:
    peer = copy.deepcopy(report)
    peer["rank"] = rank
    details = peer.get("rank_details")
    if isinstance(details, dict):
        launcher = details.get("launcher")
        if isinstance(launcher, dict):
            launcher.update(
                rank=rank,
                local_rank=rank,
                logical_cuda_device=rank,
            )
        assert details.get("runtime_determinism") == dict(report).get(  # type: ignore[arg-type]
            "rank_details"
        ).get("runtime_determinism")
    return peer


def _install_pipeline_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    model_load_calls: list[bool],
    accelerator_calls: list[str],
) -> Path:
    config_path = (tmp_path / "config.yaml").resolve()
    config_path.write_text("test: true\n", encoding="utf-8")
    config = _config(tmp_path)
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=config_path,
        to_artifact_dict=lambda: {"test": True},
    )
    token_identity = SimpleNamespace(tokenizer_vocab_size=32)
    components = SimpleNamespace(token_identity=token_identity, tokenizer=object())
    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", str(tmp_path / "cache-root"))
    monkeypatch.setattr(execution_plan, "load_train_config", lambda path: resolved)
    monkeypatch.setattr(
        pipeline,
        "collect_execution_provenance",
        lambda **kwargs: {"schema_version": 1},
    )
    monkeypatch.setattr(
        pipeline,
        "require_pinned_runtime_baseline",
        lambda **kwargs: _runtime_baseline_receipt(),
    )

    def load_components(config: object, *, load_model: bool) -> object:
        model_load_calls.append(load_model)
        if load_model:
            raise AssertionError("model loader must not run before cache admission")
        return components

    monkeypatch.setattr(pipeline, "load_qwen_components", load_components)
    monkeypatch.setattr(
        pipeline, "build_token_vocabulary_groups", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        pipeline, "resolve_qwen_runtime_controls", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        pipeline,
        "build_packing_cache_fingerprint",
        lambda *args, **kwargs: FINGERPRINT,
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_planned_step_schedule",
        lambda *args, **kwargs: SimpleNamespace(
            resolved_max_steps=1,
            runtime_batch=SimpleNamespace(
                world_size=int(kwargs["world_size"]),
                resolved_grad_accum_steps=1,
                effective_batch_size=int(kwargs["world_size"]),
            ),
        ),
    )

    def build_accelerator(precision: str) -> object:
        accelerator_calls.append(precision)
        raise AssertionError("Accelerator must not be constructed before admission")

    monkeypatch.setattr(pipeline, "_build_accelerator", build_accelerator)
    return config_path


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [
        ("0", None),
        (None, "1"),
        ("", "1"),
        (" 0", "1"),
        ("+0", "1"),
        ("00", "1"),
        ("0", ""),
        ("0", "01"),
        ("0", "1.0"),
        ("rank", "2"),
        ("-1", "2"),
        ("2", "2"),
        ("0", "0"),
    ],
)
def test_partial_or_malformed_launcher_identity_fails_before_any_preflight_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank: str | None,
    world_size: str | None,
) -> None:
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    config_path = _install_pipeline_fakes(
        monkeypatch,
        tmp_path=tmp_path,
        model_load_calls=model_load_calls,
        accelerator_calls=accelerator_calls,
    )
    if rank is None:
        monkeypatch.delenv("RANK", raising=False)
    else:
        monkeypatch.setenv("RANK", rank)
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    construction_calls: list[str] = []
    monkeypatch.setattr(
        control_plane,
        "_build_model_free_preflight_gatherer",
        lambda world: construction_calls.append("control-plane"),
    )
    monkeypatch.setattr(
        pipeline,
        "_initialize_model_free_run_owner",
        lambda **kwargs: construction_calls.append("writer"),
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_model_free_training_preflight",
        lambda **kwargs: construction_calls.append("cache"),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run_training_pipeline(config_path)

    assert exc_info.value.code == "runtime.preflight_launch_identity_invalid"
    assert construction_calls == []
    assert model_load_calls == []
    assert accelerator_calls == []
    assert not (tmp_path / "artifacts").exists()


def test_direct_launch_defaults_only_when_both_identity_fields_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert execution_plan._resolve_model_free_launch_identity() == (0, 1)


def test_pinned_runtime_baseline_admission_broadcasts_rank_zero_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "schema_version": 3,
        "baseline_sha256": "b" * 64,
        "attention_backend": "flash_attention_2",
        "admitted": True,
        "mismatches": [],
        "reference_only": {
            "ms-swift": {
                "matches_recorded_reference": True,
                "mismatches": [],
            }
        },
    }
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        pipeline,
        "require_pinned_runtime_baseline",
        lambda **kwargs: calls.append(dict(kwargs)) or receipt,
    )

    def gather(report: object) -> tuple[object, object]:
        return report, {**dict(report), "rank": 1}  # type: ignore[arg-type]

    observed = pipeline._resolve_shared_pinned_runtime_baseline(
        {"schema_version": 1},
        attention_backend="flash_attention_2",
        rank=0,
        world_size=2,
        rank_report_gatherer=gather,
    )

    assert observed == receipt
    assert calls == [
        {
            "provenance": {"schema_version": 1},
            "attention_backend": "flash_attention_2",
        }
    ]


def test_pinned_runtime_baseline_admission_rejects_a_drifted_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "require_pinned_runtime_baseline",
        lambda **kwargs: _runtime_baseline_receipt(),
    )

    def gather(report: object) -> tuple[object, object]:
        peer = {
            **dict(report),  # type: ignore[arg-type]
            "rank": 1,
            "status": "failed",
            "error_type": "RuntimeContractError",
            "error_code": "runtime.pinned_runtime_baseline_rejected",
            "rank_details": {
                "admission_status": "failed",
                "baseline": None,
            },
        }
        return report, peer

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._resolve_shared_pinned_runtime_baseline(
            {"schema_version": 1},
            attention_backend="flash_attention_2",
            rank=0,
            world_size=2,
            rank_report_gatherer=gather,
        )

    assert exc_info.value.code == "runtime.distributed_phase_failed"
    assert exc_info.value.context["failed_ranks"] == [1]


def test_pinned_runtime_baseline_failure_precedes_cache_model_and_accelerator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    config_path = _install_pipeline_fakes(
        monkeypatch,
        tmp_path=tmp_path,
        model_load_calls=model_load_calls,
        accelerator_calls=accelerator_calls,
    )
    monkeypatch.setattr(
        pipeline,
        "require_pinned_runtime_baseline",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("baseline drift")),
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_model_free_training_preflight",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cache preflight must not run after baseline rejection")
        ),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run_training_pipeline(config_path)

    assert exc_info.value.code == "runtime.pinned_runtime_baseline_rejected"
    assert model_load_calls == []
    assert accelerator_calls == []
    state = json.loads(
        (tmp_path / "artifacts" / "run" / "run.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "failed"
    assert state["measurement"]["terminal_phase"] == ("config_provenance_resolution")
    assert (
        state["measurement"]["phases"]["config_provenance_resolution"]["status"]
        == "failed"
    )


def test_exact_resume_lineage_is_admitted_and_converged_before_run_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = (tmp_path / "checkpoint").resolve()
    manifest = SimpleNamespace(
        parent_run_id="parent-run",
        parent_segment_id="parent-segment",
        checkpoint_step=3,
        continuation_index=1,
        world_size=2,
        aggregate_digest="a" * 64,
    )
    monkeypatch.setattr(pipeline, "load_training_state_manifest", lambda path: manifest)
    monkeypatch.setattr(pipeline, "_file_sha256", lambda path: "b" * 64)

    def gather(report: object) -> tuple[object, object]:
        return report, {**dict(report), "rank": 1}  # type: ignore[arg-type]

    lineage = pipeline._resolve_resume_continuation_lineage(
        SimpleNamespace(
            resume=SimpleNamespace(
                mode="exact_same_world_size",
                checkpoint_dir=str(checkpoint_dir),
            )
        ),
        rank=0,
        world_size=2,
        rank_report_gatherer=gather,
    )

    assert lineage == {
        "parent_run_id": "parent-run",
        "parent_segment_id": "parent-segment",
        "parent_checkpoint_identity": {
            "resolved_path": str(checkpoint_dir),
            "checkpoint_step": 3,
            "training_state_manifest_file_sha256": "b" * 64,
            "training_state_aggregate_digest": "a" * 64,
        },
        "parent_continuation_index": 1,
        "continuation_index": 2,
    }


def test_exact_resume_lineage_rejects_wrong_world_before_run_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "load_training_state_manifest",
        lambda path: SimpleNamespace(world_size=2),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._resolve_resume_continuation_lineage(
            SimpleNamespace(
                resume=SimpleNamespace(
                    mode="exact_same_world_size",
                    checkpoint_dir=str(tmp_path / "checkpoint"),
                )
            ),
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
        )

    assert exc_info.value.code == "training.resume_world_size_mismatch"


def test_provider_resolution_requires_one_exact_receipt_across_launcher_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", raising=False)

    def gather(report: object) -> tuple[object, object]:
        peer = {**dict(report), "rank": 1}  # type: ignore[arg-type]
        peer_details = dict(peer["rank_details"])
        peer_resolution = dict(peer_details["resolution"])
        peer_resolution.update(
            resolved_mode="overlapped",
            provider_disposition="overlapped",
            input_build_owner="provider_producer_cpu",
            lookahead_depth=1,
        )
        peer_details["resolution"] = peer_resolution
        peer["rank_details"] = peer_details
        return report, peer

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._resolve_converged_forward_input_provider_mode(
            "synchronous",
            rank=0,
            world_size=2,
            rank_report_gatherer=gather,
        )

    assert exc_info.value.code == (
        "training.forward_input_provider_resolution_mismatch"
    )
    assert sorted(exc_info.value.context["rank_resolutions"]) == ["0", "1"]


def test_provider_resolution_accepts_exact_same_mode_source_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", "overlapped")
    captured: list[dict[str, object]] = []

    def gather(report: object) -> tuple[object, object]:
        return report, {**dict(report), "rank": 1}  # type: ignore[arg-type]

    resolved = pipeline._resolve_converged_forward_input_provider_mode(
        "synchronous",
        rank=0,
        world_size=2,
        rank_report_gatherer=gather,
        receipt_sink=lambda receipt: captured.append(dict(receipt)),
    )

    assert resolved.resolved_mode == "overlapped"
    assert resolved.source == "deprecated_environment_override"
    assert captured[0]["rank_details"]["0"] == captured[0]["rank_details"]["1"]


def test_eval_reduction_rejects_default_vs_explicit_same_effective_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", raising=False)

    def gather(report: object) -> tuple[object, object]:
        peer = {**dict(report), "rank": 1}  # type: ignore[arg-type]
        peer_details = dict(peer["rank_details"])
        peer_details["resolution"] = {
            "control": "replicated",
            "effective_mode": "replicated",
            "source": "COORDEXP_SWIFT_EVAL_REDUCTION_MODE",
            "pack_count": 1,
            "world_size": 2,
        }
        peer["rank_details"] = peer_details
        return report, peer

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._resolve_converged_eval_reduction_receipt(
            pack_count=1,
            rank=0,
            world_size=2,
            rank_report_gatherer=gather,
        )

    assert exc_info.value.code == "training.eval_reduction_resolution_mismatch"


def test_invalid_provider_override_fails_shared_artifact_before_cache_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    config_path = _install_pipeline_fakes(
        monkeypatch,
        tmp_path=tmp_path,
        model_load_calls=model_load_calls,
        accelerator_calls=accelerator_calls,
    )
    monkeypatch.setenv("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", "invalid-provider")

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run_training_pipeline(config_path)

    assert exc_info.value.code == "training.forward_input_provider_mode_invalid"
    assert model_load_calls == []
    assert accelerator_calls == []
    run_state = json.loads(
        (tmp_path / "artifacts" / "run" / "run.json").read_text(encoding="utf-8")
    )
    assert run_state["status"] == "failed"
    assert run_state["measurement"]["terminal_phase"] == (
        "config_provenance_resolution"
    )
    phase = run_state["measurement"]["phases"]["config_provenance_resolution"]
    assert phase["status"] == "failed"
    assert phase["rank_details"]["0"]["resolution_status"] == "failed"
    assert phase["rank_details"]["0"]["error_code"] == (
        "training.forward_input_provider_mode_invalid"
    )


def test_post_writer_handshake_failure_finalizes_one_artifact_and_closes_gatherer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    config_path = _install_pipeline_fakes(
        monkeypatch,
        tmp_path=tmp_path,
        model_load_calls=model_load_calls,
        accelerator_calls=accelerator_calls,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")

    class _HandshakeGatherer:
        calls = 0
        closed = False

        def __call__(self, report: object) -> tuple[object, object]:
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("secret-shaped post-init broadcast failure")
            peer = _peer_rank_report(report, rank=1)
            peer["payload"] = None
            return report, peer

        def close(self) -> None:
            self.closed = True

    gatherer = _HandshakeGatherer()
    monkeypatch.setattr(
        control_plane,
        "_build_model_free_preflight_gatherer",
        lambda world_size: gatherer,
    )

    with pytest.raises(RuntimeError, match="post-init broadcast failure"):
        pipeline.run_training_pipeline(config_path)

    assert gatherer.calls == 3
    assert gatherer.closed is True
    assert model_load_calls == []
    assert accelerator_calls == []
    run_files = list((tmp_path / "artifacts").rglob("run.json"))
    assert run_files == [tmp_path / "artifacts" / "run" / "run.json"]
    state = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["completed_steps"] == 0
    assert state["consumed_packs"] == 0
    assert state["checkpoint_event_count"] == 0
    assert state["terminal_error"] == (
        "artifact initialization handshake failed: RuntimeError"
    )
    assert "secret-shaped" not in json.dumps(state)


def test_cache_rank_report_projects_only_bounded_allowlisted_context_into_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    config_path = _install_pipeline_fakes(
        monkeypatch,
        tmp_path=tmp_path,
        model_load_calls=model_load_calls,
        accelerator_calls=accelerator_calls,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    captured_phase_reports: list[dict[str, object]] = []

    class _Gatherer:
        calls = 0
        closed = False

        def __call__(self, report: object) -> tuple[object, object]:
            self.calls += 1
            local = dict(report)  # type: ignore[arg-type]
            if local.get("kind") == "model_free_control_broadcast":
                peer = {**local, "rank": 1, "payload": None}
                return report, peer
            captured_phase_reports.append(local)
            peer = _peer_rank_report(report, rank=1)
            peer.update(status="completed", error_type=None, error_code=None)
            peer.pop("cache_preflight_error", None)
            return report, peer

        def close(self) -> None:
            self.closed = True

    gatherer = _Gatherer()
    monkeypatch.setattr(
        control_plane,
        "_build_model_free_preflight_gatherer",
        lambda world_size: gatherer,
    )
    expected_target = cache_dir_for_fingerprint(
        (tmp_path / "cache-root").resolve(), FINGERPRINT
    )
    oversized_secret = "secret-shaped:" + "x" * 100_000

    def fail_cache_preflight(**kwargs: object) -> object:
        raise RuntimeContractError(
            oversized_secret,
            code="training.pack_cache_not_prepared",
            context={
                "split": "train",
                "cache_root": str((tmp_path / "cache-root").resolve()),
                "expected_cache_target": str(expected_target),
                "cache_version": PACKING_CACHE_VERSION,
                "fingerprint": FINGERPRINT,
                "validation_category": "expected_target_missing",
                "automatic_recovery": "single_process_preparation_required",
                "preparation_argv": [
                    "python",
                    "-m",
                    "src.prepare_train_cache",
                    "--config",
                    str(config_path),
                ],
                "preparation_env": {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "FLASH_ATTENTION_DETERMINISTIC": "1",
                },
                "secret_token": oversized_secret,
                "unknown": {"nested": oversized_secret},
            },
        )

    monkeypatch.setattr(
        pipeline, "_resolve_model_free_training_preflight", fail_cache_preflight
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run_training_pipeline(config_path)

    assert exc_info.value.code == "training.pack_cache_not_prepared"
    assert "secret-shaped" not in str(exc_info.value)
    assert "secret_token" not in exc_info.value.context
    assert gatherer.closed is True
    assert [report["phase"] for report in captured_phase_reports] == [
        "config_provenance_resolution",
        "upstream_runtime_baseline_admission",
        "profile_sync_timing_resolution",
        "config_provenance_resolution",
        "cache_preflight",
    ]
    serialized_report = json.dumps(captured_phase_reports)
    assert len(serialized_report) < 10_000
    assert "secret-shaped" not in serialized_report
    assert "secret_token" not in serialized_report
    run_state = json.loads(
        (tmp_path / "artifacts" / "run" / "run.json").read_text(encoding="utf-8")
    )
    assert run_state["status"] == "failed"
    assert run_state["policy_identities"]["upstream_runtime_baseline"] == (
        _runtime_baseline_receipt()
    )
    assert "secret-shaped" not in json.dumps(run_state)
    assert "secret_token" not in json.dumps(run_state)
    assert model_load_calls == []
    assert accelerator_calls == []


def _damage_expected_target(tmp_path: Path, damage: str) -> Path:
    cache_root = tmp_path / "cache-root"
    cache_dir = cache_dir_for_fingerprint(cache_root, FINGERPRINT)
    if damage == "missing":
        return cache_dir
    if damage == "stale":
        retired_dir = (
            tmp_path / "cache-root" / "coordexp-swift-pack-cache-v2" / FINGERPRINT
        )
        retired_dir.mkdir(parents=True)
        (retired_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "version": "coordexp-swift-pack-cache-v2",
                    "status": "complete",
                    "fingerprint": FINGERPRINT,
                }
            ),
            encoding="utf-8",
        )
        return cache_dir
    cache_dir.mkdir(parents=True)
    if damage == "unpublished":
        return cache_dir
    if damage == "immutable_collision":
        (cache_dir / "manifest.json").write_text("{broken", encoding="utf-8")
        return cache_dir
    if damage == "corrupt":
        cache_dir.rmdir()
        manifest = write_micro_step_cache(
            cache_dir,
            (_micro_step(),),
            cache_root=cache_root,
            fingerprint=FINGERPRINT,
            determinants=DETERMINANTS,
            materialization=MATERIALIZATION,
            determinant_revalidator=lambda: DETERMINANTS,
            augmentation=AUGMENTATION,
        )
        chunk = cache_dir / manifest["chunks"][0]["path"]
        chunk.write_bytes(chunk.read_bytes() + b"corrupt")
        return cache_dir
    raise AssertionError(f"unknown damage: {damage}")


def test_pre_writer_failure_closes_preflight_gatherer_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    config_path = _install_pipeline_fakes(
        monkeypatch,
        tmp_path=tmp_path,
        model_load_calls=model_load_calls,
        accelerator_calls=accelerator_calls,
    )
    close_calls: list[str] = []
    gatherer = SimpleNamespace(close=lambda: close_calls.append("closed"))
    monkeypatch.setattr(
        control_plane,
        "_build_model_free_preflight_gatherer",
        lambda world_size: gatherer,
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_shared_run_directory",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("run-directory failure before writer initialization")
        ),
    )

    with pytest.raises(RuntimeError, match="before writer initialization"):
        pipeline.run_training_pipeline(config_path)

    assert close_calls == ["closed"]
    assert model_load_calls == []
    assert accelerator_calls == []
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("builder_outcome", ["raises", "returns_none"])
def test_preflight_gatherer_construction_failure_destroys_owned_group(
    monkeypatch: pytest.MonkeyPatch,
    builder_outcome: str,
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

        def init_process_group(self, **kwargs: object) -> None:
            self.init_calls += 1
            self.initialized = True

        def destroy_process_group(self) -> None:
            self.destroy_calls += 1
            self.initialized = False

    distributed = _Distributed()
    monkeypatch.setattr(control_plane.torch, "distributed", distributed)
    if builder_outcome == "raises":
        monkeypatch.setattr(
            control_plane,
            "_build_rank_report_gatherer",
            lambda world_size: (_ for _ in ()).throw(
                RuntimeError("rank-report gatherer construction failed")
            ),
        )
        expected_exception: type[BaseException] = RuntimeError
    else:
        monkeypatch.setattr(
            control_plane, "_build_rank_report_gatherer", lambda world_size: None
        )
        expected_exception = RuntimeContractError

    with pytest.raises(expected_exception):
        control_plane._build_model_free_preflight_gatherer(2)

    assert distributed.init_calls == 1
    assert distributed.destroy_calls == 1
    assert distributed.initialized is False


@pytest.mark.parametrize(
    ("damage", "expected_code", "category", "expected_terminal_phase"),
    [
        (
            "missing",
            "training.pack_cache_not_prepared",
            "expected_target_missing",
            "cache_publication_admission",
        ),
        (
            "stale",
            "training.pack_cache_not_prepared",
            "expected_target_missing",
            "cache_publication_admission",
        ),
        (
            "unpublished",
            "training.pack_cache_immutable_collision",
            "publication_manifest_missing",
            "cache_publication_admission",
        ),
        (
            "corrupt",
            "training.pack_cache_immutable_collision",
            "required_payload_digest_mismatch",
            "train_rank_hydration",
        ),
        (
            "immutable_collision",
            "training.pack_cache_immutable_collision",
            "publication_manifest_malformed",
            "cache_publication_admission",
        ),
    ],
)
def test_each_required_cache_failure_is_before_model_and_accelerator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    expected_code: str,
    category: str,
    expected_terminal_phase: str,
) -> None:
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    config_path = _install_pipeline_fakes(
        monkeypatch,
        tmp_path=tmp_path,
        model_load_calls=model_load_calls,
        accelerator_calls=accelerator_calls,
    )
    cache_dir = _damage_expected_target(tmp_path, damage)

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline.run_training_pipeline(config_path)

    error = exc_info.value
    assert error.code == expected_code
    assert error.context["cache_root"] == str((tmp_path / "cache-root").resolve())
    assert error.context["expected_cache_target"] == str(cache_dir)
    assert error.context["cache_version"] == PACKING_CACHE_VERSION
    assert error.context["fingerprint"] == FINGERPRINT
    assert error.context["validation_category"] == category
    assert model_load_calls == [False]
    assert accelerator_calls == []
    run_state = json.loads(
        (tmp_path / "artifacts" / "run" / "run.json").read_text(encoding="utf-8")
    )
    assert run_state["status"] == "failed"
    assert run_state["measurement"]["terminal_phase"] == expected_terminal_phase
    assert run_state["measurement"]["steady_state_eligible"] is False
    if expected_code == "training.pack_cache_not_prepared":
        expected_argv = [
            "python",
            "-m",
            "src.prepare_train_cache",
            "--config",
            str(config_path),
        ]
        assert error.context["preparation_argv"] == expected_argv
        assert error.context["preparation_env"] == {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
        }
        assert error.context["preparation_command"] == (
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "FLASH_ATTENTION_DETERMINISTIC=1 " + " ".join(expected_argv)
        )
    else:
        assert error.context["automatic_recovery"] == "unavailable"
        assert "preparation_command" not in error.context
        assert "src.prepare_train_cache" not in str(error)


@pytest.mark.parametrize(
    "damage", ["missing", "stale", "unpublished", "corrupt", "immutable_collision"]
)
def test_direct_and_distributed_cache_failure_errors_are_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    config_path = _install_pipeline_fakes(
        monkeypatch,
        tmp_path=tmp_path,
        model_load_calls=model_load_calls,
        accelerator_calls=accelerator_calls,
    )
    config = _config(tmp_path)
    _damage_expected_target(tmp_path, damage)

    def body(
        *,
        world_size: int,
        rank_report_gatherer: object | None = None,
    ) -> object:
        return pipeline._resolve_model_free_training_preflight(
            config=config,
            config_path=config_path,
            repo_root=tmp_path,
            rank=0,
            world_size=world_size,
            rank_report_gatherer=rank_report_gatherer,
        )

    with pytest.raises(RuntimeContractError) as direct_info:
        control_plane._run_rank_converged_phase(
            "cache_preflight",
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
            body=lambda: body(world_size=1),
        )

    def gather(report: object) -> tuple[object, object]:
        peer = {**dict(report), "rank": 1}  # type: ignore[arg-type]
        return report, peer

    with pytest.raises(RuntimeContractError) as distributed_info:
        control_plane._run_rank_converged_phase(
            "cache_preflight",
            rank=0,
            world_size=2,
            rank_report_gatherer=gather,
            body=lambda: body(world_size=2, rank_report_gatherer=gather),
        )

    assert distributed_info.value.code == direct_info.value.code
    assert distributed_info.value.message == direct_info.value.message
    assert distributed_info.value.context == direct_info.value.context
    assert model_load_calls == [False, False]
    assert accelerator_calls == []


def _distributed_preflight_failure_worker(
    rank: int,
    port: int,
    root: str,
    output: mp.Queue,
) -> None:
    task_root = Path(root)
    config_path = (task_root / "config.yaml").resolve()
    config = _config(task_root)
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=config_path,
        to_artifact_dict=lambda: {"test": True},
    )
    model_load_calls: list[bool] = []
    accelerator_calls: list[str] = []
    token_identity = SimpleNamespace(tokenizer_vocab_size=32)
    components = SimpleNamespace(token_identity=token_identity, tokenizer=object())
    execution_plan.load_train_config = lambda path: resolved
    pipeline.collect_execution_provenance = lambda **kwargs: {"schema_version": 1}
    pipeline.require_pinned_runtime_baseline = (
        lambda **kwargs: _runtime_baseline_receipt()
    )

    def load_components(config: object, *, load_model: bool) -> object:
        model_load_calls.append(load_model)
        if load_model:
            raise AssertionError("model loader must not run before cache admission")
        return components

    pipeline.load_qwen_components = load_components
    pipeline.build_token_vocabulary_groups = lambda *args, **kwargs: object()
    pipeline.resolve_qwen_runtime_controls = lambda *args, **kwargs: object()
    pipeline.build_packing_cache_fingerprint = lambda *args, **kwargs: FINGERPRINT
    pipeline.resolve_planned_step_schedule = lambda *args, **kwargs: SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=_WORLD_SIZE,
            resolved_grad_accum_steps=1,
            effective_batch_size=_WORLD_SIZE,
        ),
    )

    def build_accelerator(precision: str) -> object:
        accelerator_calls.append(precision)
        raise AssertionError("Accelerator must not be constructed before admission")

    pipeline._build_accelerator = build_accelerator
    try:
        import os

        os.environ.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(_WORLD_SIZE),
                "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(task_root / "cache-root"),
            }
        )
        try:
            pipeline.run_training_pipeline(config_path)
        except RuntimeContractError as exc:
            output.put(
                {
                    "rank": rank,
                    "code": exc.code,
                    "message": exc.message,
                    "context": dict(exc.context),
                    "model_load_calls": model_load_calls,
                    "accelerator_calls": accelerator_calls,
                    "group_initialized_after": dist.is_initialized(),
                }
            )
        else:
            output.put({"rank": rank, "unexpected_success": True})
    except BaseException as exc:
        output.put({"rank": rank, "worker_error": f"{type(exc).__name__}: {exc}"})


def _distributed_provider_resolution_mismatch_worker(
    rank: int,
    port: int,
    root: str,
    output: mp.Queue,
) -> None:
    task_root = Path(root)
    config_path = (task_root / "config.yaml").resolve()
    config = _config(task_root)
    config.training.forward_input_provider_mode = "synchronous"
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=config_path,
        to_artifact_dict=lambda: {"test": True},
    )
    execution_plan.load_train_config = lambda path: resolved
    pipeline.collect_execution_provenance = lambda **kwargs: {"schema_version": 1}
    pipeline.require_pinned_runtime_baseline = (
        lambda **kwargs: _runtime_baseline_receipt()
    )
    pipeline._resolve_model_free_training_preflight = lambda **kwargs: (
        (_ for _ in ()).throw(
            AssertionError("cache admission must not run after provider mismatch")
        )
    )
    pipeline._build_accelerator = lambda precision: (
        (_ for _ in ()).throw(
            AssertionError("Accelerator must not run after provider mismatch")
        )
    )
    try:
        import os

        os.environ.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(_WORLD_SIZE),
                "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(task_root / "cache-root"),
            }
        )
        os.environ.pop("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", None)
        if rank == 1:
            # Same resolved mode, different source: exact receipt convergence
            # must reject even a semantically no-op rank-local override.
            os.environ["COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE"] = "synchronous"
        try:
            pipeline.run_training_pipeline(config_path)
        except RuntimeContractError as exc:
            output.put(
                {
                    "rank": rank,
                    "code": exc.code,
                    "message": exc.message,
                    "context": dict(exc.context),
                    "group_initialized_after": dist.is_initialized(),
                }
            )
        else:
            output.put({"rank": rank, "unexpected_success": True})
    except BaseException as exc:
        output.put({"rank": rank, "worker_error": f"{type(exc).__name__}: {exc}"})


def _distributed_preflight_success_worker(
    rank: int,
    preflight_port: int,
    accelerator_port: int,
    root: str,
    output: mp.Queue,
) -> None:
    task_root = Path(root)
    config_path = (task_root / "config.yaml").resolve()
    config = _config(task_root)
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=config_path,
        to_artifact_dict=lambda: {"test": True},
    )
    token_identity = SimpleNamespace(tokenizer_vocab_size=32)
    components = SimpleNamespace(token_identity=token_identity, tokenizer=object())
    observations: dict[str, object] = {}
    execution_plan.load_train_config = lambda path: resolved
    pipeline.collect_execution_provenance = lambda **kwargs: {"schema_version": 1}
    pipeline.require_pinned_runtime_baseline = (
        lambda **kwargs: _runtime_baseline_receipt()
    )
    pipeline.load_qwen_components = lambda config, *, load_model: (
        (_ for _ in ()).throw(AssertionError("model load must remain stubbed"))
        if load_model
        else components
    )
    pipeline.build_token_vocabulary_groups = lambda *args, **kwargs: object()
    pipeline.resolve_qwen_runtime_controls = lambda *args, **kwargs: object()
    pipeline.build_packing_cache_fingerprint = lambda *args, **kwargs: FINGERPRINT
    pipeline.resolve_planned_step_schedule = lambda *args, **kwargs: SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=_WORLD_SIZE,
            resolved_grad_accum_steps=1,
            effective_batch_size=_WORLD_SIZE,
        ),
    )

    def build_accelerator(precision: str) -> object:
        observations["group_initialized_at_accelerator"] = dist.is_initialized()
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{accelerator_port}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=10),
        )
        return SimpleNamespace(
            process_index=rank,
            num_processes=_WORLD_SIZE,
            is_main_process=rank == 0,
        )

    def run_initialized_training(**kwargs: object) -> dict[str, object]:
        accelerator = kwargs["accelerator"]
        assert int(accelerator.process_index) == rank  # type: ignore[union-attr]
        assert int(accelerator.num_processes) == _WORLD_SIZE  # type: ignore[union-attr]
        observations["group_initialized_at_transition"] = dist.is_initialized()
        writer = kwargs["writer"]
        if writer is not None:
            writer.finalize(  # type: ignore[union-attr]
                status="completed",
                updated_at="completed",
                completed_steps=0,
                consumed_packs=0,
                checkpoint_event_count=0,
                optimizer_update_status=None,
                finite_status=None,
            )
        return {"rank": rank, "transition": "completed"}

    pipeline._build_accelerator = build_accelerator
    pipeline.validate_accelerator_runtime = lambda *args, **kwargs: None
    pipeline._run_initialized_training = run_initialized_training
    try:
        import os

        os.environ.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(preflight_port),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(_WORLD_SIZE),
                "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(task_root / "cache-root"),
            }
        )
        result = pipeline.run_training_pipeline(config_path)
        if dist.is_initialized():
            dist.destroy_process_group()
        output.put(
            {
                "rank": rank,
                "result": result,
                **observations,
                "group_initialized_after": dist.is_initialized(),
            }
        )
    except BaseException as exc:
        if dist.is_initialized():
            dist.destroy_process_group()
        output.put({"rank": rank, "worker_error": f"{type(exc).__name__}: {exc}"})


def _distributed_accelerator_identity_mismatch_worker(
    rank: int,
    preflight_port: int,
    accelerator_port: int,
    root: str,
    output: mp.Queue,
) -> None:
    task_root = Path(root)
    config_path = (task_root / "config.yaml").resolve()
    config = _config(task_root)
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=config_path,
        to_artifact_dict=lambda: {"test": True},
    )
    model_load_calls: list[bool] = []
    gatherer_build_world_sizes: list[int] = []
    gatherer_close_world_sizes: list[int] = []
    token_identity = SimpleNamespace(tokenizer_vocab_size=32)
    components = SimpleNamespace(token_identity=token_identity, tokenizer=object())
    execution_plan.load_train_config = lambda path: resolved
    pipeline.collect_execution_provenance = lambda **kwargs: {"schema_version": 1}
    pipeline.require_pinned_runtime_baseline = (
        lambda **kwargs: _runtime_baseline_receipt()
    )

    def load_components(config: object, *, load_model: bool) -> object:
        model_load_calls.append(load_model)
        if load_model:
            raise AssertionError("model loader must not run on identity mismatch")
        return components

    pipeline.load_qwen_components = load_components
    pipeline.build_token_vocabulary_groups = lambda *args, **kwargs: object()
    pipeline.resolve_qwen_runtime_controls = lambda *args, **kwargs: object()
    pipeline.build_packing_cache_fingerprint = lambda *args, **kwargs: FINGERPRINT
    pipeline.resolve_planned_step_schedule = lambda *args, **kwargs: SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=_WORLD_SIZE,
            resolved_grad_accum_steps=1,
            effective_batch_size=_WORLD_SIZE,
        ),
    )
    real_build_rank_report_gatherer = control_plane._build_rank_report_gatherer

    def build_rank_report_gatherer(world_size: int) -> object:
        gatherer_build_world_sizes.append(world_size)
        gatherer = real_build_rank_report_gatherer(world_size)
        if gatherer is None:
            raise AssertionError("two-rank test requires a rank report gatherer")
        real_close = gatherer.close

        def close() -> None:
            real_close()
            gatherer_close_world_sizes.append(world_size)

        gatherer.close = close
        return gatherer

    control_plane._build_rank_report_gatherer = build_rank_report_gatherer

    def build_accelerator(precision: str) -> object:
        if dist.is_initialized():
            raise AssertionError("temporary preflight group survived Accelerator setup")
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{accelerator_port}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=10),
        )
        return SimpleNamespace(
            process_index=rank,
            num_processes=_WORLD_SIZE if rank == 0 else _WORLD_SIZE + 1,
            is_main_process=rank == 0,
            distributed_type=SimpleNamespace(name="MULTI_GPU"),
            mixed_precision=precision,
            gradient_accumulation_steps=1,
        )

    pipeline._build_accelerator = build_accelerator
    pipeline._run_initialized_training = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("training assembly must not run on identity mismatch")
    )
    try:
        import os

        os.environ.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(preflight_port),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(_WORLD_SIZE),
                "COORDEXP_SWIFT_PACK_CACHE_ROOT": str(task_root / "cache-root"),
            }
        )
        try:
            pipeline.run_training_pipeline(config_path)
        except RuntimeContractError as exc:
            if dist.is_initialized():
                dist.destroy_process_group()
            output.put(
                {
                    "rank": rank,
                    "code": exc.code,
                    "message": exc.message,
                    "context": dict(exc.context),
                    "model_load_calls": model_load_calls,
                    "gatherer_build_world_sizes": gatherer_build_world_sizes,
                    "gatherer_close_world_sizes": gatherer_close_world_sizes,
                    "group_initialized_after": dist.is_initialized(),
                }
            )
        else:
            output.put({"rank": rank, "unexpected_success": True})
    except BaseException as exc:
        if dist.is_initialized():
            dist.destroy_process_group()
        output.put({"rank": rank, "worker_error": f"{type(exc).__name__}: {exc}"})


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_rank_one_required_payload_failure_converges_and_tears_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.yaml").write_text("test: true\n", encoding="utf-8")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    cache_root = tmp_path / "cache-root"
    cache_dir = cache_dir_for_fingerprint(cache_root, FINGERPRINT)
    manifest = write_micro_step_cache(
        cache_dir,
        (_micro_step(), _micro_step()),
        cache_root=cache_root,
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        determinant_revalidator=lambda: DETERMINANTS,
        augmentation=AUGMENTATION,
        chunk_size=1,
    )
    rank_one_chunk = cache_dir / manifest["chunks"][1]["path"]
    rank_one_chunk.write_bytes(rank_one_chunk.read_bytes() + b"rank-one-corrupt")
    context = mp.get_context("spawn")
    output: mp.Queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_distributed_preflight_failure_worker,
            args=(rank, port, str(tmp_path), output),
            daemon=False,
        )
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    messages = _reap_spawned_rank_processes(
        list(enumerate(processes)),
        output,
        description="distributed preflight failure",
        expected_results=_WORLD_SIZE,
    )
    assert [process.exitcode for process in processes] == [0, 0]
    assert not [message for message in messages if "worker_error" in message]
    by_rank = {int(message["rank"]): message for message in messages}
    assert sorted(by_rank) == [0, 1]
    assert (
        by_rank[0]["code"]
        == by_rank[1]["code"]
        == ("training.pack_cache_immutable_collision")
    )
    assert by_rank[0]["message"] == by_rank[1]["message"]
    assert by_rank[0]["context"] == by_rank[1]["context"]
    assert by_rank[0]["context"]["validation_category"] == (
        "required_payload_digest_mismatch"
    )
    assert by_rank[0]["model_load_calls"] == [False]
    assert by_rank[1]["model_load_calls"] == [False]
    assert by_rank[0]["accelerator_calls"] == []
    assert by_rank[1]["accelerator_calls"] == []
    assert by_rank[0]["group_initialized_after"] is False
    assert by_rank[1]["group_initialized_after"] is False
    run_files = list((tmp_path / "artifacts").rglob("run.json"))
    assert run_files == [tmp_path / "artifacts" / "run" / "run.json"]
    run_state = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert run_state["status"] == "failed"
    assert run_state["measurement"]["terminal_phase"] == "train_rank_hydration"
    assert run_state["measurement"]["steady_state_eligible"] is False


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_provider_env_source_mismatch_fails_before_cache_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.yaml").write_text("test: true\n", encoding="utf-8")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    context = mp.get_context("spawn")
    output: mp.Queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_distributed_provider_resolution_mismatch_worker,
            args=(rank, port, str(tmp_path), output),
            daemon=False,
        )
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    messages = _reap_spawned_rank_processes(
        list(enumerate(processes)),
        output,
        description="distributed provider resolution mismatch",
        expected_results=_WORLD_SIZE,
    )
    assert [process.exitcode for process in processes] == [0, 0]
    assert not [message for message in messages if "worker_error" in message]
    by_rank = {int(message["rank"]): message for message in messages}
    assert sorted(by_rank) == [0, 1]
    assert (
        by_rank[0]["code"]
        == by_rank[1]["code"]
        == ("training.forward_input_provider_resolution_mismatch")
    )
    assert by_rank[0]["context"] == by_rank[1]["context"]
    assert by_rank[0]["group_initialized_after"] is False
    assert by_rank[1]["group_initialized_after"] is False
    run_state = json.loads(
        (tmp_path / "artifacts" / "run" / "run.json").read_text(encoding="utf-8")
    )
    assert run_state["status"] == "failed"
    assert run_state["measurement"]["terminal_phase"] == (
        "config_provenance_resolution"
    )
    phase = run_state["measurement"]["phases"]["config_provenance_resolution"]
    assert phase["status"] == "failed"
    assert phase["rank_details"]["0"]["resolution"]["source"] == "strict_config"
    assert phase["rank_details"]["1"]["resolution"]["source"] == (
        "deprecated_environment_override"
    )


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_preflight_success_tears_down_gloo_before_accelerator_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.yaml").write_text("test: true\n", encoding="utf-8")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    cache_root = tmp_path / "cache-root"
    cache_dir = cache_dir_for_fingerprint(cache_root, FINGERPRINT)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(), _micro_step()),
        cache_root=cache_root,
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        determinant_revalidator=lambda: DETERMINANTS,
        augmentation=AUGMENTATION,
        chunk_size=1,
    )
    context = mp.get_context("spawn")
    output: mp.Queue = context.Queue()
    preflight_port = _free_port()
    accelerator_port = _free_port()
    while accelerator_port == preflight_port:
        accelerator_port = _free_port()
    processes = [
        context.Process(
            target=_distributed_preflight_success_worker,
            args=(
                rank,
                preflight_port,
                accelerator_port,
                str(tmp_path),
                output,
            ),
            daemon=False,
        )
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    messages = _reap_spawned_rank_processes(
        list(enumerate(processes)),
        output,
        description="distributed preflight success transition",
        expected_results=_WORLD_SIZE,
    )
    assert [process.exitcode for process in processes] == [0, 0]
    assert not [message for message in messages if "worker_error" in message]
    by_rank = {int(message["rank"]): message for message in messages}
    assert sorted(by_rank) == [0, 1]
    for rank, message in by_rank.items():
        assert message["result"] == {"rank": rank, "transition": "completed"}
        assert message["group_initialized_at_accelerator"] is False
        assert message["group_initialized_at_transition"] is True
        assert message["group_initialized_after"] is False
    run_files = list((tmp_path / "artifacts").rglob("run.json"))
    assert run_files == [tmp_path / "artifacts" / "run" / "run.json"]
    run_state = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert run_state["status"] == "completed"
    assert run_state["measurement"]["phases"]["cache_admission"]["status"] == (
        "completed"
    )


@pytest.mark.skipif(not _have_gloo(), reason="requires torch.distributed gloo backend")
def test_two_rank_rank_one_accelerator_identity_mismatch_converges_and_tears_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.yaml").write_text("test: true\n", encoding="utf-8")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    cache_root = tmp_path / "cache-root"
    cache_dir = cache_dir_for_fingerprint(cache_root, FINGERPRINT)
    write_micro_step_cache(
        cache_dir,
        (_micro_step(), _micro_step()),
        cache_root=cache_root,
        fingerprint=FINGERPRINT,
        determinants=DETERMINANTS,
        materialization=MATERIALIZATION,
        determinant_revalidator=lambda: DETERMINANTS,
        augmentation=AUGMENTATION,
        chunk_size=1,
    )
    context = mp.get_context("spawn")
    output: mp.Queue = context.Queue()
    preflight_port = _free_port()
    accelerator_port = _free_port()
    while accelerator_port == preflight_port:
        accelerator_port = _free_port()
    processes = [
        context.Process(
            target=_distributed_accelerator_identity_mismatch_worker,
            args=(
                rank,
                preflight_port,
                accelerator_port,
                str(tmp_path),
                output,
            ),
            daemon=False,
        )
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    messages = _reap_spawned_rank_processes(
        list(enumerate(processes)),
        output,
        description="distributed Accelerator identity mismatch",
        expected_results=_WORLD_SIZE,
    )
    assert [process.exitcode for process in processes] == [0, 0]
    assert not [message for message in messages if "worker_error" in message]
    by_rank = {int(message["rank"]): message for message in messages}
    assert sorted(by_rank) == [0, 1]
    assert (
        by_rank[0]["code"] == by_rank[1]["code"] == ("runtime.distributed_phase_failed")
    )
    assert by_rank[0]["message"] == by_rank[1]["message"]
    assert by_rank[0]["context"] == by_rank[1]["context"]
    assert by_rank[0]["context"] == {
        "phase": "accelerator_runtime_preflight",
        "failed_ranks": [1],
        "failure_kinds": [
            "RuntimeContractError:runtime.preflight_accelerate_identity_mismatch"
        ],
    }
    for message in by_rank.values():
        assert message["model_load_calls"] == [False]
        assert message["gatherer_build_world_sizes"] == [_WORLD_SIZE, _WORLD_SIZE]
        assert message["gatherer_close_world_sizes"] == [_WORLD_SIZE, _WORLD_SIZE]
        assert message["group_initialized_after"] is False
    run_files = list((tmp_path / "artifacts").rglob("run.json"))
    assert run_files == [tmp_path / "artifacts" / "run" / "run.json"]
    run_state = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert run_state["status"] == "failed"
    assert run_state["measurement"]["terminal_phase"] == "cache_admission"
    assert run_state["measurement"]["phases"]["cache_admission"]["status"] == ("failed")
    assert (
        "runtime.preflight_accelerate_identity_mismatch" in run_state["terminal_error"]
    )


def test_rank_converged_phase_preserves_primary_failure_when_receipt_sink_fails() -> (
    None
):
    primary = RuntimeContractError(
        "primary cache admission failure",
        code="training.primary_cache_failure",
    )

    def fail_body() -> None:
        raise primary

    def fail_receipt_sink(receipt: object) -> None:
        assert receipt
        raise RuntimeContractError(
            "secondary receipt persistence failure",
            code="run_writer.secondary_receipt_failure",
        )

    with pytest.raises(RuntimeContractError) as exc_info:
        control_plane._run_rank_converged_phase(
            "cache_preflight",
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
            body=fail_body,
            receipt_sink=fail_receipt_sink,
            resource_collector=lambda: {
                "schema_version": 1,
                "cpu": {
                    "scope": "current_process",
                    "max_rss_bytes": 1,
                    "io_read_bytes": 2,
                    "io_write_bytes": 3,
                },
                "gpu": {
                    "scope": "current_process_current_device",
                    "initialized": False,
                    "unavailable_reason": "cuda_not_initialized",
                },
            },
        )

    assert exc_info.value is primary
    assert exc_info.value.code == "training.primary_cache_failure"
    notes = getattr(exc_info.value, "__notes__", ())
    assert any(
        "run_writer.secondary_receipt_failure" in note
        and "secondary receipt persistence failure" not in note
        for note in notes
    )
