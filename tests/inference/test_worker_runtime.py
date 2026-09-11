from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from src.common.errors import ArtifactContractError, RuntimeContractError


def test_worker_environment_narrows_cuda_visible_devices_and_isolates_runtime_cache(
    tmp_path: Path,
) -> None:
    from src.inference import worker

    runtime_cache_root = tmp_path / "runtime-cache"
    env = worker.build_worker_environment(
        base_env={"CUDA_VISIBLE_DEVICES": "2,3,5", "KEEP": "yes"},
        parent_visible_device_token="5",
        runtime_cache_root=runtime_cache_root,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["KEEP"] == "yes"
    assert env["VLLM_CACHE_ROOT"] == str(runtime_cache_root / "vllm")
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(runtime_cache_root / "torchinductor")


def test_worker_cuda_validation_requires_exactly_one_visible_device() -> None:
    from src.inference import worker

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.validate_worker_cuda_environment(
            rank=0,
            world_size=2,
            parent_visible_device_token="5",
            environ={"CUDA_VISIBLE_DEVICES": "5"},
            torch_module=_torch(device_count=2, current_device=0),
        )

    assert exc_info.value.code == "inference.worker_cuda_binding_invalid"
    assert exc_info.value.context["cuda_device_count"] == 2


def test_worker_cuda_metadata_records_binding_and_model_device() -> None:
    from src.inference import worker

    metadata = worker.build_worker_runtime_metadata(
        rank=2,
        world_size=4,
        parent_visible_device_token="5",
        environ={"CUDA_VISIBLE_DEVICES": "5"},
        torch_module=_torch(device_count=1, current_device=0),
        model=_model_with_device("cuda:0"),
    )

    assert metadata == {
        "rank": 2,
        "world_size": 4,
        "parent_visible_device_token": "5",
        "worker_cuda_visible_devices": "5",
        "cuda_device_count": 1,
        "cuda_current_device": 0,
        "logical_device": "cuda:0",
        "model_first_parameter_device": "cuda:0",
        "runtime_cache": {
            "vllm_cache_root": "",
            "torchinductor_cache_dir": "",
        },
    }


def test_worker_cuda_validation_rejects_nonzero_current_device() -> None:
    from src.inference import worker

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.validate_worker_cuda_environment(
            rank=1,
            world_size=2,
            parent_visible_device_token="3",
            environ={"CUDA_VISIBLE_DEVICES": "3"},
            torch_module=_torch(device_count=1, current_device=1),
        )

    assert exc_info.value.code == "inference.worker_cuda_binding_invalid"
    assert exc_info.value.context["cuda_current_device"] == 1


@pytest.mark.parametrize(
    ("rank", "world_size", "parent_visible_device_token"),
    [
        (2, 2, "3"),
        (0, 0, "3"),
        (0, 1, ""),
        (0, 1, "-1"),
        (0, 1, "2,3"),
    ],
)
def test_worker_cuda_validation_rejects_impossible_rank_world_or_token_metadata(
    rank: int,
    world_size: int,
    parent_visible_device_token: str,
) -> None:
    from src.inference import worker

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.validate_worker_cuda_environment(
            rank=rank,
            world_size=world_size,
            parent_visible_device_token=parent_visible_device_token,
            environ={"CUDA_VISIBLE_DEVICES": parent_visible_device_token},
            torch_module=_torch(device_count=1, current_device=0),
        )

    assert exc_info.value.code == "inference.worker_identity_invalid"


def test_worker_launch_uses_private_fresh_interpreter_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker

    launched: dict[str, Any] = {}

    class FakeProcess:
        returncode = None

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)
    runtime_cache_root = tmp_path / "runtime-cache"

    def fake_mkdtemp(*, prefix: str) -> str:
        launched["cache_prefix"] = prefix
        runtime_cache_root.mkdir()
        return str(runtime_cache_root)

    monkeypatch.setattr(worker.tempfile, "mkdtemp", fake_mkdtemp)

    process = worker.launch_worker_subprocess(
        rank=1,
        world_size=2,
        parent_visible_device_token="3",
        resolved_config_json=tmp_path / "resolved.json",
        shard_plan_json=tmp_path / "rank-001.json",
        output_dir=tmp_path / "shards" / "rank-001",
        execution_model_json=tmp_path / "execution_model.json",
        base_env={"CUDA_VISIBLE_DEVICES": "2,3"},
    )

    assert isinstance(process, FakeProcess)
    assert launched["command"][:3] == [
        worker.sys.executable,
        "-m",
        "src.inference.worker",
    ]
    assert "--rank" in launched["command"]
    assert "--resolved-config-json" in launched["command"]
    assert "--execution-model-json" in launched["command"]
    assert launched["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert launched["kwargs"]["env"]["VLLM_CACHE_ROOT"] == str(
        runtime_cache_root / "vllm"
    )
    assert launched["kwargs"]["env"]["TORCHINDUCTOR_CACHE_DIR"] == str(
        runtime_cache_root / "torchinductor"
    )
    assert launched["cache_prefix"] == "coordexp-infras-infer-rank-1-"
    assert process._coordexp_runtime_cache_root == str(runtime_cache_root)
    assert launched["kwargs"]["start_new_session"] is True
    assert "shell" not in launched["kwargs"]


def test_worker_wait_removes_private_runtime_cache(tmp_path: Path) -> None:
    from src.inference import worker

    runtime_cache_root = tmp_path / "worker-cache"
    runtime_cache_root.mkdir()

    class FinishedProcess:
        pid = None
        returncode = 0
        _coordexp_runtime_cache_root = str(runtime_cache_root)

        def poll(self) -> int:
            return 0

    assert worker.wait_for_worker_processes([(0, FinishedProcess())]) == {0: 0}
    assert not runtime_cache_root.exists()


def test_terminate_worker_processes_stops_all_owned_processes(tmp_path: Path) -> None:
    from src.inference import worker

    processes: list[Any] = []
    for rank in (0, 1):
        cache_root = tmp_path / f"cache-{rank}"
        cache_root.mkdir()

        class LiveProcess:
            pid = None
            returncode: int | None = None

            def terminate(self) -> None:
                self.returncode = -15

            def wait(self, *, timeout: float | None = None) -> int:
                assert timeout is not None
                assert self.returncode is not None
                return self.returncode

        process = LiveProcess()
        process._coordexp_runtime_cache_root = str(cache_root)
        processes.append(process)

    assert worker.terminate_worker_processes(list(enumerate(processes))) == {
        0: -15,
        1: -15,
    }
    assert all(process.returncode == -15 for process in processes)
    assert not any((tmp_path / f"cache-{rank}").exists() for rank in (0, 1))


def test_worker_wait_timeout_terminates_owned_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker

    signals: list[tuple[int, object]] = []

    class FakeProcess:
        pid = 4321
        returncode: int | None = None

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float | None = None) -> int:
            if self.returncode is None:
                raise worker.subprocess.TimeoutExpired("worker", timeout)
            return self.returncode

    process = FakeProcess()

    process_group_alive = True

    def fake_killpg(pid: int, sig: object) -> None:
        nonlocal process_group_alive
        signals.append((pid, sig))
        if sig == worker.signal.SIGKILL:
            process.returncode = -9
            process_group_alive = False

    monkeypatch.setattr(worker, "_signal_process_group", fake_killpg)
    monkeypatch.setattr(worker, "_process_group_exists", lambda _: process_group_alive)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.wait_for_worker_processes(((0, process),), timeout_seconds=1e-9)

    assert exc_info.value.code == "inference.worker_timeout"
    assert exc_info.value.context["timed_out_ranks"] == [0]
    assert signals == [(4321, worker.signal.SIGTERM), (4321, worker.signal.SIGKILL)]


def test_worker_termination_kills_descendants_after_leader_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker

    signals: list[object] = []
    process_group_alive = True

    class ExitedLeader:
        pid = 4321
        returncode = -15

        def wait(self, *, timeout: float | None = None) -> int:
            return self.returncode

    def signal_group(_: int, sig: object) -> None:
        nonlocal process_group_alive
        signals.append(sig)
        if sig == worker.signal.SIGKILL:
            process_group_alive = False

    monkeypatch.setattr(worker, "WORKER_TERMINATION_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(worker, "_signal_process_group", signal_group)
    monkeypatch.setattr(worker, "_process_group_exists", lambda _: process_group_alive)

    assert worker._terminate_worker_process_tree(ExitedLeader()) == -15
    assert signals == [worker.signal.SIGTERM, worker.signal.SIGKILL]


def test_worker_termination_fails_when_process_group_survives_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker

    signals: list[object] = []

    class ExitedLeader:
        pid = 4321
        returncode = -15

        def wait(self, *, timeout: float | None = None) -> int:
            return self.returncode

    monkeypatch.setattr(worker, "WORKER_TERMINATION_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(
        worker,
        "_signal_process_group",
        lambda _, sig: signals.append(sig),
    )
    monkeypatch.setattr(worker, "_process_group_exists", lambda _: True)

    with pytest.raises(RuntimeContractError) as exc_info:
        worker._terminate_worker_process_tree(ExitedLeader())

    assert exc_info.value.code == "inference.worker_process_tree_survived"
    assert signals == [worker.signal.SIGTERM, worker.signal.SIGKILL]


def test_worker_termination_attempts_every_rank_before_reporting_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker

    processes = [SimpleNamespace(returncode=None), SimpleNamespace(returncode=-9)]
    attempted: list[object] = []
    cache_removed: list[object] = []

    def terminate(process: object) -> int:
        attempted.append(process)
        if process is processes[0]:
            raise RuntimeContractError(
                "survived",
                code="inference.worker_process_tree_survived",
            )
        return -9

    monkeypatch.setattr(worker, "_terminate_worker_process_tree", terminate)
    monkeypatch.setattr(
        worker,
        "_remove_worker_runtime_cache",
        lambda process: cache_removed.append(process),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.terminate_worker_processes(tuple(enumerate(processes)))

    assert exc_info.value.code == "inference.worker_process_tree_survived"
    assert exc_info.value.context["failed_ranks"] == [0]
    assert exc_info.value.context["worker_return_codes"] == {0: None, 1: -9}
    assert attempted == processes
    assert cache_removed == processes


def test_worker_wait_rejects_surviving_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker

    class FakeProcess:
        pid = 5678
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, *, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(worker, "WORKER_TERMINATION_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(worker, "_process_group_exists", lambda _: True)
    monkeypatch.setattr(worker, "_terminate_worker_process_tree", lambda _: -9)

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.wait_for_worker_processes(((2, FakeProcess()),))

    assert exc_info.value.code == "inference.worker_orphan_process"
    assert exc_info.value.context["rank"] == 2


def test_worker_orphan_failure_terminates_pending_sibling_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker

    class FinishedOrphan:
        pid = 5678
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, *, timeout: float | None = None) -> int:
            return 0

    class PendingProcess:
        pid = 6789
        returncode: int | None = None

        def poll(self) -> None:
            return None

    orphan = FinishedOrphan()
    pending = PendingProcess()
    terminated: list[object] = []

    monkeypatch.setattr(worker, "WORKER_TERMINATION_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(
        worker,
        "_process_group_exists",
        lambda process_group_id: process_group_id == 5678,
    )

    def terminate(process: object) -> int:
        terminated.append(process)
        return -9

    monkeypatch.setattr(worker, "_terminate_worker_process_tree", terminate)

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.wait_for_worker_processes(
            ((0, orphan), (1, pending)),
        )

    assert exc_info.value.code == "inference.worker_orphan_process"
    assert terminated == [orphan, pending]


def test_worker_main_executes_assigned_rank_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel
    from src.inference import worker

    plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0", "row-1"),
        per_device_batch_size=1,
        visible_cuda_tokens=("0", "1"),
    )
    resolved = SimpleNamespace(
        config=SimpleNamespace(
            name="unit",
            backend=SimpleNamespace(type="hf"),
        ),
        fingerprint="infer-fp",
    )
    calls: dict[str, Any] = {}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        worker,
        "load_resolved_infer_config_artifact",
        lambda path: resolved,
    )
    monkeypatch.setattr(
        worker,
        "load_data_parallel_plan_artifact",
        lambda path: plan,
    )
    monkeypatch.setattr(
        worker,
        "build_worker_runtime_metadata",
        lambda **kwargs: {
            "rank": kwargs["rank"],
            "world_size": kwargs["world_size"],
            "parent_visible_device_token": kwargs["parent_visible_device_token"],
            "worker_cuda_visible_devices": kwargs["parent_visible_device_token"],
            "cuda_device_count": 1,
            "cuda_current_device": 0,
            "logical_device": "cuda:0",
            "model_first_parameter_device": "cuda:0",
        },
    )

    def fake_run_shard(**kwargs: Any) -> int:
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(worker.pipeline, "run_shard", fake_run_shard)

    result = worker.main(
        [
            "--rank",
            "1",
            "--world-size",
            "2",
            "--parent-visible-device-token",
            "1",
            "--resolved-config-json",
            str(tmp_path / "resolved.json"),
            "--shard-plan-json",
            str(tmp_path / "plan.json"),
            "--output-dir",
            str(tmp_path / "shards" / "rank-001"),
        ]
    )

    assert result == 0
    assert calls["resolved"] is resolved
    assert calls["output_dir"] == tmp_path / "shards" / "rank-001"
    assert calls["row_indices"] == (1,)
    assert calls["rank_plan"] == plan.ranks[1]
    assert calls["worker_metadata"]["shard_plan_fingerprint"] == plan.fingerprint
    assert calls["worker_metadata"]["worker_logical_device"] == "cuda:0"
    assert calls["worker_metadata"]["model_first_parameter_device"] == "cuda:0"


def test_hf_worker_rejects_execution_model_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker

    monkeypatch.setattr(
        worker,
        "load_resolved_infer_config_artifact",
        lambda path: SimpleNamespace(
            config=SimpleNamespace(backend=SimpleNamespace(type="hf"))
        ),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.main(
            [
                "--rank",
                "0",
                "--world-size",
                "1",
                "--parent-visible-device-token",
                "0",
                "--resolved-config-json",
                str(tmp_path / "resolved.json"),
                "--shard-plan-json",
                str(tmp_path / "plan.json"),
                "--output-dir",
                str(tmp_path / "shards" / "rank-000"),
                "--execution-model-json",
                str(tmp_path / "execution-model.json"),
            ]
        )

    assert exc_info.value.code == "inference.worker_hf_execution_model_forbidden"


def test_worker_revalidates_execution_model_before_running_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import data_parallel, worker
    from src.inference.execution_model import resolve_execution_model

    base = _write_execution_snapshot(tmp_path / "base")
    receipt = resolve_execution_model(
        base_model_path=base,
        target_dtype="bf16",
    )
    receipt_path = tmp_path / "execution_model.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"changed-after-controller")
    plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0",),
        per_device_batch_size=1,
        visible_cuda_tokens=("0",),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        worker,
        "load_resolved_infer_config_artifact",
        lambda path: SimpleNamespace(
            config=SimpleNamespace(
                name="unit",
                backend=SimpleNamespace(type="vllm"),
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "load_data_parallel_plan_artifact",
        lambda path: plan,
    )
    monkeypatch.setattr(
        worker,
        "build_worker_runtime_metadata",
        lambda **kwargs: {
            "rank": 0,
            "world_size": 1,
            "parent_visible_device_token": "0",
            "worker_cuda_visible_devices": "0",
            "cuda_device_count": 1,
            "cuda_current_device": 0,
            "logical_device": "cuda:0",
            "model_first_parameter_device": None,
        },
    )
    run_called = False

    def fail_if_called(**kwargs: Any) -> int:
        nonlocal run_called
        run_called = True
        return 0

    monkeypatch.setattr(worker.pipeline, "run_shard", fail_if_called)

    with pytest.raises(RuntimeContractError, match="snapshot"):
        worker.main(
            [
                "--rank",
                "0",
                "--world-size",
                "1",
                "--parent-visible-device-token",
                "0",
                "--resolved-config-json",
                str(tmp_path / "resolved.json"),
                "--shard-plan-json",
                str(tmp_path / "plan.json"),
                "--output-dir",
                str(tmp_path / "shards" / "rank-000"),
                "--execution-model-json",
                str(receipt_path),
            ]
        )
    assert run_called is False


def _patch_worker_for_execution_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: dict[str, Any],
) -> None:
    from src.inference import data_parallel, worker

    plan = data_parallel.plan_data_parallel_shards(
        row_ids=("row-0",),
        per_device_batch_size=1,
        visible_cuda_tokens=("0",),
    )
    resolved = SimpleNamespace(
        config=SimpleNamespace(name="unit", backend=SimpleNamespace(type="hf")),
        fingerprint="infer-fp",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(worker, "load_resolved_infer_config_artifact", lambda path: resolved)
    monkeypatch.setattr(worker, "load_data_parallel_plan_artifact", lambda path: plan)
    monkeypatch.setattr(
        worker,
        "build_worker_runtime_metadata",
        lambda **kwargs: {
            "rank": kwargs["rank"],
            "world_size": kwargs["world_size"],
            "parent_visible_device_token": kwargs["parent_visible_device_token"],
            "worker_cuda_visible_devices": kwargs["parent_visible_device_token"],
            "cuda_device_count": 1,
            "cuda_current_device": 0,
            "logical_device": "cuda:0",
            "model_first_parameter_device": "cuda:0",
        },
    )

    def fake_run_shard(**kwargs: Any) -> int:
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(worker.pipeline, "run_shard", fake_run_shard)


def _worker_argv(
    tmp_path: Path,
    *,
    execution_context_json: str | None = None,
    execution_context_file_sha256: str | None = None,
    execution_context_value_fingerprint: str | None = None,
    execution_context_journal_plan_reference: str | None = None,
) -> list[str]:
    argv = [
        "--rank",
        "0",
        "--world-size",
        "1",
        "--parent-visible-device-token",
        "0",
        "--resolved-config-json",
        str(tmp_path / "resolved.json"),
        "--shard-plan-json",
        str(tmp_path / "plan.json"),
        "--output-dir",
        str(tmp_path / "shards" / "rank-000"),
    ]
    if execution_context_json is not None:
        argv.extend(["--execution-context-json", execution_context_json])
    if execution_context_file_sha256 is not None:
        argv.extend(["--execution-context-file-sha256", execution_context_file_sha256])
    if execution_context_value_fingerprint is not None:
        argv.extend(
            ["--execution-context-value-fingerprint", execution_context_value_fingerprint]
        )
    if execution_context_journal_plan_reference is not None:
        argv.extend(
            [
                "--execution-context-journal-plan-reference",
                execution_context_journal_plan_reference,
            ]
        )
    return argv


def test_worker_main_verifies_copies_and_forwards_execution_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker
    from src.inference.execution_context import materialize_execution_context_artifact

    calls: dict[str, Any] = {}
    _patch_worker_for_execution_context(monkeypatch, calls=calls)

    reference = {
        "journal_schema_version": 1,
        "execution_id": "exec-1",
        "plan_fingerprint": "a" * 64,
        "plan_file_sha256": "b" * 64,
    }
    artifact = materialize_execution_context_artifact(
        run_dir=tmp_path / "controller",
        execution_context={"probe_worker_note": "unit"},
        journal_plan_reference=reference,
    )
    assert artifact is not None

    result = worker.main(
        _worker_argv(
            tmp_path,
            execution_context_json=artifact.locator,
            execution_context_file_sha256=artifact.file_sha256,
            execution_context_value_fingerprint=artifact.value_fingerprint,
            execution_context_journal_plan_reference=json.dumps(reference, sort_keys=True),
        )
    )

    assert result == 0
    assert calls["execution_context_identity"] == artifact.identity()
    rank_local_copy = tmp_path / "shards" / "rank-000" / "execution_context.json"
    assert rank_local_copy.is_file()
    assert rank_local_copy.read_bytes() == artifact.path.read_bytes()


def test_worker_main_rejects_execution_context_journal_plan_reference_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker
    from src.inference.execution_context import materialize_execution_context_artifact

    calls: dict[str, Any] = {}
    _patch_worker_for_execution_context(monkeypatch, calls=calls)

    artifact = materialize_execution_context_artifact(
        run_dir=tmp_path / "controller",
        execution_context={"probe_worker_note": "unit"},
        journal_plan_reference={
            "journal_schema_version": 1,
            "execution_id": "exec-1",
            "plan_fingerprint": "a" * 64,
            "plan_file_sha256": "b" * 64,
        },
    )
    assert artifact is not None
    drifted_reference = {
        "journal_schema_version": 1,
        "execution_id": "exec-DIFFERENT",
        "plan_fingerprint": "a" * 64,
        "plan_file_sha256": "b" * 64,
    }

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.main(
            _worker_argv(
                tmp_path,
                execution_context_json=artifact.locator,
                execution_context_file_sha256=artifact.file_sha256,
                execution_context_value_fingerprint=artifact.value_fingerprint,
                execution_context_journal_plan_reference=json.dumps(
                    drifted_reference, sort_keys=True
                ),
            )
        )

    assert exc_info.value.code == "inference.execution_context_plan_reference_mismatch"
    assert calls == {}
    assert not (tmp_path / "shards" / "rank-000" / "execution_context.json").exists()


def test_worker_main_rejects_malformed_execution_context_digest_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import worker
    from src.inference.execution_context import materialize_execution_context_artifact

    calls: dict[str, Any] = {}
    _patch_worker_for_execution_context(monkeypatch, calls=calls)

    artifact = materialize_execution_context_artifact(
        run_dir=tmp_path / "controller",
        execution_context={"probe_worker_note": "unit"},
    )
    assert artifact is not None

    with pytest.raises(ArtifactContractError) as exc_info:
        worker.main(
            _worker_argv(
                tmp_path,
                execution_context_json=artifact.locator,
                execution_context_file_sha256="not-a-sha256-digest",
                execution_context_value_fingerprint=artifact.value_fingerprint,
            )
        )

    assert exc_info.value.code == "inference.execution_context_digest_format_invalid"
    assert calls == {}


def test_worker_main_rejects_execution_context_payload_schema_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from src.artifacts.json_values import canonical_json_bytes, json_sha256
    from src.inference import worker

    calls: dict[str, Any] = {}
    _patch_worker_for_execution_context(monkeypatch, calls=calls)

    tampered_payload = {
        "execution_context_schema_version": 1,
        "context": {"probe_worker_note": "unit"},
        # "context_fingerprint" key is missing entirely: schema tamper.
        "journal_plan_reference": None,
    }
    path = tmp_path / "tampered_execution_context.json"
    encoded = canonical_json_bytes(tampered_payload)
    path.write_bytes(encoded)

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.main(
            _worker_argv(
                tmp_path,
                execution_context_json=str(path),
                execution_context_file_sha256=hashlib.sha256(encoded).hexdigest(),
                execution_context_value_fingerprint=json_sha256(tampered_payload),
            )
        )

    assert exc_info.value.code == "inference.execution_context_schema_invalid"
    assert calls == {}


def test_worker_main_rejects_boolean_execution_context_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from src.artifacts.json_values import canonical_json_bytes, json_sha256
    from src.inference import worker

    calls: dict[str, Any] = {}
    _patch_worker_for_execution_context(monkeypatch, calls=calls)
    context = {"probe_worker_note": "unit"}
    payload = {
        "execution_context_schema_version": True,
        "context": context,
        "context_fingerprint": json_sha256(context),
        "journal_plan_reference": None,
    }
    path = tmp_path / "boolean-schema-execution-context.json"
    encoded = canonical_json_bytes(payload)
    path.write_bytes(encoded)

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.main(
            _worker_argv(
                tmp_path,
                execution_context_json=str(path.resolve()),
                execution_context_file_sha256=hashlib.sha256(encoded).hexdigest(),
                execution_context_value_fingerprint=json_sha256(payload),
            )
        )

    assert exc_info.value.code == "inference.execution_context_schema_invalid"
    assert calls == {}


def test_worker_main_rejects_execution_context_inner_fingerprint_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from src.artifacts.json_values import canonical_json_bytes, json_sha256
    from src.inference import worker

    calls: dict[str, Any] = {}
    _patch_worker_for_execution_context(monkeypatch, calls=calls)

    tampered_payload = {
        "execution_context_schema_version": 1,
        "context": {"probe_worker_note": "unit"},
        "context_fingerprint": "0" * 64,  # deliberately wrong for this context
        "journal_plan_reference": None,
    }
    path = tmp_path / "tampered_execution_context.json"
    encoded = canonical_json_bytes(tampered_payload)
    path.write_bytes(encoded)

    with pytest.raises(RuntimeContractError) as exc_info:
        worker.main(
            _worker_argv(
                tmp_path,
                execution_context_json=str(path),
                execution_context_file_sha256=hashlib.sha256(encoded).hexdigest(),
                execution_context_value_fingerprint=json_sha256(tampered_payload),
            )
        )

    assert exc_info.value.code == "inference.execution_context_inner_fingerprint_mismatch"
    assert calls == {}


def test_worker_command_binds_journal_plan_reference_in_launch_contract(
    tmp_path: Path,
) -> None:
    from src.inference.worker import build_worker_command

    reference = {
        "journal_schema_version": 1,
        "execution_id": "exec-1",
        "plan_fingerprint": "a" * 64,
        "plan_file_sha256": "b" * 64,
    }
    command = build_worker_command(
        rank=0,
        world_size=1,
        parent_visible_device_token="0",
        resolved_config_json=tmp_path / "resolved.json",
        shard_plan_json=tmp_path / "plan.json",
        output_dir=tmp_path / "shards" / "rank-000",
        execution_context_json=tmp_path / "execution_context.json",
        execution_context_file_sha256="c" * 64,
        execution_context_value_fingerprint="d" * 64,
        execution_context_journal_plan_reference=reference,
    )

    index = command.index("--execution-context-journal-plan-reference")
    assert json.loads(command[index + 1]) == reference


def test_worker_source_does_not_import_multiprocessing_or_request_fork_context() -> None:
    from src.inference import worker

    source = inspect.getsource(worker)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    fork_context_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get_context"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in {"fork", "forkserver"}
            ):
                fork_context_calls.append(node)

    assert "multiprocessing" not in imported_roots
    assert fork_context_calls == []


def _torch(*, device_count: int, current_device: int) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: device_count > 0,
            device_count=lambda: device_count,
            current_device=lambda: current_device,
        )
    )


def _model_with_device(device: str) -> SimpleNamespace:
    parameter = SimpleNamespace(device=device)
    return SimpleNamespace(parameters=lambda: iter([parameter]))


def _write_execution_snapshot(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "tie_word_embeddings": True,
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.zeros(
                (2, 2), dtype=torch.bfloat16
            )
        },
        root / "model.safetensors",
    )
    return root
