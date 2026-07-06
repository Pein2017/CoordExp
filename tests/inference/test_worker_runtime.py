from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.common.errors import RuntimeContractError


def test_worker_environment_narrows_cuda_visible_devices_to_assigned_token() -> None:
    from src.inference import worker

    env = worker.build_worker_environment(
        base_env={"CUDA_VISIBLE_DEVICES": "2,3,5", "KEEP": "yes"},
        parent_visible_device_token="5",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["KEEP"] == "yes"


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

    process = worker.launch_worker_subprocess(
        rank=1,
        world_size=2,
        parent_visible_device_token="3",
        resolved_config_json=tmp_path / "resolved.json",
        shard_plan_json=tmp_path / "rank-001.json",
        output_dir=tmp_path / "shards" / "rank-001",
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
    assert launched["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert "shell" not in launched["kwargs"]


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
    resolved = SimpleNamespace(config=SimpleNamespace(name="unit"), fingerprint="infer-fp")
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
