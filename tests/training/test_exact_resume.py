from __future__ import annotations

import copy
import dataclasses
import random
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

from src.artifacts import training_state
from src.training import exact_resume
from src.artifacts.training_state import (
    REQUIRED_IDENTITY_KINDS,
    TrainingStatePublicationPlan,
    build_resume_compatibility_projection,
    load_training_state_manifest,
)
from src.common.errors import ArtifactContractError
from src.training.exact_resume import (
    DistributedExactResumeRestoreReceipt,
    DistributedExactResumeRestoreStatus,
    DistributedExactResumeStatus,
    RankCudaDeviceBinding,
    RankRngSnapshot,
    admit_and_restore_current_rank,
    build_exact_resume_identities,
    build_exact_resume_topology_identity,
    capture_rank_rng_snapshot,
    prepare_distributed_exact_resume_contribution,
    publish_distributed_exact_resume,
    restore_distributed_exact_resume,
    serialize_current_rank_training_state,
)


_DIGESTS = {
    name: f"{index + 1:x}" * 64
    for index, name in enumerate(REQUIRED_IDENTITY_KINDS)
    if name != "resolved_config"
}


def _runtime(
    *, initialized: bool
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    if initialized:
        loss = model(torch.arange(6, dtype=torch.float32).reshape(2, 3)).square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return model, optimizer, scheduler


def _runtime_with_scaler(
    *, initialized: bool
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.amp.GradScaler,
]:
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu")
    if initialized:
        loss = model(torch.arange(6, dtype=torch.float32).reshape(2, 3)).square().sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return model, optimizer, scheduler, scaler


def _resolved_config() -> dict[str, Any]:
    return {
        "config": {
            "resume": {"checkpoint_dir": None, "mode": "disabled"},
            "run": {
                "artifact_root": "/outputs",
                "name": "parent",
                "output_dir": "/outputs/parent",
            },
            "runtime": {
                "determinism": {"mode": "strict_cuda_replay_v1"},
                "seed": 17,
            },
            "training": {"seed": 17},
        },
        "resolution": {
            "entry_config_path": "/configs/parent.yaml",
            "fingerprint": "parent",
            "loader_version": "coordexp-swift-config-v1",
            "path_origins": {},
            "schema_version": 1,
            "sources": [],
        },
    }


def test_resume_compatibility_binds_runtime_determinism_but_not_continuation() -> None:
    parent = _resolved_config()
    child = copy.deepcopy(parent)
    child["config"]["run"]["name"] = "child"
    child["config"]["resume"] = {
        "checkpoint_dir": "/outputs/parent/checkpoints/step-3",
        "mode": "exact_same_world_size",
    }
    strict_parent = build_resume_compatibility_projection(parent)
    strict_child = build_resume_compatibility_projection(child)
    assert strict_parent == strict_child
    assert strict_parent["semantic_config"]["runtime"] == {
        "determinism": {"mode": "strict_cuda_replay_v1"},
        "seed": 17,
    }

    legacy_child = copy.deepcopy(child)
    legacy_child["config"]["runtime"]["determinism"]["mode"] = "legacy"
    assert build_resume_compatibility_projection(legacy_child) != strict_parent


def _identities(
    resolved_config: Mapping[str, Any] | None = None,
) -> Mapping[str, str]:
    return build_exact_resume_identities(
        base_model=_DIGESTS["base_model"],
        cache=_DIGESTS["cache"],
        dependencies=_DIGESTS["dependencies"],
        policy=_DIGESTS["policy"],
        resolved_config=resolved_config or _resolved_config(),
        topology=_DIGESTS["topology"],
        trainable_surface=_DIGESTS["trainable_surface"],
    )


def _rng_snapshot(*, rank: int, world_size: int) -> RankRngSnapshot:
    return RankRngSnapshot(
        rank=rank,
        world_size=world_size,
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_states=(torch.arange(32, dtype=torch.uint8) + rank,),
        cuda_device_topology=("cuda:0",),
        cuda_device_binding=RankCudaDeviceBinding(
            rank=rank,
            world_size=world_size,
            logical_device=rank,
            physical_device=f"GPU-{rank}",
        ),
    )


def _rank_payload(*, rank: int, world_size: int, scaler_applicable: bool = False):
    if scaler_applicable:
        model, optimizer, scheduler, scaler = _runtime_with_scaler(initialized=True)
    else:
        model, optimizer, scheduler = _runtime(initialized=True)
        scaler = None
    return serialize_current_rank_training_state(
        rank=rank,
        world_size=world_size,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cursor={
            "data": {"epoch": 2, "ordinal": 20 + rank},
            "pack": {"ordinal": 10 + rank, "pending": []},
        },
        next_rank_local_micro_step=30 + rank,
        accumulation_microstep=0,
        rng_snapshot=_rng_snapshot(rank=rank, world_size=world_size),
    )


def _plan(
    *, world_size: int, scaler_applicable: bool = False
) -> TrainingStatePublicationPlan:
    return TrainingStatePublicationPlan(
        parent_run_id="run-parent",
        parent_segment_id="segment-2",
        checkpoint_step=17,
        continuation_index=2,
        world_size=world_size,
        identities=_identities(),
        scheduler_applicable=True,
        scaler_applicable=scaler_applicable,
        resolved_config=_resolved_config(),
        resume_compatibility=build_resume_compatibility_projection(_resolved_config()),
        accumulation_microstep=0,
    )


def _checkpoint(tmp_path: Path, name: str = "step-17") -> Path:
    checkpoint = tmp_path / name
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"inference-payload")
    return checkpoint


class _ThreadCollectives:
    def __init__(self, world_size: int) -> None:
        self._barrier = threading.Barrier(world_size)
        self._condition = threading.Condition()
        self._generation = 0
        self._values: dict[int, list[DistributedExactResumeStatus]] = {}
        self._results: dict[int, tuple[DistributedExactResumeStatus, ...]] = {}
        self.gathered: list[DistributedExactResumeStatus] = []
        self.barrier_calls = 0

    def barrier(self) -> None:
        self.barrier_calls += 1
        self._barrier.wait(timeout=10)

    def gather(
        self, value: DistributedExactResumeStatus
    ) -> Sequence[DistributedExactResumeStatus]:
        assert not _contains_bytes(value)
        with self._condition:
            generation = self._generation
            values = self._values.setdefault(generation, [])
            values.append(value)
            self.gathered.append(value)
            if len(values) == self._barrier.parties:
                result = tuple(sorted(values, key=lambda item: item.rank))
                self._results[generation] = result
                self._generation += 1
                self._condition.notify_all()
            else:
                assert self._condition.wait_for(
                    lambda: generation in self._results, timeout=10
                )
            return self._results[generation]


def _contains_bytes(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if dataclasses.is_dataclass(value):
        return any(
            _contains_bytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, Mapping):
        return any(
            _contains_bytes(key) or _contains_bytes(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_bytes(item) for item in value)
    return False


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return list(left) == list(right) and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _state_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _gloo_restore_failure_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_dir: str,
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    model, optimizer, scheduler, scaler = _runtime_with_scaler(initialized=False)
    random.seed(100 + rank)
    np.random.seed(200 + rank)
    torch.manual_seed(300 + rank)
    model_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    scaler_before = copy.deepcopy(scaler.state_dict())
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_cpu_before = torch.get_rng_state().clone()
    cuda_before = (torch.arange(32, dtype=torch.uint8) + 100 + rank,)
    cuda_state = [state.clone() for state in cuda_before]
    cuda_get_calls = 0
    cuda_set_calls = 0
    seed_calls: list[str] = []
    gathered_phases: list[str] = []

    def get_cuda() -> Sequence[torch.Tensor]:
        nonlocal cuda_get_calls
        cuda_get_calls += 1
        return tuple(state.clone() for state in cuda_state)

    def set_cuda(states: Sequence[torch.Tensor]) -> None:
        nonlocal cuda_set_calls
        cuda_set_calls += 1
        if rank == 1 and cuda_set_calls == 1:
            raise RuntimeError("injected rank-one CUDA RNG apply failure")
        cuda_state[:] = [state.clone() for state in states]

    def gather(
        status: DistributedExactResumeRestoreStatus,
    ) -> Sequence[DistributedExactResumeRestoreStatus]:
        assert not _contains_bytes(status)
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, status)
        assert all(
            isinstance(item, DistributedExactResumeRestoreStatus)
            and not _contains_bytes(item)
            for item in gathered
        )
        gathered_phases.append(status.phase)
        return tuple(gathered)

    torch.cuda.get_rng_state_all = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("restore must not enumerate visible CUDA devices")
    )
    random.seed = lambda *_args, **_kwargs: seed_calls.append("python")  # type: ignore[assignment]
    np.random.seed = lambda *_args, **_kwargs: seed_calls.append("numpy")  # type: ignore[assignment]
    torch.manual_seed = lambda *_args, **_kwargs: seed_calls.append("torch")  # type: ignore[assignment]
    error: ArtifactContractError | None = None
    try:
        restore_distributed_exact_resume(
            checkpoint_dir,
            checkpoint_step=17,
            expected_manifest_digest=load_training_state_manifest(
                checkpoint_dir
            ).aggregate_digest,
            rank=rank,
            world_size=world_size,
            identities=_identities(),
            resolved_config=_resolved_config(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cuda_device_topology=("cuda:0",),
            gather_status=gather,
            get_torch_cuda_rng_states=get_cuda,
            set_torch_cuda_rng_states=set_cuda,
        )
    except ArtifactContractError as exc:
        error = exc
    finally:
        dist.destroy_process_group()

    assert error is not None
    result_queue.put(
        {
            "code": error.code,
            "context": error.context,
            "cuda_restored": _state_equal(tuple(cuda_state), cuda_before),
            "cuda_get_calls": cuda_get_calls,
            "gathered_phases": gathered_phases,
            "model_restored": all(
                torch.equal(dict(model.named_parameters())[name], value)
                for name, value in model_before.items()
            ),
            "numpy_restored": _state_equal(np.random.get_state(), numpy_before),
            "optimizer_restored": _state_equal(
                optimizer.state_dict(), optimizer_before
            ),
            "python_restored": random.getstate() == python_before,
            "rank": rank,
            "scaler_restored": _state_equal(scaler.state_dict(), scaler_before),
            "scheduler_restored": _state_equal(
                scheduler.state_dict(), scheduler_before
            ),
            "seed_calls": seed_calls,
            "torch_cpu_restored": torch.equal(torch.get_rng_state(), torch_cpu_before),
        }
    )


def _gloo_publication_failure_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_dir: str,
    failure_mode: str,
    payloads: Sequence[Any],
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    gathered_phases: list[str] = []

    def gather(
        status: DistributedExactResumeStatus,
    ) -> Sequence[DistributedExactResumeStatus]:
        assert not _contains_bytes(status)
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, status)
        assert all(
            isinstance(item, DistributedExactResumeStatus) and not _contains_bytes(item)
            for item in gathered
        )
        gathered_phases.append(status.phase)
        return tuple(gathered)

    if failure_mode == "commit" and rank == 0:
        exact_resume.commit_training_state_contributions = (  # type: ignore[assignment]
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ArtifactContractError(
                    "injected pre-manifest commit failure",
                    code="training_state.injected_commit_failure",
                )
            )
        )
    payload_rank = 0 if failure_mode == "contribute" and rank == 1 else rank
    error: ArtifactContractError | None = None
    try:
        publish_distributed_exact_resume(
            checkpoint_dir,
            plan=_plan(world_size=world_size),
            rank=rank,
            local_payload=payloads[payload_rank],
            barrier=dist.barrier,
            gather_status=gather,
        )
    except ArtifactContractError as exc:
        error = exc
    finally:
        dist.destroy_process_group()

    assert error is not None
    checkpoint = Path(checkpoint_dir)
    result_queue.put(
        {
            "code": error.code,
            "context": error.context,
            "gathered_phases": gathered_phases,
            "rank": rank,
            "residue": sorted(
                path.name for path in checkpoint.glob(".training_state.*.tmp")
            ),
            "target_exists": (checkpoint / "training_state").exists(),
        }
    )


def test_identity_builder_covers_the_strict_core_identity_set() -> None:
    identities = _identities()

    assert tuple(identities) == REQUIRED_IDENTITY_KINDS
    assert len(identities["resolved_config"]) == 64
    assert len(identities["resume_compatibility"]) == 64

    with pytest.raises(ArtifactContractError) as exc_info:
        build_exact_resume_identities(
            base_model="not-a-digest",
            cache=_DIGESTS["cache"],
            dependencies=_DIGESTS["dependencies"],
            policy=_DIGESTS["policy"],
            resolved_config=_resolved_config(),
            topology=_DIGESTS["topology"],
            trainable_surface=_DIGESTS["trainable_surface"],
        )
    assert exc_info.value.code == "training_state.schema"

    with pytest.raises(ArtifactContractError) as config_info:
        build_exact_resume_identities(
            base_model=_DIGESTS["base_model"],
            cache=_DIGESTS["cache"],
            dependencies=_DIGESTS["dependencies"],
            policy=_DIGESTS["policy"],
            resolved_config=["not", "a", "mapping"],  # type: ignore[arg-type]
            topology=_DIGESTS["topology"],
            trainable_surface=_DIGESTS["trainable_surface"],
        )
    assert config_info.value.code == "training_state.schema"


def test_publication_status_gather_rejects_wrong_world_size() -> None:
    local_status = DistributedExactResumeStatus(
        phase="begin",
        rank=0,
        world_size=2,
        ok=True,
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        exact_resume._gather_statuses(
            lambda _status: (local_status,),
            local_status,
            phase="begin",
            world_size=1,
        )

    assert exc_info.value.code == "training_state.distributed_control"


def test_restore_status_gather_rejects_wrong_world_size() -> None:
    local_status = DistributedExactResumeRestoreStatus(
        phase="restore_admit",
        rank=0,
        world_size=2,
        ok=False,
        error_code="training_state.test_failure",
        error_type="TestFailure",
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        exact_resume._gather_restore_statuses(
            lambda _status: (local_status,),
            local_status,
            phase="restore_admit",
            world_size=1,
        )

    assert exc_info.value.code == "training_state.distributed_control"


def test_identity_uses_core_resume_projection_while_preserving_full_configs() -> None:
    parent = _resolved_config()
    child = copy.deepcopy(parent)
    child["config"]["run"] = {
        "artifact_root": "/outputs/child",
        "name": "child",
        "output_dir": "/outputs/child/segment",
    }
    child["config"]["resume"] = {
        "checkpoint_dir": "/outputs/parent/checkpoints/step-17",
        "mode": "exact_same_world_size",
    }
    child["resolution"]["entry_config_path"] = "/configs/child.yaml"
    parent_identities = build_exact_resume_identities(
        base_model=_DIGESTS["base_model"],
        cache=_DIGESTS["cache"],
        dependencies=_DIGESTS["dependencies"],
        policy=_DIGESTS["policy"],
        resolved_config=parent,
        topology=_DIGESTS["topology"],
        trainable_surface=_DIGESTS["trainable_surface"],
    )
    child_identities = build_exact_resume_identities(
        base_model=_DIGESTS["base_model"],
        cache=_DIGESTS["cache"],
        dependencies=_DIGESTS["dependencies"],
        policy=_DIGESTS["policy"],
        resolved_config=child,
        topology=_DIGESTS["topology"],
        trainable_surface=_DIGESTS["trainable_surface"],
    )

    assert parent != child
    assert parent_identities["resolved_config"] != child_identities["resolved_config"]
    assert (
        parent_identities["resume_compatibility"]
        == child_identities["resume_compatibility"]
    )

    child["config"]["training"]["seed"] = 18
    drifted = build_exact_resume_identities(
        base_model=_DIGESTS["base_model"],
        cache=_DIGESTS["cache"],
        dependencies=_DIGESTS["dependencies"],
        policy=_DIGESTS["policy"],
        resolved_config=child,
        topology=_DIGESTS["topology"],
        trainable_surface=_DIGESTS["trainable_surface"],
    )
    assert drifted["resume_compatibility"] != parent_identities["resume_compatibility"]


def test_rng_capture_is_rank_bound_and_records_exact_cuda_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_state = torch.arange(16, dtype=torch.uint8) + 1
    observed_devices: list[int] = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: observed_devices.append(device) or cuda_state,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: pytest.fail("rank capture must not enumerate visible CUDA devices"),
    )

    snapshot = capture_rank_rng_snapshot(
        rank=1,
        world_size=2,
        physical_cuda_device="GPU-deadbeef",
    )

    assert snapshot.rank == 1
    assert snapshot.world_size == 2
    assert snapshot.python_state == random.getstate()
    observed_numpy = snapshot.numpy_state
    current_numpy = np.random.get_state()
    assert observed_numpy[0] == current_numpy[0]
    assert np.array_equal(observed_numpy[1], current_numpy[1])
    assert torch.equal(snapshot.torch_cpu_state, torch.get_rng_state())
    assert snapshot.cuda_device_topology == ("cuda:0",)
    assert snapshot.cuda_device_binding == RankCudaDeviceBinding(
        rank=1,
        world_size=2,
        logical_device=1,
        physical_device="GPU-deadbeef",
    )
    assert observed_devices == [1]
    assert torch.equal(snapshot.torch_cuda_states[0], cuda_state)
    assert snapshot.torch_cuda_states[0] is not cuda_state

    with pytest.raises(ArtifactContractError) as device_info:
        capture_rank_rng_snapshot(
            rank=1,
            world_size=2,
            logical_cuda_device=0,
            physical_cuda_device="GPU-deadbeef",
        )
    assert device_info.value.code == "training_state.incompatible"

    bindings = (
        RankCudaDeviceBinding(0, 2, 0, "GPU-a"),
        RankCudaDeviceBinding(1, 2, 1, "GPU-b"),
    )
    assert build_exact_resume_topology_identity(bindings) != (
        build_exact_resume_topology_identity(
            (bindings[0], RankCudaDeviceBinding(1, 2, 1, "GPU-c"))
        )
    )


def test_two_rank_publication_uses_control_only_gathers_and_commits_once(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=2)
    payloads = [_rank_payload(rank=rank, world_size=2) for rank in range(2)]
    collectives = _ThreadCollectives(2)

    def publish(rank: int):
        return publish_distributed_exact_resume(
            checkpoint,
            plan=plan,
            rank=rank,
            local_payload=payloads[rank],
            barrier=collectives.barrier,
            gather_status=collectives.gather,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, range(2)))

    assert results[0].manifest.aggregate_digest == results[1].manifest.aggregate_digest
    assert results[0].manifest.world_size == 2
    assert (checkpoint / "training_state" / "manifest.json").is_file()
    assert [status.phase for status in collectives.gathered].count("begin") == 2
    assert [status.phase for status in collectives.gathered].count("contribute") == 2
    assert [status.phase for status in collectives.gathered].count("commit") == 2
    assert collectives.barrier_calls == 4
    assert not any(_contains_bytes(status) for status in collectives.gathered)


def test_prepublication_serialization_failure_converges_before_contribution(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=2)
    collectives = _ThreadCollectives(2)

    def prepare(rank: int) -> ArtifactContractError:
        model, optimizer, scheduler = _runtime(initialized=True)
        snapshot_rank = 0 if rank == 1 else rank
        try:
            prepare_distributed_exact_resume_contribution(
                plan=plan,
                rank=rank,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                cursor={
                    "data": {"ordinal": 20 + rank},
                    "pack": {"ordinal": 10 + rank},
                },
                next_rank_local_micro_step=30 + rank,
                rng_snapshot=_rng_snapshot(
                    rank=snapshot_rank,
                    world_size=2,
                ),
                gather_status=collectives.gather,
            )
        except ArtifactContractError as exc:
            return exc
        raise AssertionError("serialization unexpectedly succeeded")

    with ThreadPoolExecutor(max_workers=2) as executor:
        errors = list(executor.map(prepare, range(2)))

    assert [error.code for error in errors] == [
        "training_state.distributed_publication_failed",
        "training_state.distributed_publication_failed",
    ]
    assert errors[0].context == errors[1].context
    assert errors[0].context["failures"] == [
        {
            "error_code": "training_state.rank_mismatch",
            "error_type": "ArtifactContractError",
            "phase": "serialize",
            "rank": 1,
        }
    ]
    assert [status.phase for status in collectives.gathered] == [
        "serialize",
        "serialize",
    ]
    assert not any(_contains_bytes(status) for status in collectives.gathered)
    assert not (checkpoint / "training_state").exists()


def test_rank_local_contribution_failure_converges_without_commit(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=2)
    rank_zero_payload = _rank_payload(rank=0, world_size=2)
    collectives = _ThreadCollectives(2)

    def publish(rank: int) -> ArtifactContractError:
        try:
            publish_distributed_exact_resume(
                checkpoint,
                plan=plan,
                rank=rank,
                local_payload=rank_zero_payload,
                barrier=collectives.barrier,
                gather_status=collectives.gather,
            )
        except ArtifactContractError as exc:
            return exc
        raise AssertionError("publication unexpectedly succeeded")

    with ThreadPoolExecutor(max_workers=2) as executor:
        errors = list(executor.map(publish, range(2)))

    assert [error.code for error in errors] == [
        "training_state.distributed_publication_failed",
        "training_state.distributed_publication_failed",
    ]
    assert errors[0].context == errors[1].context
    assert errors[0].context["failures"] == [
        {
            "error_code": "training_state.rank_mismatch",
            "error_type": "ArtifactContractError",
            "phase": "contribute",
            "rank": 1,
        }
    ]
    assert not (checkpoint / "training_state").exists()
    assert not list(checkpoint.glob(".training_state.*.tmp"))
    assert [status.phase for status in collectives.gathered].count("abort") == 2
    assert collectives.barrier_calls == 4
    assert not any(_contains_bytes(status) for status in collectives.gathered)


@pytest.mark.parametrize("failure_mode", ["contribute", "commit"])
def test_two_rank_gloo_pre_manifest_failures_abort_owned_stage_without_residue(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    init_path = tmp_path / f"publication-{failure_mode}-gloo-init"
    init_method = f"file://{init_path}"
    context = torch.multiprocessing.get_context("fork")
    result_queue = context.Queue()
    payloads = tuple(_rank_payload(rank=rank, world_size=2) for rank in range(2))
    torch.multiprocessing.start_processes(
        _gloo_publication_failure_worker,
        args=(
            2,
            init_method,
            str(checkpoint),
            failure_mode,
            payloads,
            result_queue,
        ),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    results = sorted(
        (result_queue.get(timeout=10) for _rank in range(2)),
        key=lambda item: item["rank"],
    )
    result_queue.close()

    assert [result["code"] for result in results] == [
        "training_state.distributed_publication_failed",
        "training_state.distributed_publication_failed",
    ]
    assert results[0]["context"] == results[1]["context"]
    assert all(result["residue"] == [] for result in results)
    assert all(result["target_exists"] is False for result in results)
    assert all(
        result["gathered_phases"] == ["begin", "contribute", "commit", "abort"]
        for result in results
    )
    expected_code = (
        "training_state.rank_mismatch"
        if failure_mode == "contribute"
        else "training_state.injected_commit_failure"
    )
    expected_rank = 1 if failure_mode == "contribute" else 0
    assert results[0]["context"]["failures"] == [
        {
            "error_code": expected_code,
            "error_type": "ArtifactContractError",
            "phase": "contribute" if failure_mode == "contribute" else "commit",
            "rank": expected_rank,
        }
    ]


def test_post_manifest_commit_failure_retains_forensic_stage_and_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=2)
    payloads = [_rank_payload(rank=rank, world_size=2) for rank in range(2)]
    collectives = _ThreadCollectives(2)
    real_commit = training_state.commit_training_state_contributions

    def fail_after_manifest(checkpoint_dir: Path | str, session: Any):
        return real_commit(
            checkpoint_dir,
            session,
            on_manifest_written=lambda: (_ for _ in ()).throw(
                RuntimeError("injected post-manifest failure")
            ),
        )

    monkeypatch.setattr(
        exact_resume,
        "commit_training_state_contributions",
        fail_after_manifest,
    )

    def publish(rank: int) -> ArtifactContractError:
        try:
            publish_distributed_exact_resume(
                checkpoint,
                plan=plan,
                rank=rank,
                local_payload=payloads[rank],
                barrier=collectives.barrier,
                gather_status=collectives.gather,
            )
        except ArtifactContractError as exc:
            return exc
        raise AssertionError("publication unexpectedly succeeded")

    with ThreadPoolExecutor(max_workers=2) as executor:
        errors = list(executor.map(publish, range(2)))

    assert [error.code for error in errors] == [
        "training_state.distributed_publication_abort_failed",
        "training_state.distributed_publication_abort_failed",
    ]
    assert errors[0].context == errors[1].context
    assert errors[0].context["abort_failures"] == [
        {
            "error_code": "training_state.terminal_forensic_only",
            "error_type": "ArtifactContractError",
            "phase": "abort",
            "rank": 0,
        }
    ]
    stages = list(checkpoint.glob(".training_state.*.tmp"))
    assert len(stages) == 1
    assert (stages[0] / training_state.TRAINING_STATE_MANIFEST).is_file()
    assert (stages[0] / training_state.TRAINING_STATE_TERMINAL_FORENSIC).is_file()
    assert not (checkpoint / training_state.TRAINING_STATE_DIRECTORY).exists()


def test_admit_restore_returns_authenticated_next_cursor_position(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=1)
    payload = _rank_payload(rank=0, world_size=1)
    collectives = _ThreadCollectives(1)
    publish_distributed_exact_resume(
        checkpoint,
        plan=plan,
        rank=0,
        local_payload=payload,
        barrier=collectives.barrier,
        gather_status=collectives.gather,
    )
    model, optimizer, scheduler = _runtime(initialized=False)
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_cpu = torch.get_rng_state()
    restored_cuda: list[torch.Tensor] = []
    try:
        restored = admit_and_restore_current_rank(
            checkpoint,
            checkpoint_step=17,
            rank=0,
            world_size=1,
            identities=_identities(),
            resolved_config=_resolved_config(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            cuda_device_topology=("cuda:0",),
            get_torch_cuda_rng_states=lambda: (torch.zeros(32, dtype=torch.uint8),),
            set_torch_cuda_rng_states=lambda states: restored_cuda.extend(
                state.clone() for state in states
            ),
        )
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.set_rng_state(original_cpu)

    assert restored.next_rank_local_micro_step == 30
    assert restored.cursor["data"]["next_rank_local_micro_step"] == 30
    assert restored.cursor["data"]["state"] == {"epoch": 2, "ordinal": 20}
    assert restored.cursor["pack"]["state"] == {"ordinal": 10, "pending": []}
    assert len(restored_cuda) == 1


def test_resume_projection_allows_continuation_metadata_and_rejects_semantic_drift(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=1)
    collectives = _ThreadCollectives(1)
    publish_distributed_exact_resume(
        checkpoint,
        plan=plan,
        rank=0,
        local_payload=_rank_payload(rank=0, world_size=1),
        barrier=collectives.barrier,
        gather_status=collectives.gather,
    )
    child = copy.deepcopy(_resolved_config())
    child["config"]["run"] = {
        "artifact_root": "/outputs/child",
        "name": "child",
        "output_dir": "/outputs/child/segment",
    }
    child["config"]["resume"] = {
        "checkpoint_dir": str(checkpoint),
        "mode": "exact_same_world_size",
    }
    child["resolution"]["entry_config_path"] = "/configs/child.yaml"
    model, optimizer, scheduler = _runtime(initialized=False)
    restored_cuda: list[torch.Tensor] = []
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_cpu = torch.get_rng_state()
    try:
        restored = admit_and_restore_current_rank(
            checkpoint,
            checkpoint_step=17,
            rank=0,
            world_size=1,
            identities=_identities(child),
            resolved_config=child,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            cuda_device_topology=("cuda:0",),
            get_torch_cuda_rng_states=lambda: (torch.zeros(32, dtype=torch.uint8),),
            set_torch_cuda_rng_states=lambda states: restored_cuda.extend(states),
        )
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.set_rng_state(original_cpu)
    assert restored.next_rank_local_micro_step == 30
    assert len(restored_cuda) == 1

    drifted = copy.deepcopy(child)
    drifted["config"]["training"]["seed"] = 18
    model, optimizer, scheduler = _runtime(initialized=False)
    model_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    setter_calls: list[str] = []
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_and_restore_current_rank(
            checkpoint,
            checkpoint_step=17,
            rank=0,
            world_size=1,
            identities=_identities(drifted),
            resolved_config=drifted,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            cuda_device_topology=("cuda:0",),
            set_torch_cuda_rng_states=lambda _states: setter_calls.append("cuda"),
        )
    assert exc_info.value.code == "training_state.incompatible"
    assert setter_calls == []
    assert all(
        torch.equal(dict(model.named_parameters())[name], value)
        for name, value in model_before.items()
    )


def test_distributed_restore_success_returns_control_only_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=1)
    publication_collectives = _ThreadCollectives(1)
    publish_distributed_exact_resume(
        checkpoint,
        plan=plan,
        rank=0,
        local_payload=_rank_payload(rank=0, world_size=1),
        barrier=publication_collectives.barrier,
        gather_status=publication_collectives.gather,
    )
    model, optimizer, scheduler = _runtime(initialized=False)
    restore_collectives = _ThreadCollectives(1)
    cuda_state = [torch.arange(32, dtype=torch.uint8) + 100]
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_cpu = torch.get_rng_state()

    def get_cuda() -> Sequence[torch.Tensor]:
        return tuple(state.clone() for state in cuda_state)

    def set_cuda(states: Sequence[torch.Tensor]) -> None:
        cuda_state[:] = [state.clone() for state in states]

    try:
        receipt = restore_distributed_exact_resume(
            checkpoint,
            checkpoint_step=17,
            expected_manifest_digest=load_training_state_manifest(
                checkpoint
            ).aggregate_digest,
            rank=0,
            world_size=1,
            identities=_identities(),
            resolved_config=_resolved_config(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            cuda_device_topology=("cuda:0",),
            gather_status=restore_collectives.gather,  # type: ignore[arg-type]
            get_torch_cuda_rng_states=get_cuda,
            set_torch_cuda_rng_states=set_cuda,
        )
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.set_rng_state(original_cpu)

    assert isinstance(receipt, DistributedExactResumeRestoreReceipt)
    assert receipt.rank == 0
    assert receipt.world_size == 1
    assert receipt.restored.next_rank_local_micro_step == 30
    assert [status.phase for status in receipt.admission_statuses] == ["restore_admit"]
    assert [status.phase for status in receipt.apply_statuses] == ["restore_apply"]
    assert not _contains_bytes(receipt.admission_statuses)
    assert not _contains_bytes(receipt.apply_statuses)


def test_manifest_swap_after_prior_admission_fails_before_any_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, "selected")
    replacement = _checkpoint(tmp_path, "replacement")
    selected_plan = _plan(world_size=1)
    replacement_plan = dataclasses.replace(
        selected_plan,
        parent_run_id="run-replacement",
        parent_segment_id="segment-replacement",
        continuation_index=3,
    )
    selected_collectives = _ThreadCollectives(1)
    replacement_collectives = _ThreadCollectives(1)
    publish_distributed_exact_resume(
        checkpoint,
        plan=selected_plan,
        rank=0,
        local_payload=_rank_payload(rank=0, world_size=1),
        barrier=selected_collectives.barrier,
        gather_status=selected_collectives.gather,
    )
    publish_distributed_exact_resume(
        replacement,
        plan=replacement_plan,
        rank=0,
        local_payload=_rank_payload(rank=0, world_size=1),
        barrier=replacement_collectives.barrier,
        gather_status=replacement_collectives.gather,
    )
    prior_admitted_digest = load_training_state_manifest(checkpoint).aggregate_digest
    replacement_digest = load_training_state_manifest(replacement).aggregate_digest
    assert replacement_digest != prior_admitted_digest
    original_state = checkpoint / "training_state"
    original_state.rename(checkpoint / "training_state.prior")
    (replacement / "training_state").rename(original_state)

    model, optimizer, scheduler = _runtime(initialized=False)
    model_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    get_cuda_calls: list[str] = []
    set_cuda_calls: list[str] = []
    restore_collectives = _ThreadCollectives(1)

    with pytest.raises(ArtifactContractError) as exc_info:
        restore_distributed_exact_resume(
            checkpoint,
            checkpoint_step=17,
            expected_manifest_digest=prior_admitted_digest,
            rank=0,
            world_size=1,
            identities=_identities(),
            resolved_config=_resolved_config(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            cuda_device_topology=("cuda:0",),
            gather_status=restore_collectives.gather,  # type: ignore[arg-type]
            get_torch_cuda_rng_states=lambda: get_cuda_calls.append("get") or (),
            set_torch_cuda_rng_states=lambda _states: set_cuda_calls.append("set"),
        )

    assert exc_info.value.code == "training_state.distributed_restore_failed"
    assert exc_info.value.context["admission_failures"] == [
        {
            "error_code": "training_state.admitted_manifest_drift",
            "error_type": "ArtifactContractError",
            "phase": "restore_admit",
            "rank": 0,
        }
    ]
    assert get_cuda_calls == []
    assert set_cuda_calls == []
    assert all(
        torch.equal(dict(model.named_parameters())[name], value)
        for name, value in model_before.items()
    )
    assert optimizer.state_dict() == optimizer_before
    assert scheduler.state_dict() == scheduler_before


def test_two_rank_gloo_apply_failure_rolls_back_every_runtime_owner(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=2, scaler_applicable=True)
    payloads = [
        _rank_payload(rank=rank, world_size=2, scaler_applicable=True)
        for rank in range(2)
    ]
    publication_collectives = _ThreadCollectives(2)

    def publish(rank: int):
        return publish_distributed_exact_resume(
            checkpoint,
            plan=plan,
            rank=rank,
            local_payload=payloads[rank],
            barrier=publication_collectives.barrier,
            gather_status=publication_collectives.gather,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(publish, range(2)))

    init_path = tmp_path / "restore-gloo-init"
    init_method = f"file://{init_path}"
    context = torch.multiprocessing.get_context("fork")
    result_queue = context.Queue()
    torch.multiprocessing.start_processes(
        _gloo_restore_failure_worker,
        args=(2, init_method, str(checkpoint), result_queue),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    results = sorted(
        (result_queue.get(timeout=10) for _rank in range(2)),
        key=lambda item: item["rank"],
    )
    result_queue.close()

    assert [result["code"] for result in results] == [
        "training_state.distributed_restore_failed",
        "training_state.distributed_restore_failed",
    ]
    assert results[0]["context"] == results[1]["context"]
    assert results[0]["context"]["apply_failures"] == [
        {
            "error_code": "training_state.restore_failed",
            "error_type": "ArtifactContractError",
            "phase": "restore_apply",
            "rank": 1,
        }
    ]
    assert results[0]["context"]["rolled_back_ranks"] == [0, 1]
    for result in results:
        assert result["gathered_phases"] == [
            "restore_admit",
            "restore_apply",
            "restore_rollback",
        ]
        assert result["seed_calls"] == []
        assert result["cuda_get_calls"] == 2
        assert result["model_restored"] is True
        assert result["optimizer_restored"] is True
        assert result["scheduler_restored"] is True
        assert result["scaler_restored"] is True
        assert result["python_restored"] is True
        assert result["numpy_restored"] is True
        assert result["torch_cpu_restored"] is True
        assert result["cuda_restored"] is True


@pytest.mark.parametrize("mismatch", ["identity", "topology"])
def test_identity_and_topology_mismatch_fail_before_runtime_mutation(
    tmp_path: Path, mismatch: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    plan = _plan(world_size=1)
    collectives = _ThreadCollectives(1)
    publish_distributed_exact_resume(
        checkpoint,
        plan=plan,
        rank=0,
        local_payload=_rank_payload(rank=0, world_size=1),
        barrier=collectives.barrier,
        gather_status=collectives.gather,
    )
    model, optimizer, scheduler = _runtime(initialized=False)
    model_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    setter_calls: list[str] = []
    identities = dict(_identities())
    topology = ("cuda:0",)
    if mismatch == "identity":
        identities["cache"] = "f" * 64
    else:
        topology = ("cuda:1",)

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_and_restore_current_rank(
            checkpoint,
            checkpoint_step=17,
            rank=0,
            world_size=1,
            identities=identities,
            resolved_config=_resolved_config(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            cuda_device_topology=topology,
            set_torch_cuda_rng_states=lambda _states: setter_calls.append("cuda"),
        )

    assert exc_info.value.code == "training_state.incompatible"
    assert setter_calls == []
    assert all(
        torch.equal(dict(model.named_parameters())[name], value)
        for name, value in model_before.items()
    )
    assert optimizer.state_dict() == optimizer_before


def test_mid_accumulation_serialization_is_rejected_explicitly() -> None:
    model, optimizer, scheduler = _runtime(initialized=True)

    with pytest.raises(ArtifactContractError) as exc_info:
        serialize_current_rank_training_state(
            rank=0,
            world_size=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            cursor={
                "data": {"ordinal": 20},
                "pack": {"ordinal": 10},
            },
            next_rank_local_micro_step=30,
            accumulation_microstep=1,
            rng_snapshot=_rng_snapshot(rank=0, world_size=1),
        )

    assert exc_info.value.code == "training_state.unsupported_mid_accumulation"
