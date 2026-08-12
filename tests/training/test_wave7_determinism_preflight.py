from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/probes/coordexp_swift/wave7_determinism_preflight.py"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed["receipt_payload_sha256"] = _digest(payload)
    path.write_bytes(_canonical(signed) + b"\n")
    return signed


def _write_native_reference(
    preflight: ModuleType, path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.artifacts import provenance

    source = Path(provenance.__file__).resolve()
    payload = {
        "schema": preflight.NATIVE_REFERENCE_SCHEMA,
        "created_at": "2026-08-10T00:00:00+00:00",
        "repository_root": str(REPO_ROOT),
        "source_identity": {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "attention_backend": "flash_attention_2",
        "pinned_runtime_baseline": {
            "schema_version": 3,
            "baseline_sha256": provenance.PINNED_RUNTIME_BASELINE_SHA256,
            "admitted": True,
        },
        "cuda": {"current_device": 0},
        "mapped_native_execution": {
            "schema_version": 1,
            "cuda_initialized": True,
            "admitted": True,
            "mismatches": [],
            "components": {},
            "mapped_cudnn_components": [],
        },
        "terminal_status": "passed",
    }
    receipt = dict(payload)
    receipt["receipt_sha256"] = _digest(payload)
    path.write_bytes(_canonical(receipt) + b"\n")
    binding = {
        "path": str(path.resolve()),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "payload_sha256": receipt["receipt_sha256"],
        "schema": receipt["schema"],
        "status": receipt["terminal_status"],
    }
    return receipt, binding


def _canonical_native_reference(
    preflight: ModuleType,
) -> tuple[Path, dict[str, Any]]:
    path = (REPO_ROOT / preflight._CANONICAL_NATIVE_REFERENCE_RELATIVE_PATH).resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    return path, {
        "path": str(path),
        "file_sha256": preflight._CANONICAL_NATIVE_REFERENCE_FILE_SHA256,
        "payload_sha256": preflight._CANONICAL_NATIVE_REFERENCE_PAYLOAD_SHA256,
        "schema": receipt["schema"],
        "status": receipt["terminal_status"],
    }


def _late_expectations(tmp_path: Path) -> dict[str, Any]:
    result = {}
    for component, distribution_name, filename in (
        ("flash_attention_2", "flash-attn", "flash_attn_2_cuda.test.so"),
        ("libcublas", "nvidia-cublas-cu12", "libcublas.so.12"),
        ("libnccl", "nvidia-nccl-cu12", "libnccl.so.2"),
    ):
        path = tmp_path / filename
        path.write_bytes(component.encode())
        result[component] = {
            "distribution": distribution_name,
            "soname": filename,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


def _write_runtime_admission(
    preflight: ModuleType,
    path: Path,
    *,
    native_binding: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    from src.artifacts import provenance

    baseline = provenance.pinned_runtime_baseline()
    provenance_payload = {
        "schema_version": 1,
        "repository": {},
        "dependencies": baseline["dependencies"],
        "runtime": baseline["runtime"],
    }
    source = Path(provenance.__file__).resolve()
    comparison = provenance.require_pinned_runtime_baseline(
        provenance=provenance_payload,
        attention_backend="flash_attention_2",
    )
    return _write_receipt(
        path,
        {
            "schema": preflight.RUNTIME_ADMISSION_SCHEMA,
            "status": "passed",
            "model_loaded": False,
            "cuda_initialized": False,
            "repository_root": str(REPO_ROOT),
            "source_identity": {
                "path": str(source),
                "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "native_reference": native_binding,
            "attention_backend": "flash_attention_2",
            "provenance": provenance_payload,
            "pinned_runtime_baseline": comparison,
            "late_native_expectations": _late_expectations(tmp_path),
        },
    )


def _write_fake_accelerate(path: Path, body: str = "") -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('Accelerate test 1.0')\n"
        "    raise SystemExit(0)\n" + body,
        encoding="utf-8",
    )
    path.chmod(0o700)


@pytest.fixture
def preflight() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wave7_determinism_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(
    preflight: ModuleType, tmp_path: Path
) -> tuple[dict[str, Path], dict[str, Any]]:
    native_path, native_binding = _canonical_native_reference(preflight)
    runtime_path = tmp_path / "runtime.json"
    _write_runtime_admission(
        preflight,
        runtime_path,
        native_binding=native_binding,
        tmp_path=tmp_path,
    )
    executable = tmp_path / "accelerate"
    _write_fake_accelerate(executable)
    paths = {
        "plan": tmp_path / "plan.json",
        "marker": tmp_path / "attempt.json",
        "terminal": tmp_path / "terminal.json",
        "ranks": tmp_path / "ranks",
        "runtime": runtime_path,
        "native": native_path,
    }
    plan = preflight.prepare_plan(
        plan_path=paths["plan"],
        attempt_marker_path=paths["marker"],
        terminal_receipt_path=paths["terminal"],
        rank_receipt_root=paths["ranks"],
        runtime_receipt_path=runtime_path,
        expected_runtime_file_sha256=hashlib.sha256(
            runtime_path.read_bytes()
        ).hexdigest(),
        native_runtime_receipt_path=native_path,
        expected_native_runtime_file_sha256=hashlib.sha256(
            native_path.read_bytes()
        ).hexdigest(),
        accelerate_executable=str(executable),
    )
    return paths, plan


def _rank_payload(
    preflight: ModuleType,
    *,
    plan: dict[str, Any],
    launch_id: str,
    rank: int,
    workload_digest: str = "d" * 64,
) -> dict[str, Any]:
    local_rank = rank
    runtime = json.loads(Path(plan["runtime_admission"]["path"]).read_text())
    late_components = {
        name: {
            "path": identity["path"],
            "size_bytes": identity["size_bytes"],
            "file_sha256": identity["file_sha256"],
            "soname": identity["soname"],
            "origin_matches_admission": True,
        }
        for name, identity in runtime["late_native_expectations"].items()
    }
    workload_components = {
        "cublas": {"digest": workload_digest},
        "flash_attention_2": {"digest": workload_digest},
        "nccl_all_reduce": {
            "digest": workload_digest,
            "expected_sum": 28.0,
        },
    }
    payload = {
        "schema": preflight.RANK_RECEIPT_SCHEMA,
        "status": "passed",
        "plan_sha256": plan["plan_payload_sha256"],
        "launch_id": launch_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": 8,
        "device_index": local_rank,
        "device_identity": {
            "current_device": rank,
            "visible_index": rank,
            "gpu_uuid": f"GPU-test-{rank}",
            "pci_bus_id": f"00000000:{rank:02X}:00.0",
        },
        "pre_cuda": {
            "cuda_initialized": False,
            "environment": dict(plan["environment"]),
            "rank_environment": {
                "RANK": str(rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": "8",
            },
        },
        "determinism_policy": preflight._expected_policy_receipt(),
        "runtime_admission": dict(plan["runtime_admission"]),
        "native_runtime_reference": dict(plan["native_runtime_reference"]),
        "mapped_native_execution": {
            "schema_version": 2,
            "cuda_initialized": True,
            "admitted": True,
            "mismatches": [],
            "pre_workload_attestation_sha256": "a" * 64,
            "post_workload_attestation_sha256": "b" * 64,
            "late_native_mappings": {
                "schema_version": 1,
                "cuda_initialized": True,
                "admitted": True,
                "mismatches": [],
                "components": late_components,
            },
        },
        "workloads": {
            **workload_components,
            "aggregate_sha256": _digest(workload_components),
        },
        "cleanup": {"process_group_destroyed": True},
        "claim_scope": dict(plan["claim_scope"]),
    }
    return preflight.finalize_receipt(payload)


def _install_fake_launches(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
    *,
    digest_by_launch: dict[str, str] | None = None,
    fail_launch: str | None = None,
) -> list[str]:
    observed: list[str] = []

    def fake_launch(**kwargs: Any) -> dict[str, Any]:
        launch_id = kwargs["launch_id"]
        observed.append(launch_id)
        assert Path(plan["artifact_targets"]["attempt_marker"]).exists()
        if launch_id == fail_launch:
            raise preflight.PreflightError(
                "synthetic launch failure", code="test.launch_failure"
            )
        digest = (digest_by_launch or {}).get(launch_id, "d" * 64)
        rank_dir = Path(plan["artifact_targets"]["rank_receipt_root"])
        for rank in range(8):
            target = rank_dir / launch_id / f"rank-{rank:05d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                _canonical(
                    _rank_payload(
                        preflight,
                        plan=plan,
                        launch_id=launch_id,
                        rank=rank,
                        workload_digest=digest,
                    )
                )
                + b"\n"
            )
        pid = 9000 + len(observed)
        member = {
            "pid": pid,
            "state": "R",
            "parent_pid": 1,
            "process_group_id": pid,
            "session_id": pid,
            "start_time_ticks": 100 + len(observed),
        }
        return {
            "launch_id": launch_id,
            "pid": pid,
            "returncode": 0,
            "termination": "exited",
            "process_scope": {
                "session_id": pid,
                "leader": {
                    "pid": pid,
                    "start_time_ticks": member["start_time_ticks"],
                },
                "observed_members": [member],
                "term_sent": False,
                "kill_sent": False,
                "remaining_members": [],
                "leader_reaped": True,
                "cleanup_failure": None,
                "cleanup_verified": True,
            },
            "cleanup_verified": True,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(preflight, "_run_accelerate_launch", fake_launch)
    shared = [{"gpu_uuid": "GPU-test-0", "pid": 111}]
    baseline_samples = [
        _gpu_sample(index, compute_processes=shared) for index in range(2)
    ]
    baseline = {
        "samples": baseline_samples,
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
        ],
        "preexisting_compute_processes": shared,
    }
    monkeypatch.setattr(
        preflight,
        "_collect_stable_gpu_idle_samples",
        lambda _plan: baseline,
    )
    monkeypatch.setattr(
        preflight,
        "_collect_postlaunch_gpu_sweep",
        lambda *, plan, baseline, phase: preflight._build_postlaunch_gpu_sweep(
            plan=plan,
            baseline=baseline,
            phase=phase,
            samples=[
                _gpu_sample(0, compute_processes=shared),
                _gpu_sample(1, compute_processes=[]),
            ],
        ),
    )
    return observed


def _gpu_sample(
    sample_index: int,
    *,
    compute_processes: list[dict[str, Any]] | None = None,
    memory_used_mib: int = 4096,
    memory_total_mib: int = 81920,
    sample_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "sample_monotonic_ns": (
            sample_index * 2_000_000_000
            if sample_monotonic_ns is None
            else sample_monotonic_ns
        ),
        "gpus": [
            {
                "index": index,
                "gpu_uuid": f"GPU-test-{index}",
                "utilization_percent": (index * 17 + sample_index) % 101,
                "memory_used_mib": memory_used_mib,
                "memory_total_mib": memory_total_mib,
                "memory_headroom_mib": memory_total_mib - memory_used_mib,
            }
            for index in range(8)
        ],
        "compute_processes": list(compute_processes or []),
    }


def test_stable_shared_gpu_baseline_is_admitted_and_bound(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    shared = [
        {"gpu_uuid": "GPU-test-0", "pid": 111},
        {"gpu_uuid": "GPU-test-7", "pid": 777},
    ]
    samples = [
        _gpu_sample(0, compute_processes=shared),
        _gpu_sample(1, compute_processes=list(reversed(shared))),
    ]
    monkeypatch.setattr(
        preflight,
        "_collect_gpu_sample",
        lambda _contract, index: samples[index],
    )
    monkeypatch.setattr(preflight.time, "sleep", lambda _seconds: None)

    baseline = preflight._collect_stable_gpu_idle_samples(plan)

    assert baseline == {
        "samples": samples,
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
        ],
        "preexisting_compute_processes": shared,
    }


def test_shared_gpu_baseline_compute_inventory_change_is_rejected(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    samples = [
        _gpu_sample(
            0,
            compute_processes=[{"gpu_uuid": "GPU-test-0", "pid": 111}],
        ),
        _gpu_sample(
            1,
            compute_processes=[{"gpu_uuid": "GPU-test-0", "pid": 222}],
        ),
    ]
    monkeypatch.setattr(
        preflight,
        "_collect_gpu_sample",
        lambda _contract, index: samples[index],
    )
    monkeypatch.setattr(preflight.time, "sleep", lambda _seconds: None)

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight._collect_stable_gpu_idle_samples(plan)

    assert exc_info.value.code == "controller.gpu_baseline_changed"


def test_shared_gpu_baseline_memory_ceiling_is_rejected(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    samples = [_gpu_sample(index, memory_used_mib=49153) for index in range(2)]
    monkeypatch.setattr(
        preflight,
        "_collect_gpu_sample",
        lambda _contract, index: samples[index],
    )
    monkeypatch.setattr(preflight.time, "sleep", lambda _seconds: None)

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight._collect_stable_gpu_idle_samples(plan)

    assert exc_info.value.code == "controller.gpu_baseline_headroom"


def test_shared_gpu_baseline_new_memory_ceiling_is_admitted(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    samples = [_gpu_sample(index, memory_used_mib=49152) for index in range(2)]
    monkeypatch.setattr(
        preflight,
        "_collect_gpu_sample",
        lambda _contract, index: samples[index],
    )
    monkeypatch.setattr(preflight.time, "sleep", lambda _seconds: None)

    baseline = preflight._collect_stable_gpu_idle_samples(plan)

    assert baseline["samples"] == samples


def test_shared_gpu_baseline_missing_memory_total_is_rejected(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    samples = [_gpu_sample(index) for index in range(2)]
    samples[0]["gpus"][0].pop("memory_total_mib")
    monkeypatch.setattr(
        preflight,
        "_collect_gpu_sample",
        lambda _contract, index: samples[index],
    )
    monkeypatch.setattr(preflight.time, "sleep", lambda _seconds: None)

    with pytest.raises(preflight.PreflightError):
        preflight._collect_stable_gpu_idle_samples(plan)


def test_shared_gpu_baseline_wrong_memory_total_is_rejected(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    samples = [_gpu_sample(index, memory_total_mib=81919) for index in range(2)]
    monkeypatch.setattr(
        preflight,
        "_collect_gpu_sample",
        lambda _contract, index: samples[index],
    )
    monkeypatch.setattr(preflight.time, "sleep", lambda _seconds: None)

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight._collect_stable_gpu_idle_samples(plan)

    assert exc_info.value.code == "controller.gpu_memory_geometry"


def test_postlaunch_subset_of_preexisting_compute_inventory_is_admitted(
    preflight: ModuleType, tmp_path: Path
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    baseline = {
        "samples": [],
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
        ],
        "preexisting_compute_processes": [
            {"gpu_uuid": "GPU-test-0", "pid": 111},
            {"gpu_uuid": "GPU-test-7", "pid": 777},
        ],
    }
    samples = [
        _gpu_sample(
            0,
            compute_processes=[{"gpu_uuid": "GPU-test-7", "pid": 777}],
        ),
        _gpu_sample(1, compute_processes=[]),
    ]

    sweep = preflight._build_postlaunch_gpu_sweep(
        plan=plan,
        baseline=baseline,
        phase="post-launch-a",
        samples=samples,
    )

    assert sweep["admitted"] is True
    assert sweep["new_compute_processes"] == []
    preflight._validate_postlaunch_gpu_sweep(sweep, plan=plan, baseline=baseline)


def test_postlaunch_new_compute_row_is_terminal_failure_evidence(
    preflight: ModuleType, tmp_path: Path
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    baseline = {
        "samples": [],
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
        ],
        "preexisting_compute_processes": [
            {"gpu_uuid": "GPU-test-0", "pid": 111},
        ],
    }
    samples = [
        _gpu_sample(
            0,
            compute_processes=[
                {"gpu_uuid": "GPU-test-0", "pid": 111},
                {"gpu_uuid": "GPU-test-3", "pid": 333},
            ],
        ),
        _gpu_sample(
            1,
            compute_processes=[{"gpu_uuid": "GPU-test-0", "pid": 111}],
        ),
    ]

    sweep = preflight._build_postlaunch_gpu_sweep(
        plan=plan,
        baseline=baseline,
        phase="post-launch-a",
        samples=samples,
    )

    assert sweep["admitted"] is False
    assert sweep["new_compute_processes"] == [{"gpu_uuid": "GPU-test-3", "pid": 333}]
    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight._validate_postlaunch_gpu_sweep(sweep, plan=plan, baseline=baseline)
    assert exc_info.value.code == "controller.gpu_new_compute_process"


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [_gpu_sample(0)],
    ],
    ids=["zero-samples", "one-sample"],
)
def test_postlaunch_sweep_requires_exactly_two_samples(
    preflight: ModuleType,
    tmp_path: Path,
    samples: list[dict[str, Any]],
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    baseline = {
        "samples": [_gpu_sample(0), _gpu_sample(1)],
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
        ],
        "preexisting_compute_processes": [],
    }

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight._build_postlaunch_gpu_sweep(
            plan=plan,
            baseline=baseline,
            phase="post-launch-a",
            samples=samples,
        )

    assert exc_info.value.code == "controller.gpu_postlaunch_samples"


def test_postlaunch_sweep_rejects_less_than_two_second_separation(
    preflight: ModuleType, tmp_path: Path
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    baseline = {
        "samples": [_gpu_sample(0), _gpu_sample(1)],
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
        ],
        "preexisting_compute_processes": [],
    }
    samples = [
        _gpu_sample(0, sample_monotonic_ns=7_000_000_000),
        _gpu_sample(1, sample_monotonic_ns=8_000_000_000),
    ]

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight._build_postlaunch_gpu_sweep(
            plan=plan,
            baseline=baseline,
            phase="post-launch-a",
            samples=samples,
        )

    assert exc_info.value.code == "controller.gpu_postlaunch_interval"


def test_postlaunch_sweep_accepts_exact_two_second_separation(
    preflight: ModuleType, tmp_path: Path
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    baseline = {
        "samples": [_gpu_sample(0), _gpu_sample(1)],
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
        ],
        "preexisting_compute_processes": [],
    }
    samples = [
        _gpu_sample(0, sample_monotonic_ns=7_000_000_000),
        _gpu_sample(1, sample_monotonic_ns=9_000_000_000),
    ]

    sweep = preflight._build_postlaunch_gpu_sweep(
        plan=plan,
        baseline=baseline,
        phase="post-launch-a",
        samples=samples,
    )

    assert sweep["admitted"] is True
    assert sweep["samples"] == samples


def test_new_schema_rejects_idle_v1_plan_substitution(
    preflight: ModuleType, tmp_path: Path
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    legacy = dict(plan)
    legacy.pop("plan_payload_sha256")
    legacy["schema"] = "coordexp-swift-wave7-determinism-preflight-plan-v1"
    legacy.pop("gpu_shared_preexisting_baseline_contract")
    legacy["gpu_idle_contract"] = {
        "device_indices": list(range(8)),
        "sample_count": 2,
        "sample_interval_seconds": 2.0,
        "max_utilization_percent": 0,
        "max_memory_used_mib": 1024,
        "require_no_compute_processes": True,
    }
    legacy = preflight._finalize_plan(legacy)

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight._validate_plan(legacy, plan_path=paths["plan"].resolve())

    assert exc_info.value.code == "preflight.plan_schema"


def test_direct_script_cli_bootstraps_repo_imports_and_preserves_command_grammar(
    tmp_path: Path,
) -> None:
    _exercise_direct_script_cli(tmp_path)


def test_prepare_plan_freezes_exact_two_launch_contract(
    preflight: ModuleType, tmp_path: Path
) -> None:
    paths, plan = _fixture(preflight, tmp_path)

    assert plan["schema"] == preflight.PLAN_SCHEMA
    assert plan["status"] == "prepared"
    assert plan["environment"] == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "FLASH_ATTENTION_DETERMINISTIC": "1",
    }
    assert [item["launch_id"] for item in plan["launch_contract"]] == [
        "launch-a",
        "launch-b",
    ]
    assert all(item["num_processes"] == 8 for item in plan["launch_contract"])
    assert plan["workload_contract"]["flash_attention_2"] == {
        "batch_total_tokens": 8,
        "causal": True,
        "cu_seqlens": [0, 3, 8],
        "deterministic": True,
        "dropout_p": 0.0,
        "head_dim": 64,
        "heads": 2,
        "max_seqlen": 5,
    }
    assert plan["claim_scope"]["classification"] == "plumbing_only"
    assert plan["schema"].endswith("plan-v4")
    assert plan["gpu_shared_preexisting_baseline_contract"] == {
        "device_indices": list(range(8)),
        "sample_count": 2,
        "sample_interval_seconds": 2.0,
        "utilization_percent_range": [0, 100],
        "required_total_memory_mib": 81920,
        "max_prelaunch_memory_used_mib": 49152,
        "min_prelaunch_memory_headroom_mib": 32768,
        "compute_process_identity_fields": ["gpu_uuid", "pid"],
        "require_stable_device_inventory": True,
        "require_stable_compute_process_inventory": True,
        "postlaunch_compute_process_policy": "subset_of_preexisting_baseline",
    }
    assert (
        "performance_under_shared_gpu_occupancy"
        in plan["claim_scope"]["does_not_establish"]
    )
    assert (
        "resource_comparisons_under_shared_gpu_occupancy"
        in plan["claim_scope"]["does_not_establish"]
    )
    assert paths["plan"].exists()
    assert not paths["marker"].exists()
    assert not paths["terminal"].exists()
    assert not paths["ranks"].exists()

    with pytest.raises(preflight.PreflightError, match="already exists"):
        preflight.prepare_plan(
            plan_path=paths["plan"],
            attempt_marker_path=paths["marker"],
            terminal_receipt_path=paths["terminal"],
            rank_receipt_root=paths["ranks"],
            runtime_receipt_path=paths["runtime"],
            expected_runtime_file_sha256=plan["runtime_admission"]["file_sha256"],
            native_runtime_receipt_path=paths["native"],
            expected_native_runtime_file_sha256=plan["native_runtime_reference"][
                "file_sha256"
            ],
        )


def test_run_publishes_marker_then_two_launch_terminal(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    observed = _install_fake_launches(preflight, monkeypatch, plan)

    terminal = preflight.run_plan(
        paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
    )

    assert observed == ["launch-a", "launch-b"]
    assert terminal["schema"] == preflight.TERMINAL_RECEIPT_SCHEMA
    assert terminal["status"] == "passed"
    assert terminal["world_size"] == 8
    assert terminal["launch_count"] == 2
    assert terminal["mismatches"] == []
    assert terminal["claim_scope"]["classification"] == "plumbing_only"
    assert len(terminal["rank_receipts"]) == 16
    assert terminal["comparisons"]["launch_digest_equal"] is True
    assert paths["marker"].exists()
    assert paths["terminal"].exists()
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    assert marker["schema"].endswith("marker-v4")
    assert marker["gpu_shared_preexisting_baseline"][
        "preexisting_compute_processes"
    ] == [{"gpu_uuid": "GPU-test-0", "pid": 111}]
    assert (
        marker["gpu_shared_preexisting_baseline"]["samples"][0]["gpus"][0][
            "memory_total_mib"
        ]
        == 81920
    )
    assert (
        marker["gpu_shared_preexisting_baseline"]["samples"][0]["gpus"][0][
            "memory_headroom_mib"
        ]
        == 77824
    )
    assert [sweep["phase"] for sweep in terminal["gpu_shared_occupancy_sweeps"]] == [
        "post-launch-a",
        "post-launch-b",
        "preterminal",
    ]
    assert all(
        sweep["admitted"] is True for sweep in terminal["gpu_shared_occupancy_sweeps"]
    )
    assert (
        terminal["gpu_shared_occupancy_sweeps"][0]["samples"][0]["gpus"][0][
            "memory_total_mib"
        ]
        == 81920
    )


def test_run_rejects_resigned_replacement_plan_before_any_side_effect(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    expected_plan_file_sha256 = _file_sha256(paths["plan"])
    alternate_targets = {
        "attempt_marker": str(tmp_path / "alternate-attempt.json"),
        "terminal_receipt": str(tmp_path / "alternate-terminal.json"),
        "rank_receipt_root": str(tmp_path / "alternate-ranks"),
    }
    alternate_payload = dict(plan)
    alternate_payload.pop("plan_payload_sha256")
    alternate_payload["artifact_targets"] = alternate_targets
    alternate_plan = preflight._finalize_plan(alternate_payload)
    paths["plan"].write_bytes(_canonical(alternate_plan) + b"\n")
    gpu_calls: list[str] = []
    launch_calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "_collect_stable_gpu_idle_samples",
        lambda _plan: gpu_calls.append("gpu"),
    )
    monkeypatch.setattr(
        preflight,
        "_run_accelerate_launch",
        lambda **_kwargs: launch_calls.append("launch"),
    )

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight.run_plan(
            paths["plan"],
            expected_plan_file_sha256=expected_plan_file_sha256,
        )

    assert exc_info.value.code == "preflight.plan_file_sha"
    assert gpu_calls == []
    assert launch_calls == []
    assert not paths["marker"].exists()
    assert not paths["terminal"].exists()
    assert not paths["ranks"].exists()
    assert not Path(alternate_targets["attempt_marker"]).exists()
    assert not Path(alternate_targets["terminal_receipt"]).exists()
    assert not Path(alternate_targets["rank_receipt_root"]).exists()


@pytest.mark.parametrize("case", ["missing", "symlink", "non_regular"])
def test_run_rejects_plan_that_is_not_an_exact_regular_nonsymlink_file(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    paths, _plan = _fixture(preflight, tmp_path)
    expected_plan_file_sha256 = _file_sha256(paths["plan"])
    requested_plan = paths["plan"]
    if case == "missing":
        requested_plan.unlink()
    elif case == "symlink":
        target = tmp_path / "plan-target.json"
        requested_plan.replace(target)
        requested_plan.symlink_to(target)
    else:
        requested_plan.unlink()
        requested_plan.mkdir()
    gpu_calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "_collect_stable_gpu_idle_samples",
        lambda _plan: gpu_calls.append("gpu"),
    )

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight.run_plan(
            requested_plan,
            expected_plan_file_sha256=expected_plan_file_sha256,
        )

    assert exc_info.value.code == "preflight.plan_file"
    assert gpu_calls == []
    assert not paths["marker"].exists()
    assert not paths["terminal"].exists()
    assert not paths["ranks"].exists()


def test_run_rejects_malformed_expected_plan_file_sha_before_parsing(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _plan = _fixture(preflight, tmp_path)
    load_calls: list[Path] = []
    original_load_json = preflight._load_json
    monkeypatch.setattr(
        preflight,
        "_load_json",
        lambda path: (load_calls.append(path), original_load_json(path))[1],
    )

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight.run_plan(paths["plan"], expected_plan_file_sha256="A" * 64)

    assert exc_info.value.code == "preflight.plan_file_sha"
    assert load_calls == []
    assert not paths["marker"].exists()
    assert not paths["terminal"].exists()
    assert not paths["ranks"].exists()


def test_run_rehashes_plan_after_gpu_admission_immediately_before_marker(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    expected_plan_file_sha256 = _file_sha256(paths["plan"])
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    observed_launches = _install_fake_launches(preflight, monkeypatch, plan)
    collect_gpu_baseline = preflight._collect_stable_gpu_idle_samples

    def mutate_plan_after_gpu_admission(loaded_plan: dict[str, Any]) -> dict[str, Any]:
        baseline = collect_gpu_baseline(loaded_plan)
        replacement_payload = dict(plan)
        replacement_payload.pop("plan_payload_sha256")
        replacement_payload["claim_scope"] = dict(plan["claim_scope"])
        replacement_payload["claim_scope"]["classification"] = "replacement"
        replacement = preflight._finalize_plan(replacement_payload)
        paths["plan"].write_bytes(_canonical(replacement) + b"\n")
        return baseline

    monkeypatch.setattr(
        preflight,
        "_collect_stable_gpu_idle_samples",
        mutate_plan_after_gpu_admission,
    )

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight.run_plan(
            paths["plan"],
            expected_plan_file_sha256=expected_plan_file_sha256,
        )

    assert exc_info.value.code == "preflight.plan_file_sha"
    assert observed_launches == []
    assert not paths["marker"].exists()
    assert not paths["ranks"].exists()
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["mismatches"] == ["preflight.plan_file_sha"]


def test_run_cli_requires_expected_plan_file_sha256(preflight: ModuleType) -> None:
    with pytest.raises(SystemExit):
        preflight._parser().parse_args(["run", "--plan", "/tmp/plan.json"])


def test_new_postlaunch_compute_row_publishes_failed_terminal_evidence(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    observed = _install_fake_launches(preflight, monkeypatch, plan)

    def collect_new_row(
        *, plan: dict[str, Any], baseline: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        return preflight._build_postlaunch_gpu_sweep(
            plan=plan,
            baseline=baseline,
            phase=phase,
            samples=[
                _gpu_sample(
                    0,
                    compute_processes=[
                        {"gpu_uuid": "GPU-test-0", "pid": 111},
                        {"gpu_uuid": "GPU-test-2", "pid": 222},
                    ],
                ),
                _gpu_sample(
                    1,
                    compute_processes=[{"gpu_uuid": "GPU-test-0", "pid": 111}],
                ),
            ],
        )

    monkeypatch.setattr(preflight, "_collect_postlaunch_gpu_sweep", collect_new_row)

    with pytest.raises(preflight.PreflightError) as exc_info:
        preflight.run_plan(
            paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
        )

    assert exc_info.value.code == "controller.gpu_new_compute_process"
    assert observed == ["launch-a"]
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["mismatches"] == ["controller.gpu_new_compute_process"]
    sweeps = terminal["gpu_shared_occupancy_sweeps"]
    assert len(sweeps) == 1
    assert sweeps[0]["phase"] == "post-launch-a"
    assert sweeps[0]["admitted"] is False
    assert sweeps[0]["new_compute_processes"] == [
        {"gpu_uuid": "GPU-test-2", "pid": 222}
    ]


def test_outer_environment_conflict_fails_closed_before_attempt_marker(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _plan = _fixture(preflight, tmp_path)
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
    called: list[str] = []
    monkeypatch.setattr(
        preflight,
        "_run_accelerate_launch",
        lambda **kwargs: called.append(kwargs["launch_id"]),
    )

    with pytest.raises(preflight.PreflightError, match="environment"):
        preflight.run_plan(
            paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
        )

    assert called == []
    assert not paths["marker"].exists()
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["launch_count"] == 0
    assert terminal["rank_receipts"] == []
    assert terminal["mismatches"] == ["controller.outer_environment"]
    assert terminal["cleanup"]["all_process_groups_exited"] is None


def test_launch_a_digest_mismatch_stops_before_launch_b(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    observed = _install_fake_launches(
        preflight,
        monkeypatch,
        plan,
        digest_by_launch={"launch-a": "e" * 64},
    )

    def reject_first_launch(
        loaded_plan: dict[str, Any], launch_id: str
    ) -> list[dict[str, Any]]:
        preflight._load_rank_receipts_original(loaded_plan, launch_id)
        raise preflight.PreflightError("rank digest rejected", code="rank.digest")

    monkeypatch.setattr(
        preflight,
        "_load_rank_receipts",
        reject_first_launch,
    )

    with pytest.raises(preflight.PreflightError, match="rank digest"):
        preflight.run_plan(
            paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
        )

    assert observed == ["launch-a"]
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["launch_count"] == 1
    assert terminal["mismatches"] == ["rank.digest"]


def test_rank_inventory_is_exact_and_receipts_are_immutable(
    preflight: ModuleType, tmp_path: Path
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    rank_root = Path(plan["artifact_targets"]["rank_receipt_root"])
    for rank in range(7):
        target = rank_root / "launch-a" / f"rank-{rank:05d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            _canonical(
                _rank_payload(
                    preflight,
                    plan=plan,
                    launch_id="launch-a",
                    rank=rank,
                )
            )
            + b"\n"
        )

    with pytest.raises(preflight.PreflightError, match="inventory"):
        preflight._load_rank_receipts(plan, "launch-a")

    target = rank_root / "launch-a" / "rank-00007.json"
    target.write_bytes(
        _canonical(
            _rank_payload(
                preflight,
                plan=plan,
                launch_id="launch-a",
                rank=7,
            )
        )
        + b"\n"
    )
    extra = rank_root / "launch-a" / "rank-00008.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="inventory"):
        preflight._load_rank_receipts(plan, "launch-a")


def test_failed_terminal_and_attempt_targets_are_absent_only(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    _install_fake_launches(preflight, monkeypatch, plan, fail_launch="launch-a")

    with pytest.raises(preflight.PreflightError):
        preflight.run_plan(
            paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
        )
    first_terminal = paths["terminal"].read_bytes()

    with pytest.raises(preflight.PreflightError, match="already exists"):
        preflight.run_plan(
            paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
        )
    assert paths["terminal"].read_bytes() == first_terminal


def test_worker_binds_live_post_cuda_native_attestation(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    token = "controller-token"
    monkeypatch.setenv("COORDEXP_WAVE7_CONTROLLER_TOKEN", token)
    monkeypatch.setenv("COORDEXP_WAVE7_CONTROLLER_PID", "1234")
    calls: list[str] = []

    class FakeHooks:
        def cuda_is_initialized(self) -> bool:
            return "cuda:3" in calls

        def apply_seed_policy(self, seed: int) -> dict[str, Any]:
            calls.append("seed")
            assert seed == 17
            return preflight._expected_policy_receipt()

        def initialize_cuda(self, local_rank: int) -> None:
            calls.append(f"cuda:{local_rank}")

        def observe_device_identity(self, local_rank: int) -> dict[str, Any]:
            calls.append("device")
            return {
                "current_device": local_rank,
                "visible_index": local_rank,
                "gpu_uuid": "GPU-test-3",
                "pci_bus_id": "00000000:03:00.0",
            }

        def attest_native(self, provenance: dict[str, Any]) -> dict[str, Any]:
            calls.append("native")
            assert provenance["status"] == "passed"
            return {
                "schema_version": 1,
                "cuda_initialized": True,
                "admitted": True,
                "mismatches": [],
                "components": {"libcublas": {"sha256": "b" * 64}},
                "mapped_cudnn_components": [],
            }

        def run_workloads(self, rank: int, world_size: int) -> dict[str, Any]:
            calls.append("workloads")
            components = {
                "cublas": {"digest": "c" * 64},
                "flash_attention_2": {"digest": "f" * 64},
                "nccl_all_reduce": {
                    "digest": "e" * 64,
                    "expected_sum": 28.0,
                },
            }
            return {**components, "aggregate_sha256": _digest(components)}

        def attest_late_native(self, provenance: dict[str, Any]) -> dict[str, Any]:
            calls.append("late-native")
            components = {
                name: {
                    "path": identity["path"],
                    "size_bytes": identity["size_bytes"],
                    "file_sha256": identity["file_sha256"],
                    "soname": identity["soname"],
                    "origin_matches_admission": True,
                }
                for name, identity in provenance["late_native_expectations"].items()
            }
            return {
                "schema_version": 1,
                "cuda_initialized": True,
                "admitted": True,
                "mismatches": [],
                "components": components,
            }

        def cleanup(self) -> bool:
            calls.append("cleanup")
            return True

    marker = preflight.finalize_receipt(
        {
            "schema": preflight.ATTEMPT_MARKER_SCHEMA,
            "status": "started",
            "plan_sha256": plan["plan_payload_sha256"],
            "controller": {
                "pid": 1234,
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            },
            "environment": dict(plan["environment"]),
            "gpu_shared_preexisting_baseline": {
                "samples": [_gpu_sample(0), _gpu_sample(1)],
                "device_inventory": [
                    {"index": index, "gpu_uuid": f"GPU-test-{index}"}
                    for index in range(8)
                ],
                "preexisting_compute_processes": [],
            },
            "commands": [
                {
                    "launch_id": launch_id,
                    "argv": preflight._redacted_command_for_launch(
                        plan, launch_id=launch_id, controller_pid=1234
                    ),
                    "environment": dict(plan["environment"]),
                }
                for launch_id in ("launch-a", "launch-b")
            ],
            "artifact_targets": dict(plan["artifact_targets"]),
        }
    )
    paths["marker"].write_bytes(_canonical(marker) + b"\n")

    receipt = preflight.run_worker(
        paths["plan"],
        launch_id="launch-a",
        controller_pid=1234,
        controller_token=token,
        hooks=FakeHooks(),
    )

    assert calls == [
        "seed",
        "cuda:3",
        "device",
        "native",
        "workloads",
        "native",
        "late-native",
        "cleanup",
    ]
    assert receipt["mapped_native_execution"]["admitted"] is True
    assert receipt["mapped_native_execution"][
        "pre_workload_attestation_sha256"
    ] == _digest(
        {
            "schema_version": 1,
            "cuda_initialized": True,
            "admitted": True,
            "mismatches": [],
            "components": {"libcublas": {"sha256": "b" * 64}},
            "mapped_cudnn_components": [],
        }
    )
    assert (
        Path(plan["artifact_targets"]["rank_receipt_root"]) / "launch-a/rank-00003.json"
    ).exists()


def test_controller_constructs_exact_child_environment_for_subprocess(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = tmp_path / "child.json"
    executable = tmp_path / "fake-accelerate"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
if sys.argv[1:] == ["--version"]:
    print("Accelerate child-test 1.0")
    raise SystemExit(0)
payload = {
    "argv": sys.argv[1:],
    "environment": {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "FLASH_ATTENTION_DETERMINISTIC": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"),
    },
    "controller_pid": os.environ.get("COORDEXP_WAVE7_CONTROLLER_PID"),
    "controller_token": os.environ.get("COORDEXP_WAVE7_CONTROLLER_TOKEN"),
}
Path(%r).write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps(payload))
print(payload["controller_token"], file=sys.stderr)
"""
        % str(capture),
        encoding="utf-8",
    )
    executable.chmod(0o700)
    paths, plan = _fixture(preflight, tmp_path)
    plan_path = paths["plan"]
    plan_path.unlink()
    plan = preflight.prepare_plan(
        plan_path=plan_path,
        attempt_marker_path=paths["marker"],
        terminal_receipt_path=paths["terminal"],
        rank_receipt_root=paths["ranks"],
        runtime_receipt_path=paths["runtime"],
        expected_runtime_file_sha256=hashlib.sha256(
            paths["runtime"].read_bytes()
        ).hexdigest(),
        native_runtime_receipt_path=paths["native"],
        expected_native_runtime_file_sha256=hashlib.sha256(
            paths["native"].read_bytes()
        ).hexdigest(),
        accelerate_executable=str(executable),
    )
    for name in plan["environment"]:
        monkeypatch.delenv(name, raising=False)

    result = preflight._run_accelerate_launch(
        plan=plan,
        launch_id="launch-a",
        controller_pid=4321,
        controller_token="fresh-token",
    )

    child = json.loads(capture.read_text(encoding="utf-8"))
    assert result["returncode"] == 0
    assert child["environment"] == plan["environment"]
    assert child["controller_pid"] == "4321"
    assert child["controller_token"] == "fresh-token"
    assert "fresh-token" not in child["argv"]
    assert child["argv"][:5] == [
        "launch",
        "--multi_gpu",
        "--num_processes",
        "8",
        "--main_process_port",
    ]
    assert child["argv"][5] == str(plan["launch_contract"][0]["main_process_port"])
    assert "fresh-token" not in result["stdout_tail"]
    assert "fresh-token" not in result["stderr_tail"]
    assert "<redacted-controller-token>" in result["stdout_tail"]
    assert "<redacted-controller-token>" in result["stderr_tail"]
    assert result["process_scope"]["cleanup_verified"] is True


def test_process_group_cleanup_is_bounded(
    preflight: ModuleType,
) -> None:
    import subprocess

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    termination = preflight._terminate_process_group(
        process, terminate_grace=2.0, kill_grace=2.0
    )

    assert termination in {"terminated", "killed"}
    assert process.poll() is not None


@pytest.mark.parametrize("partial_evidence", ["valid", "malformed", "changing"])
def test_keyboard_interrupt_after_spawn_reaps_scope_and_contains_partial_evidence(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    partial_evidence: str,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    shared = [{"gpu_uuid": "GPU-test-0", "pid": 111}]
    baseline = {
        "samples": [_gpu_sample(index, compute_processes=shared) for index in range(2)],
        "device_inventory": [
            {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
        ],
        "preexisting_compute_processes": shared,
    }
    monkeypatch.setattr(
        preflight,
        "_collect_stable_gpu_idle_samples",
        lambda _plan: baseline,
    )
    rank_root = Path(plan["artifact_targets"]["rank_receipt_root"])
    first_rank_path = rank_root / "launch-a/rank-00000.json"
    second_rank_path = rank_root / "launch-a/rank-00001.json"
    first_rank = (
        b"{not-json\n"
        if partial_evidence == "malformed"
        else _canonical(
            _rank_payload(preflight, plan=plan, launch_id="launch-a", rank=0)
        )
        + b"\n"
    )
    second_rank = (
        _canonical(_rank_payload(preflight, plan=plan, launch_id="launch-a", rank=1))
        + b"\n"
    )
    ready_path = tmp_path / "child-ready"
    launch_port = plan["launch_contract"][0]["main_process_port"]
    child_source = (
        "import socket, time\n"
        "from pathlib import Path\n"
        f"first = Path({str(first_rank_path)!r})\n"
        f"second = Path({str(second_rank_path)!r})\n"
        f"ready = Path({str(ready_path)!r})\n"
        "first.parent.mkdir(parents=True, exist_ok=True)\n"
        f"listener = socket.socket(); listener.bind(('127.0.0.1', {launch_port})); "
        "listener.listen(1)\n"
        f"first.write_bytes({first_rank!r})\n"
        "ready.write_text('ready', encoding='utf-8')\n"
        "time.sleep(0.4)\n"
        f"second.write_bytes({second_rank!r})\n"
        "time.sleep(60)\n"
    )
    real_popen = preflight.subprocess.Popen
    children: list[subprocess.Popen[str]] = []

    def start_cpu_child(_argv: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        child = real_popen([sys.executable, "-c", child_source], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(preflight.subprocess, "Popen", start_cpu_child)
    if partial_evidence == "changing":
        stable_inventory = preflight._partial_rank_file_inventory

        def changing_inventory(
            loaded_plan: dict[str, Any],
        ) -> tuple[tuple[str, int, int, str], ...]:
            inventory = stable_inventory(loaded_plan)
            encoded = first_rank_path.read_bytes()
            if encoded.endswith(b" \n"):
                first_rank_path.write_bytes(encoded[:-2] + b"\n")
            else:
                first_rank_path.write_bytes(encoded[:-1] + b" \n")
            return inventory

        monkeypatch.setattr(
            preflight,
            "_partial_rank_file_inventory",
            changing_inventory,
        )
    real_sleep = time.sleep
    interrupted = False

    def interrupt_after_first_receipt(seconds: float) -> None:
        nonlocal interrupted
        if not interrupted:
            deadline = time.monotonic() + 5.0
            while not ready_path.exists() and time.monotonic() < deadline:
                real_sleep(0.01)
            assert ready_path.exists()
            interrupted = True
            raise KeyboardInterrupt
        real_sleep(min(seconds, 0.01))

    monkeypatch.setattr(preflight.time, "sleep", interrupt_after_first_receipt)
    terminal_bytes: bytes | None = None
    try:
        with pytest.raises(preflight.PreflightError) as exc_info:
            preflight.run_plan(
                paths["plan"],
                expected_plan_file_sha256=_file_sha256(paths["plan"]),
            )
        assert exc_info.value.code == "KeyboardInterrupt"
        terminal_bytes = paths["terminal"].read_bytes()
        terminal = json.loads(terminal_bytes)
        expected_mismatches = ["KeyboardInterrupt"]
        if partial_evidence != "valid":
            expected_mismatches.append("rank.partial_evidence_binding")
        assert terminal["mismatches"] == expected_mismatches
        assert terminal["launch_count"] == 1
        assert terminal["cleanup"] == {
            "all_process_groups_exited": True,
            "bounded": True,
        }
        process = terminal["processes"][0]
        assert process["cleanup_verified"] is True
        assert process["process_scope"]["term_sent"] is True
        assert process["process_scope"]["remaining_members"] == []
        assert process["process_scope"]["leader_reaped"] is True
        if partial_evidence == "valid":
            assert terminal["rank_receipts"] == [
                {
                    "path": str(first_rank_path),
                    "file_sha256": _file_sha256(first_rank_path),
                    "payload_sha256": json.loads(first_rank)["receipt_payload_sha256"],
                    "schema": preflight.RANK_RECEIPT_SCHEMA,
                    "status": "passed",
                }
            ]
        else:
            assert terminal["rank_receipts"] == []
        assert preflight._session_members(process["pid"]) == []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", launch_port))
        real_sleep(0.5)
        assert not second_rank_path.exists()
        assert paths["terminal"].read_bytes() == terminal_bytes
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, 9)
            child.wait(timeout=2.0)


def test_cleanup_reap_timeout_is_retained_as_failed_terminal_evidence(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    real_wait = process.wait

    def reap_timeout(timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired(process.args, timeout)

    monkeypatch.setattr(process, "wait", reap_timeout)
    try:
        termination, scope = preflight._cleanup_process_scope(
            process, terminate_grace=2.0, kill_grace=2.0
        )
    finally:
        monkeypatch.setattr(process, "wait", real_wait)
        real_wait(timeout=2.0)

    assert termination in {"terminated", "killed"}
    assert scope["cleanup_verified"] is False
    assert scope["leader_reaped"] is False
    assert scope["cleanup_failure"] == "leader_reap_timeout"
    assert scope["observed_members"]
    assert {member["process_group_id"] for member in scope["observed_members"]} == {
        process.pid
    }

    _paths, plan = _fixture(preflight, tmp_path)
    record = {
        "launch_id": "launch-a",
        "pid": process.pid,
        "returncode": process.returncode,
        "termination": termination,
        "process_scope": scope,
        "cleanup_verified": False,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    terminal = preflight.finalize_receipt(
        preflight._terminal_payload(
            plan=plan,
            status="failed",
            mismatches=["controller.cleanup_failed"],
            marker=None,
            processes=[record],
            receipts=[],
            comparisons=None,
            gpu_shared_occupancy_sweeps=[],
        )
    )

    validated = preflight._validate_terminal_receipt(terminal, plan=plan)

    assert validated["cleanup"]["all_process_groups_exited"] is False
    assert validated["processes"][0]["process_scope"]["leader"]["pid"] == process.pid


def test_cleanup_survivor_identity_and_process_group_are_retained(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    identity = preflight._proc_identity(process.pid)
    assert identity is not None
    monkeypatch.setattr(
        preflight,
        "_wait_session_empty",
        lambda _session_id, _timeout: [dict(identity)],
    )
    try:
        termination, scope = preflight._cleanup_process_scope(
            process, terminate_grace=0.01, kill_grace=2.0
        )
    finally:
        process.wait(timeout=2.0)

    assert termination == "killed"
    assert scope["cleanup_verified"] is False
    assert scope["cleanup_failure"] == "surviving_members_after_sigkill"
    assert scope["remaining_members"] == [identity]
    assert scope["remaining_members"][0]["process_group_id"] == process.pid


def test_rank_workload_component_mutation_is_rejected(
    preflight: ModuleType, tmp_path: Path
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    rank_root = Path(plan["artifact_targets"]["rank_receipt_root"])
    for rank in range(8):
        receipt = _rank_payload(
            preflight,
            plan=plan,
            launch_id="launch-a",
            rank=rank,
        )
        if rank == 4:
            payload = dict(receipt)
            payload.pop("receipt_payload_sha256")
            payload["workloads"] = dict(payload["workloads"])
            payload["workloads"]["cublas"] = {"digest": "e" * 64}
            receipt = preflight.finalize_receipt(payload)
        target = rank_root / "launch-a" / f"rank-{rank:05d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_canonical(receipt) + b"\n")

    with pytest.raises(preflight.PreflightError, match="workload"):
        preflight._load_rank_receipts(plan, "launch-a")


def test_verify_reloads_full_live_evidence_and_rejects_deleted_rank(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    _install_fake_launches(preflight, monkeypatch, plan)
    terminal = preflight.run_plan(
        paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
    )
    marker_text = paths["marker"].read_text(encoding="utf-8")
    assert "--controller-token" not in marker_text
    assert "COORDEXP_WAVE7_CONTROLLER_TOKEN" not in marker_text

    verified = preflight.verify_receipt(
        paths["terminal"],
        expected_file_sha256=hashlib.sha256(paths["terminal"].read_bytes()).hexdigest(),
        plan_path=paths["plan"],
        expected_plan_file_sha256=hashlib.sha256(
            paths["plan"].read_bytes()
        ).hexdigest(),
    )
    assert verified == terminal

    (paths["ranks"] / "launch-b/rank-00007.json").unlink()
    with pytest.raises(preflight.PreflightError, match="inventory"):
        preflight.verify_receipt(
            paths["terminal"],
            expected_file_sha256=hashlib.sha256(
                paths["terminal"].read_bytes()
            ).hexdigest(),
            plan_path=paths["plan"],
            expected_plan_file_sha256=hashlib.sha256(
                paths["plan"].read_bytes()
            ).hexdigest(),
        )


def test_verify_rejects_minimal_or_failed_terminal(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    _install_fake_launches(preflight, monkeypatch, plan, fail_launch="launch-a")
    with pytest.raises(preflight.PreflightError):
        preflight.run_plan(
            paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
        )
    with pytest.raises(preflight.PreflightError, match="not passed"):
        preflight.verify_receipt(
            paths["terminal"],
            expected_file_sha256=hashlib.sha256(
                paths["terminal"].read_bytes()
            ).hexdigest(),
            plan_path=paths["plan"],
            expected_plan_file_sha256=hashlib.sha256(
                paths["plan"].read_bytes()
            ).hexdigest(),
        )


def test_accelerate_path_is_frozen_and_mutation_is_rejected(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    _write_fake_accelerate(alternate / "accelerate")
    monkeypatch.setenv("PATH", f"{alternate}:{os.environ['PATH']}")

    loaded = preflight._validate_plan(
        preflight._load_json(paths["plan"]), plan_path=paths["plan"].resolve()
    )
    assert loaded["accelerate_identity"]["path"] == plan["accelerate_identity"]["path"]

    installed_distribution = preflight.distribution
    monkeypatch.setattr(
        preflight,
        "distribution",
        lambda name: SimpleNamespace(version="999.0-drift")
        if name == "accelerate"
        else installed_distribution(name),
    )
    with pytest.raises(preflight.PreflightError, match="identity drifted"):
        preflight._validate_plan(
            preflight._load_json(paths["plan"]), plan_path=paths["plan"].resolve()
        )
    monkeypatch.setattr(preflight, "distribution", installed_distribution)

    frozen = Path(plan["accelerate_identity"]["path"])
    frozen.write_text(
        frozen.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8"
    )
    with pytest.raises(preflight.PreflightError, match="identity drifted"):
        preflight._validate_plan(
            preflight._load_json(paths["plan"]), plan_path=paths["plan"].resolve()
        )


def test_declared_vs_observed_device_is_rejected(
    preflight: ModuleType, tmp_path: Path
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    receipt = _rank_payload(preflight, plan=plan, launch_id="launch-a", rank=2)
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256")
    payload["device_identity"] = dict(payload["device_identity"])
    payload["device_identity"]["current_device"] = 3
    mutated = preflight.finalize_receipt(payload)
    with pytest.raises(preflight.PreflightError, match="observed CUDA device"):
        preflight._validate_rank_receipt(
            mutated, plan=plan, launch_id="launch-a", rank=2
        )


def test_late_native_identity_is_bound_to_runtime_admission(
    preflight: ModuleType, tmp_path: Path
) -> None:
    _paths, plan = _fixture(preflight, tmp_path)
    receipt = _rank_payload(preflight, plan=plan, launch_id="launch-a", rank=1)
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256")
    payload["mapped_native_execution"] = dict(payload["mapped_native_execution"])
    late = dict(payload["mapped_native_execution"]["late_native_mappings"])
    late["components"] = {
        name: dict(identity) for name, identity in late["components"].items()
    }
    late["components"]["libnccl"]["file_sha256"] = "0" * 64
    payload["mapped_native_execution"]["late_native_mappings"] = late
    mutated = preflight.finalize_receipt(payload)
    with pytest.raises(preflight.PreflightError, match="component drifted"):
        preflight._validate_rank_receipt(
            mutated, plan=plan, launch_id="launch-a", rank=1
        )


def test_orphan_descendant_is_captured_and_reaped(
    preflight: ModuleType, tmp_path: Path
) -> None:
    paths, _old_plan = _fixture(preflight, tmp_path)
    paths["plan"].unlink()
    executable = tmp_path / "orphan-accelerate"
    _write_fake_accelerate(
        executable,
        body=(
            "import os, time\n"
            "child = os.fork()\n"
            "if child == 0:\n"
            "    sink = os.open('/dev/null', os.O_WRONLY)\n"
            "    os.dup2(sink, 1); os.dup2(sink, 2)\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
            "raise SystemExit(0)\n"
        ),
    )
    plan = preflight.prepare_plan(
        plan_path=paths["plan"],
        attempt_marker_path=paths["marker"],
        terminal_receipt_path=paths["terminal"],
        rank_receipt_root=paths["ranks"],
        runtime_receipt_path=paths["runtime"],
        expected_runtime_file_sha256=hashlib.sha256(
            paths["runtime"].read_bytes()
        ).hexdigest(),
        native_runtime_receipt_path=paths["native"],
        expected_native_runtime_file_sha256=hashlib.sha256(
            paths["native"].read_bytes()
        ).hexdigest(),
        accelerate_executable=str(executable),
    )

    result = preflight._run_accelerate_launch(
        plan=plan,
        launch_id="launch-a",
        controller_pid=os.getpid(),
        controller_token="orphan-test-token",
    )

    assert result["returncode"] == 0
    assert result["process_scope"]["cleanup_verified"] is True
    assert len(result["process_scope"]["observed_members"]) >= 2
    assert result["process_scope"]["term_sent"] is True


def test_port_collision_fails_before_attempt_marker(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, plan = _fixture(preflight, tmp_path)
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    handle.bind(("127.0.0.1", plan["launch_contract"][0]["main_process_port"]))
    handle.listen(1)
    try:
        with pytest.raises(preflight.PreflightError, match="port"):
            preflight.run_plan(
                paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
            )
    finally:
        handle.close()
    assert not paths["marker"].exists()
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    assert terminal["mismatches"] == ["controller.port_unavailable"]


def test_native_reference_requires_canonical_path_and_immutable_hashes(
    preflight: ModuleType, tmp_path: Path
) -> None:
    canonical, _binding = _canonical_native_reference(preflight)
    receipt, observed = preflight._validate_native_reference_receipt(
        canonical,
        expected_file_sha256=preflight._CANONICAL_NATIVE_REFERENCE_FILE_SHA256,
    )
    assert receipt["receipt_sha256"] == (
        preflight._CANONICAL_NATIVE_REFERENCE_PAYLOAD_SHA256
    )
    assert observed["path"] == str(canonical)

    copied = tmp_path / "copied-native-reference.json"
    copied.write_bytes(canonical.read_bytes())
    with pytest.raises(preflight.PreflightError, match="does not match"):
        preflight._validate_native_reference_receipt(
            copied,
            expected_file_sha256=preflight._CANONICAL_NATIVE_REFERENCE_FILE_SHA256,
        )

    synthetic = tmp_path / "synthetic-native-reference.json"
    _write_native_reference(preflight, synthetic)
    with pytest.raises(preflight.PreflightError, match="does not match"):
        preflight._validate_native_reference_receipt(
            synthetic,
            expected_file_sha256=hashlib.sha256(synthetic.read_bytes()).hexdigest(),
        )


def test_controller_token_is_private_and_scrubbed_from_persisted_artifacts(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _old_plan = _fixture(preflight, tmp_path)
    paths["plan"].unlink()
    capture = tmp_path / "private-channel-capture.json"
    executable = tmp_path / "echo-private-channel"
    _write_fake_accelerate(
        executable,
        body=(
            "import json, os\n"
            "from pathlib import Path\n"
            f"capture = Path({str(capture)!r})\n"
            "payload = {'argv': sys.argv[1:], "
            "'token': os.environ.get('COORDEXP_WAVE7_CONTROLLER_TOKEN')}\n"
            "capture.write_text(json.dumps(payload), encoding='utf-8')\n"
            "print(json.dumps(payload))\n"
            "print(payload['token'], file=sys.stderr)\n"
        ),
    )
    plan = preflight.prepare_plan(
        plan_path=paths["plan"],
        attempt_marker_path=paths["marker"],
        terminal_receipt_path=paths["terminal"],
        rank_receipt_root=paths["ranks"],
        runtime_receipt_path=paths["runtime"],
        expected_runtime_file_sha256=hashlib.sha256(
            paths["runtime"].read_bytes()
        ).hexdigest(),
        native_runtime_receipt_path=paths["native"],
        expected_native_runtime_file_sha256=(
            preflight._CANONICAL_NATIVE_REFERENCE_FILE_SHA256
        ),
        accelerate_executable=str(executable),
    )
    for name, value in plan["environment"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        preflight,
        "_collect_stable_gpu_idle_samples",
        lambda _plan: {
            "samples": [_gpu_sample(0), _gpu_sample(1)],
            "device_inventory": [
                {"index": index, "gpu_uuid": f"GPU-test-{index}"} for index in range(8)
            ],
            "preexisting_compute_processes": [],
        },
    )
    monkeypatch.setattr(
        preflight,
        "_collect_postlaunch_gpu_sweep",
        lambda *, plan, baseline, phase: preflight._build_postlaunch_gpu_sweep(
            plan=plan,
            baseline=baseline,
            phase=phase,
            samples=[_gpu_sample(0), _gpu_sample(1)],
        ),
    )

    with pytest.raises(preflight.PreflightError, match="inventory"):
        preflight.run_plan(
            paths["plan"], expected_plan_file_sha256=_file_sha256(paths["plan"])
        )

    private = json.loads(capture.read_text(encoding="utf-8"))
    token = private["token"]
    assert isinstance(token, str) and token
    assert token not in private["argv"]
    assert "--controller-token" not in private["argv"]
    marker_text = paths["marker"].read_text(encoding="utf-8")
    terminal_text = paths["terminal"].read_text(encoding="utf-8")
    assert token not in marker_text
    assert token not in terminal_text
    terminal = json.loads(terminal_text)
    process = terminal["processes"][0]
    assert token not in process["stdout_tail"]
    assert token not in process["stderr_tail"]
    assert "<redacted-controller-token>" in process["stdout_tail"]
    assert "<redacted-controller-token>" in process["stderr_tail"]


def _exercise_direct_script_cli(
    tmp_path: Path,
) -> None:
    native_reference = (
        REPO_ROOT
        / "outputs/probes/coordexp_swift/wave8_native_runtime/2026-08-10-r2/receipt.json"
    )
    native_file_sha256 = (
        "d88c9c4ded7c698786c54721f8fbbffdad1467e467f845f72dbdb8cfe9eb75ac"
    )
    runtime_receipt = tmp_path / "runtime-admission.json"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    admitted = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "admit-runtime",
            "--receipt",
            str(runtime_receipt),
            "--native-reference",
            str(native_reference),
            "--expected-native-reference-file-sha256",
            native_file_sha256,
        ],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    assert admitted.returncode == 0, admitted.stderr
    assert runtime_receipt.exists()

    accelerate = Path(sys.executable).resolve().with_name("accelerate")
    assert accelerate.is_file()
    plan_path = tmp_path / "plan.json"
    marker_path = tmp_path / "marker.json"
    terminal_path = tmp_path / "terminal.json"
    rank_root = tmp_path / "ranks"
    prepared = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            "--plan",
            str(plan_path),
            "--attempt-marker",
            str(marker_path),
            "--terminal-receipt",
            str(terminal_path),
            "--rank-receipt-root",
            str(rank_root),
            "--runtime-receipt",
            str(runtime_receipt),
            "--expected-runtime-file-sha256",
            hashlib.sha256(runtime_receipt.read_bytes()).hexdigest(),
            "--native-runtime-receipt",
            str(native_reference),
            "--expected-native-runtime-file-sha256",
            native_file_sha256,
            "--accelerate-executable",
            str(accelerate),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert prepared.returncode == 0, prepared.stderr
    plan = json.loads(prepared.stdout)
    assert plan["accelerate_identity"]["path"] == str(accelerate)
    assert plan["accelerate_identity"]["distribution_name"] == "accelerate"
    assert plan["accelerate_identity"]["distribution_version"] == "1.10.1"
    for launch in plan["launch_contract"]:
        assert launch["argv_template"][7:9] == [str(SCRIPT), "_worker"]
        assert "--controller-token" not in launch["argv_template"]

    worker_environment = environment.copy()
    worker_environment.update(plan["environment"])
    worker_environment["COORDEXP_WAVE7_CONTROLLER_PID"] = "4321"
    worker_environment["COORDEXP_WAVE7_CONTROLLER_TOKEN"] = "private-test-token"
    worker = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "_worker",
            "--plan",
            str(plan_path),
            "--launch-id",
            "launch-a",
            "--controller-pid",
            "4321",
        ],
        cwd=REPO_ROOT,
        env=worker_environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert worker.returncode == 2
    assert "preflight.json_read" in worker.stderr
    assert "ModuleNotFoundError" not in worker.stderr

    terminal_path.write_text("{}\n", encoding="utf-8")
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--plan",
            str(plan_path),
            "--expected-plan-file-sha256",
            _file_sha256(plan_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 2
    assert "preflight.artifact_collision" in run.stderr
    assert "ModuleNotFoundError" not in run.stderr
