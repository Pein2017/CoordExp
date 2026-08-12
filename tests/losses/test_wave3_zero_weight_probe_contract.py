from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from src.qwen import parity as parity_module


def _load_probe_module():
    name = "coordexp_wave3_zero_weight_probe_test_module"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/probes/coordexp_swift/wave3_zero_weight_gpu.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reset_accelerate_shared_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from accelerate.state import AcceleratorState, PartialState

    monkeypatch.setattr(PartialState, "_shared_state", {})
    monkeypatch.setattr(AcceleratorState, "_shared_state", {})


def _fake_exact_accelerator(*, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        distributed_type=SimpleNamespace(name="NO"),
        process_index=0,
        local_process_index=0,
        num_processes=1,
        device=torch.device(device),
        mixed_precision="bf16",
        native_amp=True,
        gradient_accumulation_steps=1,
        scaler=None,
    )


def _refinalize(
    probe, payload: dict[str, object], hash_field: str
) -> dict[str, object]:
    value = deepcopy(payload)
    value.pop(hash_field, None)
    return probe._finalize(value, hash_field=hash_field)


def _fake_model_weight_identity() -> dict[str, object]:
    receipt = (
        Path(__file__).resolve().parents[2]
        / "openspec/changes/harden-optimize-coordexp-swift-training-infrastructure/"
        "receipts/wave2-v3-plan.json"
    )
    return deepcopy(
        json.loads(receipt.read_text(encoding="utf-8"))["model_weight_identity"]
    )


def _write_tiny_indexed_model(root: Path) -> dict[str, object]:
    root.mkdir()
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 23},
                "weight_map": {
                    "layer.a": "model-00001-of-00002.safetensors",
                    "layer.b": "model-00002-of-00002.safetensors",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return parity_module.base_model_weight_identity(root)


def _valid_plan(probe, tmp_path: Path) -> dict[str, object]:
    targets = {
        "plan": str((tmp_path / "plan.json").resolve()),
        "attempt_marker": str((tmp_path / "attempt.json").resolve()),
        "receipt": str((tmp_path / "receipt.json").resolve()),
        "publication_failure": str((tmp_path / "publication-failure.json").resolve()),
    }
    prepare_argv = probe._prepare_argv(
        config_path=probe.FROZEN_CONFIG_PATH,
        cache_dir=probe.W0_CACHE_DIR,
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device="cuda:0",
    )
    run_argv = probe._artifact_argv(
        command="run",
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device="cuda:0",
    )
    controller_argv = probe._artifact_argv(
        command="controller",
        plan_path=targets["plan"],
        receipt_path=targets["receipt"],
        attempt_marker_path=targets["attempt_marker"],
        publication_failure_path=targets["publication_failure"],
        device="cuda:0",
    )
    return probe._finalize(
        {
            "schema": probe.PLAN_SCHEMA,
            "status": "prepared",
            "model_weight_identity": _fake_model_weight_identity(),
            "config": {
                "entry_path": str(probe.FROZEN_CONFIG_PATH.resolve()),
                "fingerprint": probe.FROZEN_CONFIG_FINGERPRINT,
                "resolved_sha256": probe.FROZEN_CONFIG_FINGERPRINT,
                "forward_input_provider_mode": (probe.FORWARD_INPUT_PROVIDER_VALUE),
                "compatibility_projection": (
                    probe.frozen_config_compatibility_projection()
                ),
                "runtime_config_attestation": (
                    probe.frozen_runtime_config_attestation()
                ),
                "model_identity": _fake_model_source_artifact(load_model=False),
            },
            "cache": {
                "path": str(probe.W0_CACHE_DIR.resolve()),
                "access": "read_only_private_immutable",
                "version": probe.W0_CACHE_VERSION,
                "fingerprint": probe.W0_CACHE_FINGERPRINT,
                "manifest_sha256": probe.W0_MANIFEST_SHA256,
                "chunk_sha256": "2" * 64,
                "shared_cache_writes": False,
            },
            "workload": {
                "selection": dict(probe.W0_FIRST_MEASURED_SELECTION),
                "pack_index": probe.W0_FIRST_MEASURED_PACK_ORDINAL,
                "example_ids": ["example"],
                "segment_bounds": [[0, 1]],
                "segment_boundaries": [0, 1],
                "pack_length": 1,
                "input_sha256": "3" * 64,
                "supervision_sha256": "4" * 64,
            },
            "loss_contract": {
                "zero_weight": 0.0,
                "nonzero_control_weight": 0.25,
                "raw_diagnostic_tolerance": {
                    "rtol": probe.LOSS_RTOL,
                    "atol": probe.LOSS_ATOL,
                },
                "bf16_derived_tolerance": {
                    "rtol": probe.BF16_RTOL,
                    "atol": probe.BF16_ATOL,
                },
                "raw_oracle": "one_frozen_fp32_logits_tensor_two_diagnostic_paths",
                "zero_total_reference": (
                    "base_and_nonzero_terms_plus_graph_connected_raw_times_zero"
                ),
                "optimized_zero_total": "base_and_nonzero_terms_only",
                "expected_trainable_gradient_count": probe.EXPECTED_TRAINABLE_COUNT,
            },
            "execution_contract": {
                "arm_order": list(probe.ARM_ORDER),
                "oracle_arm_precedes_measured_arms": True,
                "model_forward_count_per_arm": 1,
                "backward_count_per_measured_arm": 1,
                "state_restore_between_arms": (
                    "trainable_values_named_buffers_train_mode_and_rng_exact"
                ),
                "wall_clock": (
                    "external_monotonic_watchdog_and_cuda_synchronized_perf_counter"
                ),
                "arm_wall_ceiling_seconds": probe.ARM_WALL_CEILING_SECONDS,
                "run_wall_ceiling_seconds": probe.RUN_WALL_CEILING_SECONDS,
                "host_rss_ceiling_bytes": probe.HOST_RSS_CEILING_BYTES,
                "device_memory_ceiling_bytes": probe.DEVICE_MEMORY_CEILING_BYTES,
                "controller_poll_seconds": probe.CONTROLLER_POLL_SECONDS,
                "process_cleanup_timeout_seconds": (
                    probe.PROCESS_CLEANUP_TIMEOUT_SECONDS
                ),
                "max_tracked_processes": probe.MAX_TRACKED_PROCESSES,
                "terminal_order": (
                    "process_tree_cleanup_then_two_sample_gpu_subset_sweep"
                ),
                "shared_gpu_total_memory_mib": probe.SHARED_GPU_TOTAL_MEMORY_MIB,
                "shared_gpu_max_preexisting_memory_mib": (
                    probe.SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB
                ),
                "shared_gpu_required_headroom_mib": (
                    probe.SHARED_GPU_REQUIRED_HEADROOM_MIB
                ),
                "shared_gpu_preflight_sample_count": probe.SHARED_GPU_SAMPLE_COUNT,
                "shared_gpu_minimum_sample_interval_seconds": (
                    probe.SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS
                ),
                "shared_gpu_utilization_disposition": (
                    "observational_only_not_admission_or_promotion"
                ),
                "shared_gpu_baseline_process_policy": (
                    "stable_exact_gpu_uuid_driver_pid_then_post_probe_subset"
                ),
                "baseline_process_action": "observe_only_never_signal_or_reset",
                "claim_boundary": probe.shared_gpu_claim_boundary(),
                "requested_device": "cuda:0",
                "retry": False,
                "sample_switch": False,
                "tolerance_change": False,
                "prepare_argv": prepare_argv,
                "prepare_argv_sha256": probe.sha256_json(prepare_argv),
                "run_argv": run_argv,
                "run_argv_sha256": probe.sha256_json(run_argv),
                "controller_argv": controller_argv,
                "controller_argv_sha256": probe.sha256_json(controller_argv),
            },
            "artifact_targets": targets,
            "source_identity": {name: "5" * 64 for name in probe.SOURCE_OWNERS},
            "provenance": {
                "schema_version": 1,
                "repository": {},
                "dependencies": {},
                "runtime": {},
            },
        },
        hash_field="plan_sha256",
    )


def _valid_passed_receipt(probe, plan: dict[str, object]) -> dict[str, object]:
    inventory = _fake_inventory(probe)
    buffers = _fake_buffers(probe)
    transition = {
        "before": buffers,
        "after": buffers,
        "restored": buffers,
        "after_matches_before": True,
        "restoration_matches_initial": True,
    }
    runtime_sources = _fake_runtime_sources(probe)
    worker_argv = probe._artifact_argv(
        command="_worker",
        plan_path=plan["artifact_targets"]["plan"],
        receipt_path=plan["artifact_targets"]["receipt"],
        attempt_marker_path=plan["artifact_targets"]["attempt_marker"],
        publication_failure_path=plan["artifact_targets"]["publication_failure"],
        device="cuda:0",
    )
    idle = {
        "schema": probe.SHARED_GPU_BASELINE_SCHEMA,
        "mode": "shared_preexisting_compute",
        "requested_device": "cuda:0",
        "cuda_visible_devices": None,
        "selector": "0",
        "gpu_identity": {
            "physical_index": 0,
            "uuid": "GPU-fake",
            "total_memory_mib": probe.SHARED_GPU_TOTAL_MEMORY_MIB,
        },
        "limits": {
            "sample_count": probe.SHARED_GPU_SAMPLE_COUNT,
            "minimum_sample_interval_seconds": (
                probe.SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS
            ),
            "max_preexisting_memory_mib": (probe.SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB),
            "required_headroom_mib": probe.SHARED_GPU_REQUIRED_HEADROOM_MIB,
        },
        "utilization_disposition": ("observational_only_not_admission_or_promotion"),
        "baseline_compute_processes": [],
        "samples": [
            {
                "sample_index": index,
                "monotonic_ns": 1 + index * 2_000_000_000,
                "memory_used_mib": 0,
                "headroom_mib": probe.SHARED_GPU_TOTAL_MEMORY_MIB,
                "utilization_percent": 0,
                "compute_processes": [],
            }
            for index in range(probe.SHARED_GPU_SAMPLE_COUNT)
        ],
    }
    pre_marker_observation = {
        "schema": probe.SHARED_GPU_OBSERVATION_SCHEMA,
        "stage": "pre_marker",
        "monotonic_ns": 4_000_000_002,
        "gpu_identity": idle["gpu_identity"],
        "memory_used_mib": 0,
        "headroom_mib": probe.SHARED_GPU_TOTAL_MEMORY_MIB,
        "utilization_percent": 0,
        "compute_processes": [],
        "missing_baseline_processes": [],
        "new_processes": [],
    }
    marker = probe._finalize(
        {
            "schema": probe.MARKER_SCHEMA,
            "status": "attempt_started",
            "plan_sha256": plan["plan_sha256"],
            "model_weight_identity": plan["model_weight_identity"],
            "receipt_target": plan["artifact_targets"]["receipt"],
            "publication_failure_target": plan["artifact_targets"][
                "publication_failure"
            ],
            "requested_device": "cuda:0",
            "command_identity": {
                "run_argv": plan["execution_contract"]["run_argv"],
                "run_argv_sha256": plan["execution_contract"]["run_argv_sha256"],
                "controller_argv": plan["execution_contract"]["controller_argv"],
                "controller_argv_sha256": plan["execution_contract"][
                    "controller_argv_sha256"
                ],
                "worker_argv": worker_argv,
                "worker_argv_sha256": probe.sha256_json(worker_argv),
            },
            "publication_owner_token_sha256": "6" * 64,
            "source_identity": plan["source_identity"],
            "runtime_config_attestation": (probe.frozen_runtime_config_attestation()),
            "runtime_source_identities": runtime_sources,
            "cpu_installation": {
                owner: {
                    "parameter_count": 1,
                    "buffer_count": 1,
                    "all_cpu": True,
                }
                for owner in ("model", "adapter", "embedding_delta")
            },
            "provenance_sha256": probe.sha256_json(plan["provenance"]),
            "shared_gpu_preflight": idle,
            "pre_marker_gpu_observation": pre_marker_observation,
            "claim_boundary": probe.shared_gpu_claim_boundary(),
            "concrete_trainable_inventory": inventory,
            "initial_buffer_inventory": buffers,
            "published_monotonic_ns": 1,
        },
        hash_field="marker_sha256",
    )
    marker_path = Path(plan["artifact_targets"]["attempt_marker"])
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    gradients = _fake_gradients(probe, inventory)
    resources = _fake_resources()
    arm_artifacts: dict[str, dict[str, object]] = {}
    totals = {
        "zero_reference": 1.0,
        "zero_optimized": 1.0,
        "nonzero_optimized": 2.0,
        "nonzero_reference": 2.0,
    }
    for name in probe.ARM_ORDER:
        mode = "zero" if name.startswith("zero_") else "nonzero"
        implementation = "reference" if name.endswith("reference") else "optimized"
        diagnostic = not (mode == "zero" and implementation == "optimized")
        arm_artifacts[name] = {
            "name": name,
            "mode": mode,
            "implementation": implementation,
            "model_forward_count": 1,
            "backward_count": 1,
            "wall_seconds": 1.0,
            "total_loss": totals[name],
            "base_only_total_loss": 1.0,
            "diagnostic_requires_grad": diagnostic,
            "diagnostic_grad_fn": "FakeBackward" if diagnostic else None,
            "nonzero_control_differentiable": mode == "nonzero",
            "graph_residue": {
                "saved_graph_tensor_count": 1,
                "live_saved_graph_tensors_after_backward_and_release": 0,
                "diagnostic_tensor_live_after_release": False,
                "optimized_zero_diagnostic_released": not diagnostic,
            },
            "resources_before": resources,
            "resources_after": resources,
            "gradients": gradients,
            "buffers": transition,
        }
    gradient_map = {
        row["name"]: torch.ones(row["shape"], dtype=torch.float32)
        for row in gradients["rows"]
    }
    comparisons = {
        "raw_diagnostic_same_logits": probe.compare_scalar(
            1.0, 1.0, rtol=probe.LOSS_RTOL, atol=probe.LOSS_ATOL
        ),
        "zero_reference_base_only_total": probe.compare_scalar(
            1.0, 1.0, rtol=probe.LOSS_RTOL, atol=probe.LOSS_ATOL
        ),
        "zero_optimized_base_only_total": probe.compare_scalar(
            1.0, 1.0, rtol=probe.LOSS_RTOL, atol=probe.LOSS_ATOL
        ),
        "zero_total": probe.compare_scalar(
            1.0, 1.0, rtol=probe.BF16_RTOL, atol=probe.BF16_ATOL
        ),
        "zero_gradients": probe.compare_gradient_maps(gradient_map, gradient_map),
        "nonzero_total": probe.compare_scalar(
            2.0, 2.0, rtol=probe.BF16_RTOL, atol=probe.BF16_ATOL
        ),
        "nonzero_gradients": probe.compare_gradient_maps(gradient_map, gradient_map),
        "nonzero_control_graph": {
            "reference_differentiable": True,
            "optimized_differentiable": True,
            "passed": True,
        },
    }
    oracle = {
        "status": "completed",
        "model_forward_count": 1,
        "backward_count": 0,
        "same_logits_object": True,
        "logits_dtype": "torch.float32",
        "logits_sha256": "9" * 64,
        "reference_requires_grad": True,
        "optimized_requires_grad": False,
        "optimized_grad_fn": None,
        "raw_diagnostic_comparison": comparisons["raw_diagnostic_same_logits"],
        "buffers": transition,
    }
    efficiency_result = probe._efficiency_from_artifacts(
        arm_artifacts["zero_reference"], arm_artifacts["zero_optimized"]
    )
    return probe._finalize(
        {
            "schema": probe.RECEIPT_SCHEMA,
            "status": "passed",
            "plan_sha256": plan["plan_sha256"],
            "model_weight_identity": plan["model_weight_identity"],
            "plan_binding": probe._plan_binding(plan),
            "source_identity": plan["source_identity"],
            "terminal_reason": None,
            "attempt_marker": {
                "status": "published",
                "path": str(marker_path),
                "marker_sha256": marker["marker_sha256"],
            },
            "completed_phases": list(probe.PASSED_PHASE_ORDER),
            "completed_arm_order": list(probe.ARM_ORDER),
            "counts": {"model_forwards": 5, "backwards": 4},
            "oracle": oracle,
            "arms": {
                name: {"status": "completed", "artifact": arm_artifacts[name]}
                for name in probe.ARM_ORDER
            },
            "comparisons": comparisons,
            "efficiency": {"status": "completed", "result": efficiency_result},
            "runtime": {
                "status": "gpu_ready",
                "requested_device": "cuda:0",
                "precision": "bf16",
                "memory_savers": {},
                "cache_access": "read_only_no_writes",
                "model_weight_identity": plan["model_weight_identity"],
                "runtime_config_attestation": (
                    probe.frozen_runtime_config_attestation()
                ),
                "runtime_source_identities": runtime_sources,
                "concrete_trainable_inventory": inventory,
                "initial_buffer_inventory": buffers,
                "resource_limits": probe._resource_limits(),
                "shared_gpu_contract": {
                    "status": "post_probe_verified",
                    "claim_boundary": probe.shared_gpu_claim_boundary(),
                    "baseline_process_action": ("observe_only_never_signal_or_reset"),
                    "baseline": idle,
                    "pre_marker_observation": pre_marker_observation,
                    "post_probe_preterminal_observation": (
                        _fake_shared_gpu_sweep(probe)
                    ),
                },
                "process_cleanup": _fake_process_cleanup(probe),
                "controller": {
                    "status": "exited",
                    "worker_pid": 1,
                    "termination": None,
                    "max_host_rss_bytes": 1024,
                    "max_gpu_memory_bytes": 1024,
                },
            },
        },
        hash_field="receipt_sha256",
    )


def _fake_inventory(probe) -> dict[str, object]:
    from src.qwen.parity import frozen_trainable_inventory_declaration

    suffixes = {
        "lora_A": ".lora_A.default.weight",
        "lora_B": ".lora_B.default.weight",
        "dora_magnitude": ".lora_magnitude_vector.default.weight",
        "special_token_delta": ".shared_embed_delta",
    }
    rows = [
        {
            "name": f"layer_{index:03d}{suffixes[group]}",
            "group": group,
            "shape": [1],
            "parameter_storage_dtype": "torch.float32",
            "expected_gradient_dtype": "torch.float32",
            "compute_provenance_dtype": "torch.bfloat16",
        }
        for group, count in probe.EXPECTED_TRAINABLE_STRUCTURE.items()
        for index in range(count)
    ]
    rows.sort(key=lambda row: row["name"])
    body = {
        "schema": "coordexp-swift-wave2-concrete-trainable-inventory-v1",
        "declaration_sha256": probe.sha256_json(
            frozen_trainable_inventory_declaration()
        ),
        "total_count": len(rows),
        "group_counts": dict(probe.EXPECTED_TRAINABLE_STRUCTURE),
        "parameters": rows,
    }
    return {**body, "inventory_sha256": probe.sha256_json(body)}


def _fake_buffers(probe) -> dict[str, object]:
    rows = [
        {
            "name": "buffer",
            "shape": [1],
            "dtype": "torch.float32",
            "numel": 1,
            "finite": True,
            "sha256": "8" * 64,
        }
    ]
    body = {"count": len(rows), "rows": rows}
    return {**body, "inventory_sha256": probe.sha256_json(body)}


def _fake_gradients(probe, inventory: dict[str, object]) -> dict[str, object]:
    rows = [
        {
            "name": row["name"],
            "shape": row["shape"],
            "storage_dtype": row["parameter_storage_dtype"],
            "gradient_dtype": "torch.float32",
            "comparison_dtype": "torch.float32",
            "numel": 1,
            "finite": True,
            "sha256": "7" * 64,
            "l2_norm": 1.0,
        }
        for row in inventory["parameters"]
    ]
    return {
        "count": len(rows),
        "rows": rows,
        "rows_sha256": probe.sha256_json(rows),
        "aggregate_nonzero": True,
    }


def _fake_resources() -> dict[str, object]:
    return {
        "schema_version": 1,
        "cpu": {
            "scope": "current_process",
            "max_rss_bytes": 1024,
            "io_read_bytes": 0,
            "io_write_bytes": 0,
        },
        "gpu": {
            "scope": "current_process_current_device",
            "initialized": True,
            "device_index": 0,
            "max_memory_allocated_bytes": 1024,
            "max_memory_reserved_bytes": 1024,
        },
    }


def _fake_runtime_sources(probe) -> dict[str, object]:
    artifacts = {
        "model": _fake_model_source_artifact(load_model=True),
        "adapter": {
            "plan": {
                "mode": "initialize_new",
                "adapter_type": "dora",
                "source_gate": {"status": "passed"},
            },
            "setup_receipt": {
                "mode": "initialize_new",
                "adapter_type": "dora",
                "trainable_names": ["adapter"],
            },
        },
        "embedding_delta": {
            "install_receipt": {
                "semantics": "additive_delta",
                "tensor_key": "shared_embed_delta",
                "delta_parameter_names": ["shared_embed_delta"],
            },
            "load_receipt": None,
        },
    }
    return {
        owner: {"artifact": artifact, "sha256": probe.sha256_json(artifact)}
        for owner, artifact in artifacts.items()
    }


def _fake_model_source_artifact(*, load_model: bool) -> dict[str, object]:
    return {
        "base_model_path": _fake_model_weight_identity()["root"],
        "base_config_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "load_model": load_model,
        "attn_implementation": "flash_attention_2",
        "processor": {"class": "fake"},
        "model": {"model_type": "qwen3_vl"},
        "tokens": {"vocab_size": 1},
        "package_versions": {"torch": "test"},
        "runtime_patches": {},
    }


def _fake_shared_gpu_baseline(probe) -> dict[str, object]:
    return {
        "schema": probe.SHARED_GPU_BASELINE_SCHEMA,
        "mode": "shared_preexisting_compute",
        "requested_device": "cuda:0",
        "cuda_visible_devices": None,
        "selector": "0",
        "gpu_identity": {
            "physical_index": 0,
            "uuid": "GPU-fake",
            "total_memory_mib": probe.SHARED_GPU_TOTAL_MEMORY_MIB,
        },
        "limits": {
            "sample_count": probe.SHARED_GPU_SAMPLE_COUNT,
            "minimum_sample_interval_seconds": (
                probe.SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS
            ),
            "max_preexisting_memory_mib": (probe.SHARED_GPU_MAX_PREEXISTING_MEMORY_MIB),
            "required_headroom_mib": probe.SHARED_GPU_REQUIRED_HEADROOM_MIB,
        },
        "utilization_disposition": ("observational_only_not_admission_or_promotion"),
        "baseline_compute_processes": [],
        "samples": [
            {
                "sample_index": index,
                "monotonic_ns": 1 + index * 2_000_000_000,
                "memory_used_mib": 0,
                "headroom_mib": probe.SHARED_GPU_TOTAL_MEMORY_MIB,
                "utilization_percent": 0,
                "compute_processes": [],
            }
            for index in range(probe.SHARED_GPU_SAMPLE_COUNT)
        ],
    }


def _fake_shared_gpu_observation(probe, *, stage: str) -> dict[str, object]:
    baseline = _fake_shared_gpu_baseline(probe)
    return {
        "schema": probe.SHARED_GPU_OBSERVATION_SCHEMA,
        "stage": stage,
        "monotonic_ns": 5_000_000_002,
        "gpu_identity": baseline["gpu_identity"],
        "memory_used_mib": 0,
        "headroom_mib": probe.SHARED_GPU_TOTAL_MEMORY_MIB,
        "utilization_percent": 0,
        "compute_processes": [],
        "missing_baseline_processes": [],
        "new_processes": [],
    }


def _fake_shared_gpu_sweep(probe) -> dict[str, object]:
    samples = []
    for index in range(probe.SHARED_GPU_SAMPLE_COUNT):
        observation = {
            **_fake_shared_gpu_observation(probe, stage="post_probe_preterminal"),
            "monotonic_ns": 5_000_000_002 + index * 2_000_000_000,
        }
        samples.append(
            {
                "sample_index": index,
                "monotonic_ns": observation["monotonic_ns"],
                "status": "passed",
                "observation": observation,
                "error": None,
            }
        )
    return {
        "schema": probe.SHARED_GPU_SUBSET_SWEEP_SCHEMA,
        "stage": "post_probe_preterminal",
        "sample_count": probe.SHARED_GPU_SAMPLE_COUNT,
        "minimum_sample_interval_seconds": (
            probe.SHARED_GPU_MINIMUM_SAMPLE_INTERVAL_SECONDS
        ),
        "status": "passed",
        "samples": samples,
        "terminal_reason": None,
    }


def _fake_process_cleanup(probe, *, status: str = "passed") -> dict[str, object]:
    row = {
        "pid": 1,
        "state": "S",
        "parent_pid": 0,
        "process_group_id": 1,
        "session_id": 1,
        "start_time_ticks": 1,
    }
    passed = status == "passed"
    return {
        "schema": probe.PROCESS_CLEANUP_SCHEMA,
        "status": status,
        "reason": "worker_terminal_cleanup",
        "leader_pid": 1,
        "cleanup_timeout_seconds": probe.PROCESS_CLEANUP_TIMEOUT_SECONDS,
        "tracked_processes": [row],
        "term_signals": [],
        "kill_signals": [],
        "leader_absent": passed,
        "descendants_absent": passed,
        "sessions_absent": passed,
        "surviving_processes": [] if passed else [row],
        "completed_monotonic_ns": 9_000_000_000,
        "terminal_reason": None
        if passed
        else {
            "code": "wave3.process_cleanup_incomplete",
            "type": "Wave3ProbeError",
            "message": "injected cleanup survivor",
        },
    }


def test_plan_schema_accepts_only_the_frozen_exact_contract(tmp_path: Path) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    assert probe.validate_plan(plan) == plan

    mutations = []
    extra = deepcopy(plan)
    extra["unexpected"] = True
    mutations.append((extra, "wave3.plan_fields"))
    wrong_config = deepcopy(plan)
    wrong_config["config"]["fingerprint"] = "f" * 64
    mutations.append((wrong_config, "wave3.config_identity"))
    writable_cache = deepcopy(plan)
    writable_cache["cache"]["shared_cache_writes"] = True
    mutations.append((writable_cache, "wave3.cache_identity"))
    reordered = deepcopy(plan)
    reordered["execution_contract"]["arm_order"] = list(reversed(probe.ARM_ORDER))
    mutations.append((reordered, "wave3.arm_order"))
    widened = deepcopy(plan)
    widened["loss_contract"]["raw_diagnostic_tolerance"]["rtol"] = 1.0
    mutations.append((widened, "wave3.tolerance"))
    collided = deepcopy(plan)
    collided["artifact_targets"]["receipt"] = collided["artifact_targets"]["plan"]
    mutations.append((collided, "wave3.artifact_targets"))
    substituted_argv = deepcopy(plan)
    substituted_argv["execution_contract"]["run_argv"] = ["substituted"]
    substituted_argv["execution_contract"]["run_argv_sha256"] = probe.sha256_json(
        ["substituted"]
    )
    mutations.append((substituted_argv, "wave3.run_argv"))
    missing_sidecar = deepcopy(plan)
    missing_sidecar["artifact_targets"].pop("publication_failure")
    mutations.append((missing_sidecar, "wave3.artifact_targets"))
    host_limit = deepcopy(plan)
    host_limit["execution_contract"]["host_rss_ceiling_bytes"] -= 1
    mutations.append((host_limit, "wave3.forward_count"))

    for mutated, code in mutations:
        mutated = _refinalize(probe, mutated, "plan_sha256")
        with pytest.raises(probe.Wave3ProbeError) as caught:
            probe.validate_plan(mutated)
        assert caught.value.code == code


def test_active_v4_plan_requires_strict_base_model_weight_identity(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)

    assert probe.PLAN_SCHEMA.endswith("-v4")
    assert probe.MARKER_SCHEMA.endswith("-v4")
    assert probe.RECEIPT_SCHEMA.endswith("-v4")
    plan = _valid_plan(probe, tmp_path)
    assert (
        plan["execution_contract"]["shared_gpu_utilization_disposition"]
        == "observational_only_not_admission_or_promotion"
    )
    assert (
        probe.EXPECTED_BASE_MODEL_WEIGHT_AGGREGATE_SHA256
        == "e128f5f42f1a042702efc1eed5a787da36a4d586a17b538671260996056284aa"
    )
    assert probe.validate_plan(plan) == plan
    assert (
        parity_module.validate_model_weight_identity(plan["model_weight_identity"])
        == plan["model_weight_identity"]
    )

    mutated = deepcopy(plan)
    mutated["model_weight_identity"]["unexpected"] = True
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.validate_plan(_refinalize(probe, mutated, "plan_sha256"))
    assert caught.value.code == "wave3.model_weight_identity"

    deep_mutation = deepcopy(plan)
    deep_mutation["model_weight_identity"]["shards"][0]["sha256"] = "b" * 64
    weight_body = dict(deep_mutation["model_weight_identity"])
    weight_body.pop("aggregate_sha256")
    deep_mutation["model_weight_identity"]["aggregate_sha256"] = (
        parity_module.sha256_json(weight_body)
    )
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.validate_plan(_refinalize(probe, deep_mutation, "plan_sha256"))
    assert caught.value.code == "wave3.model_weight_identity"


@pytest.mark.parametrize("mutation", ["index", "shard"])
def test_fresh_weight_rehash_rejects_index_or_shard_mutation(
    tmp_path: Path, mutation: str
) -> None:
    probe = _load_probe_module()
    root = tmp_path / "model"
    planned = _write_tiny_indexed_model(root)
    assert probe._assert_current_model_weight_identity(planned) == planned

    target = (
        root / "model.safetensors.index.json"
        if mutation == "index"
        else root / "model-00002-of-00002.safetensors"
    )
    if mutation == "index":
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["metadata"]["total_size"] += 1
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    else:
        target.write_bytes(target.read_bytes() + b"-mutated")

    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._assert_current_model_weight_identity(planned)
    assert caught.value.code == "wave3.model_weight_identity"


def test_historical_r2_reader_is_exact_failed_only_and_non_executable() -> None:
    probe = _load_probe_module()
    root = probe.HISTORICAL_R2_ROOT
    plan_bytes = (root / "plan.json").read_bytes()
    marker_bytes = (root / "attempt-marker.json").read_bytes()
    receipt_bytes = (root / "terminal-receipt.json").read_bytes()

    summary = probe.load_historical_r2_failed_evidence()
    assert summary == {
        "status": "historical_non_executable",
        "root": str(root),
        "plan_file_sha256": probe.HISTORICAL_R2_PLAN_FILE_SHA256,
        "plan_sha256": probe.HISTORICAL_R2_PLAN_SHA256,
        "marker_file_sha256": probe.HISTORICAL_R2_MARKER_FILE_SHA256,
        "marker_sha256": probe.HISTORICAL_R2_MARKER_SHA256,
        "receipt_file_sha256": probe.HISTORICAL_R2_RECEIPT_FILE_SHA256,
        "receipt_sha256": probe.HISTORICAL_R2_RECEIPT_SHA256,
        "terminal_reason_code": "wave3.host_watchdog",
    }
    historical_plan = json.loads(plan_bytes)
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.validate_plan(historical_plan)
    assert caught.value.code == "wave3.schema"

    for field, value in (
        ("plan_bytes", plan_bytes + b" "),
        ("marker_bytes", marker_bytes + b" "),
        ("receipt_bytes", receipt_bytes + b" "),
    ):
        payloads = {
            "plan_bytes": plan_bytes,
            "marker_bytes": marker_bytes,
            "receipt_bytes": receipt_bytes,
        }
        payloads[field] = value
        with pytest.raises(probe.Wave3ProbeError) as caught:
            probe._validate_historical_r2_failed_bytes(**payloads)
        assert caught.value.code == "wave3.historical_r2"

    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.load_historical_r2_failed_evidence(root.parent)
    assert caught.value.code == "wave3.historical_r2"

    r3_plan_path = root.parent / "2026-08-10-r3/plan.json"
    r3_plan_bytes = r3_plan_path.read_bytes()
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.validate_plan(json.loads(r3_plan_bytes))
    assert caught.value.code == "wave3.schema"
    assert r3_plan_path.read_bytes() == r3_plan_bytes


def test_historical_r3_v3_failed_chain_is_exact_and_non_executable() -> None:
    """Break caught: mutating or accidentally re-admitting the consumed v3 chain."""

    probe = _load_probe_module()
    summary = probe.load_historical_r3_v3_failed_evidence()
    assert summary == {
        "status": "historical_non_executable",
        "root": str(probe.HISTORICAL_R3_V3_ROOT),
        "plan_file_sha256": "f4936e01729722ce297f9d9a09bd791967fcb741ada827d32b1321deaa86288e",
        "plan_sha256": "0d2f9c575968ab1f79c530ac6a501190b183fc6e29fd835cbeb5ce555f98a90a",
        "marker_file_sha256": "27b2b957d6f96b9a454d3394773fbc65672b673824e14b45c0f5373847c9b297",
        "marker_sha256": "3df128f6ea3b4c12cf822dbcb5258f5a0da0f74d58a7a6bd46a92e09fbf400d7",
        "receipt_file_sha256": "6283baa2b72c0e730bb671179980423564188b9d910778cf29ba66042298958c",
        "receipt_sha256": "b0be5a695abfb4c657d60afba82988f8381159664859e62ea19b55ace0dd148e",
        "terminal_reason_code": "wave3.accelerator",
    }
    historical_plan = json.loads(
        (probe.HISTORICAL_R3_V3_ROOT / "plan.json").read_text(encoding="utf-8")
    )
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.validate_plan(historical_plan)
    assert caught.value.code == "wave3.schema"


def test_worker_fresh_weight_drift_fails_before_model_load_gpu_or_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    evidence = probe._empty_evidence(
        requested_device="cuda:0",
        marker_path=Path(plan["artifact_targets"]["attempt_marker"]),
        plan=plan,
    )
    resolved = probe.load_train_config(probe.FROZEN_CONFIG_PATH)
    monkeypatch.setattr(probe, "load_train_config", lambda _path: resolved)
    monkeypatch.setattr(
        probe, "_load_frozen_micro_step", lambda _path: ({}, SimpleNamespace())
    )
    monkeypatch.setattr(probe, "_workload_identity", lambda _micro: plan["workload"])
    monkeypatch.setattr(
        probe,
        "_assert_current_model_weight_identity",
        lambda _expected: (_ for _ in ()).throw(
            probe.Wave3ProbeError(
                "injected shard drift", code="wave3.model_weight_identity"
            )
        ),
    )
    for owner in (
        "load_qwen_components",
        "_assert_gpu_idle",
        "publish_json_absent",
        "_build_exact_one_rank_accelerator",
    ):
        monkeypatch.setattr(
            probe,
            owner,
            lambda *_args, _owner=owner, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"{_owner} must not run")
            ),
        )

    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._execute_gpu_probe(
            plan,
            device=torch.device("cuda:0"),
            evidence=evidence,
            run_started=probe.time.monotonic(),
            marker_target=Path(plan["artifact_targets"]["attempt_marker"]),
            publication_owner_token_sha256="6" * 64,
            worker_argv=probe._artifact_argv(
                command="_worker",
                plan_path=plan["artifact_targets"]["plan"],
                receipt_path=plan["artifact_targets"]["receipt"],
                attempt_marker_path=plan["artifact_targets"]["attempt_marker"],
                publication_failure_path=plan["artifact_targets"][
                    "publication_failure"
                ],
                device="cuda:0",
            ),
            shared_gpu_baseline={},
            emit=lambda _event: None,
        )
    assert caught.value.code == "wave3.model_weight_identity"
    assert evidence["runtime"]["status"] == "not_started"
    assert evidence["attempt_marker"]["status"] == "not_published"
    assert not Path(plan["artifact_targets"]["attempt_marker"]).exists()


@pytest.mark.parametrize("command", ["run", "controller", "_worker"])
@pytest.mark.parametrize("plan_kind", ["r2", "r3", "synthetic_v2"])
def test_v2_entry_gate_is_typed_and_has_zero_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    plan_kind: str,
) -> None:
    probe = _load_probe_module()
    if plan_kind == "r2":
        plan_path = probe.HISTORICAL_R2_ROOT / "plan.json"
    elif plan_kind == "r3":
        plan_path = probe.HISTORICAL_R2_ROOT.parent / "2026-08-10-r3/plan.json"
    else:
        synthetic = _valid_plan(probe, tmp_path)
        synthetic["schema"] = probe.HISTORICAL_R2_PLAN_SCHEMA
        plan_path = tmp_path / "synthetic-v2-plan.json"
        plan_path.write_text(json.dumps(synthetic), encoding="utf-8")

    side_effects: list[str] = []

    def forbidden(owner: str):
        def fail(*_args, **_kwargs):
            side_effects.append(owner)
            raise AssertionError(f"{owner} must not run")

        return fail

    monkeypatch.setattr(probe, "_parse_cuda_device", forbidden("cuda_parse"))
    monkeypatch.setattr(probe.subprocess, "Popen", forbidden("popen"))
    monkeypatch.setattr(
        probe, "_publish_terminal_receipt", forbidden("receipt_publish")
    )
    monkeypatch.setattr(probe, "publish_json_absent", forbidden("artifact_publish"))
    monkeypatch.setattr(probe, "_failure_receipt", forbidden("failure_receipt"))
    monkeypatch.setattr(probe, "build_plan", forbidden("cache_or_plan_build"))
    monkeypatch.setattr(probe, "load_qwen_components", forbidden("model_load"))
    targets = {
        "receipt": tmp_path / f"{command}-receipt.json",
        "attempt_marker": tmp_path / f"{command}-marker.json",
        "publication_failure": tmp_path / f"{command}-sidecar.json",
    }
    exit_code = probe.main(
        [
            command,
            "--plan",
            str(plan_path),
            "--receipt",
            str(targets["receipt"]),
            "--attempt-marker",
            str(targets["attempt_marker"]),
            "--publication-failure",
            str(targets["publication_failure"]),
            "--device",
            "cuda:0",
        ]
    )

    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert failure["status"] == "failed"
    assert failure["code"] == "wave3.schema"
    assert failure["command"] == command
    assert side_effects == []
    assert not any(path.exists() for path in targets.values())


def test_controller_binding_gate_precedes_cuda_and_publishable_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    plan_path = Path(plan["artifact_targets"]["plan"])
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    wrong_receipt = tmp_path / "wrong-receipt.json"
    side_effects: list[str] = []

    def forbidden(owner: str):
        def fail(*_args, **_kwargs):
            side_effects.append(owner)
            raise AssertionError(f"{owner} must not run")

        return fail

    monkeypatch.setattr(probe, "_parse_cuda_device", forbidden("cuda_parse"))
    monkeypatch.setattr(probe.subprocess, "Popen", forbidden("popen"))
    monkeypatch.setattr(
        probe, "_publish_terminal_receipt", forbidden("receipt_publish")
    )
    monkeypatch.setattr(probe, "_failure_receipt", forbidden("failure_receipt"))

    exit_code = probe.main(
        [
            "controller",
            "--plan",
            str(plan_path),
            "--receipt",
            str(wrong_receipt),
            "--attempt-marker",
            str(plan["artifact_targets"]["attempt_marker"]),
            "--publication-failure",
            str(plan["artifact_targets"]["publication_failure"]),
            "--device",
            "cuda:0",
        ]
    )

    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert failure["code"] == "wave3.run_binding"
    assert failure["command"] == "controller"
    assert side_effects == []
    assert not wrong_receipt.exists()
    assert not Path(plan["artifact_targets"]["attempt_marker"]).exists()
    assert not Path(plan["artifact_targets"]["publication_failure"]).exists()


def test_plan_hash_detects_unfinalized_mutation(tmp_path: Path) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    plan["status"] = "changed"
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.validate_plan(plan)
    assert caught.value.code == "wave3.hash"


def test_config_identity_allows_only_exact_enumerated_compatibility_projection(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    resolved = probe.load_train_config(probe.FROZEN_CONFIG_PATH)
    assert resolved.fingerprint == probe.FROZEN_CONFIG_FINGERPRINT
    assert probe.sha256_json(resolved.config_dict) == probe.FROZEN_CONFIG_FINGERPRINT
    assert (
        resolved.config.training.forward_input_provider_mode
        == probe.FORWARD_INPUT_PROVIDER_VALUE
    )
    projection = probe.build_config_compatibility_projection(resolved.config_dict)
    assert projection == probe.frozen_config_compatibility_projection()
    assert projection["schema"] == (
        "coordexp-swift-wave3-config-compatibility-projection-v2"
    )
    assert projection["removed_path_values"] == [
        {
            "path": "training.forward_input_provider_mode",
            "value": "synchronous",
        },
        {"path": "packing.policy", "value": "source_order_next_fit"},
        {"path": "packing.window_size", "value": None},
        {"path": "packing.lookahead", "value": None},
        {"path": "packing.seed", "value": 0},
        {"path": "packing.worker_count", "value": 1},
        {"path": "packing.fragment_item_budget", "value": 1024},
        {"path": "packing.fragment_byte_budget", "value": 4_194_304},
        {"path": "packing.cursor_byte_budget", "value": 65_536},
        {"path": "packing.max_packs_per_fragment", "value": None},
        {
            "path": "resume",
            "value": {"checkpoint_dir": None, "mode": "disabled"},
        },
    ]
    assert (
        projection["projected_config_sha256"] == probe.LEGACY_FROZEN_CONFIG_FINGERPRINT
    )
    plan = _valid_plan(probe, tmp_path)

    mutations = []
    non_synchronous = deepcopy(plan)
    non_synchronous["config"]["forward_input_provider_mode"] = "overlapped"
    mutations.append((non_synchronous, "wave3.config_identity"))
    extra_removed = deepcopy(plan)
    extra_removed["config"]["compatibility_projection"]["removed_path_values"].append(
        {"path": "training.precision", "value": "bf16"}
    )
    mutations.append((extra_removed, "wave3.config_projection"))
    missing_removed = deepcopy(plan)
    missing_removed["config"]["compatibility_projection"]["removed_path_values"].pop()
    mutations.append((missing_removed, "wave3.config_projection"))
    wrong_default = deepcopy(plan)
    wrong_default["config"]["compatibility_projection"]["removed_path_values"][1][
        "value"
    ] = "window_binpack"
    mutations.append((wrong_default, "wave3.config_projection"))
    wrong_digest = deepcopy(plan)
    wrong_digest["config"]["compatibility_projection"]["projected_config_sha256"] = (
        "f" * 64
    )
    mutations.append((wrong_digest, "wave3.config_projection"))
    wrong_legacy = deepcopy(plan)
    wrong_legacy["config"]["compatibility_projection"]["legacy_config_fingerprint"] = (
        "e" * 64
    )
    mutations.append((wrong_legacy, "wave3.config_projection"))

    for mutated, code in mutations:
        with pytest.raises(probe.Wave3ProbeError) as caught:
            probe.validate_plan(_refinalize(probe, mutated, "plan_sha256"))
        assert caught.value.code == code

    live = deepcopy(resolved.config_dict)
    live["training"]["forward_input_provider_mode"] = "overlapped"
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.attest_runtime_config(plan["config"], live)
    assert caught.value.code in {"wave3.config_identity", "wave3.config_provider"}

    for mutator in (
        lambda value: value["packing"].__setitem__("worker_count", 2),
        lambda value: value["packing"].pop("fragment_byte_budget"),
        lambda value: value["resume"].__setitem__("mode", "exact"),
        lambda value: value.pop("resume"),
        lambda value: value["runtime"].__setitem__(
            "seed", value["runtime"]["seed"] + 1
        ),
    ):
        drifted = deepcopy(resolved.config_dict)
        mutator(drifted)
        with pytest.raises(probe.Wave3ProbeError) as caught:
            probe.build_config_compatibility_projection(drifted)
        assert caught.value.code == "wave3.config_identity"


def test_marker_and_receipt_schemas_are_exact_and_plan_bound(tmp_path: Path) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    receipt = _valid_passed_receipt(probe, plan)
    marker = json.loads(
        Path(plan["artifact_targets"]["attempt_marker"]).read_text(encoding="utf-8")
    )
    assert probe.validate_marker(marker, expected_plan=plan) == marker
    assert probe.validate_receipt(receipt, expected_plan=plan) == receipt
    assert marker["model_weight_identity"] == plan["model_weight_identity"]
    assert receipt["model_weight_identity"] == plan["model_weight_identity"]
    assert (
        receipt["plan_binding"]["model_weight_identity_sha256"]
        == plan["model_weight_identity"]["aggregate_sha256"]
    )
    assert (
        receipt["runtime"]["model_weight_identity"] == marker["model_weight_identity"]
    )

    for artifact, hash_field, validator in (
        (marker, "marker_sha256", probe.validate_marker),
        (receipt, "receipt_sha256", probe.validate_receipt),
    ):
        mutated = deepcopy(artifact)
        mutated["model_weight_identity"]["shards"][0]["sha256"] = "c" * 64
        weight_body = dict(mutated["model_weight_identity"])
        weight_body.pop("aggregate_sha256")
        mutated["model_weight_identity"]["aggregate_sha256"] = (
            parity_module.sha256_json(weight_body)
        )
        mutated = _refinalize(probe, mutated, hash_field)
        with pytest.raises(probe.Wave3ProbeError):
            validator(mutated, expected_plan=plan)

    bad_receipt = deepcopy(receipt)
    bad_receipt["counts"]["model_forwards"] = 4
    bad_receipt = _refinalize(probe, bad_receipt, "receipt_sha256")
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.validate_receipt(bad_receipt, expected_plan=plan)
    assert caught.value.code == "wave3.receipt"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.pop("oracle"),
        lambda value: value["arms"].pop("zero_reference"),
        lambda value: value["arms"].update({"extra": {}}),
        lambda value: value["arms"]["zero_reference"]["artifact"].update(
            {"name": "zero_optimized"}
        ),
        lambda value: value["comparisons"].update({"zero_total": None}),
        lambda value: value["runtime"]["concrete_trainable_inventory"].update(
            {"total_count": 0}
        ),
        lambda value: value["runtime"].update({"runtime_source_identities": {}}),
        lambda value: value["comparisons"]["zero_total"].update({"threshold": 999.0}),
        lambda value: value.update({"unexpected": True}),
    ],
)
def test_passed_receipt_deep_mutations_reject(tmp_path: Path, mutator) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    receipt = _valid_passed_receipt(probe, plan)
    mutated = deepcopy(receipt)
    mutated.pop("receipt_sha256")
    mutator(mutated)
    mutated = probe._finalize(mutated, hash_field="receipt_sha256")
    with pytest.raises(probe.Wave3ProbeError):
        probe.validate_receipt(mutated, expected_plan=plan)


def test_prepare_argv_predeclares_all_artifacts_device_and_read_only_cache(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    argv = probe._prepare_argv(
        config_path=probe.FROZEN_CONFIG_PATH,
        cache_dir=probe.W0_CACHE_DIR,
        plan_path=tmp_path / "plan.json",
        receipt_path=tmp_path / "receipt.json",
        attempt_marker_path=tmp_path / "attempt.json",
        publication_failure_path=tmp_path / "publication-failure.json",
        device="cuda:00",
    )
    assert argv[2] == "prepare"
    assert argv[argv.index("--device") + 1] == "cuda:0"
    assert argv[argv.index("--cache-dir") + 1] == str(probe.W0_CACHE_DIR.resolve())
    for flag in (
        "--plan",
        "--receipt",
        "--attempt-marker",
        "--publication-failure",
    ):
        assert Path(argv[argv.index(flag) + 1]).is_absolute()


def test_prepare_collision_is_fail_closed_before_plan_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    occupied = tmp_path / "plan.json"
    occupied.write_text("owned", encoding="utf-8")
    monkeypatch.setattr(
        probe,
        "build_plan",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    args = argparse.Namespace(
        plan=occupied,
        receipt=tmp_path / "receipt.json",
        attempt_marker=tmp_path / "attempt.json",
        publication_failure=tmp_path / "publication-failure.json",
        config=probe.FROZEN_CONFIG_PATH,
        cache_dir=probe.W0_CACHE_DIR,
        device="cuda:0",
    )
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.prepare_command(args)
    assert caught.value.code == "wave3.immutable_collision"
    assert occupied.read_text(encoding="utf-8") == "owned"


def test_prepare_publishes_once_without_touching_other_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    monkeypatch.setattr(probe, "build_plan", lambda **_kwargs: plan)
    args = argparse.Namespace(
        plan=tmp_path / "plan.json",
        receipt=tmp_path / "receipt.json",
        attempt_marker=tmp_path / "attempt.json",
        publication_failure=tmp_path / "publication-failure.json",
        config=probe.FROZEN_CONFIG_PATH,
        cache_dir=probe.W0_CACHE_DIR,
        device="cuda:0",
    )
    assert probe.prepare_command(args) == 0
    assert json.loads(args.plan.read_text(encoding="utf-8")) == plan
    assert not args.receipt.exists()
    assert not args.attempt_marker.exists()
    assert not args.publication_failure.exists()
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.prepare_command(args)
    assert caught.value.code == "wave3.immutable_collision"


def test_publish_json_absent_never_replaces_existing_bytes(tmp_path: Path) -> None:
    probe = _load_probe_module()
    target = tmp_path / "artifact.json"
    probe.publish_json_absent(target, {"value": 1})
    original = target.read_bytes()
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.publish_json_absent(target, {"value": 2})
    assert caught.value.code == "wave3.immutable_collision"
    assert target.read_bytes() == original
    broken_link = tmp_path / "broken.json"
    broken_link.symlink_to(tmp_path / "missing.json")
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.publish_json_absent(broken_link, {"value": 3})
    assert caught.value.code == "wave3.immutable_collision"
    assert broken_link.is_symlink()


def test_atomic_publication_prelink_failure_leaves_no_final_or_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    target = tmp_path / "artifact.json"

    def fail_before_link(*_args, **_kwargs):
        raise OSError("injected pre-link failure")

    monkeypatch.setattr(probe, "write_strict_json_atomic", fail_before_link)
    with pytest.raises(probe.Wave3ArtifactPublicationError) as caught:
        probe.publish_json_absent(target, {"value": 1})
    assert caught.value.linked_by_this_call is False
    assert caught.value.reloaded_exact is False
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure_kind", ["directory_fsync", "temporary_cleanup"])
def test_atomic_publication_postlink_failure_is_typed_and_exactly_reloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    probe = _load_probe_module()
    target = tmp_path / "artifact.json"
    linked: list[str] = []
    if failure_kind == "directory_fsync":
        original_fsync = parity_module.os.fsync
        calls = 0

        def fail_directory_fsync(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected directory fsync failure")
            return original_fsync(fd)

        monkeypatch.setattr(parity_module.os, "fsync", fail_directory_fsync)
    else:
        original_unlink = Path.unlink

        def fail_temporary_cleanup(path, *args, **kwargs):
            if path.parent == tmp_path and path.name.startswith(".artifact.json."):
                raise OSError("injected temporary cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_temporary_cleanup)

    with pytest.raises(probe.Wave3ArtifactPublicationError) as caught:
        probe.publish_json_absent(
            target,
            {"value": 1},
            on_linked=lambda: linked.append("owned_link"),
        )
    assert caught.value.linked_by_this_call is True
    assert caught.value.reloaded_exact is True
    assert linked == ["owned_link"]
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}


def test_atomic_publication_directory_open_failure_is_typed_and_exactly_reloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    target = tmp_path / "artifact.json"
    linked: list[str] = []
    original_open = parity_module.os.open

    def fail_directory_open(path, flags, *args, **kwargs):
        if Path(path) == tmp_path:
            raise OSError("injected directory open failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(parity_module.os, "open", fail_directory_open)
    with pytest.raises(probe.Wave3ArtifactPublicationError) as caught:
        probe.publish_json_absent(
            target,
            {"value": 1},
            on_linked=lambda: linked.append("owned_link"),
        )
    assert caught.value.linked_by_this_call is True
    assert caught.value.reloaded_exact is True
    assert linked == ["owned_link"]
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}


@pytest.mark.parametrize("existing", [{"value": 1}, {"foreign": True}])
def test_atomic_publication_collision_never_claims_ownership_or_recovery(
    tmp_path: Path, existing: dict[str, object]
) -> None:
    probe = _load_probe_module()
    target = tmp_path / "artifact.json"
    probe.publish_json_absent(target, existing)
    original = target.read_bytes()
    linked: list[str] = []
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.publish_json_absent(
            target,
            {"value": 1},
            on_linked=lambda: linked.append("owned_link"),
        )
    assert caught.value.code == "wave3.immutable_collision"
    assert not isinstance(caught.value, probe.Wave3ArtifactPublicationError)
    assert linked == []
    assert target.read_bytes() == original


def test_atomic_publication_rejects_mutated_postlink_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    target = tmp_path / "artifact.json"
    original_writer = probe.write_strict_json_atomic

    def mutate_after_link(path, payload, *, on_linked=None):
        def linked():
            if on_linked is not None:
                on_linked()
            Path(path).write_text('{"mutated":true}\n', encoding="utf-8")

        return original_writer(path, payload, on_linked=linked)

    monkeypatch.setattr(probe, "write_strict_json_atomic", mutate_after_link)
    with pytest.raises(probe.Wave3ArtifactPublicationError) as caught:
        probe.publish_json_absent(target, {"value": 1})
    assert caught.value.linked_by_this_call is True
    assert caught.value.reloaded_exact is False
    assert json.loads(target.read_text(encoding="utf-8")) == {"mutated": True}


def test_same_logits_guard_rejects_second_forward_and_tensor_substitution() -> None:
    probe = _load_probe_module()
    logits = torch.randn(2, 3)
    probe.assert_single_forward_same_logits(
        forward_count=1,
        backward_count=0,
        logits=logits,
        diagnostic_logits=(logits, logits),
    )
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.assert_single_forward_same_logits(
            forward_count=2,
            backward_count=0,
            logits=logits,
            diagnostic_logits=(logits,),
        )
    assert caught.value.code == "wave3.forward_count"
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.assert_single_forward_same_logits(
            forward_count=1,
            backward_count=0,
            logits=logits,
            diagnostic_logits=(logits.clone(),),
        )
    assert caught.value.code == "wave3.same_logits"


def test_oracle_executes_one_forward_and_reuses_exact_fp32_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
    forward_calls = 0
    context_ids: list[int] = []

    def fake_forward(*_args, **_kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return SimpleNamespace(
            logits=logits,
            logits_position_ids=torch.tensor([0]),
        )

    class Runner:
        def __init__(self, *, detached: bool) -> None:
            self.detached = detached

        def compute_micro_step(self, context, _plan, *, local_micro_step_index: int):
            assert local_micro_step_index == 0
            context_ids.append(id(context.logits))
            raw = context.logits.sum()
            if self.detached:
                raw = raw.detach()
            term = SimpleNamespace(raw_loss=raw)
            return SimpleNamespace(term_by_name=lambda _name: term)

    monkeypatch.setattr(probe, "run_qwen_forward", fake_forward)
    monkeypatch.setattr(
        probe, "LossContext", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    artifact = probe._execute_oracle_arm(
        model=object(),
        forward_inputs=SimpleNamespace(input_ids=torch.tensor([0])),
        micro_step=SimpleNamespace(
            expected_vocab_size=2,
            token_sequence=object(),
            vocab_groups=object(),
        ),
        differentiable=Runner(detached=False),
        optimized=Runner(detached=True),
        loss_plan=object(),
    )
    assert forward_calls == 1
    assert context_ids == [id(logits), id(logits)]
    assert artifact["same_logits_object"] is True
    assert artifact["raw_diagnostic_comparison"]["passed"] is True
    assert artifact["reference_requires_grad"] is True
    assert artifact["optimized_requires_grad"] is False


@pytest.mark.parametrize(
    ("left", "right", "rtol", "atol", "passed"),
    [
        (1.0 + 5.0e-6, 1.0, 1.0e-5, 1.0e-6, True),
        (1.0 + 2.0e-5, 1.0, 1.0e-5, 1.0e-6, False),
        (1.009, 1.0, 5.0e-3, 5.0e-3, True),
        (1.011, 1.0, 5.0e-3, 5.0e-3, False),
        (float("nan"), 1.0, 1.0, 1.0, False),
    ],
)
def test_scalar_comparison_uses_the_declared_fixed_band(
    left: float, right: float, rtol: float, atol: float, passed: bool
) -> None:
    probe = _load_probe_module()
    assert probe.compare_scalar(left, right, rtol=rtol, atol=atol)["passed"] is passed


def test_gradient_comparison_is_elementwise_fp32_and_exact_name_bound() -> None:
    probe = _load_probe_module()
    reference = {"p": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    within = {"p": torch.tensor([1.009, 2.0], dtype=torch.float16)}
    outside = {"p": torch.tensor([1.02, 2.0], dtype=torch.float32)}
    assert probe.compare_gradient_maps(reference, within)["passed"] is True
    assert probe.compare_gradient_maps(reference, outside)["passed"] is False
    missing = probe.compare_gradient_maps(reference, {"q": torch.tensor([1.0])})
    assert missing["passed"] is False
    assert missing["missing"] == ["p"]
    assert missing["extra"] == ["q"]


def test_failure_receipt_preserves_completed_prefix_and_evidence(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    evidence = probe._empty_evidence(
        requested_device="cuda:0",
        marker_path=Path(plan["artifact_targets"]["attempt_marker"]),
        plan=plan,
    )
    evidence["completed_phases"] = ["plan_revalidated", "shared_gpu_preflight"]
    failure = probe._failure_receipt(
        plan,
        evidence=evidence,
        exc=probe.Wave3ProbeError("boom", code="wave3.test_failure"),
    )
    assert probe.validate_receipt(failure, expected_plan=plan) == failure
    assert failure["completed_arm_order"] == []
    assert failure["counts"] == {"model_forwards": 0, "backwards": 0}
    assert failure["terminal_reason"]["code"] == "wave3.test_failure"
    assert failure["model_weight_identity"] == plan["model_weight_identity"]
    assert (
        failure["plan_binding"]["model_weight_identity_sha256"]
        == plan["model_weight_identity"]["aggregate_sha256"]
    )

    for mutator in (
        lambda value: value.pop("oracle"),
        lambda value: value.update({"arms": {}}),
        lambda value: value["comparisons"].update({"extra": None}),
        lambda value: value.update({"terminal_reason": {}}),
        lambda value: value["runtime"].update({"resource_limits": {}}),
    ):
        mutated = deepcopy(failure)
        mutated.pop("receipt_sha256")
        mutator(mutated)
        mutated = probe._finalize(mutated, hash_field="receipt_sha256")
        with pytest.raises(probe.Wave3ProbeError):
            probe.validate_receipt(mutated, expected_plan=plan)


def test_marker_rejects_inventory_source_and_command_substitution(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    _valid_passed_receipt(probe, plan)
    marker_path = Path(plan["artifact_targets"]["attempt_marker"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    mutations = []
    empty_source = deepcopy(marker)
    empty_source["runtime_source_identities"] = {}
    mutations.append(empty_source)
    inventory = deepcopy(marker)
    inventory["concrete_trainable_inventory"]["parameters"] = []
    mutations.append(inventory)
    command = deepcopy(marker)
    command["command_identity"]["worker_argv"] = ["substituted"]
    command["command_identity"]["worker_argv_sha256"] = probe.sha256_json(
        ["substituted"]
    )
    mutations.append(command)
    extra = deepcopy(marker)
    extra["unexpected"] = True
    mutations.append(extra)
    for mutated in mutations:
        mutated = _refinalize(probe, mutated, "marker_sha256")
        with pytest.raises(probe.Wave3ProbeError):
            probe.validate_marker(mutated, expected_plan=plan)


@pytest.mark.parametrize("configured_device", [None, "cuda:2"])
def test_accelerator_admission_sets_or_preserves_exact_requested_device_without_cuda(
    monkeypatch: pytest.MonkeyPatch, configured_device: str | None
) -> None:
    probe = _load_probe_module()
    admission = getattr(probe, "_admit_exact_one_rank_accelerator_environment", None)
    assert callable(admission)
    _reset_accelerate_shared_state(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    if configured_device is None:
        monkeypatch.delenv("ACCELERATE_TORCH_DEVICE", raising=False)
    else:
        monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", configured_device)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pre-marker admission must not touch CUDA")
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(
            AssertionError("pre-marker admission must not touch CUDA")
        ),
    )

    admission(device=torch.device("cuda:2"))

    assert os.environ["ACCELERATE_TORCH_DEVICE"] == "cuda:2"


def test_accelerate_1_10_indexless_direct_default_is_pinned_before_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import accelerate
    from accelerate.state import PartialState

    probe = _load_probe_module()
    admission = getattr(probe, "_admit_exact_one_rank_accelerator_environment", None)
    validator = getattr(probe, "_validate_exact_accelerator_runtime", None)
    assert callable(admission)
    assert callable(validator)
    assert accelerate.__version__ == "1.10.1"
    source = inspect.getsource(PartialState.default_device.fget)
    assert 'return torch.device("cuda")' in source
    _reset_accelerate_shared_state(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("ACCELERATE_TORCH_DEVICE", raising=False)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    indexless_direct_default = torch.device("cuda")
    assert indexless_direct_default != torch.device("cuda:0")
    admission(device=torch.device("cuda:0"))
    fake = _fake_exact_accelerator(
        device=os.environ.get("ACCELERATE_TORCH_DEVICE", "cuda")
    )

    identity = validator(
        fake,
        expected_device=torch.device("cuda:0"),
        expected_mixed_precision="bf16",
    )
    assert identity["device"] == "cuda:0"


def test_accelerator_admission_rejects_conflicting_device_without_mutation_or_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    admission = getattr(probe, "_admit_exact_one_rank_accelerator_environment", None)
    assert callable(admission)
    _reset_accelerate_shared_state(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cuda:1")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejected admission must not touch CUDA")
        ),
    )

    with pytest.raises(probe.Wave3ProbeError) as caught:
        admission(device=torch.device("cuda:0"))

    assert caught.value.code == "wave3.accelerator_device_override"
    assert os.environ["ACCELERATE_TORCH_DEVICE"] == "cuda:1"


@pytest.mark.parametrize(
    "contamination",
    ["partial_state", "accelerator_state", "distributed", "launcher"],
)
def test_accelerator_contamination_fails_before_marker_publication_or_cuda(
    monkeypatch: pytest.MonkeyPatch, contamination: str
) -> None:
    from accelerate.state import AcceleratorState, PartialState

    probe = _load_probe_module()
    admission = getattr(probe, "_admit_exact_one_rank_accelerator_environment", None)
    assert callable(admission)
    _reset_accelerate_shared_state(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "is_initialized",
        lambda: contamination == "distributed",
    )
    monkeypatch.delenv("ACCELERATE_TORCH_DEVICE", raising=False)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    if contamination == "partial_state":
        monkeypatch.setattr(PartialState, "_shared_state", {"device": "cuda"})
    elif contamination == "accelerator_state":
        monkeypatch.setattr(AcceleratorState, "_shared_state", {"device": "cuda"})
    elif contamination == "launcher":
        monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejected admission must not touch CUDA")
        ),
    )

    with pytest.raises(probe.Wave3ProbeError) as caught:
        admission(device=torch.device("cuda:0"))

    assert caught.value.code == "wave3.accelerator_not_fresh"
    assert "ACCELERATE_TORCH_DEVICE" not in os.environ
    execution_source = inspect.getsource(probe._execute_gpu_probe)
    admission_position = execution_source.index(
        "_admit_exact_one_rank_accelerator_environment"
    )
    marker_position = execution_source.index("publish_json_absent(marker_target")
    build_position = execution_source.index("_build_exact_one_rank_accelerator")
    assert admission_position < marker_position < build_position
    assert "torch.cuda" not in inspect.getsource(admission)


def test_exact_accelerator_runtime_accepts_complete_one_rank_bf16_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    validator = getattr(probe, "_validate_exact_accelerator_runtime", None)
    assert callable(validator)
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cuda:2")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    accelerator = _fake_exact_accelerator(device="cuda:2")

    identity = validator(
        accelerator,
        expected_device=torch.device("cuda:2"),
        expected_mixed_precision="bf16",
    )

    assert identity == {
        "distributed_type": "NO",
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "device": "cuda:2",
        "cuda_current_device": 2,
        "mixed_precision": "bf16",
        "native_amp": True,
        "gradient_accumulation_steps": 1,
        "scaler": None,
        "accelerate_torch_device": "cuda:2",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "distributed_type",
        "rank",
        "local_rank",
        "world_size",
        "indexless_device",
        "current_device",
        "mixed_precision",
        "native_amp",
        "accumulation",
        "scaler",
        "device_environment",
    ],
)
def test_exact_accelerator_runtime_rejects_each_identity_drift(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    probe = _load_probe_module()
    validator = getattr(probe, "_validate_exact_accelerator_runtime", None)
    assert callable(validator)
    accelerator = _fake_exact_accelerator(device="cuda:2")
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cuda:2")
    current_device = 2
    if mutation == "distributed_type":
        accelerator.distributed_type = SimpleNamespace(name="MULTI_GPU")
    elif mutation == "rank":
        accelerator.process_index = 1
    elif mutation == "local_rank":
        accelerator.local_process_index = 1
    elif mutation == "world_size":
        accelerator.num_processes = 2
    elif mutation == "indexless_device":
        accelerator.device = torch.device("cuda")
    elif mutation == "current_device":
        current_device = 1
    elif mutation == "mixed_precision":
        accelerator.mixed_precision = "fp16"
    elif mutation == "native_amp":
        accelerator.native_amp = False
    elif mutation == "accumulation":
        accelerator.gradient_accumulation_steps = 2
    elif mutation == "scaler":
        accelerator.scaler = object()
    elif mutation == "device_environment":
        monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)

    with pytest.raises(probe.Wave3ProbeError) as caught:
        validator(
            accelerator,
            expected_device=torch.device("cuda:2"),
            expected_mixed_precision="bf16",
        )

    assert caught.value.code == "wave3.accelerator_identity"


def test_marker_publication_occurs_after_cpu_install_inventory_and_before_gpu_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    evidence = probe._empty_evidence(
        requested_device="cuda:0",
        marker_path=Path(plan["artifact_targets"]["attempt_marker"]),
        plan=plan,
    )
    evidence["completed_phases"] = ["plan_revalidated", "shared_gpu_preflight"]
    calls: list[str] = []

    class FakeModel:
        def parameters(self):
            return []

        def buffers(self):
            return []

        def named_parameters(self):
            return []

        def named_buffers(self):
            return []

    model = FakeModel()
    live_config_dict = probe.load_train_config(probe.FROZEN_CONFIG_PATH).config_dict
    resolved = SimpleNamespace(
        fingerprint=probe.FROZEN_CONFIG_FINGERPRINT,
        config_dict=live_config_dict,
        config=SimpleNamespace(
            training=SimpleNamespace(
                precision="bf16", forward_input_provider_mode="synchronous"
            ),
            adapter=object(),
            model=SimpleNamespace(special_token_embeddings=object()),
        ),
    )
    components = SimpleNamespace(
        model=model,
        token_identity=object(),
        base_model_path=Path("/model"),
        base_config_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
    )
    adapter_plan = SimpleNamespace(mode="initialize_new")
    adapter_result = SimpleNamespace(model=model)
    special_result = SimpleNamespace(model=model)
    inventory = _fake_inventory(probe)
    runtime_sources = _fake_runtime_sources(probe)
    load_calls = 0

    def load_config(_path):
        nonlocal load_calls
        load_calls += 1
        calls.append(f"config_loaded_{load_calls}")
        return resolved

    monkeypatch.setattr(probe, "load_train_config", load_config)
    monkeypatch.setattr(
        probe,
        "_assert_current_model_weight_identity",
        lambda _expected: calls.append("weights_rehashed")
        or plan["model_weight_identity"],
    )
    attestation_calls = 0

    def attest(*_args, **_kwargs):
        nonlocal attestation_calls
        attestation_calls += 1
        calls.append(f"config_attested_{attestation_calls}")
        return probe.frozen_runtime_config_attestation()

    monkeypatch.setattr(probe, "attest_runtime_config", attest)
    monkeypatch.setattr(
        probe,
        "_load_frozen_micro_step",
        lambda _path: ({}, SimpleNamespace()),
    )
    monkeypatch.setattr(probe, "_workload_identity", lambda _micro: plan["workload"])

    def load_components(*_args, **_kwargs):
        calls.append("model_cpu")
        return components

    monkeypatch.setattr(probe, "load_qwen_components", load_components)
    monkeypatch.setattr(
        probe,
        "build_adapter_setup_plan",
        lambda *_args, **_kwargs: adapter_plan,
    )

    def setup_adapter(*_args, **_kwargs):
        calls.append("adapter_cpu")
        return adapter_result

    monkeypatch.setattr(probe, "setup_dora_adapter", setup_adapter)
    monkeypatch.setattr(
        probe, "build_default_special_token_selection", lambda *_args: object()
    )

    def install_delta(*_args, **_kwargs):
        calls.append("delta_cpu")
        return special_result

    monkeypatch.setattr(probe, "install_special_token_embedding_deltas", install_delta)
    monkeypatch.setattr(probe, "enable_training_memory_savers", lambda _model: {})

    def concrete(_model):
        calls.append("inventory_589")
        return inventory

    monkeypatch.setattr(probe, "_concrete_trainable_inventory", concrete)
    monkeypatch.setattr(
        probe, "_runtime_source_identities", lambda **_kwargs: runtime_sources
    )
    admission = getattr(probe, "_admit_exact_one_rank_accelerator_environment", None)
    assert callable(admission)
    monkeypatch.setattr(
        probe,
        "_admit_exact_one_rank_accelerator_environment",
        lambda **_kwargs: calls.append("accelerator_admitted"),
    )
    monkeypatch.setattr(
        probe,
        "_assert_gpu_process_subset",
        lambda _device, _baseline, *, stage: _fake_shared_gpu_observation(
            probe, stage=stage
        ),
    )
    original_publish = probe.publish_json_absent

    def publish(path, payload, **kwargs):
        calls.append("marker_published")
        return original_publish(path, payload, **kwargs)

    monkeypatch.setattr(probe, "publish_json_absent", publish)

    def gpu_setup(**_kwargs):
        calls.append("gpu_setup")
        raise probe.Wave3ProbeError("stop", code="wave3.test_stop")

    monkeypatch.setattr(probe, "_build_exact_one_rank_accelerator", gpu_setup)
    with pytest.raises(probe.Wave3ProbeError, match="stop"):
        probe._execute_gpu_probe(
            plan,
            device=torch.device("cuda:0"),
            evidence=evidence,
            run_started=probe.time.monotonic(),
            marker_target=Path(plan["artifact_targets"]["attempt_marker"]),
            publication_owner_token_sha256="6" * 64,
            worker_argv=probe._artifact_argv(
                command="_worker",
                plan_path=plan["artifact_targets"]["plan"],
                receipt_path=plan["artifact_targets"]["receipt"],
                attempt_marker_path=plan["artifact_targets"]["attempt_marker"],
                publication_failure_path=plan["artifact_targets"][
                    "publication_failure"
                ],
                device="cuda:0",
            ),
            shared_gpu_baseline=_fake_shared_gpu_baseline(probe),
            emit=lambda _event: None,
        )
    assert calls == [
        "config_loaded_1",
        "config_attested_1",
        "weights_rehashed",
        "model_cpu",
        "adapter_cpu",
        "delta_cpu",
        "inventory_589",
        "config_loaded_2",
        "config_attested_2",
        "accelerator_admitted",
        "marker_published",
        "gpu_setup",
    ]


def test_fresh_pre_marker_config_reload_rejects_post_setup_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    initial = probe.load_train_config(probe.FROZEN_CONFIG_PATH)
    drifted_dict = deepcopy(initial.config_dict)
    drifted_dict["training"]["precision"] = "fp32"
    drifted = SimpleNamespace(
        fingerprint=probe.sha256_json(drifted_dict),
        config_dict=drifted_dict,
        config=SimpleNamespace(
            training=SimpleNamespace(
                precision="fp32", forward_input_provider_mode="synchronous"
            )
        ),
    )
    monkeypatch.setattr(probe, "load_train_config", lambda _path: drifted)

    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._attest_fresh_runtime_config_immediately_before_marker(
            plan,
            initial_resolved_config=initial.config_dict,
        )
    assert caught.value.code == "wave3.config_identity"
    assert not Path(plan["artifact_targets"]["attempt_marker"]).exists()


def test_fresh_config_reload_is_after_inventory_and_before_shared_marker_gpu() -> None:
    probe = _load_probe_module()
    source = inspect.getsource(probe._execute_gpu_probe)
    inventory = source.index("inventory = _concrete_trainable_inventory(model)")
    fresh = source.index("_attest_fresh_runtime_config_immediately_before_marker")
    admission = source.index("_admit_exact_one_rank_accelerator_environment")
    shared = source.index("pre_marker_observation = _assert_gpu_process_subset")
    marker = source.index("publish_json_absent(marker_target")
    accelerator = source.index("_build_exact_one_rank_accelerator")
    assert inventory < fresh < admission < shared < marker < accelerator


def test_marker_postlink_fsync_failure_preserves_typed_consumed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    _valid_passed_receipt(probe, plan)
    original_marker = json.loads(
        Path(plan["artifact_targets"]["attempt_marker"]).read_text(encoding="utf-8")
    )
    target = tmp_path / "attempt-postlink.json"
    linked: list[str] = []
    original_fsync = parity_module.os.fsync
    calls = 0

    def fail_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected marker directory fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(parity_module.os, "fsync", fail_directory_fsync)
    with pytest.raises(probe.Wave3ArtifactPublicationError) as caught:
        probe.publish_json_absent(
            target,
            original_marker,
            on_linked=lambda: linked.append("attempt_started"),
        )
    assert caught.value.linked_by_this_call is True
    assert caught.value.reloaded_exact is True
    assert linked == ["attempt_started"]
    persisted = probe._load_json(target)
    assert probe.validate_marker(persisted, expected_plan=plan) == original_marker


def test_marker_recovery_never_claims_a_foreign_collision(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    _valid_passed_receipt(probe, plan)
    marker_path = Path(plan["artifact_targets"]["attempt_marker"])
    evidence = probe._empty_evidence(
        requested_device="cuda:0", marker_path=marker_path, plan=plan
    )
    evidence["completed_phases"] = ["plan_revalidated", "shared_gpu_preflight"]
    probe._consume_marker_if_present(
        marker_path,
        plan=plan,
        evidence=evidence,
        publication_owner_token_sha256="7" * 64,
    )
    assert evidence["attempt_marker"]["status"] == "not_published"
    assert evidence["completed_phases"] == [
        "plan_revalidated",
        "shared_gpu_preflight",
    ]
    probe._consume_marker_if_present(
        marker_path,
        plan=plan,
        evidence=evidence,
        publication_owner_token_sha256="6" * 64,
    )
    assert evidence["attempt_marker"]["status"] == "published"
    assert evidence["completed_phases"] == list(probe.PASSED_PHASE_ORDER[:4])


def test_named_buffer_snapshots_bind_shape_dtype_finite_hash_and_restoration() -> None:
    probe = _load_probe_module()
    model = torch.nn.Module()
    model.register_buffer("state", torch.tensor([1.0, 2.0]))
    initial = probe._snapshot_named_buffers(model)
    with torch.no_grad():
        model.state.add_(1.0)
    changed = probe._snapshot_named_buffers(model)
    with torch.no_grad():
        model.state.copy_(torch.tensor([1.0, 2.0]))
    restored = probe._snapshot_named_buffers(model)
    transition = probe._buffer_transition(
        before=initial, after=changed, restored=restored, initial=initial
    )
    assert transition["after_matches_before"] is False
    assert transition["restoration_matches_initial"] is True
    broken = deepcopy(restored)
    broken["rows"][0]["finite"] = False
    broken["inventory_sha256"] = probe.sha256_json(
        {"count": broken["count"], "rows": broken["rows"]}
    )
    with pytest.raises(probe.Wave3ProbeError):
        probe._validate_buffer_inventory(broken)


def test_external_watchdog_terminates_hung_arm_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    state = {
        "status": "running",
        "worker_pid": process.pid,
        "termination": None,
        "max_host_rss_bytes": 0,
        "max_gpu_memory_bytes": 0,
        "current_arm": "zero_reference",
        "current_arm_started": probe.time.monotonic() - 1.0,
        "terminal_receipt": None,
    }
    monkeypatch.setattr(probe, "ARM_WALL_CEILING_SECONDS", 0.01)
    monkeypatch.setattr(probe, "_process_tree_rss_bytes", lambda _pid: 0)
    monkeypatch.setattr(probe, "_gpu_memory_used_bytes", lambda _device: 0)
    try:
        with pytest.raises(probe.Wave3ProbeError) as caught:
            probe._watch_worker(
                process,
                device=torch.device("cuda:0"),
                run_started=probe.time.monotonic(),
                evidence={},
                controller_state=state,
            )
        assert caught.value.code == "wave3.arm_wall"
        probe._terminate_worker(
            process, controller_state=state, reason=caught.value.code
        )
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()


def test_process_tree_rss_skips_child_that_exits_after_status_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    status_reads = {41002: 0}

    def read_proc_text(path: Path, *, encoding: str) -> str:
        assert encoding == "utf-8"
        proc_path = str(path)
        if proc_path == "/proc/41001/status":
            return "State:\tS (sleeping)\nVmRSS:\t11 kB\n"
        if proc_path == "/proc/41001/task/41001/children":
            return "41002\n"
        if proc_path == "/proc/41002/status":
            status_reads[41002] += 1
            if status_reads[41002] == 1:
                return "State:\tR (running)\n"
            raise FileNotFoundError(proc_path)
        if proc_path == "/proc/41002/task/41002/children":
            return ""
        raise AssertionError(f"unexpected proc path: {proc_path}")

    monkeypatch.setattr(probe.Path, "read_text", read_proc_text)

    assert probe._process_tree_rss_bytes(41001) == 11 * 1024
    assert status_reads[41002] == 2


@pytest.mark.parametrize("terminal_state", ["Z", "X", "x"])
def test_process_tree_rss_skips_rechecked_terminal_state_without_vmrss(
    monkeypatch: pytest.MonkeyPatch,
    terminal_state: str,
) -> None:
    probe = _load_probe_module()
    zombie_status_reads = 0

    def read_proc_text(path: Path, *, encoding: str) -> str:
        nonlocal zombie_status_reads
        assert encoding == "utf-8"
        proc_path = str(path)
        if proc_path == "/proc/42001/status":
            return "State:\tS (sleeping)\nVmRSS:\t13 kB\n"
        if proc_path == "/proc/42001/task/42001/children":
            return "42002\n"
        if proc_path == "/proc/42002/status":
            zombie_status_reads += 1
            return f"State:\t{terminal_state} (terminal)\n"
        if proc_path == "/proc/42002/task/42002/children":
            return ""
        raise AssertionError(f"unexpected proc path: {proc_path}")

    monkeypatch.setattr(probe.Path, "read_text", read_proc_text)

    assert probe._process_tree_rss_bytes(42001) == 13 * 1024
    assert zombie_status_reads == 2


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        ("State:\tS (sleeping)\n", "lacks VmRSS"),
        ("State:\tS (sleeping)\nVmRSS:\tnot-a-number kB\n", "malformed"),
    ],
)
def test_process_tree_rss_still_rejects_live_missing_or_malformed_vmrss(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    failure: str,
) -> None:
    probe = _load_probe_module()

    def read_proc_text(path: Path, *, encoding: str) -> str:
        assert encoding == "utf-8"
        if str(path) == "/proc/43001/status":
            return status
        if str(path) == "/proc/43001/task/43001/children":
            return ""
        raise AssertionError(f"unexpected proc path: {path}")

    monkeypatch.setattr(probe.Path, "read_text", read_proc_text)

    with pytest.raises(probe.Wave3ProbeError, match=failure) as caught:
        probe._process_tree_rss_bytes(43001)
    assert caught.value.code == "wave3.host_watchdog"


@pytest.mark.parametrize("blocked_leaf", ["status", "children"])
@pytest.mark.parametrize("blocked_error", [PermissionError, OSError])
def test_process_tree_rss_still_rejects_proc_access_errors(
    monkeypatch: pytest.MonkeyPatch,
    blocked_leaf: str,
    blocked_error: type[OSError],
) -> None:
    probe = _load_probe_module()

    def read_proc_text(path: Path, *, encoding: str) -> str:
        assert encoding == "utf-8"
        proc_path = str(path)
        if proc_path.endswith(f"/{blocked_leaf}"):
            raise blocked_error(proc_path)
        if proc_path == "/proc/44001/status":
            return "State:\tS (sleeping)\nVmRSS:\t17 kB\n"
        if proc_path == "/proc/44001/task/44001/children":
            return ""
        raise AssertionError(f"unexpected proc path: {proc_path}")

    monkeypatch.setattr(probe.Path, "read_text", read_proc_text)

    with pytest.raises(probe.Wave3ProbeError, match="unavailable") as caught:
        probe._process_tree_rss_bytes(44001)
    assert caught.value.code == "wave3.host_watchdog"


def test_process_tree_rss_survives_disposable_cpu_child_churn() -> None:
    probe = _load_probe_module()
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys,time\n"
                "deadline=time.monotonic()+2.0\n"
                "children=[]\n"
                "while time.monotonic()<deadline:\n"
                " children.append(subprocess.Popen([sys.executable,'-c','pass']))\n"
                " children=[child for child in children if child.poll() is None]\n"
                " time.sleep(0.001)\n"
                "for child in children: child.wait()\n"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    samples = 0
    try:
        while worker.poll() is None:
            assert probe._process_tree_rss_bytes(worker.pid) >= 0
            samples += 1
    finally:
        if worker.poll() is None:
            worker.terminate()
        worker.wait(timeout=10)
    assert samples > 0


def test_internal_worker_rejects_direct_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    monkeypatch.delenv(probe.CONTROLLER_PID_ENV, raising=False)
    monkeypatch.delenv(probe.CONTROLLER_TOKEN_ENV, raising=False)
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._require_controller_parent()
    assert caught.value.code == "wave3.controller_parent"


@pytest.mark.parametrize(
    ("rss", "gpu", "code"),
    [
        (64 * 1024**3 + 1, 0, "wave3.host_rss"),
        (0, 76 * 1024**3 + 1, "wave3.device_memory"),
    ],
)
def test_external_watchdog_enforces_host_and_gpu_memory(
    monkeypatch: pytest.MonkeyPatch, rss: int, gpu: int, code: str
) -> None:
    probe = _load_probe_module()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    state = {
        "status": "running",
        "worker_pid": process.pid,
        "termination": None,
        "max_host_rss_bytes": 0,
        "max_gpu_memory_bytes": 0,
        "current_arm": None,
        "current_arm_started": None,
        "terminal_receipt": None,
    }
    monkeypatch.setattr(probe, "_process_tree_rss_bytes", lambda _pid: rss)
    monkeypatch.setattr(probe, "_gpu_memory_used_bytes", lambda _device: gpu)
    try:
        with pytest.raises(probe.Wave3ProbeError) as caught:
            probe._watch_worker(
                process,
                device=torch.device("cuda:0"),
                run_started=probe.time.monotonic(),
                evidence={},
                controller_state=state,
            )
        assert caught.value.code == code
    finally:
        probe._terminate_worker(process, controller_state=state, reason=code)


def test_receipt_publication_failure_emits_one_immutable_sidecar(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    receipt = _valid_passed_receipt(probe, plan)
    receipt_target = Path(plan["artifact_targets"]["receipt"])
    sidecar_target = Path(plan["artifact_targets"]["publication_failure"])
    receipt_target.write_text("occupied", encoding="utf-8")
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._publish_terminal_receipt(
            receipt_target=receipt_target,
            publication_failure_target=sidecar_target,
            receipt=receipt,
            plan=plan,
        )
    assert caught.value.code == "wave3.receipt_publication"
    sidecar = json.loads(sidecar_target.read_text(encoding="utf-8"))
    assert probe.validate_publication_failure(sidecar, expected_plan=plan) == sidecar
    original = sidecar_target.read_bytes()
    with pytest.raises(probe.Wave3ProbeError):
        probe._publish_terminal_receipt(
            receipt_target=receipt_target,
            publication_failure_target=sidecar_target,
            receipt=receipt,
            plan=plan,
        )
    assert sidecar_target.read_bytes() == original


def test_terminal_receipt_postlink_fsync_failure_recovers_only_its_own_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    receipt = _valid_passed_receipt(probe, plan)
    receipt_target = Path(plan["artifact_targets"]["receipt"])
    sidecar_target = Path(plan["artifact_targets"]["publication_failure"])
    original_fsync = parity_module.os.fsync
    calls = 0

    def fail_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected receipt directory fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(parity_module.os, "fsync", fail_directory_fsync)
    status = probe._publish_terminal_receipt(
        receipt_target=receipt_target,
        publication_failure_target=sidecar_target,
        receipt=receipt,
        plan=plan,
    )
    assert status["receipt_linked_and_reloaded"] is True
    assert status["post_link_recovery"] is True
    assert status["publication_warning"]["type"] == "OSError"
    assert (
        probe.validate_receipt(probe._load_json(receipt_target), expected_plan=plan)
        == receipt
    )
    assert not sidecar_target.exists()


def test_publication_sidecar_postlink_cleanup_failure_is_recovered_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    receipt = _valid_passed_receipt(probe, plan)
    receipt_target = Path(plan["artifact_targets"]["receipt"])
    sidecar_target = Path(plan["artifact_targets"]["publication_failure"])
    receipt_target.write_text("occupied", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_sidecar_cleanup(path, *args, **kwargs):
        if path.parent == tmp_path and path.name.startswith(
            ".publication-failure.json."
        ):
            raise OSError("injected sidecar cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_sidecar_cleanup)
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._publish_terminal_receipt(
            receipt_target=receipt_target,
            publication_failure_target=sidecar_target,
            receipt=receipt,
            plan=plan,
        )
    assert caught.value.code == "wave3.receipt_publication"
    sidecar = probe._load_json(sidecar_target)
    assert probe.validate_publication_failure(sidecar, expected_plan=plan) == sidecar


def test_sidecar_publication_failure_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    receipt = _valid_passed_receipt(probe, plan)
    calls: list[Path] = []

    def fail(path, _payload):
        calls.append(Path(path))
        raise probe.Wave3ProbeError("write failed", code="wave3.test_write")

    monkeypatch.setattr(probe, "publish_json_absent", fail)
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._publish_terminal_receipt(
            receipt_target=Path(plan["artifact_targets"]["receipt"]),
            publication_failure_target=Path(
                plan["artifact_targets"]["publication_failure"]
            ),
            receipt=receipt,
            plan=plan,
        )
    assert caught.value.code == "wave3.test_write"
    assert calls == [
        Path(plan["artifact_targets"]["receipt"]),
        Path(plan["artifact_targets"]["publication_failure"]),
    ]


def test_valid_v4_cpu_run_preserves_failure_receipt_before_cache_model_or_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    plan_path = Path(plan["artifact_targets"]["plan"])
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        probe,
        "build_plan",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cache must not load")),
    )
    monkeypatch.setattr(
        probe,
        "_execute_gpu_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model must not load")
        ),
    )
    args = argparse.Namespace(
        plan=plan_path,
        receipt=Path(plan["artifact_targets"]["receipt"]),
        attempt_marker=Path(plan["artifact_targets"]["attempt_marker"]),
        publication_failure=Path(plan["artifact_targets"]["publication_failure"]),
        device="cuda:0",
    )
    assert probe.run_command(args) == 1
    assert not args.attempt_marker.exists()
    failure = json.loads(args.receipt.read_text(encoding="utf-8"))
    assert (
        probe.validate_receipt(failure, expected_plan=plan)["terminal_reason"]["code"]
        == "wave3.device_preflight"
    )
    assert failure["plan_sha256"] == plan["plan_sha256"]
    assert failure["counts"] == {"model_forwards": 0, "backwards": 0}


def test_shared_gpu_gate_binds_visible_device_and_rejects_over_budget_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    monotonic_values = iter((1_000_000_000, 3_000_000_000))
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: next(monotonic_values))
    calls: list[list[str]] = []

    def idle_run(argv, **_kwargs):
        calls.append(argv)
        return _shared_gpu_completed_process_query(
            argv,
            gpu_row="7, GPU-id, 81920, 100, 0",
            process_rows="",
        )

    monkeypatch.setattr(probe.subprocess, "run", idle_run)
    receipt = probe._capture_shared_gpu_baseline(torch.device("cuda:0"))
    assert len(receipt["samples"]) == 2
    assert all(argv[-1] == "7" for argv in calls)

    monotonic_values = iter((4_000_000_000, 6_000_000_000))
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda argv, **_kwargs: _shared_gpu_completed_process_query(
            argv,
            gpu_row="7, GPU-id, 81920, 49153, 0",
            process_rows="",
        ),
    )
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._capture_shared_gpu_baseline(torch.device("cuda:0"))
    assert caught.value.code == "wave3.gpu_busy"


def test_total_deadline_and_arm_order_are_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(
        probe.time,
        "monotonic",
        lambda: float(probe.RUN_WALL_CEILING_SECONDS + 1),
    )
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._assert_deadline(0.0)
    assert caught.value.code == "wave3.run_wall"
    assert probe.ARM_ORDER == (
        "zero_reference",
        "zero_optimized",
        "nonzero_optimized",
        "nonzero_reference",
    )


def test_probe_source_has_no_retry_or_cache_publication_route() -> None:
    probe = _load_probe_module()
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "for retry" not in source
    assert "while retry" not in source
    assert "prepare_training_pack_caches" not in source
    assert "publish_cache" not in source
    assert "_load_validated_chunk" in source


def _shared_gpu_completed_process_query(
    argv: list[str],
    *,
    gpu_row: str = "7, GPU-shared, 81920, 42000, 97",
    process_rows: str = "GPU-shared, 7001, 21000\nGPU-shared, 7002, 21000\n",
) -> subprocess.CompletedProcess[str]:
    if any("--query-gpu=" in item for item in argv):
        return subprocess.CompletedProcess(argv, 0, f"{gpu_row}\n", "")
    if any("--query-compute-apps=" in item for item in argv):
        return subprocess.CompletedProcess(argv, 0, process_rows, "")
    raise AssertionError(f"unexpected command: {argv}")


def test_shared_gpu_baseline_binds_two_stable_samples_with_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: admitting one sample or ignoring stable driver PID identity."""

    probe = _load_probe_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    sleeps: list[float] = []
    monotonic_values = iter((1_000_000_000, 3_000_000_000))
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: next(monotonic_values))
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda argv, **_kwargs: _shared_gpu_completed_process_query(argv),
    )

    baseline = probe._capture_shared_gpu_baseline(torch.device("cuda:0"))

    assert sleeps == [2.0]
    assert baseline["schema"] == "coordexp-swift-wave3-shared-gpu-baseline-v1"
    assert baseline["gpu_identity"] == {
        "physical_index": 7,
        "uuid": "GPU-shared",
        "total_memory_mib": 81920,
    }
    assert baseline["limits"] == {
        "sample_count": 2,
        "minimum_sample_interval_seconds": 2.0,
        "max_preexisting_memory_mib": 49152,
        "required_headroom_mib": 32768,
    }
    assert (
        baseline["utilization_disposition"]
        == "observational_only_not_admission_or_promotion"
    )
    assert baseline["baseline_compute_processes"] == [
        {"gpu_uuid": "GPU-shared", "driver_pid": 7001},
        {"gpu_uuid": "GPU-shared", "driver_pid": 7002},
    ]
    assert [row["memory_used_mib"] for row in baseline["samples"]] == [42000, 42000]
    assert [row["headroom_mib"] for row in baseline["samples"]] == [39920, 39920]
    assert [row["utilization_percent"] for row in baseline["samples"]] == [97, 97]
    assert probe._validate_shared_gpu_baseline(baseline) == baseline


@pytest.mark.parametrize(
    ("second_gpu_row", "second_process_rows", "expected_code"),
    [
        ("7, GPU-shared, 81920, 49153, 0", "GPU-shared, 7001, 1\n", "wave3.gpu_busy"),
        (
            "7, GPU-shared, 81920, 42000, 0",
            "GPU-shared, 7002, 1\n",
            "wave3.gpu_baseline",
        ),
        ("7, GPU-other, 81920, 42000, 0", "GPU-other, 7001, 1\n", "wave3.gpu_baseline"),
    ],
)
def test_shared_gpu_baseline_rejects_memory_or_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    second_gpu_row: str,
    second_process_rows: str,
    expected_code: str,
) -> None:
    """Break caught: admitting over-budget or changing shared GPU ownership."""

    probe = _load_probe_module()
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    monotonic_values = iter((1_000_000_000, 3_000_000_000))
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: next(monotonic_values))
    calls = 0

    def run(argv, **_kwargs):
        nonlocal calls
        sample = calls // 2
        calls += 1
        if sample == 0:
            return _shared_gpu_completed_process_query(
                argv,
                gpu_row="7, GPU-shared, 81920, 42000, 0",
                process_rows="GPU-shared, 7001, 1\n",
            )
        return _shared_gpu_completed_process_query(
            argv, gpu_row=second_gpu_row, process_rows=second_process_rows
        )

    monkeypatch.setattr(probe.subprocess, "run", run)
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._capture_shared_gpu_baseline(torch.device("cuda:7"))
    assert caught.value.code == expected_code


def test_post_probe_gpu_process_gate_allows_baseline_exit_but_rejects_new_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: allowing an owned/orphan GPU row to survive preterminal."""

    probe = _load_probe_module()
    baseline = {
        "schema": "coordexp-swift-wave3-shared-gpu-baseline-v1",
        "mode": "shared_preexisting_compute",
        "requested_device": "cuda:0",
        "cuda_visible_devices": "7",
        "selector": "7",
        "gpu_identity": {
            "physical_index": 7,
            "uuid": "GPU-shared",
            "total_memory_mib": 81920,
        },
        "limits": {
            "sample_count": 2,
            "minimum_sample_interval_seconds": 2.0,
            "max_preexisting_memory_mib": 49152,
            "required_headroom_mib": 32768,
        },
        "utilization_disposition": ("observational_only_not_admission_or_promotion"),
        "baseline_compute_processes": [
            {"gpu_uuid": "GPU-shared", "driver_pid": 7001},
            {"gpu_uuid": "GPU-shared", "driver_pid": 7002},
        ],
        "samples": [
            {
                "sample_index": index,
                "monotonic_ns": 1_000_000_000 + index * 2_000_000_000,
                "memory_used_mib": 42000,
                "headroom_mib": 39920,
                "utilization_percent": 99,
                "compute_processes": [
                    {"gpu_uuid": "GPU-shared", "driver_pid": 7001},
                    {"gpu_uuid": "GPU-shared", "driver_pid": 7002},
                ],
            }
            for index in range(2)
        ],
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda argv, **_kwargs: _shared_gpu_completed_process_query(
            argv,
            gpu_row="7, GPU-shared, 81920, 21000, 88",
            process_rows="GPU-shared, 7002, 21000\n",
        ),
    )
    observation = probe._assert_gpu_process_subset(
        torch.device("cuda:0"), baseline, stage="post_probe_preterminal"
    )
    assert observation["compute_processes"] == [
        {"gpu_uuid": "GPU-shared", "driver_pid": 7002}
    ]
    assert observation["missing_baseline_processes"] == [
        {"gpu_uuid": "GPU-shared", "driver_pid": 7001}
    ]

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda argv, **_kwargs: _shared_gpu_completed_process_query(
            argv,
            gpu_row="7, GPU-shared, 81920, 22000, 88",
            process_rows="GPU-shared, 7002, 21000\nGPU-shared, 9009, 1000\n",
        ),
    )
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe._assert_gpu_process_subset(
            torch.device("cuda:0"), baseline, stage="post_probe_preterminal"
        )
    assert caught.value.code == "wave3.gpu_new_process"


def test_post_probe_preterminal_gate_requires_two_subset_sweeps_two_seconds_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: publishing terminal evidence from one instantaneous postcheck."""

    probe = _load_probe_module()
    baseline = _fake_shared_gpu_baseline(probe)
    observations = iter(
        (
            {
                **_fake_shared_gpu_observation(probe, stage="post_probe_preterminal"),
                "monotonic_ns": 10_000_000_000,
            },
            {
                **_fake_shared_gpu_observation(probe, stage="post_probe_preterminal"),
                "monotonic_ns": 12_000_000_000,
            },
        )
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        probe,
        "_assert_gpu_process_subset",
        lambda *_args, **_kwargs: next(observations),
    )
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: sleeps.append(seconds))

    sweep = probe._capture_gpu_process_subset_sweep(
        torch.device("cuda:0"), baseline, stage="post_probe_preterminal"
    )

    assert sleeps == [2.0]
    assert sweep["schema"] == "coordexp-swift-wave3-shared-gpu-subset-sweep-v2"
    assert sweep["sample_count"] == 2
    assert sweep["minimum_sample_interval_seconds"] == 2.0
    assert sweep["status"] == "passed"
    assert sweep["terminal_reason"] is None
    assert [row["monotonic_ns"] for row in sweep["samples"]] == [
        10_000_000_000,
        12_000_000_000,
    ]
    assert (
        probe._validate_gpu_process_subset_sweep(
            sweep, baseline=baseline, expected_stage="post_probe_preterminal"
        )
        == sweep
    )


def test_failed_postcheck_still_records_two_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: first postcheck error aborts without the mandatory second attempt."""

    probe = _load_probe_module()
    attempts = 0
    timestamps = iter((20_000_000_000, 22_000_000_000))
    sleeps: list[float] = []

    def fail(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise probe.Wave3ProbeError("injected query failure", code="wave3.test_post")

    monkeypatch.setattr(probe, "_assert_gpu_process_subset", fail)
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: next(timestamps))
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: sleeps.append(seconds))
    sweep = probe._capture_gpu_process_subset_sweep(
        torch.device("cuda:0"),
        _fake_shared_gpu_baseline(probe),
        stage="post_probe_preterminal",
    )
    assert attempts == 2
    assert sleeps == [2.0]
    assert sweep["status"] == "failed"
    assert len(sweep["samples"]) == 2
    assert all(row["status"] == "failed" for row in sweep["samples"])
    assert sweep["terminal_reason"]["code"] == "wave3.test_post"


def test_active_schema_and_efficiency_claim_are_shared_load_nonpromotional(
    tmp_path: Path,
) -> None:
    """Break caught: executing v3 or promoting shared-load resource deltas."""

    probe = _load_probe_module()
    assert probe.PLAN_SCHEMA == "coordexp-swift-wave3-zero-weight-plan-v4"
    assert probe.MARKER_SCHEMA.endswith("-v4")
    assert probe.RECEIPT_SCHEMA.endswith("-v4")
    assert probe.shared_gpu_claim_boundary() == {
        "admissible": ["correctness", "plumbing", "numerical_equivalence"],
        "nonpromotional": [
            "timing",
            "resource_usage",
            "efficiency",
            "zero_weight_performance",
        ],
        "promotion_allowed": False,
        "reason": "shared_preexisting_gpu_load",
    }
    efficiency = probe._efficiency_from_artifacts(
        {
            "wall_seconds": 2.0,
            "resources_after": {
                "gpu": {
                    "max_memory_allocated_bytes": 10,
                    "max_memory_reserved_bytes": 20,
                },
                "cpu": {"max_rss_bytes": 30},
            },
        },
        {
            "wall_seconds": 1.0,
            "resources_after": {
                "gpu": {
                    "max_memory_allocated_bytes": 5,
                    "max_memory_reserved_bytes": 10,
                },
                "cpu": {"max_rss_bytes": 15},
            },
        },
    )
    assert efficiency["promotion_claim"] == "nonpromotional_shared_load_observation"
    historical = json.loads(
        (
            Path(probe.__file__).resolve().parents[3]
            / "outputs/probes/coordexp_swift/wave3_zero_weight/"
            "2026-08-10-r3-v3/plan.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(probe.Wave3ProbeError) as caught:
        probe.validate_plan(historical)
    assert caught.value.code == "wave3.schema"


def test_process_cleanup_terminates_nested_session_and_verifies_total_absence() -> None:
    """Break caught: leader exit or killpg leaves a nested setsid child alive."""

    probe = _load_probe_module()
    child_code = "import time; time.sleep(60)"
    leader_code = (
        "import subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
        "print(p.pid,flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline().strip())
    state = {
        "owned_process_identities": {},
        "owned_session_ids": [],
        "termination": None,
    }
    try:
        probe._record_owned_process_tree(process.pid, state)
        cleanup = probe._cleanup_worker_process_tree(
            process,
            controller_state=state,
            reason="test_nested_session",
        )
        assert cleanup["status"] == "passed"
        assert cleanup["leader_absent"] is True
        assert cleanup["descendants_absent"] is True
        assert cleanup["sessions_absent"] is True
        assert cleanup["surviving_processes"] == []
        assert process.poll() is not None
        assert not Path(f"/proc/{child_pid}").exists()
        assert any(
            row["pid"] == child_pid and row["session_id"] == child_pid
            for row in cleanup["tracked_processes"]
        )
    finally:
        for pid in (child_pid, process.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_failure_receipt_retains_cleanup_and_post_sweep_failures(
    tmp_path: Path,
) -> None:
    """Break caught: failed cleanup/sweep is omitted or presented as passed."""

    probe = _load_probe_module()
    plan = _valid_plan(probe, tmp_path)
    passed = _valid_passed_receipt(probe, plan)

    cleanup_failed = deepcopy(passed)
    cleanup_failed["status"] = "failed"
    cleanup_failed["terminal_reason"] = {
        "code": "wave3.process_cleanup_incomplete",
        "type": "Wave3ProbeError",
        "message": "injected cleanup failure",
    }
    cleanup_failed["runtime"]["process_cleanup"] = _fake_process_cleanup(
        probe, status="failed"
    )
    cleanup_failed = _refinalize(probe, cleanup_failed, "receipt_sha256")
    assert (
        probe.validate_receipt(cleanup_failed, expected_plan=plan)["runtime"][
            "process_cleanup"
        ]["status"]
        == "failed"
    )

    cannot_claim_passed = deepcopy(cleanup_failed)
    cannot_claim_passed["status"] = "passed"
    cannot_claim_passed["terminal_reason"] = None
    cannot_claim_passed = _refinalize(probe, cannot_claim_passed, "receipt_sha256")
    with pytest.raises(probe.Wave3ProbeError):
        probe.validate_receipt(cannot_claim_passed, expected_plan=plan)

    sweep_failed = deepcopy(passed)
    sweep_failed["status"] = "failed"
    sweep_failed["terminal_reason"] = {
        "code": "wave3.gpu_postcheck",
        "type": "Wave3ProbeError",
        "message": "injected post sweep failure",
    }
    failed_sweep = sweep_failed["runtime"]["shared_gpu_contract"][
        "post_probe_preterminal_observation"
    ]
    failed_error = {
        "code": "wave3.gpu_new_process",
        "type": "Wave3ProbeError",
        "message": "injected new GPU process",
    }
    failed_sweep["samples"][0]["status"] = "failed"
    failed_sweep["samples"][0]["error"] = failed_error
    failed_sweep["status"] = "failed"
    failed_sweep["terminal_reason"] = failed_error
    sweep_failed["runtime"]["shared_gpu_contract"]["status"] = "post_probe_failed"
    sweep_failed = _refinalize(probe, sweep_failed, "receipt_sha256")
    validated = probe.validate_receipt(sweep_failed, expected_plan=plan)
    assert validated["runtime"]["shared_gpu_contract"]["status"] == (
        "post_probe_failed"
    )


@pytest.mark.parametrize(
    "reason",
    ["success", "wave3.arm_wall", "wave3.host_rss", "worker_failed"],
)
def test_every_spawned_worker_path_cleans_before_two_sample_post_sweep(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """Break caught: success or failure path skips cleanup or reverses ordering."""

    probe = _load_probe_module()
    calls: list[str] = []
    cleanup = _fake_process_cleanup(probe)
    sweep = _fake_shared_gpu_sweep(probe)
    monkeypatch.setattr(
        probe,
        "_cleanup_worker_process_tree",
        lambda *_args, **_kwargs: calls.append("cleanup") or cleanup,
    )
    monkeypatch.setattr(
        probe,
        "_capture_gpu_process_subset_sweep",
        lambda *_args, **_kwargs: calls.append("post_sweep") or sweep,
    )
    state: dict[str, object] = {}
    probe._finalize_spawned_worker(
        SimpleNamespace(pid=1),
        device=torch.device("cuda:0"),
        shared_gpu_baseline=_fake_shared_gpu_baseline(probe),
        controller_state=state,
        reason=reason,
    )
    assert calls == ["cleanup", "post_sweep"]
    assert state["process_cleanup"] == cleanup
    assert state["post_probe_preterminal_observation"] == sweep


def test_cleanup_exception_is_retained_and_does_not_skip_post_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    calls: list[str] = []
    monkeypatch.setattr(
        probe,
        "_cleanup_worker_process_tree",
        lambda *_args, **_kwargs: calls.append("cleanup")
        or (_ for _ in ()).throw(
            probe.Wave3ProbeError("injected cleanup", code="wave3.test_cleanup")
        ),
    )
    monkeypatch.setattr(
        probe,
        "_capture_gpu_process_subset_sweep",
        lambda *_args, **_kwargs: calls.append("post_sweep")
        or _fake_shared_gpu_sweep(probe),
    )
    state: dict[str, object] = {"owned_process_identities": {}}
    probe._finalize_spawned_worker(
        SimpleNamespace(pid=123),
        device=torch.device("cuda:0"),
        shared_gpu_baseline=_fake_shared_gpu_baseline(probe),
        controller_state=state,
        reason="worker_failed",
    )
    assert calls == ["cleanup", "post_sweep"]
    assert state["process_cleanup"]["status"] == "failed"
    assert state["process_cleanup"]["terminal_reason"]["code"] == ("wave3.test_cleanup")
