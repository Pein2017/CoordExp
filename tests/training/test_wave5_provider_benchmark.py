from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import pytest
import torch
from safetensors.torch import save_file

from src.config.loader import load_train_config
from src.training import pack_cache
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
)
from src.training.supervised_trainer import SupervisedMicroStep


def _load_probe_module():
    name = "coordexp_wave5_provider_benchmark_test_module"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/probes/coordexp_swift/wave5_provider_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_probe_module()
    from src.qwen.parity import base_model_weight_identity

    model_root = tmp_path / "tiny-model"
    model_root.mkdir()
    save_file({"weight": torch.ones(1)}, str(model_root / "model.safetensors"))
    identity = base_model_weight_identity(model_root)
    identity["root"] = str(
        Path(
            load_train_config(module.BASE_CONFIG_PATH).config.model.base_model
        ).resolve()
    )
    identity["aggregate_sha256"] = module.sha256_json(
        {key: value for key, value in identity.items() if key != "aggregate_sha256"}
    )
    monkeypatch.setattr(
        module, "_current_base_model_weight_identity", lambda: deepcopy(identity)
    )
    historical_root = tmp_path / "historical-w0-run"
    historical_root.mkdir()
    historical_config = deepcopy(module._five_step_reference_config())
    historical_config["run"]["name"] = "wave0_compatibility_reference_5step"
    historical_config["run"]["artifact_root"] = str(tmp_path / "historical-output")
    historical_config["training"].pop("forward_input_provider_mode")
    for field in (
        "policy",
        "window_size",
        "lookahead",
        "seed",
        "worker_count",
        "fragment_item_budget",
        "fragment_byte_budget",
        "cursor_byte_budget",
        "max_packs_per_fragment",
    ):
        historical_config["packing"].pop(field)
    historical_config.pop("resume")
    historical_fingerprint = module.sha256_json(historical_config)
    resolved_path = historical_root / "resolved_config.json"
    _write_json(
        resolved_path,
        {
            "config": historical_config,
            "resolution": {
                "fingerprint": historical_fingerprint,
                "entry_config_path": str(tmp_path / "historical-w0.yaml"),
                "loader_version": "coordexp-swift-config-v1",
                "schema_version": 1,
                "sources": [],
                "path_origins": {},
            },
        },
    )
    run_path = historical_root / "run.json"
    _write_json(
        run_path,
        {
            "status": "completed",
            "completed_steps": 5,
            "checkpoint_event_count": 1,
            "config_fingerprint": historical_fingerprint,
            "forward_input_provider_mode": "synchronous",
            "measurement": {
                "context": {
                    "warmup_exclusion_steps": 2,
                    "profile_sync_timings": {"enabled": False, "source": "default"},
                }
            },
            "policy_identities": {
                "eval_reduction": {
                    "schema_version": 1,
                    "mode": "disjoint_shard",
                    "source": "default",
                }
            },
        },
    )
    monkeypatch.setattr(module, "HISTORICAL_W0_RESOLVED_CONFIG_PATH", resolved_path)
    monkeypatch.setattr(
        module,
        "HISTORICAL_W0_RESOLVED_CONFIG_FILE_SHA256",
        module.sha256_file(resolved_path),
    )
    monkeypatch.setattr(module, "HISTORICAL_W0_RUN_RECEIPT_PATH", run_path)
    monkeypatch.setattr(
        module,
        "HISTORICAL_W0_RUN_RECEIPT_FILE_SHA256",
        module.sha256_file(run_path),
    )
    historical_receipt = deepcopy(module.HISTORICAL_W0_RECEIPT)
    historical_receipt["derived_five_step_config_fingerprint"] = historical_fingerprint
    monkeypatch.setattr(module, "HISTORICAL_W0_RECEIPT", historical_receipt)
    monkeypatch.setattr(
        module,
        "HISTORICAL_W0_BASE_WEIGHT_AGGREGATE_SHA256",
        identity["aggregate_sha256"],
    )
    monotonic_ns = 1_000_000_000

    def next_monotonic_ns():
        nonlocal monotonic_ns
        current = monotonic_ns
        monotonic_ns += 2_000_000_000
        return current

    monkeypatch.setattr(module.time, "monotonic_ns", next_monotonic_ns)
    weight_receipt_path = historical_root / "weight-identity-receipt.json"
    _write_json(weight_receipt_path, {"model_weight_identity": identity})
    monkeypatch.setattr(
        module, "HISTORICAL_W0_WEIGHT_IDENTITY_RECEIPT_PATH", weight_receipt_path
    )
    monkeypatch.setattr(
        module,
        "HISTORICAL_W0_WEIGHT_IDENTITY_RECEIPT_FILE_SHA256",
        module.sha256_file(weight_receipt_path),
    )
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _bind_authenticated_cpu_provenance(probe, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.artifacts.provenance import pinned_runtime_baseline

    baseline = pinned_runtime_baseline()
    dependencies = deepcopy(baseline["dependencies"])
    dependencies.update(deepcopy(baseline.get("reference_only", {})))

    def collect(*, repository_root):
        assert Path(repository_root) == probe.REPO_ROOT
        return {
            "dependencies": deepcopy(dependencies),
            "runtime": deepcopy(baseline["runtime"]),
        }

    monkeypatch.setattr(probe, "_collect_execution_baseline_provenance", collect)


def _cache_determinants(*, split: str) -> dict:
    from src.packing.planner import (
        PACK_PLAN_SCHEMA,
        PACK_PLAN_SCHEMA_VERSION,
        build_pack_plan_policy_identity,
    )

    semantic = {
        "version": pack_cache.PACKING_CACHE_VERSION,
        "split": split,
        "dataset": {"purpose": "wave5-probe-test"},
        "template": {"purpose": "wave5-probe-test"},
        "packing": {
            "schema": PACK_PLAN_SCHEMA,
            "schema_version": PACK_PLAN_SCHEMA_VERSION,
            "global_max_length": 12_000,
            "policy_identity": build_pack_plan_policy_identity(),
            "fragment_pack_budget": None,
        },
        "processor": {"purpose": "wave5-probe-test"},
        "ordering": {"purpose": "wave5-probe-test"},
        "augmentation": {"split": split, "enabled": False},
        "qwen": {
            "processor_identity": {"purpose": "wave5-probe-test"},
            "token_identity": {"purpose": "wave5-probe-test"},
            "encoding_identity": {"purpose": "wave5-probe-test"},
            "model_config_assets": {"purpose": "wave5-probe-test"},
            "processor_assets": {"purpose": "wave5-probe-test"},
            "tokenizer_assets": {"purpose": "wave5-probe-test"},
        },
        "realized_vocab_groups": {
            "vocab_size": 6,
            "desc_text": [0, 1],
            "schema": [2],
            "coordinate": [3],
            "eos": [4],
            "blocked": [5],
        },
        "micro_step_runtime_config": {
            "fa2_model_dtype": "bf16",
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


def _semantic_micro_step(
    index: int,
    *,
    split: str,
    pack_plan: dict | None = None,
) -> SupervisedMicroStep:
    metadata = {
        "split": split,
        "pack_id": index,
        "example_ids": [f"example-{split}-{index}"],
        "augmentation_receipt": {
            "split": split,
            "mode": "disabled",
            "object_ordering": "source_order",
        },
    }
    if pack_plan is not None:
        metadata["pack_plan"] = deepcopy(pack_plan)
    return SupervisedMicroStep(
        pack=f"pack-{split}-{index}",
        encoded_examples=(f"example-{split}-{index}",),
        position_inputs={"position_ids": (index, index + 1)},
        token_sequence={"labels": (index + 2, index + 3)},
        vocab_groups={"desc_text": (0, 1), "coordinate": (2, 3)},
        metadata=metadata,
        expected_vocab_size=6,
        fa2_model_dtype="bf16",
    )


def _pack_plan_receipt(probe, *, split: str, count: int) -> dict:
    determinant_split = "train" if split == "train" else "eval.forward"
    plan_sha256 = probe.sha256_json(
        {"split": determinant_split, "mode": "complete_plan", "count": count}
    )
    return {
        "schema_version": 1,
        "mode": "complete_plan",
        "policy_identity": _cache_determinants(split=determinant_split)["packing"][
            "policy_identity"
        ],
        "plan_sha256": plan_sha256,
        "fragment_chain_sha256": None,
        "fragment_count": 1,
        "source_input_count": count,
        "emitted_pack_count": count,
        "fragment_sha256": plan_sha256,
    }


def _write_compatibility_caches(probe, tmp_path: Path, monkeypatch=None):
    historical_root = tmp_path / "historical-v2"
    current_root = tmp_path / "current-v3"
    historical_binding = {}
    current_fingerprints = {}
    for split, count in (("train", 32), ("eval", 8)):
        determinant_split = "train" if split == "train" else "eval.forward"
        historical_steps = tuple(
            _semantic_micro_step(index, split=split) for index in range(count)
        )
        pack_plan = _pack_plan_receipt(probe, split=split, count=count)
        current_steps = tuple(
            _semantic_micro_step(index, split=split, pack_plan=pack_plan)
            for index in range(count)
        )
        determinants = _cache_determinants(split=determinant_split)
        fingerprint = pack_cache.packing_cache_fingerprint_from_determinants(
            determinants
        )
        cache_dir = pack_cache.cache_dir_for_fingerprint(current_root, fingerprint)
        pack_cache.write_micro_step_cache(
            cache_dir,
            current_steps,
            cache_root=current_root,
            fingerprint=fingerprint,
            determinants=determinants,
            augmentation={
                "split": determinant_split,
                "mode": "disabled",
                "policy": "geometry_flips",
                "enabled": False,
                "seed": 17,
                "input_example_count": count,
                "output_example_count": count,
                "presentation_count": count,
                "object_ordering": "source_order",
            },
            materialization=pack_cache.build_packing_cache_materialization(),
            determinant_revalidator=lambda determinants=determinants: determinants,
        )
        current_fingerprints[split] = fingerprint

        historical_fingerprint = probe.sha256_json(f"historical-{split}")
        historical_dir = historical_root / historical_fingerprint
        chunk_path = historical_dir / "chunks/chunk-00000.pkl"
        chunk_path.parent.mkdir(parents=True)
        chunk_path.write_bytes(
            pickle.dumps(historical_steps, protocol=pickle.HIGHEST_PROTOCOL)
        )
        manifest = {
            "version": "coordexp-swift-pack-cache-v2",
            "status": "complete",
            "fingerprint": historical_fingerprint,
            "micro_step_count": count,
            "chunk_size": 512,
            "chunks": [
                {
                    "start": 0,
                    "end": count,
                    "count": count,
                    "path": "chunks/chunk-00000.pkl",
                    "sha256": probe.sha256_file(chunk_path),
                }
            ],
            "determinants": {"split": determinant_split},
        }
        manifest_path = historical_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        historical_binding[split] = {
            "fingerprint": historical_fingerprint,
            "manifest_sha256": probe.sha256_file(manifest_path),
            "micro_step_count": count,
            "chunk_sha256": probe.sha256_file(chunk_path),
        }
    if monkeypatch is None:
        probe.HISTORICAL_W0_CACHE_BINDING = historical_binding
    else:
        monkeypatch.setattr(probe, "HISTORICAL_W0_CACHE_BINDING", historical_binding)
    return historical_root, current_root, current_fingerprints


def _compatibility(probe, tmp_path: Path):
    historical_root, cache_root, fingerprints = _write_compatibility_caches(
        probe, tmp_path
    )
    path = tmp_path / "compatibility.json"
    attestation = probe.build_and_publish_compatibility_attestation(
        historical_cache_root=historical_root,
        cache_root=cache_root,
        train_fingerprint=fingerprints["train"],
        eval_fingerprint=fingerprints["eval"],
        output_path=path,
    )
    return cache_root, path, attestation


def _plan(
    probe,
    tmp_path: Path,
    *,
    publish: bool = False,
    gpu_execution_policy: str = "idle_promotional",
):
    root = tmp_path / "wave5"
    root.mkdir(parents=True)
    cache_root, attestation_path, _ = _compatibility(probe, tmp_path)
    plan = probe.build_frozen_plan(
        root=root,
        cache_root=cache_root,
        compatibility_attestation_path=attestation_path,
        gpu_execution_policy=gpu_execution_policy,
    )
    if publish:
        probe.publish_json_absent(plan["artifact_targets"]["plan"], plan)
    return plan


def _gpu_rows(
    probe,
    *,
    memory_used_bytes: int = 0,
    utilization: int = 0,
    memory_total_bytes: int | None = None,
):
    total = (
        probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES
        if memory_total_bytes is None
        else memory_total_bytes
    )
    return [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_total_bytes": total,
            "memory_used_bytes": memory_used_bytes,
            "memory_headroom_bytes": total - memory_used_bytes,
            "utilization_percent": utilization,
        }
        for index in range(probe.WORLD_SIZE)
    ]


def _baseline_apps(probe, *, pid_offset: int = 700_000):
    return [
        {
            "gpu_uuid": f"GPU-{index}",
            "pid": pid_offset + index,
            "process_name": f"preexisting-rank-{index}",
        }
        for index in range(probe.WORLD_SIZE)
    ]


def _gpu_baseline_fixture(probe, plan, *, publish: bool = True):
    policy = plan["gpu_execution_contract"]["policy"]
    apps = _baseline_apps(probe) if policy == "shared_nonpromotional" else []
    baseline = probe._finalize(
        {
            "schema": probe.GPU_BASELINE_SCHEMA,
            "status": "admitted",
            "plan_sha256": plan["plan_sha256"],
            "baseline_path": plan["artifact_targets"]["gpu_baseline"],
            "execution_policy": policy,
            "performance_promotion_eligible": plan["gpu_execution_contract"][
                "performance_promotion_eligible"
            ],
            "claim_scope": plan["gpu_execution_contract"]["claim_scope"],
            "sample_count": probe.GPU_IDLE_STABLE_SAMPLE_COUNT,
            "sample_interval_seconds": probe.GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
            "expected_total_memory_bytes_per_gpu": plan["gpu_execution_contract"][
                "expected_total_memory_bytes_per_gpu"
            ],
            "max_used_bytes_per_gpu": plan["gpu_execution_contract"][
                "baseline_max_used_bytes_per_gpu"
            ],
            "minimum_headroom_bytes_per_gpu": plan["gpu_execution_contract"][
                "minimum_headroom_bytes_per_gpu"
            ],
            "utilization_semantics": plan["gpu_execution_contract"][
                "utilization_semantics"
            ],
            "gpu_inventory": [
                {"gpu_index": index, "gpu_uuid": f"GPU-{index}"}
                for index in range(probe.WORLD_SIZE)
            ],
            "baseline_compute_apps": apps,
            "baseline_compute_app_identities": [
                {"gpu_uuid": row["gpu_uuid"], "driver_pid": row["pid"]} for row in apps
            ],
            "samples": [
                {
                    "sample_index": sample_index,
                    "monotonic_ns": 1_000_000_000 + sample_index * 2_000_000_000,
                    "compute_apps": deepcopy(apps),
                    "gpus": _gpu_rows(
                        probe,
                        memory_used_bytes=(1 if apps else 0),
                        utilization=(100 if apps else 0),
                    ),
                }
                for sample_index in range(probe.GPU_IDLE_STABLE_SAMPLE_COUNT)
            ],
        },
        hash_field="gpu_baseline_sha256",
    )
    path = Path(plan["artifact_targets"]["gpu_baseline"])
    if publish and not path.exists():
        _write_json(path, baseline)
    return baseline


def _mock_controller_gpu_baseline(probe, plan, monkeypatch):
    monkeypatch.setattr(
        probe,
        "_stable_gpu_execution_baseline",
        lambda received: _gpu_baseline_fixture(probe, received, publish=False),
    )
    baseline_apps = _gpu_baseline_fixture(probe, plan, publish=False)[
        "baseline_compute_apps"
    ]
    monkeypatch.setattr(
        probe,
        "_sample_nvidia_compute_apps",
        lambda: deepcopy(baseline_apps),
    )


def _refinalize(probe, value, hash_field):
    result = deepcopy(value)
    result.pop(hash_field, None)
    return probe._finalize(result, hash_field=hash_field)


def _production_fixture(
    probe,
    plan,
    triad_index: int,
    arm: str,
    *,
    step_seconds: float = 10.0,
    entry_to_terminal_seconds: float | None = None,
):
    targets = probe._arm_targets(plan, triad_index, arm)
    run_dir = Path(targets["run_dir"])
    checkpoint = run_dir / "checkpoints/step-5"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    _write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path": "frozen-model",
            "bias": "none",
            "inference_mode": True,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "peft_type": "LORA",
            "r": 8,
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
            "use_dora": True,
        },
    )
    save_file(
        {
            "base_model.model.q_proj.lora_A.default.weight": torch.ones(2, 3),
            "base_model.model.q_proj.lora_B.default.weight": torch.ones(3, 2),
            "base_model.model.q_proj.lora_magnitude_vector.default.weight": (
                torch.ones(3)
            ),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    special_tokens = checkpoint / "special_token_embeddings"
    special_tokens.mkdir()
    save_file(
        {DEFAULT_EMBED_DELTA_TENSOR_KEY: torch.ones(1, 2)},
        str(special_tokens / "special_token_embeddings.safetensors"),
    )
    _write_json(
        special_tokens / "special_token_embeddings.json",
        {
            "semantics": SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
            "tensor_key": DEFAULT_EMBED_DELTA_TENSOR_KEY,
            "tensor_shape": [1, 2],
            "tensor_dtype": "float32",
            "token_strings": ["<coord>"],
            "token_ids": [5],
            "base_model_path": "frozen-model",
            "base_config_sha256": "base-config-sha",
            "tokenizer_sha256": "tokenizer-sha",
            "tie_word_embeddings": True,
        },
    )
    _write_json(
        run_dir / "checkpoints/final.json",
        {"step": 5, "checkpoint_path": "checkpoints/step-5"},
    )
    provider = probe.ARM_SPECS[arm]["provider_mode"]
    config = deepcopy(plan["config_compatibility"]["five_step_reference_config"])
    config["run"].update(
        {
            "name": "train",
            "artifact_root": targets["output_root"],
            "collision_policy": "fail",
        }
    )
    config["training"]["forward_input_provider_mode"] = provider
    fingerprint = probe.sha256_json({"arm": arm, "config": config})
    _write_json(
        run_dir / "resolved_config.json",
        {"config": config, "resolution": {"fingerprint": fingerprint}},
    )
    phases = {
        name: {"status": "completed", "duration_seconds": float(index + 1) / 10}
        for index, name in enumerate(probe.EXPECTED_PHASES)
    }
    run = {
        "status": "completed",
        "completed_steps": 5,
        "forward_input_provider_mode": provider,
        "forward_input_provider_resolution": {
            "configured_mode": provider,
            "resolved_mode": provider,
            "source": "strict_config",
            "environment_variable": None,
            "is_semantic_override": False,
            "provider_disposition": (
                "none" if provider == "legacy_fused" else provider
            ),
            "input_build_owner": {
                "legacy_fused": "trainer_fused_device_direct",
                "synchronous": "provider_consumer_cpu",
                "overlapped": "provider_producer_cpu",
            }[provider],
            "device_transfer_owner": (
                "trainer_fused_build"
                if provider == "legacy_fused"
                else "provider_consumer"
            ),
            "lookahead_depth": int(provider == "overlapped"),
        },
        "config_fingerprint": fingerprint,
        "measurement": {
            "context": probe._expected_measurement_context(
                plan, arm=arm, workload_identity=fingerprint
            ),
            "entry_to_terminal": {
                "status": "completed",
                "started_at": "2026-08-10T00:00:00+00:00",
                "completed_at": "2026-08-10T00:01:40+00:00",
                "duration_seconds": (
                    10.0 * step_seconds
                    if entry_to_terminal_seconds is None
                    else entry_to_terminal_seconds
                ),
                "clock": "monotonic",
                "boundary": probe.MEASUREMENT_ENTRY_TO_TERMINAL_BOUNDARY,
            },
            "phases": phases,
            "resource_high_water": {
                "cpu": {
                    "max_rss_bytes": 2 * 1024**3,
                    "io_read_bytes": 1000,
                    "io_write_bytes": 2000,
                },
                "gpu": {
                    "initialized": True,
                    "max_memory_allocated_bytes": 10 * 1024**3,
                    "max_memory_reserved_bytes": 12 * 1024**3,
                },
            },
        },
        "provenance": {
            "repository": "same",
            "dependencies": {"torch": "same", "transformers": "same"},
        },
        "policy_identities": {
            "packing": "source_order_next_fit",
            "cache": "v3",
            "eval_reduction": {
                "schema_version": 1,
                "control": "auto",
                "effective_mode": "disjoint_shard",
                "source": "default",
                "pack_count": 8,
                "world_size": probe.WORLD_SIZE,
            },
            "input_provider": {
                "schema_version": 1,
                "mode": provider,
                "configured_mode": provider,
                "resolved_mode": provider,
                "source": "strict_config",
                "environment_variable": None,
                "is_semantic_override": False,
                "provider_disposition": (
                    "none" if provider == "legacy_fused" else provider
                ),
                "input_build_owner": {
                    "legacy_fused": "trainer_fused_device_direct",
                    "synchronous": "provider_consumer_cpu",
                    "overlapped": "provider_producer_cpu",
                }[provider],
                "device_transfer_owner": (
                    "trainer_fused_build"
                    if provider == "legacy_fused"
                    else "provider_consumer"
                ),
                "lookahead_depth": int(provider == "overlapped"),
            },
        },
        "materializations": {"train": "same-v3", "eval": "same-v3"},
    }
    _write_json(run_dir / "run.json", run)
    rows = []
    for step in range(1, 6):
        rows.append(
            {
                "step": step,
                "split": "train",
                "micro_step_count": 3,
                "optimizer_update_status": "applied",
                "finite_status": "finite",
                "loss/total": 1.0 / step,
                "lr": 1e-5,
                "step_duration_seconds": step_seconds,
                "input_build_seconds": 1.0,
                "input_wait_seconds": 0.1,
                "per_rank_measurement": {
                    str(rank): {
                        "step_duration_seconds": step_seconds + rank * 0.001,
                        "input_build_seconds": 1.0 + rank * 0.01,
                        "input_wait_seconds": 0.1 + rank * 0.001,
                        "resource/cpu_max_rss_bytes": float(2 * 1024**3),
                        "resource/cpu_io_read_bytes": float(1000 + rank),
                        "resource/cpu_io_write_bytes": float(2000 + rank),
                        "resource/gpu_max_memory_allocated_bytes": float(10 * 1024**3),
                        "resource/gpu_max_memory_reserved_bytes": float(12 * 1024**3),
                    }
                    for rank in range(probe.WORLD_SIZE)
                },
            }
        )
    rows.append(
        {
            "step": 3,
            "split": "eval",
            "example_count": 64,
            "pack_count": 8,
            "loss/total": 0.25,
            "optimizer_update_status": "applied",
            "finite_status": "finite",
            "eval_duration_seconds": 5.0,
            "per_rank_measurement": {
                str(rank): {
                    "eval_duration_seconds": 5.0,
                    "resource/cpu_max_rss_bytes": float(2 * 1024**3),
                    "resource/cpu_io_read_bytes": float(1000 + rank),
                    "resource/cpu_io_write_bytes": float(2000 + rank),
                    "resource/gpu_max_memory_allocated_bytes": float(10 * 1024**3),
                    "resource/gpu_max_memory_reserved_bytes": float(12 * 1024**3),
                }
                for rank in range(probe.WORLD_SIZE)
            },
        }
    )
    (run_dir / "logging.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    production_argv = probe._production_argv(plan, triad_index, arm)
    gpu_baseline = _gpu_baseline_fixture(probe, plan)
    baseline_apps = gpu_baseline["baseline_compute_apps"]
    owned_apps = [
        {
            "gpu_uuid": f"GPU-{index}",
            "pid": 100 + index,
            "process_name": "python",
        }
        for index in range(probe.WORLD_SIZE)
    ]
    lifetime_ownership, owned_bindings = probe._replay_gpu_process_ownership_sample(
        [*baseline_apps, *owned_apps],
        expected_gpu_uuids=[f"GPU-{index}" for index in range(probe.WORLD_SIZE)],
        baseline_compute_apps=baseline_apps,
        prior_bindings={},
        allow_owned_compute_apps=True,
        field="fixture.active",
    )
    final_ownership, _ = probe._replay_gpu_process_ownership_sample(
        baseline_apps,
        expected_gpu_uuids=[f"GPU-{index}" for index in range(probe.WORLD_SIZE)],
        baseline_compute_apps=baseline_apps,
        prior_bindings=owned_bindings,
        allow_owned_compute_apps=False,
        field="fixture.final",
    )
    post_phase_sweep = {
        "expected_sample_count": probe.GPU_POST_PHASE_SAMPLE_COUNT,
        "sample_count": probe.GPU_POST_PHASE_SAMPLE_COUNT,
        "sample_interval_seconds": probe.GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS,
        "samples": [
            {
                "sample_index": sample_index,
                "monotonic_ns": 1_000_000_000 + sample_index * 2_000_000_000,
                "gpu_process_ownership": deepcopy(final_ownership),
            }
            for sample_index in range(probe.GPU_POST_PHASE_SAMPLE_COUNT)
        ],
        "status": "passed",
        "sampling_error": None,
        "cleanup": None,
    }
    resource = probe._finalize(
        {
            "schema": probe.RESOURCE_SCHEMA,
            "status": "complete",
            "production_argv": production_argv,
            "production_argv_sha256": probe.sha256_json(production_argv),
            "elapsed_seconds": 100.0,
            "gpu_execution_baseline": gpu_baseline,
            "resource_breach": None,
            "monitor_error": None,
            "termination": None,
            "samples": [
                {
                    "elapsed_seconds": 1.0,
                    "process_tree_rss_bytes": 2 * 1024**3,
                    "artifact_bytes": 1024,
                    "gpu_process_ownership": lifetime_ownership,
                    "gpus": [
                        {
                            "gpu_index": index,
                            "gpu_uuid": f"GPU-{index}",
                            "memory_total_bytes": (
                                probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES
                            ),
                            "memory_headroom_bytes": (
                                probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES - 14 * 1024**3
                            ),
                            "memory_used_bytes": 14 * 1024**3,
                            "utilization_percent": 90,
                        }
                        for index in range(probe.WORLD_SIZE)
                    ],
                }
            ],
            "post_phase_gpu_process_sweep": post_phase_sweep,
        },
        hash_field="resource_sha256",
    )
    _write_json(Path(targets["resource_samples"]), resource)
    stream_root = Path(targets["executed_streams"])
    stream_root.mkdir(parents=True)
    for rank in range(probe.WORLD_SIZE):
        events = [
            {
                "planned_step_id": step,
                "local_micro_step_index": micro_step,
                "pack": {"pack_index": rank * 100 + step * 10 + micro_step},
                "tensor_values": [
                    {
                        "path": "forward_inputs.input_ids",
                        "sha256": probe.sha256_json(
                            [rank, step, micro_step, "input_ids"]
                        ),
                    }
                ],
                "tensor_metadata": [
                    {
                        "path": "forward_inputs.input_ids",
                        "dtype": "torch.int64",
                        "shape": [1, 2],
                        "stride": [2, 1],
                        "requires_grad": False,
                    }
                ],
                "supervision": {"labels": [step, micro_step]},
                "position_ids": {"position_ids": [rank, step, micro_step]},
                "loss_inputs": {
                    "expected_vocab_size": 6,
                    "logits_to_keep": [0, 1],
                },
            }
            for step in range(1, 6)
            for micro_step in range(3)
        ]
        receipt = probe._build_rank_stream_receipt(
            rank=rank,
            provider_mode=provider,
            events=events,
        )
        _write_json(stream_root / f"rank-{rank}.json", receipt)


def _observation(
    probe,
    plan,
    triad_index: int,
    arm: str,
    step_seconds: float,
    *,
    entry_to_terminal_seconds: float | None = None,
):
    _production_fixture(
        probe,
        plan,
        triad_index,
        arm,
        step_seconds=step_seconds,
        entry_to_terminal_seconds=entry_to_terminal_seconds,
    )
    return probe.derive_observation(plan, triad_index=triad_index, arm=arm)


def test_current_v3_compatibility_binds_canonical_manifests_and_stream(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    assert probe.validate_plan(plan) == plan
    assert (
        plan["w0_binding"]["current_v3"]["train"]["format_version"]
        == "coordexp-swift-pack-cache-v3"
    )
    assert (
        plan["w0_binding"]["historical_receipt"]["status"]
        == "historical_evidence_only_not_loader_input"
    )
    parity = plan["w0_binding"]["stream_parity"]
    assert all(item["projected_equal"] for item in parity.values())
    assert parity["supervision_sha256"]["raw_equal"] is False
    assert all(
        item["raw_equal"]
        for name, item in parity.items()
        if name != "supervision_sha256"
    )


def test_retired_v2_and_stream_compatibility_drift_fail_closed(probe, tmp_path):
    cache_root, path, attestation = _compatibility(probe, tmp_path)
    attestation["current_v3"]["train"]["format_version"] = (
        "coordexp-swift-pack-cache-v2"
    )
    attestation = _refinalize(probe, attestation, "attestation_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="retired"):
        probe.validate_compatibility_attestation(attestation, cache_root=cache_root)
    original = probe.load_strict_json(path)
    original["stream_parity"]["pack_ids_sha256"]["projected_equal"] = False
    original = _refinalize(probe, original, "attestation_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="parity"):
        probe.validate_compatibility_attestation(original, cache_root=cache_root)


def test_compatibility_attestation_recomputes_authenticated_v2_v3_streams(
    probe, tmp_path, monkeypatch
):
    historical_root, current_root, fingerprints = _write_compatibility_caches(
        probe, tmp_path, monkeypatch
    )
    target = tmp_path / "compatibility-attestation.json"

    attestation = probe.build_and_publish_compatibility_attestation(
        historical_cache_root=historical_root,
        cache_root=current_root,
        train_fingerprint=fingerprints["train"],
        eval_fingerprint=fingerprints["eval"],
        output_path=target,
    )

    assert probe.load_strict_json(target) == attestation
    assert attestation["status"] == "passed"
    assert attestation["oracle"]["historical_v2"]["train"]["decoded_count"] == 32
    assert attestation["oracle"]["current_v3"]["eval"]["decoded_count"] == 8
    assert set(attestation["stream_parity"]) == set(probe.STREAM_PARITY_FIELDS)
    assert all(
        item["projected_equal"] for item in attestation["stream_parity"].values()
    )
    assert attestation["stream_parity"]["supervision_sha256"]["raw_equal"] is False
    assert attestation["compatibility_projection"] == {
        "schema": probe.STREAM_COMPATIBILITY_PROJECTION_SCHEMA,
        "historical_cache_version": "coordexp-swift-pack-cache-v2",
        "current_cache_version": probe.PACK_CACHE_VERSION,
        "historical_required_absent_paths": ["metadata.pack_plan"],
        "current_removed_paths": ["metadata.pack_plan"],
        "all_other_stream_fields": "exact_sha256",
    }
    for split in ("train", "eval"):
        historical = attestation["oracle"]["historical_v2"][split]["stream_projection"]
        current = attestation["oracle"]["current_v3"][split]["stream_projection"]
        assert (
            historical["raw_semantic_digests"]
            == historical["projected_semantic_digests"]
        )
        assert (
            current["raw_semantic_digests"]["supervision_sha256"]
            != current["projected_semantic_digests"]["supervision_sha256"]
        )
        assert (
            current["pack_plan_provenance_sha256"]
            == attestation["current_v3"][split]["pack_plan_provenance_sha256"]
        )
    with pytest.raises(probe.Wave5BenchmarkError, match="exists"):
        probe.build_and_publish_compatibility_attestation(
            historical_cache_root=historical_root,
            cache_root=current_root,
            train_fingerprint=fingerprints["train"],
            eval_fingerprint=fingerprints["eval"],
            output_path=target,
        )


def test_stream_projection_requires_versioned_pack_plan_presence(probe):
    receipt = _pack_plan_receipt(probe, split="train", count=1)
    historical_with_provenance = (
        _semantic_micro_step(0, split="train", pack_plan=receipt),
    )
    current_without_provenance = (_semantic_micro_step(0, split="train"),)
    with pytest.raises(probe.Wave5BenchmarkError) as historical_error:
        probe._stream_semantic_projection(
            historical_with_provenance, role="historical_v2"
        )
    assert historical_error.value.code == "wave5.pack_plan_provenance"
    with pytest.raises(probe.Wave5BenchmarkError) as current_error:
        probe._stream_semantic_projection(
            current_without_provenance,
            role="current_v3",
            packing_determinant=_cache_determinants(split="train")["packing"],
        )
    assert current_error.value.code == "wave5.pack_plan_provenance"


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing", None),
        ("extra", None),
        ("schema_version", 2),
        ("mode", "bounded_online_fragments"),
        ("policy_identity", {"policy": "source_order_next_fit"}),
        ("plan_sha256", "1" * 64),
        ("fragment_chain_sha256", "3" * 64),
        ("fragment_sha256", "2" * 64),
        ("fragment_count", 2),
        ("source_input_count", 2),
        ("emitted_pack_count", 2),
    ],
)
def test_current_v3_pack_plan_projection_rejects_every_receipt_drift(
    probe, mutation, value
):
    receipt = _pack_plan_receipt(probe, split="train", count=1)
    if mutation == "missing":
        receipt.pop("fragment_sha256")
    elif mutation == "extra":
        receipt["unexpected"] = "drift"
    else:
        receipt[mutation] = value
    steps = (_semantic_micro_step(0, split="train", pack_plan=receipt),)
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe._stream_semantic_projection(
            steps,
            role="current_v3",
            packing_determinant=_cache_determinants(split="train")["packing"],
        )
    assert caught.value.code == "wave5.pack_plan_provenance"


@pytest.mark.parametrize("mutation", ["missing", "extra", "drift"])
def test_current_v3_pack_plan_projection_binds_manifest_determinant(probe, mutation):
    receipt = _pack_plan_receipt(probe, split="train", count=1)
    steps = (_semantic_micro_step(0, split="train", pack_plan=receipt),)
    determinant = _cache_determinants(split="train")["packing"]
    accepted = probe._stream_semantic_projection(
        steps, role="current_v3", packing_determinant=determinant
    )
    assert accepted["packing_determinant_sha256"] == probe.sha256_json(determinant)
    drifted = deepcopy(determinant)
    if mutation == "missing":
        drifted.pop("fragment_pack_budget")
    elif mutation == "extra":
        drifted["unexpected"] = "drift"
    else:
        drifted["global_max_length"] += 1
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe._stream_semantic_projection(
            steps, role="current_v3", packing_determinant=drifted
        )
    assert caught.value.code == "wave5.pack_plan_provenance"


def test_stream_projection_removes_only_metadata_pack_plan(probe):
    historical = (_semantic_micro_step(0, split="train"),)
    receipt = _pack_plan_receipt(probe, split="train", count=1)
    current_step = _semantic_micro_step(0, split="train", pack_plan=receipt)
    old_projection = probe._stream_semantic_projection(historical, role="historical_v2")
    new_projection = probe._stream_semantic_projection(
        (current_step,),
        role="current_v3",
        packing_determinant=_cache_determinants(split="train")["packing"],
    )
    assert (
        old_projection["projected_semantic_digests"]
        == new_projection["projected_semantic_digests"]
    )
    drifted_metadata = deepcopy(current_step.metadata)
    drifted_metadata["augmentation_receipt"]["object_ordering"] = "drifted"
    drifted_projection = probe._stream_semantic_projection(
        (replace(current_step, metadata=drifted_metadata),),
        role="current_v3",
        packing_determinant=_cache_determinants(split="train")["packing"],
    )
    assert (
        old_projection["projected_semantic_digests"]["supervision_sha256"]
        != drifted_projection["projected_semantic_digests"]["supervision_sha256"]
    )


def test_refuted_v1_compatibility_receipt_cannot_masquerade(probe, tmp_path):
    cache_root, _, attestation = _compatibility(probe, tmp_path)
    attestation["schema"] = probe.LEGACY_REFUTED_COMPATIBILITY_SCHEMA
    attestation = _refinalize(probe, attestation, "attestation_sha256")
    with pytest.raises(probe.Wave5BenchmarkError):
        probe.validate_compatibility_attestation(attestation, cache_root=cache_root)


def test_compatibility_cli_is_the_only_attestation_publisher(
    probe, tmp_path, monkeypatch
):
    historical_root, current_root, fingerprints = _write_compatibility_caches(
        probe, tmp_path, monkeypatch
    )
    target = tmp_path / "cli-attestation.json"
    assert (
        probe.main(
            [
                "attest-compatibility",
                "--historical-cache-root",
                str(historical_root),
                "--cache-root",
                str(current_root),
                "--train-fingerprint",
                fingerprints["train"],
                "--eval-fingerprint",
                fingerprints["eval"],
                "--output",
                str(target),
            ]
        )
        == 0
    )
    assert probe.load_strict_json(target)["command"]["output_path"] == str(target)


def test_direct_file_cli_bootstraps_repo_imports(probe, tmp_path):
    historical_root = tmp_path / "historical"
    current_root = tmp_path / "current"
    historical_root.mkdir()
    current_root.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(probe.SCRIPT_PATH),
            "attest-compatibility",
            "--historical-cache-root",
            str(historical_root),
            "--cache-root",
            str(current_root),
            "--train-fingerprint",
            "1" * 64,
            "--eval-fingerprint",
            "2" * 64,
            "--output",
            str(tmp_path / "absent.json"),
        ],
        cwd=probe.REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stderr
    assert "required path is absent" in result.stderr


@pytest.mark.parametrize("mutation", ["empty_chunks", "forged_hash", "payload_drift"])
def test_compatibility_command_rejects_untrusted_or_drifted_payloads(
    probe, tmp_path, monkeypatch, mutation
):
    historical_root, current_root, fingerprints = _write_compatibility_caches(
        probe, tmp_path, monkeypatch
    )
    if mutation == "empty_chunks":
        binding = deepcopy(probe.HISTORICAL_W0_CACHE_BINDING)
        fingerprint = binding["eval"]["fingerprint"]
        manifest_path = historical_root / fingerprint / "manifest.json"
        manifest = probe.load_strict_json(manifest_path)
        manifest["chunks"] = []
        _write_json(manifest_path, manifest)
        binding["eval"]["manifest_sha256"] = probe.sha256_file(manifest_path)
        monkeypatch.setattr(probe, "HISTORICAL_W0_CACHE_BINDING", binding)
    elif mutation == "forged_hash":
        binding = deepcopy(probe.HISTORICAL_W0_CACHE_BINDING)
        binding["train"]["manifest_sha256"] = "0" * 64
        monkeypatch.setattr(probe, "HISTORICAL_W0_CACHE_BINDING", binding)
    else:
        drift_root = tmp_path / "drift-v3"
        determinants = _cache_determinants(split="train")
        fingerprint = pack_cache.packing_cache_fingerprint_from_determinants(
            determinants
        )
        cache_dir = pack_cache.cache_dir_for_fingerprint(drift_root, fingerprint)
        drifted = tuple(
            _semantic_micro_step(index + 100, split="train") for index in range(32)
        )
        pack_cache.write_micro_step_cache(
            cache_dir,
            drifted,
            cache_root=drift_root,
            fingerprint=fingerprint,
            determinants=determinants,
            augmentation={
                "split": "train",
                "mode": "disabled",
                "policy": "geometry_flips",
                "enabled": False,
                "seed": 17,
                "input_example_count": 32,
                "output_example_count": 32,
                "presentation_count": 32,
                "object_ordering": "source_order",
            },
            materialization=pack_cache.build_packing_cache_materialization(),
            determinant_revalidator=lambda: determinants,
        )
        eval_source = current_root / probe.PACK_CACHE_VERSION / fingerprints["eval"]
        eval_target = drift_root / probe.PACK_CACHE_VERSION / fingerprints["eval"]
        import shutil

        shutil.copytree(eval_source, eval_target)
        current_root = drift_root
        fingerprints["train"] = fingerprint
    with pytest.raises(probe.Wave5BenchmarkError):
        probe.build_and_publish_compatibility_attestation(
            historical_cache_root=historical_root,
            cache_root=current_root,
            train_fingerprint=fingerprints["train"],
            eval_fingerprint=fingerprints["eval"],
            output_path=tmp_path / "rejected.json",
        )


def test_plan_freezes_exact_arms_triads_and_generated_runner(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    assert plan["triad_orders"] == [["R", "D", "O"], ["D", "O", "R"], ["O", "R", "D"]]
    assert [plan["arms"][arm]["provider_mode"] for arm in ("R", "D", "O")] == [
        "synchronous",
        "legacy_fused",
        "overlapped",
    ]
    assert plan["measurement_contract"]["accepted_paired_observations"] == 3
    assert plan["measurement_contract"]["warmup_exclusion_steps"] == 2
    assert plan["measurement_contract"]["wall_clock_scope"] == (
        "training_entry_to_terminal_artifact"
    )
    assert plan["measurement_contract"]["profile_sync_timings"] == {
        "enabled": False,
        "source": "default",
    }
    assert set(plan["measurement_contract"]["comparison_arms"]) == {"R", "D", "O"}
    assert plan["measurement_contract"]["blind_retry"] is False
    assert plan["runner"]["argv_template"][2] == "arm"
    assert plan["execution_baseline"]["repository"]["aggregate_sha256"]
    assert set(plan["execution_baseline"]["repository"]["owners"]) >= {
        "provider",
        "pipeline",
        "trainer",
        "artifact_writer",
        "checkpoint_writer",
    }
    assert plan["execution_baseline"]["runtime"]["baseline_sha256"]
    attempt = probe.build_attempt(plan)
    assert attempt["execution_baseline"] == plan["execution_baseline"]
    assert attempt["measurement_contract"] == plan["measurement_contract"]
    compatibility = plan["config_compatibility"]
    projection = compatibility["compatibility_projection"]
    assert projection["schema"] == (
        "coordexp-swift-wave5-w0-current-config-compatibility-projection-v2"
    )
    assert projection["removed_path_values"] == [
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
        compatibility["current_parent_projection_sha256"]
        == compatibility["historical_w0_projection_sha256"]
    )
    assert compatibility["historical_w0"]["resolved_config_file_sha256"]
    assert compatibility["historical_w0"]["run_receipt_file_sha256"]
    assert compatibility["historical_w0"]["base_model_weight_identity"][
        "receipt_file_sha256"
    ]
    assert compatibility["normative_current_full_config_sha256"]
    assert plan["base_model_weight_identity"]["aggregate_sha256"]
    assert (
        plan["historical_w0_base_weight_aggregate_sha256"]
        == plan["base_model_weight_identity"]["aggregate_sha256"]
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "value"])
def test_config_compatibility_projection_rejects_every_inventory_drift(
    probe, tmp_path, mutation
):
    plan = _plan(probe, tmp_path)
    rows = plan["config_compatibility"]["compatibility_projection"][
        "removed_path_values"
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append({"path": "runtime.seed", "value": 17})
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["value"] = "window_binpack"
    plan = _refinalize(probe, plan, "plan_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="config compatibility"):
        probe.validate_plan(plan)


def test_config_compatibility_revalidates_the_complete_live_config(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path)
    original = probe._five_step_reference_config

    def drifted_live_config():
        config = original()
        config["runtime"]["seed"] += 1
        return config

    monkeypatch.setattr(probe, "_five_step_reference_config", drifted_live_config)
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe.validate_plan(plan)
    assert caught.value.code == "wave5.config_compatibility"


def test_production_runner_authenticates_exact_measurement_context(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    argv = probe._production_argv(plan, 1, "O")
    assert argv[-10:] == [
        "--comparison-arm",
        "O",
        "--warmup-exclusion-steps",
        "2",
        "--wall-clock-scope",
        "training_entry_to_terminal_artifact",
        "--profile-sync-timings",
        "disabled-default",
        "--measurement-contract-sha256",
        probe.sha256_json(plan["measurement_contract"]),
    ]


def test_production_environment_clears_eval_override_and_observation_attests_default(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path)
    monkeypatch.setenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", "replicated")
    environment = probe._production_environment(plan)
    assert "COORDEXP_SWIFT_EVAL_REDUCTION_MODE" not in environment
    assert plan["semantic_contract"]["eval_reduction"] == {
        "schema_version": 1,
        "control": "auto",
        "effective_mode": "disjoint_shard",
        "source": "default",
        "pack_count": 8,
        "world_size": probe.WORLD_SIZE,
    }
    assert plan["config_compatibility"]["historical_w0"]["eval_reduction"] == {
        "schema_version": 1,
        "mode": "disjoint_shard",
        "source": "default",
    }

    _production_fixture(probe, plan, 0, "R")
    run_path = Path(probe._arm_targets(plan, 0, "R")["run_dir"]) / "run.json"
    run = probe.load_strict_json(run_path)
    run["policy_identities"]["eval_reduction"] = {
        "schema_version": 1,
        "mode": "replicated",
        "source": "COORDEXP_SWIFT_EVAL_REDUCTION_MODE",
    }
    _write_json(run_path, run)
    with pytest.raises(probe.Wave5BenchmarkError, match="eval reduction"):
        probe.derive_observation(plan, triad_index=0, arm="R")


def test_instrumented_runner_passes_authenticated_context_to_public_entrypoint(
    probe, tmp_path, monkeypatch
):
    from src import train as train_module
    from src.training import pipeline as pipeline_module

    plan = _plan(probe, tmp_path)
    config_path = probe._write_arm_config(plan, 0, "O")
    output = tmp_path / "streams"
    captured = {}
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", str(probe.WORLD_SIZE))
    monkeypatch.setattr(
        probe, "_install_executed_stream_instrumentation", lambda _: None
    )

    def fake_pipeline(path, *, measurement_context):
        captured["path"] = path
        captured["measurement_context"] = deepcopy(measurement_context)
        return {"status": "completed"}

    def fake_main(argv, *, runner):
        assert argv == ["--config", str(config_path)]
        runner(config_path)
        return 0

    monkeypatch.setattr(pipeline_module, "run_training_pipeline", fake_pipeline)
    monkeypatch.setattr(train_module, "main", fake_main)
    assert (
        probe.run_instrumented_train(
            config_path,
            output,
            comparison_arm="O",
            warmup_exclusion_steps=2,
            wall_clock_scope="training_entry_to_terminal_artifact",
            profile_sync_timings="disabled-default",
            measurement_contract_sha256=probe.sha256_json(plan["measurement_contract"]),
        )
        == 0
    )
    assert captured["measurement_context"] == {
        "comparison_arm": "O",
        "wall_clock_scope": "training_entry_to_terminal_artifact",
        "warmup_exclusion_steps": 2,
    }


def test_plan_rejects_config_projection_or_weight_identity_mutation(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path)
    plan["config_compatibility"]["five_step_reference_config"]["runtime"]["seed"] += 1
    plan = _refinalize(probe, plan, "plan_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="config compatibility"):
        probe.validate_plan(plan)

    plan = _plan(probe, tmp_path / "weight")
    plan["base_model_weight_identity"]["shards"][0]["sha256"] = "0" * 64
    plan["base_model_weight_identity"]["aggregate_sha256"] = probe.sha256_json(
        {
            key: value
            for key, value in plan["base_model_weight_identity"].items()
            if key != "aggregate_sha256"
        }
    )
    plan = _refinalize(probe, plan, "plan_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="weight identity"):
        probe.validate_plan(plan)

    live_plan = _plan(probe, tmp_path / "live-weight")
    drifted = deepcopy(live_plan["base_model_weight_identity"])
    drifted["shards"][0]["sha256"] = "1" * 64
    drifted["aggregate_sha256"] = probe.sha256_json(
        {key: value for key, value in drifted.items() if key != "aggregate_sha256"}
    )
    monkeypatch.setattr(
        probe, "_current_base_model_weight_identity", lambda: deepcopy(drifted)
    )
    with pytest.raises(probe.Wave5BenchmarkError, match="weight identity"):
        probe.validate_plan(live_plan)


def test_plan_reauthenticates_historical_w0_artifacts_and_weight_aggregate(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path)
    resolved_path = probe.HISTORICAL_W0_RESOLVED_CONFIG_PATH
    original_resolved = probe.load_strict_json(resolved_path)
    resolved = deepcopy(original_resolved)
    resolved["config"]["runtime"]["seed"] += 1
    _write_json(resolved_path, resolved)
    with pytest.raises(probe.Wave5BenchmarkError, match="historical W0"):
        probe.validate_plan(plan)
    _write_json(resolved_path, original_resolved)

    replacement = _plan(probe, tmp_path / "weight")
    monkeypatch.setattr(probe, "HISTORICAL_W0_BASE_WEIGHT_AGGREGATE_SHA256", "0" * 64)
    with pytest.raises(probe.Wave5BenchmarkError, match="weight identity"):
        probe.validate_plan(replacement)


def test_execution_baseline_cache_returns_deeply_frozen_copies(probe):
    first = probe._execution_baseline()
    original = first["repository"]["aggregate_sha256"]
    first["repository"]["aggregate_sha256"] = "0" * 64
    assert probe._execution_baseline()["repository"]["aggregate_sha256"] == original


def test_plan_rejects_execution_owner_or_runtime_baseline_drift(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    plan["execution_baseline"]["repository"]["owners"]["trainer"]["sha256"] = "0" * 64
    plan = _refinalize(probe, plan, "plan_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="execution baseline"):
        probe.validate_plan(plan)


def test_cpu_provenance_seam_cannot_bypass_real_runtime_admission(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path)
    _bind_authenticated_cpu_provenance(probe, monkeypatch)
    probe._revalidate_execution_baseline(plan)

    monkeypatch.setattr(
        probe,
        "_collect_execution_baseline_provenance",
        lambda *, repository_root: {"dependencies": {}, "runtime": {}},
    )
    with pytest.raises(probe.Wave5BenchmarkError, match="runtime dependency baseline"):
        probe._revalidate_execution_baseline(plan)


def test_target_alias_and_arbitrary_command_are_rejected(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    plan["artifact_targets"]["terminal"] = plan["artifact_targets"]["attempt"]
    plan = _refinalize(probe, plan, "plan_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="targets"):
        probe.validate_plan(plan)
    plan = _plan(probe, tmp_path / "second")
    plan["runner"]["argv_template"] = [sys.executable, "-c", "arbitrary"]
    plan = _refinalize(probe, plan, "plan_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="arbitrary"):
        probe.validate_plan(plan)


def test_missing_and_forged_attempt_marker_fail_closed(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    with pytest.raises(probe.Wave5BenchmarkError, match="absent"):
        probe.load_authenticated_attempt(plan, plan["artifact_targets"]["attempt"])
    attempt = probe.build_attempt(plan)
    attempt["commands"][0]["argv"][-1] = "O"
    attempt = _refinalize(probe, attempt, "attempt_sha256")
    _write_json(Path(plan["artifact_targets"]["attempt"]), attempt)
    with pytest.raises(probe.Wave5BenchmarkError, match="forged"):
        probe.load_authenticated_attempt(plan, plan["artifact_targets"]["attempt"])


def test_synthetic_child_observation_is_rejected(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    _production_fixture(probe, plan, 0, "R")
    _write_json(Path(probe._arm_targets(plan, 0, "R")["observation"]), {"forged": True})
    with pytest.raises(probe.Wave5BenchmarkError, match="synthetic"):
        probe.derive_observation(plan, triad_index=0, arm="R")


def test_artifact_derivation_binds_full_schema(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    observation = _observation(probe, plan, 0, "R", 10.0)
    assert (
        probe.validate_observation(observation, plan=plan, triad_index=0, arm="R")
        == observation
    )
    assert set(observation["phases"]) == set(probe.EXPECTED_PHASES)
    assert observation["pack_utilization"]["train"]["micro_step_count"] == 32
    assert observation["evaluation"]["denominators"]["example_count"] == 64
    assert observation["checkpoint"]["inventory"]["file_count"] == 4
    assert observation["checkpoint"]["loadability"]["loadable"] is True
    assert observation["resources"]["arm_process_lifetime"]["samples"]
    assert observation["resources"]["summary"]["resource_gate_passed"] is True
    assert observation["position"] == 1
    assert observation["measurement_context"]["comparison_arm"] == "R"
    assert observation["timing"]["entry_to_terminal_seconds"] == 100.0
    assert (
        observation["resources"]["summary"]["gpu_utilization_representative"][
            "all_device_samples"
        ]["median"]
        == 90.0
    )
    assert observation["timing"][
        "steady_state_optimizer_step_wall_seconds"
    ] == pytest.approx(30.021)
    assert observation["timing"]["per_step"][0][
        "input_build_skew_seconds"
    ] == pytest.approx(0.07)
    assert set(observation["semantic"]) == set(probe.SEMANTIC_PARITY_FIELDS)
    assert set(observation["executed_provider_streams"]["ranks"]) == {
        str(rank) for rank in range(probe.WORLD_SIZE)
    }
    assert observation["executed_provider_streams"]["event_count_per_rank"] == 15
    pack_id_parity = plan["w0_binding"]["stream_parity"]["pack_ids_sha256"]
    assert (
        pack_id_parity["current_v3_raw_sha256"]
        == pack_id_parity["current_v3_projected_sha256"]
    )
    assert (
        observation["semantic"]["pack_ids_sha256"]
        != pack_id_parity["current_v3_projected_sha256"]
    )


@pytest.mark.parametrize(
    "mutation",
    ("extra_top_level_field", "complete_with_monitor_error", "missing_final_sample"),
)
def test_resource_receipt_rejects_hash_valid_shape_and_status_mutations(
    probe, tmp_path, mutation
):
    plan = _plan(probe, tmp_path)
    _production_fixture(probe, plan, 0, "R")
    resource_path = Path(probe._arm_targets(plan, 0, "R")["resource_samples"])
    resource = probe.load_strict_json(resource_path)
    if mutation == "extra_top_level_field":
        resource["unexpected"] = "hash-valid"
    elif mutation == "complete_with_monitor_error":
        resource["monitor_error"] = {
            "type": "Wave5BenchmarkError",
            "code": "wave5.monitor",
            "message": "contradicts complete status",
        }
    else:
        resource.pop("post_phase_gpu_process_sweep")
    resource = _refinalize(probe, resource, "resource_sha256")
    _write_json(resource_path, resource)

    with pytest.raises(probe.Wave5BenchmarkError, match="resource receipt"):
        probe.derive_observation(plan, triad_index=0, arm="R")


def test_observation_rejects_non_arm_config_drift_but_provider_factor_passes(
    probe, tmp_path
):
    plan = _plan(probe, tmp_path)
    _production_fixture(probe, plan, 0, "R")
    targets = probe._arm_targets(plan, 0, "R")
    resolved_path = Path(targets["run_dir"]) / "resolved_config.json"
    resolved = probe.load_strict_json(resolved_path)
    resolved["config"]["runtime"]["seed"] += 1
    _write_json(resolved_path, resolved)
    with pytest.raises(probe.Wave5BenchmarkError, match="config compatibility"):
        probe.derive_observation(plan, triad_index=0, arm="R")

    provider_plan = _plan(probe, tmp_path / "provider")
    observation = _observation(probe, provider_plan, 0, "D", 10.0)
    assert observation["provider_mode"] == "legacy_fused"
    assert (
        observation["execution_identity"]["config_compatibility_projection_sha256"]
        == (provider_plan["config_compatibility"]["current_parent_projection_sha256"])
    )


def test_observation_position_is_canonical_one_based(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    observation = _observation(probe, plan, 0, "D", 10.0)
    assert observation["position"] == 2
    for invalid in (0, 3):
        mutated = deepcopy(observation)
        mutated["position"] = invalid
        mutated = _refinalize(probe, mutated, "observation_sha256")
        with pytest.raises(probe.Wave5BenchmarkError, match="identity"):
            probe.validate_observation(mutated, plan=plan, triad_index=0, arm="D")


def test_observation_measurement_context_and_e2e_duration_are_authenticated(
    probe, tmp_path
):
    plan = _plan(probe, tmp_path)
    observation = _observation(probe, plan, 0, "R", 10.0)
    mutated = deepcopy(observation)
    mutated["measurement_context"]["comparison_arm"] = "D"
    mutated = _refinalize(probe, mutated, "observation_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="provider factor"):
        probe.validate_observation(mutated, plan=plan, triad_index=0, arm="R")

    mutated = deepcopy(observation)
    mutated["timing"]["entry_to_terminal_seconds"] = 1.0
    mutated = _refinalize(probe, mutated, "observation_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="timing"):
        probe.validate_observation(mutated, plan=plan, triad_index=0, arm="R")


def test_mutated_executed_rank_stream_cannot_be_replaced_by_cache_attestation(
    probe, tmp_path
):
    plan = _plan(probe, tmp_path)
    _production_fixture(probe, plan, 0, "R")
    targets = probe._arm_targets(plan, 0, "R")
    rank_path = Path(targets["executed_streams"]) / "rank-3.json"
    receipt = probe.load_strict_json(rank_path)
    receipt["events"][0]["supervision"]["labels"] = [999]
    receipt = _refinalize(probe, receipt, "stream_receipt_sha256")
    _write_json(rank_path, receipt)
    with pytest.raises(probe.Wave5BenchmarkError, match="executed stream"):
        probe.derive_observation(plan, triad_index=0, arm="R")


def test_gpu_idle_preflight_requires_two_stable_final_samples(probe, monkeypatch):
    idle = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_total_bytes": probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "memory_headroom_bytes": probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "memory_used_bytes": 0,
            "utilization_percent": 0,
        }
        for index in range(probe.WORLD_SIZE)
    ]
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(idle))
    monkeypatch.setattr(probe, "_sample_nvidia_compute_apps", lambda: [])
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    receipt = probe._stable_gpu_idle_samples()
    assert receipt["sample_count"] == 2
    assert len(receipt["samples"]) == 2
    assert receipt["gpu_inventory"] == [
        {"gpu_index": index, "gpu_uuid": f"GPU-{index}"}
        for index in range(probe.WORLD_SIZE)
    ]
    assert [sample["gpus"] for sample in receipt["samples"]] == [idle, idle]

    busy = deepcopy(idle)
    busy[0]["utilization_percent"] = 100
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(busy))
    with pytest.raises(probe.Wave5BenchmarkError, match="busy"):
        probe._stable_gpu_idle_samples()

    over_one_gib = deepcopy(idle)
    over_one_gib[0]["memory_used_bytes"] = 1024**3 + 1
    over_one_gib[0]["memory_headroom_bytes"] -= 1024**3 + 1
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(over_one_gib))
    with pytest.raises(probe.Wave5BenchmarkError, match="busy"):
        probe._stable_gpu_idle_samples()


def test_gpu_idle_preflight_rejects_index_uuid_mapping_drift(probe, monkeypatch):
    first = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_total_bytes": probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "memory_headroom_bytes": probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "memory_used_bytes": 0,
            "utilization_percent": 0,
        }
        for index in range(probe.WORLD_SIZE)
    ]
    second = deepcopy(first)
    second[4]["gpu_uuid"] = "GPU-replaced"
    samples = iter([first, second])
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(next(samples)))
    monkeypatch.setattr(probe, "_sample_nvidia_compute_apps", lambda: [])
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)

    with pytest.raises(probe.Wave5BenchmarkError, match="mapping"):
        probe._stable_gpu_idle_samples()

    duplicated = deepcopy(first)
    duplicated[7]["gpu_uuid"] = duplicated[6]["gpu_uuid"]
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(duplicated))
    with pytest.raises(probe.Wave5BenchmarkError, match="eight GPUs"):
        probe._stable_gpu_idle_samples()


def test_shared_gpu_preflight_binds_stable_preexisting_driver_pids(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path, gpu_execution_policy="shared_nonpromotional")
    gpus = _gpu_rows(
        probe,
        memory_used_bytes=probe.SHARED_GPU_BASELINE_MAX_USED_BYTES,
        utilization=100,
    )
    apps = _baseline_apps(probe)
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(gpus))
    monkeypatch.setattr(probe, "_sample_nvidia_compute_apps", lambda: deepcopy(apps))
    sleeps = []
    monkeypatch.setattr(probe.time, "sleep", sleeps.append)

    baseline = probe._stable_gpu_execution_baseline(plan)

    assert sleeps == [probe.GPU_BASELINE_SAMPLE_INTERVAL_SECONDS]
    assert probe.GPU_BASELINE_SAMPLE_INTERVAL_SECONDS >= 2.0
    assert baseline["execution_policy"] == "shared_nonpromotional"
    assert baseline["performance_promotion_eligible"] is False
    assert baseline["utilization_semantics"] == "observational_only"
    assert baseline["baseline_compute_apps"] == apps
    assert baseline["expected_total_memory_bytes_per_gpu"] == (81_920 * 1024**2)
    assert baseline["minimum_headroom_bytes_per_gpu"] == 32_768 * 1024**2
    assert all(
        row["memory_headroom_bytes"] == 32_768 * 1024**2
        for sample in baseline["samples"]
        for row in sample["gpus"]
    )
    assert baseline["baseline_compute_app_identities"] == [
        {"gpu_uuid": row["gpu_uuid"], "driver_pid": row["pid"]} for row in apps
    ]
    assert baseline["gpu_inventory"] == [
        {"gpu_index": index, "gpu_uuid": f"GPU-{index}"}
        for index in range(probe.WORLD_SIZE)
    ]


def test_shared_gpu_preflight_rejects_memory_and_driver_pid_drift(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path, gpu_execution_policy="shared_nonpromotional")
    over_limit = _gpu_rows(
        probe,
        memory_used_bytes=probe.SHARED_GPU_BASELINE_MAX_USED_BYTES + 1,
        utilization=0,
    )
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(over_limit))
    monkeypatch.setattr(
        probe, "_sample_nvidia_compute_apps", lambda: _baseline_apps(probe)
    )
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe._stable_gpu_execution_baseline(plan)
    assert caught.value.code == "wave5.gpu_busy"

    gpus = _gpu_rows(probe, memory_used_bytes=1, utilization=100)
    app_samples = [_baseline_apps(probe), _baseline_apps(probe, pid_offset=800_000)]
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(gpus))
    monkeypatch.setattr(
        probe,
        "_sample_nvidia_compute_apps",
        lambda: deepcopy(app_samples.pop(0)),
    )
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe._stable_gpu_execution_baseline(plan)
    assert caught.value.code == "wave5.gpu_baseline_drift"

    wrong_total = _gpu_rows(
        probe,
        memory_used_bytes=1,
        memory_total_bytes=probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES - 1,
    )
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(wrong_total))
    monkeypatch.setattr(
        probe, "_sample_nvidia_compute_apps", lambda: _baseline_apps(probe)
    )
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe._stable_gpu_execution_baseline(plan)
    assert caught.value.code == "wave5.gpu_sample"


def test_shared_gpu_ownership_allows_baseline_during_arm_but_not_new_rows_after(
    probe,
):
    expected = [f"GPU-{index}" for index in range(probe.WORLD_SIZE)]
    baseline = _baseline_apps(probe)
    owned = [
        {
            "gpu_uuid": f"GPU-{index}",
            "pid": 900_000 + index,
            "process_name": f"owned-rank-{index}",
        }
        for index in range(probe.WORLD_SIZE)
    ]
    active, bindings = probe._replay_gpu_process_ownership_sample(
        [*baseline, *owned],
        expected_gpu_uuids=expected,
        baseline_compute_apps=baseline,
        prior_bindings={},
        allow_owned_compute_apps=True,
        field="active",
    )
    assert active["status"] == "passed"
    assert active["baseline_compute_apps"] == baseline
    assert active["owned_compute_apps"] == owned

    clean, replayed = probe._replay_gpu_process_ownership_sample(
        baseline[:-1],
        expected_gpu_uuids=expected,
        baseline_compute_apps=baseline,
        prior_bindings=bindings,
        allow_owned_compute_apps=False,
        field="after-arm",
    )
    assert clean["status"] == "passed"
    assert clean["compute_apps"] == baseline[:-1]
    assert replayed == bindings

    leaked, _ = probe._replay_gpu_process_ownership_sample(
        [*baseline, owned[0]],
        expected_gpu_uuids=expected,
        baseline_compute_apps=baseline,
        prior_bindings=bindings,
        allow_owned_compute_apps=False,
        field="after-arm-leak",
    )
    assert leaked["status"] == "failed"
    assert leaked["violation_codes"] == ["nonbaseline_compute_app_after_phase"]
    assert leaked["foreign_compute_apps"] == [owned[0]]


def test_post_phase_gpu_sweep_is_exactly_two_samples_at_least_two_seconds_apart(
    probe, monkeypatch
):
    baseline = {
        "gpu_inventory": [
            {"gpu_index": index, "gpu_uuid": f"GPU-{index}"}
            for index in range(probe.WORLD_SIZE)
        ],
        "baseline_compute_apps": _baseline_apps(probe),
    }
    samples = [baseline["baseline_compute_apps"][:-1], []]
    monkeypatch.setattr(
        probe,
        "_sample_nvidia_compute_apps",
        lambda: deepcopy(samples.pop(0)),
    )
    sleeps = []
    monkeypatch.setattr(probe.time, "sleep", sleeps.append)
    monotonic_ns = iter([1_000_000_000, 3_000_000_000])
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: next(monotonic_ns))
    sweep = probe._post_phase_gpu_process_sweep(
        baseline,
        prior_bindings={},
        field="after-arm",
    )
    assert sweep["expected_sample_count"] == 2
    assert sweep["sample_count"] == 2
    assert sweep["sample_interval_seconds"] >= 2.0
    assert len(sweep["samples"]) == 2
    assert [sample["monotonic_ns"] for sample in sweep["samples"]] == [
        1_000_000_000,
        3_000_000_000,
    ]
    assert all(
        sample["gpu_process_ownership"]["status"] == "passed"
        for sample in sweep["samples"]
    )
    assert sleeps == [probe.GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_index", 0.5),
        ("memory_total_bytes", 81_920 * 1024**2 + 0.5),
        ("memory_used_bytes", 0.5),
        ("memory_headroom_bytes", 81_920 * 1024**2 - 0.5),
        ("utilization_percent", 0.5),
        ("utilization_percent", True),
    ],
)
def test_gpu_samples_reject_nonintegral_and_boolean_nvidia_fields(probe, field, value):
    rows = _gpu_rows(probe)
    rows[0][field] = value
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe._validate_gpu_sample(rows, field="fractional")
    assert caught.value.code == "wave5.gpu_sample"


@pytest.mark.parametrize(
    ("timestamps", "accepted"),
    [
        ([1_000_000_000, 3_000_000_000], True),
        ([1_000_000_000, 2_000_000_000], False),
        ([1_000_000_000], False),
        ([], False),
    ],
    ids=("exact-two-seconds", "one-second", "one-sample", "zero-samples"),
)
def test_gpu_baseline_authenticates_actual_sample_timestamps(
    probe, tmp_path, timestamps, accepted
):
    plan = _plan(probe, tmp_path)
    baseline = _gpu_baseline_fixture(probe, plan, publish=False)
    baseline["samples"] = baseline["samples"][: len(timestamps)]
    baseline["sample_count"] = len(timestamps)
    for sample, monotonic_ns in zip(baseline["samples"], timestamps, strict=True):
        sample["monotonic_ns"] = monotonic_ns
    baseline = _refinalize(probe, baseline, "gpu_baseline_sha256")
    if accepted:
        assert probe._validate_gpu_execution_baseline(baseline, plan=plan) == baseline
    else:
        with pytest.raises(probe.Wave5BenchmarkError):
            probe._validate_gpu_execution_baseline(baseline, plan=plan)


def test_resource_v5_publishes_authenticated_sampling_error_sweep(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path, publish=True)
    _production_fixture(probe, plan, 0, "R")
    targets = probe._arm_targets(plan, 0, "R")
    resource = probe.load_strict_json(targets["resource_samples"])

    def unavailable():
        raise probe.Wave5BenchmarkError(
            "injected post-phase inventory failure",
            code="wave5.gpu_process_sample",
        )

    monkeypatch.setattr(probe, "_sample_nvidia_compute_apps", unavailable)
    sweep = probe._post_phase_gpu_process_sweep(
        resource["gpu_execution_baseline"],
        prior_bindings={},
        field="injected",
    )
    assert sweep["status"] == "sampling_error"
    assert sweep["sample_count"] == 0
    assert sweep["sampling_error"] == {
        "type": "Wave5BenchmarkError",
        "code": "wave5.gpu_process_sample",
        "message": "injected post-phase inventory failure",
    }
    assert sweep["cleanup"] == {
        "termination_recorded": False,
        "descendant_attestation": "not_required",
        "descendants_remaining_count": 0,
    }
    resource["status"] = "monitor_error"
    resource["monitor_error"] = deepcopy(sweep["sampling_error"])
    resource["post_phase_gpu_process_sweep"] = sweep
    resource = _refinalize(probe, resource, "resource_sha256")
    published = tmp_path / "published-sampling-error-resource-v5.json"
    probe.publish_json_absent(published, resource)
    assert (
        probe._validate_resource_receipt(probe.load_strict_json(published), plan=plan)
        == resource
    )


def test_resource_v5_rejects_short_or_too_close_post_phase_sweeps(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    _production_fixture(probe, plan, 0, "R")
    targets = probe._arm_targets(plan, 0, "R")
    resource = probe.load_strict_json(targets["resource_samples"])
    assert probe._validate_resource_receipt(resource, plan=plan) == resource
    mutations = []
    for sample_count in (0, 1):
        mutated = deepcopy(resource)
        mutated["post_phase_gpu_process_sweep"]["samples"] = mutated[
            "post_phase_gpu_process_sweep"
        ]["samples"][:sample_count]
        mutated["post_phase_gpu_process_sweep"]["sample_count"] = sample_count
        mutations.append(mutated)
    too_close = deepcopy(resource)
    too_close["post_phase_gpu_process_sweep"]["samples"][1]["monotonic_ns"] = (
        too_close["post_phase_gpu_process_sweep"]["samples"][0]["monotonic_ns"]
        + 1_000_000_000
    )
    mutations.append(too_close)
    for mutated in mutations:
        mutated = _refinalize(probe, mutated, "resource_sha256")
        with pytest.raises(probe.Wave5BenchmarkError):
            probe._validate_resource_receipt(mutated, plan=plan)


@pytest.mark.parametrize(
    "field",
    [
        "gpu_index",
        "memory_total_bytes",
        "memory_used_bytes",
        "memory_headroom_bytes",
        "utilization_percent",
    ],
)
def test_resource_v5_rejects_refinalized_fractional_nvidia_fields(
    probe, tmp_path, field
):
    plan = _plan(probe, tmp_path)
    _production_fixture(probe, plan, 0, "R")
    targets = probe._arm_targets(plan, 0, "R")
    resource = probe.load_strict_json(targets["resource_samples"])
    resource["samples"][0]["gpus"][0][field] += 0.5
    resource = _refinalize(probe, resource, "resource_sha256")
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe._validate_resource_receipt(resource, plan=plan)
    assert caught.value.code == "wave5.gpu_sample"


def test_shared_plan_and_decision_are_explicitly_nonpromotional(probe, tmp_path):
    plan = _plan(probe, tmp_path, gpu_execution_policy="shared_nonpromotional")
    assert plan["gpu_execution_contract"] == {
        "policy": "shared_nonpromotional",
        "baseline_sample_count": 2,
        "baseline_sample_interval_seconds": (
            probe.GPU_BASELINE_SAMPLE_INTERVAL_SECONDS
        ),
        "expected_total_memory_bytes_per_gpu": (probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES),
        "baseline_max_used_bytes_per_gpu": (probe.SHARED_GPU_BASELINE_MAX_USED_BYTES),
        "minimum_headroom_bytes_per_gpu": (probe.GPU_MINIMUM_HEADROOM_BYTES),
        "post_phase_sample_count": 2,
        "post_phase_sample_interval_seconds": (
            probe.GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS
        ),
        "utilization_semantics": "observational_only",
        "performance_promotion_eligible": False,
        "claim_scope": "correctness_plumbing_and_failure_semantics_only",
    }
    observations = []
    for triad_index, order in enumerate(probe.TRIAD_ORDERS):
        for arm in order:
            observations.append(
                _observation(probe, plan, triad_index, arm, step_seconds=10.0)
            )
    decision = probe.decide_provider(plan, observations)
    assert decision["status"] == "decided_non_promoting"
    assert decision["performance_promotion_eligible"] is False
    assert decision["performance_claims_allowed"] is False
    assert decision["recommended_default"] == "R"
    assert decision["recommended_provider_mode"] == "synchronous"
    assert {row["status"] for row in decision["candidate_dispositions"].values()} == {
        "nonpromotional_shared_execution"
    }


def test_historical_v3_plan_schema_is_never_executable(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    historical = deepcopy(plan)
    historical["schema"] = probe.HISTORICAL_PLAN_SCHEMA_V3
    historical = _refinalize(probe, historical, "plan_sha256")
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe.validate_plan(historical)
    assert caught.value.code == "wave5.schema"

    preserved = probe.load_historical_r2_plan()
    assert preserved == {
        "status": "historical_non_executable",
        "schema": probe.HISTORICAL_PLAN_SCHEMA_V3,
        "path": str(probe.HISTORICAL_R2_PLAN_PATH),
        "file_sha256": probe.HISTORICAL_R2_PLAN_FILE_SHA256,
        "plan_sha256": "07b113189189b61a3d9c3c80e83c84db0170848a189b07d2c2eb55d73f2b70bf",
    }
    copied = tmp_path / "copied-r2-plan.json"
    copied.write_bytes(probe.HISTORICAL_R2_PLAN_PATH.read_bytes())
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe.load_historical_r2_plan(copied)
    assert caught.value.code == "wave5.historical_plan"


def test_gpu_uuid_binding_replay_rejects_foreign_multiple_and_replacement(probe):
    expected = [f"GPU-{index}" for index in range(probe.WORLD_SIZE)]
    first_raw = [
        {
            "gpu_uuid": uuid,
            "pid": 900_000 + index,
            "process_name": f"python-rank-{index}",
        }
        for index, uuid in enumerate(expected)
    ]
    first, bindings = probe._replay_gpu_process_ownership_sample(
        first_raw,
        expected_gpu_uuids=expected,
        prior_bindings={},
        field="sample[0]",
    )
    assert first["status"] == "passed"
    assert first["compute_apps"] == first_raw
    assert first["bound_compute_apps"] == first_raw

    absent, bindings_after_absence = probe._replay_gpu_process_ownership_sample(
        [],
        expected_gpu_uuids=expected,
        prior_bindings=bindings,
        field="sample[1]",
    )
    assert absent["status"] == "passed"
    assert absent["compute_apps"] == []
    assert bindings_after_absence == bindings

    cases = [
        (
            [
                {
                    "gpu_uuid": "GPU-foreign",
                    "pid": 42,
                    "process_name": "foreign",
                }
            ],
            "unexpected_gpu_uuid",
        ),
        (
            [
                {
                    "gpu_uuid": expected[0],
                    "pid": first_raw[0]["pid"],
                    "process_name": first_raw[0]["process_name"],
                },
                {
                    "gpu_uuid": expected[0],
                    "pid": first_raw[0]["pid"] + 1,
                    "process_name": first_raw[0]["process_name"],
                },
            ],
            "multiple_compute_apps_per_gpu_uuid",
        ),
        (
            [{**first_raw[0], "pid": first_raw[0]["pid"] + 1}],
            "compute_app_binding_replaced",
        ),
        (
            [{**first_raw[0], "process_name": "python-replaced"}],
            "compute_app_binding_replaced",
        ),
    ]
    for raw, violation_code in cases:
        sample, replayed_bindings = probe._replay_gpu_process_ownership_sample(
            raw,
            expected_gpu_uuids=expected,
            prior_bindings=bindings,
            field=violation_code,
        )
        assert sample["status"] == "failed"
        assert sample["violation_codes"] == [violation_code]
        assert sample["foreign_compute_apps"] == raw
        assert replayed_bindings == bindings


def test_gpu_uuid_binding_sequence_requires_complete_final_replay(probe):
    expected = [f"GPU-{index}" for index in range(probe.WORLD_SIZE)]
    raw = [
        {
            "gpu_uuid": uuid,
            "pid": 800_000 + index,
            "process_name": f"python-rank-{index}",
        }
        for index, uuid in enumerate(expected[:-1])
    ]
    sample, bindings = probe._replay_gpu_process_ownership_sample(
        raw,
        expected_gpu_uuids=expected,
        prior_bindings={},
        field="sample[0]",
    )
    final, _ = probe._replay_gpu_process_ownership_sample(
        [],
        expected_gpu_uuids=expected,
        prior_bindings=bindings,
        field="final",
    )
    with pytest.raises(probe.Wave5BenchmarkError, match="all eight"):
        probe._validate_gpu_process_ownership_sequence(
            [sample],
            final_sample=final,
            expected_gpu_uuids=expected,
            require_complete=True,
            field="arm_process_lifetime",
        )

    complete_raw = raw + [
        {
            "gpu_uuid": expected[-1],
            "pid": 800_007,
            "process_name": "python-rank-7",
        }
    ]
    complete, complete_bindings = probe._replay_gpu_process_ownership_sample(
        complete_raw,
        expected_gpu_uuids=expected,
        prior_bindings={},
        field="sample[0]",
    )
    complete_final, _ = probe._replay_gpu_process_ownership_sample(
        [],
        expected_gpu_uuids=expected,
        prior_bindings=complete_bindings,
        field="final",
    )
    mutated_final = deepcopy(complete_final)
    mutated_final["compute_apps"] = [
        {"gpu_uuid": "GPU-foreign", "pid": 1, "process_name": "mutated"}
    ]
    with pytest.raises(probe.Wave5BenchmarkError, match="replay"):
        probe._validate_gpu_process_ownership_sequence(
            [complete],
            final_sample=mutated_final,
            expected_gpu_uuids=expected,
            require_complete=True,
            field="arm_process_lifetime",
        )


@pytest.mark.parametrize(
    ("observed_gpu_count", "expected_error_code"),
    [(8, None), (7, "wave5.gpu_process_ownership")],
    ids=("complete-host-pids", "incomplete-host-pids"),
)
def test_r2_host_local_pid_mismatch_uses_complete_uuid_binding(
    probe, tmp_path, monkeypatch, observed_gpu_count, expected_error_code
):
    plan = _plan(probe, tmp_path, publish=True)
    attempt = probe.build_attempt(plan)
    probe.publish_json_absent(plan["artifact_targets"]["attempt"], attempt)
    _gpu_baseline_fixture(probe, plan)
    targets = probe._arm_targets(plan, 0, "R")
    Path(targets["directory"]).mkdir(parents=True)
    idle = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_total_bytes": probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "memory_headroom_bytes": probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "memory_used_bytes": 0,
            "utilization_percent": 0,
        }
        for index in range(probe.WORLD_SIZE)
    ]
    events = []
    monkeypatch.setattr(probe, "_write_arm_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(probe, "_production_argv", lambda *_args, **_kwargs: ["arm"])
    monkeypatch.setattr(probe, "_production_environment", lambda _plan: {})
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(idle))
    monkeypatch.setattr(
        probe,
        "_process_tree_rss_rows",
        lambda _pid: [{"pid": 41, "rss_bytes": 1}],
    )
    monkeypatch.setattr(probe, "_artifact_tree_bytes", lambda _root: 0)
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    raw_samples = iter(
        [
            [
                {
                    "gpu_uuid": f"GPU-{index}",
                    "pid": 700_000 + index,
                    "process_name": f"python-rank-{index}",
                }
                for index in range(observed_gpu_count)
            ],
            [],
            [],
        ]
    )
    monkeypatch.setattr(
        probe,
        "_sample_nvidia_compute_apps",
        lambda: deepcopy(next(raw_samples)),
    )

    class CompletedArm:
        pid = 41
        returncode = 0

        def __init__(self):
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else self.returncode

    def popen(*_args, **_kwargs):
        assert events == []
        events.append("popen")
        return CompletedArm()

    terminated = []
    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    monkeypatch.setattr(probe, "_terminate", lambda process: terminated.append(process))

    def call():
        return probe.run_production_arm(
            plan["artifact_targets"]["plan"],
            plan["artifact_targets"]["attempt"],
            0,
            "R",
        )

    if expected_error_code is None:
        assert call() == 0
    else:
        with pytest.raises(probe.Wave5BenchmarkError) as caught:
            call()
        assert caught.value.code == expected_error_code
    assert events == ["popen"]
    assert terminated == []
    receipt = probe.load_strict_json(targets["resource_samples"])
    assert receipt["post_phase_gpu_process_sweep"]["sample_count"] == 2
    assert all(
        sample["gpu_process_ownership"]["compute_apps"] == []
        for sample in receipt["post_phase_gpu_process_sweep"]["samples"]
    )
    assert (
        len(
            receipt["post_phase_gpu_process_sweep"]["samples"][-1][
                "gpu_process_ownership"
            ]["bound_compute_apps"]
        )
        == observed_gpu_count
    )
    if expected_error_code is None:
        assert receipt["status"] == "complete"
    else:
        assert receipt["status"] == "monitor_error"
        assert receipt["monitor_error"]["code"] == expected_error_code


def test_foreign_gpu_process_appearing_late_stops_disposable_cpu_arm(
    probe, tmp_path, monkeypatch
):
    plan = _plan(probe, tmp_path, publish=True)
    attempt = probe.build_attempt(plan)
    probe.publish_json_absent(plan["artifact_targets"]["attempt"], attempt)
    _gpu_baseline_fixture(probe, plan)
    targets = probe._arm_targets(plan, 0, "R")
    Path(targets["directory"]).mkdir(parents=True)
    monkeypatch.setattr(probe, "_write_arm_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        probe,
        "_production_argv",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            "import time; time.sleep(120)",
        ],
    )
    monkeypatch.setattr(
        probe, "_production_environment", lambda _plan: os.environ.copy()
    )
    idle = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_total_bytes": probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "memory_headroom_bytes": probe.GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "memory_used_bytes": 0,
            "utilization_percent": 0,
        }
        for index in range(probe.WORLD_SIZE)
    ]
    monkeypatch.setattr(probe, "_sample_nvidia", lambda: deepcopy(idle))
    compute_app_samples = iter(
        [
            [],
            [
                {
                    "gpu_uuid": "GPU-foreign",
                    "pid": os.getpid(),
                    "process_name": "pytest-foreign-owner",
                }
            ],
            [],
            [],
        ]
    )
    monkeypatch.setattr(
        probe,
        "_sample_nvidia_compute_apps",
        lambda: deepcopy(next(compute_app_samples)),
    )

    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe.run_production_arm(
            plan["artifact_targets"]["plan"],
            plan["artifact_targets"]["attempt"],
            0,
            "R",
        )
    assert caught.value.code == "wave5.foreign_gpu_process"
    receipt = probe.load_strict_json(targets["resource_samples"])
    assert receipt["status"] == "resource_limit_exceeded"
    assert receipt["monitor_error"]["code"] == "wave5.foreign_gpu_process"
    ownership = receipt["samples"][-1]["gpu_process_ownership"]
    assert ownership["status"] == "failed"
    assert ownership["violation_codes"] == ["unexpected_gpu_uuid"]
    assert ownership["foreign_compute_apps"] == [
        {
            "gpu_uuid": "GPU-foreign",
            "pid": os.getpid(),
            "process_name": "pytest-foreign-owner",
        }
    ]
    assert all(
        sample["gpu_process_ownership"]["compute_apps"] == []
        for sample in receipt["post_phase_gpu_process_sweep"]["samples"]
    )
    assert receipt["termination"]["descendant_attestation"] == "passed"


def test_controller_classifies_foreign_gpu_process_as_nonpromoting_resource_stop(
    probe, tmp_path, monkeypatch
):
    _bind_authenticated_cpu_provenance(probe, monkeypatch)
    plan = _plan(probe, tmp_path, publish=True)
    _mock_controller_gpu_baseline(probe, plan, monkeypatch)

    def foreign_process_stop(*_args, **_kwargs):
        raise probe.Wave5BenchmarkError(
            "foreign GPU process appeared late",
            code="wave5.foreign_gpu_process",
        )

    monkeypatch.setattr(probe, "_execute_arm", foreign_process_stop)
    with pytest.raises(probe.Wave5BenchmarkError) as caught:
        probe.run_controller(plan["artifact_targets"]["plan"])
    assert caught.value.code == "wave5.foreign_gpu_process"
    terminal = probe.load_strict_json(plan["artifact_targets"]["terminal"])
    decision = probe.load_strict_json(plan["artifact_targets"]["decision"])
    assert terminal["status"] == "resource-stop"
    assert decision["status"] == "complete_non_promoting"
    assert decision["recommended_default"] is None
    assert {row["status"] for row in decision["candidate_dispositions"].values()} == {
        "resource-stop"
    }


def test_in_run_resource_breach_hard_stops_process(probe, monkeypatch):
    stopped = []
    process = object()
    monkeypatch.setattr(probe, "_terminate", lambda received: stopped.append(received))
    breach = {
        "code": "host_rss_all_rank_ceiling",
        "observed": probe.HOST_RSS_ALL_RANK_CEILING_BYTES + 1,
        "ceiling": probe.HOST_RSS_ALL_RANK_CEILING_BYTES,
    }
    with pytest.raises(probe.Wave5BenchmarkError, match="resource ceiling"):
        probe._stop_for_resource_breach(process, breach)
    assert stopped == [process]


def test_allocator_rank_ceiling_is_inclusive_and_enforced(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    _production_fixture(probe, plan, 0, "R")
    targets = probe._arm_targets(plan, 0, "R")
    logging_path = Path(targets["run_dir"]) / "logging.jsonl"
    rows = [json.loads(line) for line in logging_path.read_text().splitlines()]
    for row in rows:
        for rank_row in row["per_rank_measurement"].values():
            rank_row["resource/gpu_max_memory_allocated_bytes"] = float(
                probe.GPU_MEMORY_CEILING_BYTES
            )
            rank_row["resource/gpu_max_memory_reserved_bytes"] = float(
                probe.GPU_MEMORY_CEILING_BYTES
            )
    logging_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    observation = probe.derive_observation(plan, triad_index=0, arm="R")
    assert observation["resources"]["summary"][
        "gpu_allocator_max_reserved_bytes_per_rank_max"
    ] == float(probe.GPU_MEMORY_CEILING_BYTES)

    rejected_plan = _plan(probe, tmp_path / "rejected")
    _production_fixture(probe, rejected_plan, 0, "R")
    rejected_targets = probe._arm_targets(rejected_plan, 0, "R")
    rejected_log = Path(rejected_targets["run_dir"]) / "logging.jsonl"
    rejected_rows = [json.loads(line) for line in rejected_log.read_text().splitlines()]
    rejected_rows[2]["per_rank_measurement"]["4"][
        "resource/gpu_max_memory_reserved_bytes"
    ] = float(probe.GPU_MEMORY_CEILING_BYTES + 1)
    rejected_log.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected_rows),
        encoding="utf-8",
    )
    with pytest.raises(probe.Wave5BenchmarkError, match="resource"):
        probe.derive_observation(rejected_plan, triad_index=0, arm="R")


def test_controller_preserves_arm_resource_stop_code(probe, tmp_path, monkeypatch):
    plan = _plan(probe, tmp_path)
    targets = probe._arm_targets(plan, 0, "R")
    resource = probe._finalize(
        {
            "schema": probe.RESOURCE_SCHEMA,
            "status": "preflight_error",
            "production_argv": probe._production_argv(plan, 0, "R"),
            "production_argv_sha256": probe.sha256_json(
                probe._production_argv(plan, 0, "R")
            ),
            "elapsed_seconds": 0.1,
            "gpu_execution_baseline": None,
            "resource_breach": None,
            "monitor_error": {
                "type": "Wave5BenchmarkError",
                "code": "wave5.gpu_busy",
                "message": "selected GPU is above 1 GiB",
            },
            "termination": None,
            "samples": [],
            "post_phase_gpu_process_sweep": None,
        },
        hash_field="resource_sha256",
    )
    _write_json(Path(targets["resource_samples"]), resource)

    class FailedArm:
        @staticmethod
        def wait(*, timeout):
            assert timeout > 0
            return 1

    monkeypatch.setattr(probe.subprocess, "Popen", lambda *args, **kwargs: FailedArm())
    with pytest.raises(probe.Wave5BenchmarkError) as exc_info:
        probe._execute_arm(plan, 0, "R", timeout_seconds=1.0)
    assert exc_info.value.code == "wave5.gpu_busy"


def test_per_rank_rss_ceiling_applies_to_legacy_and_all_provider_arms(probe):
    gpu_rows = [
        {"gpu_index": rank, "memory_used_bytes": 0, "utilization_percent": 0}
        for rank in range(probe.WORLD_SIZE)
    ]
    for arm in ("R", "D", "O"):
        breach = probe._resource_limit_breach(
            arm=arm,
            gpu_rows=gpu_rows,
            process_rss_rows=[
                {
                    "pid": 100 + rank,
                    "rss_bytes": (
                        probe.HOST_RSS_PER_RANK_CEILING_BYTES + 1 if rank == 4 else 1
                    ),
                }
                for rank in range(probe.WORLD_SIZE)
            ],
            artifact_bytes=0,
        )
        assert breach == {
            "code": "host_rss_per_rank_ceiling",
            "arm": arm,
            "pid": 104,
            "observed": probe.HOST_RSS_PER_RANK_CEILING_BYTES + 1,
            "ceiling": probe.HOST_RSS_PER_RANK_CEILING_BYTES,
        }


def test_termination_covers_nested_production_process_group(probe, monkeypatch):
    signals = []
    protected_baseline_driver_pids = {row["pid"] for row in _baseline_apps(probe)}

    class Process:
        pid = 101

        @staticmethod
        def wait(*, timeout):
            assert timeout == probe.TERMINATION_GRACE_SECONDS
            return 0

    monkeypatch.setattr(probe, "_descendant_process_groups", lambda _pid: {101, 202})
    monkeypatch.setattr(
        probe.os, "killpg", lambda pgid, signum: signals.append((pgid, signum))
    )
    monkeypatch.setattr(probe, "_live_descendant_pids", lambda _pid: set())
    receipt = probe._terminate(Process())
    assert signals == [(101, probe.signal.SIGTERM), (202, probe.signal.SIGTERM)]
    assert {pgid for pgid, _signum in signals}.isdisjoint(
        protected_baseline_driver_pids
    )
    assert receipt["descendant_attestation"] == "passed"
    assert receipt["descendants_remaining"] == []


def test_termination_kills_disposable_cpu_process_tree_and_attests_no_descendants(
    probe,
):
    code = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c',"
        "\"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); time.sleep(120)\"]); "
        "print(child.pid, flush=True); time.sleep(120)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline().strip())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not probe._live_descendant_pids(process.pid):
        time.sleep(0.01)
    receipt = probe._terminate(process)
    assert receipt["descendant_attestation"] == "passed"
    assert receipt["descendants_remaining"] == []
    assert process.poll() is not None
    assert not Path(f"/proc/{child_pid}").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "provider",
        "provider_policy",
        "phase",
        "phase_status",
        "eval",
        "eval_exact",
        "checkpoint",
        "checkpoint_payload",
        "special_token_payload",
        "config",
        "train_duplicate",
    ],
)
def test_artifact_derivation_mutations_fail_closed(probe, tmp_path, mutation):
    plan = _plan(probe, tmp_path)
    _production_fixture(probe, plan, 1, "D")
    targets = probe._arm_targets(plan, 1, "D")
    run_dir = Path(targets["run_dir"])
    if mutation == "provider":
        run = probe.load_strict_json(run_dir / "run.json")
        run["forward_input_provider_mode"] = "synchronous"
        _write_json(run_dir / "run.json", run)
    elif mutation == "provider_policy":
        run = probe.load_strict_json(run_dir / "run.json")
        run["policy_identities"]["input_provider"]["resolved_mode"] = "overlapped"
        _write_json(run_dir / "run.json", run)
    elif mutation == "phase":
        run = probe.load_strict_json(run_dir / "run.json")
        run["measurement"]["phases"].pop("model_loading")
        _write_json(run_dir / "run.json", run)
    elif mutation == "phase_status":
        run = probe.load_strict_json(run_dir / "run.json")
        run["measurement"]["phases"]["evaluation_execution"]["status"] = "not_run"
        _write_json(run_dir / "run.json", run)
    elif mutation == "eval":
        rows = [
            json.loads(line)
            for line in (run_dir / "logging.jsonl").read_text().splitlines()
        ]
        [row for row in rows if row["split"] == "eval"][0]["example_count"] = 0
        (run_dir / "logging.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    elif mutation == "checkpoint":
        _write_json(
            run_dir / "checkpoints/final.json",
            {"step": 4, "checkpoint_path": "checkpoints/step-5"},
        )
    elif mutation == "eval_exact":
        rows = [
            json.loads(line)
            for line in (run_dir / "logging.jsonl").read_text().splitlines()
        ]
        [row for row in rows if row["split"] == "eval"][0]["example_count"] = 63
        (run_dir / "logging.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    elif mutation == "checkpoint_payload":
        (run_dir / "checkpoints/step-5/adapter/adapter_config.json").unlink()
    elif mutation == "special_token_payload":
        (
            run_dir
            / "checkpoints/step-5/special_token_embeddings/special_token_embeddings.json"
        ).unlink()
    elif mutation == "train_duplicate":
        rows = [
            json.loads(line)
            for line in (run_dir / "logging.jsonl").read_text().splitlines()
        ]
        train = [row for row in rows if row["split"] == "train"]
        train[-1]["step"] = 4
        eval_row = [row for row in rows if row["split"] == "eval"]
        (run_dir / "logging.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in [*train, *eval_row])
        )
    else:
        resolved = probe.load_strict_json(run_dir / "resolved_config.json")
        resolved["config"]["training"]["max_steps"] = 4
        _write_json(run_dir / "resolved_config.json", resolved)
    with pytest.raises(probe.Wave5BenchmarkError):
        probe.derive_observation(plan, triad_index=1, arm="D")


def test_decision_uses_three_paired_triads_strict_parity_and_noise(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    seconds = {"R": 10.0, "D": 10.5, "O": 8.0}
    observations = [
        _observation(probe, plan, index, arm, seconds[arm] + index * 0.01)
        for index, order in enumerate(probe.TRIAD_ORDERS)
        for arm in order
    ]
    decision = probe.decide_provider(plan, observations)
    assert decision["semantic_equivalence_passed"] is True
    assert decision["candidate_dispositions"]["O"]["paired_observation_count"] == 3
    assert decision["candidate_dispositions"]["O"]["status"] == "experimental"
    assert decision["recommended_default"] == "O"
    summary = decision["measurement_summary"]
    assert set(summary["per_arm"]) == {"R", "D", "O"}
    assert summary["per_arm"]["R"]["steady_state_seconds"]["count"] == 3
    assert set(summary["per_arm"]["R"]["steady_state_seconds"]) == {
        "count",
        "values",
        "median",
        "mad",
        "minimum",
        "maximum",
        "range",
    }
    assert len(summary["paired_comparisons"]["O"]["triads"]) == 3
    assert (
        summary["paired_comparisons"]["O"]["summaries"]["steady_state_delta_seconds"][
            "count"
        ]
        == 3
    )
    assert summary["failure_summary"] == {
        "expected_arm_executions": 9,
        "successful_observations": 9,
        "failed_observations": 0,
        "arm_failures": [],
    }
    observations[-1]["semantic"]["loss_results_sha256"] = "0" * 64
    observations[-1] = _refinalize(probe, observations[-1], "observation_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="not derived"):
        probe.decide_provider(plan, observations)


def test_promotion_requires_both_steady_and_entry_to_terminal_gain_beyond_noise(
    probe, tmp_path
):
    plan = _plan(probe, tmp_path)
    assert plan["measurement_contract"]["noise_rule"] == (
        "both_median_paired_steady_state_and_entry_to_terminal_gain_gt_"
        "max(0.05,2*respective_mad)"
    )
    observations = []
    for index, order in enumerate(probe.TRIAD_ORDERS):
        for arm in order:
            observations.append(
                _observation(
                    probe,
                    plan,
                    index,
                    arm,
                    {"R": 10.0, "D": 10.5, "O": 9.9}[arm],
                    entry_to_terminal_seconds={
                        "R": 100.0,
                        "D": 105.0,
                        "O": 80.0,
                    }[arm],
                )
            )
    decision = probe.decide_provider(plan, observations)
    candidate = decision["candidate_dispositions"]["O"]
    assert candidate["clears_frozen_entry_to_terminal_noise"] is True
    assert candidate["clears_frozen_steady_state_noise"] is False
    assert candidate["entry_to_terminal_nonregression_passed"] is True
    assert candidate["status"] == "inconclusive"
    assert decision["recommended_default"] == "R"


def test_noise_uses_mad_of_paired_gains_not_absolute_reference_times(probe, tmp_path):
    plan = _plan(probe, tmp_path)
    reference = [10.0, 20.0, 30.0]
    observations = []
    for index, order in enumerate(probe.TRIAD_ORDERS):
        for arm in order:
            factor = {"R": 1.0, "D": 1.1, "O": 0.8}[arm]
            observations.append(
                _observation(probe, plan, index, arm, reference[index] * factor)
            )
    decision = probe.decide_provider(plan, observations)
    assert decision["candidate_dispositions"]["O"]["paired_gain_mad_fraction"] < 1e-4
    assert decision["candidate_dispositions"]["O"]["frozen_noise_fraction"] == 0.05
    assert decision["candidate_dispositions"]["O"]["status"] == "experimental"
    assert decision["recommended_default"] == "O"


def test_decision_rejects_over_two_percent_entry_to_terminal_regression(
    probe, tmp_path
):
    plan = _plan(probe, tmp_path)
    observations = []
    for index, order in enumerate(probe.TRIAD_ORDERS):
        for arm in order:
            observations.append(
                _observation(
                    probe,
                    plan,
                    index,
                    arm,
                    {"R": 10.0, "D": 10.5, "O": 8.0}[arm],
                    entry_to_terminal_seconds={"R": 100.0, "D": 100.0, "O": 103.0}[arm],
                )
            )
    decision = probe.decide_provider(plan, observations)
    candidate = decision["candidate_dispositions"]["O"]
    assert candidate["clears_frozen_steady_state_noise"] is True
    assert candidate["clears_frozen_entry_to_terminal_noise"] is False
    assert candidate["clears_frozen_noise"] is False
    assert candidate["median_entry_to_terminal_regression_fraction"] == pytest.approx(
        0.03
    )
    assert candidate["entry_to_terminal_nonregression_passed"] is False
    assert candidate["status"] == "rejected"
    assert decision["recommended_default"] == "R"


@pytest.mark.parametrize(
    ("candidate_e2e", "expected_status"),
    [(99.0, "inconclusive"), (101.0, "rejected")],
)
def test_promotion_is_owned_by_paired_e2e_gain_beyond_frozen_noise(
    probe, tmp_path, candidate_e2e, expected_status
):
    plan = _plan(probe, tmp_path)
    observations = []
    for index, order in enumerate(probe.TRIAD_ORDERS):
        for arm in order:
            observations.append(
                _observation(
                    probe,
                    plan,
                    index,
                    arm,
                    {"R": 10.0, "D": 10.5, "O": 8.0}[arm],
                    entry_to_terminal_seconds={
                        "R": 100.0,
                        "D": 105.0,
                        "O": candidate_e2e,
                    }[arm],
                )
            )
    decision = probe.decide_provider(plan, observations)
    candidate = decision["candidate_dispositions"]["O"]
    assert candidate["clears_frozen_steady_state_noise"] is True
    assert candidate["clears_frozen_entry_to_terminal_noise"] is False
    assert candidate["status"] == expected_status
    assert decision["recommended_default"] == "R"


def test_cpu_fake_controller_publishes_authenticated_chain(
    probe, tmp_path, monkeypatch
):
    _bind_authenticated_cpu_provenance(probe, monkeypatch)
    plan = _plan(probe, tmp_path, publish=True)
    _mock_controller_gpu_baseline(probe, plan, monkeypatch)

    def fake_execute(received_plan, triad_index, arm, *, timeout_seconds):
        assert probe._arm_argv(received_plan, triad_index, arm)[2] == "arm"
        assert timeout_seconds > 0
        _production_fixture(
            probe,
            received_plan,
            triad_index,
            arm,
            step_seconds={"R": 10.0, "D": 10.5, "O": 8.0}[arm],
        )

    monkeypatch.setattr(probe, "_execute_arm", fake_execute)
    terminal = probe.run_controller(plan["artifact_targets"]["plan"])
    assert terminal["status"] == "completed"
    baseline = probe.load_authenticated_gpu_baseline(plan)
    assert terminal["gpu_baseline_sha256"] == baseline["gpu_baseline_sha256"]
    assert terminal["terminal_gpu_process_sweep"]["status"] == "passed"
    assert terminal["terminal_gpu_process_sweep"]["sample_count"] == 2
    assert all(
        sample["gpu_process_ownership"]["compute_apps"] == []
        for sample in terminal["terminal_gpu_process_sweep"]["samples"]
    )
    assert len(terminal["observation_sha256"]) == 9
    assert probe.validate_terminal(terminal, plan=plan) == terminal
    terminal_mutations = []
    for sample_count in (0, 1):
        mutated = deepcopy(terminal)
        mutated["terminal_gpu_process_sweep"]["samples"] = mutated[
            "terminal_gpu_process_sweep"
        ]["samples"][:sample_count]
        mutated["terminal_gpu_process_sweep"]["sample_count"] = sample_count
        terminal_mutations.append(mutated)
    too_close = deepcopy(terminal)
    too_close["terminal_gpu_process_sweep"]["samples"][1]["monotonic_ns"] = (
        too_close["terminal_gpu_process_sweep"]["samples"][0]["monotonic_ns"]
        + 1_000_000_000
    )
    terminal_mutations.append(too_close)
    for mutated in terminal_mutations:
        mutated = _refinalize(probe, mutated, "terminal_sha256")
        with pytest.raises(probe.Wave5BenchmarkError):
            probe.validate_terminal(mutated, plan=plan)
    assert Path(plan["artifact_targets"]["decision"]).exists()


def test_post_matrix_decision_exception_still_publishes_terminal_receipt(
    probe, tmp_path, monkeypatch
):
    _bind_authenticated_cpu_provenance(probe, monkeypatch)
    plan = _plan(probe, tmp_path, publish=True)
    _mock_controller_gpu_baseline(probe, plan, monkeypatch)

    def fake_execute(received_plan, triad_index, arm, *, timeout_seconds):
        _production_fixture(
            probe,
            received_plan,
            triad_index,
            arm,
            step_seconds={"R": 10.0, "D": 10.5, "O": 8.0}[arm],
        )

    monkeypatch.setattr(probe, "_execute_arm", fake_execute)
    monkeypatch.setattr(
        probe,
        "decide_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            probe.Wave5BenchmarkError("decision injected", code="test.decision")
        ),
    )
    with pytest.raises(probe.Wave5BenchmarkError, match="decision injected"):
        probe.run_controller(plan["artifact_targets"]["plan"])
    terminal = probe.load_strict_json(plan["artifact_targets"]["terminal"])
    assert terminal["status"] == "failed"
    assert len(terminal["observation_sha256"]) == 9
    assert probe.validate_terminal(terminal, plan=plan) == terminal
    decision = probe.load_strict_json(plan["artifact_targets"]["decision"])
    assert decision["status"] == "complete_non_promoting"
    assert decision["declared_stop"]["stage"] == "decision"
    assert decision["recommended_default"] == "R"


def test_terminal_deep_loads_observations_and_decision(probe, tmp_path, monkeypatch):
    _bind_authenticated_cpu_provenance(probe, monkeypatch)
    plan = _plan(probe, tmp_path, publish=True)
    _mock_controller_gpu_baseline(probe, plan, monkeypatch)

    def fake_execute(received_plan, triad_index, arm, *, timeout_seconds):
        _production_fixture(
            probe,
            received_plan,
            triad_index,
            arm,
            step_seconds={"R": 10.0, "D": 10.5, "O": 8.0}[arm],
        )

    monkeypatch.setattr(probe, "_execute_arm", fake_execute)
    terminal = probe.run_controller(plan["artifact_targets"]["plan"])
    first_target = Path(
        probe._arm_targets(plan, 0, probe.TRIAD_ORDERS[0][0])["observation"]
    )
    original_observation = probe.load_strict_json(first_target)
    tampered = deepcopy(original_observation)
    tampered["status"] = "forged"
    tampered = _refinalize(probe, tampered, "observation_sha256")
    _write_json(first_target, tampered)
    with pytest.raises(probe.Wave5BenchmarkError):
        probe.validate_terminal(terminal, plan=plan)

    _write_json(first_target, original_observation)
    decision_path = Path(plan["artifact_targets"]["decision"])
    decision = probe.load_strict_json(decision_path)
    decision["recommended_default"] = "D"
    decision = _refinalize(probe, decision, "decision_sha256")
    _write_json(decision_path, decision)
    with pytest.raises(probe.Wave5BenchmarkError):
        probe.validate_terminal(terminal, plan=plan)


def test_failed_terminal_requires_typed_error_and_observation_prefix(
    probe, tmp_path, monkeypatch
):
    _bind_authenticated_cpu_provenance(probe, monkeypatch)
    plan = _plan(probe, tmp_path, publish=True)
    _mock_controller_gpu_baseline(probe, plan, monkeypatch)
    calls = []

    def fail_second(received_plan, triad_index, arm, *, timeout_seconds):
        calls.append((triad_index, arm))
        if len(calls) == 2:
            raise probe.Wave5BenchmarkError("injected", code="test.injected")
        _production_fixture(probe, received_plan, triad_index, arm, step_seconds=10.0)

    monkeypatch.setattr(probe, "_execute_arm", fail_second)
    with pytest.raises(probe.Wave5BenchmarkError, match="injected"):
        probe.run_controller(plan["artifact_targets"]["plan"])
    terminal = probe.load_strict_json(plan["artifact_targets"]["terminal"])
    assert probe.validate_terminal(terminal, plan=plan) == terminal
    assert terminal["status"] == "failed"
    assert len(terminal["observation_sha256"]) == 1
    decision = probe.load_strict_json(plan["artifact_targets"]["decision"])
    assert terminal["decision_sha256"] == decision["decision_sha256"]
    assert decision["status"] == "complete_non_promoting"
    assert decision["declared_stop"]["category"] == "failure"
    assert decision["recommended_default"] == "R"
    assert {item["status"] for item in decision["candidate_dispositions"].values()} <= {
        "rejected",
        "inconclusive",
        "experimental",
        "resource-stop",
    }

    oversized = deepcopy(terminal)
    oversized["error"]["message"] = "x" * 4097
    oversized = _refinalize(probe, oversized, "terminal_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="error"):
        probe.validate_terminal(oversized, plan=plan)

    terminal["error"] = None
    terminal = _refinalize(probe, terminal, "terminal_sha256")
    with pytest.raises(probe.Wave5BenchmarkError, match="error"):
        probe.validate_terminal(terminal, plan=plan)


def test_incomplete_matrix_requires_declared_stop_and_resource_stop_is_nonpromoting(
    probe, tmp_path
):
    plan = _plan(probe, tmp_path)
    observation = _observation(probe, plan, 0, "R", 10.0)
    with pytest.raises(probe.Wave5BenchmarkError, match="incomplete"):
        probe.decide_provider(plan, [observation])
    stop = {
        "category": "resource-stop",
        "stage": "arm_execution",
        "completed_evidence": {},
        "error": {
            "type": "Wave5BenchmarkError",
            "code": "wave5.resource_ceiling",
            "message": "rank 4 crossed RSS ceiling",
        },
    }
    decision = probe.decide_provider(plan, [observation], declared_stop=stop)
    assert decision["status"] == "complete_non_promoting"
    assert decision["recommended_default"] == "R"
    assert {item["status"] for item in decision["candidate_dispositions"].values()} == {
        "resource-stop"
    }
    assert (
        decision["measurement_summary"]["failure_summary"]["successful_observations"]
        == 1
    )


def test_resource_stop_before_any_observation_never_passes_resource_gate(
    probe, tmp_path
):
    plan = _plan(probe, tmp_path)
    stop = {
        "category": "resource-stop",
        "stage": "arm_execution",
        "completed_evidence": {},
        "error": {
            "type": "Wave5BenchmarkError",
            "code": "wave5.gpu_busy",
            "message": "preflight inventory changed",
        },
    }
    decision = probe.decide_provider(plan, [], declared_stop=stop)
    assert decision["status"] == "complete_non_promoting"
    assert decision["resource_gate_passed"] is False
    assert decision["recommended_default"] is None


def test_true_terminal_publication_failure_gets_sidecar(probe, tmp_path, monkeypatch):
    _bind_authenticated_cpu_provenance(probe, monkeypatch)
    plan = _plan(probe, tmp_path, publish=True)
    _mock_controller_gpu_baseline(probe, plan, monkeypatch)

    def fake_execute(received_plan, triad_index, arm, *, timeout_seconds):
        _production_fixture(probe, received_plan, triad_index, arm, step_seconds=10.0)

    original_publish = probe.publish_json_absent

    def fail_terminal(path, payload):
        if str(path) == plan["artifact_targets"]["terminal"]:
            raise probe.Wave5BenchmarkError(
                "injected terminal publication failure", code="test.publish"
            )
        return original_publish(path, payload)

    monkeypatch.setattr(probe, "_execute_arm", fake_execute)
    monkeypatch.setattr(probe, "publish_json_absent", fail_terminal)
    with pytest.raises(probe.Wave5BenchmarkError, match="injected"):
        probe.run_controller(plan["artifact_targets"]["plan"])
    sidecar = probe.load_strict_json(
        plan["artifact_targets"]["terminal_publication_failure"]
    )
    assert sidecar["status"] == "terminal_publication_failed"
    assert (
        sidecar["attempt_sha256"]
        == probe.load_strict_json(plan["artifact_targets"]["attempt"])["attempt_sha256"]
    )
