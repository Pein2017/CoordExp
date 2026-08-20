from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, replace
from functools import wraps
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import threading
from types import MethodType, SimpleNamespace
import sys

import pytest
import torch

import src.artifacts.identity as identity_module
import src.qwen.parity as parity_module
from src.losses import SegmentBalancedDenominator
from src.losses.runner import PlannedStepLossPlan
from src.qwen.fa2 import Fa2VarlenPlan
from src.qwen.images import QwenImageEncoding, QwenNoResizeImagePlan
from src.qwen.parity import (
    CONFIG_COMPATIBILITY_PROJECTION_SCHEMA,
    FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
    FROZEN_PARENT_V2_MODEL_WEIGHT_SHA256,
    FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT,
    GradientRecord,
    MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA,
    MODEL_WEIGHT_IDENTITY_SCHEMA,
    PARITY_FAILURE_EVIDENCE_SCHEMA,
    PARITY_ATTEMPT_MARKER_SCHEMA,
    PARITY_PLAN_SCHEMA,
    PARITY_RECEIPT_SCHEMA,
    ParityContractError,
    SemanticAtomKey,
    assert_dependency_provenance_equal,
    assert_model_weight_identity_equal,
    assert_plan_revalidated,
    attest_qwen_component_identity,
    base_model_weight_identity,
    base_model_weight_identity_with_execution_policy,
    compare_gradient_inventories,
    compare_packed_gradient_repeat,
    compare_keyed_logits,
    compare_semantic_atom_inventories,
    compare_shared_denominators,
    config_compatibility_projection,
    finalize_plan,
    finalize_attempt_marker,
    frozen_parent_v2_identity,
    frozen_trainable_inventory_declaration,
    frozen_tolerances,
    load_strict_json,
    merged_boundary_forward_inputs,
    negative_discriminator,
    repo_identity,
    selected_logits_by_semantic_key,
    semantic_atom_inventory,
    semantic_atom_key_inventory_sha256,
    sha256_json,
    validate_dependency_provenance,
    validate_denominator_comparison_artifact,
    validate_measurement_contract,
    validate_model_weight_identity,
    validate_parity_plan,
    validate_parity_receipt,
    validate_config_compatibility_projection,
    validate_v3_config_identity,
    validate_cross_arm_bf16_loss_term_scalars,
    validate_concrete_trainable_inventory,
    write_strict_json_atomic,
)


@pytest.fixture(autouse=True)
def _wave2_flash_attention_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")


def _load_probe_module():
    name = "coordexp_wave2_packed_parity_test_module"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/probes/coordexp_swift/wave2_packed_parity.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Atom:
    example_id: str
    logical_target_position: int
    logical_target_end: int
    token_id: int
    token_type: str = "coordinate"
    object_id: str | None = "object-1"
    field: str | None = "bbox"
    source: str | None = "target"


@dataclass(frozen=True)
class _Sequence:
    atoms: tuple[_Atom, ...]


@dataclass(frozen=True)
class _Receipt:
    fa2_varlen_plan: Fa2VarlenPlan
    fa2_branch_proof_policy: str | None = None


@dataclass(frozen=True)
class _Inputs:
    pack_index: int
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    fa2_varlen_plan: Fa2VarlenPlan
    receipt: _Receipt
    logits_to_keep: int = 0
    logits_position_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _EncodedWithImage:
    example_id: str
    image_encoding: QwenImageEncoding

    def to_artifact_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "image_encoding": self.image_encoding.to_artifact_dict(),
        }


class _Context:
    def __init__(self, logits: torch.Tensor, atoms: tuple[_Atom, ...]) -> None:
        self._logits = logits
        self._atoms = atoms

    def select_logits_fp32(self, *, token_types: object = None):
        del token_types
        targets = torch.tensor([atom.token_id for atom in self._atoms])
        return self._logits, targets, self._atoms


@dataclass(frozen=True)
class _ArtifactDenominator:
    artifact: dict[str, object]

    def to_artifact_dict(self) -> dict[str, object]:
        return dict(self.artifact)


def _atom(example_id: str, position: int, token_id: int = 3) -> _Atom:
    return _Atom(
        example_id=example_id,
        logical_target_position=position,
        logical_target_end=position + 1,
        token_id=token_id,
    )


def _weight_identity() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": MODEL_WEIGHT_IDENTITY_SCHEMA,
        "root": "/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent",
        "mode": "indexed_safetensors",
        "index": {
            "path": "model.safetensors.index.json",
            "size_bytes": 56_186,
            "sha256": "7ab471424a936028921a6c952e662457bb4fa6aedb1fb13232a0993a0ad97d61",
        },
        "declaration_count": 625,
        "shards": [
            {
                "path": "model-00001-of-00002.safetensors",
                "size_bytes": 4_992_656_024,
                "sha256": "2dff41296f9d817f9698bef43e31ce58fd06b5080978d31279d06f6766a695a4",
            },
            {
                "path": "model-00002-of-00002.safetensors",
                "size_bytes": 3_523_560_656,
                "sha256": "915ce8d4baabd778e76d32bd3d57c6c65c806c877a4fda47fc4b6f6f9e1510a5",
            },
        ],
        "shard_count": 2,
        "total_bytes": 8_516_216_680,
        "bounds": {
            "max_index_bytes": parity_module.MAX_WEIGHT_INDEX_BYTES,
            "max_declarations": parity_module.MAX_WEIGHT_DECLARATIONS,
            "max_shards": parity_module.MAX_WEIGHT_SHARDS,
            "max_shard_bytes": parity_module.MAX_WEIGHT_SHARD_BYTES,
            "max_total_bytes": parity_module.MAX_WEIGHT_TOTAL_BYTES,
        },
    }
    identity = {**body, "aggregate_sha256": sha256_json(body)}
    assert identity["aggregate_sha256"] == FROZEN_PARENT_V2_MODEL_WEIGHT_SHA256
    return identity


def _dependency_identity(*, binary_available: bool = True) -> dict[str, object]:
    components = {
        name: {}
        for name in (
            "ms-swift",
            "transformers",
            "flash-attn",
            "flash_attn_2_cuda",
            "torch",
            "accelerate",
            "peft",
            "tokenizers",
        )
    }
    status = "available" if binary_available else "unavailable"
    components["flash_attn_2_cuda"] = {
        "role": "runtime_dependency",
        "origin_kind": "binary",
        "imported_origin": {
            "status": status,
            **({"value": "/env/flash_attn_2_cuda.so"} if binary_available else {}),
        },
        "sha256": {
            "status": status,
            **({"value": "2" * 64} if binary_available else {}),
        },
    }
    cuda_runtime_origin = "/env/nvidia/cuda_runtime/lib/libcudart.so.12"
    components["cuda-runtime"] = {
        "distribution": "nvidia-cuda-runtime-cu12",
        "import_name": None,
        "distribution_version": {"status": "available", "value": "12.8.90"},
        "distribution_record": {
            "status": "available",
            "value": {"sha256": "3" * 64, "size_bytes": 11_369},
        },
        "role": "runtime_dependency",
        "origin_resolution": "loaded_shared_object",
        "loaded_soname": "libcudart.so.12",
        "distribution_relative_path": ("nvidia/cuda_runtime/lib/libcudart.so.12"),
        "distribution_origin": {
            "status": "available",
            "value": cuda_runtime_origin,
        },
        "imported_origin": {
            "status": "available",
            "value": cuda_runtime_origin,
        },
        "loaded_origin_matches_distribution": True,
        "origin_kind": "binary",
        "sha256": {"status": "available", "value": "4" * 64},
        "size_bytes": {"status": "available", "value": 728_800},
        "elf_build_id": {
            "status": "available",
            "value": "7b1714ea2d766ca35afe1e3dd34a75b41b78999f",
        },
        "source_repository": {
            "status": "unavailable",
            "reason": "not_source_origin",
        },
        "source_identities": {},
        "native_identities": {},
    }
    return {
        "schema": "coordexp-swift-wave2-dependency-identity-v1",
        "collector": "src.artifacts.provenance.collect_dependency_provenance",
        "collected": components,
        "accelerate_runtime_sources": [
            {
                "relative_path": relative,
                "resolved_path": f"/env/accelerate/{relative}",
                "sha256": str(index) * 64,
            }
            for index, relative in enumerate(
                (
                    "accelerator.py",
                    "state.py",
                    "utils/modeling.py",
                    "utils/operations.py",
                ),
                start=3,
            )
        ],
    }


def _runtime_patch_receipt(
    *,
    policy: str = "enabled",
    loaded: bool,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "name": "qwen3_vl_patch_embed_linearization",
        "policy": policy,
        "applied": False,
        "reason": "model_not_loaded",
        "owner_path": None,
        "original_class": None,
        "owner_class": None,
        "patched_class": None,
        "projection_class": None,
        "original_forward_sha256": None,
        "replacement_forward_sha256": None,
        "in_channels": None,
        "temporal_patch_size": None,
        "patch_size": None,
        "embed_dim": None,
        "weight_shape": None,
        "bias": None,
        "kernel_size": None,
        "stride": None,
        "padding": None,
        "dilation": None,
        "groups": None,
        "equivalence_probe": None,
    }
    if not loaded:
        return receipt
    receipt.update(
        {
            "owner_path": "model.visual.patch_embed",
            "reason": "policy_disabled",
        }
    )
    if policy == "disabled":
        return receipt
    receipt.update(
        {
            "applied": True,
            "reason": "conv3d_kernel_stride_equivalent_linear_projection",
            "original_class": "Qwen3VLVisionPatchEmbed",
            "owner_class": "Qwen3VLVisionPatchEmbed",
            "patched_class": "LinearizedQwen3VLPatchEmbed",
            "projection_class": "Conv3d",
            "original_forward_sha256": "a" * 64,
            "replacement_forward_sha256": "b" * 64,
            "in_channels": 3,
            "temporal_patch_size": 2,
            "patch_size": 14,
            "embed_dim": 8,
            "weight_shape": [8, 3, 2, 14, 14],
            "bias": True,
            "kernel_size": [2, 14, 14],
            "stride": [2, 14, 14],
            "padding": [0, 0, 0],
            "dilation": [1, 1, 1],
            "groups": 1,
            "equivalence_probe": {
                "scope": "cpu_float32_forward_and_grad_small_canary",
                "sample_count": 2,
                "input_features": 1176,
                "max_abs_diff": 0.0,
                "grad_max_abs_diff": 0.0,
            },
        }
    )
    return receipt


def _component_identity(
    *,
    load_model: bool,
    policy: str = "enabled",
) -> dict[str, object]:
    return {
        "base_model_path": "/models/qwen",
        "base_config_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "load_model": load_model,
        "attn_implementation": "flash_attention_2",
        "processor": {"processor_class": "Qwen3VLProcessor"},
        "model": {"model_type": "qwen3_vl"},
        "tokens": {"tokenizer_vocab_size": 10},
        "package_versions": {
            "tokenizers": "1",
            "torch": "2",
            "transformers": "3",
        },
        "runtime_patches": {
            "qwen3_vl_patch_embed_linearization": _runtime_patch_receipt(
                policy=policy,
                loaded=load_model,
            )
        },
    }


def _segment_denominator(
    term_name: str,
    *,
    context_count: int,
    selected_atom_count: int,
    denominator_scope: str = "planned_step",
    eligible_segment_count: int = 2,
    skipped_segment_count: int = 0,
) -> SegmentBalancedDenominator:
    return SegmentBalancedDenominator(
        term_name=term_name,
        denominator_scope=denominator_scope,
        eligible_segment_count=eligible_segment_count,
        selected_atom_count=selected_atom_count,
        skipped_segment_count=skipped_segment_count,
        context_count=context_count,
    )


def _production_loss_plan(
    context_count: int,
    *,
    denominators: dict[str, object] | None = None,
    denominator_scope: str = "planned_step",
) -> PlannedStepLossPlan:
    if denominators is None:
        denominators = {
            "base_ce": _segment_denominator(
                "base_ce", context_count=context_count, selected_atom_count=138
            ),
            "coord_gaussian_rps": _segment_denominator(
                "coord_gaussian_rps",
                context_count=context_count,
                selected_atom_count=56,
            ),
            "token_type_gate": _segment_denominator(
                "token_type_gate",
                context_count=context_count,
                selected_atom_count=138,
            ),
        }
    base = denominators["base_ce"]
    assert isinstance(base, SegmentBalancedDenominator)
    return PlannedStepLossPlan(
        denominators=denominators,
        counts={
            "count/supervised_atoms": base.selected_atom_count,
            "count/eligible_segments": base.eligible_segment_count,
            "count/skipped_segments": base.skipped_segment_count,
            "count/packs": context_count,
            "count/examples": 2,
        },
        token_type_gate_groups=("coordinate",),
        denominator_scope=denominator_scope,
        world_size=1,
        rank=0,
        backend_gradient_scale=1.0,
    )


def _normalization_preflight() -> dict[str, object]:
    return compare_shared_denominators(
        _production_loss_plan(1),
        _production_loss_plan(2),
        packed_context_count=1,
        separate_context_count=2,
    )


def _loss_artifact(*, delta: float = 0.0) -> dict[str, object]:
    base_raw = 1.0 + delta
    coord_raw = 0.25 + delta
    gate_raw = delta
    return {
        "terms": [
            {
                "name": "base_ce",
                "raw_loss": base_raw,
                "weighted_loss": base_raw * 0.75,
                "weight": 0.75,
                "segment_mean_numerator": 2.0 + delta,
                "token_weighted_diagnostic": 0.5 + delta,
            },
            {
                "name": "coord_gaussian_rps",
                "raw_loss": coord_raw,
                "weighted_loss": coord_raw,
                "weight": 1.0,
                "segment_mean_numerator": 1.0 + delta,
                "token_weighted_diagnostic": 0.25 + delta,
            },
            {
                "name": "token_type_gate",
                "raw_loss": gate_raw,
                "weighted_loss": gate_raw,
                "weight": 1.0,
                "segment_mean_numerator": 0.5 + delta,
                "token_weighted_diagnostic": 0.125 + delta,
            },
        ]
    }


def _device_sample(index: int = 0) -> dict[str, object]:
    return {
        "sample_index": index,
        "monotonic_ns": index + 1,
        "physical_index": 0,
        "uuid": "GPU-test",
        "memory_used_bytes": 1,
        "utilization_percent": 0,
    }


def _measurement(*, v3: bool = False) -> dict[str, object]:
    phase_names = (
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        *(("packed_primary", "packed_repeat") if v3 else ("packed_clean",)),
        "separate_reference",
        "negative_control",
        "comparison",
    )
    maximum_io_bytes = len(phase_names) - 1
    phases = [
        {
            "name": name,
            "start_ns": index * 10,
            "end_ns": index * 10 + 5,
            "duration_ns": 5,
            "status": "completed",
        }
        for index, name in enumerate(phase_names, start=1)
    ]
    boundary_samples = [
        {
            "phase": name,
            "monotonic_ns": index + 1,
            "host": {
                "rss_hwm_bytes": 1024,
                "io_read_bytes": index,
                "io_write_bytes": index,
            },
            "gpu": _device_sample(index),
            "torch_cuda": {
                "max_allocated_bytes": 2048,
                "max_reserved_bytes": 4096,
            },
        }
        for index, name in enumerate(phase_names)
    ]
    return {
        "schema": "coordexp-swift-wave2-measurement-v1",
        "runtime_launch_identity": {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "process_count": 1,
            "mode": "single_process_explicit_device",
            "device": "cuda:0",
            "mixed_precision": "bf16",
        },
        "gpu_idle_preflight": {
            "status": "passed",
            "requested_device": "cuda:0",
            "physical_index": 0,
            "uuid": "GPU-test",
            "memory_limit_bytes": 1024**3,
            "utilization_limit_percent": 5,
            "sampler": {
                "command": "nvidia-smi query",
                "sample_count": 3,
                "interval_seconds": 0.2,
                "cuda_visible_device_mapping": "0",
            },
            "checks": [
                {
                    "name": name,
                    "samples": [_device_sample(index) for index in range(3)],
                }
                for name in (
                    "initial_before_cpu_model_work",
                    "final_before_gpu_work",
                )
            ],
        },
        "policies": {
            "seed": 17,
            "config_fingerprint": "a" * 64,
            "packing_policy": "source_order_next_fit_two_selected_examples",
            "provider_policy": "synchronous_cpu_build_then_explicit_device_transfer",
            "attention_backend": "flash_attention_2",
            "attention_proof_policy": "bounded_first_packed_forward",
        },
        "phases": phases,
        "resources": {
            "host": {
                "rss_hwm_bytes": 1024,
                "io_read_bytes_hwm": maximum_io_bytes,
                "io_write_bytes_hwm": maximum_io_bytes,
                "source": "test",
            },
            "gpu": {
                "device": "cuda:0",
                "torch_peak_allocated_bytes": 2048,
                "torch_peak_reserved_bytes": 4096,
                "device_used_hwm_bytes": 8192,
                "source": "test",
                "device_sampler": {
                    "status": "completed",
                    "interval_seconds": 0.25,
                    "sample_count": 2,
                    "maximum_samples": 8192,
                    "hwm_memory_used_bytes": 8192,
                    "hwm_utilization_percent": 1,
                    "first_monotonic_ns": 1,
                    "last_monotonic_ns": 2,
                },
            },
            "ceilings": {
                "status": "passed",
                "host_bytes": 64 * 1024**3,
                "device_bytes": 76 * 1024**3,
                "comparison": {
                    "host_rss_below": True,
                    "torch_reserved_below": True,
                    "device_sampler_below": True,
                },
            },
            "phase_boundary_samples": boundary_samples,
        },
        "pack_utilization": {
            "pack_length": 5,
            "global_max_length": 10,
            "unused_tokens": 5,
            "utilization_ratio": 0.5,
            "segment_count": 2,
        },
        "semantic_result": {
            "clean_parity_passed": True,
            "negative_forward_signal_detected": True,
            "all_layer_proof_passed": True,
        },
        "eligibility": {
            "terminal_eligible": True,
            "steady_state_timing_eligible": False,
            "steady_state_reason": "single proof probe",
        },
        "not_applicable": {
            name: {"status": "not_applicable", "reason": "test"}
            for name in ("cache", "eval", "checkpoint", "warmup", "per_step")
        },
    }


def _v3_config_identity() -> dict[str, object]:
    projection = {
        "schema": CONFIG_COMPATIBILITY_PROJECTION_SCHEMA,
        "current_config_sha256": FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT,
        "removed_path_values": [
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
            {"path": "runtime.determinism", "value": {"mode": "legacy"}},
        ],
        "projected_config_sha256": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "parent_config_fingerprint": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "policy": "remove_exact_enumerated_later_strict_defaults",
    }
    return {
        "entry_path": "ignored.yaml",
        "fingerprint": FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT,
        "schema_version": 1,
        "loader_version": "coordexp-swift-config-v1",
        "resolved_config_sha256": FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT,
        "sources": [{"path": "ignored.yaml", "sha256": "f" * 64}],
        "compatibility_projection": projection,
        "runtime_config_attestation": {
            "schema": "coordexp-swift-wave2-runtime-config-attestation-v1",
            "status": "passed",
            "config_fingerprint": FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT,
            "field_path": "training.forward_input_provider_mode",
            "required_value": "synchronous",
            "resolved_value": "synchronous",
            "compatibility_projection_sha256": sha256_json(projection),
        },
    }


def _plan_body() -> dict[str, object]:
    example_ids = list(frozen_parent_v2_identity()["example_ids"])
    atoms = semantic_atom_inventory(
        _Sequence(
            (
                _atom(example_ids[0], 1),
                _atom(example_ids[1], 2, token_id=4),
            )
        )
    )
    return {
        "schema": PARITY_PLAN_SCHEMA,
        "status": "prepared",
        "config_identity": _v3_config_identity(),
        "repo_identity": {"head": "repo"},
        "dependency_identity": _dependency_identity(),
        "model_identity": _component_identity(load_model=False),
        "model_weight_identity": _weight_identity(),
        "parent_v2": frozen_parent_v2_identity(),
        "selection": {
            "split": "train",
            "source_indices": [0, 1],
            "example_ids": example_ids,
            "selection_policy": "explicit_source_indices_no_switching",
        },
        "samples": [{"example_id": item} for item in example_ids],
        "arms": {
            "packed_primary": {
                "example_ids": example_ids,
                "segment_boundaries": [0, 1436, 2822],
                "input_ids_sha256": "1" * 64,
                "position_ids_sha256": "2" * 64,
                "supervision_sha256": "3" * 64,
                "image_content_sha256": ["4" * 64, "5" * 64],
                "loss_wiring": "one_microstep_one_shared_denominator",
                "proof_scope": "packed_forward_only_before_backward",
            },
            "packed_repeat": {
                "example_ids": example_ids,
                "segment_boundaries": [0, 1436, 2822],
                "input_ids_sha256": "1" * 64,
                "position_ids_sha256": "2" * 64,
                "supervision_sha256": "3" * 64,
                "image_content_sha256": ["4" * 64, "5" * 64],
                "loss_wiring": "one_microstep_one_immediate_backward",
                "proof_scope": "disabled_identical_repeat",
            },
            "separate_reference": {
                "microstep_count": 2,
                "loss_wiring": "two_microsteps_two_immediate_backwards_one_initial_clear",
            },
            "packed_merged_boundary_negative": {"segment_boundaries": [0, 2822]},
        },
        "semantic_atom_inventory": list(atoms),
        "trainable_mechanism": {"adapter": "dora"},
        "trainable_inventory_declaration": frozen_trainable_inventory_declaration(),
        "determinism": {
            "seed": 17,
            "torch_manual_seed": 17,
            "cuda_manual_seed_all": 17,
            "flash_attention_deterministic": "1",
        },
        "tolerances": frozen_tolerances(),
        "loss_normalization_preflight": _normalization_preflight(),
        "source_owners": [
            {"path": path, "sha256": str(index) * 64}
            for index, path in enumerate(
                (
                    "src/qwen/parity.py",
                    "scripts/probes/coordexp_swift/wave2_packed_parity.py",
                    "src/artifacts/provenance.py",
                    "src/artifacts/resources.py",
                ),
                start=6,
            )
        ],
    }


def _passed_receipt(plan: dict[str, object]) -> dict[str, object]:
    forward_paths = {
        "unmeasured_no_proof_warmup": 1,
        "timed_proof_off": 1,
        "packed_clean": 1,
        "separate_reference": 2,
        "packed_merged_boundary_negative": 1,
    }
    autocast_row = {
        "module_type": "Qwen3VLTextAttention",
        "layer_idx": 0,
        "cuda_autocast_enabled": True,
        "cuda_autocast_dtype": "torch.bfloat16",
    }
    expected_model_identity = _component_identity(load_model=False)
    loaded_model_identity = _component_identity(load_model=True)
    loss_term_scalars = _load_probe_module()._compare_loss_terms(
        _loss_artifact(), _loss_artifact()
    )
    return {
        "schema": PARITY_RECEIPT_SCHEMA,
        "terminal_status": "passed",
        "plan_sha256": plan["plan_sha256"],
        "source_identity": {
            "config_identity": json.loads(json.dumps(plan["config_identity"])),
            "repo_identity": json.loads(json.dumps(plan["repo_identity"])),
            "dependency_identity": json.loads(json.dumps(plan["dependency_identity"])),
            "model_identity": loaded_model_identity,
            "model_weight_identity": json.loads(
                json.dumps(plan["model_weight_identity"])
            ),
            "source_owners": json.loads(json.dumps(plan["source_owners"])),
        },
        "model_identity_attestation": attest_qwen_component_identity(
            expected_model_identity,
            loaded_model_identity,
        ),
        "execution": {
            "device": "cuda:0",
            "model_dtype": "torch.bfloat16",
            "train_mode": True,
            "use_cache": False,
            "optimizer": None,
            "memory_savers": {},
            "adapter": {},
            "special_token_embeddings": {},
            "trainable_value_identity_before": ["same"],
            "trainable_value_identity_after": ["same"],
            "accelerator": {
                "distributed_type": "NO",
                "rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "device": "cuda:0",
                "cuda_current_device": 0,
                "mixed_precision": "bf16",
                "native_amp": True,
                "gradient_accumulation_steps": 1,
                "scaler": None,
                "accelerate_torch_device": "cuda:0",
                "prepared_model_attestation": {
                    "binding_branch": "bound_method",
                    "prepared_forward_type": "method",
                    "wrapper_owner_type": "function",
                    "output_wrapper_type": "ConvertOutputsToFp32",
                    "wrapper_identity_chain_verified": True,
                    "unwrapped_original_identity_verified": True,
                },
                "prepared": True,
                "prepare_route": "accelerator.prepare",
                "backward_route": "accelerator.backward",
                "output_conversion": "convert_outputs_to_fp32",
                "observed_forward_logits_dtypes": {
                    path: ["torch.float32"] * count
                    for path, count in forward_paths.items()
                },
                "observed_inner_autocast": {
                    path: [dict(autocast_row) for _ in range(count)]
                    for path, count in forward_paths.items()
                },
            },
        },
        "arms": {},
        "proof": {"status": "pass"},
        "comparisons": {
            "semantic_atoms": {"passed": True},
            "denominators": _normalization_preflight(),
            "supervised_logits": {"passed": True},
            "loss": {"passed": True},
            "cross_arm_bf16_loss_term_scalars": loss_term_scalars,
            "gradients": {"passed": True},
        },
        "negative_discriminator": {
            "detected": True,
            "forward_detected_by": ["supervised_logits"],
            "gradient_only_is_insufficient": True,
        },
        "timings": {
            "clock": "time.perf_counter_ns_with_cuda_synchronize",
            "scope": "forward_only_same_packed_inputs_train_mode_no_backward",
            "proof_on_forward_ns": 12,
            "proof_off_forward_ns": 10,
            "proof_overhead_ns": 2,
            "ordering": [
                "unmeasured_no_proof_warmup",
                "timed_proof_off",
                "timed_proof_on_packed_clean",
            ],
            "timed_sample_count_per_mode": 1,
            "warmup_sample_count": 1,
            "steady_step_inclusion": False,
        },
        "gpu_memory": {
            "scope": "whole_probe_including_model_transfer_and_all_forward_backward_arms",
            "peak_allocated_bytes": 2048,
            "peak_reserved_bytes": 4096,
        },
        "measurement": _measurement(),
        "failure": None,
    }


def _synthetic_concrete_inventory() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    patterns = (
        ("lora_A", 196, ".lora_A.default.weight"),
        ("lora_B", 196, ".lora_B.default.weight"),
        ("dora_magnitude", 196, ".lora_magnitude_vector.default.weight"),
        ("special_token_delta", 1, ".shared_embed_delta"),
    )
    for group, count, suffix in patterns:
        for index in range(count):
            rows.append(
                {
                    "name": f"model.synthetic_{group}_{index:03d}{suffix}",
                    "group": group,
                    "shape": [1],
                    "parameter_storage_dtype": "torch.float32",
                    "expected_gradient_dtype": "torch.float32",
                    "compute_provenance_dtype": "torch.bfloat16",
                }
            )
    rows.sort(key=lambda row: str(row["name"]))
    body: dict[str, object] = {
        "schema": "coordexp-swift-wave2-concrete-trainable-inventory-v1",
        "declaration_sha256": sha256_json(frozen_trainable_inventory_declaration()),
        "total_count": 589,
        "group_counts": {
            "lora_A": 196,
            "lora_B": 196,
            "dora_magnitude": 196,
            "special_token_delta": 1,
        },
        "parameters": rows,
    }
    value = {**body, "inventory_sha256": sha256_json(body)}
    return validate_concrete_trainable_inventory(value)


def _synthetic_gradient_records() -> tuple[GradientRecord, ...]:
    inventory = _synthetic_concrete_inventory()
    return tuple(
        GradientRecord(
            str(row["name"]),
            tuple(row["shape"]),
            str(row["parameter_storage_dtype"]),
            torch.ones(1),
            gradient_dtype="torch.float32",
            gradient_provenance_dtype="torch.bfloat16",
        )
        for row in inventory["parameters"]
    )


def _synthetic_semantic_keys() -> tuple[SemanticAtomKey, ...]:
    example_ids = frozen_parent_v2_identity()["example_ids"]
    return tuple(
        sorted(
            (
                SemanticAtomKey.from_atom(_atom(example_ids[0], 1)),
                SemanticAtomKey.from_atom(_atom(example_ids[1], 2, token_id=4)),
            )
        )
    )


def _synthetic_arm(
    name: str,
    *,
    microsteps: int,
    forward_only: bool = False,
) -> dict[str, object]:
    records = () if forward_only else _synthetic_gradient_records()
    if name in {"packed_primary", "packed_repeat"}:
        boundaries = [[0, 1436, 2822]]
    elif name == "separate_reference":
        boundaries = [[0, 1436], [0, 1386]]
    else:
        boundaries = [[0, 2822]]
    proof = _v3_success_proof() if name == "packed_primary" else None
    events = []
    if not forward_only:
        events = [
            {
                "microstep_index": index,
                "forward_ordinal": index + 1,
                "loss_ordinal": index + 1,
                "backward_ordinal": index + 1,
                "immediate_after_loss": True,
                "sync_gradients": index == microsteps - 1,
                "accumulation_context": (
                    "sync_gradients"
                    if index == microsteps - 1
                    else "accelerator.no_sync"
                ),
            }
            for index in range(microsteps)
        ]
    return {
        "name": name,
        "microstep_count": microsteps,
        "total_loss_fp32": 1.0,
        "loss_artifact": {**_loss_artifact(), "total_loss": 1.0},
        "semantic_logit_rows": 2,
        "semantic_key_inventory_sha256": semantic_atom_key_inventory_sha256(
            _synthetic_semantic_keys()
        ),
        "gradient_inventory": [record.to_inventory_dict() for record in records],
        "forward_receipts": [
            {
                "fa2_varlen": {
                    "segment_boundaries": boundaries[index],
                    "proof": proof if index == 0 else None,
                }
            }
            for index in range(microsteps)
        ],
        "forward_logits_dtypes": ["torch.float32"] * microsteps,
        "inner_autocast": [
            {
                "module_type": "Qwen3VLTextAttention",
                "layer_idx": 0,
                "cuda_autocast_enabled": True,
                "cuda_autocast_dtype": "torch.bfloat16",
            }
            for _ in range(microsteps)
        ],
        "backward_cadence": {
            "microstep_count": microsteps,
            "gradient_clear_count": 1,
            "harness_backward_call_count": 0 if forward_only else microsteps,
            "harness_policy": (
                "forward_only_negative_control"
                if forward_only
                else "one_immediate_accelerator_backward_per_microstep"
            ),
            "events": events,
            "production_backward_call_count": 0 if forward_only else microsteps,
            "production_policy": (
                "not_applicable_forward_only_negative"
                if forward_only
                else "one_runtime_backward_per_microstep_with_accumulation_context"
            ),
            "cadence_matches_production": True,
        },
    }


def _passed_receipt(plan: dict[str, object]) -> dict[str, object]:  # noqa: F811
    concrete = _synthetic_concrete_inventory()
    records = _synthetic_gradient_records()
    gradient_comparison = compare_gradient_inventories(records, records)
    repeat_comparison = compare_packed_gradient_repeat(records, records)
    loss_terms = _load_probe_module()._compare_loss_terms(
        _loss_artifact(), _loss_artifact()
    )
    forward_paths = {
        "unmeasured_no_proof_warmup": 1,
        "timed_proof_off": 1,
        "packed_primary": 1,
        "packed_repeat": 1,
        "separate_reference": 2,
        "packed_merged_boundary_negative": 1,
    }
    autocast_row = {
        "module_type": "Qwen3VLTextAttention",
        "layer_idx": 0,
        "cuda_autocast_enabled": True,
        "cuda_autocast_dtype": "torch.bfloat16",
    }
    marker_dir = Path(tempfile.mkdtemp(prefix="coordexp-wave2-v3-test-marker-"))
    marker_path = marker_dir / "attempt.json"
    receipt_target = marker_dir / "receipt.json"
    loaded_model_identity = _component_identity(load_model=True)
    loaded_source_identity = {
        "config_identity": json.loads(json.dumps(plan["config_identity"])),
        "repo_identity": json.loads(json.dumps(plan["repo_identity"])),
        "dependency_identity": json.loads(json.dumps(plan["dependency_identity"])),
        "model_identity": loaded_model_identity,
        "model_weight_identity": json.loads(json.dumps(plan["model_weight_identity"])),
        "source_owners": json.loads(json.dumps(plan["source_owners"])),
    }
    marker = finalize_attempt_marker(
        {
            "schema": PARITY_ATTEMPT_MARKER_SCHEMA,
            "status": "attempt_started",
            "plan_sha256": plan["plan_sha256"],
            "receipt_target": str(receipt_target),
            "command_identity": {"schema": "test-command-v1"},
            "source_identity": loaded_source_identity,
            "concrete_trainable_inventory": concrete,
        }
    )
    write_strict_json_atomic(marker_path, marker)
    example_ids = frozen_parent_v2_identity()["example_ids"]
    atoms = (
        _atom(example_ids[0], 1),
        _atom(example_ids[1], 2, token_id=4),
    )
    clean_logits = {
        SemanticAtomKey.from_atom(atom): torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        for atom in atoms
    }
    logits_comparison = compare_keyed_logits(clean_logits, clean_logits)
    loss_comparison = parity_module.compare_tensors(
        torch.tensor(1.0),
        torch.tensor(1.0),
        rtol=parity_module.BF16_RTOL,
        atol=parity_module.BF16_ATOL,
    ).to_artifact_dict()
    loss_comparison["passed"] = loss_comparison.pop("allclose")
    comparison = {
        "supervised_logits": logits_comparison,
        "loss": loss_comparison,
        "cross_arm_bf16_loss_term_scalars": loss_terms,
        "gradients": gradient_comparison,
    }
    negative_logits = compare_keyed_logits(
        clean_logits, {key: value + 1.0 for key, value in clean_logits.items()}
    )
    negative_loss = parity_module.compare_tensors(
        torch.tensor(1.0),
        torch.tensor(1.0),
        rtol=parity_module.BF16_RTOL,
        atol=parity_module.BF16_ATOL,
    ).to_artifact_dict()
    negative_varlen = {"segment_boundaries": [0, 2822], "proof": None}
    return {
        "schema": PARITY_RECEIPT_SCHEMA,
        "terminal_status": "passed",
        "plan_sha256": plan["plan_sha256"],
        "parent_v2": frozen_parent_v2_identity(),
        "attempt_marker": {
            "path": str(marker_path),
            "schema": marker["schema"],
            "status": marker["status"],
            "marker_sha256": marker["marker_sha256"],
            "expected_receipt_target": str(receipt_target),
        },
        "trainable_inventory": {"concrete": concrete},
        "source_identity": loaded_source_identity,
        "model_identity_attestation": attest_qwen_component_identity(
            _component_identity(load_model=False), loaded_model_identity
        ),
        "execution": {
            "device": "cuda:0",
            "model_dtype": "torch.bfloat16",
            "train_mode": True,
            "use_cache": False,
            "optimizer": None,
            "memory_savers": {},
            "adapter": {},
            "special_token_embeddings": {},
            "trainable_value_identity_before": ["same"],
            "trainable_value_identity_after": ["same"],
            "accelerator": {
                "distributed_type": "NO",
                "rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "device": "cuda:0",
                "cuda_current_device": 0,
                "mixed_precision": "bf16",
                "native_amp": True,
                "gradient_accumulation_steps": 1,
                "scaler": None,
                "accelerate_torch_device": "cuda:0",
                "prepared_model_attestation": {
                    "binding_branch": "bound_method",
                    "prepared_forward_type": "method",
                    "wrapper_owner_type": "function",
                    "output_wrapper_type": "ConvertOutputsToFp32",
                    "wrapper_identity_chain_verified": True,
                    "unwrapped_original_identity_verified": True,
                },
                "prepared": True,
                "prepare_route": "accelerator.prepare",
                "backward_route": "accelerator.backward",
                "output_conversion": "convert_outputs_to_fp32",
                "observed_forward_logits_dtypes": {
                    path: ["torch.float32"] * count
                    for path, count in forward_paths.items()
                },
                "observed_inner_autocast": {
                    path: [dict(autocast_row) for _ in range(count)]
                    for path, count in forward_paths.items()
                },
            },
        },
        "arms": {
            "packed_primary": _synthetic_arm("packed_primary", microsteps=1),
            "packed_repeat": _synthetic_arm("packed_repeat", microsteps=1),
            "separate_reference": _synthetic_arm("separate_reference", microsteps=2),
            "packed_merged_boundary_negative": _synthetic_arm(
                "packed_merged_boundary_negative", microsteps=1, forward_only=True
            ),
        },
        "proof": _v3_success_proof(),
        "comparisons": {
            "semantic_atoms": compare_semantic_atom_inventories(
                _Sequence(atoms), _Sequence(atoms)
            ),
            "denominators": _normalization_preflight(),
            "packed_primary_vs_separate": json.loads(json.dumps(comparison)),
            "packed_repeat_vs_separate": json.loads(json.dumps(comparison)),
            "packed_repeat_measurability": repeat_comparison,
        },
        "negative_discriminator": {
            "detected": True,
            "boundary_changed": True,
            "clean_boundaries": [0, 1436, 2822],
            "negative_boundaries": [0, 2822],
            "forward_detected_by": ["supervised_logits"],
            "diagnostic_detected_by": [],
            "gradient_only_is_insufficient": True,
            "attestation": {
                "status": "rejected_against_frozen_clean_boundary",
                "expected_clean_boundaries": [0, 1436, 2822],
                "observed_negative_boundaries": [0, 2822],
                "boundary_mismatch_detected": True,
                "proof_disabled": True,
                "executed_negative_varlen_receipt": negative_varlen,
            },
            "supervised_logits": negative_logits,
            "total_loss": negative_loss,
            "gradients": {"status": "not_executed", "acceptance_metric": False},
        },
        "timings": {
            "clock": "time.perf_counter_ns_with_cuda_synchronize",
            "scope": "forward_only_same_packed_inputs_train_mode_no_backward",
            "proof_on_forward_ns": 12,
            "proof_off_forward_ns": 10,
            "proof_overhead_ns": 2,
            "ordering": [
                "unmeasured_no_proof_warmup",
                "timed_proof_off",
                "timed_proof_on_packed_primary",
            ],
            "timed_sample_count_per_mode": 1,
            "warmup_sample_count": 1,
            "steady_step_inclusion": False,
        },
        "gpu_memory": {
            "scope": "whole_probe_including_model_transfer_and_all_forward_backward_arms",
            "peak_allocated_bytes": 2048,
            "peak_reserved_bytes": 4096,
        },
        "measurement": _measurement(v3=True),
        "failure": None,
    }


def test_plan_is_deterministic_self_authenticating_and_strict_json(tmp_path) -> None:
    first = finalize_plan(_plan_body())
    second = finalize_plan(_plan_body())
    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]

    path = write_strict_json_atomic(tmp_path / "plan.json", first)
    assert load_strict_json(path) == first
    assert validate_parity_plan(first) == first

    mutated = json.loads(json.dumps(first))
    mutated["determinism"]["seed"] = 18
    with pytest.raises(ParityContractError, match="frozen parent|fingerprint"):
        validate_parity_plan(mutated)

    unknown = dict(first)
    unknown["unexpected"] = True
    with pytest.raises(ParityContractError, match="fields"):
        validate_parity_plan(unknown)

    (tmp_path / "nan.json").write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ParityContractError, match="strict JSON"):
        load_strict_json(tmp_path / "nan.json")


@pytest.mark.parametrize("collision_kind", ["identical", "foreign"])
def test_atomic_json_collision_never_fires_link_ownership_callback(
    tmp_path: Path,
    collision_kind: str,
) -> None:
    target = tmp_path / "artifact.json"
    payload = {"value": "intended"}
    existing = payload if collision_kind == "identical" else {"value": "foreign"}
    write_strict_json_atomic(target, existing)
    linked: list[str] = []

    with pytest.raises(ParityContractError) as caught:
        write_strict_json_atomic(
            target,
            payload,
            on_linked=lambda: linked.append("linked"),
        )

    assert caught.value.code == "qwen.parity.artifact_collision"
    assert linked == []
    assert load_strict_json(target) == existing


def test_atomic_json_prelink_failure_never_fires_link_ownership_callback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "artifact.json"
    linked: list[str] = []

    def fail_link(_source, _target) -> None:
        raise OSError("injected pre-link failure")

    monkeypatch.setattr(identity_module.os, "link", fail_link)
    with pytest.raises(OSError, match="pre-link failure"):
        write_strict_json_atomic(
            target,
            {"value": "intended"},
            on_linked=lambda: linked.append("linked"),
        )
    assert linked == []
    assert not target.exists()


def test_atomic_json_directory_open_failure_is_not_reported_as_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "artifact.json"
    payload = {"value": "intended"}
    linked: list[dict[str, object]] = []

    def fail_directory_open(_path, _flags):
        raise OSError("injected directory open failure")

    monkeypatch.setattr(identity_module.os, "open", fail_directory_open)
    with pytest.raises(ParityContractError) as caught:
        write_strict_json_atomic(
            target,
            payload,
            on_linked=lambda: linked.append(load_strict_json(target)),
        )

    assert caught.value.code == "qwen.parity.artifact_directory_sync"
    assert linked == [payload]
    assert load_strict_json(target) == payload


@pytest.mark.parametrize("failure_kind", ["directory_fsync", "temporary_cleanup"])
def test_atomic_json_postlink_failure_fires_callback_for_this_link_only(
    tmp_path: Path,
    monkeypatch,
    failure_kind: str,
) -> None:
    target = tmp_path / "artifact.json"
    payload = {"value": "intended"}
    linked: list[dict[str, object]] = []
    if failure_kind == "directory_fsync":
        original_fsync = identity_module.os.fsync
        fsync_calls = 0

        def fail_directory_fsync(fd) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("injected directory fsync failure")
            original_fsync(fd)

        monkeypatch.setattr(identity_module.os, "fsync", fail_directory_fsync)
        expected = "directory fsync"
    else:
        original_unlink = Path.unlink

        def fail_temporary_cleanup(path, *args, **kwargs):
            if path.parent == tmp_path and path.name.startswith(".artifact.json."):
                raise OSError("injected temporary cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_temporary_cleanup)
        expected = "temporary cleanup"

    def record_link() -> None:
        linked.append(load_strict_json(target))

    with pytest.raises(OSError, match=expected):
        write_strict_json_atomic(target, payload, on_linked=record_link)

    assert linked == [payload]
    assert load_strict_json(target) == payload


def test_v3_schema_and_simple_fp32_tolerance_contract_rejects_stale_v2() -> None:
    assert PARITY_PLAN_SCHEMA.endswith("plan-v3")
    assert PARITY_RECEIPT_SCHEMA.endswith("receipt-v3")
    assert frozen_tolerances() == {
        "bf16_derived_fp32_comparison": {
            "source_compute_dtype": "torch.bfloat16",
            "comparison_dtype": "torch.float32",
            "rtol": 5.0e-3,
            "atol": 5.0e-3,
        },
        "same_packed_repeat": {
            "comparison_dtype": "torch.float32",
            "max_abs": 2.5e-3,
        },
        "negative_control": "outside_at_least_one_frozen_supervised_logit_or_total_loss_tolerance",
        "storage_dtype_selects_tolerance": False,
        "adaptive_tolerance": False,
        "widening_after_results": False,
    }

    stale_plan = _plan_body()
    stale_plan["schema"] = "coordexp-swift-wave2-packed-parity-plan-v2"
    with pytest.raises(ParityContractError, match="schema or status"):
        finalize_plan(stale_plan)

    plan = finalize_plan(_plan_body())
    stale_receipt = _passed_receipt(plan)
    stale_receipt["schema"] = "coordexp-swift-wave2-packed-parity-receipt-v2"
    with pytest.raises(ParityContractError, match="schema is unsupported"):
        validate_parity_receipt(
            stale_receipt,
            expected_plan=plan,
        )


def test_v3_config_projection_refuses_the_migrated_live_config() -> None:
    """The Wave-2 v3 runtime identity is completed GPU evidence.

    `standardize-coordexp-swift-supervised-losses` deliberately migrated this
    supported config (gate `mode`/`0.1`, coordinate term moved to
    `losses.auxiliary`), so the live file is a different experiment from the one
    that was launched. The frozen constant is NOT re-pinned: the projection must
    fail closed on the live file, while the archived identity still
    authenticates the exact enumerated projection without touching it.
    """

    from src.config import load_train_config

    config_path = (
        Path(__file__).resolve().parents[2] / "configs/coordexp_swift/smoke/"
        "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_"
        "llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
    )
    resolved = load_train_config(config_path)
    assert resolved.fingerprint != FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT
    assert resolved.config.training.forward_input_provider_mode == "synchronous"

    with pytest.raises(ParityContractError) as caught:
        config_compatibility_projection(resolved.config_dict)
    assert caught.value.code == "qwen.parity.runtime_config_drift"
    assert caught.value.context["expected"] == FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT
    assert caught.value.context["observed"] == resolved.fingerprint

    archived = _v3_config_identity()
    assert validate_v3_config_identity(archived) == archived
    archived_projection = archived["compatibility_projection"]
    assert (
        archived_projection["schema"]
        == "coordexp-swift-wave2-config-compatibility-projection-v2"
    )
    assert archived_projection["removed_path_values"] == [
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
        {"path": "runtime.determinism", "value": {"mode": "legacy"}},
    ]
    assert (
        archived_projection["projected_config_sha256"]
        == FROZEN_PARENT_V2_CONFIG_FINGERPRINT
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "non_synchronous",
        "nondefault_packing",
        "missing_packing_default",
        "nondefault_resume",
        "missing_resume",
        "unrelated_config_drift",
    ],
)
@pytest.mark.skip(
    reason=(
        "historicized: standardize-coordexp-swift-supervised-losses migrated "
        "the live patchproof config, so the unmutated baseline already drifts "
        "from FROZEN_V3_RUNTIME_CONFIG_FINGERPRINT and every mutation would "
        "pass vacuously; the unconditional drift refusal is asserted by "
        "test_v3_config_projection_refuses_the_migrated_live_config"
    )
)
def test_v3_config_projection_rejects_any_live_config_drift(mutation: str) -> None:
    from src.config import load_train_config

    config_path = (
        Path(__file__).resolve().parents[2] / "configs/coordexp_swift/smoke/"
        "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_"
        "llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
    )
    config = json.loads(json.dumps(load_train_config(config_path).config_dict))
    if mutation == "non_synchronous":
        config["training"]["forward_input_provider_mode"] = "bounded_overlap"
    elif mutation == "nondefault_packing":
        config["packing"]["policy"] = "window_binpack"
    elif mutation == "missing_packing_default":
        del config["packing"]["cursor_byte_budget"]
    elif mutation == "nondefault_resume":
        config["resume"]["mode"] = "exact"
    elif mutation == "missing_resume":
        del config["resume"]
    else:
        config["runtime"]["seed"] += 1
    with pytest.raises(ParityContractError) as caught:
        config_compatibility_projection(config)
    assert caught.value.code == "qwen.parity.runtime_config_drift"


@pytest.mark.parametrize(
    "mutation",
    ["extra_removed_path", "missing_removed_path", "wrong_value", "wrong_digest"],
)
def test_v3_config_projection_rejects_fabricated_attestation(mutation: str) -> None:
    projection = json.loads(
        json.dumps(_v3_config_identity()["compatibility_projection"])
    )
    if mutation == "extra_removed_path":
        projection["removed_path_values"].append({"path": "runtime.seed", "value": 17})
    elif mutation == "missing_removed_path":
        projection["removed_path_values"].pop()
    elif mutation == "wrong_value":
        projection["removed_path_values"][1]["value"] = "window_binpack"
    else:
        projection["projected_config_sha256"] = "0" * 64
    with pytest.raises(ParityContractError) as caught:
        validate_config_compatibility_projection(projection)
    assert caught.value.code == "qwen.parity.config_projection_attestation"


def test_historical_v1_config_projection_remains_readable_without_live_replay() -> None:
    projection = {
        "schema": "coordexp-swift-wave2-config-compatibility-projection-v1",
        "current_config_sha256": (
            "0f8fda29362a46e67cecccdd5fee7d7539fafc7d91b52f49b4d4b036556224a1"
        ),
        "removed_path_values": [
            {
                "path": "training.forward_input_provider_mode",
                "value": "synchronous",
            }
        ],
        "projected_config_sha256": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "parent_config_fingerprint": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        "policy": "remove_exactly_one_later_strict_default",
    }
    identity = _v3_config_identity()
    identity["fingerprint"] = projection["current_config_sha256"]
    identity["resolved_config_sha256"] = projection["current_config_sha256"]
    identity["compatibility_projection"] = projection
    identity["runtime_config_attestation"] = {
        **identity["runtime_config_attestation"],
        "config_fingerprint": projection["current_config_sha256"],
        "compatibility_projection_sha256": sha256_json(projection),
    }

    assert validate_v3_config_identity(identity) == identity


def test_v3_config_projection_does_not_weaken_generic_plan_equality() -> None:
    expected = finalize_plan(_plan_body())
    observed_body = json.loads(json.dumps(expected))
    del observed_body["plan_sha256"]
    observed_body["repo_identity"]["head"] = "drifted"
    observed = finalize_plan(observed_body)
    with pytest.raises(ParityContractError) as caught:
        assert_plan_revalidated(expected, observed)
    assert caught.value.code == "qwen.parity.plan_drift"


def test_probe_parent_match_allows_only_versioned_config_projection() -> None:
    probe = _load_probe_module()
    current = _plan_body()
    current_arms = current["arms"]
    parent_config = {
        key: json.loads(json.dumps(value))
        for key, value in current["config_identity"].items()
        if key
        in {
            "entry_path",
            "schema_version",
            "loader_version",
            "sources",
        }
    }
    parent_config.update(
        {
            "fingerprint": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
            "resolved_config_sha256": FROZEN_PARENT_V2_CONFIG_FINGERPRINT,
        }
    )
    parent = {
        **json.loads(json.dumps(current)),
        "config_identity": parent_config,
        "arms": {
            "packed_clean": json.loads(json.dumps(current_arms["packed_primary"])),
            "separate_reference": json.loads(
                json.dumps(current_arms["separate_reference"])
            ),
            "packed_merged_boundary_negative": json.loads(
                json.dumps(current_arms["packed_merged_boundary_negative"])
            ),
        },
    }
    probe._assert_v3_matches_parent_v2(current, parent)

    drifted = json.loads(json.dumps(current))
    drifted["config_identity"]["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ParityContractError) as caught:
        probe._assert_v3_matches_parent_v2(drifted, parent)
    assert caught.value.code == "qwen.parity.config_projection_identity_drift"


def test_plan_rejects_loss_normalization_preflight_mutation() -> None:
    body = _plan_body()
    preflight = body["loss_normalization_preflight"]
    preflight["packed"]["base_ce"]["selected_atom_count"] += 1
    with pytest.raises(ParityContractError, match="denominator"):
        finalize_plan(body)


@pytest.mark.parametrize("value", [None, "0", "true"])
def test_v3_plan_requires_frozen_flash_attention_determinism(value: object) -> None:
    body = _plan_body()
    if value is None:
        body["determinism"].pop("flash_attention_deterministic")
    else:
        body["determinism"]["flash_attention_deterministic"] = value
    with pytest.raises(ParityContractError) as caught:
        finalize_plan(body)
    assert caught.value.code == "qwen.parity.flash_attention_deterministic"


def test_v3_loss_artifact_reports_weight_math_without_inventing_plan_weights() -> None:
    artifact = {**_loss_artifact(), "total_loss": 1.0}
    validated = parity_module._validate_v3_loss_artifact(
        artifact,
        owner="test.loss_artifact",
        expected_term_names=("base_ce", "coord_gaussian_rps", "token_type_gate"),
        expected_term_weights=None,
    )
    assert validated == artifact

    mutated = json.loads(json.dumps(artifact))
    mutated["terms"][0]["weighted_loss"] += 0.1
    mutated["total_loss"] += 0.1
    with pytest.raises(ParityContractError) as caught:
        parity_module._validate_v3_loss_artifact(
            mutated,
            owner="test.loss_artifact",
            expected_term_names=(
                "base_ce",
                "coord_gaussian_rps",
                "token_type_gate",
            ),
            expected_term_weights=None,
        )
    assert caught.value.code == "qwen.parity.arm_loss_weight"


def test_v3_loss_artifact_binds_authoritative_plan_weights_when_exposed() -> None:
    body = _plan_body()
    body["trainable_mechanism"] = {
        "losses": {
            "protected": {
                "base_ce": {"weight": 0.75},
                "coord_gaussian_rps": {"weight": 1.0},
                "token_type_gate": {"weight": 1.0},
            }
        }
    }
    plan = finalize_plan(body)
    weights = parity_module._planned_loss_term_weights(plan)
    assert weights == {
        "base_ce": 0.75,
        "coord_gaussian_rps": 1.0,
        "token_type_gate": 1.0,
    }
    artifact = {**_loss_artifact(), "total_loss": 1.0}
    mutated = json.loads(json.dumps(artifact))
    mutated["terms"][0]["weight"] = 0.5
    mutated["terms"][0]["weighted_loss"] = mutated["terms"][0]["raw_loss"] * 0.5
    mutated["total_loss"] = sum(term["weighted_loss"] for term in mutated["terms"])
    with pytest.raises(ParityContractError) as caught:
        parity_module._validate_v3_loss_artifact(
            mutated,
            owner="test.loss_artifact",
            expected_term_names=tuple(weights),
            expected_term_weights=weights,
        )
    assert caught.value.code == "qwen.parity.arm_loss_weight_binding"


def test_atomic_json_publication_is_absent_target_only(tmp_path) -> None:
    target = tmp_path / "plan.json"
    write_strict_json_atomic(target, {"value": "first"})
    original = target.read_bytes()
    with pytest.raises(ParityContractError, match="already exists"):
        write_strict_json_atomic(target, {"value": "second"})
    assert target.read_bytes() == original


def test_repo_identity_accepts_git_sha1_head_and_ignores_untracked_state(
    tmp_path, monkeypatch
) -> None:
    head = "a" * 40

    def fake_git(root, *args):
        del root
        if args == ("rev-parse", "HEAD"):
            return f"{head}\n"
        if args == ("status", "--short", "--untracked-files=no"):
            return " M tracked.py\n"
        if args == ("diff", "--binary", "HEAD", "--", "."):
            return "tracked diff\n"
        raise AssertionError(args)

    monkeypatch.setattr(identity_module, "_git", fake_git)
    identity = repo_identity(tmp_path)
    assert identity["head"] == head
    assert identity["dirty"] is True


def test_semantic_key_alignment_ignores_pack_position_and_detects_drift() -> None:
    packed = _Sequence((_atom("a", 1), _atom("b", 2, token_id=4)))
    separate = (
        _Sequence((_atom("a", 1),)),
        _Sequence((_atom("b", 2, token_id=4),)),
    )
    comparison = compare_semantic_atom_inventories(packed, separate)
    assert comparison["passed"] is True

    drifted = (_Sequence((_atom("a", 1),)), _Sequence((_atom("b", 3, token_id=4),)))
    comparison = compare_semantic_atom_inventories(packed, drifted)
    assert comparison["passed"] is False
    assert comparison["missing"]
    assert comparison["extra"]


def test_selected_logits_are_keyed_and_compare_full_rows() -> None:
    atoms = (_atom("a", 1), _atom("b", 2, token_id=4))
    left = selected_logits_by_semantic_key(
        _Context(
            torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), atoms
        )
    )
    right = {key: value.clone() for key, value in left.items()}
    assert compare_keyed_logits(left, right)["passed"] is True

    changed_key = sorted(right)[0]
    right[changed_key][0] += 1.0
    result = compare_keyed_logits(left, right)
    assert result["passed"] is False
    assert result["max_abs_diff"] == pytest.approx(1.0)


def test_shared_denominator_comparison_accepts_production_semantics_and_arm_counts() -> (
    None
):
    packed = _production_loss_plan(1)
    separate_shared = _production_loss_plan(2)
    comparison = compare_shared_denominators(
        packed,
        separate_shared,
        packed_context_count=1,
        separate_context_count=2,
    )
    assert comparison["passed"] is True
    assert comparison["failures"] == []
    assert comparison["semantic_fields"] == [
        "term_name",
        "denominator_scope",
        "eligible_segment_count",
        "selected_atom_count",
        "skipped_segment_count",
    ]
    assert comparison["expected_context_counts"] == {
        "packed": 1,
        "separate_shared": 2,
    }
    assert comparison["packed"]["base_ce"]["context_count"] == 1
    assert comparison["separate_shared"]["base_ce"]["context_count"] == 2


@pytest.mark.parametrize(
    "field",
    [
        "term_name",
        "denominator_scope",
        "eligible_segment_count",
        "selected_atom_count",
        "skipped_segment_count",
    ],
)
def test_shared_denominator_comparison_rejects_every_semantic_field(
    field: str,
) -> None:
    packed = _production_loss_plan(1)
    separate = _production_loss_plan(2)
    denominators = dict(separate.denominators)
    if field == "term_name":
        original = denominators.pop("token_type_gate")
        denominators["token_type_gate_drift"] = replace(
            original, term_name="token_type_gate_drift"
        )
        separate = _production_loss_plan(2, denominators=denominators)
    elif field == "denominator_scope":
        denominators = {
            name: replace(denominator, denominator_scope="planned_step_global")
            for name, denominator in denominators.items()
        }
        separate = _production_loss_plan(
            2,
            denominators=denominators,
            denominator_scope="planned_step_global",
        )
    else:
        original = denominators["token_type_gate"]
        denominators["token_type_gate"] = replace(
            original,
            **{field: getattr(original, field) + 1},
        )
        separate = _production_loss_plan(2, denominators=denominators)
    comparison = compare_shared_denominators(
        packed,
        separate,
        packed_context_count=1,
        separate_context_count=2,
    )
    assert comparison["passed"] is False
    assert comparison["failures"]
    if field == "term_name":
        assert comparison["failures"][0]["reason"] == "term_inventory_mismatch"
    else:
        assert any(failure.get("field") == field for failure in comparison["failures"])


@pytest.mark.parametrize("arm", ["packed", "separate_shared"])
def test_shared_denominator_comparison_rejects_wrong_arm_context_count(
    arm: str,
) -> None:
    packed = _production_loss_plan(1)
    separate = _production_loss_plan(2)
    target = packed if arm == "packed" else separate
    wrong_count = 2 if arm == "packed" else 1
    mutated = {
        name: replace(denominator, context_count=wrong_count)
        for name, denominator in target.denominators.items()
    }
    if arm == "packed":
        packed = _production_loss_plan(1, denominators=mutated)
    else:
        separate = _production_loss_plan(2, denominators=mutated)
    comparison = compare_shared_denominators(
        packed,
        separate,
        packed_context_count=1,
        separate_context_count=2,
    )
    assert comparison["passed"] is False
    assert {
        (failure.get("arm"), failure.get("reason"))
        for failure in comparison["failures"]
    } >= {(arm, "context_count_mismatch")}


@pytest.mark.parametrize("arm", ["packed", "separate_shared"])
def test_shared_denominator_comparison_rejects_wrong_plan_pack_count(
    arm: str,
) -> None:
    packed = _production_loss_plan(1)
    separate = _production_loss_plan(2)
    target = packed if arm == "packed" else separate
    counts = dict(target.counts)
    counts["count/packs"] = 2 if arm == "packed" else 1
    if arm == "packed":
        packed = replace(packed, counts=counts)
    else:
        separate = replace(separate, counts=counts)
    comparison = compare_shared_denominators(
        packed,
        separate,
        packed_context_count=1,
        separate_context_count=2,
    )
    assert comparison["passed"] is False
    assert {
        (failure.get("arm"), failure.get("reason"))
        for failure in comparison["failures"]
    } >= {(arm, "plan_context_count_mismatch")}


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_shared_denominator_comparison_rejects_term_inventory(
    mutation: str,
) -> None:
    packed = _production_loss_plan(1)
    denominators = dict(_production_loss_plan(2).denominators)
    if mutation == "missing":
        denominators.pop("coord_gaussian_rps")
    else:
        denominators["extra_term"] = _segment_denominator(
            "extra_term", context_count=2, selected_atom_count=1
        )
    separate = _production_loss_plan(2, denominators=denominators)
    comparison = compare_shared_denominators(
        packed,
        separate,
        packed_context_count=1,
        separate_context_count=2,
    )
    assert comparison["passed"] is False
    assert comparison["failures"][0]["reason"] == "term_inventory_mismatch"


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_shared_denominator_comparison_rejects_artifact_field_schema(
    mutation: str,
) -> None:
    packed = _production_loss_plan(1)
    separate = _production_loss_plan(2)
    denominators = dict(separate.denominators)
    artifact = denominators["token_type_gate"].to_artifact_dict()
    if mutation == "missing":
        artifact.pop("selected_atom_count")
    else:
        artifact["unexpected"] = 1
    denominators["token_type_gate"] = _ArtifactDenominator(artifact)
    malformed = _production_loss_plan(2, denominators=denominators)
    with pytest.raises(ParityContractError, match="production schema"):
        compare_shared_denominators(
            packed,
            malformed,
            packed_context_count=1,
            separate_context_count=2,
        )


def test_cross_arm_bf16_loss_term_scalars_use_combined_torch_allclose_band() -> None:
    probe = _load_probe_module()
    inside = probe._compare_loss_terms(_loss_artifact(), _loss_artifact(delta=5.0e-3))
    outside = probe._compare_loss_terms(_loss_artifact(), _loss_artifact(delta=5.1e-3))
    assert inside["passed"] is True
    assert outside["passed"] is False
    assert validate_cross_arm_bf16_loss_term_scalars(inside) == inside
    assert validate_cross_arm_bf16_loss_term_scalars(outside) == outside
    assert inside["source_forward_dtype"] == "torch.bfloat16"
    assert inside["comparison_dtype"] == "torch.float32"
    assert inside["rtol"] == 5.0e-3
    assert inside["atol"] == 5.0e-3
    zero_weight_row = inside["terms"][1]["fields"][1]
    assert zero_weight_row["packed_value"] == 0.25
    assert zero_weight_row["raw_delta"] == pytest.approx(-5.0e-3)
    assert zero_weight_row["allclose"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "source_dtype",
        "comparison_dtype",
        "rtol",
        "field_inventory",
        "duplicate_term",
        "field_order",
        "raw_delta",
        "nested_allclose",
        "nested_extra",
        "top_passed",
    ],
)
def test_cross_arm_bf16_loss_term_scalar_validator_rejects_mutations(
    mutation: str,
) -> None:
    block = _load_probe_module()._compare_loss_terms(_loss_artifact(), _loss_artifact())
    if mutation == "source_dtype":
        block["source_forward_dtype"] = "torch.float32"
    elif mutation == "comparison_dtype":
        block["comparison_dtype"] = "torch.bfloat16"
    elif mutation == "rtol":
        block["rtol"] = 1.0e-5
    elif mutation == "field_inventory":
        block["fields"].pop()
    elif mutation == "duplicate_term":
        block["terms"].append(json.loads(json.dumps(block["terms"][0])))
    elif mutation == "field_order":
        block["terms"][0]["fields"].reverse()
    elif mutation == "raw_delta":
        block["terms"][0]["fields"][0]["raw_delta"] = 1.0
    elif mutation == "nested_allclose":
        block["terms"][0]["fields"][0]["allclose"] = False
    elif mutation == "nested_extra":
        block["terms"][0]["fields"][0]["unexpected"] = True
    else:
        block["passed"] = False
    with pytest.raises(ParityContractError, match="loss-term|strict schema"):
        validate_cross_arm_bf16_loss_term_scalars(block)


@pytest.mark.parametrize(
    "mutation",
    [
        "semantic_fields",
        "expected_context_counts",
        "failures",
        "packed_context",
        "semantic_value",
        "term_inventory",
        "plan_pack_count",
        "plan_groups",
        "plan_semantic_count",
    ],
)
def test_denominator_comparison_validator_rejects_complete_artifact_mutations(
    mutation: str,
) -> None:
    artifact = _normalization_preflight()
    if mutation == "semantic_fields":
        artifact["semantic_fields"].pop()
    elif mutation == "expected_context_counts":
        artifact["expected_context_counts"]["separate_shared"] = 1
    elif mutation == "failures":
        artifact["failures"] = [{"reason": "mutated"}]
    elif mutation == "packed_context":
        artifact["packed"]["base_ce"]["context_count"] = 2
    elif mutation == "semantic_value":
        artifact["separate_shared"]["base_ce"]["selected_atom_count"] += 1
    elif mutation == "term_inventory":
        artifact["separate_shared"].pop("coord_gaussian_rps")
    elif mutation == "plan_pack_count":
        artifact["plan_normalization"]["packed"]["counts"]["count/packs"] = 2
    elif mutation == "plan_groups":
        artifact["plan_normalization"]["packed"]["token_type_gate_groups"] = []
    else:
        artifact["plan_normalization"]["separate_shared"]["counts"][
            "count/supervised_atoms"
        ] += 1
    with pytest.raises(ParityContractError, match="denominator|normalization"):
        validate_denominator_comparison_artifact(artifact)


def test_gradient_inventory_requires_exact_names_shapes_dtypes_none_and_tolerance() -> (
    None
):
    left = (
        GradientRecord("bf16", (2,), "torch.bfloat16", torch.tensor([1.0, 2.0])),
        GradientRecord("fp32", (1,), "torch.float32", torch.tensor([0.0])),
    )
    right = (
        GradientRecord("bf16", (2,), "torch.bfloat16", torch.tensor([1.001, 2.0])),
        GradientRecord("fp32", (1,), "torch.float32", torch.tensor([0.0])),
    )
    assert compare_gradient_inventories(left, right)["passed"] is True

    missing = compare_gradient_inventories(left, right[:1])
    assert missing["passed"] is False
    assert missing["missing_names"] == ["fp32"]

    none_mismatch = compare_gradient_inventories(
        left,
        (
            right[0],
            GradientRecord("fp32", (1,), "torch.float32", None),
        ),
    )
    assert none_mismatch["passed"] is False
    assert none_mismatch["failures"][0]["reason"] == "none_status"

    dtype_mismatch = compare_gradient_inventories(
        left,
        (
            GradientRecord("bf16", (2,), "torch.float32", torch.tensor([1.0, 2.0])),
            right[1],
        ),
    )
    assert dtype_mismatch["passed"] is False
    assert dtype_mismatch["failures"][0]["reason"] == (
        "parameter_storage_dtype_mismatch"
    )

    tolerance_failure = compare_gradient_inventories(
        left,
        (
            GradientRecord("bf16", (2,), "torch.bfloat16", torch.tensor([2.0, 2.0])),
            right[1],
        ),
    )
    assert tolerance_failure["passed"] is False
    assert tolerance_failure["failures"][0]["reason"] == "tolerance_failure"

    nonfinite = (
        GradientRecord(
            "bf16", (2,), "torch.bfloat16", torch.tensor([float("nan"), 2.0])
        ),
    )
    with pytest.raises(ParityContractError, match="non-finite"):
        compare_gradient_inventories(nonfinite, nonfinite)

    both_none = (
        GradientRecord("bf16", (2,), "torch.bfloat16", None),
        GradientRecord("fp32", (1,), "torch.float32", None),
    )
    all_none_result = compare_gradient_inventories(both_none, both_none)
    assert all_none_result["passed"] is False
    assert all_none_result["parity_passed"] is True
    assert all_none_result["coverage_passed"] is False
    assert all_none_result["nonzero_gradient_signal"] is False
    assert {failure["reason"] for failure in all_none_result["failures"]} == {
        "none_status",
        "no_nonzero_gradient_signal",
    }

    all_zero = (
        GradientRecord("bf16", (2,), "torch.bfloat16", torch.zeros(2)),
        GradientRecord("fp32", (1,), "torch.float32", torch.zeros(1)),
    )
    all_zero_result = compare_gradient_inventories(all_zero, all_zero)
    assert all_zero_result["passed"] is False
    assert all_zero_result["parity_passed"] is True
    assert all_zero_result["coverage_passed"] is False
    assert all_zero_result["failures"][-1]["reason"] == "no_nonzero_gradient_signal"

    repeat_zero_result = compare_packed_gradient_repeat(all_zero, all_zero)
    assert repeat_zero_result["status"] == "unmeasurable"
    assert repeat_zero_result["passed"] is False
    assert repeat_zero_result["coverage_passed"] is False
    assert repeat_zero_result["failures"][-1]["reason"] == (
        "no_nonzero_gradient_signal"
    )


def test_fp32_storage_with_bf16_compute_provenance_uses_bf16_band() -> None:
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(17, 19, dtype=torch.float32))
    first = torch.randn(64, 17, dtype=torch.float32)
    second = torch.randn(64, 17, dtype=torch.float32)

    def branch_loss(inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = inputs @ weight
        return output.float().square().sum() / 1000.0

    def gradient(mode: str) -> torch.Tensor:
        weight.grad = None
        if mode == "packed":
            branch_loss(torch.cat((first, second), dim=0)).backward()
        elif mode == "summed":
            (branch_loss(first) + branch_loss(second)).backward()
        else:
            branch_loss(first).backward()
            branch_loss(second).backward()
        assert weight.grad is not None
        return weight.grad.detach().clone()

    packed = gradient("packed")
    summed = gradient("summed")
    streaming = gradient("streaming")
    assert not torch.allclose(packed, summed, rtol=1.0e-4, atol=1.0e-5)
    assert not torch.allclose(packed, streaming, rtol=1.0e-4, atol=1.0e-5)
    assert torch.allclose(packed, summed, rtol=5.0e-3, atol=5.0e-3)

    comparison = compare_gradient_inventories(
        (
            GradientRecord(
                "weight",
                tuple(weight.shape),
                "torch.float32",
                packed,
                gradient_dtype="torch.float32",
                gradient_provenance_dtype="torch.bfloat16",
            ),
        ),
        (
            GradientRecord(
                "weight",
                tuple(weight.shape),
                "torch.float32",
                summed,
                gradient_dtype="torch.float32",
                gradient_provenance_dtype="torch.bfloat16",
            ),
        ),
    )
    assert comparison["passed"] is True
    assert comparison["coverage_passed"] is True
    assert comparison["parity_passed"] is True
    assert comparison["source_compute_dtype"] == "torch.bfloat16"
    assert comparison["comparison_dtype"] == "torch.float32"
    assert comparison["parameters"][0]["rtol"] == 5.0e-3
    assert comparison["parameters"][0]["atol"] == 5.0e-3
    assert "tolerance_policy" not in comparison


def test_gradient_failure_names_are_complete_within_bounded_receipt_capacity() -> None:
    left = tuple(
        GradientRecord(
            f"parameter_{index:03d}",
            (1,),
            "torch.float32",
            torch.ones(1),
        )
        for index in range(100)
    )
    result = compare_gradient_inventories(left, ())
    assert len(result["missing_names"]) == 100
    assert len(result["missing_names"]) == 100
    assert result["parameters"] == []


def test_gradient_comparison_fails_closed_above_full_row_capacity() -> None:
    shared_gradient = torch.ones(1)
    records = tuple(
        GradientRecord(
            f"parameter_{index:04d}",
            (1,),
            "torch.float32",
            shared_gradient,
        )
        for index in range(parity_module.MAX_GRADIENT_PARAMETER_SAMPLES + 1)
    )
    with pytest.raises(ParityContractError) as exc_info:
        compare_gradient_inventories(records, records)
    assert exc_info.value.code == "qwen.parity.gradient_artifact_bound"


def test_probe_rejects_trainable_inventory_above_receipt_capacity_before_arms() -> None:
    probe = _load_probe_module()
    row = {"name": "weight"}
    inventory = [row] * (parity_module.MAX_GRADIENT_PARAMETER_SAMPLES + 1)
    with pytest.raises(ParityContractError) as exc_info:
        probe._assert_gradient_receipt_capacity(inventory)
    assert exc_info.value.code == "qwen.parity.gradient_artifact_bound"


def test_harness_backward_cadence_matches_production_streaming() -> None:
    probe = _load_probe_module()
    harness_source = inspect.getsource(probe._execute_arm)
    from src.training.supervised_trainer import SupervisedTrainer

    production_source = inspect.getsource(SupervisedTrainer._run_streaming_planned_step)
    assert "accelerator.backward(bundle.total_loss)" in harness_source
    assert "for index, (inputs, tokens)" in harness_source
    assert "self.runtime.backward(" in production_source
    assert "for local_micro_step_index, micro_step" in production_source

    arm = probe.ExecutedArm(
        name="separate_reference",
        total_loss=torch.tensor(1.0),
        loss_artifact={},
        keyed_logits={},
        gradients=(),
        forward_receipts=({}, {}),
        forward_logits_dtypes=("torch.float32", "torch.float32"),
        autocast_observations=({}, {}),
        forward_elapsed_ns=1,
        backward_call_count=2,
        backward_events=(
            {
                "microstep_index": 0,
                "forward_ordinal": 1,
                "loss_ordinal": 1,
                "backward_ordinal": 1,
                "immediate_after_loss": True,
                "sync_gradients": False,
                "accumulation_context": "no_sync",
            },
            {
                "microstep_index": 1,
                "forward_ordinal": 2,
                "loss_ordinal": 2,
                "backward_ordinal": 2,
                "immediate_after_loss": True,
                "sync_gradients": True,
                "accumulation_context": "accumulate",
            },
        ),
        gradient_clear_count=1,
    )
    cadence = probe._arm_artifact(arm)["backward_cadence"]
    assert cadence["harness_backward_call_count"] == 2
    assert cadence["production_backward_call_count"] == 2
    assert cadence["cadence_matches_production"] is True


def test_boundary_corruption_changes_only_fa2_plan() -> None:
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    plan = Fa2VarlenPlan(
        segment_boundaries=(0, 2, 5),
        segment_lengths=(2, 3),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu.clone(),
        max_length_q=3,
        max_length_k=3,
        attention_mask=None,
    )
    clean = _Inputs(
        pack_index=0,
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        position_ids=torch.arange(20).reshape(4, 1, 5),
        pixel_values=torch.ones(2, 4),
        image_grid_thw=torch.tensor([[1, 1, 1], [1, 1, 1]]),
        fa2_varlen_plan=plan,
        receipt=_Receipt(plan, "first_micro_step"),
        logits_to_keep=torch.tensor([0, 3, 4]),
    )
    corrupted = merged_boundary_forward_inputs(clean)
    assert corrupted.fa2_varlen_plan.segment_boundaries == (0, 5)
    assert torch.equal(clean.input_ids, corrupted.input_ids)
    assert torch.equal(clean.position_ids, corrupted.position_ids)
    assert torch.equal(clean.pixel_values, corrupted.pixel_values)
    assert torch.equal(clean.image_grid_thw, corrupted.image_grid_thw)
    assert corrupted.receipt.fa2_branch_proof_policy == "disabled"


def test_negative_discriminator_requires_boundary_change_and_numerical_failure() -> (
    None
):
    detected = negative_discriminator(
        clean_boundaries=(0, 2, 5),
        negative_boundaries=(0, 5),
        logits_allclose=False,
        loss_allclose=True,
        gradients_allclose=True,
    )
    assert detected == {
        "detected": True,
        "boundary_changed": True,
        "clean_boundaries": [0, 2, 5],
        "negative_boundaries": [0, 5],
        "forward_detected_by": ["supervised_logits"],
        "diagnostic_detected_by": [],
        "gradient_only_is_insufficient": True,
    }
    assert (
        negative_discriminator(
            clean_boundaries=(0, 2, 5),
            negative_boundaries=(0, 5),
            logits_allclose=True,
            loss_allclose=True,
            gradients_allclose=True,
        )["detected"]
        is False
    )
    gradient_only = negative_discriminator(
        clean_boundaries=(0, 2, 5),
        negative_boundaries=(0, 5),
        logits_allclose=True,
        loss_allclose=True,
        gradients_allclose=False,
    )
    assert gradient_only["detected"] is False
    assert gradient_only["diagnostic_detected_by"] == ["trainable_gradients"]
    assert (
        negative_discriminator(
            clean_boundaries=(0, 2, 5),
            negative_boundaries=(0, 2, 5),
            logits_allclose=False,
            loss_allclose=True,
            gradients_allclose=True,
        )["detected"]
        is False
    )


@pytest.mark.parametrize("policy", ["enabled", "disabled"])
def test_qwen_component_identity_attestation_accepts_exact_transition(
    policy: str,
) -> None:
    expected = _component_identity(load_model=False, policy=policy)
    loaded = _component_identity(load_model=True, policy=policy)
    attestation = attest_qwen_component_identity(expected, loaded)
    assert attestation["status"] == "pass"
    assert attestation["runtime_patch_transition"]["policy"] == policy
    assert attestation["runtime_patch_transition"]["expected_reason"] == (
        "model_not_loaded"
    )
    assert attestation["runtime_patch_transition"]["loaded_reason"] == (
        "conv3d_kernel_stride_equivalent_linear_projection"
        if policy == "enabled"
        else "policy_disabled"
    )


def test_qwen_component_identity_attestation_rejects_stable_projection_drift() -> None:
    expected = _component_identity(load_model=False)
    loaded = _component_identity(load_model=True)
    loaded["processor"] = {"processor_class": "DriftedProcessor"}
    with pytest.raises(ParityContractError, match="differs from the prepared plan"):
        attest_qwen_component_identity(expected, loaded)


@pytest.mark.parametrize(
    ("expected_load_model", "loaded_load_model"),
    [(True, True), (False, False)],
)
def test_qwen_component_identity_attestation_rejects_load_state_drift(
    expected_load_model: bool,
    loaded_load_model: bool,
) -> None:
    expected = _component_identity(load_model=expected_load_model)
    loaded = _component_identity(load_model=loaded_load_model)
    with pytest.raises(ParityContractError, match="load state"):
        attest_qwen_component_identity(expected, loaded)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_patch", "exactly one runtime patch"),
        ("reason", "owner or reason"),
        ("shape", "shape invariants"),
        ("probe", "exceeds tolerance"),
    ],
)
def test_qwen_component_identity_attestation_rejects_runtime_patch_drift(
    mutation: str,
    message: str,
) -> None:
    expected = _component_identity(load_model=False)
    loaded = _component_identity(load_model=True)
    patches = loaded["runtime_patches"]
    assert isinstance(patches, dict)
    patch = patches["qwen3_vl_patch_embed_linearization"]
    assert isinstance(patch, dict)
    if mutation == "extra_patch":
        patches["unexpected"] = dict(patch)
    elif mutation == "reason":
        patch["reason"] = "already_linearized"
    elif mutation == "shape":
        patch["weight_shape"] = [8, 3, 1, 14, 14]
    else:
        probe = patch["equivalence_probe"]
        assert isinstance(probe, dict)
        probe["grad_max_abs_diff"] = 1.1e-4
    with pytest.raises(ParityContractError, match=message):
        attest_qwen_component_identity(expected, loaded)


def test_receipt_validation_requires_terminal_clean_gate_and_negative() -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    assert (
        validate_parity_receipt(
            receipt,
            expected_plan=plan,
        )
        == receipt
    )

    failed_clean = json.loads(json.dumps(receipt))
    failed_clean["comparisons"]["packed_primary_vs_separate"]["gradients"]["passed"] = (
        False
    )
    with pytest.raises(ParityContractError, match="gradient|failed"):
        validate_parity_receipt(failed_clean, expected_plan=plan)

    missing_negative = json.loads(json.dumps(receipt))
    missing_negative["negative_discriminator"]["detected"] = False
    with pytest.raises(ParityContractError, match="negative.*inconsistent"):
        validate_parity_receipt(missing_negative, expected_plan=plan)

    gradient_only_negative = json.loads(json.dumps(receipt))
    gradient_only_negative["negative_discriminator"]["forward_detected_by"] = []
    with pytest.raises(ParityContractError, match="negative.*inconsistent"):
        validate_parity_receipt(
            gradient_only_negative,
            expected_plan=plan,
        )

    failure = {
        **receipt,
        "terminal_status": "failed",
        "attempt_marker": {},
        "trainable_inventory": {},
        "source_identity": {
            "config_identity": plan["config_identity"],
            "repo_identity": plan["repo_identity"],
            "dependency_identity": plan["dependency_identity"],
            "model_identity": plan["model_identity"],
            "model_weight_identity": plan["model_weight_identity"],
            "source_owners": plan["source_owners"],
        },
        "model_identity_attestation": {},
        "execution": {"requested_device": "cuda:0"},
        "failure": {
            "type": "ParityContractError",
            "message": "bounded failure",
            "evidence": {
                "schema": PARITY_FAILURE_EVIDENCE_SCHEMA,
                "stage_reached": "plan_revalidated",
                "completed_phases": [],
                "completed_fields": [
                    "source_identity",
                    "execution",
                ],
            },
        },
        "comparisons": {},
        "arms": {},
        "proof": {},
        "negative_discriminator": {},
        "timings": {},
        "gpu_memory": {},
        "measurement": {},
    }
    assert validate_parity_receipt(failure, expected_plan=plan) == failure
    with pytest.raises(ParityContractError, match="requires its authenticated plan"):
        validate_parity_receipt(receipt)


def _failure_proof() -> dict[str, object]:
    boundaries = [0, 2, 5]
    topology_id = "qwen-test-topology-v1"
    module_name = "model.language_model.layers.0.self_attn"
    module_class = "Qwen3VLTextAttention"
    identity = f"{module_name}#0"
    event = {
        "kind": "text",
        "identity": identity,
        "topology_id": topology_id,
        "module_name": module_name,
        "module_class": module_class,
        "layer_idx": 0,
        "configured_backend": "flash_attention_2",
        "registry_key": "flash_attention_2",
        "registry_had_local_override": False,
        "attention_mask": None,
        "completion_status": "completed",
        "cu_seq_lens_q": boundaries,
        "cu_seq_lens_k": boundaries,
        "max_length_q": 3,
        "max_length_k": 3,
        "flash_fn_call_count": 0,
        "flash_varlen_fn_call_count": 1,
        "pad_fn_call_count": 0,
        "unpad_fn_call_count": 0,
        "varlen_calls": [{}],
    }
    return {
        "status": "pass",
        "observed_branch": "padding_free_varlen",
        "segment_boundaries": boundaries,
        "cu_seq_lens_q": boundaries,
        "cu_seq_lens_k": boundaries,
        "max_length_q": 3,
        "max_length_k": 3,
        "resolved_attention_implementation": "flash_attention_2",
        "model_dtype": "torch.bfloat16",
        "branch_evidence_from_explicit_varlen_kwargs": True,
        "flash_fn_called": False,
        "flash_varlen_fn_called": True,
        "pad_fn_called": False,
        "unpad_fn_called": False,
        "observed_call": {},
        "topology": {
            "topology_id": topology_id,
            "text_model_name": "model.language_model",
            "text_model_class": "Qwen3VLTextModel",
            "configured_text_layer_count": 1,
            "text_layers": [
                {
                    "topology_id": topology_id,
                    "identity": identity,
                    "module_name": module_name,
                    "module_class": module_class,
                    "layer_idx": 0,
                    "configured_backend": "flash_attention_2",
                }
            ],
        },
        "expected_text_layer_count": 1,
        "observed_text_layer_count": 1,
        "text_layer_events": [event],
        "vision_attention_events": [],
        "unrelated_attention_events": [],
    }


def _v3_success_proof() -> dict[str, object]:
    proof = _failure_proof()
    boundaries = [0, 1436, 2822]
    proof["segment_boundaries"] = boundaries
    proof["cu_seq_lens_q"] = boundaries
    proof["cu_seq_lens_k"] = boundaries
    proof["max_length_q"] = 1436
    proof["max_length_k"] = 1436
    event = proof["text_layer_events"][0]
    event["cu_seq_lens_q"] = boundaries
    event["cu_seq_lens_k"] = boundaries
    event["max_length_q"] = 1436
    event["max_length_k"] = 1436
    second_layer = json.loads(json.dumps(proof["topology"]["text_layers"][0]))
    second_layer["module_name"] = "model.language_model.layers.1.self_attn"
    second_layer["layer_idx"] = 1
    second_layer["identity"] = f"{second_layer['module_name']}#1"
    proof["topology"]["text_layers"].append(second_layer)
    proof["topology"]["configured_text_layer_count"] = 2
    second_event = json.loads(json.dumps(event))
    second_event["module_name"] = second_layer["module_name"]
    second_event["layer_idx"] = 1
    second_event["identity"] = second_layer["identity"]
    proof["text_layer_events"].append(second_event)
    proof["expected_text_layer_count"] = 2
    proof["observed_text_layer_count"] = 2
    return proof


def _failure_arm(
    name: str,
    *,
    microsteps: int,
    proof: dict[str, object],
    gradient_count: int = 1,
) -> dict[str, object]:
    boundaries = [0, 5] if name.endswith("negative") else [0, 2, 5]
    forward_receipts = [
        {
            "fa2_varlen": {
                "segment_boundaries": boundaries,
                "proof": proof if name == "packed_clean" else None,
            }
        }
        for _ in range(microsteps)
    ]
    return {
        "name": name,
        "microstep_count": microsteps,
        "total_loss_fp32": 1.0,
        "loss_artifact": {"status": "complete"},
        "semantic_logit_rows": 1,
        "gradient_inventory": [
            {
                "name": f"weight_{index:04d}",
                "shape": [1],
                "parameter_dtype": "torch.float32",
                "parameter_storage_dtype": "torch.float32",
                "gradient_dtype": "torch.float32",
                "gradient_provenance_dtype": "torch.bfloat16",
                "grad_status": "finite",
                "gradient_sha256": "f" * 64,
            }
            for index in range(min(gradient_count, 64))
        ],
        "forward_receipts": forward_receipts,
        "forward_logits_dtypes": ["torch.float32"] * microsteps,
        "inner_autocast": [{} for _ in range(microsteps)],
        "backward_cadence": {
            "microstep_count": microsteps,
            "harness_backward_call_count": 1,
            "harness_policy": "sum_then_single_backward",
            "production_backward_call_count": microsteps,
            "production_policy": "streaming_backward_per_microstep",
            "cadence_matches_production": microsteps == 1,
        },
        "gradient_inventory_count": gradient_count,
        "gradient_inventory_omitted": max(0, gradient_count - 64),
    }


def _valid_rich_clean_failed_receipt(*, gradient_count: int = 1) -> dict[str, object]:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    passed = _passed_receipt(plan)
    passed["source_identity"] = {
        "config_identity": plan["config_identity"],
        "repo_identity": plan["repo_identity"],
        "dependency_identity": plan["dependency_identity"],
        "model_identity": _component_identity(load_model=True),
        "model_weight_identity": plan["model_weight_identity"],
        "source_owners": plan["source_owners"],
    }
    atom = _atom("a", 1)
    atom_key = SemanticAtomKey.from_atom(atom)
    clean_logits = {atom_key: torch.tensor([0.0, 1.0, 2.0, 3.0])}
    reference_logits = {atom_key: clean_logits[atom_key].clone()}
    negative_logits_values = {atom_key: clean_logits[atom_key] + 1.0}
    clean_gradient_records = tuple(
        GradientRecord(
            f"weight_{index:04d}",
            (1,),
            "torch.float32",
            torch.ones(1),
            gradient_dtype="torch.float32",
            gradient_provenance_dtype="torch.bfloat16",
        )
        for index in range(gradient_count)
    )
    reference_gradient_records = tuple(
        GradientRecord(
            f"weight_{index:04d}",
            (1,),
            "torch.float32",
            torch.ones(1) + (1.0 if index == 0 else 0.0),
            gradient_dtype="torch.float32",
            gradient_provenance_dtype="torch.bfloat16",
        )
        for index in range(gradient_count)
    )
    gradients = compare_gradient_inventories(
        clean_gradient_records, reference_gradient_records
    )
    negative_gradients = compare_gradient_inventories(
        clean_gradient_records, clean_gradient_records
    )
    comparisons = {
        "semantic_atoms": compare_semantic_atom_inventories(
            _Sequence((atom,)), _Sequence((atom,))
        ),
        "denominators": _normalization_preflight(),
        "supervised_logits": compare_keyed_logits(clean_logits, reference_logits),
        "loss": {
            "passed": True,
            "rtol": parity_module.BF16_RTOL,
            "atol": parity_module.BF16_ATOL,
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "compared_value_count": 1,
        },
        "cross_arm_bf16_loss_term_scalars": probe._compare_loss_terms(
            _loss_artifact(), _loss_artifact()
        ),
        "gradients": gradients,
    }
    negative_logits = compare_keyed_logits(clean_logits, negative_logits_values)
    negative_loss = parity_module.compare_tensors(
        torch.tensor(1.0),
        torch.tensor(2.0),
        rtol=parity_module.BF16_RTOL,
        atol=parity_module.BF16_ATOL,
    ).to_artifact_dict()
    discriminator = negative_discriminator(
        clean_boundaries=[0, 2, 5],
        negative_boundaries=[0, 5],
        logits_allclose=bool(negative_logits["passed"]),
        loss_allclose=bool(negative_loss["allclose"]),
        gradients_allclose=bool(negative_gradients["passed"]),
    )
    proof = _failure_proof()
    arms = {
        "packed_clean": _failure_arm(
            "packed_clean",
            microsteps=1,
            proof=proof,
            gradient_count=gradient_count,
        ),
        "separate_reference": _failure_arm(
            "separate_reference",
            microsteps=2,
            proof=proof,
            gradient_count=gradient_count,
        ),
        "packed_merged_boundary_negative": _failure_arm(
            "packed_merged_boundary_negative",
            microsteps=1,
            proof=proof,
            gradient_count=gradient_count,
        ),
    }
    executed_negative = arms["packed_merged_boundary_negative"]["forward_receipts"][0][
        "fa2_varlen"
    ]
    negative = {
        **discriminator,
        "attestation": {
            "status": "rejected_against_frozen_clean_boundary",
            "expected_clean_boundaries": [0, 2, 5],
            "observed_negative_boundaries": [0, 5],
            "boundary_mismatch_detected": True,
            "proof_disabled": True,
            "executed_negative_varlen_receipt": executed_negative,
        },
        "supervised_logits": negative_logits,
        "total_loss": negative_loss,
        "gradients": negative_gradients,
    }
    evidence = probe._FailureEvidenceAccumulator(requested_device="cuda:0")
    evidence.reach("comparisons")
    for phase in (
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "negative_control",
    ):
        evidence.complete_phase(phase)
    for name, value in (
        ("source_identity", passed["source_identity"]),
        ("model_identity_attestation", passed["model_identity_attestation"]),
        ("execution", passed["execution"]),
        ("attempt_marker", passed["attempt_marker"]),
        ("trainable_inventory", passed["trainable_inventory"]),
        ("arms", arms),
        ("proof", proof),
        ("comparisons", comparisons),
        ("negative_discriminator", negative),
        (
            "timings",
            {
                "clock": "time.perf_counter_ns_with_cuda_synchronize",
                "scope": "forward_only_same_packed_inputs_train_mode_no_backward",
                "proof_on_forward_ns": 12,
                "proof_off_forward_ns": 10,
                "proof_overhead_ns": 2,
            },
        ),
        (
            "measurement",
            {
                "schema": "coordexp-swift-wave2-failure-measurement-v1",
                "completed_phases": [
                    {
                        "name": phase,
                        "start_ns": index * 10,
                        "end_ns": index * 10 + 5,
                        "duration_ns": 5,
                        "status": "completed",
                    }
                    for index, phase in enumerate(evidence.completed_phases)
                ],
                "phase_boundary_samples": [],
            },
        ),
    ):
        evidence.record(name, value)
    receipt = probe._failure_receipt(
        plan,
        device_text="cuda:0",
        exc=ParityContractError(
            "clean packed-versus-separate parity failed",
            code="qwen.parity.clean_failed",
            context={},
        ),
        evidence=evidence,
    )
    return receipt


def _valid_rich_clean_failed_receipt(  # noqa: F811
    *, gradient_count: int = 589
) -> dict[str, object]:
    if gradient_count != 589:
        raise ValueError("v3 rich fixture preserves the exact 589-row inventory")
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    completed_execution = receipt["execution"]
    accelerator = completed_execution["accelerator"]
    retained_execution = {
        "requested_device": "cuda:0",
        "device": "cuda:0",
        "gpu_idle_preflight": json.loads(
            json.dumps(receipt["measurement"]["gpu_idle_preflight"])
        ),
        "accelerator": {
            key: accelerator[key]
            for key in (
                "distributed_type",
                "rank",
                "local_rank",
                "world_size",
                "device",
                "cuda_current_device",
                "mixed_precision",
                "native_amp",
                "gradient_accumulation_steps",
                "scaler",
                "accelerate_torch_device",
            )
        },
        "model_dtype": completed_execution["model_dtype"],
        "train_mode": completed_execution["train_mode"],
        "use_cache": completed_execution["use_cache"],
        "optimizer": completed_execution["optimizer"],
        "memory_savers": completed_execution["memory_savers"],
        "adapter": completed_execution["adapter"],
        "special_token_embeddings": completed_execution["special_token_embeddings"],
        "trainable_value_identity_before": completed_execution[
            "trainable_value_identity_before"
        ],
        "prepared_model_attestation": accelerator["prepared_model_attestation"],
    }
    receipt["execution"] = probe._merge_completed_failure_execution(
        retained_execution,
        completed_execution,
    )
    for arm in receipt["arms"].values():
        inventory = arm["gradient_inventory"]
        arm["gradient_inventory"] = inventory[:64]
        arm["gradient_inventory_count"] = len(inventory)
        arm["gradient_inventory_omitted"] = max(0, len(inventory) - 64)
    receipt["timings"] = {
        "clock": "time.perf_counter_ns_with_cuda_synchronize",
        "scope": "forward_only_same_packed_inputs_train_mode_no_backward",
        "proof_on_forward_ns": 12,
        "proof_off_forward_ns": 10,
        "proof_overhead_ns": 2,
    }
    completed_phases = [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "negative_control",
    ]
    receipt["measurement"] = {
        "schema": "coordexp-swift-wave2-failure-measurement-v1",
        "completed_phases": [
            {
                "name": phase,
                "start_ns": index * 10,
                "end_ns": index * 10 + 5,
                "duration_ns": 5,
                "status": "completed",
            }
            for index, phase in enumerate(completed_phases)
        ],
        "phase_boundary_samples": [],
    }
    receipt["gpu_memory"] = {}
    receipt["terminal_status"] = "failed"
    receipt["failure"] = {
        "type": "ParityContractError",
        "message": "injected post-comparison failure",
        "code": "qwen.parity.injected_post_comparison",
        "evidence": {
            "schema": PARITY_FAILURE_EVIDENCE_SCHEMA,
            "stage_reached": "comparisons",
            "completed_phases": completed_phases,
            "completed_fields": [
                "source_identity",
                "model_identity_attestation",
                "execution",
                "attempt_marker",
                "trainable_inventory",
                "arms",
                "proof",
                "comparisons",
                "negative_discriminator",
                "timings",
                "measurement",
            ],
        },
    }
    return receipt


def test_valid_rich_clean_failed_receipt_preserves_completed_evidence() -> None:
    receipt = _valid_rich_clean_failed_receipt()
    plan = finalize_plan(_plan_body())
    assert validate_parity_receipt(receipt, expected_plan=plan) == receipt
    assert receipt["terminal_status"] == "failed"
    execution = receipt["execution"]
    assert execution["requested_device"] == execution["device"] == "cuda:0"
    assert (
        execution["gpu_idle_preflight"] == _measurement(v3=True)["gpu_idle_preflight"]
    )
    assert [check["name"] for check in execution["gpu_idle_preflight"]["checks"]] == [
        "initial_before_cpu_model_work",
        "final_before_gpu_work",
    ]
    failure = receipt["failure"]
    assert isinstance(failure, dict)
    evidence = failure["evidence"]
    assert evidence["schema"] == PARITY_FAILURE_EVIDENCE_SCHEMA
    assert evidence["stage_reached"] == "comparisons"
    assert evidence["completed_phases"][-1] == "negative_control"


def test_immutable_wave2_v3_failure_receipt_remains_readable_without_rewrite() -> None:
    receipt_dir = (
        Path(__file__).resolve().parents[2]
        / "openspec/changes/archive/2026-08-12-harden-optimize-coordexp-swift-training-infrastructure/receipts"
    )
    plan_path = receipt_dir / "wave2-v3-plan.json"
    receipt_path = receipt_dir / "wave2-v3-terminal-receipt.json"
    plan_bytes = plan_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    assert parity_module.sha256_file(plan_path) == (
        "141b4294dacae69f39907303d3ae75c4c4182d8da7625b399280ed99ed66187d"
    )
    assert parity_module.sha256_file(receipt_path) == (
        "781838ada548c9c0d9db4dacf1b43b6bc0e767303fa2906006c95a3ce70e6958"
    )
    plan = load_strict_json(plan_path)
    receipt = load_strict_json(receipt_path)
    marker_reference = receipt["attempt_marker"]
    assert receipt["schema"] == PARITY_RECEIPT_SCHEMA
    assert receipt["terminal_status"] == "failed"
    assert receipt["failure"]["code"] == "qwen.parity.clean_failed"
    assert marker_reference["path"] == (
        "/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/"
        "harden-optimize-coordexp-swift-training-infrastructure/receipts/"
        "wave2-v3-attempt-marker.json"
    )
    assert marker_reference["expected_receipt_target"] == (
        "/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/"
        "harden-optimize-coordexp-swift-training-infrastructure/receipts/"
        "wave2-v3-terminal-receipt.json"
    )

    archived_marker = load_strict_json(receipt_dir / "wave2-v3-attempt-marker.json")
    assert archived_marker["schema"] == PARITY_ATTEMPT_MARKER_SCHEMA
    assert archived_marker["status"] == "attempt_started"
    assert archived_marker["marker_sha256"] == marker_reference["marker_sha256"]
    assert parity_module.validate_attempt_marker(
        archived_marker,
        expected_plan=plan,
    ) == archived_marker
    assert plan_path.read_bytes() == plan_bytes
    assert receipt_path.read_bytes() == receipt_bytes


@pytest.mark.parametrize("mutation", ["unrelated_message", "missing_execution_field"])
def test_immutable_wave2_v3_failure_exception_rejects_mutated_receipts(
    mutation: str,
) -> None:
    receipt_dir = (
        Path(__file__).resolve().parents[2]
        / "openspec/changes/archive/2026-08-12-harden-optimize-coordexp-swift-training-infrastructure/receipts"
    )
    plan = load_strict_json(receipt_dir / "wave2-v3-plan.json")
    receipt = load_strict_json(receipt_dir / "wave2-v3-terminal-receipt.json")
    if mutation == "unrelated_message":
        receipt["failure"]["message"] += " mutated"
    else:
        receipt["execution"].pop("use_cache")

    with pytest.raises(ParityContractError):
        validate_parity_receipt(receipt, expected_plan=plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_requested_device",
        "missing_preflight",
        "missing_final_check",
        "forged_final_sample",
    ],
)
def test_v3_completed_failure_rejects_missing_or_forged_preflight(
    mutation: str,
) -> None:
    receipt = json.loads(json.dumps(_valid_rich_clean_failed_receipt()))
    plan = finalize_plan(_plan_body())
    execution = receipt["execution"]
    if mutation == "missing_requested_device":
        execution.pop("requested_device")
    elif mutation == "missing_preflight":
        execution.pop("gpu_idle_preflight")
    elif mutation == "missing_final_check":
        execution["gpu_idle_preflight"]["checks"].pop()
    else:
        final_sample = execution["gpu_idle_preflight"]["checks"][1]["samples"][0]
        final_sample["uuid"] = "GPU-forged"
    with pytest.raises(ParityContractError):
        validate_parity_receipt(receipt, expected_plan=plan)


_V3_STAGE_PHASES = {
    "initialized": [],
    "receipt_target_preflight": [],
    "plan_loaded": [],
    "plan_revalidated": [],
    "cuda_preflight": [],
    "normalization_preflight": [],
    "attempt_started": [],
    "accelerator_ready": [],
    "model_setup": ["model_setup"],
    "input_construction": ["model_setup", "input_construction"],
    "warmup_forward": [
        "model_setup",
        "input_construction",
        "warmup_forward",
    ],
    "proof_off_forward": [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
    ],
    "packed_primary": [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        "packed_primary",
    ],
    "packed_repeat": [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        "packed_primary",
        "packed_repeat",
    ],
    "separate_reference": [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        "packed_primary",
        "packed_repeat",
        "separate_reference",
    ],
    "negative_control": [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "negative_control",
    ],
    "comparisons": [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "negative_control",
    ],
    "finalization": [
        "model_setup",
        "input_construction",
        "warmup_forward",
        "proof_off_forward",
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "negative_control",
        "comparison",
    ],
}
_V3_STAGE_ARMS = {
    "packed_primary": ("packed_primary",),
    "packed_repeat": ("packed_primary", "packed_repeat"),
    "separate_reference": (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
    ),
    "negative_control": (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "packed_merged_boundary_negative",
    ),
    "comparisons": (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "packed_merged_boundary_negative",
    ),
    "finalization": (
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "packed_merged_boundary_negative",
    ),
}


def _v3_failure_at_stage(stage: str) -> tuple[dict[str, object], dict[str, object]]:
    plan = finalize_plan(_plan_body())
    receipt = _valid_rich_clean_failed_receipt()
    stage_names = list(_V3_STAGE_PHASES)
    stage_index = stage_names.index(stage)
    marker_index = stage_names.index("attempt_started")
    model_index = stage_names.index("model_setup")
    arm_names = _V3_STAGE_ARMS.get(stage, ())
    base_execution = {
        "requested_device": "cuda:0",
        "device": "cuda:0",
        "gpu_idle_preflight": {
            "status": "passed",
            "requested_device": "cuda:0",
        },
    }
    accelerator_identity = {
        key: receipt["execution"]["accelerator"][key]
        for key in (
            "distributed_type",
            "rank",
            "local_rank",
            "world_size",
            "device",
            "cuda_current_device",
            "mixed_precision",
            "native_amp",
            "gradient_accumulation_steps",
            "scaler",
            "accelerate_torch_device",
        )
    }
    if stage_index < 2:
        receipt["plan_sha256"] = "0" * 64
        receipt["source_identity"] = {}
    elif stage_index < marker_index:
        receipt["source_identity"] = {
            "config_identity": plan["config_identity"],
            "repo_identity": plan["repo_identity"],
            "dependency_identity": plan["dependency_identity"],
            "model_identity": plan["model_identity"],
            "model_weight_identity": plan["model_weight_identity"],
            "source_owners": plan["source_owners"],
        }
    if stage_index < stage_names.index("cuda_preflight"):
        receipt["execution"] = {"requested_device": "cuda:0"}
    elif stage in {"cuda_preflight", "normalization_preflight", "attempt_started"}:
        receipt["execution"] = base_execution
    elif stage == "accelerator_ready":
        receipt["execution"] = {
            **base_execution,
            "accelerator": accelerator_identity,
        }
    elif stage not in {"comparisons", "finalization"}:
        receipt["execution"] = {
            **base_execution,
            "accelerator": accelerator_identity,
            "model_dtype": "torch.bfloat16",
            "train_mode": True,
            "use_cache": False,
            "optimizer": None,
            "memory_savers": {},
            "adapter": {},
            "special_token_embeddings": {},
            "trainable_value_identity_before": ["same"],
            "prepared_model_attestation": receipt["execution"]["accelerator"][
                "prepared_model_attestation"
            ],
        }
    if stage_index < marker_index:
        receipt["attempt_marker"] = {}
        receipt["trainable_inventory"] = {}
    if stage_index < model_index:
        receipt["model_identity_attestation"] = {}
        receipt["measurement"] = {}
    else:
        receipt["measurement"] = {
            "schema": "coordexp-swift-wave2-failure-measurement-v1",
            "completed_phases": [
                {
                    "name": phase,
                    "start_ns": index * 10,
                    "end_ns": index * 10 + 5,
                    "duration_ns": 5,
                    "status": "completed",
                }
                for index, phase in enumerate(_V3_STAGE_PHASES[stage])
            ],
            "phase_boundary_samples": [],
        }
    if not arm_names:
        receipt["arms"] = {}
        receipt["proof"] = {}
        receipt["timings"] = {}
    else:
        receipt["arms"] = {name: receipt["arms"][name] for name in arm_names}
    if stage not in {"comparisons", "finalization"}:
        receipt["comparisons"] = {}
        receipt["negative_discriminator"] = {}
    receipt["gpu_memory"] = {}
    field_order = (
        "source_identity",
        "model_identity_attestation",
        "execution",
        "attempt_marker",
        "trainable_inventory",
        "arms",
        "proof",
        "comparisons",
        "negative_discriminator",
        "timings",
        "gpu_memory",
        "measurement",
    )
    receipt["failure"]["evidence"] = {
        "schema": PARITY_FAILURE_EVIDENCE_SCHEMA,
        "stage_reached": stage,
        "completed_phases": list(_V3_STAGE_PHASES[stage]),
        "completed_fields": [name for name in field_order if receipt[name]],
    }
    return plan, receipt


@pytest.mark.parametrize("stage", list(_V3_STAGE_PHASES))
def test_v3_rich_failure_accepts_exact_stage_phase_and_arm_prefix(stage: str) -> None:
    plan, receipt = _v3_failure_at_stage(stage)
    expected_plan = None if receipt["plan_sha256"] == "0" * 64 else plan
    assert validate_parity_receipt(receipt, expected_plan=expected_plan) == receipt


@pytest.mark.parametrize("stage", list(_V3_STAGE_PHASES))
def test_v3_rich_failure_rejects_stage_phase_contradiction(stage: str) -> None:
    plan, receipt = _v3_failure_at_stage(stage)
    phases = receipt["failure"]["evidence"]["completed_phases"]
    if phases:
        phases.pop()
    else:
        phases.append("model_setup")
    expected_plan = None if receipt["plan_sha256"] == "0" * 64 else plan
    with pytest.raises(ParityContractError, match="completed phases"):
        validate_parity_receipt(receipt, expected_plan=expected_plan)


def test_v3_rich_failure_rejects_future_or_mutated_partial_arm() -> None:
    plan, receipt = _v3_failure_at_stage("model_setup")
    template = _valid_rich_clean_failed_receipt()
    receipt["arms"] = {"packed_primary": template["arms"]["packed_primary"]}
    receipt["proof"] = template["proof"]
    fields = receipt["failure"]["evidence"]["completed_fields"]
    fields.remove("measurement")
    fields.extend(["arms", "proof", "measurement"])
    with pytest.raises(ParityContractError, match="completed fields|arm inventory"):
        validate_parity_receipt(receipt, expected_plan=plan)

    plan, receipt = _v3_failure_at_stage("packed_repeat")
    receipt["arms"]["packed_repeat"]["backward_cadence"][
        "harness_backward_call_count"
    ] = 2
    with pytest.raises(ParityContractError, match="cadence"):
        validate_parity_receipt(receipt, expected_plan=plan)


def test_v3_negative_undetected_failure_remains_publishable() -> None:
    plan, receipt = _v3_failure_at_stage("comparisons")
    negative = receipt["negative_discriminator"]
    negative["supervised_logits"] = json.loads(
        json.dumps(
            receipt["comparisons"]["packed_primary_vs_separate"]["supervised_logits"]
        )
    )
    negative["forward_detected_by"] = []
    negative["detected"] = False
    receipt["failure"]["code"] = "qwen.parity.negative_undetected"
    receipt["failure"]["message"] = "merged boundary produced no forward signal"
    assert validate_parity_receipt(receipt, expected_plan=plan) == receipt


def test_v3_repeat_unmeasurable_failure_remains_publishable() -> None:
    plan, receipt = _v3_failure_at_stage("comparisons")
    primary = _synthetic_gradient_records()
    repeat = list(_synthetic_gradient_records())
    repeat[0] = replace(repeat[0], grad=repeat[0].grad + 0.01)
    repeat_artifact = compare_packed_gradient_repeat(primary, tuple(repeat))
    assert repeat_artifact["status"] == "unmeasurable"
    receipt["comparisons"]["packed_repeat_measurability"] = repeat_artifact
    receipt["terminal_status"] = "unmeasurable"
    receipt["failure"]["code"] = "qwen.parity.packed_repeat_unmeasurable"
    receipt["failure"]["message"] = "packed repeat exceeded the fixed gate"
    assert validate_parity_receipt(receipt, expected_plan=plan) == receipt
    assert [
        check["name"] for check in receipt["execution"]["gpu_idle_preflight"]["checks"]
    ] == ["initial_before_cpu_model_work", "final_before_gpu_work"]


@pytest.mark.parametrize(
    "mutation",
    [
        "proof",
        "proof_identity",
        "proof_duplicate",
        "proof_extra",
        "proof_reordered",
        "gradient_inventory_binding",
        "reversed_no_sync",
        "empty_forward_evidence",
        "arm_loss_binding",
        "loss_term_binding",
        "loss_term_inventory",
        "loss_term_weighted_sum",
        "repeat_value_count",
        "repeat_value_count_short",
        "execution",
        "timings",
        "measurement",
        "gpu_memory",
        "marker_inventory",
    ],
)
def test_v3_passed_receipt_rejects_deep_attestation_mutations(mutation: str) -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    if mutation == "proof":
        receipt["proof"]["text_layer_events"] = []
    elif mutation == "proof_identity":
        receipt["proof"]["text_layer_events"][0]["layer_idx"] = 1
    elif mutation == "proof_duplicate":
        receipt["proof"]["text_layer_events"][1] = json.loads(
            json.dumps(receipt["proof"]["text_layer_events"][0])
        )
    elif mutation == "proof_extra":
        receipt["proof"]["text_layer_events"].append(
            json.loads(json.dumps(receipt["proof"]["text_layer_events"][1]))
        )
    elif mutation == "proof_reordered":
        receipt["proof"]["text_layer_events"].reverse()
    elif mutation == "gradient_inventory_binding":
        for comparison_name in (
            "packed_primary_vs_separate",
            "packed_repeat_vs_separate",
        ):
            artifact = receipt["comparisons"][comparison_name]["gradients"]
            row = artifact["parameters"][0]
            artifact["parameters"] = [row]
            artifact["left_count"] = 1
            artifact["right_count"] = 1
            artifact["matched_count"] = 1
            artifact["compared_value_count"] = row["compared_value_count"]
            artifact["max_abs_diff"] = row["max_abs_diff"]
    elif mutation == "reversed_no_sync":
        events = receipt["arms"]["separate_reference"]["backward_cadence"]["events"]
        events[0]["sync_gradients"] = True
        events[0]["accumulation_context"] = "sync_gradients"
        events[1]["sync_gradients"] = False
        events[1]["accumulation_context"] = "accelerator.no_sync"
    elif mutation == "empty_forward_evidence":
        arm = receipt["arms"]["separate_reference"]
        arm["forward_receipts"] = []
        arm["forward_logits_dtypes"] = []
        arm["inner_autocast"] = []
    elif mutation == "arm_loss_binding":
        arm = receipt["arms"]["separate_reference"]
        arm["total_loss_fp32"] = 1000.0
        arm["loss_artifact"]["total_loss"] = 1000.0
    elif mutation == "loss_term_binding":
        term = receipt["arms"]["separate_reference"]["loss_artifact"]["terms"][0]
        term["raw_loss"] = 999.0
        term["segment_mean_numerator"] = 999.0
        term["token_weighted_diagnostic"] = 999.0
    elif mutation == "loss_term_inventory":
        receipt["arms"]["separate_reference"]["loss_artifact"]["terms"][0]["name"] = (
            "forged_term"
        )
    elif mutation == "loss_term_weighted_sum":
        receipt["arms"]["separate_reference"]["loss_artifact"]["terms"][0][
            "weighted_loss"
        ] = 999.0
    elif mutation == "repeat_value_count":
        receipt["comparisons"]["packed_repeat_measurability"][
            "compared_value_count"
        ] = 0
    elif mutation == "repeat_value_count_short":
        receipt["comparisons"]["packed_repeat_measurability"][
            "compared_value_count"
        ] -= 1
    elif mutation == "execution":
        receipt["execution"]["accelerator"]["native_amp"] = False
    elif mutation == "timings":
        receipt["timings"]["ordering"][-1] = "timed_proof_on_wrong_arm"
    elif mutation == "measurement":
        receipt["measurement"]["phases"][4]["name"] = "packed_wrong"
    elif mutation == "gpu_memory":
        receipt["gpu_memory"]["peak_reserved_bytes"] += 1
    else:
        receipt["trainable_inventory"]["concrete"]["parameters"][0]["shape"] = [2]
    with pytest.raises(ParityContractError):
        validate_parity_receipt(receipt, expected_plan=plan)


def test_v3_offline_marker_reference_binds_expected_receipt_target() -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    receipt["attempt_marker"]["expected_receipt_target"] = str(
        Path(receipt["attempt_marker"]["expected_receipt_target"]).with_name(
            "foreign-receipt.json"
        )
    )
    with pytest.raises(ParityContractError) as caught:
        validate_parity_receipt(receipt, expected_plan=plan)
    assert caught.value.code == "qwen.parity.marker_receipt_binding"


def _drop_same_semantic_logit_subset(artifact: dict[str, object]) -> None:
    rows = artifact["rows"]
    assert isinstance(rows, list) and len(rows) == 2
    artifact["rows"] = rows[:1]
    artifact["left_count"] = 1
    artifact["right_count"] = 1
    artifact["row_count"] = 1
    artifact["max_abs_diff"] = rows[0]["max_abs_diff"]
    artifact["max_rel_diff"] = rows[0]["max_rel_diff"]


def _mutate_aligned_semantic_key_inventory(
    artifact: dict[str, object], mutation: str
) -> None:
    rows = artifact["rows"]
    assert isinstance(rows, list) and len(rows) == 2
    if mutation == "reorder":
        rows.reverse()
    else:
        key = rows[0]["key"]
        if mutation == "example_id":
            key["example_id"] = "zz-substituted-example"
        elif mutation == "token_id":
            key["token_id"] += 100
        elif mutation == "position":
            key["logical_target_position"] += 100
            key["logical_target_end"] += 100
        else:
            raise AssertionError(f"unsupported semantic-key mutation: {mutation}")
        rows.sort(key=lambda row: SemanticAtomKey(**row["key"]))
    artifact["semantic_key_inventory_sha256"] = semantic_atom_key_inventory_sha256(
        tuple(row["key"] for row in rows)
    )


def _semantic_logit_surface(
    receipt: dict[str, object], surface: str
) -> dict[str, object]:
    if surface == "negative":
        return receipt["negative_discriminator"]["supervised_logits"]
    comparison_name = {
        "primary": "packed_primary_vs_separate",
        "repeat": "packed_repeat_vs_separate",
    }[surface]
    return receipt["comparisons"][comparison_name]["supervised_logits"]


@pytest.mark.parametrize("surface", ["primary", "repeat", "negative"])
@pytest.mark.parametrize("mutation", ["example_id", "token_id", "position", "reorder"])
def test_v3_passed_receipt_rejects_same_cardinality_semantic_key_drift(
    surface: str,
    mutation: str,
) -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    artifact = _semantic_logit_surface(receipt, surface)
    _mutate_aligned_semantic_key_inventory(artifact, mutation)
    assert artifact["left_count"] == artifact["right_count"] == 2
    assert artifact["row_count"] == 2

    with pytest.raises(ParityContractError):
        validate_parity_receipt(receipt, expected_plan=plan)


@pytest.mark.parametrize("surface", ["primary", "repeat", "negative"])
@pytest.mark.parametrize("mutation", ["example_id", "token_id", "position", "reorder"])
def test_v3_comparisons_failure_rejects_same_cardinality_semantic_key_drift(
    surface: str,
    mutation: str,
) -> None:
    plan, receipt = _v3_failure_at_stage("comparisons")
    artifact = _semantic_logit_surface(receipt, surface)
    _mutate_aligned_semantic_key_inventory(artifact, mutation)
    assert artifact["left_count"] == artifact["right_count"] == 2
    assert artifact["row_count"] == 2

    with pytest.raises(ParityContractError):
        validate_parity_receipt(receipt, expected_plan=plan)


@pytest.mark.parametrize(
    "surface",
    [
        "semantic_left",
        "semantic_right",
        "packed_primary",
        "packed_repeat",
        "separate_reference",
        "packed_merged_boundary_negative",
    ],
)
@pytest.mark.parametrize("terminal", ["passed", "comparisons_failure"])
def test_v3_receipt_rejects_plan_semantic_key_digest_drift(
    surface: str,
    terminal: str,
) -> None:
    if terminal == "passed":
        plan = finalize_plan(_plan_body())
        receipt = _passed_receipt(plan)
    else:
        plan, receipt = _v3_failure_at_stage("comparisons")
    if surface == "semantic_left":
        receipt["comparisons"]["semantic_atoms"]["left_key_inventory_sha256"] = "f" * 64
    elif surface == "semantic_right":
        receipt["comparisons"]["semantic_atoms"]["right_key_inventory_sha256"] = (
            "f" * 64
        )
    else:
        receipt["arms"][surface]["semantic_key_inventory_sha256"] = "f" * 64

    with pytest.raises(ParityContractError):
        validate_parity_receipt(receipt, expected_plan=plan)


@pytest.mark.parametrize(
    "comparison_name",
    ["packed_primary_vs_separate", "packed_repeat_vs_separate"],
)
def test_v3_passed_receipt_rejects_same_subset_semantic_logit_drop(
    comparison_name: str,
) -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    artifact = receipt["comparisons"][comparison_name]["supervised_logits"]
    _drop_same_semantic_logit_subset(artifact)
    assert artifact["passed"] is True

    with pytest.raises(ParityContractError) as caught:
        validate_parity_receipt(receipt, expected_plan=plan)
    assert caught.value.code == "qwen.parity.semantic_logit_coverage"


@pytest.mark.parametrize(
    "surface",
    [
        "semantic_left",
        "semantic_right",
        "primary_logits_left",
        "primary_logits_right",
        "primary_logits_rows",
        "repeat_logits_left",
        "repeat_logits_right",
        "repeat_logits_rows",
        "negative_logits_left",
        "negative_logits_right",
        "negative_logits_rows",
        "packed_primary_arm",
        "packed_repeat_arm",
        "separate_reference_arm",
        "negative_arm",
    ],
)
def test_v3_passed_receipt_binds_every_semantic_count_surface(surface: str) -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    huge = 10**100
    if surface == "semantic_left":
        receipt["comparisons"]["semantic_atoms"]["left_count"] = huge
    elif surface == "semantic_right":
        receipt["comparisons"]["semantic_atoms"]["right_count"] = huge
    elif surface.startswith("primary_logits"):
        side = surface.rsplit("_", 1)[1]
        receipt["comparisons"]["packed_primary_vs_separate"]["supervised_logits"][
            "row_count" if side == "rows" else f"{side}_count"
        ] = huge
    elif surface.startswith("repeat_logits"):
        side = surface.rsplit("_", 1)[1]
        receipt["comparisons"]["packed_repeat_vs_separate"]["supervised_logits"][
            "row_count" if side == "rows" else f"{side}_count"
        ] = huge
    elif surface.startswith("negative_logits"):
        side = surface.rsplit("_", 1)[1]
        receipt["negative_discriminator"]["supervised_logits"][
            "row_count" if side == "rows" else f"{side}_count"
        ] = huge
    else:
        arm_name = {
            "packed_primary_arm": "packed_primary",
            "packed_repeat_arm": "packed_repeat",
            "separate_reference_arm": "separate_reference",
            "negative_arm": "packed_merged_boundary_negative",
        }[surface]
        receipt["arms"][arm_name]["semantic_logit_rows"] = huge

    with pytest.raises(ParityContractError):
        validate_parity_receipt(receipt, expected_plan=plan)


@pytest.mark.parametrize(
    "surface",
    [
        "semantic_atoms",
        "packed_primary_logits",
        "packed_repeat_logits",
        "negative_logits",
        "packed_primary_arm",
        "packed_repeat_arm",
        "separate_reference_arm",
        "negative_arm",
    ],
)
def test_v3_comparisons_failure_binds_every_semantic_count_surface(
    surface: str,
) -> None:
    plan, receipt = _v3_failure_at_stage("comparisons")
    if surface == "semantic_atoms":
        receipt["comparisons"]["semantic_atoms"]["left_count"] = 1
        receipt["comparisons"]["semantic_atoms"]["right_count"] = 1
    elif surface.endswith("_logits"):
        if surface == "negative_logits":
            artifact = receipt["negative_discriminator"]["supervised_logits"]
        else:
            comparison_name = (
                "packed_primary_vs_separate"
                if surface.startswith("packed_primary")
                else "packed_repeat_vs_separate"
            )
            artifact = receipt["comparisons"][comparison_name]["supervised_logits"]
        _drop_same_semantic_logit_subset(artifact)
    else:
        arm_name = {
            "packed_primary_arm": "packed_primary",
            "packed_repeat_arm": "packed_repeat",
            "separate_reference_arm": "separate_reference",
            "negative_arm": "packed_merged_boundary_negative",
        }[surface]
        receipt["arms"][arm_name]["semantic_logit_rows"] = 1

    with pytest.raises(ParityContractError):
        validate_parity_receipt(receipt, expected_plan=plan)


def test_v3_pre_marker_failure_with_resource_summary_is_publishable() -> None:
    plan, receipt = _v3_failure_at_stage("normalization_preflight")
    receipt["gpu_memory"] = {
        "scope": "whole_probe_until_failure",
        "peak_allocated_bytes": 11,
        "peak_reserved_bytes": 13,
    }
    receipt["measurement"] = {
        "schema": "coordexp-swift-wave2-failure-measurement-v1",
        "completed_phases": [],
        "phase_boundary_samples": [],
        "failure_resource_summary": {
            "device": "cuda:0",
            "torch_peak_allocated_bytes": 11,
            "torch_peak_reserved_bytes": 13,
            "device_sampler": {"status": "completed"},
        },
    }
    receipt["failure"]["evidence"]["completed_fields"] = [
        "source_identity",
        "execution",
        "gpu_memory",
        "measurement",
    ]
    assert validate_parity_receipt(receipt, expected_plan=plan) == receipt


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_passed",
        "forged_completed_field",
        "stage_phase_contradiction",
        "fabricated_execution",
        "fabricated_arm",
        "fabricated_negative",
        "fabricated_config_identity",
        "fabricated_repo_identity",
        "truncated_source_owners",
        "forged_gradient_pass",
    ],
)
def test_rich_failed_receipt_rejects_strict_mutations(mutation: str) -> None:
    receipt = _valid_rich_clean_failed_receipt()
    plan = finalize_plan(_plan_body())
    mutated = json.loads(json.dumps(receipt))
    if mutation == "extra_passed":
        mutated["failure"]["evidence"]["passed"] = True
    elif mutation == "forged_completed_field":
        mutated["failure"]["evidence"]["completed_fields"].remove("comparisons")
    elif mutation == "stage_phase_contradiction":
        mutated["failure"]["evidence"]["stage_reached"] = "initialized"
    elif mutation == "fabricated_execution":
        mutated["attempt_marker"]["marker_sha256"] = "f" * 64
    elif mutation == "fabricated_arm":
        mutated["arms"]["packed_primary"]["backward_cadence"][
            "harness_backward_call_count"
        ] = 2
    elif mutation == "fabricated_negative":
        mutated["negative_discriminator"]["detected"] = False
    elif mutation == "fabricated_config_identity":
        mutated["source_identity"]["config_identity"]["fingerprint"] = "forged"
    elif mutation == "fabricated_repo_identity":
        mutated["source_identity"]["repo_identity"]["head"] = "forged"
    elif mutation == "truncated_source_owners":
        mutated["source_identity"]["source_owners"].pop()
    else:
        mutated["comparisons"]["packed_primary_vs_separate"]["gradients"]["parameters"][
            0
        ]["rtol"] = 1.0e-4
    with pytest.raises(ParityContractError):
        validate_parity_receipt(mutated, expected_plan=plan)


def test_rich_failure_requires_exact_authenticated_paired_plan() -> None:
    receipt = _valid_rich_clean_failed_receipt()
    with pytest.raises(ParityContractError) as missing_info:
        validate_parity_receipt(receipt)
    assert missing_info.value.code == "qwen.parity.receipt_plan_required"

    drifted_body = _plan_body()
    drifted_body["repo_identity"]["head"] = "different-repo-state"
    drifted_plan = finalize_plan(drifted_body)
    with pytest.raises(ParityContractError) as drift_info:
        validate_parity_receipt(receipt, expected_plan=drifted_plan)
    assert drift_info.value.code == "qwen.parity.receipt_plan_binding"


def test_preplan_rich_failure_remains_standalone_valid() -> None:
    probe = _load_probe_module()
    evidence = probe._FailureEvidenceAccumulator(requested_device="cuda:0")
    evidence.record("execution", {"requested_device": "cuda:0"})
    receipt = probe._failure_receipt(
        None,
        device_text="cuda:0",
        exc=ParityContractError(
            "injected pre-plan failure",
            code="qwen.parity.injected_preplan",
            context={},
        ),
        evidence=evidence,
    )
    assert validate_parity_receipt(receipt) == receipt


@pytest.mark.parametrize(
    ("comparison", "message"),
    [
        ("denominators", "status is not passed"),
        ("cross_arm_bf16_loss_term_scalars", "top-level status is inconsistent"),
    ],
)
def test_receipt_validation_requires_all_loss_normalization_gates(
    comparison: str,
    message: str,
) -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    if comparison == "denominators":
        receipt["comparisons"][comparison]["passed"] = False
    else:
        receipt["comparisons"]["packed_primary_vs_separate"][comparison]["passed"] = (
            False
        )
    with pytest.raises(ParityContractError, match=message):
        validate_parity_receipt(receipt, expected_plan=plan)


def test_receipt_validation_rejects_model_identity_attestation_mutations() -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)

    attestation_mutation = json.loads(json.dumps(receipt))
    attestation_mutation["model_identity_attestation"]["stable_projection_sha256"] = (
        "f" * 64
    )
    with pytest.raises(ParityContractError, match="attestation is inconsistent"):
        validate_parity_receipt(
            attestation_mutation,
            expected_plan=plan,
        )

    source_mutation = json.loads(json.dumps(receipt))
    source_mutation["source_identity"]["model_identity"]["tokenizer_sha256"] = "e" * 64
    with pytest.raises(ParityContractError, match="prepared plan"):
        validate_parity_receipt(source_mutation, expected_plan=plan)

    plan_mutation = _component_identity(load_model=False)
    plan_mutation["base_config_sha256"] = "e" * 64
    with pytest.raises(ParityContractError, match="contradicts the plan"):
        validate_parity_receipt(
            receipt,
            expected_plan=plan,
            expected_plan_model_identity=plan_mutation,
        )


@pytest.mark.parametrize(
    "mutation",
    ["config_identity", "repo_identity", "truncated_source_owners"],
)
def test_passed_receipt_rejects_fabricated_plan_owned_source(
    mutation: str,
) -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    if mutation == "config_identity":
        receipt["source_identity"]["config_identity"]["fingerprint"] = "forged"
    elif mutation == "repo_identity":
        receipt["source_identity"]["repo_identity"]["head"] = "forged"
    else:
        receipt["source_identity"]["source_owners"].pop()
    with pytest.raises(ParityContractError) as exc_info:
        validate_parity_receipt(receipt, expected_plan=plan)
    assert exc_info.value.code == "qwen.parity.receipt_source_binding"


def test_passed_receipt_requires_exact_authenticated_plan() -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)
    with pytest.raises(ParityContractError) as missing_info:
        validate_parity_receipt(receipt)
    assert missing_info.value.code == "qwen.parity.receipt_plan_required"

    drifted_body = _plan_body()
    drifted_body["repo_identity"]["head"] = "different-repo-state"
    drifted_plan = finalize_plan(drifted_body)
    with pytest.raises(ParityContractError) as drift_info:
        validate_parity_receipt(receipt, expected_plan=drifted_plan)
    assert drift_info.value.code == "qwen.parity.receipt_plan_binding"


def test_receipt_validation_rejects_cadence_repeat_and_tolerance_drift() -> None:
    plan = finalize_plan(_plan_body())
    receipt = _passed_receipt(plan)

    cadence = json.loads(json.dumps(receipt))
    cadence["arms"]["separate_reference"]["backward_cadence"][
        "harness_backward_call_count"
    ] = 1
    with pytest.raises(ParityContractError, match="cadence"):
        validate_parity_receipt(cadence, expected_plan=plan)

    retained_graph = json.loads(json.dumps(receipt))
    retained_graph["arms"]["separate_reference"]["backward_cadence"]["events"][0][
        "immediate_after_loss"
    ] = False
    with pytest.raises(ParityContractError, match="event order"):
        validate_parity_receipt(retained_graph, expected_plan=plan)

    repeat = json.loads(json.dumps(receipt))
    repeat["comparisons"]["packed_repeat_measurability"]["max_abs_diff"] = 0.003
    with pytest.raises(ParityContractError, match="repeat"):
        validate_parity_receipt(repeat, expected_plan=plan)

    storage_selected = json.loads(json.dumps(receipt))
    storage_selected["comparisons"]["packed_primary_vs_separate"]["gradients"][
        "parameters"
    ][0]["tolerance_selected_by"] = "parameter_storage_dtype"
    with pytest.raises(ParityContractError, match="fields"):
        validate_parity_receipt(storage_selected, expected_plan=plan)


def test_semantic_key_is_exactly_the_declared_identity() -> None:
    key = SemanticAtomKey.from_atom(_atom("a", 7, token_id=9))
    assert key.to_artifact_dict() == {
        "example_id": "a",
        "logical_target_position": 7,
        "logical_target_end": 8,
        "token_id": 9,
        "token_type": "coordinate",
        "object_id": "object-1",
        "field": "bbox",
        "source": "target",
    }


def test_indexed_and_standalone_weight_identities_detect_content_drift(
    tmp_path: Path,
) -> None:
    indexed = tmp_path / "indexed"
    indexed.mkdir()
    (indexed / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (indexed / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 11},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    before = base_model_weight_identity(indexed)
    assert before["mode"] == "indexed_safetensors"
    assert validate_model_weight_identity(before) == before

    (indexed / "model-00002-of-00002.safetensors").write_bytes(b"drifted")
    after_shard_drift = base_model_weight_identity(indexed)
    with pytest.raises(ParityContractError, match="changed after plan preparation"):
        assert_model_weight_identity_equal(before, after_shard_drift)

    (indexed / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 11, "revision": 2},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    after_index_drift = base_model_weight_identity(indexed)
    with pytest.raises(ParityContractError, match="changed after plan preparation"):
        assert_model_weight_identity_equal(before, after_index_drift)

    standalone = tmp_path / "standalone"
    standalone.mkdir()
    (standalone / "model.safetensors").write_bytes(b"standalone")
    standalone_identity = base_model_weight_identity(standalone)
    assert standalone_identity["mode"] == "standalone_safetensors"
    assert validate_model_weight_identity(standalone_identity) == standalone_identity


def test_indexed_weight_hashing_overlaps_and_matches_serial_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = tmp_path / "indexed"
    indexed.mkdir()
    for index, payload in ((1, b"first"), (2, b"second")):
        (indexed / f"model-{index:05d}-of-00002.safetensors").write_bytes(payload)
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    serial = base_model_weight_identity(indexed, max_workers=1)

    original = identity_module._stable_file_identity
    overlap = threading.Barrier(2)
    active_lock = threading.Lock()
    active = 0
    peak_active = 0

    def _overlapping_identity(path: Path, **kwargs):
        nonlocal active, peak_active
        if path.suffix == ".safetensors":
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                overlap.wait(timeout=5.0)
                return original(path, **kwargs)
            finally:
                with active_lock:
                    active -= 1
        return original(path, **kwargs)

    monkeypatch.setattr(identity_module, "_stable_file_identity", _overlapping_identity)
    parallel, policy = base_model_weight_identity_with_execution_policy(
        indexed,
        max_workers=2,
    )

    assert parallel == serial
    assert peak_active == 2
    assert policy == {
        "schema": MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA,
        "strategy": "thread_pool_file_sha256",
        "resolved_workers": 2,
        "payload_file_count": 2,
    }


def test_parallel_weight_hashing_reports_lexicographically_first_shard_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = tmp_path / "indexed"
    indexed.mkdir()
    for index in (1, 2):
        (indexed / f"model-{index:05d}-of-00002.safetensors").write_bytes(b"payload")
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "z": "model-00002-of-00002.safetensors",
                    "a": "model-00001-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    original = identity_module._stable_file_identity
    overlap = threading.Barrier(2)

    def _failing_identity(path: Path, **kwargs):
        if path.suffix != ".safetensors":
            return original(path, **kwargs)
        overlap.wait(timeout=5.0)
        code = "test.weight.first" if "00001" in path.name else "test.weight.second"
        raise ParityContractError(
            f"failed {path.name}",
            code=code,
            context={"path": path.name},
        )

    monkeypatch.setattr(identity_module, "_stable_file_identity", _failing_identity)

    with pytest.raises(ParityContractError) as error:
        base_model_weight_identity(indexed, max_workers=2)
    assert error.value.code == "test.weight.first"


@pytest.mark.parametrize("max_workers", [True, False, 0, -1, 1.5])
def test_weight_hashing_rejects_invalid_worker_counts(
    tmp_path: Path,
    max_workers: object,
) -> None:
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    (standalone / "model.safetensors").write_bytes(b"standalone")

    with pytest.raises(ParityContractError) as error:
        base_model_weight_identity(standalone, max_workers=max_workers)  # type: ignore[arg-type]
    assert error.value.code == "qwen.parity.weight_hash_workers"


def test_weight_hash_policy_caps_workers_and_uses_one_for_standalone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = tmp_path / "indexed"
    indexed.mkdir()
    for index in (1, 2):
        (indexed / f"model-{index:05d}-of-00002.safetensors").write_bytes(b"payload")
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    _, capped = base_model_weight_identity_with_execution_policy(
        indexed,
        max_workers=99,
    )
    assert capped["resolved_workers"] == 2

    monkeypatch.setattr(identity_module.os, "cpu_count", lambda: 1)
    _, cpu_default = base_model_weight_identity_with_execution_policy(indexed)
    assert cpu_default["resolved_workers"] == 1

    standalone = tmp_path / "standalone"
    standalone.mkdir()
    (standalone / "model.safetensors").write_bytes(b"standalone")
    _, standalone_policy = base_model_weight_identity_with_execution_policy(
        standalone,
        max_workers=99,
    )
    assert standalone_policy == {
        "schema": MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA,
        "strategy": "thread_pool_file_sha256",
        "resolved_workers": 1,
        "payload_file_count": 1,
    }


def test_weight_hash_executor_failure_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = tmp_path / "indexed"
    indexed.mkdir()
    for index in (1, 2):
        (indexed / f"model-{index:05d}-of-00002.safetensors").write_bytes(b"payload")
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    class _BrokenExecutor:
        def __init__(self, *, max_workers: int) -> None:
            del max_workers
            raise RuntimeError("executor unavailable")

    monkeypatch.setattr(identity_module, "ThreadPoolExecutor", _BrokenExecutor)
    with pytest.raises(ParityContractError) as error:
        base_model_weight_identity(indexed, max_workers=2)
    assert error.value.code == "qwen.parity.weight_hash_executor"


def test_executor_submit_failure_does_not_mask_prior_shard_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = tmp_path / "indexed"
    indexed.mkdir()
    for index in (1, 2):
        (indexed / f"model-{index:05d}-of-00002.safetensors").write_bytes(b"payload")
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    shard_error = ParityContractError(
        "first shard failed",
        code="test.weight.first",
        context={"path": "model-00001-of-00002.safetensors"},
    )

    class _SubmitFailureExecutor:
        def __init__(self, *, max_workers: int) -> None:
            del max_workers
            self.calls = 0

        def submit(self, fn, *args, **kwargs):
            del fn, args, kwargs
            self.calls += 1
            if self.calls == 1:
                future: Future[object] = Future()
                future.set_exception(shard_error)
                return future
            raise RuntimeError("submit failed")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is True
            assert cancel_futures is True

    monkeypatch.setattr(
        identity_module,
        "ThreadPoolExecutor",
        _SubmitFailureExecutor,
    )
    with pytest.raises(ParityContractError) as error:
        base_model_weight_identity(indexed, max_workers=2)
    assert error.value.code == "test.weight.first"


def test_weight_identity_rejects_index_escape_and_unindexed_multi_shard(
    tmp_path: Path,
) -> None:
    indexed = tmp_path / "escape"
    indexed.mkdir()
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ParityContractError, match="safe relative"):
        base_model_weight_identity(indexed)

    unindexed = tmp_path / "unindexed"
    unindexed.mkdir()
    (unindexed / "model.safetensors").write_bytes(b"one")
    (unindexed / "extra.safetensors").write_bytes(b"two")
    with pytest.raises(ParityContractError, match="one standalone"):
        base_model_weight_identity(unindexed)


def test_dependency_identity_requires_available_flash_binary_and_detects_drift() -> (
    None
):
    expected = _dependency_identity()
    assert validate_dependency_provenance(expected) == expected

    unavailable = _dependency_identity(binary_available=False)
    with pytest.raises(ParityContractError, match="binary provenance is unavailable"):
        validate_dependency_provenance(unavailable)

    drifted = json.loads(json.dumps(expected))
    drifted["accelerate_runtime_sources"][0]["sha256"] = "f" * 64
    with pytest.raises(ParityContractError, match="changed after plan preparation"):
        assert_dependency_provenance_equal(expected, drifted)

    missing_source = json.loads(json.dumps(expected))
    missing_source["accelerate_runtime_sources"].pop()
    with pytest.raises(ParityContractError, match="incomplete or unordered"):
        validate_dependency_provenance(missing_source)

    unknown_component = json.loads(json.dumps(expected))
    unknown_component["collected"]["unexpected-runtime"] = {}
    with pytest.raises(ParityContractError) as unknown_error:
        validate_dependency_provenance(unknown_component)
    assert unknown_error.value.code == "qwen.parity.dependency_inventory"

    mismatched_cuda_runtime = json.loads(json.dumps(expected))
    mismatched_cuda_runtime["collected"]["cuda-runtime"][
        "loaded_origin_matches_distribution"
    ] = False
    with pytest.raises(ParityContractError) as cuda_error:
        validate_dependency_provenance(mismatched_cuda_runtime)
    assert cuda_error.value.code == "qwen.parity.cuda_runtime_provenance"

    cuda_runtime_residue = json.loads(json.dumps(expected))
    cuda_runtime_residue["collected"]["cuda-runtime"]["unexpected"] = True
    with pytest.raises(ParityContractError, match="fields differ"):
        validate_dependency_provenance(cuda_runtime_residue)

    missing_cuda_runtime = json.loads(json.dumps(expected))
    missing_cuda_runtime["collected"].pop("cuda-runtime")
    with pytest.raises(ParityContractError) as missing_cuda_error:
        validate_dependency_provenance(missing_cuda_runtime)
    assert missing_cuda_error.value.code == "qwen.parity.dependency_inventory"


def test_exact_frozen_parent_v2_plan_file_allows_legacy_dependency_inventory() -> None:
    plan_path = (
        Path(__file__).resolve().parents[2]
        / "openspec/changes/archive/2026-08-12-harden-optimize-coordexp-swift-training-infrastructure/"
        "receipts/wave2-v2-parent-plan.json"
    )

    validated = parity_module.validate_parent_v2_plan_file(plan_path)

    assert validated["plan_sha256"] == parity_module.FROZEN_PARENT_V2_PLAN_SHA256
    assert "cuda-runtime" not in validated["dependency_identity"]["collected"]


def test_measurement_contract_rejects_peak_summary_drift_and_ceiling() -> None:
    measurement = _measurement()
    assert validate_measurement_contract(measurement) == measurement

    summary_drift = json.loads(json.dumps(measurement))
    summary_drift["resources"]["gpu"]["torch_peak_reserved_bytes"] = 4095
    with pytest.raises(ParityContractError, match="summary disagrees"):
        validate_measurement_contract(summary_drift)

    transient_ceiling = json.loads(json.dumps(measurement))
    ceiling = transient_ceiling["resources"]["ceilings"]["device_bytes"]
    transient_ceiling["resources"]["gpu"]["torch_peak_reserved_bytes"] = ceiling
    for sample in transient_ceiling["resources"]["phase_boundary_samples"]:
        sample["torch_cuda"]["max_reserved_bytes"] = ceiling
    transient_ceiling["resources"]["ceilings"]["comparison"]["torch_reserved_below"] = (
        False
    )
    with pytest.raises(ParityContractError, match="ceiling status"):
        validate_measurement_contract(transient_ceiling)


def test_measurement_contract_requires_final_idle_check_and_bounded_sampler() -> None:
    missing_final = _measurement()
    missing_final["gpu_idle_preflight"]["checks"].pop()
    with pytest.raises(ParityContractError, match="preflight is incomplete"):
        validate_measurement_contract(missing_final)

    overrun = _measurement()
    sampler = overrun["resources"]["gpu"]["device_sampler"]
    sampler["sample_count"] = sampler["maximum_samples"] + 1
    with pytest.raises(ParityContractError, match="sample bound"):
        validate_measurement_contract(overrun)


def test_runtime_image_processor_is_attached_without_semantic_drift(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    image_plan = QwenNoResizeImagePlan(
        example_id="example",
        image_path=tmp_path / "image.jpg",
        width=28,
        height=28,
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        required_spatial_factor=28,
        raw_pixels=28 * 28,
        raw_patch_rows=4,
        expected_pixel_values_width=14 * 14 * 3 * 2,
        image_grid_thw=(1, 2, 2),
        merged_visual_tokens=1,
        max_raw_pixels=28 * 28,
        max_merged_visual_tokens=1,
    )
    encoded = _EncodedWithImage(
        example_id="example",
        image_encoding=QwenImageEncoding(
            plan=image_plan,
            pixel_values=None,
            image_grid_thw_tensor=None,
        ),
    )
    processor = object()
    before = encoded.to_artifact_dict()
    (attached,) = probe._attach_runtime_image_processor(
        (encoded,), image_processor=processor
    )
    assert attached.image_encoding.image_processor is processor
    assert attached.to_artifact_dict() == before
    with pytest.raises(ParityContractError, match="runtime processor"):
        probe._attach_runtime_image_processor((encoded,), image_processor=None)


def test_exact_accelerator_validator_requires_one_rank_bf16_native_amp(
    monkeypatch,
) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cuda:2")
    accelerator = SimpleNamespace(
        distributed_type=SimpleNamespace(name="NO"),
        process_index=0,
        local_process_index=0,
        num_processes=1,
        device=torch.device("cuda:2"),
        mixed_precision="bf16",
        native_amp=True,
        gradient_accumulation_steps=1,
        scaler=None,
    )
    identity = probe._validate_exact_accelerator_runtime(
        accelerator,
        expected_device=torch.device("cuda:2"),
        expected_mixed_precision="bf16",
    )
    assert identity["device"] == "cuda:2"
    assert identity["native_amp"] is True

    accelerator.native_amp = False
    with pytest.raises(ParityContractError, match="frozen one-rank BF16 seam"):
        probe._validate_exact_accelerator_runtime(
            accelerator,
            expected_device=torch.device("cuda:2"),
            expected_mixed_precision="bf16",
        )


def test_prepared_model_attestation_accepts_bound_method_wrapper() -> None:
    from accelerate.utils.operations import convert_outputs_to_fp32

    probe = _load_probe_module()

    class BoundForwardModule(torch.nn.Module):
        def forward(self, value=None):
            return value

    model = BoundForwardModule()
    model._original_forward = model.forward
    original_forward = model.forward.__func__

    @wraps(original_forward)
    def autocast_forward(self, value=None):
        return original_forward(self, value)

    model.forward = MethodType(
        convert_outputs_to_fp32(autocast_forward),
        model,
    )
    artifact = probe._assert_prepared_accelerator_model(
        SimpleNamespace(_models=[model]),
        model=model,
        expected_device=torch.device("cpu"),
    )
    assert artifact == {
        "binding_branch": "bound_method",
        "prepared_forward_type": "method",
        "wrapper_owner_type": "function",
        "output_wrapper_type": "ConvertOutputsToFp32",
        "wrapper_identity_chain_verified": True,
        "unwrapped_original_identity_verified": True,
    }


def test_prepared_model_attestation_accepts_direct_callable_wrapper() -> None:
    from accelerate.utils.operations import convert_outputs_to_fp32

    probe = _load_probe_module()

    class DirectForwardModule(torch.nn.Module):
        def forward(self, value=None):
            return value

    model = DirectForwardModule()

    def original_forward(value=None):
        return value

    model._original_forward = original_forward

    @wraps(original_forward)
    def autocast_forward(value=None):
        return original_forward(value)

    model.forward = convert_outputs_to_fp32(autocast_forward)
    artifact = probe._assert_prepared_accelerator_model(
        SimpleNamespace(_models=[model]),
        model=model,
        expected_device=torch.device("cpu"),
    )
    assert artifact == {
        "binding_branch": "direct_callable",
        "prepared_forward_type": "function",
        "wrapper_owner_type": "function",
        "output_wrapper_type": "ConvertOutputsToFp32",
        "wrapper_identity_chain_verified": True,
        "unwrapped_original_identity_verified": True,
    }


def test_prepared_model_attestation_rejects_missing_output_wrapper() -> None:
    probe = _load_probe_module()

    class PlainModule(torch.nn.Module):
        def forward(self, value=None):
            return value

    model = PlainModule()
    model._original_forward = model.forward
    with pytest.raises(ParityContractError, match="ConvertOutputsToFp32"):
        probe._assert_prepared_accelerator_model(
            SimpleNamespace(_models=[model]),
            model=model,
            expected_device=torch.device("cpu"),
        )


def test_resource_gate_rejects_cumulative_reserved_peak_before_next_arm(
    monkeypatch,
) -> None:
    probe = _load_probe_module()
    device = torch.device("cuda:0")
    monkeypatch.setattr(probe, "_sync", lambda _device: None)
    monkeypatch.setattr(
        probe,
        "_host_resource_snapshot",
        lambda: {"rss_hwm_bytes": 1, "io_read_bytes": 0, "io_write_bytes": 0},
    )
    monkeypatch.setattr(
        probe,
        "_nvidia_smi_sample",
        lambda _device, sample_index: {
            **_device_sample(sample_index),
            "physical_index": 0,
        },
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 1)
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda _device: probe.DEVICE_MEMORY_CEILING_BYTES,
    )
    sampler = SimpleNamespace(
        raise_if_failed=lambda: None,
        artifact=lambda: {"hwm_memory_used_bytes": 1},
    )
    with pytest.raises(ParityContractError, match="device-memory ceiling"):
        probe._enforce_resource_ceilings(
            device,
            phase="packed_clean",
            samples=[],
            device_sampler=sampler,
        )


def test_run_refuses_existing_receipt_before_plan_or_model_work(
    tmp_path: Path,
    monkeypatch,
) -> None:
    probe = _load_probe_module()
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"sentinel":true}\n', encoding="utf-8")
    original = receipt.read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan/model work must not begin")

    monkeypatch.setattr(probe, "load_strict_json", forbidden)
    monkeypatch.setattr(probe, "build_plan", forbidden)
    result = probe.run_command(
        SimpleNamespace(plan=str(plan), receipt=str(receipt), device="cpu")
    )
    assert result == 1
    assert receipt.read_bytes() == original


def _patch_run_to_fail_after_plan(monkeypatch, probe, plan) -> None:
    monkeypatch.setattr(probe, "load_strict_json", lambda _path: plan)
    monkeypatch.setattr(probe, "validate_parity_plan", lambda value: value)
    monkeypatch.setattr(probe, "build_plan", lambda **_kwargs: (plan, object()))
    monkeypatch.setattr(probe, "assert_plan_revalidated", lambda *_args: None)

    def fail_probe(
        _plan,
        _materials,
        *,
        device_text,
        failure_evidence,
    ):
        del _plan, _materials, device_text
        failure_evidence.reach("plan_revalidated")
        raise ParityContractError(
            "injected primary failure",
            code="qwen.parity.injected_primary",
            context={},
        )

    monkeypatch.setattr(probe, "_execute_real_probe", fail_probe)


def test_run_publishes_reloadable_rich_post_comparison_clean_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    template = _valid_rich_clean_failed_receipt(gradient_count=589)
    monkeypatch.setattr(probe, "load_strict_json", lambda _path: plan)
    monkeypatch.setattr(probe, "validate_parity_plan", lambda value: value)
    monkeypatch.setattr(probe, "build_plan", lambda **_kwargs: (plan, object()))
    monkeypatch.setattr(probe, "assert_plan_revalidated", lambda *_args: None)

    def fail_after_comparisons(
        _plan,
        _materials,
        *,
        device_text,
        receipt_path,
        attempt_marker_path,
        command_identity,
        failure_evidence,
    ):
        del _plan, _materials, receipt_path, attempt_marker_path, command_identity
        assert device_text == "cuda:0"
        for phase in template["failure"]["evidence"]["completed_phases"]:
            failure_evidence.complete_phase(phase)
        for field in (
            "source_identity",
            "model_identity_attestation",
            "execution",
            "attempt_marker",
            "trainable_inventory",
            "arms",
            "proof",
            "timings",
            "measurement",
        ):
            failure_evidence.record(field, template[field])
        failure_evidence.record("comparisons", template["comparisons"])
        failure_evidence.record(
            "negative_discriminator", template["negative_discriminator"]
        )
        failure_evidence.reach("comparisons")
        raise ParityContractError(
            "clean packed-versus-separate parity failed",
            code="qwen.parity.clean_failed",
            context={},
        )

    monkeypatch.setattr(probe, "_execute_real_probe", fail_after_comparisons)
    receipt_path = tmp_path / "receipt.json"
    attempt_marker_path = tmp_path / "attempt.json"
    result = probe.run_command(
        SimpleNamespace(
            plan="ignored.json",
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(attempt_marker_path),
            device="cuda:0",
        )
    )
    assert result == 1
    status = json.loads(capsys.readouterr().err)
    assert status["receipt_persisted"] is True
    persisted = load_strict_json(receipt_path)
    assert validate_parity_receipt(persisted, expected_plan=plan) == persisted
    assert persisted["failure"].get("code") == "qwen.parity.clean_failed", persisted[
        "failure"
    ]
    assert persisted["failure"]["evidence"] == template["failure"]["evidence"]
    assert (
        len(
            persisted["comparisons"]["packed_primary_vs_separate"]["gradients"][
                "parameters"
            ]
        )
        == 589
    )
    assert len(persisted["arms"]["packed_primary"]["gradient_inventory"]) == 64
    assert persisted["arms"]["packed_primary"]["gradient_inventory_count"] == 589
    for field in template["failure"]["evidence"]["completed_fields"]:
        assert persisted[field] == template[field]
    assert receipt_path.stat().st_size < 8 * 1024 * 1024


def test_post_marker_exception_is_persisted_as_attempt_started_failure(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    concrete = _synthetic_concrete_inventory()
    loaded_source_identity = {
        "config_identity": json.loads(json.dumps(plan["config_identity"])),
        "repo_identity": json.loads(json.dumps(plan["repo_identity"])),
        "dependency_identity": json.loads(json.dumps(plan["dependency_identity"])),
        "model_identity": _component_identity(load_model=True),
        "model_weight_identity": json.loads(json.dumps(plan["model_weight_identity"])),
        "source_owners": json.loads(json.dumps(plan["source_owners"])),
    }
    evidence = probe._FailureEvidenceAccumulator(requested_device="cuda:0")
    evidence.record(
        "execution",
        {
            "requested_device": "cuda:0",
            "device": "cuda:0",
            "gpu_idle_preflight": {
                "status": "passed",
                "requested_device": "cuda:0",
            },
        },
    )
    marker_path = tmp_path / "attempt.json"
    receipt_path = tmp_path / "receipt.json"
    injected = RuntimeError("injected after persisted marker")

    with pytest.raises(RuntimeError, match="after persisted marker"):
        probe._publish_attempt_start_marker(
            marker_path,
            plan=plan,
            receipt_target=receipt_path,
            command_identity={"schema": "test-command-v1"},
            source_identity=loaded_source_identity,
            concrete_inventory=concrete,
            failure_evidence=evidence,
            inject_after=lambda: (_ for _ in ()).throw(injected),
        )

    failure_receipt = probe._failure_receipt(
        plan,
        device_text="cuda:0",
        exc=injected,
        evidence=evidence,
    )
    assert marker_path.exists()
    assert failure_receipt["attempt_marker"]["path"] == str(marker_path)
    assert failure_receipt["trainable_inventory"] == {"concrete": concrete}
    assert failure_receipt["source_identity"] == loaded_source_identity
    assert failure_receipt["failure"]["evidence"]["stage_reached"] == (
        "attempt_started"
    )
    assert validate_parity_receipt(failure_receipt, expected_plan=plan) == (
        failure_receipt
    )


@pytest.mark.parametrize("failure_kind", ["reload", "persisted_validation"])
def test_post_marker_recheck_failure_retains_reloadable_terminal_evidence(
    tmp_path: Path,
    monkeypatch,
    failure_kind: str,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    concrete = _synthetic_concrete_inventory()
    loaded_source_identity = {
        "config_identity": json.loads(json.dumps(plan["config_identity"])),
        "repo_identity": json.loads(json.dumps(plan["repo_identity"])),
        "dependency_identity": json.loads(json.dumps(plan["dependency_identity"])),
        "model_identity": _component_identity(load_model=True),
        "model_weight_identity": json.loads(json.dumps(plan["model_weight_identity"])),
        "source_owners": json.loads(json.dumps(plan["source_owners"])),
    }
    evidence = probe._FailureEvidenceAccumulator(requested_device="cuda:0")
    evidence.record(
        "execution",
        {
            "requested_device": "cuda:0",
            "device": "cuda:0",
            "gpu_idle_preflight": {
                "status": "passed",
                "requested_device": "cuda:0",
            },
        },
    )
    marker_path = tmp_path / "attempt.json"
    receipt_path = tmp_path / "receipt.json"
    original_loader = probe.load_strict_json
    original_validator = probe.validate_attempt_marker
    if failure_kind == "reload":
        injected = RuntimeError("injected persisted marker reload failure")

        def fail_reload(_path):
            raise injected

        monkeypatch.setattr(probe, "load_strict_json", fail_reload)
    else:
        injected = ParityContractError(
            "injected persisted marker validation failure",
            code="qwen.parity.injected_marker_validation",
            context={},
        )
        validation_calls = 0

        def fail_persisted_validation(marker, **kwargs):
            nonlocal validation_calls
            validation_calls += 1
            if validation_calls == 2:
                raise injected
            return original_validator(marker, **kwargs)

        monkeypatch.setattr(
            probe,
            "validate_attempt_marker",
            fail_persisted_validation,
        )

    kwargs = {
        "plan": plan,
        "receipt_target": receipt_path,
        "command_identity": {"schema": "test-command-v1"},
        "source_identity": loaded_source_identity,
        "concrete_inventory": concrete,
        "failure_evidence": evidence,
    }
    with pytest.raises(type(injected), match="persisted marker"):
        probe._publish_attempt_start_marker(marker_path, **kwargs)

    monkeypatch.setattr(probe, "load_strict_json", original_loader)
    monkeypatch.setattr(probe, "validate_attempt_marker", original_validator)
    failure_receipt = probe._failure_receipt(
        plan,
        device_text="cuda:0",
        exc=injected,
        evidence=evidence,
    )
    assert marker_path.exists()
    assert failure_receipt["attempt_marker"]["path"] == str(marker_path)
    assert failure_receipt["trainable_inventory"] == {"concrete": concrete}
    assert failure_receipt["source_identity"] == loaded_source_identity
    assert failure_receipt["failure"]["evidence"]["stage_reached"] == (
        "attempt_started"
    )
    assert validate_parity_receipt(failure_receipt, expected_plan=plan) == (
        failure_receipt
    )
    write_strict_json_atomic(receipt_path, failure_receipt)
    reloaded_receipt = load_strict_json(receipt_path)
    assert validate_parity_receipt(reloaded_receipt, expected_plan=plan) == (
        reloaded_receipt
    )
    with pytest.raises(ParityContractError) as caught:
        probe._publish_attempt_start_marker(marker_path, **kwargs)
    assert caught.value.code == "qwen.parity.artifact_collision"


@pytest.mark.parametrize("failure_kind", ["directory_fsync", "temporary_cleanup"])
def test_post_link_marker_writer_failure_recovers_exact_consumed_attempt(
    tmp_path: Path,
    monkeypatch,
    failure_kind: str,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    concrete = _synthetic_concrete_inventory()
    loaded_source_identity = {
        "config_identity": json.loads(json.dumps(plan["config_identity"])),
        "repo_identity": json.loads(json.dumps(plan["repo_identity"])),
        "dependency_identity": json.loads(json.dumps(plan["dependency_identity"])),
        "model_identity": _component_identity(load_model=True),
        "model_weight_identity": json.loads(json.dumps(plan["model_weight_identity"])),
        "source_owners": json.loads(json.dumps(plan["source_owners"])),
    }
    evidence = probe._FailureEvidenceAccumulator(requested_device="cuda:0")
    evidence.record(
        "execution",
        {
            "requested_device": "cuda:0",
            "device": "cuda:0",
            "gpu_idle_preflight": {
                "status": "passed",
                "requested_device": "cuda:0",
            },
        },
    )
    marker_path = tmp_path / "attempt.json"
    receipt_path = tmp_path / "receipt.json"
    if failure_kind == "directory_fsync":
        original_fsync = identity_module.os.fsync
        fsync_calls = 0

        def fail_directory_fsync(fd):
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("injected post-link directory fsync failure")
            return original_fsync(fd)

        monkeypatch.setattr(identity_module.os, "fsync", fail_directory_fsync)
        expected_message = "directory fsync"
    else:
        original_unlink = Path.unlink

        def fail_temporary_cleanup(path, *args, **kwargs):
            if (
                path.parent == tmp_path
                and path.name.startswith(".attempt.json.")
                and path.name.endswith(".tmp")
            ):
                raise OSError("injected post-link temporary cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_temporary_cleanup)
        expected_message = "temporary cleanup"

    kwargs = {
        "plan": plan,
        "receipt_target": receipt_path,
        "command_identity": {"schema": "test-command-v1"},
        "source_identity": loaded_source_identity,
        "concrete_inventory": concrete,
        "failure_evidence": evidence,
    }
    with pytest.raises(OSError, match=expected_message) as caught:
        probe._publish_attempt_start_marker(marker_path, **kwargs)

    if failure_kind == "directory_fsync":
        monkeypatch.setattr(identity_module.os, "fsync", original_fsync)
    else:
        monkeypatch.setattr(Path, "unlink", original_unlink)
    failure_receipt = probe._failure_receipt(
        plan,
        device_text="cuda:0",
        exc=caught.value,
        evidence=evidence,
    )
    assert marker_path.exists()
    assert failure_receipt["attempt_marker"]["path"] == str(marker_path)
    assert failure_receipt["trainable_inventory"] == {"concrete": concrete}
    assert failure_receipt["source_identity"] == loaded_source_identity
    assert failure_receipt["failure"]["evidence"]["stage_reached"] == (
        "attempt_started"
    )
    assert validate_parity_receipt(failure_receipt, expected_plan=plan) == (
        failure_receipt
    )
    write_strict_json_atomic(receipt_path, failure_receipt)
    reloaded_receipt = load_strict_json(receipt_path)
    assert validate_parity_receipt(reloaded_receipt, expected_plan=plan) == (
        reloaded_receipt
    )
    with pytest.raises(ParityContractError) as second:
        probe._publish_attempt_start_marker(marker_path, **kwargs)
    assert second.value.code == "qwen.parity.artifact_collision"


def _patch_run_to_return_passed_receipt(
    monkeypatch,
    probe,
    *,
    plan: dict[str, object],
    receipt: dict[str, object],
) -> None:
    monkeypatch.setattr(probe, "build_plan", lambda **_kwargs: (plan, object()))
    monkeypatch.setattr(probe, "assert_plan_revalidated", lambda *_args: None)

    def execute(
        _plan,
        _materials,
        *,
        receipt_path,
        attempt_marker_path,
        command_identity,
        failure_evidence,
        **_kwargs,
    ):
        del _plan, _materials, _kwargs
        receipt["attempt_marker"] = probe._publish_attempt_start_marker(
            attempt_marker_path,
            plan=plan,
            receipt_target=receipt_path,
            command_identity=command_identity,
            source_identity=receipt["source_identity"],
            concrete_inventory=receipt["trainable_inventory"]["concrete"],
            failure_evidence=failure_evidence,
        )
        return receipt

    monkeypatch.setattr(probe, "_execute_real_probe", execute)


@pytest.mark.parametrize("failure_kind", ["directory_fsync", "temporary_cleanup"])
def test_run_recovers_exact_passed_receipt_after_its_own_postlink_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
    failure_kind: str,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    passed = _passed_receipt(plan)
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_return_passed_receipt(
        monkeypatch,
        probe,
        plan=plan,
        receipt=passed,
    )
    receipt_links = 0
    original_link = identity_module.os.link

    def count_receipt_link(source, target) -> None:
        nonlocal receipt_links
        if Path(target) == receipt_path:
            receipt_links += 1
        original_link(source, target)

    monkeypatch.setattr(identity_module.os, "link", count_receipt_link)
    if failure_kind == "directory_fsync":
        original_fsync = identity_module.os.fsync
        fsync_calls = 0

        def fail_receipt_directory_fsync(fd) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 4:
                raise OSError("injected passed receipt directory fsync failure")
            original_fsync(fd)

        monkeypatch.setattr(identity_module.os, "fsync", fail_receipt_directory_fsync)
    else:
        original_unlink = Path.unlink

        def fail_receipt_temporary_cleanup(path, *args, **kwargs):
            if path.parent == tmp_path and path.name.startswith(".receipt.json."):
                raise OSError("injected passed receipt temporary cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_receipt_temporary_cleanup)

    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    captured = capsys.readouterr()
    status = json.loads(captured.out)
    assert result == 0
    assert status["terminal_status"] == "passed"
    assert status["receipt_persisted"] is True
    assert status["post_link_recovery"] is True
    assert status["publication_warning"]["type"] == "OSError"
    assert captured.err == ""
    assert receipt_links == 1
    assert not (tmp_path / "receipt.json.publication-failure.json").exists()
    persisted = load_strict_json(receipt_path)
    assert persisted == passed
    assert (
        validate_parity_receipt(
            persisted,
            expected_plan=plan,
            expected_receipt_target=receipt_path,
        )
        == persisted
    )


@pytest.mark.parametrize("collision_kind", ["identical", "foreign"])
def test_run_never_promotes_preexisting_receipt_collision(
    tmp_path: Path,
    monkeypatch,
    capsys,
    collision_kind: str,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    passed = _passed_receipt(plan)
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_return_passed_receipt(
        monkeypatch,
        probe,
        plan=plan,
        receipt=passed,
    )
    original_execute = probe._execute_real_probe

    def collide(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        existing = result if collision_kind == "identical" else {"foreign": True}
        write_strict_json_atomic(receipt_path, existing)
        return result

    monkeypatch.setattr(probe, "_execute_real_probe", collide)
    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    status = json.loads(capsys.readouterr().err)
    assert result == 1
    assert status["terminal_status"] == "failed"
    assert status["receipt_persisted"] is False
    assert "post_link_recovery" not in status
    assert status["publication_failure"]["sidecar_persisted"] is True


def test_run_never_promotes_invalid_postlink_receipt(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    passed = _passed_receipt(plan)
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_return_passed_receipt(
        monkeypatch,
        probe,
        plan=plan,
        receipt=passed,
    )
    original_fsync = identity_module.os.fsync
    fsync_calls = 0

    def corrupt_after_receipt_link(fd) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 4:
            receipt_path.write_text("{invalid\n", encoding="utf-8")
            raise OSError("injected invalid post-link receipt")
        original_fsync(fd)

    monkeypatch.setattr(identity_module.os, "fsync", corrupt_after_receipt_link)
    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    status = json.loads(capsys.readouterr().err)
    assert result == 1
    assert status["receipt_persisted"] is False
    assert "post_link_recovery" not in status
    assert status["publication_failure"]["sidecar_persisted"] is True


def _patch_run_to_raise_simple_failure(monkeypatch, probe, plan) -> None:
    monkeypatch.setattr(probe, "build_plan", lambda **_kwargs: (plan, object()))
    monkeypatch.setattr(probe, "assert_plan_revalidated", lambda *_args: None)

    def fail_probe(
        _plan,
        _materials,
        *,
        failure_evidence,
        **_kwargs,
    ):
        del _plan, _materials, _kwargs
        failure_evidence.reach("plan_revalidated")
        raise ParityContractError(
            "injected primary failure",
            code="qwen.parity.injected_primary",
            context={},
        )

    monkeypatch.setattr(probe, "_execute_real_probe", fail_probe)


def _patch_run_to_raise_unmeasurable_failure(
    monkeypatch, probe, plan, template
) -> None:
    monkeypatch.setattr(probe, "build_plan", lambda **_kwargs: (plan, object()))
    monkeypatch.setattr(probe, "assert_plan_revalidated", lambda *_args: None)

    def fail_probe(
        _plan,
        _materials,
        *,
        failure_evidence,
        **_kwargs,
    ):
        del _plan, _materials, _kwargs
        for phase in template["failure"]["evidence"]["completed_phases"]:
            failure_evidence.complete_phase(phase)
        for field in template["failure"]["evidence"]["completed_fields"]:
            failure_evidence.record(field, template[field])
        failure_evidence.reach("comparisons")
        raise ParityContractError(
            "packed repeat exceeded the fixed gate",
            code="qwen.parity.packed_repeat_unmeasurable",
            context={},
        )

    monkeypatch.setattr(probe, "_execute_real_probe", fail_probe)


@pytest.mark.parametrize("terminal_status", ["failed", "unmeasurable"])
@pytest.mark.parametrize("failure_kind", ["directory_fsync", "temporary_cleanup"])
def test_run_recovers_exact_failure_style_receipt_after_own_postlink_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
    terminal_status: str,
    failure_kind: str,
) -> None:
    probe = _load_probe_module()
    if terminal_status == "unmeasurable":
        plan, template = _v3_failure_at_stage("comparisons")
        primary = _synthetic_gradient_records()
        repeat = list(_synthetic_gradient_records())
        repeat[0] = replace(repeat[0], grad=repeat[0].grad + 0.01)
        template["comparisons"]["packed_repeat_measurability"] = (
            compare_packed_gradient_repeat(primary, tuple(repeat))
        )
        template["terminal_status"] = "unmeasurable"
        template["failure"]["code"] = "qwen.parity.packed_repeat_unmeasurable"
        template["failure"]["message"] = "packed repeat exceeded the fixed gate"
        _patch_run_to_raise_unmeasurable_failure(
            monkeypatch,
            probe,
            plan,
            template,
        )
    else:
        plan = finalize_plan(_plan_body())
        _patch_run_to_raise_simple_failure(monkeypatch, probe, plan)
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    write_strict_json_atomic(plan_path, plan)
    receipt_links = 0
    original_link = identity_module.os.link

    def count_receipt_link(source, target) -> None:
        nonlocal receipt_links
        if Path(target) == receipt_path:
            receipt_links += 1
        original_link(source, target)

    monkeypatch.setattr(identity_module.os, "link", count_receipt_link)
    if failure_kind == "directory_fsync":
        original_fsync = identity_module.os.fsync
        fsync_calls = 0

        def fail_receipt_directory_fsync(fd) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("injected failure receipt directory fsync failure")
            original_fsync(fd)

        monkeypatch.setattr(identity_module.os, "fsync", fail_receipt_directory_fsync)
    else:
        original_unlink = Path.unlink

        def fail_receipt_temporary_cleanup(path, *args, **kwargs):
            if path.parent == tmp_path and path.name.startswith(".receipt.json."):
                raise OSError("injected failure receipt temporary cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_receipt_temporary_cleanup)

    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    status = json.loads(capsys.readouterr().err)
    assert result == 1
    assert status["terminal_status"] == terminal_status
    assert status["receipt_persisted"] is True
    assert status["post_link_recovery"] is True
    assert status["publication_warning"]["type"] == "OSError"
    assert receipt_links == 1
    assert not (tmp_path / "receipt.json.publication-failure.json").exists()
    persisted = load_strict_json(receipt_path)
    assert persisted["terminal_status"] == terminal_status
    assert validate_parity_receipt(persisted, expected_plan=plan) == persisted


@pytest.mark.parametrize("collision_kind", ["identical", "foreign"])
def test_failure_receipt_collision_never_claims_postlink_recovery(
    tmp_path: Path,
    monkeypatch,
    capsys,
    collision_kind: str,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_raise_simple_failure(monkeypatch, probe, plan)
    original_writer = probe.write_strict_json_atomic

    def collide(path, payload, **kwargs):
        if Path(path) == receipt_path and not receipt_path.exists():
            existing = payload if collision_kind == "identical" else {"foreign": True}
            original_writer(path, existing)
        return original_writer(path, payload, **kwargs)

    monkeypatch.setattr(probe, "write_strict_json_atomic", collide)
    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    status = json.loads(capsys.readouterr().err)
    assert result == 1
    assert status["receipt_persisted"] is False
    assert "post_link_recovery" not in status
    assert status["publication_failure"]["sidecar_persisted"] is True


def test_failure_receipt_different_postlink_payload_is_not_recovered(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_raise_simple_failure(monkeypatch, probe, plan)
    original_fsync = identity_module.os.fsync
    fsync_calls = 0

    def replace_after_link(fd) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            receipt_path.write_text('{"foreign":true}\n', encoding="utf-8")
            raise OSError("injected different failure receipt")
        original_fsync(fd)

    monkeypatch.setattr(identity_module.os, "fsync", replace_after_link)
    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    status = json.loads(capsys.readouterr().err)
    assert result == 1
    assert status["receipt_persisted"] is False
    assert "post_link_recovery" not in status
    assert status["publication_failure"]["sidecar_persisted"] is True


class _PostSamplerMutationFailureDict(dict):
    def __setitem__(self, key, value):
        if key == "device_sampler":
            raise RuntimeError("injected post-sampler mutation failure")
        return super().__setitem__(key, value)


@pytest.mark.parametrize(
    "failure_kind",
    ["device_ceiling", "success_validation", "post_sampler_mutation"],
)
def test_run_post_sampler_failure_publishes_failure_style_terminal_receipt(
    tmp_path: Path,
    monkeypatch,
    capsys,
    failure_kind: str,
) -> None:
    probe = _load_probe_module()
    plan, finalization_template = _v3_failure_at_stage("finalization")
    success_receipt = _passed_receipt(plan)
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    write_strict_json_atomic(plan_path, plan)
    monkeypatch.setattr(probe, "build_plan", lambda **_kwargs: (plan, object()))
    monkeypatch.setattr(probe, "assert_plan_revalidated", lambda *_args: None)
    monkeypatch.setattr(
        probe,
        "_preflight_cuda",
        lambda device_text: (
            torch.device(device_text),
            {"status": "passed", "requested_device": device_text},
        ),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 11)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 13)

    class Sampler:
        def __init__(self, _device) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self, *, raise_error=True):
            del raise_error
            return {"status": "completed", "hwm_memory_used_bytes": 17}

    monkeypatch.setattr(probe, "_BoundedDeviceSampler", Sampler)

    def execute_with_sampler(
        _plan,
        _materials,
        *,
        receipt_path,
        attempt_marker_path,
        command_identity,
        failure_evidence,
        **_kwargs,
    ):
        del _plan, _materials, _kwargs
        marker_reference = probe._publish_attempt_start_marker(
            attempt_marker_path,
            plan=plan,
            receipt_target=receipt_path,
            command_identity=command_identity,
            source_identity=success_receipt["source_identity"],
            concrete_inventory=success_receipt["trainable_inventory"]["concrete"],
            failure_evidence=failure_evidence,
        )
        success_receipt["attempt_marker"] = marker_reference
        for phase in finalization_template["failure"]["evidence"]["completed_phases"]:
            failure_evidence.complete_phase(phase)
        for field in (
            "model_identity_attestation",
            "execution",
            "arms",
            "proof",
            "comparisons",
            "negative_discriminator",
            "timings",
            "measurement",
        ):
            failure_evidence.record(field, finalization_template[field])
        failure_evidence.reach("finalization")
        if failure_kind == "device_ceiling":
            success_receipt["measurement"]["resources"]["host"]["rss_hwm_bytes"] = (
                probe.HOST_MEMORY_CEILING_BYTES
            )
        elif failure_kind == "post_sampler_mutation":
            success_receipt["measurement"]["resources"]["gpu"] = (
                _PostSamplerMutationFailureDict(
                    success_receipt["measurement"]["resources"]["gpu"]
                )
            )
        return success_receipt

    monkeypatch.setattr(
        probe,
        "_execute_real_probe_with_sampler",
        execute_with_sampler,
    )
    original_receipt_validator = probe.validate_parity_receipt
    if failure_kind == "success_validation":

        def fail_success_validation(receipt, **kwargs):
            if receipt["terminal_status"] == "passed":
                raise ParityContractError(
                    "injected success receipt self-validation failure",
                    code="qwen.parity.injected_success_validation",
                    context={},
                )
            return original_receipt_validator(receipt, **kwargs)

        monkeypatch.setattr(
            probe,
            "validate_parity_receipt",
            fail_success_validation,
        )

    args = SimpleNamespace(
        plan=str(plan_path),
        parent_v2_plan="parent-v2.json",
        receipt=str(receipt_path),
        attempt_marker=str(marker_path),
        device="cuda:0",
    )
    assert probe.run_command(args) == 1
    status = json.loads(capsys.readouterr().err)
    assert status["receipt_persisted"] is True
    assert not (tmp_path / "receipt.json.publication-failure.json").exists()
    persisted = load_strict_json(receipt_path)
    assert validate_parity_receipt(persisted, expected_plan=plan) == persisted
    assert persisted["terminal_status"] == "failed"
    assert persisted["attempt_marker"]["path"] == str(marker_path)
    assert persisted["failure"]["evidence"]["stage_reached"] == "finalization"
    assert persisted["measurement"]["schema"] == (
        "coordexp-swift-wave2-failure-measurement-v1"
    )
    assert persisted["gpu_memory"]["scope"] == "whole_probe_until_failure"
    assert marker_path.exists()
    with pytest.raises(ParityContractError) as consumed:
        probe.assert_absent_artifact_target(marker_path)
    assert consumed.value.code == "qwen.parity.artifact_collision"


def test_primary_failure_receipt_validation_failure_publishes_sidecar(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    _patch_run_to_fail_after_plan(monkeypatch, probe, plan)

    def reject_receipt(_receipt, **_kwargs):
        raise ParityContractError(
            "injected receipt validation failure",
            code="qwen.parity.injected_receipt_validation",
            context={},
        )

    monkeypatch.setattr(probe, "validate_parity_receipt", reject_receipt)
    receipt = tmp_path / "receipt.json"
    result = probe.run_command(
        SimpleNamespace(plan="ignored.json", receipt=str(receipt), device="cuda:0")
    )
    assert result == 1
    assert not receipt.exists()
    sidecar = tmp_path / "receipt.json.publication-failure.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["terminal_status"] == "publication_failed"
    assert payload["receipt_persisted"] is False
    status = json.loads(capsys.readouterr().err)
    assert status["receipt_persisted"] is False
    assert status["publication_failure"]["sidecar_persisted"] is True


def test_atomic_failure_receipt_publication_failure_publishes_sidecar(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    _patch_run_to_fail_after_plan(monkeypatch, probe, plan)
    receipt = tmp_path / "receipt.json"
    original_writer = probe.write_strict_json_atomic

    def fail_primary_write(path, payload, **kwargs):
        if Path(path) == receipt:
            raise OSError("injected atomic publication failure")
        return original_writer(path, payload, **kwargs)

    monkeypatch.setattr(probe, "write_strict_json_atomic", fail_primary_write)
    result = probe.run_command(
        SimpleNamespace(plan="ignored.json", receipt=str(receipt), device="cuda:0")
    )
    assert result == 1
    assert not receipt.exists()
    sidecar = tmp_path / "receipt.json.publication-failure.json"
    assert sidecar.exists()
    status = json.loads(capsys.readouterr().err)
    assert status["receipt_persisted"] is False
    assert status["publication_failure"]["sidecar_persisted"] is True
    assert status["publication_failure"]["publication_failure"]["type"] == ("OSError")
    assert "post_link_recovery" not in status
    assert "sidecar_post_link_recovery" not in status["publication_failure"]


@pytest.mark.parametrize("failure_kind", ["directory_fsync", "temporary_cleanup"])
def test_publication_sidecar_recovers_exact_payload_after_own_postlink_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
    failure_kind: str,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    sidecar_path = tmp_path / "receipt.json.publication-failure.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_raise_simple_failure(monkeypatch, probe, plan)
    original_writer = probe.write_strict_json_atomic

    def fail_primary_receipt(path, payload, **kwargs):
        if Path(path) == receipt_path:
            raise OSError("injected failure receipt pre-link failure")
        return original_writer(path, payload, **kwargs)

    monkeypatch.setattr(probe, "write_strict_json_atomic", fail_primary_receipt)
    if failure_kind == "directory_fsync":
        original_fsync = identity_module.os.fsync
        fsync_calls = 0

        def fail_sidecar_directory_fsync(fd) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("injected sidecar directory fsync failure")
            original_fsync(fd)

        monkeypatch.setattr(identity_module.os, "fsync", fail_sidecar_directory_fsync)
    else:
        original_unlink = Path.unlink

        def fail_sidecar_temporary_cleanup(path, *args, **kwargs):
            if path.parent == tmp_path and path.name.startswith(
                ".receipt.json.publication-failure.json."
            ):
                raise OSError("injected sidecar temporary cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_sidecar_temporary_cleanup)

    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    status = json.loads(capsys.readouterr().err)
    publication = status["publication_failure"]
    assert result == 1
    assert status["receipt_persisted"] is False
    assert publication["sidecar_persisted"] is True
    assert publication["sidecar_post_link_recovery"] is True
    assert publication["sidecar_publication_warning"]["type"] == "OSError"
    assert sidecar_path.exists()
    persisted = load_strict_json(sidecar_path)
    assert persisted["terminal_status"] == "publication_failed"
    assert persisted["receipt_persisted"] is False
    assert "sidecar_post_link_recovery" not in persisted


def test_publication_sidecar_prelink_failure_is_not_recovered(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    sidecar_path = tmp_path / "receipt.json.publication-failure.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_raise_simple_failure(monkeypatch, probe, plan)

    def fail_all_publications(path, _payload, **_kwargs):
        target = Path(path)
        if target == receipt_path:
            raise OSError("injected failure receipt pre-link failure")
        if target == sidecar_path:
            raise OSError("injected sidecar pre-link failure")
        raise AssertionError(f"unexpected publication target: {target}")

    monkeypatch.setattr(probe, "write_strict_json_atomic", fail_all_publications)
    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    publication = json.loads(capsys.readouterr().err)["publication_failure"]
    assert result == 1
    assert publication["sidecar_persisted"] is False
    assert "sidecar_post_link_recovery" not in publication
    assert publication["sidecar_failure"]["type"] == "OSError"
    assert not sidecar_path.exists()


@pytest.mark.parametrize("collision_kind", ["identical", "foreign"])
def test_publication_sidecar_collision_never_claims_postlink_recovery(
    tmp_path: Path,
    monkeypatch,
    capsys,
    collision_kind: str,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    sidecar_path = tmp_path / "receipt.json.publication-failure.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_raise_simple_failure(monkeypatch, probe, plan)
    original_writer = probe.write_strict_json_atomic

    def collide_sidecar(path, payload, **kwargs):
        target = Path(path)
        if target == receipt_path:
            raise OSError("injected failure receipt pre-link failure")
        if target == sidecar_path and not sidecar_path.exists():
            existing = payload if collision_kind == "identical" else {"foreign": True}
            original_writer(target, existing)
        return original_writer(target, payload, **kwargs)

    monkeypatch.setattr(probe, "write_strict_json_atomic", collide_sidecar)
    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    publication = json.loads(capsys.readouterr().err)["publication_failure"]
    assert result == 1
    assert publication["sidecar_persisted"] is False
    assert "sidecar_post_link_recovery" not in publication
    assert publication["sidecar_failure"]["code"] == ("qwen.parity.artifact_collision")
    persisted = load_strict_json(sidecar_path)
    if collision_kind == "identical":
        assert persisted["terminal_status"] == "publication_failed"
    else:
        assert persisted == {"foreign": True}


def test_publication_sidecar_different_postlink_payload_is_not_recovered(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    probe = _load_probe_module()
    plan = finalize_plan(_plan_body())
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "attempt.json"
    sidecar_path = tmp_path / "receipt.json.publication-failure.json"
    write_strict_json_atomic(plan_path, plan)
    _patch_run_to_raise_simple_failure(monkeypatch, probe, plan)
    original_writer = probe.write_strict_json_atomic

    def fail_primary_receipt(path, payload, **kwargs):
        if Path(path) == receipt_path:
            raise OSError("injected failure receipt pre-link failure")
        return original_writer(path, payload, **kwargs)

    monkeypatch.setattr(probe, "write_strict_json_atomic", fail_primary_receipt)
    original_fsync = identity_module.os.fsync
    fsync_calls = 0

    def replace_sidecar_after_link(fd) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            sidecar_path.write_text('{"foreign":true}\n', encoding="utf-8")
            raise OSError("injected different sidecar payload")
        original_fsync(fd)

    monkeypatch.setattr(identity_module.os, "fsync", replace_sidecar_after_link)
    result = probe.run_command(
        SimpleNamespace(
            plan=str(plan_path),
            parent_v2_plan="parent-v2.json",
            receipt=str(receipt_path),
            attempt_marker=str(marker_path),
            device="cuda:0",
        )
    )
    publication = json.loads(capsys.readouterr().err)["publication_failure"]
    assert result == 1
    assert publication["sidecar_persisted"] is False
    assert "sidecar_post_link_recovery" not in publication
    assert publication["sidecar_failure"]["type"] == "OSError"
    assert load_strict_json(sidecar_path) == {"foreign": True}


def test_receipt_and_publication_sidecar_collision_never_overwrites(
    tmp_path: Path, capsys
) -> None:
    probe = _load_probe_module()
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    sidecar = tmp_path / "receipt.json.publication-failure.json"
    receipt.write_bytes(b"primary-sentinel\n")
    sidecar.write_bytes(b"sidecar-sentinel\n")

    result = probe.run_command(
        SimpleNamespace(plan=str(plan), receipt=str(receipt), device="cpu")
    )
    assert result == 1
    assert receipt.read_bytes() == b"primary-sentinel\n"
    assert sidecar.read_bytes() == b"sidecar-sentinel\n"
    status = json.loads(capsys.readouterr().err)
    assert status["receipt_persisted"] is False
    assert status["publication_failure"]["sidecar_persisted"] is False
    assert status["publication_failure"]["sidecar_failure"]["code"] == (
        "qwen.parity.artifact_collision"
    )
