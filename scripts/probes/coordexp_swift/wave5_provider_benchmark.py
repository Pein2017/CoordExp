#!/usr/bin/env python3
"""Fail-closed Wave 5 provider benchmark and receipt harness.

The controller accepts no caller-selected executable.  A prepared plan binds this
file, the current Python interpreter, ``src/train.py``, the typed W0-derived
configuration recipe, and immutable current-v3 cache manifests.  Each arm runs in
its own process group.  Only the controller derives observations, from the
deterministic production run directory after the authenticated arm process exits.
The retained idle policy is the only performance-promotional route.  The explicit
shared policy binds stable pre-existing driver processes and produces correctness,
plumbing, and failure-semantics evidence only.

The historical W0 cache is evidence only.  It is never passed to the current
loader.  A dated compatibility attestation must relate the historical W0 stream
receipt to canonical, loader-admissible ``coordexp-swift-pack-cache-v3`` targets.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


HISTORICAL_PLAN_SCHEMA_V3 = "coordexp-swift-wave5-provider-benchmark-plan-v3"
PLAN_SCHEMA = "coordexp-swift-wave5-provider-benchmark-plan-v4"
ATTEMPT_SCHEMA = "coordexp-swift-wave5-provider-benchmark-attempt-v4"
OBSERVATION_SCHEMA = "coordexp-swift-wave5-provider-observation-v4"
DECISION_SCHEMA = "coordexp-swift-wave5-provider-decision-v4"
TERMINAL_SCHEMA = "coordexp-swift-wave5-provider-benchmark-terminal-v4"
PUBLICATION_FAILURE_SCHEMA = "coordexp-swift-wave5-terminal-publication-failure-v1"
LEGACY_REFUTED_COMPATIBILITY_SCHEMA = "coordexp-swift-wave0-current-v3-compatibility-v1"
COMPATIBILITY_SCHEMA = "coordexp-swift-wave0-current-v3-compatibility-v2"
STREAM_COMPATIBILITY_PROJECTION_SCHEMA = (
    "coordexp-swift-wave5-v2-v3-stream-compatibility-projection-v1"
)
PACK_PLAN_PROVENANCE_SCHEMA = "coordexp-swift-wave5-current-v3-pack-plan-provenance-v1"
GPU_BASELINE_SCHEMA = "coordexp-swift-wave5-gpu-execution-baseline-v1"
RESOURCE_SCHEMA = "coordexp-swift-wave5-arm-resource-samples-v5"
RANK_STREAM_SCHEMA = "coordexp-swift-wave5-executed-rank-stream-v1"
CONFIG_COMPATIBILITY_PROJECTION_SCHEMA = (
    "coordexp-swift-wave5-w0-current-config-compatibility-projection-v2"
)
PACK_CACHE_VERSION = "coordexp-swift-pack-cache-v3"

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_PATH = Path(__file__).resolve()
TRAIN_PATH = (REPO_ROOT / "src/train.py").resolve()
BASE_CONFIG_PATH = (
    REPO_ROOT / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_"
    "llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
).resolve()
HISTORICAL_R2_PLAN_PATH = (
    REPO_ROOT
    / "outputs/probes/coordexp_swift/wave5_provider_benchmark/2026-08-10-r2/plan.json"
).resolve()
HISTORICAL_R2_PLAN_FILE_SHA256 = (
    "154d13d9f989fa2c0f56f6c364c0d3ba60e481f74a95bc1fe621dd8f2945ec7a"
)

ARM_SPECS: dict[str, dict[str, str]] = {
    "R": {"provider_mode": "synchronous", "role": "compatibility_reference"},
    "D": {"provider_mode": "legacy_fused", "role": "candidate"},
    "O": {"provider_mode": "overlapped", "role": "candidate"},
}
TRIAD_ORDERS = (("R", "D", "O"), ("D", "O", "R"), ("O", "R", "D"))
REFERENCE_ARM = "R"
CANDIDATE_ARMS = ("D", "O")
WORLD_SIZE = 8
WARMUP_STEPS = (1, 2)
MEASURED_STEPS = (3, 4, 5)
MIN_PAIRED_OBSERVATIONS = 3
ARM_TIMEOUT_SECONDS = 30 * 60
MATRIX_WALL_CEILING_SECONDS = 3 * 60 * 60
TERMINATION_GRACE_SECONDS = 60.0
STEADY_NOISE_FLOOR_FRACTION = 0.05
HOST_RSS_PER_RANK_CEILING_BYTES = 64 * 1024**3
HOST_RSS_ALL_RANK_CEILING_BYTES = 256 * 1024**3
GPU_MEMORY_CEILING_BYTES = 76 * 1024**3
ARTIFACT_CEILING_BYTES = 20 * 1024**3
GPU_IDLE_MEMORY_LIMIT_BYTES = 1 * 1024**3
GPU_IDLE_UTILIZATION_LIMIT_PERCENT = 5.0
GPU_IDLE_STABLE_SAMPLE_COUNT = 2
GPU_BASELINE_SAMPLE_INTERVAL_SECONDS = 2.0
GPU_IDLE_SAMPLE_INTERVAL_SECONDS = GPU_BASELINE_SAMPLE_INTERVAL_SECONDS
GPU_EXPECTED_TOTAL_MEMORY_BYTES = 81_920 * 1024**2
SHARED_GPU_BASELINE_MAX_USED_BYTES = 49_152 * 1024**2
GPU_MINIMUM_HEADROOM_BYTES = 32_768 * 1024**2
GPU_POST_PHASE_SAMPLE_COUNT = 2
GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS = 2.0
MAX_MONOTONIC_NS = 2**63 - 1
GPU_EXECUTION_POLICIES = ("idle_promotional", "shared_nonpromotional")
MEASUREMENT_WALL_CLOCK_SCOPE = "training_entry_to_terminal_artifact"
MEASUREMENT_ENTRY_TO_TERMINAL_BOUNDARY = (
    "training_entry_to_terminal_state_durable_before_measurement_annotation"
)
PROFILE_SYNC_TIMINGS = {"enabled": False, "source": "default"}
HISTORICAL_W0_EXPECTED_EVAL_REDUCTION = {
    "schema_version": 1,
    "mode": "disjoint_shard",
    "source": "default",
}
EXPECTED_EVAL_REDUCTION = {
    "schema_version": 1,
    "control": "auto",
    "effective_mode": "disjoint_shard",
    "source": "default",
    "pack_count": 8,
    "world_size": WORLD_SIZE,
}
CURRENT_CONFIG_COMPATIBILITY_DEFAULT_PATH_VALUES = (
    ("packing.policy", "source_order_next_fit"),
    ("packing.window_size", None),
    ("packing.lookahead", None),
    ("packing.seed", 0),
    ("packing.worker_count", 1),
    ("packing.fragment_item_budget", 1024),
    ("packing.fragment_byte_budget", 4_194_304),
    ("packing.cursor_byte_budget", 65_536),
    ("packing.max_packs_per_fragment", None),
    ("resume", {"checkpoint_dir": None, "mode": "disabled"}),
)
E2E_REGRESSION_LIMIT_FRACTION = 0.02

EXPECTED_PHASES = (
    "config_provenance_resolution",
    "cache_identity_resolution",
    "cache_preparation",
    "cache_publication",
    "cache_publication_admission",
    "cache_admission",
    "train_rank_hydration",
    "evaluation_hydration",
    "model_loading",
    "optimizer_runtime_assembly",
    "first_optimizer_step",
    "steady_state",
    "evaluation_execution",
    "checkpoint_publication",
)
EXECUTION_OWNER_PATHS = {
    "provider": REPO_ROOT / "src/training/forward_input_provider.py",
    "pipeline": REPO_ROOT / "src/training/pipeline.py",
    # The training facade kept its name but no longer owns the behavior it is
    # fingerprinted for: these are its post-decomposition owners.
    "execution_plan": REPO_ROOT / "src/training/execution_plan.py",
    "control_plane": REPO_ROOT / "src/training/control_plane.py",
    "session": REPO_ROOT / "src/training/session.py",
    "cache_workflow": REPO_ROOT / "src/training/cache_workflow.py",
    "cache_contract": REPO_ROOT / "src/training/cache_contract.py",
    "reporting": REPO_ROOT / "src/training/reporting.py",
    "trainer": REPO_ROOT / "src/training/supervised_trainer.py",
    "micro_steps": REPO_ROOT / "src/training/micro_steps.py",
    "artifact_writer": REPO_ROOT / "src/artifacts/run_writer.py",
    "run_schema": REPO_ROOT / "src/artifacts/run_schema.py",
    "run_state": REPO_ROOT / "src/artifacts/run_state.py",
    "checkpoint_writer": REPO_ROOT / "src/artifacts/checkpoints.py",
    "provenance": REPO_ROOT / "src/artifacts/provenance.py",
    "config_loader": REPO_ROOT / "src/config/loader.py",
    "config_models": REPO_ROOT / "src/config/models.py",
    "train_entrypoint": TRAIN_PATH,
    "benchmark_instrumentation": SCRIPT_PATH,
}
STREAM_PARITY_FIELDS = (
    "pack_ids_sha256",
    "tensor_values_sha256",
    "tensor_metadata_sha256",
    "supervision_sha256",
    "position_ids_sha256",
    "loss_inputs_sha256",
)
SEMANTIC_PARITY_FIELDS = (
    *STREAM_PARITY_FIELDS,
    "loss_results_sha256",
    "optimizer_semantics_sha256",
    "evaluation_results_sha256",
    "checkpoint_output_sha256",
)

HISTORICAL_W0_RECEIPT = {
    "schema": "coordexp-swift-wave0-historical-stream-receipt-v1",
    "status": "historical_evidence_only_not_loader_input",
    "derived_five_step_config_fingerprint": (
        "1088934be1d193307b0f11393ade898d615716e8e49ed00f98fe7ba2236fde2a"
    ),
    "seed": 17,
    "world_size": WORLD_SIZE,
    "grad_accum_steps": 3,
    "warmup_steps": list(WARMUP_STEPS),
    "measured_steps": list(MEASURED_STEPS),
    "local_pack_presentations_per_rank": 15,
    "global_rank_pack_presentations": 120,
    "train_pack_count": 32,
    "eval_pack_count": 8,
    "eval_steps": [3],
    "checkpoint_steps": [5],
}

HISTORICAL_W0_CACHE_BINDING = {
    "train": {
        "fingerprint": "c6be15b8d7840524829c70e1fa5729accf600ae366d97fccf32778eca84c7de1",
        "manifest_sha256": "636c701efd1cf94af0f88bdb1d86fcef0529d47609c2b48fbf433a7172da7a4e",
        "micro_step_count": 32,
        "chunk_sha256": "b5becdfc84c6569554af01bbeb41a153ae168b4d3796f8e7209078349f6db665",
    },
    "eval": {
        "fingerprint": "60330b24519f0931e5c84765874c70e0cb5d54cd472bd6f79fc30977edab626c",
        "manifest_sha256": "d8ca6858a56fb895f3ba037bfa27d96104cde6881aded837ec4473834ecc73bf",
        "micro_step_count": 8,
        "chunk_sha256": "fa8ea831c1912572ab75d132c57da5f1114b18650236971209004c1934b4ce59",
    },
}

HISTORICAL_W0_RESOLVED_CONFIG_PATH = Path(
    "/tmp/coordexp-wave0-baseline-FIU33RQX/outputs/"
    "wave0_compatibility_reference_5step/resolved_config.json"
)
HISTORICAL_W0_RESOLVED_CONFIG_FILE_SHA256 = (
    "f424b5e7924af113283c0e369da6d22a264cef4d97079c35d7c28d9cea7c9ef7"
)
HISTORICAL_W0_RUN_RECEIPT_PATH = HISTORICAL_W0_RESOLVED_CONFIG_PATH.with_name(
    "run.json"
)
HISTORICAL_W0_RUN_RECEIPT_FILE_SHA256 = (
    "2c5879129e9e156119e38985496e3d7751453509be3da8b8f70280ea22370ef9"
)
HISTORICAL_W0_BASE_WEIGHT_AGGREGATE_SHA256 = (
    "e128f5f42f1a042702efc1eed5a787da36a4d586a17b538671260996056284aa"
)
HISTORICAL_W0_WEIGHT_IDENTITY_RECEIPT_PATH = (
    REPO_ROOT
    / "openspec/changes/archive/2026-08-12-harden-optimize-coordexp-swift-training-infrastructure/"
    "receipts/wave2-v3-plan.json"
)
HISTORICAL_W0_WEIGHT_IDENTITY_RECEIPT_FILE_SHA256 = (
    "141b4294dacae69f39907303d3ae75c4c4182d8da7625b399280ed99ed66187d"
)


class Wave5BenchmarkError(RuntimeError):
    """A typed Wave 5 contract failure."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise Wave5BenchmarkError(
            "value is not strict JSON", code="wave5.json"
        ) from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finalize(payload: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    if hash_field in result:
        raise Wave5BenchmarkError("hash already present", code="wave5.hash")
    result[hash_field] = sha256_json(result)
    return result


def _validate_finalized(
    payload: Mapping[str, Any], *, schema: str, hash_field: str
) -> dict[str, Any]:
    value = deepcopy(dict(payload))
    if value.get("schema") != schema:
        raise Wave5BenchmarkError("schema mismatch", code="wave5.schema")
    digest = value.pop(hash_field, None)
    if not _is_sha256(digest) or digest != sha256_json(value):
        raise Wave5BenchmarkError("artifact hash mismatch", code="wave5.hash")
    value[hash_field] = digest
    return value


def load_strict_json(path: str | Path) -> dict[str, Any]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Wave5BenchmarkError(
            "cannot read strict JSON", code="wave5.json_read"
        ) from exc
    if not isinstance(value, dict):
        raise Wave5BenchmarkError("JSON root is not an object", code="wave5.json_shape")
    canonical_json_bytes(value)
    return value


def load_historical_r2_plan(
    path: str | Path = HISTORICAL_R2_PLAN_PATH,
) -> dict[str, str]:
    actual = _lexical(path)
    if actual != HISTORICAL_R2_PLAN_PATH or not actual.is_file():
        raise Wave5BenchmarkError(
            "historical r2 plan path is not exact", code="wave5.historical_plan"
        )
    if sha256_file(actual) != HISTORICAL_R2_PLAN_FILE_SHA256:
        raise Wave5BenchmarkError(
            "historical r2 plan bytes changed", code="wave5.historical_plan"
        )
    payload = _validate_finalized(
        load_strict_json(actual),
        schema=HISTORICAL_PLAN_SCHEMA_V3,
        hash_field="plan_sha256",
    )
    return {
        "status": "historical_non_executable",
        "schema": HISTORICAL_PLAN_SCHEMA_V3,
        "path": str(actual),
        "file_sha256": HISTORICAL_R2_PLAN_FILE_SHA256,
        "plan_sha256": payload["plan_sha256"],
    }


def _lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _canonical_nonsymlink(path: str | Path, *, must_exist: bool = False) -> Path:
    candidate = _lexical(path)
    if candidate != Path(path).expanduser().resolve(strict=False):
        raise Wave5BenchmarkError(
            "path is not canonical", code="wave5.target_canonical"
        )
    current = candidate
    while True:
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise Wave5BenchmarkError(
                    "path traverses symlink", code="wave5.target_symlink"
                )
        if current == current.parent:
            break
        current = current.parent
    if must_exist and not candidate.exists():
        raise Wave5BenchmarkError(
            "required path is absent", code="wave5.target_missing"
        )
    return candidate


def _require_within(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Wave5BenchmarkError(
            "target escapes exact root", code="wave5.target_root"
        ) from exc
    if path == root:
        raise Wave5BenchmarkError(
            "root itself is not an artifact target", code="wave5.target_root"
        )


def _validate_distinct_paths(root: Path, paths: Sequence[Path]) -> None:
    normalized: set[Path] = set()
    inodes: set[tuple[int, int]] = set()
    for path in paths:
        checked = _canonical_nonsymlink(path)
        _require_within(root, checked)
        if checked in normalized:
            raise Wave5BenchmarkError(
                "artifact target aliases another target", code="wave5.target_alias"
            )
        normalized.add(checked)
        if checked.exists():
            stat = checked.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in inodes:
                raise Wave5BenchmarkError(
                    "artifact targets are hardlink aliases", code="wave5.target_alias"
                )
            inodes.add(inode)


def publish_json_absent(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _canonical_nonsymlink(path)
    if target.exists():
        raise Wave5BenchmarkError(
            "artifact target exists", code="wave5.artifact_collision"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    _canonical_nonsymlink(target.parent, must_exist=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise Wave5BenchmarkError(
                "artifact target exists", code="wave5.artifact_collision"
            ) from exc
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if load_strict_json(target) != dict(payload):
        raise Wave5BenchmarkError(
            "published artifact reload mismatch", code="wave5.publish_reload"
        )
    return target


def _parse_attested_at(value: object) -> None:
    if not isinstance(value, str):
        raise Wave5BenchmarkError(
            "compatibility date missing", code="wave5.compatibility_date"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Wave5BenchmarkError(
            "compatibility date invalid", code="wave5.compatibility_date"
        ) from exc
    if parsed.tzinfo is None or parsed.date().isoformat() < "2026-08-10":
        raise Wave5BenchmarkError(
            "compatibility attestation predates v3 contract",
            code="wave5.compatibility_date",
        )


def _semantic_json_value(value: Any) -> Any:
    """Return a strict, type-preserving JSON projection for cached semantics."""

    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "value_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _semantic_json_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise Wave5BenchmarkError(
                "semantic mapping key is not text", code="wave5.stream_projection"
            )
        return {key: _semantic_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_semantic_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        projected = [_semantic_json_value(item) for item in value]
        return sorted(projected, key=canonical_json_bytes)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        canonical_json_bytes(value)
        return value
    raise Wave5BenchmarkError(
        f"unsupported semantic value {type(value).__name__}",
        code="wave5.stream_projection",
    )


def _tensor_streams(
    value: Any, *, path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    values: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    def walk(item: Any, item_path: str) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            values.append(
                {"path": item_path, "sha256": hashlib.sha256(raw).hexdigest()}
            )
            metadata.append(
                {
                    "path": item_path,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "requires_grad": bool(item.requires_grad),
                }
            )
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                walk(getattr(item, field.name), f"{item_path}.{field.name}")
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                walk(item[key], f"{item_path}[{key!r}]")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")

    walk(value, path)
    return values, metadata


def _pack_identity(step: Any) -> dict[str, Any]:
    pack = step.pack
    segment_rows = []
    for segment in getattr(pack, "segments", ()) or ():
        segment_rows.append(
            {
                name: getattr(segment, name, None)
                for name in (
                    "pack_index",
                    "segment_index",
                    "example_index",
                    "example_id",
                    "start",
                    "end",
                )
            }
        )
    return {
        "metadata_pack_id": (
            step.metadata.get("pack_id") if isinstance(step.metadata, Mapping) else None
        ),
        "pack_index": getattr(pack, "pack_index", None),
        "segments": segment_rows,
        "example_ids": [
            getattr(example, "example_id", str(example))
            for example in step.encoded_examples
        ],
        "pack_fallback": None if segment_rows else _semantic_json_value(pack),
    }


def _stream_semantic_digests(
    micro_steps: Sequence[Any], *, remove_pack_plan: bool = False
) -> dict[str, str]:
    if not micro_steps:
        raise Wave5BenchmarkError(
            "decoded cache stream is empty", code="wave5.cache_empty"
        )
    tensor_values: list[dict[str, Any]] = []
    tensor_metadata: list[dict[str, Any]] = []
    for index, step in enumerate(micro_steps):
        values, metadata = _tensor_streams(step, path=f"micro_steps[{index}]")
        tensor_values.extend(values)
        tensor_metadata.extend(metadata)
    supervision = []
    for index, step in enumerate(micro_steps):
        metadata = step.metadata
        if not isinstance(metadata, Mapping):
            raise Wave5BenchmarkError(
                f"micro-step metadata is malformed at index {index}",
                code="wave5.stream_projection",
            )
        projected_metadata = dict(metadata)
        if remove_pack_plan:
            projected_metadata.pop("pack_plan")
        supervision.append(
            _semantic_json_value(
                {
                    "encoded_examples": step.encoded_examples,
                    "token_sequence": step.token_sequence,
                    "vocab_groups": step.vocab_groups,
                    "metadata": projected_metadata,
                }
            )
        )
    projections = {
        "pack_ids_sha256": [_pack_identity(step) for step in micro_steps],
        "tensor_values_sha256": tensor_values,
        "tensor_metadata_sha256": tensor_metadata,
        "supervision_sha256": supervision,
        "position_ids_sha256": [
            _semantic_json_value(step.position_inputs) for step in micro_steps
        ],
        "loss_inputs_sha256": [
            _semantic_json_value(
                {
                    "token_sequence": step.token_sequence,
                    "vocab_groups": step.vocab_groups,
                    "expected_vocab_size": step.expected_vocab_size,
                    "extra_model_kwargs": step.extra_model_kwargs,
                    "fa2_model_dtype": step.fa2_model_dtype,
                    "capture_fa2_branch": step.capture_fa2_branch,
                    "require_fa2_branch_proof": step.require_fa2_branch_proof,
                    "fa2_branch_proof_policy": step.fa2_branch_proof_policy,
                }
            )
            for step in micro_steps
        ],
    }
    return {name: sha256_json(value) for name, value in projections.items()}


def _expected_current_packing_determinant() -> dict[str, Any]:
    from src.packing.planner import (
        PACK_PLAN_SCHEMA,
        PACK_PLAN_SCHEMA_VERSION,
        build_pack_plan_policy_identity,
    )

    packing = _five_step_reference_config()["packing"]
    return {
        "schema": PACK_PLAN_SCHEMA,
        "schema_version": PACK_PLAN_SCHEMA_VERSION,
        "global_max_length": int(packing["global_max_length"]),
        "policy_identity": build_pack_plan_policy_identity(
            policy=str(packing["policy"]),
            window_size=packing["window_size"],
            lookahead=packing["lookahead"],
            seed=int(packing["seed"]),
            worker_count=int(packing["worker_count"]),
            cursor_byte_budget=int(packing["cursor_byte_budget"]),
            fragment_item_budget=int(packing["fragment_item_budget"]),
            fragment_byte_budget=int(packing["fragment_byte_budget"]),
        ),
        "fragment_pack_budget": packing["max_packs_per_fragment"],
    }


def _stream_semantic_projection(
    micro_steps: Sequence[Any],
    *,
    role: str,
    packing_determinant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if role not in {"historical_v2", "current_v3"}:
        raise Wave5BenchmarkError(
            "stream compatibility projection role is invalid",
            code="wave5.stream_projection",
        )
    if not micro_steps:
        raise Wave5BenchmarkError(
            "decoded cache stream is empty", code="wave5.cache_empty"
        )
    raw = _stream_semantic_digests(micro_steps)
    if role == "historical_v2":
        if packing_determinant is not None:
            raise Wave5BenchmarkError(
                "historical v2 projection cannot bind a v3 packing determinant",
                code="wave5.pack_plan_provenance",
            )
        for index, step in enumerate(micro_steps):
            metadata = step.metadata
            if not isinstance(metadata, Mapping) or "pack_plan" in metadata:
                raise Wave5BenchmarkError(
                    f"historical v2 metadata.pack_plan must be absent at index {index}",
                    code="wave5.pack_plan_provenance",
                )
        provenance = {
            "schema": PACK_PLAN_PROVENANCE_SCHEMA,
            "role": role,
            "path": "metadata.pack_plan",
            "disposition": "required_absent",
            "micro_step_count": len(micro_steps),
        }
        return {
            "schema": STREAM_COMPATIBILITY_PROJECTION_SCHEMA,
            "role": role,
            "raw_semantic_digests": raw,
            "projected_semantic_digests": dict(raw),
            "packing_determinant_sha256": None,
            "pack_plan_provenance_sha256": sha256_json(provenance),
        }

    expected_determinant = _expected_current_packing_determinant()
    if (
        not isinstance(packing_determinant, Mapping)
        or dict(packing_determinant) != expected_determinant
    ):
        raise Wave5BenchmarkError(
            "current v3 packing determinant is not the exact Wave5 determinant",
            code="wave5.pack_plan_provenance",
        )
    expected_receipt_keys = {
        "schema_version",
        "mode",
        "policy_identity",
        "plan_sha256",
        "fragment_chain_sha256",
        "fragment_count",
        "source_input_count",
        "emitted_pack_count",
        "fragment_sha256",
    }
    expected_policy = expected_determinant["policy_identity"]
    plans: list[dict[str, Any]] = []
    for index, step in enumerate(micro_steps):
        metadata = step.metadata
        if not isinstance(metadata, Mapping) or "pack_plan" not in metadata:
            raise Wave5BenchmarkError(
                f"current v3 metadata.pack_plan is missing at index {index}",
                code="wave5.pack_plan_provenance",
            )
        receipt = metadata["pack_plan"]
        if not isinstance(receipt, Mapping) or set(receipt) != expected_receipt_keys:
            raise Wave5BenchmarkError(
                f"current v3 pack-plan fields are not exact at index {index}",
                code="wave5.pack_plan_provenance",
            )
        if (
            receipt.get("schema_version") != 1
            or receipt.get("mode") != "complete_plan"
            or receipt.get("policy_identity") != expected_policy
            or not _is_sha256(receipt.get("plan_sha256"))
            or receipt.get("fragment_chain_sha256") is not None
            or receipt.get("fragment_count") != 1
            or receipt.get("fragment_sha256") != receipt.get("plan_sha256")
        ):
            raise Wave5BenchmarkError(
                f"current v3 pack-plan identity is invalid at index {index}",
                code="wave5.pack_plan_provenance",
            )
        plans.append(dict(receipt))
    plan = plans[0]
    if any(item != plan for item in plans[1:]):
        raise Wave5BenchmarkError(
            "current v3 pack-plan receipt changed within one split",
            code="wave5.pack_plan_provenance",
        )
    source_input_count = sum(len(step.encoded_examples) for step in micro_steps)
    if (
        isinstance(plan["source_input_count"], bool)
        or plan["source_input_count"] != source_input_count
        or isinstance(plan["emitted_pack_count"], bool)
        or plan["emitted_pack_count"] != len(micro_steps)
    ):
        raise Wave5BenchmarkError(
            "current v3 pack-plan counts do not match the decoded stream",
            code="wave5.pack_plan_provenance",
        )
    determinant_sha256 = sha256_json(expected_determinant)
    provenance = {
        "schema": PACK_PLAN_PROVENANCE_SCHEMA,
        "role": role,
        "path": "metadata.pack_plan",
        "disposition": "removed_after_exact_validation",
        "packing_determinant_sha256": determinant_sha256,
        "policy_identity_sha256": sha256_json(expected_policy),
        "plan_sha256": plan["plan_sha256"],
        "fragment_sha256_sequence": [item["fragment_sha256"] for item in plans],
        "fragment_count": plan["fragment_count"],
        "source_input_count": plan["source_input_count"],
        "emitted_pack_count": plan["emitted_pack_count"],
        "micro_step_count": len(micro_steps),
    }
    return {
        "schema": STREAM_COMPATIBILITY_PROJECTION_SCHEMA,
        "role": role,
        "raw_semantic_digests": raw,
        "projected_semantic_digests": _stream_semantic_digests(
            micro_steps, remove_pack_plan=True
        ),
        "packing_determinant_sha256": determinant_sha256,
        "pack_plan_provenance_sha256": sha256_json(provenance),
    }


def _rank_stream_digests(events: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if not events:
        raise Wave5BenchmarkError(
            "executed stream is empty", code="wave5.executed_stream"
        )
    projections = {
        "pack_ids_sha256": [event.get("pack") for event in events],
        "tensor_values_sha256": [event.get("tensor_values") for event in events],
        "tensor_metadata_sha256": [event.get("tensor_metadata") for event in events],
        "supervision_sha256": [event.get("supervision") for event in events],
        "position_ids_sha256": [event.get("position_ids") for event in events],
        "loss_inputs_sha256": [event.get("loss_inputs") for event in events],
    }
    return {name: sha256_json(value) for name, value in projections.items()}


def _build_rank_stream_receipt(
    *,
    rank: int,
    provider_mode: str,
    events: Sequence[Mapping[str, Any]],
    status: str = "complete",
    error: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if rank not in range(WORLD_SIZE) or provider_mode not in {
        spec["provider_mode"] for spec in ARM_SPECS.values()
    }:
        raise Wave5BenchmarkError(
            "executed stream identity invalid", code="wave5.executed_stream"
        )
    copied_events = deepcopy([dict(event) for event in events])
    digests = _rank_stream_digests(copied_events) if copied_events else None
    return _finalize(
        {
            "schema": RANK_STREAM_SCHEMA,
            "status": status,
            "rank": rank,
            "world_size": WORLD_SIZE,
            "provider_mode": provider_mode,
            "event_count": len(copied_events),
            "events": copied_events,
            "semantic_digests": digests,
            "error": None if error is None else dict(error),
        },
        hash_field="stream_receipt_sha256",
    )


def _validate_rank_stream_receipt(
    payload: Mapping[str, Any], *, rank: int, provider_mode: str
) -> dict[str, Any]:
    receipt = _validate_finalized(
        payload, schema=RANK_STREAM_SCHEMA, hash_field="stream_receipt_sha256"
    )
    events = receipt.get("events")
    if (
        set(receipt)
        != {
            "schema",
            "status",
            "rank",
            "world_size",
            "provider_mode",
            "event_count",
            "events",
            "semantic_digests",
            "error",
            "stream_receipt_sha256",
        }
        or receipt.get("status") != "complete"
        or receipt.get("rank") != rank
        or receipt.get("world_size") != WORLD_SIZE
        or receipt.get("provider_mode") != provider_mode
        or not isinstance(events, list)
        or receipt.get("event_count") != len(events)
        or len(events) != 15
        or receipt.get("error") is not None
    ):
        raise Wave5BenchmarkError(
            "executed stream receipt is incomplete", code="wave5.executed_stream"
        )
    coordinates = [
        (event.get("planned_step_id"), event.get("local_micro_step_index"))
        for event in events
        if isinstance(event, Mapping)
    ]
    if coordinates != [(step, micro) for step in range(1, 6) for micro in range(3)]:
        raise Wave5BenchmarkError(
            "executed stream coordinates are not exact",
            code="wave5.executed_stream",
        )
    if receipt.get("semantic_digests") != _rank_stream_digests(events):
        raise Wave5BenchmarkError(
            "executed stream semantic digests changed",
            code="wave5.executed_stream",
        )
    return receipt


def _execution_repository_identity() -> dict[str, Any]:
    owners = {
        name: _file_identity(path)
        for name, path in sorted(EXECUTION_OWNER_PATHS.items())
    }
    return {"owners": owners, "aggregate_sha256": sha256_json(owners)}


@lru_cache(maxsize=1)
def _execution_baseline_bytes() -> bytes:
    from src.artifacts.provenance import pinned_runtime_baseline

    runtime = pinned_runtime_baseline()
    return canonical_json_bytes(
        {
            "repository": _execution_repository_identity(),
            "runtime": {
                "baseline": runtime,
                "baseline_sha256": sha256_json(runtime),
            },
        }
    )


def _execution_baseline() -> dict[str, Any]:
    value = json.loads(_execution_baseline_bytes())
    if not isinstance(value, dict):
        raise Wave5BenchmarkError(
            "execution baseline cache is malformed", code="wave5.execution_baseline"
        )
    return value


def _collect_execution_baseline_provenance(
    *, repository_root: str | Path
) -> dict[str, Any]:
    from src.artifacts.provenance import collect_execution_provenance

    return collect_execution_provenance(repository_root=repository_root)


def _revalidate_execution_baseline(plan: Mapping[str, Any]) -> None:
    from src.artifacts.provenance import (
        pinned_runtime_baseline,
        require_pinned_runtime_baseline,
    )

    expected = plan.get("execution_baseline")
    runtime = pinned_runtime_baseline()
    current = {
        "repository": _execution_repository_identity(),
        "runtime": {"baseline": runtime, "baseline_sha256": sha256_json(runtime)},
    }
    if expected != current:
        raise Wave5BenchmarkError(
            "execution baseline changed", code="wave5.execution_baseline"
        )
    provenance = _collect_execution_baseline_provenance(repository_root=REPO_ROOT)
    try:
        require_pinned_runtime_baseline(
            provenance=provenance, attention_backend="flash_attention_2"
        )
    except Exception as exc:
        raise Wave5BenchmarkError(
            "runtime dependency baseline changed",
            code="wave5.execution_baseline",
        ) from exc


def _measurement_contract() -> dict[str, Any]:
    return {
        "world_size": WORLD_SIZE,
        "comparison_arms": list(ARM_SPECS),
        "warmup_steps": list(WARMUP_STEPS),
        "warmup_exclusion_steps": len(WARMUP_STEPS),
        "measured_steps": list(MEASURED_STEPS),
        "wall_clock_scope": MEASUREMENT_WALL_CLOCK_SCOPE,
        "entry_to_terminal_boundary": MEASUREMENT_ENTRY_TO_TERMINAL_BOUNDARY,
        "profile_sync_timings": deepcopy(PROFILE_SYNC_TIMINGS),
        "accepted_paired_observations": MIN_PAIRED_OBSERVATIONS,
        "phase_names": list(EXPECTED_PHASES),
        "arm_timeout_seconds": ARM_TIMEOUT_SECONDS,
        "matrix_wall_ceiling_seconds": MATRIX_WALL_CEILING_SECONDS,
        "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
        "noise_rule": (
            "both_median_paired_steady_state_and_entry_to_terminal_gain_gt_"
            "max(0.05,2*respective_mad)"
        ),
        "entry_to_terminal_regression_limit_fraction": (E2E_REGRESSION_LIMIT_FRACTION),
        "retry": False,
        "blind_retry": False,
    }


def _expected_measurement_context(
    plan: Mapping[str, Any], *, arm: str, workload_identity: str
) -> dict[str, Any]:
    contract = plan.get("measurement_contract")
    if contract != _measurement_contract() or arm not in ARM_SPECS:
        raise Wave5BenchmarkError(
            "measurement contract changed", code="wave5.measurement_context"
        )
    if not isinstance(workload_identity, str) or not workload_identity:
        raise Wave5BenchmarkError(
            "measurement workload identity missing",
            code="wave5.measurement_context",
        )
    return {
        "comparison_arm": arm,
        "wall_clock_scope": contract["wall_clock_scope"],
        "warmup_exclusion_steps": contract["warmup_exclusion_steps"],
        "workload_identity": workload_identity,
        "world_size": WORLD_SIZE,
        "profile_sync_timings": deepcopy(contract["profile_sync_timings"]),
    }


def _five_step_reference_config() -> dict[str, Any]:
    from src.config.loader import load_train_config

    config = deepcopy(load_train_config(BASE_CONFIG_PATH).config_dict)
    config["training"]["max_steps"] = 5
    config["eval"]["forward"]["steps"] = [3]
    config["checkpoint"]["steps"] = [5]
    return config


def _current_config_compatibility_removed_path_values() -> list[dict[str, Any]]:
    return json.loads(
        canonical_json_bytes(
            [
                {"path": path, "value": value}
                for path, value in CURRENT_CONFIG_COMPATIBILITY_DEFAULT_PATH_VALUES
            ]
        )
    )


def _remove_exact_config_path_values(
    config: dict[str, Any], path_values: Sequence[Mapping[str, Any]]
) -> None:
    for row in path_values:
        if set(row) != {"path", "value"}:
            raise Wave5BenchmarkError(
                "config compatibility removal row is malformed",
                code="wave5.config_compatibility",
            )
        path = row.get("path")
        if not isinstance(path, str) or not path:
            raise Wave5BenchmarkError(
                "config compatibility removal path is malformed",
                code="wave5.config_compatibility",
            )
        owner: Any = config
        parts = path.split(".")
        for part in parts[:-1]:
            if not isinstance(owner, dict) or part not in owner:
                raise Wave5BenchmarkError(
                    "config compatibility removal path is absent",
                    code="wave5.config_compatibility",
                )
            owner = owner[part]
        leaf = parts[-1]
        if not isinstance(owner, dict) or leaf not in owner:
            raise Wave5BenchmarkError(
                "config compatibility removal path is absent",
                code="wave5.config_compatibility",
            )
        if owner[leaf] != row["value"]:
            raise Wave5BenchmarkError(
                "config compatibility value is not the exact current default",
                code="wave5.config_compatibility",
            )
        del owner[leaf]


def _project_arm_config(
    config: Mapping[str, Any], *, historical_w0: bool = False
) -> dict[str, Any]:
    projected = deepcopy(dict(config))
    try:
        for field in ("name", "artifact_root", "collision_policy"):
            projected["run"].pop(field)
        provider = projected["training"].pop(
            "forward_input_provider_mode", None if historical_w0 else object()
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise Wave5BenchmarkError(
            "config compatibility projection is malformed",
            code="wave5.config_compatibility",
        ) from exc
    allowed_providers = {spec["provider_mode"] for spec in ARM_SPECS.values()}
    if provider not in allowed_providers and not (historical_w0 and provider is None):
        raise Wave5BenchmarkError(
            "config provider compatibility factor is invalid",
            code="wave5.config_compatibility",
        )
    if not historical_w0:
        _remove_exact_config_path_values(
            projected,
            _current_config_compatibility_removed_path_values(),
        )
    return projected


def _load_authenticated_historical_w0_weight_identity(
    *, expected_model_root: str
) -> dict[str, Any]:
    from src.qwen.parity import validate_model_weight_identity

    receipt_path = _canonical_nonsymlink(
        HISTORICAL_W0_WEIGHT_IDENTITY_RECEIPT_PATH, must_exist=True
    )
    if sha256_file(receipt_path) != HISTORICAL_W0_WEIGHT_IDENTITY_RECEIPT_FILE_SHA256:
        raise Wave5BenchmarkError(
            "historical W0 weight receipt authentication failed",
            code="wave5.historical_w0",
        )
    receipt = load_strict_json(receipt_path)
    identity = receipt.get("model_weight_identity")
    try:
        checked = validate_model_weight_identity(identity)
    except Exception as exc:
        raise Wave5BenchmarkError(
            "historical W0 weight receipt is invalid",
            code="wave5.historical_w0",
        ) from exc
    if (
        checked.get("aggregate_sha256") != HISTORICAL_W0_BASE_WEIGHT_AGGREGATE_SHA256
        or checked.get("root") != expected_model_root
    ):
        raise Wave5BenchmarkError(
            "historical W0 weight identity disagrees with resolved config",
            code="wave5.historical_w0",
        )
    return {
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": (HISTORICAL_W0_WEIGHT_IDENTITY_RECEIPT_FILE_SHA256),
        "identity": checked,
    }


def _load_authenticated_historical_w0() -> dict[str, Any]:
    resolved_path = _canonical_nonsymlink(
        HISTORICAL_W0_RESOLVED_CONFIG_PATH, must_exist=True
    )
    run_path = _canonical_nonsymlink(HISTORICAL_W0_RUN_RECEIPT_PATH, must_exist=True)
    if (
        sha256_file(resolved_path) != HISTORICAL_W0_RESOLVED_CONFIG_FILE_SHA256
        or sha256_file(run_path) != HISTORICAL_W0_RUN_RECEIPT_FILE_SHA256
    ):
        raise Wave5BenchmarkError(
            "historical W0 artifact authentication failed",
            code="wave5.historical_w0",
        )
    resolved = load_strict_json(resolved_path)
    run = load_strict_json(run_path)
    config = resolved.get("config")
    resolution = resolved.get("resolution")
    fingerprint = HISTORICAL_W0_RECEIPT["derived_five_step_config_fingerprint"]
    expected_eval_reduction = deepcopy(HISTORICAL_W0_EXPECTED_EVAL_REDUCTION)
    measurement = run.get("measurement")
    context = measurement.get("context") if isinstance(measurement, Mapping) else None
    policies = run.get("policy_identities")
    if (
        not isinstance(config, Mapping)
        or not isinstance(resolution, Mapping)
        or resolution.get("fingerprint") != fingerprint
        or sha256_json(config) != fingerprint
        or run.get("config_fingerprint") != fingerprint
        or run.get("status") != "completed"
        or run.get("completed_steps") != 5
        or run.get("checkpoint_event_count") != 1
        or run.get("forward_input_provider_mode") != "synchronous"
        or not isinstance(context, Mapping)
        or context.get("warmup_exclusion_steps") != len(WARMUP_STEPS)
        or context.get("profile_sync_timings") != PROFILE_SYNC_TIMINGS
        or not isinstance(policies, Mapping)
        or policies.get("eval_reduction") != expected_eval_reduction
    ):
        raise Wave5BenchmarkError(
            "historical W0 resolved config or receipt is invalid",
            code="wave5.historical_w0",
        )
    projection = _project_arm_config(config, historical_w0=True)
    weight_identity = _load_authenticated_historical_w0_weight_identity(
        expected_model_root=str(config["model"]["base_model"])
    )
    return {
        "resolved_config_path": str(resolved_path),
        "resolved_config_file_sha256": HISTORICAL_W0_RESOLVED_CONFIG_FILE_SHA256,
        "run_receipt_path": str(run_path),
        "run_receipt_file_sha256": HISTORICAL_W0_RUN_RECEIPT_FILE_SHA256,
        "config_fingerprint": fingerprint,
        "resolved_config": deepcopy(dict(config)),
        "projection": projection,
        "projection_sha256": sha256_json(projection),
        "eval_reduction": expected_eval_reduction,
        "base_model_weight_identity": weight_identity,
    }


def _config_compatibility_contract() -> dict[str, Any]:
    from src.config.loader import load_train_config

    historical = _load_authenticated_historical_w0()
    normative_current = load_train_config(BASE_CONFIG_PATH)
    normative_current_config = deepcopy(normative_current.config_dict)
    reference = _five_step_reference_config()
    projection = _project_arm_config(reference)
    projection_sha256 = sha256_json(projection)
    if projection_sha256 != historical["projection_sha256"]:
        raise Wave5BenchmarkError(
            "current config is not compatible with authenticated W0 after only "
            "declared arm factors and exact current defaults",
            code="wave5.config_compatibility",
        )
    compatibility_projection = {
        "schema": CONFIG_COMPATIBILITY_PROJECTION_SCHEMA,
        "current_config_sha256": sha256_json(reference),
        "removed_path_values": (_current_config_compatibility_removed_path_values()),
        "projected_config_sha256": projection_sha256,
        "historical_config_sha256": historical["config_fingerprint"],
        "historical_projected_config_sha256": historical["projection_sha256"],
        "policy": "remove_exact_enumerated_current_defaults_after_arm_factor",
    }
    return {
        "schema": "coordexp-swift-wave5-w0-current-config-compatibility-v1",
        "historical_w0": historical,
        "normative_current_full_config": normative_current_config,
        "normative_current_full_config_sha256": normative_current.fingerprint,
        "five_step_reference_config": reference,
        "current_five_step_config_sha256": sha256_json(reference),
        "allowed_arm_difference_paths": [
            "run.name",
            "run.artifact_root",
            "run.collision_policy",
            "training.forward_input_provider_mode",
        ],
        "compatibility_projection": compatibility_projection,
        "current_parent_projection": projection,
        "current_parent_projection_sha256": projection_sha256,
        "historical_w0_projection_sha256": historical["projection_sha256"],
    }


def _weight_file_stat_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.name == "model.safetensors.index.json"
        or path.name == "model.safetensors"
        or (path.name.startswith("model-") and path.name.endswith(".safetensors"))
    )
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in candidates
    )


@lru_cache(maxsize=4)
def _base_model_weight_identity_for_signature(
    root: str, signature: tuple[tuple[str, int, int], ...]
) -> dict[str, Any]:
    del signature
    from src.qwen.parity import base_model_weight_identity

    return base_model_weight_identity(root)


def _current_base_model_weight_identity() -> dict[str, Any]:
    root = Path(_five_step_reference_config()["model"]["base_model"]).resolve()
    try:
        signature = _weight_file_stat_signature(root)
    except OSError as exc:
        raise Wave5BenchmarkError(
            "base-model weight identity unavailable",
            code="wave5.weight_identity",
        ) from exc
    return deepcopy(_base_model_weight_identity_for_signature(str(root), signature))


def _revalidate_plan_compatibility(plan: Mapping[str, Any]) -> None:
    from src.qwen.parity import (
        assert_model_weight_identity_equal,
        validate_model_weight_identity,
    )

    if plan.get("config_compatibility") != _config_compatibility_contract():
        raise Wave5BenchmarkError(
            "config compatibility binding changed",
            code="wave5.config_compatibility",
        )
    expected_weights = plan.get("base_model_weight_identity")
    if not isinstance(expected_weights, Mapping):
        raise Wave5BenchmarkError(
            "base-model weight identity missing", code="wave5.weight_identity"
        )
    try:
        validate_model_weight_identity(expected_weights)
        if (
            plan.get("historical_w0_base_weight_aggregate_sha256")
            != HISTORICAL_W0_BASE_WEIGHT_AGGREGATE_SHA256
            or expected_weights.get("aggregate_sha256")
            != HISTORICAL_W0_BASE_WEIGHT_AGGREGATE_SHA256
            or expected_weights.get("root")
            != plan["config_compatibility"]["normative_current_full_config"]["model"][
                "base_model"
            ]
        ):
            raise Wave5BenchmarkError(
                "current and historical W0 base-model weight identity differ",
                code="wave5.weight_identity",
            )
        assert_model_weight_identity_equal(
            expected_weights, _current_base_model_weight_identity()
        )
    except Exception as exc:
        raise Wave5BenchmarkError(
            "base-model weight identity changed",
            code="wave5.weight_identity",
        ) from exc


def _load_historical_v2_split(
    root: Path, *, split: str
) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    from src.training import pack_cache
    from src.training.supervised_trainer import SupervisedMicroStep

    binding = HISTORICAL_W0_CACHE_BINDING[split]
    cache_dir = _canonical_nonsymlink(root / binding["fingerprint"], must_exist=True)
    _require_within(root, cache_dir)
    manifest_path = _canonical_nonsymlink(cache_dir / "manifest.json", must_exist=True)
    if sha256_file(manifest_path) != binding["manifest_sha256"]:
        raise Wave5BenchmarkError(
            "historical W0 manifest authentication failed",
            code="wave5.historical_manifest",
        )
    manifest = load_strict_json(manifest_path)
    chunks = manifest.get("chunks")
    if (
        manifest.get("version") != "coordexp-swift-pack-cache-v2"
        or manifest.get("status") != "complete"
        or manifest.get("fingerprint") != binding["fingerprint"]
        or manifest.get("micro_step_count") != binding["micro_step_count"]
        or not isinstance(chunks, list)
        or len(chunks) != 1
    ):
        raise Wave5BenchmarkError(
            "historical W0 manifest is not the authenticated oracle",
            code="wave5.historical_manifest",
        )
    chunk = chunks[0]
    if (
        not isinstance(chunk, Mapping)
        or chunk.get("start") != 0
        or chunk.get("end") != binding["micro_step_count"]
        or chunk.get("count") != binding["micro_step_count"]
        or chunk.get("sha256") != binding["chunk_sha256"]
    ):
        raise Wave5BenchmarkError(
            "historical W0 chunk declaration changed",
            code="wave5.historical_chunk",
        )
    relative = Path(str(chunk.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise Wave5BenchmarkError(
            "historical W0 chunk path is unsafe", code="wave5.historical_chunk"
        )
    chunk_path = _canonical_nonsymlink(cache_dir / relative, must_exist=True)
    _require_within(cache_dir, chunk_path)
    try:
        snapshot, actual_sha256 = pack_cache._read_chunk_snapshot(cache_dir, chunk_path)
        if actual_sha256 != binding["chunk_sha256"]:
            snapshot.close()
            raise Wave5BenchmarkError(
                "historical W0 chunk authentication failed",
                code="wave5.historical_chunk",
            )
        with snapshot:
            decoded = pack_cache._RestrictedCacheUnpickler(snapshot).load()
    except Wave5BenchmarkError:
        raise
    except Exception as exc:
        raise Wave5BenchmarkError(
            "historical W0 chunk decode failed", code="wave5.historical_chunk"
        ) from exc
    if (
        not isinstance(decoded, tuple)
        or len(decoded) != binding["micro_step_count"]
        or not decoded
        or not all(isinstance(step, SupervisedMicroStep) for step in decoded)
    ):
        raise Wave5BenchmarkError(
            "historical W0 decoded payload is invalid",
            code="wave5.historical_payload",
        )
    return manifest, decoded, _stream_semantic_projection(decoded, role="historical_v2")


def _load_current_v3_split(
    root: Path, *, fingerprint: str, expected_count: int
) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    from src.training import pack_cache

    cache_dir = pack_cache.cache_dir_for_fingerprint(root, fingerprint)
    try:
        manifest = pack_cache.load_cache_manifest(
            cache_dir,
            cache_root=root,
            expected_fingerprint=fingerprint,
            level="payloads",
        )
        decoded = pack_cache.load_all_micro_steps_from_cache(
            cache_dir,
            cache_root=root,
            expected_fingerprint=fingerprint,
        )
    except Exception as exc:
        raise Wave5BenchmarkError(
            "current v3 cache public admission failed", code="wave5.current_v3"
        ) from exc
    if len(decoded) != expected_count or not decoded:
        raise Wave5BenchmarkError(
            "current v3 decoded payload count mismatch", code="wave5.current_v3"
        )
    determinants = manifest.get("determinants")
    packing_determinant = (
        determinants.get("packing") if isinstance(determinants, Mapping) else None
    )
    return (
        manifest,
        decoded,
        _stream_semantic_projection(
            decoded,
            role="current_v3",
            packing_determinant=packing_determinant,
        ),
    )


def build_and_publish_compatibility_attestation(
    *,
    historical_cache_root: str | Path,
    cache_root: str | Path,
    train_fingerprint: str,
    eval_fingerprint: str,
    output_path: str | Path,
) -> dict[str, Any]:
    historical_root = _canonical_nonsymlink(historical_cache_root, must_exist=True)
    current_root = _canonical_nonsymlink(cache_root, must_exist=True)
    target = _canonical_nonsymlink(output_path)
    if not all(_is_sha256(value) for value in (train_fingerprint, eval_fingerprint)):
        raise Wave5BenchmarkError(
            "current v3 fingerprints are invalid", code="wave5.current_v3"
        )
    historical = {}
    current = {}
    historical_projections = {}
    current_projections = {}
    for split, fingerprint, count in (
        ("train", train_fingerprint, 32),
        ("eval", eval_fingerprint, 8),
    ):
        old_manifest, old_steps, old_projection = _load_historical_v2_split(
            historical_root, split=split
        )
        new_manifest, new_steps, new_projection = _load_current_v3_split(
            current_root, fingerprint=fingerprint, expected_count=count
        )
        historical_projections[split] = old_projection
        current_projections[split] = new_projection
        historical[split] = {
            "fingerprint": old_manifest["fingerprint"],
            "manifest_sha256": HISTORICAL_W0_CACHE_BINDING[split]["manifest_sha256"],
            "decoded_count": len(old_steps),
            "stream_projection": old_projection,
        }
        real_tokens = sum(
            max(1, int(getattr(step.pack, "length", 1))) for step in new_steps
        )
        capacity_tokens = sum(
            max(
                max(1, int(getattr(step.pack, "length", 1))),
                int(getattr(step.pack, "global_max_length", 1)),
            )
            for step in new_steps
        )
        manifest_file = (
            Path(current_root) / PACK_CACHE_VERSION / fingerprint / "manifest.json"
        )
        current[split] = {
            "format_version": PACK_CACHE_VERSION,
            "fingerprint": fingerprint,
            "manifest_path": str(manifest_file),
            "manifest_sha256": sha256_file(manifest_file),
            "micro_step_count": len(new_steps),
            "pack_stream_sha256": new_projection["projected_semantic_digests"][
                "pack_ids_sha256"
            ],
            "packing_determinant_sha256": new_projection["packing_determinant_sha256"],
            "pack_plan_provenance_sha256": new_projection[
                "pack_plan_provenance_sha256"
            ],
            "real_tokens": real_tokens,
            "capacity_tokens": capacity_tokens,
            "utilization_fraction": real_tokens / capacity_tokens,
        }
    parity = {}
    for field in STREAM_PARITY_FIELDS:
        old_raw = sha256_json(
            {
                split: historical_projections[split]["raw_semantic_digests"][field]
                for split in ("train", "eval")
            }
        )
        new_raw = sha256_json(
            {
                split: current_projections[split]["raw_semantic_digests"][field]
                for split in ("train", "eval")
            }
        )
        old_projected = sha256_json(
            {
                split: historical_projections[split]["projected_semantic_digests"][
                    field
                ]
                for split in ("train", "eval")
            }
        )
        new_projected = sha256_json(
            {
                split: current_projections[split]["projected_semantic_digests"][field]
                for split in ("train", "eval")
            }
        )
        parity[field] = {
            "historical_raw_sha256": old_raw,
            "current_v3_raw_sha256": new_raw,
            "raw_equal": old_raw == new_raw,
            "historical_projected_sha256": old_projected,
            "current_v3_projected_sha256": new_projected,
            "projected_equal": old_projected == new_projected,
        }
    if (
        not all(item["projected_equal"] for item in parity.values())
        or parity["supervision_sha256"]["raw_equal"] is not False
        or any(
            item["raw_equal"] is not True
            for field, item in parity.items()
            if field != "supervision_sha256"
        )
    ):
        raise Wave5BenchmarkError(
            "historical W0 and current v3 semantic payloads drifted",
            code="wave5.stream_parity",
        )
    command = {
        "schema": "coordexp-swift-wave5-compatibility-command-v1",
        "script": _file_identity(SCRIPT_PATH),
        "historical_cache_root": str(historical_root),
        "cache_root": str(current_root),
        "train_fingerprint": train_fingerprint,
        "eval_fingerprint": eval_fingerprint,
        "output_path": str(target),
    }
    command["command_sha256"] = sha256_json(command)
    attestation = _finalize(
        {
            "schema": COMPATIBILITY_SCHEMA,
            "status": "passed",
            "method": ("exact_current_v3_pack_stream_projection_against_w0_receipt"),
            "attested_at": _utc_now(),
            "cache_root": str(current_root),
            "historical_w0_receipt": deepcopy(HISTORICAL_W0_RECEIPT),
            "compatibility_projection": {
                "schema": STREAM_COMPATIBILITY_PROJECTION_SCHEMA,
                "historical_cache_version": "coordexp-swift-pack-cache-v2",
                "current_cache_version": PACK_CACHE_VERSION,
                "historical_required_absent_paths": ["metadata.pack_plan"],
                "current_removed_paths": ["metadata.pack_plan"],
                "all_other_stream_fields": "exact_sha256",
            },
            "command": command,
            "oracle": {
                "historical_v2": historical,
                "current_v3": {
                    split: {
                        "fingerprint": current[split]["fingerprint"],
                        "manifest_sha256": current[split]["manifest_sha256"],
                        "decoded_count": current[split]["micro_step_count"],
                        "stream_projection": current_projections[split],
                    }
                    for split in ("train", "eval")
                },
            },
            "current_v3": current,
            "stream_parity": parity,
        },
        hash_field="attestation_sha256",
    )
    publish_json_absent(target, attestation)
    return attestation


def validate_compatibility_attestation(
    payload: Mapping[str, Any], *, cache_root: str | Path
) -> dict[str, Any]:
    attestation = _validate_finalized(
        payload, schema=COMPATIBILITY_SCHEMA, hash_field="attestation_sha256"
    )
    root = _canonical_nonsymlink(cache_root, must_exist=True)
    if attestation.get("status") != "passed" or attestation.get("method") != (
        "exact_current_v3_pack_stream_projection_against_w0_receipt"
    ):
        raise Wave5BenchmarkError(
            "compatibility attestation did not pass", code="wave5.compatibility"
        )
    _parse_attested_at(attestation.get("attested_at"))
    if attestation.get("cache_root") != str(root):
        raise Wave5BenchmarkError(
            "compatibility cache root mismatch", code="wave5.compatibility"
        )
    if attestation.get("historical_w0_receipt") != HISTORICAL_W0_RECEIPT:
        raise Wave5BenchmarkError(
            "historical W0 receipt mismatch", code="wave5.compatibility"
        )
    if attestation.get("compatibility_projection") != {
        "schema": STREAM_COMPATIBILITY_PROJECTION_SCHEMA,
        "historical_cache_version": "coordexp-swift-pack-cache-v2",
        "current_cache_version": PACK_CACHE_VERSION,
        "historical_required_absent_paths": ["metadata.pack_plan"],
        "current_removed_paths": ["metadata.pack_plan"],
        "all_other_stream_fields": "exact_sha256",
    }:
        raise Wave5BenchmarkError(
            "stream compatibility projection contract changed",
            code="wave5.stream_projection",
        )
    command = attestation.get("command")
    if not isinstance(command, Mapping):
        raise Wave5BenchmarkError(
            "compatibility command binding missing", code="wave5.compatibility"
        )
    command_without_hash = dict(command)
    command_sha256 = command_without_hash.pop("command_sha256", None)
    if (
        set(command_without_hash)
        != {
            "schema",
            "script",
            "historical_cache_root",
            "cache_root",
            "train_fingerprint",
            "eval_fingerprint",
            "output_path",
        }
        or command_without_hash.get("schema")
        != "coordexp-swift-wave5-compatibility-command-v1"
        or command_without_hash.get("script") != _file_identity(SCRIPT_PATH)
        or command_without_hash.get("cache_root") != str(root)
        or not _is_sha256(command_without_hash.get("train_fingerprint"))
        or not _is_sha256(command_without_hash.get("eval_fingerprint"))
        or not isinstance(command_without_hash.get("output_path"), str)
        or command_sha256 != sha256_json(command_without_hash)
    ):
        raise Wave5BenchmarkError(
            "compatibility command authentication failed",
            code="wave5.compatibility_command",
        )
    historical_root = _canonical_nonsymlink(
        command_without_hash.get("historical_cache_root", ""), must_exist=True
    )
    oracle = attestation.get("oracle")
    if not isinstance(oracle, Mapping) or set(oracle) != {
        "historical_v2",
        "current_v3",
    }:
        raise Wave5BenchmarkError(
            "compatibility oracle binding missing", code="wave5.compatibility"
        )
    current = attestation.get("current_v3")
    if (
        not isinstance(current, Mapping)
        or set(current) != {"train", "eval"}
        or not isinstance(oracle["historical_v2"], Mapping)
        or set(oracle["historical_v2"]) != {"train", "eval"}
        or not isinstance(oracle["current_v3"], Mapping)
        or set(oracle["current_v3"]) != {"train", "eval"}
    ):
        raise Wave5BenchmarkError(
            "compatibility cache or oracle binding incomplete",
            code="wave5.compatibility",
        )
    historical_projections: dict[str, dict[str, Any]] = {}
    current_projections: dict[str, dict[str, Any]] = {}
    current_binding_fields = {
        "format_version",
        "fingerprint",
        "manifest_path",
        "manifest_sha256",
        "micro_step_count",
        "pack_stream_sha256",
        "packing_determinant_sha256",
        "pack_plan_provenance_sha256",
        "real_tokens",
        "capacity_tokens",
        "utilization_fraction",
    }
    for split, historical_count, fingerprint_key in (
        ("train", 32, "train_fingerprint"),
        ("eval", 8, "eval_fingerprint"),
    ):
        fingerprint = command_without_hash[fingerprint_key]
        item = current[split]
        if not isinstance(item, Mapping) or set(item) != current_binding_fields:
            raise Wave5BenchmarkError(
                "v3 cache binding fields incomplete", code="wave5.cache"
            )
        if item.get("format_version") != PACK_CACHE_VERSION:
            raise Wave5BenchmarkError(
                "retired or invalid cache binding", code="wave5.cache_version"
            )
        if item.get("fingerprint") != fingerprint:
            raise Wave5BenchmarkError(
                "v3 cache fingerprint binding changed", code="wave5.cache"
            )
        old_manifest, old_steps, old_projection = _load_historical_v2_split(
            historical_root, split=split
        )
        new_manifest, new_steps, new_projection = _load_current_v3_split(
            root, fingerprint=fingerprint, expected_count=historical_count
        )
        historical_projections[split] = old_projection
        current_projections[split] = new_projection
        manifest_path = (
            root / PACK_CACHE_VERSION / fingerprint / "manifest.json"
        ).resolve()
        manifest_sha256 = sha256_file(manifest_path)
        real_tokens = sum(
            max(1, int(getattr(step.pack, "length", 1))) for step in new_steps
        )
        capacity_tokens = sum(
            max(
                max(1, int(getattr(step.pack, "length", 1))),
                int(getattr(step.pack, "global_max_length", 1)),
            )
            for step in new_steps
        )
        expected_current = {
            "format_version": PACK_CACHE_VERSION,
            "fingerprint": fingerprint,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "micro_step_count": len(new_steps),
            "pack_stream_sha256": new_projection["projected_semantic_digests"][
                "pack_ids_sha256"
            ],
            "packing_determinant_sha256": new_projection["packing_determinant_sha256"],
            "pack_plan_provenance_sha256": new_projection[
                "pack_plan_provenance_sha256"
            ],
            "real_tokens": real_tokens,
            "capacity_tokens": capacity_tokens,
            "utilization_fraction": real_tokens / capacity_tokens,
        }
        expected_historical_oracle = {
            "fingerprint": old_manifest["fingerprint"],
            "manifest_sha256": HISTORICAL_W0_CACHE_BINDING[split]["manifest_sha256"],
            "decoded_count": len(old_steps),
            "stream_projection": old_projection,
        }
        expected_current_oracle = {
            "fingerprint": fingerprint,
            "manifest_sha256": manifest_sha256,
            "decoded_count": len(new_steps),
            "stream_projection": new_projection,
        }
        if (
            dict(current[split]) != expected_current
            or oracle["historical_v2"][split] != expected_historical_oracle
            or oracle["current_v3"][split] != expected_current_oracle
            or new_manifest.get("micro_step_count") != historical_count
        ):
            raise Wave5BenchmarkError(
                "compatibility oracle identity invalid", code="wave5.compatibility"
            )
    expected_parity = {}
    for field in STREAM_PARITY_FIELDS:
        historical_raw = sha256_json(
            {
                split: historical_projections[split]["raw_semantic_digests"][field]
                for split in ("train", "eval")
            }
        )
        current_raw = sha256_json(
            {
                split: current_projections[split]["raw_semantic_digests"][field]
                for split in ("train", "eval")
            }
        )
        historical_projected = sha256_json(
            {
                split: historical_projections[split]["projected_semantic_digests"][
                    field
                ]
                for split in ("train", "eval")
            }
        )
        current_projected = sha256_json(
            {
                split: current_projections[split]["projected_semantic_digests"][field]
                for split in ("train", "eval")
            }
        )
        expected_parity[field] = {
            "historical_raw_sha256": historical_raw,
            "current_v3_raw_sha256": current_raw,
            "raw_equal": historical_raw == current_raw,
            "historical_projected_sha256": historical_projected,
            "current_v3_projected_sha256": current_projected,
            "projected_equal": historical_projected == current_projected,
        }
    parity = attestation.get("stream_parity")
    if (
        parity != expected_parity
        or not all(item["projected_equal"] for item in expected_parity.values())
        or expected_parity["supervision_sha256"]["raw_equal"] is not False
        or any(
            item["raw_equal"] is not True
            for field, item in expected_parity.items()
            if field != "supervision_sha256"
        )
    ):
        raise Wave5BenchmarkError(
            "W0/current-v3 stream parity failed", code="wave5.stream_parity"
        )
    return attestation


def _file_identity(path: Path) -> dict[str, Any]:
    checked = _canonical_nonsymlink(path, must_exist=True)
    return {"path": str(checked), "sha256": sha256_file(checked)}


def _arm_key(triad_index: int, arm: str) -> str:
    order = TRIAD_ORDERS[triad_index]
    return f"triad-{triad_index + 1}-position-{order.index(arm) + 1}-arm-{arm}"


def _targets_for_root(root: Path) -> dict[str, Any]:
    arms: dict[str, dict[str, str]] = {}
    for triad_index, order in enumerate(TRIAD_ORDERS):
        for arm in order:
            directory = root / "arms" / _arm_key(triad_index, arm)
            arms[_arm_key(triad_index, arm)] = {
                "directory": str(directory),
                "config": str(directory / "arm.yaml"),
                "output_root": str(directory / "output"),
                "run_dir": str(directory / "output" / "train"),
                "resource_samples": str(directory / "resource-samples.json"),
                "stdout": str(directory / "stdout.log"),
                "stderr": str(directory / "stderr.log"),
                "executed_streams": str(directory / "executed-streams"),
                "observation": str(directory / "observation.json"),
            }
    return {
        "plan": str(root / "plan.json"),
        "attempt": str(root / "attempt.json"),
        "gpu_baseline": str(root / "gpu-execution-baseline.json"),
        "terminal": str(root / "terminal.json"),
        "terminal_publication_failure": str(root / "terminal-publication-failure.json"),
        "decision": str(root / "decision.json"),
        "arms": arms,
    }


def _all_target_paths(targets: Mapping[str, Any]) -> list[Path]:
    result = [
        Path(targets[name])
        for name in (
            "plan",
            "attempt",
            "gpu_baseline",
            "terminal",
            "terminal_publication_failure",
            "decision",
        )
    ]
    arms = targets["arms"]
    for item in arms.values():
        result.extend(Path(value) for value in item.values())
    return result


def _gpu_execution_contract(policy: str) -> dict[str, Any]:
    if policy == "idle_promotional":
        return {
            "policy": policy,
            "baseline_sample_count": GPU_IDLE_STABLE_SAMPLE_COUNT,
            "baseline_sample_interval_seconds": GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
            "expected_total_memory_bytes_per_gpu": GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "baseline_max_used_bytes_per_gpu": GPU_IDLE_MEMORY_LIMIT_BYTES,
            "minimum_headroom_bytes_per_gpu": (
                GPU_EXPECTED_TOTAL_MEMORY_BYTES - GPU_IDLE_MEMORY_LIMIT_BYTES
            ),
            "post_phase_sample_count": GPU_POST_PHASE_SAMPLE_COUNT,
            "post_phase_sample_interval_seconds": (
                GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS
            ),
            "utilization_semantics": "idle_gate_at_or_below_5_percent",
            "performance_promotion_eligible": True,
            "claim_scope": "matched_idle_semantic_and_performance_evidence",
        }
    if policy == "shared_nonpromotional":
        return {
            "policy": policy,
            "baseline_sample_count": GPU_IDLE_STABLE_SAMPLE_COUNT,
            "baseline_sample_interval_seconds": GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
            "expected_total_memory_bytes_per_gpu": GPU_EXPECTED_TOTAL_MEMORY_BYTES,
            "baseline_max_used_bytes_per_gpu": SHARED_GPU_BASELINE_MAX_USED_BYTES,
            "minimum_headroom_bytes_per_gpu": GPU_MINIMUM_HEADROOM_BYTES,
            "post_phase_sample_count": GPU_POST_PHASE_SAMPLE_COUNT,
            "post_phase_sample_interval_seconds": (
                GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS
            ),
            "utilization_semantics": "observational_only",
            "performance_promotion_eligible": False,
            "claim_scope": "correctness_plumbing_and_failure_semantics_only",
        }
    raise Wave5BenchmarkError(
        "unsupported GPU execution policy", code="wave5.gpu_execution_policy"
    )


def build_frozen_plan(
    *,
    root: str | Path,
    cache_root: str | Path,
    compatibility_attestation_path: str | Path,
    gpu_execution_policy: str = "idle_promotional",
) -> dict[str, Any]:
    if gpu_execution_policy not in GPU_EXECUTION_POLICIES:
        raise Wave5BenchmarkError(
            "unsupported GPU execution policy", code="wave5.gpu_execution_policy"
        )
    exact_root = _canonical_nonsymlink(root, must_exist=True)
    exact_cache_root = _canonical_nonsymlink(cache_root, must_exist=True)
    attestation_path = _canonical_nonsymlink(
        compatibility_attestation_path, must_exist=True
    )
    attestation = validate_compatibility_attestation(
        load_strict_json(attestation_path), cache_root=exact_cache_root
    )
    if attestation["command"]["output_path"] != str(attestation_path):
        raise Wave5BenchmarkError(
            "compatibility output path is not command-bound",
            code="wave5.compatibility_command",
        )
    targets = _targets_for_root(exact_root)
    _validate_distinct_paths(exact_root, _all_target_paths(targets))
    config_compatibility = _config_compatibility_contract()
    current_weights = _current_base_model_weight_identity()
    if (
        current_weights.get("aggregate_sha256")
        != HISTORICAL_W0_BASE_WEIGHT_AGGREGATE_SHA256
    ):
        raise Wave5BenchmarkError(
            "current base-model weight identity differs from historical W0",
            code="wave5.weight_identity",
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "prepared",
        "root": str(exact_root),
        "arms": deepcopy(ARM_SPECS),
        "triad_orders": [list(order) for order in TRIAD_ORDERS],
        "w0_binding": {
            "historical_receipt": deepcopy(HISTORICAL_W0_RECEIPT),
            "current_v3": deepcopy(attestation["current_v3"]),
            "stream_parity": deepcopy(attestation["stream_parity"]),
            "compatibility_attestation_path": str(attestation_path),
            "compatibility_attestation_sha256": attestation["attestation_sha256"],
        },
        "execution_baseline": _execution_baseline(),
        "config_compatibility": config_compatibility,
        "base_model_weight_identity": current_weights,
        "historical_w0_base_weight_aggregate_sha256": (
            HISTORICAL_W0_BASE_WEIGHT_AGGREGATE_SHA256
        ),
        "typed_config_recipe": {
            "base_config": _file_identity(BASE_CONFIG_PATH),
            "train_entrypoint": _file_identity(TRAIN_PATH),
            "allowed_overrides": [
                "run.name",
                "run.artifact_root",
                "run.collision_policy",
                "training.forward_input_provider_mode",
            ],
            "world_size": WORLD_SIZE,
            "max_steps": 5,
            "eval_steps": [3],
            "checkpoint_steps": [5],
        },
        "runner": {
            "python": _file_identity(Path(sys.executable).resolve()),
            "script": _file_identity(SCRIPT_PATH),
            "subcommand": "arm",
            "argv_template": [
                "{python}",
                "{script}",
                "arm",
                "--plan",
                "{plan}",
                "--attempt-marker",
                "{attempt}",
                "--triad-index",
                "{triad_index}",
                "--arm",
                "{arm}",
            ],
            "shell": False,
        },
        "measurement_contract": _measurement_contract(),
        "semantic_contract": {
            "parity_fields": list(SEMANTIC_PARITY_FIELDS),
            "equality": "exact_sha256",
            "promotion_requires_all_parity": True,
            "eval_reduction": deepcopy(EXPECTED_EVAL_REDUCTION),
        },
        "resource_contract": {
            "host_rss_per_rank_ceiling_bytes": HOST_RSS_PER_RANK_CEILING_BYTES,
            "host_rss_all_rank_ceiling_bytes": HOST_RSS_ALL_RANK_CEILING_BYTES,
            "gpu_memory_ceiling_bytes": GPU_MEMORY_CEILING_BYTES,
            "artifact_ceiling_bytes": ARTIFACT_CEILING_BYTES,
            "scope": (
                "all_train_and_eval_ranks_plus_arm_process_lifetime_gpu_samples_"
                "with_compute_process_tree_ownership_and_complete_output_tree"
            ),
        },
        "gpu_execution_contract": _gpu_execution_contract(gpu_execution_policy),
        "artifact_targets": targets,
    }
    return _finalize(plan, hash_field="plan_sha256")


def validate_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = _validate_finalized(payload, schema=PLAN_SCHEMA, hash_field="plan_sha256")
    root = _canonical_nonsymlink(plan.get("root", ""), must_exist=True)
    expected_targets = _targets_for_root(root)
    if plan.get("artifact_targets") != expected_targets:
        raise Wave5BenchmarkError(
            "artifact targets are not canonical", code="wave5.targets"
        )
    _validate_distinct_paths(root, _all_target_paths(expected_targets))
    runner = plan.get("runner")
    if (
        not isinstance(runner, Mapping)
        or runner.get("argv_template")
        != [
            "{python}",
            "{script}",
            "arm",
            "--plan",
            "{plan}",
            "--attempt-marker",
            "{attempt}",
            "--triad-index",
            "{triad_index}",
            "--arm",
            "{arm}",
        ]
        or runner.get("shell") is not False
    ):
        raise Wave5BenchmarkError(
            "arbitrary runner command rejected", code="wave5.command"
        )
    if runner.get("python") != _file_identity(
        Path(sys.executable).resolve()
    ) or runner.get("script") != _file_identity(SCRIPT_PATH):
        raise Wave5BenchmarkError(
            "runner executable identity mismatch", code="wave5.command_identity"
        )
    recipe = plan.get("typed_config_recipe")
    if (
        not isinstance(recipe, Mapping)
        or recipe.get("base_config") != _file_identity(BASE_CONFIG_PATH)
        or recipe.get("train_entrypoint") != _file_identity(TRAIN_PATH)
    ):
        raise Wave5BenchmarkError(
            "production config or train identity mismatch",
            code="wave5.production_identity",
        )
    if plan.get("execution_baseline") != _execution_baseline():
        raise Wave5BenchmarkError(
            "execution baseline changed", code="wave5.execution_baseline"
        )
    _revalidate_plan_compatibility(plan)
    binding = plan.get("w0_binding")
    if not isinstance(binding, Mapping):
        raise Wave5BenchmarkError("W0 binding missing", code="wave5.compatibility")
    attestation_path = _canonical_nonsymlink(
        binding.get("compatibility_attestation_path", ""), must_exist=True
    )
    attestation = validate_compatibility_attestation(
        load_strict_json(attestation_path),
        cache_root=Path(binding["current_v3"]["train"]["manifest_path"]).parents[2],
    )
    if binding != {
        "historical_receipt": HISTORICAL_W0_RECEIPT,
        "current_v3": attestation["current_v3"],
        "stream_parity": attestation["stream_parity"],
        "compatibility_attestation_path": str(attestation_path),
        "compatibility_attestation_sha256": attestation["attestation_sha256"],
    }:
        raise Wave5BenchmarkError(
            "W0 compatibility binding changed", code="wave5.compatibility"
        )
    if plan.get("arms") != ARM_SPECS or plan.get("triad_orders") != [
        list(order) for order in TRIAD_ORDERS
    ]:
        raise Wave5BenchmarkError(
            "arm or ordering contract changed", code="wave5.order"
        )
    expected = build_frozen_plan(
        root=root,
        cache_root=Path(attestation["cache_root"]),
        compatibility_attestation_path=attestation_path,
        gpu_execution_policy=plan.get("gpu_execution_contract", {}).get("policy"),
    )
    if plan != expected:
        raise Wave5BenchmarkError(
            "plan differs from frozen contract", code="wave5.plan"
        )
    return plan


def _arm_targets(plan: Mapping[str, Any], triad_index: int, arm: str) -> dict[str, str]:
    if not 0 <= triad_index < len(TRIAD_ORDERS) or arm not in TRIAD_ORDERS[triad_index]:
        raise Wave5BenchmarkError("invalid arm coordinates", code="wave5.arm")
    return dict(plan["artifact_targets"]["arms"][_arm_key(triad_index, arm)])


def _arm_argv(plan: Mapping[str, Any], triad_index: int, arm: str) -> list[str]:
    targets = plan["artifact_targets"]
    return [
        plan["runner"]["python"]["path"],
        plan["runner"]["script"]["path"],
        "arm",
        "--plan",
        targets["plan"],
        "--attempt-marker",
        targets["attempt"],
        "--triad-index",
        str(triad_index),
        "--arm",
        arm,
    ]


def _production_argv(plan: Mapping[str, Any], triad_index: int, arm: str) -> list[str]:
    targets = _arm_targets(plan, triad_index, arm)
    measurement = plan["measurement_contract"]
    return [
        plan["runner"]["python"]["path"],
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(WORLD_SIZE),
        plan["runner"]["script"]["path"],
        "instrumented-train",
        "--config",
        targets["config"],
        "--stream-output-root",
        targets["executed_streams"],
        "--comparison-arm",
        arm,
        "--warmup-exclusion-steps",
        str(measurement["warmup_exclusion_steps"]),
        "--wall-clock-scope",
        measurement["wall_clock_scope"],
        "--profile-sync-timings",
        "disabled-default",
        "--measurement-contract-sha256",
        sha256_json(measurement),
    ]


def build_attempt(plan: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_plan(plan)
    commands = [
        {
            "triad_index": index,
            "arm": arm,
            "argv": _arm_argv(checked, index, arm),
            "argv_sha256": sha256_json(_arm_argv(checked, index, arm)),
        }
        for index, order in enumerate(TRIAD_ORDERS)
        for arm in order
    ]
    return _finalize(
        {
            "schema": ATTEMPT_SCHEMA,
            "status": "consumed_no_retry",
            "created_at": _utc_now(),
            "plan_path": checked["artifact_targets"]["plan"],
            "plan_sha256": checked["plan_sha256"],
            "attempt_path": checked["artifact_targets"]["attempt"],
            "execution_baseline": deepcopy(checked["execution_baseline"]),
            "gpu_execution_contract": deepcopy(checked["gpu_execution_contract"]),
            "measurement_contract": deepcopy(checked["measurement_contract"]),
            "config_compatibility_sha256": sha256_json(checked["config_compatibility"]),
            "base_model_weight_identity_sha256": checked["base_model_weight_identity"][
                "aggregate_sha256"
            ],
            "runner": deepcopy(checked["runner"]),
            "commands": commands,
        },
        hash_field="attempt_sha256",
    )


def load_authenticated_attempt(
    plan: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    checked = validate_plan(plan)
    actual_path = _canonical_nonsymlink(path, must_exist=True)
    if str(actual_path) != checked["artifact_targets"]["attempt"]:
        raise Wave5BenchmarkError(
            "attempt marker path mismatch", code="wave5.attempt_path"
        )
    attempt = _validate_finalized(
        load_strict_json(actual_path),
        schema=ATTEMPT_SCHEMA,
        hash_field="attempt_sha256",
    )
    expected = build_attempt(checked)
    expected_fixed = {
        key: value
        for key, value in expected.items()
        if key not in {"created_at", "attempt_sha256"}
    }
    actual_fixed = {
        key: value
        for key, value in attempt.items()
        if key not in {"created_at", "attempt_sha256"}
    }
    if not isinstance(attempt.get("created_at"), str) or actual_fixed != expected_fixed:
        raise Wave5BenchmarkError("forged attempt marker", code="wave5.attempt")
    return attempt


def _write_arm_config(plan: Mapping[str, Any], triad_index: int, arm: str) -> Path:
    import yaml
    from src.config.loader import load_train_config

    targets = _arm_targets(plan, triad_index, arm)
    config_path = Path(targets["config"])
    if config_path.exists() or Path(targets["directory"]).exists():
        raise Wave5BenchmarkError(
            "arm artifact collision", code="wave5.artifact_collision"
        )
    config_path.parent.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "extends": str(BASE_CONFIG_PATH),
        "run": {
            "name": "train",
            "artifact_root": targets["output_root"],
            "collision_policy": "fail",
        },
        "training": {
            "forward_input_provider_mode": ARM_SPECS[arm]["provider_mode"],
            "max_steps": 5,
        },
        "eval": {"forward": {"steps": [3]}},
        "checkpoint": {"steps": [5]},
    }
    with config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
    resolved = load_train_config(config_path)
    cfg = resolved.config
    if (
        cfg.training.forward_input_provider_mode != ARM_SPECS[arm]["provider_mode"]
        or cfg.training.max_steps != 5
        or list(cfg.eval.forward.steps) != [3]
        or list(cfg.checkpoint.steps) != [5]
        or str(cfg.run.artifact_root) != targets["output_root"]
    ):
        raise Wave5BenchmarkError(
            "typed arm config did not resolve exactly", code="wave5.arm_config"
        )
    if (
        _project_arm_config(resolved.config_dict)
        != plan["config_compatibility"]["current_parent_projection"]
    ):
        raise Wave5BenchmarkError(
            "arm config compatibility projection changed",
            code="wave5.config_compatibility",
        )
    return config_path


def _executed_stream_event(
    *,
    micro_step: Any,
    forward_inputs: Any,
    planned_step_id: int,
    local_micro_step_index: int,
) -> dict[str, Any]:
    tensor_values, tensor_metadata = _tensor_streams(
        forward_inputs, path="forward_inputs"
    )
    return {
        "planned_step_id": planned_step_id,
        "local_micro_step_index": local_micro_step_index,
        "pack": _pack_identity(micro_step),
        "tensor_values": tensor_values,
        "tensor_metadata": tensor_metadata,
        "supervision": _semantic_json_value(
            {
                "example_ids": [
                    getattr(example, "example_id", str(index))
                    for index, example in enumerate(micro_step.encoded_examples)
                ],
                "token_sequence": micro_step.token_sequence,
                "vocab_groups": micro_step.vocab_groups,
                "metadata": micro_step.metadata,
            }
        ),
        "position_ids": _semantic_json_value(
            {
                "position_inputs": micro_step.position_inputs,
                "executed_position_ids": forward_inputs.position_ids,
            }
        ),
        "loss_inputs": _semantic_json_value(
            {
                "token_sequence": micro_step.token_sequence,
                "vocab_groups": micro_step.vocab_groups,
                "expected_vocab_size": micro_step.expected_vocab_size,
                "extra_model_kwargs": micro_step.extra_model_kwargs,
                "fa2_model_dtype": micro_step.fa2_model_dtype,
                "capture_fa2_branch": micro_step.capture_fa2_branch,
                "require_fa2_branch_proof": micro_step.require_fa2_branch_proof,
                "fa2_branch_proof_policy": micro_step.fa2_branch_proof_policy,
                "logits_to_keep": forward_inputs.logits_to_keep,
            }
        ),
    }


class _ExecutedStreamRecorder:
    def __init__(self, *, rank: int, provider_mode: str) -> None:
        self.rank = rank
        self.provider_mode = provider_mode
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        *,
        micro_step: Any,
        forward_inputs: Any,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> None:
        self.events.append(
            _executed_stream_event(
                micro_step=micro_step,
                forward_inputs=forward_inputs,
                planned_step_id=planned_step_id,
                local_micro_step_index=local_micro_step_index,
            )
        )

    def receipt(
        self, *, status: str, error: Mapping[str, str] | None
    ) -> dict[str, Any]:
        return _build_rank_stream_receipt(
            rank=self.rank,
            provider_mode=self.provider_mode,
            events=self.events,
            status=status,
            error=error,
        )


def _install_executed_stream_instrumentation(
    recorder: _ExecutedStreamRecorder,
) -> None:
    from src.training import forward_input_provider as provider_module
    from src.training import supervised_trainer as trainer_module

    context = threading.local()
    original_streaming_step = (
        trainer_module.SupervisedTrainer._run_streaming_planned_step
    )

    def instrumented_streaming_step(self: Any, **kwargs: Any) -> Any:
        context.planned_step_id = int(kwargs["planned_step_id"])
        context.local_micro_step_index = 0
        try:
            return original_streaming_step(self, **kwargs)
        finally:
            context.planned_step_id = None
            context.local_micro_step_index = None

    trainer_module.SupervisedTrainer._run_streaming_planned_step = (
        instrumented_streaming_step
    )

    for provider_class in (
        provider_module.SynchronousForwardInputProvider,
        provider_module.OverlappedForwardInputProvider,
    ):
        original_begin = provider_class.begin_planned_step
        original_take = provider_class.take

        def instrumented_begin(
            self: Any,
            planned_step_id: int,
            moved_micro_steps: Sequence[Any],
            *,
            _original: Any = original_begin,
        ) -> None:
            self._wave5_planned_step_id = int(planned_step_id)
            _original(self, planned_step_id, moved_micro_steps)

        def instrumented_take(
            self: Any,
            ordinal: int,
            micro_step: Any,
            *,
            _original: Any = original_take,
        ) -> Any:
            forward_inputs = _original(self, ordinal, micro_step)
            recorder.record(
                micro_step=micro_step,
                forward_inputs=forward_inputs,
                planned_step_id=int(self._wave5_planned_step_id),
                local_micro_step_index=int(ordinal),
            )
            return forward_inputs

        provider_class.begin_planned_step = instrumented_begin
        provider_class.take = instrumented_take

    original_default_forward = trainer_module._default_qwen_forward
    original_build = trainer_module.build_qwen_forward_inputs

    def instrumented_default_forward(model: Any, micro_step: Any) -> Any:
        context.micro_step = micro_step
        try:
            return original_default_forward(model, micro_step)
        finally:
            context.micro_step = None

    def instrumented_build(*args: Any, **kwargs: Any) -> Any:
        forward_inputs = original_build(*args, **kwargs)
        micro_step = getattr(context, "micro_step", None)
        planned_step_id = getattr(context, "planned_step_id", None)
        local_index = getattr(context, "local_micro_step_index", None)
        if (
            micro_step is not None
            and planned_step_id is not None
            and local_index is not None
        ):
            recorder.record(
                micro_step=micro_step,
                forward_inputs=forward_inputs,
                planned_step_id=int(planned_step_id),
                local_micro_step_index=int(local_index),
            )
            context.local_micro_step_index = int(local_index) + 1
        return forward_inputs

    trainer_module._default_qwen_forward = instrumented_default_forward
    trainer_module.build_qwen_forward_inputs = instrumented_build


def run_instrumented_train(
    config_path: str | Path,
    stream_output_root: str | Path,
    *,
    comparison_arm: str,
    warmup_exclusion_steps: int,
    wall_clock_scope: str,
    profile_sync_timings: str,
    measurement_contract_sha256: str,
) -> int:
    from src.config.loader import load_train_config
    from src.train import main as train_main
    from src.training.pipeline import run_training_pipeline

    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Wave5BenchmarkError(
            "instrumented rank identity unavailable", code="wave5.executed_stream"
        ) from exc
    if rank not in range(WORLD_SIZE) or world_size != WORLD_SIZE:
        raise Wave5BenchmarkError(
            "instrumented rank identity mismatch", code="wave5.executed_stream"
        )
    config = load_train_config(config_path).config
    provider_mode = str(config.training.forward_input_provider_mode)
    if (
        comparison_arm not in ARM_SPECS
        or ARM_SPECS[comparison_arm]["provider_mode"] != provider_mode
        or warmup_exclusion_steps != len(WARMUP_STEPS)
        or wall_clock_scope != MEASUREMENT_WALL_CLOCK_SCOPE
        or profile_sync_timings != "disabled-default"
        or measurement_contract_sha256 != sha256_json(_measurement_contract())
        or os.environ.get("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS") is not None
        or os.environ.get("COORDEXP_SWIFT_EVAL_REDUCTION_MODE") is not None
    ):
        raise Wave5BenchmarkError(
            "instrumented measurement context mismatch",
            code="wave5.measurement_context",
        )
    measurement_context = {
        "comparison_arm": comparison_arm,
        "wall_clock_scope": wall_clock_scope,
        "warmup_exclusion_steps": warmup_exclusion_steps,
    }
    recorder = _ExecutedStreamRecorder(rank=rank, provider_mode=provider_mode)
    _install_executed_stream_instrumentation(recorder)
    error: dict[str, str] | None = None
    try:
        return train_main(
            ["--config", str(config_path)],
            runner=lambda path: run_training_pipeline(
                path, measurement_context=measurement_context
            ),
        )
    except BaseException as exc:
        error = {
            "type": type(exc).__name__[:128],
            "code": str(getattr(exc, "code", "wave5.instrumented_train"))[:128],
            "message": str(exc)[:4096],
        }
        raise
    finally:
        receipt = recorder.receipt(
            status="complete" if error is None else "failed", error=error
        )
        target = Path(stream_output_root) / f"rank-{rank}.json"
        publish_json_absent(target, receipt)


def _sample_nvidia() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if (
            len(fields) == 5
            and fields[0].isdigit()
            and fields[1]
            and fields[2].isdigit()
            and fields[3].isdigit()
            and fields[4].isdigit()
        ):
            rows.append(
                {
                    "gpu_index": int(fields[0]),
                    "gpu_uuid": fields[1],
                    "memory_total_bytes": int(fields[2]) * 1024**2,
                    "memory_used_bytes": int(fields[3]) * 1024**2,
                    "memory_headroom_bytes": (int(fields[2]) - int(fields[3]))
                    * 1024**2,
                    "utilization_percent": int(fields[4]),
                }
            )
    return rows


def _sample_nvidia_compute_apps() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Wave5BenchmarkError(
            "GPU compute-process inventory unavailable",
            code="wave5.gpu_process_sample",
        ) from exc
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if (
            len(fields) != 3
            or not fields[0]
            or not fields[1].isdigit()
            or int(fields[1]) <= 0
            or not fields[2]
        ):
            raise Wave5BenchmarkError(
                "GPU compute-process inventory is malformed",
                code="wave5.gpu_process_sample",
            )
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2][:4096],
            }
        )
    return sorted(
        rows, key=lambda row: (row["gpu_uuid"], row["pid"], row["process_name"])
    )


def _validate_gpu_compute_apps(rows: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise Wave5BenchmarkError(
            f"{field} is not a process list", code="wave5.gpu_process_sample"
        )
    validated = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"gpu_uuid", "pid", "process_name"}
            or not isinstance(row.get("gpu_uuid"), str)
            or not row["gpu_uuid"]
            or isinstance(row.get("pid"), bool)
            or not isinstance(row.get("pid"), int)
            or row["pid"] <= 0
            or not isinstance(row.get("process_name"), str)
            or not row["process_name"]
            or len(row["process_name"]) > 4096
        ):
            raise Wave5BenchmarkError(
                f"{field} contains a malformed process row",
                code="wave5.gpu_process_sample",
            )
        validated.append(
            {
                "gpu_uuid": row["gpu_uuid"],
                "pid": row["pid"],
                "process_name": row["process_name"],
            }
        )
    return sorted(
        validated,
        key=lambda row: (row["gpu_uuid"], row["pid"], row["process_name"]),
    )


def _validate_gpu_sample(rows: Any, *, field: str) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, list)
        or len(rows) != WORLD_SIZE
        or any(not isinstance(row, Mapping) for row in rows)
        or any(
            set(row)
            != {
                "gpu_index",
                "gpu_uuid",
                "memory_total_bytes",
                "memory_used_bytes",
                "memory_headroom_bytes",
                "utilization_percent",
            }
            for row in rows
        )
        or any(
            isinstance(row.get("gpu_index"), bool)
            or not isinstance(row.get("gpu_index"), int)
            for row in rows
        )
        or [row.get("gpu_index") for row in rows] != list(range(WORLD_SIZE))
        or any(
            not isinstance(row.get("gpu_uuid"), str) or not row["gpu_uuid"]
            for row in rows
        )
        or len({row["gpu_uuid"] for row in rows}) != WORLD_SIZE
    ):
        raise Wave5BenchmarkError(
            f"{field} does not cover exactly eight GPUs", code="wave5.gpu_sample"
        )
    validated = []
    for row in rows:
        integral_fields = (
            "memory_total_bytes",
            "memory_used_bytes",
            "memory_headroom_bytes",
            "utilization_percent",
        )
        if any(
            isinstance(row.get(name), bool)
            or not isinstance(row.get(name), int)
            or row[name] < 0
            for name in integral_fields
        ):
            raise Wave5BenchmarkError(
                f"{field} contains a nonintegral NVIDIA field",
                code="wave5.gpu_sample",
            )
        total = row["memory_total_bytes"]
        memory = row["memory_used_bytes"]
        headroom = row["memory_headroom_bytes"]
        utilization = row["utilization_percent"]
        if (
            total != GPU_EXPECTED_TOTAL_MEMORY_BYTES
            or memory > total
            or headroom != total - memory
            or utilization > 100
        ):
            raise Wave5BenchmarkError(
                f"{field} GPU memory geometry is invalid", code="wave5.gpu_sample"
            )
        validated.append(
            {
                "gpu_index": row["gpu_index"],
                "gpu_uuid": str(row["gpu_uuid"]),
                "memory_total_bytes": total,
                "memory_used_bytes": memory,
                "memory_headroom_bytes": headroom,
                "utilization_percent": utilization,
            }
        )
    return validated


def _validate_monotonic_ns(value: Any, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_MONOTONIC_NS
    ):
        raise Wave5BenchmarkError(
            f"{field} is not a bounded monotonic timestamp",
            code="wave5.sample_timing",
        )
    return value


def _validate_sample_timing(
    samples: Any,
    *,
    expected_count: int,
    interval_seconds: float,
    field: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(samples, list) or len(samples) != expected_count:
        raise Wave5BenchmarkError(
            f"{field} does not contain exactly {expected_count} samples",
            code="wave5.sample_timing",
        )
    timestamps = [
        _validate_monotonic_ns(sample.get("monotonic_ns"), field=f"{field}[{index}]")
        if isinstance(sample, Mapping)
        else _validate_monotonic_ns(None, field=f"{field}[{index}]")
        for index, sample in enumerate(samples)
    ]
    minimum_delta_ns = int(interval_seconds * 1_000_000_000)
    if any(
        timestamps[index] - timestamps[index - 1] < minimum_delta_ns
        for index in range(1, len(timestamps))
    ):
        raise Wave5BenchmarkError(
            f"{field} samples are too close together",
            code="wave5.sample_timing",
        )
    return samples


def _gpu_inventory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"gpu_index": int(row["gpu_index"]), "gpu_uuid": str(row["gpu_uuid"])}
        for row in rows
    ]


def _compute_app_identities(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            {"gpu_uuid": str(row["gpu_uuid"]), "driver_pid": int(row["pid"])}
            for row in rows
        ],
        key=lambda row: (row["gpu_uuid"], row["driver_pid"]),
    )


def _stable_gpu_samples_for_policy(policy: str) -> dict[str, Any]:
    contract = _gpu_execution_contract(policy)
    samples = []
    gpu_inventory: list[dict[str, Any]] | None = None
    baseline_compute_apps: list[dict[str, Any]] | None = None
    baseline_compute_app_identities: list[dict[str, Any]] | None = None
    for sample_index in range(GPU_IDLE_STABLE_SAMPLE_COUNT):
        monotonic_ns = time.monotonic_ns()
        rows = _validate_gpu_sample(
            _sample_nvidia(), field=f"gpu_execution_baseline[{sample_index}]"
        )
        compute_apps = _validate_gpu_compute_apps(
            _sample_nvidia_compute_apps(),
            field=f"gpu_execution_baseline[{sample_index}].compute_apps",
        )
        if any(
            row["memory_used_bytes"] > contract["baseline_max_used_bytes_per_gpu"]
            or row["memory_headroom_bytes"] < contract["minimum_headroom_bytes_per_gpu"]
            for row in rows
        ) or (
            policy == "idle_promotional"
            and (
                compute_apps
                or any(
                    row["utilization_percent"] > GPU_IDLE_UTILIZATION_LIMIT_PERCENT
                    for row in rows
                )
            )
        ):
            raise Wave5BenchmarkError(
                "one or more benchmark GPUs are busy", code="wave5.gpu_busy"
            )
        sample_inventory = _gpu_inventory(rows)
        if gpu_inventory is None:
            gpu_inventory = sample_inventory
        elif sample_inventory != gpu_inventory:
            raise Wave5BenchmarkError(
                "GPU index-to-UUID mapping changed during preflight",
                code="wave5.gpu_sample",
            )
        identities = _compute_app_identities(compute_apps)
        expected_uuids = {row["gpu_uuid"] for row in sample_inventory}
        if any(row["gpu_uuid"] not in expected_uuids for row in compute_apps):
            raise Wave5BenchmarkError(
                "GPU compute-process baseline contains an unexpected UUID",
                code="wave5.gpu_baseline_drift",
            )
        if baseline_compute_app_identities is None:
            baseline_compute_apps = deepcopy(compute_apps)
            baseline_compute_app_identities = identities
        elif identities != baseline_compute_app_identities:
            raise Wave5BenchmarkError(
                "GPU compute-process baseline changed during preflight",
                code="wave5.gpu_baseline_drift",
            )
        samples.append(
            {
                "sample_index": sample_index,
                "monotonic_ns": monotonic_ns,
                "gpus": rows,
                "compute_apps": compute_apps,
            }
        )
        if sample_index + 1 < GPU_IDLE_STABLE_SAMPLE_COUNT:
            time.sleep(GPU_BASELINE_SAMPLE_INTERVAL_SECONDS)
    _validate_sample_timing(
        samples,
        expected_count=GPU_IDLE_STABLE_SAMPLE_COUNT,
        interval_seconds=GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
        field="gpu_execution_baseline",
    )
    return {
        "execution_policy": policy,
        "performance_promotion_eligible": contract["performance_promotion_eligible"],
        "claim_scope": contract["claim_scope"],
        "sample_count": GPU_IDLE_STABLE_SAMPLE_COUNT,
        "sample_interval_seconds": GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
        "expected_total_memory_bytes_per_gpu": contract[
            "expected_total_memory_bytes_per_gpu"
        ],
        "max_used_bytes_per_gpu": contract["baseline_max_used_bytes_per_gpu"],
        "minimum_headroom_bytes_per_gpu": contract["minimum_headroom_bytes_per_gpu"],
        "utilization_semantics": contract["utilization_semantics"],
        "gpu_inventory": gpu_inventory,
        "baseline_compute_apps": baseline_compute_apps,
        "baseline_compute_app_identities": baseline_compute_app_identities,
        "samples": samples,
    }


def _stable_gpu_idle_samples() -> dict[str, Any]:
    """Compatibility seam for CPU tests of the retained promotional idle policy."""

    baseline = _stable_gpu_samples_for_policy("idle_promotional")
    return {
        "sample_count": baseline["sample_count"],
        "memory_limit_bytes": baseline["max_used_bytes_per_gpu"],
        "utilization_limit_percent": GPU_IDLE_UTILIZATION_LIMIT_PERCENT,
        "gpu_inventory": baseline["gpu_inventory"],
        "samples": baseline["samples"],
    }


def _stable_gpu_execution_baseline(plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = plan.get("gpu_execution_contract")
    if not isinstance(contract, Mapping):
        raise Wave5BenchmarkError(
            "GPU execution contract missing", code="wave5.gpu_execution_policy"
        )
    expected_contract = _gpu_execution_contract(str(contract.get("policy")))
    if dict(contract) != expected_contract:
        raise Wave5BenchmarkError(
            "GPU execution contract changed", code="wave5.gpu_execution_policy"
        )
    baseline = _stable_gpu_samples_for_policy(expected_contract["policy"])
    return _finalize(
        {
            "schema": GPU_BASELINE_SCHEMA,
            "status": "admitted",
            "plan_sha256": plan.get("plan_sha256"),
            "baseline_path": plan.get("artifact_targets", {}).get("gpu_baseline"),
            **baseline,
        },
        hash_field="gpu_baseline_sha256",
    )


def _validate_gpu_execution_baseline(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    baseline = _validate_finalized(
        value, schema=GPU_BASELINE_SCHEMA, hash_field="gpu_baseline_sha256"
    )
    if set(baseline) != {
        "schema",
        "status",
        "plan_sha256",
        "baseline_path",
        "execution_policy",
        "performance_promotion_eligible",
        "claim_scope",
        "sample_count",
        "sample_interval_seconds",
        "expected_total_memory_bytes_per_gpu",
        "max_used_bytes_per_gpu",
        "minimum_headroom_bytes_per_gpu",
        "utilization_semantics",
        "gpu_inventory",
        "baseline_compute_apps",
        "baseline_compute_app_identities",
        "samples",
        "gpu_baseline_sha256",
    }:
        raise Wave5BenchmarkError(
            "GPU execution baseline shape is invalid", code="wave5.gpu_baseline"
        )
    contract = plan.get("gpu_execution_contract")
    if not isinstance(contract, Mapping):
        raise Wave5BenchmarkError(
            "GPU execution contract missing", code="wave5.gpu_execution_policy"
        )
    expected_contract = _gpu_execution_contract(str(contract.get("policy")))
    samples = baseline.get("samples")
    inventory = baseline.get("gpu_inventory")
    frozen_apps = _validate_gpu_compute_apps(
        baseline.get("baseline_compute_apps"), field="gpu_baseline.compute_apps"
    )
    frozen_identities = _compute_app_identities(frozen_apps)
    if (
        baseline.get("status") != "admitted"
        or baseline.get("plan_sha256") != plan.get("plan_sha256")
        or baseline.get("baseline_path")
        != plan.get("artifact_targets", {}).get("gpu_baseline")
        or baseline.get("execution_policy") != expected_contract["policy"]
        or baseline.get("performance_promotion_eligible")
        != expected_contract["performance_promotion_eligible"]
        or baseline.get("claim_scope") != expected_contract["claim_scope"]
        or baseline.get("sample_count") != GPU_IDLE_STABLE_SAMPLE_COUNT
        or baseline.get("sample_interval_seconds")
        != GPU_BASELINE_SAMPLE_INTERVAL_SECONDS
        or baseline.get("expected_total_memory_bytes_per_gpu")
        != expected_contract["expected_total_memory_bytes_per_gpu"]
        or baseline.get("max_used_bytes_per_gpu")
        != expected_contract["baseline_max_used_bytes_per_gpu"]
        or baseline.get("minimum_headroom_bytes_per_gpu")
        != expected_contract["minimum_headroom_bytes_per_gpu"]
        or baseline.get("utilization_semantics")
        != expected_contract["utilization_semantics"]
        or baseline.get("baseline_compute_app_identities") != frozen_identities
        or not isinstance(samples, list)
        or len(samples) != GPU_IDLE_STABLE_SAMPLE_COUNT
        or not isinstance(inventory, list)
    ):
        raise Wave5BenchmarkError(
            "GPU execution baseline disagrees with plan", code="wave5.gpu_baseline"
        )
    _validate_sample_timing(
        samples,
        expected_count=GPU_IDLE_STABLE_SAMPLE_COUNT,
        interval_seconds=GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
        field="gpu_baseline.samples",
    )
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != {
            "sample_index",
            "monotonic_ns",
            "gpus",
            "compute_apps",
        }:
            raise Wave5BenchmarkError(
                "GPU execution baseline sample is malformed",
                code="wave5.gpu_baseline",
            )
        rows = _validate_gpu_sample(
            sample.get("gpus"), field=f"gpu_baseline.samples[{sample_index}]"
        )
        apps = _validate_gpu_compute_apps(
            sample.get("compute_apps"),
            field=f"gpu_baseline.samples[{sample_index}].compute_apps",
        )
        if (
            isinstance(sample.get("sample_index"), bool)
            or not isinstance(sample.get("sample_index"), int)
            or sample.get("sample_index") != sample_index
            or _gpu_inventory(rows) != inventory
            or _compute_app_identities(apps) != frozen_identities
            or any(
                row["memory_used_bytes"]
                > expected_contract["baseline_max_used_bytes_per_gpu"]
                or row["memory_headroom_bytes"]
                < expected_contract["minimum_headroom_bytes_per_gpu"]
                for row in rows
            )
            or (
                expected_contract["policy"] == "idle_promotional"
                and (
                    apps
                    or any(
                        row["utilization_percent"] > GPU_IDLE_UTILIZATION_LIMIT_PERCENT
                        for row in rows
                    )
                )
            )
        ):
            raise Wave5BenchmarkError(
                "GPU execution baseline sample drifted",
                code="wave5.gpu_baseline",
            )
    return baseline


def load_authenticated_gpu_baseline(
    plan: Mapping[str, Any], path: str | Path | None = None
) -> dict[str, Any]:
    target = plan.get("artifact_targets", {}).get("gpu_baseline")
    actual = _canonical_nonsymlink(target if path is None else path, must_exist=True)
    if str(actual) != target:
        raise Wave5BenchmarkError(
            "GPU execution baseline path mismatch", code="wave5.gpu_baseline"
        )
    return _validate_gpu_execution_baseline(load_strict_json(actual), plan=plan)


def _post_phase_gpu_process_sample(
    baseline: Mapping[str, Any],
    *,
    prior_bindings: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    sample, _ = _replay_gpu_process_ownership_sample(
        _sample_nvidia_compute_apps(),
        expected_gpu_uuids=[row["gpu_uuid"] for row in baseline["gpu_inventory"]],
        baseline_compute_apps=baseline["baseline_compute_apps"],
        prior_bindings=prior_bindings,
        allow_owned_compute_apps=False,
        field=field,
    )
    return sample


def _post_phase_gpu_process_sweep(
    baseline: Mapping[str, Any],
    *,
    prior_bindings: Mapping[str, Mapping[str, Any]],
    field: str,
    termination: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    samples = []
    try:
        for sample_index in range(GPU_POST_PHASE_SAMPLE_COUNT):
            monotonic_ns = time.monotonic_ns()
            ownership = _post_phase_gpu_process_sample(
                baseline,
                prior_bindings=prior_bindings,
                field=f"{field}.samples[{sample_index}]",
            )
            samples.append(
                {
                    "sample_index": sample_index,
                    "monotonic_ns": monotonic_ns,
                    "gpu_process_ownership": ownership,
                }
            )
            if sample_index + 1 < GPU_POST_PHASE_SAMPLE_COUNT:
                time.sleep(GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS)
        _validate_sample_timing(
            samples,
            expected_count=GPU_POST_PHASE_SAMPLE_COUNT,
            interval_seconds=GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS,
            field=field,
        )
    except BaseException as exc:
        return {
            "expected_sample_count": GPU_POST_PHASE_SAMPLE_COUNT,
            "sample_count": len(samples),
            "sample_interval_seconds": GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS,
            "samples": samples,
            "status": "sampling_error",
            "sampling_error": {
                "type": type(exc).__name__[:128],
                "code": str(getattr(exc, "code", "wave5.gpu_process_sample"))[:128],
                "message": str(exc)[:4096],
            },
            "cleanup": _bounded_cleanup_evidence(termination),
        }
    return {
        "expected_sample_count": GPU_POST_PHASE_SAMPLE_COUNT,
        "sample_count": GPU_POST_PHASE_SAMPLE_COUNT,
        "sample_interval_seconds": GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS,
        "samples": samples,
        "status": (
            "passed"
            if all(
                sample["gpu_process_ownership"]["status"] == "passed"
                for sample in samples
            )
            else "failed"
        ),
        "sampling_error": None,
        "cleanup": None,
    }


def _bounded_cleanup_evidence(
    termination: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if termination is None:
        return {
            "termination_recorded": False,
            "descendant_attestation": "not_required",
            "descendants_remaining_count": 0,
        }
    attestation = termination.get("descendant_attestation")
    remaining = termination.get("descendants_remaining")
    if attestation not in {"passed", "failed", "unavailable"}:
        attestation = "unavailable"
    if not isinstance(remaining, list) or any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        for pid in remaining
    ):
        remaining = []
        attestation = "unavailable"
    return {
        "termination_recorded": True,
        "descendant_attestation": attestation,
        "descendants_remaining_count": len(remaining),
    }


def _process_tree_rss_rows(root_pid: int) -> list[dict[str, int]]:
    pending = [int(root_pid)]
    seen: set[int] = set()
    rows: list[dict[str, int]] = []
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, UnicodeError) as exc:
            raise Wave5BenchmarkError(
                "process-tree RSS watchdog unavailable", code="wave5.resource_watchdog"
            ) from exc
        rss_kib = 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) != 3 or parts[2] != "kB":
                    raise Wave5BenchmarkError(
                        "process RSS field malformed", code="wave5.resource_watchdog"
                    )
                rss_kib = int(parts[1])
                break
        rows.append({"pid": pid, "rss_bytes": rss_kib * 1024})
        pending.extend(int(value) for value in children.split())
    return sorted(rows, key=lambda row: row["pid"])


def _process_tree_rss_bytes(root_pid: int) -> int:
    return sum(row["rss_bytes"] for row in _process_tree_rss_rows(root_pid))


def _validate_expected_gpu_uuids(value: Any, *, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != WORLD_SIZE
        or any(not isinstance(uuid, str) or not uuid for uuid in value)
        or len(set(value)) != WORLD_SIZE
    ):
        raise Wave5BenchmarkError(
            f"{field} does not identify exactly eight GPUs",
            code="wave5.gpu_process_ownership",
        )
    return list(value)


def _validate_gpu_bindings(
    value: Any, *, expected_gpu_uuids: Sequence[str], field: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise Wave5BenchmarkError(
            f"{field} is malformed", code="wave5.gpu_process_ownership"
        )
    expected = set(expected_gpu_uuids)
    if any(not isinstance(uuid, str) or uuid not in expected for uuid in value):
        raise Wave5BenchmarkError(
            f"{field} contains an unexpected GPU UUID",
            code="wave5.gpu_process_ownership",
        )
    bindings: dict[str, dict[str, Any]] = {}
    for uuid, row in value.items():
        validated = _validate_gpu_compute_apps([row], field=f"{field}.{uuid}")
        if validated[0]["gpu_uuid"] != uuid:
            raise Wave5BenchmarkError(
                f"{field} key disagrees with its row",
                code="wave5.gpu_process_ownership",
            )
        bindings[uuid] = validated[0]
    return bindings


def _ordered_gpu_bindings(
    bindings: Mapping[str, Mapping[str, Any]], expected_gpu_uuids: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        deepcopy(dict(bindings[uuid]))
        for uuid in expected_gpu_uuids
        if uuid in bindings
    ]


def _replay_gpu_process_ownership_sample(
    raw_compute_apps: Any,
    *,
    expected_gpu_uuids: Sequence[str],
    baseline_compute_apps: Sequence[Mapping[str, Any]] | None = None,
    prior_bindings: Mapping[str, Mapping[str, Any]],
    allow_owned_compute_apps: bool = True,
    field: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected_uuids = _validate_expected_gpu_uuids(
        list(expected_gpu_uuids), field=f"{field}.expected_gpu_uuids"
    )
    compute_apps = _validate_gpu_compute_apps(
        raw_compute_apps, field=f"{field}.compute_apps"
    )
    baseline_rows = _validate_gpu_compute_apps(
        [] if baseline_compute_apps is None else list(baseline_compute_apps),
        field=f"{field}.baseline_compute_apps",
    )
    baseline_identities = {(row["gpu_uuid"], row["pid"]) for row in baseline_rows}
    observed_baseline = [
        row
        for row in compute_apps
        if (row["gpu_uuid"], row["pid"]) in baseline_identities
    ]
    owned_compute_apps = [
        row
        for row in compute_apps
        if (row["gpu_uuid"], row["pid"]) not in baseline_identities
    ]
    bindings = _validate_gpu_bindings(
        prior_bindings,
        expected_gpu_uuids=expected_uuids,
        field=f"{field}.prior_bindings",
    )
    rows_by_uuid: dict[str, list[dict[str, Any]]] = {}
    for row in owned_compute_apps:
        rows_by_uuid.setdefault(row["gpu_uuid"], []).append(row)
    foreign: list[dict[str, Any]] = []
    violation_codes: list[str] = []
    for uuid, rows in rows_by_uuid.items():
        if uuid not in expected_uuids:
            foreign.extend(rows)
            if "unexpected_gpu_uuid" not in violation_codes:
                violation_codes.append("unexpected_gpu_uuid")
    for uuid in expected_uuids:
        rows = rows_by_uuid.get(uuid, [])
        if len(rows) > 1:
            foreign.extend(rows)
            if "multiple_compute_apps_per_gpu_uuid" not in violation_codes:
                violation_codes.append("multiple_compute_apps_per_gpu_uuid")
            continue
        if not rows:
            continue
        row = rows[0]
        if not allow_owned_compute_apps:
            foreign.append(row)
            if "nonbaseline_compute_app_after_phase" not in violation_codes:
                violation_codes.append("nonbaseline_compute_app_after_phase")
        elif uuid not in bindings:
            bindings[uuid] = deepcopy(row)
        elif row != bindings[uuid]:
            foreign.append(row)
            if "compute_app_binding_replaced" not in violation_codes:
                violation_codes.append("compute_app_binding_replaced")
    foreign = _validate_gpu_compute_apps(foreign, field=f"{field}.foreign_compute_apps")
    sample = {
        "schema_version": 2,
        "compute_apps": compute_apps,
        "bound_compute_apps": _ordered_gpu_bindings(bindings, expected_uuids),
        "foreign_compute_apps": foreign,
        "violation_codes": violation_codes,
        "status": "failed" if violation_codes else "passed",
    }
    if baseline_compute_apps is not None:
        sample.update(
            {
                "baseline_compute_apps": observed_baseline,
                "owned_compute_apps": owned_compute_apps,
                "allow_owned_compute_apps": allow_owned_compute_apps,
            }
        )
    return sample, bindings


def _gpu_process_ownership_sample(
    *,
    expected_gpu_uuids: Sequence[str],
    baseline_compute_apps: Sequence[Mapping[str, Any]] | None = None,
    prior_bindings: Mapping[str, Mapping[str, Any]],
    allow_owned_compute_apps: bool = True,
    field: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    return _replay_gpu_process_ownership_sample(
        _sample_nvidia_compute_apps(),
        expected_gpu_uuids=expected_gpu_uuids,
        baseline_compute_apps=baseline_compute_apps,
        prior_bindings=prior_bindings,
        allow_owned_compute_apps=allow_owned_compute_apps,
        field=field,
    )


def _validate_gpu_process_ownership_sequence(
    samples: Any,
    *,
    final_sample: Any,
    expected_gpu_uuids: Sequence[str],
    baseline_compute_apps: Sequence[Mapping[str, Any]] | None = None,
    require_complete: bool,
    field: str,
) -> dict[str, dict[str, Any]]:
    expected_uuids = _validate_expected_gpu_uuids(
        list(expected_gpu_uuids), field=f"{field}.expected_gpu_uuids"
    )
    if not isinstance(samples, list):
        raise Wave5BenchmarkError(
            f"{field} samples are malformed", code="wave5.gpu_process_ownership"
        )
    bindings: dict[str, dict[str, Any]] = {}
    replay_values = [*samples, final_sample]
    for sample_index, value in enumerate(replay_values):
        sample_field = (
            f"{field}.samples[{sample_index}]"
            if sample_index < len(samples)
            else f"{field}.final_sample"
        )
        expected_fields = {
            "schema_version",
            "compute_apps",
            "bound_compute_apps",
            "foreign_compute_apps",
            "violation_codes",
            "status",
        }
        if baseline_compute_apps is not None:
            expected_fields.update(
                {
                    "baseline_compute_apps",
                    "owned_compute_apps",
                    "allow_owned_compute_apps",
                }
            )
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise Wave5BenchmarkError(
                f"{sample_field} is malformed", code="wave5.gpu_process_ownership"
            )
        expected_sample, bindings = _replay_gpu_process_ownership_sample(
            value.get("compute_apps"),
            expected_gpu_uuids=expected_uuids,
            baseline_compute_apps=baseline_compute_apps,
            prior_bindings=bindings,
            allow_owned_compute_apps=sample_index < len(samples),
            field=sample_field,
        )
        if dict(value) != expected_sample:
            raise Wave5BenchmarkError(
                f"{sample_field} disagrees with raw compute-app replay",
                code="wave5.gpu_process_ownership",
            )
    if require_complete and set(bindings) != set(expected_uuids):
        raise Wave5BenchmarkError(
            f"{field} did not bind all eight GPU UUIDs",
            code="wave5.gpu_process_ownership",
        )
    return bindings


def _validate_gpu_idle_preflight_receipt(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "sample_count",
        "memory_limit_bytes",
        "utilization_limit_percent",
        "gpu_inventory",
        "samples",
    }:
        raise Wave5BenchmarkError(
            "resource receipt GPU preflight is malformed", code="wave5.resource"
        )
    samples = value.get("samples")
    inventory = value.get("gpu_inventory")
    if (
        value.get("sample_count") != GPU_IDLE_STABLE_SAMPLE_COUNT
        or value.get("memory_limit_bytes") != GPU_IDLE_MEMORY_LIMIT_BYTES
        or value.get("utilization_limit_percent") != GPU_IDLE_UTILIZATION_LIMIT_PERCENT
        or not isinstance(samples, list)
        or len(samples) != GPU_IDLE_STABLE_SAMPLE_COUNT
        or not isinstance(inventory, list)
    ):
        raise Wave5BenchmarkError(
            "resource receipt GPU preflight is incomplete", code="wave5.resource"
        )
    _validate_sample_timing(
        samples,
        expected_count=GPU_IDLE_STABLE_SAMPLE_COUNT,
        interval_seconds=GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
        field="resource_receipt.preflight",
    )
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != {
            "sample_index",
            "monotonic_ns",
            "gpus",
            "compute_apps",
        }:
            raise Wave5BenchmarkError(
                "resource receipt GPU preflight sample is malformed",
                code="wave5.resource",
            )
        rows = _validate_gpu_sample(
            sample.get("gpus"), field=f"resource_receipt.preflight[{sample_index}]"
        )
        compute_apps = _validate_gpu_compute_apps(
            sample.get("compute_apps"),
            field=f"resource_receipt.preflight[{sample_index}].compute_apps",
        )
        if (
            sample.get("sample_index") != sample_index
            or _gpu_inventory(rows) != inventory
            or compute_apps
            or any(
                row["memory_used_bytes"] > GPU_IDLE_MEMORY_LIMIT_BYTES
                or row["utilization_percent"] > GPU_IDLE_UTILIZATION_LIMIT_PERCENT
                for row in rows
            )
        ):
            raise Wave5BenchmarkError(
                "resource receipt GPU preflight disagrees with its frozen inventory",
                code="wave5.resource",
            )
    return deepcopy(inventory)


def _validate_resource_error(value: Any, *, field: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"type", "code", "message"}
        or not all(
            isinstance(value.get(name), str) and value[name]
            for name in ("type", "code", "message")
        )
        or len(value["type"]) > 128
        or len(value["code"]) > 128
        or len(value["message"]) > 4096
    ):
        raise Wave5BenchmarkError(
            f"resource receipt {field} is malformed", code="wave5.resource"
        )
    return dict(value)


def _validate_post_phase_sweep_envelope(
    value: Any,
    *,
    field: str,
    termination: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    expected_fields = {
        "expected_sample_count",
        "sample_count",
        "sample_interval_seconds",
        "samples",
        "status",
        "sampling_error",
        "cleanup",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or isinstance(value.get("expected_sample_count"), bool)
        or value.get("expected_sample_count") != GPU_POST_PHASE_SAMPLE_COUNT
        or isinstance(value.get("sample_count"), bool)
        or not isinstance(value.get("sample_count"), int)
        or value.get("sample_interval_seconds")
        != GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS
        or not isinstance(value.get("samples"), list)
        or value.get("sample_count") != len(value["samples"])
    ):
        raise Wave5BenchmarkError(
            f"{field} envelope is malformed", code="wave5.resource"
        )
    samples = value["samples"]
    ownership_samples = []
    for sample_index, sample in enumerate(samples):
        if (
            not isinstance(sample, Mapping)
            or set(sample) != {"sample_index", "monotonic_ns", "gpu_process_ownership"}
            or isinstance(sample.get("sample_index"), bool)
            or not isinstance(sample.get("sample_index"), int)
            or sample.get("sample_index") != sample_index
            or not isinstance(sample.get("gpu_process_ownership"), Mapping)
        ):
            raise Wave5BenchmarkError(
                f"{field} sample is malformed", code="wave5.resource"
            )
        _validate_monotonic_ns(
            sample.get("monotonic_ns"), field=f"{field}.samples[{sample_index}]"
        )
        ownership_samples.append(dict(sample["gpu_process_ownership"]))
    status = value.get("status")
    if status in {"passed", "failed"}:
        _validate_sample_timing(
            samples,
            expected_count=GPU_POST_PHASE_SAMPLE_COUNT,
            interval_seconds=GPU_POST_PHASE_SAMPLE_INTERVAL_SECONDS,
            field=f"{field}.samples",
        )
        if value.get("sampling_error") is not None or value.get("cleanup") is not None:
            raise Wave5BenchmarkError(
                f"{field} successful envelope is contradictory",
                code="wave5.resource",
            )
    elif status == "sampling_error":
        if len(samples) >= GPU_POST_PHASE_SAMPLE_COUNT:
            raise Wave5BenchmarkError(
                f"{field} unavailable envelope is contradictory",
                code="wave5.resource",
            )
        _validate_resource_error(
            value.get("sampling_error"), field=f"{field}.sampling_error"
        )
        if value.get("cleanup") != _bounded_cleanup_evidence(termination):
            raise Wave5BenchmarkError(
                f"{field} cleanup evidence changed", code="wave5.resource"
            )
    else:
        raise Wave5BenchmarkError(f"{field} status is invalid", code="wave5.resource")
    return ownership_samples, status


def _validate_resource_receipt(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    resource = _validate_finalized(
        value, schema=RESOURCE_SCHEMA, hash_field="resource_sha256"
    )
    if set(resource) != {
        "schema",
        "status",
        "production_argv",
        "production_argv_sha256",
        "elapsed_seconds",
        "gpu_execution_baseline",
        "resource_breach",
        "monitor_error",
        "termination",
        "samples",
        "post_phase_gpu_process_sweep",
        "resource_sha256",
    }:
        raise Wave5BenchmarkError(
            "resource receipt top-level shape is malformed", code="wave5.resource"
        )
    argv = resource.get("production_argv")
    samples = resource.get("samples")
    status = resource.get("status")
    breach = resource.get("resource_breach")
    monitor_error = resource.get("monitor_error")
    termination = resource.get("termination")
    post_phase_sweep = resource.get("post_phase_gpu_process_sweep")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(argument, str) or not argument for argument in argv)
        or resource.get("production_argv_sha256") != sha256_json(argv)
        or not isinstance(samples, list)
        or (termination is not None and not isinstance(termination, Mapping))
    ):
        raise Wave5BenchmarkError(
            "resource receipt execution envelope is malformed", code="wave5.resource"
        )
    _finite(resource.get("elapsed_seconds"), "resource_receipt.elapsed_seconds")
    if status == "preflight_error":
        if (
            resource.get("gpu_execution_baseline") is not None
            or breach is not None
            or termination is not None
            or samples
            or post_phase_sweep is not None
        ):
            raise Wave5BenchmarkError(
                "resource receipt preflight_error shape is contradictory",
                code="wave5.resource",
            )
        _validate_resource_error(monitor_error, field="monitor_error")
        return deepcopy(resource)
    baseline = _validate_gpu_execution_baseline(
        resource.get("gpu_execution_baseline"), plan=plan
    )
    if baseline != load_authenticated_gpu_baseline(plan):
        raise Wave5BenchmarkError(
            "resource receipt GPU baseline binding changed", code="wave5.resource"
        )
    inventory = baseline["gpu_inventory"]
    ownership_samples = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise Wave5BenchmarkError(
                "resource receipt lifetime sample is malformed",
                code="wave5.resource",
            )
        gpu_rows = _validate_gpu_sample(
            sample.get("gpus"),
            field=f"resource_receipt.samples[{sample_index}].gpus",
        )
        if _gpu_inventory(gpu_rows) != inventory:
            raise Wave5BenchmarkError(
                "resource receipt lifetime GPU inventory changed",
                code="wave5.resource",
            )
        ownership_samples.append(sample.get("gpu_process_ownership"))
    post_phase_samples, sweep_status = _validate_post_phase_sweep_envelope(
        post_phase_sweep,
        field="resource_receipt.post_phase_gpu_process_sweep",
        termination=termination,
    )
    expected_gpu_uuids = [row["gpu_uuid"] for row in inventory]
    bindings: dict[str, dict[str, Any]] = {}
    for sample_index, ownership in enumerate(ownership_samples):
        expected_ownership, bindings = _replay_gpu_process_ownership_sample(
            ownership.get("compute_apps") if isinstance(ownership, Mapping) else None,
            expected_gpu_uuids=expected_gpu_uuids,
            baseline_compute_apps=baseline["baseline_compute_apps"],
            prior_bindings=bindings,
            allow_owned_compute_apps=True,
            field=f"resource_receipt.gpu_process_ownership.samples[{sample_index}]",
        )
        if ownership != expected_ownership:
            raise Wave5BenchmarkError(
                "resource receipt lifetime GPU ownership is not derived",
                code="wave5.resource",
            )
    for sample_index, ownership in enumerate(post_phase_samples):
        expected_ownership, bindings = _replay_gpu_process_ownership_sample(
            ownership.get("compute_apps"),
            expected_gpu_uuids=expected_gpu_uuids,
            baseline_compute_apps=baseline["baseline_compute_apps"],
            prior_bindings=bindings,
            allow_owned_compute_apps=False,
            field=(
                f"resource_receipt.post_phase_gpu_process_sweep.samples[{sample_index}]"
            ),
        )
        if ownership != expected_ownership:
            raise Wave5BenchmarkError(
                "resource receipt post-phase GPU sweep is not derived",
                code="wave5.resource",
            )
    if sweep_status in {"passed", "failed"} and sweep_status != (
        "passed"
        if all(sample.get("status") == "passed" for sample in post_phase_samples)
        else "failed"
    ):
        raise Wave5BenchmarkError(
            "resource receipt post-phase GPU sweep status is not derived",
            code="wave5.resource",
        )
    ownership_values = [*ownership_samples, *post_phase_samples]
    if status == "complete":
        if (
            breach is not None
            or monitor_error is not None
            or termination is not None
            or sweep_status != "passed"
            or len(bindings) != WORLD_SIZE
            or any(sample.get("status") != "passed" for sample in ownership_values)
        ):
            raise Wave5BenchmarkError(
                "resource receipt complete status is contradictory",
                code="wave5.resource",
            )
    elif status == "monitor_error":
        if breach is not None or any(
            sample.get("status") != "passed" for sample in ownership_values
        ):
            raise Wave5BenchmarkError(
                "resource receipt monitor_error shape is contradictory",
                code="wave5.resource",
            )
        _validate_resource_error(monitor_error, field="monitor_error")
    elif status == "resource_limit_exceeded":
        if (
            not isinstance(breach, Mapping)
            or not isinstance(breach.get("code"), str)
            or not breach["code"]
        ):
            raise Wave5BenchmarkError(
                "resource receipt resource_limit_exceeded breach is malformed",
                code="wave5.resource",
            )
        _validate_resource_error(monitor_error, field="monitor_error")
    else:
        raise Wave5BenchmarkError(
            "resource receipt status is invalid", code="wave5.resource"
        )
    return deepcopy(resource)


def _artifact_tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Wave5BenchmarkError(
                "arm output contains a symlink", code="wave5.output_symlink"
            )
        if path.is_file():
            total += path.stat().st_size
    return total


def _resource_limit_breach(
    *,
    arm: str,
    gpu_rows: Sequence[Mapping[str, Any]],
    process_rss_rows: Sequence[Mapping[str, Any]],
    artifact_bytes: int,
) -> dict[str, Any] | None:
    for row in process_rss_rows:
        rss = int(row["rss_bytes"])
        if rss > HOST_RSS_PER_RANK_CEILING_BYTES:
            return {
                "code": "host_rss_per_rank_ceiling",
                "arm": arm,
                "pid": int(row["pid"]),
                "observed": rss,
                "ceiling": HOST_RSS_PER_RANK_CEILING_BYTES,
            }
    process_rss_bytes = sum(int(row["rss_bytes"]) for row in process_rss_rows)
    if process_rss_bytes > HOST_RSS_ALL_RANK_CEILING_BYTES:
        return {
            "code": "host_rss_all_rank_ceiling",
            "arm": arm,
            "observed": process_rss_bytes,
            "ceiling": HOST_RSS_ALL_RANK_CEILING_BYTES,
        }
    if artifact_bytes > ARTIFACT_CEILING_BYTES:
        return {
            "code": "artifact_ceiling",
            "arm": arm,
            "observed": artifact_bytes,
            "ceiling": ARTIFACT_CEILING_BYTES,
        }
    for row in gpu_rows:
        memory = int(row["memory_used_bytes"])
        if memory > GPU_MEMORY_CEILING_BYTES:
            return {
                "code": "gpu_memory_ceiling",
                "arm": arm,
                "gpu_index": int(row["gpu_index"]),
                "observed": memory,
                "ceiling": GPU_MEMORY_CEILING_BYTES,
            }
    return None


def _stop_for_resource_breach(
    process: subprocess.Popen[Any], breach: Mapping[str, Any]
) -> None:
    _terminate(process)
    raise Wave5BenchmarkError(
        f"arm exceeded hard resource ceiling: {breach.get('code')}",
        code="wave5.resource_ceiling",
    )


def _production_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    cache_root = Path(
        plan["w0_binding"]["current_v3"]["train"]["manifest_path"]
    ).parents[2]
    environment = os.environ.copy()
    environment["COORDEXP_SWIFT_PACK_CACHE_ROOT"] = str(cache_root)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index) for index in range(WORLD_SIZE)
    )
    for selector in (
        "COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE",
        "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS",
        "COORDEXP_SWIFT_EVAL_REDUCTION_MODE",
    ):
        environment.pop(selector, None)
    return environment


def run_production_arm(
    plan_path: str | Path, attempt_path: str | Path, triad_index: int, arm: str
) -> int:
    plan = validate_plan(load_strict_json(plan_path))
    _revalidate_plan_compatibility(plan)
    load_authenticated_attempt(plan, attempt_path)
    _write_arm_config(plan, triad_index, arm)
    targets = _arm_targets(plan, triad_index, arm)
    argv = _production_argv(plan, triad_index, arm)
    environment = _production_environment(plan)
    started = time.monotonic()
    try:
        gpu_execution_baseline = load_authenticated_gpu_baseline(plan)
    except BaseException as exc:
        resource = _finalize(
            {
                "schema": RESOURCE_SCHEMA,
                "status": "preflight_error",
                "production_argv": argv,
                "production_argv_sha256": sha256_json(argv),
                "elapsed_seconds": time.monotonic() - started,
                "gpu_execution_baseline": None,
                "resource_breach": None,
                "monitor_error": {
                    "type": type(exc).__name__[:128],
                    "code": str(getattr(exc, "code", "wave5.gpu_preflight"))[:128],
                    "message": str(exc)[:4096],
                },
                "termination": None,
                "samples": [],
                "post_phase_gpu_process_sweep": None,
            },
            hash_field="resource_sha256",
        )
        publish_json_absent(targets["resource_samples"], resource)
        raise
    samples: list[dict[str, Any]] = []
    resource_breach: dict[str, Any] | None = None
    monitor_error: BaseException | None = None
    termination: dict[str, Any] | None = None
    expected_gpu_uuids = [
        row["gpu_uuid"] for row in gpu_execution_baseline["gpu_inventory"]
    ]
    baseline_compute_apps = gpu_execution_baseline["baseline_compute_apps"]
    gpu_bindings: dict[str, dict[str, Any]] = {}
    post_phase_gpu_process_sweep: dict[str, Any] | None = None
    with (
        Path(targets["stdout"]).open("x", encoding="utf-8") as stdout,
        Path(targets["stderr"]).open("x", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                gpu_rows = _validate_gpu_sample(
                    _sample_nvidia(), field="arm_process_lifetime"
                )
                if _gpu_inventory(gpu_rows) != gpu_execution_baseline["gpu_inventory"]:
                    raise Wave5BenchmarkError(
                        "GPU index-to-UUID mapping changed after preflight",
                        code="wave5.gpu_sample",
                    )
                process_rss_rows = _process_tree_rss_rows(process.pid)
                gpu_process_ownership, gpu_bindings = _gpu_process_ownership_sample(
                    expected_gpu_uuids=expected_gpu_uuids,
                    baseline_compute_apps=baseline_compute_apps,
                    prior_bindings=gpu_bindings,
                    allow_owned_compute_apps=True,
                    field=f"arm_process_lifetime.samples[{len(samples)}]",
                )
                artifact_bytes = _artifact_tree_bytes(Path(targets["output_root"]))
                sample = {
                    "elapsed_seconds": time.monotonic() - started,
                    "gpus": gpu_rows,
                    "gpu_process_ownership": gpu_process_ownership,
                    "process_rss_rows": process_rss_rows,
                    "process_tree_rss_bytes": sum(
                        row["rss_bytes"] for row in process_rss_rows
                    ),
                    "artifact_bytes": artifact_bytes,
                }
                samples.append(sample)
                if gpu_process_ownership["status"] != "passed":
                    resource_breach = {
                        "code": "foreign_gpu_process",
                        "arm": arm,
                        "foreign_compute_apps": deepcopy(
                            gpu_process_ownership["foreign_compute_apps"]
                        ),
                    }
                    termination = _terminate(process)
                    raise Wave5BenchmarkError(
                        "foreign GPU compute process appeared during the arm",
                        code="wave5.foreign_gpu_process",
                    )
                resource_breach = _resource_limit_breach(
                    arm=arm,
                    gpu_rows=gpu_rows,
                    process_rss_rows=process_rss_rows,
                    artifact_bytes=artifact_bytes,
                )
                if resource_breach is not None:
                    termination = _terminate(process)
                    raise Wave5BenchmarkError(
                        "arm exceeded hard resource ceiling: "
                        f"{resource_breach.get('code')}",
                        code="wave5.resource_ceiling",
                    )
                time.sleep(1.0)
        except BaseException as exc:
            monitor_error = exc
            if termination is None:
                termination = _terminate(process)
            if termination.get("descendant_attestation") != "passed":
                monitor_error = Wave5BenchmarkError(
                    "process-tree cleanup could not attest zero live descendants",
                    code="wave5.termination",
                )
        try:
            post_phase_gpu_process_sweep = _post_phase_gpu_process_sweep(
                gpu_execution_baseline,
                prior_bindings=gpu_bindings,
                field="arm_process_lifetime.post_phase",
                termination=termination,
            )
            if post_phase_gpu_process_sweep["status"] == "sampling_error":
                sampling_error = post_phase_gpu_process_sweep["sampling_error"]
                if monitor_error is None:
                    monitor_error = Wave5BenchmarkError(
                        sampling_error["message"], code=sampling_error["code"]
                    )
            elif (
                post_phase_gpu_process_sweep["status"] == "failed"
                and resource_breach is None
            ):
                foreign_rows = [
                    row
                    for sample in post_phase_gpu_process_sweep["samples"]
                    for row in sample["gpu_process_ownership"]["foreign_compute_apps"]
                ]
                resource_breach = {
                    "code": "foreign_gpu_process",
                    "arm": arm,
                    "foreign_compute_apps": deepcopy(foreign_rows),
                }
                monitor_error = Wave5BenchmarkError(
                    "foreign GPU compute process appeared after the arm",
                    code="wave5.foreign_gpu_process",
                )
            if (
                process.returncode == 0
                and set(gpu_bindings) != set(expected_gpu_uuids)
                and monitor_error is None
            ):
                monitor_error = Wave5BenchmarkError(
                    "successful arm did not bind all eight GPU UUIDs",
                    code="wave5.gpu_process_ownership",
                )
        except BaseException as exc:
            if monitor_error is None:
                monitor_error = exc
    resource = _finalize(
        {
            "schema": RESOURCE_SCHEMA,
            "status": (
                "resource_limit_exceeded"
                if resource_breach is not None
                else "monitor_error"
                if monitor_error is not None
                else "complete"
            ),
            "production_argv": argv,
            "production_argv_sha256": sha256_json(argv),
            "elapsed_seconds": time.monotonic() - started,
            "gpu_execution_baseline": gpu_execution_baseline,
            "resource_breach": resource_breach,
            "monitor_error": (
                None
                if monitor_error is None
                else {
                    "type": type(monitor_error).__name__[:128],
                    "code": str(getattr(monitor_error, "code", "wave5.monitor"))[:128],
                    "message": str(monitor_error)[:4096],
                }
            ),
            "termination": termination,
            "samples": samples,
            "post_phase_gpu_process_sweep": post_phase_gpu_process_sweep,
        },
        hash_field="resource_sha256",
    )
    publish_json_absent(targets["resource_samples"], resource)
    if monitor_error is not None:
        raise monitor_error
    return int(process.returncode)


def _finite(value: object, field: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise Wave5BenchmarkError(f"invalid numeric field {field}", code="wave5.metric")
    return float(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
            if not isinstance(value, dict):
                raise ValueError("row")
            rows.append(value)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Wave5BenchmarkError(
            "logging JSONL invalid", code="wave5.logging"
        ) from exc
    return rows


def _artifact_inventory(root: Path) -> dict[str, Any]:
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Wave5BenchmarkError(
                "output contains symlink", code="wave5.output_symlink"
            )
        if path.is_file():
            size = path.stat().st_size
            total += size
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "files": files,
        "file_count": len(files),
        "total_bytes": total,
        "inventory_sha256": sha256_json(files),
    }


def _load_executed_streams(root: Path, *, provider_mode: str) -> dict[str, Any]:
    checked_root = _canonical_nonsymlink(root, must_exist=True)
    expected_names = {f"rank-{rank}.json" for rank in range(WORLD_SIZE)}
    observed_names = {path.name for path in checked_root.iterdir()}
    if observed_names != expected_names:
        raise Wave5BenchmarkError(
            "executed stream rank coverage is incomplete",
            code="wave5.executed_stream",
        )
    ranks: dict[str, Any] = {}
    for rank in range(WORLD_SIZE):
        path = _canonical_nonsymlink(
            checked_root / f"rank-{rank}.json", must_exist=True
        )
        receipt = _validate_rank_stream_receipt(
            load_strict_json(path), rank=rank, provider_mode=provider_mode
        )
        ranks[str(rank)] = {
            "path": str(path.relative_to(checked_root.parent)),
            "file_sha256": sha256_file(path),
            "stream_receipt_sha256": receipt["stream_receipt_sha256"],
            "event_count": receipt["event_count"],
            "semantic_digests": deepcopy(receipt["semantic_digests"]),
        }
    aggregate = {
        field: sha256_json(
            [
                {"rank": rank, "sha256": ranks[str(rank)]["semantic_digests"][field]}
                for rank in range(WORLD_SIZE)
            ]
        )
        for field in STREAM_PARITY_FIELDS
    }
    return {
        "schema": RANK_STREAM_SCHEMA,
        "world_size": WORLD_SIZE,
        "provider_mode": provider_mode,
        "event_count_per_rank": 15,
        "ranks": ranks,
        "aggregate_semantic_digests": aggregate,
    }


def _validate_production_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    from peft import PeftConfig
    from safetensors import safe_open
    from src.qwen.special_token_embeddings import (
        inspect_special_token_embedding_delta_payload,
    )

    adapter_dir = _canonical_nonsymlink(checkpoint_dir / "adapter", must_exist=True)
    try:
        config = PeftConfig.from_pretrained(str(adapter_dir))
    except Exception as exc:
        raise Wave5BenchmarkError(
            "checkpoint adapter config is not publicly loadable",
            code="wave5.checkpoint",
        ) from exc
    tensor_path = _canonical_nonsymlink(
        adapter_dir / "adapter_model.safetensors", must_exist=True
    )
    try:
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            keys = tuple(sorted(handle.keys()))
            shapes = {key: list(handle.get_tensor(key).shape) for key in keys}
    except Exception as exc:
        raise Wave5BenchmarkError(
            "checkpoint adapter tensor payload is not publicly loadable",
            code="wave5.checkpoint",
        ) from exc
    required = {
        "lora_A": any("lora_A" in key for key in keys),
        "lora_B": any("lora_B" in key for key in keys),
        "lora_magnitude_vector": any("lora_magnitude_vector" in key for key in keys),
    }
    if (
        not keys
        or not all(required.values())
        or any("embed_tokens" in key or "lm_head" in key for key in keys)
    ):
        raise Wave5BenchmarkError(
            "checkpoint adapter schema is not inference-minimal",
            code="wave5.checkpoint",
        )
    special_token_dir = _canonical_nonsymlink(
        checkpoint_dir / "special_token_embeddings", must_exist=True
    )
    try:
        special_token_receipt = inspect_special_token_embedding_delta_payload(
            special_token_dir
        )
    except Exception as exc:
        raise Wave5BenchmarkError(
            "checkpoint special-token payload is not publicly loadable",
            code="wave5.checkpoint",
        ) from exc
    return {
        "validators": [
            "peft.PeftConfig.from_pretrained",
            "safetensors.safe_open",
            "src.qwen.special_token_embeddings.inspect_special_token_embedding_delta_payload",
        ],
        "peft_type": str(config.peft_type),
        "task_type": None if config.task_type is None else str(config.task_type),
        "tensor_count": len(keys),
        "tensor_inventory_sha256": sha256_json(shapes),
        "special_token_payload": special_token_receipt,
        "loadable": True,
    }


def _semantic_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        key
        for key in row
        if any(
            token in key
            for token in (
                "duration",
                "seconds",
                "rss",
                "memory",
                "io_",
                "resource",
                "per_rank",
            )
        )
    }
    return {key: row[key] for key in sorted(row) if key not in excluded}


def _expected_provider_resolution(mode: str) -> dict[str, Any]:
    if mode not in {spec["provider_mode"] for spec in ARM_SPECS.values()}:
        raise Wave5BenchmarkError(
            "unknown provider factor", code="wave5.provider_identity"
        )
    return {
        "configured_mode": mode,
        "resolved_mode": mode,
        "source": "strict_config",
        "environment_variable": None,
        "is_semantic_override": False,
        "provider_disposition": "none" if mode == "legacy_fused" else mode,
        "input_build_owner": {
            "legacy_fused": "trainer_fused_device_direct",
            "synchronous": "provider_consumer_cpu",
            "overlapped": "provider_producer_cpu",
        }[mode],
        "device_transfer_owner": (
            "trainer_fused_build" if mode == "legacy_fused" else "provider_consumer"
        ),
        "lookahead_depth": int(mode == "overlapped"),
    }


def _project_provider_factor(
    policies: Mapping[str, Any], *, expected_mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_resolution = _expected_provider_resolution(expected_mode)
    provider = policies.get("input_provider")
    expected_provider = {
        "schema_version": 1,
        "mode": expected_mode,
        **expected_resolution,
    }
    if provider != expected_provider:
        raise Wave5BenchmarkError(
            "resolved provider policy does not match the arm factor",
            code="wave5.provider_identity",
        )
    non_provider = deepcopy(dict(policies))
    del non_provider["input_provider"]
    if not non_provider:
        raise Wave5BenchmarkError(
            "non-provider policy identity is empty", code="wave5.identity"
        )
    return non_provider, expected_provider


def _timing_and_resource_summary(
    measured: Sequence[Mapping[str, Any]],
    eval_row: Mapping[str, Any],
    phases: Mapping[str, Mapping[str, Any]],
    resource: Mapping[str, Any],
    output_inventory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        resource.get("status") != "complete"
        or resource.get("resource_breach") is not None
    ):
        raise Wave5BenchmarkError(
            "arm resource receipt is not successful", code="wave5.resource"
        )
    baseline = resource.get("gpu_execution_baseline")
    baseline_inventory = (
        baseline.get("gpu_inventory") if isinstance(baseline, Mapping) else None
    )
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("sample_count") != GPU_IDLE_STABLE_SAMPLE_COUNT
        or baseline.get("sample_interval_seconds")
        != GPU_BASELINE_SAMPLE_INTERVAL_SECONDS
        or not isinstance(baseline.get("samples"), list)
        or len(baseline["samples"]) != GPU_IDLE_STABLE_SAMPLE_COUNT
        or not isinstance(baseline_inventory, list)
    ):
        raise Wave5BenchmarkError(
            "stable GPU execution baseline is incomplete",
            code="wave5.gpu_preflight",
        )
    for sample_index, sample in enumerate(baseline["samples"]):
        if (
            not isinstance(sample, Mapping)
            or sample.get("sample_index") != sample_index
        ):
            raise Wave5BenchmarkError(
                "stable GPU idle sample is malformed", code="wave5.gpu_preflight"
            )
        rows = _validate_gpu_sample(
            sample.get("gpus"), field=f"gpu_execution_baseline[{sample_index}]"
        )
        if _gpu_inventory(rows) != baseline_inventory:
            raise Wave5BenchmarkError(
                "stable GPU baseline index-to-UUID mapping changed",
                code="wave5.gpu_preflight",
            )
    per_step = []
    rank_high_water = {
        str(rank): {
            "cpu_max_rss_bytes": 0.0,
            "cpu_io_read_bytes": 0.0,
            "cpu_io_write_bytes": 0.0,
            "gpu_max_memory_allocated_bytes": 0.0,
            "gpu_max_memory_reserved_bytes": 0.0,
        }
        for rank in range(WORLD_SIZE)
    }
    resource_fields = {
        "cpu_max_rss_bytes": "resource/cpu_max_rss_bytes",
        "cpu_io_read_bytes": "resource/cpu_io_read_bytes",
        "cpu_io_write_bytes": "resource/cpu_io_write_bytes",
        "gpu_max_memory_allocated_bytes": ("resource/gpu_max_memory_allocated_bytes"),
        "gpu_max_memory_reserved_bytes": "resource/gpu_max_memory_reserved_bytes",
    }
    for row in measured:
        per_rank = row["per_rank_measurement"]
        step_wall = []
        build = []
        wait = []
        for rank in range(WORLD_SIZE):
            rank_row = per_rank[str(rank)]
            step_wall.append(
                _finite(rank_row.get("step_duration_seconds"), "step_duration")
            )
            build.append(_finite(rank_row.get("input_build_seconds"), "input_build"))
            wait.append(_finite(rank_row.get("input_wait_seconds"), "input_wait"))
            for summary_field, source_field in resource_fields.items():
                value = _finite(rank_row.get(source_field), source_field)
                rank_high_water[str(rank)][summary_field] = max(
                    rank_high_water[str(rank)][summary_field], value
                )
        per_step.append(
            {
                "step": row["step"],
                "all_rank_max_step_wall_seconds": max(step_wall),
                "input_build_all_rank_max_seconds": max(build),
                "input_build_skew_seconds": max(build) - min(build),
                "input_wait_all_rank_max_seconds": max(wait),
                "input_wait_skew_seconds": max(wait) - min(wait),
            }
        )
    eval_rank = eval_row.get("per_rank_measurement")
    if not isinstance(eval_rank, Mapping) or set(eval_rank) != {
        str(rank) for rank in range(WORLD_SIZE)
    }:
        raise Wave5BenchmarkError(
            "evaluation rank resource coverage incomplete", code="wave5.eval_coverage"
        )
    for rank in range(WORLD_SIZE):
        rank_row = eval_rank[str(rank)]
        _finite(rank_row.get("eval_duration_seconds"), "eval_duration")
        for summary_field, source_field in resource_fields.items():
            value = _finite(rank_row.get(source_field), source_field)
            rank_high_water[str(rank)][summary_field] = max(
                rank_high_water[str(rank)][summary_field], value
            )
    samples = resource.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Wave5BenchmarkError("GPU resource samples missing", code="wave5.resource")
    gpu_memory_max = 0.0
    gpu_utilization_max = 0.0
    gpu_utilization_values: list[float] = []
    gpu_utilization_by_device: dict[str, list[float]] = {
        str(index): [] for index in range(WORLD_SIZE)
    }
    gpu_process_ownership_sample_count = 0
    ownership_samples = []
    for sample in samples:
        gpus = sample.get("gpus") if isinstance(sample, Mapping) else None
        if not isinstance(gpus, list) or {
            row.get("gpu_index") for row in gpus if isinstance(row, Mapping)
        } != set(range(WORLD_SIZE)):
            raise Wave5BenchmarkError(
                "GPU sample scope is not exactly eight devices", code="wave5.resource"
            )
        gpus = _validate_gpu_sample(gpus, field="arm_process_lifetime.gpus")
        if _gpu_inventory(gpus) != baseline_inventory:
            raise Wave5BenchmarkError(
                "arm GPU index-to-UUID mapping changed",
                code="wave5.gpu_process_ownership",
            )
        process_tree_rss = _finite(
            sample.get("process_tree_rss_bytes"), "process_tree_rss_bytes"
        )
        ownership = sample.get("gpu_process_ownership")
        ownership_samples.append(ownership)
        gpu_process_ownership_sample_count += 1
        process_rss_rows = sample.get("process_rss_rows")
        if process_rss_rows is not None:
            if not isinstance(process_rss_rows, list) or not process_rss_rows:
                raise Wave5BenchmarkError(
                    "per-process RSS samples malformed", code="wave5.resource"
                )
            rss_values = [
                _finite(row.get("rss_bytes"), "process_rss")
                for row in process_rss_rows
                if isinstance(row, Mapping)
            ]
            if (
                len(rss_values) != len(process_rss_rows)
                or sum(rss_values) != process_tree_rss
            ):
                raise Wave5BenchmarkError(
                    "per-process RSS samples disagree with tree total",
                    code="wave5.resource",
                )
            if any(value > HOST_RSS_PER_RANK_CEILING_BYTES for value in rss_values):
                raise Wave5BenchmarkError(
                    "per-rank RSS ceiling exceeded", code="wave5.resource"
                )
        _finite(sample.get("artifact_bytes"), "artifact_bytes")
        gpu_memory_max = max(
            gpu_memory_max,
            *[_finite(row.get("memory_used_bytes"), "gpu.memory") for row in gpus],
        )
        gpu_utilization_max = max(
            gpu_utilization_max,
            *[
                _finite(row.get("utilization_percent"), "gpu.utilization")
                for row in gpus
            ],
        )
        for row in gpus:
            utilization = _finite(row.get("utilization_percent"), "gpu.utilization")
            gpu_utilization_values.append(utilization)
            gpu_utilization_by_device[str(int(row["gpu_index"]))].append(utilization)
    _validate_gpu_process_ownership_sequence(
        ownership_samples,
        final_sample=resource["post_phase_gpu_process_sweep"]["samples"][0][
            "gpu_process_ownership"
        ],
        expected_gpu_uuids=[row["gpu_uuid"] for row in baseline_inventory],
        baseline_compute_apps=baseline["baseline_compute_apps"],
        require_complete=True,
        field="arm_process_lifetime.gpu_process_ownership",
    )
    if any(
        isinstance(ownership, Mapping) and ownership.get("status") != "passed"
        for ownership in [
            *ownership_samples,
            *[
                sample["gpu_process_ownership"]
                for sample in resource["post_phase_gpu_process_sweep"]["samples"]
            ],
        ]
    ):
        raise Wave5BenchmarkError(
            "arm resource receipt contains a foreign GPU process",
            code="wave5.foreign_gpu_process",
        )
    rss_max = max(row["cpu_max_rss_bytes"] for row in rank_high_water.values())
    rss_sum = sum(row["cpu_max_rss_bytes"] for row in rank_high_water.values())
    allocator_allocated_max = max(
        row["gpu_max_memory_allocated_bytes"] for row in rank_high_water.values()
    )
    allocator_reserved_max = max(
        row["gpu_max_memory_reserved_bytes"] for row in rank_high_water.values()
    )
    resource_gate = (
        rss_max <= HOST_RSS_PER_RANK_CEILING_BYTES
        and rss_sum <= HOST_RSS_ALL_RANK_CEILING_BYTES
        and gpu_memory_max <= GPU_MEMORY_CEILING_BYTES
        and allocator_allocated_max <= GPU_MEMORY_CEILING_BYTES
        and allocator_reserved_max <= GPU_MEMORY_CEILING_BYTES
        and output_inventory["total_bytes"] <= ARTIFACT_CEILING_BYTES
    )
    time_to_first = sum(
        _finite(phases[name].get("duration_seconds"), f"phase.{name}")
        for name in EXPECTED_PHASES[: EXPECTED_PHASES.index("first_optimizer_step") + 1]
    )
    timing = {
        "time_to_first_optimizer_step_seconds": time_to_first,
        "steady_state_optimizer_step_wall_seconds": sum(
            row["all_rank_max_step_wall_seconds"] for row in per_step
        ),
        "per_step": per_step,
        "all_phase_durations_seconds": {
            name: _finite(phases[name].get("duration_seconds"), f"phase.{name}")
            for name in EXPECTED_PHASES
        },
    }
    resources = {
        "gpu_execution_policy": baseline["execution_policy"],
        "performance_promotion_eligible": baseline["performance_promotion_eligible"],
        "shared_load_metrics_observational_only": (
            not baseline["performance_promotion_eligible"]
        ),
        "scope": (
            "all_train_and_eval_ranks_plus_arm_process_lifetime_gpu_samples_"
            "and_complete_output_tree"
        ),
        "rank_high_water": rank_high_water,
        "cpu_max_rss_bytes_per_rank_max": rss_max,
        "cpu_max_rss_bytes_all_rank_sum": rss_sum,
        "cpu_io_read_bytes_all_rank_sum": sum(
            row["cpu_io_read_bytes"] for row in rank_high_water.values()
        ),
        "cpu_io_write_bytes_all_rank_sum": sum(
            row["cpu_io_write_bytes"] for row in rank_high_water.values()
        ),
        "gpu_memory_used_bytes_process_lifetime_max": gpu_memory_max,
        "gpu_allocator_max_allocated_bytes_per_rank_max": allocator_allocated_max,
        "gpu_allocator_max_reserved_bytes_per_rank_max": allocator_reserved_max,
        "gpu_utilization_percent_process_lifetime_max": gpu_utilization_max,
        "gpu_utilization_representative": {
            "sample_count": len(samples),
            "all_device_samples": _distribution(gpu_utilization_values),
            "per_device": {
                index: _distribution(values)
                for index, values in gpu_utilization_by_device.items()
            },
        },
        "gpu_process_ownership_sample_count": gpu_process_ownership_sample_count,
        "gpu_process_ownership_attested": True,
        "artifact_bytes": output_inventory["total_bytes"],
        "resource_gate_passed": resource_gate,
    }
    return timing, resources


def derive_observation(
    plan: Mapping[str, Any], *, triad_index: int, arm: str
) -> dict[str, Any]:
    checked = validate_plan(plan)
    _revalidate_plan_compatibility(checked)
    targets = _arm_targets(checked, triad_index, arm)
    observation_path = Path(targets["observation"])
    if observation_path.exists():
        raise Wave5BenchmarkError(
            "synthetic child observation rejected", code="wave5.synthetic_observation"
        )
    run_dir = _canonical_nonsymlink(targets["run_dir"], must_exist=True)
    run = load_strict_json(run_dir / "run.json")
    resolved = load_strict_json(run_dir / "resolved_config.json")
    resource = _validate_resource_receipt(
        load_strict_json(targets["resource_samples"]), plan=checked
    )
    expected_production_argv = _production_argv(checked, triad_index, arm)
    if resource.get("production_argv") != expected_production_argv or resource.get(
        "production_argv_sha256"
    ) != sha256_json(expected_production_argv):
        raise Wave5BenchmarkError(
            "production argv authentication failed", code="wave5.command_identity"
        )
    logs = _read_jsonl(run_dir / "logging.jsonl")
    train_rows = [row for row in logs if row.get("split") == "train"]
    eval_rows = [row for row in logs if row.get("split") == "eval"]
    config = resolved.get("config", {})
    resolution = resolved.get("resolution", {})
    if (
        run.get("status") != "completed"
        or run.get("completed_steps") != 5
        or run.get("forward_input_provider_mode") != ARM_SPECS[arm]["provider_mode"]
        or config.get("training", {}).get("forward_input_provider_mode")
        != ARM_SPECS[arm]["provider_mode"]
        or config.get("training", {}).get("max_steps") != 5
        or config.get("eval", {}).get("forward", {}).get("steps") != [3]
        or config.get("checkpoint", {}).get("steps") != [5]
        or run.get("config_fingerprint") != resolution.get("fingerprint")
    ):
        raise Wave5BenchmarkError(
            "production run/config identity mismatch", code="wave5.run_identity"
        )
    projected_config = _project_arm_config(config)
    if projected_config != checked["config_compatibility"]["current_parent_projection"]:
        raise Wave5BenchmarkError(
            "resolved config compatibility projection changed",
            code="wave5.config_compatibility",
        )
    measurement = run.get("measurement")
    expected_measurement_context = _expected_measurement_context(
        checked, arm=arm, workload_identity=resolution["fingerprint"]
    )
    entry_to_terminal = (
        measurement.get("entry_to_terminal")
        if isinstance(measurement, Mapping)
        else None
    )
    if (
        not isinstance(measurement, Mapping)
        or measurement.get("context") != expected_measurement_context
        or not isinstance(entry_to_terminal, Mapping)
        or set(entry_to_terminal)
        != {
            "status",
            "started_at",
            "completed_at",
            "duration_seconds",
            "clock",
            "boundary",
        }
        or entry_to_terminal.get("status") != "completed"
        or entry_to_terminal.get("clock") != "monotonic"
        or entry_to_terminal.get("boundary") != MEASUREMENT_ENTRY_TO_TERMINAL_BOUNDARY
    ):
        raise Wave5BenchmarkError(
            "run measurement context is not exact",
            code="wave5.measurement_context",
        )
    _parse_attested_at(entry_to_terminal.get("started_at"))
    _parse_attested_at(entry_to_terminal.get("completed_at"))
    entry_to_terminal_seconds = _finite(
        entry_to_terminal.get("duration_seconds"), "entry_to_terminal"
    )
    phases = measurement.get("phases") if isinstance(measurement, Mapping) else None
    if not isinstance(phases, Mapping) or set(phases) != set(EXPECTED_PHASES):
        raise Wave5BenchmarkError("phase coverage incomplete", code="wave5.phases")
    phase_receipts = {}
    for name in EXPECTED_PHASES:
        phase = phases[name]
        if not isinstance(phase, Mapping) or "status" not in phase:
            raise Wave5BenchmarkError("phase receipt invalid", code="wave5.phases")
        allowed_statuses = (
            {"completed", "not_run"}
            if name in {"cache_preparation", "cache_publication"}
            else {"completed"}
        )
        if phase["status"] not in allowed_statuses:
            raise Wave5BenchmarkError("phase did not complete", code="wave5.phases")
        _finite(phase.get("duration_seconds"), f"phase.{name}")
        phase_receipts[name] = deepcopy(dict(phase))
    if len(train_rows) != 5 or [row.get("step") for row in train_rows] != [
        1,
        2,
        3,
        4,
        5,
    ]:
        raise Wave5BenchmarkError(
            "training rows must be the exact unique five-step sequence",
            code="wave5.train_coverage",
        )
    measured = [row for row in train_rows if row.get("step") in MEASURED_STEPS]
    if {row.get("step") for row in measured} != set(MEASURED_STEPS):
        raise Wave5BenchmarkError(
            "measured step coverage incomplete", code="wave5.train_coverage"
        )
    for row in train_rows:
        if (
            row.get("optimizer_update_status") != "applied"
            or row.get("finite_status") != "finite"
        ):
            raise Wave5BenchmarkError(
                "optimizer/loss semantics failed", code="wave5.optimizer"
            )
        per_rank = row.get("per_rank_measurement")
        if not isinstance(per_rank, Mapping) or set(map(str, range(WORLD_SIZE))) != set(
            map(str, per_rank)
        ):
            raise Wave5BenchmarkError(
                "train rank coverage incomplete", code="wave5.train_coverage"
            )
    if (
        len(eval_rows) != 1
        or eval_rows[0].get("step") != 3
        or eval_rows[0].get("example_count") != 64
        or eval_rows[0].get("pack_count") != 8
    ):
        raise Wave5BenchmarkError(
            "evaluation coverage/denominators incomplete", code="wave5.eval_coverage"
        )
    eval_semantic = _semantic_projection(eval_rows[0])
    eval_denominators = {
        key: value
        for key, value in eval_semantic.items()
        if any(
            token in key for token in ("count", "denominator", "numerator", "weight")
        )
    }
    final = load_strict_json(run_dir / "checkpoints/final.json")
    if final != {"step": 5, "checkpoint_path": "checkpoints/step-5"}:
        raise Wave5BenchmarkError(
            "checkpoint final receipt invalid", code="wave5.checkpoint"
        )
    checkpoint_dir = _canonical_nonsymlink(
        run_dir / final["checkpoint_path"], must_exist=True
    )
    checkpoint_inventory = _artifact_inventory(checkpoint_dir)
    if not checkpoint_inventory["files"]:
        raise Wave5BenchmarkError("checkpoint output empty", code="wave5.checkpoint")
    checkpoint_loadability = _validate_production_checkpoint(checkpoint_dir)
    output_inventory = _artifact_inventory(run_dir)
    timing, resource_summary = _timing_and_resource_summary(
        measured, eval_rows[0], phase_receipts, resource, output_inventory
    )
    timing["entry_to_terminal_seconds"] = entry_to_terminal_seconds
    provenance = run.get("provenance")
    policies = run.get("policy_identities")
    materializations = run.get("materializations")
    if not all(
        isinstance(value, Mapping) and value
        for value in (provenance, policies, materializations)
    ):
        raise Wave5BenchmarkError(
            "dependency/cache policy identity incomplete", code="wave5.identity"
        )
    expected_provider_mode = ARM_SPECS[arm]["provider_mode"]
    if policies.get("eval_reduction") != checked["semantic_contract"]["eval_reduction"]:
        raise Wave5BenchmarkError(
            "eval reduction policy is not disjoint_shard/default",
            code="wave5.eval_reduction",
        )
    expected_provider_resolution = _expected_provider_resolution(expected_provider_mode)
    if run.get("forward_input_provider_resolution") != expected_provider_resolution:
        raise Wave5BenchmarkError(
            "run provider resolution does not match the arm factor",
            code="wave5.provider_identity",
        )
    non_provider_policies, provider_factor = _project_provider_factor(
        policies, expected_mode=expected_provider_mode
    )
    non_provider_config = deepcopy(config)
    non_provider_config["training"].pop("forward_input_provider_mode", None)
    non_provider_config["run"].pop("name", None)
    non_provider_config["run"].pop("artifact_root", None)
    non_provider_config["run"].pop("collision_policy", None)
    executed_streams = _load_executed_streams(
        Path(targets["executed_streams"]), provider_mode=expected_provider_mode
    )
    train_semantics = [_semantic_projection(row) for row in train_rows]
    optimizer_projection = [
        {
            key: row.get(key)
            for key in (
                "step",
                "micro_step_count",
                "optimizer_update_status",
                "finite_status",
                "lr",
            )
        }
        for row in train_rows
    ]
    semantic = {
        **executed_streams["aggregate_semantic_digests"],
        "loss_results_sha256": sha256_json(train_semantics),
        "optimizer_semantics_sha256": sha256_json(optimizer_projection),
        "evaluation_results_sha256": sha256_json(eval_semantic),
        "checkpoint_output_sha256": checkpoint_inventory["inventory_sha256"],
    }
    observation = {
        "schema": OBSERVATION_SCHEMA,
        "status": "derived_from_production_artifacts",
        "plan_sha256": checked["plan_sha256"],
        "triad_index": triad_index,
        "triad_order": list(TRIAD_ORDERS[triad_index]),
        "position": TRIAD_ORDERS[triad_index].index(arm) + 1,
        "arm": arm,
        "provider_mode": ARM_SPECS[arm]["provider_mode"],
        "measurement_context": expected_measurement_context,
        "source_artifacts": {
            "run_json_sha256": sha256_file(run_dir / "run.json"),
            "resolved_config_sha256": sha256_file(run_dir / "resolved_config.json"),
            "logging_jsonl_sha256": sha256_file(run_dir / "logging.jsonl"),
            "resource_samples_sha256": resource["resource_sha256"],
            "executed_stream_receipts_sha256": {
                rank: row["file_sha256"]
                for rank, row in executed_streams["ranks"].items()
            },
            "final_checkpoint_receipt_sha256": sha256_file(
                run_dir / "checkpoints/final.json"
            ),
        },
        "execution_identity": {
            "resolved_config_fingerprint": resolution["fingerprint"],
            "non_provider_config_sha256": sha256_json(non_provider_config),
            "config_compatibility_projection_sha256": sha256_json(projected_config),
            "model_config_sha256": sha256_json(config.get("model")),
            "base_model_weight_identity_sha256": checked["base_model_weight_identity"][
                "aggregate_sha256"
            ],
            "cache_binding_sha256": sha256_json(checked["w0_binding"]["current_v3"]),
            "dependency_provenance_sha256": sha256_json(provenance),
            "non_provider_policy_identities_sha256": sha256_json(non_provider_policies),
            "provider_factor": provider_factor,
            "provider_factor_sha256": sha256_json(provider_factor),
            "materializations_sha256": sha256_json(materializations),
        },
        "phases": phase_receipts,
        "executed_provider_streams": executed_streams,
        "pack_utilization": {
            split: {
                key: checked["w0_binding"]["current_v3"][split][key]
                for key in (
                    "micro_step_count",
                    "real_tokens",
                    "capacity_tokens",
                    "utilization_fraction",
                    "pack_stream_sha256",
                )
            }
            for split in ("train", "eval")
        },
        "training": {
            "warmup_steps": list(WARMUP_STEPS),
            "measured_steps": list(MEASURED_STEPS),
            "row_count": len(train_rows),
            "measured_rows": deepcopy(measured),
            "loss_results": train_semantics,
            "optimizer_semantics": optimizer_projection,
        },
        "evaluation": {
            "event_count": 1,
            "steps": [3],
            "example_count": eval_rows[0]["example_count"],
            "pack_count": eval_rows[0]["pack_count"],
            "denominators": eval_denominators,
            "results": eval_semantic,
            "rank_resources": deepcopy(eval_rows[0]["per_rank_measurement"]),
        },
        "checkpoint": {
            "final_receipt": final,
            "inventory": checkpoint_inventory,
            "loadability": checkpoint_loadability,
        },
        "resources": {
            "run_high_water": deepcopy(measurement.get("resource_high_water")),
            "arm_process_lifetime": resource,
            "complete_output_inventory": output_inventory,
            "summary": resource_summary,
        },
        "timing": timing,
        "semantic": semantic,
    }
    finalized = _finalize(observation, hash_field="observation_sha256")
    validate_observation(finalized, plan=checked, triad_index=triad_index, arm=arm)
    return finalized


def validate_observation(
    payload: Mapping[str, Any], *, plan: Mapping[str, Any], triad_index: int, arm: str
) -> dict[str, Any]:
    checked = validate_plan(plan)
    obs = _validate_finalized(
        payload, schema=OBSERVATION_SCHEMA, hash_field="observation_sha256"
    )
    if set(obs) != {
        "schema",
        "status",
        "plan_sha256",
        "triad_index",
        "triad_order",
        "position",
        "arm",
        "provider_mode",
        "measurement_context",
        "source_artifacts",
        "execution_identity",
        "phases",
        "executed_provider_streams",
        "pack_utilization",
        "training",
        "evaluation",
        "checkpoint",
        "resources",
        "timing",
        "semantic",
        "observation_sha256",
    }:
        raise Wave5BenchmarkError(
            "observation schema is not exact", code="wave5.observation"
        )
    if (
        obs.get("status") != "derived_from_production_artifacts"
        or obs.get("plan_sha256") != checked["plan_sha256"]
        or obs.get("triad_index") != triad_index
        or obs.get("arm") != arm
        or obs.get("triad_order") != list(TRIAD_ORDERS[triad_index])
        or obs.get("position") != TRIAD_ORDERS[triad_index].index(arm) + 1
        or obs.get("provider_mode") != ARM_SPECS[arm]["provider_mode"]
    ):
        raise Wave5BenchmarkError(
            "observation identity mismatch", code="wave5.observation"
        )
    semantic = obs.get("semantic")
    if (
        not isinstance(semantic, Mapping)
        or set(semantic) != set(SEMANTIC_PARITY_FIELDS)
        or not all(_is_sha256(value) for value in semantic.values())
    ):
        raise Wave5BenchmarkError("semantic schema incomplete", code="wave5.semantic")
    expected_semantic = {
        **obs["executed_provider_streams"]["aggregate_semantic_digests"],
        "loss_results_sha256": sha256_json(obs["training"]["loss_results"]),
        "optimizer_semantics_sha256": sha256_json(
            obs["training"]["optimizer_semantics"]
        ),
        "evaluation_results_sha256": sha256_json(obs["evaluation"]["results"]),
        "checkpoint_output_sha256": obs["checkpoint"]["inventory"]["inventory_sha256"],
    }
    if semantic != expected_semantic:
        raise Wave5BenchmarkError(
            "semantic values are not derived from receipt fields", code="wave5.semantic"
        )
    if obs.get("execution_identity", {}).get("cache_binding_sha256") != sha256_json(
        checked["w0_binding"]["current_v3"]
    ):
        raise Wave5BenchmarkError("cache identity mismatch", code="wave5.identity")
    if set(obs.get("phases", {})) != set(EXPECTED_PHASES):
        raise Wave5BenchmarkError("phase schema incomplete", code="wave5.phases")
    if obs.get("training", {}).get("measured_steps") != list(MEASURED_STEPS) or obs.get(
        "evaluation", {}
    ).get("steps") != [3]:
        raise Wave5BenchmarkError("coverage schema mismatch", code="wave5.coverage")
    inventory = obs.get("resources", {}).get("complete_output_inventory", {})
    if (
        inventory.get("total_bytes", ARTIFACT_CEILING_BYTES + 1)
        > ARTIFACT_CEILING_BYTES
    ):
        raise Wave5BenchmarkError("artifact ceiling exceeded", code="wave5.resource")
    targets = _arm_targets(checked, triad_index, arm)
    run_dir = Path(targets["run_dir"])
    run = load_strict_json(run_dir / "run.json")
    policies = run.get("policy_identities")
    if not isinstance(policies, Mapping):
        raise Wave5BenchmarkError("policy identity missing", code="wave5.identity")
    if policies.get("eval_reduction") != checked["semantic_contract"]["eval_reduction"]:
        raise Wave5BenchmarkError(
            "eval reduction policy changed", code="wave5.eval_reduction"
        )
    non_provider_policies, provider_factor = _project_provider_factor(
        policies, expected_mode=ARM_SPECS[arm]["provider_mode"]
    )
    execution_identity = obs.get("execution_identity")
    resolved = load_strict_json(run_dir / "resolved_config.json")
    resolved_config = resolved.get("config")
    resolution = resolved.get("resolution")
    if not isinstance(resolved_config, Mapping) or not isinstance(resolution, Mapping):
        raise Wave5BenchmarkError(
            "resolved config evidence is malformed", code="wave5.config_compatibility"
        )
    projected_config = _project_arm_config(resolved_config)
    expected_context = _expected_measurement_context(
        checked, arm=arm, workload_identity=str(resolution.get("fingerprint", ""))
    )
    if (
        not isinstance(execution_identity, Mapping)
        or projected_config
        != checked["config_compatibility"]["current_parent_projection"]
        or execution_identity.get("config_compatibility_projection_sha256")
        != checked["config_compatibility"]["current_parent_projection_sha256"]
        or execution_identity.get("base_model_weight_identity_sha256")
        != checked["base_model_weight_identity"]["aggregate_sha256"]
        or obs.get("measurement_context") != expected_context
        or execution_identity.get("non_provider_policy_identities_sha256")
        != sha256_json(non_provider_policies)
        or execution_identity.get("provider_factor") != provider_factor
        or execution_identity.get("provider_factor_sha256")
        != sha256_json(provider_factor)
    ):
        raise Wave5BenchmarkError(
            "provider factor projection is not exact", code="wave5.provider_identity"
        )
    expected_sources = {
        "run_json_sha256": sha256_file(run_dir / "run.json"),
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.json"),
        "logging_jsonl_sha256": sha256_file(run_dir / "logging.jsonl"),
        "resource_samples_sha256": obs["resources"]["arm_process_lifetime"][
            "resource_sha256"
        ],
        "executed_stream_receipts_sha256": {
            rank: row["file_sha256"]
            for rank, row in obs["executed_provider_streams"]["ranks"].items()
        },
        "final_checkpoint_receipt_sha256": sha256_file(
            run_dir / "checkpoints/final.json"
        ),
    }
    if obs.get("source_artifacts") != expected_sources:
        raise Wave5BenchmarkError(
            "source artifact identity changed", code="wave5.source_artifact"
        )
    expected_streams = _load_executed_streams(
        Path(targets["executed_streams"]),
        provider_mode=ARM_SPECS[arm]["provider_mode"],
    )
    if obs.get("executed_provider_streams") != expected_streams:
        raise Wave5BenchmarkError(
            "executed stream evidence changed", code="wave5.executed_stream"
        )
    if _artifact_inventory(run_dir) != inventory:
        raise Wave5BenchmarkError(
            "complete output inventory changed", code="wave5.source_artifact"
        )
    checkpoint_dir = run_dir / obs["checkpoint"]["final_receipt"]["checkpoint_path"]
    if _artifact_inventory(checkpoint_dir) != obs["checkpoint"]["inventory"]:
        raise Wave5BenchmarkError(
            "checkpoint inventory changed", code="wave5.source_artifact"
        )
    if obs["checkpoint"].get("loadability") != _validate_production_checkpoint(
        checkpoint_dir
    ):
        raise Wave5BenchmarkError(
            "checkpoint loadability evidence changed", code="wave5.checkpoint"
        )
    reconstructed_eval = {
        **obs["evaluation"]["results"],
        "per_rank_measurement": obs["evaluation"]["rank_resources"],
    }
    expected_timing, expected_resources = _timing_and_resource_summary(
        obs["training"]["measured_rows"],
        reconstructed_eval,
        obs["phases"],
        obs["resources"]["arm_process_lifetime"],
        inventory,
    )
    run_measurement = run.get("measurement")
    entry_to_terminal = (
        run_measurement.get("entry_to_terminal")
        if isinstance(run_measurement, Mapping)
        else None
    )
    if (
        not isinstance(run_measurement, Mapping)
        or run_measurement.get("context") != expected_context
        or not isinstance(entry_to_terminal, Mapping)
        or entry_to_terminal.get("boundary") != MEASUREMENT_ENTRY_TO_TERMINAL_BOUNDARY
        or entry_to_terminal.get("status") != "completed"
        or entry_to_terminal.get("clock") != "monotonic"
    ):
        raise Wave5BenchmarkError(
            "measurement context evidence changed",
            code="wave5.measurement_context",
        )
    expected_timing["entry_to_terminal_seconds"] = _finite(
        entry_to_terminal.get("duration_seconds"), "entry_to_terminal"
    )
    if (
        obs.get("timing") != expected_timing
        or obs["resources"].get("summary") != expected_resources
    ):
        raise Wave5BenchmarkError(
            "timing or resource summary is not derived", code="wave5.resource"
        )
    if (
        obs.get("resources", {}).get("summary", {}).get("resource_gate_passed")
        is not True
    ):
        raise Wave5BenchmarkError("resource gate failed", code="wave5.resource")
    return obs


def _step_wall(observation: Mapping[str, Any]) -> float:
    durations = []
    for row in observation["training"]["measured_rows"]:
        per_rank = row["per_rank_measurement"]
        values = [
            _finite(rank_row.get("step_duration_seconds"), "step_duration")
            for rank_row in per_rank.values()
        ]
        durations.append(max(values))
    return sum(durations)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    checked = [
        _finite(value, "distribution", minimum=-float("inf")) for value in values
    ]
    if not checked:
        raise Wave5BenchmarkError(
            "measurement distribution is empty", code="wave5.decision"
        )
    median = statistics.median(checked)
    mad = statistics.median(abs(value - median) for value in checked)
    minimum = min(checked)
    maximum = max(checked)
    return {
        "count": len(checked),
        "values": checked,
        "median": median,
        "mad": mad,
        "minimum": minimum,
        "maximum": maximum,
        "range": maximum - minimum,
    }


def _observation_measurements(observation: Mapping[str, Any]) -> dict[str, float]:
    per_step = observation["timing"]["per_step"]
    resources = observation["resources"]["summary"]
    return {
        "entry_to_terminal_seconds": _finite(
            observation["timing"]["entry_to_terminal_seconds"],
            "entry_to_terminal_seconds",
        ),
        "time_to_first_optimizer_step_seconds": _finite(
            observation["timing"]["time_to_first_optimizer_step_seconds"],
            "time_to_first_optimizer_step_seconds",
        ),
        "steady_state_seconds": _step_wall(observation),
        "input_build_all_rank_max_seconds": sum(
            _finite(row["input_build_all_rank_max_seconds"], "input_build")
            for row in per_step
        ),
        "input_wait_all_rank_max_seconds": sum(
            _finite(row["input_wait_all_rank_max_seconds"], "input_wait")
            for row in per_step
        ),
        "input_build_skew_seconds": max(
            _finite(row["input_build_skew_seconds"], "input_build_skew")
            for row in per_step
        ),
        "input_wait_skew_seconds": max(
            _finite(row["input_wait_skew_seconds"], "input_wait_skew")
            for row in per_step
        ),
        "cpu_max_rss_bytes_per_rank_max": _finite(
            resources["cpu_max_rss_bytes_per_rank_max"], "cpu_rss"
        ),
        "cpu_max_rss_bytes_all_rank_sum": _finite(
            resources["cpu_max_rss_bytes_all_rank_sum"], "cpu_rss_sum"
        ),
        "cpu_io_read_bytes_all_rank_sum": _finite(
            resources["cpu_io_read_bytes_all_rank_sum"], "cpu_io_read"
        ),
        "cpu_io_write_bytes_all_rank_sum": _finite(
            resources["cpu_io_write_bytes_all_rank_sum"], "cpu_io_write"
        ),
        "gpu_memory_used_bytes_process_lifetime_max": _finite(
            resources["gpu_memory_used_bytes_process_lifetime_max"], "gpu_memory"
        ),
        "gpu_utilization_percent_process_lifetime_max": _finite(
            resources["gpu_utilization_percent_process_lifetime_max"],
            "gpu_utilization",
        ),
        "gpu_utilization_percent_representative_median": _finite(
            resources["gpu_utilization_representative"]["all_device_samples"]["median"],
            "gpu_utilization_representative",
        ),
        "artifact_bytes": _finite(resources["artifact_bytes"], "artifact_bytes"),
    }


def _measurement_summary(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    arm_failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    per_arm = {}
    measurements: dict[tuple[int, str], dict[str, float]] = {}
    for key, observation in indexed.items():
        measurements[key] = _observation_measurements(observation)
    for arm in ARM_SPECS:
        arm_indices = [index for index in range(3) if (index, arm) in measurements]
        if not arm_indices:
            continue
        rows = [measurements[(index, arm)] for index in arm_indices]
        phases = {
            name: _distribution(
                [
                    _finite(
                        indexed[(index, arm)]["phases"][name]["duration_seconds"],
                        f"phase.{name}",
                    )
                    for index in arm_indices
                ]
            )
            for name in EXPECTED_PHASES
        }
        per_arm[arm] = {
            "provider_mode": ARM_SPECS[arm]["provider_mode"],
            **{field: _distribution([row[field] for row in rows]) for field in rows[0]},
            "phase_durations_seconds": phases,
        }
    paired = {}

    def delta_name(field: str) -> str:
        for suffix in ("_seconds", "_bytes"):
            if field.endswith(suffix):
                return f"{field[: -len(suffix)]}_delta{suffix}"
        return f"{field}_delta"

    for arm in CANDIDATE_ARMS:
        triads = []
        for index in range(3):
            if (index, REFERENCE_ARM) not in measurements or (
                index,
                arm,
            ) not in measurements:
                continue
            reference = measurements[(index, REFERENCE_ARM)]
            candidate = measurements[(index, arm)]
            deltas = {
                delta_name(field): candidate[field] - reference[field]
                for field in reference
            }
            deltas["steady_state_gain_fraction"] = (
                reference["steady_state_seconds"] - candidate["steady_state_seconds"]
            ) / reference["steady_state_seconds"]
            deltas["entry_to_terminal_gain_fraction"] = (
                reference["entry_to_terminal_seconds"]
                - candidate["entry_to_terminal_seconds"]
            ) / reference["entry_to_terminal_seconds"]
            triads.append({"triad_index": index, **deltas})
        if not triads:
            continue
        fields_to_summarize = [key for key in triads[0] if key != "triad_index"]
        paired[arm] = {
            "provider_mode": ARM_SPECS[arm]["provider_mode"],
            "triads": triads,
            "summaries": {
                field: _distribution([row[field] for row in triads])
                for field in fields_to_summarize
            },
        }
    return {
        "per_arm": per_arm,
        "paired_comparisons": paired,
        "failure_summary": {
            "expected_arm_executions": 9,
            "successful_observations": len(indexed),
            "failed_observations": 9 - len(indexed),
            "arm_failures": deepcopy([dict(item) for item in arm_failures]),
        },
    }


def _validate_declared_stop(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "category",
        "stage",
        "error",
        "completed_evidence",
    }:
        raise Wave5BenchmarkError("declared stop is malformed", code="wave5.stop")
    category = value.get("category")
    stage = value.get("stage")
    error = value.get("error")
    if (
        category not in {"failure", "resource-stop"}
        or stage not in {"arm_execution", "decision"}
        or not isinstance(error, Mapping)
    ):
        raise Wave5BenchmarkError("declared stop is malformed", code="wave5.stop")
    if set(error) != {"type", "code", "message"} or not all(
        isinstance(error.get(field), str) and error[field]
        for field in ("type", "code", "message")
    ):
        raise Wave5BenchmarkError("declared stop error is malformed", code="wave5.stop")
    if (
        len(error["type"]) > 128
        or len(error["code"]) > 128
        or len(error["message"]) > 4096
    ):
        raise Wave5BenchmarkError("declared stop error is unbounded", code="wave5.stop")
    completed_evidence = value.get("completed_evidence")
    if not isinstance(completed_evidence, Mapping) or not all(
        isinstance(name, str) and _is_sha256(digest)
        for name, digest in completed_evidence.items()
    ):
        raise Wave5BenchmarkError(
            "declared stop evidence is malformed", code="wave5.stop"
        )
    return {
        "category": category,
        "stage": stage,
        "error": dict(error),
        "completed_evidence": dict(completed_evidence),
    }


def _completed_arm_evidence(
    plan: Mapping[str, Any], triad_index: int, arm: str
) -> dict[str, str]:
    targets = _arm_targets(plan, triad_index, arm)
    candidates = {
        "resource_samples": Path(targets["resource_samples"]),
        "stdout": Path(targets["stdout"]),
        "stderr": Path(targets["stderr"]),
        "run_json": Path(targets["run_dir"]) / "run.json",
        "resolved_config": Path(targets["run_dir"]) / "resolved_config.json",
        "logging_jsonl": Path(targets["run_dir"]) / "logging.jsonl",
        "final_checkpoint_receipt": (
            Path(targets["run_dir"]) / "checkpoints/final.json"
        ),
    }
    stream_root = Path(targets["executed_streams"])
    candidates.update(
        {
            f"executed_stream_rank_{rank}": stream_root / f"rank-{rank}.json"
            for rank in range(WORLD_SIZE)
        }
    )
    return {
        name: sha256_file(path)
        for name, path in sorted(candidates.items())
        if path.is_file() and not path.is_symlink()
    }


def _completed_matrix_evidence(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        f"observation_{index + 1}": str(observation["observation_sha256"])
        for index, observation in enumerate(observations)
    }


def _derive_provider_decision(
    plan: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    declared_stop: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked = validate_plan(plan)
    indexed = {}
    for obs in observations:
        key = (obs.get("triad_index"), obs.get("arm"))
        if key in indexed:
            raise Wave5BenchmarkError("duplicate observation", code="wave5.matrix")
        indexed[key] = validate_observation(
            obs, plan=checked, triad_index=key[0], arm=key[1]
        )
    expected = {
        (index, arm) for index, order in enumerate(TRIAD_ORDERS) for arm in order
    }
    stop = None if declared_stop is None else _validate_declared_stop(declared_stop)
    gpu_execution_contract = checked["gpu_execution_contract"]
    performance_promotion_eligible = bool(
        gpu_execution_contract["performance_promotion_eligible"]
    )
    if not set(indexed).issubset(expected):
        raise Wave5BenchmarkError("matrix coordinates are invalid", code="wave5.matrix")
    coordinates = [
        (index, arm) for index, order in enumerate(TRIAD_ORDERS) for arm in order
    ]
    if list(indexed) != coordinates[: len(indexed)]:
        raise Wave5BenchmarkError("matrix is not an exact prefix", code="wave5.matrix")
    complete = set(indexed) == expected
    if not complete and stop is None:
        raise Wave5BenchmarkError("matrix is incomplete", code="wave5.matrix")
    if complete and stop is not None and stop["stage"] != "decision":
        raise Wave5BenchmarkError(
            "complete matrix cannot declare an early stop", code="wave5.stop"
        )
    if not complete and stop is not None and stop["stage"] != "arm_execution":
        raise Wave5BenchmarkError(
            "incomplete matrix stop stage is invalid", code="wave5.stop"
        )
    reference = next(
        (obs for (index, arm), obs in indexed.items() if arm == REFERENCE_ARM), None
    )
    reference_semantics = None if reference is None else reference["semantic"]
    identity_fields = (
        "non_provider_config_sha256",
        "model_config_sha256",
        "cache_binding_sha256",
        "dependency_provenance_sha256",
        "non_provider_policy_identities_sha256",
        "materializations_sha256",
    )
    semantic_green = reference is not None and all(
        obs["semantic"] == reference_semantics for obs in indexed.values()
    )
    identity_green = reference is not None and all(
        all(
            obs["execution_identity"][field] == reference["execution_identity"][field]
            for field in identity_fields
        )
        for obs in indexed.values()
    )
    resource_green = bool(indexed) and all(
        obs["resources"]["summary"]["resource_gate_passed"] is True
        for obs in indexed.values()
    )
    failed_coordinate = None if complete else coordinates[len(indexed)]
    expected_stop_evidence = None
    if stop is not None and stop["stage"] == "decision":
        expected_stop_evidence = _completed_matrix_evidence(
            [indexed[key] for key in coordinates]
        )
    elif stop is not None:
        expected_stop_evidence = _completed_arm_evidence(
            checked, failed_coordinate[0], failed_coordinate[1]
        )
    if stop is not None and stop["completed_evidence"] != expected_stop_evidence:
        raise Wave5BenchmarkError(
            "declared stop completed evidence changed", code="wave5.stop"
        )
    failures = []
    if stop is not None:
        failure = dict(stop)
        if failed_coordinate is not None:
            failure.update(
                {"triad_index": failed_coordinate[0], "arm": failed_coordinate[1]}
            )
        failures.append(failure)
    measurement_summary = _measurement_summary(indexed, arm_failures=failures)
    dispositions = {}
    promotable = []
    for arm in CANDIDATE_ARMS:
        paired_indices = [
            index
            for index in range(3)
            if (index, "R") in indexed and (index, arm) in indexed
        ]
        gains = [
            (_step_wall(indexed[(index, "R")]) - _step_wall(indexed[(index, arm)]))
            / _step_wall(indexed[(index, "R")])
            for index in paired_indices
        ]
        steady_gain = statistics.median(gains) if gains else 0.0
        steady_gain_mad = (
            statistics.median(abs(value - steady_gain) for value in gains)
            if gains
            else 0.0
        )
        steady_noise = max(STEADY_NOISE_FLOOR_FRACTION, 2 * steady_gain_mad)
        e2e_gains = [
            (
                indexed[(index, "R")]["timing"]["entry_to_terminal_seconds"]
                - indexed[(index, arm)]["timing"]["entry_to_terminal_seconds"]
            )
            / indexed[(index, "R")]["timing"]["entry_to_terminal_seconds"]
            for index in paired_indices
        ]
        e2e_gain = statistics.median(e2e_gains) if e2e_gains else 0.0
        e2e_gain_mad = (
            statistics.median(abs(value - e2e_gain) for value in e2e_gains)
            if e2e_gains
            else 0.0
        )
        e2e_noise = max(STEADY_NOISE_FLOOR_FRACTION, 2 * e2e_gain_mad)
        median_e2e_regression = -e2e_gain
        e2e_nonregression = (
            bool(e2e_gains) and median_e2e_regression <= E2E_REGRESSION_LIMIT_FRACTION
        )
        okay = (
            performance_promotion_eligible
            and complete
            and stop is None
            and semantic_green
            and identity_green
            and resource_green
            and len(gains) >= MIN_PAIRED_OBSERVATIONS
            and steady_gain > steady_noise
            and e2e_gain > e2e_noise
            and e2e_nonregression
        )
        if stop is not None and stop["category"] == "resource-stop":
            disposition = "resource-stop"
        elif stop is not None:
            disposition = "inconclusive"
        elif not complete:
            disposition = "inconclusive"
        elif not performance_promotion_eligible:
            disposition = "nonpromotional_shared_execution"
        elif (
            not semantic_green
            or not identity_green
            or not resource_green
            or steady_gain <= 0
            or e2e_gain <= 0
            or not e2e_nonregression
        ):
            disposition = "rejected"
        elif okay:
            disposition = "experimental"
        else:
            disposition = "inconclusive"
        dispositions[arm] = {
            "paired_observation_count": len(gains),
            "median_paired_gain_fraction": e2e_gain,
            "paired_gain_mad_fraction": e2e_gain_mad,
            "frozen_noise_fraction": e2e_noise,
            "clears_frozen_noise": e2e_gain > e2e_noise,
            "median_paired_steady_state_gain_fraction": steady_gain,
            "paired_steady_state_gain_mad_fraction": steady_gain_mad,
            "frozen_steady_state_noise_fraction": steady_noise,
            "clears_frozen_steady_state_noise": steady_gain > steady_noise,
            "median_paired_entry_to_terminal_gain_fraction": e2e_gain,
            "paired_entry_to_terminal_gain_mad_fraction": e2e_gain_mad,
            "frozen_entry_to_terminal_noise_fraction": e2e_noise,
            "clears_frozen_entry_to_terminal_noise": e2e_gain > e2e_noise,
            "median_entry_to_terminal_regression_fraction": median_e2e_regression,
            "entry_to_terminal_regression_limit_fraction": (
                E2E_REGRESSION_LIMIT_FRACTION
            ),
            "entry_to_terminal_nonregression_passed": e2e_nonregression,
            "status": disposition,
            "paired_measurements": deepcopy(
                measurement_summary["paired_comparisons"].get(arm)
            ),
        }
        if okay:
            promotable.append(arm)
    recommended = max(
        promotable,
        key=lambda code: dispositions[code][
            "median_paired_entry_to_terminal_gain_fraction"
        ],
        default="R",
    )
    if not complete or stop is not None:
        recommended = "R" if reference is not None else None
    return _finalize(
        {
            "schema": DECISION_SCHEMA,
            "status": (
                "decided"
                if complete and stop is None and performance_promotion_eligible
                else "decided_non_promoting"
                if complete and stop is None
                else "complete_non_promoting"
            ),
            "plan_sha256": checked["plan_sha256"],
            "gpu_execution_policy": gpu_execution_contract["policy"],
            "execution_evidence_scope": gpu_execution_contract["claim_scope"],
            "performance_promotion_eligible": performance_promotion_eligible,
            "performance_claims_allowed": performance_promotion_eligible,
            "semantic_equivalence_passed": semantic_green,
            "identity_parity_passed": identity_green,
            "resource_gate_passed": resource_green,
            "measurement_summary": measurement_summary,
            "declared_stop": stop,
            "recommendation_measurable": recommended is not None,
            "candidate_dispositions": dispositions,
            "recommended_default": recommended,
            "recommended_provider_mode": (
                "unmeasurable"
                if recommended is None
                else ARM_SPECS[recommended]["provider_mode"]
            ),
            "observation_sha256": [
                indexed[key]["observation_sha256"]
                for key in coordinates[: len(indexed)]
            ],
        },
        hash_field="decision_sha256",
    )


def decide_provider(
    plan: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    declared_stop: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _derive_provider_decision(plan, observations, declared_stop=declared_stop)


def validate_decision(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decision = _validate_finalized(
        payload, schema=DECISION_SCHEMA, hash_field="decision_sha256"
    )
    expected = _derive_provider_decision(
        plan, observations, declared_stop=decision.get("declared_stop")
    )
    if decision != expected:
        raise Wave5BenchmarkError(
            "decision is not derived from the authenticated matrix",
            code="wave5.decision",
        )
    return decision


def _descendant_process_groups(root_pid: int) -> set[int]:
    pending = [int(root_pid)]
    seen: set[int] = set()
    groups: set[int] = {int(root_pid)}
    current_group = os.getpgrp()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            group = os.getpgid(pid)
            if group != current_group:
                groups.add(group)
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, UnicodeError) as exc:
            raise Wave5BenchmarkError(
                "process-tree termination inventory unavailable",
                code="wave5.termination",
            ) from exc
        pending.extend(int(value) for value in children.split())
    groups.discard(current_group)
    return groups


def _live_descendant_pids(root_pid: int) -> set[int]:
    pending = [int(root_pid)]
    seen: set[int] = set()
    descendants: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, UnicodeError) as exc:
            raise Wave5BenchmarkError(
                "process-tree descendant inventory unavailable",
                code="wave5.termination",
            ) from exc
        child_pids = {int(value) for value in children.split()}
        descendants.update(child_pids)
        pending.extend(child_pids)
    descendants.discard(int(root_pid))
    return descendants


def _pid_is_live(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (OSError, UnicodeError):
        return True
    return not any(
        line.startswith("State:") and "Z" in line for line in status.splitlines()
    )


def _signal_process_groups(groups: Sequence[int], signum: int) -> None:
    for group in sorted(set(groups)):
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            continue


def _terminate(process: subprocess.Popen[Any]) -> dict[str, Any]:
    inventory_error: dict[str, str] | None = None
    try:
        captured_descendants = _live_descendant_pids(process.pid)
        groups = _descendant_process_groups(process.pid)
    except BaseException as exc:
        captured_descendants = set()
        groups = {int(process.pid)}
        inventory_error = {
            "type": type(exc).__name__[:128],
            "code": str(getattr(exc, "code", "wave5.termination"))[:128],
        }
    groups.add(int(process.pid))
    signals_sent = ["SIGTERM"]
    _signal_process_groups(groups, signal.SIGTERM)
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            groups.update(_descendant_process_groups(process.pid))
        except BaseException:
            pass
        signals_sent.append("SIGKILL")
        _signal_process_groups(groups, signal.SIGKILL)
        try:
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    remaining = sorted(pid for pid in captured_descendants if _pid_is_live(pid))
    try:
        remaining = sorted(set(remaining) | _live_descendant_pids(process.pid))
    except BaseException:
        if inventory_error is None:
            inventory_error = {
                "type": "Wave5BenchmarkError",
                "code": "wave5.termination",
            }
    if remaining:
        if "SIGKILL" not in signals_sent:
            signals_sent.append("SIGKILL")
        _signal_process_groups(groups, signal.SIGKILL)
        remaining = sorted(pid for pid in remaining if _pid_is_live(pid))
    return {
        "captured_process_groups": sorted(groups),
        "captured_descendant_pids": sorted(captured_descendants),
        "signals": signals_sent,
        "inventory_error": inventory_error,
        "descendant_attestation": (
            "passed"
            if inventory_error is None and not remaining
            else "failed"
            if remaining
            else "unavailable"
        ),
        "descendants_remaining": remaining,
        "root_returncode": (
            process.poll() if callable(getattr(process, "poll", None)) else None
        ),
    }


def _execute_arm(
    plan: Mapping[str, Any], triad_index: int, arm: str, *, timeout_seconds: float
) -> None:
    argv = _arm_argv(plan, triad_index, arm)
    if argv != _arm_argv(validate_plan(plan), triad_index, arm):
        raise Wave5BenchmarkError(
            "generated argv authentication failed", code="wave5.command"
        )
    process = subprocess.Popen(argv, cwd=REPO_ROOT, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate(process)
        raise Wave5BenchmarkError(
            "arm exceeded hard timeout", code="wave5.timeout"
        ) from exc
    if returncode:
        resource_path = Path(_arm_targets(plan, triad_index, arm)["resource_samples"])
        if resource_path.is_file():
            resource = _validate_resource_receipt(
                load_strict_json(resource_path), plan=plan
            )
            monitor_error = resource.get("monitor_error")
            if isinstance(monitor_error, Mapping) and monitor_error.get("code"):
                raise Wave5BenchmarkError(
                    str(monitor_error.get("message", "production arm failed")),
                    code=str(monitor_error["code"]),
                )
            if resource.get("status") == "resource_limit_exceeded":
                raise Wave5BenchmarkError(
                    "production arm crossed a declared resource ceiling",
                    code="wave5.resource_ceiling",
                )
        raise Wave5BenchmarkError("production arm failed", code="wave5.arm_failed")


def run_controller(plan_path: str | Path) -> dict[str, Any]:
    plan = validate_plan(load_strict_json(plan_path))
    if (
        str(_canonical_nonsymlink(plan_path, must_exist=True))
        != plan["artifact_targets"]["plan"]
    ):
        raise Wave5BenchmarkError("plan loaded from wrong path", code="wave5.plan_path")
    targets = plan["artifact_targets"]
    for path in _all_target_paths(targets):
        if str(path) != targets["plan"] and path.exists():
            raise Wave5BenchmarkError(
                "immutable target collision", code="wave5.artifact_collision"
            )
    _revalidate_plan_compatibility(plan)
    attempt = build_attempt(plan)
    publish_json_absent(targets["attempt"], attempt)
    attempt = load_authenticated_attempt(plan, targets["attempt"])
    started = time.monotonic()
    observations = []
    matrix_coordinates = [
        (index, arm) for index, order in enumerate(TRIAD_ORDERS) for arm in order
    ]
    terminal_status = "failed"
    error: dict[str, str] | None = None
    decision = None
    decision_started = False
    gpu_baseline: dict[str, Any] | None = None
    terminal_gpu_process_sweep: dict[str, Any] | None = None
    try:
        _revalidate_execution_baseline(plan)
        gpu_baseline = _stable_gpu_execution_baseline(plan)
        publish_json_absent(targets["gpu_baseline"], gpu_baseline)
        load_authenticated_gpu_baseline(plan)
        for triad_index, order in enumerate(TRIAD_ORDERS):
            for arm in order:
                remaining = MATRIX_WALL_CEILING_SECONDS - (time.monotonic() - started)
                if remaining <= 0:
                    raise Wave5BenchmarkError(
                        "matrix wall ceiling exceeded", code="wave5.matrix_timeout"
                    )
                _execute_arm(
                    plan,
                    triad_index,
                    arm,
                    timeout_seconds=min(ARM_TIMEOUT_SECONDS, remaining),
                )
                observation = derive_observation(plan, triad_index=triad_index, arm=arm)
                publish_json_absent(
                    _arm_targets(plan, triad_index, arm)["observation"], observation
                )
                observations.append(observation)
        decision_started = True
        terminal_gpu_process_sweep = _post_phase_gpu_process_sweep(
            gpu_baseline,
            prior_bindings={},
            field="controller.pre_decision",
        )
        if terminal_gpu_process_sweep["status"] == "sampling_error":
            sampling_error = terminal_gpu_process_sweep["sampling_error"]
            raise Wave5BenchmarkError(
                sampling_error["message"], code=sampling_error["code"]
            )
        if terminal_gpu_process_sweep["status"] == "failed":
            raise Wave5BenchmarkError(
                "nonbaseline GPU compute process survived the matrix",
                code="wave5.gpu_process_leak",
            )
        decision = decide_provider(plan, observations)
        publish_json_absent(targets["decision"], decision)
        terminal_status = "completed"
    except Exception as exc:
        error = {
            "type": type(exc).__name__[:128],
            "code": str(getattr(exc, "code", "wave5.unexpected"))[:128],
            "message": str(exc)[:4096],
        }
        if gpu_baseline is not None and terminal_gpu_process_sweep is None:
            try:
                terminal_gpu_process_sweep = _post_phase_gpu_process_sweep(
                    gpu_baseline,
                    prior_bindings={},
                    field="controller.failure_terminal",
                )
                if terminal_gpu_process_sweep["status"] == "sampling_error":
                    error = dict(terminal_gpu_process_sweep["sampling_error"])
                elif terminal_gpu_process_sweep["status"] == "failed":
                    error = {
                        "type": "Wave5BenchmarkError",
                        "code": "wave5.gpu_process_leak",
                        "message": (
                            "nonbaseline GPU compute process survived controller cleanup"
                        ),
                    }
            except Exception as terminal_sample_error:
                error = {
                    "type": type(terminal_sample_error).__name__[:128],
                    "code": str(
                        getattr(
                            terminal_sample_error,
                            "code",
                            "wave5.gpu_process_sample",
                        )
                    )[:128],
                    "message": str(terminal_sample_error)[:4096],
                }
        resource_stop_codes = {
            "wave5.resource_ceiling",
            "wave5.resource_watchdog",
            "wave5.gpu_busy",
            "wave5.timeout",
            "wave5.matrix_timeout",
            "wave5.gpu_sample",
            "wave5.gpu_process_sample",
            "wave5.gpu_process_ownership",
            "wave5.foreign_gpu_process",
            "wave5.gpu_process_leak",
            "wave5.gpu_baseline",
            "wave5.gpu_baseline_drift",
            "wave5.termination",
        }
        stop_category = (
            "resource-stop"
            if error["code"] in resource_stop_codes
            or error["code"].startswith("wave5.resource")
            else "failure"
        )
        decision_stage = decision_started and len(observations) == len(
            matrix_coordinates
        )
        decision = _derive_provider_decision(
            plan,
            observations,
            declared_stop={
                "category": stop_category,
                "stage": "decision" if decision_stage else "arm_execution",
                "error": error,
                "completed_evidence": (
                    _completed_matrix_evidence(observations)
                    if decision_stage
                    else _completed_arm_evidence(
                        plan, *matrix_coordinates[len(observations)]
                    )
                ),
            },
        )
        publish_json_absent(targets["decision"], decision)
        terminal_status = (
            stop_category if stop_category == "resource-stop" else "failed"
        )
    terminal = _finalize(
        {
            "schema": TERMINAL_SCHEMA,
            "status": terminal_status,
            "finished_at": _utc_now(),
            "plan_path": targets["plan"],
            "plan_sha256": plan["plan_sha256"],
            "attempt_path": targets["attempt"],
            "attempt_sha256": attempt["attempt_sha256"],
            "attempt_file_sha256": sha256_file(targets["attempt"]),
            "authenticated_commands_sha256": sha256_json(attempt["commands"]),
            "gpu_baseline_sha256": (
                None if gpu_baseline is None else gpu_baseline["gpu_baseline_sha256"]
            ),
            "terminal_gpu_process_sweep": terminal_gpu_process_sweep,
            "observation_sha256": [obs["observation_sha256"] for obs in observations],
            "decision_sha256": None
            if decision is None
            else decision["decision_sha256"],
            "error": error,
        },
        hash_field="terminal_sha256",
    )
    try:
        publish_json_absent(targets["terminal"], terminal)
    except Exception as publication_error:
        sidecar = _finalize(
            {
                "schema": PUBLICATION_FAILURE_SCHEMA,
                "status": "terminal_publication_failed",
                "created_at": _utc_now(),
                "plan_path": targets["plan"],
                "plan_sha256": plan["plan_sha256"],
                "attempt_path": targets["attempt"],
                "attempt_sha256": attempt["attempt_sha256"],
                "intended_terminal_sha256": terminal["terminal_sha256"],
                "error_type": type(publication_error).__name__,
                "error_code": getattr(publication_error, "code", "wave5.publish"),
            },
            hash_field="publication_failure_sha256",
        )
        publish_json_absent(targets["terminal_publication_failure"], sidecar)
        raise
    validate_terminal(load_strict_json(targets["terminal"]), plan=plan)
    if error is not None:
        raise Wave5BenchmarkError(error["message"], code=error["code"])
    return terminal


def validate_terminal(
    payload: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_plan(plan)
    terminal = _validate_finalized(
        payload, schema=TERMINAL_SCHEMA, hash_field="terminal_sha256"
    )
    if set(terminal) != {
        "schema",
        "status",
        "finished_at",
        "plan_path",
        "plan_sha256",
        "attempt_path",
        "attempt_sha256",
        "attempt_file_sha256",
        "authenticated_commands_sha256",
        "gpu_baseline_sha256",
        "terminal_gpu_process_sweep",
        "observation_sha256",
        "decision_sha256",
        "error",
        "terminal_sha256",
    }:
        raise Wave5BenchmarkError("terminal schema is not exact", code="wave5.terminal")
    _parse_attested_at(terminal.get("finished_at"))
    attempt_path = terminal.get("attempt_path")
    attempt = load_authenticated_attempt(checked, attempt_path)
    if (
        terminal.get("plan_path") != checked["artifact_targets"]["plan"]
        or terminal.get("plan_sha256") != checked["plan_sha256"]
        or terminal.get("attempt_sha256") != attempt["attempt_sha256"]
        or terminal.get("attempt_file_sha256") != sha256_file(attempt_path)
        or terminal.get("authenticated_commands_sha256")
        != sha256_json(attempt["commands"])
    ):
        raise Wave5BenchmarkError(
            "terminal authentication failed", code="wave5.terminal"
        )
    baseline_path = checked["artifact_targets"]["gpu_baseline"]
    terminal_gpu_process_sweep = terminal.get("terminal_gpu_process_sweep")
    if Path(baseline_path).is_file():
        baseline = load_authenticated_gpu_baseline(checked, baseline_path)
        if terminal.get("gpu_baseline_sha256") != baseline["gpu_baseline_sha256"]:
            raise Wave5BenchmarkError(
                "terminal GPU baseline hash changed", code="wave5.terminal"
            )
        terminal_ownership_samples, sweep_status = _validate_post_phase_sweep_envelope(
            terminal_gpu_process_sweep,
            field="terminal.gpu_process_sweep",
            termination=None,
        )
        expected_terminal_samples = []
        for sample_index, sample in enumerate(terminal_ownership_samples):
            expected_sample, _ = _replay_gpu_process_ownership_sample(
                sample.get("compute_apps"),
                expected_gpu_uuids=[
                    row["gpu_uuid"] for row in baseline["gpu_inventory"]
                ],
                baseline_compute_apps=baseline["baseline_compute_apps"],
                prior_bindings={},
                allow_owned_compute_apps=False,
                field=f"terminal.gpu_process_sweep.samples[{sample_index}]",
            )
            expected_terminal_samples.append(expected_sample)
        expected_sweep_status = (
            "passed"
            if all(sample["status"] == "passed" for sample in expected_terminal_samples)
            else "failed"
        )
        if (
            terminal_ownership_samples != expected_terminal_samples
            or (
                sweep_status != "sampling_error"
                and sweep_status != expected_sweep_status
            )
            or (
                sweep_status == "sampling_error"
                and terminal.get("error")
                != terminal_gpu_process_sweep.get("sampling_error")
            )
        ):
            raise Wave5BenchmarkError(
                "terminal GPU process sweep is not derived", code="wave5.terminal"
            )
    elif (
        terminal.get("gpu_baseline_sha256") is not None
        or terminal_gpu_process_sweep is not None
    ):
        raise Wave5BenchmarkError(
            "terminal references an absent GPU baseline", code="wave5.terminal"
        )
    observation_sha256 = terminal.get("observation_sha256")
    if not isinstance(observation_sha256, list) or len(observation_sha256) > 9:
        raise Wave5BenchmarkError(
            "terminal observation chain is malformed", code="wave5.terminal"
        )
    coordinates = [
        (index, arm) for index, order in enumerate(TRIAD_ORDERS) for arm in order
    ]
    observations = []
    for expected_sha256, (triad_index, arm) in zip(
        observation_sha256, coordinates, strict=False
    ):
        observation_path = _arm_targets(checked, triad_index, arm)["observation"]
        if not Path(observation_path).is_file():
            raise Wave5BenchmarkError(
                "terminal observation artifact is absent", code="wave5.terminal"
            )
        observation = validate_observation(
            load_strict_json(observation_path),
            plan=checked,
            triad_index=triad_index,
            arm=arm,
        )
        if observation["observation_sha256"] != expected_sha256:
            raise Wave5BenchmarkError(
                "terminal observation hash chain changed", code="wave5.terminal"
            )
        observations.append(observation)
    status = terminal.get("status")
    error = terminal.get("error")
    decision_sha256 = terminal.get("decision_sha256")
    if status == "completed":
        if (
            error is not None
            or len(observations) != len(coordinates)
            or not isinstance(terminal_gpu_process_sweep, Mapping)
            or terminal_gpu_process_sweep.get("status") != "passed"
        ):
            raise Wave5BenchmarkError(
                "completed terminal has error or incomplete observations",
                code="wave5.terminal_error",
            )
        if not _is_sha256(decision_sha256):
            raise Wave5BenchmarkError(
                "completed terminal decision hash missing", code="wave5.terminal"
            )
        decision_path = checked["artifact_targets"]["decision"]
        if not Path(decision_path).is_file():
            raise Wave5BenchmarkError(
                "completed terminal decision artifact absent", code="wave5.terminal"
            )
        decision = validate_decision(
            load_strict_json(decision_path), plan=checked, observations=observations
        )
        if decision["decision_sha256"] != decision_sha256:
            raise Wave5BenchmarkError(
                "terminal decision hash chain changed", code="wave5.terminal"
            )
    elif status in {"failed", "resource-stop"}:
        error_lengths_valid = isinstance(error, Mapping) and (
            len(str(error.get("type", ""))) <= 128
            and len(str(error.get("code", ""))) <= 128
            and len(str(error.get("message", ""))) <= 4096
        )
        error_code_valid = isinstance(error, Mapping) and all(
            character.islower() or character.isdigit() or character in "_.-"
            for character in str(error.get("code", ""))
        )
        if (
            not isinstance(error, Mapping)
            or set(error) != {"type", "code", "message"}
            or not all(
                isinstance(error.get(field), str) and error[field] for field in error
            )
            or not error_lengths_valid
            or not error_code_valid
        ):
            raise Wave5BenchmarkError(
                "failed terminal error or prefix is invalid",
                code="wave5.terminal_error",
            )
        if not _is_sha256(decision_sha256):
            raise Wave5BenchmarkError(
                "stopped terminal decision hash missing", code="wave5.terminal"
            )
        decision_path = checked["artifact_targets"]["decision"]
        if not Path(decision_path).is_file():
            raise Wave5BenchmarkError(
                "stopped terminal decision artifact absent", code="wave5.terminal"
            )
        decision = validate_decision(
            load_strict_json(decision_path), plan=checked, observations=observations
        )
        if decision["decision_sha256"] != decision_sha256:
            raise Wave5BenchmarkError(
                "terminal decision hash chain changed", code="wave5.terminal"
            )
        expected_category = "resource-stop" if status == "resource-stop" else "failure"
        if decision.get("declared_stop", {}).get("category") != expected_category:
            raise Wave5BenchmarkError(
                "terminal and decision stop categories differ", code="wave5.terminal"
            )
        if decision.get("declared_stop", {}).get("error") != error:
            raise Wave5BenchmarkError(
                "terminal and decision errors differ", code="wave5.terminal"
            )
        expected_stage = (
            "decision" if len(observations) == len(coordinates) else "arm_execution"
        )
        if decision.get("declared_stop", {}).get("stage") != expected_stage:
            raise Wave5BenchmarkError(
                "terminal observation count and stop stage differ",
                code="wave5.terminal",
            )
    else:
        raise Wave5BenchmarkError("terminal status is invalid", code="wave5.terminal")
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    compatibility = sub.add_parser("attest-compatibility")
    compatibility.add_argument("--historical-cache-root", required=True)
    compatibility.add_argument("--cache-root", required=True)
    compatibility.add_argument("--train-fingerprint", required=True)
    compatibility.add_argument("--eval-fingerprint", required=True)
    compatibility.add_argument("--output", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--cache-root", required=True)
    prepare.add_argument("--compatibility-attestation", required=True)
    prepare.add_argument(
        "--gpu-execution-policy",
        choices=GPU_EXECUTION_POLICIES,
        default="idle_promotional",
    )
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    arm = sub.add_parser("arm")
    arm.add_argument("--plan", required=True)
    arm.add_argument("--attempt-marker", required=True)
    arm.add_argument("--triad-index", required=True, type=int)
    arm.add_argument("--arm", required=True, choices=tuple(ARM_SPECS))
    instrumented = sub.add_parser("instrumented-train")
    instrumented.add_argument("--config", required=True)
    instrumented.add_argument("--stream-output-root", required=True)
    instrumented.add_argument(
        "--comparison-arm", required=True, choices=tuple(ARM_SPECS)
    )
    instrumented.add_argument("--warmup-exclusion-steps", required=True, type=int)
    instrumented.add_argument("--wall-clock-scope", required=True)
    instrumented.add_argument("--profile-sync-timings", required=True)
    instrumented.add_argument("--measurement-contract-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "attest-compatibility":
        build_and_publish_compatibility_attestation(
            historical_cache_root=args.historical_cache_root,
            cache_root=args.cache_root,
            train_fingerprint=args.train_fingerprint,
            eval_fingerprint=args.eval_fingerprint,
            output_path=args.output,
        )
        return 0
    if args.command == "prepare":
        root = _canonical_nonsymlink(args.root, must_exist=True)
        plan = build_frozen_plan(
            root=root,
            cache_root=args.cache_root,
            compatibility_attestation_path=args.compatibility_attestation,
            gpu_execution_policy=args.gpu_execution_policy,
        )
        publish_json_absent(plan["artifact_targets"]["plan"], plan)
        return 0
    if args.command == "run":
        run_controller(args.plan)
        return 0
    if args.command == "instrumented-train":
        return run_instrumented_train(
            args.config,
            args.stream_output_root,
            comparison_arm=args.comparison_arm,
            warmup_exclusion_steps=args.warmup_exclusion_steps,
            wall_clock_scope=args.wall_clock_scope,
            profile_sync_timings=args.profile_sync_timings,
            measurement_contract_sha256=args.measurement_contract_sha256,
        )
    return run_production_arm(
        args.plan, args.attempt_marker, args.triad_index, args.arm
    )


if __name__ == "__main__":
    raise SystemExit(main())
