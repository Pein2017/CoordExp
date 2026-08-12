#!/usr/bin/env python3
"""Authenticated CPU-only Wave 6 comparison of current PackPlan policies.

``prepare`` binds the current integration owners and immutable historical W0-CPU
encoded stream into an absent plan. ``run`` performs three fresh-process paired
worker comparisons over the frozen five-arm grid without training/model/GPU/cache
writes and publishes one absent terminal receipt. Planner utilization is
descriptive only: changed-order candidates require a matched five-step training
comparison.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import (
    dataclass,
    fields as dataclass_fields,
    is_dataclass,
    make_dataclass,
)
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import re
import resource
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.loader import load_train_config  # noqa: E402
from src.packing.planner import (  # noqa: E402
    DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
    DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
    DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
    ONLINE_WINDOW_BINPACK,
    PACK_PLAN_CURSOR_SCHEMA,
    PACK_PLAN_CURSOR_SCHEMA_VERSION,
    PACK_PLAN_SCHEMA,
    PACK_PLAN_SCHEMA_VERSION,
    PACK_PLAN_STREAM_RECEIPT_SCHEMA,
    PACK_PLAN_STREAM_RECEIPT_SCHEMA_VERSION,
    SOURCE_ORDER_NEXT_FIT,
    WINDOW_BINPACK,
    PackPlan,
    PackPlanCursor,
    PackPlanStreamReceipt,
    build_pack_plan_policy_identity,
    create_pack_plan,
    replay_pack_plan,
    stream_online_pack_plan_fragments,
    verify_pack_plan_stream_fragments,
)
from src.qwen.parity import (  # noqa: E402
    ParityContractError,
    assert_absent_artifact_target,
    write_strict_json_atomic,
)
import src.packing.supervision as packing_supervision  # noqa: E402
import src.qwen.images as qwen_images  # noqa: E402
from src.training import pack_cache  # noqa: E402
import src.training.pipeline as training_pipeline  # noqa: E402


PLAN_SCHEMA = "coordexp-swift-wave6-pack-plan-comparison-plan-v1"
RECEIPT_SCHEMA = "coordexp-swift-wave6-pack-plan-comparison-receipt-v1"
OBSERVATION_SCHEMA = "coordexp-swift-wave6-pack-plan-observation-v1"
CONTROLLER_RECEIPT_SCHEMA = "coordexp-swift-wave6-pack-plan-controller-receipt-v1"
FAILURE_RECEIPT_SCHEMA = "coordexp-swift-wave6-pack-plan-failure-receipt-v1"
RESEARCH_MEANING_SCHEMA = "coordexp-swift-wave6-pack-plan-research-meaning-v1"
ENCODED_STREAM_SCHEMA = "coordexp-swift-wave0-train-encoded-stream-v1"
SOURCE_BINDING_SCHEMA = "coordexp-swift-wave6-source-owners-v1"
CONFIG_BINDING_SCHEMA = "coordexp-swift-wave6-config-binding-v1"
W0_BINDING_SCHEMA = "coordexp-swift-wave0-cache-binding-v1"
PRODUCTION_INTEGRATION_SCHEMA = "coordexp-swift-wave6-production-integration-v1"
SOURCE_QUIESCENCE_SCHEMA = "coordexp-swift-wave6-source-quiescence-v1"

R2_RESEARCH_MEANING_BINDING = {
    "plan_sha256": ("e890e6a60ebf25e56f928fb1f7d96f602c2000787d1d4b7133acd1d4f78d1dab"),
    "plan_file_sha256": (
        "6e358950ab5b14ef94e493ddda57d955dd5e4200d6466483529380b1e4eb1478"
    ),
    "controller_receipt_sha256": (
        "d243c0d3af1b490f6ed768fad42f733e89d03eb010969a3dc3dc92804a0ec587"
    ),
    "controller_receipt_file_sha256": (
        "e6b2bfc6c50a904b1f2c2910ac27c68d9d8fbda75b54440157b5b83febdc5888"
    ),
}

BASE_CONFIG_PATH = (
    REPO_ROOT / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_"
    "llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
).resolve()
SCRIPT_PATH = Path(__file__).resolve()

ARM_GRID: tuple[Mapping[str, Any], ...] = (
    {
        "arm_id": "source_order_next_fit",
        "policy": SOURCE_ORDER_NEXT_FIT,
        "window_size": None,
        "lookahead": None,
    },
    {
        "arm_id": "window_binpack_w8",
        "policy": WINDOW_BINPACK,
        "window_size": 8,
        "lookahead": None,
    },
    {
        "arm_id": "window_binpack_w32",
        "policy": WINDOW_BINPACK,
        "window_size": 32,
        "lookahead": None,
    },
    {
        "arm_id": "online_window_binpack_l8",
        "policy": ONLINE_WINDOW_BINPACK,
        "window_size": None,
        "lookahead": 8,
    },
    {
        "arm_id": "online_window_binpack_l32",
        "policy": ONLINE_WINDOW_BINPACK,
        "window_size": None,
        "lookahead": 32,
    },
)

WORKER_COUNTS = (1, 8)
STREAM_FRAGMENT_PACK_BUDGET = 7
RESUME_FRAGMENT_PACK_BUDGET = 7
MAX_FRAGMENT_COUNT = 1_024
MAX_INPUT_COUNT = 10_000
MAX_ERROR_CHARS = 1_024
CHILD_TIMEOUT_SECONDS = 15 * 60
POLICY_PAIR_ORDERS = (
    (SOURCE_ORDER_NEXT_FIT, "candidate"),
    ("candidate", SOURCE_ORDER_NEXT_FIT),
    (SOURCE_ORDER_NEXT_FIT, "candidate"),
)
CANDIDATE_GRID = ARM_GRID[1:]
HISTORICAL_W0_CONFIG_FINGERPRINT = (
    "de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d"
)
HISTORICAL_W0_SPLITS = {
    "train": {
        "fingerprint": (
            "c6be15b8d7840524829c70e1fa5729accf600ae366d97fccf32778eca84c7de1"
        ),
        "manifest_sha256": (
            "636c701efd1cf94af0f88bdb1d86fcef0529d47609c2b48fbf433a7172da7a4e"
        ),
        "chunk_sha256": (
            "b5becdfc84c6569554af01bbeb41a153ae168b4d3796f8e7209078349f6db665"
        ),
        "micro_step_count": 32,
        "example_count": 256,
    },
    "eval": {
        "fingerprint": (
            "60330b24519f0931e5c84765874c70e0cb5d54cd472bd6f79fc30977edab626c"
        ),
        "manifest_sha256": (
            "d8ca6858a56fb895f3ba037bfa27d96104cde6881aded837ec4473834ecc73bf"
        ),
        "chunk_sha256": (
            "fa8ea831c1912572ab75d132c57da5f1114b18650236971209004c1934b4ce59"
        ),
        "micro_step_count": 8,
        "example_count": 64,
    },
}
FROZEN_W0_TRAIN_STREAM = {
    "input_count": 256,
    "encoded_length_sum": 357_559,
    "encoded_inputs_sha256": (
        "0da569af3fc744df3a3eb54260b25cc81bbbad8195ed072c23e042f30b0744c6"
    ),
    "encoded_lengths_sha256": (
        "f452f3a4630e85f7fdd5ca5c4a54bd44b5d0640d4f52b3ec4d6208b056f806ba"
    ),
    "intra_image_order_identities_sha256": (
        "bd48842c1522067442055e536fcfa415ba83a6a560dbd635090db3a5c4fe0196"
    ),
    "example_ids_sha256": (
        "590336f48bf369e3b217900bbeb9b4a40a3c666408cce4afdd8a837f90d7d6a1"
    ),
}


class Wave6ProbeError(RuntimeError):
    """Typed fail-closed Wave 6 probe error."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class Wave6ChildProcessError(Wave6ProbeError):
    """Typed child failure whose bounded message preserves the stderr tail."""


@dataclass(frozen=True)
class ProbeRawExample:
    example_id: str
    input_ids: tuple[int, ...]
    row_ids: tuple[str, ...]
    encoded_dataclass_identity: Mapping[str, Any]


@dataclass(frozen=True)
class ProbeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    row_ids: tuple[str, ...]

    @property
    def intra_image_order_identity(self) -> str:
        payload = json.dumps(self.row_ids, separators=(",", ":")).encode()
        return f"test-row-order-sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise Wave6ProbeError("value is not strict JSON", code="wave6.json") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Wave6ProbeError("identity file is unreadable", code="wave6.file") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def finalize_artifact(payload: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    if hash_field in result:
        raise Wave6ProbeError("artifact is already finalized", code="wave6.hash")
    result[hash_field] = sha256_json(result)
    return result


def _validate_finalized(
    payload: Mapping[str, Any], *, schema: str, hash_field: str
) -> dict[str, Any]:
    value = deepcopy(dict(payload))
    if value.get("schema") != schema:
        raise Wave6ProbeError("artifact schema mismatch", code="wave6.schema")
    observed = value.pop(hash_field, None)
    if not _is_sha256(observed) or observed != sha256_json(value):
        raise Wave6ProbeError("artifact hash mismatch", code="wave6.hash")
    value[hash_field] = observed
    return value


def load_strict_json(path: str | Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Wave6ProbeError(
            "cannot read strict JSON", code="wave6.json_read"
        ) from exc
    if not isinstance(payload, dict):
        raise Wave6ProbeError("JSON root is not an object", code="wave6.json_read")
    canonical_json_bytes(payload)
    return payload


def publish_json_absent(path: str | Path, payload: Mapping[str, Any]) -> Path:
    try:
        target = assert_absent_artifact_target(path)
        write_strict_json_atomic(target, payload)
    except ParityContractError as exc:
        code = (
            "wave6.immutable_collision"
            if exc.code == "qwen.parity.artifact_collision"
            else "wave6.artifact_publication"
        )
        raise Wave6ProbeError("artifact publication failed", code=code) from exc
    if load_strict_json(target) != dict(payload):
        raise Wave6ProbeError(
            "published artifact failed exact reload", code="wave6.artifact_persistence"
        )
    return target


def _absent_target(path: str | Path) -> Path:
    try:
        return assert_absent_artifact_target(path)
    except ParityContractError as exc:
        code = (
            "wave6.immutable_collision"
            if exc.code == "qwen.parity.artifact_collision"
            else "wave6.artifact_target"
        )
        raise Wave6ProbeError("artifact target is not absent", code=code) from exc


def _require_exact_fields(
    value: Mapping[str, Any], fields: set[str], *, code: str
) -> None:
    if set(value) != fields:
        raise Wave6ProbeError("artifact fields are not exact", code=code)


def _mapping(value: Any, *, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Wave6ProbeError("expected a mapping", code=code)
    return value


def _source_owners() -> dict[str, Any]:
    paths = {
        "probe": SCRIPT_PATH,
        "planner": REPO_ROOT / "src/packing/planner.py",
        "config_models": REPO_ROOT / "src/config/models.py",
        "config_loader": REPO_ROOT / "src/config/loader.py",
        "qwen_encoding": REPO_ROOT / "src/qwen/encoding.py",
        "qwen_images": REPO_ROOT / "src/qwen/images.py",
        "pack_cache": REPO_ROOT / "src/training/pack_cache.py",
    }
    owners = {
        owner: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for owner, path in paths.items()
    }
    payload = {"schema": SOURCE_BINDING_SCHEMA, "owners": owners}
    return {**owners, "binding_sha256": sha256_json(payload)}


def _symbol_identity(path: Path, symbol: Any) -> dict[str, str]:
    try:
        source = inspect.getsource(symbol)
    except (OSError, TypeError) as exc:
        raise Wave6ProbeError(
            "integration symbol source is unavailable",
            code="wave6.production_integration",
        ) from exc
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "qualname": str(
            getattr(symbol, "__qualname__", getattr(symbol, "__name__", ""))
        ),
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _production_integration_contract() -> dict[str, Any]:
    pipeline_path = REPO_ROOT / "src/training/pipeline.py"
    supervision_path = REPO_ROOT / "src/packing/supervision.py"
    cache_path = REPO_ROOT / "src/training/pack_cache.py"
    qwen_images_path = REPO_ROOT / "src/qwen/images.py"
    pipeline_symbols = {
        name: _symbol_identity(pipeline_path, getattr(training_pipeline, name))
        for name in (
            "_materialize_raw_examples_for_dataset",
            "_build_encoded_examples_for_dataset",
            "_encode_examples_with_fork_process_pool",
            "_render_and_encode_example",
            "_materialize_pack_plan",
            "_build_micro_steps_for_dataset",
        )
    }
    manifest_symbols = {
        name: _symbol_identity(cache_path, getattr(pack_cache, name))
        for name in (
            "write_micro_step_cache",
            "_validate_manifest",
            "_load_validated_manifest",
        )
    }
    payload_symbols = {
        name: _symbol_identity(cache_path, getattr(pack_cache, name))
        for name in (
            "_load_validated_chunk",
            "_read_chunk_snapshot",
            "_RestrictedCacheUnpickler",
        )
    }
    determinant_symbols = {
        name: _symbol_identity(cache_path, getattr(pack_cache, name))
        for name in (
            "build_packing_cache_determinants",
            "_build_determinant_entries",
            "_validate_determinant_registry",
        )
    }
    payload = {
        "schema": PRODUCTION_INTEGRATION_SCHEMA,
        "pipeline_symbols": pipeline_symbols,
        "supervision_owner": {
            "path": str(supervision_path.resolve()),
            "sha256": sha256_file(supervision_path),
            "build_packed_supervision": _symbol_identity(
                supervision_path, packing_supervision.build_packed_supervision
            ),
        },
        "encoded_materialization_serialization": {
            "owner_path": str(qwen_images_path.resolve()),
            "owner_sha256": sha256_file(qwen_images_path),
            "qwen_image_getstate": _symbol_identity(
                qwen_images_path, qwen_images.QwenImageEncoding.__getstate__
            ),
            "canonicalization": ("in_memory_pickle_protocol_5_roundtrip_per_example"),
        },
        "private_v3_cache": {
            "owner_path": str(cache_path.resolve()),
            "owner_sha256": sha256_file(cache_path),
            "cache_version": pack_cache.PACKING_CACHE_VERSION,
            "registry_schema_version": (
                pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION
            ),
            "determinant_owners": dict(pack_cache.PACKING_CACHE_DETERMINANT_OWNERS),
            "determinant_symbols": determinant_symbols,
            "manifest_symbols": manifest_symbols,
            "payload_symbols": payload_symbols,
            "payload_schema": pack_cache._supervised_micro_step_schema_identity(),
        },
    }
    return {**payload, "binding_sha256": sha256_json(payload)}


def _source_state() -> dict[str, Any]:
    payload = {
        "source_owners": _source_owners(),
        "production_integration": _production_integration_contract(),
    }
    return {**payload, "sha256": sha256_json(payload)}


def _validate_production_integration(value: Any) -> dict[str, Any]:
    observed = dict(_mapping(value, code="wave6.production_integration"))
    if observed != _production_integration_contract():
        raise Wave6ProbeError(
            "production integration binding drifted",
            code="wave6.production_integration",
        )
    return observed


def _validate_source_quiescence(value: Any, *, current_sha256: str) -> dict[str, Any]:
    observed = dict(_mapping(value, code="wave6.source_quiescence"))
    expected_fields = {
        "schema",
        "required",
        "before_sha256",
        "after_sha256",
        "stable",
    }
    _require_exact_fields(observed, expected_fields, code="wave6.source_quiescence")
    if observed != {
        "schema": SOURCE_QUIESCENCE_SCHEMA,
        "required": True,
        "before_sha256": current_sha256,
        "after_sha256": current_sha256,
        "stable": True,
    }:
        raise Wave6ProbeError(
            "source was not quiescent across plan preparation",
            code="wave6.source_quiescence",
        )
    return observed


def _validate_source_owners(value: Any) -> dict[str, Any]:
    observed = dict(_mapping(value, code="wave6.source_owner"))
    expected = _source_owners()
    if observed != expected:
        raise Wave6ProbeError("source-owner binding drifted", code="wave6.source_owner")
    return observed


def _pack_plan_contract() -> dict[str, Any]:
    policies = []
    for arm in ARM_GRID:
        policies.append(
            {
                "arm_id": arm["arm_id"],
                "identity": build_pack_plan_policy_identity(
                    policy=str(arm["policy"]),
                    window_size=arm["window_size"],
                    lookahead=arm["lookahead"],
                    seed=17,
                    worker_count=1,
                ),
            }
        )
    return {
        "schema": PACK_PLAN_SCHEMA,
        "schema_version": PACK_PLAN_SCHEMA_VERSION,
        "cursor_schema": PACK_PLAN_CURSOR_SCHEMA,
        "cursor_schema_version": PACK_PLAN_CURSOR_SCHEMA_VERSION,
        "stream_receipt_schema": PACK_PLAN_STREAM_RECEIPT_SCHEMA,
        "stream_receipt_schema_version": PACK_PLAN_STREAM_RECEIPT_SCHEMA_VERSION,
        "policies": policies,
    }


def _config_binding(*, planner_global_max_length: int) -> dict[str, Any]:
    resolved = load_train_config(BASE_CONFIG_PATH)
    config = resolved.config
    payload = {
        "schema": CONFIG_BINDING_SCHEMA,
        "entry_path": str(BASE_CONFIG_PATH),
        "entry_sha256": sha256_file(BASE_CONFIG_PATH),
        "resolved_fingerprint": resolved.fingerprint,
        "sources": [source.to_artifact_dict() for source in resolved.sources],
        "resolved_contract": {
            "train_sample_limit": config.data.train.sample_limit,
            "eval_sample_limit": None
            if config.data.eval is None
            else config.data.eval.sample_limit,
            "train_order": config.data.train_order,
            "object_ordering": config.template.object_ordering,
            "runtime_seed": config.runtime.seed,
            "configured_global_max_length": config.packing.global_max_length,
            "configured_policy": config.packing.policy,
            "planner_global_max_length": planner_global_max_length,
        },
    }
    return {**payload, "binding_sha256": sha256_json(payload)}


def _validate_config_binding(value: Any, *, scope: str) -> dict[str, Any]:
    observed = dict(_mapping(value, code="wave6.config"))
    contract = _mapping(observed.get("resolved_contract"), code="wave6.config")
    planner_length = contract.get("planner_global_max_length")
    if (
        not isinstance(planner_length, int)
        or isinstance(planner_length, bool)
        or planner_length <= 0
    ):
        raise Wave6ProbeError("planner length is invalid", code="wave6.config")
    expected = _config_binding(planner_global_max_length=planner_length)
    if observed != expected:
        raise Wave6ProbeError("config binding drifted", code="wave6.config")
    fixed = {
        "train_sample_limit": 256,
        "eval_sample_limit": 64,
        "train_order": "source_order",
        "object_ordering": "geo_sorted",
        "runtime_seed": 17,
        "configured_global_max_length": 12_000,
        "configured_policy": SOURCE_ORDER_NEXT_FIT,
    }
    if any(
        contract.get(key) != expected_value for key, expected_value in fixed.items()
    ):
        raise Wave6ProbeError(
            "current config is outside W0 contract", code="wave6.config"
        )
    if scope == "frozen_w0_cpu" and planner_length != 12_000:
        raise Wave6ProbeError("W0 planner length drifted", code="wave6.config")
    return observed


def _execution_contract() -> dict[str, Any]:
    return {
        "cpu_only": True,
        "training": "forbidden",
        "model_loading": "forbidden",
        "gpu": "forbidden",
        "cache_writes": "forbidden",
        "artifact_writes": ["prepared_plan", "success_or_failure_terminal_receipt"],
        "adaptive_thresholds": False,
        "retry_count": 0,
        "worker_counts": list(WORKER_COUNTS),
        "worker_count_disposition": (
            "semantic_pending_upstream_materialization_equality"
        ),
        "pair_orders": [list(order) for order in POLICY_PAIR_ORDERS],
        "accepted_pair_count_per_candidate": 3,
        "candidate_count": len(CANDIDATE_GRID),
        "fresh_arm_process_count": len(CANDIDATE_GRID) * 3 * 2,
        "fresh_process_per_observation": True,
        "child_timeout_seconds": CHILD_TIMEOUT_SECONDS,
        "stream_fragment_pack_budget": STREAM_FRAGMENT_PACK_BUDGET,
        "resume_fragment_pack_budget": RESUME_FRAGMENT_PACK_BUDGET,
        "cursor_byte_budget": DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
        "fragment_item_budget": DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
        "fragment_byte_budget": DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
        "wall_clock": (
            "time.perf_counter_public_planner_entry_to_return_single_observation_per_arm"
        ),
        "rss": "current_process_rss_and_process_lifetime_high_water",
        "threshold_policy": "none_cpu_planner_metrics_are_descriptive_only",
    }


def _arm_v3_determinant_from_packing(packing: Any) -> dict[str, Any]:
    policy_identity = build_pack_plan_policy_identity(
        policy=str(packing.policy),
        window_size=packing.window_size,
        lookahead=packing.lookahead,
        seed=int(packing.seed),
        worker_count=int(packing.worker_count),
        cursor_byte_budget=int(packing.cursor_byte_budget),
        fragment_item_budget=int(packing.fragment_item_budget),
        fragment_byte_budget=int(packing.fragment_byte_budget),
    )
    owner = str(pack_cache.PACKING_CACHE_DETERMINANT_OWNERS["packing_config"])
    payload = {
        "cache_version": pack_cache.PACKING_CACHE_VERSION,
        "registry_schema_version": (
            pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION
        ),
        "determinant_owners": dict(pack_cache.PACKING_CACHE_DETERMINANT_OWNERS),
        "packing_config_owner_source_identity": {
            "path": owner,
            "sha256": sha256_file(REPO_ROOT / owner),
        },
        "packing": {
            "schema": PACK_PLAN_SCHEMA,
            "schema_version": PACK_PLAN_SCHEMA_VERSION,
            "global_max_length": int(packing.global_max_length),
            "policy_identity": policy_identity,
            "fragment_pack_budget": packing.max_packs_per_fragment,
        },
    }
    return {**payload, "projection_fingerprint": sha256_json(payload)}


def _arm_v3_determinant_bindings(*, global_max_length: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in ARM_GRID:
        result[str(spec["arm_id"])] = {
            str(worker_count): _arm_v3_determinant_from_packing(
                _packing_config_for_observation(
                    SimpleNamespace(),
                    spec=spec,
                    worker_count=worker_count,
                    global_max_length=global_max_length,
                ).packing
            )
            for worker_count in WORKER_COUNTS
        }
    return result


def _stream_identity(
    examples: Sequence[Any], *, global_max_length: int, status: str
) -> dict[str, Any]:
    if not examples or len(examples) > MAX_INPUT_COUNT:
        raise Wave6ProbeError(
            "encoded stream count is invalid", code="wave6.encoded_stream"
        )
    plan = create_pack_plan(
        examples,
        global_max_length=global_max_length,
        policy=SOURCE_ORDER_NEXT_FIT,
        seed=17,
        worker_count=1,
    )
    inputs = [item.to_dict() for item in plan.inputs]
    payload = {
        "schema": ENCODED_STREAM_SCHEMA,
        "identity_status": status,
        "input_count": len(inputs),
        "encoded_length_sum": sum(item["encoded_length"] for item in inputs),
        "encoded_inputs_sha256": sha256_json(inputs),
        "encoded_lengths_sha256": sha256_json(
            [item["encoded_length"] for item in inputs]
        ),
        "intra_image_order_identities_sha256": sha256_json(
            [item["intra_image_order_identity"] for item in inputs]
        ),
        "example_ids_sha256": sha256_json([item["example_id"] for item in inputs]),
    }
    return {**payload, "identity_sha256": sha256_json(payload)}


def _validate_stream_identity(value: Any, *, scope: str) -> dict[str, Any]:
    stream = dict(_mapping(value, code="wave6.encoded_stream"))
    _require_exact_fields(
        stream,
        {
            "schema",
            "identity_status",
            "input_count",
            "encoded_length_sum",
            "encoded_inputs_sha256",
            "encoded_lengths_sha256",
            "intra_image_order_identities_sha256",
            "example_ids_sha256",
            "identity_sha256",
        },
        code="wave6.encoded_stream",
    )
    digest = stream.pop("identity_sha256")
    if (
        stream.get("schema") != ENCODED_STREAM_SCHEMA
        or not _is_sha256(digest)
        or digest != sha256_json(stream)
    ):
        raise Wave6ProbeError(
            "encoded stream authentication failed", code="wave6.encoded_stream"
        )
    stream["identity_sha256"] = digest
    count = stream.get("input_count")
    length_sum = stream.get("encoded_length_sum")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or count > MAX_INPUT_COUNT
        or not isinstance(length_sum, int)
        or isinstance(length_sum, bool)
        or length_sum <= 0
        or not all(
            _is_sha256(stream[field])
            for field in (
                "encoded_inputs_sha256",
                "encoded_lengths_sha256",
                "intra_image_order_identities_sha256",
                "example_ids_sha256",
            )
        )
    ):
        raise Wave6ProbeError(
            "encoded stream fields are invalid", code="wave6.encoded_stream"
        )
    expected_status = (
        "frozen_historical_w0_exact"
        if scope == "frozen_w0_cpu"
        else "fixture_not_historical_w0"
    )
    if stream.get("identity_status") != expected_status:
        raise Wave6ProbeError(
            "encoded stream scope is invalid", code="wave6.encoded_stream"
        )
    if scope == "frozen_w0_cpu" and any(
        stream.get(key) != expected for key, expected in FROZEN_W0_TRAIN_STREAM.items()
    ):
        raise Wave6ProbeError(
            "historical W0 stream drifted", code="wave6.encoded_stream"
        )
    return stream


def _fixture_w0_binding() -> dict[str, Any]:
    payload = {
        "schema": W0_BINDING_SCHEMA,
        "status": "temporary_fixture_no_cache",
        "historical_config_fingerprint": None,
        "cache_root": None,
        "splits": {},
        "access_policy": "no_cache_input",
    }
    return {**payload, "binding_sha256": sha256_json(payload)}


def _validate_w0_binding(value: Any, *, scope: str) -> dict[str, Any]:
    binding = dict(_mapping(value, code="wave6.w0_binding"))
    _require_exact_fields(
        binding,
        {
            "schema",
            "status",
            "historical_config_fingerprint",
            "cache_root",
            "splits",
            "access_policy",
            "binding_sha256",
        },
        code="wave6.w0_binding",
    )
    digest = binding.pop("binding_sha256")
    if (
        binding.get("schema") != W0_BINDING_SCHEMA
        or not _is_sha256(digest)
        or digest != sha256_json(binding)
    ):
        raise Wave6ProbeError(
            "W0 binding authentication failed", code="wave6.w0_binding"
        )
    binding["binding_sha256"] = digest
    if scope == "temporary_fixture":
        if binding != _fixture_w0_binding():
            raise Wave6ProbeError(
                "fixture W0 binding is invalid", code="wave6.w0_binding"
            )
    else:
        splits = _mapping(binding.get("splits"), code="wave6.w0_binding")
        if (
            binding.get("status") != "historical_v2_read_only_encoded_evidence"
            or binding.get("historical_config_fingerprint")
            != HISTORICAL_W0_CONFIG_FINGERPRINT
            or binding.get("access_policy") != "authenticate_read_only_no_cache_writes"
            or not isinstance(binding.get("cache_root"), str)
            or set(splits) != {"train", "eval"}
        ):
            raise Wave6ProbeError(
                "historical W0 binding is invalid", code="wave6.w0_binding"
            )
        root = Path(binding["cache_root"])
        for split, expected in HISTORICAL_W0_SPLITS.items():
            observed = _mapping(splits[split], code="wave6.w0_binding")
            expected_cache_dir = root / expected["fingerprint"]
            expected_split = {
                "fingerprint": expected["fingerprint"],
                "manifest_path": str(expected_cache_dir / "manifest.json"),
                "manifest_sha256": expected["manifest_sha256"],
                "chunk_path": str(expected_cache_dir / "chunks/chunk-00000.pkl"),
                "chunk_sha256": expected["chunk_sha256"],
                "micro_step_count": expected["micro_step_count"],
                "example_count": expected["example_count"],
            }
            if dict(observed) != expected_split:
                raise Wave6ProbeError(
                    "historical W0 split binding is invalid",
                    code="wave6.w0_binding",
                )
    return binding


def _materialization_workload(examples: Sequence[Any], *, scope: str) -> dict[str, Any]:
    if scope == "temporary_fixture":
        example_types = {type(example) for example in examples}
        if len(example_types) != 1:
            raise Wave6ProbeError(
                "fixture examples must share one encoded dataclass type",
                code="wave6.materialization_workload",
            )
        example_type = next(iter(example_types))
        fixture_field_names = (
            [field.name for field in dataclass_fields(example_type)]
            if is_dataclass(example_type)
            else []
        )
        if fixture_field_names != ["example_id", "input_ids", "row_ids"]:
            raise Wave6ProbeError(
                "fixture encoded dataclass shape is not supported",
                code="wave6.materialization_workload",
            )
        encoded_dataclass_identity = {
            "module": example_type.__module__,
            "qualname": example_type.__qualname__,
            "field_names": fixture_field_names,
        }
        fixture_examples = []
        for example in examples:
            example_id = getattr(example, "example_id", None)
            input_ids = getattr(example, "input_ids", None)
            row_ids = getattr(example, "row_ids", None)
            if (
                not isinstance(example_id, str)
                or not example_id
                or not isinstance(input_ids, tuple)
                or not isinstance(row_ids, tuple)
            ):
                raise Wave6ProbeError(
                    "fixture cannot be materialized through the production seam",
                    code="wave6.materialization_workload",
                )
            fixture_examples.append(
                {
                    "example_id": example_id,
                    "input_ids": [int(token) for token in input_ids],
                    "row_ids": [str(row_id) for row_id in row_ids],
                }
            )
        payload = {
            "mode": "deterministic_fixture_through_production_encoder_seam",
            "split": "train",
            "fixture_examples": fixture_examples,
            "encoded_dataclass_identity": encoded_dataclass_identity,
            "real_config_path": None,
            "model_loading": "forbidden",
        }
    else:
        payload = {
            "mode": "real_w0_train_through_production_encoder_seam",
            "split": "train",
            "fixture_examples": None,
            "encoded_dataclass_identity": None,
            "real_config_path": str(BASE_CONFIG_PATH),
            "model_loading": "forbidden",
        }
    return {**payload, "binding_sha256": sha256_json(payload)}


def _validate_materialization_workload(value: Any, *, scope: str) -> dict[str, Any]:
    workload = dict(_mapping(value, code="wave6.materialization_workload"))
    _require_exact_fields(
        workload,
        {
            "mode",
            "split",
            "fixture_examples",
            "encoded_dataclass_identity",
            "real_config_path",
            "model_loading",
            "binding_sha256",
        },
        code="wave6.materialization_workload",
    )
    digest = workload.pop("binding_sha256", None)
    if not _is_sha256(digest) or digest != sha256_json(workload):
        raise Wave6ProbeError(
            "materialization workload authentication failed",
            code="wave6.materialization_workload",
        )
    workload["binding_sha256"] = digest
    if workload.get("split") != "train" or workload.get("model_loading") != "forbidden":
        raise Wave6ProbeError(
            "materialization workload contract is invalid",
            code="wave6.materialization_workload",
        )
    if scope == "temporary_fixture":
        examples = workload.get("fixture_examples")
        encoded_dataclass_identity = workload.get("encoded_dataclass_identity")
        if (
            workload.get("mode")
            != "deterministic_fixture_through_production_encoder_seam"
            or workload.get("real_config_path") is not None
            or not isinstance(examples, list)
            or not examples
            or len(examples) > MAX_INPUT_COUNT
            or not isinstance(encoded_dataclass_identity, Mapping)
            or set(encoded_dataclass_identity) != {"module", "qualname", "field_names"}
            or not isinstance(encoded_dataclass_identity.get("module"), str)
            or not encoded_dataclass_identity["module"]
            or not isinstance(encoded_dataclass_identity.get("qualname"), str)
            or not encoded_dataclass_identity["qualname"].isidentifier()
            or encoded_dataclass_identity.get("field_names")
            != ["example_id", "input_ids", "row_ids"]
        ):
            raise Wave6ProbeError(
                "fixture materialization workload is invalid",
                code="wave6.materialization_workload",
            )
    elif workload != _materialization_workload((), scope="frozen_w0_cpu"):
        raise Wave6ProbeError(
            "real materialization workload drifted",
            code="wave6.materialization_workload",
        )
    return workload


def _build_plan(
    examples: Sequence[Any],
    *,
    scope: str,
    global_max_length: int,
    w0_binding: Mapping[str, Any],
    plan_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    plan_target = _absent_target(plan_path)
    receipt_target = _absent_target(receipt_path)
    failure_target = receipt_target.with_name(
        f"{receipt_target.stem}.failure{receipt_target.suffix}"
    )
    _absent_target(failure_target)
    if len({plan_target, receipt_target, failure_target}) != 3:
        raise Wave6ProbeError(
            "plan and terminal targets must differ", code="wave6.artifact_target"
        )
    source_before = _source_state()
    status = (
        "frozen_historical_w0_exact"
        if scope == "frozen_w0_cpu"
        else "fixture_not_historical_w0"
    )
    payload = {
        "schema": PLAN_SCHEMA,
        "status": "prepared",
        "scope": scope,
        "source_owners": source_before["source_owners"],
        "production_integration": source_before["production_integration"],
        "pack_plan_contract": _pack_plan_contract(),
        "config": _config_binding(planner_global_max_length=global_max_length),
        "materialization_workload": _materialization_workload(
            tuple(examples), scope=scope
        ),
        "w0_cpu_cache_binding": dict(w0_binding),
        "w0_cpu_encoded_stream": _stream_identity(
            tuple(examples), global_max_length=global_max_length, status=status
        ),
        "candidate_grid": [dict(item) for item in ARM_GRID],
        "arm_v3_determinant_bindings": _arm_v3_determinant_bindings(
            global_max_length=global_max_length
        ),
        "execution_contract": _execution_contract(),
        "artifact_targets": {
            "plan": str(plan_target),
            "receipt": str(receipt_target),
            "failure_receipt": str(failure_target),
        },
        "run_argv": [
            sys.executable,
            str(SCRIPT_PATH),
            "run",
            "--plan",
            str(plan_target),
        ],
    }
    source_after = _source_state()
    if source_before != source_after:
        raise Wave6ProbeError(
            "source changed during plan preparation",
            code="wave6.source_quiescence",
        )
    payload["preparation_source_quiescence"] = {
        "schema": SOURCE_QUIESCENCE_SCHEMA,
        "required": True,
        "before_sha256": source_before["sha256"],
        "after_sha256": source_after["sha256"],
        "stable": True,
    }
    plan = finalize_artifact(payload, hash_field="plan_sha256")
    validate_plan(plan, require_receipt_absent=True)
    return plan


def build_fixture_plan(
    examples: Sequence[Any],
    *,
    plan_path: str | Path,
    receipt_path: str | Path,
    global_max_length: int,
) -> dict[str, Any]:
    """Build an authenticated no-cache fixture plan for focused tests."""

    return _build_plan(
        examples,
        scope="temporary_fixture",
        global_max_length=global_max_length,
        w0_binding=_fixture_w0_binding(),
        plan_path=plan_path,
        receipt_path=receipt_path,
    )


def validate_plan(
    payload: Mapping[str, Any], *, require_receipt_absent: bool
) -> dict[str, Any]:
    plan = _validate_finalized(payload, schema=PLAN_SCHEMA, hash_field="plan_sha256")
    _require_exact_fields(
        plan,
        {
            "schema",
            "status",
            "scope",
            "source_owners",
            "production_integration",
            "preparation_source_quiescence",
            "pack_plan_contract",
            "config",
            "materialization_workload",
            "w0_cpu_cache_binding",
            "w0_cpu_encoded_stream",
            "candidate_grid",
            "arm_v3_determinant_bindings",
            "execution_contract",
            "artifact_targets",
            "run_argv",
            "plan_sha256",
        },
        code="wave6.plan",
    )
    scope = plan.get("scope")
    if plan.get("status") != "prepared" or scope not in {
        "frozen_w0_cpu",
        "temporary_fixture",
    }:
        raise Wave6ProbeError("plan status or scope is invalid", code="wave6.plan")
    _validate_source_owners(plan["source_owners"])
    integration = _validate_production_integration(plan["production_integration"])
    current_state = {
        "source_owners": plan["source_owners"],
        "production_integration": integration,
    }
    _validate_source_quiescence(
        plan["preparation_source_quiescence"],
        current_sha256=sha256_json(current_state),
    )
    if plan["pack_plan_contract"] != _pack_plan_contract():
        raise Wave6ProbeError(
            "PackPlan contract drifted", code="wave6.pack_plan_contract"
        )
    _validate_config_binding(plan["config"], scope=str(scope))
    _validate_materialization_workload(
        plan["materialization_workload"], scope=str(scope)
    )
    _validate_w0_binding(plan["w0_cpu_cache_binding"], scope=str(scope))
    _validate_stream_identity(plan["w0_cpu_encoded_stream"], scope=str(scope))
    if plan["candidate_grid"] != [dict(item) for item in ARM_GRID]:
        raise Wave6ProbeError("candidate grid drifted", code="wave6.candidate_grid")
    expected_arm_v3 = _arm_v3_determinant_bindings(
        global_max_length=int(
            plan["config"]["resolved_contract"]["planner_global_max_length"]
        )
    )
    if plan["arm_v3_determinant_bindings"] != expected_arm_v3:
        raise Wave6ProbeError(
            "arm current-v3 determinant binding drifted",
            code="wave6.arm_v3_determinant",
        )
    if plan["execution_contract"] != _execution_contract():
        raise Wave6ProbeError(
            "execution contract drifted", code="wave6.execution_contract"
        )
    targets = _mapping(plan["artifact_targets"], code="wave6.artifact_target")
    _require_exact_fields(
        targets,
        {"plan", "receipt", "failure_receipt"},
        code="wave6.artifact_target",
    )
    plan_target = Path(str(targets["plan"]))
    receipt_target = Path(str(targets["receipt"]))
    failure_target = Path(str(targets["failure_receipt"]))
    if (
        not plan_target.is_absolute()
        or not receipt_target.is_absolute()
        or not failure_target.is_absolute()
        or len({plan_target, receipt_target, failure_target}) != 3
        or plan["run_argv"]
        != [sys.executable, str(SCRIPT_PATH), "run", "--plan", str(plan_target)]
    ):
        raise Wave6ProbeError(
            "artifact target or run argv is invalid", code="wave6.artifact_target"
        )
    if require_receipt_absent:
        _absent_target(receipt_target)
        _absent_target(failure_target)
    canonical_json_bytes(plan)
    return plan


def _current_rss_bytes() -> int:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        pages = int(fields[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages <= 0 or page_size <= 0:
            raise ValueError
        return pages * page_size
    except (OSError, ValueError, IndexError) as exc:
        raise Wave6ProbeError("process RSS is unavailable", code="wave6.rss") from exc


def _process_high_water_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value <= 0:
        raise Wave6ProbeError("process high-water RSS is unavailable", code="wave6.rss")
    return value * 1024 if sys.platform != "darwin" else value


def _arm_kwargs(spec: Mapping[str, Any], *, worker_count: int) -> dict[str, Any]:
    return {
        "policy": spec["policy"],
        "window_size": spec["window_size"],
        "lookahead": spec["lookahead"],
        "seed": 17,
        "worker_count": worker_count,
        "cursor_byte_budget": DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
        "fragment_item_budget": DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
        "fragment_byte_budget": DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
    }


def _roundtrip_replay(plan: PackPlan, examples: Sequence[Any]) -> None:
    decoded = PackPlan.from_json(plan.to_json())
    if decoded != plan or decoded.to_json() != plan.to_json():
        raise Wave6ProbeError("PackPlan round-trip drifted", code="wave6.replay")
    replayed = replay_pack_plan(decoded, examples)
    expected = [pack.input_ordinals for pack in plan.packs]
    observed = [
        tuple(segment.example_index for segment in pack.segments) for pack in replayed
    ]
    if observed != expected:
        raise Wave6ProbeError("PackPlan replay membership drifted", code="wave6.replay")
    for pack in replayed:
        for segment in pack.segments:
            if pack.input_ids[segment.start : segment.end] != tuple(
                examples[segment.example_index].input_ids
            ):
                raise Wave6ProbeError(
                    "PackPlan replay tokens drifted", code="wave6.replay"
                )


def _dedupe_inputs(plans: Sequence[PackPlan]) -> list[dict[str, Any]]:
    by_ordinal: dict[int, dict[str, Any]] = {}
    for plan in plans:
        for item in plan.inputs:
            value = item.to_dict()
            existing = by_ordinal.get(item.input_ordinal)
            if existing is not None and existing != value:
                raise Wave6ProbeError(
                    "fragment input identity drifted", code="wave6.replay"
                )
            by_ordinal[item.input_ordinal] = value
    return [by_ordinal[index] for index in sorted(by_ordinal)]


def _semantic_projection(
    plans: Sequence[PackPlan], terminal_cursor: PackPlanCursor
) -> dict[str, Any]:
    return {
        "inputs": _dedupe_inputs(plans),
        "packs": [pack.to_dict() for plan in plans for pack in plan.packs],
        "rejected_examples": [
            item.to_dict() for plan in plans for item in plan.rejected_examples
        ],
        "terminal_cursor": terminal_cursor.to_payload_dict(),
    }


def _build_bundle(
    examples: Sequence[Any],
    *,
    spec: Mapping[str, Any],
    global_max_length: int,
    worker_count: int,
    strict_replay: bool,
) -> dict[str, Any]:
    kwargs = _arm_kwargs(spec, worker_count=worker_count)
    rss_before = _current_rss_bytes()
    started_at = time.perf_counter()
    if spec["policy"] != ONLINE_WINDOW_BINPACK:
        plan = create_pack_plan(examples, global_max_length=global_max_length, **kwargs)
        planning_wall_seconds = time.perf_counter() - started_at
        rss_after = _current_rss_bytes()
        rss_high_water = _process_high_water_bytes()
        if strict_replay:
            _roundtrip_replay(plan, examples)
        plans = [plan]
        terminal_cursor = plan.replay_cursor
        resume = {
            "applicable": False,
            "fragmented_resume_exact": True,
            "terminal_cursor_complete": True,
            "strict_fragment_chain_verified": True,
            "fragment_count": 1,
        }
        bounded = {
            "pending_limit": 1
            if spec["policy"] == SOURCE_ORDER_NEXT_FIT
            else spec["window_size"],
            "max_pending_items_observed": plan.max_pending_items_observed,
            "cursor_byte_budget": DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
            "max_cursor_bytes_observed": plan.max_serialized_cursor_bytes_observed,
            "fragment_item_budget": DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
            "max_fragment_items_observed": len(plan.inputs),
            "fragment_byte_budget": DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
            "max_fragment_bytes_observed": plan.serialized_size_bytes,
        }
    else:
        fragments: list[PackPlan] = []
        receipt = stream_online_pack_plan_fragments(
            lambda: iter(examples),
            fragment_sink=fragments.append,
            global_max_length=global_max_length,
            lookahead=int(spec["lookahead"]),
            max_packs_per_fragment=STREAM_FRAGMENT_PACK_BUDGET,
            seed=17,
            worker_count=worker_count,
            cursor_byte_budget=DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
            fragment_item_budget=DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
            fragment_byte_budget=DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
        )
        planning_wall_seconds = time.perf_counter() - started_at
        rss_after = _current_rss_bytes()
        rss_high_water = _process_high_water_bytes()
        receipt = PackPlanStreamReceipt.from_json(receipt.to_json())
        fragments = [PackPlan.from_json(fragment.to_json()) for fragment in fragments]
        verify_pack_plan_stream_fragments(receipt, fragments)
        if strict_replay:
            for fragment in fragments:
                _roundtrip_replay(fragment, examples)
        plans = fragments
        terminal_cursor = receipt.terminal_cursor
        resume = {
            "applicable": True,
            "fragmented_resume_exact": True,
            "terminal_cursor_complete": terminal_cursor.complete,
            "strict_fragment_chain_verified": True,
            "fragment_count": receipt.fragment_count,
        }
        bounded = {
            "pending_limit": spec["lookahead"],
            "max_pending_items_observed": receipt.max_pending_items_observed,
            "cursor_byte_budget": DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
            "max_cursor_bytes_observed": receipt.max_cursor_bytes_observed,
            "fragment_item_budget": DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
            "max_fragment_items_observed": receipt.max_fragment_items_observed,
            "fragment_byte_budget": DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
            "max_fragment_bytes_observed": receipt.max_fragment_bytes_observed,
        }
    projection = _semantic_projection(plans, terminal_cursor)
    bounded["within_all_declared_bounds"] = bool(
        bounded["max_pending_items_observed"] <= bounded["pending_limit"]
        and bounded["max_cursor_bytes_observed"] <= bounded["cursor_byte_budget"]
        and bounded["max_fragment_items_observed"] <= bounded["fragment_item_budget"]
        and bounded["max_fragment_bytes_observed"] <= bounded["fragment_byte_budget"]
    )
    return {
        "plans": plans,
        "terminal_cursor": terminal_cursor,
        "projection": projection,
        "projection_sha256": sha256_json(projection),
        "resume": resume,
        "bounded": bounded,
        "measurement": {
            "planning_wall_seconds": planning_wall_seconds,
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
            "process_rss_high_water_bytes": rss_high_water,
        },
    }


def _fragmented_resume_oracle(
    examples: Sequence[Any],
    *,
    spec: Mapping[str, Any],
    global_max_length: int,
    expected: Mapping[str, Any],
    worker_count: int = 1,
) -> dict[str, Any]:
    if spec["policy"] != ONLINE_WINDOW_BINPACK:
        return dict(expected["resume"])
    cursor: PackPlanCursor | None = None
    predecessor: PackPlan | None = None
    parts: list[PackPlan] = []
    while cursor is None or not cursor.complete:
        if len(parts) >= MAX_FRAGMENT_COUNT:
            raise Wave6ProbeError(
                "resume fragment count exceeded", code="wave6.resume_cursor"
            )
        part = create_pack_plan(
            examples,
            global_max_length=global_max_length,
            **_arm_kwargs(spec, worker_count=worker_count),
            replay_cursor=cursor,
            replay_plan=predecessor,
            max_packs=RESUME_FRAGMENT_PACK_BUDGET,
        )
        part = PackPlan.from_json(part.to_json())
        _roundtrip_replay(part, examples)
        parts.append(part)
        cursor = PackPlanCursor.from_json(part.replay_cursor.to_json())
        predecessor = part
    expected_memberships = [
        pack["input_ordinals"] for pack in expected["projection"]["packs"]
    ]
    resumed_memberships = [
        list(pack.input_ordinals) for part in parts for pack in part.packs
    ]
    exact = bool(
        resumed_memberships == expected_memberships
        and cursor == expected["terminal_cursor"]
        and cursor.complete
    )
    if not exact:
        raise Wave6ProbeError(
            "fragmented resume differs from stream", code="wave6.resume_cursor"
        )
    return {
        "applicable": True,
        "fragmented_resume_exact": True,
        "terminal_cursor_complete": True,
        "strict_fragment_chain_verified": True,
        "fragment_count": len(parts),
    }


def _co_present_pairs(memberships: Sequence[Sequence[int]]) -> list[list[int]]:
    pairs: set[tuple[int, int]] = set()
    for membership in memberships:
        for left_index, left in enumerate(membership):
            for right in membership[left_index + 1 :]:
                pairs.add(tuple(sorted((int(left), int(right)))))
    return [list(pair) for pair in sorted(pairs)]


def _ordering_metrics(
    memberships: Sequence[Sequence[int]], *, reference_pairs: Sequence[Sequence[int]]
) -> dict[str, Any]:
    flattened = [int(ordinal) for membership in memberships for ordinal in membership]
    positions = {ordinal: index for index, ordinal in enumerate(flattened)}
    displacement = [
        abs(positions[ordinal] - ordinal) for ordinal in range(len(flattened))
    ]
    pairs = _co_present_pairs(memberships)
    pair_set = {tuple(pair) for pair in pairs}
    reference_set = {tuple(pair) for pair in reference_pairs}
    return {
        "flattened_order_sha256": sha256_json(flattened),
        "moved_example_count": sum(value > 0 for value in displacement),
        "absolute_displacement_sum": sum(displacement),
        "absolute_displacement_max": max(displacement, default=0),
        "co_present_pair_count": len(pairs),
        "co_present_pairs_sha256": sha256_json(pairs),
        "added_vs_reference_count": len(pair_set - reference_set),
        "removed_vs_reference_count": len(reference_set - pair_set),
        "retained_vs_reference_count": len(pair_set & reference_set),
    }


def _research_disposition(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        f"{arm['arm_id']}__matched5step_candidate"
        for arm in arms
        if arm["arm_id"] != "source_order_next_fit"
        and arm["ordering_change"]["moved_example_count"] > 0
        and arm["cpu_disposition"] == "survives_cpu_gate_requires_matched_training"
    ]
    return {
        "planner_utilization_can_promote": False,
        "promotion_authorized": False,
        "optimization_dynamics_unmeasured": True,
        "claim_boundary": (
            "CPU evidence establishes deterministic planner mechanics and descriptive "
            "resource/order metrics only; it does not establish quality, optimization "
            "dynamics, distributed skew, or end-to-end training benefit."
        ),
        "smallest_matched_training": {
            "step_count": 5,
            "reference_arm": "source_order_next_fit__matched5step_reference",
            "candidate_arms": candidates,
            "only_changed_factor": "packing_policy_and_its_declared_window_or_lookahead",
            "identical_fields": [
                "images",
                "intra_image_rows_and_order",
                "encoded_tokens_and_supervision",
                "optimizer_budget",
                "five_step_schedule",
                "evaluation",
                "checkpoint_policy",
                "artifact_semantics",
            ],
        },
    }


def _fixture_render_and_encode_example(
    raw_example: ProbeRawExample, *, config: Any, components: Any
) -> Any:
    del config, components
    encoded_type = _fixture_encoded_type(raw_example)
    return encoded_type(
        example_id=raw_example.example_id,
        input_ids=raw_example.input_ids,
        row_ids=raw_example.row_ids,
    )


def _fixture_row_identity(example: Any) -> str:
    payload = json.dumps(example.row_ids, separators=(",", ":")).encode()
    return f"test-row-order-sha256:{hashlib.sha256(payload).hexdigest()}"


def _fixture_encoded_type(raw_example: ProbeRawExample) -> type[Any]:
    identity = getattr(raw_example, "encoded_dataclass_identity", None)
    if not isinstance(identity, Mapping):
        raise Wave6ProbeError(
            "fixture encoded class identity is missing",
            code="wave6.encoded_materialization",
        )
    module_name = str(identity["module"])
    class_name = str(identity["qualname"])
    module = sys.modules.get(module_name)
    if module is None:
        module = ModuleType(module_name)
        sys.modules[module_name] = module
    existing = getattr(module, class_name, None)
    if existing is not None:
        if (
            not is_dataclass(existing)
            or [field.name for field in dataclass_fields(existing)]
            != identity["field_names"]
        ):
            raise Wave6ProbeError(
                "fixture encoded class identity collides",
                code="wave6.encoded_materialization",
            )
        return existing
    encoded_type = make_dataclass(
        class_name,
        [
            ("example_id", str),
            ("input_ids", tuple[int, ...]),
            ("row_ids", tuple[str, ...]),
        ],
        frozen=True,
        namespace={
            "intra_image_order_identity": property(_fixture_row_identity),
        },
    )
    encoded_type.__module__ = module_name
    encoded_type.__qualname__ = class_name
    setattr(module, class_name, encoded_type)
    return encoded_type


def _fixture_materialization(
    workload: Mapping[str, Any], *, worker_count: int
) -> tuple[Any, tuple[Any, ...], dict[str, Any], None]:
    raw_examples = tuple(
        ProbeRawExample(
            example_id=str(item["example_id"]),
            input_ids=tuple(int(token) for token in item["input_ids"]),
            row_ids=tuple(str(row_id) for row_id in item["row_ids"]),
            encoded_dataclass_identity=workload["encoded_dataclass_identity"],
        )
        for item in workload["fixture_examples"]
    )
    config = SimpleNamespace()
    components = SimpleNamespace()
    _fixture_encoded_type(raw_examples[0])
    original = training_pipeline._render_and_encode_example
    training_pipeline._render_and_encode_example = _fixture_render_and_encode_example
    try:
        encoded = training_pipeline._build_encoded_examples_for_dataset(
            config,
            components,
            raw_examples,
            materialization_workers=worker_count,
        )
    finally:
        training_pipeline._render_and_encode_example = original
    return (
        config,
        encoded,
        {
            "mode": "deterministic_fixture",
            "raw_example_count": len(raw_examples),
            "encoded_dataclass_identity": workload["encoded_dataclass_identity"],
            "private_v3_determinants": None,
        },
        None,
    )


def _real_materialization(
    workload: Mapping[str, Any], *, worker_count: int
) -> tuple[Any, tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    resolved = load_train_config(workload["real_config_path"])
    config = resolved.config
    components = training_pipeline.load_qwen_components(config, load_model=False)
    augmentation = training_pipeline._materialize_raw_examples_for_dataset(
        config,
        config.data.train,
        split="train",
    )
    encoded = training_pipeline._build_encoded_examples_for_dataset(
        config,
        components,
        augmentation.examples,
        materialization_workers=worker_count,
    )
    vocab_groups = training_pipeline.build_token_vocabulary_groups(
        components.token_identity,
        tokenizer=components.tokenizer,
    )
    determinant_config = config.model_copy(
        update={
            "packing": config.packing.model_copy(update={"worker_count": worker_count})
        }
    )
    determinants = pack_cache.build_packing_cache_determinants(
        determinant_config,
        components,
        dataset=determinant_config.data.train,
        split="train",
        vocab_groups=vocab_groups,
    )
    return (
        config,
        encoded,
        {
            "mode": "real_w0_train",
            "raw_example_count": len(augmentation.examples),
            "private_v3_determinants": {
                "version": determinants["version"],
                "registry_schema_version": determinants["registry_schema_version"],
                "aggregate_fingerprint": determinants["aggregate_fingerprint"],
                "packing": determinants["packing"],
            },
        },
        {"components": components, "vocab_groups": vocab_groups},
    )


def _canonicalize_encoded_materialization(
    examples: Sequence[Any],
) -> tuple[Any, ...]:
    """Put every worker result on the production serialization boundary."""

    canonical: list[Any] = []
    try:
        for example in examples:
            canonical.append(pickle.loads(pickle.dumps(example, protocol=5)))
    except (
        AttributeError,
        EOFError,
        ImportError,
        pickle.PickleError,
        TypeError,
    ) as exc:
        raise Wave6ProbeError(
            "encoded materialization serialization failed",
            code="wave6.encoded_materialization",
        ) from exc
    if len(canonical) != len(examples):
        raise Wave6ProbeError(
            "encoded materialization serialization changed example count",
            code="wave6.encoded_materialization",
        )
    return tuple(canonical)


def _packing_config_for_observation(
    base_config: Any,
    *,
    spec: Mapping[str, Any],
    worker_count: int,
    global_max_length: int,
) -> Any:
    values = {
        "global_max_length": global_max_length,
        "policy": spec["policy"],
        "window_size": spec["window_size"],
        "lookahead": spec["lookahead"],
        "seed": 17,
        "worker_count": worker_count,
        "cursor_byte_budget": DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
        "max_packs_per_fragment": (
            STREAM_FRAGMENT_PACK_BUDGET
            if spec["policy"] == ONLINE_WINDOW_BINPACK
            else None
        ),
        "fragment_item_budget": DEFAULT_PACK_PLAN_FRAGMENT_ITEM_BUDGET,
        "fragment_byte_budget": DEFAULT_PACK_PLAN_FRAGMENT_BYTE_BUDGET,
    }
    packing = getattr(base_config, "packing", None)
    if packing is not None and callable(getattr(packing, "model_copy", None)):
        return base_config.model_copy(
            update={"packing": packing.model_copy(update=values)}
        )
    return SimpleNamespace(packing=SimpleNamespace(**values))


def _current_v3_arm_determinant(
    base_config: Any,
    *,
    spec: Mapping[str, Any],
    worker_count: int,
    global_max_length: int,
    real_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    arm_config = _packing_config_for_observation(
        base_config,
        spec=spec,
        worker_count=worker_count,
        global_max_length=global_max_length,
    )
    projection = _arm_v3_determinant_from_packing(arm_config.packing)
    aggregate_fingerprint = None
    if real_context is not None:
        determinants = pack_cache.build_packing_cache_determinants(
            arm_config,
            real_context["components"],
            dataset=arm_config.data.train,
            split="train",
            vocab_groups=real_context["vocab_groups"],
        )
        if (
            determinants["version"] != projection["cache_version"]
            or determinants["registry_schema_version"]
            != projection["registry_schema_version"]
            or determinants["packing"] != projection["packing"]
            or determinants["aggregate_fingerprint"]
            != pack_cache.packing_cache_fingerprint_from_determinants(determinants)
        ):
            raise Wave6ProbeError(
                "real current-v3 arm determinant differs from projection",
                code="wave6.arm_v3_determinant",
            )
        aggregate_fingerprint = determinants["aggregate_fingerprint"]
    return {
        **projection,
        "full_aggregate_fingerprint": aggregate_fingerprint,
        "full_determinants": determinants if real_context is not None else None,
    }


def _production_arm_observation(
    encoded_examples: Sequence[Any],
    *,
    base_config: Any,
    spec: Mapping[str, Any],
    worker_count: int,
    global_max_length: int,
) -> dict[str, Any]:
    config = _packing_config_for_observation(
        base_config,
        spec=spec,
        worker_count=worker_count,
        global_max_length=global_max_length,
    )
    started_at = time.perf_counter()
    packs, pipeline_receipt, fragment_by_pack = (
        training_pipeline._materialize_pack_plan(config, encoded_examples)
    )
    planner_wall = time.perf_counter() - started_at
    supervision = packing_supervision.build_packed_supervision(
        packs, tuple(encoded_examples)
    )
    memberships = [
        [segment.example_index for segment in pack.segments] for pack in packs
    ]
    semantic_order = {
        "memberships": memberships,
        "pack_lengths": [pack.length for pack in packs],
        "example_ids": [
            [segment.example_id for segment in pack.segments] for pack in packs
        ],
        "supervision": supervision.to_artifact_dict(),
    }
    upstream_plan_or_fragment_chain_sha256 = (
        pipeline_receipt["plan_sha256"]
        if pipeline_receipt["plan_sha256"] is not None
        else pipeline_receipt["fragment_chain_sha256"]
    )
    if not _is_sha256(upstream_plan_or_fragment_chain_sha256):
        raise Wave6ProbeError(
            "production pipeline did not emit a full plan identity",
            code="wave6.production_plan_identity",
        )
    strict_bundle = _build_bundle(
        encoded_examples,
        spec=spec,
        global_max_length=global_max_length,
        worker_count=worker_count,
        strict_replay=True,
    )
    replay_memberships = [
        pack["input_ordinals"] for pack in strict_bundle["projection"]["packs"]
    ]
    if replay_memberships != memberships:
        raise Wave6ProbeError(
            "production materialization differs from strict planner replay",
            code="wave6.semantic_oracle",
        )
    resume = _fragmented_resume_oracle(
        encoded_examples,
        spec=spec,
        global_max_length=global_max_length,
        expected=strict_bundle,
        worker_count=worker_count,
    )
    pack_fragment_identity_sha256 = sha256_json(
        [[index, fragment_by_pack[index]] for index in sorted(fragment_by_pack)]
    )
    pipeline_receipt_sha256 = sha256_json(pipeline_receipt)
    semantic_order_sha256 = sha256_json(semantic_order)
    full_plan_sha256 = sha256_json(
        {
            "pipeline_receipt": pipeline_receipt,
            "pack_fragment_identity_sha256": pack_fragment_identity_sha256,
            "semantic_order": semantic_order,
        }
    )
    used_tokens = sum(pack.length for pack in packs)
    capacity_tokens = len(packs) * global_max_length
    return {
        "arm_id": spec["arm_id"],
        "policy": spec["policy"],
        "window_size": spec["window_size"],
        "lookahead": spec["lookahead"],
        "production_plan_identity": {
            "mode": pipeline_receipt["mode"],
            "full_plan_sha256": full_plan_sha256,
            "upstream_plan_or_fragment_chain_sha256": (
                upstream_plan_or_fragment_chain_sha256
            ),
            "pipeline_receipt_sha256": pipeline_receipt_sha256,
            "pack_fragment_identity_sha256": pack_fragment_identity_sha256,
            "worker_count_bound_semantically": True,
        },
        "semantic_order_sha256": semantic_order_sha256,
        "pack_memberships": memberships,
        "semantic_oracle": {
            "each_once_exact": True,
            "intra_image_row_identity_order_exact": True,
            "strict_pack_plan_roundtrip": True,
            "strict_replay_exact": True,
            "production_pipeline_matches_strict_replay": True,
        },
        "resume_cursor_oracle": resume,
        "boundedness": strict_bundle["bounded"],
        "metrics": {
            "used_tokens": used_tokens,
            "capacity_tokens": capacity_tokens,
            "pack_count": len(packs),
            "tail_waste": capacity_tokens - used_tokens,
            "utilization": used_tokens / capacity_tokens,
            "planning_wall_seconds": planner_wall,
        },
    }


def execute_materialization_observation(
    plan: Mapping[str, Any],
    *,
    candidate_arm_id: str,
    repetition_index: int,
    position_index: int,
    arm_id: str,
    injected_failure_stage: str | None = None,
) -> dict[str, Any]:
    rss_before = _current_rss_bytes()
    started_at = time.perf_counter()
    checked = validate_plan(plan, require_receipt_absent=True)
    candidate_specs = {str(spec["arm_id"]): spec for spec in CANDIDATE_GRID}
    candidate_spec = candidate_specs.get(candidate_arm_id)
    if (
        candidate_spec is None
        or not isinstance(repetition_index, int)
        or isinstance(repetition_index, bool)
        or not 0 <= repetition_index < len(POLICY_PAIR_ORDERS)
        or position_index not in {0, 1}
    ):
        raise Wave6ProbeError(
            "policy-pair coordinate is invalid",
            code="wave6.observation_coordinate",
        )
    expected_label = POLICY_PAIR_ORDERS[repetition_index][position_index]
    expected_arm_id = (
        candidate_arm_id if expected_label == "candidate" else str(expected_label)
    )
    if arm_id != expected_arm_id:
        raise Wave6ProbeError(
            "arm differs from frozen policy-pair order",
            code="wave6.observation_coordinate",
        )
    spec = next(spec for spec in ARM_GRID if spec["arm_id"] == arm_id)
    if injected_failure_stage == "child":
        raise Wave6ProbeError("injected child failure", code="wave6.injected_child")
    workload = checked["materialization_workload"]
    materializer = (
        _fixture_materialization
        if checked["scope"] == "temporary_fixture"
        else _real_materialization
    )
    raw_materialized = {
        worker_count: materializer(workload, worker_count=worker_count)
        for worker_count in WORKER_COUNTS
    }
    materialized = {
        worker_count: (
            result[0],
            _canonicalize_encoded_materialization(result[1]),
            {
                **result[2],
                "encoded_materialization_canonicalization": (
                    "in_memory_pickle_protocol_5_roundtrip_per_example"
                ),
            },
            result[3],
        )
        for worker_count, result in raw_materialized.items()
    }
    if injected_failure_stage == "planner":
        raise Wave6ProbeError("injected planner failure", code="wave6.injected_planner")
    global_max_length = int(
        checked["config"]["resolved_contract"]["planner_global_max_length"]
    )
    streams = {
        worker_count: _stream_identity(
            materialized[worker_count][1],
            global_max_length=global_max_length,
            status=checked["w0_cpu_encoded_stream"]["identity_status"],
        )
        for worker_count in WORKER_COUNTS
    }
    if (
        any(stream != checked["w0_cpu_encoded_stream"] for stream in streams.values())
        or streams[1] != streams[8]
    ):
        raise Wave6ProbeError(
            "independent encoded materialization differs from frozen stream",
            code="wave6.encoded_materialization",
        )
    worker_executions = {
        str(worker_count): _production_arm_observation(
            materialized[worker_count][1],
            base_config=materialized[worker_count][0],
            spec=spec,
            worker_count=worker_count,
            global_max_length=global_max_length,
        )
        for worker_count in WORKER_COUNTS
    }
    one = worker_executions["1"]
    eight = worker_executions["8"]
    semantic_order_exact = bool(
        one["semantic_order_sha256"] == eight["semantic_order_sha256"]
        and one["pack_memberships"] == eight["pack_memberships"]
    )
    if not semantic_order_exact:
        raise Wave6ProbeError(
            "worker-count production plan semantics differ",
            code="wave6.worker_equality",
        )
    determinants = {
        str(worker_count): _current_v3_arm_determinant(
            materialized[worker_count][0],
            spec=spec,
            worker_count=worker_count,
            global_max_length=global_max_length,
            real_context=materialized[worker_count][3],
        )
        for worker_count in WORKER_COUNTS
    }
    arm = {
        "arm_id": spec["arm_id"],
        "policy": spec["policy"],
        "window_size": spec["window_size"],
        "lookahead": spec["lookahead"],
        "worker_executions": worker_executions,
        "worker_comparison": {
            "encoded_materialization_exact": True,
            "semantic_order_exact": True,
            "full_plan_sha256_worker_1": one["production_plan_identity"][
                "full_plan_sha256"
            ],
            "full_plan_sha256_worker_8": eight["production_plan_identity"][
                "full_plan_sha256"
            ],
            "full_plan_sha256_equal": one["production_plan_identity"][
                "full_plan_sha256"
            ]
            == eight["production_plan_identity"]["full_plan_sha256"],
            "worker_count_disposition": (
                "semantic_pending_upstream_materialization_equality"
            ),
        },
        "current_v3_determinants": determinants,
        "semantic_oracle": one["semantic_oracle"],
        "resume_cursor_oracle": one["resume_cursor_oracle"],
        "boundedness": one["boundedness"],
    }
    wall_seconds = time.perf_counter() - started_at
    rss_after = _current_rss_bytes()
    payload = {
        "schema": OBSERVATION_SCHEMA,
        "status": "accepted",
        "plan_sha256": checked["plan_sha256"],
        "candidate_arm_id": candidate_arm_id,
        "repetition_index": repetition_index,
        "position_index": position_index,
        "arm_id": arm_id,
        "materialization": {
            "production_encoder_seam": (
                "src.training.pipeline._build_encoded_examples_for_dataset"
            ),
            "worker_counts": list(WORKER_COUNTS),
            "encoded_materialization_exact": True,
            "encoded_stream": streams[1],
            "workers": {
                str(worker_count): materialized[worker_count][2]
                for worker_count in WORKER_COUNTS
            },
        },
        "arm": arm,
        "process": {
            "pid": os.getpid(),
            "measurement_scope": (
                "single_policy_arm_entry_to_return_including_worker_1_8_proof"
            ),
            "wall_seconds": wall_seconds,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": max(0, rss_after - rss_before),
            "rss_high_water_bytes": _process_high_water_bytes(),
            "rss_high_water_scope": "fresh_process_lifetime",
        },
    }
    return finalize_artifact(payload, hash_field="observation_sha256")


def execute_comparison(
    plan: Mapping[str, Any], examples: Sequence[Any]
) -> dict[str, Any]:
    checked = validate_plan(plan, require_receipt_absent=True)
    examples = tuple(examples)
    contract = _mapping(checked["config"]["resolved_contract"], code="wave6.config")
    global_max_length = int(contract["planner_global_max_length"])
    expected_stream = checked["w0_cpu_encoded_stream"]
    status = expected_stream["identity_status"]
    observed_stream = _stream_identity(
        examples, global_max_length=global_max_length, status=status
    )
    if observed_stream != expected_stream:
        raise Wave6ProbeError(
            "encoded stream changed since plan", code="wave6.encoded_stream_identity"
        )

    import torch

    if torch.cuda.is_initialized():
        raise Wave6ProbeError(
            "CUDA was initialized before CPU comparison", code="wave6.gpu_forbidden"
        )
    source_plan = create_pack_plan(
        examples,
        global_max_length=global_max_length,
        policy=SOURCE_ORDER_NEXT_FIT,
        seed=17,
        worker_count=1,
    )
    expected_inputs = [item.to_dict() for item in source_plan.inputs]
    measured: list[dict[str, Any]] = []
    for spec in ARM_GRID:
        one = _build_bundle(
            examples,
            spec=spec,
            global_max_length=global_max_length,
            worker_count=1,
            strict_replay=True,
        )
        eight = _build_bundle(
            examples,
            spec=spec,
            global_max_length=global_max_length,
            worker_count=8,
            strict_replay=False,
        )
        if one["projection"] != eight["projection"]:
            raise Wave6ProbeError(
                "worker-count planner semantics differ", code="wave6.worker_equality"
            )
        if one["projection"]["inputs"] != expected_inputs:
            raise Wave6ProbeError(
                "planner input or row identity drifted", code="wave6.semantic_oracle"
            )
        resume = _fragmented_resume_oracle(
            examples,
            spec=spec,
            global_max_length=global_max_length,
            expected=one,
        )
        memberships = [pack["input_ordinals"] for pack in one["projection"]["packs"]]
        rejected = [
            item["input_ordinal"] for item in one["projection"]["rejected_examples"]
        ]
        dispositions = [
            ordinal for membership in memberships for ordinal in membership
        ] + rejected
        if sorted(dispositions) != list(range(len(examples))) or len(
            dispositions
        ) != len(set(dispositions)):
            raise Wave6ProbeError(
                "planner each-once coverage failed", code="wave6.semantic_oracle"
            )
        used = sum(pack["encoded_length"] for pack in one["projection"]["packs"])
        pack_count = len(memberships)
        capacity = pack_count * global_max_length
        measured.append(
            {
                "spec": dict(spec),
                "bundle": one,
                "memberships": memberships,
                "rejected": rejected,
                "resume": resume,
                "metrics": {
                    "used_tokens": used,
                    "capacity_tokens": capacity,
                    "pack_count": pack_count,
                    "tail_waste": capacity - used,
                    "utilization": 0.0 if capacity == 0 else used / capacity,
                    **one["measurement"],
                },
                "worker": {
                    "compared_worker_counts": [1, 8],
                    "exact_semantic_projection_equal": True,
                    "worker_1_projection_sha256": one["projection_sha256"],
                    "worker_8_projection_sha256": eight["projection_sha256"],
                    "scope": "identical_already_encoded_examples_planner_semantics",
                },
            }
        )
    reference_pairs = _co_present_pairs(measured[0]["memberships"])
    arms = []
    for item in measured:
        ordering = _ordering_metrics(
            item["memberships"], reference_pairs=reference_pairs
        )
        is_reference = item["spec"]["policy"] == SOURCE_ORDER_NEXT_FIT
        disposition = (
            "compatibility_reference"
            if is_reference
            else (
                "survives_cpu_gate_requires_matched_training"
                if ordering["moved_example_count"] > 0
                else "cpu_equivalent_order_no_changed_order_training_arm"
            )
        )
        arms.append(
            {
                **item["spec"],
                "policy_identity": build_pack_plan_policy_identity(
                    policy=item["spec"]["policy"],
                    window_size=item["spec"]["window_size"],
                    lookahead=item["spec"]["lookahead"],
                    seed=17,
                    worker_count=1,
                ),
                "pack_memberships": item["memberships"],
                "rejected_ordinals": item["rejected"],
                "metrics": item["metrics"],
                "ordering_change": ordering,
                "semantic_oracle": {
                    "each_once_exact": True,
                    "intra_image_row_identity_order_exact": True,
                    "strict_pack_plan_roundtrip": True,
                    "strict_replay_exact": True,
                    "worker_count_1_vs_8_exact": True,
                },
                "worker_equality": item["worker"],
                "resume_cursor_oracle": item["resume"],
                "boundedness": item["bundle"]["bounded"],
                "cpu_disposition": disposition,
            }
        )
    if torch.cuda.is_initialized():
        raise Wave6ProbeError(
            "CUDA initialized during CPU comparison", code="wave6.gpu_forbidden"
        )
    receipt_payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "completed_cpu_only_training_required",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": checked["artifact_targets"]["plan"],
        "plan_sha256": checked["plan_sha256"],
        "plan_file_sha256": sha256_file(checked["artifact_targets"]["plan"]),
        "source_owners": checked["source_owners"],
        "config": checked["config"],
        "w0_cpu_cache_binding": checked["w0_cpu_cache_binding"],
        "w0_cpu_encoded_stream": observed_stream,
        "execution_attestation": {
            "cpu_only": True,
            "cuda_initialized_before_or_after": False,
            "training_executed": False,
            "model_loaded": False,
            "cache_written": False,
            "adaptive_thresholds_used": False,
            "retries_executed": 0,
            "measurement_repetitions_per_arm": 1,
        },
        "arms": arms,
        "research_disposition": _research_disposition(arms),
    }
    receipt = finalize_artifact(receipt_payload, hash_field="receipt_sha256")
    validate_receipt(receipt, expected_plan=checked)
    return receipt


def _validate_metrics(
    arm: Mapping[str, Any], *, global_max_length: int, expected_used: int
) -> None:
    metrics = _mapping(arm.get("metrics"), code="wave6.metrics")
    _require_exact_fields(
        metrics,
        {
            "used_tokens",
            "capacity_tokens",
            "pack_count",
            "tail_waste",
            "utilization",
            "planning_wall_seconds",
            "process_rss_before_bytes",
            "process_rss_after_bytes",
            "process_rss_high_water_bytes",
        },
        code="wave6.metrics",
    )
    memberships = arm.get("pack_memberships")
    if not isinstance(memberships, list) or not all(
        isinstance(item, list) and item for item in memberships
    ):
        raise Wave6ProbeError("pack memberships are invalid", code="wave6.metrics")
    pack_count = len(memberships)
    capacity = pack_count * global_max_length
    expected = {
        "used_tokens": expected_used,
        "capacity_tokens": capacity,
        "pack_count": pack_count,
        "tail_waste": capacity - expected_used,
        "utilization": 0.0 if capacity == 0 else expected_used / capacity,
    }
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise Wave6ProbeError("planner metrics are inconsistent", code="wave6.metrics")
    for field in ("planning_wall_seconds",):
        value = metrics.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise Wave6ProbeError("wall metric is invalid", code="wave6.metrics")
    for field in (
        "process_rss_before_bytes",
        "process_rss_after_bytes",
        "process_rss_high_water_bytes",
    ):
        value = metrics.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise Wave6ProbeError("RSS metric is invalid", code="wave6.metrics")


def validate_receipt(
    payload: Mapping[str, Any], *, expected_plan: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_plan(expected_plan, require_receipt_absent=False)
    receipt = _validate_finalized(
        payload, schema=RECEIPT_SCHEMA, hash_field="receipt_sha256"
    )
    _require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "finished_at",
            "plan_path",
            "plan_sha256",
            "plan_file_sha256",
            "source_owners",
            "config",
            "w0_cpu_cache_binding",
            "w0_cpu_encoded_stream",
            "execution_attestation",
            "arms",
            "research_disposition",
            "receipt_sha256",
        },
        code="wave6.receipt",
    )
    if receipt.get("status") != "completed_cpu_only_training_required":
        raise Wave6ProbeError("receipt status is invalid", code="wave6.receipt")
    try:
        datetime.fromisoformat(str(receipt["finished_at"]))
    except ValueError as exc:
        raise Wave6ProbeError(
            "receipt timestamp is invalid", code="wave6.receipt"
        ) from exc
    if (
        receipt.get("plan_path") != plan["artifact_targets"]["plan"]
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("plan_file_sha256")
        != sha256_file(plan["artifact_targets"]["plan"])
        or receipt.get("source_owners") != plan["source_owners"]
        or receipt.get("config") != plan["config"]
        or receipt.get("w0_cpu_cache_binding") != plan["w0_cpu_cache_binding"]
        or receipt.get("w0_cpu_encoded_stream") != plan["w0_cpu_encoded_stream"]
    ):
        raise Wave6ProbeError(
            "receipt plan binding drifted", code="wave6.receipt_binding"
        )
    expected_attestation = {
        "cpu_only": True,
        "cuda_initialized_before_or_after": False,
        "training_executed": False,
        "model_loaded": False,
        "cache_written": False,
        "adaptive_thresholds_used": False,
        "retries_executed": 0,
        "measurement_repetitions_per_arm": 1,
    }
    if receipt.get("execution_attestation") != expected_attestation:
        raise Wave6ProbeError(
            "execution attestation is invalid", code="wave6.execution_attestation"
        )
    arms = receipt.get("arms")
    if not isinstance(arms, list) or len(arms) != len(ARM_GRID):
        raise Wave6ProbeError("receipt arms are invalid", code="wave6.arms")
    input_count = int(plan["w0_cpu_encoded_stream"]["input_count"])
    used_tokens = int(plan["w0_cpu_encoded_stream"]["encoded_length_sum"])
    global_max_length = int(
        plan["config"]["resolved_contract"]["planner_global_max_length"]
    )
    reference_pairs: list[list[int]] | None = None
    for arm, spec in zip(arms, ARM_GRID, strict=True):
        required = {
            "arm_id",
            "policy",
            "window_size",
            "lookahead",
            "policy_identity",
            "pack_memberships",
            "rejected_ordinals",
            "metrics",
            "ordering_change",
            "semantic_oracle",
            "worker_equality",
            "resume_cursor_oracle",
            "boundedness",
            "cpu_disposition",
        }
        _require_exact_fields(
            _mapping(arm, code="wave6.arms"), required, code="wave6.arms"
        )
        if any(
            arm.get(field) != spec[field]
            for field in ("arm_id", "policy", "window_size", "lookahead")
        ):
            raise Wave6ProbeError("arm grid binding drifted", code="wave6.arms")
        expected_policy = build_pack_plan_policy_identity(
            policy=spec["policy"],
            window_size=spec["window_size"],
            lookahead=spec["lookahead"],
            seed=17,
            worker_count=1,
        )
        if arm.get("policy_identity") != expected_policy:
            raise Wave6ProbeError("arm policy identity drifted", code="wave6.arms")
        memberships = arm["pack_memberships"]
        rejected = arm["rejected_ordinals"]
        dispositions = [
            ordinal for membership in memberships for ordinal in membership
        ] + list(rejected)
        if sorted(dispositions) != list(range(input_count)) or len(dispositions) != len(
            set(dispositions)
        ):
            raise Wave6ProbeError(
                "each-once coverage is invalid", code="wave6.semantic_oracle"
            )
        semantic = arm.get("semantic_oracle")
        if semantic != {
            "each_once_exact": True,
            "intra_image_row_identity_order_exact": True,
            "strict_pack_plan_roundtrip": True,
            "strict_replay_exact": True,
            "worker_count_1_vs_8_exact": True,
        }:
            raise Wave6ProbeError(
                "semantic oracle is invalid", code="wave6.semantic_oracle"
            )
        _validate_metrics(
            arm, global_max_length=global_max_length, expected_used=used_tokens
        )
        pairs = _co_present_pairs(memberships)
        if reference_pairs is None:
            reference_pairs = pairs
        expected_ordering = _ordering_metrics(
            memberships, reference_pairs=reference_pairs
        )
        if arm.get("ordering_change") != expected_ordering:
            raise Wave6ProbeError(
                "ordering metrics are invalid", code="wave6.ordering_metrics"
            )
        worker = arm.get("worker_equality")
        if (
            not isinstance(worker, Mapping)
            or set(worker)
            != {
                "compared_worker_counts",
                "exact_semantic_projection_equal",
                "worker_1_projection_sha256",
                "worker_8_projection_sha256",
                "scope",
            }
            or worker.get("compared_worker_counts") != [1, 8]
            or worker.get("exact_semantic_projection_equal") is not True
            or worker.get("worker_1_projection_sha256")
            != worker.get("worker_8_projection_sha256")
            or not _is_sha256(worker.get("worker_1_projection_sha256"))
            or worker.get("scope")
            != "identical_already_encoded_examples_planner_semantics"
        ):
            raise Wave6ProbeError(
                "worker equality is invalid", code="wave6.worker_equality"
            )
        resume = arm.get("resume_cursor_oracle")
        expected_applicable = spec["policy"] == ONLINE_WINDOW_BINPACK
        if (
            not isinstance(resume, Mapping)
            or set(resume)
            != {
                "applicable",
                "fragmented_resume_exact",
                "terminal_cursor_complete",
                "strict_fragment_chain_verified",
                "fragment_count",
            }
            or resume.get("applicable") is not expected_applicable
            or any(
                resume.get(field) is not True
                for field in (
                    "fragmented_resume_exact",
                    "terminal_cursor_complete",
                    "strict_fragment_chain_verified",
                )
            )
            or not isinstance(resume.get("fragment_count"), int)
            or not 1 <= resume["fragment_count"] <= MAX_FRAGMENT_COUNT
        ):
            raise Wave6ProbeError(
                "resume cursor oracle is invalid", code="wave6.resume_cursor"
            )
        bounded = arm.get("boundedness")
        if (
            not isinstance(bounded, Mapping)
            or set(bounded)
            != {
                "pending_limit",
                "max_pending_items_observed",
                "cursor_byte_budget",
                "max_cursor_bytes_observed",
                "fragment_item_budget",
                "max_fragment_items_observed",
                "fragment_byte_budget",
                "max_fragment_bytes_observed",
                "within_all_declared_bounds",
            }
            or bounded.get("within_all_declared_bounds") is not True
            or bounded["max_pending_items_observed"] > bounded["pending_limit"]
            or bounded["max_cursor_bytes_observed"] > bounded["cursor_byte_budget"]
            or bounded["max_fragment_items_observed"] > bounded["fragment_item_budget"]
            or bounded["max_fragment_bytes_observed"] > bounded["fragment_byte_budget"]
        ):
            raise Wave6ProbeError(
                "planner bounds are invalid", code="wave6.boundedness"
            )
        expected_disposition = (
            "compatibility_reference"
            if spec["policy"] == SOURCE_ORDER_NEXT_FIT
            else (
                "survives_cpu_gate_requires_matched_training"
                if expected_ordering["moved_example_count"] > 0
                else "cpu_equivalent_order_no_changed_order_training_arm"
            )
        )
        if arm.get("cpu_disposition") != expected_disposition:
            raise Wave6ProbeError(
                "CPU disposition is invalid", code="wave6.research_disposition"
            )
    if receipt.get("research_disposition") != _research_disposition(arms):
        raise Wave6ProbeError(
            "research disposition is invalid", code="wave6.research_disposition"
        )
    canonical_json_bytes(receipt)
    return receipt


def _validate_arm_execution(
    execution: Any,
    *,
    spec: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _mapping(execution, code="wave6.observation")
    _require_exact_fields(
        value,
        {
            "arm_id",
            "policy",
            "window_size",
            "lookahead",
            "production_plan_identity",
            "semantic_order_sha256",
            "pack_memberships",
            "semantic_oracle",
            "resume_cursor_oracle",
            "boundedness",
            "metrics",
        },
        code="wave6.observation",
    )
    if any(
        value.get(field) != spec[field]
        for field in ("arm_id", "policy", "window_size", "lookahead")
    ):
        raise Wave6ProbeError("arm differs from grid", code="wave6.observation")
    identity = _mapping(
        value["production_plan_identity"], code="wave6.production_plan_identity"
    )
    if (
        set(identity)
        != {
            "mode",
            "full_plan_sha256",
            "upstream_plan_or_fragment_chain_sha256",
            "pipeline_receipt_sha256",
            "pack_fragment_identity_sha256",
            "worker_count_bound_semantically",
        }
        or any(
            not _is_sha256(identity[field])
            for field in (
                "full_plan_sha256",
                "upstream_plan_or_fragment_chain_sha256",
                "pipeline_receipt_sha256",
                "pack_fragment_identity_sha256",
            )
        )
        or identity["worker_count_bound_semantically"] is not True
        or not _is_sha256(value["semantic_order_sha256"])
    ):
        raise Wave6ProbeError(
            "production plan identity is invalid",
            code="wave6.production_plan_identity",
        )
    memberships = value["pack_memberships"]
    if not isinstance(memberships, list):
        raise Wave6ProbeError("memberships are invalid", code="wave6.semantic_oracle")
    ordinals = [ordinal for membership in memberships for ordinal in membership]
    input_count = int(plan["w0_cpu_encoded_stream"]["input_count"])
    if sorted(ordinals) != list(range(input_count)) or len(ordinals) != len(
        set(ordinals)
    ):
        raise Wave6ProbeError(
            "production membership is not each-once", code="wave6.semantic_oracle"
        )
    expected_semantic = {
        "each_once_exact": True,
        "intra_image_row_identity_order_exact": True,
        "strict_pack_plan_roundtrip": True,
        "strict_replay_exact": True,
        "production_pipeline_matches_strict_replay": True,
    }
    if value["semantic_oracle"] != expected_semantic:
        raise Wave6ProbeError(
            "production semantic oracle is invalid", code="wave6.semantic_oracle"
        )
    resume = _mapping(value["resume_cursor_oracle"], code="wave6.resume_cursor")
    if (
        resume.get("applicable") is not (spec["policy"] == ONLINE_WINDOW_BINPACK)
        or any(
            resume.get(field) is not True
            for field in (
                "fragmented_resume_exact",
                "terminal_cursor_complete",
                "strict_fragment_chain_verified",
            )
        )
        or not isinstance(resume.get("fragment_count"), int)
        or not 1 <= resume["fragment_count"] <= MAX_FRAGMENT_COUNT
    ):
        raise Wave6ProbeError("resume oracle is invalid", code="wave6.resume_cursor")
    bounded = _mapping(value["boundedness"], code="wave6.boundedness")
    if (
        bounded.get("within_all_declared_bounds") is not True
        or bounded["max_pending_items_observed"] > bounded["pending_limit"]
        or bounded["max_cursor_bytes_observed"] > bounded["cursor_byte_budget"]
        or bounded["max_fragment_items_observed"] > bounded["fragment_item_budget"]
        or bounded["max_fragment_bytes_observed"] > bounded["fragment_byte_budget"]
    ):
        raise Wave6ProbeError("planner bounds are invalid", code="wave6.boundedness")
    metrics = _mapping(value["metrics"], code="wave6.metrics")
    expected_used = int(plan["w0_cpu_encoded_stream"]["encoded_length_sum"])
    capacity = len(memberships) * int(
        plan["config"]["resolved_contract"]["planner_global_max_length"]
    )
    if any(
        metrics.get(field) != expected
        for field, expected in {
            "used_tokens": expected_used,
            "capacity_tokens": capacity,
            "pack_count": len(memberships),
            "tail_waste": capacity - expected_used,
            "utilization": expected_used / capacity,
        }.items()
    ) or not isinstance(metrics.get("planning_wall_seconds"), (int, float)):
        raise Wave6ProbeError("arm metrics are invalid", code="wave6.metrics")
    return value


def _validate_observed_arm_v3_determinant(
    value: Any,
    *,
    expected_projection: Mapping[str, Any],
    real_scope: bool,
) -> None:
    observed = dict(_mapping(value, code="wave6.arm_v3_determinant"))
    aggregate = observed.pop("full_aggregate_fingerprint", None)
    full_determinants = observed.pop("full_determinants", None)
    if observed != expected_projection:
        raise Wave6ProbeError(
            "arm current-v3 projection is invalid",
            code="wave6.arm_v3_determinant",
        )
    if not real_scope:
        if aggregate is not None or full_determinants is not None:
            raise Wave6ProbeError(
                "fixture arm must not claim a full current-v3 determinant",
                code="wave6.arm_v3_determinant",
            )
        return
    if not _is_sha256(aggregate) or not isinstance(full_determinants, Mapping):
        raise Wave6ProbeError(
            "real arm requires its full current-v3 determinant",
            code="wave6.arm_v3_determinant",
        )
    try:
        recomputed = pack_cache.packing_cache_fingerprint_from_determinants(
            full_determinants
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise Wave6ProbeError(
            "real arm full current-v3 determinant is invalid",
            code="wave6.arm_v3_determinant",
        ) from exc
    if (
        recomputed != aggregate
        or full_determinants.get("aggregate_fingerprint") != aggregate
        or full_determinants.get("version") != expected_projection["cache_version"]
        or full_determinants.get("registry_schema_version")
        != expected_projection["registry_schema_version"]
        or full_determinants.get("packing") != expected_projection["packing"]
    ):
        raise Wave6ProbeError(
            "real arm full current-v3 determinant is not bound to its projection",
            code="wave6.arm_v3_determinant",
        )


def validate_observation(
    payload: Mapping[str, Any],
    *,
    expected_plan: Mapping[str, Any],
    require_terminal_absent: bool = True,
) -> dict[str, Any]:
    plan = validate_plan(expected_plan, require_receipt_absent=require_terminal_absent)
    observation = _validate_finalized(
        payload, schema=OBSERVATION_SCHEMA, hash_field="observation_sha256"
    )
    _require_exact_fields(
        observation,
        {
            "schema",
            "status",
            "plan_sha256",
            "candidate_arm_id",
            "repetition_index",
            "position_index",
            "arm_id",
            "materialization",
            "arm",
            "process",
            "observation_sha256",
        },
        code="wave6.observation",
    )
    candidates = {str(spec["arm_id"]): spec for spec in CANDIDATE_GRID}
    candidate_arm_id = observation.get("candidate_arm_id")
    repetition = observation.get("repetition_index")
    position = observation.get("position_index")
    if (
        observation.get("status") != "accepted"
        or observation.get("plan_sha256") != plan["plan_sha256"]
        or candidate_arm_id not in candidates
        or not isinstance(repetition, int)
        or isinstance(repetition, bool)
        or not 0 <= repetition < len(POLICY_PAIR_ORDERS)
        or position not in {0, 1}
    ):
        raise Wave6ProbeError(
            "observation coordinate is invalid", code="wave6.observation_coordinate"
        )
    label = POLICY_PAIR_ORDERS[repetition][position]
    expected_arm_id = candidate_arm_id if label == "candidate" else str(label)
    if observation.get("arm_id") != expected_arm_id:
        raise Wave6ProbeError(
            "observation arm coordinate is invalid",
            code="wave6.observation_coordinate",
        )
    materialization = _mapping(
        observation["materialization"], code="wave6.encoded_materialization"
    )
    workers = _mapping(
        materialization.get("workers"), code="wave6.encoded_materialization"
    )
    if (
        materialization.get("production_encoder_seam")
        != "src.training.pipeline._build_encoded_examples_for_dataset"
        or materialization.get("worker_counts") != [1, 8]
        or materialization.get("encoded_materialization_exact") is not True
        or materialization.get("encoded_stream") != plan["w0_cpu_encoded_stream"]
        or set(workers) != {"1", "8"}
        or any(not isinstance(worker, Mapping) for worker in workers.values())
        or any(
            worker.get("raw_example_count")
            != plan["w0_cpu_encoded_stream"]["input_count"]
            for worker in workers.values()
            if isinstance(worker, Mapping)
        )
    ):
        raise Wave6ProbeError(
            "encoded materialization is invalid", code="wave6.encoded_materialization"
        )
    for worker in workers.values():
        if plan["scope"] == "temporary_fixture":
            if (
                worker.get("mode") != "deterministic_fixture"
                or worker.get("private_v3_determinants") is not None
            ):
                raise Wave6ProbeError(
                    "fixture materialization is invalid",
                    code="wave6.encoded_materialization",
                )
        else:
            source_determinants = _mapping(
                worker.get("private_v3_determinants"), code="wave6.private_v3"
            )
            if (
                worker.get("mode") != "real_w0_train"
                or source_determinants.get("version")
                != pack_cache.PACKING_CACHE_VERSION
                or source_determinants.get("registry_schema_version")
                != pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION
                or not _is_sha256(source_determinants.get("aggregate_fingerprint"))
            ):
                raise Wave6ProbeError(
                    "real materialization determinant is invalid",
                    code="wave6.private_v3",
                )
    process = _mapping(observation["process"], code="wave6.process_metrics")
    if (
        set(process)
        != {
            "pid",
            "measurement_scope",
            "wall_seconds",
            "rss_before_bytes",
            "rss_after_bytes",
            "rss_delta_bytes",
            "rss_high_water_bytes",
            "rss_high_water_scope",
        }
        or not isinstance(process["pid"], int)
        or process["pid"] <= 0
        or process["measurement_scope"]
        != "single_policy_arm_entry_to_return_including_worker_1_8_proof"
        or process["rss_high_water_scope"] != "fresh_process_lifetime"
        or not isinstance(process["wall_seconds"], (int, float))
        or isinstance(process["wall_seconds"], bool)
        or not math.isfinite(process["wall_seconds"])
        or process["wall_seconds"] < 0
        or any(
            not isinstance(process[field], int)
            or isinstance(process[field], bool)
            or process[field] < 0
            for field in (
                "rss_before_bytes",
                "rss_after_bytes",
                "rss_delta_bytes",
                "rss_high_water_bytes",
            )
        )
        or process["rss_delta_bytes"]
        != max(0, process["rss_after_bytes"] - process["rss_before_bytes"])
        or process["rss_high_water_bytes"] <= 0
    ):
        raise Wave6ProbeError(
            "process metrics are invalid", code="wave6.process_metrics"
        )
    spec = next(spec for spec in ARM_GRID if spec["arm_id"] == expected_arm_id)
    arm = _mapping(observation["arm"], code="wave6.observation")
    _require_exact_fields(
        arm,
        {
            "arm_id",
            "policy",
            "window_size",
            "lookahead",
            "worker_executions",
            "worker_comparison",
            "current_v3_determinants",
            "semantic_oracle",
            "resume_cursor_oracle",
            "boundedness",
        },
        code="wave6.observation",
    )
    if any(
        arm.get(field) != spec[field]
        for field in ("arm_id", "policy", "window_size", "lookahead")
    ):
        raise Wave6ProbeError("observation arm is invalid", code="wave6.observation")
    executions = _mapping(arm["worker_executions"], code="wave6.observation")
    if set(executions) != {"1", "8"}:
        raise Wave6ProbeError("worker executions are invalid", code="wave6.observation")
    one = _validate_arm_execution(executions["1"], spec=spec, plan=plan)
    eight = _validate_arm_execution(executions["8"], spec=spec, plan=plan)
    expected_worker_comparison = {
        "encoded_materialization_exact": True,
        "semantic_order_exact": True,
        "full_plan_sha256_worker_1": one["production_plan_identity"][
            "full_plan_sha256"
        ],
        "full_plan_sha256_worker_8": eight["production_plan_identity"][
            "full_plan_sha256"
        ],
        "full_plan_sha256_equal": one["production_plan_identity"]["full_plan_sha256"]
        == eight["production_plan_identity"]["full_plan_sha256"],
        "worker_count_disposition": (
            "semantic_pending_upstream_materialization_equality"
        ),
    }
    if (
        one["semantic_order_sha256"] != eight["semantic_order_sha256"]
        or one["pack_memberships"] != eight["pack_memberships"]
        or arm["worker_comparison"] != expected_worker_comparison
        or arm["semantic_oracle"] != one["semantic_oracle"]
        or arm["resume_cursor_oracle"] != one["resume_cursor_oracle"]
        or arm["boundedness"] != one["boundedness"]
    ):
        raise Wave6ProbeError(
            "worker equality is invalid", code="wave6.worker_equality"
        )
    determinants = _mapping(
        arm["current_v3_determinants"], code="wave6.arm_v3_determinant"
    )
    if set(determinants) != {"1", "8"}:
        raise Wave6ProbeError(
            "arm determinant workers differ", code="wave6.arm_v3_determinant"
        )
    for worker_count in WORKER_COUNTS:
        expected = plan["arm_v3_determinant_bindings"][expected_arm_id][
            str(worker_count)
        ]
        _validate_observed_arm_v3_determinant(
            determinants[str(worker_count)],
            expected_projection=expected,
            real_scope=plan["scope"] == "frozen_w0_cpu",
        )
    return observation


def _paired_observations(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for candidate_spec in CANDIDATE_GRID:
        candidate_arm_id = str(candidate_spec["arm_id"])
        for repetition_index, order in enumerate(POLICY_PAIR_ORDERS):
            repetition = sorted(
                (
                    item
                    for item in observations
                    if item["candidate_arm_id"] == candidate_arm_id
                    and item["repetition_index"] == repetition_index
                ),
                key=lambda item: item["position_index"],
            )
            if len(repetition) != 2 or [
                item["position_index"] for item in repetition
            ] != [0, 1]:
                raise Wave6ProbeError(
                    "policy-paired observation is incomplete",
                    code="wave6.paired_observation",
                )
            expected_arm_ids = [
                candidate_arm_id if label == "candidate" else str(label)
                for label in order
            ]
            if [item["arm_id"] for item in repetition] != expected_arm_ids:
                raise Wave6ProbeError(
                    "policy-pair order differs from A/B contract",
                    code="wave6.paired_observation",
                )
            by_arm = {item["arm_id"]: item for item in repetition}
            source = by_arm[SOURCE_ORDER_NEXT_FIT]
            candidate = by_arm[candidate_arm_id]
            source_execution = source["arm"]["worker_executions"]["1"]
            candidate_execution = candidate["arm"]["worker_executions"]["1"]
            ordering = _ordering_metrics(
                candidate_execution["pack_memberships"],
                reference_pairs=_co_present_pairs(source_execution["pack_memberships"]),
            )
            pairs.append(
                {
                    "candidate_arm_id": candidate_arm_id,
                    "repetition_index": repetition_index,
                    "order": list(order),
                    "source_pid": source["process"]["pid"],
                    "candidate_pid": candidate["process"]["pid"],
                    "source_observation_sha256": source["observation_sha256"],
                    "candidate_observation_sha256": candidate["observation_sha256"],
                    "ordering_change": ordering,
                    "resource_deltas": {
                        "wall_seconds": candidate["process"]["wall_seconds"]
                        - source["process"]["wall_seconds"],
                        "rss_delta_bytes": candidate["process"]["rss_delta_bytes"]
                        - source["process"]["rss_delta_bytes"],
                        "rss_high_water_bytes": candidate["process"][
                            "rss_high_water_bytes"
                        ]
                        - source["process"]["rss_high_water_bytes"],
                    },
                    "planner_deltas": {
                        "utilization": candidate_execution["metrics"]["utilization"]
                        - source_execution["metrics"]["utilization"],
                        "pack_count": candidate_execution["metrics"]["pack_count"]
                        - source_execution["metrics"]["pack_count"],
                        "tail_waste": candidate_execution["metrics"]["tail_waste"]
                        - source_execution["metrics"]["tail_waste"],
                    },
                }
            )
    return pairs


def _metric_summary(values: Sequence[int | float]) -> dict[str, Any]:
    if len(values) != 3:
        raise Wave6ProbeError(
            "metric summary requires three observations", code="wave6.aggregate"
        )
    observed = list(values)
    return {
        "observations": observed,
        "median": statistics.median(observed),
        "minimum": min(observed),
        "maximum": max(observed),
        "range": max(observed) - min(observed),
    }


def _controller_aggregate(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = _paired_observations(observations)
    by_sha = {item["observation_sha256"]: item for item in observations}
    candidates: dict[str, Any] = {}
    for spec in CANDIDATE_GRID:
        arm_id = str(spec["arm_id"])
        candidate_pairs = [pair for pair in pairs if pair["candidate_arm_id"] == arm_id]
        source_observations = [
            by_sha[pair["source_observation_sha256"]] for pair in candidate_pairs
        ]
        candidate_observations = [
            by_sha[pair["candidate_observation_sha256"]] for pair in candidate_pairs
        ]
        raw_summaries = {
            role: {
                output_name: _metric_summary(
                    [item["process"][field] for item in role_observations]
                )
                for output_name, field in (
                    ("wall_seconds", "wall_seconds"),
                    ("rss_delta_bytes", "rss_delta_bytes"),
                    ("rss_high_water_bytes", "rss_high_water_bytes"),
                )
            }
            for role, role_observations in (
                ("source", source_observations),
                ("candidate", candidate_observations),
            )
        }
        candidates[arm_id] = {
            **raw_summaries,
            "wall_seconds_delta": _metric_summary(
                [pair["resource_deltas"]["wall_seconds"] for pair in candidate_pairs]
            ),
            "rss_delta_bytes_delta": _metric_summary(
                [pair["resource_deltas"]["rss_delta_bytes"] for pair in candidate_pairs]
            ),
            "rss_hwm_bytes_delta": _metric_summary(
                [
                    pair["resource_deltas"]["rss_high_water_bytes"]
                    for pair in candidate_pairs
                ]
            ),
            "utilization_delta": _metric_summary(
                [pair["planner_deltas"]["utilization"] for pair in candidate_pairs]
            ),
            "pack_count_delta": _metric_summary(
                [pair["planner_deltas"]["pack_count"] for pair in candidate_pairs]
            ),
            "tail_waste_delta": _metric_summary(
                [pair["planner_deltas"]["tail_waste"] for pair in candidate_pairs]
            ),
        }
    return {
        "accepted_pair_count_per_candidate": 3,
        "fresh_process_count": len(observations),
        "candidates": candidates,
    }


def _controller_disposition(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not observations:
        raise Wave6ProbeError(
            "controller disposition requires observations",
            code="wave6.research_disposition",
        )
    pairs = _paired_observations(observations)
    research_arms: list[dict[str, Any]] = [
        {
            "arm_id": "source_order_next_fit",
            "ordering_change": {"moved_example_count": 0},
            "cpu_disposition": "compatibility_reference",
        }
    ]
    for spec in CANDIDATE_GRID:
        arm_id = str(spec["arm_id"])
        ordering = next(
            pair["ordering_change"]
            for pair in pairs
            if pair["candidate_arm_id"] == arm_id
        )
        research_arms.append(
            {
                "arm_id": arm_id,
                "ordering_change": ordering,
                "cpu_disposition": (
                    "survives_cpu_gate_requires_matched_training"
                    if ordering["moved_example_count"] > 0
                    else "cpu_equivalent_order_no_changed_order_training_arm"
                ),
            }
        )
    return {
        **_research_disposition(research_arms),
        "worker_count_remains_semantic": True,
        "next_gate": "matched_five_step_training_for_surviving_changed_order_arms",
    }


def validate_controller_receipt(
    payload: Mapping[str, Any], *, expected_plan: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_plan(expected_plan, require_receipt_absent=False)
    receipt = _validate_finalized(
        payload,
        schema=CONTROLLER_RECEIPT_SCHEMA,
        hash_field="controller_receipt_sha256",
    )
    _require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "finished_at",
            "plan_path",
            "plan_sha256",
            "plan_file_sha256",
            "pair_orders",
            "observations",
            "paired_observations",
            "aggregate",
            "research_disposition",
            "controller_receipt_sha256",
        },
        code="wave6.controller_receipt",
    )
    if (
        receipt.get("status") != "completed_cpu_only_training_required"
        or receipt.get("plan_path") != plan["artifact_targets"]["plan"]
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("plan_file_sha256")
        != sha256_file(plan["artifact_targets"]["plan"])
        or receipt.get("pair_orders") != [list(order) for order in POLICY_PAIR_ORDERS]
    ):
        raise Wave6ProbeError(
            "controller receipt binding is invalid",
            code="wave6.controller_receipt",
        )
    observations = receipt.get("observations")
    expected_observation_count = len(CANDIDATE_GRID) * 3 * 2
    if (
        not isinstance(observations, list)
        or len(observations) != expected_observation_count
    ):
        raise Wave6ProbeError(
            "controller observation count is invalid",
            code="wave6.controller_receipt",
        )
    checked_observations = [
        validate_observation(
            item,
            expected_plan=plan,
            require_terminal_absent=False,
        )
        for item in observations
    ]
    coordinates = [
        (
            item["candidate_arm_id"],
            item["repetition_index"],
            item["position_index"],
            item["arm_id"],
        )
        for item in checked_observations
    ]
    expected_coordinates = []
    for spec in CANDIDATE_GRID:
        candidate_arm_id = str(spec["arm_id"])
        for repetition_index, order in enumerate(POLICY_PAIR_ORDERS):
            for position_index, label in enumerate(order):
                expected_coordinates.append(
                    (
                        candidate_arm_id,
                        repetition_index,
                        position_index,
                        candidate_arm_id if label == "candidate" else str(label),
                    )
                )
    if (
        coordinates != expected_coordinates
        or len({item["process"]["pid"] for item in checked_observations})
        != expected_observation_count
    ):
        raise Wave6ProbeError(
            "controller observations are not fresh ordered processes",
            code="wave6.controller_receipt",
        )
    if receipt.get("paired_observations") != _paired_observations(
        checked_observations
    ) or receipt.get("aggregate") != _controller_aggregate(checked_observations):
        raise Wave6ProbeError(
            "controller paired aggregate is invalid", code="wave6.aggregate"
        )
    if receipt.get("research_disposition") != _controller_disposition(
        checked_observations
    ):
        raise Wave6ProbeError(
            "controller research disposition is invalid",
            code="wave6.research_disposition",
        )
    return receipt


def _canonical_research_source(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    lexical = Path(os.path.abspath(os.fspath(requested)))
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise Wave6ProbeError(
            "research-meaning source is unavailable",
            code="wave6.research_meaning",
        ) from exc
    if lexical != resolved or requested.is_symlink() or not resolved.is_file():
        raise Wave6ProbeError(
            "research-meaning source is not a canonical file",
            code="wave6.research_meaning",
        )
    return resolved


def _validate_research_meaning_binding(value: Mapping[str, Any]) -> dict[str, str]:
    binding = dict(_mapping(value, code="wave6.research_meaning"))
    fields = {
        "plan_sha256",
        "plan_file_sha256",
        "controller_receipt_sha256",
        "controller_receipt_file_sha256",
    }
    if set(binding) != fields or not all(
        _is_sha256(binding[field]) for field in fields
    ):
        raise Wave6ProbeError(
            "research-meaning source binding is invalid",
            code="wave6.research_meaning",
        )
    return {field: str(binding[field]) for field in fields}


def _authenticated_research_meaning_sources(
    *,
    plan_path: str | Path,
    receipt_path: str | Path,
    expected_binding: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, str]]:
    binding = _validate_research_meaning_binding(expected_binding)
    canonical_plan_path = _canonical_research_source(plan_path)
    canonical_receipt_path = _canonical_research_source(receipt_path)
    plan = _validate_finalized(
        load_strict_json(canonical_plan_path),
        schema=PLAN_SCHEMA,
        hash_field="plan_sha256",
    )
    receipt = _validate_finalized(
        load_strict_json(canonical_receipt_path),
        schema=CONTROLLER_RECEIPT_SCHEMA,
        hash_field="controller_receipt_sha256",
    )
    _require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "finished_at",
            "plan_path",
            "plan_sha256",
            "plan_file_sha256",
            "pair_orders",
            "observations",
            "paired_observations",
            "aggregate",
            "research_disposition",
            "controller_receipt_sha256",
        },
        code="wave6.research_meaning",
    )
    observed_binding = {
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": sha256_file(canonical_plan_path),
        "controller_receipt_sha256": receipt["controller_receipt_sha256"],
        "controller_receipt_file_sha256": sha256_file(canonical_receipt_path),
    }
    if (
        observed_binding != binding
        or receipt["status"] != "completed_cpu_only_training_required"
        or receipt["plan_path"] != str(canonical_plan_path)
        or receipt["plan_sha256"] != plan["plan_sha256"]
        or receipt["plan_file_sha256"] != observed_binding["plan_file_sha256"]
        or receipt["pair_orders"] != [list(order) for order in POLICY_PAIR_ORDERS]
    ):
        raise Wave6ProbeError(
            "research-meaning sources differ from the immutable binding",
            code="wave6.research_meaning",
        )
    return (
        canonical_plan_path,
        canonical_receipt_path,
        plan,
        receipt,
        binding,
    )


def _finite_metric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validated_research_meaning_pairs(
    receipt: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    observations = receipt.get("observations")
    pairs = receipt.get("paired_observations")
    expected_count = len(CANDIDATE_GRID) * len(POLICY_PAIR_ORDERS) * 2
    if not isinstance(observations, list) or len(observations) != expected_count:
        raise Wave6ProbeError(
            "research-meaning observation count is invalid",
            code="wave6.research_meaning",
        )
    observation_sha256: list[str] = []
    for observation in observations:
        if not isinstance(observation, Mapping) or not _is_sha256(
            observation.get("observation_sha256")
        ):
            raise Wave6ProbeError(
                "research-meaning observation identity is invalid",
                code="wave6.research_meaning",
            )
        observation_sha256.append(str(observation["observation_sha256"]))
    if len(set(observation_sha256)) != expected_count:
        raise Wave6ProbeError(
            "research-meaning observations are not unique",
            code="wave6.research_meaning",
        )
    if not isinstance(pairs, list) or len(pairs) * 2 != expected_count:
        raise Wave6ProbeError(
            "research-meaning pair count is invalid",
            code="wave6.research_meaning",
        )
    pair_fields = {
        "candidate_arm_id",
        "repetition_index",
        "order",
        "source_pid",
        "candidate_pid",
        "source_observation_sha256",
        "candidate_observation_sha256",
        "ordering_change",
        "resource_deltas",
        "planner_deltas",
    }
    ordering_fields = {
        "flattened_order_sha256",
        "moved_example_count",
        "absolute_displacement_sum",
        "absolute_displacement_max",
        "co_present_pair_count",
        "co_present_pairs_sha256",
        "added_vs_reference_count",
        "removed_vs_reference_count",
        "retained_vs_reference_count",
    }
    expected_coordinates = [
        (str(spec["arm_id"]), repetition_index, list(order))
        for spec in CANDIDATE_GRID
        for repetition_index, order in enumerate(POLICY_PAIR_ORDERS)
    ]
    checked: list[dict[str, Any]] = []
    referenced_sha256: list[str] = []
    pids: list[int] = []
    for pair, expected_coordinate in zip(pairs, expected_coordinates, strict=True):
        value = dict(_mapping(pair, code="wave6.research_meaning"))
        ordering = _mapping(value.get("ordering_change"), code="wave6.research_meaning")
        resources = _mapping(
            value.get("resource_deltas"), code="wave6.research_meaning"
        )
        planner = _mapping(value.get("planner_deltas"), code="wave6.research_meaning")
        coordinate = (
            value.get("candidate_arm_id"),
            value.get("repetition_index"),
            value.get("order"),
        )
        if (
            set(value) != pair_fields
            or coordinate != expected_coordinate
            or set(ordering) != ordering_fields
            or not _is_sha256(ordering.get("flattened_order_sha256"))
            or not _is_sha256(ordering.get("co_present_pairs_sha256"))
            or not all(
                isinstance(ordering[field], int)
                and not isinstance(ordering[field], bool)
                and ordering[field] >= 0
                for field in ordering_fields
                - {"flattened_order_sha256", "co_present_pairs_sha256"}
            )
            or set(resources)
            != {"wall_seconds", "rss_delta_bytes", "rss_high_water_bytes"}
            or set(planner) != {"utilization", "pack_count", "tail_waste"}
            or not all(_finite_metric(metric) for metric in resources.values())
            or not all(_finite_metric(metric) for metric in planner.values())
            or not _is_sha256(value.get("source_observation_sha256"))
            or not _is_sha256(value.get("candidate_observation_sha256"))
            or not isinstance(value.get("source_pid"), int)
            or isinstance(value.get("source_pid"), bool)
            or not isinstance(value.get("candidate_pid"), int)
            or isinstance(value.get("candidate_pid"), bool)
        ):
            raise Wave6ProbeError(
                "research-meaning raw pair is invalid",
                code="wave6.research_meaning",
            )
        referenced_sha256.extend(
            [
                str(value["source_observation_sha256"]),
                str(value["candidate_observation_sha256"]),
            ]
        )
        pids.extend([int(value["source_pid"]), int(value["candidate_pid"])])
        checked.append(deepcopy(value))
    if (
        set(referenced_sha256) != set(observation_sha256)
        or len(set(referenced_sha256)) != expected_count
        or len(set(pids)) != expected_count
    ):
        raise Wave6ProbeError(
            "research-meaning pairs do not cover the raw observations exactly once",
            code="wave6.research_meaning",
        )
    for spec in CANDIDATE_GRID:
        arm_id = str(spec["arm_id"])
        moved = [
            pair["ordering_change"]["moved_example_count"]
            for pair in checked
            if pair["candidate_arm_id"] == arm_id
        ]
        expected_changed_order = arm_id != "window_binpack_w8"
        if len(moved) != 3 or any(
            (count > 0) != expected_changed_order for count in moved
        ):
            raise Wave6ProbeError(
                "research-meaning semantic survivor set differs from r2",
                code="wave6.research_meaning",
            )
    return checked, observation_sha256


def _paired_research_metric_summary(values: Sequence[int | float]) -> dict[str, Any]:
    if len(values) != 3 or not all(_finite_metric(value) for value in values):
        raise Wave6ProbeError(
            "research-meaning metric requires three finite observations",
            code="wave6.research_meaning",
        )
    observations = list(values)
    median = statistics.median(observations)
    absolute_deviations = [abs(value - median) for value in observations]
    return {
        "observations": observations,
        "median": median,
        "minimum": min(observations),
        "maximum": max(observations),
        "median_absolute_deviation": statistics.median(absolute_deviations),
    }


def _research_meaning_metric_summaries(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = (
        ("wall_seconds_delta", "resource_deltas", "wall_seconds"),
        ("rss_delta_bytes_delta", "resource_deltas", "rss_delta_bytes"),
        ("rss_hwm_bytes_delta", "resource_deltas", "rss_high_water_bytes"),
        ("utilization_delta", "planner_deltas", "utilization"),
        ("pack_count_delta", "planner_deltas", "pack_count"),
        ("tail_waste_delta", "planner_deltas", "tail_waste"),
    )
    return {
        str(spec["arm_id"]): {
            output_name: _paired_research_metric_summary(
                [
                    pair[group][field]
                    for pair in pairs
                    if pair["candidate_arm_id"] == spec["arm_id"]
                ]
            )
            for output_name, group, field in metrics
        }
        for spec in CANDIDATE_GRID
    }


def _research_meaning_disposition() -> dict[str, Any]:
    survivors = [
        "window_binpack_w32",
        "online_window_binpack_l8",
        "online_window_binpack_l32",
    ]
    return {
        "candidate_dispositions": {
            "window_binpack_w8": "non_survivor_semantically_equivalent_order",
            **{
                arm_id: "semantic_survivor_requires_matched_five_step_training"
                for arm_id in survivors
            },
        },
        "semantic_survivors": survivors,
        "non_survivors": ["window_binpack_w8"],
        "planner_utilization_can_promote": False,
        "promotion_authorized": False,
        "optimization_dynamics_unmeasured": True,
        "worker_count_remains_semantic": True,
        "claim_boundary": (
            "CPU evidence establishes deterministic planner mechanics and descriptive "
            "resource/order metrics only; it does not establish quality, optimization "
            "dynamics, distributed skew, or end-to-end training benefit."
        ),
        "next_gate": "matched_five_step_training_for_surviving_changed_order_arms",
        "smallest_matched_training": {
            "step_count": 5,
            "reference_arm": "source_order_next_fit__matched5step_reference",
            "candidate_arms": [
                f"{arm_id}__matched5step_candidate" for arm_id in survivors
            ],
            "only_changed_factor": (
                "packing_policy_and_its_declared_window_or_lookahead"
            ),
            "identical_fields": [
                "images",
                "intra_image_rows_and_order",
                "encoded_tokens_and_supervision",
                "optimizer_budget",
                "five_step_schedule",
                "evaluation",
                "checkpoint_policy",
                "artifact_semantics",
            ],
        },
    }


def _build_research_meaning_packet_from_binding(
    *,
    plan_path: str | Path,
    receipt_path: str | Path,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    (
        canonical_plan_path,
        canonical_receipt_path,
        _plan,
        receipt,
        binding,
    ) = _authenticated_research_meaning_sources(
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=expected_binding,
    )
    pairs, observation_sha256 = _validated_research_meaning_pairs(receipt)
    payload = {
        "schema": RESEARCH_MEANING_SCHEMA,
        "status": "authenticated_r2_cpu_research_meaning",
        "source_artifacts": {
            "plan": {
                "path": str(canonical_plan_path),
                "plan_sha256": binding["plan_sha256"],
                "file_sha256": binding["plan_file_sha256"],
            },
            "controller_receipt": {
                "path": str(canonical_receipt_path),
                "controller_receipt_sha256": binding["controller_receipt_sha256"],
                "file_sha256": binding["controller_receipt_file_sha256"],
            },
        },
        "raw_observation_sha256": observation_sha256,
        "raw_paired_deltas": pairs,
        "paired_metric_summaries": _research_meaning_metric_summaries(pairs),
        "research_disposition": _research_meaning_disposition(),
    }
    return finalize_artifact(payload, hash_field="research_meaning_packet_sha256")


def build_research_meaning_packet(
    *, plan_path: str | Path, receipt_path: str | Path
) -> dict[str, Any]:
    return _build_research_meaning_packet_from_binding(
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=R2_RESEARCH_MEANING_BINDING,
    )


def _validate_research_meaning_packet_from_binding(
    payload: Mapping[str, Any],
    *,
    plan_path: str | Path,
    receipt_path: str | Path,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    packet = _validate_finalized(
        payload,
        schema=RESEARCH_MEANING_SCHEMA,
        hash_field="research_meaning_packet_sha256",
    )
    expected = _build_research_meaning_packet_from_binding(
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=expected_binding,
    )
    if packet != expected:
        raise Wave6ProbeError(
            "research-meaning packet differs from immutable raw r2 evidence",
            code="wave6.research_meaning",
        )
    return packet


def validate_research_meaning_packet(
    payload: Mapping[str, Any],
    *,
    plan_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    return _validate_research_meaning_packet_from_binding(
        payload,
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=R2_RESEARCH_MEANING_BINDING,
    )


def _publish_research_meaning_packet_from_binding(
    *,
    plan_path: str | Path,
    receipt_path: str | Path,
    packet_path: str | Path,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    target = _absent_target(packet_path)
    packet = _build_research_meaning_packet_from_binding(
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=expected_binding,
    )
    _validate_research_meaning_packet_from_binding(
        packet,
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=expected_binding,
    )
    publish_json_absent(target, packet)
    return packet


def publish_research_meaning_packet(
    *, plan_path: str | Path, receipt_path: str | Path, packet_path: str | Path
) -> dict[str, Any]:
    return _publish_research_meaning_packet_from_binding(
        plan_path=plan_path,
        receipt_path=receipt_path,
        packet_path=packet_path,
        expected_binding=R2_RESEARCH_MEANING_BINDING,
    )


def _bounded_error(exc: BaseException) -> dict[str, str]:
    message = str(exc)
    return {
        "type": type(exc).__name__[:128],
        "code": str(getattr(exc, "code", "wave6.child_failure"))[:128],
        "message": message[-MAX_ERROR_CHARS:],
    }


def validate_failure_receipt(
    payload: Mapping[str, Any], *, expected_plan: Mapping[str, Any]
) -> dict[str, Any]:
    # A terminal source-drift failure must remain publishable and verifiable
    # against the already-authenticated plan even though live owner validation
    # is now expected to fail.
    plan = _validate_finalized(
        expected_plan, schema=PLAN_SCHEMA, hash_field="plan_sha256"
    )
    targets = _mapping(plan.get("artifact_targets"), code="wave6.failure_receipt")
    if set(targets) != {"plan", "receipt", "failure_receipt"}:
        raise Wave6ProbeError(
            "failure plan targets are invalid", code="wave6.failure_receipt"
        )
    failure = _validate_finalized(
        payload,
        schema=FAILURE_RECEIPT_SCHEMA,
        hash_field="failure_receipt_sha256",
    )
    _require_exact_fields(
        failure,
        {
            "schema",
            "status",
            "finished_at",
            "plan_path",
            "plan_sha256",
            "failed_coordinate",
            "completed_observation_sha256",
            "error",
            "retry_count",
            "failure_receipt_sha256",
        },
        code="wave6.failure_receipt",
    )
    error = _mapping(failure.get("error"), code="wave6.failure_receipt")
    coordinate = _mapping(
        failure.get("failed_coordinate"), code="wave6.failure_receipt"
    )
    try:
        datetime.fromisoformat(str(failure["finished_at"]))
    except ValueError as exc:
        raise Wave6ProbeError(
            "terminal failure timestamp is invalid", code="wave6.failure_receipt"
        ) from exc
    repetition = coordinate.get("repetition_index")
    position = coordinate.get("position_index")
    candidate_arm_id = coordinate.get("candidate_arm_id")
    candidates = {str(spec["arm_id"]) for spec in CANDIDATE_GRID}
    if (
        failure.get("status") != "failed_terminal"
        or failure.get("plan_path") != plan["artifact_targets"]["plan"]
        or failure.get("plan_sha256") != plan["plan_sha256"]
        or failure.get("retry_count") != 0
        or Path(plan["artifact_targets"]["receipt"]).exists()
        or set(coordinate)
        != {"candidate_arm_id", "repetition_index", "position_index", "arm_id"}
        or candidate_arm_id not in candidates
        or not isinstance(repetition, int)
        or isinstance(repetition, bool)
        or not 0 <= repetition < len(POLICY_PAIR_ORDERS)
        or position not in {0, 1}
        or coordinate.get("arm_id")
        != (
            candidate_arm_id
            if POLICY_PAIR_ORDERS[repetition][position] == "candidate"
            else POLICY_PAIR_ORDERS[repetition][position]
        )
        or set(error) != {"type", "code", "message"}
        or any(not isinstance(error[field], str) or not error[field] for field in error)
        or len(error["message"]) > MAX_ERROR_CHARS
        or not isinstance(failure.get("completed_observation_sha256"), list)
        or len(failure["completed_observation_sha256"]) > len(CANDIDATE_GRID) * 3 * 2
        or len(set(failure["completed_observation_sha256"]))
        != len(failure["completed_observation_sha256"])
        or not all(
            _is_sha256(value) for value in failure["completed_observation_sha256"]
        )
    ):
        raise Wave6ProbeError(
            "terminal failure receipt is invalid", code="wave6.failure_receipt"
        )
    return failure


def _strict_json_text(payload: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise Wave6ProbeError(
            "child did not emit strict JSON", code="wave6.child_failure"
        ) from exc
    if not isinstance(value, dict):
        raise Wave6ProbeError(
            "child observation is not an object", code="wave6.child_failure"
        )
    return value


def _child_process_error(
    *, returncode: int, stdout: str, stderr: str
) -> Wave6ChildProcessError:
    diagnostic = stderr or stdout or "child emitted no diagnostic"
    exception_type = "UnknownChildError"
    for line in reversed(diagnostic.splitlines()):
        match = re.match(
            r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?::|$)",
            line.strip(),
        )
        if match is not None:
            exception_type = match.group("type")
            break
    prefix = f"child_exit_code={returncode} child_exception_type={exception_type}\n"
    tail_budget = max(0, MAX_ERROR_CHARS - len(prefix))
    message = prefix + diagnostic[-tail_budget:]
    return Wave6ChildProcessError(message, code="wave6.child_failure")


def _run_observation_child(
    plan: Mapping[str, Any],
    *,
    candidate_arm_id: str,
    repetition_index: int,
    position_index: int,
    arm_id: str,
    injected_failure_stage: str | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "observation",
        "--plan",
        plan["artifact_targets"]["plan"],
        "--candidate-arm-id",
        candidate_arm_id,
        "--repetition-index",
        str(repetition_index),
        "--position-index",
        str(position_index),
        "--arm-id",
        arm_id,
    ]
    if injected_failure_stage is not None:
        command.extend(["--inject-failure-stage", injected_failure_stage])
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["TRANSFORMERS_VERBOSITY"] = "error"
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=CHILD_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Wave6ProbeError(
            "observation child could not complete", code="wave6.child_failure"
        ) from exc
    if completed.returncode != 0:
        raise _child_process_error(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return validate_observation(_strict_json_text(completed.stdout), expected_plan=plan)


def run_controller(
    plan_path: str | Path,
    *,
    injected_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(plan_path).expanduser().resolve()
    plan = validate_plan(load_strict_json(target), require_receipt_absent=True)
    if target != Path(plan["artifact_targets"]["plan"]):
        raise Wave6ProbeError(
            "invoked plan path differs from bound target",
            code="wave6.artifact_target",
        )
    observations: list[dict[str, Any]] = []
    failed_coordinate: dict[str, int | str] | None = None
    try:
        for candidate_spec in CANDIDATE_GRID:
            candidate_arm_id = str(candidate_spec["arm_id"])
            for repetition_index, order in enumerate(POLICY_PAIR_ORDERS):
                for position_index, label in enumerate(order):
                    arm_id = candidate_arm_id if label == "candidate" else str(label)
                    failed_coordinate = {
                        "candidate_arm_id": candidate_arm_id,
                        "repetition_index": repetition_index,
                        "position_index": position_index,
                        "arm_id": arm_id,
                    }
                    stage = None
                    if injected_failure is not None and all(
                        injected_failure.get(field) == failed_coordinate[field]
                        for field in (
                            "candidate_arm_id",
                            "repetition_index",
                            "position_index",
                        )
                    ):
                        stage = str(injected_failure.get("stage"))
                    observations.append(
                        _run_observation_child(
                            plan,
                            candidate_arm_id=candidate_arm_id,
                            repetition_index=repetition_index,
                            position_index=position_index,
                            arm_id=arm_id,
                            injected_failure_stage=stage,
                        )
                    )
        paired = _paired_observations(observations)
        receipt = finalize_artifact(
            {
                "schema": CONTROLLER_RECEIPT_SCHEMA,
                "status": "completed_cpu_only_training_required",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "plan_path": plan["artifact_targets"]["plan"],
                "plan_sha256": plan["plan_sha256"],
                "plan_file_sha256": sha256_file(plan["artifact_targets"]["plan"]),
                "pair_orders": [list(order) for order in POLICY_PAIR_ORDERS],
                "observations": observations,
                "paired_observations": paired,
                "aggregate": _controller_aggregate(observations),
                "research_disposition": _controller_disposition(observations),
            },
            hash_field="controller_receipt_sha256",
        )
        validate_controller_receipt(receipt, expected_plan=plan)
    except BaseException as exc:
        failure = finalize_artifact(
            {
                "schema": FAILURE_RECEIPT_SCHEMA,
                "status": "failed_terminal",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "plan_path": plan["artifact_targets"]["plan"],
                "plan_sha256": plan["plan_sha256"],
                "failed_coordinate": failed_coordinate,
                "completed_observation_sha256": [
                    item["observation_sha256"] for item in observations
                ],
                "error": _bounded_error(exc),
                "retry_count": 0,
            },
            hash_field="failure_receipt_sha256",
        )
        validate_failure_receipt(failure, expected_plan=plan)
        publish_json_absent(plan["artifact_targets"]["failure_receipt"], failure)
        if isinstance(exc, Wave6ProbeError) and exc.code == "wave6.child_failure":
            raise
        raise Wave6ProbeError(str(exc), code="wave6.child_failure") from exc
    publish_json_absent(plan["artifact_targets"]["receipt"], receipt)
    return receipt


def _canonical_existing_directory(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    lexical = Path(os.path.abspath(os.fspath(requested)))
    resolved = requested.resolve(strict=True)
    if lexical != resolved or not resolved.is_dir() or requested.is_symlink():
        raise Wave6ProbeError(
            "W0 root is not a real directory", code="wave6.w0_binding"
        )
    current = resolved
    while current != current.parent:
        if current.is_symlink():
            raise Wave6ProbeError(
                "W0 root traverses a symlink", code="wave6.w0_binding"
            )
        current = current.parent
    return resolved


def _load_historical_w0(
    w0_cache_root: str | Path, *, load_train_examples: bool = True
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    root = _canonical_existing_directory(w0_cache_root)
    split_bindings: dict[str, Any] = {}
    train_examples: tuple[Any, ...] | None = None
    for split, expected in HISTORICAL_W0_SPLITS.items():
        cache_dir = root / expected["fingerprint"]
        manifest_path = cache_dir / "manifest.json"
        manifest = load_strict_json(manifest_path)
        if (
            sha256_file(manifest_path) != expected["manifest_sha256"]
            or manifest.get("version") != "coordexp-swift-pack-cache-v2"
            or manifest.get("status") != "complete"
            or manifest.get("fingerprint") != expected["fingerprint"]
            or manifest.get("micro_step_count") != expected["micro_step_count"]
            or not isinstance(manifest.get("chunks"), list)
            or len(manifest["chunks"]) != 1
            or manifest["chunks"][0].get("sha256") != expected["chunk_sha256"]
        ):
            raise Wave6ProbeError(
                "historical W0 manifest drifted", code="wave6.w0_binding"
            )
        chunk_path = cache_dir / str(manifest["chunks"][0]["path"])
        if sha256_file(chunk_path) != expected["chunk_sha256"]:
            raise Wave6ProbeError(
                "historical W0 chunk drifted", code="wave6.w0_binding"
            )
        split_bindings[split] = {
            "fingerprint": expected["fingerprint"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": expected["manifest_sha256"],
            "chunk_path": str(chunk_path),
            "chunk_sha256": expected["chunk_sha256"],
            "micro_step_count": expected["micro_step_count"],
            "example_count": expected["example_count"],
        }
        if split == "train" and load_train_examples:
            _start, steps = pack_cache._load_validated_chunk(
                cache_dir, manifest["chunks"][0]
            )
            train_examples = tuple(
                example for step in steps for example in step.encoded_examples
            )
            if len(train_examples) != expected["example_count"]:
                raise Wave6ProbeError(
                    "historical W0 example count drifted", code="wave6.w0_binding"
                )
    if load_train_examples and train_examples is None:
        raise Wave6ProbeError(
            "historical W0 train payload was not loaded", code="wave6.w0_binding"
        )
    payload = {
        "schema": W0_BINDING_SCHEMA,
        "status": "historical_v2_read_only_encoded_evidence",
        "historical_config_fingerprint": HISTORICAL_W0_CONFIG_FINGERPRINT,
        "cache_root": str(root),
        "splits": split_bindings,
        "access_policy": "authenticate_read_only_no_cache_writes",
    }
    binding = {**payload, "binding_sha256": sha256_json(payload)}
    _validate_w0_binding(binding, scope="frozen_w0_cpu")
    return () if train_examples is None else train_examples, binding


def prepare_plan(
    *, w0_cache_root: str | Path, plan_path: str | Path, receipt_path: str | Path
) -> dict[str, Any]:
    examples, binding = _load_historical_w0(w0_cache_root)
    plan = _build_plan(
        examples,
        scope="frozen_w0_cpu",
        global_max_length=12_000,
        w0_binding=binding,
        plan_path=plan_path,
        receipt_path=receipt_path,
    )
    publish_json_absent(plan["artifact_targets"]["plan"], plan)
    return plan


def run_plan(plan_path: str | Path) -> dict[str, Any]:
    """Run the fail-closed fresh-process controller bound by ``plan_path``."""

    return run_controller(plan_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--w0-cache-root", required=True)
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--receipt", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    research_meaning = subparsers.add_parser("research-meaning")
    research_meaning.add_argument("--plan", required=True)
    research_meaning.add_argument("--receipt", required=True)
    research_meaning.add_argument("--output", required=True)
    observation = subparsers.add_parser("observation")
    observation.add_argument("--plan", required=True)
    observation.add_argument(
        "--candidate-arm-id",
        required=True,
        choices=tuple(str(spec["arm_id"]) for spec in CANDIDATE_GRID),
    )
    observation.add_argument("--repetition-index", required=True, type=int)
    observation.add_argument("--position-index", required=True, type=int)
    observation.add_argument(
        "--arm-id",
        required=True,
        choices=tuple(str(spec["arm_id"]) for spec in ARM_GRID),
    )
    observation.add_argument("--inject-failure-stage", choices=("child", "planner"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "prepare":
        plan = prepare_plan(
            w0_cache_root=args.w0_cache_root,
            plan_path=args.plan,
            receipt_path=args.receipt,
        )
        print(
            json.dumps(
                {
                    "plan": plan["artifact_targets"]["plan"],
                    "plan_sha256": plan["plan_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "observation":
        plan = load_strict_json(args.plan)
        observation = execute_materialization_observation(
            plan,
            candidate_arm_id=args.candidate_arm_id,
            repetition_index=args.repetition_index,
            position_index=args.position_index,
            arm_id=args.arm_id,
            injected_failure_stage=args.inject_failure_stage,
        )
        print(json.dumps(observation, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "research-meaning":
        packet = publish_research_meaning_packet(
            plan_path=args.plan,
            receipt_path=args.receipt,
            packet_path=args.output,
        )
        print(
            json.dumps(
                {
                    "packet": str(_canonical_research_source(args.output)),
                    "research_meaning_packet_sha256": packet[
                        "research_meaning_packet_sha256"
                    ],
                    "status": packet["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = run_plan(args.plan)
    plan = load_strict_json(args.plan)
    print(
        json.dumps(
            {
                "receipt": plan["artifact_targets"]["receipt"],
                "receipt_sha256": receipt["controller_receipt_sha256"],
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
