"""Packing-cache preparation, admission, and hydration orchestration.

`src/training/pack_cache.py` stays the low-level owner of determinant
construction, fingerprinting, immutable publication, restricted
deserialization, manifest validation, and rank/eval payload loading.  This
module owns the higher-level operations that used to live in the training
assembly facade:

- `prepare_training_pack_caches(config_path)` and its preparation receipt;
- absent-target build orchestration, multi-worker render/tokenize/pack
  materialization, and split aggregation;
- model-free train/eval cache fingerprint resolution and admission;
- rank-local train/eval hydration and image-processor attachment;
- conversion of low-level cache failures into the actionable preparation
  command and the bounded rank diagnostic.

Training remains fail-closed on a missing or invalid cache.  Preparation
remains a separate single-process command and is never hidden inside
distributed or model startup: this module must not import
`src/training/session.py` or `src/training/pipeline.py`, and must never request
a model-bearing component load.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shlex
import time
from typing import Any

from src.artifacts import RunWriter
from src.artifacts.provenance import (
    collect_execution_provenance,
    require_pinned_runtime_baseline,
)
from src.artifacts.resources import (
    collect_resource_snapshot,
    merge_resource_high_water,
)
from src.augmentation.factory import build_augmentation_processor
from src.augmentation.processor import AugmentationMaterializationResult
from src.common.errors import RuntimeContractError
from src.config.loader import load_train_config
from src.config.resolve import resolve_qwen_runtime_controls
from src.data import load_raw_examples
from src.eval.forward import (
    EVAL_REDUCTION_DISJOINT_SHARD,
    EVAL_REDUCTION_REPLICATED,
    resolve_active_eval_reduction_mode,
    resolve_eval_reduction_control,
)
from src.losses import build_token_vocabulary_groups
from src.packing import (
    ONLINE_WINDOW_BINPACK,
    PackPlan,
    PackedSequence,
    build_pack_plan_policy_identity,
    build_packed_supervision,
    create_pack_plan,
    replay_pack_plan,
    stream_online_pack_plan_fragments,
    verify_pack_plan_stream_fragments,
)
from src.qwen import (
    QwenImageEncoding,
    attach_qwen_image_processor,
    build_qwen_position_inputs,
    encode_rendered_example,
    load_qwen_components,
)
from src.runtime import TrainingSeedReceipt, seed_training_runtime
from src.supervision import (
    build_token_sequence_from_packed_supervision,
    index_token_atoms_by_pack,
)
from src.templates import render_example
from src.training import control_plane
from src.training.micro_steps import SupervisedMicroStep
from src.training.pack_cache import (
    DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    PACKING_CACHE_VERSION,
    PackingCacheInvalidError,
    build_packing_cache_determinants,
    build_packing_cache_fingerprint,
    build_packing_cache_materialization,
    cache_dir_for_fingerprint,
    load_cache_manifest,
    load_rank_eval_micro_steps_from_cache,
    load_rank_micro_steps_from_cache,
    manifest_path,
    packing_cache_fingerprint_from_determinants,
    write_micro_step_cache,
)
from src.training.schedule import resolve_planned_step_schedule


TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval.forward"
_PACK_CACHE_WORKER_CONTEXT: dict[str, Any] | None = None
_PACK_CACHE_ROOT_ENV = "COORDEXP_SWIFT_PACK_CACHE_ROOT"
_FORWARD_INPUT_PROVIDER_MODE_ENV = "COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE"
_EVAL_REDUCTION_MODE_ENV = "COORDEXP_SWIFT_EVAL_REDUCTION_MODE"
_PROFILE_SYNC_TIMINGS_ENV = "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS"
_RECEIPT_ENVIRONMENT_SELECTORS = frozenset(
    {
        _PACK_CACHE_ROOT_ENV,
        _FORWARD_INPUT_PROVIDER_MODE_ENV,
        _EVAL_REDUCTION_MODE_ENV,
        _PROFILE_SYNC_TIMINGS_ENV,
    }
)

#: Verification level `--require-all-hit` uses for both existing targets.
REQUIRE_ALL_HIT_VERIFICATION_LEVEL = "payloads"


@dataclass(frozen=True)
class CachePreflight:
    """One fail-before-build verification of both published split targets.

    The record carries only what the `--require-all-hit` route resolves: the
    two semantic fingerprints, the two immutable targets, and the terminal
    receipt.  It abstracts no storage backend and no cache version.
    """

    train_fingerprint: str
    eval_fingerprint: str
    train_target: Path
    eval_target: Path
    receipt: Mapping[str, Any]

    def to_receipt_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _environment_selector_source(name: str) -> str:
    """Return a bounded selector source without persisting its raw value."""

    if name not in _RECEIPT_ENVIRONMENT_SELECTORS:
        raise RuntimeContractError(
            "environment selector is not approved for the run receipt",
            code="runtime.environment_selector_unsupported",
            context={"name": name},
        )
    return name if name in os.environ else "default"


def _resolve_pack_cache_root(repo_root: Path) -> tuple[Path, dict[str, str]]:
    """Resolve the production cache root and its compact allowlisted source."""

    raw = os.environ.get(_PACK_CACHE_ROOT_ENV)
    root = (
        repo_root / ".cache" / "coordexp_swift" / "packing"
        if raw is None
        else Path(raw)
    )
    return root, {
        "resolved_root": str(root.resolve()),
        "source": _environment_selector_source(_PACK_CACHE_ROOT_ENV),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _launcher_device_mapping(*, rank: int, world_size: int) -> dict[str, Any]:
    raw_local_rank = os.environ.get("LOCAL_RANK")
    if raw_local_rank is None:
        if world_size != 1:
            raise RuntimeContractError(
                "distributed strict runtime identity requires LOCAL_RANK",
                code="runtime.determinism_launcher_mapping_invalid",
                context={"rank": rank, "world_size": world_size},
            )
        local_rank = 0
    elif re.fullmatch(r"0|[1-9][0-9]*", raw_local_rank) is None:
        raise RuntimeContractError(
            "LOCAL_RANK must be a strict nonnegative decimal integer",
            code="runtime.determinism_launcher_mapping_invalid",
            context={"rank": rank, "world_size": world_size},
        )
    else:
        local_rank = int(raw_local_rank)
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_devices: list[str] | None = None
    if raw_visible is not None:
        entries = raw_visible.split(",")
        if (
            not entries
            or len(entries) > 64
            or any(
                not entry or len(entry) > 128 or not entry.isascii()
                for entry in entries
            )
        ):
            raise RuntimeContractError(
                "CUDA_VISIBLE_DEVICES violates the bounded launcher mapping contract",
                code="runtime.determinism_launcher_mapping_invalid",
                context={"rank": rank, "world_size": world_size},
            )
        if local_rank >= len(entries):
            raise RuntimeContractError(
                "LOCAL_RANK is outside CUDA_VISIBLE_DEVICES",
                code="runtime.determinism_launcher_mapping_invalid",
                context={
                    "local_rank": local_rank,
                    "visible_device_count": len(entries),
                },
            )
        visible_devices = entries
    return {
        "cuda_visible_devices": visible_devices,
        "local_rank": local_rank,
        "logical_cuda_device": local_rank,
        "rank": rank,
        "world_size": world_size,
    }


def _establish_converged_runtime_determinism(
    runtime_config: Any,
    *,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
    phase: str,
) -> dict[str, Any]:
    """Establish one pre-CUDA policy and attest it on the CPU control plane."""

    local_detail: dict[str, Any] = {
        "runtime_determinism": None,
        "application": None,
        "launcher": None,
    }
    local_receipt: TrainingSeedReceipt | None = None
    converged_details: list[dict[str, Any]] = []

    def establish_local() -> TrainingSeedReceipt:
        nonlocal local_receipt
        launcher = _launcher_device_mapping(rank=rank, world_size=world_size)
        receipt = seed_training_runtime(
            int(runtime_config.seed),
            determinism_mode=str(runtime_config.determinism.mode),
            phase=phase,
        )
        local_receipt = receipt
        local_detail.update(
            runtime_determinism=receipt.to_policy_identity_dict(),
            application=receipt.to_artifact_dict(),
            launcher=launcher,
        )
        return receipt

    def validate_receipts(receipt: Mapping[str, Any]) -> None:
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            return
        details = [
            dict(rank_details[str(index)])
            for index in range(world_size)
            if str(index) in rank_details
        ]
        if len(details) != world_size:
            raise RuntimeContractError(
                "runtime determinism receipt lacks the complete launcher rank set",
                code="runtime.determinism_rank_mismatch",
            )
        policies = [detail.get("runtime_determinism") for detail in details]
        canonical = policies[0]
        if not isinstance(canonical, Mapping) or any(
            policy != canonical for policy in policies[1:]
        ):
            raise RuntimeContractError(
                "runtime determinism policy differs across launcher ranks",
                code="runtime.determinism_rank_mismatch",
            )
        launchers = [detail.get("launcher") for detail in details]
        for expected_rank, item in enumerate(launchers):
            if not isinstance(item, Mapping):
                raise RuntimeContractError(
                    "runtime determinism launcher mapping is incomplete",
                    code="runtime.determinism_rank_mismatch",
                )
            expected_fields = {
                "cuda_visible_devices",
                "local_rank",
                "logical_cuda_device",
                "rank",
                "world_size",
            }
            local_rank = item.get("local_rank")
            visible_devices = item.get("cuda_visible_devices")
            if (
                set(item) != expected_fields
                or item.get("rank") != expected_rank
                or item.get("world_size") != world_size
                or isinstance(local_rank, bool)
                or not isinstance(local_rank, int)
                or local_rank < 0
                or item.get("logical_cuda_device") != local_rank
                or (
                    visible_devices is not None
                    and (
                        not isinstance(visible_devices, list)
                        or local_rank >= len(visible_devices)
                    )
                )
            ):
                raise RuntimeContractError(
                    "runtime determinism launcher mapping differs from the rank set",
                    code="runtime.determinism_rank_mismatch",
                )
        applications = [detail.get("application") for detail in details]
        if any(not isinstance(item, Mapping) for item in applications):
            raise RuntimeContractError(
                "runtime determinism application receipt is incomplete",
                code="runtime.determinism_rank_mismatch",
            )
        if canonical.get("mode") == "strict_cuda_replay_v1" and any(
            item.get("cuda_initialized") is not False
            for item in applications
            if isinstance(item, Mapping)
        ):
            raise RuntimeContractError(
                "strict runtime determinism was not established before CUDA",
                code="runtime.determinism_rank_mismatch",
            )
        converged_details[:] = details

    control_plane._run_rank_converged_phase(
        "config_provenance_resolution",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=establish_local,
        local_details=lambda: local_detail,
        receipt_sink=validate_receipts,
    )
    if local_receipt is None or len(converged_details) != world_size:
        raise RuntimeContractError(
            "runtime determinism convergence returned no admitted receipt",
            code="runtime.determinism_rank_mismatch",
        )
    return {
        **local_receipt.to_policy_identity_dict(),
        "application_receipt": local_receipt.to_artifact_dict(),
        "launcher_attestations": [
            dict(detail["launcher"]) for detail in converged_details
        ],
        "pre_apply_cuda_initialized": bool(local_receipt.cuda_initialized),
        "pre_apply_cuda_initialized_by_rank": {
            str(index): bool(detail["application"]["cuda_initialized"])
            for index, detail in enumerate(converged_details)
        },
    }


def _runtime_determinism_run_policy(
    converged: Mapping[str, Any],
    *,
    pinned_runtime_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_sha256 = pinned_runtime_baseline.get("baseline_sha256")
    if (
        not isinstance(baseline_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", baseline_sha256) is None
    ):
        raise RuntimeContractError(
            "runtime determinism policy requires the admitted baseline digest",
            code="runtime.determinism_baseline_invalid",
        )
    return {
        **dict(converged),
        "pinned_runtime_baseline_sha256": baseline_sha256,
    }


def _resolve_eval_reduction_receipt(
    *, pack_count: int | None, world_size: int
) -> dict[str, Any]:
    """Resolve the complete rank-local eval selector without model state."""

    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise RuntimeContractError(
            "eval reduction world size must be a positive integer",
            code="training.eval_reduction_resolution_invalid",
            context={"world_size": world_size},
        )
    if pack_count is not None and (
        isinstance(pack_count, bool)
        or not isinstance(pack_count, int)
        or pack_count <= 0
    ):
        raise RuntimeContractError(
            "eval reduction pack count must be positive when an eval cache exists",
            code="training.eval_reduction_resolution_invalid",
            context={"pack_count": pack_count},
        )
    control = resolve_eval_reduction_control()
    effective_mode = (
        EVAL_REDUCTION_REPLICATED
        if pack_count is None
        else resolve_active_eval_reduction_mode(
            pack_count=pack_count,
            world_size=world_size,
        )
    )
    return {
        "control": control,
        "effective_mode": effective_mode,
        "source": _environment_selector_source(_EVAL_REDUCTION_MODE_ENV),
        "pack_count": pack_count,
        "world_size": world_size,
    }


def _resolve_converged_eval_reduction_receipt(
    *,
    pack_count: int | None,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
) -> dict[str, Any]:
    """Freeze one exact eval selector receipt on the model-free control plane."""

    local_detail: dict[str, Any] = {"resolution": None}

    def resolve_local() -> dict[str, Any]:
        resolved = _resolve_eval_reduction_receipt(
            pack_count=pack_count,
            world_size=world_size,
        )
        local_detail["resolution"] = dict(resolved)
        return resolved

    def validate_receipts(receipt: Mapping[str, Any]) -> None:
        rank_details = receipt.get("rank_details")
        if not isinstance(rank_details, Mapping):
            return
        resolutions = {
            str(rank_key): dict(detail["resolution"])
            for rank_key, detail in rank_details.items()
            if isinstance(detail, Mapping)
            and isinstance(detail.get("resolution"), Mapping)
        }
        if len(resolutions) != world_size:
            return
        canonical = resolutions.get("0")
        if canonical is None or any(
            resolution != canonical for resolution in resolutions.values()
        ):
            raise RuntimeContractError(
                "eval reduction resolution differs across launcher ranks",
                code="training.eval_reduction_resolution_mismatch",
                context={"rank_resolutions": resolutions},
            )

    observed = control_plane._run_rank_converged_phase(
        "eval_reduction_resolution",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=resolve_local,
        local_details=lambda: local_detail,
        receipt_sink=validate_receipts,
    )
    expected_fields = {
        "control",
        "effective_mode",
        "source",
        "pack_count",
        "world_size",
    }
    if not isinstance(observed, Mapping) or set(observed) != expected_fields:
        raise RuntimeContractError(
            "eval reduction resolution returned an invalid receipt",
            code="training.eval_reduction_resolution_invalid",
        )
    return dict(observed)


def _packing_policy_receipt(config: Any) -> dict[str, Any]:
    packing = config.packing
    return {
        "schema_version": 2,
        "global_max_length": int(packing.global_max_length),
        "planner": build_pack_plan_policy_identity(
            policy=str(packing.policy),
            window_size=packing.window_size,
            lookahead=packing.lookahead,
            seed=int(packing.seed),
            worker_count=int(packing.worker_count),
            cursor_byte_budget=int(packing.cursor_byte_budget),
            fragment_item_budget=int(packing.fragment_item_budget),
            fragment_byte_budget=int(packing.fragment_byte_budget),
        ),
        "fragment_pack_budget": packing.max_packs_per_fragment,
        "train_order": str(config.data.train_order),
        "intra_image_object_order": str(config.template.object_ordering),
    }


def prepare_training_pack_caches(
    config_path: str | Path,
    *,
    require_all_hit: bool = False,
) -> dict[str, Any]:
    """Materialize all packing caches before distributed model startup.

    ``require_all_hit=True`` selects the fail-before-build verification mode.
    It resolves both split fingerprints and targets, admits and fully validates
    both existing publications, and returns its verification receipt.  It never
    reaches a render, tokenize, pack, absent-target build, temporary
    publication, or immutable publication path: the branch below is taken
    before the first build-capable call, so a target that disappears or becomes
    invalid between discovery and use fails instead of triggering a rebuild.
    """

    entry_started_at = _utc_now()
    entry_started_monotonic = time.monotonic()
    initial_resources = collect_resource_snapshot()
    repo_root = Path.cwd().resolve()
    resolved_config = load_train_config(config_path)
    config = resolved_config.config
    runtime_determinism = _establish_converged_runtime_determinism(
        config.runtime,
        rank=0,
        world_size=1,
        rank_report_gatherer=None,
        phase="pack_cache_preparation",
    )
    provenance = collect_execution_provenance(repository_root=repo_root)
    pinned_runtime_baseline = require_pinned_runtime_baseline(
        provenance=provenance,
        attention_backend=str(config.model.attn_implementation),
    )
    components = load_qwen_components(config, load_model=False)
    vocab_groups = build_token_vocabulary_groups(
        components.token_identity,
        tokenizer=components.tokenizer,
    )
    config_phase_seconds = time.monotonic() - entry_started_monotonic
    pack_cache_root, pack_cache_root_receipt = _resolve_pack_cache_root(repo_root)
    if require_all_hit:
        # Fail-before-build gate.  Everything above is read-only resolution;
        # nothing below this branch may run in verification mode.
        return _verify_published_pack_cache_targets(
            config,
            components,
            vocab_groups,
            config_path=config_path,
            resolved_config=resolved_config,
            cache_root=pack_cache_root,
            cache_root_receipt=pack_cache_root_receipt,
            provenance=provenance,
            pinned_runtime_baseline=pinned_runtime_baseline,
            runtime_determinism=runtime_determinism,
            initial_resources=initial_resources,
            entry_started_at=entry_started_at,
            entry_started_monotonic=entry_started_monotonic,
            config_phase_seconds=config_phase_seconds,
        ).to_receipt_dict()
    train_cache = _resolve_or_build_train_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        accelerator=None,
        rank=0,
        verification_level="payloads",
        cache_root=pack_cache_root,
    )
    eval_cache = _resolve_eval_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        accelerator=None,
        rank=0,
        cache_root=pack_cache_root,
    )
    final_resources = collect_resource_snapshot()
    resource_high_water = merge_resource_high_water(initial_resources, final_resources)
    split_caches = [train_cache, *([] if eval_cache is None else [eval_cache])]
    aggregate_phases = {
        phase: _aggregate_cache_phase(split_caches, phase)
        for phase in ("cache_preparation", "cache_publication", "cache_admission")
    }
    return {
        "entry_config_path": str(resolved_config.entry_config_path),
        "resolved_config_fingerprint": resolved_config.fingerprint,
        "model_loaded": False,
        "provenance": provenance,
        "policy_identities": {
            "upstream_runtime_baseline": pinned_runtime_baseline,
            "runtime_determinism": _runtime_determinism_run_policy(
                runtime_determinism,
                pinned_runtime_baseline=pinned_runtime_baseline,
            ),
            "packing": _packing_policy_receipt(config),
            "cache": {
                "schema_version": 1,
                "root": pack_cache_root_receipt,
                "train_fingerprint": str(train_cache["fingerprint"]),
                "eval_fingerprint": (
                    None if eval_cache is None else str(eval_cache["fingerprint"])
                ),
            },
        },
        "measurement": {
            "schema_version": 1,
            "context": {
                "comparison_arm": "compatibility_reference",
                "wall_clock_scope": "prepare_training_pack_caches_entry_to_return",
                "workload_identity": resolved_config.fingerprint,
                "world_size": 1,
            },
            "started_at": entry_started_at,
            "completed_at": _utc_now(),
            "duration_seconds": time.monotonic() - entry_started_monotonic,
            "phases": {
                "config_provenance_resolution": {
                    "status": "completed",
                    "duration_seconds": config_phase_seconds,
                },
                **aggregate_phases,
            },
            "resource_high_water": resource_high_water,
        },
        "train": _pack_cache_preparation_receipt(train_cache),
        "eval": (
            None if eval_cache is None else _pack_cache_preparation_receipt(eval_cache)
        ),
    }


def _verify_published_pack_cache_targets(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    config_path: str | Path,
    resolved_config: Any,
    cache_root: Path,
    cache_root_receipt: Mapping[str, Any],
    provenance: Mapping[str, Any],
    pinned_runtime_baseline: Mapping[str, Any],
    runtime_determinism: Mapping[str, Any],
    initial_resources: Mapping[str, Any],
    entry_started_at: str,
    entry_started_monotonic: float,
    config_phase_seconds: float,
) -> CachePreflight:
    """Validate both published split targets without any build authority.

    This route calls no builder: it resolves the two semantic fingerprints,
    admits both immutable publications at the strongest verification level, and
    fails closed on a missing, incomplete, or invalid target.  A successful
    discovery performed before the process started is not sufficient evidence,
    so admission happens here, inside the invoked workflow.
    """

    if config.data.eval is None:
        raise RuntimeContractError(
            "cache verification requires both declared split targets; this "
            "config declares no evaluation split, so two valid hits cannot be "
            "proven",
            code="training.pack_cache_verification_split_undeclared",
            context={
                "cache_root": str(cache_root),
                "cache_version": PACKING_CACHE_VERSION,
                "declared_splits": [TRAIN_SPLIT],
                "required_splits": [TRAIN_SPLIT, EVAL_SPLIT],
            },
        )
    train_fingerprint = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
        vocab_groups=vocab_groups,
    )
    eval_fingerprint = build_packing_cache_fingerprint(
        config,
        components,
        dataset=config.data.eval,
        split=EVAL_SPLIT,
        vocab_groups=vocab_groups,
    )
    train_cache = _admit_model_free_pack_cache(
        config,
        components,
        vocab_groups=vocab_groups,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
        cache_root=cache_root,
        config_path=config_path,
        verification_level=REQUIRE_ALL_HIT_VERIFICATION_LEVEL,
        resolved_fingerprint=str(train_fingerprint),
    )
    eval_cache = _admit_model_free_pack_cache(
        config,
        components,
        vocab_groups=vocab_groups,
        dataset=config.data.eval,
        split=EVAL_SPLIT,
        cache_root=cache_root,
        config_path=config_path,
        verification_level=REQUIRE_ALL_HIT_VERIFICATION_LEVEL,
        resolved_fingerprint=str(eval_fingerprint),
    )
    final_resources = collect_resource_snapshot()
    resource_high_water = merge_resource_high_water(initial_resources, final_resources)
    receipt = {
        "entry_config_path": str(resolved_config.entry_config_path),
        "resolved_config_fingerprint": resolved_config.fingerprint,
        "model_loaded": False,
        "cache_materialization_authorized": False,
        "verification_level": REQUIRE_ALL_HIT_VERIFICATION_LEVEL,
        "verified_splits": [TRAIN_SPLIT, EVAL_SPLIT],
        "provenance": provenance,
        "policy_identities": {
            "upstream_runtime_baseline": pinned_runtime_baseline,
            "runtime_determinism": _runtime_determinism_run_policy(
                runtime_determinism,
                pinned_runtime_baseline=pinned_runtime_baseline,
            ),
            "packing": _packing_policy_receipt(config),
            "cache": {
                "schema_version": 1,
                "root": cache_root_receipt,
                "train_fingerprint": str(train_fingerprint),
                "eval_fingerprint": str(eval_fingerprint),
            },
        },
        "measurement": {
            "schema_version": 1,
            "context": {
                "comparison_arm": "require_all_hit_verification",
                "wall_clock_scope": (
                    "prepare_training_pack_caches_entry_to_return"
                ),
                "workload_identity": resolved_config.fingerprint,
                "world_size": 1,
            },
            "started_at": entry_started_at,
            "completed_at": _utc_now(),
            "duration_seconds": time.monotonic() - entry_started_monotonic,
            "phases": {
                "config_provenance_resolution": {
                    "status": "completed",
                    "duration_seconds": config_phase_seconds,
                },
                "cache_preparation": {
                    "status": "not_run",
                    "reason": "verification_has_no_materialization_authority",
                    "duration_seconds": 0.0,
                },
                "cache_publication": {
                    "status": "not_run",
                    "reason": "verification_has_no_materialization_authority",
                    "duration_seconds": 0.0,
                },
                "cache_admission": _aggregate_cache_phase(
                    [train_cache, eval_cache], "cache_admission"
                ),
            },
            "resource_high_water": resource_high_water,
        },
        "train": _pack_cache_preparation_receipt(train_cache),
        "eval": _pack_cache_preparation_receipt(eval_cache),
    }
    return CachePreflight(
        train_fingerprint=str(train_fingerprint),
        eval_fingerprint=str(eval_fingerprint),
        train_target=Path(train_cache["cache_dir"]),
        eval_target=Path(eval_cache["cache_dir"]),
        receipt=receipt,
    )


def _aggregate_cache_phase(
    split_caches: Sequence[Mapping[str, Any]], phase: str
) -> dict[str, Any]:
    receipts = [cache["phase_receipt"][phase] for cache in split_caches]
    statuses = {str(receipt["status"]) for receipt in receipts}
    duration_seconds = sum(float(receipt["duration_seconds"]) for receipt in receipts)
    if statuses == {"not_run_cache_hit"}:
        return {
            "status": "not_run",
            "reason": "all_cache_hits",
            "duration_seconds": duration_seconds,
        }
    result: dict[str, Any] = {
        "status": "completed",
        "duration_seconds": duration_seconds,
    }
    if "not_run_cache_hit" in statuses:
        result["reason"] = "mixed_cache_hits_and_builds"
    return result


def _pack_cache_preparation_receipt(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(cache["status"]),
        "build_status": str(cache["build_status"]),
        "cache_dir": str(cache["cache_dir"]),
        "format_version": str(cache["format_version"]),
        "fingerprint": str(cache["fingerprint"]),
        "manifest_path": str(cache["manifest_path"]),
        "manifest_sha256": str(cache["manifest_sha256"]),
        "micro_step_count": int(cache["micro_step_count"]),
        "phase_receipt": dict(cache["phase_receipt"]),
    }


def build_base_micro_steps(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    materialization_workers: int | None = None,
) -> tuple[SupervisedMicroStep, ...]:
    return _build_micro_steps_for_dataset(
        config,
        components,
        vocab_groups,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
        materialization_workers=materialization_workers,
    )


def _resolve_or_build_train_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    accelerator: Any,
    rank: int = 0,
    verification_level: str,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    return _resolve_or_build_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
        accelerator=accelerator,
        rank=rank,
        verification_level=verification_level,
        cache_root=cache_root,
        build_micro_steps=lambda workers: build_base_micro_steps(
            config,
            components,
            vocab_groups,
            materialization_workers=workers,
        ),
    )


def _resolve_eval_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    accelerator: Any,
    rank: int = 0,
    cache_root: Path | None = None,
) -> dict[str, Any] | None:
    if config.data.eval is None:
        return None
    return _resolve_or_build_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        dataset=config.data.eval,
        split="eval.forward",
        accelerator=accelerator,
        rank=rank,
        verification_level="payloads",
        cache_root=cache_root,
        build_micro_steps=lambda workers: _build_micro_steps_for_dataset(
            config,
            components,
            vocab_groups,
            dataset=config.data.eval,
            split="eval.forward",
            materialization_workers=workers,
        ),
    )


def _resolve_or_build_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    dataset: Any,
    split: str,
    accelerator: Any,
    rank: int = 0,
    build_micro_steps: Callable[[int], Sequence[SupervisedMicroStep]],
    materialization_workers: int | None = None,
    verification_level: str,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    if cache_root is None:
        cache_root, _ = _resolve_pack_cache_root(repo_root)
    world_size = 1 if accelerator is None else int(accelerator.num_processes)
    determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=dataset,
        split=split,
        vocab_groups=vocab_groups,
    )
    fingerprint = packing_cache_fingerprint_from_determinants(determinants)

    def revalidate_determinants() -> Mapping[str, Any]:
        return build_packing_cache_determinants(
            config,
            components,
            dataset=dataset,
            split=split,
            vocab_groups=vocab_groups,
        )

    resolved_materialization_workers = _resolve_pack_cache_materialization_workers(
        materialization_workers
    )
    materialization = build_packing_cache_materialization(
        workers=resolved_materialization_workers,
        strategy=PACKING_CACHE_MATERIALIZATION_STRATEGY,
    )
    cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
    cache_complete_before = False
    preparation_seconds = 0.0
    publication_seconds = 0.0
    admission_seconds = 0.0
    try:
        admission_started = time.monotonic()
        try:
            manifest = load_cache_manifest(
                cache_dir,
                cache_root=cache_root,
                expected_fingerprint=fingerprint,
                level=verification_level,
            )
            admission_seconds = time.monotonic() - admission_started
            cache_complete_before = True
        except PackingCacheInvalidError as exc:
            admission_seconds = time.monotonic() - admission_started
            if cache_dir.exists():
                raise _cache_preflight_error(
                    cache_dir=cache_dir,
                    cache_root=cache_root,
                    config_path=repo_root / "<cache-preparation>",
                    fingerprint=fingerprint,
                    split=split,
                    error=exc,
                ) from exc
            if world_size > 1:
                raise RuntimeContractError(
                    "distributed training requires prepared packing caches; run "
                    "`python -m src.prepare_train_cache --config <path>` before "
                    "`accelerate launch`",
                    code="training.pack_cache_not_prepared",
                    context={
                        "cache_root": str(cache_root),
                        "expected_cache_target": str(cache_dir),
                        "cache_version": PACKING_CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "world_size": world_size,
                    },
                )
            preparation_started = time.monotonic()
            micro_steps = tuple(build_micro_steps(resolved_materialization_workers))
            preparation_seconds = time.monotonic() - preparation_started
            publication_started = time.monotonic()
            manifest = write_micro_step_cache(
                cache_dir,
                micro_steps,
                cache_root=cache_root,
                fingerprint=fingerprint,
                determinants=determinants,
                materialization=materialization,
                determinant_revalidator=revalidate_determinants,
                augmentation=_augmentation_receipt_from_micro_steps(micro_steps),
            )
            publication_seconds = time.monotonic() - publication_started
            admission_started = time.monotonic()
            manifest = load_cache_manifest(
                cache_dir,
                cache_root=cache_root,
                expected_fingerprint=fingerprint,
                level=verification_level,
            )
            admission_seconds = time.monotonic() - admission_started
    except RuntimeContractError:
        raise
    except PackingCacheInvalidError as exc:
        if cache_dir.exists():
            raise RuntimeContractError(
                "required v3 packing cache target is occupied by an invalid "
                f"immutable publication: {cache_dir}",
                code="training.pack_cache_immutable_collision",
                context={
                    "split": split,
                    "cache_root": str(cache_root),
                    "expected_cache_target": str(cache_dir),
                    "cache_version": PACKING_CACHE_VERSION,
                    "fingerprint": fingerprint,
                    "validation_category": _occupied_cache_validation_category(
                        cache_dir,
                        expected_fingerprint=fingerprint,
                        error=exc,
                    ),
                    "automatic_recovery": "unavailable",
                },
            ) from exc
        raise RuntimeContractError(
            "packing cache publication failed before the expected target became visible",
            code="training.pack_cache_resolution_failed",
            context={
                "split": split,
                "cache_root": str(cache_root),
                "expected_cache_target": str(cache_dir),
                "cache_version": PACKING_CACHE_VERSION,
                "fingerprint": fingerprint,
                "error_type": type(exc).__name__,
            },
        ) from exc
    except BaseException as exc:
        raise RuntimeContractError(
            "failed to resolve packing cache",
            code="training.pack_cache_resolution_failed",
            context={
                "rank": rank,
                "cache_root": str(cache_root),
                "expected_cache_target": str(cache_dir),
                "cache_version": PACKING_CACHE_VERSION,
                "fingerprint": fingerprint,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1024],
            },
        ) from exc
    cache_manifest_path = manifest_path(cache_dir)
    return {
        "cache_dir": cache_dir,
        "format_version": str(manifest["version"]),
        "fingerprint": fingerprint,
        "micro_step_count": int(manifest["micro_step_count"]),
        "chunk_count": len(manifest["chunks"]),
        "chunk_size": int(manifest["chunk_size"]),
        "status": manifest["status"],
        "build_status": (
            "waited" if rank != 0 else ("hit" if cache_complete_before else "built")
        ),
        "manifest_path": cache_manifest_path,
        "manifest_sha256": _file_sha256(cache_manifest_path),
        "determinants_sha256": _sha256_json(manifest["determinants"]),
        "chunk_sha256s": [str(chunk["sha256"]) for chunk in manifest["chunks"]],
        "materialization": manifest.get("materialization"),
        "augmentation": manifest.get("augmentation"),
        "phase_receipt": {
            "cache_preparation": {
                "status": "completed"
                if not cache_complete_before
                else "not_run_cache_hit",
                "duration_seconds": preparation_seconds,
            },
            "cache_publication": {
                "status": "completed"
                if not cache_complete_before
                else "not_run_cache_hit",
                "duration_seconds": publication_seconds,
            },
            "cache_admission": {
                "status": "completed",
                "duration_seconds": admission_seconds,
                "verification_level": verification_level,
            },
        },
    }


def _cache_preparation_argv(config_path: str | Path) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        "src.prepare_train_cache",
        "--config",
        str(Path(config_path).expanduser().resolve()),
    )


def _cache_preparation_environment() -> dict[str, str]:
    return dict(control_plane._STRICT_CACHE_PREPARATION_ENVIRONMENT)


def _occupied_cache_validation_category(
    cache_dir: Path,
    *,
    expected_fingerprint: str,
    error: BaseException,
) -> str:
    manifest_file = manifest_path(cache_dir)
    if not manifest_file.is_file():
        return "publication_manifest_missing"
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "publication_manifest_malformed"
    if not isinstance(payload, Mapping):
        return "publication_manifest_malformed"
    if payload.get("version") != PACKING_CACHE_VERSION:
        return "retired_or_unknown_version"
    if payload.get("status") != "complete":
        return "publication_incomplete"
    if payload.get("fingerprint") != expected_fingerprint:
        return "semantic_fingerprint_mismatch"
    error_text = str(error).lower()
    if "sha256" in error_text or "checksum" in error_text or "digest" in error_text:
        return "required_payload_digest_mismatch"
    if "pickle" in error_text or "payload" in error_text or "micro-step" in error_text:
        return "required_payload_invalid"
    if "chunk" in error_text:
        return "chunk_plan_or_payload_invalid"
    return "current_publication_invalid"


def _cache_preflight_error(
    *,
    cache_dir: Path,
    cache_root: Path,
    config_path: str | Path,
    fingerprint: str,
    split: str,
    error: BaseException,
) -> RuntimeContractError:
    common_context: dict[str, Any] = {
        "split": split,
        "cache_root": str(cache_root),
        "expected_cache_target": str(cache_dir),
        "cache_version": PACKING_CACHE_VERSION,
        "fingerprint": fingerprint,
    }
    if not cache_dir.exists():
        preparation_argv = _cache_preparation_argv(config_path)
        preparation_env = _cache_preparation_environment()
        preparation_command = shlex.join(
            [
                *(f"{name}={value}" for name, value in preparation_env.items()),
                *preparation_argv,
            ]
        )
        return RuntimeContractError(
            "required packing cache is not prepared; "
            f"expected cache target: {cache_dir}; prepare with: {preparation_command}",
            code="training.pack_cache_not_prepared",
            context={
                **common_context,
                "validation_category": "expected_target_missing",
                "automatic_recovery": "single_process_preparation_required",
                "preparation_argv": list(preparation_argv),
                "preparation_env": preparation_env,
                "preparation_command": preparation_command,
            },
            cause=error,
        )
    return RuntimeContractError(
        "required v3 packing cache target is occupied by an invalid immutable "
        f"publication: {cache_dir}",
        code="training.pack_cache_immutable_collision",
        context={
            **common_context,
            "validation_category": _occupied_cache_validation_category(
                cache_dir,
                expected_fingerprint=fingerprint,
                error=error,
            ),
            "automatic_recovery": "unavailable",
        },
        cause=error,
    )


def _admit_model_free_pack_cache(
    config: Any,
    components: Any,
    *,
    vocab_groups: Any,
    dataset: Any,
    split: str,
    cache_root: Path,
    config_path: str | Path,
    verification_level: str,
    resolved_fingerprint: str | None = None,
) -> dict[str, Any]:
    fingerprint = (
        build_packing_cache_fingerprint(
            config,
            components,
            dataset=dataset,
            split=split,
            vocab_groups=vocab_groups,
        )
        if resolved_fingerprint is None
        else resolved_fingerprint
    )
    cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
    admission_started = time.monotonic()
    try:
        manifest = load_cache_manifest(
            cache_dir,
            cache_root=cache_root,
            expected_fingerprint=fingerprint,
            level=verification_level,
        )
    except PackingCacheInvalidError as exc:
        raise _cache_preflight_error(
            cache_dir=cache_dir,
            cache_root=cache_root,
            config_path=config_path,
            fingerprint=fingerprint,
            split=split,
            error=exc,
        ) from exc
    admission_seconds = time.monotonic() - admission_started
    cache_manifest_path = manifest_path(cache_dir)
    return {
        "cache_dir": cache_dir,
        "format_version": str(manifest["version"]),
        "fingerprint": fingerprint,
        "micro_step_count": int(manifest["micro_step_count"]),
        "chunk_count": len(manifest["chunks"]),
        "chunk_size": int(manifest["chunk_size"]),
        "status": manifest["status"],
        "build_status": "hit",
        "manifest_path": cache_manifest_path,
        "manifest_sha256": _file_sha256(cache_manifest_path),
        "determinants_sha256": _sha256_json(manifest["determinants"]),
        "chunk_sha256s": [str(chunk["sha256"]) for chunk in manifest["chunks"]],
        "materialization": manifest.get("materialization"),
        "augmentation": manifest.get("augmentation"),
        "phase_receipt": {
            "cache_preparation": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_publication": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_admission": {
                "status": "completed",
                "duration_seconds": admission_seconds,
                "verification_level": verification_level,
            },
        },
    }


def _resolve_model_free_training_preflight(
    *,
    config: Any,
    config_path: str | Path,
    repo_root: Path,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None = None,
    phase_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and validate all required cache state before expensive setup."""

    started = time.monotonic()
    trace = {} if phase_trace is None else phase_trace

    def measured(phase: str, body: Callable[[], Any]) -> Any:
        phase_started = time.monotonic()
        status = "completed"
        failure: dict[str, str] | None = None
        try:
            return body()
        except Exception as exc:
            status = "failed"
            code = getattr(exc, "code", "python_exception")
            failure = {
                "error_type": type(exc).__name__[:128],
                "error_code": str(code)[:128],
            }
            raise
        finally:
            receipt: dict[str, Any] = {
                "status": status,
                "duration_seconds": max(0.0, time.monotonic() - phase_started),
                "resource_snapshot": collect_resource_snapshot(),
            }
            if failure is not None:
                receipt["failure"] = failure
            trace[phase] = receipt

    def resolve_identities() -> dict[str, Any]:
        components = load_qwen_components(config, load_model=False)
        vocab_groups = build_token_vocabulary_groups(
            components.token_identity,
            tokenizer=components.tokenizer,
        )
        resolve_qwen_runtime_controls(
            config,
            tokenizer_vocab_size=components.token_identity.tokenizer_vocab_size,
            model_logits_dtype=config.training.precision,
        )
        cache_root, cache_root_receipt = _resolve_pack_cache_root(repo_root)
        return {
            "components": components,
            "vocab_groups": vocab_groups,
            "cache_root": cache_root,
            "cache_root_receipt": cache_root_receipt,
            "train_fingerprint": build_packing_cache_fingerprint(
                config,
                components,
                dataset=config.data.train,
                split=TRAIN_SPLIT,
                vocab_groups=vocab_groups,
            ),
            "eval_fingerprint": (
                None
                if config.data.eval is None
                else build_packing_cache_fingerprint(
                    config,
                    components,
                    dataset=config.data.eval,
                    split="eval.forward",
                    vocab_groups=vocab_groups,
                )
            ),
        }

    identities = measured("cache_identity_resolution", resolve_identities)
    trace["cache_identity_resolution"]["details"] = {
        "train_identity_resolved": True,
        "eval_identity_resolved": identities["eval_fingerprint"] is not None,
    }
    components = identities["components"]
    vocab_groups = identities["vocab_groups"]
    cache_root = identities["cache_root"]
    cache_root_receipt = identities["cache_root_receipt"]

    def admit_publications() -> tuple[dict[str, Any], Any, dict[str, Any] | None]:
        train_cache = _admit_model_free_pack_cache(
            config,
            components,
            vocab_groups=vocab_groups,
            dataset=config.data.train,
            split=TRAIN_SPLIT,
            cache_root=cache_root,
            config_path=config_path,
            verification_level="manifest",
            resolved_fingerprint=str(identities["train_fingerprint"]),
        )
        schedule = resolve_planned_step_schedule(
            config,
            packs_per_epoch=int(train_cache["micro_step_count"]),
            world_size=world_size,
            source_config_path=str(config_path),
        )
        eval_cache = None
        if config.data.eval is not None:
            eval_cache = _admit_model_free_pack_cache(
                config,
                components,
                vocab_groups=vocab_groups,
                dataset=config.data.eval,
                split="eval.forward",
                cache_root=cache_root,
                config_path=config_path,
                verification_level="payloads",
                resolved_fingerprint=str(identities["eval_fingerprint"]),
            )
        return train_cache, schedule, eval_cache

    train_cache, schedule, eval_cache = measured(
        "cache_publication_admission", admit_publications
    )
    trace["cache_publication_admission"]["details"] = {
        "train_verification_level": "manifest",
        "eval_verification_level": (
            "not_applicable" if eval_cache is None else "payloads"
        ),
        "train_manifest_micro_step_count": int(train_cache["micro_step_count"]),
        "eval_manifest_micro_step_count": (
            0 if eval_cache is None else int(eval_cache["micro_step_count"])
        ),
        "immutable_publication_admitted": True,
    }

    eval_reduction = _resolve_converged_eval_reduction_receipt(
        pack_count=(
            None if eval_cache is None else int(eval_cache["micro_step_count"])
        ),
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
    )
    trace["cache_publication_admission"]["details"]["eval_reduction"] = dict(
        eval_reduction
    )

    def hydrate_train_rank() -> tuple[SupervisedMicroStep, ...]:
        try:
            return load_rank_micro_steps_from_cache(
                train_cache["cache_dir"],
                cache_root=cache_root,
                expected_fingerprint=str(train_cache["fingerprint"]),
                schedule=schedule,
                rank=rank,
                world_size=world_size,
            )
        except PackingCacheInvalidError as exc:
            raise _cache_preflight_error(
                cache_dir=Path(train_cache["cache_dir"]),
                cache_root=cache_root,
                config_path=config_path,
                fingerprint=str(train_cache["fingerprint"]),
                split=TRAIN_SPLIT,
                error=exc,
            ) from exc

    train_micro_steps = measured("train_rank_hydration", hydrate_train_rank)
    trace["train_rank_hydration"]["details"] = {
        "retained_micro_step_count": len(train_micro_steps),
        "rank": rank,
        "world_size": world_size,
    }
    return {
        "components": components,
        "vocab_groups": vocab_groups,
        "cache_root": cache_root,
        "cache_root_receipt": cache_root_receipt,
        "train_cache": train_cache,
        "train_micro_steps": train_micro_steps,
        "eval_cache": eval_cache,
        "eval_reduction": eval_reduction,
        "schedule": schedule,
        "rank": rank,
        "world_size": world_size,
        "duration_seconds": time.monotonic() - started,
        "phase_trace": trace,
    }


def _hydrate_eval_micro_steps_from_cache(
    eval_cache: Mapping[str, Any],
    *,
    cache_root: Path,
    rank: int,
    world_size: int,
    rank_report_gatherer: Callable[[Any], Sequence[Any]] | None,
    reduction_receipt: Mapping[str, Any] | None = None,
    receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[tuple[SupervisedMicroStep, ...], str, int]:
    """Hydrate one exact eval ordinal assignment without a full-load fallback."""

    hydration_details: dict[str, Any] = {}

    def hydrate() -> tuple[tuple[SupervisedMicroStep, ...], str, int]:
        raw_pack_count = eval_cache.get("micro_step_count")
        if (
            not isinstance(raw_pack_count, int)
            or isinstance(raw_pack_count, bool)
            or raw_pack_count <= 0
        ):
            raise RuntimeContractError(
                "evaluation cache must declare a positive micro-step count",
                code="training.eval_cache_count_invalid",
                context={"micro_step_count": raw_pack_count},
            )
        pack_count = int(raw_pack_count)
        resolved_reduction = (
            _resolve_eval_reduction_receipt(
                pack_count=pack_count,
                world_size=world_size,
            )
            if reduction_receipt is None
            else dict(reduction_receipt)
        )
        expected_fields = {
            "control",
            "effective_mode",
            "source",
            "pack_count",
            "world_size",
        }
        if (
            set(resolved_reduction) != expected_fields
            or resolved_reduction.get("pack_count") != pack_count
            or resolved_reduction.get("world_size") != world_size
            or resolved_reduction.get("effective_mode")
            not in {EVAL_REDUCTION_REPLICATED, EVAL_REDUCTION_DISJOINT_SHARD}
        ):
            raise RuntimeContractError(
                "eval hydration disagrees with the model-free reduction receipt",
                code="training.eval_reduction_receipt_mismatch",
                context={
                    "admitted_pack_count": resolved_reduction.get("pack_count"),
                    "hydration_pack_count": pack_count,
                    "admitted_world_size": resolved_reduction.get("world_size"),
                    "hydration_world_size": world_size,
                },
            )
        reduction_mode = str(resolved_reduction["effective_mode"])

        if reduction_mode == EVAL_REDUCTION_DISJOINT_SHARD:
            # The selective loader already applies the canonical modulo
            # assignment. Production must not partition this shard again.
            loader_rank = rank
            loader_world_size = world_size
        elif world_size > 1 and pack_count < world_size:
            # Structural replicated fallback: empty disjoint ranks cannot
            # participate in the accepted evaluation contract. Hydrate the
            # exact full ordinal sequence through the selective API itself.
            loader_rank = 0
            loader_world_size = 1
        else:
            # Single-rank execution and the explicit internal replicated
            # control both request the one exact all-ordinal assignment. This
            # is not a recovery path for a failed selective load.
            loader_rank = 0
            loader_world_size = 1

        shard = load_rank_eval_micro_steps_from_cache(
            eval_cache["cache_dir"],
            cache_root=cache_root,
            expected_fingerprint=str(eval_cache["fingerprint"]),
            rank=loader_rank,
            world_size=loader_world_size,
        )
        total_ordinal_count = int(shard.total_ordinal_count)
        if total_ordinal_count != pack_count:
            raise RuntimeContractError(
                "selective eval hydration disagrees with the admitted pack count",
                code="training.eval_hydration_total_count_mismatch",
                context={
                    "admitted_pack_count": pack_count,
                    "loaded_total_ordinal_count": total_ordinal_count,
                },
            )
        expected_ordinals = tuple(
            range(loader_rank, total_ordinal_count, loader_world_size)
        )
        observed_ordinals = tuple(shard.canonical_ordinals)
        if observed_ordinals != expected_ordinals:
            raise RuntimeContractError(
                "selective eval hydration returned a noncanonical ordinal assignment",
                code="training.eval_hydration_ordinal_mismatch",
                context={
                    "loader_rank": loader_rank,
                    "loader_world_size": loader_world_size,
                    "expected_count": len(expected_ordinals),
                    "observed_count": len(observed_ordinals),
                },
            )
        micro_steps = tuple(shard.micro_steps)
        if len(micro_steps) != len(expected_ordinals):
            raise RuntimeContractError(
                "selective eval hydration payload count disagrees with its ordinals",
                code="training.eval_hydration_payload_count_mismatch",
                context={
                    "ordinal_count": len(expected_ordinals),
                    "payload_count": len(micro_steps),
                },
            )
        hydration_details.update(
            {
                "assignment_mode": reduction_mode,
                "loader_rank": loader_rank,
                "loader_world_size": loader_world_size,
                "canonical_ordinal_total_count": total_ordinal_count,
                "canonical_ordinal_assigned_count": len(expected_ordinals),
                "retained_micro_step_count": len(micro_steps),
                "selective_decode": {
                    "decoded_micro_step_count": len(micro_steps),
                    "decoded_chunk_count": {
                        "status": "unavailable",
                        "reason": "selective_loader_counter_not_exposed",
                    },
                    "payload_bytes_read": {
                        "status": "unavailable",
                        "reason": "selective_loader_byte_counter_not_exposed",
                    },
                },
            }
        )
        return micro_steps, reduction_mode, total_ordinal_count

    return control_plane._run_rank_converged_phase(
        "evaluation_hydration",
        rank=rank,
        world_size=world_size,
        rank_report_gatherer=rank_report_gatherer,
        body=hydrate,
        local_details=lambda: hydration_details,
        receipt_sink=receipt_sink,
    )


def _bind_cache_materialization(
    writer: RunWriter, split: str, cache: Mapping[str, Any]
) -> None:
    writer.bind_materialization(
        split,
        cache_format_version=str(cache["format_version"]),
        semantic_fingerprint=str(cache["fingerprint"]),
        determinant_digest=str(cache["determinants_sha256"]),
    )


def _attach_image_processors_to_micro_steps(
    micro_steps: Sequence[SupervisedMicroStep],
    *,
    image_processor: Any,
) -> tuple[SupervisedMicroStep, ...]:
    if image_processor is None:
        raise RuntimeContractError(
            "cached Qwen image encodings require a runtime image_processor",
            code="training.qwen_image_processor_missing",
            context={},
        )
    return tuple(
        replace(
            micro_step,
            encoded_examples=tuple(
                _attach_image_processor_to_encoded_example(
                    encoded_example,
                    image_processor=image_processor,
                )
                for encoded_example in micro_step.encoded_examples
            ),
        )
        for micro_step in micro_steps
    )


def _attach_image_processor_to_encoded_example(
    encoded_example: Any,
    *,
    image_processor: Any,
) -> Any:
    image_encoding = getattr(encoded_example, "image_encoding", None)
    if not isinstance(image_encoding, QwenImageEncoding):
        return encoded_example
    return replace(
        encoded_example,
        image_encoding=attach_qwen_image_processor(image_encoding, image_processor),
    )


def _apply_fa2_branch_proof_policy(
    micro_steps: Sequence[SupervisedMicroStep],
    config: Any,
) -> tuple[SupervisedMicroStep, ...]:
    policy = config.model.fa2_branch_proof
    configured: list[SupervisedMicroStep] = []
    for local_index, micro_step in enumerate(micro_steps):
        capture = policy == "every_forward" or (
            policy == "first_micro_step" and local_index == 0
        )
        configured.append(
            replace(
                micro_step,
                fa2_branch_evidence=None,
                capture_fa2_branch=capture,
                require_fa2_branch_proof=capture,
                fa2_branch_proof_policy=policy,
            )
        )
    return tuple(configured)


def _qwen_image_processor(components: Any) -> Any:
    processor = getattr(components, "processor", None)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeContractError(
            "Qwen components must expose processor.image_processor for lazy image packing",
            code="training.qwen_image_processor_missing",
            context={"processor_type": type(processor).__name__},
        )
    return image_processor


def _materialize_raw_examples_for_dataset(
    config: Any,
    dataset: Any,
    *,
    split: str,
) -> AugmentationMaterializationResult:
    raw_examples = load_raw_examples(dataset)
    processor = build_augmentation_processor(config, split=split)
    return processor.materialize(
        raw_examples,
        split=split,
        object_ordering=config.template.object_ordering,
    )


def _render_and_encode_example(
    raw_example: Any,
    *,
    config: Any,
    components: Any,
) -> Any:
    rendered = render_example(
        raw_example,
        config.template,
        object_order_seed=_object_order_seed(config, raw_example.example_id),
    )
    return encode_rendered_example(
        raw_example,
        rendered,
        components=components,
        processor_config=config.model.processor,
        global_max_length=config.packing.global_max_length,
        materialize_image_pixels=False,
    )


def _encode_example_worker(index: int) -> tuple[int, Any]:
    context = _PACK_CACHE_WORKER_CONTEXT
    if context is None:
        raise RuntimeContractError(
            "packing cache worker context was not initialized",
            code="training.pack_cache_worker_context_missing",
            context={"index": index},
        )
    raw_examples = context["raw_examples"]
    raw_example = raw_examples[index]
    encoded = _render_and_encode_example(
        raw_example,
        config=context["config"],
        components=context["components"],
    )
    return index, encoded


def _restore_encoded_example_order(
    indexed_results: Sequence[tuple[int, Any]],
    *,
    expected_count: int,
) -> tuple[Any, ...]:
    ordered: list[Any | None] = [None for _ in range(expected_count)]
    seen: set[int] = set()
    for index, encoded_example in indexed_results:
        if index < 0 or index >= expected_count:
            raise RuntimeContractError(
                "packing cache worker returned an out-of-range example index",
                code="training.pack_cache_worker_index",
                context={"index": index, "expected_count": expected_count},
            )
        if index in seen:
            raise RuntimeContractError(
                "packing cache worker returned a duplicate example index",
                code="training.pack_cache_worker_index_duplicate",
                context={"index": index},
            )
        seen.add(index)
        ordered[index] = encoded_example
    if len(seen) != expected_count:
        missing = sorted(set(range(expected_count)) - seen)
        raise RuntimeContractError(
            "packing cache workers did not return every encoded example",
            code="training.pack_cache_worker_index_missing",
            context={"missing_indices": missing[:16], "missing_count": len(missing)},
        )
    return tuple(encoded_example for encoded_example in ordered)


def _encode_examples_with_fork_process_pool(
    config: Any,
    components: Any,
    raw_examples: Sequence[Any],
    *,
    workers: int,
) -> tuple[Any, ...]:
    mp_context = _fork_multiprocessing_context(workers)
    global _PACK_CACHE_WORKER_CONTEXT
    _PACK_CACHE_WORKER_CONTEXT = {
        "config": config,
        "components": components,
        "raw_examples": tuple(raw_examples),
    }
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(_encode_example_worker, index)
                for index in range(len(raw_examples))
            ]
            indexed_results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
    finally:
        _PACK_CACHE_WORKER_CONTEXT = None
    return _restore_encoded_example_order(
        indexed_results,
        expected_count=len(raw_examples),
    )


def _build_encoded_examples_for_dataset(
    config: Any,
    components: Any,
    raw_examples: Sequence[Any],
    *,
    materialization_workers: int | None = None,
) -> tuple[Any, ...]:
    workers = _resolve_pack_cache_materialization_workers(materialization_workers)
    if workers == 1:
        return tuple(
            _render_and_encode_example(
                raw_example,
                config=config,
                components=components,
            )
            for raw_example in raw_examples
        )
    return _encode_examples_with_fork_process_pool(
        config,
        components,
        raw_examples,
        workers=workers,
    )


def _resolve_pack_cache_materialization_workers(workers: int | None) -> int:
    resolved_workers = (
        DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS if workers is None else workers
    )
    if isinstance(resolved_workers, bool) or not isinstance(resolved_workers, int):
        raise RuntimeContractError(
            "packing cache materialization workers must be an integer",
            code="training.pack_cache_workers_invalid",
            context={"workers": resolved_workers},
        )
    if resolved_workers <= 0:
        raise RuntimeContractError(
            "packing cache materialization workers must be positive",
            code="training.pack_cache_workers_invalid",
            context={"workers": resolved_workers},
        )
    return resolved_workers


def _fork_multiprocessing_context(workers: int) -> Any:
    available_start_methods = tuple(multiprocessing.get_all_start_methods())
    if "fork" not in available_start_methods:
        raise RuntimeContractError(
            "parallel packing-cache materialization requires multiprocessing fork",
            code="training.pack_cache_workers_unavailable",
            context={
                "workers": workers,
                "available_start_methods": available_start_methods,
            },
        )
    try:
        return multiprocessing.get_context("fork")
    except ValueError as exc:
        raise RuntimeContractError(
            "parallel packing-cache materialization could not acquire fork context",
            code="training.pack_cache_workers_unavailable",
            context={
                "workers": workers,
                "available_start_methods": available_start_methods,
            },
            cause=exc,
        ) from exc


def _materialize_pack_plan(
    config: Any,
    encoded_examples: Sequence[Any],
) -> tuple[tuple[PackedSequence, ...], dict[str, Any], dict[int, str]]:
    """Plan and authenticate exact memberships before supervision is built."""

    packing = config.packing
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
    common_kwargs = {
        "global_max_length": int(packing.global_max_length),
        "seed": int(packing.seed),
        "worker_count": int(packing.worker_count),
        "cursor_byte_budget": int(packing.cursor_byte_budget),
        "fragment_item_budget": int(packing.fragment_item_budget),
        "fragment_byte_budget": int(packing.fragment_byte_budget),
    }
    if packing.policy != ONLINE_WINDOW_BINPACK:
        plan = create_pack_plan(
            encoded_examples,
            policy=str(packing.policy),
            window_size=packing.window_size,
            lookahead=packing.lookahead,
            **common_kwargs,
        )
        _reject_pack_plan_omissions((plan,))
        packs = replay_pack_plan(plan, encoded_examples)
        receipt = {
            "schema_version": 1,
            "mode": "complete_plan",
            "policy_identity": policy_identity,
            "plan_sha256": plan.canonical_sha256,
            "fragment_chain_sha256": None,
            "fragment_count": 1,
            "source_input_count": len(encoded_examples),
            "emitted_pack_count": len(packs),
        }
        return (
            packs,
            receipt,
            {pack.pack_index: plan.canonical_sha256 for pack in packs},
        )

    fragments: list[PackPlan] = []
    stream_receipt = stream_online_pack_plan_fragments(
        lambda: iter(encoded_examples),
        fragment_sink=fragments.append,
        lookahead=int(packing.lookahead),
        max_packs_per_fragment=int(packing.max_packs_per_fragment),
        **common_kwargs,
    )
    verify_pack_plan_stream_fragments(stream_receipt, fragments)
    _reject_pack_plan_omissions(fragments)
    materialized_fragments = tuple(
        (fragment, replay_pack_plan(fragment, encoded_examples))
        for fragment in fragments
    )
    packs = tuple(
        pack for _, fragment_packs in materialized_fragments for pack in fragment_packs
    )
    fragment_by_pack = {
        pack.pack_index: fragment.canonical_sha256
        for fragment, fragment_packs in materialized_fragments
        for pack in fragment_packs
    }
    receipt = {
        "schema_version": 1,
        "mode": "bounded_online_fragments",
        "policy_identity": policy_identity,
        "plan_sha256": None,
        "fragment_chain_sha256": stream_receipt.fragment_chain_sha256,
        "fragment_count": stream_receipt.fragment_count,
        "source_input_count": stream_receipt.source_input_count,
        "emitted_pack_count": stream_receipt.emitted_pack_count,
        "max_fragment_items_observed": stream_receipt.max_fragment_items_observed,
        "max_fragment_bytes_observed": stream_receipt.max_fragment_bytes_observed,
        "max_pending_items_observed": stream_receipt.max_pending_items_observed,
        "max_cursor_bytes_observed": stream_receipt.max_cursor_bytes_observed,
    }
    return packs, receipt, fragment_by_pack


def _reject_pack_plan_omissions(plans: Sequence[PackPlan]) -> None:
    rejected = [item for plan in plans for item in plan.rejected_examples]
    if not rejected:
        return
    raise RuntimeContractError(
        "pack planning rejected one or more encoded examples",
        code="training.pack_plan_rejected_examples",
        context={
            "rejected_count": len(rejected),
            "example_ids": [item.example_id for item in rejected[:16]],
        },
    )


def _build_micro_steps_for_dataset(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    dataset: Any,
    split: str,
    materialization_workers: int | None = None,
) -> tuple[SupervisedMicroStep, ...]:
    if dataset is None:
        raise RuntimeContractError(
            "micro-step construction requires an explicit dataset split",
            code="training.dataset_split_missing",
            context={"split": split},
        )
    augmentation_result = _materialize_raw_examples_for_dataset(
        config,
        dataset,
        split=split,
    )
    raw_examples = augmentation_result.examples
    encoded_examples = _build_encoded_examples_for_dataset(
        config,
        components,
        raw_examples,
        materialization_workers=materialization_workers,
    )
    packs, pack_plan_receipt, fragment_by_pack = _materialize_pack_plan(
        config,
        encoded_examples,
    )
    supervision = build_packed_supervision(packs, encoded_examples)
    token_atoms_by_pack = index_token_atoms_by_pack(supervision)
    micro_steps: list[SupervisedMicroStep] = []
    for pack in packs:
        pack_examples = _encoded_examples_for_pack(pack, encoded_examples)
        position_inputs = build_qwen_position_inputs(
            pack,
            pack_examples,
            image_token_id=_image_token_id(components),
        )
        token_sequence = build_token_sequence_from_packed_supervision(
            pack,
            token_atoms_by_pack.get(pack.pack_index, ()),
        )
        micro_steps.append(
            SupervisedMicroStep(
                pack=pack,
                encoded_examples=pack_examples,
                position_inputs=position_inputs,
                token_sequence=token_sequence,
                vocab_groups=vocab_groups,
                metadata={
                    "split": split,
                    "pack_id": pack.pack_index,
                    "example_ids": [segment.example_id for segment in pack.segments],
                    "augmentation_receipt": augmentation_result.receipt,
                    "pack_plan": {
                        **pack_plan_receipt,
                        "fragment_sha256": fragment_by_pack[pack.pack_index],
                    },
                },
                expected_vocab_size=components.token_identity.tokenizer_vocab_size,
                fa2_model_dtype=config.training.precision,
                capture_fa2_branch=config.model.fa2_branch_proof == "every_forward",
                require_fa2_branch_proof=config.model.fa2_branch_proof
                == "every_forward",
            )
        )
    if not micro_steps:
        raise RuntimeContractError(
            "micro-step construction produced no packs",
            code="training.empty_micro_step_plan",
            context={"split": split},
        )
    return tuple(micro_steps)


def _augmentation_receipt_from_micro_steps(
    micro_steps: Sequence[SupervisedMicroStep],
) -> dict[str, Any] | None:
    for micro_step in micro_steps:
        metadata = getattr(micro_step, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        receipt = metadata.get("augmentation_receipt")
        if isinstance(receipt, Mapping):
            return dict(receipt)
    return None


def _encoded_examples_for_pack(
    pack: PackedSequence,
    encoded_examples: Sequence[Any],
) -> tuple[Any, ...]:
    examples_by_id = {
        str(getattr(example, "example_id")): example for example in encoded_examples
    }
    return tuple(examples_by_id[segment.example_id] for segment in pack.segments)


def _object_order_seed(config: Any, example_id: str) -> int | None:
    del example_id
    if config.template.object_ordering == "random":
        return int(config.runtime.seed)
    return None


def _image_token_id(components: Any) -> int | None:
    tokenizer = getattr(components, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    token_id = convert("<|image_pad|>")
    return None if token_id is None else int(token_id)
