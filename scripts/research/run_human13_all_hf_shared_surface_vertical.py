#!/usr/bin/env python3
"""Guarded Task-4 entry and audit contracts for the all-HF Human-13 vertical.

This module owns the lifecycle seam only.  It does not load a model, reserve a
GPU, or create an output directory by itself.  Live callers inject the already
admitted Task-1/2/3 owners through :class:`OneImageServices`; the public CLI is
therefore a value-only dry run unless a later execution owner supplies an
explicit adapter.  The audit projection intentionally delegates parsing and
matching to ``analyze_human13_k_union``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast
from weakref import ReferenceType, ref


ALL_HF_VERTICAL_UNIT_ID = (
    "2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical"
)
CONFIG_SCHEMA_VERSION = "human13_all_hf_shared_surface_vertical_config.v1"
TERMINAL_SCHEMA_VERSION = "human13_all_hf_shared_surface_vertical_terminal.v2"
LEGACY_TERMINAL_SCHEMA_VERSION = "human13_all_hf_shared_surface_vertical_terminal.v1"
PHASE_LEDGER_SCHEMA_VERSION = "human13_all_hf_phase_ledger.v1"
RESOURCE_SCHEMA_VERSION = "human13_all_hf_shared_surface_vertical_resource.v1"
RESERVATION_IDENTITY_SCHEMA_VERSION = "human13_one_image_reservation_identity.v1"
SOURCE_ASSEMBLY_SCHEMA_VERSION = "human13_all_hf_source_assembly.v1"
SEED_GROUPS = (
    (35001, 35002, 35003, 35004),
    (35005, 35006, 35007, 35008),
    (35009, 35010, 35011, 35012),
    (35013, 35014, 35015, 35016),
)
AUDIT_REPETITION_PENALTIES = (1.0, 1.10)
MIN_PRODUCTION_GPU_TOTAL_BYTES = 75 << 30
MIN_TRAINING_GPU_FREE_BYTES = 64 << 30
MIN_AUDIT_GPU_FREE_BYTES = 24 << 30
ZERO_MODEL_ACTIONS = MappingProxyType(
    {
        "model_loads": 0,
        "forwards": 0,
        "backwards": 0,
        "optimizer_steps": 0,
        "gpu_allocations": 0,
        "network_actions": 0,
        "output_creations": 0,
    }
)
_CAP_STOP_REASONS = frozenset(
    {"cap", "cap_stop", "length", "length_truncated", "max_new_tokens"}
)
_NATURAL_STOP_REASONS = frozenset({"eos", "im_end", "natural_stop"})
_SUPPORTED_STOP_REASONS = _CAP_STOP_REASONS | _NATURAL_STOP_REASONS
_CANONICAL_AUDIT_PARSER = "compact_object_box_closed_only"
_CANONICAL_PARSER_STATUSES = frozenset(
    {"accepted", "accepted_with_drops", "empty", "all_spans_dropped"}
)
_SOURCE_AUDIT_ARM_ID = "frozen_source"
_PROPOSAL_AUDIT_ARM_ID = "private_proposal"
_DIGEST_RE = frozenset("0123456789abcdef")
_TERMINAL_SEALS: dict[int, tuple[ReferenceType[Any], str]] = {}
_TERMINAL_ISSUER_TOKEN = object()


def _install_canonical_module_alias() -> None:
    """Keep ``python -m`` and package imports on one class identity.

    The services owner imports this module by its package name.  When the
    public entry is launched with ``python -m``, Python initially registers it
    only as ``__main__``; without this alias, terminal receipt classes are
    duplicated and strict typed persistence rejects the receipt at the end of
    an otherwise completed lifecycle.
    """

    if __name__ == "__main__":
        sys.modules.setdefault(
            "scripts.research.run_human13_all_hf_shared_surface_vertical",
            sys.modules[__name__],
        )


_install_canonical_module_alias()


def _ensure_repo_root_on_sys_path(repo_root: Path) -> None:
    """Make direct script execution resolve the repository package imports.

    ``python scripts/research/<entry>.py`` puts only the script directory on
    ``sys.path``.  The production execution path imports sibling owners via
    ``scripts.research`` after parsing ``--repo-root``; install that explicit
    root before the first dynamic import without relying on the caller's
    working directory or ``PYTHONPATH``.
    """

    root = str(repo_root.expanduser().resolve())
    sys.path[:] = [entry for entry in sys.path if entry != root]
    sys.path.insert(0, root)


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _phase_ledger_sha256(values: Sequence[str]) -> str:
    hashes = tuple(_digest(value, field="phase receipt SHA-256") for value in values)
    return _sha256(
        {
            "schema_version": PHASE_LEDGER_SCHEMA_VERSION,
            "phase_receipt_sha256s": list(hashes),
        }
    )


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _DIGEST_RE for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _rp(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result not in AUDIT_REPETITION_PENALTIES:
        raise ValueError(f"{field} must be exactly 1.0 or 1.10")
    return result


def _ordered_rps(value: Sequence[object]) -> tuple[float, float]:
    result = tuple(_rp(item, field="audit repetition penalty") for item in value)
    if result != AUDIT_REPETITION_PENALTIES:
        raise ValueError("audit repetition penalties must be exactly (1.0, 1.10)")
    return cast(tuple[float, float], result)


def _as_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


@dataclass(frozen=True)
class RunReservationIdentity:
    """Mode-neutral identity of one atomically admitted immutable run root."""

    reservation_mode: Literal["fresh_primary", "lost_owner_recovery"]
    run_id: str
    output_root: str
    reservation_sha256: str
    recovery_successor_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.reservation_mode not in {"fresh_primary", "lost_owner_recovery"}:
            raise ValueError("reservation mode is unsupported")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("reservation run_id must be nonempty")
        if not isinstance(self.output_root, str) or not Path(self.output_root).is_absolute():
            raise ValueError("reservation output_root must be absolute")
        _digest(self.reservation_sha256, field="reservation_sha256")
        if self.reservation_mode == "fresh_primary":
            if self.recovery_successor_sha256 is not None:
                raise ValueError("fresh primary reservation must not bind recovery")
        elif self.recovery_successor_sha256 is None:
            raise ValueError("lost-owner recovery must bind its recovery receipt")
        if self.recovery_successor_sha256 is not None:
            _digest(
                self.recovery_successor_sha256,
                field="recovery_successor_sha256",
            )

    def _payload(self) -> dict[str, Any]:
        value = {
            "schema_version": RESERVATION_IDENTITY_SCHEMA_VERSION,
            "reservation_mode": self.reservation_mode,
            "run_id": self.run_id,
            "output_root": self.output_root,
            "reservation_sha256": self.reservation_sha256,
            "recovery_successor_sha256": self.recovery_successor_sha256,
        }
        return value

    @property
    def content_sha256(self) -> str:
        return _sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return self._payload() | {"content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunReservationIdentity:
        if value.get("schema_version") != RESERVATION_IDENTITY_SCHEMA_VERSION:
            raise ValueError("reservation identity schema differs")
        identity = cls(
            reservation_mode=value["reservation_mode"],
            run_id=value["run_id"],
            output_root=value["output_root"],
            reservation_sha256=value["reservation_sha256"],
            recovery_successor_sha256=value.get("recovery_successor_sha256"),
        )
        if value.get("content_sha256") != identity.content_sha256:
            raise ValueError("reservation identity content hash differs")
        return identity


@dataclass(frozen=True)
class EntryConfig:
    """Frozen value-only plan for the one-image Task-4 entry."""

    unit_id: str
    image_id: int
    seed_groups: tuple[tuple[int, int, int, int], ...]
    training_repetition_penalty: float
    dtype: Literal["bfloat16"]
    attention_backend: Literal["flash_attention_2"]
    use_cache: Literal[False]
    learning_rate: float
    optimizer_name: Literal["adamw_torch"]
    output_root: str
    audit_repetition_penalties: tuple[float, float]
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_epsilon: float = 1.0e-8
    optimizer_weight_decay: float = 0.0
    source_checkpoint_path: str | None = None
    base_model_path: str | None = None
    adapter_path: str | None = None
    special_embedding_path: str | None = None
    source_adapter_sha256: str | None = None
    special_embedding_sha256: str | None = None
    manifest_sha256: str | None = None
    training_temperature: float = 0.4
    training_top_p: float = 1.0
    training_top_k: int | None = None
    training_max_new_tokens: int = 512
    training_stop_token: str = "<|im_end|>"
    audit_backend: Literal["hf"] = "hf"
    audit_dtype: Literal["float32"] = "float32"
    audit_attention_backend: Literal["sdpa"] = "sdpa"
    audit_batch_size: int = 1
    collision_policy: Literal["fail"] = "fail"
    resource_training_gpu: int = 0
    resource_audit_gpu: int = 1
    resource_retry_policy: Literal["none"] = "none"
    resource_promotion: Literal[False] = False
    parser: str = "compact_object_box_closed_only"
    matcher_algorithm: str = "cardinality_first_max_total_iou"
    matcher_duplicate_iou: float = 0.95
    matcher_owner_iou: float = 0.50

    def __post_init__(self) -> None:
        if self.unit_id != ALL_HF_VERTICAL_UNIT_ID:
            raise ValueError("entry unit_id differs from the frozen Task-4 unit")
        if self.image_id != 1584:
            raise ValueError("entry image_id must remain 1584")
        groups = tuple(tuple(int(seed) for seed in group) for group in self.seed_groups)
        if groups != SEED_GROUPS:
            raise ValueError("entry seed groups differ from frozen 35001..35016 K16")
        object.__setattr__(self, "seed_groups", groups)
        if self.training_repetition_penalty != 1.0:
            raise ValueError("training repetition penalty must be exactly 1.0")
        if self.dtype != "bfloat16" or self.attention_backend != "flash_attention_2":
            raise ValueError("training surface must remain BF16/FlashAttention-2")
        if self.use_cache is not False:
            raise ValueError("cache is forbidden")
        if self.learning_rate != 3.0e-6 or self.optimizer_name != "adamw_torch":
            raise ValueError("optimizer must remain fresh AdamW at 3e-6")
        if self.optimizer_betas != (0.9, 0.999):
            raise ValueError("optimizer betas must remain exactly (0.9, 0.999)")
        if self.optimizer_epsilon != 1.0e-8:
            raise ValueError("optimizer epsilon must remain exactly 1e-8")
        if self.optimizer_weight_decay != 0.0:
            raise ValueError("optimizer weight decay must remain exactly 0")
        if self.training_temperature != 0.4:
            raise ValueError("training temperature must remain exactly 0.4")
        if self.training_top_p != 1.0:
            raise ValueError("training top_p must remain exactly 1.0")
        if self.training_top_k is not None:
            raise ValueError("training top_k must remain unset")
        if self.training_max_new_tokens != 512:
            raise ValueError("training max_new_tokens must remain exactly 512")
        if self.training_stop_token != "<|im_end|>":
            raise ValueError("training stop token differs from the frozen stop token")
        if (
            self.audit_backend != "hf"
            or self.audit_dtype != "float32"
            or self.audit_attention_backend != "sdpa"
            or self.audit_batch_size != 1
        ):
            raise ValueError("audit surface differs from frozen HF fp32/SDPA batch-one")
        if self.collision_policy != "fail":
            raise ValueError("output collision policy must remain fail")
        if (
            self.resource_training_gpu != 0
            or self.resource_audit_gpu != 1
            or self.resource_retry_policy != "none"
            or self.resource_promotion is not False
        ):
            raise ValueError(
                "resource contract must remain GPU0/GPU1, no retry, no promotion"
            )
        if not isinstance(self.output_root, str) or not self.output_root:
            raise ValueError("output_root must be nonempty")
        object.__setattr__(
            self,
            "audit_repetition_penalties",
            _ordered_rps(self.audit_repetition_penalties),
        )
        for field in (
            "source_adapter_sha256",
            "special_embedding_sha256",
            "manifest_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                _digest(value, field=field)
        if self.matcher_algorithm != "cardinality_first_max_total_iou":
            raise ValueError(
                "entry matcher must remain cardinality-first max-total-IoU"
            )
        if self.parser != "compact_object_box_closed_only":
            raise ValueError("entry parser differs from the canonical parser")
        for field in ("matcher_duplicate_iou", "matcher_owner_iou"):
            threshold = _finite_float(getattr(self, field), field=field)
            expected = 0.95 if field == "matcher_duplicate_iou" else 0.50
            if threshold != expected:
                raise ValueError(f"{field} must remain exactly {expected}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EntryConfig:
        if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError("entry config schema_version differs")
        training = value.get("training", {})
        surface = value.get("surface", {})
        optimizer = value.get("optimizer", {})
        audit = value.get("audit", {})
        run = value.get("run", {})
        source = value.get("source", {})
        training = _as_mapping(training, field="training")
        surface = _as_mapping(surface, field="surface")
        optimizer = _as_mapping(optimizer, field="optimizer")
        audit = _as_mapping(audit, field="audit")
        run = _as_mapping(run, field="run")
        source = _as_mapping(source, field="source")
        groups_value = value.get("seed_groups", SEED_GROUPS)
        if not isinstance(groups_value, (list, tuple)):
            raise ValueError("seed_groups must be an array")
        groups = tuple(tuple(int(seed) for seed in group) for group in groups_value)
        image_id = value.get("image_id")
        if not isinstance(image_id, int):
            raise ValueError("image_id is required")
        training_temperature = training.get("temperature", 0.4)
        training_top_p = training.get("top_p", 1.0)
        training_top_k = training.get("top_k")
        training_max_new_tokens = training.get("max_new_tokens", 512)
        training_stop_token = training.get("stop_token", "<|im_end|>")
        if not isinstance(training_max_new_tokens, int) or isinstance(
            training_max_new_tokens, bool
        ):
            raise ValueError("training max_new_tokens must be an integer")
        audit_backend = audit.get("backend", "hf")
        audit_dtype = audit.get("dtype", "float32")
        audit_attention_backend = audit.get("attention_backend", "sdpa")
        audit_batch_size = audit.get("batch_size", 1)
        if not isinstance(audit_batch_size, int) or isinstance(audit_batch_size, bool):
            raise ValueError("audit batch_size must be an integer")
        optimizer_betas_value = optimizer.get("betas", (0.9, 0.999))
        if (
            not isinstance(optimizer_betas_value, (list, tuple))
            or len(optimizer_betas_value) != 2
        ):
            raise ValueError("optimizer betas must contain exactly two values")
        optimizer_betas = (
            float(optimizer_betas_value[0]),
            float(optimizer_betas_value[1]),
        )
        resource = _as_mapping(value.get("resource", {}), field="resource")
        resource_training_gpu = resource.get("training_gpu", 0)
        resource_audit_gpu = resource.get("audit_gpu", 1)
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (resource_training_gpu, resource_audit_gpu)
        ):
            raise ValueError("resource GPU roles must be integers")
        output_root = run.get("output_root", value.get("output_root"))
        if not isinstance(output_root, str):
            raise ValueError("run.output_root is required")
        return cls(
            unit_id=str(value.get("unit_id", "")),
            image_id=image_id,
            seed_groups=groups,  # type: ignore[arg-type]
            training_repetition_penalty=float(
                training.get(
                    "repetition_penalty", value.get("training_repetition_penalty", 1.0)
                )
            ),
            dtype=surface.get("dtype", "bfloat16"),  # type: ignore[arg-type]
            attention_backend=surface.get("attention_backend", "flash_attention_2"),  # type: ignore[arg-type]
            use_cache=surface.get("use_cache", False),  # type: ignore[arg-type]
            learning_rate=float(optimizer.get("learning_rate", 3.0e-6)),
            optimizer_name=optimizer.get("name", "adamw_torch"),  # type: ignore[arg-type]
            optimizer_betas=optimizer_betas,
            optimizer_epsilon=float(optimizer.get("epsilon", 1.0e-8)),
            optimizer_weight_decay=float(optimizer.get("weight_decay", 0.0)),
            output_root=output_root,
            audit_repetition_penalties=tuple(
                audit.get("repetition_penalties", AUDIT_REPETITION_PENALTIES)
            ),  # type: ignore[arg-type]
            training_temperature=float(training_temperature),
            training_top_p=float(training_top_p),
            training_top_k=training_top_k,
            training_max_new_tokens=training_max_new_tokens,
            training_stop_token=str(training_stop_token),
            audit_backend=audit_backend,  # type: ignore[arg-type]
            audit_dtype=audit_dtype,  # type: ignore[arg-type]
            audit_attention_backend=audit_attention_backend,  # type: ignore[arg-type]
            audit_batch_size=audit_batch_size,
            collision_policy=run.get("collision_policy", "fail"),  # type: ignore[arg-type]
            resource_training_gpu=resource_training_gpu,
            resource_audit_gpu=resource_audit_gpu,
            resource_retry_policy=resource.get("retry_policy", "none"),  # type: ignore[arg-type]
            resource_promotion=resource.get("promotion", False),  # type: ignore[arg-type]
            source_checkpoint_path=source.get("checkpoint_path"),
            base_model_path=source.get("base_model_path"),
            adapter_path=source.get("adapter_path"),
            special_embedding_path=source.get("special_embedding_path"),
            source_adapter_sha256=source.get("adapter_sha256"),
            special_embedding_sha256=source.get("special_embedding_sha256"),
            manifest_sha256=value.get("manifest_sha256"),
            parser=str(audit.get("parser", "compact_object_box_closed_only")),
            matcher_algorithm=str(
                audit.get("matcher_algorithm", "cardinality_first_max_total_iou")
            ),
            matcher_duplicate_iou=float(audit.get("duplicate_iou_threshold", 0.95)),
            matcher_owner_iou=float(audit.get("owner_iou_threshold", 0.50)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EntryConfig:
        try:
            import yaml

            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"entry config is unavailable: {path}") from error
        if not isinstance(raw, Mapping):
            raise ValueError("entry config must contain an object")
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "unit_id": self.unit_id,
            "image_id": self.image_id,
            "seed_groups": [list(group) for group in self.seed_groups],
            "training": {
                "repetition_penalty": self.training_repetition_penalty,
                "temperature": self.training_temperature,
                "top_p": self.training_top_p,
                "top_k": self.training_top_k,
                "max_new_tokens": self.training_max_new_tokens,
                "stop_token": self.training_stop_token,
            },
            "surface": {
                "dtype": self.dtype,
                "attention_backend": self.attention_backend,
                "use_cache": self.use_cache,
            },
            "optimizer": {
                "name": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "betas": list(self.optimizer_betas),
                "epsilon": self.optimizer_epsilon,
                "weight_decay": self.optimizer_weight_decay,
            },
            "audit": {
                "backend": self.audit_backend,
                "dtype": self.audit_dtype,
                "attention_backend": self.audit_attention_backend,
                "batch_size": self.audit_batch_size,
                "repetition_penalties": list(self.audit_repetition_penalties),
                "parser": self.parser,
                "matcher_algorithm": self.matcher_algorithm,
                "duplicate_iou_threshold": self.matcher_duplicate_iou,
                "owner_iou_threshold": self.matcher_owner_iou,
            },
            "run": {
                "output_root": self.output_root,
                "collision_policy": self.collision_policy,
            },
            "resource": {
                "training_gpu": self.resource_training_gpu,
                "audit_gpu": self.resource_audit_gpu,
                "retry_policy": self.resource_retry_policy,
                "promotion": self.resource_promotion,
            },
            "source": {
                "checkpoint_path": self.source_checkpoint_path,
                "base_model_path": self.base_model_path,
                "adapter_path": self.adapter_path,
                "special_embedding_path": self.special_embedding_path,
                "adapter_sha256": self.source_adapter_sha256,
                "special_embedding_sha256": self.special_embedding_sha256,
            },
            "manifest_sha256": self.manifest_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class GPUResource:
    index: int
    total_memory_bytes: int
    free_memory_bytes: int
    suitable: bool = True

    def __post_init__(self) -> None:
        _nonnegative_int(self.index, field="GPU index")
        total = _nonnegative_int(self.total_memory_bytes, field="GPU total memory")
        free = _nonnegative_int(self.free_memory_bytes, field="GPU free memory")
        if total <= 0 or free > total:
            raise ValueError("GPU memory bounds are invalid")
        if not isinstance(self.suitable, bool):
            raise ValueError("GPU suitability must be a boolean value receipt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "total_memory_bytes": self.total_memory_bytes,
            "free_memory_bytes": self.free_memory_bytes,
            "suitable": self.suitable,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GPUResource:
        return cls(
            index=value["index"],
            total_memory_bytes=value["total_memory_bytes"],
            free_memory_bytes=value["free_memory_bytes"],
            suitable=value.get("suitable", True),
        )


@dataclass(frozen=True)
class DualGPUResourceReceipt:
    cards: tuple[GPUResource, ...]
    training_gpu: int = 0
    audit_gpu: int = 1

    def __post_init__(self) -> None:
        cards = tuple(self.cards)
        if self.training_gpu != 0 or self.audit_gpu != 1:
            raise ValueError("GPU roles must bind GPU 0 training and GPU 1 audit")
        if self.training_gpu == self.audit_gpu:
            raise ValueError("training and audit GPUs must be distinct")
        if not cards or len({card.index for card in cards}) != len(cards):
            raise ValueError("GPU resource receipt must contain distinct cards")
        by_index = {card.index: card for card in cards}
        for role, index in (("training", self.training_gpu), ("audit", self.audit_gpu)):
            card = by_index.get(index)
            if card is None:
                raise ValueError(f"missing {role} GPU {index}")
            if card.suitable is not True or card.free_memory_bytes <= 0:
                raise ValueError(f"{role} GPU {index} is not suitable")

    @property
    def content_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": RESOURCE_SCHEMA_VERSION,
                "cards": [
                    {
                        "index": card.index,
                        "total_memory_bytes": card.total_memory_bytes,
                        "free_memory_bytes": card.free_memory_bytes,
                        "suitable": card.suitable,
                    }
                    for card in self.cards
                ],
                "training_gpu": self.training_gpu,
                "audit_gpu": self.audit_gpu,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "cards": [card.to_dict() for card in self.cards],
            "training_gpu": self.training_gpu,
            "audit_gpu": self.audit_gpu,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DualGPUResourceReceipt:
        if value.get("schema_version") != RESOURCE_SCHEMA_VERSION:
            raise ValueError("dual-GPU resource schema differs")
        receipt = cls(
            cards=tuple(GPUResource.from_dict(item) for item in value["cards"]),
            training_gpu=value.get("training_gpu", 0),
            audit_gpu=value.get("audit_gpu", 1),
        )
        if value.get("content_sha256") != receipt.content_sha256:
            raise ValueError("dual-GPU resource content hash differs")
        return receipt


def validate_dual_gpu_resources(
    cards: Sequence[GPUResource] | DualGPUResourceReceipt,
) -> DualGPUResourceReceipt:
    """Validate the exact two-role GPU assignment without touching CUDA."""

    if isinstance(cards, DualGPUResourceReceipt):
        return cards
    values = tuple(cards)
    if len(values) < 2:
        raise ValueError("dual-GPU entry requires two distinct suitable cards")
    if any(not isinstance(card, GPUResource) for card in values):
        raise ValueError("GPU resources must be value receipts")
    try:
        return DualGPUResourceReceipt(values)
    except ValueError as error:
        raise ValueError(str(error)) from error


def observe_production_gpu_resources() -> tuple[GPUResource, GPUResource]:
    """Read the two physical CUDA cards before any production model assembly."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("production execution requires physical CUDA GPUs 0 and 1")
    cards: list[GPUResource] = []
    for index in (0, 1):
        free, total = torch.cuda.mem_get_info(index)
        cards.append(
            GPUResource(
                index=index,
                total_memory_bytes=int(total),
                free_memory_bytes=int(free),
                suitable=True,
            )
        )
    return cards[0], cards[1]


def admit_production_gpu_resources(
    cards: Sequence[GPUResource],
) -> DualGPUResourceReceipt:
    """Fail closed on role, availability, or the declared live memory floor."""

    values = tuple(cards)
    if tuple(card.index for card in values) != (0, 1):
        raise RuntimeError("production resources must be exact physical GPUs 0 and 1")
    receipt = validate_dual_gpu_resources(values)
    by_index = {card.index: card for card in receipt.cards}
    for index, minimum_free in (
        (0, MIN_TRAINING_GPU_FREE_BYTES),
        (1, MIN_AUDIT_GPU_FREE_BYTES),
    ):
        card = by_index[index]
        if card.total_memory_bytes < MIN_PRODUCTION_GPU_TOTAL_BYTES:
            raise RuntimeError(f"GPU {index} is not an 80-GB-class production card")
        if card.free_memory_bytes < minimum_free:
            raise RuntimeError(
                f"GPU {index} has insufficient free memory for its production role"
            )
    return receipt


@dataclass(frozen=True)
class ExecutionAuthority:
    user_model_gpu_authority: bool
    model_authority_receipt: str = "explicit_user_model_authority"
    gpu_authority_receipt: str = "explicit_user_gpu_authority"

    def require(self) -> None:
        if self.user_model_gpu_authority is not True:
            raise PermissionError(
                "live execution requires --user-model-gpu-authority and explicit model/GPU authority"
            )
        if not self.model_authority_receipt or not self.gpu_authority_receipt:
            raise PermissionError("model and GPU authority receipts must be nonempty")

    @property
    def content_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": "human13_execution_authority.v1",
                "user_model_gpu_authority": self.user_model_gpu_authority,
                "model_authority_receipt": self.model_authority_receipt,
                "gpu_authority_receipt": self.gpu_authority_receipt,
            }
        )


@dataclass(frozen=True)
class OutputRootReceipt:
    path: str
    existed_before: Literal[False]
    immutable: Literal[True] = True

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("output root receipt must use an absolute path")
        if self.existed_before is not False or self.immutable is not True:
            raise ValueError("output root must be newly confirmed absent and immutable")

    @property
    def content_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": "human13_output_root_absence.v1",
                "path": self.path,
                "existed_before": self.existed_before,
                "immutable": self.immutable,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_output_root_absence.v1",
            "path": self.path,
            "existed_before": self.existed_before,
            "immutable": self.immutable,
            "content_sha256": self.content_sha256,
        }


def confirm_absent_output_root(path: str | Path) -> OutputRootReceipt:
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise ValueError("output root must be an absolute path")
    resolved = target.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"output root must be newly absent: {resolved}")
    return OutputRootReceipt(str(resolved), False)


@dataclass(frozen=True)
class SourceAssemblyReceipt:
    source_plan_sha256: str
    training_gpu: int
    audit_gpu: int
    training_surface: str
    audit_surface: str
    checkpoint_path: str | None = None
    base_model_path: str | None = None
    adapter_path: str | None = None
    special_embedding_path: str | None = None
    adapter_sha256: str | None = None
    special_embedding_sha256: str | None = None
    checkpoint_sha256: str | None = None
    tokenizer_sha256: str | None = None
    prompt_policy_fingerprint: str | None = None
    panel_sha256: str | None = None
    image_sha256: str | None = None
    manifest_sha256: str | None = None
    source_validation_sha256: str | None = None
    assembly_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _digest(self.source_plan_sha256, field="source_plan_sha256")
        if self.training_gpu != 0 or self.audit_gpu != 1:
            raise ValueError(
                "source assembly roles must bind GPU 0 training and GPU 1 audit"
            )
        if self.training_surface != "bf16/flash_attention_2":
            raise ValueError("training assembly surface differs from BF16/FA2")
        if self.audit_surface != "fp32/sdpa/batch1":
            raise ValueError(
                "audit assembly surface differs from HF fp32/SDPA batch-one"
            )
        for field in (
            "adapter_sha256",
            "special_embedding_sha256",
            "checkpoint_sha256",
            "tokenizer_sha256",
            "prompt_policy_fingerprint",
            "panel_sha256",
            "image_sha256",
            "manifest_sha256",
            "source_validation_sha256",
            "assembly_receipt_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                _digest(value, field=field)
        for field in (
            "checkpoint_path",
            "base_model_path",
            "adapter_path",
            "special_embedding_path",
        ):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be a non-empty path")

    @property
    def content_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": SOURCE_ASSEMBLY_SCHEMA_VERSION,
                "source_plan_sha256": self.source_plan_sha256,
                "training_gpu": self.training_gpu,
                "audit_gpu": self.audit_gpu,
                "training_surface": self.training_surface,
                "audit_surface": self.audit_surface,
                "checkpoint_path": self.checkpoint_path,
                "base_model_path": self.base_model_path,
                "adapter_path": self.adapter_path,
                "special_embedding_path": self.special_embedding_path,
                "adapter_sha256": self.adapter_sha256,
                "special_embedding_sha256": self.special_embedding_sha256,
                "checkpoint_sha256": self.checkpoint_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "prompt_policy_fingerprint": self.prompt_policy_fingerprint,
                "panel_sha256": self.panel_sha256,
                "image_sha256": self.image_sha256,
                "manifest_sha256": self.manifest_sha256,
                "source_validation_sha256": self.source_validation_sha256,
                "assembly_receipt_sha256": self.assembly_receipt_sha256,
            }
        )

    @property
    def source_validation_content_sha256(self) -> str:
        return _sha256(
            {
                "checkpoint_path": self.checkpoint_path,
                "base_model_path": self.base_model_path,
                "adapter_path": self.adapter_path,
                "special_embedding_path": self.special_embedding_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                "adapter_sha256": self.adapter_sha256,
                "special_embedding_sha256": self.special_embedding_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "prompt_policy_fingerprint": self.prompt_policy_fingerprint,
                "panel_sha256": self.panel_sha256,
                "image_sha256": self.image_sha256,
                "manifest_sha256": self.manifest_sha256,
            }
        )

    @property
    def assembly_receipt_content_sha256(self) -> str:
        return _sha256(
            {
                "source_plan_sha256": self.source_plan_sha256,
                "training_gpu": self.training_gpu,
                "audit_gpu": self.audit_gpu,
                "training_surface": self.training_surface,
                "audit_surface": self.audit_surface,
                "source_validation_sha256": self.source_validation_content_sha256,
            }
        )


@dataclass(frozen=True)
class ManifestImageContext:
    """Typed bridge carrying an ImageRecord and its parent manifest binding."""

    image: Any
    binding: Any


def _split_manifest_context(
    value: Any,
    manifest_binding: Any | None,
) -> tuple[Any, Any | None]:
    if isinstance(value, ManifestImageContext):
        if manifest_binding is not None and manifest_binding is not value.binding:
            raise ValueError("manifest image context and explicit binding differ")
        return value.image, value.binding
    return value, manifest_binding if manifest_binding is not None else getattr(
        value, "binding", None
    )


@dataclass(frozen=True)
class ResourceReceipt:
    resources: DualGPUResourceReceipt
    output_root: OutputRootReceipt | None
    phase_count: int
    retry_count: int
    promoted_checkpoint: Literal[False]
    reservation_identity: RunReservationIdentity | None = None
    sampled_request_count: int = 16
    sampled_group_count: int = 4
    sample_forward_count: int = 2048
    replay_forward_count: int = 2048
    source_owner_forward_count: int = 0
    total_forward_count: int = 4096
    no_cache_forward_count: int = 4096
    backward_count: int = 1
    token_cap: int = 512
    peak_host_rss_bytes: int | None = None
    cuda_peak_allocated_bytes: int | None = None
    cuda_peak_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resources, DualGPUResourceReceipt):
            raise ValueError("resource receipt requires dual-GPU evidence")
        _nonnegative_int(self.phase_count, field="phase_count")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise ValueError("retry_count must be a nonnegative integer")
        if self.promoted_checkpoint is not False:
            raise ValueError("proposal checkpoint promotion is forbidden")
        if self.reservation_identity is not None and not isinstance(
            self.reservation_identity, RunReservationIdentity
        ):
            raise ValueError("resource reservation identity is malformed")
        if self.reservation_identity is not None and (
            self.output_root is None
            or self.output_root.path != self.reservation_identity.output_root
        ):
            raise ValueError(
                "reservation identity must match the resource output root"
            )
        for field in (
            "sampled_request_count",
            "sampled_group_count",
            "sample_forward_count",
            "replay_forward_count",
            "source_owner_forward_count",
            "total_forward_count",
            "no_cache_forward_count",
            "backward_count",
            "token_cap",
        ):
            _nonnegative_int(getattr(self, field), field=field)
        if (
            self.sampled_request_count > 16
            or self.sampled_group_count > 4
            or self.backward_count > 1
            or self.token_cap != 512
        ):
            raise ValueError("resource receipt exceeds the frozen K16 one-update budget")
        if self.sampled_request_count != self.sampled_group_count * 4:
            raise ValueError("sampled request/group counts differ")
        if self.sample_forward_count > 4 * self.token_cap:
            raise ValueError("sample forward count exceeds the frozen K16 bound")
        if self.replay_forward_count > self.sample_forward_count:
            raise ValueError("replay forward count exceeds sampled step coverage")
        if self.total_forward_count != (
            self.sample_forward_count
            + self.replay_forward_count
            + self.source_owner_forward_count
        ):
            raise ValueError("resource forward-count total differs")
        if self.no_cache_forward_count != self.total_forward_count:
            raise ValueError("every resource forward must remain no-cache")
        for field in (
            "peak_host_rss_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            value = getattr(self, field)
            if value is not None:
                _nonnegative_int(value, field=field)

    @property
    def training_gpu(self) -> int:
        return self.resources.training_gpu

    @property
    def audit_gpu(self) -> int:
        return self.resources.audit_gpu

    @property
    def content_sha256(self) -> str:
        return _sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        value = {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "resources": self.resources.to_dict(),
            "output_root": None
            if self.output_root is None
            else self.output_root.to_dict(),
            "phase_count": self.phase_count,
            "retry_count": self.retry_count,
            "promoted_checkpoint": self.promoted_checkpoint,
            "sampled_request_count": self.sampled_request_count,
            "sampled_group_count": self.sampled_group_count,
            "sample_forward_count": self.sample_forward_count,
            "replay_forward_count": self.replay_forward_count,
            "source_owner_forward_count": self.source_owner_forward_count,
            "total_forward_count": self.total_forward_count,
            "no_cache_forward_count": self.no_cache_forward_count,
            "backward_count": self.backward_count,
            "token_cap": self.token_cap,
            "peak_host_rss_bytes": self.peak_host_rss_bytes,
            "cuda_peak_allocated_bytes": self.cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": self.cuda_peak_reserved_bytes,
        }
        if self.reservation_identity is not None:
            value["reservation_identity"] = self.reservation_identity.to_dict()
        return value

    def to_dict(self) -> dict[str, Any]:
        value = self._payload()
        return value | {"content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceReceipt:
        if value.get("schema_version") != RESOURCE_SCHEMA_VERSION:
            raise ValueError("resource receipt schema differs")
        root_value = value.get("output_root")
        root = None
        if root_value is not None:
            if not isinstance(root_value, Mapping):
                raise ValueError("resource output_root must be an object")
            root = OutputRootReceipt(
                path=root_value["path"],
                existed_before=root_value["existed_before"],
                immutable=root_value.get("immutable", True),
            )
            if root_value.get("content_sha256") != root.content_sha256:
                raise ValueError("resource output root content hash differs")
        resources = DualGPUResourceReceipt.from_dict(value["resources"])
        identity_value = value.get("reservation_identity")
        identity = None
        if identity_value is not None:
            identity = RunReservationIdentity.from_dict(
                _as_mapping(identity_value, field="reservation_identity")
            )
        receipt = cls(
            resources=resources,
            output_root=root,
            phase_count=value["phase_count"],
            retry_count=value["retry_count"],
            promoted_checkpoint=value["promoted_checkpoint"],
            reservation_identity=identity,
            sampled_request_count=value.get("sampled_request_count", 16),
            sampled_group_count=value.get("sampled_group_count", 4),
            sample_forward_count=value.get("sample_forward_count", 2048),
            replay_forward_count=value.get("replay_forward_count", 2048),
            source_owner_forward_count=value.get("source_owner_forward_count", 0),
            total_forward_count=value.get("total_forward_count", 4096),
            no_cache_forward_count=value.get("no_cache_forward_count", 4096),
            backward_count=value.get("backward_count", 1),
            token_cap=value.get("token_cap", 512),
            peak_host_rss_bytes=value.get("peak_host_rss_bytes"),
            cuda_peak_allocated_bytes=value.get("cuda_peak_allocated_bytes"),
            cuda_peak_reserved_bytes=value.get("cuda_peak_reserved_bytes"),
        )
        if value.get("content_sha256") != receipt.content_sha256:
            raise ValueError("resource receipt content hash differs")
        return receipt


@dataclass(frozen=True)
class ActionAttemptReceipt:
    """Append-only physical-boundary attempt evidence.

    ``model_actions`` remains the admitted-session counter. This receipt is
    separate so a loader that touched a checkpoint and then failed cannot be
    mistaken for an admitted model/session.
    """

    schema_version: Literal["human13_action_attempt.v1"]
    boundary: Literal["training_open", "audit_open", "audit_evaluator"]
    resource_role: Literal["gpu0_training", "gpu1_audit"]
    attempted_count: int
    completed_count: int
    failed_count: int
    session_admitted: bool
    canonical_device_identity_sha256: str | None = None
    exception_type: str | None = None
    exception_message_sha256: str | None = None
    stdout_sha256: str | None = None
    stdout_tail: str | None = None
    stderr_sha256: str | None = None
    stderr_tail: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "human13_action_attempt.v1":
            raise ValueError("action attempt schema differs")
        if self.boundary not in {"training_open", "audit_open", "audit_evaluator"}:
            raise ValueError("action attempt boundary is unsupported")
        if self.resource_role not in {"gpu0_training", "gpu1_audit"}:
            raise ValueError("action attempt resource role is unsupported")
        for field in ("attempted_count", "completed_count", "failed_count"):
            _nonnegative_int(getattr(self, field), field=field)
        if self.attempted_count != 1 or self.completed_count + self.failed_count != 1:
            raise ValueError("one action attempt must be exactly completed or failed")
        expected_admitted = self.boundary in {"training_open", "audit_open"} and (
            self.completed_count == 1
        )
        if self.session_admitted is not expected_admitted:
            raise ValueError("admitted-session status differs from attempt outcome")
        for field in (
            "canonical_device_identity_sha256",
            "exception_message_sha256",
            "stdout_sha256",
            "stderr_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                _digest(value, field=field)
        if self.failed_count and (
            not self.exception_type or self.exception_message_sha256 is None
        ):
            raise ValueError("failed action attempt lacks exception evidence")
        for field in ("stdout_tail", "stderr_tail"):
            value = getattr(self, field)
            if value is not None and len(value) > 512:
                raise ValueError(f"{field} exceeds bounded provenance tail")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "boundary": self.boundary,
            "resource_role": self.resource_role,
            "attempted_count": self.attempted_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "session_admitted": self.session_admitted,
            "canonical_device_identity_sha256": self.canonical_device_identity_sha256,
            "exception_type": self.exception_type,
            "exception_message_sha256": self.exception_message_sha256,
            "stdout_sha256": self.stdout_sha256,
            "stdout_tail": self.stdout_tail,
            "stderr_sha256": self.stderr_sha256,
            "stderr_tail": self.stderr_tail,
        }

    @property
    def content_sha256(self) -> str:
        return _sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return self._payload() | {"content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionAttemptReceipt:
        if value.get("schema_version") != "human13_action_attempt.v1":
            raise ValueError("action attempt schema differs")
        result = cls(
            schema_version=value["schema_version"],
            boundary=value["boundary"],
            resource_role=value["resource_role"],
            attempted_count=value["attempted_count"],
            completed_count=value["completed_count"],
            failed_count=value["failed_count"],
            session_admitted=value["session_admitted"],
            canonical_device_identity_sha256=value.get(
                "canonical_device_identity_sha256"
            ),
            exception_type=value.get("exception_type"),
            exception_message_sha256=value.get("exception_message_sha256"),
            stdout_sha256=value.get("stdout_sha256"),
            stdout_tail=value.get("stdout_tail"),
            stderr_sha256=value.get("stderr_sha256"),
            stderr_tail=value.get("stderr_tail"),
        )
        if value.get("content_sha256") != result.content_sha256:
            raise ValueError("action attempt content hash differs")
        return result


@dataclass(frozen=True)
class AuditAnalysis:
    repetition_penalty: float
    source_owner_ids: tuple[str, ...]
    proposal_owner_ids: tuple[str, ...]
    h_gain_owner_ids: tuple[str, ...]
    g_loss_owner_ids: tuple[str, ...]
    m_gain_owner_ids: tuple[str, ...]
    net_unique_delta: int
    source_duplicate_rows: int
    proposal_duplicate_rows: int
    source_unmatched_rows: int
    proposal_unmatched_rows: int
    source_malformed_rows: int
    proposal_malformed_rows: int
    source_cap_stops: int
    proposal_cap_stops: int
    row_count: int
    token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repetition_penalty",
            _rp(self.repetition_penalty, field="repetition_penalty"),
        )
        for field in (
            "source_owner_ids",
            "proposal_owner_ids",
            "h_gain_owner_ids",
            "g_loss_owner_ids",
            "m_gain_owner_ids",
        ):
            values = tuple(str(item) for item in getattr(self, field))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be sorted and unique")
            object.__setattr__(self, field, values)
        for field in (
            "net_unique_delta",
            "source_duplicate_rows",
            "proposal_duplicate_rows",
            "source_unmatched_rows",
            "proposal_unmatched_rows",
            "source_malformed_rows",
            "proposal_malformed_rows",
            "source_cap_stops",
            "proposal_cap_stops",
            "row_count",
            "token_count",
        ):
            value = getattr(self, field)
            if field == "net_unique_delta":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{field} must be an integer")
            else:
                _nonnegative_int(value, field=field)

    @property
    def duplicate_delta(self) -> int:
        return self.proposal_duplicate_rows - self.source_duplicate_rows

    @property
    def g_owner_ids(self) -> tuple[str, ...]:
        """Proposal owners retained under this audit (compatibility alias)."""

        return self.proposal_owner_ids

    @property
    def h_gain(self) -> tuple[str, ...]:
        return self.h_gain_owner_ids

    @property
    def g_loss(self) -> tuple[str, ...]:
        return self.g_loss_owner_ids

    @property
    def m_gains(self) -> tuple[str, ...]:
        return self.m_gain_owner_ids

    @property
    def duplicate_rows(self) -> int:
        return self.proposal_duplicate_rows

    @property
    def unmatched_rows(self) -> int:
        return self.proposal_unmatched_rows

    @property
    def malformed_rows(self) -> int:
        return self.proposal_malformed_rows

    @property
    def cap_stops(self) -> int:
        return self.proposal_cap_stops

    @property
    def rows(self) -> int:
        return self.row_count

    @property
    def tokens(self) -> int:
        return self.token_count

    @property
    def malformed_delta(self) -> int:
        return self.proposal_malformed_rows - self.source_malformed_rows

    @property
    def cap_delta(self) -> int:
        return self.proposal_cap_stops - self.source_cap_stops

    @property
    def burdens_nonincreasing(self) -> bool:
        return (
            self.duplicate_delta <= 0
            and self.malformed_delta <= 0
            and self.cap_delta <= 0
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "repetition_penalty": self.repetition_penalty,
            "source_owner_ids": list(self.source_owner_ids),
            "proposal_owner_ids": list(self.proposal_owner_ids),
            "h_gain_owner_ids": list(self.h_gain_owner_ids),
            "g_loss_owner_ids": list(self.g_loss_owner_ids),
            "m_gain_owner_ids": list(self.m_gain_owner_ids),
            "net_unique_delta": self.net_unique_delta,
            "source_duplicate_rows": self.source_duplicate_rows,
            "proposal_duplicate_rows": self.proposal_duplicate_rows,
            "source_unmatched_rows": self.source_unmatched_rows,
            "proposal_unmatched_rows": self.proposal_unmatched_rows,
            "source_malformed_rows": self.source_malformed_rows,
            "proposal_malformed_rows": self.proposal_malformed_rows,
            "source_cap_stops": self.source_cap_stops,
            "proposal_cap_stops": self.proposal_cap_stops,
            "row_count": self.row_count,
            "token_count": self.token_count,
        }
        return value


@dataclass(frozen=True)
class AuditPairAnalysis:
    by_repetition_penalty: Mapping[float, AuditAnalysis]

    def __post_init__(self) -> None:
        values = {
            float(key): value for key, value in self.by_repetition_penalty.items()
        }
        if set(values) != set(AUDIT_REPETITION_PENALTIES):
            raise ValueError("audit analysis must cover RP 1.0 and RP 1.10 exactly")
        if any(not isinstance(value, AuditAnalysis) for value in values.values()):
            raise ValueError("audit analysis values must be typed AuditAnalysis")
        object.__setattr__(self, "by_repetition_penalty", MappingProxyType(values))

    @property
    def content_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": "human13_dual_rp_audit_analysis.v1",
                "audits": [
                    self.by_repetition_penalty[rp].to_dict()
                    for rp in AUDIT_REPETITION_PENALTIES
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_dual_rp_audit_analysis.v1",
            "audits": [
                self.by_repetition_penalty[rp].to_dict()
                for rp in AUDIT_REPETITION_PENALTIES
            ],
            "content_sha256": self.content_sha256,
        }


def _image_matcher(
    image: Any, manifest_binding: Any | None = None
) -> tuple[float, float]:
    binding = (
        manifest_binding
        if manifest_binding is not None
        else getattr(image, "binding", None)
    )
    matcher = getattr(binding, "matcher", None)
    if matcher is None:
        raise ValueError("manifest image context lacks the frozen matcher identity")
    if (
        getattr(matcher, "algorithm", None) != "cardinality_first_max_total_iou"
        or getattr(matcher, "same_category", None) is not True
        or getattr(matcher, "duplicate_comparison", None) != "strictly_greater"
        or getattr(matcher, "target_row_rule", None)
        != "max_owner_iou_then_seed_then_row_index"
    ):
        raise ValueError(
            "manifest matcher identity differs from the frozen canonical matcher"
        )
    duplicate = getattr(matcher, "duplicate_iou_threshold", 0.95)
    owner = getattr(matcher, "owner_iou_threshold", 0.50)
    if float(duplicate) != 0.95 or float(owner) != 0.50:
        raise ValueError(
            "manifest matcher thresholds differ from the frozen canonical matcher"
        )
    return 0.95, 0.50


def _manifest_surface_identity(
    image: Any, manifest_binding: Any | None = None
) -> dict[str, str]:
    binding = (
        manifest_binding
        if manifest_binding is not None
        else getattr(image, "binding", None)
    )
    panel = getattr(binding, "panel", None)
    surface = getattr(binding, "surface", None)
    binding_values = {
        "panel_sha256": getattr(panel, "panel_sha256", None),
        "tokenizer_sha256": getattr(surface, "tokenizer_sha256", None),
        "prompt_policy_fingerprint": getattr(
            surface, "prompt_policy_fingerprint", None
        ),
    }
    has_bound_surface = any(isinstance(value, str) for value in binding_values.values())
    values: dict[str, Any] = {}
    for field, bound_value in binding_values.items():
        local_value = getattr(image, field, None)
        if manifest_binding is not None and has_bound_surface:
            if not isinstance(bound_value, str):
                raise ValueError(f"manifest binding {field} is required")
            if local_value is not None:
                if not isinstance(local_value, str) or _digest(
                    local_value, field=f"manifest_image.{field}"
                ) != _digest(bound_value, field=f"manifest_binding.{field}"):
                    raise ValueError(
                        f"manifest image {field} conflicts with manifest binding"
                    )
            values[field] = bound_value
        else:
            values[field] = local_value if local_value is not None else bound_value
    result: dict[str, str] = {}
    for field, value in values.items():
        if not isinstance(value, str):
            raise ValueError(f"manifest image {field} is required")
        result[field] = _digest(value, field=f"manifest_image.{field}")
    return result


def _manifest_image_identity(
    config: EntryConfig, image: Any, manifest_binding: Any | None = None
) -> dict[str, Any]:
    if image is None:
        raise ValueError(
            "one-image live entry requires the admitted image manifest record"
        )
    image_id = getattr(image, "image_id", None)
    if (
        isinstance(image_id, bool)
        or not isinstance(image_id, int)
        or image_id != config.image_id
    ):
        raise ValueError("manifest image_id does not match the sealed entry config")
    if not config.manifest_sha256:
        raise ValueError("manifest_sha256 is required before live execution")
    _digest(config.manifest_sha256, field="manifest_sha256")
    for field in (
        "source_checkpoint_path",
        "base_model_path",
        "adapter_path",
        "special_embedding_path",
        "source_adapter_sha256",
        "special_embedding_sha256",
    ):
        value = getattr(config, field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} is required before live execution")
    _digest(config.source_adapter_sha256, field="source_adapter_sha256")
    _digest(config.special_embedding_sha256, field="special_embedding_sha256")
    identity: dict[str, Any] = {"image_id": image_id}
    identity.update(_manifest_surface_identity(image, manifest_binding))
    for field in ("panel_row_sha256", "image_sha256"):
        value = getattr(image, field, None)
        if not isinstance(value, str):
            raise ValueError(f"manifest image {field} is required")
        identity[field] = _digest(value, field=f"manifest_image.{field}")
    for field in ("g_owner_ids", "h_owner_ids", "m_owner_ids"):
        values = getattr(image, field, None)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"manifest image {field} is required")
        if any(not isinstance(item, str) or not item for item in values):
            raise ValueError(
                f"manifest image {field} owner IDs must be nonempty strings"
            )
        normalized = tuple(values)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"manifest image {field} must be unique")
        identity[field] = normalized
    strata = (
        identity["g_owner_ids"],
        identity["h_owner_ids"],
        identity["m_owner_ids"],
    )
    if len(set().union(*strata)) != sum(len(values) for values in strata):
        raise ValueError("manifest image G/H/M owner strata must be disjoint")
    owners = getattr(image, "owners", ())
    if (
        isinstance(owners, (list, tuple))
        and owners
        and all(isinstance(getattr(owner, "stratum", None), str) for owner in owners)
    ):
        owner_strata: dict[str, set[str]] = {"G": set(), "H": set(), "M": set()}
        for owner in owners:
            stratum = str(getattr(owner, "stratum")).upper()
            if stratum not in owner_strata:
                raise ValueError("manifest owner stratum must be G, H, or M")
            owner_id = str(getattr(owner, "owner_id", ""))
            if not owner_id:
                raise ValueError("manifest owner_id must be nonempty")
            owner_strata[stratum].add(owner_id)
        for stratum, field in (
            ("G", "g_owner_ids"),
            ("H", "h_owner_ids"),
            ("M", "m_owner_ids"),
        ):
            if owner_strata[stratum] != set(identity[field]):
                raise ValueError(f"manifest owner strata do not match {field}")
    return identity


def _validate_audit_provenance(
    image: Any,
    output: Mapping[str, Any],
    *,
    expected_manifest_sha256: str | None = None,
    manifest_binding: Any | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_path: str | None = None,
    expected_parser: str = _CANONICAL_AUDIT_PARSER,
    expected_arm_id: str | None = None,
    expected_source_identity: Mapping[str, Any] | None = None,
) -> None:
    image_id = getattr(image, "image_id", None)
    if output.get("image_id") != image_id:
        raise ValueError("audit image_id does not match the manifest image")
    provenance = output.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("audit provenance must be an object")
    if provenance.get("image_id") != image_id:
        raise ValueError("audit provenance.image_id does not match the manifest image")
    parser = output.get("parser")
    provenance_parser = provenance.get("parser")
    if not isinstance(parser, str) or not parser:
        raise ValueError("audit parser is required")
    if not isinstance(provenance_parser, str) or not provenance_parser:
        raise ValueError("audit provenance.parser is required")
    if parser != provenance_parser:
        raise ValueError("audit parser differs between output and provenance")
    if parser != expected_parser:
        raise ValueError("audit parser differs from the canonical parser")
    parser_status = output.get("parser_status")
    if parser_status not in _CANONICAL_PARSER_STATUSES:
        raise ValueError("audit parser_status is outside the canonical parser")
    arm_id = output.get("arm_id")
    provenance_arm_id = provenance.get("arm_id")
    if not isinstance(arm_id, str) or not arm_id:
        raise ValueError("audit arm_id is required")
    if not isinstance(provenance_arm_id, str) or not provenance_arm_id:
        raise ValueError("audit provenance.arm_id is required")
    if arm_id != provenance_arm_id:
        raise ValueError("audit arm_id differs between output and provenance")
    if expected_arm_id is not None and arm_id != expected_arm_id:
        raise ValueError("audit arm_id differs from the expected Source/proposal arm")
    milestone = output.get("milestone")
    provenance_milestone = provenance.get("milestone")
    if (
        isinstance(milestone, bool)
        or not isinstance(milestone, int)
        or milestone < 0
        or isinstance(provenance_milestone, bool)
        or not isinstance(provenance_milestone, int)
        or provenance_milestone < 0
    ):
        raise ValueError("audit milestone must be a nonnegative integer")
    if milestone != provenance_milestone:
        raise ValueError("audit milestone differs between output and provenance")
    trajectory_id = output.get("trajectory_id")
    provenance_trajectory_id = provenance.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("audit trajectory_id is required")
    if not isinstance(provenance_trajectory_id, str) or not provenance_trajectory_id:
        raise ValueError("audit provenance.trajectory_id is required")
    if trajectory_id != provenance_trajectory_id:
        raise ValueError("audit trajectory_id differs between output and provenance")
    source_trajectory_id = provenance.get("source_trajectory_id")
    if not isinstance(source_trajectory_id, str) or not source_trajectory_id:
        raise ValueError("audit provenance.source_trajectory_id is required")
    trajectories = getattr(image, "trajectories", ())
    if isinstance(trajectories, (list, tuple)) and trajectories:
        expected_source_trajectory_id = getattr(trajectories[0], "trajectory_id", None)
        if (
            isinstance(expected_source_trajectory_id, str)
            and source_trajectory_id != expected_source_trajectory_id
        ):
            raise ValueError("audit source trajectory differs from the manifest Source")
    for field in ("run_id", "run_root"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"audit provenance.{field} is required")
    metadata_fields = (
        "decode_mode",
        "backend",
        "physical_batch_size",
        "do_sample",
    )
    for field in metadata_fields:
        if (
            field in output
            and field in provenance
            and output[field] != provenance[field]
        ):
            raise ValueError(f"audit {field} differs between output and provenance")
    metadata = {
        field: output[field] if field in output else provenance.get(field)
        for field in metadata_fields
    }
    if metadata != {
        "decode_mode": "original_prompt_clean_greedy",
        "backend": "hf",
        "physical_batch_size": 1,
        "do_sample": False,
    }:
        raise ValueError(
            "audit surface must be original-prompt clean HF greedy batch-one"
        )
    if "repetition_penalty" in provenance:
        if _rp(
            provenance["repetition_penalty"],
            field="audit provenance.repetition_penalty",
        ) != _rp(output.get("repetition_penalty"), field="audit repetition penalty"):
            raise ValueError(
                "audit repetition penalty differs between output and provenance"
            )
    if (
        "checkpoint_payload_sha256" in output
        and "checkpoint_payload_sha256" in provenance
        and output["checkpoint_payload_sha256"]
        != provenance["checkpoint_payload_sha256"]
    ):
        raise ValueError(
            "audit checkpoint_payload_sha256 differs between output and provenance"
        )
    checkpoint_payload_sha256 = (
        output["checkpoint_payload_sha256"]
        if "checkpoint_payload_sha256" in output
        else provenance.get("checkpoint_payload_sha256")
    )
    if not isinstance(checkpoint_payload_sha256, str):
        raise ValueError("audit provenance.checkpoint_payload_sha256 is required")
    _digest(
        checkpoint_payload_sha256, field="audit provenance.checkpoint_payload_sha256"
    )
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_payload_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("audit checkpoint identity differs from Source assembly")
    source_identity = provenance.get("source_checkpoint_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("audit provenance.source_checkpoint_identity is required")
    required_identity_fields = {
        "checkpoint_path",
        "base_model_path",
        "adapter_sha256",
        "special_embedding_sha256",
    }
    if set(source_identity) != required_identity_fields:
        raise ValueError("audit source checkpoint identity fields differ")
    for field in ("checkpoint_path", "base_model_path"):
        if not isinstance(source_identity[field], str) or not source_identity[field]:
            raise ValueError(f"audit source identity.{field} must be non-empty")
    for field in ("adapter_sha256", "special_embedding_sha256"):
        _digest(source_identity[field], field=f"audit source identity.{field}")
    if expected_source_identity is not None and dict(source_identity) != dict(
        expected_source_identity
    ):
        raise ValueError(
            "audit source checkpoint identity differs from the sealed source"
        )
    if "checkpoint_path" not in provenance:
        raise ValueError("audit provenance.checkpoint_path is required")
    checkpoint_paths = [
        container["checkpoint_path"]
        for container in (output, provenance)
        if "checkpoint_path" in container
    ]
    if len(checkpoint_paths) == 2 and checkpoint_paths[0] != checkpoint_paths[1]:
        raise ValueError("audit checkpoint_path differs between output and provenance")
    if checkpoint_paths:
        checkpoint_path = checkpoint_paths[0]
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("audit checkpoint_path must be a non-empty path")
        if expected_checkpoint_path is None:
            raise ValueError("audit checkpoint_path requires a sealed expected path")
        if checkpoint_path != expected_checkpoint_path:
            raise ValueError("audit checkpoint_path differs from the sealed checkpoint")
    for field in (
        "manifest_sha256",
        "panel_sha256",
        "panel_row_sha256",
        "image_sha256",
        "tokenizer_sha256",
        "prompt_policy_fingerprint",
    ):
        value = provenance.get(field)
        if not isinstance(value, str):
            raise ValueError(f"audit provenance.{field} is required")
        _digest(value, field=f"audit provenance.{field}")
        if field == "manifest_sha256" and expected_manifest_sha256 is not None:
            if value != expected_manifest_sha256:
                raise ValueError(
                    "audit provenance.manifest_sha256 differs from the sealed manifest"
                )
        expected = (
            _manifest_surface_identity(image, manifest_binding).get(field)
            if field
            in {"panel_sha256", "tokenizer_sha256", "prompt_policy_fingerprint"}
            else getattr(image, field, None)
        )
        if field != "manifest_sha256" and value != expected:
            raise ValueError(
                f"audit provenance.{field} differs from the manifest image"
            )


def _validate_source_assembly(
    config: EntryConfig,
    resources: DualGPUResourceReceipt,
    image_identity: Mapping[str, Any],
    source: SourceAssemblyReceipt,
) -> None:
    if (
        source.training_gpu != resources.training_gpu
        or source.audit_gpu != resources.audit_gpu
    ):
        raise ValueError(
            "source assembly GPU roles differ from the admitted dual-GPU receipt"
        )
    if source.source_plan_sha256 != config.content_sha256:
        raise ValueError("source assembly plan differs from the sealed entry config")
    expected_paths = {
        "checkpoint_path": config.source_checkpoint_path,
        "base_model_path": config.base_model_path,
        "adapter_path": config.adapter_path,
        "special_embedding_path": config.special_embedding_path,
    }
    for field, expected in expected_paths.items():
        actual = getattr(source, field)
        if not isinstance(actual, str) or actual != expected:
            raise ValueError(f"source assembly {field} differs from the sealed config")
    expected_digests = {
        "adapter_sha256": config.source_adapter_sha256,
        "special_embedding_sha256": config.special_embedding_sha256,
        "manifest_sha256": config.manifest_sha256,
        "panel_sha256": image_identity["panel_sha256"],
        "image_sha256": image_identity["image_sha256"],
        "tokenizer_sha256": image_identity["tokenizer_sha256"],
        "prompt_policy_fingerprint": image_identity["prompt_policy_fingerprint"],
    }
    for field, expected in expected_digests.items():
        actual = getattr(source, field)
        if not isinstance(actual, str) or actual != expected:
            raise ValueError(
                f"source assembly {field} differs from the admitted provenance"
            )
        _digest(actual, field=f"source assembly.{field}")
    if source.checkpoint_sha256 is None:
        raise ValueError("source assembly checkpoint_sha256 is required")
    _digest(source.checkpoint_sha256, field="source assembly.checkpoint_sha256")
    for field in ("source_validation_sha256", "assembly_receipt_sha256"):
        value = getattr(source, field)
        if value is None:
            raise ValueError(f"source assembly {field} is required")
        _digest(value, field=f"source assembly.{field}")
    if source.source_validation_sha256 != source.source_validation_content_sha256:
        raise ValueError(
            "source assembly validation digest does not bind source identity"
        )
    if source.assembly_receipt_sha256 != source.assembly_receipt_content_sha256:
        raise ValueError(
            "source assembly receipt digest does not bind the assembled roles"
        )


def _project_audit(
    image: Any,
    output: Mapping[str, Any],
    *,
    expected_manifest_sha256: str | None = None,
    expected_repetition_penalty: float | None = None,
    manifest_binding: Any | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_path: str | None = None,
    expected_parser: str = _CANONICAL_AUDIT_PARSER,
    expected_arm_id: str | None = None,
    expected_source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from scripts.research.analyze_human13_k_union import (
        _match_prefix,
        _ordered_predictions,
    )

    _validate_audit_provenance(
        image,
        output,
        expected_manifest_sha256=expected_manifest_sha256,
        manifest_binding=manifest_binding,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_path=expected_checkpoint_path,
        expected_parser=expected_parser,
        expected_arm_id=expected_arm_id,
        expected_source_identity=expected_source_identity,
    )
    if expected_repetition_penalty is not None:
        observed_rp = output.get("repetition_penalty")
        if (
            observed_rp is None
            or _rp(observed_rp, field="audit repetition penalty")
            != expected_repetition_penalty
        ):
            raise ValueError("audit repetition_penalty differs from the requested RP")
    predictions = _ordered_predictions(output)
    duplicate_threshold, owner_threshold = _image_matcher(image, manifest_binding)
    projected = _match_prefix(
        image,
        predictions,
        duplicate_iou_threshold=duplicate_threshold,
        owner_iou_threshold=owner_threshold,
    )
    raw_tokens = output.get("generated_token_ids", ())
    if not isinstance(raw_tokens, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in raw_tokens
    ):
        raise ValueError("audit generated_token_ids must be nonnegative integers")
    malformed_value = output.get(
        "malformed_row_count", output.get("dropped_prediction_count", 0)
    )
    malformed = _nonnegative_int(malformed_value, field="malformed_row_count")
    stop_reason = output.get("stop_reason")
    if stop_reason not in _SUPPORTED_STOP_REASONS:
        raise ValueError("audit stop_reason is unsupported")
    canonical_malformed = len(projected["invalid_rows"])
    effective_malformed = max(malformed, canonical_malformed)
    return {
        "owner_ids": tuple(sorted(str(item) for item in projected["owner_matches"])),
        "duplicate_rows": len(projected["duplicate_rows"]),
        "unmatched_rows": len(projected["unmatched_rows"]),
        "malformed_rows": effective_malformed,
        "cap_stops": int(stop_reason in _CAP_STOP_REASONS),
        "row_count": len(predictions)
        + max(0, effective_malformed - canonical_malformed),
        "token_count": len(raw_tokens),
    }


def acquired_h_owner_ids_from_trajectory(
    trajectory_ledger: object,
    manifest_image: object,
) -> tuple[str, ...]:
    """Derive exact acquired first-hit H from the admitted trajectory ledger."""

    from scripts.research.human13_trajectory_credit import (
        _require_scientific_ledger_admission,
    )

    ledger = _require_scientific_ledger_admission(trajectory_ledger)
    target_image_id = getattr(manifest_image, "image_id", None)
    if target_image_id is None and len(ledger.images) == 1:
        target_image_id = ledger.images[0].image_id
    image_ledgers = tuple(
        item
        for item in ledger.images
        if item.image_id == target_image_id
    )
    if len(image_ledgers) != 1:
        raise ValueError("trajectory ledger lacks the exact manifest image")
    manifest_h_ids = {
        str(owner_id) for owner_id in getattr(manifest_image, "h_owner_ids", ())
    }
    if not manifest_h_ids:
        manifest_h_ids = {
            str(getattr(owner, "owner_id"))
            for owner in getattr(manifest_image, "owners", ())
            if str(getattr(owner, "stratum", "")).upper() == "H"
        }
    acquired = {
        row.matched_owner_id
        for trajectory in image_ledgers[0].trajectories
        for row in trajectory.rows
        if row.outcome == "trusted_first_hit"
        and row.owner_stratum == "H"
        and row.matched_owner_id is not None
    }
    if any(
        not isinstance(owner_id, str)
        or not owner_id
        or owner_id not in manifest_h_ids
        for owner_id in acquired
    ):
        raise ValueError(
            "admitted trajectory H first hits differ from the manifest H set"
        )
    return tuple(sorted(cast(set[str], acquired)))


def analyze_audit_pair(
    *,
    image: Any,
    source_outputs: Mapping[float, Mapping[str, Any]],
    proposal_outputs: Mapping[float, Mapping[str, Any]],
    acquired_h_owner_ids: Sequence[str],
    expected_manifest_sha256: str | None = None,
    manifest_binding: Any | None = None,
    expected_source_checkpoint_sha256: str | None = None,
    expected_proposal_checkpoint_sha256: str | None = None,
    expected_source_checkpoint_path: str | None = None,
    expected_proposal_checkpoint_path: str | None = None,
    expected_parser: str = _CANONICAL_AUDIT_PARSER,
    expected_source_arm_id: str = _SOURCE_AUDIT_ARM_ID,
    expected_proposal_arm_id: str = _PROPOSAL_AUDIT_ARM_ID,
    expected_source_identity: Mapping[str, Any] | None = None,
) -> AuditPairAnalysis:
    """Project both clean-greedy surfaces through the canonical parser/matcher."""

    expected = set(AUDIT_REPETITION_PENALTIES)
    if (
        set(float(key) for key in source_outputs) != expected
        or set(float(key) for key in proposal_outputs) != expected
    ):
        raise ValueError("source and proposal audits must cover both RPs exactly")
    if (
        expected_source_checkpoint_sha256 is None
        or expected_proposal_checkpoint_sha256 is None
        or expected_source_checkpoint_path is None
        or expected_proposal_checkpoint_path is None
    ):
        raise ValueError(
            "audit analyzer requires sealed Source/proposal checkpoint identities and paths"
        )
    if expected_source_checkpoint_sha256 == expected_proposal_checkpoint_sha256:
        raise ValueError("proposal checkpoint payload must differ from Source")
    if expected_source_checkpoint_path == expected_proposal_checkpoint_path:
        raise ValueError("proposal checkpoint path must differ from Source")
    _digest(
        expected_source_checkpoint_sha256,
        field="expected_source_checkpoint_sha256",
    )
    _digest(
        expected_proposal_checkpoint_sha256,
        field="expected_proposal_checkpoint_sha256",
    )
    if expected_source_identity is None:
        raise ValueError("audit analyzer requires sealed source identity")
    owner_groups = tuple(
        set(getattr(image, field, ()))
        for field in ("g_owner_ids", "h_owner_ids", "m_owner_ids")
    )
    if len(set().union(*owner_groups)) != sum(len(group) for group in owner_groups):
        raise ValueError("manifest image G/H/M owner strata must be disjoint")
    owners = getattr(image, "owners", ())
    if (
        isinstance(owners, (list, tuple))
        and owners
        and all(isinstance(getattr(owner, "stratum", None), str) for owner in owners)
    ):
        owner_strata: dict[str, set[str]] = {"G": set(), "H": set(), "M": set()}
        for owner in owners:
            stratum = str(getattr(owner, "stratum")).upper()
            if stratum not in owner_strata:
                raise ValueError("manifest owner stratum must be G, H, or M")
            owner_strata[stratum].add(str(getattr(owner, "owner_id", "")))
        for stratum, group in zip(("G", "H", "M"), owner_groups, strict=True):
            if owner_strata[stratum] != group:
                raise ValueError(f"manifest owner strata do not match {stratum}")
    manifest_h_ids = {str(item) for item in getattr(image, "h_owner_ids", ())}
    if any(not isinstance(item, str) or not item for item in acquired_h_owner_ids):
        raise ValueError("acquired H owner IDs must be nonempty strings")
    h_ids = set(acquired_h_owner_ids)
    if not h_ids <= manifest_h_ids:
        raise ValueError(
            "acquired H owner IDs must be a subset of manifest image H owners"
        )
    m_ids = {str(item) for item in getattr(image, "m_owner_ids", ())}
    result: dict[float, AuditAnalysis] = {}
    identity_fingerprints: set[str] = set()
    for rp in AUDIT_REPETITION_PENALTIES:
        for output in (source_outputs[rp], proposal_outputs[rp]):
            provenance = output.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError("audit provenance must be an object")
            identity_fingerprints.add(
                _sha256(
                    {
                        "source_checkpoint_identity": provenance.get(
                            "source_checkpoint_identity"
                        ),
                        "tokenizer_sha256": provenance.get("tokenizer_sha256"),
                        "prompt_policy_fingerprint": provenance.get(
                            "prompt_policy_fingerprint"
                        ),
                    }
                )
            )
        source = _project_audit(
            image,
            source_outputs[rp],
            expected_manifest_sha256=expected_manifest_sha256,
            expected_repetition_penalty=rp,
            manifest_binding=manifest_binding,
            expected_checkpoint_sha256=expected_source_checkpoint_sha256,
            expected_checkpoint_path=expected_source_checkpoint_path,
            expected_parser=expected_parser,
            expected_arm_id=expected_source_arm_id,
            expected_source_identity=expected_source_identity,
        )
        proposal = _project_audit(
            image,
            proposal_outputs[rp],
            expected_manifest_sha256=expected_manifest_sha256,
            expected_repetition_penalty=rp,
            manifest_binding=manifest_binding,
            expected_checkpoint_sha256=expected_proposal_checkpoint_sha256,
            expected_checkpoint_path=expected_proposal_checkpoint_path,
            expected_parser=expected_parser,
            expected_arm_id=expected_proposal_arm_id,
            expected_source_identity=expected_source_identity,
        )
        source_ids = set(source["owner_ids"])
        proposal_ids = set(proposal["owner_ids"])
        result[rp] = AuditAnalysis(
            repetition_penalty=rp,
            source_owner_ids=tuple(sorted(source_ids)),
            proposal_owner_ids=tuple(sorted(proposal_ids)),
            h_gain_owner_ids=tuple(sorted((proposal_ids & h_ids) - source_ids)),
            g_loss_owner_ids=tuple(sorted(source_ids - proposal_ids)),
            m_gain_owner_ids=tuple(sorted((proposal_ids & m_ids) - source_ids)),
            net_unique_delta=len(proposal_ids) - len(source_ids),
            source_duplicate_rows=source["duplicate_rows"],
            proposal_duplicate_rows=proposal["duplicate_rows"],
            source_unmatched_rows=source["unmatched_rows"],
            proposal_unmatched_rows=proposal["unmatched_rows"],
            source_malformed_rows=source["malformed_rows"],
            proposal_malformed_rows=proposal["malformed_rows"],
            source_cap_stops=source["cap_stops"],
            proposal_cap_stops=proposal["cap_stops"],
            row_count=proposal["row_count"],
            token_count=proposal["token_count"],
        )
    if len(identity_fingerprints) != 1:
        raise ValueError("audit source/proposal checkpoint identity differs")
    return AuditPairAnalysis(result)


@dataclass(frozen=True)
class ContinuationGateInput:
    audit_pair: AuditPairAnalysis
    one_image_terminal_sha256: str | None = None


@dataclass(frozen=True)
class ContinuationGateReceipt:
    admitted: bool
    reasons: tuple[str, ...]
    audit_pair_sha256: str

    @property
    def content_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": "human13_continuation_gate.v1",
                "admitted": self.admitted,
                "reasons": list(self.reasons),
                "audit_pair_sha256": self.audit_pair_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_continuation_gate.v1",
            "admitted": self.admitted,
            "reasons": list(self.reasons),
            "audit_pair_sha256": self.audit_pair_sha256,
            "content_sha256": self.content_sha256,
        }


def evaluate_continuation_gate(
    value: AuditPairAnalysis | ContinuationGateInput,
) -> ContinuationGateReceipt:
    pair = value.audit_pair if isinstance(value, ContinuationGateInput) else value
    one = pair.by_repetition_penalty[1.0]
    reasons: list[str] = []
    if not one.h_gain_owner_ids:
        reasons.append("rp1.0_h_gain_empty")
    if one.net_unique_delta <= 0:
        reasons.append("rp1.0_net_unique_nonpositive")
    for rp in AUDIT_REPETITION_PENALTIES:
        audit = pair.by_repetition_penalty[rp]
        if audit.g_loss_owner_ids:
            reasons.append(f"rp{rp:g}_g_loss_nonzero")
        if audit.duplicate_delta > 0:
            reasons.append(f"rp{rp:g}_duplicate_increase")
        if audit.malformed_delta > 0:
            reasons.append(f"rp{rp:g}_malformed_increase")
        if audit.cap_delta > 0:
            reasons.append(f"rp{rp:g}_cap_increase")
    return ContinuationGateReceipt(
        admitted=not reasons,
        reasons=tuple(reasons),
        audit_pair_sha256=pair.content_sha256,
    )


class OneImageServices(Protocol):
    """Injected production seams; implementations own model/GPU actions."""

    def preflight_source_assembly(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> SourceAssemblyReceipt: ...

    def open_training(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> object: ...

    def open_audit(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> object: ...

    def source_audit(
        self, audit_session: object, repetition_penalty: float
    ) -> Mapping[str, Any]: ...

    def request_source_only_close(self) -> None: ...

    def acquire_and_replay(
        self, training_session: object, config: EntryConfig
    ) -> object: ...

    def apply_private_update(
        self, training_session: object, acquisition: object, config: EntryConfig
    ) -> object: ...

    def write_private_proposal(
        self, training_session: object, proposal: object, output_root: Path
    ) -> object: ...

    def proposal_audit(
        self, audit_session: object, private: object, repetition_penalty: float
    ) -> Mapping[str, Any]: ...

    def rollback_and_reproduce_source(
        self, training_session: object, proposal: object
    ) -> bool: ...

    def cleanup_private_proposal(self, private: object) -> None: ...

    def close(
        self, training_session: object | None, audit_session: object | None
    ) -> None: ...


@dataclass(frozen=True)
class OneImageTerminalReceipt:
    terminal_status: Literal[
        "dry_run",
        "preflight_admitted",
        "parity_failure",
        "update_failure",
        "completed_null_or_unsafe",
        "passing_one_image",
    ]
    resource_receipt: ResourceReceipt
    model_actions: Mapping[str, int]
    phase_receipts: tuple[str, ...] = ()
    source_assembly_sha256: str | None = None
    audit_pair_sha256: str | None = None
    continuation_gate_sha256: str | None = None
    phase_receipt_sha256s: tuple[str, ...] = ()
    phase_ledger_sha256: str | None = None
    private_proposal_cleaned: bool = False
    source_reproduced: bool = False
    failure_reason: str | None = None
    action_attempts: tuple[ActionAttemptReceipt, ...] = ()
    schema_version: Literal[
        "human13_all_hf_shared_surface_vertical_terminal.v1",
        "human13_all_hf_shared_surface_vertical_terminal.v2",
    ] = TERMINAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            LEGACY_TERMINAL_SCHEMA_VERSION,
            TERMINAL_SCHEMA_VERSION,
        }:
            raise ValueError("terminal receipt schema differs")
        if self.schema_version == LEGACY_TERMINAL_SCHEMA_VERSION and self.action_attempts:
            raise ValueError("legacy terminal cannot carry action attempts")
        if self.terminal_status not in {
            "dry_run",
            "preflight_admitted",
            "parity_failure",
            "update_failure",
            "completed_null_or_unsafe",
            "passing_one_image",
        }:
            raise ValueError("terminal status is outside the frozen Task-4 vocabulary")
        actions = {
            str(key): _nonnegative_int(value, field=f"model_actions.{key}")
            for key, value in self.model_actions.items()
        }
        object.__setattr__(self, "model_actions", MappingProxyType(actions))
        object.__setattr__(self, "phase_receipts", tuple(self.phase_receipts))
        object.__setattr__(
            self, "phase_receipt_sha256s", tuple(self.phase_receipt_sha256s)
        )
        object.__setattr__(self, "action_attempts", tuple(self.action_attempts))
        if any(not isinstance(value, ActionAttemptReceipt) for value in self.action_attempts):
            raise ValueError("terminal action attempts are malformed")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.model_actions.values()
        ):
            raise ValueError("model action counters must be nonnegative integers")
        for field in (
            "source_assembly_sha256",
            "audit_pair_sha256",
            "continuation_gate_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                _digest(value, field=field)
        for value in self.phase_receipt_sha256s:
            _digest(value, field="phase_receipt_sha256s")
        if self.phase_receipt_sha256s:
            if self.phase_ledger_sha256 is None:
                raise ValueError("terminal phase ledger digest is required")
            _digest(self.phase_ledger_sha256, field="phase_ledger_sha256")
            if self.phase_ledger_sha256 != _phase_ledger_sha256(
                self.phase_receipt_sha256s
            ):
                raise ValueError("terminal phase ledger digest differs")
        elif self.phase_ledger_sha256 is not None:
            raise ValueError("empty terminal phase ledger must not carry a digest")
        if self.terminal_status == "dry_run" and any(self.model_actions.values()):
            raise ValueError(
                "dry-run terminal must contain zero model/GPU/output actions"
            )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "terminal_status": self.terminal_status,
            "resource_receipt": self.resource_receipt.to_dict(),
            "resource_receipt_sha256": self.resource_receipt.content_sha256,
            "model_actions": dict(self.model_actions),
            "phase_receipts": list(self.phase_receipts),
            "source_assembly_sha256": self.source_assembly_sha256,
            "audit_pair_sha256": self.audit_pair_sha256,
            "continuation_gate_sha256": self.continuation_gate_sha256,
            "phase_receipt_sha256s": list(self.phase_receipt_sha256s),
            "phase_receipt_count": len(self.phase_receipt_sha256s),
            "phase_ledger_sha256": self.phase_ledger_sha256,
            "private_proposal_cleaned": self.private_proposal_cleaned,
            "source_reproduced": self.source_reproduced,
            "failure_reason": self.failure_reason,
        }
        if self.schema_version == TERMINAL_SCHEMA_VERSION:
            value["action_attempts"] = [
                action.to_dict() for action in self.action_attempts
            ]
            value["action_attempt_count"] = len(self.action_attempts)
        if include_hash:
            value["content_sha256"] = _sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OneImageTerminalReceipt:
        schema_version = value.get("schema_version")
        if schema_version not in {
            LEGACY_TERMINAL_SCHEMA_VERSION,
            TERMINAL_SCHEMA_VERSION,
        }:
            raise ValueError("terminal receipt schema differs")
        resource_value = value.get("resource_receipt")
        if not isinstance(resource_value, Mapping):
            raise ValueError("terminal receipt lacks detached resource receipt")
        resource = ResourceReceipt.from_dict(resource_value)
        if value.get("resource_receipt_sha256") != resource.content_sha256:
            raise ValueError("terminal resource lineage differs")
        phase_hashes = tuple(value.get("phase_receipt_sha256s", ()))
        serialized_count = value.get("phase_receipt_count")
        if (
            isinstance(serialized_count, bool)
            or not isinstance(serialized_count, int)
            or serialized_count != len(phase_hashes)
        ):
            raise ValueError("terminal phase receipt count differs")
        action_values = value.get("action_attempts", ())
        if schema_version == LEGACY_TERMINAL_SCHEMA_VERSION and (
            "action_attempts" in value or "action_attempt_count" in value
        ):
            raise ValueError("legacy terminal cannot carry action attempts")
        if not isinstance(action_values, (list, tuple)):
            raise ValueError("terminal action attempts must be an array")
        action_count = value.get("action_attempt_count", 0)
        if (
            isinstance(action_count, bool)
            or not isinstance(action_count, int)
            or action_count != len(action_values)
        ):
            raise ValueError("terminal action attempt count differs")
        action_attempts = tuple(ActionAttemptReceipt.from_dict(item) for item in action_values)
        result = cls(
            terminal_status=value["terminal_status"],
            resource_receipt=resource,
            model_actions=value["model_actions"],
            phase_receipts=tuple(value.get("phase_receipts", ())),
            source_assembly_sha256=value.get("source_assembly_sha256"),
            audit_pair_sha256=value.get("audit_pair_sha256"),
            continuation_gate_sha256=value.get("continuation_gate_sha256"),
            phase_receipt_sha256s=phase_hashes,
            phase_ledger_sha256=value.get("phase_ledger_sha256"),
            private_proposal_cleaned=value.get("private_proposal_cleaned", False),
            source_reproduced=value.get("source_reproduced", False),
            failure_reason=value.get("failure_reason"),
            action_attempts=action_attempts,
            schema_version=schema_version,
        )
        expected = value.get("content_sha256")
        if expected != result.content_sha256:
            raise ValueError("terminal receipt content hash differs")
        return result


def _seal_terminal(
    value: OneImageTerminalReceipt, *, issuer: object | None = None
) -> OneImageTerminalReceipt:
    if issuer is not _TERMINAL_ISSUER_TOKEN:
        raise PermissionError("only the guarded one-image issuer may seal a terminal")
    identity = id(value)

    def cleanup(_: ReferenceType[Any], *, key: int = identity) -> None:
        _TERMINAL_SEALS.pop(key, None)

    _TERMINAL_SEALS[identity] = (ref(value, cleanup), value.content_sha256)
    return value


def _require_terminal_seal(value: OneImageTerminalReceipt) -> None:
    entry = _TERMINAL_SEALS.get(id(value))
    if entry is None or entry[0]() is not value or entry[1] != value.content_sha256:
        raise PermissionError("one-image terminal is not an issued sealed receipt")


def _resource_receipt(
    resources: DualGPUResourceReceipt,
    output_root: OutputRootReceipt | None,
    phase_count: int,
    *,
    retry_count: int = 0,
    forward_counts: Mapping[str, int] | None = None,
    backward_count: int | None = None,
    reservation_identity: RunReservationIdentity | None = None,
) -> ResourceReceipt:
    sampled_request_count = 16
    sampled_group_count = 4
    counts = {
        "sample_forward_count": 2048,
        "replay_forward_count": 2048,
        "source_owner_forward_count": 0,
        "total_forward_count": 4096,
        "no_cache_forward_count": 4096,
    }
    if forward_counts is not None:
        counts = {
            field: forward_counts[field]
            for field in counts
        }
        sampled_request_count = forward_counts.get("sampled_request_count", 16)
        sampled_group_count = forward_counts.get("sampled_group_count", 4)
    if backward_count is None:
        backward_count = 1
    return ResourceReceipt(
        resources=resources,
        output_root=output_root,
        phase_count=phase_count,
        retry_count=retry_count,
        promoted_checkpoint=False,
        reservation_identity=reservation_identity,
        sampled_request_count=sampled_request_count,
        sampled_group_count=sampled_group_count,
        backward_count=backward_count,
        **counts,
    )


def _shared_surface_forward_counts(services: object) -> Mapping[str, int] | None:
    receipt = getattr(services, "shared_surface_resource_receipt", None)
    if receipt is None:
        return None
    to_dict = getattr(receipt, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("shared-surface resource receipt must be serializable")
    value = to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("shared-surface resource receipt must serialize to an object")
    fields = (
        "sample_forward_count",
        "replay_forward_count",
        "source_owner_forward_count",
        "total_forward_count",
        "no_cache_forward_count",
    )
    counts: dict[str, int] = {
        field: _nonnegative_int(value.get(field), field=f"shared_surface.{field}")
        for field in fields
    }
    sampled_hashes = value.get("sampled_group_sha256s", ())
    sampled_seed_groups = value.get("sampled_seed_groups", ())
    if not isinstance(sampled_hashes, (list, tuple)):
        raise TypeError("shared-surface sampled group hashes must be a sequence")
    if not isinstance(sampled_seed_groups, (list, tuple)):
        raise TypeError("shared-surface sampled seed groups must be a sequence")
    counts["sampled_group_count"] = len(sampled_hashes)
    counts["sampled_request_count"] = (
        sum(
            len(group)
            for group in sampled_seed_groups
            if isinstance(group, (list, tuple))
        )
        if sampled_seed_groups
        else len(sampled_hashes) * 4
    )
    return counts


def dry_run(config: EntryConfig) -> OneImageTerminalReceipt:
    """Return a value-only estimate; no root check or service call occurs."""

    resources = DualGPUResourceReceipt(
        cards=(
            GPUResource(0, 1, 1),
            GPUResource(1, 1, 1),
        )
    )
    receipt = _resource_receipt(resources, None, 0)
    return _seal_terminal(
        OneImageTerminalReceipt(
            terminal_status="dry_run",
            resource_receipt=receipt,
            model_actions=ZERO_MODEL_ACTIONS,
            phase_receipts=("dry_run_planned",),
        ),
        issuer=_TERMINAL_ISSUER_TOKEN,
    )


def _classify_failure(error: BaseException) -> str:
    name = type(error).__name__.lower()
    text = str(error).lower()
    if "parity" in name or "parity" in text:
        return "parity_failure"
    return "update_failure"


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _private_from_error(error: BaseException) -> object | None:
    for field in ("private_proposal", "private", "proposal"):
        value = getattr(error, field, None)
        if value is not None:
            return value
    return None


def _private_from_services(services: object) -> object | None:
    for field in ("private_proposal", "partial_private_proposal", "private"):
        value = getattr(services, field, None)
        if value is not None and not callable(value):
            return value
    return None


def _checkpoint_payload_digest(value: object, *, field: str) -> str:
    candidates = ("checkpoint_payload_sha256", "checkpoint_sha256")
    observed: object = None
    for candidate in candidates:
        if isinstance(value, Mapping) and candidate in value:
            observed = value[candidate]
            break
        candidate_value = getattr(value, candidate, None)
        if candidate_value is not None:
            observed = candidate_value
            break
    if not isinstance(observed, str):
        raise ValueError(f"{field} must carry a checkpoint payload digest")
    return _digest(observed, field=field)


def _checkpoint_path(value: object, *, field: str) -> str:
    observed: object = None
    if isinstance(value, Mapping) and "checkpoint_path" in value:
        observed = value["checkpoint_path"]
    else:
        observed = getattr(value, "checkpoint_path", None)
    if not isinstance(observed, str) or not observed:
        raise ValueError(f"{field} must carry a checkpoint path")
    return observed


def _action_counters(
    services: object,
    *,
    phases: Sequence[str],
    status: str,
) -> dict[str, int]:
    """Detach injected action counters without assuming a concrete runtime."""

    observed = getattr(services, "action_counters", None)
    if callable(observed):
        observed = observed()
    if isinstance(observed, Mapping):
        result = dict(ZERO_MODEL_ACTIONS)
        for key, value in observed.items():
            if key in result:
                result[key] = _nonnegative_int(value, field=f"action.{key}")
        return result
    result = dict(ZERO_MODEL_ACTIONS)
    if not phases:
        return result
    result["model_loads"] = int(
        "gpu0_training_surface_open" in phases
        or "gpu1_hf_fp32_sdpa_audit_surface_open" in phases
    ) + int("gpu1_hf_fp32_sdpa_audit_surface_open" in phases)
    result["gpu_allocations"] = int(result["model_loads"] > 0) * 2
    result["forwards"] = sum(
        phase.startswith("source_audit_rp_") or phase.startswith("proposal_audit_rp_")
        for phase in phases
    )
    result["backwards"] = int("one_private_update" in phases)
    result["optimizer_steps"] = int("one_private_update" in phases)
    result["output_creations"] = int("private_proposal_written" in phases)
    del status
    return result


def run_one_image(
    config: EntryConfig,
    *,
    authority: ExecutionAuthority,
    resources: Sequence[GPUResource] | DualGPUResourceReceipt,
    output_root: str | Path,
    services: OneImageServices,
    manifest_image: Any | ManifestImageContext | None = None,
    manifest_binding: Any | None = None,
    execute: bool = True,
    preflight_only: bool = False,
) -> OneImageTerminalReceipt:
    """Run one guarded lifecycle through injected Task-1/2/3 owners.

    Admission is complete before the first injected method that may load a
    model.  Every exception still attempts private-proposal cleanup and closes
    both borrowed sessions.  No retry or fallback branch exists.
    """

    if execute is not True:
        return dry_run(config)
    authority.require()
    resource_receipt = validate_dual_gpu_resources(resources)
    root_receipt = confirm_absent_output_root(output_root)
    manifest_image, manifest_binding = _split_manifest_context(
        manifest_image, manifest_binding
    )
    image_identity = _manifest_image_identity(config, manifest_image, manifest_binding)
    _image_matcher(manifest_image, manifest_binding)
    phases: list[str] = []
    training_session: object | None = None
    audit_session: object | None = None
    private_proposal: object | None = None
    proposal_object: object | None = None
    source_outputs: dict[float, Mapping[str, Any]] = {}
    proposal_outputs: dict[float, Mapping[str, Any]] = {}
    source_assembly: SourceAssemblyReceipt | None = None
    audit_pair: AuditPairAnalysis | None = None
    gate: ContinuationGateReceipt | None = None
    cleaned = False
    reproduced = False
    rollback_attempted = False
    failure: BaseException | None = None
    reservation_state_exposed = hasattr(services, "reservation_identity")
    status: Literal[
        "preflight_admitted",
        "parity_failure",
        "update_failure",
        "completed_null_or_unsafe",
        "passing_one_image",
    ] = "update_failure"

    class _PreflightAdmitted(Exception):
        pass

    try:
        source_assembly = services.preflight_source_assembly(config, resource_receipt)
        phases.append("admission_source_assembly")
        if not isinstance(source_assembly, SourceAssemblyReceipt):
            raise ValueError(
                "source assembly adapter must return SourceAssemblyReceipt"
            )
        _validate_source_assembly(
            config, resource_receipt, image_identity, source_assembly
        )
        source_identity = {
            "checkpoint_path": source_assembly.checkpoint_path,
            "base_model_path": source_assembly.base_model_path,
            "adapter_sha256": source_assembly.adapter_sha256,
            "special_embedding_sha256": source_assembly.special_embedding_sha256,
        }
        source_checkpoint_sha256 = cast(str, source_assembly.checkpoint_sha256)
        training_session = services.open_training(config, resource_receipt)
        phases.append("gpu0_training_surface_open")
        audit_session = services.open_audit(config, resource_receipt)
        phases.append("gpu1_hf_fp32_sdpa_audit_surface_open")
        for rp in config.audit_repetition_penalties:
            try:
                source_outputs[rp] = services.source_audit(audit_session, rp)
            except BaseException as error:
                observed_phase = getattr(error, "_source_audit_phase", None)
                if (
                    isinstance(observed_phase, str)
                    and observed_phase.startswith("source_audit_rp_")
                    and observed_phase not in phases
                ):
                    phases.append(observed_phase)
                raise
            _project_audit(
                manifest_image,
                source_outputs[rp],
                expected_manifest_sha256=config.manifest_sha256,
                expected_repetition_penalty=rp,
                manifest_binding=manifest_binding,
                expected_checkpoint_sha256=source_checkpoint_sha256,
                expected_checkpoint_path=source_assembly.checkpoint_path,
                expected_parser=config.parser,
                expected_arm_id=_SOURCE_AUDIT_ARM_ID,
                expected_source_identity=source_identity,
            )
            phases.append(f"source_audit_rp_{rp:g}")
        pre_acquisition_admission = getattr(
            services, "pre_acquisition_admission", None
        )
        if callable(pre_acquisition_admission):
            pre_acquisition_admission(training_session, source_outputs)
            phases.append("pre_acquisition_update_admission")
        if preflight_only:
            phases.append("preflight_source_surfaces_admitted")
            request_source_only_close = getattr(
                services, "request_source_only_close", None
            )
            if callable(request_source_only_close):
                request_source_only_close()
                phases.append("source_only_close_requested")
            raise _PreflightAdmitted()
        acquisition = services.acquire_and_replay(training_session, config)
        phases.append("k16_acquisition_replay")
        if getattr(acquisition, "parity_passed", True) is not True:
            raise RuntimeError("shared-surface replay parity failure")
        proposal = services.apply_private_update(training_session, acquisition, config)
        proposal_object = proposal
        phases.append("one_private_update")
        private_proposal = services.write_private_proposal(
            training_session, proposal, Path(root_receipt.path)
        )
        proposal_checkpoint_path = _checkpoint_path(
            private_proposal, field="private proposal checkpoint_path"
        )
        proposal_checkpoint_sha256 = _checkpoint_payload_digest(
            private_proposal, field="private proposal checkpoint_sha256"
        )
        if proposal_checkpoint_sha256 == source_checkpoint_sha256:
            raise ValueError(
                "private proposal checkpoint payload must differ from Source"
            )
        if proposal_checkpoint_path == source_assembly.checkpoint_path:
            raise ValueError("private proposal checkpoint path must differ from Source")
        phases.append("private_proposal_written")
        for rp in config.audit_repetition_penalties:
            proposal_outputs[rp] = services.proposal_audit(
                audit_session, private_proposal, rp
            )
            phases.append(f"proposal_audit_rp_{rp:g}")
        proposal_input = getattr(acquisition, "cuda_proposal_input", None)
        admitted_trajectory = getattr(proposal_input, "trajectory_ledger", None)
        raw_acquired_h = (
            acquired_h_owner_ids_from_trajectory(
                admitted_trajectory,
                manifest_image,
            )
            if admitted_trajectory is not None
            else getattr(acquisition, "trusted_h_owner_ids", ())
        )
        if not isinstance(raw_acquired_h, (list, tuple)):
            raise ValueError("acquired H owner IDs must be a sequence")
        if any(not isinstance(item, str) or not item for item in raw_acquired_h):
            raise ValueError("acquired H owner IDs must be nonempty strings")
        acquired_h_set = set(raw_acquired_h)
        if not acquired_h_set <= set(image_identity["h_owner_ids"]):
            raise ValueError(
                "acquired H owner IDs include owners outside the manifest H set"
            )
        acquired_h = tuple(sorted(acquired_h_set))
        audit_pair = analyze_audit_pair(
            image=manifest_image,
            source_outputs=source_outputs,
            proposal_outputs=proposal_outputs,
            acquired_h_owner_ids=acquired_h,
            expected_manifest_sha256=config.manifest_sha256,
            manifest_binding=manifest_binding,
            expected_source_checkpoint_sha256=source_checkpoint_sha256,
            expected_proposal_checkpoint_sha256=proposal_checkpoint_sha256,
            expected_source_checkpoint_path=source_assembly.checkpoint_path,
            expected_proposal_checkpoint_path=proposal_checkpoint_path,
            expected_parser=config.parser,
            expected_source_arm_id=_SOURCE_AUDIT_ARM_ID,
            expected_proposal_arm_id=_PROPOSAL_AUDIT_ARM_ID,
            expected_source_identity=source_identity,
        )
        gate = evaluate_continuation_gate(audit_pair)
        phases.append("dual_rp_audit_analyzed")
        rollback_attempted = True
        reproduced = bool(
            services.rollback_and_reproduce_source(training_session, proposal)
        )
        phases.append("rollback_source_reproduction")
        if not reproduced:
            raise RuntimeError("Source reproduction failed after private audits")
        status = "passing_one_image" if gate.admitted else "completed_null_or_unsafe"
    except _PreflightAdmitted:
        failure = None
        status = "preflight_admitted"
    except BaseException as error:  # cleanup and classification are terminal-owned
        failure = error
        status = cast(
            Literal[
                "preflight_admitted",
                "parity_failure",
                "update_failure",
                "completed_null_or_unsafe",
                "passing_one_image",
            ],
            _classify_failure(error),
        )
    finally:
        try:
            retry_count = int(getattr(services, "retry_count", 0))
        except (TypeError, ValueError):
            retry_count = 1
        if retry_count != 0 or bool(getattr(services, "fallback_used", False)):
            if failure is None:
                failure = RuntimeError("adaptive retry or fallback is forbidden")
            status = "update_failure"
        if (
            proposal_object is not None
            and not rollback_attempted
            and training_session is not None
        ):
            try:
                rollback_attempted = True
                reproduced = bool(
                    services.rollback_and_reproduce_source(
                        training_session, proposal_object
                    )
                )
                phases.append("rollback_source_reproduction")
                if not reproduced:
                    if failure is None:
                        failure = RuntimeError(
                            "Source reproduction failed after private update"
                        )
                    status = "update_failure"
            except BaseException as rollback_error:
                if failure is None:
                    failure = rollback_error
                status = "update_failure"
        if private_proposal is None and failure is not None:
            private_proposal = _private_from_error(failure)
        if private_proposal is None:
            private_proposal = _private_from_services(services)
        if private_proposal is not None:
            try:
                services.cleanup_private_proposal(private_proposal)
                cleaned = True
                phases.append("private_proposal_cleanup")
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                status = "update_failure"
        if training_session is not None or audit_session is not None:
            try:
                services.close(training_session, audit_session)
                phases.append("sessions_closed")
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                status = "update_failure"
        if not reproduced and bool(getattr(services, "source_reproduced", False)):
            reproduced = True
            if "rollback_source_reproduction" not in phases:
                phases.append("rollback_source_reproduction")
    reservation_identity = getattr(services, "reservation_identity", None)
    if reservation_identity is not None and not isinstance(
        reservation_identity, RunReservationIdentity
    ):
        raise TypeError("service reservation identity is malformed")
    if failure is not None and reservation_state_exposed and reservation_identity is None:
        raise failure
    service_phase_hashes = tuple(getattr(services, "phase_receipt_sha256s", ()))
    service_phase_ledger_sha256 = getattr(services, "phase_ledger_sha256", None)
    action_counters = _action_counters(services, phases=phases, status=status)
    shared_forward_counts = _shared_surface_forward_counts(services)
    if (
        shared_forward_counts is None
        and reservation_identity is not None
        and all(value == 0 for value in action_counters.values())
    ):
        shared_forward_counts = {
            "sample_forward_count": 0,
            "replay_forward_count": 0,
            "source_owner_forward_count": 0,
            "total_forward_count": 0,
            "no_cache_forward_count": 0,
            "sampled_group_count": 0,
            "sampled_request_count": 0,
        }
    terminal = _seal_terminal(
        OneImageTerminalReceipt(
            terminal_status=status,
            resource_receipt=_resource_receipt(
                resource_receipt,
                root_receipt,
                len(phases),
                retry_count=retry_count,
                forward_counts=shared_forward_counts,
                backward_count=action_counters["backwards"],
                reservation_identity=reservation_identity,
            ),
            model_actions=action_counters,
            phase_receipts=tuple(phases),
            source_assembly_sha256=None
            if source_assembly is None
            else source_assembly.content_sha256,
            audit_pair_sha256=None if audit_pair is None else audit_pair.content_sha256,
            continuation_gate_sha256=None if gate is None else gate.content_sha256,
            phase_receipt_sha256s=service_phase_hashes,
            phase_ledger_sha256=service_phase_ledger_sha256,
            private_proposal_cleaned=cleaned,
            source_reproduced=reproduced,
            failure_reason=None if failure is None else _error_text(failure),
            action_attempts=tuple(getattr(services, "action_attempts", lambda: ())()),
        ),
        issuer=_TERMINAL_ISSUER_TOKEN,
    )
    persist_terminal = getattr(services, "persist_terminal", None)
    if callable(persist_terminal) and (
        not reservation_state_exposed or reservation_identity is not None
    ):
        persist_terminal(terminal)
    return terminal


@dataclass(frozen=True)
class FullPanelAdmissionReceipt:
    admitted: bool
    one_image_terminal_sha256: str
    expected_terminal_sha256: str
    model_actions: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_actions",
            MappingProxyType(
                {
                    str(key): _nonnegative_int(value, field=f"model_actions.{key}")
                    for key, value in self.model_actions.items()
                }
            ),
        )


def run_full_panel(
    config: EntryConfig,
    *,
    one_image_terminal_hash: str | None,
    expected_terminal_hash: str,
    one_image_terminal: OneImageTerminalReceipt | None = None,
    authority: ExecutionAuthority,
    resources: Sequence[GPUResource] | DualGPUResourceReceipt,
    output_root: str | Path,
    services: OneImageServices | None = None,
) -> FullPanelAdmissionReceipt:
    """Admit only the exact passing one-image terminal; do not execute Task 5."""

    authority.require()
    _digest(expected_terminal_hash, field="expected_terminal_hash")
    if one_image_terminal_hash is None:
        raise PermissionError(
            "full-panel execution requires a passing one-image terminal hash"
        )
    _digest(one_image_terminal_hash, field="one_image_terminal_hash")
    if one_image_terminal_hash != expected_terminal_hash:
        raise PermissionError(
            "full-panel terminal hash differs from the exact passing one-image result"
        )
    if not isinstance(one_image_terminal, OneImageTerminalReceipt):
        raise PermissionError(
            "full-panel admission requires a sealed one-image terminal receipt"
        )
    _require_terminal_seal(one_image_terminal)
    if one_image_terminal.content_sha256 != one_image_terminal_hash:
        raise PermissionError("sealed one-image terminal content hash differs")
    if one_image_terminal.terminal_status != "passing_one_image":
        raise PermissionError("sealed one-image terminal is not a passing result")
    if (
        not one_image_terminal.private_proposal_cleaned
        or not one_image_terminal.source_reproduced
        or "rollback_source_reproduction" not in one_image_terminal.phase_receipts
        or "private_proposal_cleanup" not in one_image_terminal.phase_receipts
        or "sessions_closed" not in one_image_terminal.phase_receipts
        or one_image_terminal.source_assembly_sha256 is None
        or one_image_terminal.audit_pair_sha256 is None
        or one_image_terminal.continuation_gate_sha256 is None
    ):
        raise PermissionError(
            "sealed one-image terminal lacks cleanup, rollback, or gate evidence"
        )
    validate_dual_gpu_resources(resources)
    confirm_absent_output_root(output_root)
    del config, services
    return FullPanelAdmissionReceipt(
        admitted=True,
        one_image_terminal_sha256=one_image_terminal_hash,
        expected_terminal_sha256=expected_terminal_hash,
        model_actions=ZERO_MODEL_ACTIONS,
    )


@dataclass(frozen=True)
class ProductionExecution:
    """Fully constructed explicit-execute arguments for the guarded runner."""

    services: OneImageServices
    resources: DualGPUResourceReceipt
    output_root: Path
    manifest_image: Any
    manifest_binding: Any


def _validate_reservation_cli(args: argparse.Namespace) -> None:
    mode = args.reservation_mode
    stale = args.stale_reservation
    recovery_authority = args.recovery_authority
    stale_owner_pid = args.stale_owner_pid
    if args.full_panel and (
        mode != "fresh_primary"
        or stale is not None
        or recovery_authority is not None
        or stale_owner_pid is not None
    ):
        raise ValueError(
            "full-panel admission requires fresh primary reservation defaults"
        )
    if mode == "fresh_primary":
        if stale is not None or recovery_authority is not None or stale_owner_pid is not None:
            raise ValueError(
                "fresh primary reservation forbids stale-parent recovery inputs"
            )
        return
    if mode != "lost_owner_recovery":
        raise ValueError("reservation mode is unsupported")
    if stale is None:
        raise ValueError("lost-owner recovery requires an explicit stale reservation")
    if not isinstance(recovery_authority, str) or not recovery_authority:
        raise ValueError("lost-owner recovery requires explicit recovery authority")


def load_task5_runtime_factory(spec: str) -> Any:
    """Load the explicit existing-owner Task-5 semantic factory."""

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--runtime-factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("Task-5 runtime factory must be callable")
    return factory


def build_production_execution(
    *,
    config: EntryConfig,
    args: argparse.Namespace,
    gpu_observer: Callable[[], Sequence[GPUResource]] | None = None,
    manifest_loader: Callable[..., Any] | None = None,
    backend_factory: Callable[..., Any] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    hf_native_owner: Any | None = None,
) -> ProductionExecution:
    """Construct the live owner only after explicit ``--execute`` admission."""

    if args.manifest is None or args.attempt_id is None:
        raise ValueError("one-image --execute requires --manifest and --attempt-id")
    _validate_reservation_cli(args)
    repo_root = Path(args.repo_root).expanduser().resolve()
    _ensure_repo_root_on_sys_path(repo_root)
    resources = admit_production_gpu_resources(
        (gpu_observer or observe_production_gpu_resources)()
    )
    from scripts.research.build_human13_k_union_manifest import load_manifest
    from scripts.research.human13_live_model import HUMAN13_SOURCE_INFER_CONFIG
    from scripts.research.human13_hf_native_one_image_owner import (
        build_repository_hf_native_owner,
    )
    from scripts.research.human13_one_image_services import (
        ExistingOwnersProductionBackend,
        ProductionOneImageServices,
        default_task5_runtime_factory,
    )

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = (manifest_loader or load_manifest)(
        manifest_path, require_full_panel=True
    )
    selected = tuple(image for image in manifest.images if image.image_id == 1584)
    if len(selected) != 1:
        raise ValueError("loaded manifest must contain exactly one image-1584 record")
    output_root = (args.output_root or Path(config.output_root)).expanduser().resolve()
    source_config = (
        Path(args.source_config).expanduser().resolve()
        if args.source_config is not None
        else (repo_root / HUMAN13_SOURCE_INFER_CONFIG).resolve()
    )
    stale = (
        None
        if args.stale_reservation is None
        else Path(args.stale_reservation).expanduser().resolve()
    )
    backend_kwargs: dict[str, Any] = dict(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=repo_root,
        source_config_path=source_config,
        runtime_factory=runtime_factory or default_task5_runtime_factory,
    )
    if hf_native_owner is None:
        # The repository owner is fail-closed: missing same-session projector,
        # pre-acquisition graph row, or registry admission remains a typed HOLD.
        hf_native_owner = build_repository_hf_native_owner()
    if hf_native_owner is not None:
        # Explicit public construction dependency: never recover scientific
        # owners from a private attribute on the shared-surface session.
        backend_kwargs["hf_native_owner"] = hf_native_owner
    backend = (backend_factory or ExistingOwnersProductionBackend)(**backend_kwargs)
    services = ProductionOneImageServices(
        backend=backend,
        reservation_mode=args.reservation_mode,
        stale_reservation_path=stale,
        successor_root=output_root,
        attempt_id=args.attempt_id,
        recovery_authority=args.recovery_authority,
        stale_owner_pid=getattr(args, "stale_owner_pid", None),
    )
    return ProductionExecution(
        services=services,
        resources=resources,
        output_root=output_root,
        manifest_image=selected[0],
        manifest_binding=manifest.binding,
    )


def _default_config_path() -> Path:
    return Path(
        "configs/coordexp_swift/research/human13_all_hf_shared_surface_vertical/01_image1584.yaml"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-config", type=Path, default=None)
    parser.add_argument(
        "--reservation-mode",
        choices=("fresh_primary", "lost_owner_recovery"),
        default="fresh_primary",
    )
    parser.add_argument("--stale-reservation", type=Path, default=None)
    parser.add_argument("--recovery-authority", default=None)
    parser.add_argument(
        "--stale-owner-pid",
        type=int,
        default=None,
        help="Explicit lost-owner PID witness for a legacy reservation without pid",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run production source admission and close without K16/update",
    )
    parser.add_argument("--user-model-gpu-authority", action="store_true")
    parser.add_argument("--full-panel", action="store_true")
    parser.add_argument("--one-image-terminal-sha256", default=None)
    parser.add_argument("--expected-terminal-sha256", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_reservation_cli(args)
    config = EntryConfig.from_yaml(args.config)
    if not args.execute:
        print(
            json.dumps(dry_run(config).to_dict(), sort_keys=True, separators=(",", ":"))
        )
        return 0
    authority = ExecutionAuthority(args.user_model_gpu_authority)
    output_root = args.output_root or Path(config.output_root)
    if args.full_panel:
        authority.require()
        expected = args.expected_terminal_sha256
        if expected is None:
            raise PermissionError(
                "--expected-terminal-sha256 is required for --full-panel"
            )
        receipt = run_full_panel(
            config,
            one_image_terminal_hash=args.one_image_terminal_sha256,
            expected_terminal_hash=expected,
            authority=authority,
            resources=admit_production_gpu_resources(
                observe_production_gpu_resources()
            ),
            output_root=output_root,
        )
        print(
            json.dumps(
                {
                    "admitted": receipt.admitted,
                    "model_actions": dict(receipt.model_actions),
                },
                sort_keys=True,
            )
        )
        return 0
    authority.require()
    execution = build_production_execution(config=config, args=args)
    terminal = run_one_image(
        config,
        authority=authority,
        resources=execution.resources,
        output_root=execution.output_root,
        services=execution.services,
        manifest_image=execution.manifest_image,
        manifest_binding=execution.manifest_binding,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(terminal.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALL_HF_VERTICAL_UNIT_ID",
    "AUDIT_REPETITION_PENALTIES",
    "ActionAttemptReceipt",
    "AuditAnalysis",
    "AuditPairAnalysis",
    "ContinuationGateInput",
    "ContinuationGateReceipt",
    "DualGPUResourceReceipt",
    "EntryConfig",
    "ExecutionAuthority",
    "FullPanelAdmissionReceipt",
    "GPUResource",
    "ManifestImageContext",
    "OneImageServices",
    "OneImageTerminalReceipt",
    "LEGACY_TERMINAL_SCHEMA_VERSION",
    "OutputRootReceipt",
    "ProductionExecution",
    "ResourceReceipt",
    "SourceAssemblyReceipt",
    "admit_production_gpu_resources",
    "analyze_audit_pair",
    "build_parser",
    "build_production_execution",
    "confirm_absent_output_root",
    "dry_run",
    "evaluate_continuation_gate",
    "main",
    "load_task5_runtime_factory",
    "observe_production_gpu_resources",
    "run_full_panel",
    "run_one_image",
    "validate_dual_gpu_resources",
]
