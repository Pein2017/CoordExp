"""SFT runner - pure YAML config-driven, no CLI arguments for hyperparameters"""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from multiprocessing import Manager
from typing import Any, Literal, Mapping, cast

import torch

try:
    from torch.distributed.elastic.multiprocessing.errors import (
        record as _torch_elastic_record,
    )
except Exception:

    def _torch_elastic_record(fn):
        return fn


from swift.llm.train.rlhf import SwiftRLHF
from swift.llm.train.sft import SwiftSft
from swift.trainers import TrainerFactory
from swift.utils import get_dist_setting

from .data_collators import build_dataset_metrics_collator
from .coord_tokens.template_adapter import apply_coord_template_adapter
from .coord_tokens.offset_adapter import (
    install_coord_offset_adapter,
    reattach_coord_offset_hooks,
)
from .bootstrap.pipeline_manifest import build_pipeline_manifest
from .bootstrap.trainer_setup import (
    build_trainer_callbacks,
    compose_trainer_class,
    instantiate_trainer,
)
from .config import ConfigLoader
from .config.prompts import get_template_prompt_hash, resolve_dense_prompt_variant_key
from .config.strict_dataclass import dataclass_asdict_no_none
from .datasets import (
    BaseCaptionDataset,
    RandomSampleDataset,
    build_static_packed_dataset,
)
from .trainers import with_final_checkpoint
from .utils import (
    FileLoggingConfig,
    enable_output_dir_file_logging,
    enable_verbose_logging,
    get_logger,
    set_log_level,
)
from .optim import register_coord_offset_optimizer
from .bootstrap.run_metadata import (
    attach_encoded_sample_cache_run_metadata as _attach_encoded_sample_cache_run_metadata_impl,
    collect_dependency_provenance as _collect_dependency_provenance_impl,
    collect_launcher_metadata_from_env as _collect_launcher_metadata_from_env_impl,
    write_run_metadata_file,
)


def resolve_trainer_cls(train_args):
    trainer_variant = getattr(train_args, "trainer_variant", None)
    if trainer_variant == "stage2_ab_training":
        raise ValueError(
            "custom.trainer_variant=stage2_ab_training has been removed; use stage2_two_channel"
        )
    if trainer_variant == "rollout_matching_sft":
        raise ValueError(
            "custom.trainer_variant=rollout_matching_sft has been removed; use stage2_rollout_aligned"
        )
    if trainer_variant == "stage2_two_channel":
        from .trainers.stage2_two_channel import Stage2TwoChannelTrainer

        trainer_cls = Stage2TwoChannelTrainer
    elif trainer_variant == "stage2_rollout_aligned":
        from .trainers.stage2_rollout_aligned import Stage2RolloutAlignedTrainer

        trainer_cls = Stage2RolloutAlignedTrainer
    elif (
        getattr(train_args, "rlhf_type", None) == "gkd"
        and trainer_variant == "gkd_monitor"
    ):
        from .trainers.gkd_monitor import GKDTrainerWithMetrics

        trainer_cls = GKDTrainerWithMetrics
    else:
        trainer_cls = TrainerFactory.get_trainer_cls(train_args)

    return with_final_checkpoint(trainer_cls)


def _resolve_model_checkpoint_path(training_config: Any) -> str | None:
    model_section = getattr(training_config, "model", None)
    if isinstance(model_section, Mapping):
        model_checkpoint = model_section.get("model")
    else:
        model_checkpoint = getattr(model_section, "model", None)
    if not isinstance(model_checkpoint, str):
        return None
    model_checkpoint = model_checkpoint.strip()
    return model_checkpoint or None


def _resolve_dense_prompt_identity(custom_config: Any) -> dict[str, Any]:
    use_summary = bool(getattr(custom_config, "use_summary", False))
    prompt_variant = resolve_dense_prompt_variant_key(
        (
            getattr(custom_config, "extra", {}) or {}
        ).get("prompt_variant")
        if isinstance(getattr(custom_config, "extra", {}) or {}, Mapping)
        else None
    )
    if use_summary:
        prompt_template_hash = None
    else:
        prompt_template_hash = get_template_prompt_hash(
            ordering=str(getattr(custom_config, "object_ordering", "sorted") or "sorted"),
            coord_mode="coord_tokens",
            prompt_variant=prompt_variant,
            object_field_order=str(
                getattr(custom_config, "object_field_order", "desc_first")
                or "desc_first"
            ),
            bbox_format=str(getattr(custom_config, "bbox_format", "xyxy") or "xyxy"),
        )
    return {
        "prompt_variant": prompt_variant,
        "prompt_template_hash": prompt_template_hash,
    }


# Use the model's native chat_template (JSON/Jinja) shipped with the tokenizer

logger = get_logger(__name__)


@dataclass(frozen=True)
class PackingRuntimeConfig:
    enabled: bool = False
    mode: Literal["dynamic", "static"] = "static"
    packing_length: int = 0
    buffer_size: int = 512
    min_fill_ratio: float = 0.65
    drop_last: bool = True
    allow_single_long: bool = True
    eval_packing: bool = True
    wait_timeout_s: float = 7200.0
    length_cache_persist_every: int | None = None
    length_precompute_workers: int = 8


@dataclass(frozen=True)
class EncodedSampleCacheRuntimeConfig:
    enabled: bool = False
    root_dir: str | None = None
    ineligible_policy: Literal["error", "bypass"] = "error"
    wait_timeout_s: float = 7200.0
    max_resident_shards: int = 4


@dataclass(frozen=True)
class StaticPackingCacheRuntimeConfig:
    root_dir: str | None = None


def _parse_packing_config(
    training_cfg: Any, template: Any, train_args: Any
) -> PackingRuntimeConfig:
    cfg = training_cfg or {}
    enabled = bool(cfg.get("packing", False))
    if not enabled:
        return PackingRuntimeConfig(enabled=False)

    mode_raw = cfg.get("packing_mode", "static")
    mode = str(mode_raw or "static").strip().lower()
    if mode not in {"dynamic", "static"}:
        raise ValueError(
            "training.packing_mode must be one of {'dynamic', 'static'}, "
            f"got {mode_raw!r}"
        )
    if mode == "dynamic":
        logger.warning(
            "training.packing_mode=dynamic is deprecated for Stage-1 and will fail fast when dataset-level packing is applied. "
            "Use training.packing_mode=static."
        )

    default_length = getattr(template, "max_length", None) or getattr(
        train_args, "max_model_len", None
    )

    packing_length = int(default_length or 0)
    if packing_length <= 0:
        raise ValueError(
            "packing is enabled but no valid packing_length/template.max_length is set"
        )

    buffer_size = int(cfg.get("packing_buffer", 512) or 512)
    min_fill_ratio = float(cfg.get("packing_min_fill_ratio", 0.65))
    if not (0 < min_fill_ratio <= 1):
        raise ValueError("packing_min_fill_ratio must be in (0,1]")
    drop_last = bool(cfg.get("packing_drop_last", True))
    allow_single_long = bool(cfg.get("packing_allow_single_long", True))
    eval_packing = bool(cfg.get("eval_packing", True))

    wait_timeout_raw = cfg.get("packing_wait_timeout_s", 7200.0)
    try:
        wait_timeout_s = float(wait_timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.packing_wait_timeout_s must be a numeric value (seconds)"
        ) from exc
    if not math.isfinite(wait_timeout_s):
        raise ValueError(
            f"training.packing_wait_timeout_s must be finite, got {wait_timeout_raw!r}"
        )
    if wait_timeout_s < 0:
        raise ValueError(
            "training.packing_wait_timeout_s must be >= 0 (set 0 to wait indefinitely)"
        )

    persist_every_raw = cfg.get("packing_length_cache_persist_every", None)
    length_cache_persist_every: int | None = None
    if persist_every_raw is not None:
        try:
            length_cache_persist_every = int(persist_every_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "training.packing_length_cache_persist_every must be an integer when set"
            ) from exc
        if length_cache_persist_every <= 0:
            raise ValueError(
                "training.packing_length_cache_persist_every must be > 0 when set"
            )

    precompute_workers_raw = cfg.get("packing_length_precompute_workers", 8)
    try:
        length_precompute_workers = int(precompute_workers_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.packing_length_precompute_workers must be an integer when set"
        ) from exc
    if length_precompute_workers <= 0:
        raise ValueError(
            "training.packing_length_precompute_workers must be > 0 (default: 8). "
            f"Got {precompute_workers_raw!r}."
        )

    return PackingRuntimeConfig(
        enabled=True,
        mode=mode,
        packing_length=packing_length,
        buffer_size=buffer_size,
        min_fill_ratio=min_fill_ratio,
        drop_last=drop_last,
        allow_single_long=allow_single_long,
        eval_packing=eval_packing,
        wait_timeout_s=wait_timeout_s,
        length_cache_persist_every=length_cache_persist_every,
        length_precompute_workers=length_precompute_workers,
    )


def _parse_encoded_sample_cache_config(
    training_cfg: Any, train_args: Any
) -> EncodedSampleCacheRuntimeConfig:
    cfg = dict((training_cfg or {}).get("encoded_sample_cache") or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return EncodedSampleCacheRuntimeConfig(enabled=False)

    root_dir_raw = cfg.get("root_dir")
    if root_dir_raw is None:
        output_dir_raw = getattr(train_args, "output_dir", None)
        if not output_dir_raw:
            raise ValueError(
                "training.output_dir must be set when training.encoded_sample_cache.enabled=true"
            )
        root_dir = str(
            Path(str(output_dir_raw)).resolve() / "cache" / "encoded_samples"
        )
    else:
        root_dir = str(Path(str(root_dir_raw)).resolve())
        if not root_dir:
            raise ValueError(
                "training.encoded_sample_cache.root_dir must be a non-empty string when provided"
            )

    policy = str(cfg.get("ineligible_policy", "error") or "error").strip().lower()
    if policy not in {"error", "bypass"}:
        raise ValueError(
            "training.encoded_sample_cache.ineligible_policy must be one of "
            "{'error', 'bypass'}"
        )

    wait_timeout_raw = cfg.get("wait_timeout_s", 7200)
    try:
        wait_timeout_s = float(wait_timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.encoded_sample_cache.wait_timeout_s must be a numeric value (seconds)"
        ) from exc
    if not math.isfinite(wait_timeout_s):
        raise ValueError(
            "training.encoded_sample_cache.wait_timeout_s must be finite, "
            f"got {wait_timeout_raw!r}"
        )
    if wait_timeout_s < 0:
        raise ValueError(
            "training.encoded_sample_cache.wait_timeout_s must be >= 0 "
            "(set 0 to wait indefinitely)"
        )

    max_resident_raw = cfg.get("max_resident_shards", 4)
    try:
        max_resident_shards = int(max_resident_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.encoded_sample_cache.max_resident_shards must be an integer"
        ) from exc
    if max_resident_shards <= 0:
        raise ValueError(
            "training.encoded_sample_cache.max_resident_shards must be > 0"
        )

    return EncodedSampleCacheRuntimeConfig(
        enabled=True,
        root_dir=root_dir,
        ineligible_policy=cast(Literal["error", "bypass"], policy),
        wait_timeout_s=wait_timeout_s,
        max_resident_shards=max_resident_shards,
    )


def _parse_static_packing_cache_config(
    training_cfg: Any,
) -> StaticPackingCacheRuntimeConfig:
    cfg = dict((training_cfg or {}).get("static_packing_cache") or {})
    root_dir_raw = cfg.get("root_dir")
    if root_dir_raw is None:
        return StaticPackingCacheRuntimeConfig(root_dir=None)

    root_dir = str(Path(str(root_dir_raw)).resolve())
    if not root_dir:
        raise ValueError(
            "training.static_packing_cache.root_dir must be a non-empty string when provided"
        )
    return StaticPackingCacheRuntimeConfig(root_dir=root_dir)


def _set_train_arg(train_args: Any, field: str, value: Any) -> None:
    setattr(train_args, field, value)
    nested = getattr(train_args, "training_args", None)
    if nested is not None:
        setattr(nested, field, value)


def _parse_checkpoint_mode(
    training_cfg: Any,
) -> Literal["artifact_only", "restartable"]:
    cfg = training_cfg or {}
    mode_raw = cfg.get("checkpoint_mode", "artifact_only")
    mode = str(mode_raw or "artifact_only").strip().lower()
    if mode not in {"artifact_only", "restartable"}:
        raise ValueError(
            "training.checkpoint_mode must be one of {'artifact_only', 'restartable'}"
        )
    return cast(Literal["artifact_only", "restartable"], mode)


def _apply_checkpoint_mode(
    train_args: Any,
    *,
    checkpoint_mode: Literal["artifact_only", "restartable"],
) -> None:
    _set_train_arg(train_args, "checkpoint_mode", str(checkpoint_mode))
    if checkpoint_mode != "restartable":
        return

    if bool(getattr(train_args, "save_only_model", False)):
        logger.info(
            "checkpoint_mode=restartable: forcing save_only_model=false so optimizer, scheduler, RNG, and trainer state are persisted."
        )
    _set_train_arg(train_args, "save_only_model", False)


def _resolve_resume_from_checkpoint(train_args: Any) -> str | None:
    for candidate in (train_args, getattr(train_args, "training_args", None)):
        value = getattr(candidate, "resume_from_checkpoint", None)
        if value is None:
            continue
        raw = str(value).strip()
        if raw:
            return raw
    return None


def _build_source_path_identity(path_hint: Any) -> dict[str, Any] | None:
    if path_hint is None:
        return None

    raw_path = str(path_hint).strip()
    if not raw_path:
        return None

    source_path = Path(raw_path).expanduser()
    resolved_path = source_path.resolve(strict=False)
    identity: dict[str, Any] = {
        "raw_path": raw_path,
        "resolved_path": str(resolved_path),
    }

    try:
        stat_result = resolved_path.stat()
    except OSError:
        identity["exists"] = False
        return identity

    identity["exists"] = True
    identity["size_bytes"] = int(stat_result.st_size)
    identity["mtime_ns"] = int(
        getattr(
            stat_result,
            "st_mtime_ns",
            int(stat_result.st_mtime * 1_000_000_000),
        )
    )
    if resolved_path.is_file():
        try:
            if int(stat_result.st_size) <= 64 * 1024 * 1024:
                sha256 = hashlib.sha256()
                with resolved_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        sha256.update(chunk)
                identity["sha256"] = sha256.hexdigest()
            else:
                identity["sha256_skipped_reason"] = "file_too_large"
        except OSError:
            identity["sha256_skipped_reason"] = "unreadable"
    return identity


def _build_data_source_provenance(
    *,
    split: str,
    dataset_jsonl: str | None,
    dataset_seed: int,
    sample_limit: int | None,
    sample_with_replacement: bool | None = None,
) -> dict[str, Any]:
    return {
        "split": str(split),
        "dataset_seed": int(dataset_seed),
        "dataset_jsonl": str(dataset_jsonl) if dataset_jsonl else None,
        "dataset_source_jsonl": _build_source_path_identity(dataset_jsonl),
        "sample_limit": int(sample_limit) if sample_limit is not None else None,
        "sample_with_replacement": bool(sample_with_replacement)
        if sample_with_replacement is not None
        else None,
    }


def _build_effective_runtime_payload(
    *,
    training_config: Any,
    train_args: Any,
    trainer_variant: str | None,
    dataset_seed: int,
    checkpoint_mode: Literal["artifact_only", "restartable"],
    packing_cfg: PackingRuntimeConfig,
    encoded_sample_cache_cfg: EncodedSampleCacheRuntimeConfig,
    train_jsonl: str | None,
    val_jsonl: str | None,
    pipeline_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    template_cfg = getattr(training_config, "template", {}) or {}
    return {
        "trainer_variant": str(trainer_variant or ""),
        "dataset_seed": int(dataset_seed),
        "run_name": str(getattr(train_args, "run_name", "") or ""),
        "output_dir": str(getattr(train_args, "output_dir", "") or ""),
        "logging_dir": str(getattr(train_args, "logging_dir", "") or ""),
        "checkpoint_mode": str(checkpoint_mode),
        "resume_from_checkpoint": _resolve_resume_from_checkpoint(train_args),
        "save_only_model": bool(getattr(train_args, "save_only_model", False)),
        "save_strategy": str(getattr(train_args, "save_strategy", "") or ""),
        "save_last_epoch": bool(getattr(train_args, "save_last_epoch", True)),
        "seed": int(getattr(train_args, "seed", 0) or 0),
        "per_device_train_batch_size": int(
            getattr(train_args, "per_device_train_batch_size", 1) or 1
        ),
        "per_device_eval_batch_size": int(
            getattr(train_args, "per_device_eval_batch_size", 1) or 1
        ),
        "gradient_accumulation_steps": int(
            getattr(train_args, "gradient_accumulation_steps", 1) or 1
        ),
        "max_steps": int(getattr(train_args, "max_steps", -1) or -1),
        "num_train_epochs": float(getattr(train_args, "num_train_epochs", 0.0) or 0.0),
        "dataloader_drop_last": bool(
            getattr(train_args, "dataloader_drop_last", False)
        ),
        "global_max_length": getattr(training_config, "global_max_length", None),
        "template_max_length": getattr(train_args, "max_model_len", None),
        "template_max_pixels": template_cfg.get("max_pixels")
        if isinstance(template_cfg, Mapping)
        else None,
        "packing": dataclass_asdict_no_none(packing_cfg),
        "encoded_sample_cache": dataclass_asdict_no_none(encoded_sample_cache_cfg),
        "dataset_source_train_jsonl": _build_source_path_identity(train_jsonl),
        "dataset_source_val_jsonl": _build_source_path_identity(val_jsonl),
        "pipeline_manifest_checksum": str(pipeline_manifest.get("checksum", ""))
        if isinstance(pipeline_manifest, Mapping)
        else "",
        "launcher": _collect_launcher_metadata_from_env(),
    }


def _recompute_gas_for_packing(
    *,
    train_args: Any,
    training_cfg: Mapping[str, Any],
    original_per_device_bs: int,
    original_gas: int,
    world_size: int,
) -> int:
    world_size = max(int(world_size), 1)
    effective_batch_raw = training_cfg.get("effective_batch_size")

    if effective_batch_raw is not None:
        try:
            requested_global = int(effective_batch_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "training.effective_batch_size must be an integer when packing is enabled"
            ) from exc
        if requested_global <= 0:
            raise ValueError(
                f"training.effective_batch_size must be > 0, got {effective_batch_raw!r}"
            )
        if requested_global % world_size != 0:
            raise ValueError(
                "training.effective_batch_size must be divisible by world_size when packing is enabled. "
                f"Got effective_batch_size={requested_global}, world_size={world_size}."
            )
        source = "training.effective_batch_size"
    else:
        requested_global = max(
            1,
            int(original_per_device_bs) * int(original_gas) * int(world_size),
        )
        source = "pre-adjust global effective batch"

    new_gas = max(1, math.ceil(requested_global / world_size))
    realized_global = int(new_gas) * int(world_size)

    current_gas = int(getattr(train_args, "gradient_accumulation_steps", 1) or 1)
    if current_gas != new_gas:
        _set_train_arg(train_args, "gradient_accumulation_steps", int(new_gas))
        logger.warning(
            "packing enabled: recomputed gradient_accumulation_steps=%s after forcing per_device_train_batch_size=1 (%s=%s, world_size=%s, previous_gas=%s)",
            new_gas,
            source,
            requested_global,
            world_size,
            current_gas,
        )

    logger.info(
        "packing effective batch is exact: requested_global_packs_per_step=%s source=%s world_size=%s realized_global_packs_per_step=%s",
        requested_global,
        source,
        world_size,
        realized_global,
    )

    return int(new_gas)


def _validate_static_packing_accumulation_windows(
    *,
    packing_cfg: PackingRuntimeConfig,
    trainer_variant: str | None,
    per_rank_batches_est: int,
    gradient_accumulation_steps: int,
    world_size: int,
    dataloader_drop_last: bool,
) -> None:
    if (not packing_cfg.enabled) or packing_cfg.mode != "static":
        return

    if str(trainer_variant or "") in {"stage2_two_channel", "stage2_rollout_aligned"}:
        return

    gas = max(1, int(gradient_accumulation_steps))
    if gas <= 1:
        return

    per_rank_batches = max(0, int(per_rank_batches_est))
    world_size_i = max(1, int(world_size))
    drop_last_flag = bool(dataloader_drop_last)
    remainder = int(per_rank_batches % gas)

    if per_rank_batches < gas:
        logger.warning(
            "static packing accumulation alignment: per_rank_batches_est=%s is smaller than gradient_accumulation_steps=%s (world_size=%s, dataloader_drop_last=%s). Trainer will flush a partial accumulation window at epoch end; packs/optimizer-step will be below target in this epoch.",
            per_rank_batches,
            gas,
            world_size_i,
            drop_last_flag,
        )
        return

    if remainder != 0:
        logger.warning(
            "static packing accumulation alignment: per_rank_batches_est=%s is not divisible by gradient_accumulation_steps=%s (remainder=%s, world_size=%s, dataloader_drop_last=%s). Trainer will flush a partial accumulation window at epoch end; packs/optimizer-step will be inexact for boundary steps.",
            per_rank_batches,
            gas,
            remainder,
            world_size_i,
            drop_last_flag,
        )


def _validate_stage2_step_budget_windows(
    *,
    trainer_variant: str | None,
    packing_enabled: bool,
    per_rank_batches_est: int,
    gradient_accumulation_steps: int,
    dataloader_drop_last: bool,
) -> None:
    """Fail-fast guard for Stage2 step-budgeted accumulation windows.

    Stage2 executes forward/backward only on the last micro-step of each
    accumulation window. Underfull windows would otherwise allow optimizer.step()
    with stale/zero gradients.
    """
    if str(trainer_variant or "") != "stage2_two_channel":
        return
    if not bool(packing_enabled):
        return

    gas = max(1, int(gradient_accumulation_steps))
    if gas <= 1:
        return

    per_rank_batches = max(0, int(per_rank_batches_est))
    if per_rank_batches < gas:
        raise ValueError(
            "stage2-ab requires per-rank batches >= gradient_accumulation_steps. "
            f"Got per_rank_batches_est={int(per_rank_batches)} "
            f"but gradient_accumulation_steps={int(gas)}. "
            "Mitigations: increase custom.train_sample_limit, reduce world_size, "
            "or reduce training.effective_batch_size."
        )

    remainder = int(per_rank_batches % gas)
    if (not bool(dataloader_drop_last)) and remainder != 0:
        raise ValueError(
            "stage2-ab step-budgeted mode does not support a partial gradient-accumulation window. "
            f"Got dataloader_drop_last=false with per_rank_batches_est={int(per_rank_batches)} and "
            f"gradient_accumulation_steps={int(gas)} (remainder={int(remainder)}). "
            "Mitigations: set training.dataloader_drop_last=true (recommended), or adjust "
            "dataset size/world_size so per-rank batches is a multiple of gradient_accumulation_steps."
        )


def _build_static_packing_fingerprint(
    *,
    training_config: Any,
    custom_config: Any,
    template: Any,
    train_args: Any,
    dataset_seed: int,
    packing_cfg: PackingRuntimeConfig,
    train_jsonl: str | None,
    dataset_split: str = "train",
    eval_sample_limit: int | None = None,
    eval_sample_with_replacement: bool | None = None,
) -> dict[str, Any]:
    template_cfg = getattr(training_config, "template", {}) or {}
    training_cfg = getattr(training_config, "training", {}) or {}
    coord_tokens_payload = _coord_tokens_fingerprint_payload(custom_config)
    prompt_identity = _resolve_dense_prompt_identity(custom_config)

    split = str(dataset_split or "train").strip().lower()
    if split not in {"train", "eval"}:
        raise ValueError(
            f"dataset_split must be one of {{'train', 'eval'}}, got {dataset_split!r}"
        )

    return {
        "dataset_seed": int(dataset_seed),
        "dataset_split": split,
        "packing_mode": packing_cfg.mode,
        "packing_length": int(packing_cfg.packing_length),
        "global_max_length": getattr(training_config, "global_max_length", None),
        "template_max_length": getattr(template, "max_length", None),
        "train_args_max_model_len": getattr(train_args, "max_model_len", None),
        "template_system": template_cfg.get("system")
        if isinstance(template_cfg, Mapping)
        else None,
        "template_truncation_strategy": template_cfg.get("truncation_strategy")
        if isinstance(template_cfg, Mapping)
        else None,
        "custom_user_prompt": getattr(custom_config, "user_prompt", None),
        "custom_emit_norm": getattr(custom_config, "emit_norm", None),
        # Preserve the legacy null-valued fusion keys so older static-packing
        # caches remain addressable after fusion was disabled in the schema.
        "custom_fusion_config": getattr(custom_config, "fusion_config", None),
        "custom_json_format": getattr(custom_config, "json_format", None),
        "custom_bbox_format": getattr(custom_config, "bbox_format", None),
        "custom_object_ordering": getattr(custom_config, "object_ordering", None),
        "custom_object_field_order": getattr(custom_config, "object_field_order", None),
        "custom_prompt_variant": prompt_identity["prompt_variant"],
        "custom_prompt_template_hash": prompt_identity["prompt_template_hash"],
        "custom_coord_mode": "coord_tokens",
        "custom_use_summary": bool(getattr(custom_config, "use_summary", False)),
        "custom_offline_max_pixels": getattr(custom_config, "offline_max_pixels", None),
        "coord_tokens": coord_tokens_payload,
        "dataset_jsonl": str(train_jsonl) if train_jsonl else None,
        "custom_train_jsonl": str(train_jsonl) if train_jsonl else None,
        "dataset_source_fusion_config": _build_source_path_identity(
            getattr(custom_config, "fusion_config", None)
        ),
        "dataset_source_jsonl": _build_source_path_identity(train_jsonl),
        "dataset_source_train_jsonl": _build_source_path_identity(train_jsonl),
        "eval_sample_limit": int(eval_sample_limit)
        if split == "eval" and eval_sample_limit is not None
        else None,
        "eval_sample_with_replacement": bool(eval_sample_with_replacement)
        if split == "eval" and eval_sample_with_replacement is not None
        else None,
        "system_prompt_dense": getattr(custom_config, "system_prompt_dense", None),
        "system_prompt_summary": getattr(custom_config, "system_prompt_summary", None),
        "train_dataloader_shuffle": training_cfg.get("train_dataloader_shuffle")
        if isinstance(training_cfg, Mapping)
        else None,
    }


def _normalize_optional_sample_limit(sample_limit: Any) -> int | None:
    if isinstance(sample_limit, int):
        return int(sample_limit)
    if isinstance(sample_limit, str) and sample_limit.isdigit():
        return int(sample_limit)
    return None


def _coord_tokens_fingerprint_payload(custom_config: Any) -> Any:
    coord_tokens_cfg = getattr(custom_config, "coord_tokens", None)
    if coord_tokens_cfg is None:
        return None
    if is_dataclass(coord_tokens_cfg):
        return dataclass_asdict_no_none(coord_tokens_cfg)
    if isinstance(coord_tokens_cfg, Mapping):
        return dict(coord_tokens_cfg)
    return None


def _validate_bbox_format_contract(
    *,
    custom_config: Any,
    trainer_variant: str | None,
) -> None:
    bbox_format = str(getattr(custom_config, "bbox_format", "xyxy") or "xyxy").strip().lower()
    variant = str(trainer_variant or "").strip()
    if variant in {"stage2_two_channel", "stage2_rollout_aligned"} and bbox_format != "xyxy":
        raise ValueError(
            "custom.bbox_format=cxcywh is currently unsupported for stage-2 trainer variants. "
            "Stage-2 target construction still assumes canonical xyxy ordering; use custom.bbox_format=xyxy."
        )


def _build_encoded_sample_cache_fingerprint(
    *,
    training_config: Any,
    custom_config: Any,
    template: Any,
    train_args: Any,
    dataset_seed: int,
    dataset_jsonl: str | None,
    dataset_split: str,
    dataset_mode: str,
    sample_limit: int | None = None,
    system_prompt_dense: str | None = None,
    system_prompt_summary: str | None = None,
) -> dict[str, Any]:
    template_cfg = getattr(training_config, "template", {}) or {}
    prompt_identity = _resolve_dense_prompt_identity(custom_config)
    split = str(dataset_split or "train").strip().lower()
    if split not in {"train", "eval"}:
        raise ValueError(
            f"dataset_split must be one of {{'train', 'eval'}}, got {dataset_split!r}"
        )

    coord_tokens_payload = _coord_tokens_fingerprint_payload(custom_config)

    return {
        "cache_schema_version": 1,
        "dataset_seed": int(dataset_seed),
        "dataset_split": split,
        "dataset_mode": str(dataset_mode),
        "dataset_jsonl": str(dataset_jsonl) if dataset_jsonl else None,
        "dataset_source_jsonl": _build_source_path_identity(dataset_jsonl),
        "sample_limit": int(sample_limit) if sample_limit is not None else None,
        "global_max_length": getattr(training_config, "global_max_length", None),
        "template_max_length": getattr(template, "max_length", None),
        "train_args_max_model_len": getattr(train_args, "max_model_len", None),
        "template_system": template_cfg.get("system")
        if isinstance(template_cfg, Mapping)
        else None,
        "template_truncation_strategy": template_cfg.get("truncation_strategy")
        if isinstance(template_cfg, Mapping)
        else None,
        "custom_user_prompt": getattr(custom_config, "user_prompt", None),
        "custom_emit_norm": getattr(custom_config, "emit_norm", None),
        "custom_json_format": getattr(custom_config, "json_format", None),
        "custom_bbox_format": getattr(custom_config, "bbox_format", None),
        "custom_object_ordering": getattr(custom_config, "object_ordering", None),
        "custom_object_field_order": getattr(custom_config, "object_field_order", None),
        "custom_prompt_variant": prompt_identity["prompt_variant"],
        "custom_prompt_template_hash": prompt_identity["prompt_template_hash"],
        "custom_coord_mode": "coord_tokens",
        "custom_use_summary": bool(getattr(custom_config, "use_summary", False)),
        "custom_offline_max_pixels": getattr(custom_config, "offline_max_pixels", None),
        "coord_tokens": coord_tokens_payload,
        "system_prompt_dense": system_prompt_dense,
        "system_prompt_summary": system_prompt_summary,
    }


def _build_encoded_sample_cache_request(
    *,
    runtime_cfg: EncodedSampleCacheRuntimeConfig,
    training_config: Any,
    custom_config: Any,
    template: Any,
    train_args: Any,
    dataset_seed: int,
    dataset_jsonl: str | None,
    dataset_split: str,
    dataset_mode: str,
    sample_limit: int | None = None,
    system_prompt_dense: str | None = None,
    system_prompt_summary: str | None = None,
) -> dict[str, Any] | None:
    if not runtime_cfg.enabled:
        return None

    fingerprint = _build_encoded_sample_cache_fingerprint(
        training_config=training_config,
        custom_config=custom_config,
        template=template,
        train_args=train_args,
        dataset_seed=dataset_seed,
        dataset_jsonl=dataset_jsonl,
        dataset_split=dataset_split,
        dataset_mode=dataset_mode,
        sample_limit=sample_limit,
        system_prompt_dense=system_prompt_dense,
        system_prompt_summary=system_prompt_summary,
    )
    fingerprint_sha256 = hashlib.sha256(
        json.dumps(
            fingerprint, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    cache_dir = Path(str(runtime_cfg.root_dir)) / fingerprint_sha256
    return {
        "enabled": True,
        "root_dir": str(runtime_cfg.root_dir),
        "ineligible_policy": runtime_cfg.ineligible_policy,
        "wait_timeout_s": float(runtime_cfg.wait_timeout_s),
        "max_resident_shards": int(runtime_cfg.max_resident_shards),
        "dataset_split": str(dataset_split),
        "dataset_jsonl": str(dataset_jsonl) if dataset_jsonl else None,
        "fingerprint": fingerprint,
        "fingerprint_sha256": fingerprint_sha256,
        "cache_dir": str(cache_dir),
    }


def _resolve_static_packing_cache_dir(
    *,
    runtime_cfg: StaticPackingCacheRuntimeConfig,
    training_config: Any,
    train_args: Any,
    dataset_jsonl: str | None,
    fusion_config_path: str | None,
    dataset_split: str,
    packing_cfg: PackingRuntimeConfig,
) -> Path:
    if runtime_cfg.root_dir is not None:
        base_root = Path(str(runtime_cfg.root_dir))
        source = "configured"
    elif dataset_jsonl:
        base_root = Path(str(dataset_jsonl)).expanduser().resolve(strict=False).parent
        base_root = base_root / "cache" / "static_packing"
        source = "dataset_jsonl"
    elif fusion_config_path:
        base_root = (
            Path(str(fusion_config_path)).expanduser().resolve(strict=False).parent
            / "cache"
            / "static_packing"
        )
        source = "fusion_config"
    else:
        output_dir_raw = getattr(train_args, "output_dir", None)
        if not output_dir_raw:
            raise ValueError(
                "training.output_dir must be set when training.packing_mode=static and no dataset/fusion path is available"
            )
        base_root = Path(str(output_dir_raw)).resolve() / "static_packing_auto"
        source = "output_dir"

    global_max_length_raw = getattr(training_config, "global_max_length", None)
    try:
        global_max_length = int(global_max_length_raw)
    except (TypeError, ValueError):
        global_max_length = int(packing_cfg.packing_length)
    if global_max_length <= 0:
        global_max_length = int(packing_cfg.packing_length)

    split = str(dataset_split or "train").strip().lower()
    if split not in {"train", "eval"}:
        raise ValueError(
            f"dataset_split must be one of {{'train', 'eval'}}, got {dataset_split!r}"
        )

    cache_dir = (
        base_root
        / f"global_max_length_{global_max_length}"
        / ("train" if split == "train" else "eval")
    )
    logger.info(
        "Static packing cache base resolved: split=%s source=%s base_root=%s cache_dir=%s",
        split,
        source,
        base_root,
        cache_dir,
    )
    return cache_dir


def _build_encoded_sample_cache_bypass_info(
    request: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "bypassed",
        "reason": str(reason),
        "policy": str(request.get("ineligible_policy") or "error"),
        "wait_timeout_s": float(request.get("wait_timeout_s", 7200.0) or 0.0),
        "dataset_split": str(request.get("dataset_split") or "train"),
        "dataset_jsonl": request.get("dataset_jsonl"),
        "fingerprint": dict(request.get("fingerprint") or {}),
        "fingerprint_sha256": request.get("fingerprint_sha256"),
        "root_dir": request.get("root_dir"),
        "cache_dir": request.get("cache_dir"),
        "manifest_path": str(
            Path(str(request.get("cache_dir") or ".")) / "manifest.json"
        ),
    }


def _append_dataset_epoch_callback(callbacks: list[Any], dataset: Any) -> list[Any]:
    if not callable(getattr(dataset, "set_epoch", None)):
        return callbacks
    from .callbacks import DatasetEpochCallback

    callbacks.append(DatasetEpochCallback(dataset))
    return callbacks


def _attach_encoded_sample_cache_run_metadata(
    meta: dict[str, Any],
    *,
    train_cache_info: Mapping[str, Any] | None,
    eval_cache_info: Mapping[str, Any] | None,
) -> None:
    _attach_encoded_sample_cache_run_metadata_impl(
        meta,
        train_cache_info=train_cache_info,
        eval_cache_info=eval_cache_info,
    )


def _is_rollout_matching_variant(trainer_variant: str | None) -> bool:
    return str(trainer_variant or "") in {
        "stage2_rollout_aligned",
        "stage2_two_channel",
    }


def _validate_stage1_static_packing_policy(
    *,
    packing_cfg: PackingRuntimeConfig,
    trainer_variant: str | None,
) -> None:
    if not packing_cfg.enabled:
        return
    if _is_rollout_matching_variant(trainer_variant):
        return

    if packing_cfg.mode != "static":
        raise ValueError(
            "training.packing_mode=dynamic is deprecated and unsupported for Stage-1 dataset-level packing. "
            "Use training.packing_mode=static."
        )


def _validate_attention_backend_for_packing(*, training_config: Any) -> None:
    """Fail-fast guard: packed training requires a padding-free-safe attention backend.

    CoordExp relies on packing as the primary efficiency lever. In practice, our
    packing paths assume a flash-attn style implementation (padding-free / varlen).
    Misconfiguration here can lead to silent correctness or performance issues.
    """

    training_section = getattr(training_config, "training", None)
    packing_enabled = False
    if isinstance(training_section, Mapping):
        packing_enabled = bool(training_section.get("packing", False))

    if not packing_enabled:
        return

    model_section = getattr(training_config, "model", None)
    attn_impl_raw = None
    if isinstance(model_section, Mapping):
        attn_impl_raw = model_section.get("attn_impl")

    attn_impl = str(attn_impl_raw or "").strip().lower()
    allowed = {"flash_attention_2", "flash_attn_2"}

    if attn_impl not in allowed:
        raise ValueError(
            "training.packing=true requires model.attn_impl to be a flash-attn backend for padding-free packed training. "
            f"Got model.attn_impl={attn_impl_raw!r}. Allowed: {sorted(allowed)}. "
            "Fix: set model.attn_impl: flash_attention_2 (recommended) or disable training.packing."
        )


def _parse_sample_size(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        size = value
    elif isinstance(value, str) and value.isdigit():
        size = int(value)
    else:
        raise ValueError(f"{field_name} must be an int or digit string, got {value!r}")
    if size <= 0:
        raise ValueError(f"{field_name} must be > 0, got {size}")
    return size


def _resolve_dataset_seed(*, training_config: Any, train_args: Any) -> int:
    """Resolve the dataset RNG seed for JSONL sampling / ordering.

    Contract:
    - Prefer the YAML-specified `training.seed` (single source of truth).
    - Fallback to `train_args.seed` when training config seed is missing.

    This avoids hidden nondeterminism from hard-coded constants and ensures dataset
    sampling is coupled to the run seed used elsewhere in the training stack.
    """

    training_section = getattr(training_config, "training", None)
    seed_raw = None
    if isinstance(training_section, Mapping):
        seed_raw = training_section.get("seed")

    if seed_raw is None:
        seed_raw = getattr(train_args, "seed", None)

    if seed_raw is None:
        logger.warning("training.seed is missing; defaulting dataset_seed=42")
        return 42

    try:
        seed = int(seed_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"training.seed must be an int (or int-like); got {seed_raw!r}"
        ) from exc

    return seed


def _collect_dependency_provenance() -> dict[str, Any]:
    return _collect_dependency_provenance_impl()


def _collect_launcher_metadata_from_env() -> dict[str, str]:
    return _collect_launcher_metadata_from_env_impl()


def _apply_rollout_decode_batch_size_override(
    *, train_args: Any, training_config: Any
) -> int:
    """Override eval batching for rollout-aware trainer variants.

    Rollout-aware variants use `rollout_matching.eval_decode_batch_size` as the
    eval-step decode microbatch source of truth. We mirror that by forcing
    `per_device_eval_batch_size` to the resolved eval decode batch size.

    Returns the resolved eval decode batch size.
    """

    trainer_variant = getattr(train_args, "trainer_variant", None)
    if trainer_variant not in {"stage2_rollout_aligned", "stage2_two_channel"}:
        return 1

    rollout_cfg_obj = getattr(training_config, "rollout_matching", None)
    if rollout_cfg_obj is None:
        rollout_cfg_for_batch: Any = {}
    elif is_dataclass(rollout_cfg_obj):
        rollout_cfg_for_batch = dataclass_asdict_no_none(rollout_cfg_obj)
    else:
        rollout_cfg_for_batch = rollout_cfg_obj

    if rollout_cfg_for_batch is None:
        rollout_cfg_for_batch = {}
    if not isinstance(rollout_cfg_for_batch, Mapping):
        raise TypeError("rollout_matching must be a mapping when provided")

    eval_decode_bs_raw = rollout_cfg_for_batch.get("eval_decode_batch_size", None)
    if eval_decode_bs_raw is None:
        raise ValueError(
            "rollout_matching.eval_decode_batch_size must be provided explicitly"
        )
    try:
        rollout_eval_decode_bs = int(eval_decode_bs_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "rollout_matching.eval_decode_batch_size must be an int"
        ) from exc
    if rollout_eval_decode_bs <= 0:
        raise ValueError("rollout_matching.eval_decode_batch_size must be > 0")

    if getattr(train_args, "training_args", None) is not None:
        current_eval_bs_raw = getattr(
            train_args.training_args,
            "per_device_eval_batch_size",
            rollout_eval_decode_bs,
        )
        try:
            current_eval_bs = int(current_eval_bs_raw)
        except (TypeError, ValueError):
            current_eval_bs = int(rollout_eval_decode_bs)
        if int(current_eval_bs) != int(rollout_eval_decode_bs):
            logger.warning(
                "Overriding per_device_eval_batch_size=%s with rollout eval_decode_batch_size=%s for rollout trainer variants.",
                int(current_eval_bs),
                int(rollout_eval_decode_bs),
            )
        train_args.training_args.per_device_eval_batch_size = int(
            rollout_eval_decode_bs
        )

    setattr(train_args, "per_device_eval_batch_size", int(rollout_eval_decode_bs))
    return int(rollout_eval_decode_bs)


def parse_args():
    """Parse minimal runtime arguments.

    All training configuration comes from YAML files.
    CLI only accepts runtime settings like config path and debug mode.
    """
    parser = argparse.ArgumentParser(
        description="SFT training with YAML configuration - zero CLI hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with config
  python -m src.sft --config configs/qwen3vl_lora.yaml
  
  # With inheritance from base config
  python -m src.sft --config configs/qwen3vl_lora.yaml --base_config configs/base.yaml
  
  # Debug mode
  python -m src.sft --config configs/debug.yaml --debug
        """,
    )

    # Required: config file path
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (required)",
    )

    # Optional: base config for inheritance
    parser.add_argument(
        "--base_config",
        type=str,
        default=None,
        help="Path to base YAML config for inheritance (optional)",
    )

    # Runtime: debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and print full config",
    )

    # Runtime: verbose mode (all ranks log)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable logging from all ranks in distributed training",
    )

    return parser.parse_args()


def _build_pipeline_manifest(
    cfg: Mapping[str, Any] | None,
    *,
    default_objective: list[str],
    default_diagnostics: list[str],
    trainer_variant: str,
    config_path: str,
    run_name: str,
    seed: int,
    coord_soft_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_pipeline_manifest(
        cfg,
        default_objective=default_objective,
        default_diagnostics=default_diagnostics,
        trainer_variant=trainer_variant,
        config_path=config_path,
        run_name=run_name,
        seed=seed,
        coord_soft_cfg=coord_soft_cfg,
    )


def _scope_logging_dir_under_run_name(train_args: Any) -> str | None:
    """Nest explicit TensorBoard roots under `run_name`.

    ms-swift versions `output_dir` when `add_version` is enabled, but it leaves an
    authored `logging_dir` untouched. Without this alignment, `tensorboard --logdir`
    sees event files directly under the root and labels the run as `.` instead of
    the authored `training.run_name`.
    """
    training_args = getattr(train_args, "training_args", None)
    run_name = getattr(train_args, "run_name", None)
    if not run_name and training_args is not None:
        run_name = getattr(training_args, "run_name", None)

    base_logging_dir = getattr(train_args, "logging_dir", None)
    if base_logging_dir is None and training_args is not None:
        base_logging_dir = getattr(training_args, "logging_dir", None)

    if not run_name or not base_logging_dir:
        return None

    run_name_str = str(run_name)
    base_logging_dir_str = str(base_logging_dir)
    base_logging_dir_norm = os.path.normpath(base_logging_dir_str)
    if os.path.basename(base_logging_dir_norm) == run_name_str:
        final_logging_dir = base_logging_dir_str
    else:
        final_logging_dir = os.path.join(base_logging_dir_str, run_name_str)
        setattr(train_args, "logging_dir", final_logging_dir)
        if training_args is not None:
            setattr(training_args, "logging_dir", final_logging_dir)
    return final_logging_dir


def _strip_trailing_trainer_state_logging_row(logging_path: str | Path) -> bool:
    """Remove the final ms-swift trainer-state append from a flat metrics JSONL.

    Upstream ms-swift appends a final status payload with `log_history`, checkpoint
    pointers, and memory to the same `logging.jsonl` stream used for flat metric rows.
    That payload is valid JSON but not a metric event, so downstream JSONL viewers
    and editors render it poorly. Keep the metrics stream flat by dropping only that
    trailing trainer-state object after training finishes.
    """

    path = Path(logging_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return False

    if not lines:
        return False

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False

    if not isinstance(payload, Mapping):
        return False
    if not isinstance(payload.get("log_history"), list):
        return False
    if "global_step" not in payload:
        return False
    if any("/" in str(key) for key in payload):
        return False
    if any(key in payload for key in ("loss", "step", "global_step/max_steps")):
        return False

    sanitized = "\n".join(lines[:-1])
    if sanitized:
        sanitized += "\n"
    path.write_text(sanitized, encoding="utf-8")
    return True


@_torch_elastic_record
def main():
    """Main training entry point - pure config-driven."""
    args = parse_args()
    config_path = os.path.abspath(str(args.config))

    # Configure logging based on runtime flags
    if args.verbose:
        enable_verbose_logging()

    if args.debug:
        set_log_level(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("  MS-Swift Training with YAML Configuration")
    logger.info("=" * 70)
    logger.info(f"Config file: {args.config}")
    if args.base_config:
        logger.info(f"Base config: {args.base_config}")
    logger.info("=" * 70)

    # Load configuration from YAML
    logger.info("Loading configuration...")
    train_args, training_config = ConfigLoader.load_training_config(
        args.config, args.base_config
    )
    # Ensure custom optimizer variant is available before trainer setup
    register_coord_offset_optimizer()
    custom_config = training_config.custom
    debug_config = getattr(training_config, "debug", None)
    # Keep directory targets aligned across ms-swift wrappers.
    run_name = getattr(train_args, "run_name", None)
    training_args = getattr(train_args, "training_args", None)
    debug_output_override_applied = False

    def _set_train_dir_attr(attr_name: str, value: str) -> None:
        setattr(train_args, attr_name, value)
        if training_args is not None:
            setattr(training_args, attr_name, value)

    # Debug: collapse output_dir + logging_dir into a single folder for easy cleanup.
    if debug_config is not None and getattr(debug_config, "enabled", False):
        debug_output_dir = getattr(debug_config, "output_dir", None)
        if debug_output_dir:
            debug_output_dir_s = str(debug_output_dir)
            logger.warning(
                "Debug output override enabled: setting output_dir=logging_dir=%s",
                debug_output_dir_s,
            )
            _set_train_dir_attr("output_dir", debug_output_dir_s)
            _set_train_dir_attr("logging_dir", debug_output_dir_s)
            debug_output_override_applied = True

    if run_name and not debug_output_override_applied:
        _scope_logging_dir_under_run_name(train_args)

    # Optional: mirror logs into output_dir for quick review (rank 0 only).
    try:
        file_log_raw = getattr(custom_config, "extra", {}).get("log_file")
        if isinstance(file_log_raw, dict):
            file_log_cfg = FileLoggingConfig(
                enabled=bool(file_log_raw.get("enabled", True)),
                filename=str(file_log_raw.get("filename", "train.log")),
                capture_stdout=bool(file_log_raw.get("capture_stdout", False)),
                capture_stderr=bool(file_log_raw.get("capture_stderr", False)),
            )
        else:
            file_log_cfg = FileLoggingConfig(enabled=False)
        log_path = enable_output_dir_file_logging(
            getattr(train_args, "output_dir", "") or "", file_log_cfg
        )
        if log_path:
            logger.info(f"File logging enabled: {log_path}")
    except (OSError, RuntimeError, TypeError, ValueError):
        raise

    # Debug mode: print full configuration
    if args.debug:
        logger.debug("TrainArguments:")
        for key, value in vars(train_args).items():
            if not key.startswith("_"):
                logger.debug(f"  {key}: {value}")
        logger.debug("Training configuration sections:")
        logger.debug(f"  model={training_config.model}")
        logger.debug(f"  data={training_config.data}")
        logger.debug(f"  template={training_config.template}")
        logger.debug(f"  training={training_config.training}")
        logger.debug(f"  tuner={training_config.tuner}")
        logger.debug(f"  rlhf={training_config.rlhf}")
        logger.debug(f"  deepspeed={training_config.deepspeed}")
        logger.debug(f"  debug={getattr(training_config, 'debug', None)}")
        logger.debug(f"  prompts={training_config.prompts}")
        logger.debug("Custom dataset config:")
        for key, value in asdict(custom_config).items():
            logger.debug(f"  {key}: {value}")
        logger.debug("=" * 70)

    # Auto-configure ROOT_IMAGE_DIR from the training JSONL path.
    train_jsonl = custom_config.train_jsonl or custom_config.extra.get("jsonl")
    if not train_jsonl:
        raise ValueError(
            "Config must specify 'custom.train_jsonl'/'custom.jsonl'"
        )

    if os.environ.get("ROOT_IMAGE_DIR") in (None, ""):
        root_dir = os.path.abspath(os.path.dirname(str(train_jsonl)))
        os.environ["ROOT_IMAGE_DIR"] = root_dir
        logger.info(f"Set ROOT_IMAGE_DIR={root_dir} (from custom.train_jsonl)")

    # Initialize SwiftSft with TrainArguments object directly
    logger.info("Initializing ms-swift pipeline...")
    rlhf_type = getattr(train_args, "rlhf_type", None)
    pipeline_cls = SwiftRLHF if rlhf_type else SwiftSft
    sft = pipeline_cls(train_args)
    coord_cfg = custom_config.coord_tokens
    if coord_cfg.enabled:
        apply_coord_template_adapter(sft.template, coord_cfg)

    coord_offset_cfg = getattr(custom_config, "coord_offset", None)
    if coord_offset_cfg and coord_offset_cfg.enabled:
        adapter = install_coord_offset_adapter(
            sft.model,
            coord_ids=coord_offset_cfg.ids or None,
            tie_head=getattr(coord_offset_cfg, "tie_head", True),
            dtype=coord_offset_cfg.dtype,
        )
        modules_to_save: list[str] = list(
            getattr(train_args, "modules_to_save", []) or []
        )
        if adapter.module_name not in modules_to_save:
            modules_to_save.append(adapter.module_name)
            setattr(train_args, "modules_to_save", modules_to_save)
        # Sanity check against vocab size when available
        vocab_size = getattr(getattr(sft.model, "config", None), "vocab_size", None)
        max_id = int(adapter.coord_ids.max().item())
        if isinstance(vocab_size, int) and max_id >= vocab_size:
            raise ValueError(
                f"coord_offset id {max_id} exceeds model vocab_size={vocab_size}. "
                "Adjust coord_offset.ids to fit the loaded tokenizer."
            )
        logger.info(
            f"Coord-offset adapter enabled: ids={adapter.coord_ids.numel()}, "
            f"embed_lr={coord_offset_cfg.embed_lr or getattr(train_args, 'learning_rate', None)}, "
            f"head_lr={coord_offset_cfg.head_lr or getattr(train_args, 'learning_rate', None)}, "
            f"tie_head={getattr(coord_offset_cfg, 'tie_head', True)}, "
            f"dtype={coord_offset_cfg.dtype or 'auto'}"
        )
    logger.info(f"Model: {train_args.model}")
    logger.info(f"Training type: {train_args.train_type}")
    if rlhf_type:
        logger.info(f"RLHF mode: {rlhf_type}")
    if train_args.train_type == "lora":
        logger.info(
            f"LoRA rank: {train_args.lora_rank}, alpha: {train_args.lora_alpha}"
        )

    # Early validation: ensure teacher/student vocabulary compatibility in GKD mode
    if rlhf_type == "gkd":
        teacher_model = getattr(sft, "teacher_model", None)
        if teacher_model is None:
            raise ValueError(
                "GKD mode requires a teacher_model. Set rlhf.teacher_model in the YAML and ensure it loads."
            )
        student_vocab = getattr(getattr(sft.model, "config", None), "vocab_size", None)
        teacher_vocab = getattr(
            getattr(teacher_model, "config", None), "vocab_size", None
        )
        if isinstance(student_vocab, int) and isinstance(teacher_vocab, int):
            if student_vocab != teacher_vocab:
                raise ValueError(
                    "Teacher/student tokenizer vocabulary size mismatch detected: "
                    f"expected {student_vocab}, got {teacher_vocab}. "
                    "Use a teacher checkpoint with a matching tokenizer/vocabulary (e.g., the same Qwen3‑VL family)."
                )

    # NOTE: Do NOT override processor normalization/rescale.
    # Qwen3-VL expects its native image preprocessing. We already pass do_resize=False at encode time.

    # Augmentation is deprecated in this repo: keep it disabled unconditionally.
    # This avoids optional external dependency failures in data loading.
    augmenter = None
    bypass_prob = float(custom_config.bypass_prob)
    if custom_config.augmentation:
        logger.warning(
            "custom.augmentation is configured but ignored because augmentation is disabled in this repo."
        )

    curriculum_state = None
    curriculum_scheduler = None
    if custom_config.augmentation_curriculum:
        logger.warning(
            "custom.augmentation_curriculum is configured but ignored because augmentation is disabled."
        )

    # Sample limits for quick smoke tests.
    #
    # When debug.enabled=true, use debug.{train,val}_sample_limit (optional) and
    # ignore custom.* limits. Otherwise use custom.{train,val}_sample_limit with
    # no shared fallback (explicit is better than implicit).
    debug_enabled = bool(
        debug_config is not None and getattr(debug_config, "enabled", False)
    )
    heartbeat_env_raw = str(os.environ.get("COORDEXP_TRAIN_HEARTBEAT", "")).strip()
    heartbeat_env = heartbeat_env_raw.lower()
    heartbeat_enabled = debug_enabled or heartbeat_env in {
        "1",
        "true",
        "yes",
        "on",
    }
    if heartbeat_env and heartbeat_env not in {
        "0",
        "false",
        "no",
        "off",
        "1",
        "true",
        "yes",
        "on",
    }:
        logger.warning(
            "Unrecognized COORDEXP_TRAIN_HEARTBEAT=%r; expected one of "
            "{0,1,false,true,no,yes,off,on}. Treating as disabled unless debug.enabled=true.",
            heartbeat_env_raw,
        )
    if debug_enabled:
        train_sample_limit = getattr(debug_config, "train_sample_limit", None)
        val_sample_limit = getattr(debug_config, "val_sample_limit", None)
        sample_limit_ns = "debug"
        if train_sample_limit is None and val_sample_limit is None:
            logger.warning(
                "debug.enabled=true but no debug.{train,val}_sample_limit set; "
                "dataset will not be sample-limited."
            )
    else:
        train_sample_limit = custom_config.train_sample_limit
        val_sample_limit = custom_config.val_sample_limit
        sample_limit_ns = "custom"

    val_sample_with_replacement = bool(
        getattr(custom_config, "val_sample_with_replacement", False)
    )
    val_sample_size = None
    if val_sample_with_replacement:
        val_sample_size = _parse_sample_size(
            val_sample_limit, f"{sample_limit_ns}.val_sample_limit"
        )
        if val_sample_size is None:
            raise ValueError(
                "custom.val_sample_with_replacement requires an eval sample size. "
                f"Set {sample_limit_ns}.val_sample_limit to a positive integer (or disable val_sample_with_replacement)."
            )
    if train_sample_limit:
        logger.info(f"{sample_limit_ns}.train_sample_limit: {train_sample_limit}")
    if val_sample_limit and not val_sample_with_replacement:
        logger.info(f"{sample_limit_ns}.val_sample_limit: {val_sample_limit}")
    elif val_sample_with_replacement:
        logger.info(f"Val sample size per eval (with replacement): {val_sample_size}")

    # Build training dataset from the resolved single-dataset JSONL contract.
    # Require minimal explicit keys; others have sane defaults

    # Extract mode control parameters
    use_summary = bool(custom_config.use_summary)

    # Prepare system prompts for the selected mode
    # The system prompt is set on the template by ConfigLoader.resolve_prompts
    system_prompt_dense = getattr(sft.template, "system", None)
    system_prompt_summary = custom_config.system_prompt_summary

    if use_summary:
        logger.info("Summary mode enabled (custom.use_summary=true)")
        if system_prompt_summary is None:
            system_prompt_summary = getattr(sft.template, "system", None)
        if system_prompt_summary is None:
            try:
                from .config.prompts import SYSTEM_PROMPT_SUMMARY

                system_prompt_summary = SYSTEM_PROMPT_SUMMARY
                logger.info("Loaded default SYSTEM_PROMPT_SUMMARY")
            except ImportError as exc:
                raise ValueError(
                    "custom.use_summary is true but no summary system prompt was provided."
                ) from exc
        system_prompt_dense = None
    else:
        logger.info("Dense mode only (custom.use_summary=false)")

    dataset_seed = _resolve_dataset_seed(
        training_config=training_config, train_args=train_args
    )
    checkpoint_mode = _parse_checkpoint_mode(training_config.training)
    _apply_checkpoint_mode(train_args, checkpoint_mode=checkpoint_mode)
    encoded_sample_cache_cfg = _parse_encoded_sample_cache_config(
        training_config.training, train_args
    )
    static_packing_cache_cfg = _parse_static_packing_cache_config(
        training_config.training
    )
    train_encoded_sample_cache_info: dict[str, Any] | None = None
    eval_encoded_sample_cache_info: dict[str, Any] | None = None
    train_encoded_sample_cache_request = _build_encoded_sample_cache_request(
        runtime_cfg=encoded_sample_cache_cfg,
        training_config=training_config,
        custom_config=custom_config,
        template=sft.template,
        train_args=train_args,
        dataset_seed=dataset_seed,
        dataset_jsonl=str(train_jsonl) if train_jsonl else None,
        dataset_split="train",
        dataset_mode="summary" if use_summary else "dense",
        sample_limit=_normalize_optional_sample_limit(train_sample_limit),
        system_prompt_dense=system_prompt_dense,
        system_prompt_summary=system_prompt_summary,
    )
    dataset: Any
    trainer_variant = getattr(train_args, "trainer_variant", None)
    _validate_bbox_format_contract(
        custom_config=custom_config,
        trainer_variant=trainer_variant,
    )
    logger.info(
        "Serialization order config: object_ordering=%s object_field_order=%s",
        custom_config.object_ordering,
        custom_config.object_field_order,
    )
    logger.info(f"Loading training dataset: {train_jsonl}")
    dataset = BaseCaptionDataset.from_jsonl(
        train_jsonl,
        template=sft.template,
        user_prompt=custom_config.user_prompt,
        emit_norm=custom_config.emit_norm,
        json_format=custom_config.json_format,
        augmenter=augmenter,
        bypass_prob=bypass_prob,
        curriculum_state=curriculum_state,
        sample_limit=train_sample_limit,
        use_summary=use_summary,
        system_prompt_dense=system_prompt_dense,
        system_prompt_summary=system_prompt_summary,
        coord_tokens=custom_config.coord_tokens,
        offline_max_pixels=custom_config.offline_max_pixels,
        seed=dataset_seed,
        object_ordering=custom_config.object_ordering,
        object_field_order=custom_config.object_field_order,
        bbox_format=custom_config.bbox_format,
        encoded_sample_cache=train_encoded_sample_cache_request,
    )
    if train_encoded_sample_cache_request is not None:
        train_encoded_sample_cache_info = dataset.get_encoded_sample_cache_info()
    packing_cfg = _parse_packing_config(
        training_config.training, sft.template, train_args
    )
    _validate_attention_backend_for_packing(training_config=training_config)
    base_dataset_len = None
    try:
        base_dataset_len = len(dataset)
    except TypeError:
        base_dataset_len = None
    # Stage_2 rollout-matching supports post-rollout packing inside the trainer only.
    # Do not apply dataset-level packing wrappers for this trainer variant.
    is_rollout_matching_variant = _is_rollout_matching_variant(trainer_variant)
    _validate_stage1_static_packing_policy(
        packing_cfg=packing_cfg,
        trainer_variant=trainer_variant,
    )

    if packing_cfg.enabled and not is_rollout_matching_variant:
        training_map = getattr(training_config, "training", {}) or {}
        if not isinstance(training_map, Mapping):
            training_map = {}

        try:
            orig_bs = int(getattr(train_args, "per_device_train_batch_size", 1) or 1)
        except (TypeError, ValueError):
            orig_bs = 1
        try:
            orig_gas = int(getattr(train_args, "gradient_accumulation_steps", 1) or 1)
        except (TypeError, ValueError):
            orig_gas = 1

        _, _, packing_world_size, _ = get_dist_setting()
        packing_world_size = max(int(packing_world_size), 1)

        if orig_bs != 1:
            logger.warning(
                "packing enabled: forcing per_device_train_batch_size=1 (was %s)",
                orig_bs,
            )
            _set_train_arg(train_args, "per_device_train_batch_size", 1)

        _recompute_gas_for_packing(
            train_args=train_args,
            training_cfg=training_map,
            original_per_device_bs=orig_bs,
            original_gas=orig_gas,
            world_size=packing_world_size,
        )

        logger.info(
            "Packing unit semantics: one per-device dataloader item equals one packed sequence; effective_batch_size is interpreted in packed-sequence units."
        )

        train_dataloader_shuffle = bool(
            getattr(train_args, "train_dataloader_shuffle", True)
        )
        static_fingerprint = _build_static_packing_fingerprint(
            training_config=training_config,
            custom_config=custom_config,
            template=sft.template,
            train_args=train_args,
            dataset_seed=dataset_seed,
            packing_cfg=packing_cfg,
            train_jsonl=str(train_jsonl) if train_jsonl else None,
            dataset_split="train",
        )
        static_cache_dir = _resolve_static_packing_cache_dir(
            runtime_cfg=static_packing_cache_cfg,
            training_config=training_config,
            train_args=train_args,
            dataset_jsonl=str(train_jsonl) if train_jsonl else None,
            fusion_config_path=None,
            dataset_split="train",
            packing_cfg=packing_cfg,
        )

        dataset = build_static_packed_dataset(
            dataset,
            template=sft.template,
            packing_length=packing_cfg.packing_length,
            min_fill_ratio=packing_cfg.min_fill_ratio,
            packing_drop_last=packing_cfg.drop_last,
            dataloader_drop_last=bool(
                getattr(train_args, "dataloader_drop_last", False)
            ),
            allow_single_long=packing_cfg.allow_single_long,
            cache_dir=static_cache_dir,
            fingerprint=static_fingerprint,
            world_size=packing_world_size,
            train_dataloader_shuffle=train_dataloader_shuffle,
            wait_timeout_s=packing_cfg.wait_timeout_s,
            length_cache_persist_every=packing_cfg.length_cache_persist_every,
            length_precompute_workers=packing_cfg.length_precompute_workers,
        )
        base_dataset_len = len(dataset)

        logger.info(
            "Packing enabled (static): length=%s min_fill=%.2f packing_drop_last=%s allow_single_long=%s wait_timeout_s=%s length_cache_persist_every=%s length_precompute_workers=%s",
            packing_cfg.packing_length,
            packing_cfg.min_fill_ratio,
            packing_cfg.drop_last,
            packing_cfg.allow_single_long,
            packing_cfg.wait_timeout_s,
            packing_cfg.length_cache_persist_every,
            packing_cfg.length_precompute_workers,
        )
        logger.info(
            "Static packing plan: train_dataloader_shuffle=%s N_raw_packs=%s N_aligned_packs=%s raw_checksum=%s aligned_checksum=%s",
            train_dataloader_shuffle,
            len(dataset.raw_plan),
            len(dataset),
            dataset.raw_plan_checksum,
            dataset.aligned_plan_checksum,
        )
        logger.info(
            "Static packing DDP alignment: world_size=%s dataloader_drop_last=%s pad_needed=%s repeated_pack_indices=%s",
            dataset.world_size,
            dataset.dataloader_drop_last,
            dataset.pad_needed,
            dataset.repeated_pack_indices,
        )
        logger.info(
            "Static packing stats: packs=%s avg_fill=%.3f single_long=%s skipped_long=%s",
            len(dataset),
            dataset.avg_fill,
            dataset.single_long,
            dataset.skipped_long,
        )
    elif packing_cfg.enabled and is_rollout_matching_variant:
        logger.info(
            "Packing enabled for rollout-matching: dataset packing is disabled; "
            "trainer will apply dynamic post-rollout packing for the teacher-forced forward pass."
        )

    if (
        packing_cfg.enabled
        and packing_cfg.eval_packing
        and not is_rollout_matching_variant
    ):
        eval_bs = getattr(train_args, "per_device_eval_batch_size", None)
        if eval_bs != 1:
            logger.warning(
                "eval_packing enabled: forcing per_device_eval_batch_size=1 (was %s)",
                eval_bs,
            )
            train_args.per_device_eval_batch_size = 1
            if getattr(train_args, "training_args", None) is not None:
                train_args.training_args.per_device_eval_batch_size = 1

    try:
        train_len = len(dataset)
    except TypeError:
        train_len = base_dataset_len
    logger.info(f"Training dataset size (reported/approx): {train_len}")

    # ------------------------------------------------------------------
    # DDP dispatch sanity (fail-soft): avoid degenerate empty shards.
    #
    # In vanilla DDP, every rank must execute the same number of forward/backward
    # steps. Transformers' default DistributedSampler achieves this by either:
    # - truncating to floor(N/world_size) when dataloader_drop_last=True, or
    # - padding (repeating) to ceil(N/world_size) when dataloader_drop_last=False.
    #
    # A common footgun for debug/smoke runs is setting a tiny train_sample_limit
    # with dataloader_drop_last=True: if N < world_size then floor(...) == 0 and
    # training becomes a zero-step no-op.
    # ------------------------------------------------------------------
    try:
        _, _, world_size_raw, _ = get_dist_setting()
        world_size = int(world_size_raw)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        world_size = 1
    world_size = max(int(world_size), 1)

    base_len_i: int | None = None
    if base_dataset_len is not None:
        try:
            base_len_i = int(base_dataset_len)
        except (TypeError, ValueError):
            base_len_i = None

    try:
        drop_last_flag = bool(getattr(train_args, "dataloader_drop_last", False))
    except (AttributeError, TypeError, ValueError):
        drop_last_flag = False

    try:
        per_device_train_bs = int(
            getattr(train_args, "per_device_train_batch_size", 1) or 1
        )
    except (AttributeError, TypeError, ValueError):
        per_device_train_bs = 1
    per_device_train_bs = max(1, int(per_device_train_bs))

    try:
        gas = int(getattr(train_args, "gradient_accumulation_steps", 1) or 1)
    except (AttributeError, TypeError, ValueError):
        gas = 1
    gas = max(1, int(gas))

    if base_len_i is not None and base_len_i <= 0:
        raise ValueError(
            "Training dataset is empty (len==0). Fix custom.train_jsonl or sample limits."
        )

    per_rank_batches_est_for_static: int | None = None

    if base_len_i is not None and world_size > 1:
        per_rank_floor = int(base_len_i // world_size)
        remainder = int(base_len_i % world_size)

        # If drop_last would yield 0 samples per rank, handle it explicitly.
        #
        # For most trainer variants, switching drop_last off makes training feasible
        # (DistributedSampler will pad by repeating indices). For stage2-ab step-budgeted
        # trainers with gradient_accumulation_steps>1, this still cannot produce a full
        # accumulation window, so we fail fast instead of silently doing zero-grad steps.
        if drop_last_flag and base_len_i > 0 and per_rank_floor <= 0:
            if str(trainer_variant or "") == "stage2_two_channel" and gas > 1:
                raise ValueError(
                    "stage2-ab requires at least one full gradient-accumulation window per rank. "
                    f"Got dataset_len={int(base_len_i)} world_size={int(world_size)} -> per_rank_floor=0 with "
                    f"dataloader_drop_last=true and gradient_accumulation_steps={int(gas)}. "
                    "Mitigations: reduce gpus/world_size, increase custom.train_sample_limit, "
                    "or reduce training.effective_batch_size so gradient_accumulation_steps becomes 1."
                )

            logger.warning(
                (
                    "DDP dispatch: dataloader_drop_last=true with dataset_len=%s and "
                    "world_size=%s would yield 0 samples per rank. Forcing "
                    "dataloader_drop_last=false so every rank receives data (with repetition)."
                ),
                base_len_i,
                world_size,
            )
            train_args.dataloader_drop_last = False
            if getattr(train_args, "training_args", None) is not None:
                train_args.training_args.dataloader_drop_last = False
            drop_last_flag = False

        per_rank_samples_est = (
            int(per_rank_floor)
            if drop_last_flag
            else int((base_len_i + world_size - 1) // world_size)
        )
        if drop_last_flag:
            per_rank_batches_est = int(per_rank_samples_est // per_device_train_bs)
        else:
            per_rank_batches_est = int(
                (per_rank_samples_est + per_device_train_bs - 1) // per_device_train_bs
            )
        per_rank_batches_est_for_static = int(per_rank_batches_est)

        if drop_last_flag:
            logger.info(
                "DDP dispatch: dataloader_drop_last=true -> per_rank=%s (dropped_remainder=%s)",
                per_rank_floor,
                remainder,
            )
        else:
            padded = int(per_rank_samples_est * world_size - base_len_i)
            logger.info(
                "DDP dispatch: dataloader_drop_last=false -> per_rank=%s (padded_repeats=%s)",
                per_rank_samples_est,
                padded,
            )

    if base_len_i is not None and per_rank_batches_est_for_static is None:
        if drop_last_flag:
            per_rank_batches_est_for_static = int(base_len_i // per_device_train_bs)
        else:
            per_rank_batches_est_for_static = int(
                (base_len_i + per_device_train_bs - 1) // per_device_train_bs
            )

    if per_rank_batches_est_for_static is not None:
        _validate_stage2_step_budget_windows(
            trainer_variant=trainer_variant,
            packing_enabled=bool(packing_cfg.enabled),
            per_rank_batches_est=int(per_rank_batches_est_for_static),
            gradient_accumulation_steps=int(gas),
            dataloader_drop_last=bool(drop_last_flag),
        )

    if per_rank_batches_est_for_static is not None:
        _validate_static_packing_accumulation_windows(
            packing_cfg=packing_cfg,
            trainer_variant=trainer_variant,
            per_rank_batches_est=per_rank_batches_est_for_static,
            gradient_accumulation_steps=gas,
            world_size=world_size,
            dataloader_drop_last=drop_last_flag,
        )

    # Calculate total_steps and initialize curriculum_state if needed
    if curriculum_scheduler is not None:
        if curriculum_scheduler._requires_total_steps:
            # Calculate total_steps from dataset length, epochs, batch size, etc.
            num_train_epochs = getattr(train_args, "num_train_epochs", None)
            per_device_train_batch_size = getattr(
                train_args, "per_device_train_batch_size", None
            )
            gradient_accumulation_steps = getattr(
                train_args, "gradient_accumulation_steps", None
            )
            max_steps = getattr(train_args, "max_steps", None)

            if max_steps is not None and max_steps > 0:
                total_steps = max_steps
            elif (
                num_train_epochs is not None
                and per_device_train_batch_size is not None
                and gradient_accumulation_steps is not None
                and base_dataset_len is not None
            ):
                _, _, world_size, _ = get_dist_setting()
                if world_size <= 0:
                    world_size = 1
                len_dataset = base_dataset_len
                total_train_batch_size = (
                    per_device_train_batch_size
                    * gradient_accumulation_steps
                    * world_size
                )
                num_update_steps_per_epoch = max(
                    len_dataset // total_train_batch_size, 1
                )
                total_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)
            else:
                raise ValueError(
                    "Cannot calculate total_steps for curriculum scheduler. "
                    "Need either max_steps or (num_train_epochs, per_device_train_batch_size, gradient_accumulation_steps) with known dataset length"
                )
            curriculum_scheduler.set_total_steps(total_steps)
            logger.info(f"Curriculum scheduler: set total_steps={total_steps}")

        # Now get initial state and create curriculum_state
        initial_state = curriculum_scheduler.get_state(0)
        manager = Manager()
        curriculum_state = manager.dict(
            {
                "step": 0,
                "bypass_prob": initial_state["bypass_prob"],
                "ops": copy.deepcopy(initial_state["ops"]),
            }
        )
        # Update dataset's curriculum_state
        if hasattr(dataset, "set_curriculum_state"):
            dataset.set_curriculum_state(curriculum_state)
        elif hasattr(dataset, "preprocessor") and dataset.preprocessor is not None:
            dataset.preprocessor.curriculum_state = curriculum_state

    # Optional: multimodal health check (only in --debug mode)
    if args.debug:
        try:
            sample = dataset[0]
            img_grid = sample.get("image_grid_thw")
            pv = sample.get("pixel_values")
            input_ids = sample.get("input_ids")
            logger.debug(f"HealthCheck: keys={list(sample.keys())}")
            if img_grid is None or pv is None:
                raise ValueError(
                    "Encoded sample missing image_grid_thw/pixel_values. Check image paths and template preprocessing."
                )
            # Print basic shapes
            grid_shape_raw = getattr(img_grid, "shape", None)
            grid_shape = (
                tuple(grid_shape_raw)
                if isinstance(grid_shape_raw, (list, tuple))
                else None
            )
            pv_shape_raw = getattr(pv, "shape", None)
            pv_shape = (
                tuple(pv_shape_raw) if isinstance(pv_shape_raw, (list, tuple)) else None
            )
            logger.debug(
                f"image_grid_thw shape: {grid_shape}; pixel_values shape: {pv_shape}"
            )

            # Token count sanity vs grid tokens
            image_token_id = getattr(dataset.template, "image_token_id", None)
            merge = getattr(
                getattr(dataset.template, "processor", None), "image_processor", None
            )
            merge_size = getattr(merge, "merge_size", 1)
            expected = None
            if hasattr(img_grid, "prod"):
                try:
                    expected = int(
                        img_grid.prod(dim=-1).sum().item() // (merge_size**2)
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    expected = None
            if (
                isinstance(image_token_id, int)
                and isinstance(input_ids, list)
                and expected is not None
            ):
                actual = sum(1 for t in input_ids if t == image_token_id)
                logger.debug(f"image tokens: expected≈{expected}, actual={actual}")
                if actual == 0 or abs(actual - expected) > max(8, expected // 10):
                    logger.warning(
                        "Image token mismatch. Investigate chat_template and image processing."
                    )
        except (
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            logger.warning(f"HealthCheck failed: {e}")

    # Optional: dump conversation text-only (no tokens, no images) and full tokens
    dump_conv = bool(custom_config.dump_conversation_text or args.debug)
    try:
        dataset_nonempty = len(dataset) > 0
    except TypeError:
        dataset_nonempty = (base_dataset_len or 0) > 0

    if dump_conv and dataset_nonempty:
        try:
            template = dataset.template
            template.set_mode("pt")
            try:
                sample_encoded = dataset[0]
            finally:
                template.set_mode("train")

            input_ids = (
                sample_encoded.get("input_ids")
                if isinstance(sample_encoded, dict)
                else None
            )
            if input_ids is None:
                raise ValueError(
                    "Sample does not contain input_ids for dumping conversation text."
                )
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()

            raw_text = template.tokenizer.decode(input_ids, skip_special_tokens=False)

            def _compress_image_pad(text: str) -> str:
                def repl(match: re.Match) -> str:
                    count = match.group(0).count("<|image_pad|>")
                    return f"<|image_pad|>*{count}"

                return re.sub(r"(?:<\|image_pad\|>)+", repl, text)

            raw_text = _compress_image_pad(raw_text)

            assistant_gt = None
            try:
                record_clone = None
                base_records = getattr(dataset, "base_records", None)
                if isinstance(base_records, list) and base_records:
                    record_clone = copy.deepcopy(base_records[0])
                else:
                    pools = getattr(dataset, "_record_pools", None)
                    if isinstance(pools, dict):
                        for pool in pools.values():
                            if isinstance(pool, list) and pool:
                                record_clone = copy.deepcopy(pool[0])
                                break
                if record_clone is None:
                    raise ValueError(
                        "No base record available for assistant GT extraction"
                    )
                builder = dataset._create_builder(dataset.mode)
                merged = builder.build_many([record_clone])
                assistant_turn = next(
                    (
                        turn
                        for turn in merged.get("messages", [])
                        if turn.get("role") == "assistant"
                    ),
                    None,
                )
                if assistant_turn:
                    contents = assistant_turn.get("content") or []
                    for item in contents:
                        if isinstance(item, dict) and item.get("type") == "text":
                            assistant_gt = item.get("text")
                            break
            except (
                AttributeError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
            ) as inner_e:
                logger.warning(f"Failed to extract assistant GT: {inner_e}")

            logger.debug("Conversation (raw):\n" + raw_text)
            if assistant_gt:
                logger.debug("Assistant GT:\n" + assistant_gt)

            dump_path = custom_config.dump_conversation_path or "conversation_text.txt"
            if not os.path.isabs(dump_path):
                base_output_dir = getattr(train_args, "output_dir", None)
                base_dir = (
                    base_output_dir if isinstance(base_output_dir, str) else os.getcwd()
                )
                dump_path = os.path.join(base_dir, dump_path)
            dump_dir = os.path.dirname(dump_path) or "."
            os.makedirs(dump_dir, exist_ok=True)
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
                if not raw_text.endswith("\n"):
                    f.write("\n")
                if assistant_gt:
                    if not raw_text.endswith("\n"):
                        f.write("\n")
                    f.write("\n--- Assistant GT ---\n")
                    f.write(assistant_gt)
                    if not assistant_gt.endswith("\n"):
                        f.write("\n")
            logger.info(f"Conversation text saved to: {dump_path}")
        except (
            AttributeError,
            IndexError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as e:
            logger.warning(f"Failed to dump conversation text: {e}")

    # Build validation dataset from a single JSONL.
    eval_dataset = None
    val_jsonl = custom_config.val_jsonl
    eval_encoded_sample_cache_request = _build_encoded_sample_cache_request(
        runtime_cfg=encoded_sample_cache_cfg,
        training_config=training_config,
        custom_config=custom_config,
        template=sft.template,
        train_args=train_args,
        dataset_seed=dataset_seed,
        dataset_jsonl=str(val_jsonl) if val_jsonl else None,
        dataset_split="eval",
        dataset_mode="summary" if use_summary else "dense",
        sample_limit=_normalize_optional_sample_limit(val_sample_limit),
        system_prompt_dense=system_prompt_dense,
        system_prompt_summary=system_prompt_summary,
    )
    if val_jsonl:
        logger.info(f"Loading validation dataset: {val_jsonl}")
        eval_sample_limit = None if val_sample_with_replacement else val_sample_limit
        eval_dataset = BaseCaptionDataset.from_jsonl(
            val_jsonl,
            template=sft.template,
            user_prompt=custom_config.user_prompt,
            emit_norm=custom_config.emit_norm,
            json_format=custom_config.json_format,
            augmenter=None,  # No augmentation for validation
            bypass_prob=0.0,  # Explicit: no bypass for validation
            sample_limit=eval_sample_limit,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            coord_tokens=custom_config.coord_tokens,
            offline_max_pixels=custom_config.offline_max_pixels,
            seed=dataset_seed,
            object_ordering=custom_config.object_ordering,
            object_field_order=custom_config.object_field_order,
            bbox_format=custom_config.bbox_format,
            encoded_sample_cache=eval_encoded_sample_cache_request,
        )
        base_eval_len = len(eval_dataset)
        if eval_encoded_sample_cache_request is not None:
            eval_encoded_sample_cache_info = (
                eval_dataset.get_encoded_sample_cache_info()
            )
        if val_sample_with_replacement:
            eval_dataset = RandomSampleDataset(
                eval_dataset, sample_size=val_sample_size, seed=dataset_seed + 11
            )
            logger.info(
                "Validation dataset size: %s (sampled with replacement from %s)",
                len(eval_dataset),
                base_eval_len,
            )
        else:
            logger.info(f"Validation dataset size: {base_eval_len}")

    if (
        packing_cfg.enabled
        and packing_cfg.eval_packing
        and eval_dataset is not None
        and not is_rollout_matching_variant
    ):
        if val_sample_with_replacement:
            raise ValueError(
                "Static eval packing requires a deterministic, index-stable eval dataset. "
                "Set custom.val_sample_with_replacement=false or disable training.eval_packing."
            )

        eval_sample_limit_i: int | None = None
        if isinstance(val_sample_limit, int):
            eval_sample_limit_i = int(val_sample_limit)
        elif isinstance(val_sample_limit, str) and val_sample_limit.isdigit():
            eval_sample_limit_i = int(val_sample_limit)

        eval_fingerprint = _build_static_packing_fingerprint(
            training_config=training_config,
            custom_config=custom_config,
            template=sft.template,
            train_args=train_args,
            dataset_seed=dataset_seed,
            packing_cfg=packing_cfg,
            train_jsonl=str(val_jsonl) if val_jsonl else None,
            dataset_split="eval",
            eval_sample_limit=eval_sample_limit_i,
            eval_sample_with_replacement=bool(val_sample_with_replacement),
        )
        eval_cache_dir = _resolve_static_packing_cache_dir(
            runtime_cfg=static_packing_cache_cfg,
            training_config=training_config,
            train_args=train_args,
            dataset_jsonl=str(val_jsonl) if val_jsonl else None,
            fusion_config_path=None,
            dataset_split="eval",
            packing_cfg=packing_cfg,
        )

        eval_dataset = build_static_packed_dataset(
            eval_dataset,
            template=sft.template,
            packing_length=packing_cfg.packing_length,
            min_fill_ratio=packing_cfg.min_fill_ratio,
            packing_drop_last=False,
            dataloader_drop_last=False,
            allow_single_long=packing_cfg.allow_single_long,
            cache_dir=eval_cache_dir,
            fingerprint=eval_fingerprint,
            world_size=packing_world_size,
            train_dataloader_shuffle=False,
            wait_timeout_s=packing_cfg.wait_timeout_s,
            length_cache_persist_every=packing_cfg.length_cache_persist_every,
            length_precompute_workers=packing_cfg.length_precompute_workers,
        )
        logger.info(
            "Packing enabled for eval (static): length=%s min_fill=%.2f packing_drop_last=%s dataloader_drop_last=%s allow_single_long=%s wait_timeout_s=%s length_cache_persist_every=%s length_precompute_workers=%s",
            packing_cfg.packing_length,
            packing_cfg.min_fill_ratio,
            False,
            False,
            packing_cfg.allow_single_long,
            packing_cfg.wait_timeout_s,
            packing_cfg.length_cache_persist_every,
            packing_cfg.length_precompute_workers,
        )
        logger.info(
            "Static eval packing plan: N_raw_packs=%s N_aligned_packs=%s raw_checksum=%s aligned_checksum=%s",
            len(eval_dataset.raw_plan),
            len(eval_dataset),
            eval_dataset.raw_plan_checksum,
            eval_dataset.aligned_plan_checksum,
        )
        logger.info(
            "Static eval packing DDP alignment: world_size=%s dataloader_drop_last=%s pad_needed=%s repeated_pack_indices=%s",
            eval_dataset.world_size,
            eval_dataset.dataloader_drop_last,
            eval_dataset.pad_needed,
            eval_dataset.repeated_pack_indices,
        )
        logger.info(
            "Static eval packing stats: packs=%s avg_fill=%.3f single_long=%s skipped_long=%s",
            len(eval_dataset),
            eval_dataset.avg_fill,
            eval_dataset.single_long,
            eval_dataset.skipped_long,
        )

    # Sample printing disabled to avoid dumping labels/ids

    # CRITICAL: Apply tuner (LoRA/adapters) before creating trainer
    logger.info("Preparing model with tuner...")
    sft.model = sft.prepare_model(
        train_args, sft.model, template=sft.template, train_dataset=dataset
    )
    # After PEFT wrapping, reattach coord-offset hooks to active modules so offsets train/save correctly
    if coord_offset_cfg and coord_offset_cfg.enabled:
        reattached = reattach_coord_offset_hooks(sft.model)
        if reattached is None:
            logger.warning(
                "coord_offset_adapter not found after prepare_model; hooks not reattached"
            )
        else:
            logger.info("Reattached coord_offset hooks on wrapped model")
    logger.info(f"Model after tuner: {type(sft.model).__name__}")

    # Setup trainer
    logger.info("Setting up trainer...")
    trainer_variant = getattr(train_args, "trainer_variant", None)
    if trainer_variant in {"stage2_rollout_aligned", "stage2_two_channel"}:
        # Keep raw fields (messages/assistant_payload) for rollout→parse→match construction.
        if getattr(train_args, "training_args", None) is not None:
            train_args.training_args.remove_unused_columns = False

        # Eval rollout batching knob: rollout_matching.eval_decode_batch_size.
        # Rollout trainer variants use this value for eval dataloader batch size.
        _apply_rollout_decode_batch_size_override(
            train_args=train_args,
            training_config=training_config,
        )

        try:
            from swift.trainers.rlhf_trainer.utils import identity_data_collator

            base_collator = identity_data_collator
        except ImportError as exc:
            raise RuntimeError(
                "rollout-matching trainer requires ms-swift identity_data_collator"
            ) from exc
    else:
        base_collator = sft._get_data_collator()
    token_type_cfg = getattr(custom_config, "token_type_metrics", None)
    coord_soft_ce_w1_cfg = getattr(custom_config, "coord_soft_ce_w1", None)
    bbox_geo_cfg = getattr(custom_config, "bbox_geo", None)
    bbox_size_aux_cfg = getattr(custom_config, "bbox_size_aux", None)
    instability_monitor_cfg = None
    loss_gradient_monitor_cfg = None
    proxy_supervision_cfg = None
    extra_cfg = getattr(custom_config, "extra", None)
    if isinstance(extra_cfg, Mapping):
        raw_instab = extra_cfg.get("instability_monitor")
        if isinstance(raw_instab, dict):
            instability_monitor_cfg = raw_instab
        raw_gradmon = extra_cfg.get("loss_gradient_monitor")
        if isinstance(raw_gradmon, Mapping):
            loss_gradient_monitor_cfg = dict(raw_gradmon)
        raw_proxy = extra_cfg.get("proxy_supervision")
        if isinstance(raw_proxy, Mapping):
            proxy_supervision_cfg = dict(raw_proxy)
    if trainer_variant in {"stage2_rollout_aligned", "stage2_two_channel"}:
        # Rollout-matching does its own encoding and loss masking inside the trainer.
        data_collator = base_collator
    else:
        data_collator = build_dataset_metrics_collator(
            sft.template,
            base_collator,
            token_type_cfg=token_type_cfg,
            instability_monitor_cfg=instability_monitor_cfg,
            proxy_supervision_cfg=proxy_supervision_cfg,
        )

    heartbeat_writer = None
    heartbeat_callback = None
    if heartbeat_enabled:
        from .callbacks.train_heartbeat import (
            HeartbeatDataCollator,
            TrainHeartbeatCallback,
            TrainHeartbeatWriter,
        )

        rank_s = str(os.environ.get("RANK", "0") or "0")
        output_dir_for_heartbeat = Path(
            str(getattr(train_args, "output_dir", ".") or ".")
        )
        heartbeat_path = (
            output_dir_for_heartbeat / f"train_heartbeat.rank{rank_s}.jsonl"
        )
        heartbeat_writer = TrainHeartbeatWriter(path=heartbeat_path, enabled=True)
        heartbeat_writer.emit("heartbeat_enabled", rank=rank_s)
        data_collator = HeartbeatDataCollator(data_collator, writer=heartbeat_writer)
        heartbeat_callback = TrainHeartbeatCallback(heartbeat_writer)
        logger.info("Train heartbeat enabled: %s", str(heartbeat_path))

    stage1_eval_detection_callback = None
    if trainer_variant not in {"stage2_rollout_aligned", "stage2_two_channel"}:
        stage1_eval_cfg = getattr(custom_config, "eval_detection", None)
        if stage1_eval_cfg is not None and bool(
            getattr(stage1_eval_cfg, "enabled", False)
        ):
            if not val_jsonl:
                raise ValueError(
                    "custom.eval_detection.enabled=true requires custom.val_jsonl"
                )
            if val_sample_with_replacement:
                raise ValueError(
                    "custom.eval_detection does not support custom.val_sample_with_replacement=true"
                )

            custom_extra = getattr(custom_config, "extra", {}) or {}
            prompt_variant = None
            if isinstance(custom_extra, Mapping):
                raw_prompt_variant = custom_extra.get("prompt_variant")
                if isinstance(raw_prompt_variant, str) and raw_prompt_variant.strip():
                    prompt_variant = str(raw_prompt_variant).strip()
            if prompt_variant is None:
                raise ValueError(
                    "custom.eval_detection requires custom.extra.prompt_variant so eval generation "
                    "matches the authored Stage-1 prompt contract"
                )

            model_checkpoint = _resolve_model_checkpoint_path(training_config)
            if model_checkpoint is None:
                raise ValueError(
                    "custom.eval_detection requires a non-empty model.model checkpoint path"
                )

            resolved_eval_limit = _normalize_optional_sample_limit(val_sample_limit)
            configured_eval_limit = _normalize_optional_sample_limit(
                getattr(stage1_eval_cfg, "limit", None)
            )
            if resolved_eval_limit is None:
                callback_eval_limit = configured_eval_limit
            elif configured_eval_limit is None:
                callback_eval_limit = resolved_eval_limit
            else:
                callback_eval_limit = min(
                    int(resolved_eval_limit), int(configured_eval_limit)
                )

            from .callbacks.stage1_detection_eval import Stage1DetectionEvalCallback

            stage1_eval_detection_callback = Stage1DetectionEvalCallback(
                gt_jsonl=str(val_jsonl),
                output_root=str(getattr(train_args, "output_dir", ".") or "."),
                model_checkpoint=str(model_checkpoint),
                prompt_variant=str(prompt_variant),
                bbox_format=str(custom_config.bbox_format),
                object_field_order=str(custom_config.object_field_order),
                object_ordering=str(custom_config.object_ordering),
                metrics=str(stage1_eval_cfg.metrics),
                use_segm=bool(stage1_eval_cfg.use_segm),
                strict_parse=bool(stage1_eval_cfg.strict_parse),
                iou_thrs=getattr(stage1_eval_cfg, "iou_thrs", None),
                f1ish_iou_thrs=list(stage1_eval_cfg.f1ish_iou_thrs),
                f1ish_pred_scope=str(stage1_eval_cfg.f1ish_pred_scope),
                semantic_model=str(stage1_eval_cfg.semantic_model),
                semantic_threshold=float(stage1_eval_cfg.semantic_threshold),
                semantic_device=str(stage1_eval_cfg.semantic_device),
                semantic_batch_size=int(stage1_eval_cfg.semantic_batch_size),
                lvis_max_dets=int(stage1_eval_cfg.lvis_max_dets),
                pred_score_source=str(stage1_eval_cfg.pred_score_source),
                pred_score_version=int(stage1_eval_cfg.pred_score_version),
                constant_score=float(stage1_eval_cfg.constant_score),
                batch_size=int(stage1_eval_cfg.batch_size),
                max_new_tokens=int(stage1_eval_cfg.max_new_tokens),
                temperature=float(stage1_eval_cfg.temperature),
                top_p=float(stage1_eval_cfg.top_p),
                repetition_penalty=float(stage1_eval_cfg.repetition_penalty),
                limit=callback_eval_limit,
                seed=dataset_seed,
                lvis_annotations_json=getattr(
                    stage1_eval_cfg, "lvis_annotations_json", None
                ),
            )
            logger.info(
                "Stage-1 detection eval enabled: metrics=%s limit=%s prompt_variant=%s",
                str(stage1_eval_cfg.metrics),
                callback_eval_limit,
                prompt_variant,
            )

    trainer_cls = compose_trainer_class(
        trainer_cls=resolve_trainer_cls(train_args),
        trainer_variant=str(trainer_variant or ""),
        instability_monitor_cfg=instability_monitor_cfg,
        token_type_cfg=token_type_cfg,
        bbox_geo_cfg=bbox_geo_cfg,
        bbox_size_aux_cfg=bbox_size_aux_cfg,
        coord_soft_ce_w1_cfg=coord_soft_ce_w1_cfg,
    )

    callbacks = build_trainer_callbacks(
        base_callbacks=sft.callbacks.copy() if sft.callbacks else [],
        dataset=dataset,
        append_dataset_epoch_callback_fn=_append_dataset_epoch_callback,
        stage1_eval_detection_callback=stage1_eval_detection_callback,
        heartbeat_callback=heartbeat_callback,
        curriculum_scheduler=curriculum_scheduler,
        curriculum_state=curriculum_state,
        save_delay_cfg=getattr(train_args, "save_delay_config", None),
        save_delay_steps=getattr(train_args, "save_delay_steps", None),
        save_delay_epochs=getattr(train_args, "save_delay_epochs", None),
        logger=logger,
    )

    trainer_kwargs = (
        sft._get_trainer_kwargs() if hasattr(sft, "_get_trainer_kwargs") else {}
    )
    trainer = instantiate_trainer(
        trainer_cls=trainer_cls,
        sft_model=sft.model,
        training_args=train_args.training_args,
        data_collator=data_collator,
        dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        template=sft.template,
        trainer_kwargs=trainer_kwargs,
        heartbeat_writer=heartbeat_writer,
    )
    # Rollout-matching evaluators emit rollout/* metrics only.
    # Guard against inherited defaults like eval_token_acc, which would crash
    # best-checkpoint selection at evaluation time.
    if (
        trainer_variant in {"stage2_rollout_aligned", "stage2_two_channel"}
        and eval_dataset is not None
        and getattr(train_args, "training_args", None) is not None
    ):
        metric_for_best_model = str(
            getattr(train_args.training_args, "metric_for_best_model", "") or ""
        ).strip()
        if "token_acc" in metric_for_best_model:
            logger.warning(
                "metric_for_best_model=%s is incompatible with rollout metrics; overriding to rollout/f1.",
                metric_for_best_model,
            )
            train_args.training_args.metric_for_best_model = "rollout/f1"
            trainer.args.metric_for_best_model = "rollout/f1"
            if getattr(train_args.training_args, "greater_is_better", None) is None:
                train_args.training_args.greater_is_better = True
                trainer.args.greater_is_better = True

    coord_soft_cfg_for_manifest: Mapping[str, Any] | None = None
    if coord_soft_ce_w1_cfg is not None:
        if isinstance(coord_soft_ce_w1_cfg, Mapping):
            coord_soft_cfg_for_manifest = dict(coord_soft_ce_w1_cfg)
        elif is_dataclass(coord_soft_ce_w1_cfg):
            coord_soft_cfg_for_manifest = dataclass_asdict_no_none(coord_soft_ce_w1_cfg)

    def _resolve_pipeline_manifest(
        cfg: Mapping[str, Any] | None,
        *,
        default_objective: list[str],
        default_diagnostics: list[str],
        coord_soft_cfg: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _build_pipeline_manifest(
            cfg,
            default_objective=default_objective,
            default_diagnostics=default_diagnostics,
            trainer_variant=str(trainer_variant or ""),
            config_path=str(config_path),
            run_name=str(getattr(train_args, "run_name", "") or ""),
            seed=int(getattr(train_args.training_args, "seed", 0) or 0),
            coord_soft_cfg=coord_soft_cfg,
        )

    if trainer_variant in {"stage2_rollout_aligned", "stage2_two_channel"}:
        try:
            rollout_cfg_obj = getattr(training_config, "rollout_matching", None)
            if rollout_cfg_obj is None:
                rollout_cfg_raw = {}
            elif is_dataclass(rollout_cfg_obj):
                rollout_cfg_raw = dataclass_asdict_no_none(rollout_cfg_obj)
            else:
                rollout_cfg_raw = rollout_cfg_obj

            if rollout_cfg_raw is None:
                rollout_cfg_raw = {}
            if not isinstance(rollout_cfg_raw, Mapping):
                raise TypeError("rollout_matching must be a mapping when provided")

            rollout_cfg: dict[str, Any] = dict(rollout_cfg_raw)

            if trainer_variant == "stage2_two_channel" and isinstance(
                rollout_cfg.get("pipeline"), Mapping
            ):
                raise ValueError(
                    "rollout_matching.pipeline is not allowed when custom.trainer_variant=stage2_two_channel. "
                    "Use stage2_ab.pipeline instead."
                )

            # BREAKING: decoding knobs moved under rollout_matching.decoding.*.
            legacy_decoding_keys = [
                k for k in ("temperature", "top_p", "top_k") if k in rollout_cfg
            ]
            if legacy_decoding_keys:
                keys_s = ", ".join(
                    f"rollout_matching.{k}" for k in legacy_decoding_keys
                )
                raise ValueError(
                    "Legacy rollout decoding keys have been removed: "
                    f"{keys_s}. Use rollout_matching.decoding.* instead. "
                    "(No backward compatibility.)"
                )

            # BREAKING: rollout_buffer was an old sync reuse optimization and is removed.
            if "rollout_buffer" in rollout_cfg:
                raise ValueError(
                    "rollout_matching.rollout_buffer has been removed. "
                    "Remove this section from your config. (No backward compatibility.)"
                )

            decoding_raw = rollout_cfg.get("decoding", None)
            if decoding_raw is None:
                decoding: dict[str, Any] = {}
            elif isinstance(decoding_raw, Mapping):
                decoding = dict(decoding_raw)
            else:
                raise TypeError(
                    "rollout_matching.decoding must be a mapping when provided"
                )
            rollout_cfg["decoding"] = decoding

            custom_extra = getattr(custom_config, "extra", {}) or {}
            prompt_variant_from_extra: str | None = None
            if isinstance(custom_extra, Mapping):
                raw_prompt_variant = custom_extra.get("prompt_variant")
                if isinstance(raw_prompt_variant, str) and raw_prompt_variant.strip():
                    prompt_variant_from_extra = str(raw_prompt_variant).strip()
            if (
                prompt_variant_from_extra is not None
                and rollout_cfg.get("eval_prompt_variant", None) is None
            ):
                # Keep eval-step rollouts aligned with custom.extra.prompt_variant unless
                # rollout_matching.eval_prompt_variant explicitly overrides it.
                rollout_cfg["eval_prompt_variant"] = str(prompt_variant_from_extra)

            # Inject packing runtime knobs (stage_2 uses dynamic post-rollout packing; dataset packing is disabled).
            rollout_cfg.update(
                {
                    "packing_enabled": bool(packing_cfg.enabled),
                    "packing_length": int(packing_cfg.packing_length),
                    "packing_buffer": int(packing_cfg.buffer_size),
                    "packing_min_fill_ratio": float(packing_cfg.min_fill_ratio),
                    "packing_drop_last": bool(packing_cfg.drop_last),
                    "prompt_variant": prompt_variant_from_extra,
                    "object_ordering": str(custom_config.object_ordering),
                    "object_field_order": str(custom_config.object_field_order),
                    "bbox_format": str(custom_config.bbox_format),
                }
            )
            setattr(trainer, "rollout_matching_cfg", rollout_cfg)
            setattr(
                trainer, "object_field_order", str(custom_config.object_field_order)
            )

            validate_hook = getattr(trainer, "_validate_rollout_matching_cfg", None)
            if callable(validate_hook):
                validate_hook()

            if trainer_variant == "stage2_rollout_aligned":
                rollout_manifest = _resolve_pipeline_manifest(
                    rollout_cfg,
                    default_objective=[
                        "token_ce",
                        "bbox_geo",
                        "bbox_size_aux",
                        "coord_reg",
                    ],
                    default_diagnostics=["coord_diag"],
                    coord_soft_cfg=coord_soft_cfg_for_manifest,
                )
            else:
                rollout_manifest = {
                    "payload": {
                        "objective": [],
                        "diagnostics": [],
                        "extra": {"variant": str(trainer_variant or "")},
                    },
                    "objective": [],
                    "diagnostics": [],
                    "extra": {"variant": str(trainer_variant or "")},
                    "checksum": "",
                    "run_context": {
                        "config": str(config_path),
                        "run_name": str(getattr(train_args, "run_name", "") or ""),
                        "seed": int(getattr(train_args.training_args, "seed", 0) or 0),
                    },
                }
            setattr(trainer, "rollout_pipeline_manifest", rollout_manifest)

            logger.info(
                "Rollout-matching config injected: rollout_backend=%s packing_enabled=%s pipeline_checksum=%s objective=%s diagnostics=%s config=%s run_name=%s seed=%s",
                rollout_cfg.get("rollout_backend", "hf"),
                rollout_cfg.get("packing_enabled", False),
                rollout_manifest.get("checksum", ""),
                [m.get("name") for m in rollout_manifest.get("objective", [])],
                [m.get("name") for m in rollout_manifest.get("diagnostics", [])],
                str(config_path),
                str(getattr(train_args, "run_name", "") or ""),
                int(getattr(train_args.training_args, "seed", 0) or 0),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to inject rollout_matching_cfg into trainer. "
                "This is required for rollout-matching/stage2-ab trainer variants."
            ) from exc

    if trainer_variant == "stage2_two_channel":
        stage2_ab_typed = getattr(training_config, "stage2_ab", None)
        if stage2_ab_typed is None:
            raise ValueError(
                "training_config.stage2_ab is required for stage2_two_channel; "
                "check config parsing (top-level stage2_ab section)."
            )
        stage2_ab_cfg: dict[str, Any] = asdict(stage2_ab_typed)

        setattr(trainer, "stage2_ab_cfg", stage2_ab_cfg)

        sched = stage2_ab_cfg.get("schedule")
        b_ratio = sched.get("b_ratio") if isinstance(sched, Mapping) else None

        stage2_manifest = _resolve_pipeline_manifest(
            stage2_ab_cfg,
            default_objective=[
                "token_ce",
                "loss_duplicate_burst_unlikelihood",
                "bbox_geo",
                "bbox_size_aux",
                "coord_reg",
            ],
            default_diagnostics=["coord_diag"],
            coord_soft_cfg=coord_soft_cfg_for_manifest,
        )
        setattr(trainer, "stage2_pipeline_manifest", stage2_manifest)

        logger.info(
            "Stage2-AB config injected: b_ratio=%s pipeline_checksum=%s objective=%s diagnostics=%s config=%s run_name=%s seed=%s",
            b_ratio,
            stage2_manifest.get("checksum", ""),
            [m.get("name") for m in stage2_manifest.get("objective", [])],
            [m.get("name") for m in stage2_manifest.get("diagnostics", [])],
            str(config_path),
            str(getattr(train_args, "run_name", "") or ""),
            int(getattr(train_args.training_args, "seed", 0) or 0),
        )
    if coord_soft_ce_w1_cfg is not None:
        setattr(trainer, "coord_soft_ce_w1_cfg", coord_soft_ce_w1_cfg)
    if bbox_geo_cfg is not None:
        setattr(trainer, "bbox_geo_cfg", bbox_geo_cfg)
    if bbox_size_aux_cfg is not None:
        setattr(trainer, "bbox_size_aux_cfg", bbox_size_aux_cfg)
    setattr(trainer, "bbox_format", str(custom_config.bbox_format))
    if token_type_cfg is not None:
        setattr(trainer, "token_type_metrics_cfg", token_type_cfg)
    if instability_monitor_cfg is not None:
        setattr(trainer, "instability_monitor_cfg", instability_monitor_cfg)
        # Provide JSONL paths so the monitor can dump offending records by base_idx.
        setattr(trainer, "instability_train_jsonl", str(train_jsonl))
        if val_jsonl:
            setattr(trainer, "instability_val_jsonl", str(val_jsonl))
    if loss_gradient_monitor_cfg is not None:
        setattr(trainer, "loss_gradient_monitor_cfg", dict(loss_gradient_monitor_cfg))
        if (
            bool(loss_gradient_monitor_cfg.get("enabled", False))
            and int(os.environ.get("RANK", "0") or "0") == 0
        ):
            param_block = loss_gradient_monitor_cfg.get("param_block", {})
            param_strategy = (
                str(param_block.get("strategy", "auto_last_lm_layernorm"))
                if isinstance(param_block, Mapping)
                else "auto_last_lm_layernorm"
            )
            logger.info(
                "LossGradientMonitor enabled: interval_steps=%s require_sync_gradients=%s param_strategy=%s",
                loss_gradient_monitor_cfg.get("interval_steps", 50),
                loss_gradient_monitor_cfg.get("require_sync_gradients", True),
                param_strategy,
            )

    resume_from_checkpoint = _resolve_resume_from_checkpoint(train_args)
    if checkpoint_mode == "restartable" and resume_from_checkpoint:
        from .trainers.final_checkpoint import prepare_restartable_checkpoint_resume

        restore_payload = prepare_restartable_checkpoint_resume(
            trainer, resume_from_checkpoint
        )
        logger.info(
            "Restartable resume preflight passed: checkpoint=%s global_step=%s",
            resume_from_checkpoint,
            restore_payload.get("global_step"),
        )
    elif checkpoint_mode == "artifact_only" and resume_from_checkpoint:
        logger.warning(
            "Resuming from %s with checkpoint_mode=artifact_only. Model-selection artifacts may not provide exact optimizer/RNG/runtime-state fidelity.",
            resume_from_checkpoint,
        )

    selected_pipeline_manifest = getattr(trainer, "stage2_pipeline_manifest", None)
    if not isinstance(selected_pipeline_manifest, Mapping):
        selected_pipeline_manifest = getattr(trainer, "rollout_pipeline_manifest", None)
    if not isinstance(selected_pipeline_manifest, Mapping):
        selected_pipeline_manifest = None

    train_sample_limit_i = _normalize_optional_sample_limit(train_sample_limit)
    val_sample_limit_i = _normalize_optional_sample_limit(val_sample_limit)
    train_data_provenance = _build_data_source_provenance(
        split="train",
        dataset_jsonl=str(train_jsonl) if train_jsonl else None,
        dataset_seed=dataset_seed,
        sample_limit=train_sample_limit_i,
    )
    eval_data_provenance = (
        _build_data_source_provenance(
            split="eval",
            dataset_jsonl=str(val_jsonl) if val_jsonl else None,
            dataset_seed=dataset_seed,
            sample_limit=val_sample_limit_i,
            sample_with_replacement=bool(val_sample_with_replacement),
        )
        if eval_dataset is not None
        else None
    )
    effective_runtime = _build_effective_runtime_payload(
        training_config=training_config,
        train_args=train_args,
        trainer_variant=trainer_variant,
        dataset_seed=dataset_seed,
        checkpoint_mode=checkpoint_mode,
        packing_cfg=packing_cfg,
        encoded_sample_cache_cfg=encoded_sample_cache_cfg,
        train_jsonl=str(train_jsonl) if train_jsonl else None,
        val_jsonl=str(val_jsonl) if val_jsonl else None,
        pipeline_manifest=selected_pipeline_manifest,
    )

    # Start training
    logger.info("=" * 70)
    logger.info("  Starting Training")
    logger.info("=" * 70)
    logger.info(f"  Output directory: {train_args.output_dir}")
    logger.info(f"  Epochs: {train_args.num_train_epochs}")
    per_device_batch = getattr(train_args, "per_device_train_batch_size", None)
    grad_accum_steps = getattr(train_args, "gradient_accumulation_steps", None)
    if isinstance(per_device_batch, int) and isinstance(grad_accum_steps, int):
        logger.info(f"  Effective batch size: {per_device_batch * grad_accum_steps}")
    else:
        logger.info(
            "  Effective batch size: unavailable (missing batch or accumulation settings)"
        )
    logger.info("=" * 70)

    # Reproducibility: record git SHA and dirty flag (rank 0 only).
    try:
        rank_raw = os.environ.get("RANK") or os.environ.get("SLURM_PROCID")
        is_rank0 = True if rank_raw is None else (int(rank_raw) == 0)
    except (TypeError, ValueError):
        is_rank0 = True

    if is_rank0:
        out_dir = getattr(train_args, "output_dir", None)
        written = None
        if out_dir:
            # ------------------------------------------------------------------
            # Required run manifest files (fail-fast)
            # ------------------------------------------------------------------
            from src.utils.run_manifest import write_run_manifest_files

            written = write_run_manifest_files(
                output_dir=Path(str(out_dir)),
                training_config=training_config,
                config_path=str(getattr(args, "config", "") or ""),
                base_config_path=str(getattr(args, "base_config", "") or "")
                if getattr(args, "base_config", None)
                else None,
                dataset_seed=dataset_seed,
                effective_runtime=effective_runtime,
                pipeline_manifest=selected_pipeline_manifest,
                train_data_provenance=train_data_provenance,
                eval_data_provenance=eval_data_provenance,
            )
            logger.info(
                "Wrote run manifest files: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(written.items())),
            )

        # ------------------------------------------------------------------
        # Required run metadata (fail-fast): git + upstream provenance
        # ------------------------------------------------------------------
        if not out_dir:
            raise ValueError(
                "train_args.output_dir is not set; cannot write run metadata"
            )

        out_path = write_run_metadata_file(
            output_dir=Path(str(out_dir)),
            config_path=str(getattr(args, "config", "") or ""),
            base_config_path=str(getattr(args, "base_config", "") or "")
            if getattr(args, "base_config", None)
            else None,
            run_name=str(getattr(train_args, "run_name", "") or ""),
            dataset_seed=dataset_seed,
            repo_root=Path(__file__).resolve().parents[1],
            manifest_files=written,
            train_cache_info=train_encoded_sample_cache_info,
            eval_cache_info=eval_encoded_sample_cache_info,
        )
        logger.info("Wrote run metadata: %s", str(out_path))

    if heartbeat_writer is not None:
        heartbeat_writer.emit("train_call_enter")
    try:
        sft.train(trainer)
        if heartbeat_writer is not None:
            heartbeat_writer.emit("train_call_return")
    except torch.cuda.OutOfMemoryError:
        if heartbeat_writer is not None:
            heartbeat_writer.emit("train_call_oom")
        debug_info = getattr(dataset, "last_sample_debug", None)
        logger.error(f"CUDA OOM encountered. Last sample debug: {debug_info}")
        raise
    except Exception as exc:
        if heartbeat_writer is not None:
            heartbeat_writer.emit("train_call_exception", exc_type=type(exc).__name__)
        raise
    finally:
        if heartbeat_writer is not None:
            heartbeat_writer.emit("train_call_finally")
        # Explicit cleanup to prevent DeepSpeed cleanup errors during GC
        # This addresses a known DeepSpeed issue where __del__ can fail
        # when accessing bf16_groups that are already partially destroyed
        try:
            # Check if DeepSpeed is enabled
            if (
                hasattr(trainer, "is_deepspeed_enabled")
                and trainer.is_deepspeed_enabled
            ):
                model_wrapped = getattr(trainer, "model_wrapped", None)
                # model_wrapped IS the DeepSpeed engine when DeepSpeed is enabled
                if model_wrapped is not None:
                    try:
                        # Patch the optimizer's destroy method to prevent IndexError
                        # This is safer than calling destroy() which can still fail
                        optimizer = getattr(model_wrapped, "optimizer", None)
                        if optimizer is not None and hasattr(optimizer, "destroy"):
                            original_destroy = optimizer.destroy

                            def safe_destroy():
                                try:
                                    original_destroy()
                                except (IndexError, AttributeError, RuntimeError):
                                    # Silently ignore errors during optimizer cleanup
                                    # These are harmless - training already completed
                                    pass

                            optimizer.destroy = safe_destroy

                        # Now safe to call engine destroy
                        if hasattr(model_wrapped, "destroy"):
                            model_wrapped.destroy()
                        logger.debug("DeepSpeed engine cleaned up successfully")
                    except (
                        AttributeError,
                        IndexError,
                        OSError,
                        RuntimeError,
                        TypeError,
                    ) as cleanup_error:
                        # Ignore cleanup errors - they're harmless at this point
                        # Training already completed successfully
                        logger.debug(
                            f"DeepSpeed cleanup warning (non-fatal): {cleanup_error}"
                        )
        except (AttributeError, IndexError, OSError, RuntimeError, TypeError) as e:
            # Non-fatal: training already completed successfully
            logger.debug(f"Cleanup warning (non-fatal): {e}")

        try:
            rank_raw = os.environ.get("RANK") or os.environ.get("SLURM_PROCID")
            is_rank0 = True if rank_raw is None else (int(rank_raw) == 0)
        except (TypeError, ValueError):
            is_rank0 = True

        if is_rank0:
            output_dir = getattr(train_args, "output_dir", None)
            if output_dir:
                logging_path = Path(str(output_dir)) / "logging.jsonl"
                if _strip_trailing_trainer_state_logging_row(logging_path):
                    logger.info(
                        "Sanitized logging.jsonl: removed trailing trainer-state payload appended by upstream ms-swift."
                    )


if __name__ == "__main__":
    main()
