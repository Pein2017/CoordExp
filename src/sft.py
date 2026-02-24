"""SFT runner - pure YAML config-driven, no CLI arguments for hyperparameters"""

import argparse
import copy
import hashlib
import importlib
import json
import sys
import logging
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import Manager
from typing import Any, Literal, Mapping, Sequence

import torch
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
from .config import ConfigLoader, SaveDelayConfig
from .config.strict_dataclass import dataclass_asdict_no_none
from .datasets import (
    BaseCaptionDataset,
    RandomSampleDataset,
    build_static_packed_dataset,
)
from .trainers.metrics.mixins import (
    AggregateTokenTypeMetricsMixin,
    CoordSoftCEW1LossMixin,
    GradAccumLossScaleMixin,
    InstabilityMonitorMixin,
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

    save_last_epoch = getattr(train_args, "save_last_epoch", True)
    if save_last_epoch:
        return with_final_checkpoint(trainer_cls)
    return trainer_cls


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


def _set_train_arg(train_args: Any, field: str, value: Any) -> None:
    setattr(train_args, field, value)
    nested = getattr(train_args, "training_args", None)
    if nested is not None:
        setattr(nested, field, value)


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
    return identity


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


def _build_static_packing_fingerprint(
    *,
    training_config: Any,
    custom_config: Any,
    template: Any,
    train_args: Any,
    dataset_seed: int,
    packing_cfg: PackingRuntimeConfig,
    train_jsonl: str | None,
    fusion_config_path: str | None,
    dataset_split: str = "train",
    eval_sample_limit: int | None = None,
    eval_sample_with_replacement: bool | None = None,
) -> dict[str, Any]:
    template_cfg = getattr(training_config, "template", {}) or {}
    training_cfg = getattr(training_config, "training", {}) or {}

    split = str(dataset_split or "train").strip().lower()
    if split not in {"train", "eval"}:
        raise ValueError(
            "dataset_split must be one of {'train', 'eval'}, "
            f"got {dataset_split!r}"
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
        "custom_json_format": getattr(custom_config, "json_format", None),
        "custom_object_ordering": getattr(custom_config, "object_ordering", None),
        "custom_object_field_order": getattr(custom_config, "object_field_order", None),
        "custom_use_summary": bool(getattr(custom_config, "use_summary", False)),
        "dataset_jsonl": str(train_jsonl) if train_jsonl else None,
        "custom_train_jsonl": str(train_jsonl) if train_jsonl else None,
        "custom_fusion_config": str(fusion_config_path)
        if fusion_config_path
        else None,
        "dataset_source_jsonl": _build_source_path_identity(train_jsonl),
        "dataset_source_train_jsonl": _build_source_path_identity(train_jsonl),
        "dataset_source_fusion_config": _build_source_path_identity(
            fusion_config_path
        ),
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


def _is_rollout_matching_variant(trainer_variant: str | None) -> bool:
    return str(trainer_variant or "") in {"stage2_rollout_aligned", "stage2_two_channel"}


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


def _safe_module_info(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}

    version = getattr(module, "__version__", None)
    module_file = getattr(module, "__file__", None)

    info: dict[str, Any] = {}
    if version is not None:
        info["version"] = str(version)
    if module_file:
        info["file"] = str(module_file)
    return info


def _find_git_repo_root(start_dir: Path) -> Path | None:
    for parent in [start_dir, *start_dir.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _safe_git_state(repo_root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        dirty = bool(status.splitlines())
        return {"sha": sha, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def _collect_dependency_provenance() -> dict[str, Any]:
    """Collect upstream dependency provenance for paper-ready reproducibility."""

    deps: dict[str, Any] = {
        "python": sys.version,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV")
        or os.environ.get("CONDA_ENV")
        or None,
        "transformers": _safe_module_info("transformers"),
        "torch": _safe_module_info("torch"),
        "vllm": _safe_module_info("vllm"),
        "swift": _safe_module_info("swift"),
    }

    swift_info = deps.get("swift")
    swift_file = None
    if isinstance(swift_info, Mapping):
        swift_file = swift_info.get("file")
    if isinstance(swift_file, str) and swift_file:
        repo_root = _find_git_repo_root(Path(swift_file).resolve().parent)
        if repo_root is not None:
            deps["ms_swift"] = {
                "repo_root": str(repo_root),
                **_safe_git_state(repo_root),
            }

    return deps


_STAGE2_LAUNCHER_METADATA_ENV_KEYS = [
    "COORDEXP_STAGE2_LAUNCHER",
    "COORDEXP_STAGE2_SERVER_BASE_URL",
    "COORDEXP_STAGE2_SERVER_MODEL",
    "COORDEXP_STAGE2_SERVER_TORCH_DTYPE",
    "COORDEXP_STAGE2_SERVER_DP",
    "COORDEXP_STAGE2_SERVER_TP",
    "COORDEXP_STAGE2_SERVER_ENFORCE_EAGER",
    "COORDEXP_STAGE2_SERVER_GPU_MEMORY_UTILIZATION",
    "COORDEXP_STAGE2_SERVER_MAX_MODEL_LEN",
    "COORDEXP_STAGE2_SERVER_ENABLE_LORA",
    "COORDEXP_STAGE2_SERVER_GPUS",
    "COORDEXP_STAGE2_LEARNER_GPUS",
]


def _collect_launcher_metadata_from_env() -> dict[str, str]:
    meta: dict[str, str] = {}
    for key in _STAGE2_LAUNCHER_METADATA_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        meta[key] = value
    return meta


def _apply_rollout_decode_batch_size_override(*, train_args: Any, training_config: Any) -> int:
    """Override eval batching for rollout-aware trainer variants.

    Rollout-aware variants treat `rollout_matching.decode_batch_size` as the single
    source of truth for rollout decode/eval microbatching. We mirror that by
    forcing `per_device_eval_batch_size` to match the resolved decode batch size.

    Returns the resolved decode batch size.
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

    decode_bs_raw = rollout_cfg_for_batch.get("decode_batch_size", 1)
    try:
        rollout_decode_bs = int(decode_bs_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("rollout_matching.decode_batch_size must be an int") from exc
    if rollout_decode_bs <= 0:
        raise ValueError("rollout_matching.decode_batch_size must be > 0")

    if getattr(train_args, "training_args", None) is not None:
        current_eval_bs_raw = getattr(
            train_args.training_args,
            "per_device_eval_batch_size",
            rollout_decode_bs,
        )
        try:
            current_eval_bs = int(current_eval_bs_raw)
        except (TypeError, ValueError):
            current_eval_bs = int(rollout_decode_bs)
        if int(current_eval_bs) != int(rollout_decode_bs):
            logger.warning(
                "Overriding per_device_eval_batch_size=%s with rollout decode_batch_size=%s for rollout trainer variants.",
                int(current_eval_bs),
                int(rollout_decode_bs),
            )
        train_args.training_args.per_device_eval_batch_size = int(rollout_decode_bs)

    setattr(train_args, "per_device_eval_batch_size", int(rollout_decode_bs))
    return int(rollout_decode_bs)


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
    if not isinstance(cfg, Mapping):
        cfg = {}
    if not isinstance(coord_soft_cfg, Mapping):
        coord_soft_cfg = {}

    pipeline_raw = cfg.get("pipeline", None)
    if not isinstance(pipeline_raw, Mapping):
        pipeline_raw = {}

    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _finite_float(value: Any, default: float) -> float:
        out = _coerce_float(value, default)
        if not math.isfinite(out):
            raise ValueError("pipeline manifest contains non-finite float (NaN/Inf)")
        if out == 0.0:
            return 0.0
        return float(out)

    def _normalize_channels(channels_raw: Any) -> list[str]:
        found: set[str] = set()
        if isinstance(channels_raw, Sequence) and not isinstance(channels_raw, (str, bytes)):
            for ch in channels_raw:
                ch_s = str(ch).strip().upper()
                if ch_s in {"A", "B"}:
                    found.add(ch_s)
        if not found:
            return ["A", "B"]
        return [ch for ch in ("A", "B") if ch in found]

    def _normalize_json_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for k in sorted(value.keys(), key=lambda x: str(x)):
                out[str(k)] = _normalize_json_value(value[k])
            return out
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_normalize_json_value(v) for v in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return _finite_float(value, 0.0)
        try:
            f = float(value)
        except (TypeError, ValueError):
            return value
        return _finite_float(f, 0.0)

    def _coord_soft_defaults() -> dict[str, Any]:
        enabled = bool(coord_soft_cfg.get("enabled", False))
        soft_default = 1.0 if enabled else 0.0
        w1_default = 1.0 if enabled else 0.0
        gate_default = 1.0 if enabled else 0.0
        return {
            "coord_ce_weight": _finite_float(coord_soft_cfg.get("ce_weight", 0.0), 0.0),
            "soft_ce_weight": _finite_float(
                coord_soft_cfg.get("soft_ce_weight", soft_default),
                soft_default,
            ),
            "w1_weight": _finite_float(
                coord_soft_cfg.get("w1_weight", w1_default),
                w1_default,
            ),
            "coord_gate_weight": _finite_float(
                coord_soft_cfg.get("gate_weight", gate_default),
                gate_default,
            ),
            "temperature": _finite_float(
                coord_soft_cfg.get("temperature", 1.0),
                1.0,
            ),
            "target_sigma": _finite_float(
                coord_soft_cfg.get("target_sigma", 2.0),
                2.0,
            ),
            "target_truncate": coord_soft_cfg.get("target_truncate", None),
        }

    def _default_module_config(name: str) -> dict[str, Any]:
        variant = str(trainer_variant or "")
        coord_soft_defaults = _coord_soft_defaults()
        coord_soft_enabled = bool(coord_soft_cfg.get("enabled", False))

        if variant == "stage2_two_channel":
            channel_b_raw = cfg.get("channel_b", {})
            channel_b = channel_b_raw if isinstance(channel_b_raw, Mapping) else {}
            desc_w = _finite_float(cfg.get("desc_ce_weight", 1.0), 1.0)

            if name == "token_ce":
                return {
                    "desc_ce_weight": desc_w,
                    "self_context_struct_ce_weight": _finite_float(
                        cfg.get("fmt_struct_ce_weight", 0.1),
                        0.1,
                    ),
                    "rollout_fn_desc_weight": desc_w,
                    "rollout_matched_prefix_struct_weight": 1.0,
                    "rollout_drop_invalid_struct_ce_multiplier": _finite_float(
                        channel_b.get("drop_invalid_struct_ce_multiplier", 1.0),
                        1.0,
                    ),
                }

            if name == "bbox_geo":
                return {
                    "smoothl1_weight": _finite_float(
                        cfg.get("bbox_smoothl1_weight", 1.0),
                        1.0,
                    ),
                    "ciou_weight": _finite_float(
                        cfg.get("bbox_ciou_weight", 1.0),
                        1.0,
                    ),
                }

            if name == "coord_reg":
                return {
                    "coord_ce_weight": _finite_float(
                        cfg.get(
                            "coord_ce_weight",
                            coord_soft_defaults.get("coord_ce_weight", 0.0),
                        ),
                        0.0,
                    ),
                    "coord_el1_weight": _finite_float(
                        cfg.get("coord_el1_weight", 0.0),
                        0.0,
                    ),
                    "coord_ehuber_weight": _finite_float(
                        cfg.get("coord_ehuber_weight", 0.0),
                        0.0,
                    ),
                    "coord_huber_delta": _finite_float(
                        cfg.get("coord_huber_delta", 0.001),
                        0.001,
                    ),
                    "coord_entropy_weight": _finite_float(
                        cfg.get("coord_entropy_weight", 0.0),
                        0.0,
                    ),
                    "coord_gate_weight": _finite_float(
                        cfg.get(
                            "coord_gate_weight",
                            coord_soft_defaults.get("coord_gate_weight", 0.0),
                        ),
                        0.0,
                    ),
                    "text_gate_weight": _finite_float(
                        cfg.get("text_gate_weight", 0.0),
                        0.0,
                    ),
                    "soft_ce_weight": _finite_float(
                        coord_soft_defaults.get("soft_ce_weight", 0.0),
                        0.0,
                    ),
                    "self_context_soft_ce_weight": _finite_float(
                        coord_soft_defaults.get("soft_ce_weight", 0.0)
                        if coord_soft_enabled
                        else 0.05,
                        0.05,
                    ),
                    "w1_weight": _finite_float(
                        coord_soft_defaults.get("w1_weight", 0.0),
                        0.0,
                    ),
                    "temperature": _finite_float(
                        coord_soft_defaults.get("temperature", 1.0),
                        1.0,
                    ),
                    "target_sigma": _finite_float(
                        coord_soft_defaults.get("target_sigma", 2.0),
                        2.0,
                    ),
                    "target_truncate": coord_soft_defaults.get("target_truncate", None),
                }

        if variant == "stage2_rollout_aligned":
            if name == "token_ce":
                return {
                    "rollout_fn_desc_weight": 1.0,
                    "rollout_matched_prefix_struct_weight": 1.0,
                }

            if name == "bbox_geo":
                return {
                    "smoothl1_weight": _finite_float(
                        cfg.get("bbox_smoothl1_weight", 1.0),
                        1.0,
                    ),
                    "ciou_weight": _finite_float(
                        cfg.get("bbox_ciou_weight", 1.0),
                        1.0,
                    ),
                }

            if name == "coord_reg":
                return {
                    "coord_ce_weight": _finite_float(
                        coord_soft_defaults.get("coord_ce_weight", 0.0),
                        0.0,
                    ),
                    "soft_ce_weight": _finite_float(
                        coord_soft_defaults.get("soft_ce_weight", 0.0),
                        0.0,
                    ),
                    "w1_weight": _finite_float(
                        coord_soft_defaults.get("w1_weight", 0.0),
                        0.0,
                    ),
                    "coord_gate_weight": _finite_float(
                        coord_soft_defaults.get("coord_gate_weight", 0.0),
                        0.0,
                    ),
                    "text_gate_weight": 0.0,
                    "temperature": _finite_float(
                        coord_soft_defaults.get("temperature", 1.0),
                        1.0,
                    ),
                    "target_sigma": _finite_float(
                        coord_soft_defaults.get("target_sigma", 2.0),
                        2.0,
                    ),
                    "target_truncate": coord_soft_defaults.get("target_truncate", None),
                }

        return {}

    def _resolve(path: str, defaults: list[str]) -> list[dict[str, Any]]:
        raw = pipeline_raw.get(path, None)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raw = None

        if raw is None:
            return [
                {
                    "name": n,
                    "enabled": True,
                    "weight": 1.0,
                    "channels": ["A", "B"],
                    "config": _default_module_config(n),
                }
                for n in defaults
            ]

        out: list[dict[str, Any]] = []
        for spec in raw:
            if not isinstance(spec, Mapping):
                continue
            name = str(spec.get("name", "") or "").strip()
            if not name:
                continue
            authored_cfg_raw = spec.get("config", {})
            authored_cfg = (
                dict(authored_cfg_raw)
                if isinstance(authored_cfg_raw, Mapping)
                else {}
            )
            merged_cfg = dict(_default_module_config(name))
            merged_cfg.update(authored_cfg)

            out.append(
                {
                    "name": name,
                    "enabled": bool(spec.get("enabled", True)),
                    "weight": max(0.0, _finite_float(spec.get("weight", 1.0), 1.0)),
                    "channels": _normalize_channels(spec.get("channels", ["A", "B"])),
                    "config": merged_cfg,
                }
            )

        return out

    objective = _resolve("objective", default_objective)
    diagnostics = _resolve("diagnostics", default_diagnostics)

    extra: dict[str, Any] = {"variant": str(trainer_variant or "")}
    variant = str(trainer_variant or "")
    if variant == "stage2_two_channel":
        extra["stage2_ab.coord_ctx_embed_mode"] = str(
            cfg.get("coord_ctx_embed_mode", "st") or "st"
        ).strip().lower()
        extra["stage2_ab.coord_decode_mode"] = str(
            cfg.get("coord_decode_mode", "exp") or "exp"
        ).strip().lower()
    elif variant == "stage2_rollout_aligned":
        extra["rollout_matching.coord_decode_mode"] = str(
            cfg.get("coord_decode_mode", "exp") or "exp"
        ).strip().lower()

    payload = _normalize_json_value(
        {
            "objective": objective,
            "diagnostics": diagnostics,
            "extra": extra,
        }
    )

    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    run_context = {
        "config": str(config_path),
        "run_name": str(run_name or ""),
        "seed": int(seed or 0),
    }

    return {
        "payload": payload,
        "objective": payload.get("objective", []),
        "diagnostics": payload.get("diagnostics", []),
        "extra": payload.get("extra", {}),
        "checksum": checksum,
        "run_context": run_context,
    }


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

    add_version = getattr(train_args, "add_version", None)
    if add_version is None and training_args is not None:
        add_version = getattr(training_args, "add_version", None)
    add_version_enabled = bool(add_version) if add_version is not None else False

    # add_version already scopes output/logging under a versioned run dir in ms-swift.
    if run_name and not add_version_enabled:
        base_logging_dir = getattr(train_args, "logging_dir", None)
        if base_logging_dir is None and training_args is not None:
            base_logging_dir = getattr(training_args, "logging_dir", None)
        if base_logging_dir:
            base_logging_dir_norm = os.path.normpath(base_logging_dir)
            if os.path.basename(base_logging_dir_norm) != str(run_name):
                final_logging_dir = os.path.join(base_logging_dir, str(run_name))
                _set_train_dir_attr("logging_dir", final_logging_dir)

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

    # Auto-configure ROOT_IMAGE_DIR from a path hint (single JSONL or fusion config).
    train_jsonl = custom_config.train_jsonl or custom_config.extra.get("jsonl")
    fusion_config_path = getattr(custom_config, "fusion_config", None)
    if not train_jsonl and not fusion_config_path:
        raise ValueError(
            "Config must specify 'custom.train_jsonl'/'custom.jsonl' or 'custom.fusion_config'"
        )

    if os.environ.get("ROOT_IMAGE_DIR") in (None, ""):
        if train_jsonl:
            root_dir = os.path.abspath(os.path.dirname(str(train_jsonl)))
            os.environ["ROOT_IMAGE_DIR"] = root_dir
            logger.info(f"Set ROOT_IMAGE_DIR={root_dir} (from custom.train_jsonl)")
        elif fusion_config_path:
            # Fusion configs are legacy/experimental. Using the fusion-config file directory
            # as a root is a heuristic; surface this explicitly to prevent silent path drift.
            root_dir = os.path.abspath(os.path.dirname(str(fusion_config_path)))
            os.environ["ROOT_IMAGE_DIR"] = root_dir
            logger.warning(
                "Set ROOT_IMAGE_DIR=%s (heuristic from custom.fusion_config path). "
                "For fusion configs, set ROOT_IMAGE_DIR explicitly (preferred) or provide custom.train_jsonl.",
                root_dir,
            )

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
    curriculum_cfg = None
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

    # Build training dataset (single JSONL or fusion config)
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

    dataset_seed = _resolve_dataset_seed(training_config=training_config, train_args=train_args)
    dataset: Any
    fusion_cfg = None
    logger.info(
        "Serialization order config: object_ordering=%s object_field_order=%s",
        custom_config.object_ordering,
        custom_config.object_field_order,
    )

    if custom_config.fusion_config:
        from .datasets.fusion import FusionConfig
        from .datasets.unified_fusion_dataset import FusionCaptionDataset

        fusion_path = str(custom_config.fusion_config)
        logger.info(f"Loading training datasets from fusion config: {fusion_path}")
        fusion_cfg = FusionConfig.from_file(fusion_path)

        # For fusion, interpret sample_limit as a per-dataset cap for debug/smoke runs.
        fusion_train_limit: int | None = None
        if isinstance(train_sample_limit, int) and train_sample_limit > 0:
            fusion_train_limit = train_sample_limit
        elif isinstance(train_sample_limit, str) and train_sample_limit.isdigit():
            fusion_train_limit = int(train_sample_limit)

        dataset = FusionCaptionDataset(
            fusion_config=fusion_cfg,
            base_template=sft.template,
            user_prompt=custom_config.user_prompt,
            emit_norm=custom_config.emit_norm,
            json_format=custom_config.json_format,
            augmenter=augmenter,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            coord_tokens=custom_config.coord_tokens,
            seed=dataset_seed,
            shuffle=True,
            sample_limit=fusion_train_limit,
            split="train",
            object_ordering=custom_config.object_ordering,
            object_field_order=custom_config.object_field_order,
        )
    else:
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
            seed=dataset_seed,
            object_ordering=custom_config.object_ordering,
            object_field_order=custom_config.object_field_order,
        )
    packing_cfg = _parse_packing_config(
        training_config.training, sft.template, train_args
    )
    _validate_attention_backend_for_packing(training_config=training_config)
    trainer_variant = getattr(train_args, "trainer_variant", None)
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

        output_dir_raw = getattr(train_args, "output_dir", None)
        if not output_dir_raw:
            raise ValueError(
                "training.output_dir must be set when training.packing_mode=static"
            )

        train_dataloader_shuffle = bool(
            getattr(train_args, "train_dataloader_shuffle", True)
        )
        static_cache_dir = Path(str(output_dir_raw)) / "static_packing"
        static_fingerprint = _build_static_packing_fingerprint(
            training_config=training_config,
            custom_config=custom_config,
            template=sft.template,
            train_args=train_args,
            dataset_seed=dataset_seed,
            packing_cfg=packing_cfg,
            train_jsonl=str(train_jsonl) if train_jsonl else None,
            fusion_config_path=str(fusion_config_path) if fusion_config_path else None,
            dataset_split="train",
        )

        dataset = build_static_packed_dataset(
            dataset,
            template=sft.template,
            packing_length=packing_cfg.packing_length,
            min_fill_ratio=packing_cfg.min_fill_ratio,
            packing_drop_last=packing_cfg.drop_last,
            dataloader_drop_last=bool(getattr(train_args, "dataloader_drop_last", False)),
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

        # Stage2-ab step-budgeted trainers execute the forward/backward only on the
        # final micro-step of the accumulation window. If the train dataloader yields
        # a partial window at the end of an epoch, the outer Trainer will still call
        # optimizer.step(), but this trainer will not have produced gradients.
        #
        # We only support two safe modes today:
        # - dataloader_drop_last=true: we drop the final partial window via a dataloader wrapper
        # - dataloader_drop_last=false with per-rank batch count exactly divisible by GAS
        if (
            str(trainer_variant or "") == "stage2_two_channel"
            and bool(packing_cfg.enabled)
            and int(gas) > 1
        ):
            if int(per_rank_batches_est) < int(gas):
                raise ValueError(
                    "stage2-ab requires per-rank batches >= gradient_accumulation_steps. "
                    f"Got dataset_len={int(base_len_i)} world_size={int(world_size)} -> per_rank_batches_est={int(per_rank_batches_est)} "
                    f"but gradient_accumulation_steps={int(gas)}. "
                    "Mitigations: reduce gpus/world_size, increase custom.train_sample_limit, "
                    "or reduce training.effective_batch_size."
                )
            if (not drop_last_flag) and (int(per_rank_batches_est) % int(gas) != 0):
                raise ValueError(
                    "stage2-ab step-budgeted mode does not support a partial gradient-accumulation window. "
                    f"Got dataloader_drop_last=false with per_rank_batches_est={int(per_rank_batches_est)} and "
                    f"gradient_accumulation_steps={int(gas)} (remainder={int(per_rank_batches_est) % int(gas)}). "
                    "Mitigations: set training.dataloader_drop_last=true (recommended), or adjust "
                    "dataset size/world_size so per-rank batches is a multiple of gradient_accumulation_steps."
                )

    if base_len_i is not None and per_rank_batches_est_for_static is None:
        if drop_last_flag:
            per_rank_batches_est_for_static = int(base_len_i // per_device_train_bs)
        else:
            per_rank_batches_est_for_static = int(
                (base_len_i + per_device_train_bs - 1) // per_device_train_bs
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
            pv_shape = tuple(pv_shape_raw) if isinstance(pv_shape_raw, (list, tuple)) else None
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
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as e:
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
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as inner_e:
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
        except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as e:
            logger.warning(f"Failed to dump conversation text: {e}")

    # Build validation dataset (single JSONL or fusion config).
    eval_dataset = None
    val_jsonl = custom_config.val_jsonl
    if fusion_cfg is not None:
        from .datasets.unified_fusion_dataset import FusionCaptionDataset

        has_any_val = any(spec.val_jsonl is not None for spec in fusion_cfg.datasets)
        if has_any_val:
            logger.info(
                "Loading validation datasets from fusion config (val_jsonl entries only)"
            )
            eval_sample_limit = (
                None if val_sample_with_replacement else val_sample_limit
            )
            fusion_eval_limit: int | None = None
            if isinstance(eval_sample_limit, int) and eval_sample_limit > 0:
                fusion_eval_limit = eval_sample_limit
            elif isinstance(eval_sample_limit, str) and eval_sample_limit.isdigit():
                fusion_eval_limit = int(eval_sample_limit)

            eval_dataset = FusionCaptionDataset(
                fusion_config=fusion_cfg,
                base_template=sft.template,
                user_prompt=custom_config.user_prompt,
                emit_norm=custom_config.emit_norm,
                json_format=custom_config.json_format,
                augmenter=None,  # No augmentation for validation
                bypass_prob=0.0,  # Explicit: no bypass for validation
                curriculum_state=None,
                use_summary=use_summary,
                system_prompt_dense=system_prompt_dense,
                system_prompt_summary=system_prompt_summary,
                coord_tokens=custom_config.coord_tokens,
                seed=dataset_seed,
                shuffle=False,  # eval ordering doesn't matter; keep stable by default
                sample_limit=fusion_eval_limit,
                split="eval",
                object_ordering=custom_config.object_ordering,
                object_field_order=custom_config.object_field_order,
            )
            base_eval_len = len(eval_dataset)
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
        else:
            logger.info(
                "Fusion config has no val_jsonl entries; skipping evaluation dataset."
            )
            val_jsonl = None
    elif val_jsonl:
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
            seed=dataset_seed,
            object_ordering=custom_config.object_ordering,
            object_field_order=custom_config.object_field_order,
        )
        base_eval_len = len(eval_dataset)
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

        eval_output_dir_raw = getattr(train_args, "output_dir", None)
        if not eval_output_dir_raw:
            raise ValueError(
                "training.output_dir must be set when training.eval_packing=true"
            )

        eval_sample_limit_i: int | None = None
        if isinstance(val_sample_limit, int):
            eval_sample_limit_i = int(val_sample_limit)
        elif isinstance(val_sample_limit, str) and val_sample_limit.isdigit():
            eval_sample_limit_i = int(val_sample_limit)

        eval_cache_dir = Path(str(eval_output_dir_raw)) / "static_packing_eval"
        eval_fingerprint = _build_static_packing_fingerprint(
            training_config=training_config,
            custom_config=custom_config,
            template=sft.template,
            train_args=train_args,
            dataset_seed=dataset_seed,
            packing_cfg=packing_cfg,
            train_jsonl=str(val_jsonl) if val_jsonl else None,
            fusion_config_path=str(fusion_config_path) if fusion_config_path else None,
            dataset_split="eval",
            eval_sample_limit=eval_sample_limit_i,
            eval_sample_with_replacement=bool(val_sample_with_replacement),
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

        # Single rollout batching knob: rollout_matching.decode_batch_size.
        # Rollout trainer variants use this value for eval dataloader batch size too.
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
    instability_monitor_cfg = None
    extra_cfg = getattr(custom_config, "extra", None)
    if isinstance(extra_cfg, Mapping):
        raw_instab = extra_cfg.get("instability_monitor")
        if isinstance(raw_instab, dict):
            instability_monitor_cfg = raw_instab
    if trainer_variant in {"stage2_rollout_aligned", "stage2_two_channel"}:
        # Rollout-matching does its own encoding and loss masking inside the trainer.
        data_collator = base_collator
    else:
        data_collator = build_dataset_metrics_collator(
            sft.template,
            base_collator,
            token_type_cfg=token_type_cfg,
            instability_monitor_cfg=instability_monitor_cfg,
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
        output_dir_for_heartbeat = Path(str(getattr(train_args, "output_dir", ".") or "."))
        heartbeat_path = output_dir_for_heartbeat / f"train_heartbeat.rank{rank_s}.jsonl"
        heartbeat_writer = TrainHeartbeatWriter(path=heartbeat_path, enabled=True)
        heartbeat_writer.emit("heartbeat_enabled", rank=rank_s)
        data_collator = HeartbeatDataCollator(data_collator, writer=heartbeat_writer)
        heartbeat_callback = TrainHeartbeatCallback(heartbeat_writer)
        logger.info("Train heartbeat enabled: %s", str(heartbeat_path))

    trainer_cls = resolve_trainer_cls(train_args)
    mixins = []
    if trainer_variant not in {"stage2_rollout_aligned", "stage2_two_channel"}:
        # Fix transformers>=4.57 grad-accum scaling when model_accepts_loss_kwargs=True
        # (ms-swift uses Seq2SeqTrainer for causal_lm). This keeps train `loss` comparable to eval_loss.
        mixins.append(GradAccumLossScaleMixin)
        if isinstance(instability_monitor_cfg, dict) and bool(
            instability_monitor_cfg.get("enabled", False)
        ):
            mixins.append(InstabilityMonitorMixin)
        if token_type_cfg and getattr(token_type_cfg, "enabled", False):
            mixins.append(AggregateTokenTypeMetricsMixin)
        if coord_soft_ce_w1_cfg and getattr(coord_soft_ce_w1_cfg, "enabled", False):
            mixins.append(CoordSoftCEW1LossMixin)
    if mixins:
        trainer_cls = type(
            f"{trainer_cls.__name__}WithMetrics",
            tuple(mixins + [trainer_cls]),
            {},
        )

    # Add callbacks (including optional heartbeat instrumentation for debug/smokes).
    callbacks = sft.callbacks.copy() if sft.callbacks else []
    if heartbeat_callback is not None:
        callbacks.append(heartbeat_callback)
    if curriculum_scheduler is not None and curriculum_state is not None:
        from .callbacks.augmentation_curriculum import (
            AugmentationCurriculumCallback,
        )

        callbacks.append(
            AugmentationCurriculumCallback(
                scheduler=curriculum_scheduler,
                curriculum_state=curriculum_state,
            )
        )
    save_delay_cfg = getattr(train_args, "save_delay_config", None)
    from .callbacks import SaveDelayCallback

    if isinstance(save_delay_cfg, SaveDelayConfig) and save_delay_cfg.active:
        callbacks.append(SaveDelayCallback(config=save_delay_cfg))
        delay_info = (
            f"step {save_delay_cfg.steps}"
            if save_delay_cfg.steps is not None
            else f"epoch {save_delay_cfg.epochs}"
        )
        logger.info(
            f"SaveDelayCallback enabled: checkpoint saves blocked until {delay_info}"
        )
    else:
        save_delay_steps = getattr(train_args, "save_delay_steps", None)
        save_delay_epochs = getattr(train_args, "save_delay_epochs", None)
        if save_delay_steps is not None and save_delay_steps > 0:
            callbacks.append(SaveDelayCallback(save_delay_steps=save_delay_steps))
            logger.info(
                f"SaveDelayCallback enabled: no checkpoints until step {save_delay_steps}"
            )
        elif save_delay_epochs is not None and float(save_delay_epochs) > 0:
            callbacks.append(
                SaveDelayCallback(save_delay_epochs=float(save_delay_epochs))
            )
            logger.info(
                f"SaveDelayCallback enabled: no checkpoints until epoch {float(save_delay_epochs):.2f}"
            )

    trainer_kwargs = (
        sft._get_trainer_kwargs() if hasattr(sft, "_get_trainer_kwargs") else {}
    )
    if heartbeat_writer is not None:
        heartbeat_writer.emit("trainer_init_start")
    trainer = trainer_cls(
        model=sft.model,
        args=train_args.training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        template=sft.template,
        **trainer_kwargs,
    )
    if heartbeat_writer is not None:
        heartbeat_writer.emit("trainer_init_done")
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
                    "object_ordering": str(custom_config.object_ordering),
                    "object_field_order": str(custom_config.object_field_order),
                }
            )
            setattr(trainer, "rollout_matching_cfg", rollout_cfg)
            setattr(trainer, "object_field_order", str(custom_config.object_field_order))

            validate_hook = getattr(trainer, "_validate_rollout_matching_cfg", None)
            if callable(validate_hook):
                validate_hook()

            rollout_manifest = _resolve_pipeline_manifest(
                rollout_cfg,
                default_objective=["token_ce", "bbox_geo", "coord_reg"],
                default_diagnostics=["coord_diag"],
                coord_soft_cfg=coord_soft_cfg_for_manifest,
            )
            setattr(trainer, "rollout_pipeline_manifest", rollout_manifest)

            logger.info(
                "Rollout-matching config injected: rollout_backend=%s packing_enabled=%s pipeline_checksum=%s objective=%s diagnostics=%s config=%s run_name=%s seed=%s",
                rollout_cfg.get("rollout_backend", "vllm"),
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
            default_objective=["token_ce", "bbox_geo", "coord_reg"],
            default_diagnostics=["coord_diag"],
            coord_soft_cfg=coord_soft_cfg_for_manifest,
        )
        setattr(trainer, "stage2_pipeline_manifest", stage2_manifest)

        logger.info(
            "Stage2-AB config injected: b_ratio=%s n_softctx_iter=%s softctx_grad_mode=%s pipeline_checksum=%s objective=%s diagnostics=%s config=%s run_name=%s seed=%s",
            b_ratio,
            stage2_ab_cfg.get("n_softctx_iter"),
            stage2_ab_cfg.get("softctx_grad_mode"),
            stage2_manifest.get("checksum", ""),
            [m.get("name") for m in stage2_manifest.get("objective", [])],
            [m.get("name") for m in stage2_manifest.get("diagnostics", [])],
            str(config_path),
            str(getattr(train_args, "run_name", "") or ""),
            int(getattr(train_args.training_args, "seed", 0) or 0),
        )
    if coord_soft_ce_w1_cfg is not None:
        setattr(trainer, "coord_soft_ce_w1_cfg", coord_soft_ce_w1_cfg)
    if token_type_cfg is not None:
        setattr(trainer, "token_type_metrics_cfg", token_type_cfg)
    if instability_monitor_cfg is not None:
        setattr(trainer, "instability_monitor_cfg", instability_monitor_cfg)
        # Provide JSONL paths so the monitor can dump offending records by base_idx.
        setattr(trainer, "instability_train_jsonl", str(train_jsonl))
        if val_jsonl:
            setattr(trainer, "instability_val_jsonl", str(val_jsonl))


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
            )
            logger.info(
                "Wrote run manifest files: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(written.items())),
            )

        # ------------------------------------------------------------------
        # Required run metadata (fail-fast): git + upstream provenance
        # ------------------------------------------------------------------
        if not out_dir:
            raise ValueError("train_args.output_dir is not set; cannot write run metadata")

        repo_root = Path(__file__).resolve().parents[1]

        def _git(*argv: str) -> str:
            return subprocess.check_output(
                ["git", *argv],
                cwd=str(repo_root),
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()

        git_sha = None
        git_branch = None
        git_dirty = None
        status_lines = []
        try:
            git_sha = _git("rev-parse", "HEAD")
            git_branch = _git("rev-parse", "--abbrev-ref", "HEAD")
            status = _git("status", "--porcelain")
            status_lines = [line for line in status.splitlines() if line.strip()]
            git_dirty = bool(status_lines)
        except Exception:
            git_sha = None
            git_branch = None
            git_dirty = None
            status_lines = []

        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": str(getattr(args, "config", "") or ""),
            "base_config": str(getattr(args, "base_config", "") or ""),
            "run_name": str(getattr(train_args, "run_name", "") or ""),
            "output_dir": str(out_dir),
            "git_sha": git_sha,
            "git_branch": git_branch,
            "git_dirty": git_dirty,
            "git_status_porcelain": status_lines[:200],
            "dataset_seed": dataset_seed,
            "upstream": _collect_dependency_provenance(),
        }

        if written is not None:
            meta["run_manifest_files"] = dict(written)

        launcher_meta = _collect_launcher_metadata_from_env()
        if launcher_meta:
            meta["launcher"] = launcher_meta

        out_path = Path(str(out_dir)) / "run_metadata.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
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
                    except (AttributeError, IndexError, OSError, RuntimeError, TypeError) as cleanup_error:
                        # Ignore cleanup errors - they're harmless at this point
                        # Training already completed successfully
                        logger.debug(
                            f"DeepSpeed cleanup warning (non-fatal): {cleanup_error}"
                        )
        except (AttributeError, IndexError, OSError, RuntimeError, TypeError) as e:
            # Non-fatal: training already completed successfully
            logger.debug(f"Cleanup warning (non-fatal): {e}")


if __name__ == "__main__":
    main()
