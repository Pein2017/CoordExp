"""SFT runner - pure YAML config-driven, no CLI arguments for hyperparameters"""

import argparse
import copy
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
from typing import Any, Mapping

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
from .datasets import BaseCaptionDataset, RandomSampleDataset, build_packed_dataset
from .datasets.augmentation.curriculum import AugmentationCurriculumScheduler
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
        from .trainers.stage2_ab_training import Stage2ABTrainingTrainer

        trainer_cls = Stage2ABTrainingTrainer
    elif trainer_variant == "rollout_matching_sft":
        from .trainers.rollout_matching_sft import RolloutMatchingSFTTrainer

        trainer_cls = RolloutMatchingSFTTrainer
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
    packing_length: int = 0
    buffer_size: int = 512
    min_fill_ratio: float = 0.65
    drop_last: bool = True
    allow_single_long: bool = True
    eval_packing: bool = False


def _parse_packing_config(
    training_cfg: Any, template: Any, train_args: Any
) -> PackingRuntimeConfig:
    cfg = training_cfg or {}
    enabled = bool(cfg.get("packing", False))
    if not enabled:
        return PackingRuntimeConfig(enabled=False)

    default_length = getattr(template, "max_length", None) or getattr(
        train_args, "max_model_len", None
    )

    # Packing length is derived from the model/template max length.
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
    eval_packing = bool(cfg.get("eval_packing", False))

    return PackingRuntimeConfig(
        enabled=True,
        packing_length=packing_length,
        buffer_size=buffer_size,
        min_fill_ratio=min_fill_ratio,
        drop_last=drop_last,
        allow_single_long=allow_single_long,
        eval_packing=eval_packing,
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
    except Exception as exc:
        raise TypeError(
            f"training.seed must be an int (or int-like); got {seed_raw!r}"
        ) from exc

    return seed


def _safe_module_info(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
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
    except Exception as exc:
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
    if trainer_variant not in {"rollout_matching_sft", "stage2_ab_training"}:
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
    except Exception as exc:
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
        except Exception:
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


def main():
    """Main training entry point - pure config-driven."""
    args = parse_args()

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
    except Exception:
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
        root_hint = train_jsonl or fusion_config_path
        if root_hint:
            root_dir = os.path.abspath(os.path.dirname(str(root_hint)))
            os.environ["ROOT_IMAGE_DIR"] = root_dir
            logger.info(f"Set ROOT_IMAGE_DIR={root_dir}")

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
            try:
                setattr(train_args, "modules_to_save", modules_to_save)
            except Exception:
                logger.warning(
                    "Failed to attach coord_offset module to modules_to_save on train_args"
                )
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

    # Configure augmentation via YAML builder (applies only to training)
    augmenter = None
    bypass_prob = float(custom_config.bypass_prob)
    aug_cfg = custom_config.augmentation
    curriculum_cfg = None
    if isinstance(aug_cfg, dict) and aug_cfg.get("enabled", True):
        try:
            # Ensure ops are registered by importing ops module
            from .datasets.augmentation import ops as _register_ops  # noqa: F401
            from .datasets.augmentation.builder import build_compose_from_config

            augmenter = build_compose_from_config(aug_cfg)
            bypass_prob = float(aug_cfg.get("bypass_prob", custom_config.bypass_prob))
            curriculum_cfg = aug_cfg.get("curriculum")
            logger.info(
                f"Augmentation pipeline built (bypass_prob={bypass_prob:.2f}, training only)"
            )
        except Exception as e:
            raise ValueError(f"Failed to build augmentation pipeline from YAML: {e}")

    curriculum_state = None
    curriculum_scheduler = None
    if curriculum_cfg is None:
        curriculum_cfg = custom_config.augmentation_curriculum
    if curriculum_cfg:
        if augmenter is None:
            raise ValueError(
                "augmentation curriculum requires a built augmentation pipeline"
            )
        try:
            scheduler = AugmentationCurriculumScheduler.from_config(
                base_bypass=bypass_prob,
                op_meta=getattr(augmenter, "_augmentation_meta", []),
                curriculum_raw=curriculum_cfg,
            )
        except Exception as exc:
            raise ValueError(f"Failed to build augmentation curriculum: {exc}") from exc
        curriculum_scheduler = scheduler
        # Note: initial_state will be computed after dataset is loaded and total_steps is calculated

    # Sample limits for quick smoke tests.
    #
    # When debug.enabled=true, use debug.{train,val}_sample_limit (optional) and
    # ignore custom.* limits. Otherwise use custom.{train,val}_sample_limit with
    # no shared fallback (explicit is better than implicit).
    debug_enabled = bool(
        debug_config is not None and getattr(debug_config, "enabled", False)
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
    except Exception:
        base_dataset_len = None
    # Stage_2 rollout-matching supports post-rollout packing inside the trainer only.
    # Do not apply dataset-level packing wrappers for this trainer variant.
    if packing_cfg.enabled and trainer_variant not in {
        "rollout_matching_sft",
        "stage2_ab_training",
    }:
        orig_bs = getattr(train_args, "per_device_train_batch_size", None)
        if orig_bs != 1:
            logger.warning(
                "packing enabled: forcing per_device_train_batch_size=1 (was %s)",
                orig_bs,
            )
            try:
                train_args.per_device_train_batch_size = 1
                if getattr(train_args, "training_args", None) is not None:
                    train_args.training_args.per_device_train_batch_size = 1
            except Exception:
                logger.warning(
                    "Failed to set per_device_train_batch_size on train_args"
                )

        # Ensure max_steps is finite when using iterable packed dataset
        max_steps = getattr(train_args, "max_steps", None)
        if max_steps is None or max_steps <= 0:
            if base_dataset_len is not None:
                grad_acc = getattr(train_args, "gradient_accumulation_steps", 1) or 1
                _, _, world_size, _ = get_dist_setting()
                world_size = max(world_size, 1)
                # Heuristic: estimate average samples per emitted pack to align optimizer steps with requested epochs.
                # Defaults to ~8 samples/pack (observed in packing guide); override via training.packing_avg_samples.
                try:
                    training_map = getattr(training_config, "training", {}) or {}
                    avg_pack_samples = float(
                        training_map.get("packing_avg_samples", 8.0)
                    )
                except Exception:
                    avg_pack_samples = 8.0
                avg_pack_samples = max(avg_pack_samples, 1e-6)
                packs_per_rank_est = math.ceil(
                    base_dataset_len / (avg_pack_samples * world_size)
                )
                steps_per_epoch_est = math.ceil(packs_per_rank_est / grad_acc)

                epochs_target = getattr(train_args, "num_train_epochs", None)
                if epochs_target is None or epochs_target <= 0:
                    est_total = steps_per_epoch_est
                else:
                    est_total = math.ceil(steps_per_epoch_est * float(epochs_target))

                train_args.max_steps = est_total
                if getattr(train_args, "training_args", None) is not None:
                    train_args.training_args.max_steps = est_total
                logger.warning(
                    (
                        "packing enabled with iterable dataset: auto-setting max_steps=%s "
                        "(base_len=%s, avg_pack_samples=%.3f, packs_per_rank_est=%s, "
                        "grad_acc=%s, world_size=%s, target_epochs=%s)"
                    ),
                    est_total,
                    base_dataset_len,
                    avg_pack_samples,
                    packs_per_rank_est,
                    grad_acc,
                    world_size,
                    epochs_target,
                )
            else:
                raise ValueError(
                    "packing enabled: max_steps must be set to a positive value when dataset length is unknown"
                )

        dataset = build_packed_dataset(
            dataset,
            template=sft.template,
            packing_length=packing_cfg.packing_length,
            buffer_size=packing_cfg.buffer_size,
            min_fill_ratio=packing_cfg.min_fill_ratio,
            drop_last=packing_cfg.drop_last,
            allow_single_long=packing_cfg.allow_single_long,
        )
        logger.info(
            "Packing enabled: length=%s buffer=%s min_fill=%.2f drop_last=%s allow_single_long=%s",
            packing_cfg.packing_length,
            packing_cfg.buffer_size,
            packing_cfg.min_fill_ratio,
            packing_cfg.drop_last,
            packing_cfg.allow_single_long,
        )
    elif packing_cfg.enabled and trainer_variant in {
        "rollout_matching_sft",
        "stage2_ab_training",
    }:
        logger.info(
            "Packing enabled for rollout-matching: dataset packing is disabled; "
            "trainer will apply dynamic post-rollout packing for the teacher-forced forward pass."
        )

    if (
        packing_cfg.enabled
        and packing_cfg.eval_packing
        and trainer_variant not in {"rollout_matching_sft", "stage2_ab_training"}
    ):
        eval_bs = getattr(train_args, "per_device_eval_batch_size", None)
        if eval_bs != 1:
            logger.warning(
                "eval_packing enabled: forcing per_device_eval_batch_size=1 (was %s)",
                eval_bs,
            )
            try:
                train_args.per_device_eval_batch_size = 1
                if getattr(train_args, "training_args", None) is not None:
                    train_args.training_args.per_device_eval_batch_size = 1
            except Exception:
                logger.warning("Failed to set per_device_eval_batch_size on train_args")

    try:
        train_len = len(dataset)
    except Exception:
        train_len = base_dataset_len
    logger.info(f"Training dataset size (reported/approx): {train_len}")

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
            try:
                grid_shape = tuple(getattr(img_grid, "shape", []))
            except Exception:
                grid_shape = None
            try:
                pv_shape = tuple(getattr(pv, "shape", []))
            except Exception:
                pv_shape = None
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
                except Exception:
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
        except Exception as e:
            logger.warning(f"HealthCheck failed: {e}")

    # Optional: dump conversation text-only (no tokens, no images) and full tokens
    dump_conv = bool(custom_config.dump_conversation_text or args.debug)
    try:
        dataset_nonempty = len(dataset) > 0
    except Exception:
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
            except Exception as inner_e:
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
        except Exception as e:
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
        and trainer_variant not in {"rollout_matching_sft", "stage2_ab_training"}
    ):
        eval_dataset = build_packed_dataset(
            eval_dataset,
            template=sft.template,
            packing_length=packing_cfg.packing_length,
            buffer_size=packing_cfg.buffer_size,
            min_fill_ratio=packing_cfg.min_fill_ratio,
            drop_last=False,
            allow_single_long=packing_cfg.allow_single_long,
        )
        logger.info(
            "Packing enabled for eval: length=%s buffer=%s min_fill=%.2f drop_last=False allow_single_long=%s",
            packing_cfg.packing_length,
            packing_cfg.buffer_size,
            packing_cfg.min_fill_ratio,
            packing_cfg.allow_single_long,
        )

    # Sample printing disabled to avoid dumping labels/ids

    # CRITICAL: Apply tuner (LoRA/adapters) before creating trainer
    logger.info("Preparing model with tuner...")
    sft.model = sft.prepare_model(
        train_args, sft.model, template=sft.template, train_dataset=dataset
    )
    # After PEFT wrapping, reattach coord-offset hooks to active modules so offsets train/save correctly
    if coord_offset_cfg and coord_offset_cfg.enabled:
        try:
            reattached = reattach_coord_offset_hooks(sft.model)
            if reattached is None:
                logger.warning(
                    "coord_offset_adapter not found after prepare_model; hooks not reattached"
                )
            else:
                logger.info("Reattached coord_offset hooks on wrapped model")
        except Exception as exc:
            logger.warning(
                f"Failed to reattach coord_offset hooks after prepare_model: {exc}"
            )
    logger.info(f"Model after tuner: {type(sft.model).__name__}")

    # Setup trainer
    logger.info("Setting up trainer...")
    trainer_variant = getattr(train_args, "trainer_variant", None)
    if trainer_variant in {"rollout_matching_sft", "stage2_ab_training"}:
        # Keep raw fields (messages/assistant_payload) for rollout→parse→match construction.
        try:
            if getattr(train_args, "training_args", None) is not None:
                train_args.training_args.remove_unused_columns = False
        except Exception:
            raise

        # Single rollout batching knob: rollout_matching.decode_batch_size.
        # Rollout trainer variants use this value for eval dataloader batch size too.
        _apply_rollout_decode_batch_size_override(
            train_args=train_args,
            training_config=training_config,
        )

        try:
            from swift.trainers.rlhf_trainer.utils import identity_data_collator

            base_collator = identity_data_collator
        except Exception as exc:
            raise RuntimeError(
                "rollout-matching trainer requires ms-swift identity_data_collator"
            ) from exc
    else:
        base_collator = sft._get_data_collator()
    token_type_cfg = getattr(custom_config, "token_type_metrics", None)
    coord_soft_ce_w1_cfg = getattr(custom_config, "coord_soft_ce_w1", None)
    instability_monitor_cfg = None
    try:
        raw_instab = getattr(custom_config, "extra", {}).get("instability_monitor")
        if isinstance(raw_instab, dict):
            instability_monitor_cfg = raw_instab
    except Exception:
        instability_monitor_cfg = None
    if trainer_variant in {"rollout_matching_sft", "stage2_ab_training"}:
        # Rollout-matching does its own encoding and loss masking inside the trainer.
        data_collator = base_collator
    else:
        data_collator = build_dataset_metrics_collator(
            sft.template,
            base_collator,
            token_type_cfg=token_type_cfg,
            instability_monitor_cfg=instability_monitor_cfg,
        )
    trainer_cls = resolve_trainer_cls(train_args)
    mixins = []
    if trainer_variant not in {"rollout_matching_sft", "stage2_ab_training"}:
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

    # Add SaveDelayCallback if save_delay_steps is configured
    callbacks = sft.callbacks.copy() if sft.callbacks else []
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
    # Rollout-matching evaluators emit rollout/* metrics only.
    # Guard against inherited defaults like eval_token_acc, which would crash
    # best-checkpoint selection at evaluation time.
    if (
        trainer_variant in {"rollout_matching_sft", "stage2_ab_training"}
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

    if trainer_variant in {"rollout_matching_sft", "stage2_ab_training"}:
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

            # Inject packing runtime knobs (stage_2 uses dynamic post-rollout packing; dataset packing is disabled).
            rollout_cfg.update(
                {
                    "packing_enabled": bool(packing_cfg.enabled),
                    "packing_length": int(packing_cfg.packing_length),
                    "packing_buffer": int(packing_cfg.buffer_size),
                    "packing_min_fill_ratio": float(packing_cfg.min_fill_ratio),
                    "packing_drop_last": bool(packing_cfg.drop_last),
                    "object_field_order": str(custom_config.object_field_order),
                }
            )
            setattr(trainer, "rollout_matching_cfg", rollout_cfg)
            setattr(trainer, "object_field_order", str(custom_config.object_field_order))

            validate_hook = getattr(trainer, "_validate_rollout_matching_cfg", None)
            if callable(validate_hook):
                validate_hook()

            logger.info(
                "Rollout-matching config injected: rollout_backend=%s packing_enabled=%s",
                rollout_cfg.get("rollout_backend", "vllm"),
                rollout_cfg.get("packing_enabled", False),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to inject rollout_matching_cfg into trainer. "
                "This is required for rollout-matching/stage2-ab trainer variants."
            ) from exc

    if trainer_variant == "stage2_ab_training":
        try:
            stage2_ab_typed = getattr(training_config, "stage2_ab", None)
            if stage2_ab_typed is None:
                raise ValueError(
                    "training_config.stage2_ab is required for stage2_ab_training; "
                    "check config parsing (top-level stage2_ab section)."
                )
            stage2_ab_cfg: dict[str, Any] = asdict(stage2_ab_typed)


            setattr(trainer, "stage2_ab_cfg", stage2_ab_cfg)

            try:
                sched = stage2_ab_cfg.get("schedule")
                b_ratio = sched.get("b_ratio") if isinstance(sched, Mapping) else None
            except Exception:
                b_ratio = None

            logger.info(
                "Stage2-AB config injected: b_ratio=%s n_softctx_iter=%s softctx_grad_mode=%s",
                b_ratio,
                stage2_ab_cfg.get("n_softctx_iter"),
                stage2_ab_cfg.get("softctx_grad_mode"),
            )
        except Exception as exc:
            logger.warning("Failed to inject stage2_ab_cfg into trainer: %s", exc)
    if coord_soft_ce_w1_cfg is not None:
        try:
            setattr(trainer, "coord_soft_ce_w1_cfg", coord_soft_ce_w1_cfg)
        except Exception:
            raise
    if token_type_cfg is not None:
        try:
            setattr(trainer, "token_type_metrics_cfg", token_type_cfg)
        except Exception:
            raise
    if instability_monitor_cfg is not None:
        try:
            setattr(trainer, "instability_monitor_cfg", instability_monitor_cfg)
        except Exception:
            raise
        # Provide JSONL paths so the monitor can dump offending records by base_idx.
        try:
            setattr(trainer, "instability_train_jsonl", str(train_jsonl))
        except Exception:
            raise
        try:
            if val_jsonl:
                setattr(trainer, "instability_val_jsonl", str(val_jsonl))
        except Exception:
            raise

    # Patch DeepSpeed __del__ to avoid noisy cleanup errors (safe no-op)
    try:
        import deepspeed  # type: ignore

        if hasattr(deepspeed.runtime.engine.DeepSpeedEngine, "__del__"):
            _orig_ds_del = deepspeed.runtime.engine.DeepSpeedEngine.__del__

            def _safe_ds_del(self):  # type: ignore[override]
                try:
                    _orig_ds_del(self)
                except Exception:
                    raise

            deepspeed.runtime.engine.DeepSpeedEngine.__del__ = _safe_ds_del  # type: ignore[assignment]
    except Exception:
        raise

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
    except Exception:
        is_rank0 = True

    if is_rank0:
        try:
            out_dir = getattr(train_args, "output_dir", None)
            if out_dir:
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
                    status_lines = [
                        line for line in status.splitlines() if line.strip()
                    ]
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
        except Exception as exc:
            logger.warning("Failed to write run metadata: %s", exc)

    try:
        sft.train(trainer)
    except torch.cuda.OutOfMemoryError:
        debug_info = getattr(dataset, "last_sample_debug", None)
        logger.error(f"CUDA OOM encountered. Last sample debug: {debug_info}")
        raise
    finally:
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
                    except Exception as cleanup_error:
                        # Ignore cleanup errors - they're harmless at this point
                        # Training already completed successfully
                        logger.debug(
                            f"DeepSpeed cleanup warning (non-fatal): {cleanup_error}"
                        )
        except Exception as e:
            # Non-fatal: training already completed successfully
            logger.debug(f"Cleanup warning (non-fatal): {e}")


if __name__ == "__main__":
    main()
