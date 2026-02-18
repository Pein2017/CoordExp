"""Pure YAML config loader - directly instantiates ms-swift objects"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from swift.llm.argument import RLHFArguments, TrainArguments
from swift.utils import get_dist_setting

from .prompts import (
    SYSTEM_PROMPT_SUMMARY,
    USER_PROMPT_SUMMARY,
    get_template_prompts,
)
from .schema import PromptOverrides, SaveDelayConfig, TrainingConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load YAML config and directly instantiate ms-swift dataclasses.

    No CLI argument parsing - direct object construction from YAML.
    All hyperparameters must be explicitly defined in YAML.
    """

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load YAML file into dictionary.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def _normalize_to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value in (0, 1, 0.0, 1.0):
                return bool(value)
            raise ValueError(f"{field_name} must be boolean (0 or 1), got {value!r}.")
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError(
                f"{field_name} string value '{value}' is not a recognized boolean representation."
            )
        raise TypeError(f"{field_name} must be a boolean value, got {type(value)!r}.")

    @staticmethod
    def _canonical_stage2_profile_kind(config_path: str) -> Optional[str]:
        config_abs = Path(config_path).resolve()
        repo_root = Path(__file__).resolve().parents[2]
        stage2_root = (repo_root / "configs" / "stage2_ab").resolve()
        for kind in ("prod", "smoke"):
            kind_root = (stage2_root / kind).resolve()
            try:
                config_abs.relative_to(kind_root)
            except ValueError:
                continue
            if config_abs.suffix.lower() == ".yaml":
                return kind
        return None

    @staticmethod
    def _lookup_nested_key(payload: Dict[str, Any], key_path: str) -> bool:
        node: Any = payload
        for segment in key_path.split("."):
            if not isinstance(node, dict) or segment not in node:
                return False
            node = node[segment]
        return True

    @staticmethod
    def _validate_stage2_leaf_contract(config_path: str) -> None:
        if ConfigLoader._canonical_stage2_profile_kind(config_path) is None:
            return

        raw_cfg = ConfigLoader.load_yaml(config_path) or {}
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Stage-2 profile must be a mapping: {config_path}")

        extends_value = raw_cfg.get("extends", raw_cfg.get("inherit"))
        extends_list = ConfigLoader._normalize_to_list(extends_value)
        if len(extends_list) != 1 or extends_list[0] != "../base.yaml":
            raise ValueError(
                "Stage-2 canonical profile leaves must extend exactly one parent '../base.yaml'. "
                f"Invalid extends in {config_path}: {extends_value!r}"
            )

        required_leaf_keys = [
            "model.model",
            "training.run_name",
            "training.output_dir",
            "training.logging_dir",
            "training.learning_rate",
            "training.vit_lr",
            "training.aligner_lr",
            "training.effective_batch_size",
            "training.eval_strategy",
            "training.eval_steps",
            "training.save_strategy",
            "training.save_steps",
            "stage2_ab.schedule.b_ratio",
            "stage2_ab.n_softctx_iter",
        ]

        missing = [
            key for key in required_leaf_keys if not ConfigLoader._lookup_nested_key(raw_cfg, key)
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                "Stage-2 canonical profile leaves must explicitly define required keys. "
                f"Missing in {config_path}: {missing_str}"
            )

    @staticmethod
    def load_yaml_with_extends(
        config_path: str, _visited: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Load YAML and resolve inheritance via 'extends'/'inherit'.

        Supports a top-level key in the YAML:
          - extends: str | list[str]     # relative to the current file
          - inherit: str | list[str]     # alias of extends

        Bases are merged in order (earlier are lower precedence).
        The current file has the highest precedence.
        Cycles are detected and will raise a ValueError.
        """
        abs_path = str(Path(config_path).resolve())
        visited: Set[str] = set(_visited or set())
        if abs_path in visited:
            raise ValueError(f"Cyclic config inheritance detected at: {abs_path}")
        visited.add(abs_path)

        current_dir = Path(abs_path).parent
        config = ConfigLoader.load_yaml(abs_path) or {}

        # Gather base paths from supported keys
        extends_value = None
        if isinstance(config, dict):
            extends_value = config.pop("extends", None)
            if extends_value is None:
                extends_value = config.pop("inherit", None)

        base_paths = ConfigLoader._normalize_to_list(extends_value)

        # Merge all bases in order
        merged_base: Dict[str, Any] = {}
        for base_ref in base_paths:
            base_path = Path(base_ref)
            if not base_path.is_absolute():
                base_path = (current_dir / base_path).resolve()
            base_cfg = ConfigLoader.load_yaml_with_extends(str(base_path), visited)
            merged_base = ConfigLoader.merge_configs(merged_base, base_cfg)

        # Finally merge current file on top
        return ConfigLoader.merge_configs(merged_base, config)

    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """Deep merge two config dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        merged = base.copy()
        for key, value in override.items():
            if (
                isinstance(value, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def resolve_prompts(config: Dict[str, Any]) -> PromptOverrides:
        prompts_config = config.get("prompts", {}) or {}
        if not isinstance(prompts_config, dict):
            raise TypeError("prompts section must be a mapping if provided")
        if prompts_config:
            raise ValueError(
                "YAML prompt overrides are disabled. Edit src/config/prompts.py instead."
            )

        use_summary = False
        custom_section = config.get("custom")
        prompt_variant: Optional[str] = None
        if custom_section is not None:
            if not isinstance(custom_section, dict):
                raise TypeError(
                    "custom section must be a mapping when resolving prompts"
                )
            if "summary_ratio" in custom_section:
                raise ValueError(
                    "custom.summary_ratio has been removed; use custom.use_summary instead."
                )
            if "use_summary" in custom_section:
                use_summary = ConfigLoader._coerce_bool(
                    custom_section["use_summary"], "custom.use_summary"
                )

            extra_cfg = custom_section.get("extra", {})
            if extra_cfg is None:
                extra_cfg = {}
            if not isinstance(extra_cfg, dict):
                raise TypeError("custom.extra must be a mapping when resolving prompts")
            prompt_variant_raw = extra_cfg.get("prompt_variant")
            if prompt_variant_raw is not None and not isinstance(prompt_variant_raw, str):
                raise TypeError(
                    "custom.extra.prompt_variant must be a string when provided"
                )
            prompt_variant = prompt_variant_raw

        if use_summary:
            default_system = SYSTEM_PROMPT_SUMMARY
            default_user = USER_PROMPT_SUMMARY
            output_variant = "summary"
        else:
            default_system, default_user = get_template_prompts(
                prompt_variant=prompt_variant,
            )
            output_variant = "dense"

        system_prompt = default_system
        user_prompt = default_user

        return PromptOverrides(
            system=str(system_prompt) if system_prompt is not None else None,
            user=str(user_prompt) if user_prompt is not None else None,
            output_variant=output_variant,
        )

    @staticmethod
    def build_train_arguments(config: TrainingConfig) -> TrainArguments:
        """Directly instantiate TrainArguments from config.

        TrainArguments is a unified dataclass that inherits from:
        - Seq2SeqTrainingArguments (HuggingFace Transformers)
        - TunerArguments (LoRA, adapters, etc.)
        - DataArguments (dataset configuration)
        - ModelArguments (model loading)
        - QuantizeArguments (quantization)
        - TemplateArguments (prompt templates)
        - SwanlabArguments (logging)

        We merge all config sections and pass to TrainArguments constructor,
        which will use ms-swift's built-in defaults for any missing fields.

        Args:
            config: Configuration dictionary from YAML

        Returns:
            Fully initialized TrainArguments object
        """
        model_section = dict(config.model)
        quant_section = dict(config.quantization)
        data_section = dict(config.data)
        template_section = dict(config.template)
        tuner_section = dict(config.tuner)
        training_section = dict(config.training)
        rlhf_section_original = dict(config.rlhf)
        rlhf_section = dict(rlhf_section_original)
        llm_kd_weight_raw = rlhf_section.pop("llm_kd_weight", None)
        if llm_kd_weight_raw is None:
            llm_kd_weight = 1.0
        else:
            try:
                llm_kd_weight = float(llm_kd_weight_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("rlhf.llm_kd_weight must be a numeric value") from exc
            if not math.isfinite(llm_kd_weight):
                raise ValueError(
                    f"rlhf.llm_kd_weight must be finite, got {llm_kd_weight_raw!r}"
                )
            if llm_kd_weight < 0:
                raise ValueError(
                    f"rlhf.llm_kd_weight must be >= 0, got {llm_kd_weight_raw!r}"
                )

        raw_save_delay_steps = training_section.pop("save_delay_steps", None)
        raw_save_delay_epochs = training_section.pop("save_delay_epochs", None)
        save_last_epoch_raw = training_section.pop("save_last_epoch", None)
        if save_last_epoch_raw is None:
            save_last_epoch = True
        else:
            save_last_epoch = ConfigLoader._coerce_bool(
                save_last_epoch_raw, "training.save_last_epoch"
            )

        # Remove packing-only knobs before TrainArguments init; they are consumed in sft.py
        _packing_keys = {
            "packing",
            "packing_length",
            "packing_buffer",
            "packing_min_fill_ratio",
            "packing_drop_last",
            "packing_allow_single_long",
            "eval_packing",
            "packing_avg_samples",
        }
        for key in _packing_keys:
            training_section.pop(key, None)

        # Auto-calculate gradient_accumulation_steps from effective_batch_size
        #
        # Stage2-AB standardizes step semantics around a true (exact) global effective batch.
        is_stage2_ab = bool(
            str(getattr(getattr(config, "custom", None), "trainer_variant", "") or "")
            == "stage2_ab_training"
        )

        effective_batch_size = training_section.pop("effective_batch_size", None)
        if is_stage2_ab and effective_batch_size is None:
            raise ValueError(
                "stage2_ab_training requires training.effective_batch_size to be set (global raw rollouts per optimizer step)."
            )

        if effective_batch_size is not None:
            try:
                effective_batch_size = int(effective_batch_size)
            except (TypeError, ValueError) as exc:
                raise ValueError("training.effective_batch_size must be an integer") from exc
            if effective_batch_size <= 0:
                raise ValueError(
                    f"training.effective_batch_size must be > 0, got {effective_batch_size}"
                )

            per_device_train_batch_size = training_section.get(
                "per_device_train_batch_size", 1
            )
            try:
                per_device_train_batch_size = int(per_device_train_batch_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "training.per_device_train_batch_size must be an integer"
                ) from exc
            if per_device_train_batch_size <= 0:
                raise ValueError(
                    f"training.per_device_train_batch_size must be > 0, got {per_device_train_batch_size}"
                )

            # Get world_size (number of GPUs) from environment
            _, _, world_size, _ = get_dist_setting()
            if world_size <= 0:
                world_size = 1

            # Calculate gradient_accumulation_steps
            # Formula: effective_batch_size = per_device_train_batch_size × world_size × gradient_accumulation_steps
            denominator = per_device_train_batch_size * world_size
            if denominator <= 0:
                denominator = 1

            if is_stage2_ab and (effective_batch_size % denominator != 0):
                raise ValueError(
                    "For stage2_ab_training, training.effective_batch_size must be divisible by "
                    f"training.per_device_train_batch_size*world_size ({per_device_train_batch_size}*{world_size}={denominator}). "
                    f"Got effective_batch_size={effective_batch_size}."
                )

            user_gas_raw = training_section.get("gradient_accumulation_steps", None)

            if is_stage2_ab:
                gradient_accumulation_steps = max(
                    1, int(effective_batch_size // denominator)
                )
            else:
                gradient_accumulation_steps = max(
                    1, math.ceil(effective_batch_size / denominator)
                )

            # Stage2-AB standardizes on effective_batch_size as the source of truth.
            if is_stage2_ab and user_gas_raw is not None:
                try:
                    user_gas = int(user_gas_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "training.gradient_accumulation_steps must be an integer when provided"
                    ) from exc
                if user_gas <= 0:
                    raise ValueError(
                        f"training.gradient_accumulation_steps must be > 0, got {user_gas_raw!r}"
                    )
                if int(user_gas) != int(gradient_accumulation_steps):
                    raise ValueError(
                        "For stage2_ab_training, training.gradient_accumulation_steps is derived from "
                        "training.effective_batch_size and must not conflict. "
                        f"Got gradient_accumulation_steps={user_gas} but expected {gradient_accumulation_steps} "
                        f"(effective_batch_size={effective_batch_size}, per_device_train_batch_size={per_device_train_batch_size}, world_size={world_size})."
                    )

            training_section["gradient_accumulation_steps"] = gradient_accumulation_steps

            logger.info(
                f"Auto-calculated gradient_accumulation_steps={gradient_accumulation_steps} "
                f"from effective_batch_size={effective_batch_size}, "
                f"per_device_train_batch_size={per_device_train_batch_size}, "
                f"world_size={world_size}"
            )

        if config.global_max_length is not None:
            model_section.setdefault("max_model_len", config.global_max_length)
            template_section.setdefault("max_length", config.global_max_length)

        if "system" not in template_section and config.prompts.system:
            template_section["system"] = config.prompts.system

        teacher_model_path = rlhf_section_original.get("teacher_model")
        rlhf_type = rlhf_section_original.get("rlhf_type")
        llm_kd_active = rlhf_type == "gkd" and llm_kd_weight > 0
        kd_requested = llm_kd_active or config.custom.visual_kd.enabled
        if kd_requested and not teacher_model_path:
            raise ValueError(
                "rlhf.teacher_model must be provided when llm KD or visual KD is enabled. "
                "Set rlhf.llm_kd_weight to 0 and disable custom.visual_kd to run without a teacher."
            )

        # Compose run-scoped output root before ms-swift appends auto-version.
        run_name_raw = training_section.get("run_name")
        output_dir_raw = training_section.get("output_dir")
        if run_name_raw and output_dir_raw:
            run_name = str(run_name_raw)
            output_dir = str(output_dir_raw)
            output_dir_path = Path(output_dir)
            if output_dir_path.name != run_name:
                training_section["output_dir"] = str(output_dir_path / run_name)

        args_dict: Dict[str, Any] = {}
        for section in (
            model_section,
            quant_section,
            data_section,
            template_section,
            tuner_section,
            training_section,
            rlhf_section,
        ):
            if section:
                args_dict.update(section)

        if config.deepspeed and config.deepspeed.enabled:
            args_dict["deepspeed"] = config.deepspeed.config

        save_delay_config = SaveDelayConfig.from_raw(
            raw_save_delay_steps, raw_save_delay_epochs
        )

        args_cls = RLHFArguments if args_dict.get("rlhf_type") else TrainArguments
        train_args = args_cls(**args_dict)

        try:
            setattr(train_args, "save_last_epoch", save_last_epoch)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "Unable to attach save_last_epoch to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        if config.custom.trainer_variant:
            try:
                setattr(train_args, "trainer_variant", config.custom.trainer_variant)
            except Exception as exc:  # pragma: no cover - explicit failure
                raise RuntimeError(
                    "Unable to attach trainer_variant to TrainArguments; update ms-swift if interface changed."
                ) from exc

        setattr(train_args, "save_delay_config", save_delay_config)
        if save_delay_config.steps is not None:
            setattr(train_args, "save_delay_steps", save_delay_config.steps)
        if save_delay_config.epochs is not None:
            setattr(train_args, "save_delay_epochs", save_delay_config.epochs)

        try:
            setattr(train_args, "visual_kd_config", config.custom.visual_kd)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach visual_kd_config to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        try:
            setattr(train_args, "llm_kd_weight", llm_kd_weight)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach llm_kd_weight to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        try:
            setattr(train_args, "coord_offset_config", config.custom.coord_offset)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach coord_offset_config to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        inner_args = getattr(train_args, "training_args", None)
        if inner_args is None:
            raise RuntimeError(
                "TrainArguments missing nested training_args; ms-swift interface may have changed."
            )

        try:
            setattr(inner_args, "save_last_epoch", save_last_epoch)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach save_last_epoch to inner training arguments; ensure ms-swift exposes this attribute."
            ) from exc

        try:
            setattr(inner_args, "visual_kd_config", config.custom.visual_kd)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach visual_kd_config to inner training arguments; ensure ms-swift exposes this attribute."
            ) from exc

        try:
            setattr(inner_args, "llm_kd_weight", llm_kd_weight)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach llm_kd_weight to inner training arguments; ensure ms-swift exposes this attribute."
            ) from exc

        try:
            setattr(inner_args, "coord_offset_config", config.custom.coord_offset)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach coord_offset_config to inner training arguments; ensure ms-swift exposes this attribute."
            ) from exc

        return train_args

    @staticmethod
    def _materialize_training_config(
        raw_config: Dict[str, Any], prompts: PromptOverrides
    ) -> TrainingConfig:
        try:
            return TrainingConfig.from_mapping(raw_config, prompts)
        except TypeError as exc:
            raise ValueError(
                "configuration must define a 'custom' mapping with dataset parameters"
            ) from exc

    @staticmethod
    def load_training_config(
        config_path: str, base_config_path: Optional[str] = None
    ) -> tuple[TrainArguments, TrainingConfig]:
        ConfigLoader._validate_stage2_leaf_contract(config_path)
        config = ConfigLoader.load_yaml_with_extends(config_path)

        if base_config_path:
            base_config = ConfigLoader.load_yaml_with_extends(base_config_path)
            config = ConfigLoader.merge_configs(base_config, config)

        prompts = ConfigLoader.resolve_prompts(config)
        materialized = ConfigLoader._materialize_training_config(config, prompts)
        train_args = ConfigLoader.build_train_arguments(materialized)

        return train_args, materialized


__all__ = ["ConfigLoader"]
