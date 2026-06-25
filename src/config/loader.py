"""Pure YAML config loader - directly instantiates ms-swift objects"""

import copy
import logging
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

import yaml
try:
    from swift.llm.argument import RLHFArguments, TrainArguments
except ImportError:
    from swift.arguments import RLHFArguments, SftArguments as TrainArguments
from swift.utils import get_dist_setting

from src.common.object_field_order import (
    normalize_object_field_order,
    normalize_object_ordering,
)
from src.common.model_paths import normalize_coordexp_base_model_path
from src.common.geometry.bbox_parameterization import normalize_bbox_format
from src.common.detection_sequence import (
    COMPACT_FULL_FORMAT,
    COORDJSON_FORMAT,
    normalize_detection_sequence_format,
)
from src.training.pipeline_registry import TrainingPipelineRegistry

from .prompts import (
    SYSTEM_PROMPT_SUMMARY,
    USER_PROMPT_SUMMARY,
    coord_mode_from_coord_tokens_enabled,
    get_template_prompts,
)
from .schema import (
    DetectionTrainingConfig,
    PromptOverrides,
    SaveDelayConfig,
    TrainingConfig,
)

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
        stage2_root = (repo_root / "configs" / "stage2" / "rollout_correction").resolve()
        for kind in ("prod", "smoke", "ablation"):
            kind_root = (stage2_root / kind).resolve()
            try:
                config_abs.relative_to(kind_root)
            except ValueError:
                continue
            if config_abs.suffix.lower() == ".yaml":
                return kind
        return None

    @staticmethod
    def _list_valued_key_paths(
        payload: Dict[str, Any], *, prefix: str = ""
    ) -> Set[str]:
        key_paths: Set[str] = set()
        for key, value in payload.items():
            if key in {"extends", "inherit"}:
                continue
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, list):
                key_paths.add(key_path)
            elif isinstance(value, dict):
                key_paths.update(
                    ConfigLoader._list_valued_key_paths(value, prefix=key_path)
                )
        return key_paths

    @staticmethod
    def _raise_list_ownership_overlap(
        *,
        config_path: str,
        key_path: str,
        owner_a: str,
        owner_b: str,
    ) -> None:
        raise ValueError(
            "Config inheritance overlaps list-valued key ownership across reusable "
            f"parents in {config_path}: {key_path} is authored by both "
            f"{owner_a} and {owner_b}. Keep one owner for each list-valued bundle."
        )

    @staticmethod
    def _collect_base_tree_list_owners(
        config_path: str, _visited: Optional[Set[str]] = None
    ) -> Dict[str, str]:
        abs_path = str(Path(config_path).resolve())
        visited: Set[str] = set(_visited or set())
        if abs_path in visited:
            raise ValueError(f"Cyclic config inheritance detected at: {abs_path}")
        visited.add(abs_path)

        raw_cfg = ConfigLoader.load_yaml(abs_path) or {}
        if not isinstance(raw_cfg, dict):
            return {}

        current_dir = Path(abs_path).parent
        extends_value = raw_cfg.get("extends", raw_cfg.get("inherit"))
        base_paths = ConfigLoader._normalize_to_list(extends_value)

        owners: Dict[str, str] = {}
        for base_ref in base_paths:
            base_path = Path(base_ref)
            if not base_path.is_absolute():
                base_path = (current_dir / base_path).resolve()
            base_owners = ConfigLoader._collect_base_tree_list_owners(
                str(base_path), visited
            )
            for key_path, owner in base_owners.items():
                existing_owner = owners.get(key_path)
                if existing_owner is not None and existing_owner != owner:
                    ConfigLoader._raise_list_ownership_overlap(
                        config_path=abs_path,
                        key_path=key_path,
                        owner_a=existing_owner,
                        owner_b=owner,
                    )
                owners[key_path] = owner

        current_cfg = dict(raw_cfg)
        current_cfg.pop("extends", None)
        current_cfg.pop("inherit", None)
        for key_path in sorted(ConfigLoader._list_valued_key_paths(current_cfg)):
            owners[key_path] = abs_path

        return owners

    @staticmethod
    def _lookup_nested_key(payload: Dict[str, Any], key_path: str) -> bool:
        node: Any = payload
        for segment in key_path.split("."):
            if not isinstance(node, dict) or segment not in node:
                return False
            node = node[segment]
        return True

    @staticmethod
    def _require_authored_training_path_string(value: Any, key: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{key} must be authored as a string")
        value_clean = value.strip()
        if not value_clean:
            raise ValueError(f"{key} must be a non-empty string")
        return value_clean

    @staticmethod
    def _join_authored_config_path(root: str, suffix: str) -> str:
        return f"{root.rstrip('/')}/{suffix.strip('/')}"

    @staticmethod
    def _materialize_training_artifact_paths(config: Dict[str, Any]) -> Dict[str, Any]:
        training_section = config.get("training")
        if training_section is None:
            return config
        if not isinstance(training_section, dict):
            raise TypeError("training section must be a mapping when materializing paths")

        artifact_subdir = training_section.get("artifact_subdir")
        if artifact_subdir in (None, "", False):
            return config

        output_root = training_section.get("output_root")
        logging_root = training_section.get("logging_root")
        if output_root in (None, "", False):
            raise ValueError(
                "training.output_root must be provided when training.artifact_subdir is set"
            )
        if logging_root in (None, "", False):
            raise ValueError(
                "training.logging_root must be provided when training.artifact_subdir is set"
            )
        if training_section.get("output_dir") not in (None, "", False):
            raise ValueError(
                "training.output_dir must not be authored together with training.artifact_subdir"
            )
        if training_section.get("logging_dir") not in (None, "", False):
            raise ValueError(
                "training.logging_dir must not be authored together with training.artifact_subdir"
            )

        output_root_clean = ConfigLoader._require_authored_training_path_string(
            output_root, "training.output_root"
        )
        logging_root_clean = ConfigLoader._require_authored_training_path_string(
            logging_root, "training.logging_root"
        )
        artifact_subdir_clean = ConfigLoader._require_authored_training_path_string(
            artifact_subdir, "training.artifact_subdir"
        )

        training_section["output_dir"] = ConfigLoader._join_authored_config_path(
            output_root_clean, artifact_subdir_clean
        )
        training_section["logging_dir"] = ConfigLoader._join_authored_config_path(
            logging_root_clean, artifact_subdir_clean
        )
        return config

    @staticmethod
    def _validate_stage2_leaf_contract(config_path: str) -> None:
        if ConfigLoader._canonical_stage2_profile_kind(config_path) is None:
            return

        raw_cfg = ConfigLoader.load_yaml(config_path) or {}
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Stage-2 profile must be a mapping: {config_path}")

        extends_value = raw_cfg.get("extends", raw_cfg.get("inherit"))
        extends_list = ConfigLoader._normalize_to_list(extends_value)
        if not extends_list:
            raise ValueError(
                "Stage-2 canonical prod/smoke/ablation profiles must declare extends/inherit so the "
                f"resolved contract is explicit. Missing in {config_path}."
            )

        resolved_cfg = ConfigLoader.load_yaml_with_extends(config_path)
        resolved_cfg = ConfigLoader._materialize_training_artifact_paths(resolved_cfg)

        required_resolved_keys = [
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
            "stage2_rollout_correction.pipeline.objective",
        ]

        missing = [
            key
            for key in required_resolved_keys
            if not ConfigLoader._lookup_nested_key(resolved_cfg, key)
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                "Stage-2 canonical prod/smoke/ablation profiles must resolve the required training keys. "
                f"Missing after inheritance in {config_path}: {missing_str}"
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
        base_list_owners: Dict[str, str] = {}
        for base_ref in base_paths:
            base_path = Path(base_ref)
            if not base_path.is_absolute():
                base_path = (current_dir / base_path).resolve()
            base_tree_owners = ConfigLoader._collect_base_tree_list_owners(
                str(base_path)
            )
            for key_path, owner in base_tree_owners.items():
                existing_owner = base_list_owners.get(key_path)
                if existing_owner is not None and existing_owner != owner:
                    ConfigLoader._raise_list_ownership_overlap(
                        config_path=abs_path,
                        key_path=key_path,
                        owner_a=existing_owner,
                        owner_b=owner,
                    )
                base_list_owners[key_path] = owner
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
        ordering_hint: str = "sorted"
        object_field_order: str | None = None
        bbox_format: str = "xyxy"
        prompt_variant: Optional[str] = None
        detection_sequence_format = COORDJSON_FORMAT
        detection_template_id: Optional[str] = None
        coord_tokens_enabled = True

        sample_factory = config.get("sample_factory")
        if sample_factory is not None:
            if not isinstance(sample_factory, dict):
                raise TypeError("sample_factory section must be a mapping")
            target_sequence = sample_factory.get("target_sequence")
            if not isinstance(target_sequence, dict):
                raise TypeError(
                    "sample_factory.target_sequence must be a mapping"
                )
            ordering_raw = target_sequence.get("object_ordering", "sorted")
            ordering_hint = (
                "random"
                if ordering_raw == "random_permutation"
                else normalize_object_ordering(
                    ordering_raw,
                    path="sample_factory.target_sequence.object_ordering",
                )
            )
            object_field_order = normalize_object_field_order(
                target_sequence.get("object_field_order", "desc_first"),
                path="sample_factory.target_sequence.object_field_order",
            )
            bbox_format = normalize_bbox_format(
                target_sequence.get("bbox_format", "xyxy"),
                path="sample_factory.target_sequence.bbox_format",
            )
            detection_template = config.get("detection_template") or {}
            if not isinstance(detection_template, dict):
                raise TypeError("detection_template section must be a mapping")
            detection_template_id_raw = detection_template.get("id")
            detection_template_id = (
                None
                if detection_template_id_raw is None
                else str(detection_template_id_raw)
            )
            if detection_template_id is not None:
                from src.detection.template_contracts import (
                    resolve_detection_template_contract,
                )

                contract = resolve_detection_template_contract(detection_template_id)
                detection_sequence_format = (
                    COMPACT_FULL_FORMAT if contract.is_compact else COORDJSON_FORMAT
                )
            prompt = config.get("prompt") or {}
            if not isinstance(prompt, dict):
                raise TypeError("prompt section must be a mapping")
            if "prompt_variant_enabled" in prompt:
                raise ValueError(
                    "prompt.prompt_variant_enabled is retired; use prompt.variant"
                )
            prompt_variant_raw = prompt.get("variant")
            if prompt_variant_raw is not None and not isinstance(prompt_variant_raw, str):
                raise TypeError("prompt.variant must be a string when provided")
            prompt_variant = prompt_variant_raw

        custom_section = config.get("custom")
        if custom_section is not None:
            if not isinstance(custom_section, dict):
                raise TypeError(
                    "custom section must be a mapping when resolving prompts"
                )
            if sample_factory is not None:
                extra_cfg = custom_section.get("extra")
                if isinstance(extra_cfg, dict) and "prompt_variant" in extra_cfg:
                    raise ValueError(
                        "custom.extra.prompt_variant is retired; use prompt.variant"
                    )
                guidance = (
                    ("detection_template_id", "detection_template.id"),
                    ("detection_sequence_format", "sample_factory.id"),
                    ("object_ordering", "sample_factory.target_sequence.object_ordering"),
                    (
                        "object_field_order",
                        "sample_factory.target_sequence.object_field_order",
                    ),
                )
                for key, new_path in guidance:
                    if key in custom_section:
                        raise ValueError(f"custom.{key} is retired; use {new_path}")
                raise ValueError(
                    "custom is obsolete for target-hierarchy detection prompt "
                    "resolution; use prompt.variant, "
                    "sample_factory.target_sequence, and detection_template.id"
                )
            if "summary_ratio" in custom_section:
                raise ValueError(
                    "custom.summary_ratio has been removed; use custom.use_summary instead."
                )
            if "use_summary" in custom_section:
                use_summary = ConfigLoader._coerce_bool(
                    custom_section["use_summary"], "custom.use_summary"
                )

            ordering_hint_raw = custom_section.get("object_ordering")
            if ordering_hint_raw is not None:
                ordering_hint = normalize_object_ordering(
                    ordering_hint_raw,
                    path="custom.object_ordering",
                )

            object_field_order_raw = custom_section.get("object_field_order", None)
            if object_field_order_raw is None:
                raise ValueError("custom.object_field_order must be provided")
            object_field_order = normalize_object_field_order(
                object_field_order_raw, path="custom.object_field_order"
            )
            bbox_format = normalize_bbox_format(
                custom_section.get("bbox_format", "xyxy"),
                path="custom.bbox_format",
            )

            coord_tokens_cfg = custom_section.get("coord_tokens")
            if coord_tokens_cfg is None:
                coord_tokens_cfg = {}
            if not isinstance(coord_tokens_cfg, dict):
                raise TypeError(
                    "custom.coord_tokens must be a mapping when provided"
                )

            coord_tokens_enabled = ConfigLoader._coerce_bool(
                coord_tokens_cfg.get("enabled", True),
                "custom.coord_tokens.enabled",
            )
            detection_sequence_format = normalize_detection_sequence_format(
                custom_section.get("detection_sequence_format", COORDJSON_FORMAT)
            )
            detection_template_id_raw = custom_section.get("detection_template_id")
            detection_template_id = (
                None
                if detection_template_id_raw is None
                else str(detection_template_id_raw)
            )
            if detection_template_id is not None:
                from src.detection.template_contracts import (
                    resolve_detection_template_contract,
                )

                detection_template_contract = resolve_detection_template_contract(
                    detection_template_id
                )
            else:
                detection_template_contract = None
            if (
                detection_sequence_format != COORDJSON_FORMAT
                or (
                    detection_template_contract is not None
                    and detection_template_contract.is_compact
                )
            ) and not coord_tokens_enabled:
                raise ValueError(
                    "compact detection rendering requires custom.coord_tokens.enabled=true"
                )

            skip_bbox_norm = ConfigLoader._coerce_bool(
                coord_tokens_cfg.get("skip_bbox_norm", True),
                "custom.coord_tokens.skip_bbox_norm",
            )
            if not skip_bbox_norm:
                raise ValueError(
                    "Pre-normalized geometry contract: custom.coord_tokens.skip_bbox_norm must be true."
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

        if object_field_order is None:
            raise ValueError("custom.object_field_order must be provided")

        if use_summary:
            default_system = SYSTEM_PROMPT_SUMMARY
            default_user = USER_PROMPT_SUMMARY
            output_variant = "summary"
        else:
            default_system, default_user = get_template_prompts(
                ordering=ordering_hint,
                coord_mode=coord_mode_from_coord_tokens_enabled(
                    coord_tokens_enabled
                ),
                prompt_variant=prompt_variant,
                object_field_order=object_field_order,
                bbox_format=bbox_format,
                detection_sequence_format=detection_sequence_format,
                detection_template_id=detection_template_id,
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
    def _runtime_trainer_variant_for_config(
        config: TrainingConfig | DetectionTrainingConfig,
    ) -> str | None:
        if isinstance(config, DetectionTrainingConfig):
            pipeline_id = getattr(getattr(config, "pipeline", None), "id", None)
            if pipeline_id == "stage2_rollout_correction":
                return "stage2_rollout_correction"
            return None
        return str(getattr(config.custom, "trainer_variant", "") or "") or None

    @staticmethod
    def build_train_arguments(config: TrainingConfig | DetectionTrainingConfig) -> TrainArguments:
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
        is_detection = isinstance(config, DetectionTrainingConfig)
        runtime_trainer_variant = ConfigLoader._runtime_trainer_variant_for_config(config)
        model_section = dict(config.model)
        model_path = model_section.get("model")
        if model_path is not None:
            model_section["model"] = normalize_coordexp_base_model_path(str(model_path))
        quant_section = dict(config.quantization)
        data_section = (
            {"dataset": ["dummy"], "val_dataset": ["dummy"]}
            if is_detection
            else dict(config.data)
        )
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
            "packing_mode",
            "packing_length",
            "packing_buffer",
            "packing_min_fill_ratio",
            "packing_drop_last",
            "packing_allow_single_long",
            "eval_packing",
            "packing_avg_samples",
            "packing_wait_timeout_s",
            "packing_length_cache_persist_every",
            "packing_length_precompute_workers",
            "encoded_sample_cache",
            "output_root",
            "logging_root",
            "artifact_subdir",
            "static_packing_cache",
            "save_model_only",
            "save_only_model",
            "checkpoint_mode",
        }
        for key in _packing_keys:
            training_section.pop(key, None)

        # Auto-calculate gradient_accumulation_steps from effective_batch_size
        #
        # Stage-2 rollout correction standardizes step semantics around a true
        # (exact) global effective batch.
        is_stage2_rollout_correction = bool(
            runtime_trainer_variant == "stage2_rollout_correction"
        )

        effective_batch_size = training_section.pop("effective_batch_size", None)
        if is_stage2_rollout_correction and effective_batch_size is None:
            raise ValueError(
                "stage2_rollout_correction requires training.effective_batch_size "
                "to be set (global raw rollouts per optimizer step)."
            )

        if effective_batch_size is not None:
            user_gas_raw = training_section.get("gradient_accumulation_steps", None)
            if user_gas_raw is not None:
                raise ValueError(
                    "training.gradient_accumulation_steps is derived from "
                    "training.effective_batch_size and must not be authored when "
                    "effective_batch_size is set"
                )
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

            if effective_batch_size % denominator != 0:
                raise ValueError(
                    "training.effective_batch_size must be divisible by "
                    f"training.per_device_train_batch_size*world_size ({per_device_train_batch_size}*{world_size}={denominator}). "
                    f"Got effective_batch_size={effective_batch_size}. This keeps derived "
                    "gradient_accumulation_steps exact; change the launch topology or "
                    "the authored effective batch."
                )

            gradient_accumulation_steps = max(1, int(effective_batch_size // denominator))

            training_section["gradient_accumulation_steps"] = gradient_accumulation_steps

            logger.info(
                f"Auto-calculated gradient_accumulation_steps={gradient_accumulation_steps} "
                f"from effective_batch_size={effective_batch_size}, "
                f"per_device_train_batch_size={per_device_train_batch_size}, "
                f"world_size={world_size}, "
                f"actual_global_effective_batch_size={denominator * gradient_accumulation_steps}"
            )

        if config.global_max_length is not None:
            model_section.setdefault("max_model_len", config.global_max_length)
            template_section.setdefault("max_length", config.global_max_length)

        if (
            not is_detection
            and "system" not in template_section
            and config.prompts.system
        ):
            template_section["system"] = config.prompts.system

        teacher_model_path = rlhf_section_original.get("teacher_model")
        rlhf_type = rlhf_section_original.get("rlhf_type")
        llm_kd_active = rlhf_type == "gkd" and llm_kd_weight > 0
        visual_kd_enabled = (
            False if is_detection else bool(config.custom.visual_kd.enabled)
        )
        kd_requested = llm_kd_active or visual_kd_enabled
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

        if is_detection:
            deepspeed_section = config.deepspeed
            if deepspeed_section and bool(deepspeed_section.get("enabled", False)):
                args_dict["deepspeed"] = deepspeed_section.get("config")
        elif config.deepspeed and config.deepspeed.enabled:
            args_dict["deepspeed"] = config.deepspeed.config

        save_delay_config = SaveDelayConfig.from_raw(
            raw_save_delay_steps, raw_save_delay_epochs
        )

        args_cls = RLHFArguments if args_dict.get("rlhf_type") else TrainArguments
        allowed_arg_names = {f.name for f in fields(args_cls)}
        if (
            "train_type" in args_dict
            and "train_type" not in allowed_arg_names
            and "tuner_type" in allowed_arg_names
        ):
            args_dict["tuner_type"] = args_dict.pop("train_type")
        train_args = args_cls(**args_dict)
        if not hasattr(train_args, "train_type") and hasattr(train_args, "tuner_type"):
            setattr(train_args, "train_type", getattr(train_args, "tuner_type"))

        try:
            setattr(train_args, "save_last_epoch", save_last_epoch)
        except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "Unable to attach save_last_epoch to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        if runtime_trainer_variant:
            try:
                setattr(train_args, "trainer_variant", runtime_trainer_variant)
            except (AttributeError, TypeError) as exc:  # pragma: no cover - explicit failure
                raise RuntimeError(
                    "Unable to attach trainer_variant to TrainArguments; update ms-swift if interface changed."
                ) from exc

        setattr(train_args, "save_delay_config", save_delay_config)
        if save_delay_config.steps is not None:
            setattr(train_args, "save_delay_steps", save_delay_config.steps)
        if save_delay_config.epochs is not None:
            setattr(train_args, "save_delay_epochs", save_delay_config.epochs)

        if not is_detection:
            try:
                setattr(train_args, "visual_kd_config", config.custom.visual_kd)
            except (AttributeError, TypeError) as exc:  # pragma: no cover
                raise RuntimeError(
                    "Unable to attach visual_kd_config to TrainArguments; ensure ms-swift exposes this attribute."
                ) from exc

        try:
            setattr(train_args, "llm_kd_weight", llm_kd_weight)
        except (AttributeError, TypeError) as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach llm_kd_weight to TrainArguments; ensure ms-swift exposes this attribute."
            ) from exc

        if is_detection:
            setattr(train_args, "detection_config", config)
        else:
            try:
                setattr(
                    train_args,
                    "token_embeddings_adapter_config",
                    config.custom.token_embeddings_adapter,
                )
            except (AttributeError, TypeError) as exc:  # pragma: no cover
                raise RuntimeError(
                    "Unable to attach token_embeddings_adapter_config to TrainArguments; ensure ms-swift exposes this attribute."
                ) from exc

        inner_args = getattr(train_args, "training_args", None)
        if inner_args is None:
            raise RuntimeError(
                "TrainArguments missing nested training_args; ms-swift interface may have changed."
            )

        if not is_detection:
            try:
                setattr(inner_args, "visual_kd_config", config.custom.visual_kd)
            except (AttributeError, TypeError) as exc:  # pragma: no cover
                raise RuntimeError(
                    "Unable to attach visual_kd_config to inner training arguments; ensure ms-swift exposes this attribute."
                ) from exc

        try:
            setattr(inner_args, "llm_kd_weight", llm_kd_weight)
        except (AttributeError, TypeError) as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to attach llm_kd_weight to inner training arguments; ensure ms-swift exposes this attribute."
            ) from exc

        if is_detection:
            setattr(inner_args, "detection_config", config)
        else:
            try:
                setattr(
                    inner_args,
                    "token_embeddings_adapter_config",
                    config.custom.token_embeddings_adapter,
                )
            except (AttributeError, TypeError) as exc:  # pragma: no cover
                raise RuntimeError(
                    "Unable to attach token_embeddings_adapter_config to inner training arguments; ensure ms-swift exposes this attribute."
                ) from exc

        return train_args

    @staticmethod
    def _materialize_training_config(
        raw_config: Dict[str, Any], prompts: PromptOverrides
    ) -> TrainingConfig | DetectionTrainingConfig:
        if ConfigLoader._is_detection_training_config_payload(raw_config):
            detection_config = ConfigLoader._sanitize_detection_training_config_payload(
                raw_config
            )
            TrainingPipelineRegistry().resolve(detection_config)
            return DetectionTrainingConfig.from_mapping(detection_config)
        try:
            return TrainingConfig.from_mapping(raw_config, prompts)
        except TypeError as exc:
            raise ValueError(
                "configuration must define a 'custom' mapping with dataset parameters"
            ) from exc

    @staticmethod
    def _sanitize_detection_training_config_payload(
        raw_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        config = copy.deepcopy(dict(raw_config))
        custom = config.pop("custom", None)
        if custom is not None:
            if not isinstance(custom, Mapping):
                raise TypeError("custom must be a mapping when provided")
            allowed_universal_residue = {
                "json_format",
                "emit_norm",
                "dump_conversation_path",
            }
            retired_guidance = {
                "trainer_variant": "pipeline.id",
                "object_ordering": "sample_factory.target_sequence.object_ordering",
                "object_field_order": "sample_factory.target_sequence.object_field_order",
                "detection_template_id": "detection_template.id",
                "detection_sequence_format": "sample_factory.id",
                "token_embeddings_adapter": "token_embeddings_adapter",
            }
            for key, new_path in retired_guidance.items():
                if key in custom:
                    raise ValueError(f"custom.{key} is retired; use {new_path}")
            unknown_custom = sorted(set(custom) - allowed_universal_residue)
            if unknown_custom:
                rendered = [f"custom.{key}" for key in unknown_custom]
                raise ValueError(
                    "custom is obsolete for target-hierarchy detection configs; "
                    f"unexpected keys: {rendered}"
                )
        data = config.get("data")
        if isinstance(data, dict):
            allowed_data_keys = {
                "train_jsonl",
                "val_jsonl",
                "image_root",
                "object_ordering",
            }
            config["data"] = {
                key: value for key, value in data.items() if key in allowed_data_keys
            }
        return config

    @staticmethod
    def _is_detection_training_config_payload(raw_config: Mapping[str, Any]) -> bool:
        keys = set(raw_config.keys())
        target_hierarchy_sentinels = {
            "sample_factory",
            "detection_template",
            "token_embeddings_adapter",
        }
        if "pipeline" in keys and (
            target_hierarchy_sentinels.intersection(keys)
            or "stage2_rollout_correction" in keys
            or "rollout_matching" in keys
        ):
            return True
        if target_hierarchy_sentinels.intersection(keys):
            return True

        target_hierarchy_markers = {
            "data",
            "pipeline",
            "sample_factory",
            "prompt",
            "detection_template",
            "token_embeddings_adapter",
            "packing",
            "evaluation",
            "validation",
        }
        legacy_detection_markers = {
            "data",
            "prompt",
            "detection_template",
            "objective",
            "packing",
            "evaluation",
            "validation",
        }
        return target_hierarchy_markers.issubset(
            keys
        ) or legacy_detection_markers.issubset(keys)

    @staticmethod
    def load_materialized_training_config(
        config_path: str, base_config_path: Optional[str] = None
    ) -> TrainingConfig | DetectionTrainingConfig:
        """Load + materialize a TrainingConfig without constructing ms-swift TrainArguments.

        This is intentionally side-effect free (no hub downloads / model probing) and is
        safe to run from arbitrary working directories.
        """

        ConfigLoader._validate_stage2_leaf_contract(config_path)
        config = ConfigLoader.load_yaml_with_extends(config_path)

        if base_config_path:
            base_config = ConfigLoader.load_yaml_with_extends(base_config_path)
            config = ConfigLoader.merge_configs(base_config, config)

        config = ConfigLoader._materialize_training_artifact_paths(config)
        prompts = (
            PromptOverrides()
            if ConfigLoader._is_detection_training_config_payload(config)
            else ConfigLoader.resolve_prompts(config)
        )
        return ConfigLoader._materialize_training_config(config, prompts)

    @staticmethod
    def load_training_config(
        config_path: str, base_config_path: Optional[str] = None
    ) -> tuple[TrainArguments, TrainingConfig | DetectionTrainingConfig]:
        ConfigLoader._validate_stage2_leaf_contract(config_path)
        config = ConfigLoader.load_yaml_with_extends(config_path)

        if base_config_path:
            base_config = ConfigLoader.load_yaml_with_extends(base_config_path)
            config = ConfigLoader.merge_configs(base_config, config)

        config = ConfigLoader._materialize_training_artifact_paths(config)
        prompts = (
            PromptOverrides()
            if ConfigLoader._is_detection_training_config_payload(config)
            else ConfigLoader.resolve_prompts(config)
        )
        materialized = ConfigLoader._materialize_training_config(config, prompts)
        train_args = ConfigLoader.build_train_arguments(materialized)

        return train_args, materialized


__all__ = ["ConfigLoader"]
