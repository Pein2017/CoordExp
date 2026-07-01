from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, Mapping

from src.bootstrap.pipeline_manifest import build_pipeline_manifest
from src.bootstrap.stage2_policy_provenance import build_stage2_policy_provenance
from src.config.prompts import resolve_dense_prompt_variant_key
from src.config.strict_dataclass import dataclass_asdict_no_none


CUSTOM_EXTRA_PROMPT_COMPAT_SOURCE = "custom.extra.prompt_variant_compat_fallback"
STAGE2_TRAINER_VARIANT = "stage2_rollout_correction"


@dataclass(frozen=True, slots=True)
class Stage2RuntimeProjection:
    """Resolved Stage-2 runtime state passed from launcher to trainer setup."""

    rollout_matching_cfg: dict[str, Any]
    rollout_pipeline_manifest: dict[str, Any]
    stage2_rollout_correction_cfg: dict[str, Any]
    stage2_pipeline_manifest: dict[str, Any]
    stage2_policy_provenance: dict[str, Any]
    policy_sources: dict[str, str]
    compatibility_fallbacks: tuple[str, ...]


def resolve_stage2_runtime_projection(
    *,
    training_config: Any,
    custom_config: Any,
    packing_cfg: Any,
    trainer_variant: str | None,
    config_path: str,
    run_name: str,
    seed: int,
    coord_soft_cfg: Mapping[str, Any] | None = None,
) -> Stage2RuntimeProjection | None:
    """Resolve Stage-2 authored config into one provenance-bearing payload."""

    if str(trainer_variant or "") != STAGE2_TRAINER_VARIANT:
        return None

    rollout_cfg = _resolve_rollout_matching_cfg(training_config)
    stage2_cfg = _resolve_stage2_correction_cfg(training_config)

    policy_sources: dict[str, str] = {}
    compatibility_fallbacks: list[str] = []
    _validate_removed_rollout_matching_surfaces(rollout_cfg)
    rollout_cfg["decoding"] = _resolve_decoding_cfg(rollout_cfg)
    _resolve_prompt_variants(
        rollout_cfg,
        training_config=training_config,
        custom_config=custom_config,
        policy_sources=policy_sources,
        compatibility_fallbacks=compatibility_fallbacks,
    )
    _inject_packing_and_geometry(
        rollout_cfg,
        training_config=training_config,
        custom_config=custom_config,
        packing_cfg=packing_cfg,
        policy_sources=policy_sources,
    )
    rollout_cfg["_policy_sources"] = dict(policy_sources)
    rollout_cfg["_compatibility_fallbacks"] = list(compatibility_fallbacks)

    rollout_manifest = _empty_rollout_manifest(
        config_path=config_path,
        run_name=run_name,
        seed=seed,
    )
    stage2_manifest = build_pipeline_manifest(
        stage2_cfg,
        default_objective=["residual_set_correction"],
        default_diagnostics=[],
        trainer_variant=str(trainer_variant or ""),
        config_path=config_path,
        run_name=run_name,
        seed=seed,
        coord_soft_cfg=coord_soft_cfg,
    )
    provenance_payload = _training_config_payload(
        training_config=training_config,
        custom_config=custom_config,
        rollout_matching_cfg=rollout_cfg,
        stage2_cfg=stage2_cfg,
    )
    stage2_policy_provenance = build_stage2_policy_provenance(
        provenance_payload,
        trainer_variant=str(trainer_variant or ""),
    )
    if stage2_policy_provenance is None:
        raise ValueError(
            "stage2_rollout_correction runtime projection could not build "
            "policy provenance"
        )
    stage2_policy_provenance["runtime_policy_sources"] = dict(policy_sources)
    stage2_policy_provenance["runtime_compatibility_fallbacks"] = list(
        compatibility_fallbacks
    )

    return Stage2RuntimeProjection(
        rollout_matching_cfg=rollout_cfg,
        rollout_pipeline_manifest=rollout_manifest,
        stage2_rollout_correction_cfg=stage2_cfg,
        stage2_pipeline_manifest=stage2_manifest,
        stage2_policy_provenance=stage2_policy_provenance,
        policy_sources=policy_sources,
        compatibility_fallbacks=tuple(compatibility_fallbacks),
    )


def apply_stage2_runtime_projection(
    trainer: Any,
    projection: Stage2RuntimeProjection | None,
) -> None:
    """Attach the resolved Stage-2 projection to the trainer boundary."""

    if projection is None:
        return
    setattr(trainer, "rollout_matching_cfg", projection.rollout_matching_cfg)
    setattr(
        trainer,
        "object_field_order",
        projection.rollout_matching_cfg["object_field_order"],
    )
    validate_hook = getattr(trainer, "_validate_rollout_matching_cfg", None)
    if callable(validate_hook):
        validate_hook()
    setattr(trainer, "rollout_pipeline_manifest", projection.rollout_pipeline_manifest)
    setattr(
        trainer,
        "stage2_rollout_correction_cfg",
        projection.stage2_rollout_correction_cfg,
    )
    setattr(trainer, "stage2_pipeline_manifest", projection.stage2_pipeline_manifest)
    setattr(trainer, "stage2_policy_provenance", projection.stage2_policy_provenance)


def _resolve_rollout_matching_cfg(training_config: Any) -> dict[str, Any]:
    raw = _read_value(training_config, "rollout_matching")
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping) and not is_dataclass(raw):
        raise TypeError("rollout_matching must be a mapping when provided")
    return _to_mapping(raw)


def _resolve_stage2_correction_cfg(training_config: Any) -> dict[str, Any]:
    raw = _read_value(training_config, "stage2_rollout_correction")
    if raw is None:
        raise ValueError(
            "training_config.stage2_rollout_correction is required for "
            "stage2_rollout_correction; check config parsing."
        )
    if not isinstance(raw, Mapping) and not is_dataclass(raw):
        raise TypeError("stage2_rollout_correction must be a mapping when provided")
    return _to_mapping(raw)


def _validate_removed_rollout_matching_surfaces(rollout_cfg: Mapping[str, Any]) -> None:
    if isinstance(rollout_cfg.get("pipeline"), Mapping):
        raise ValueError(
            "rollout_matching.pipeline has been removed. "
            "Use stage2_rollout_correction.pipeline with "
            "pipeline.id=stage2_rollout_correction instead."
        )
    legacy_decoding_keys = [
        k for k in ("temperature", "top_p", "top_k") if k in rollout_cfg
    ]
    if legacy_decoding_keys:
        keys_s = ", ".join(f"rollout_matching.{k}" for k in legacy_decoding_keys)
        raise ValueError(
            "Legacy rollout decoding keys have been removed: "
            f"{keys_s}. Use rollout_matching.decoding.* instead. "
            "(No backward compatibility.)"
        )
    if "rollout_buffer" in rollout_cfg:
        raise ValueError(
            "rollout_matching.rollout_buffer has been removed. "
            "Remove this section from your config. (No backward compatibility.)"
        )


def _resolve_decoding_cfg(rollout_cfg: Mapping[str, Any]) -> dict[str, Any]:
    decoding_raw = rollout_cfg.get("decoding", None)
    if decoding_raw is None:
        return {}
    if isinstance(decoding_raw, Mapping):
        return dict(decoding_raw)
    raise TypeError("rollout_matching.decoding must be a mapping when provided")


def _resolve_prompt_variants(
    rollout_cfg: dict[str, Any],
    *,
    training_config: Any,
    custom_config: Any,
    policy_sources: dict[str, str],
    compatibility_fallbacks: list[str],
) -> None:
    prompt_variant_from_hierarchy = _target_hierarchy_prompt_variant(training_config)
    custom_extra = _read_value(custom_config, "extra") or {}
    prompt_variant_from_extra = None
    if isinstance(custom_extra, Mapping):
        raw_prompt_variant = custom_extra.get("prompt_variant")
        if isinstance(raw_prompt_variant, str) and raw_prompt_variant.strip():
            prompt_variant_from_extra = str(raw_prompt_variant).strip()

    for key in ("prompt_variant", "eval_prompt_variant"):
        raw = rollout_cfg.get(key, None)
        if raw is not None:
            if not isinstance(raw, str):
                raise TypeError(
                    f"rollout_matching.{key} must be a string when provided"
                )
            stripped = raw.strip()
            if stripped:
                rollout_cfg[key] = resolve_dense_prompt_variant_key(stripped)
                policy_sources[f"rollout_matching.{key}"] = f"rollout_matching.{key}"
                continue
            rollout_cfg[key] = None

        if prompt_variant_from_hierarchy is not None:
            rollout_cfg[key] = resolve_dense_prompt_variant_key(
                prompt_variant_from_hierarchy
            )
            policy_sources[f"rollout_matching.{key}"] = "prompt.variant"
        elif prompt_variant_from_extra is not None:
            rollout_cfg[key] = resolve_dense_prompt_variant_key(
                prompt_variant_from_extra
            )
            policy_sources[f"rollout_matching.{key}"] = (
                CUSTOM_EXTRA_PROMPT_COMPAT_SOURCE
            )
            if "custom.extra.prompt_variant" not in compatibility_fallbacks:
                compatibility_fallbacks.append("custom.extra.prompt_variant")
        else:
            policy_sources[f"rollout_matching.{key}"] = "unset"
            rollout_cfg[key] = None


def _inject_packing_and_geometry(
    rollout_cfg: dict[str, Any],
    *,
    training_config: Any,
    custom_config: Any,
    packing_cfg: Any,
    policy_sources: dict[str, str],
) -> None:
    target_sequence = _target_sequence_mapping(training_config)
    detection_template = _to_mapping(_read_value(training_config, "detection_template"))
    if target_sequence is not None:
        template_id = str(detection_template.get("id") or "").strip()
        if template_id in {"compact", "compact_full"}:
            detection_sequence_format = "compact_full"
        elif template_id == "stage1_json_pretty":
            detection_sequence_format = "coordjson"
        else:
            detection_sequence_format = template_id or str(
                _read_value(custom_config, "detection_sequence_format")
            )
        sequence_fields = {
            "object_ordering": str(target_sequence.get("object_ordering")),
            "object_field_order": str(target_sequence.get("object_field_order")),
            "bbox_format": str(target_sequence.get("bbox_format")),
            "detection_sequence_format": detection_sequence_format,
        }
    else:
        sequence_fields = {
            "object_ordering": str(_read_value(custom_config, "object_ordering")),
            "object_field_order": str(_read_value(custom_config, "object_field_order")),
            "bbox_format": str(_read_value(custom_config, "bbox_format")),
            "detection_sequence_format": str(
                _read_value(custom_config, "detection_sequence_format")
            ),
        }
    resolved = {
        "packing_enabled": bool(_read_value(packing_cfg, "enabled")),
        "packing_length": int(_read_value(packing_cfg, "packing_length") or 0),
        "packing_buffer": int(_read_value(packing_cfg, "buffer_size") or 0),
        "packing_min_fill_ratio": float(
            _read_value(packing_cfg, "min_fill_ratio") or 0.0
        ),
        "packing_drop_last": bool(_read_value(packing_cfg, "drop_last")),
        **sequence_fields,
    }
    rollout_cfg.update(resolved)
    sources = {
        "packing.enabled": "training.packing",
        "packing.length": "training.packing_length",
        "packing.buffer": "training.packing_buffer",
        "packing.min_fill_ratio": "training.packing_min_fill_ratio",
        "packing.drop_last": "training.packing_drop_last",
    }
    if target_sequence is not None:
        sources.update(
            {
                "object_ordering": "sample_factory.target_sequence.object_ordering",
                "object_field_order": "sample_factory.target_sequence.object_field_order",
                "bbox_format": "sample_factory.target_sequence.bbox_format",
                "detection_sequence_format": "detection_template.id+sample_factory.id",
            }
        )
    else:
        sources.update(
            {
                "object_ordering": "custom.object_ordering",
                "object_field_order": "custom.object_field_order",
                "bbox_format": "custom.bbox_format",
                "detection_sequence_format": "custom.detection_sequence_format",
            }
        )
    policy_sources.update(sources)


def _target_hierarchy_prompt_variant(training_config: Any) -> str | None:
    prompt = _read_value(training_config, "prompt")
    raw_variant = _read_value(prompt, "variant")
    if isinstance(raw_variant, str) and raw_variant.strip():
        return raw_variant.strip()
    return None


def _target_sequence_mapping(training_config: Any) -> dict[str, Any] | None:
    sample_factory = _read_value(training_config, "sample_factory")
    target_sequence = _read_value(sample_factory, "target_sequence")
    if target_sequence is None:
        return None
    mapped = _to_mapping(target_sequence)
    return mapped or None


def _empty_rollout_manifest(
    *,
    config_path: str,
    run_name: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "payload": {
            "objective": [],
            "diagnostics": [],
            "extra": {
                "pipeline_id": "stage2_rollout_correction",
                "runtime_stage": "stage2",
            },
        },
        "objective": [],
        "diagnostics": [],
        "extra": {
            "pipeline_id": "stage2_rollout_correction",
            "runtime_stage": "stage2",
        },
        "checksum": "",
        "run_context": {
            "config": str(config_path),
            "run_name": str(run_name or ""),
            "seed": int(seed or 0),
        },
    }


def _training_config_payload(
    *,
    training_config: Any,
    custom_config: Any,
    rollout_matching_cfg: Mapping[str, Any],
    stage2_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    custom_payload = _to_mapping(custom_config)
    return {
        "pipeline": _to_mapping(_read_value(training_config, "pipeline")),
        "sample_factory": _to_mapping(_read_value(training_config, "sample_factory")),
        "detection_template": _to_mapping(
            _read_value(training_config, "detection_template")
        ),
        "custom": custom_payload,
        "rollout_matching": dict(rollout_matching_cfg),
        "stage2_rollout_correction": dict(stage2_cfg),
        "training": _to_mapping(_read_value(training_config, "training")),
    }


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return dataclass_asdict_no_none(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            str(k): v
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return {}


def _read_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


__all__ = [
    "CUSTOM_EXTRA_PROMPT_COMPAT_SOURCE",
    "STAGE2_TRAINER_VARIANT",
    "Stage2RuntimeProjection",
    "apply_stage2_runtime_projection",
    "resolve_stage2_runtime_projection",
]
