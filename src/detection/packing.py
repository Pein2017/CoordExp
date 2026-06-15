"""Packing contracts for detection training surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast

from src.common.detection_sequence import (
    COMPACT_FULL_FORMAT,
    COORDJSON_FORMAT,
    normalize_detection_sequence_format,
)
from src.detection.objective import DetectionTrainingMode
from src.detection.template import DetectionSequenceTemplate, TemplateId, get_detection_template
from src.detection.tokenization import TokenSpan

PackingRuntimeMode = Literal["disabled", "static", "padding_free_packed"]


def _canonicalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "packing fingerprint fields must not contain NaN or Infinity"
            )
        return value
    if isinstance(value, Mapping):
        canonical_items: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "packing fingerprint mapping keys must be strings; "
                    f"got {type(key).__name__}"
                )
            canonical_items[key] = _canonicalize_json_value(nested_value)
        return MappingProxyType(
            {
                key: canonical_items[key]
                for key in sorted(canonical_items)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonicalize_json_value(item) for item in value)
    raise TypeError(
        "packing fingerprint fields must be JSON-serializable scalars, "
        f"lists, tuples, or mappings; got {type(value).__name__}"
    )


def _json_ready_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {key: _json_ready_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return [_json_ready_value(item) for item in value]
    raise TypeError(
        "packing fingerprint canonical values must be scalars, tuples, or mappings; "
        f"got {type(value).__name__}"
    )


def _canonicalize_json_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    canonical = _canonicalize_json_value(value)
    if not isinstance(canonical, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return canonical


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _require_positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return int(value)


def _require_non_negative_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return int(value)


def resolve_detection_template_id_for_static_packing(
    detection_sequence_format: Any,
) -> TemplateId:
    detection_format = normalize_detection_sequence_format(detection_sequence_format)
    if detection_format == COORDJSON_FORMAT:
        return "stage1_json_pretty"
    if detection_format == COMPACT_FULL_FORMAT:
        return "compact_full"
    raise ValueError(
        "dataset-level static packing only supports detection templates "
        "{'stage1_json_pretty', 'compact_full'}; got "
        f"detection_sequence_format={detection_sequence_format!r}"
    )


def resolve_static_sft_training_mode(
    *,
    objective_variant: str | None = None,
    object_ordering: str = "sorted",
    trainer_variant: str | None = None,
) -> DetectionTrainingMode:
    if objective_variant == "teacher_forcing":
        raise ValueError(
            "teacher_forcing_target_ir must not use legacy recursive static "
            "packing helpers; exact atom-position packing mapping is not "
            "implemented"
        )
    if objective_variant:
        return _validate_detection_training_mode(str(objective_variant))
    if str(object_ordering or "sorted") in {"random", "random_permutation"}:
        return "random_order_sft"
    return "sorted_sft"


def _validate_detection_training_mode(value: str) -> DetectionTrainingMode:
    if value not in {
        "sorted_sft",
        "random_order_sft",
        "prefix_denoising_sft",
        "random_permutation_et_rmp_ce",
        "trie_disabled_full_suffix_ce",
        "prefix_rollin_et_rmp_ce",
    }:
        raise ValueError(f"Unsupported detection training mode: {value!r}")
    return cast(DetectionTrainingMode, value)


def _default_state_weighting_for_mode(training_mode: DetectionTrainingMode) -> str:
    if training_mode == "random_permutation_et_rmp_ce":
        return "legacy_row_mean_prefix_mixture_equivalence"
    return "none"


def _default_normalization_for_mode(training_mode: DetectionTrainingMode) -> str:
    if training_mode == "random_permutation_et_rmp_ce":
        return "legacy_row_mean_equivalence"
    if training_mode == "prefix_denoising_sft":
        return "branch_balanced_clean_noisy_ce"
    return "token_mean"


@dataclass(frozen=True)
class PackingProfile:
    mode: PackingRuntimeMode
    packing_length: int = 0
    runtime_flags: Mapping[str, Any] = field(default_factory=dict)
    experimental: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "static", "padding_free_packed"}:
            raise ValueError(f"Unsupported packing mode: {self.mode!r}")
        packing_length = (
            _require_non_negative_int(
                self.packing_length,
                field_name="packing_length",
            )
            if self.mode == "disabled"
            else _require_positive_int(
                self.packing_length,
                field_name="packing_length",
            )
        )
        object.__setattr__(self, "packing_length", packing_length)
        if self.mode == "disabled":
            if self.experimental:
                raise ValueError("disabled packing cannot be marked experimental")
        else:
            if self.mode == "padding_free_packed":
                object.__setattr__(self, "experimental", True)
            elif self.experimental:
                raise ValueError(
                    "Only padding_free_packed may be marked experimental"
                )
        object.__setattr__(
            self,
            "runtime_flags",
            _canonicalize_json_mapping(
                self.runtime_flags,
                field_name="runtime_flags",
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"


@dataclass(frozen=True)
class PackingEligibility:
    eligible: bool
    mode: PackingRuntimeMode
    training_mode: DetectionTrainingMode
    template_id: str
    reason: str
    experimental: bool = False
    requires_trie_metadata_preservation: bool = False


@dataclass(frozen=True)
class PackedSequenceMetadata:
    packed_example_id: str
    source_example_ids: tuple[str, ...]
    token_ranges: tuple[TokenSpan, ...]
    label_ranges: tuple[TokenSpan, ...]
    image_ranges: tuple[TokenSpan, ...]
    trie_metadata_ranges: tuple[TokenSpan, ...]
    template_id: str
    training_mode: str


@dataclass(frozen=True)
class PackingFingerprintInput:
    template_id: str
    template_version: int
    prompt_profile: str
    tokenizer_id: str
    object_ordering: str
    objective_variant: str
    state_weighting_policy: str
    normalization_policy: str
    loss_mask_version: str
    preprocessing_version: str
    profile: PackingProfile

    def __post_init__(self) -> None:
        if not isinstance(self.profile, PackingProfile):
            raise TypeError("profile must be a PackingProfile")
        object.__setattr__(
            self,
            "template_version",
            _require_positive_int(
                self.template_version,
                field_name="template_version",
            ),
        )
        for field_name in (
            "template_id",
            "prompt_profile",
            "tokenizer_id",
            "object_ordering",
            "objective_variant",
            "state_weighting_policy",
            "normalization_policy",
            "loss_mask_version",
            "preprocessing_version",
        ):
            raw_value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                _require_non_empty_string(raw_value, field_name=field_name),
            )


@dataclass(frozen=True)
class PackingFingerprintMetadata:
    template_id: str
    template_version: int
    prompt_profile: str
    tokenizer_id: str
    object_ordering: str
    objective_variant: str
    state_weighting_policy: str
    normalization_policy: str
    loss_mask_version: str
    preprocessing_version: str
    packing_mode: PackingRuntimeMode
    packing_length: int
    runtime_flags: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "template_version",
            _require_positive_int(
                self.template_version,
                field_name="template_version",
            ),
        )
        object.__setattr__(
            self,
            "packing_length",
            _require_non_negative_int(
                self.packing_length,
                field_name="packing_length",
            ),
        )
        for field_name in (
            "template_id",
            "prompt_profile",
            "tokenizer_id",
            "object_ordering",
            "objective_variant",
            "state_weighting_policy",
            "normalization_policy",
            "loss_mask_version",
            "preprocessing_version",
            "packing_mode",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_non_empty_string(getattr(self, field_name), field_name=field_name),
            )
        if self.packing_mode not in {"disabled", "static", "padding_free_packed"}:
            raise ValueError(f"Unsupported packing mode: {self.packing_mode!r}")
        object.__setattr__(
            self,
            "runtime_flags",
            _canonicalize_json_mapping(
                self.runtime_flags,
                field_name="runtime_flags",
            ),
        )


@dataclass(frozen=True)
class PackingFingerprint:
    metadata: PackingFingerprintMetadata
    canonical_json: str
    sha256: str


@dataclass(frozen=True)
class StaticSftPackingFingerprintRequest:
    detection_sequence_format: str
    prompt_profile: str
    tokenizer_id: str
    object_ordering: str
    profile: PackingProfile
    runtime_fields: Mapping[str, Any]
    objective_variant: str | None = None
    state_weighting_policy: str | None = None
    normalization_policy: str | None = None
    trainer_variant: str | None = None
    loss_mask_version: str = "legacy_stage1_static_packing_mask_v1"
    preprocessing_version: str = "legacy_base_caption_dataset_v1"

    def __post_init__(self) -> None:
        for field_name in (
            "detection_sequence_format",
            "prompt_profile",
            "tokenizer_id",
            "object_ordering",
            "loss_mask_version",
            "preprocessing_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_non_empty_string(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "objective_variant",
            "state_weighting_policy",
            "normalization_policy",
            "trainer_variant",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _require_non_empty_string(value, field_name=field_name),
                )
        object.__setattr__(
            self,
            "runtime_fields",
            _canonicalize_json_value(self.runtime_fields),
        )


def assess_packing_eligibility(
    *,
    template: DetectionSequenceTemplate,
    training_mode: DetectionTrainingMode,
    profile: PackingProfile,
) -> PackingEligibility:
    template_id = template.template_id
    if not profile.enabled:
        return PackingEligibility(
            eligible=False,
            mode=profile.mode,
            training_mode=training_mode,
            template_id=template_id,
            reason="packing is disabled",
        )

    if profile.mode == "padding_free_packed":
        return PackingEligibility(
            eligible=False,
            mode=profile.mode,
            training_mode=training_mode,
            template_id=template_id,
            reason=(
                "padding_free_packed is an experimental runtime mode and "
                "must not be treated as ordinary static packing"
            ),
            experimental=True,
        )

    recursive_sidecar_modes = {
        "random_permutation_et_rmp_ce",
        "prefix_rollin_et_rmp_ce",
    }
    if training_mode in recursive_sidecar_modes:
        return PackingEligibility(
            eligible=False,
            mode=profile.mode,
            training_mode=training_mode,
            template_id=template_id,
            reason=(
                f"{training_mode} packing is unsupported until trie target "
                "metadata preservation and target-position offsets are implemented"
            ),
            requires_trie_metadata_preservation=True,
        )

    if training_mode not in {"sorted_sft", "random_order_sft", "prefix_denoising_sft"}:
        raise ValueError(f"Unsupported detection packing training mode: {training_mode!r}")

    if not template.capabilities.supports_sft:
        return PackingEligibility(
            eligible=False,
            mode=profile.mode,
            training_mode=training_mode,
            template_id=template_id,
            reason=f"template {template_id!r} does not support SFT",
        )

    if not template.capabilities.supports_static_packing:
        return PackingEligibility(
            eligible=False,
            mode=profile.mode,
            training_mode=training_mode,
            template_id=template_id,
            reason=f"template {template_id!r} does not support static packing",
        )

    return PackingEligibility(
        eligible=True,
        mode=profile.mode,
        training_mode=training_mode,
        template_id=template_id,
        reason=(
            "eligible for static packing: full-sequence hard-CE example with "
            "template-declared static packing support"
        ),
    )


def require_packing_eligibility(
    *,
    template: DetectionSequenceTemplate,
    training_mode: DetectionTrainingMode,
    profile: PackingProfile,
) -> PackingEligibility:
    eligibility = assess_packing_eligibility(
        template=template,
        training_mode=training_mode,
        profile=profile,
    )
    if not eligibility.eligible:
        raise ValueError(eligibility.reason)
    return eligibility


def require_static_sft_packing_eligibility(
    *,
    detection_sequence_format: Any,
    profile: PackingProfile,
    object_ordering: str = "sorted",
    objective_variant: str | None = None,
    trainer_variant: str | None = None,
) -> PackingEligibility:
    template = get_detection_template(
        resolve_detection_template_id_for_static_packing(detection_sequence_format)
    )
    training_mode = resolve_static_sft_training_mode(
        objective_variant=objective_variant,
        object_ordering=object_ordering,
        trainer_variant=trainer_variant,
    )
    return require_packing_eligibility(
        template=template,
        training_mode=training_mode,
        profile=profile,
    )


def build_packing_fingerprint(
    payload: PackingFingerprintInput,
) -> PackingFingerprint:
    metadata = PackingFingerprintMetadata(
        template_id=payload.template_id,
        template_version=int(payload.template_version),
        prompt_profile=payload.prompt_profile,
        tokenizer_id=payload.tokenizer_id,
        object_ordering=payload.object_ordering,
        objective_variant=payload.objective_variant,
        state_weighting_policy=payload.state_weighting_policy,
        normalization_policy=payload.normalization_policy,
        loss_mask_version=payload.loss_mask_version,
        preprocessing_version=payload.preprocessing_version,
        packing_mode=payload.profile.mode,
        packing_length=int(payload.profile.packing_length),
        runtime_flags=_canonicalize_json_value(payload.profile.runtime_flags),
    )
    canonical_payload = {
        "loss_mask_version": metadata.loss_mask_version,
        "normalization_policy": metadata.normalization_policy,
        "object_ordering": metadata.object_ordering,
        "objective_variant": metadata.objective_variant,
        "packing_length": metadata.packing_length,
        "packing_mode": metadata.packing_mode,
        "preprocessing_version": metadata.preprocessing_version,
        "prompt_profile": metadata.prompt_profile,
        "runtime_flags": _json_ready_value(metadata.runtime_flags),
        "state_weighting_policy": metadata.state_weighting_policy,
        "template_id": metadata.template_id,
        "template_version": metadata.template_version,
        "tokenizer_id": metadata.tokenizer_id,
    }
    canonical_json = json.dumps(
        canonical_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return PackingFingerprint(
        metadata=metadata,
        canonical_json=canonical_json,
        sha256=sha256,
    )


def build_static_sft_packing_fingerprint(
    *,
    payload: PackingFingerprintInput,
    runtime_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the active static-SFT packing cache fingerprint.

    The legacy static-packing cache accepts a mapping as its fingerprint. Keep
    the existing runtime fields for cache debuggability, but make the detection
    packing contract the canonical owner of behaviorally meaningful template,
    objective, mask, and packing identity fields.
    """

    fingerprint = build_packing_fingerprint(payload)
    canonical_runtime_fields = _json_ready_value(
        _canonicalize_json_value(runtime_fields)
    )
    if not isinstance(canonical_runtime_fields, dict):
        raise TypeError("runtime_fields must be a mapping")
    return {
        **canonical_runtime_fields,
        "detection_packing_contract": {
            "schema_version": "detection_packing_fingerprint_v1",
            "sha256": fingerprint.sha256,
            "canonical_json": fingerprint.canonical_json,
            "metadata": _json_ready_value(fingerprint.metadata.__dict__),
        },
    }


def build_stage1_static_sft_packing_fingerprint(
    request: StaticSftPackingFingerprintRequest,
) -> dict[str, Any]:
    template = get_detection_template(
        resolve_detection_template_id_for_static_packing(
            request.detection_sequence_format
        )
    )
    training_mode = resolve_static_sft_training_mode(
        objective_variant=request.objective_variant,
        object_ordering=request.object_ordering,
        trainer_variant=request.trainer_variant,
    )
    require_packing_eligibility(
        template=template,
        training_mode=training_mode,
        profile=request.profile,
    )
    return build_static_sft_packing_fingerprint(
        payload=PackingFingerprintInput(
            template_id=template.template_id,
            template_version=template.capabilities.version,
            prompt_profile=request.prompt_profile,
            tokenizer_id=request.tokenizer_id,
            object_ordering=request.object_ordering,
            objective_variant=training_mode,
            state_weighting_policy=(
                request.state_weighting_policy
                or _default_state_weighting_for_mode(training_mode)
            ),
            normalization_policy=(
                request.normalization_policy
                or _default_normalization_for_mode(training_mode)
            ),
            loss_mask_version=request.loss_mask_version,
            preprocessing_version=request.preprocessing_version,
            profile=request.profile,
        ),
        runtime_fields=request.runtime_fields,
    )


__all__ = [
    "PackedSequenceMetadata",
    "PackingEligibility",
    "PackingFingerprint",
    "PackingFingerprintInput",
    "PackingFingerprintMetadata",
    "PackingProfile",
    "PackingRuntimeMode",
    "StaticSftPackingFingerprintRequest",
    "assess_packing_eligibility",
    "build_stage1_static_sft_packing_fingerprint",
    "build_packing_fingerprint",
    "build_static_sft_packing_fingerprint",
    "require_static_sft_packing_eligibility",
    "require_packing_eligibility",
    "resolve_detection_template_id_for_static_packing",
    "resolve_static_sft_training_mode",
]
