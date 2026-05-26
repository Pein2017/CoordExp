from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias, cast


SupervisionStage: TypeAlias = Literal["stage1", "stage2"]
SupervisionChannel: TypeAlias = Literal["primary", "rollout_correction"]
SemanticScalar: TypeAlias = str | int | float | bool | None
SemanticMetadata: TypeAlias = Mapping[str, SemanticScalar]

VALID_SUPERVISION_STAGES: frozenset[str] = frozenset(("stage1", "stage2"))
VALID_SUPERVISION_CHANNELS: frozenset[str] = frozenset(
    ("primary", "rollout_correction")
)
FORBIDDEN_SEMANTIC_METADATA_KEYS: frozenset[str] = frozenset(
    (
        "assistant_text",
        "input_ids",
        "model",
        "model_handle",
        "model_inputs",
        "model_output",
        "raw",
        "raw_config",
        "rendered",
        "rendered_prompt",
        "rendered_text",
        "rendered_assistant_text",
        "token",
        "token_ids",
        "tokens",
        "tensor",
        "tensor_shape",
        "tensors",
        "tokenizer",
    )
)
FORBIDDEN_SEMANTIC_METADATA_KEY_PREFIXES: tuple[str, ...] = (
    "assistant_text.",
    "assistant_text-",
    "assistant_text_",
    "input_ids.",
    "input_ids-",
    "input_ids_",
    "model.",
    "model-",
    "model_",
    "model_handle.",
    "model_handle-",
    "model_handle_",
    "model_inputs.",
    "model_inputs-",
    "model_inputs_",
    "model_output.",
    "model_output-",
    "model_output_",
    "raw.",
    "raw-",
    "raw_",
    "raw_config.",
    "raw_config-",
    "raw_config_",
    "rendered.",
    "rendered-",
    "rendered_",
    "rendered_prompt.",
    "rendered_prompt-",
    "rendered_prompt_",
    "rendered_text.",
    "rendered_text-",
    "rendered_text_",
    "rendered_assistant_text.",
    "rendered_assistant_text-",
    "rendered_assistant_text_",
    "token.",
    "token-",
    "token_",
    "token_ids.",
    "token_ids-",
    "token_ids_",
    "tokens.",
    "tokens-",
    "tokens_",
    "tensor.",
    "tensor-",
    "tensor_",
    "tensors.",
    "tensors-",
    "tensors_",
    "tokenizer.",
    "tokenizer-",
    "tokenizer_",
)


def freeze_semantic_metadata(
    metadata: SemanticMetadata | None,
) -> SemanticMetadata:
    """Return an immutable semantic metadata mapping.

    :param metadata: Optional scalar metadata for a supervision context or plan.
    :returns: Read-only mapping with semantic scalar values.
    :raises TypeError: If a key is not a string or a value is not a supported
        semantic scalar.
    """

    if metadata is None:
        return cast(SemanticMetadata, MappingProxyType({}))
    if not isinstance(metadata, Mapping):
        raise TypeError("semantic metadata must be a mapping")

    frozen_metadata: dict[str, SemanticScalar] = {}
    for key, value in metadata.items():
        if type(key) is not str:
            raise TypeError("semantic metadata keys must be strings")
        if (
            key in FORBIDDEN_SEMANTIC_METADATA_KEYS
            or key.startswith(FORBIDDEN_SEMANTIC_METADATA_KEY_PREFIXES)
        ):
            raise ValueError(f"semantic metadata key is not allowed: {key!r}")
        if (
            value is not None
            and type(value) is not str
            and type(value) is not int
            and type(value) is not float
            and type(value) is not bool
        ):
            raise TypeError(
                "semantic metadata values must be str, int, float, bool, or None"
            )
        if type(value) is float and not math.isfinite(value):
            raise ValueError("semantic metadata float values must be finite")
        frozen_metadata[key] = value

    return cast(SemanticMetadata, MappingProxyType(frozen_metadata))


def validate_supervision_stage(stage: object) -> SupervisionStage:
    """Return a validated supervision stage.

    :param stage: Candidate stage value.
    :returns: Validated supervision stage.
    :raises ValueError: If the stage is not supported.
    """

    if type(stage) is not str:
        raise TypeError("supervision stage must be a semantic string")
    if stage not in VALID_SUPERVISION_STAGES:
        raise ValueError(f"unsupported supervision stage: {stage!r}")

    return cast(SupervisionStage, stage)


def validate_supervision_channel(channel: object) -> SupervisionChannel:
    """Return a validated supervision channel.

    :param channel: Candidate channel value.
    :returns: Validated supervision channel.
    :raises ValueError: If the channel is not supported.
    """

    if type(channel) is not str:
        raise TypeError("supervision channel must be a semantic string")
    if channel not in VALID_SUPERVISION_CHANNELS:
        raise ValueError(f"unsupported supervision channel: {channel!r}")

    return cast(SupervisionChannel, channel)


def validate_stage_channel_pair(
    *,
    stage: SupervisionStage,
    channel: SupervisionChannel,
) -> None:
    """Validate stage/channel ownership compatibility.

    :param stage: Validated supervision stage.
    :param channel: Validated supervision channel.
    :raises ValueError: If the channel is incompatible with the stage.
    """

    if stage == "stage1" and channel != "primary":
        raise ValueError("stage1 supervision channel must be primary")


def validate_required_semantic_string(value: object, *, field_name: str) -> str:
    """Return a validated required semantic string field.

    :param value: Candidate string value.
    :param field_name: Field name for error reporting.
    :returns: Validated string value.
    :raises TypeError: If the value is not a string.
    :raises ValueError: If the value is empty.
    """

    if type(value) is not str:
        raise TypeError(f"{field_name} must be a semantic string")
    if value == "":
        raise ValueError(f"{field_name} must not be empty")

    return value


def validate_optional_semantic_string(
    value: object | None,
    *,
    field_name: str,
) -> str | None:
    """Return a validated optional semantic string field.

    :param value: Candidate optional string value.
    :param field_name: Field name for error reporting.
    :returns: Validated string value or ``None``.
    :raises TypeError: If the value is neither ``None`` nor a string.
    :raises ValueError: If the value is an empty string.
    """

    if value is None:
        return None

    return validate_required_semantic_string(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class SupervisionContext:
    """Semantic execution context shared by per-example supervision plans.

    :param context_id: Stable identifier for the semantic planning context.
    :param dataset_id: Dataset identity associated with the planned example.
    :param split: Dataset split associated with the planned example.
    :param template_id: Template family selected before rendering.
    :param stage: Training stage for this context.
    :param channel: Optional supervision channel ownership for Stage-2.
    :param experiment_id: Optional experiment or run identifier.
    :param metadata: Optional scalar semantic metadata.
    """

    context_id: str
    dataset_id: str
    split: str
    template_id: str
    stage: SupervisionStage
    channel: SupervisionChannel = "primary"
    experiment_id: str | None = None
    metadata: SemanticMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate semantic identifiers and freeze scalar metadata."""

        object.__setattr__(
            self,
            "context_id",
            validate_required_semantic_string(self.context_id, field_name="context_id"),
        )
        object.__setattr__(
            self,
            "dataset_id",
            validate_required_semantic_string(self.dataset_id, field_name="dataset_id"),
        )
        object.__setattr__(
            self,
            "split",
            validate_required_semantic_string(self.split, field_name="split"),
        )
        object.__setattr__(
            self,
            "template_id",
            validate_required_semantic_string(self.template_id, field_name="template_id"),
        )
        object.__setattr__(
            self,
            "experiment_id",
            validate_optional_semantic_string(
                self.experiment_id,
                field_name="experiment_id",
            ),
        )
        stage = validate_supervision_stage(self.stage)
        channel = validate_supervision_channel(self.channel)
        validate_stage_channel_pair(stage=stage, channel=channel)

        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "metadata", freeze_semantic_metadata(self.metadata))
