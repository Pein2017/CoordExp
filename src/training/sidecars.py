"""Semantic sidecar containers for unified training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias, cast

SidecarMetadata: TypeAlias = Mapping[str, Any]


def _freeze_mapping(metadata: Mapping[str, Any] | None) -> SidecarMetadata:
    """Return a shallow immutable sidecar metadata mapping."""

    if metadata is None:
        return cast(SidecarMetadata, MappingProxyType({}))
    if not isinstance(metadata, Mapping):
        raise TypeError("sidecar metadata must be a mapping")

    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        if type(key) is not str:
            raise TypeError("sidecar metadata keys must be plain strings")
        copied[key] = value

    return cast(SidecarMetadata, MappingProxyType(copied))


def _freeze_sequence(values: Sequence[Any]) -> tuple[Any, ...]:
    """Return a shallow immutable tuple from a non-string sequence."""

    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise TypeError("sidecar sequence fields must be sequences")

    return tuple(values)


@dataclass(frozen=True, slots=True)
class SupervisionSidecars:
    """Supervision payloads available to bridge and runner layers.

    :param spans: Semantic supervision spans or compatible span payloads.
    :param payloads: Additional supervision payload objects.
    :param metadata: Optional sidecar metadata.
    """

    spans: Sequence[Any] = field(default_factory=tuple)
    payloads: Sequence[Any] = field(default_factory=tuple)
    metadata: SidecarMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze supervision sidecar containers."""

        object.__setattr__(self, "spans", _freeze_sequence(self.spans))
        object.__setattr__(self, "payloads", _freeze_sequence(self.payloads))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class DiagnosticSidecars:
    """Diagnostics and provenance available outside model forwarding.

    :param rendered_assistant_text: Optional rendered assistant text for audits.
    :param token_roles: Optional token-role payload for diagnostics.
    :param metadata: Optional sidecar metadata.
    """

    rendered_assistant_text: str | None = None
    token_roles: Sequence[Any] = field(default_factory=tuple)
    metadata: SidecarMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze diagnostic sidecar containers."""

        if self.rendered_assistant_text is not None and type(
            self.rendered_assistant_text
        ) is not str:
            raise TypeError("rendered_assistant_text must be a string or None")

        object.__setattr__(self, "token_roles", _freeze_sequence(self.token_roles))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class DatasetSidecars:
    """Dataset identity and row provenance available outside model forwarding.

    :param sample_id: Optional stable sample identifier.
    :param dataset_id: Optional dataset identifier.
    :param split: Optional dataset split.
    :param base_idx: Optional source row index.
    :param metadata: Optional sidecar metadata.
    """

    sample_id: str | None = None
    dataset_id: str | None = None
    split: str | None = None
    base_idx: int | None = None
    metadata: SidecarMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scalar dataset sidecar identity fields."""

        for field_name in ("sample_id", "dataset_id", "split"):
            value = getattr(self, field_name)
            if value is not None and type(value) is not str:
                raise TypeError(f"{field_name} must be a string or None")
        if self.base_idx is not None and type(self.base_idx) is not int:
            raise TypeError("base_idx must be an integer or None")

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Stage2OwnershipSidecars:
    """Stage-2 ownership payloads available to bridge and runner layers.

    :param assignment_result: Optional assignment result payload.
    :param duplicate_filter_result: Optional duplicate-filtering payload.
    :param rollout_payload: Optional rollout ownership payload.
    :param metadata: Optional sidecar metadata.
    """

    assignment_result: Any = None
    duplicate_filter_result: Any = None
    rollout_payload: Any = None
    metadata: SidecarMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze Stage-2 sidecar metadata."""

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TrainingSidecars:
    """Grouped sidecars that are never model-forward inputs.

    The object is a semantic data holder only. It deliberately exposes no
    ``model_inputs`` or ``forward`` surface.
    """

    supervision: SupervisionSidecars = field(default_factory=SupervisionSidecars)
    diagnostics: DiagnosticSidecars = field(default_factory=DiagnosticSidecars)
    dataset: DatasetSidecars = field(default_factory=DatasetSidecars)
    stage2: Stage2OwnershipSidecars = field(default_factory=Stage2OwnershipSidecars)

    def __post_init__(self) -> None:
        """Validate grouped sidecar container types."""

        if type(self.supervision) is not SupervisionSidecars:
            raise TypeError("supervision sidecars must be SupervisionSidecars")
        if type(self.diagnostics) is not DiagnosticSidecars:
            raise TypeError("diagnostic sidecars must be DiagnosticSidecars")
        if type(self.dataset) is not DatasetSidecars:
            raise TypeError("dataset sidecars must be DatasetSidecars")
        if type(self.stage2) is not Stage2OwnershipSidecars:
            raise TypeError("stage2 sidecars must be Stage2OwnershipSidecars")


__all__ = [
    "DatasetSidecars",
    "DiagnosticSidecars",
    "SidecarMetadata",
    "Stage2OwnershipSidecars",
    "SupervisionSidecars",
    "TrainingSidecars",
]
