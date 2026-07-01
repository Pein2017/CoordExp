from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import fsum
from typing import Literal, TypeAlias

MetricReducer: TypeAlias = Literal["ratio", "weighted_mean", "sum", "last"]
MetricUnit: TypeAlias = Literal[
    "token",
    "span",
    "object",
    "slot",
    "image",
    "batch",
    "sample",
]
MetricScalar: TypeAlias = float | int
MetricValue: TypeAlias = float | int | str | bool | None


@dataclass(frozen=True)
class MetricIdentity:
    """Stable semantic identity for a metric before flat alias publication."""

    key: str
    reducer: MetricReducer
    unit: MetricUnit
    semantic_role: str | None = None
    token_role: str | None = None
    vocab_scope: str | None = None
    coordinate_surface: str | None = None
    geometry_type: str | None = None
    slot_name: str | None = None
    object_scope: str | None = None
    template_id: str | None = None
    parser_mode: str | None = None
    metric_surface: str | None = None
    stage: str | None = None
    channel: str | None = None
    objective_id: str | None = None
    provenance: str | None = None
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        _require_non_empty_string("key", self.key)


@dataclass(frozen=True)
class MetricEvent:
    """Typed metric observation with explicit denominator and identity axes."""

    key: str
    numerator: float | None
    denominator: float | None
    value: MetricValue
    reducer: MetricReducer
    unit: MetricUnit
    semantic_role: str | None = None
    token_role: str | None = None
    vocab_scope: str | None = None
    coordinate_surface: str | None = None
    geometry_type: str | None = None
    slot_name: str | None = None
    object_scope: str | None = None
    template_id: str | None = None
    parser_mode: str | None = None
    metric_surface: str | None = None
    stage: str | None = None
    channel: str | None = None
    objective_id: str | None = None
    provenance: str | None = None
    diagnostic_only: bool = False
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_string("key", self.key)
        aliases = tuple(self.aliases)
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"MetricEvent aliases must be unique: {aliases!r}")
        for alias in aliases:
            _require_non_empty_string("alias", alias)
        object.__setattr__(self, "aliases", aliases)

        if self.reducer in {"ratio", "weighted_mean"}:
            if self.numerator is None:
                raise ValueError(f"{self.reducer} MetricEvent requires numerator")
            if self.denominator is None:
                raise ValueError(f"{self.reducer} MetricEvent requires denominator")
            if self.denominator < 0:
                raise ValueError(f"{self.reducer} MetricEvent denominator must be >= 0")
        elif self.reducer in {"sum", "last"}:
            if self.value is None:
                raise ValueError(f"{self.reducer} MetricEvent requires value")
        else:
            raise ValueError(f"Unsupported MetricEvent reducer: {self.reducer!r}")

    @property
    def identity(self) -> MetricIdentity:
        return MetricIdentity(
            key=self.key,
            reducer=self.reducer,
            unit=self.unit,
            semantic_role=self.semantic_role,
            token_role=self.token_role,
            vocab_scope=self.vocab_scope,
            coordinate_surface=self.coordinate_surface,
            geometry_type=self.geometry_type,
            slot_name=self.slot_name,
            object_scope=self.object_scope,
            template_id=self.template_id,
            parser_mode=self.parser_mode,
            metric_surface=self.metric_surface,
            stage=self.stage,
            channel=self.channel,
            objective_id=self.objective_id,
            provenance=self.provenance,
            diagnostic_only=self.diagnostic_only,
        )


class MetricAliasRegistry:
    """Instance-scoped alias validator for compatibility-safe flat log names."""

    def __init__(self) -> None:
        self._aliases: dict[str, MetricIdentity] = {}

    def register(self, alias: str, identity: MetricIdentity) -> None:
        _require_non_empty_string("alias", alias)
        existing = self._aliases.get(alias)
        if existing is None:
            self._aliases[alias] = identity
            return
        if existing != identity:
            raise ValueError(
                "Metric alias collision for "
                f"{alias!r}: {existing.key!r} != {identity.key!r}"
            )

    def register_aliases(
        self,
        identity: MetricIdentity,
        aliases: Iterable[str],
    ) -> None:
        for alias in aliases:
            self.register(alias, identity)

    def register_event(self, event: MetricEvent) -> None:
        self.register_aliases(event.identity, event.aliases)

    def resolve(self, alias: str) -> MetricIdentity:
        return self._aliases[alias]

    def as_mapping(self) -> Mapping[str, MetricIdentity]:
        return dict(self._aliases)


def ratio_event(
    key: str,
    numerator: MetricScalar,
    denominator: MetricScalar,
    *,
    unit: MetricUnit,
    aliases: Sequence[str] = (),
    semantic_role: str | None = None,
    token_role: str | None = None,
    vocab_scope: str | None = None,
    coordinate_surface: str | None = None,
    geometry_type: str | None = None,
    slot_name: str | None = None,
    object_scope: str | None = None,
    template_id: str | None = None,
    parser_mode: str | None = None,
    metric_surface: str | None = None,
    stage: str | None = None,
    channel: str | None = None,
    objective_id: str | None = None,
    provenance: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    return MetricEvent(
        key=key,
        numerator=_as_float("numerator", numerator),
        denominator=_as_float("denominator", denominator),
        value=None,
        reducer="ratio",
        unit=unit,
        semantic_role=semantic_role,
        token_role=token_role,
        vocab_scope=vocab_scope,
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        slot_name=slot_name,
        object_scope=object_scope,
        template_id=template_id,
        parser_mode=parser_mode,
        metric_surface=metric_surface,
        stage=stage,
        channel=channel,
        objective_id=objective_id,
        provenance=provenance,
        diagnostic_only=diagnostic_only,
        aliases=tuple(aliases),
    )


def weighted_mean_event(
    key: str,
    value: MetricScalar,
    weight: MetricScalar,
    *,
    unit: MetricUnit,
    aliases: Sequence[str] = (),
    semantic_role: str | None = None,
    token_role: str | None = None,
    vocab_scope: str | None = None,
    coordinate_surface: str | None = None,
    geometry_type: str | None = None,
    slot_name: str | None = None,
    object_scope: str | None = None,
    template_id: str | None = None,
    parser_mode: str | None = None,
    metric_surface: str | None = None,
    stage: str | None = None,
    channel: str | None = None,
    objective_id: str | None = None,
    provenance: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    value_float = _as_float("value", value)
    weight_float = _as_float("weight", weight)
    return MetricEvent(
        key=key,
        numerator=value_float * weight_float,
        denominator=weight_float,
        value=value_float,
        reducer="weighted_mean",
        unit=unit,
        semantic_role=semantic_role,
        token_role=token_role,
        vocab_scope=vocab_scope,
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        slot_name=slot_name,
        object_scope=object_scope,
        template_id=template_id,
        parser_mode=parser_mode,
        metric_surface=metric_surface,
        stage=stage,
        channel=channel,
        objective_id=objective_id,
        provenance=provenance,
        diagnostic_only=diagnostic_only,
        aliases=tuple(aliases),
    )


def sum_event(
    key: str,
    value: MetricScalar,
    *,
    unit: MetricUnit,
    aliases: Sequence[str] = (),
    semantic_role: str | None = None,
    token_role: str | None = None,
    vocab_scope: str | None = None,
    coordinate_surface: str | None = None,
    geometry_type: str | None = None,
    slot_name: str | None = None,
    object_scope: str | None = None,
    template_id: str | None = None,
    parser_mode: str | None = None,
    metric_surface: str | None = None,
    stage: str | None = None,
    channel: str | None = None,
    objective_id: str | None = None,
    provenance: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    value_float = _as_float("value", value)
    return MetricEvent(
        key=key,
        numerator=value_float,
        denominator=None,
        value=value_float,
        reducer="sum",
        unit=unit,
        semantic_role=semantic_role,
        token_role=token_role,
        vocab_scope=vocab_scope,
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        slot_name=slot_name,
        object_scope=object_scope,
        template_id=template_id,
        parser_mode=parser_mode,
        metric_surface=metric_surface,
        stage=stage,
        channel=channel,
        objective_id=objective_id,
        provenance=provenance,
        diagnostic_only=diagnostic_only,
        aliases=tuple(aliases),
    )


def last_event(
    key: str,
    value: MetricValue,
    *,
    unit: MetricUnit,
    aliases: Sequence[str] = (),
    semantic_role: str | None = None,
    token_role: str | None = None,
    vocab_scope: str | None = None,
    coordinate_surface: str | None = None,
    geometry_type: str | None = None,
    slot_name: str | None = None,
    object_scope: str | None = None,
    template_id: str | None = None,
    parser_mode: str | None = None,
    metric_surface: str | None = None,
    stage: str | None = None,
    channel: str | None = None,
    objective_id: str | None = None,
    provenance: str | None = None,
    diagnostic_only: bool = False,
) -> MetricEvent:
    return MetricEvent(
        key=key,
        numerator=None,
        denominator=None,
        value=value,
        reducer="last",
        unit=unit,
        semantic_role=semantic_role,
        token_role=token_role,
        vocab_scope=vocab_scope,
        coordinate_surface=coordinate_surface,
        geometry_type=geometry_type,
        slot_name=slot_name,
        object_scope=object_scope,
        template_id=template_id,
        parser_mode=parser_mode,
        metric_surface=metric_surface,
        stage=stage,
        channel=channel,
        objective_id=objective_id,
        provenance=provenance,
        diagnostic_only=diagnostic_only,
        aliases=tuple(aliases),
    )


def reduce_metric_events(events: Iterable[MetricEvent]) -> dict[str, MetricValue]:
    grouped: dict[str, list[MetricEvent]] = {}
    identities: dict[str, MetricIdentity] = {}
    for event in events:
        existing_identity = identities.get(event.key)
        if existing_identity is None:
            identities[event.key] = event.identity
        elif existing_identity != event.identity:
            raise ValueError(
                "Metric key collision for "
                f"{event.key!r}: incompatible metric identities"
            )
        grouped.setdefault(event.key, []).append(event)

    reduced: dict[str, MetricValue] = {}
    for key, key_events in grouped.items():
        reducer = key_events[0].reducer
        if reducer == "ratio":
            numerator = fsum(_require_event_number(event.numerator) for event in key_events)
            denominator = fsum(
                _require_event_number(event.denominator) for event in key_events
            )
            reduced[key] = None if denominator == 0 else numerator / denominator
        elif reducer == "weighted_mean":
            weighted_sum = fsum(
                _require_event_number(event.numerator) for event in key_events
            )
            weight_sum = fsum(
                _require_event_number(event.denominator) for event in key_events
            )
            reduced[key] = None if weight_sum == 0 else weighted_sum / weight_sum
        elif reducer == "sum":
            reduced[key] = fsum(_require_event_number(event.value) for event in key_events)
        elif reducer == "last":
            reduced[key] = key_events[-1].value
        else:
            raise ValueError(f"Unsupported MetricEvent reducer: {reducer!r}")
    return reduced


def flatten_metric_events(
    events: Iterable[MetricEvent],
    *,
    registry: MetricAliasRegistry | None = None,
) -> dict[str, float]:
    """Reduce typed events into trainer-safe flat numeric metrics.

    Canonical event keys are always emitted when their reduced value is numeric.
    Compatible aliases are emitted only through ``MetricAliasRegistry`` identity
    checks, so legacy aliases cannot drift to a different denominator, vocab
    scope, parser surface, or diagnostic surface.

    Events whose reducer returns ``None`` (for example ratio metrics with a zero
    denominator) are omitted from the flat log map.
    """

    events_tuple = tuple(events)
    alias_registry = register_legacy_metric_aliases(registry)
    for event in events_tuple:
        alias_registry.register_event(event)
    for event in events_tuple:
        _reject_reserved_alias_key(event, alias_registry)

    reduced = reduce_metric_events(events_tuple)
    flat: dict[str, float] = {}
    aliases_by_identity = _aliases_by_identity(alias_registry)
    emitted_identities: dict[str, MetricIdentity] = {}
    for event in events_tuple:
        identity = event.identity
        value = reduced[event.key]
        if value is None:
            continue
        metric_value = _require_flat_metric_number(event.key, value)
        _publish_flat_metric(
            flat,
            emitted_identities,
            event.key,
            metric_value,
            identity,
        )
        for alias in aliases_by_identity.get(identity, ()):
            _publish_flat_metric(
                flat,
                emitted_identities,
                alias,
                metric_value,
                identity,
            )
    return flat


def _require_non_empty_string(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY = MetricIdentity(
    key="coord_token_acc/full_vocab/top1",
    reducer="ratio",
    unit="token",
    semantic_role="bbox_coord",
    token_role="coord",
    vocab_scope="full_vocab",
    coordinate_surface="coord_token",
    geometry_type="bbox",
    metric_surface="training_logits",
)
FULL_VOCAB_COORD_TOKEN_ACC_ALIASES = ("coord_token_acc",)

FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY = MetricIdentity(
    key="coord_token_acc/full_vocab/top5",
    reducer="ratio",
    unit="token",
    semantic_role="bbox_coord",
    token_role="coord",
    vocab_scope="full_vocab",
    coordinate_surface="coord_token",
    geometry_type="bbox",
    metric_surface="training_logits",
)
FULL_VOCAB_COORD_TOKEN_ACC_TOP5_ALIASES = ("coord_token_acc_top5",)

COORD_VOCAB_TOKEN_ACC_IDENTITY = MetricIdentity(
    key="coord_vocab_token_acc/top1",
    reducer="ratio",
    unit="token",
    semantic_role="bbox_coord",
    token_role="coord",
    vocab_scope="coord_vocab",
    coordinate_surface="coord_token",
    geometry_type="bbox",
    metric_surface="training_logits",
)


def register_legacy_metric_aliases(
    registry: MetricAliasRegistry | None = None,
) -> MetricAliasRegistry:
    registry = MetricAliasRegistry() if registry is None else registry
    registry.register_aliases(
        FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY,
        FULL_VOCAB_COORD_TOKEN_ACC_ALIASES,
    )
    registry.register_aliases(
        FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY,
        FULL_VOCAB_COORD_TOKEN_ACC_TOP5_ALIASES,
    )
    return registry


def _as_float(field_name: str, value: MetricScalar) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{field_name} must be numeric")
    return float(value)


def _require_event_number(value: MetricValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"MetricEvent reducer expected numeric value, got {value!r}")
    return float(value)


def _require_flat_metric_number(key: str, value: MetricValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(
            f"MetricEvent {key!r} reduced to non-numeric flat log value {value!r}"
        )
    return float(value)


def _aliases_by_identity(
    registry: MetricAliasRegistry,
) -> dict[MetricIdentity, tuple[str, ...]]:
    aliases: dict[MetricIdentity, list[str]] = {}
    for alias, identity in registry.as_mapping().items():
        aliases.setdefault(identity, []).append(alias)
    return {
        identity: tuple(sorted(identity_aliases))
        for identity, identity_aliases in aliases.items()
    }


def _reject_reserved_alias_key(
    event: MetricEvent,
    registry: MetricAliasRegistry,
) -> None:
    try:
        reserved_identity = registry.resolve(event.key)
    except KeyError:
        return
    if reserved_identity != event.identity:
        raise ValueError(
            "Metric event key is reserved as an alias for a different identity: "
            f"{event.key!r}: {reserved_identity.key!r} != {event.identity.key!r}"
        )


def _publish_flat_metric(
    flat: dict[str, float],
    emitted_identities: dict[str, MetricIdentity],
    key: str,
    value: float,
    identity: MetricIdentity,
) -> None:
    existing_identity = emitted_identities.get(key)
    if existing_identity is not None and existing_identity != identity:
        raise ValueError(
            "Metric flat key collision for "
            f"{key!r}: {existing_identity.key!r} != {identity.key!r}"
        )
    existing_value = flat.get(key)
    if existing_value is not None and existing_value != value:
        raise ValueError(
            "Metric flat key collision for "
            f"{key!r}: incompatible reduced values {existing_value!r} != {value!r}"
        )
    flat[key] = value
    emitted_identities[key] = identity


__all__ = [
    "COORD_VOCAB_TOKEN_ACC_IDENTITY",
    "FULL_VOCAB_COORD_TOKEN_ACC_ALIASES",
    "FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY",
    "FULL_VOCAB_COORD_TOKEN_ACC_TOP5_ALIASES",
    "FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY",
    "MetricAliasRegistry",
    "MetricEvent",
    "MetricIdentity",
    "MetricReducer",
    "MetricUnit",
    "MetricValue",
    "flatten_metric_events",
    "last_event",
    "ratio_event",
    "reduce_metric_events",
    "register_legacy_metric_aliases",
    "sum_event",
    "weighted_mean_event",
]
