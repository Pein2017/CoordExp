from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Literal, TypeAlias, cast


DistributionKind: TypeAlias = Literal[
    "hard_token",
    "multi_positive_token",
    "coordinate_soft_token",
    "box_regression",
]
TargetObjectiveId: TypeAlias = Literal[
    "token_ce",
    "trie_ce",
    "coord_soft_ce",
    "box_regression",
]
CoordinateSoftTargetFamily: TypeAlias = Literal["iou_gibbs_v0", "ciou_gibbs_v0"]
CoordinateSoftLossMode: TypeAlias = Literal[
    "full_vocab_ce",
    "coord_vocab_ce",
    "w1_distance",
]
SemanticBBoxInput: TypeAlias = Sequence[int | float]
SemanticBBox: TypeAlias = tuple[float, float, float, float]
MultiPositiveWeightInput: TypeAlias = Sequence[int | float] | None


def _retired_identifier(*parts: str) -> str:
    """Return a retired identifier from semantic fragments."""

    return "_".join(parts)


VALID_DISTRIBUTION_KINDS: frozenset[str] = frozenset(
    (
        "hard_token",
        "multi_positive_token",
        "coordinate_soft_token",
        "box_regression",
    )
)
RETIRED_DISTRIBUTION_KINDS: frozenset[str] = frozenset(
    (
        _retired_identifier("duplicate", "negative"),
        _retired_identifier("adjacent", "repulsion"),
        _retired_identifier(
            "forced",
            "continuation",
        ),
    )
)
VALID_TARGET_OBJECTIVE_IDS: frozenset[str] = frozenset(
    (
        "token_ce",
        "trie_ce",
        "coord_soft_ce",
        "box_regression",
    )
)
RETIRED_TARGET_OBJECTIVE_IDS: frozenset[str] = frozenset(
    (
        _retired_identifier("adjacent", "repulsion"),
        _retired_identifier("duplicate", "burst", "unlikelihood"),
        _retired_identifier("duplicate", "negative"),
        _retired_identifier(
            "forced",
            "continuation",
        ),
    )
)
VALID_COORDINATE_SOFT_TARGET_FAMILIES: frozenset[str] = frozenset(
    ("iou_gibbs_v0", "ciou_gibbs_v0")
)
VALID_COORDINATE_SOFT_LOSS_MODES: frozenset[str] = frozenset(
    ("full_vocab_ce", "coord_vocab_ce", "w1_distance")
)
SUPPORTED_OBJECTIVES_BY_DISTRIBUTION_KIND: Mapping[DistributionKind, frozenset[str]] = (
    MappingProxyType(
        {
            "hard_token": frozenset(("token_ce",)),
            "multi_positive_token": frozenset(("trie_ce",)),
            "coordinate_soft_token": frozenset(("coord_soft_ce",)),
            "box_regression": frozenset(("box_regression",)),
        }
    )
)


def _supported_kind_list() -> str:
    """Return supported distribution kinds for error messages."""

    return ", ".join(sorted(VALID_DISTRIBUTION_KINDS))


def _supported_objective_id_list() -> str:
    """Return supported objective identifiers for error messages."""

    return ", ".join(sorted(VALID_TARGET_OBJECTIVE_IDS))


def _supported_coordinate_soft_target_family_list() -> str:
    """Return supported coordinate-soft target families for error messages."""

    return ", ".join(sorted(VALID_COORDINATE_SOFT_TARGET_FAMILIES))


def _supported_coordinate_soft_loss_mode_list() -> str:
    """Return supported coordinate-soft loss modes for error messages."""

    return ", ".join(sorted(VALID_COORDINATE_SOFT_LOSS_MODES))


def validate_distribution_kind(kind: object) -> DistributionKind:
    """Return a validated target distribution kind.

    :param kind: Candidate distribution kind.
    :returns: Supported target distribution kind.
    :raises TypeError: If the kind is not a plain string.
    :raises ValueError: If the kind is retired or unsupported.
    """

    if type(kind) is not str:
        raise TypeError("target distribution kind must be a semantic string")
    if kind in RETIRED_DISTRIBUTION_KINDS:
        raise ValueError(
            "retired target distribution kind "
            f"{kind!r} is not supported; use one of: {_supported_kind_list()}"
        )
    if kind not in VALID_DISTRIBUTION_KINDS:
        raise ValueError(
            "unsupported target distribution kind "
            f"{kind!r}; use one of: {_supported_kind_list()}"
        )

    return cast(DistributionKind, kind)


def validate_target_objective_id(objective_id: object) -> TargetObjectiveId:
    """Return a validated semantic target objective identifier.

    :param objective_id: Candidate objective identifier.
    :returns: Supported target objective identifier.
    :raises TypeError: If the identifier is not a plain string.
    :raises ValueError: If the identifier is retired or unsupported.
    """

    if type(objective_id) is not str:
        raise TypeError("target objective id must be a semantic string")
    if objective_id in RETIRED_TARGET_OBJECTIVE_IDS:
        raise ValueError(
            "retired target objective id "
            f"{objective_id!r} is not supported; use one of: "
            f"{_supported_objective_id_list()}"
        )
    if objective_id not in VALID_TARGET_OBJECTIVE_IDS:
        raise ValueError(
            "unsupported target objective id "
            f"{objective_id!r}; use one of: {_supported_objective_id_list()}"
        )

    return cast(TargetObjectiveId, objective_id)


def _validate_coordinate_soft_target_family(
    target_family: object,
) -> CoordinateSoftTargetFamily:
    """Return a validated coordinate-soft target-family identifier."""

    if type(target_family) is not str:
        raise TypeError("coordinate soft target_family must be a semantic string")
    if target_family not in VALID_COORDINATE_SOFT_TARGET_FAMILIES:
        raise ValueError(
            "unsupported coordinate soft target_family "
            f"{target_family!r}; use one of: "
            f"{_supported_coordinate_soft_target_family_list()}"
        )

    return cast(CoordinateSoftTargetFamily, target_family)


def _validate_coordinate_soft_loss_mode(loss_mode: object) -> CoordinateSoftLossMode:
    """Return a validated coordinate-soft loss-mode identifier."""

    if type(loss_mode) is not str:
        raise TypeError("coordinate soft loss_mode must be a semantic string")
    if loss_mode not in VALID_COORDINATE_SOFT_LOSS_MODES:
        raise ValueError(
            "unsupported coordinate soft loss_mode "
            f"{loss_mode!r}; use one of: {_supported_coordinate_soft_loss_mode_list()}"
        )

    return cast(CoordinateSoftLossMode, loss_mode)


def _normalize_token_id(token_id: object) -> int:
    """Return a validated non-negative token id.

    :param token_id: Candidate token id.
    :returns: Non-negative integer token id.
    :raises TypeError: If the id is not an integer.
    :raises ValueError: If the id is negative.
    """

    if type(token_id) is not int:
        raise TypeError("target token ids must be non-negative integers")
    if token_id < 0:
        raise ValueError("target token ids must be non-negative")

    return token_id


def _normalize_token_ids(token_ids: object, *, field_name: str) -> tuple[int, ...]:
    """Return validated token ids as an immutable tuple.

    :param token_ids: Candidate token id sequence.
    :param field_name: Field name for error reporting.
    :returns: Tuple of non-negative integer token ids.
    :raises TypeError: If the candidate is not a sequence of integer ids.
    :raises ValueError: If the sequence is empty or contains a negative id.
    """

    if isinstance(token_ids, (str, bytes, Mapping)) or not isinstance(
        token_ids,
        Sequence,
    ):
        raise TypeError(f"{field_name} must be a sequence of target token ids")

    normalized = tuple(_normalize_token_id(token_id) for token_id in token_ids)
    if len(normalized) == 0:
        raise ValueError(f"{field_name} must contain at least one token id")

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate token ids")

    return normalized


def _normalize_multi_positive_token_weights(
    token_weights: object,
    *,
    token_count: int,
) -> tuple[float, ...] | None:
    """Return optional normalized multi-positive token weights.

    :param token_weights: Candidate optional token weights.
    :param token_count: Expected weight count.
    :returns: Immutable positive finite float weights, or ``None``.
    :raises TypeError: If weights are not a numeric sequence.
    :raises ValueError: If weights are non-finite, non-positive, or misaligned.
    """

    if token_weights is None:
        return None
    if isinstance(token_weights, (str, bytes, Mapping)) or not isinstance(
        token_weights,
        Sequence,
    ):
        raise TypeError("token_weights must be a sequence of numeric scalars")

    weights: list[float] = []
    for weight in token_weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("token_weights must contain numeric scalar weights")
        parsed = float(weight)
        if not math.isfinite(parsed):
            raise ValueError("token_weights must be finite")
        if parsed <= 0.0:
            raise ValueError("token_weights must be positive")
        weights.append(parsed)

    if len(weights) != token_count:
        raise ValueError("token_weights must align with token_ids")
    if sum(weights) <= 0.0:
        raise ValueError("token_weights must have positive total")

    return tuple(weights)


def _normalize_bbox(target_bbox: object) -> SemanticBBox:
    """Return a validated immutable bounding box target.

    :param target_bbox: Candidate bounding box target.
    :returns: Four-value floating point bounding box.
    :raises TypeError: If the bounding box is not a sequence of numbers.
    :raises ValueError: If the bounding box does not contain four finite values.
    """

    if isinstance(target_bbox, (str, bytes, Mapping)) or not isinstance(
        target_bbox,
        Sequence,
    ):
        raise TypeError("target bbox must be a four-value numeric sequence")
    if len(target_bbox) != 4:
        raise ValueError("target bbox must contain exactly four values")

    values: list[float] = []
    for value in target_bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("target bbox values must be finite numeric scalars")

        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise ValueError("target bbox values must be finite")
        values.append(coordinate)

    return (values[0], values[1], values[2], values[3])


@dataclass(frozen=True, slots=True)
class TargetDistribution:
    """Base class for semantic target distributions.

    Concrete subclasses encode the small registered family of supported target
    distributions. This base class is intentionally not directly constructable.

    :param kind: Explicit registered target distribution kind.
    """

    kind: DistributionKind = field(init=False)

    def __post_init__(self) -> None:
        """Reject direct base-class construction."""

        if type(self) is TargetDistribution:
            raise TypeError(
                "TargetDistribution is abstract; use a registered distribution class"
            )


@dataclass(frozen=True, slots=True)
class HardTokenDistribution(TargetDistribution):
    """Single hard-token target distribution.

    :param token_id: Target token id supervised at the label position.
    """

    token_id: int
    kind: Literal["hard_token"] = field(default="hard_token", init=False)

    def __post_init__(self) -> None:
        """Normalize the target token id."""

        object.__setattr__(self, "token_id", _normalize_token_id(self.token_id))


@dataclass(frozen=True, slots=True)
class MultiPositiveTokenDistribution(TargetDistribution):
    """Multi-positive hard-token target distribution.

    :param token_ids: Positive token ids accepted for the same label position.
    :param token_weights: Optional positive finite multiplicity weights.
    """

    token_ids: Sequence[int]
    token_weights: MultiPositiveWeightInput = None
    kind: Literal["multi_positive_token"] = field(
        default="multi_positive_token",
        init=False,
    )

    def __post_init__(self) -> None:
        """Normalize positive token ids and optional multiplicity weights."""

        token_ids = _normalize_token_ids(self.token_ids, field_name="token_ids")
        object.__setattr__(
            self,
            "token_ids",
            token_ids,
        )
        object.__setattr__(
            self,
            "token_weights",
            _normalize_multi_positive_token_weights(
                self.token_weights,
                token_count=len(token_ids),
            ),
        )


@dataclass(frozen=True, slots=True)
class CoordinateSoftTokenWeight:
    """Weighted coordinate-token target entry.

    :param token_id: Coordinate token id.
    :param weight: Non-negative finite target weight for the token id.
    """

    token_id: int
    weight: float

    def __post_init__(self) -> None:
        """Normalize token id and weight scalars."""

        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("coordinate soft token weights must be numeric scalars")

        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ValueError("coordinate soft token weights must be finite")
        if weight < 0.0:
            raise ValueError("coordinate soft token weights must be non-negative")

        object.__setattr__(self, "token_id", _normalize_token_id(self.token_id))
        object.__setattr__(self, "weight", weight)


CoordinateSoftTokenWeightInput: TypeAlias = (
    CoordinateSoftTokenWeight | Sequence[int | float]
)


def _normalize_coordinate_weight_entry(
    entry: object,
) -> CoordinateSoftTokenWeight:
    """Return a validated coordinate soft-token weight entry.

    :param entry: Existing entry object or two-value ``(token_id, weight)`` pair.
    :returns: Immutable coordinate soft-token weight entry.
    :raises TypeError: If the entry is not a valid pair.
    :raises ValueError: If the entry has the wrong arity or invalid values.
    """

    if type(entry) is CoordinateSoftTokenWeight:
        return entry
    if isinstance(entry, (str, bytes, Mapping)) or not isinstance(entry, Sequence):
        raise TypeError(
            "coordinate soft token weights must be entries or token-weight pairs"
        )
    if len(entry) != 2:
        raise ValueError(
            "coordinate soft token weight entries must contain exactly two values"
        )

    return CoordinateSoftTokenWeight(token_id=entry[0], weight=entry[1])


@dataclass(frozen=True, slots=True)
class CoordinateSoftTokenDistribution(TargetDistribution):
    """Soft coordinate-token target distribution.

    :param token_weights: Non-empty sequence of coordinate token-weight entries.
    :param target_family: Closed semantic recipe used to create target weights.
    :param loss_mode: Closed semantic objective mode intended for these weights.
    """

    token_weights: Sequence[CoordinateSoftTokenWeightInput]
    target_family: CoordinateSoftTargetFamily = "iou_gibbs_v0"
    loss_mode: CoordinateSoftLossMode = "full_vocab_ce"
    kind: Literal["coordinate_soft_token"] = field(
        default="coordinate_soft_token",
        init=False,
    )

    def __post_init__(self) -> None:
        """Normalize soft-token weights and validate positive mass."""

        if isinstance(self.token_weights, (str, bytes, Mapping)) or not isinstance(
            self.token_weights,
            Sequence,
        ):
            raise TypeError(
                "coordinate soft token weights must be a sequence of entries"
            )

        token_weights = tuple(
            _normalize_coordinate_weight_entry(entry) for entry in self.token_weights
        )
        if len(token_weights) == 0:
            raise ValueError(
                "coordinate soft token weights must contain at least one entry"
            )
        token_ids = tuple(entry.token_id for entry in token_weights)
        if len(set(token_ids)) != len(token_ids):
            raise ValueError(
                "coordinate soft token weights must not contain duplicate token ids"
            )

        total_weight = sum(entry.weight for entry in token_weights)
        if total_weight <= 0.0:
            raise ValueError("coordinate soft token weights must have positive total")

        object.__setattr__(self, "token_weights", token_weights)
        object.__setattr__(
            self,
            "target_family",
            _validate_coordinate_soft_target_family(self.target_family),
        )
        object.__setattr__(
            self,
            "loss_mode",
            _validate_coordinate_soft_loss_mode(self.loss_mode),
        )


@dataclass(frozen=True, slots=True)
class BoxRegressionDistribution(TargetDistribution):
    """Box-regression target distribution.

    :param target_bbox: Four finite semantic target coordinates.
    """

    target_bbox: SemanticBBoxInput
    kind: Literal["box_regression"] = field(default="box_regression", init=False)

    def __post_init__(self) -> None:
        """Normalize the box-regression target."""

        object.__setattr__(self, "target_bbox", _normalize_bbox(self.target_bbox))


class TargetDistributionRegistry:
    """Closed registry for supported semantic target distributions."""

    _REGISTERED: ClassVar[Mapping[str, type[TargetDistribution]]] = MappingProxyType(
        {
            "hard_token": HardTokenDistribution,
            "multi_positive_token": MultiPositiveTokenDistribution,
            "coordinate_soft_token": CoordinateSoftTokenDistribution,
            "box_regression": BoxRegressionDistribution,
        }
    )

    @classmethod
    def create(cls, *, kind: object, **kwargs: object) -> TargetDistribution:
        """Create a registered target distribution.

        :param kind: Supported distribution kind.
        :param kwargs: Constructor arguments for the concrete distribution.
        :returns: Concrete target distribution instance.
        :raises ValueError: If the kind is retired or unsupported.
        """

        validated_kind = validate_distribution_kind(kind)
        distribution_cls = cls._REGISTERED[validated_kind]

        return distribution_cls(**kwargs)

    @classmethod
    def register(
        cls,
        kind: object,
        distribution_cls: type[TargetDistribution],
    ) -> None:
        """Validate registration for the closed distribution family.

        The registry is intentionally static in this skeleton. This method
        exists only to make unsupported or retired registration attempts fail
        with the same actionable kind validation as construction.

        :param kind: Candidate distribution kind.
        :param distribution_cls: Candidate distribution class.
        :raises ValueError: If the kind is retired, unsupported, or dynamic.
        """

        validated_kind = validate_distribution_kind(kind)
        if distribution_cls is cls._REGISTERED[validated_kind]:
            return

        raise ValueError(
            "target distribution registry is closed; "
            f"{validated_kind!r} is already registered to "
            f"{cls._REGISTERED[validated_kind].__name__}"
        )

    @classmethod
    def registered_distribution_classes(cls) -> frozenset[type[TargetDistribution]]:
        """Return the concrete registered distribution classes.

        :returns: Immutable set of concrete registered distribution classes.
        """

        return frozenset(cls._REGISTERED.values())

    @classmethod
    def validate_objective_support(
        cls,
        distribution: TargetDistribution,
        objective_id: object,
    ) -> TargetObjectiveId:
        """Validate that a distribution is supported by a semantic objective.

        This compatibility check is intentionally metadata-only. It does not
        inspect logits, loss rows, encoded positions, tensors, or tokenizer
        state.

        :param distribution: Registered target distribution instance.
        :param objective_id: Closed target objective identifier.
        :returns: Validated objective identifier.
        :raises TypeError: If the distribution or objective identifier is not
            semantic contract data.
        :raises ValueError: If the pair is unsupported.
        """

        if type(distribution) not in cls.registered_distribution_classes():
            raise TypeError(
                "objective support validation requires a registered TargetDistribution"
            )

        validated_objective_id = validate_target_objective_id(objective_id)
        supported_objectives = SUPPORTED_OBJECTIVES_BY_DISTRIBUTION_KIND[
            distribution.kind
        ]
        if validated_objective_id not in supported_objectives:
            supported_list = ", ".join(sorted(supported_objectives))
            raise ValueError(
                "target distribution/objective pair is unsupported: "
                f"{distribution.kind!r} with {validated_objective_id!r}; "
                f"supported objective ids for {distribution.kind!r}: "
                f"{supported_list}"
            )

        return validated_objective_id
