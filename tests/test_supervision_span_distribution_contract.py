from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType

import pytest

from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import (
    BoxRegressionDistribution,
    CoordinateSoftTokenDistribution,
    CoordinateSoftTokenWeight,
    HardTokenDistribution,
    MultiPositiveTokenDistribution,
    TargetDistribution,
    TargetDistributionRegistry,
)
from src.training.supervision.spans import SupervisionSpan


class RawString(str):
    """String subclass carrying raw payload attributes for guard tests."""

    input_ids = (1, 2, 3)


def _retired_identifier(*parts: str) -> str:
    return "_".join(parts)


def test_accepted_distribution_kinds_construct_and_expose_explicit_kind() -> None:
    distributions = (
        TargetDistributionRegistry.create(kind="hard_token", token_id=11),
        TargetDistributionRegistry.create(
            kind="multi_positive_token",
            token_ids=[11, 12],
        ),
        TargetDistributionRegistry.create(
            kind="coordinate_soft_token",
            token_weights=[
                CoordinateSoftTokenWeight(token_id=20, weight=0.25),
                (21, 0.75),
            ],
        ),
        TargetDistributionRegistry.create(
            kind="box_regression",
            target_bbox=[1, 2, 3, 4],
        ),
    )

    assert [distribution.kind for distribution in distributions] == [
        "hard_token",
        "multi_positive_token",
        "coordinate_soft_token",
        "box_regression",
    ]
    assert all(type(distribution) is not TargetDistribution for distribution in distributions)
    assert all(not hasattr(distribution, "__dict__") for distribution in distributions)


def test_distribution_kind_rejects_string_subclasses() -> None:
    with pytest.raises(TypeError, match="target distribution kind"):
        TargetDistributionRegistry.create(kind=RawString("hard_token"), token_id=11)


@pytest.mark.parametrize(
    "kind",
    (
        _retired_identifier("duplicate", "negative"),
        _retired_identifier("adjacent", "repulsion"),
        _retired_identifier(
            "forced",
            "continuation",
        ),
    ),
)
def test_registry_rejects_retired_distribution_kinds(kind: str) -> None:
    with pytest.raises(ValueError, match="retired target distribution kind"):
        TargetDistributionRegistry.create(kind=kind)

    with pytest.raises(ValueError, match="retired target distribution kind"):
        TargetDistributionRegistry.register(kind, HardTokenDistribution)


def test_registry_rejects_unsupported_future_distribution_kind() -> None:
    with pytest.raises(ValueError, match="unsupported target distribution kind"):
        TargetDistributionRegistry.create(kind="unsupported_future_distribution")

    with pytest.raises(ValueError, match="unsupported target distribution kind"):
        TargetDistributionRegistry.register(
            "unsupported_future_distribution",
            HardTokenDistribution,
        )


def test_multi_positive_distribution_rejects_empty_and_normalizes_tokens() -> None:
    distribution = MultiPositiveTokenDistribution(token_ids=[3, 5])

    assert distribution.kind == "multi_positive_token"
    assert distribution.token_ids == (3, 5)
    assert distribution.token_weights is None
    assert isinstance(distribution.token_ids, tuple)

    with pytest.raises(ValueError, match="at least one token"):
        MultiPositiveTokenDistribution(token_ids=[])

    with pytest.raises(TypeError, match="token ids"):
        MultiPositiveTokenDistribution(token_ids=[1, "2"])


def test_multi_positive_distribution_normalizes_weighted_multiplicity() -> None:
    distribution = MultiPositiveTokenDistribution(
        token_ids=[3, 5],
        token_weights=[2, 1.5],
    )

    assert distribution.token_ids == (3, 5)
    assert distribution.token_weights == (2.0, 1.5)
    assert isinstance(distribution.token_weights, tuple)


@pytest.mark.parametrize(
    "token_weights,error_type,error_match",
    (
        ([True, 1.0], TypeError, "token_weights"),
        ([1.0, "2.0"], TypeError, "token_weights"),
        ([1.0, float("nan")], ValueError, "finite"),
        ([1.0, float("inf")], ValueError, "finite"),
        ([1.0, -0.5], ValueError, "positive"),
        ([1.0, 0.0], ValueError, "positive"),
        ([1.0], ValueError, "align"),
    ),
)
def test_multi_positive_distribution_rejects_invalid_token_weights(
    token_weights: object,
    error_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(error_type, match=error_match):
        MultiPositiveTokenDistribution(
            token_ids=[3, 5],
            token_weights=token_weights,
        )


def test_hard_token_distribution_rejects_bool_token_id() -> None:
    with pytest.raises(TypeError, match="target token ids"):
        HardTokenDistribution(token_id=True)


@pytest.mark.parametrize(
    "token_ids",
    (
        [3, 3],
        [3, 5, 3],
    ),
)
def test_multi_positive_distribution_rejects_duplicate_token_ids(
    token_ids: list[int],
) -> None:
    with pytest.raises(ValueError, match="duplicate token ids"):
        MultiPositiveTokenDistribution(token_ids=token_ids)


@pytest.mark.parametrize(
    "token_weights,error_type,error_match",
    (
        ([], ValueError, "at least one"),
        ([(1, -0.1)], ValueError, "non-negative"),
        ([(1, 0.0), (2, 0.0)], ValueError, "positive total"),
        ([(1, "0.5")], TypeError, "weights"),
    ),
)
def test_coordinate_soft_distribution_validates_weights(
    token_weights: object,
    error_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(error_type, match=error_match):
        CoordinateSoftTokenDistribution(token_weights=token_weights)


def test_coordinate_soft_distribution_normalizes_to_immutable_entries() -> None:
    mutable_weights = [(7, 0.25), CoordinateSoftTokenWeight(token_id=8, weight=0.75)]
    distribution = CoordinateSoftTokenDistribution(
        token_weights=mutable_weights,
        target_family="ciou_gibbs_v0",
        loss_mode="coord_vocab_ce",
    )
    mutable_weights.append((9, 0.5))

    assert distribution.kind == "coordinate_soft_token"
    assert distribution.target_family == "ciou_gibbs_v0"
    assert distribution.loss_mode == "coord_vocab_ce"
    assert distribution.token_weights == (
        CoordinateSoftTokenWeight(token_id=7, weight=0.25),
        CoordinateSoftTokenWeight(token_id=8, weight=0.75),
    )
    assert isinstance(distribution.token_weights, tuple)
    assert all(not hasattr(entry, "__dict__") for entry in distribution.token_weights)


@pytest.mark.parametrize("target_family", ("iou_gibbs_v0", "ciou_gibbs_v0"))
@pytest.mark.parametrize("loss_mode", ("full_vocab_ce", "coord_vocab_ce", "w1_distance"))
def test_coordinate_soft_distribution_accepts_closed_semantic_identity_fields(
    target_family: str,
    loss_mode: str,
) -> None:
    distribution = CoordinateSoftTokenDistribution(
        token_weights=[(7, 1.0)],
        target_family=target_family,
        loss_mode=loss_mode,
    )

    assert distribution.target_family == target_family
    assert distribution.loss_mode == loss_mode


@pytest.mark.parametrize(
    "kwargs,error_type,error_match",
    (
        (
            {"target_family": "gaussian_v0"},
            ValueError,
            "unsupported coordinate soft target_family",
        ),
        (
            {"target_family": RawString("iou_gibbs_v0")},
            TypeError,
            "coordinate soft target_family",
        ),
        (
            {"loss_mode": "kl_divergence"},
            ValueError,
            "unsupported coordinate soft loss_mode",
        ),
        (
            {"loss_mode": RawString("full_vocab_ce")},
            TypeError,
            "coordinate soft loss_mode",
        ),
    ),
)
def test_coordinate_soft_distribution_rejects_unsupported_semantic_identity_fields(
    kwargs: dict[str, object],
    error_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(error_type, match=error_match):
        CoordinateSoftTokenDistribution(token_weights=[(7, 1.0)], **kwargs)


def test_coordinate_soft_distribution_rejects_duplicate_token_ids_after_normalization() -> None:
    with pytest.raises(ValueError, match="duplicate token ids"):
        CoordinateSoftTokenDistribution(
            token_weights=[
                CoordinateSoftTokenWeight(token_id=7, weight=0.25),
                (7, 0.75),
            ],
        )


def test_box_regression_bbox_normalizes_four_finite_numbers() -> None:
    distribution = BoxRegressionDistribution(target_bbox=[1, 2, 3.5, 4])

    assert distribution.kind == "box_regression"
    assert distribution.target_bbox == (1.0, 2.0, 3.5, 4.0)
    assert isinstance(distribution.target_bbox, tuple)

    with pytest.raises(ValueError, match="exactly four"):
        BoxRegressionDistribution(target_bbox=(1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="finite"):
        BoxRegressionDistribution(target_bbox=(1.0, 2.0, 3.0, float("nan")))

    with pytest.raises(TypeError, match="numeric"):
        BoxRegressionDistribution(target_bbox=(1.0, 2.0, 3.0, True))


@pytest.mark.parametrize(
    "distribution,objective_id",
    (
        (HardTokenDistribution(token_id=11), "token_ce"),
        (MultiPositiveTokenDistribution(token_ids=[11, 12]), "trie_ce"),
        (
            CoordinateSoftTokenDistribution(token_weights=[(20, 1.0)]),
            "coord_soft_ce",
        ),
        (BoxRegressionDistribution(target_bbox=[1, 2, 3, 4]), "box_regression"),
    ),
)
def test_registry_validates_semantically_supported_distribution_objective_pairs(
    distribution: TargetDistribution,
    objective_id: str,
) -> None:
    assert (
        TargetDistributionRegistry.validate_objective_support(
            distribution,
            objective_id,
        )
        == objective_id
    )


@pytest.mark.parametrize(
    "distribution,objective_id",
    (
        (HardTokenDistribution(token_id=11), "trie_ce"),
        (MultiPositiveTokenDistribution(token_ids=[11, 12]), "token_ce"),
        (
            CoordinateSoftTokenDistribution(token_weights=[(20, 1.0)]),
            "box_regression",
        ),
        (BoxRegressionDistribution(target_bbox=[1, 2, 3, 4]), "coord_soft_ce"),
    ),
)
def test_registry_rejects_semantically_invalid_distribution_objective_pairs(
    distribution: TargetDistribution,
    objective_id: str,
) -> None:
    with pytest.raises(ValueError, match="distribution/objective pair"):
        TargetDistributionRegistry.validate_objective_support(
            distribution,
            objective_id,
        )


@pytest.mark.parametrize(
    "objective_id,error_type,error_match",
    (
        ("duplicate_burst_unlikelihood", ValueError, "retired target objective id"),
        ("unsupported_future_objective", ValueError, "unsupported target objective id"),
        (RawString("token_ce"), TypeError, "target objective id"),
    ),
)
def test_registry_rejects_retired_or_unsupported_objective_ids(
    objective_id: object,
    error_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(error_type, match=error_match):
        TargetDistributionRegistry.validate_objective_support(
            HardTokenDistribution(token_id=11),
            objective_id,
        )


def test_supervision_span_stores_label_positions_not_logit_rows() -> None:
    span = SupervisionSpan(
        sample_id="sample-1",
        context_id="ctx-1",
        role="coordinate",
        label_positions=[4, 5, 6, 7],
        distribution=BoxRegressionDistribution(target_bbox=(0.1, 0.2, 0.3, 0.4)),
        start=4,
        end=8,
        provenance="compact_full",
        metadata={"object_index": 2},
    )

    assert span.label_positions == (4, 5, 6, 7)
    assert span.role == "coordinate"
    assert span.start == 4
    assert span.end == 8
    assert span.metadata["object_index"] == 2
    assert isinstance(span.metadata, MappingProxyType)

    for forbidden_field in (
        "logit_positions",
        "logit_rows",
        "input_ids",
        "model_inputs",
        "tokenizer",
        "raw_config",
    ):
        assert not hasattr(span, forbidden_field)


@pytest.mark.parametrize(
    "kwargs,error_match",
    (
        ({"start": 4}, "both be present or both absent"),
        ({"end": 6}, "both be present or both absent"),
        ({"start": 4, "end": 4}, "greater than start"),
        ({"start": 5, "end": 4}, "greater than start"),
    ),
)
def test_supervision_span_rejects_invalid_start_end_combinations(
    kwargs: dict[str, int],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        SupervisionSpan(
            sample_id="sample-1",
            role="coordinate",
            label_positions=[4],
            distribution=HardTokenDistribution(token_id=77),
            **kwargs,
        )


@pytest.mark.parametrize(
    "label_positions,error_type,error_match",
    (
        ([], ValueError, "at least one label position"),
        ([-1], ValueError, "non-negative"),
        ([1.5], TypeError, "label positions"),
        ([True], TypeError, "label positions"),
    ),
)
def test_supervision_span_rejects_invalid_label_positions(
    label_positions: object,
    error_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(error_type, match=error_match):
        SupervisionSpan(
            sample_id="sample-1",
            role="schema",
            label_positions=label_positions,
            distribution=HardTokenDistribution(token_id=77),
        )


def test_supervision_span_rejects_non_distribution_payloads() -> None:
    with pytest.raises(TypeError, match="TargetDistribution"):
        SupervisionSpan(
            sample_id="sample-1",
            role="schema",
            label_positions=(1,),
            distribution={"kind": "hard_token", "token_id": 77},
        )


def test_supervision_span_rejects_distribution_subclasses() -> None:
    @dataclass(frozen=True, slots=True)
    class RawHardTokenDistribution(HardTokenDistribution):
        input_ids: tuple[int, ...] = (1, 2)

    with pytest.raises(TypeError, match="registered TargetDistribution"):
        SupervisionSpan(
            sample_id="sample-1",
            role="schema",
            label_positions=(1,),
            distribution=RawHardTokenDistribution(token_id=77),
        )


def test_supervision_span_copies_source_lists_and_rejects_forbidden_metadata() -> None:
    label_positions = [4, 5, 6, 7]
    metadata = {"source": "fixture", "score": 0.5}
    span = SupervisionSpan(
        sample_id="sample-1",
        role="coordinate",
        label_positions=label_positions,
        distribution=BoxRegressionDistribution(target_bbox=(0.1, 0.2, 0.3, 0.4)),
        metadata=metadata,
    )

    label_positions.append(8)
    metadata["source"] = "mutated"

    assert span.label_positions == (4, 5, 6, 7)
    assert span.metadata["source"] == "fixture"

    with pytest.raises(ValueError, match="metadata key"):
        SupervisionSpan(
            sample_id="sample-1",
            role="coordinate",
            label_positions=(4,),
            distribution=HardTokenDistribution(token_id=77),
            metadata={"input_ids.debug": "1,2,3"},
        )


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), -float("inf")))
def test_supervision_span_and_batch_reject_non_finite_float_metadata(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match="metadata float values must be finite"):
        SupervisionSpan(
            sample_id="sample-1",
            role="coordinate",
            label_positions=(4,),
            distribution=HardTokenDistribution(token_id=77),
            metadata={"score": bad_value},
        )

    with pytest.raises(ValueError, match="metadata float values must be finite"):
        SupervisionBatch(
            spans=(),
            metadata={"score": bad_value},
        )


def test_supervision_batch_is_frozen_slots_semantic_holder() -> None:
    span = SupervisionSpan(
        sample_id="sample-1",
        role="object_boundary",
        label_positions=(9,),
        distribution=HardTokenDistribution(token_id=77),
    )
    batch = SupervisionBatch(spans=[span], batch_id="batch-1", metadata={"limit": 1})

    assert batch.spans == (span,)
    assert batch.batch_id == "batch-1"
    assert batch.metadata["limit"] == 1
    assert isinstance(batch.metadata, MappingProxyType)
    assert not hasattr(batch, "__dict__")

    with pytest.raises(FrozenInstanceError):
        batch.batch_id = "other"

    for forbidden_field in (
        "input_ids",
        "model_inputs",
        "tokenizer",
        "raw_config",
        "tensor",
        "tensors",
    ):
        assert not hasattr(batch, forbidden_field)


def test_supervision_batch_copies_source_lists_and_rejects_forbidden_metadata() -> None:
    spans = [
        SupervisionSpan(
            sample_id="sample-1",
            role="object_boundary",
            label_positions=(9,),
            distribution=HardTokenDistribution(token_id=77),
        )
    ]
    batch = SupervisionBatch(spans=spans, metadata={"limit": 1})

    spans.append(
        SupervisionSpan(
            sample_id="sample-2",
            role="object_boundary",
            label_positions=(10,),
            distribution=HardTokenDistribution(token_id=78),
        )
    )

    assert len(batch.spans) == 1
    assert batch.spans[0].sample_id == "sample-1"

    with pytest.raises(ValueError, match="metadata key"):
        SupervisionBatch(spans=(), metadata={"model_inputs.debug": "bad"})


def test_supervision_batch_rejects_non_span_entries() -> None:
    with pytest.raises(TypeError, match="SupervisionSpan"):
        SupervisionBatch(
            spans=[
                {
                    "sample_id": "sample-1",
                    "label_positions": [1],
                },
            ],
        )


def test_supervision_batch_rejects_span_subclasses() -> None:
    @dataclass(frozen=True, slots=True)
    class RawSpan(SupervisionSpan):
        input_ids: tuple[int, ...] = (1, 2)

    with pytest.raises(TypeError, match="SupervisionSpan"):
        SupervisionBatch(
            spans=[
                RawSpan(
                    sample_id="sample-1",
                    role="schema",
                    label_positions=(1,),
                    distribution=HardTokenDistribution(token_id=77),
                ),
            ],
        )
