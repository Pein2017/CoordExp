from dataclasses import replace

import pytest

from src.metrics.detection_sequence import parser_success_event
from src.metrics.events import (
    COORD_VOCAB_TOKEN_ACC_IDENTITY,
    FULL_VOCAB_COORD_TOKEN_ACC_ALIASES,
    FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY,
    FULL_VOCAB_COORD_TOKEN_ACC_TOP5_ALIASES,
    FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY,
    MetricAliasRegistry,
    MetricEvent,
    MetricIdentity,
    flatten_metric_events,
    ratio_event,
    reduce_metric_events,
    register_legacy_metric_aliases,
    weighted_mean_event,
)


def _ratio_event_from_identity(
    identity: MetricIdentity,
    *,
    numerator: float,
    denominator: float,
    aliases: tuple[str, ...] = (),
) -> MetricEvent:
    return ratio_event(
        identity.key,
        numerator,
        denominator,
        unit=identity.unit,
        aliases=aliases,
        semantic_role=identity.semantic_role,
        token_role=identity.token_role,
        vocab_scope=identity.vocab_scope,
        coordinate_surface=identity.coordinate_surface,
        geometry_type=identity.geometry_type,
        slot_name=identity.slot_name,
        object_scope=identity.object_scope,
        template_id=identity.template_id,
        parser_mode=identity.parser_mode,
        metric_surface=identity.metric_surface,
        diagnostic_only=identity.diagnostic_only,
    )


def test_ratio_metric_reduces_by_summed_numerator_denominator() -> None:
    events = [
        ratio_event("x", 1, 2, unit="token"),
        ratio_event("x", 1, 10, unit="token"),
    ]

    assert reduce_metric_events(events)["x"] == pytest.approx(2 / 12)


def test_weighted_mean_reduces_by_summed_weighted_values_and_weights() -> None:
    events = [
        weighted_mean_event("loss", 2.0, 1.0, unit="token"),
        weighted_mean_event("loss", 10.0, 3.0, unit="token"),
    ]

    assert reduce_metric_events(events)["loss"] == pytest.approx(32 / 4)


def test_zero_denominator_ratio_reduces_to_none() -> None:
    events = [ratio_event("empty", 0, 0, unit="token")]

    assert reduce_metric_events(events)["empty"] is None


def test_flatten_metric_events_omits_zero_denominator_none_values() -> None:
    events = [ratio_event("empty", 0, 0, unit="token")]

    assert flatten_metric_events(events) == {}


def test_flatten_metric_events_emits_legacy_alias_from_full_vocab_coord_identity() -> None:
    event = _ratio_event_from_identity(
        FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY,
        numerator=3,
        denominator=4,
    )

    flat = flatten_metric_events([event])

    assert flat[FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.key] == pytest.approx(3 / 4)
    assert flat["coord_token_acc"] == pytest.approx(3 / 4)


def test_flatten_metric_events_emits_top5_legacy_alias_from_full_vocab_coord_identity() -> None:
    event = _ratio_event_from_identity(
        FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY,
        numerator=4,
        denominator=5,
    )

    flat = flatten_metric_events([event])

    assert flat[FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.key] == pytest.approx(4 / 5)
    assert flat["coord_token_acc_top5"] == pytest.approx(4 / 5)


def test_coord_vocab_flattening_does_not_emit_legacy_coord_alias() -> None:
    event = _ratio_event_from_identity(
        COORD_VOCAB_TOKEN_ACC_IDENTITY,
        numerator=2,
        denominator=5,
    )

    flat = flatten_metric_events([event])

    assert flat[COORD_VOCAB_TOKEN_ACC_IDENTITY.key] == pytest.approx(2 / 5)
    assert "coord_token_acc" not in flat


def test_flatten_metric_events_rejects_reserved_legacy_alias_as_canonical_key() -> None:
    event = ratio_event(
        "coord_token_acc",
        1,
        2,
        unit="token",
        semantic_role="bbox_coord",
        token_role="coord",
        vocab_scope="coord_vocab",
        coordinate_surface="coord_token",
        geometry_type="bbox",
        metric_surface="training_logits",
    )

    with pytest.raises(ValueError, match="reserved as an alias.*coord_token_acc"):
        flatten_metric_events([event])


def test_flatten_metric_events_rejects_alias_collision_before_logging() -> None:
    full_vocab_event = _ratio_event_from_identity(
        FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY,
        numerator=1,
        denominator=2,
        aliases=("shared_alias",),
    )
    coord_vocab_event = _ratio_event_from_identity(
        COORD_VOCAB_TOKEN_ACC_IDENTITY,
        numerator=1,
        denominator=2,
        aliases=("shared_alias",),
    )

    with pytest.raises(ValueError, match="alias collision.*shared_alias"):
        flatten_metric_events([full_vocab_event, coord_vocab_event])


def test_diagnostic_parser_metric_flattening_does_not_emit_strict_aliases() -> None:
    event = parser_success_event(
        1,
        2,
        parser_mode="diagnostic_salvage",
        metric_surface="diagnostic_only",
        diagnostic_only=True,
    )

    flat = flatten_metric_events([event])

    assert flat[event.key] == pytest.approx(1 / 2)
    assert "parser_valid_rate" not in flat


def test_duplicate_metric_aliases_fail() -> None:
    registry = register_legacy_metric_aliases()
    different_metric_identity = replace(
        FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY,
        key="coord_vocab_token_acc/top1",
        vocab_scope="coord_vocab",
    )

    with pytest.raises(ValueError, match="alias collision.*coord_token_acc"):
        registry.register("coord_token_acc", different_metric_identity)


def test_legacy_coord_token_acc_alias_freezes_full_vocab_top1_identity() -> None:
    registry = register_legacy_metric_aliases()

    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.key == "coord_token_acc/full_vocab/top1"
    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.reducer == "ratio"
    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.unit == "token"
    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.semantic_role == "bbox_coord"
    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.token_role == "coord"
    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.vocab_scope == "full_vocab"
    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.coordinate_surface == "coord_token"
    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.geometry_type == "bbox"
    assert FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.metric_surface == "training_logits"
    assert FULL_VOCAB_COORD_TOKEN_ACC_ALIASES == ("coord_token_acc",)
    assert registry.resolve("coord_token_acc") == FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY


def test_legacy_coord_token_acc_top5_alias_freezes_full_vocab_top5_identity() -> None:
    registry = register_legacy_metric_aliases()

    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.key == (
        "coord_token_acc/full_vocab/top5"
    )
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.reducer == "ratio"
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.unit == "token"
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.semantic_role == "bbox_coord"
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.token_role == "coord"
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.vocab_scope == "full_vocab"
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.coordinate_surface == "coord_token"
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.geometry_type == "bbox"
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY.metric_surface == "training_logits"
    assert FULL_VOCAB_COORD_TOKEN_ACC_TOP5_ALIASES == ("coord_token_acc_top5",)
    assert (
        registry.resolve("coord_token_acc_top5")
        == FULL_VOCAB_COORD_TOKEN_ACC_TOP5_IDENTITY
    )


def test_coord_vocab_metric_has_distinct_key_and_cannot_reuse_legacy_alias() -> None:
    registry = register_legacy_metric_aliases()

    assert COORD_VOCAB_TOKEN_ACC_IDENTITY.key == "coord_vocab_token_acc/top1"
    assert COORD_VOCAB_TOKEN_ACC_IDENTITY.key != FULL_VOCAB_COORD_TOKEN_ACC_IDENTITY.key
    assert COORD_VOCAB_TOKEN_ACC_IDENTITY.vocab_scope == "coord_vocab"

    with pytest.raises(ValueError, match="alias collision.*coord_token_acc"):
        registry.register_aliases(
            COORD_VOCAB_TOKEN_ACC_IDENTITY,
            ("coord_token_acc",),
        )


def test_diagnostic_metrics_cannot_share_strict_aliases() -> None:
    registry = MetricAliasRegistry()
    strict_identity = MetricIdentity(
        key="parser_valid_rate/strict_expected",
        reducer="ratio",
        unit="sample",
        parser_mode="strict_expected",
        metric_surface="strict_expected_template",
        diagnostic_only=False,
    )
    diagnostic_identity = MetricIdentity(
        key="parser_valid_rate/diagnostic_salvage",
        reducer="ratio",
        unit="sample",
        parser_mode="diagnostic_salvage",
        metric_surface="diagnostic_only",
        diagnostic_only=True,
    )

    registry.register("parser_valid_rate", strict_identity)

    with pytest.raises(ValueError, match="alias collision.*parser_valid_rate"):
        registry.register("parser_valid_rate", diagnostic_identity)


def test_parser_mode_and_metric_surface_participate_in_metric_identity() -> None:
    strict_event = ratio_event(
        "parser_valid_rate/strict_expected",
        9,
        10,
        unit="sample",
        parser_mode="strict_expected",
        metric_surface="strict_expected_template",
    )
    diagnostic_event = ratio_event(
        "parser_valid_rate/diagnostic_auto_detect",
        9,
        10,
        unit="sample",
        parser_mode="diagnostic_auto_detect",
        metric_surface="diagnostic_only",
        diagnostic_only=True,
    )

    assert strict_event.identity != diagnostic_event.identity


    assert strict_event.identity.parser_mode == "strict_expected"
    assert diagnostic_event.identity.parser_mode == "diagnostic_auto_detect"
    assert strict_event.identity.metric_surface == "strict_expected_template"
    assert diagnostic_event.identity.metric_surface == "diagnostic_only"
