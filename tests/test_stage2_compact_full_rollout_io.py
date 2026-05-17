from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.common.detection_sequence import (
    BOX_START_TOKEN,
    COMPACT_FULL_FORMAT,
    OBJECT_REF_START_TOKEN,
    parse_compact_detection_sequence,
)
from src.training.stage2.rollout_codec import (
    CompactFullRolloutCodec,
    Stage2RolloutObject,
    Stage2RolloutParseResult,
    Stage2RolloutTemplateMismatchError,
)


@dataclass(frozen=True)
class _TestGTObject:
    index: int
    geom_type: str
    points_norm1000: list[object]
    desc: str


def _gt(index: int, desc: str, points: list[object]) -> _TestGTObject:
    return _TestGTObject(
        index=index,
        geom_type="bbox_2d",
        points_norm1000=points,
        desc=desc,
    )


def test_compact_full_parse_and_append_round_trip_without_json_fallback() -> None:
    codec = CompactFullRolloutCodec()
    raw = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    )

    parse_result = codec.parse(raw)
    target = codec.build_channel_b_target(
        parse_result,
        fn_objects=(_gt(1, "dog", [10, 20, 30, 40]),),
    )

    assert parse_result.template_family == "compact_full"
    assert parse_result.parser_id == "compact_full"
    assert parse_result.invalid_rollout is False
    assert parse_result.empty_valid_object_set is False
    assert [obj.desc for obj in parse_result.valid_objects] == ["cat"]
    assert parse_result.valid_objects[0].bbox_tokens == (
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    )

    assert target.text == (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>\n"
        f"{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    assert '"objects"' not in target.text
    assert target.metadata.rollout_context == "rollout_valid_with_fn_append"
    assert target.metadata.counts_as_valid_rollout is True

    reparsed = parse_compact_detection_sequence(
        target.text,
        detection_sequence_format=COMPACT_FULL_FORMAT,
    )
    assert reparsed == {
        "objects": [
            {
                "desc": "cat",
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ],
            },
            {
                "desc": "dog",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            },
        ]
    }


@pytest.mark.parametrize(
    ("raw", "fallback_reason"),
    [
        ("not compact output", "malformed_compact_full"),
        ("", "empty_valid_object_set"),
    ],
)
def test_compact_full_invalid_or_empty_output_falls_back_to_gt_fn_append_only(
    raw: str,
    fallback_reason: str,
) -> None:
    codec = CompactFullRolloutCodec()

    parse_result = codec.parse(raw)
    target = codec.build_channel_b_target(
        parse_result,
        fn_objects=(_gt(0, "fallback cat", [100, 200, 300, 400]),),
    )

    assert parse_result.valid_objects == ()
    assert parse_result.fallback_reason == fallback_reason
    assert target.text == (
        f"{OBJECT_REF_START_TOKEN}fallback cat{BOX_START_TOKEN}"
        "<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
    )
    assert target.metadata.rollout_context == "fallback_gt_fn_append_only"
    assert target.metadata.fallback_loss_weight == 1.0
    assert target.metadata.counts_as_valid_rollout is False
    assert target.metadata.fallback_reason == fallback_reason


def test_compact_full_rejects_coordjson_rollout_text_as_template_mismatch() -> None:
    codec = CompactFullRolloutCodec()
    raw = (
        '{"objects": [{"desc": "cat", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    )

    with pytest.raises(Stage2RolloutTemplateMismatchError, match="compact_full"):
        codec.parse(raw)


@pytest.mark.parametrize(
    "bbox_norm1000",
    [
        (1, 2, 3),
        (1, 2, 3, 1000),
        (1, 2, 3, True),
        (1, 2, 3, 4.0),
    ],
)
def test_compact_full_rejects_malformed_accepted_bbox(
    bbox_norm1000: tuple[object, ...],
) -> None:
    codec = CompactFullRolloutCodec()
    parse_result = Stage2RolloutParseResult(
        template_family="compact_full",
        parser_id="compact_full",
        response_text="manual",
        valid_objects=(
            Stage2RolloutObject(
                object_id="rollout[0]",
                index=0,
                desc="cat",
                bbox_norm1000=bbox_norm1000,  # type: ignore[arg-type]
            ),
        ),
        invalid_rollout=False,
        empty_valid_object_set=False,
        truncated=False,
    )

    with pytest.raises(ValueError, match="norm1000 bbox"):
        codec.build_channel_b_target(parse_result, fn_objects=())


def test_compact_full_rejects_out_of_range_fn_bbox() -> None:
    codec = CompactFullRolloutCodec()
    parse_result = Stage2RolloutParseResult(
        template_family="compact_full",
        parser_id="compact_full",
        response_text="manual",
        valid_objects=(),
        invalid_rollout=False,
        empty_valid_object_set=False,
        truncated=False,
    )

    with pytest.raises(ValueError, match="norm1000 bbox"):
        codec.build_channel_b_target(
            parse_result,
            fn_objects=(_gt(0, "dog", [10, 20, 30, 1000]),),
        )
