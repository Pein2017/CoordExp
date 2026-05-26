from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.training.stage2.rollout_codec import (
    LegacyCoordJsonRolloutCodec,
    Stage2RolloutObject,
    Stage2RolloutParseResult,
    Stage2RolloutTemplateMismatchError,
)
from src.utils.coordjson_transpiler import parse_coordjson


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


def test_coordjson_legacy_codec_parses_and_appends_coordjson_text() -> None:
    codec = LegacyCoordJsonRolloutCodec()
    raw = (
        '{"objects": [{"desc": "cat", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    )

    parse_result = codec.parse(raw)
    target = codec.build_rollout_correction_target(
        parse_result,
        fn_objects=(_gt(1, "dog", [10, 20, 30, 40]),),
    )

    assert parse_result.template_family == "coordjson"
    assert parse_result.parser_id == "coordjson_legacy"
    assert parse_result.invalid_rollout is False
    assert parse_result.append_prefix_text == (
        '{"objects": [{"desc": "cat", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}'
    )
    assert parse_result.valid_objects[0].desc == "cat"
    assert parse_result.valid_objects[0].bbox_norm1000 == (1, 2, 3, 4)

    assert target.text == (
        '{"objects": [{"desc": "cat", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, '
        '{"desc": "dog", "bbox_2d": '
        '[<|coord_10|>, <|coord_20|>, <|coord_30|>, <|coord_40|>]}]}'
    )
    assert OBJECT_REF_START_TOKEN not in target.text
    assert BOX_START_TOKEN not in target.text
    assert target.metadata.rollout_context == "legacy_coordjson_append"
    assert target.metadata.counts_as_valid_rollout is True

    reparsed = parse_coordjson(
        target.text,
        mode="salvage",
        object_field_order="desc_first",
    )
    assert reparsed.parse_failed is False
    assert [(record.desc, record.geometry_values) for record in reparsed.records] == [
        ("cat", [1, 2, 3, 4]),
        ("dog", [10, 20, 30, 40]),
    ]


def test_coordjson_legacy_codec_rejects_compact_full_marker_text() -> None:
    codec = LegacyCoordJsonRolloutCodec()
    raw = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    )

    with pytest.raises(Stage2RolloutTemplateMismatchError, match="coordjson"):
        codec.parse(raw)


def test_coordjson_legacy_render_target_preserves_multiple_accepted_objects() -> None:
    codec = LegacyCoordJsonRolloutCodec()

    target = codec.render_target(
        accepted_objects=(
            Stage2RolloutObject(
                object_id="objects[0]",
                index=0,
                desc="cat",
                bbox_norm1000=(1, 2, 3, 4),
            ),
            Stage2RolloutObject(
                object_id="objects[1]",
                index=1,
                desc="bird",
                bbox_norm1000=(5, 6, 7, 8),
            ),
        ),
        fn_objects=(_gt(2, "dog", [10, 20, 30, 40]),),
    )

    reparsed = parse_coordjson(
        target,
        mode="salvage",
        object_field_order="desc_first",
    )

    assert reparsed.parse_failed is False
    assert [(record.desc, record.geometry_values) for record in reparsed.records] == [
        ("cat", [1, 2, 3, 4]),
        ("bird", [5, 6, 7, 8]),
        ("dog", [10, 20, 30, 40]),
    ]


def test_coordjson_legacy_appender_requires_typed_prefix_for_valid_rollout() -> None:
    codec = LegacyCoordJsonRolloutCodec()
    parse_result = Stage2RolloutParseResult(
        template_family="coordjson",
        parser_id="coordjson_legacy",
        response_text="manual",
        valid_objects=(
            Stage2RolloutObject(
                object_id="objects[0]",
                index=0,
                desc="cat",
                bbox_norm1000=(1, 2, 3, 4),
            ),
        ),
        invalid_rollout=False,
        empty_valid_object_set=False,
        truncated=False,
    )

    with pytest.raises(ValueError, match="append prefix"):
        codec.build_rollout_correction_target(parse_result, fn_objects=())


@pytest.mark.parametrize(
    "bbox_norm1000",
    [
        (1, 2, 3),
        (1, 2, 3, 1000),
        (1, 2, 3, False),
        (1, 2, 3, 4.0),
    ],
)
def test_coordjson_legacy_rejects_malformed_accepted_bbox(
    bbox_norm1000: tuple[object, ...],
) -> None:
    codec = LegacyCoordJsonRolloutCodec()
    parse_result = Stage2RolloutParseResult(
        template_family="coordjson",
        parser_id="coordjson_legacy",
        response_text="manual",
        valid_objects=(
            Stage2RolloutObject(
                object_id="objects[0]",
                index=0,
                desc="cat",
                bbox_norm1000=bbox_norm1000,  # type: ignore[arg-type]
            ),
        ),
        invalid_rollout=False,
        empty_valid_object_set=False,
        truncated=False,
        append_prefix_text='{"objects": [{"desc": "cat", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}',
    )

    with pytest.raises(ValueError, match="norm1000 bbox"):
        codec.render_target(parse_result.valid_objects, fn_objects=())


@pytest.mark.parametrize(
    "points_norm1000",
    [
        [1, 2, 3, 1000],
        [1, 2, 3],
        [1, 2, 3, False],
        [1, 2, 3, 4.0],
    ],
)
def test_coordjson_legacy_rejects_malformed_fn_bbox(
    points_norm1000: list[object],
) -> None:
    codec = LegacyCoordJsonRolloutCodec()
    parse_result = Stage2RolloutParseResult(
        template_family="coordjson",
        parser_id="coordjson_legacy",
        response_text="manual",
        valid_objects=(
            Stage2RolloutObject(
                object_id="objects[0]",
                index=0,
                desc="cat",
                bbox_norm1000=(1, 2, 3, 4),
            ),
        ),
        invalid_rollout=False,
        empty_valid_object_set=False,
        truncated=False,
        append_prefix_text='{"objects": [{"desc": "cat", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}',
    )

    with pytest.raises(ValueError, match="norm1000 bbox"):
        codec.build_rollout_correction_target(
            parse_result,
            fn_objects=(_gt(1, "dog", points_norm1000),),
        )
