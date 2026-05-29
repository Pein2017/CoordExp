from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.common.detection_sequence import (
    BOX_START_TOKEN,
    COMPACT_FULL_FORMAT,
    OBJECT_REF_START_TOKEN,
    parse_compact_detection_sequence,
)
from src.trainers.rollout_correction.rollout_views import build_rollout_correction_view
from src.training.stage2.rollout_codec import (
    CompactFullRolloutCodec,
    Stage2RolloutObject,
    Stage2RolloutParseResult,
    Stage2RolloutTemplateMismatchError,
    resolve_stage2_rollout_template_policy,
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


class _MiniTokenizer:
    eos_token_id = 999_999

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._pieces: dict[int, str] = {}

    def _id_for(self, piece: str) -> int:
        if piece not in self._ids:
            token_id = len(self._ids) + 10
            self._ids[piece] = token_id
            self._pieces[token_id] = piece
        return self._ids[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        out: list[int] = []
        i = 0
        specials = (OBJECT_REF_START_TOKEN, BOX_START_TOKEN, "<|im_end|>")
        while i < len(text):
            for special in specials:
                if text.startswith(special, i):
                    out.append(self._id_for(special))
                    i += len(special)
                    break
            else:
                if text.startswith("<|coord_", i):
                    end = text.find("|>", i)
                    if end >= 0:
                        piece = text[i : end + 2]
                        out.append(self._id_for(piece))
                        i = end + 2
                        continue
                out.append(self._id_for(text[i]))
                i += 1
        return out

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self._pieces[int(token_id)] for token_id in token_ids)


def test_compact_full_parse_and_append_round_trip_without_json_fallback() -> None:
    codec = CompactFullRolloutCodec()
    raw = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
    )

    parse_result = codec.parse(raw)
    target = codec.build_rollout_correction_target(
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
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
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


def test_compact_full_eos_padding_is_not_parse_truncation() -> None:
    codec = CompactFullRolloutCodec()
    raw = (
        f"{OBJECT_REF_START_TOKEN}chair{BOX_START_TOKEN}"
        "<|coord_453|><|coord_512|><|coord_546|><|coord_737|>"
        "<|im_end|><|endoftext|><|endoftext|><|endoftext|>"
    )

    parse_result = codec.parse(raw)

    assert parse_result.invalid_rollout is False
    assert parse_result.truncated is False
    assert [obj.desc for obj in parse_result.valid_objects] == ["chair"]
    assert parse_result.valid_objects[0].bbox_tokens == (
        "<|coord_453|>",
        "<|coord_512|>",
        "<|coord_546|>",
        "<|coord_737|>",
    )


def test_compact_full_rollout_codec_salvages_valid_rows_for_training() -> None:
    codec = CompactFullRolloutCodec()
    raw = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
        f"{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}"
        "bad<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )

    parse_result = codec.parse(raw)

    assert parse_result.invalid_rollout is False
    assert parse_result.empty_valid_object_set is False
    assert [obj.desc for obj in parse_result.valid_objects] == ["cat"]
    assert (
        parse_compact_detection_sequence(
            raw,
            detection_sequence_format=COMPACT_FULL_FORMAT,
        )
        is None
    )


def test_compact_full_rollout_view_drops_salvaged_span_mismatch_attempt() -> None:
    tok = _MiniTokenizer()
    raw = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
        f"{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}"
        "bad<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    token_ids = tok.encode(raw, add_special_tokens=False)

    view = build_rollout_correction_view(
        tokenizer=tok,
        object_field_order="desc_first",
        coord_id_to_bin={},
        duplicate_iou_threshold=0.5,
        center_radius_scale=0.5,
        max_new_tokens=256,
        rollout_result=(token_ids, raw, "unit", []),
        source_label="anchor",
        parse_rollout_for_matching_fn=None,
        points_from_coord_tokens_fn=None,
        duplicate_diagnostics_fn=lambda *_args, **_kwargs: {},
        rollout_template_policy=resolve_stage2_rollout_template_policy("compact_full"),
    )

    assert view["invalid_rollout"] == 1
    assert view["fallback_reason"] == "compact_full_span_extraction_failed"
    assert view["rollout_counts_as_valid_rollout"] == 0
    assert view["pred_objects"] == 1
    assert view["n_valid_pred"] == 0
    assert view["parsed_bbox_objects_raw"] == []
    assert view["compact_full_span_extraction_failed"] == 1
    assert view["compact_full_object_spans"] == []
    assert view["drop_reasons"]["compact_full_span_extraction_failed"] == 1


def test_compact_full_strict_preflight_rejects_salvaged_rows() -> None:
    tok = _MiniTokenizer()
    raw = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
        f"{OBJECT_REF_START_TOKEN}dog{BOX_START_TOKEN}"
        "bad<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    token_ids = tok.encode(raw, add_special_tokens=False)

    with pytest.raises(ValueError, match="strict rollout preflight.*dropped"):
        build_rollout_correction_view(
            tokenizer=tok,
            object_field_order="desc_first",
            coord_id_to_bin={},
            duplicate_iou_threshold=0.5,
            center_radius_scale=0.5,
            max_new_tokens=256,
            rollout_result=(token_ids, raw, "unit", []),
            source_label="anchor",
            parse_rollout_for_matching_fn=None,
            points_from_coord_tokens_fn=None,
            duplicate_diagnostics_fn=lambda *_args, **_kwargs: {},
            rollout_template_policy=resolve_stage2_rollout_template_policy(
                "compact_full",
                strict_rollout_preflight=True,
            ),
        )


def test_compact_full_strict_preflight_rejects_fallback_output() -> None:
    tok = _MiniTokenizer()
    raw = "not compact output"
    token_ids = tok.encode(raw, add_special_tokens=False)

    with pytest.raises(ValueError, match="strict rollout preflight.*fallback"):
        build_rollout_correction_view(
            tokenizer=tok,
            object_field_order="desc_first",
            coord_id_to_bin={},
            duplicate_iou_threshold=0.5,
            center_radius_scale=0.5,
            max_new_tokens=256,
            rollout_result=(token_ids, raw, "unit", []),
            source_label="anchor",
            parse_rollout_for_matching_fn=None,
            points_from_coord_tokens_fn=None,
            duplicate_diagnostics_fn=lambda *_args, **_kwargs: {},
            rollout_template_policy=resolve_stage2_rollout_template_policy(
                "compact_full",
                strict_rollout_preflight=True,
            ),
        )


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
    target = codec.build_rollout_correction_target(
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
        codec.build_rollout_correction_target(parse_result, fn_objects=())


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
        codec.build_rollout_correction_target(
            parse_result,
            fn_objects=(_gt(0, "dog", [10, 20, 30, 1000]),),
        )
