from __future__ import annotations

from dataclasses import dataclass
import re
from types import SimpleNamespace
import unittest

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.infer.parsing import (
    DetectionParserResult,
    diagnostic_parser_result,
    parse_stage2_detection_rollout_predictions,
    require_metric_bearing,
    strict_parser_result,
)
from src.trainers.rollout_matching.parsing import (
    parse_rollout_for_matching,
    points_from_coord_tokens,
)
from src.training.stage2.rollout_codec import (
    CompactFullRolloutCodec,
    resolve_stage2_rollout_template_policy,
)


def test_strict_parser_result_can_be_metric_bearing() -> None:
    result = strict_parser_result(
        predictions=({"desc": "cat", "bbox": [1, 2, 3, 4]},),
        parser_id="compact_full",
    )

    require_metric_bearing(result, consumer="official_eval")

    assert result.parser_policy == "strict"
    assert result.metric_bearing is True
    assert result.salvage_recovered is False
    assert result.predictions == ({"desc": "cat", "bbox": [1, 2, 3, 4]},)


def test_diagnostic_salvage_result_is_never_metric_bearing() -> None:
    result = diagnostic_parser_result(
        predictions=({"desc": "cat", "bbox": [1, 2, 3, 4]},),
        parser_id="coordjson_salvage",
        diagnostics={"reason": "recovered_from_malformed_json"},
    )

    assert result.parser_policy == "diagnostic"
    assert result.metric_bearing is False
    assert result.salvage_recovered is True

    with unittest.TestCase().assertRaisesRegex(ValueError, "metric_bearing=false"):
        require_metric_bearing(result, consumer="gt_vs_pred.jsonl")


def test_constructor_rejects_metric_bearing_diagnostic_result() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "diagnostic parser results cannot be metric-bearing",
    ):
        DetectionParserResult(
            predictions=(),
            parser_id="bad_salvage",
            parser_policy="diagnostic",
            metric_bearing=True,
            salvage_recovered=False,
        )


def test_constructor_rejects_salvage_recovered_metric_result() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "salvage-recovered parser results cannot be metric-bearing",
    ):
        DetectionParserResult(
            predictions=(),
            parser_id="bad_salvage",
            parser_policy="strict",
            metric_bearing=True,
            salvage_recovered=True,
        )


def test_constructor_rejects_non_boolean_metric_status() -> None:
    for bad_value in ("false", 1):
        with unittest.TestCase().assertRaisesRegex(
            TypeError,
            "metric_bearing must be a bool",
        ):
            DetectionParserResult(
                predictions=(),
                parser_id="bad_status",
                parser_policy="strict",
                metric_bearing=bad_value,
                salvage_recovered=False,
            )


def test_constructor_rejects_non_boolean_salvage_status() -> None:
    with unittest.TestCase().assertRaisesRegex(
        TypeError,
        "salvage_recovered must be a bool",
    ):
        DetectionParserResult(
            predictions=(),
            parser_id="bad_status",
            parser_policy="diagnostic",
            metric_bearing=False,
            salvage_recovered="false",
        )


def test_parser_policy_and_metric_status_are_artifact_visible() -> None:
    result = diagnostic_parser_result(
        predictions=(),
        parser_id="debug_salvage",
        errors=("malformed_output",),
        diagnostics={"offset": 12},
    )

    assert result.to_artifact_metadata() == {
        "parser_id": "debug_salvage",
        "parser_policy": "diagnostic",
        "metric_bearing": False,
        "salvage_recovered": True,
        "parser_error_count": 1,
    }


@dataclass(frozen=True)
class _FakeGTObject:
    index: int
    geom_type: str
    points_norm1000: list[int]
    desc: str


@dataclass(frozen=True)
class _FakeCompactObject:
    index: int
    geom_type: str
    bbox_norm1000: list[int] | None
    desc: str
    object_id: str


@dataclass(frozen=True)
class _FakeParse:
    valid_objects: tuple[object, ...]
    dropped_invalid: int = 0
    dropped_invalid_by_reason: dict[str, int] | None = None
    dropped_ambiguous: int = 0
    truncated: bool = False
    response_token_ids: tuple[int, ...] = ()
    empty_valid_object_set: bool = False
    fallback_reason: str | None = None


class _FakeCompactCodec:
    def __init__(self, _policy) -> None:
        pass

    def parse(self, _text: str) -> _FakeParse:
        return _FakeParse(
            valid_objects=(
                _FakeCompactObject(
                    index=0,
                    geom_type="bbox_2d",
                    bbox_norm1000=[1, 2, 10, 20],
                    desc="cat",
                    object_id="obj-0",
                ),
                _FakeCompactObject(
                    index=1,
                    geom_type="bbox_2d",
                    bbox_norm1000=[9, 9, 8, 8],
                    desc="bad",
                    object_id="obj-1",
                ),
            ),
            dropped_invalid_by_reason={},
        )


class _FakeTokenizer:
    def decode(self, ids, **_kwargs) -> str:
        return "decoded:" + ",".join(str(int(x)) for x in ids)


class _CoordLiteralTokenizer:
    """Token-aligned CoordJSON tokenizer stub using real coord token ids."""

    _coord_re = re.compile(r"<\|coord_(\d{1,4})\|>")

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self._next_id = 1000

    def convert_tokens_to_ids(self, tokens):
        out: list[int] = []
        for token in tokens:
            match = self._coord_re.fullmatch(str(token))
            out.append(int(match.group(1)) if match else -1)
        return out

    def encode(self, text: str, add_special_tokens: bool = False, **_kwargs):
        del add_special_tokens
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith("<|coord_", i):
                end = text.find("|>", i)
                if end < 0:
                    raise ValueError("unterminated coord token in test tokenizer input")
                token = text[i : end + 2]
                match = self._coord_re.fullmatch(token)
                if not match:
                    raise ValueError(f"bad coord token in test tokenizer input: {token}")
                ids.append(int(match.group(1)))
                i = end + 2
                continue
            char = text[i]
            token_id = self._char_to_id.get(char)
            if token_id is None:
                token_id = self._next_id
                self._next_id += 1
                self._char_to_id[char] = token_id
                self._id_to_piece[token_id] = char
            ids.append(token_id)
            i += 1
        return ids

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **_kwargs,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        pieces: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if 0 <= token_id <= 999:
                pieces.append(f"<|coord_{token_id}|>")
            else:
                pieces.append(self._id_to_piece.get(token_id, ""))
        return "".join(pieces)


def test_stage2_compact_full_parser_adapter_filters_bad_geometry() -> None:
    parsed = parse_stage2_detection_rollout_predictions(
        tokenizer=_FakeTokenizer(),
        response_token_ids=[11, 12, 13],
        response_text="",
        rollout_template_policy=SimpleNamespace(template_family="compact_full"),
        object_field_order="desc_first",
        coord_id_to_bin={},
        gt_object_factory=_FakeGTObject,
        compact_rollout_codec_factory=_FakeCompactCodec,
        parse_rollout_for_matching_fn=lambda **_kwargs: None,
        points_from_coord_tokens_fn=lambda **_kwargs: None,
    )

    assert parsed.parse.response_token_ids == (11, 12, 13)
    assert parsed.parse.dropped_invalid == 1
    assert parsed.parse.dropped_invalid_by_reason == {"bbox_invalid": 1}
    assert parsed.preds == (
        _FakeGTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[1, 2, 10, 20],
            desc="cat",
        ),
    )
    assert parsed.pred_objects_dump == (
        {
            "key": "obj-0",
            "index": 0,
            "geom_type": "bbox_2d",
            "points_norm1000": [1, 2, 10, 20],
            "desc": "cat",
        },
    )


def test_stage2_compact_full_parser_adapter_uses_real_codec_fixture() -> None:
    raw = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_10|><|coord_20|>"
    )

    parsed = parse_stage2_detection_rollout_predictions(
        tokenizer=_FakeTokenizer(),
        response_token_ids=[31, 32, 33],
        response_text=raw,
        rollout_template_policy=resolve_stage2_rollout_template_policy("compact_full"),
        object_field_order="desc_first",
        coord_id_to_bin={},
        gt_object_factory=_FakeGTObject,
        compact_rollout_codec_factory=CompactFullRolloutCodec,
        parse_rollout_for_matching_fn=lambda **_kwargs: None,
        points_from_coord_tokens_fn=lambda **_kwargs: None,
    )

    assert parsed.parse.parser_id == "compact_full"
    assert parsed.parse.response_token_ids == (31, 32, 33)
    assert parsed.parse.dropped_invalid == 0
    assert parsed.pred_meta == parsed.preds
    assert parsed.preds == (
        _FakeGTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[1, 2, 10, 20],
            desc="cat",
        ),
    )
    assert parsed.pred_objects_dump == (
        {
            "key": "rollout[0]",
            "index": 0,
            "geom_type": "bbox_2d",
            "points_norm1000": [1, 2, 10, 20],
            "desc": "cat",
        },
    )


def test_stage2_coord_token_parser_adapter_uses_injected_trainer_functions() -> None:
    fake_pred = SimpleNamespace(
        index=2,
        geom_type="bbox_2d",
        coord_token_indices=(0, 1, 2, 3),
        key="k2",
        desc="dog",
    )
    fake_parse = SimpleNamespace(
        valid_objects=(fake_pred,),
        response_token_ids=(101, 102, 103, 104),
    )
    calls: list[str] = []

    def _fake_parse_rollout_for_matching(**kwargs):
        calls.append("parse")
        assert kwargs["response_token_ids"] == (101, 102, 103, 104)
        assert kwargs["object_field_order"] == "geometry_first"
        return fake_parse

    def _fake_points_from_coord_tokens(**kwargs):
        calls.append("points")
        assert kwargs["coord_id_to_bin"] == {101: 1}
        return [4, 5, 14, 25]

    parsed = parse_stage2_detection_rollout_predictions(
        tokenizer=_FakeTokenizer(),
        response_token_ids=[101, 102, 103, 104],
        response_text="ignored",
        rollout_template_policy=SimpleNamespace(template_family="coordjson"),
        object_field_order="geometry_first",
        coord_id_to_bin={101: 1},
        gt_object_factory=_FakeGTObject,
        compact_rollout_codec_factory=_FakeCompactCodec,
        parse_rollout_for_matching_fn=_fake_parse_rollout_for_matching,
        points_from_coord_tokens_fn=_fake_points_from_coord_tokens,
    )

    assert calls == ["parse", "points"]
    assert parsed.parse is fake_parse
    assert parsed.pred_meta == (fake_pred,)
    assert parsed.preds == (
        _FakeGTObject(
            index=2,
            geom_type="bbox_2d",
            points_norm1000=[4, 5, 14, 25],
            desc="",
        ),
    )


def test_stage2_coord_token_parser_adapter_uses_real_parser_fixture() -> None:
    tokenizer = _CoordLiteralTokenizer()
    response_text = (
        '{"objects": [{"desc": "cat", "bbox_2d": ['
        "<|coord_1|>, <|coord_2|>, <|coord_10|>, <|coord_20|>"
        "]}]}"
    )
    response_token_ids = tokenizer.encode(response_text, add_special_tokens=False)

    parsed = parse_stage2_detection_rollout_predictions(
        tokenizer=tokenizer,
        response_token_ids=response_token_ids,
        response_text=response_text,
        rollout_template_policy=SimpleNamespace(template_family="coordjson"),
        object_field_order="desc_first",
        coord_id_to_bin={i: i for i in range(1000)},
        gt_object_factory=_FakeGTObject,
        compact_rollout_codec_factory=CompactFullRolloutCodec,
        parse_rollout_for_matching_fn=parse_rollout_for_matching,
        points_from_coord_tokens_fn=points_from_coord_tokens,
    )

    assert parsed.parse.invalid_rollout is False
    assert parsed.parse.response_text == response_text
    assert parsed.parse.prefix_text == response_text[:-2]
    assert [obj.desc for obj in parsed.pred_meta] == ["cat"]
    assert parsed.preds == (
        _FakeGTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[1, 2, 10, 20],
            desc="",
        ),
    )
    assert parsed.pred_objects_dump == (
        {
            "key": "objects[0]",
            "index": 0,
            "geom_type": "bbox_2d",
            "points_norm1000": [1, 2, 10, 20],
            "desc": "cat",
        },
    )
