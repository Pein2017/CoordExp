from __future__ import annotations

from types import SimpleNamespace

import torch

from src.analysis.unmatched_proposal_verifier import PreparedExample
from src.analysis.raw_text_coord_continuity_scoring import (
    build_candidate_coordinate_span,
    build_candidate_coordinate_span_multi,
    lexical_features_for_candidate,
    replace_bbox_slot_value,
    replace_bbox_slot_values,
    score_candidate_coordinate_xy_grid_batch,
    score_candidate_coordinate_sequence,
    score_candidate_coordinate_sequences_batch,
    score_span_logprobs,
)


def test_score_span_logprobs_supports_multi_token_chunk() -> None:
    logits = torch.full((1, 5, 16), -20.0)
    input_ids = torch.tensor([[1, 3, 4, 5, 2]])
    logits[0, 0, 3] = 5.0
    logits[0, 1, 4] = 4.0
    logits[0, 2, 5] = 3.0

    result = score_span_logprobs(
        logits=logits,
        input_ids=input_ids,
        batch_idx=0,
        positions=[1, 2, 3],
    )

    assert result["count"] == 3
    assert result["mean_logprob"] > -0.1
    assert result["sum_logprob"] > -0.3


def test_score_span_logprobs_rejects_nonpositive_positions() -> None:
    logits = torch.zeros((1, 2, 4))
    input_ids = torch.tensor([[0, 1]])

    try:
        score_span_logprobs(
            logits=logits,
            input_ids=input_ids,
            batch_idx=0,
            positions=[0],
        )
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected ValueError for position 0")


def test_prepared_example_accepts_legacy_four_field_shape() -> None:
    prepared = PreparedExample(
        full_text="demo",
        assistant_text="demo",
        desc_positions=[1],
        full_input_ids=[1, 2, 3],
    )

    assert prepared.assistant_start == 0
    assert prepared.assistant_input_ids == []


def test_replace_bbox_slot_value_preserves_json_boundaries() -> None:
    assistant_text = '[{"desc":"book","bbox_2d":[199,200,210,250]}]'
    replaced = replace_bbox_slot_value(
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(199, 200, 210, 250),
        candidate_value=231,
    )

    assert replaced == '[{"desc":"book","bbox_2d":[231,200,210,250]}]'


def test_replace_bbox_slot_value_can_target_later_duplicate_bbox() -> None:
    assistant_text = '[{"desc":"a","bbox_2d":[1,2,3,4]},{"desc":"b","bbox_2d":[1,2,3,4]}]'
    replaced = replace_bbox_slot_value(
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(1, 2, 3, 4),
        candidate_value=9,
        object_index=1,
    )

    assert replaced == '[{"desc":"a","bbox_2d":[1,2,3,4]},{"desc":"b","bbox_2d":[9,2,3,4]}]'


def test_replace_bbox_slot_value_supports_coordjson_top_level_object() -> None:
    assistant_text = '{"objects": [{"desc": "book", "bbox_2d": [199, 200, 210, 250]}]}'
    replaced = replace_bbox_slot_value(
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(199, 200, 210, 250),
        candidate_value=231,
    )

    assert (
        replaced
        == '{"objects": [{"desc": "book", "bbox_2d": [231, 200, 210, 250]}]}'
    )


def test_replace_bbox_slot_value_supports_fenced_json_blocks() -> None:
    assistant_text = (
        '```json\n{"objects": [{"desc": "book", "bbox_2d": [199, 200, 210, 250]}]}\n```'
    )
    replaced = replace_bbox_slot_value(
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(199, 200, 210, 250),
        candidate_value=231,
    )

    assert (
        replaced
        == '```json\n{"objects": [{"desc": "book", "bbox_2d": [231, 200, 210, 250]}]}\n```'
    )


def test_replace_bbox_slot_values_updates_multiple_slots() -> None:
    assistant_text = '{"objects": [{"desc": "book", "bbox_2d": [199, 200, 210, 250]}]}'
    replaced = replace_bbox_slot_values(
        assistant_text=assistant_text,
        original_bbox=(199, 200, 210, 250),
        candidate_values_by_slot={"x1": 231, "y1": 205},
    )

    assert (
        replaced
        == '{"objects": [{"desc": "book", "bbox_2d": [231, 205, 210, 250]}]}'
    )


def test_lexical_features_capture_numeric_and_token_shape() -> None:
    features = lexical_features_for_candidate(
        candidate_value=210,
        center_value=199,
        gt_value=200,
        tokenizer_tokens=["2", "1", "0"],
        center_tokens=["1", "9", "9"],
    )

    assert features["numeric_distance_to_center"] == 11
    assert features["numeric_distance_to_gt"] == 10
    assert features["digit_length_match"] == 1
    assert features["token_count"] == 3


def test_lexical_features_use_true_character_edit_distance() -> None:
    features = lexical_features_for_candidate(
        candidate_value=121,
        center_value=212,
        gt_value=121,
        tokenizer_tokens=["1", "2", "1"],
        center_tokens=["2", "1", "2"],
    )

    assert features["char_edit_distance"] == 2


def test_build_candidate_coordinate_span_tracks_replaced_number_tokens() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    assistant_text = '[{"desc":"book","bbox_2d":[99,200,210,250]}]'
    candidate = build_candidate_coordinate_span(
        tokenizer=_Tokenizer(),
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(99, 200, 210, 250),
        candidate_value=100,
    )

    candidate_ids = _Tokenizer().encode(candidate["candidate_assistant_text"])
    span_text = "".join(chr(candidate_ids[pos]) for pos in candidate["assistant_relative_positions"])

    assert candidate["candidate_assistant_text"] == '[{"desc":"book","bbox_2d":[100,200,210,250]}]'
    assert span_text == "100"


def test_score_candidate_coordinate_sequence_uses_prepared_span_scoring() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    class _Scorer:
        tokenizer = _Tokenizer()

        def __init__(self) -> None:
            self.prepared_texts: list[str] = []
            self.scored_spans: list[list[list[int]]] = []

        def prepare_example(
            self,
            *,
            image: object,
            assistant_text: str,
            desc_positions_rel: list[int],
            prompt_variant: str,
            object_field_order: str,
        ) -> SimpleNamespace:
            del image, desc_positions_rel, prompt_variant, object_field_order
            self.prepared_texts.append(assistant_text)
            return SimpleNamespace(assistant_start=7, full_text="unused")

        def score_prepared_spans(
            self,
            *,
            prepared: SimpleNamespace,
            image: object,
            spans: list[list[int]],
        ) -> list[dict[str, float | int]]:
            del prepared, image
            self.scored_spans.append(spans)
            return [{"count": len(spans[0]), "sum_logprob": -1.5, "mean_logprob": -0.5}]

    scorer = _Scorer()
    result = score_candidate_coordinate_sequence(
        scorer=scorer,
        image=object(),
        assistant_text='[{"desc":"book","bbox_2d":[99,200,210,250]}]',
        slot="x1",
        original_bbox=(99, 200, 210, 250),
        candidate_value=100,
        prompt_variant="plain",
        object_field_order="bbox_then_desc",
    )

    assert scorer.prepared_texts == ['[{"desc":"book","bbox_2d":[100,200,210,250]}]']
    assert scorer.scored_spans == [[[34, 35, 36]]]
    assert result["count"] == 3
    assert result["candidate_value"] == 100
    assert result["assistant_relative_positions"] == [27, 28, 29]


def test_score_candidate_coordinate_sequences_batch_batches_span_scoring() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    class _Scorer:
        tokenizer = _Tokenizer()

        def __init__(self) -> None:
            self.prepared_texts: list[str] = []
            self.scored_batches: list[dict[str, object]] = []

        def prepare_example(
            self,
            *,
            image: object,
            assistant_text: str,
            desc_positions_rel: list[int],
            prompt_variant: str,
            object_field_order: str,
        ) -> SimpleNamespace:
            del image, desc_positions_rel, prompt_variant, object_field_order
            self.prepared_texts.append(assistant_text)
            return SimpleNamespace(assistant_start=7, full_text=f"full::{assistant_text}")

        def score_prepared_batch_spans(
            self,
            *,
            examples: list[SimpleNamespace],
            images: list[object],
            spans_list: list[list[int]],
        ) -> list[dict[str, float | int]]:
            self.scored_batches.append(
                {
                    "full_texts": [example.full_text for example in examples],
                    "images": images,
                    "spans_list": spans_list,
                }
            )
            return [
                {
                    "count": len(span),
                    "sum_logprob": -float(index + 1),
                    "mean_logprob": -float(index + 1) / float(len(span)),
                }
                for index, span in enumerate(spans_list)
            ]

    scorer = _Scorer()
    image = object()

    results = score_candidate_coordinate_sequences_batch(
        scorer=scorer,
        image=image,
        assistant_text='[{"desc":"book","bbox_2d":[99,200,210,250]}]',
        slot="x1",
        original_bbox=(99, 200, 210, 250),
        candidate_values=[100, 99],
        prompt_variant="plain",
        object_field_order="bbox_then_desc",
    )

    assert scorer.prepared_texts == [
        '[{"desc":"book","bbox_2d":[100,200,210,250]}]',
        '[{"desc":"book","bbox_2d":[99,200,210,250]}]',
    ]
    assert scorer.scored_batches == [
        {
            "full_texts": [
                'full::[{"desc":"book","bbox_2d":[100,200,210,250]}]',
                'full::[{"desc":"book","bbox_2d":[99,200,210,250]}]',
            ],
            "images": [image, image],
            "spans_list": [[34, 35, 36], [34, 35]],
        }
    ]
    assert [result["candidate_value"] for result in results] == [100, 99]
    assert [result["count"] for result in results] == [3, 2]
    assert results[0]["assistant_relative_positions"] == [27, 28, 29]
    assert results[1]["assistant_relative_positions"] == [27, 28]


def test_build_candidate_coordinate_span_scores_noop_replacement_span() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    candidate = build_candidate_coordinate_span(
        tokenizer=_Tokenizer(),
        assistant_text='[{"desc":"book","bbox_2d":[99,200,210,250]}]',
        slot="x1",
        original_bbox=(99, 200, 210, 250),
        candidate_value=99,
    )

    candidate_ids = _Tokenizer().encode(candidate["candidate_assistant_text"])
    span_text = "".join(chr(candidate_ids[pos]) for pos in candidate["assistant_relative_positions"])

    assert candidate["candidate_assistant_text"] == '[{"desc":"book","bbox_2d":[99,200,210,250]}]'
    assert span_text == "99"


def test_build_candidate_coordinate_span_preserves_pretty_printed_json() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    assistant_text = '[ {"desc": "book", "bbox_2d": [99, 200, 210, 250]} ]'
    candidate = build_candidate_coordinate_span(
        tokenizer=_Tokenizer(),
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(99, 200, 210, 250),
        candidate_value=100,
    )

    candidate_ids = _Tokenizer().encode(candidate["candidate_assistant_text"])
    span_text = "".join(chr(candidate_ids[pos]) for pos in candidate["assistant_relative_positions"])

    assert candidate["candidate_assistant_text"] == '[ {"desc": "book", "bbox_2d": [100, 200, 210, 250]} ]'
    assert span_text == "100"


def test_build_candidate_coordinate_span_scores_pretty_printed_noop() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    candidate = build_candidate_coordinate_span(
        tokenizer=_Tokenizer(),
        assistant_text='[ {"desc": "book", "bbox_2d": [99, 200, 210, 250]} ]',
        slot="x1",
        original_bbox=(99, 200, 210, 250),
        candidate_value=99,
    )

    candidate_ids = _Tokenizer().encode(candidate["candidate_assistant_text"])
    span_text = "".join(chr(candidate_ids[pos]) for pos in candidate["assistant_relative_positions"])

    assert span_text == "99"


def test_build_candidate_coordinate_span_multi_tracks_joint_xy_replacement() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    candidate = build_candidate_coordinate_span_multi(
        tokenizer=_Tokenizer(),
        assistant_text='[{"desc":"book","bbox_2d":[99,200,210,250]}]',
        original_bbox=(99, 200, 210, 250),
        candidate_values_by_slot={"x1": 100, "y1": 201},
    )

    candidate_ids = _Tokenizer().encode(candidate["candidate_assistant_text"])
    span_text = "".join(
        chr(candidate_ids[pos]) for pos in candidate["assistant_relative_positions"]
    )

    assert candidate["candidate_assistant_text"] == '[{"desc":"book","bbox_2d":[100,201,210,250]}]'
    assert span_text == "100,201"


def test_score_candidate_coordinate_xy_grid_batch_batches_joint_spans() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    class _Scorer:
        tokenizer = _Tokenizer()

        def __init__(self) -> None:
            self.prepared_texts: list[str] = []
            self.scored_batches: list[dict[str, object]] = []

        def prepare_example(
            self,
            *,
            image: object,
            assistant_text: str,
            desc_positions_rel: list[int],
            prompt_variant: str,
            object_field_order: str,
        ) -> SimpleNamespace:
            del image, desc_positions_rel, prompt_variant, object_field_order
            self.prepared_texts.append(assistant_text)
            return SimpleNamespace(assistant_start=7, full_text=f"full::{assistant_text}")

        def score_prepared_batch_spans(
            self,
            *,
            examples: list[SimpleNamespace],
            images: list[object],
            spans_list: list[list[int]],
        ) -> list[dict[str, float | int]]:
            self.scored_batches.append(
                {
                    "full_texts": [example.full_text for example in examples],
                    "images": images,
                    "spans_list": spans_list,
                }
            )
            return [
                {
                    "count": len(span),
                    "sum_logprob": -float(index + 1),
                    "mean_logprob": -float(index + 1) / float(len(span)),
                }
                for index, span in enumerate(spans_list)
            ]

    scorer = _Scorer()
    image = object()

    results = score_candidate_coordinate_xy_grid_batch(
        scorer=scorer,
        image=image,
        assistant_text='[{"desc":"book","bbox_2d":[99,200,210,250]}]',
        original_bbox=(99, 200, 210, 250),
        candidate_xy_pairs=[(100, 201), (99, 200)],
        prompt_variant="plain",
        object_field_order="bbox_then_desc",
    )

    assert scorer.prepared_texts == [
        '[{"desc":"book","bbox_2d":[100,201,210,250]}]',
        '[{"desc":"book","bbox_2d":[99,200,210,250]}]',
    ]
    assert scorer.scored_batches == [
        {
            "full_texts": [
                'full::[{"desc":"book","bbox_2d":[100,201,210,250]}]',
                'full::[{"desc":"book","bbox_2d":[99,200,210,250]}]',
            ],
            "images": [image, image],
            "spans_list": [[34, 35, 36, 37, 38, 39, 40], [34, 35, 36, 37, 38, 39]],
        }
    ]
    assert [(row["candidate_x1"], row["candidate_y1"]) for row in results] == [
        (100, 201),
        (99, 200),
    ]


def test_score_candidate_coordinate_xy_grid_batch_supports_chunked_batches() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    class _Scorer:
        tokenizer = _Tokenizer()

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def prepare_example(
            self,
            *,
            image: object,
            assistant_text: str,
            desc_positions_rel: list[int],
            prompt_variant: str,
            object_field_order: str,
        ) -> SimpleNamespace:
            del image, assistant_text, desc_positions_rel, prompt_variant, object_field_order
            return SimpleNamespace(assistant_start=0, full_text="unused")

        def score_prepared_batch_spans(
            self,
            *,
            examples: list[SimpleNamespace],
            images: list[object],
            spans_list: list[list[int]],
        ) -> list[dict[str, float | int]]:
            del images
            self.batch_sizes.append(len(examples))
            return [
                {"count": len(span), "sum_logprob": -1.0, "mean_logprob": -0.5}
                for span in spans_list
            ]

    scorer = _Scorer()

    results = score_candidate_coordinate_xy_grid_batch(
        scorer=scorer,
        image=object(),
        assistant_text='[{"desc":"book","bbox_2d":[99,200,210,250]}]',
        original_bbox=(99, 200, 210, 250),
        candidate_xy_pairs=[(100, 201), (101, 202), (102, 203)],
        prompt_variant="plain",
        object_field_order="bbox_then_desc",
        batch_size=2,
    )

    assert len(results) == 3
    assert scorer.batch_sizes == [2, 1]


def test_build_candidate_coordinate_span_multi_fallback_uses_candidate_bbox() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [1 for _ in text]

    result = build_candidate_coordinate_span_multi(
        tokenizer=_Tokenizer(),
        assistant_text='{"objects": [{"desc": "person", "bbox_2d": [620, 10, 950, 635]}]}',
        original_bbox=(620, 10, 950, 635),
        candidate_values_by_slot={"x1": 600, "y1": 12},
        object_index=0,
    )

    assert result["candidate_assistant_text"] == (
        '{"objects": [{"desc": "person", "bbox_2d": [600, 12, 950, 635]}]}'
    )
    assert result["assistant_relative_positions"]
