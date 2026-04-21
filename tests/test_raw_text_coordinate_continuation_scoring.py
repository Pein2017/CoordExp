from __future__ import annotations

from types import SimpleNamespace

from src.analysis.raw_text_coordinate_continuation_scoring import (
    build_candidate_continuation_span,
    score_candidate_continuations_batch,
)


def test_build_candidate_continuation_span_tracks_appended_object_tokens() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    baseline = '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}]}'
    candidate = (
        '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}, '
        '{"desc": "book", "bbox_2d": [5, 6, 7, 8]}]}'
    )

    span = build_candidate_continuation_span(
        tokenizer=_Tokenizer(),
        baseline_assistant_text=baseline,
        candidate_assistant_text=candidate,
    )
    candidate_ids = _Tokenizer().encode(span["candidate_assistant_text"])
    span_text = "".join(
        chr(candidate_ids[pos]) for pos in span["assistant_relative_positions"]
    )

    assert span["candidate_assistant_text"] == candidate
    assert span_text == ', {"desc": "book", "bbox_2d": [5, 6, 7, 8]}'


def test_score_candidate_continuations_batch_scores_changed_chunk_only() -> None:
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
            return SimpleNamespace(assistant_start=5, full_text=f"full::{assistant_text}")

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

    baseline = '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}]}'
    candidate_rows = [
        {
            "candidate_label": "gt_next",
            "candidate_assistant_text": (
                '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}, '
                '{"desc": "book", "bbox_2d": [5, 6, 7, 8]}]}'
            ),
        },
        {
            "candidate_label": "exact_duplicate",
            "candidate_assistant_text": (
                '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}, '
                '{"desc": "book", "bbox_2d": [9, 9, 9, 9]}]}'
            ),
        },
    ]
    scorer = _Scorer()
    image = object()

    scored = score_candidate_continuations_batch(
        scorer=scorer,
        image=image,
        baseline_assistant_text=baseline,
        candidate_rows=candidate_rows,
        prompt_variant="coco_80",
        object_field_order="desc_first",
    )

    assert [row["candidate_label"] for row in scored] == [
        "gt_next",
        "exact_duplicate",
    ]
    assert scorer.prepared_texts == [
        candidate_rows[0]["candidate_assistant_text"],
        candidate_rows[1]["candidate_assistant_text"],
    ]
    assert scorer.scored_batches == [
        {
            "full_texts": [
                f"full::{candidate_rows[0]['candidate_assistant_text']}",
                f"full::{candidate_rows[1]['candidate_assistant_text']}",
            ],
            "images": [image, image],
            "spans_list": [
                list(range(59, 102)),
                list(range(59, 102)),
            ],
        }
    ]
    assert [row["assistant_relative_positions"] for row in scored] == [
        list(range(54, 97)),
        list(range(54, 97)),
    ]
    assert [row["absolute_positions"] for row in scored] == [
        list(range(59, 102)),
        list(range(59, 102)),
    ]
    assert [row["count"] for row in scored] == [43, 43]
