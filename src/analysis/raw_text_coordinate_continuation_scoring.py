"""Generic changed-chunk scoring helpers for raw-text continuation probes."""

from __future__ import annotations

from typing import Mapping, Sequence


def build_candidate_continuation_span(
    *,
    tokenizer: object,
    baseline_assistant_text: str,
    candidate_assistant_text: str,
) -> dict[str, object]:
    encode = getattr(tokenizer, "encode")
    baseline_ids = list(encode(baseline_assistant_text, add_special_tokens=False))
    candidate_ids = list(encode(candidate_assistant_text, add_special_tokens=False))
    prefix_len = 0
    for left_id, right_id in zip(baseline_ids, candidate_ids):
        if int(left_id) != int(right_id):
            break
        prefix_len += 1
    max_suffix = min(len(baseline_ids), len(candidate_ids)) - prefix_len
    suffix_len = 0
    while suffix_len < max_suffix:
        if int(baseline_ids[-(suffix_len + 1)]) != int(
            candidate_ids[-(suffix_len + 1)]
        ):
            break
        suffix_len += 1
    changed_stop = len(candidate_ids) - suffix_len
    assistant_relative_positions = list(range(prefix_len, changed_stop))
    if not assistant_relative_positions:
        raise ValueError("changed_continuation_span_empty")
    return {
        "candidate_assistant_text": candidate_assistant_text,
        "assistant_relative_positions": assistant_relative_positions,
    }


def score_candidate_continuations_batch(
    *,
    scorer: object,
    image: object,
    baseline_assistant_text: str,
    candidate_rows: Sequence[Mapping[str, object]],
    prompt_variant: str,
    object_field_order: str,
) -> list[dict[str, object]]:
    if not candidate_rows:
        return []
    prepared_examples: list[object] = []
    spans_list: list[list[int]] = []
    rows_out: list[dict[str, object]] = []
    tokenizer = getattr(scorer, "tokenizer")
    for row in candidate_rows:
        candidate = build_candidate_continuation_span(
            tokenizer=tokenizer,
            baseline_assistant_text=baseline_assistant_text,
            candidate_assistant_text=str(row["candidate_assistant_text"]),
        )
        prepared = scorer.prepare_example(
            image=image,
            assistant_text=str(candidate["candidate_assistant_text"]),
            desc_positions_rel=[],
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
        )
        assistant_start = int(getattr(prepared, "assistant_start"))
        absolute_positions = [
            assistant_start + int(pos)
            for pos in candidate["assistant_relative_positions"]
        ]
        prepared_examples.append(prepared)
        spans_list.append(absolute_positions)
        rows_out.append(
            {
                **dict(row),
                "assistant_relative_positions": list(
                    candidate["assistant_relative_positions"]
                ),
                "absolute_positions": absolute_positions,
            }
        )
    batch_score = getattr(scorer, "score_prepared_batch_spans", None)
    if callable(batch_score):
        score_rows = batch_score(
            examples=prepared_examples,
            images=[image] * len(prepared_examples),
            spans_list=spans_list,
        )
    else:
        score_rows = [
            scorer.score_prepared_spans(
                prepared=prepared,
                image=image,
                spans=[span],
            )[0]
            for prepared, span in zip(prepared_examples, spans_list)
        ]
    return [
        {
            **row,
            **score_row,
        }
        for row, score_row in zip(rows_out, score_rows)
    ]

