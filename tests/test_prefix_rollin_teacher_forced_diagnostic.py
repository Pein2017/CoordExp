from __future__ import annotations

import re

import pytest
import torch

from src.detection.data import ObjectOrderingPlan, normalize_detection_row
from src.detection.objective import build_compact_prefix_rollin_example
from src.analysis.prefix_rollin_teacher_forced_diagnostic import (
    compute_prefix_rollin_position_delta,
    score_prefix_rollin_logits,
    summarize_prefix_rollin_probe_rows,
)

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class FakeTokenizer:
    eos_token = "<|endoftext|>"
    unk_token_id = 0

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            "<|object_ref_start|>": 4,
            "<|box_start|>": 5,
        }
        self.eos_token_id = self._token_to_id[self.eos_token]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @property
    def special_tokens_map(self) -> dict[str, str]:
        return {"im_start": "<|im_start|>", "im_end": "<|im_end|>"}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(
            self(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )["input_ids"]
        )

    def decode(self, token_ids: list[int]) -> str:
        reverse = {value: key for key, value in self._token_to_id.items()}
        return "".join(reverse.get(int(token_id), f"<token:{int(token_id)}>") for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert add_generation_prompt is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n"
            f"{message['content']}<|im_end|>\n"
            for message in messages
        )
        if not tokenize:
            return rendered
        return self(rendered, return_offsets_mapping=True, add_special_tokens=False)[
            "input_ids"
        ]

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            match = _SPECIAL_TOKEN_RE.match(text, cursor)
            if match is not None:
                token_text = match.group(0)
                token_end = match.end()
            else:
                token_text = text[cursor]
                token_end = cursor + 1
            token_id = self._token_to_id.setdefault(token_text, len(self._token_to_id) + 1)
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end
        return {"input_ids": input_ids, "offset_mapping": offsets}


def _raw_row() -> dict[str, object]:
    return {
        "images": ["images/train2017/example.jpg"],
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
                "desc": "cat",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": 101,
            },
            {
                "bbox_2d": [
                    "<|coord_50|>",
                    "<|coord_60|>",
                    "<|coord_70|>",
                    "<|coord_80|>",
                ],
                "desc": "dog",
                "category_id": 18,
                "category_name": "dog",
                "coco_ann_id": 102,
            },
            {
                "bbox_2d": [
                    "<|coord_90|>",
                    "<|coord_100|>",
                    "<|coord_110|>",
                    "<|coord_120|>",
                ],
                "desc": "bus",
                "category_id": 6,
                "category_name": "bus",
                "coco_ann_id": 103,
            },
        ],
        "width": 640,
        "height": 480,
        "image_id": 9,
        "file_name": "images/train2017/example.jpg",
        "metadata": {"source": "unit", "split": "train"},
    }


def _example(*, k: int, object_count: int = 3):
    raw = _raw_row()
    raw["objects"] = raw["objects"][:object_count]
    sample = normalize_detection_row(
        raw=parse_raw_row_for_test(raw),
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=123,
            seed_source="unit",
        ),
    )
    return build_compact_prefix_rollin_example(
        objects=sample.objects,
        rollin_order=sample.objects,
        k=k,
        tokenizer=FakeTokenizer(),
        normalized_sample=sample,
        system_prompt="You are a detector.",
        user_content="<image>\nDetect every object.",
    )


def parse_raw_row_for_test(raw):
    from src.detection.data import parse_raw_detection_row

    return parse_raw_detection_row(raw)


def _branch_row(rows):
    return next(row for row in rows if row["branch_kind"] == "valid_next_object")


def _eos_row(rows):
    return next(row for row in rows if row["branch_kind"] == "semantic_eos")


def test_branch_probe_uses_next_token_shift_and_valid_mass_margin() -> None:
    example = _example(k=0, object_count=3)
    branch_target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id in {str(item) for item in example.rollin_state.remaining}
        and target.valid_token_ids
    )
    eos_id = example.stop_contract.im_end_token_id
    vocab_size = max((*branch_target.valid_token_ids, eos_id, *example.input_ids)) + 1
    logits = torch.zeros((1, len(example.input_ids), vocab_size), dtype=torch.float32)
    prediction_row = branch_target.position - 1
    for token_id in branch_target.valid_token_ids:
        logits[0, prediction_row, token_id] = 4.0
    logits[0, prediction_row, eos_id] = 1.5

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=example.input_ids,
    )
    row = _branch_row(rows)

    log_probs = torch.log_softmax(logits[0, prediction_row], dim=-1)
    expected_valid_mass = torch.logsumexp(
        log_probs[torch.tensor(branch_target.valid_token_ids)],
        dim=-1,
    )
    expected_margin = expected_valid_mass - log_probs[eos_id]
    expected_logit_margin = (
        logits[0, prediction_row, list(branch_target.valid_token_ids)].max()
        - logits[0, prediction_row, eos_id]
    )
    assert row["target_position"] == branch_target.position
    assert row["processor_position"] == branch_target.position
    assert row["valid_logprob_mass"] == pytest.approx(expected_valid_mass.item())
    assert row["eos_logprob"] == pytest.approx(log_probs[eos_id].item())
    assert row["margin_valid_mass_minus_eos_logprob"] == pytest.approx(
        expected_margin.item()
    )
    assert row["margin_valid_max_logit_minus_eos_logit"] == pytest.approx(
        expected_logit_margin.item()
    )


def test_singleton_remaining_object_is_still_reported_as_valid_next_branch() -> None:
    example = _example(k=2, object_count=3)
    remaining = {str(item) for item in example.rollin_state.remaining}
    target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id in remaining
    )
    assert target.kind == "hard_ce"
    assert len(target.valid_token_ids) == 1

    vocab_size = max(target.teacher_token_id, example.stop_contract.im_end_token_id, *example.input_ids) + 1
    logits = torch.zeros((1, len(example.input_ids), vocab_size), dtype=torch.float32)
    logits[0, target.position - 1, target.teacher_token_id] = 3.0

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=example.input_ids,
    )
    row = _branch_row(rows)

    assert row["target_kind"] == "hard_ce"
    assert row["valid_next_object_branch_present"] is True
    assert row["valid_token_ids"] == list(target.valid_token_ids)


def test_full_prefix_state_reports_only_im_end() -> None:
    example = _example(k=3, object_count=3)
    eos_target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.teacher_token_id == example.stop_contract.im_end_token_id
    )
    vocab_size = max(eos_target.teacher_token_id, *example.input_ids) + 1
    logits = torch.zeros((1, len(example.input_ids), vocab_size), dtype=torch.float32)
    logits[0, eos_target.position - 1, eos_target.teacher_token_id] = 5.0

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=example.input_ids,
    )

    assert not any(row["branch_kind"] == "valid_next_object" for row in rows)
    eos_row = _eos_row(rows)
    assert eos_row["valid_next_object_branch_present"] is False
    assert eos_row["teacher_token_id"] == example.stop_contract.im_end_token_id


def test_processor_padding_rebase_preserves_target_identity() -> None:
    example = _example(k=0, object_count=3)
    pad_id = 0
    processor_ids = (pad_id, pad_id, *example.input_ids)
    delta = compute_prefix_rollin_position_delta(
        processor_input_ids=processor_ids,
        example=example,
    )
    assert delta == 2

    target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id in {str(item) for item in example.rollin_state.remaining}
    )
    vocab_size = max(target.teacher_token_id, example.stop_contract.im_end_token_id, *processor_ids) + 1
    logits = torch.zeros((1, len(processor_ids), vocab_size), dtype=torch.float32)
    logits[0, target.position + delta - 1, target.teacher_token_id] = 3.0

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=processor_ids,
    )
    assert _branch_row(rows)["processor_position"] == target.position + delta


def test_processor_vision_placeholder_rebase_can_align_by_assistant_suffix() -> None:
    example = _example(k=0, object_count=3)
    assistant_start = example.tokenized.assistant_token_span.start
    processor_ids = (999, 998, *example.input_ids[assistant_start:])
    delta = compute_prefix_rollin_position_delta(
        processor_input_ids=processor_ids,
        example=example,
    )
    assert delta == 2 - assistant_start

    target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id in {str(item) for item in example.rollin_state.remaining}
    )
    vocab_size = max(target.teacher_token_id, example.stop_contract.im_end_token_id, *processor_ids) + 1
    logits = torch.zeros((1, len(processor_ids), vocab_size), dtype=torch.float32)
    logits[0, target.position + delta - 1, target.teacher_token_id] = 3.0

    rows = score_prefix_rollin_logits(
        logits=logits,
        example=example,
        tokenizer=FakeTokenizer(),
        processor_input_ids=processor_ids,
    )
    assert _branch_row(rows)["processor_position"] == target.position + delta


def test_summarize_prefix_rollin_probe_rows_reports_margin_health() -> None:
    rows = [
        {
            "branch_kind": "valid_next_object",
            "rollin_k": 0,
            "margin_valid_mass_minus_eos_logprob": 2.0,
        },
        {
            "branch_kind": "valid_next_object",
            "rollin_k": 1,
            "margin_valid_mass_minus_eos_logprob": -1.0,
        },
        {
            "branch_kind": "semantic_eos",
            "rollin_k": 2,
            "eos_logprob": -0.2,
        },
    ]

    summary = summarize_prefix_rollin_probe_rows(rows)

    assert summary["row_count"] == 3
    assert summary["valid_next_object_row_count"] == 2
    assert summary["semantic_eos_row_count"] == 1
    assert summary["valid_margin_mean"] == pytest.approx(0.5)
    assert summary["valid_margin_le_zero_rate"] == pytest.approx(0.5)
    assert summary["rollin_k_values"] == [0, 1, 2]
