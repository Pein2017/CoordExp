from __future__ import annotations

import re
from typing import Any

import pytest
import torch

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.loss import (
    RecursiveDetectionLossWeights,
    compute_recursive_detection_ce_batch_loss,
)
from src.detection.objective import build_compact_prefix_rollin_example
from src.detection.tokenization import TokenRole


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class SpecialTokenAwareTokenizer:
    eos_token = "<|endoftext|>"
    unk_token_id = 0

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            OBJECT_REF_START_TOKEN: 4,
            BOX_START_TOKEN: 5,
        }
        self._id_to_token = {token_id: token for token, token_id in self._token_to_id.items()}
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

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert add_generation_prompt is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if not tokenize:
            return rendered
        encoded = self(
            rendered,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        return list(encoded["input_ids"])

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
            token_id = self._token_to_id.setdefault(
                token_text,
                len(self._token_to_id) + 1,
            )
            self._id_to_token.setdefault(token_id, token_text)
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end

        return {"input_ids": input_ids, "offset_mapping": offsets}

    def token_text(self, token_id: int) -> str:
        return self._id_to_token[int(token_id)]


def _object(
    instance_id: str,
    *,
    desc: str,
    coords: tuple[int, int, int, int],
    source_index: int,
) -> NormalizedDetectionObject:
    return NormalizedDetectionObject(
        normalized_object_index=source_index,
        source_object_index=source_index,
        object_instance_id=instance_id,
        desc=desc,
        bbox_2d=CoordinateTokenBox(
            *(f"<|coord_{coord}|>" for coord in coords),
        ),
        category_id=source_index,
        category_name=desc,
        coco_ann_id=1000 + source_index,
    )


def _objects() -> tuple[NormalizedDetectionObject, NormalizedDetectionObject]:
    return (
        _object(
            "inst-a",
            desc="cat",
            coords=(10, 20, 30, 40),
            source_index=0,
        ),
        _object(
            "inst-b",
            desc="dog",
            coords=(50, 60, 70, 80),
            source_index=1,
        ),
    )


def _example(*, k: int, eos_trust_weight: float = 1.0):
    objects = _objects()
    return build_compact_prefix_rollin_example(
        objects=objects,
        rollin_order=objects,
        k=k,
        tokenizer=SpecialTokenAwareTokenizer(),
        eos_trust_weight=eos_trust_weight,
    )


def test_normalized_sample_rollin_order_mismatch_fails_closed_until_explicit_builder_exists() -> None:
    objects = _objects()
    sample = NormalizedDetectionSample(
        images=("unit.jpg",),
        objects=objects,
        width=100,
        height=100,
        image_id=1,
        file_name="unit.jpg",
        metadata=DetectionMetadata(source="unit", split="train"),
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=1,
            seed_source="unit",
        ).with_realized((0, 1)),
    )

    with pytest.raises(
        ValueError,
        match="rollin_order.*normalized_sample.*explicit emitted/suffix builder",
    ):
        build_compact_prefix_rollin_example(
            objects=objects,
            rollin_order=tuple(reversed(objects)),
            k=1,
            tokenizer=SpecialTokenAwareTokenizer(),
            normalized_sample=sample,
        )


def _token_text_by_id(example) -> dict[int, str]:
    token_text_by_id: dict[int, str] = {}
    for token_id, (start, end) in zip(
        example.input_ids,
        example.tokenized.offset_mapping,
        strict=True,
    ):
        token_text_by_id.setdefault(int(token_id), example.chat_text[start:end])
    return token_text_by_id


def test_rollin_prefix_labels_are_masked_and_suffix_labels_active() -> None:
    example = _example(k=1)
    prefix_positions = example.debug_spans["rollin_prefix"].token_positions
    suffix_positions = example.debug_spans["supervised_suffix"].token_positions

    assert prefix_positions
    assert suffix_positions
    assert all(example.labels[position] == -100 for position in prefix_positions)
    assert all(example.labels[position] != -100 for position in suffix_positions)


def test_first_suffix_token_position_after_prefix_uses_active_label() -> None:
    example = _example(k=1)
    first_suffix = example.debug_spans["supervised_suffix"].token_positions[0]
    target = example.recursive_detection_targets.token_targets[0]

    assert target.position == first_suffix
    assert first_suffix > 0
    assert example.labels[first_suffix] == target.teacher_token_id
    assert example.tokenized.input_ids[first_suffix] == target.teacher_token_id


def test_k_zero_first_remaining_desc_branch_is_multi_positive() -> None:
    example = _example(k=0)
    token_text_by_id = _token_text_by_id(example)

    branch_target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.kind == "trie_multi_positive" and target.token_role is TokenRole.DESC
    )

    assert branch_target.object_instance_id == "inst-a"
    assert branch_target.teacher_token_id in branch_target.valid_token_ids
    assert {token_text_by_id[token_id] for token_id in branch_target.valid_token_ids} == {
        "c",
        "d",
    }
    assert sum(branch_target.child_probabilities) == pytest.approx(1.0)


def test_k_zero_local_multi_positive_support_accepts_non_teacher_child() -> None:
    example = _example(k=0)
    branch_target = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.kind == "trie_multi_positive" and target.token_role is TokenRole.DESC
    )
    alternate_child = next(
        token_id
        for token_id in branch_target.valid_token_ids
        if token_id != branch_target.teacher_token_id
    )
    vocab_size = max(example.input_ids) + 1
    logits = torch.full((1, len(example.input_ids), vocab_size), -30.0)
    for target in example.recursive_detection_targets.token_targets:
        logits[0, target.position - 1, target.teacher_token_id] = 30.0
    logits[0, branch_target.position - 1, branch_target.teacher_token_id] = -30.0
    logits[0, branch_target.position - 1, alternate_child] = 30.0

    result = compute_recursive_detection_ce_batch_loss(
        logits=logits,
        targets=[example.recursive_detection_targets],
        weights=RecursiveDetectionLossWeights(
            support_weight=1.0,
            balance_weight=0.0,
        ),
    )

    assert result.loss.item() < 1e-4


def test_emitted_prefix_object_is_excluded_from_remaining_branch() -> None:
    example = _example(k=1)
    first_remaining_desc = next(
        target
        for target in example.recursive_detection_targets.token_targets
        if target.object_instance_id == "inst-b" and target.token_role is TokenRole.DESC
    )

    assert first_remaining_desc.kind == "hard_ce"
    assert first_remaining_desc.valid_token_ids == (
        first_remaining_desc.teacher_token_id,
    )


def test_boundary_separator_between_prefix_and_suffix_is_active_continuation() -> None:
    example = _example(k=1)
    first_suffix = example.debug_spans["supervised_suffix"].token_positions[0]
    target = example.recursive_detection_targets.token_targets[0]

    assert example.tokenized.token_roles[first_suffix] is TokenRole.SEPARATOR
    assert target.kind == "hard_ce"
    assert target.token_role is TokenRole.SEPARATOR
    assert example.labels[first_suffix] == target.teacher_token_id
    start, end = example.tokenized.offset_mapping[first_suffix]
    assert example.tokenized.chat_text[start:end] == "\n"


def test_first_suffix_entry_starts_after_active_boundary_separator() -> None:
    example = _example(k=1)
    suffix_positions = example.debug_spans["supervised_suffix"].token_positions
    suffix_entry_positions = example.debug_spans[
        "supervised_suffix_entries"
    ].token_positions

    assert suffix_entry_positions
    assert suffix_positions[0] < suffix_entry_positions[0]
    assert example.tokenized.token_roles[suffix_entry_positions[0]] is TokenRole.CONTROL
    assert (
        example.tokenized.input_ids[suffix_entry_positions[0]]
        == example.labels[suffix_entry_positions[0]]
    )


def test_prefix_rollin_diagnostics_describe_sampled_k_not_full_prefix_mixture() -> None:
    example = _example(k=1)
    diagnostics = example.recursive_detection_targets.state_weighting_diagnostics

    assert diagnostics.prefix_length_probabilities == (0.0, 1.0, 0.0)
    assert diagnostics.supervised_token_counts_by_prefix_length == (
        0,
        len(example.recursive_detection_targets.token_targets),
        0,
    )
    assert diagnostics.entry_exposures == (0.0, 1.0)
    assert diagnostics.separator_exposures == (1.0,)
    assert diagnostics.terminal_exposure == 1.0


def test_shifted_target_consumes_previous_logit_after_prefix_rewrite() -> None:
    example = _example(k=1)
    vocab_size = max(example.input_ids) + 1
    logits = torch.full((1, len(example.input_ids), vocab_size), -30.0)
    for target in example.recursive_detection_targets.token_targets:
        logits[0, target.position - 1, target.teacher_token_id] = 30.0

    loss = compute_recursive_detection_ce_batch_loss(
        logits=logits,
        targets=[example.recursive_detection_targets],
    )

    assert loss.loss.item() < 1e-4


def test_k_equals_n_masks_all_objects_and_only_trains_weighted_im_end() -> None:
    example = _example(k=2, eos_trust_weight=0.25)
    object_positions = example.debug_spans["rollin_prefix"].token_positions
    active_positions = tuple(
        position for position, label in enumerate(example.labels) if label != -100
    )

    assert all(example.labels[position] == -100 for position in object_positions)
    assert example.debug_spans["supervised_suffix"].token_positions == ()
    assert active_positions == tuple(example.assistant_stop_token_span.token_indices())

    eos_pos = active_positions[0]
    assert example.input_ids[eos_pos] == example.stop_contract.im_end_token_id
    assert example.labels[eos_pos] == example.stop_contract.im_end_token_id
    assert example.recursive_detection_targets.token_targets[0].position == eos_pos
    assert example.recursive_detection_targets.token_targets[0].loss_weight == 0.25


def test_k_less_than_n_weights_im_end_after_suffix_completion() -> None:
    example = _example(k=1, eos_trust_weight=0.25)
    eos_positions = set(example.assistant_stop_token_span.token_indices())
    eos_targets = [
        target
        for target in example.recursive_detection_targets.token_targets
        if target.position in eos_positions
    ]

    assert len(eos_targets) == 1
    assert eos_targets[0].teacher_token_id == example.stop_contract.im_end_token_id
    assert eos_targets[0].loss_weight == pytest.approx(0.25)


def test_eos_trust_weight_scales_k_equals_n_stop_loss_without_dropping_target() -> None:
    full_weight = _example(k=2, eos_trust_weight=1.0)
    low_weight = _example(k=2, eos_trust_weight=0.25)
    zero_weight = _example(k=2, eos_trust_weight=0.0)
    vocab_size = max(max(example.input_ids) for example in (full_weight, low_weight)) + 1

    def loss_value(example) -> float:
        logits = torch.zeros((1, len(example.input_ids), vocab_size))
        result = compute_recursive_detection_ce_batch_loss(
            logits=logits,
            targets=[example.recursive_detection_targets],
        )
        assert torch.isfinite(result.loss)
        assert len(example.recursive_detection_targets.token_targets) == 1
        return float(result.loss.item())

    full_loss = loss_value(full_weight)
    low_loss = loss_value(low_weight)
    zero_loss = loss_value(zero_weight)

    assert full_loss > 0.0
    assert low_loss == pytest.approx(0.25 * full_loss)
    assert zero_loss == pytest.approx(0.0)
    assert zero_weight.recursive_detection_targets.token_targets[0].loss_weight == 0.0


def test_prefix_rollin_type_gate_sidecar_covers_positive_tokens() -> None:
    objects = _objects()
    example = build_compact_prefix_rollin_example(
        objects=objects,
        rollin_order=objects,
        k=0,
        tokenizer=SpecialTokenAwareTokenizer(),
        type_gate_config={
            "enabled": True,
            "weights": {"struct": 2.0, "coord": 1.0, "desc": 0.2, "eos": 0.5},
        },
    )

    assert example.recursive_detection_targets.token_targets
    assert all(
        target.type_gate_token_ids
        for target in example.recursive_detection_targets.token_targets
    )
    for target in example.recursive_detection_targets.token_targets:
        positives = target.valid_token_ids or (target.teacher_token_id,)
        assert set(positives).issubset(set(target.type_gate_token_ids))
        assert target.type_gate_weight >= 0.0


def test_rendered_payload_excludes_manual_eos_and_chat_template_supplies_stop() -> None:
    example = _example(k=0)

    assert "<|im_end|>" not in example.rendered_assistant.text
    assert example.assistant_stop_token_text == "<|im_end|>"
    assert "<|endoftext|>" not in example.chat_text
    assert "<|end_of_text|>" not in example.chat_text


def test_assistant_stop_span_is_template_supplied_im_end_once_after_payload() -> None:
    example = _example(k=0)
    stop_positions = tuple(example.assistant_stop_token_span.token_indices())

    assert len(stop_positions) == 1
    assert example.input_ids[stop_positions[0]] == example.stop_contract.im_end_token_id
    assert example.assistant_stop_char_span.start == example.assistant_char_span.end
