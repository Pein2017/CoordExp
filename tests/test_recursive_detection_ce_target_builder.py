from __future__ import annotations

from dataclasses import replace
import re

import pytest

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.objective import prepare_detection_training_example
from src.detection.template import CompactFullTemplate, Stage1JsonPrettyTemplate
from src.detection.tokenization import TokenRole

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class SpecialTokenAwareTokenizer:
    def __init__(self) -> None:
        self.chat_template_calls = 0
        self.encode_calls = 0
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        self.chat_template_calls += 1
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        self.encode_calls += 1

        tokens: list[str] = []
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
            tokens.append(token_text)
            offsets.append((cursor, token_end))
            cursor = token_end

        return {
            "input_ids": [self.token_id(token_text) for token_text in tokens],
            "offset_mapping": offsets,
        }

    def token_id(self, token_text: str) -> int:
        token_id = self._token_to_id.get(token_text)
        if token_id is not None:
            return token_id

        token_id = len(self._token_to_id) + 1
        self._token_to_id[token_text] = token_id
        self._id_to_token[token_id] = token_text
        return token_id

    def token_text(self, token_id: int) -> str:
        return self._id_to_token[token_id]


def _object(
    *,
    normalized_index: int,
    source_index: int,
    instance_id: str,
    desc: str,
    coords: tuple[str, str, str, str],
) -> NormalizedDetectionObject:
    return NormalizedDetectionObject(
        normalized_object_index=normalized_index,
        source_object_index=source_index,
        object_instance_id=instance_id,
        desc=desc,
        bbox_2d=CoordinateTokenBox(*coords),
        category_id=normalized_index + 1,
        category_name=desc,
        coco_ann_id=1000 + source_index,
    )


def _sample(
    *objects: NormalizedDetectionObject,
) -> NormalizedDetectionSample:
    realized = tuple(obj.source_object_index for obj in objects)
    indexed_objects = tuple(
        replace(obj, normalized_object_index=index)
        for index, obj in enumerate(objects)
    )
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=indexed_objects,
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=17,
            seed_source="unit-test",
        ).with_realized(realized),
    )


def _target_map(prepared) -> dict[int, object]:
    assert prepared.recursive_detection_targets is not None
    return {
        target.position: target
        for target in prepared.recursive_detection_targets.token_targets
    }


def _token_texts(tokenizer: SpecialTokenAwareTokenizer, token_ids: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(tokenizer.token_text(token_id) for token_id in token_ids)


def test_compact_shared_object_ref_is_hard_ce_and_first_desc_divergence_is_multi_positive() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="img-9:ann-501:src-7",
            desc="cat",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
        _object(
            normalized_index=1,
            source_index=3,
            instance_id="img-9:ann-502:src-3",
            desc="dog",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    first_entry = prepared.tokenized.object_entries[0]
    targets = _target_map(prepared)

    object_ref_target = targets[first_entry.object_ref_start_span.start]
    assert object_ref_target.kind == "hard_ce"
    assert object_ref_target.token_role is TokenRole.CONTROL
    assert object_ref_target.object_instance_id == first_entry.object_instance_id
    assert _token_texts(tokenizer, object_ref_target.valid_token_ids) == (
        OBJECT_REF_START_TOKEN,
    )

    desc_target = targets[first_entry.desc_span.start]
    assert desc_target.kind == "trie_multi_positive"
    assert desc_target.token_role is TokenRole.DESC
    assert desc_target.object_instance_id == first_entry.object_instance_id
    assert _token_texts(tokenizer, desc_target.valid_token_ids) == ("c", "d")
    assert desc_target.child_multiplicities == (1, 1)
    assert desc_target.child_probabilities == pytest.approx((0.5, 0.5))
    assert sum(desc_target.child_probabilities) == pytest.approx(1.0)


def test_compact_same_desc_diverges_at_first_coordinate_token() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="img-9:ann-601:src-7",
            desc="car",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
        _object(
            normalized_index=1,
            source_index=3,
            instance_id="img-9:ann-602:src-3",
            desc="car",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    first_entry = prepared.tokenized.object_entries[0]
    targets = _target_map(prepared)

    bbox_start_target = targets[first_entry.bbox_start_span.start]
    assert bbox_start_target.kind == "hard_ce"
    assert _token_texts(tokenizer, bbox_start_target.valid_token_ids) == (BOX_START_TOKEN,)

    first_coord_target = targets[first_entry.coord_spans[0].start]
    assert first_coord_target.kind == "trie_multi_positive"
    assert first_coord_target.token_role is TokenRole.COORD
    assert _token_texts(tokenizer, first_coord_target.valid_token_ids) == (
        "<|coord_10|>",
        "<|coord_100|>",
    )
    assert first_coord_target.child_multiplicities == (1, 1)


def test_desc_prefix_collision_allows_box_start_as_trie_child() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="img-9:ann-701:src-7",
            desc="car",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
        _object(
            normalized_index=1,
            source_index=3,
            instance_id="img-9:ann-702:src-3",
            desc="cart",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    first_entry = prepared.tokenized.object_entries[0]
    targets = _target_map(prepared)

    bbox_start_target = targets[first_entry.bbox_start_span.start]
    assert bbox_start_target.kind == "trie_multi_positive"
    assert bbox_start_target.token_role is TokenRole.BBOX_START
    assert set(_token_texts(tokenizer, bbox_start_target.valid_token_ids)) == {
        BOX_START_TOKEN,
        "t",
    }
    assert bbox_start_target.child_multiplicities == (1, 1)


def test_exact_duplicate_entries_are_removed_one_teacher_instance_at_a_time() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    duplicate = _object(
        normalized_index=0,
        source_index=7,
        instance_id="img-9:ann-801:src-7",
        desc="cat",
        coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
    )
    sample = _sample(
        duplicate,
        replace(
            duplicate,
            normalized_object_index=1,
            source_object_index=8,
            object_instance_id="img-9:ann-802:src-8",
            coco_ann_id=1802,
        ),
        _object(
            normalized_index=2,
            source_index=3,
            instance_id="img-9:ann-803:src-3",
            desc="dog",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    first_entry, second_entry = prepared.tokenized.object_entries[:2]
    targets = _target_map(prepared)

    first_desc_target = targets[first_entry.desc_span.start]
    assert first_desc_target.kind == "trie_multi_positive"
    assert _token_texts(tokenizer, first_desc_target.valid_token_ids) == ("c", "d")
    assert first_desc_target.child_multiplicities == (2, 1)

    second_desc_target = targets[second_entry.desc_span.start]
    assert second_desc_target.kind == "trie_multi_positive"
    assert second_desc_target.object_instance_id == second_entry.object_instance_id
    assert _token_texts(tokenizer, second_desc_target.valid_token_ids) == ("c", "d")
    assert second_desc_target.child_multiplicities == (1, 1)


def test_stage1_json_template_builds_recursive_targets_from_rendered_spans() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="img-9:ann-901:src-7",
            desc="car",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
        _object(
            normalized_index=1,
            source_index=3,
            instance_id="img-9:ann-902:src-3",
            desc="car",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=Stage1JsonPrettyTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    first_entry = prepared.tokenized.object_entries[0]
    targets = _target_map(prepared)

    entry_open_target = targets[first_entry.entry_span.start]
    assert entry_open_target.kind == "hard_ce"
    assert _token_texts(tokenizer, entry_open_target.valid_token_ids) == ("{",)

    first_coord_target = targets[first_entry.coord_spans[0].start]
    assert first_coord_target.kind == "trie_multi_positive"
    assert first_coord_target.token_role is TokenRole.COORD
    assert _token_texts(tokenizer, first_coord_target.valid_token_ids) == (
        "<|coord_10|>",
        "<|coord_100|>",
    )


def test_recursive_builder_rejects_duplicate_active_object_instance_ids() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="duplicate-id",
            desc="cat",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
        _object(
            normalized_index=1,
            source_index=3,
            instance_id="duplicate-id",
            desc="dog",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
    )

    with pytest.raises(ValueError, match="duplicate object_instance_id"):
        prepare_detection_training_example(
            sample,
            template=CompactFullTemplate(),
            tokenizer=tokenizer,
            mode="random_permutation_et_rmp_ce",
        )
