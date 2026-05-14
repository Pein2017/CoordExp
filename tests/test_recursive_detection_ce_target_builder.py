from __future__ import annotations

from dataclasses import replace
import re

import pytest
import torch

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.coord_soft_targets import (
    CoordSoftTargetCandidate,
    CoordSoftTargetRuntimeConfig,
    full_vocab_coord_support_balance_ce,
)
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
from src.detection.objective import prepare_detection_training_example
from src.detection.template import CompactFullTemplate, Stage1JsonPrettyTemplate
from src.detection.tokenization import TokenRole
from src.metrics.events import reduce_metric_events

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


def _entry_coord_targets(prepared, object_instance_id: str) -> tuple[object, object, object, object]:
    targets = _target_map(prepared)
    entry = next(
        item
        for item in prepared.tokenized.object_entries
        if item.object_instance_id == object_instance_id
    )
    return tuple(targets[span.start] for span in entry.coord_spans)


def _coord_candidate_ids(target) -> tuple[str, ...]:
    return tuple(spec.object_instance_id for spec in target.coord_instance_candidates)


def _coord_candidate_tuples(target) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    return tuple(
        (spec.object_instance_id, spec.bbox_xyxy)
        for spec in target.coord_instance_candidates
    )


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
    assert tuple(
        (spec.object_instance_id, spec.slot_name, spec.bbox_xyxy)
        for spec in first_coord_target.coord_soft_targets
    ) == (
        ("img-9:ann-601:src-7", "x1", (10, 20, 30, 40)),
        ("img-9:ann-602:src-3", "x1", (100, 200, 300, 400)),
    )
    assert tuple(
        spec.probability for spec in first_coord_target.coord_soft_targets
    ) == pytest.approx((0.5, 0.5))


def test_coord_instance_candidates_use_same_desc_semantic_branch_only() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    car_a = _object(
        normalized_index=0,
        source_index=7,
        instance_id="img-9:ann-611:src-7",
        desc="car",
        coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
    )
    car_b = _object(
        normalized_index=1,
        source_index=3,
        instance_id="img-9:ann-612:src-3",
        desc="car",
        coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
    )
    dog_c = _object(
        normalized_index=2,
        source_index=5,
        instance_id="img-9:ann-613:src-5",
        desc="dog",
        coords=("<|coord_110|>", "<|coord_210|>", "<|coord_310|>", "<|coord_410|>"),
    )
    sample = _sample(car_a, car_b, dog_c)

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    car_entry_ids = [
        entry.object_instance_id
        for entry in prepared.tokenized.object_entries
        if entry.object_instance_id in {car_a.object_instance_id, car_b.object_instance_id}
    ]
    assert car_entry_ids
    first_car_targets = _entry_coord_targets(prepared, car_entry_ids[0])

    for target in first_car_targets:
        assert _coord_candidate_ids(target) == (
            car_a.object_instance_id,
            car_b.object_instance_id,
        )
        assert target.coord_slot_name in {"x1", "y1", "x2", "y2"}


def test_coord_instance_candidates_exclude_already_emitted_same_desc_instances() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    car_a = _object(
        normalized_index=0,
        source_index=7,
        instance_id="img-9:ann-621:src-7",
        desc="car",
        coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
    )
    car_b = _object(
        normalized_index=1,
        source_index=3,
        instance_id="img-9:ann-622:src-3",
        desc="car",
        coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
    )
    sample = _sample(car_a, car_b)

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    car_entry_ids = [
        entry.object_instance_id
        for entry in prepared.tokenized.object_entries
        if entry.object_instance_id in {car_a.object_instance_id, car_b.object_instance_id}
    ]
    assert len(car_entry_ids) == 2
    second_x1_target = _entry_coord_targets(prepared, car_entry_ids[1])[0]

    assert _coord_candidate_ids(second_x1_target) == (car_entry_ids[1],)


def test_coord_instance_candidates_carry_same_semantic_branch_across_coord_block() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    car_a = _object(
        normalized_index=0,
        source_index=7,
        instance_id="img-9:ann-631:src-7",
        desc="car",
        coords=(
            "<|coord_100|>",
            "<|coord_100|>",
            "<|coord_200|>",
            "<|coord_200|>",
        ),
    )
    car_b = _object(
        normalized_index=1,
        source_index=3,
        instance_id="img-9:ann-632:src-3",
        desc="car",
        coords=(
            "<|coord_103|>",
            "<|coord_101|>",
            "<|coord_350|>",
            "<|coord_260|>",
        ),
    )
    sample = _sample(car_a, car_b)

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    first_entry = prepared.tokenized.object_entries[0]
    coord_targets = _entry_coord_targets(prepared, first_entry.object_instance_id)
    expected = (
        (car_a.object_instance_id, (100, 100, 200, 200)),
        (car_b.object_instance_id, (103, 101, 350, 260)),
    )

    for target in coord_targets:
        assert _coord_candidate_tuples(target) == expected


def test_coord_instance_candidates_include_teacher_once_for_every_candidate_coord_target() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="img-9:ann-641:src-7",
            desc="car",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
        _object(
            normalized_index=1,
            source_index=3,
            instance_id="img-9:ann-642:src-3",
            desc="car",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
        _object(
            normalized_index=2,
            source_index=5,
            instance_id="img-9:ann-643:src-5",
            desc="dog",
            coords=("<|coord_110|>", "<|coord_210|>", "<|coord_310|>", "<|coord_410|>"),
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    assert prepared.recursive_detection_targets is not None
    candidate_coord_targets = [
        target
        for target in prepared.recursive_detection_targets.token_targets
        if target.coord_instance_candidates
    ]
    assert candidate_coord_targets
    for target in candidate_coord_targets:
        assert target.coord_slot_name in {"x1", "y1", "x2", "y2"}
        assert _coord_candidate_ids(target).count(target.object_instance_id) == 1


def test_coord_instance_candidates_do_not_leak_across_prepared_examples() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    first_prepared = prepare_detection_training_example(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-9:ann-651:src-7",
                desc="car",
                coords=(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
            ),
            _object(
                normalized_index=1,
                source_index=3,
                instance_id="img-9:ann-652:src-3",
                desc="car",
                coords=(
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ),
            ),
        ),
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )
    second_prepared = prepare_detection_training_example(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-9:ann-651:src-7",
                desc="car",
                coords=(
                    "<|coord_15|>",
                    "<|coord_25|>",
                    "<|coord_35|>",
                    "<|coord_45|>",
                ),
            ),
            _object(
                normalized_index=1,
                source_index=3,
                instance_id="img-9:ann-652:src-3",
                desc="car",
                coords=(
                    "<|coord_105|>",
                    "<|coord_205|>",
                    "<|coord_305|>",
                    "<|coord_405|>",
                ),
            ),
        ),
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )

    first_x1 = _entry_coord_targets(
        first_prepared,
        first_prepared.tokenized.object_entries[0].object_instance_id,
    )[0]
    second_x1 = _entry_coord_targets(
        second_prepared,
        second_prepared.tokenized.object_entries[0].object_instance_id,
    )[0]

    assert _coord_candidate_tuples(first_x1) == (
        ("img-9:ann-651:src-7", (10, 20, 30, 40)),
        ("img-9:ann-652:src-3", (100, 200, 300, 400)),
    )
    assert _coord_candidate_tuples(second_x1) == (
        ("img-9:ann-651:src-7", (15, 25, 35, 45)),
        ("img-9:ann-652:src-3", (105, 205, 305, 405)),
    )


def test_compact_trie_coordinate_soft_ce_uses_coord_metadata_not_semantic_atom() -> None:
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
    assert prepared.recursive_detection_targets is not None
    first_entry = prepared.tokenized.object_entries[0]
    first_coord_target = _target_map(prepared)[first_entry.coord_spans[0].start]
    assert first_coord_target.kind == "trie_multi_positive"
    assert first_coord_target.token_role is TokenRole.COORD
    assert first_coord_target.semantic_role.value == "entry_trie_decision"

    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="iou_gibbs_v0",
        tau=0.0090909091,
        coord_token_start=10,
        coord_token_end=1009,
    )
    logits = torch.zeros(
        (
            max(
                target.position
                for target in prepared.recursive_detection_targets.token_targets
            )
            + 1,
            1020,
        ),
        dtype=torch.float32,
    )
    candidates = tuple(
        CoordSoftTargetCandidate(
            object_instance_id=spec.object_instance_id,
            slot_name=spec.slot_name,
            bbox_xyxy=spec.bbox_xyxy,
            probability=spec.probability,
        )
        for spec in first_coord_target.coord_soft_targets
    )
    manual = full_vocab_coord_support_balance_ce(
        logits[first_coord_target.position - 1],
        candidates,
        cfg,
        support_weight=2.0,
        balance_weight=1.0,
    )
    focused_targets = replace(
        prepared.recursive_detection_targets,
        token_targets=(first_coord_target,),
        loss_atoms=tuple(
            atom
            for atom in prepared.recursive_detection_targets.loss_atoms
            if first_coord_target.position in atom.token_positions
        ),
    )

    result = compute_recursive_detection_ce_batch_loss(
        logits=logits,
        targets=(focused_targets,),
        weights=RecursiveDetectionLossWeights(
            support_weight=2.0,
            balance_weight=1.0,
            coord_soft_ce=cfg,
        ),
    )
    reduced = reduce_metric_events(result.metric_events)

    assert result.per_position_losses[0][first_coord_target.position].item() == (
        pytest.approx(manual.weighted_loss.item())
    )
    assert reduced["recursive_detection_ce/coord_soft_ce/support_mixture"] == (
        pytest.approx(1.0)
    )
    assert reduced["recursive_detection_ce/coord_soft_ce/candidate_count"] == (
        pytest.approx(2.0)
    )


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
