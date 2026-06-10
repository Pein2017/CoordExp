from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import pytest
import torch

from src.common.detection_compact_rows import (
    BOX_START_TOKEN,
    IM_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.scene import detection_scene_from_normalized_sample_bridge
from src.detection.teacher_forcing.rollin import derive_rollin_seed
from src.detection.teacher_forcing.target_builder import (
    TeacherForcingTargetBuilder,
    build_teacher_forcing_target,
)
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.validation import validate_target_ir
from src.training.teacher_forcing.vocab import RoleVocab


class TinyContextTokenizer:
    """Deterministic compact_full tokenizer with marker offsets.

    The tokenizer intentionally maps ``carrot`` to ``car`` + ``rot`` so the
    target builder must represent the post-``car`` ambiguity as {TEXT, SCHEMA}.
    """

    stop_token_id = 10000

    def __init__(self) -> None:
        self._ids: dict[str, int] = {
            "<bos>": 1,
            OBJECT_REF_START_TOKEN: 10,
            BOX_START_TOKEN: 11,
            IM_END_TOKEN: self.stop_token_id,
        }
        self._next_id = 100

    @property
    def bos_token_id(self) -> int:
        return self._ids["<bos>"]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._id_for(token)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self._id_for(token) for token, _span in self._scan(text)]

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        scanned = self._scan(text)
        return {
            "input_ids": [self._id_for(token) for token, _span in scanned],
            "offset_mapping": [span for _token, span in scanned],
        }

    def token_id(self, token: str) -> int:
        return self._id_for(token)

    def role_vocab(self) -> RoleVocab:
        return RoleVocab(
            schema_token_ids=frozenset(
                {
                    self.token_id(OBJECT_REF_START_TOKEN),
                    self.token_id(BOX_START_TOKEN),
                }
            ),
            text_token_ids=frozenset(
                self.token_id(token)
                for token in ("car", "rot", "cat", "dog", "bus", "red", "sedan")
            ),
            coord_token_ids=frozenset(self.token_id(f"<|coord_{idx}|>") for idx in range(1000)),
            stop_token_id=self.stop_token_id,
        )

    def _id_for(self, token: str) -> int:
        if token not in self._ids:
            self._ids[token] = self._next_id
            self._next_id += 1
        return self._ids[token]

    def _scan(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        parts: list[tuple[str, tuple[int, int]]] = []
        cursor = 0
        while cursor < len(text):
            matched = False
            for special in (OBJECT_REF_START_TOKEN, BOX_START_TOKEN, IM_END_TOKEN):
                if text.startswith(special, cursor):
                    parts.append((special, (cursor, cursor + len(special))))
                    cursor += len(special)
                    matched = True
                    break
            if matched:
                continue

            coord = re.match(r"<\|coord_\d+\|>", text[cursor:])
            if coord is not None:
                token = coord.group(0)
                parts.append((token, (cursor, cursor + len(token))))
                cursor += len(token)
                continue

            if text.startswith("carrot", cursor):
                parts.append(("car", (cursor, cursor + 3)))
                parts.append(("rot", (cursor + 3, cursor + 6)))
                cursor += 6
                continue

            word = re.match(r"[A-Za-z0-9_]+", text[cursor:])
            if word is not None:
                token = word.group(0)
                parts.append((token, (cursor, cursor + len(token))))
                cursor += len(token)
                continue

            parts.append((text[cursor], (cursor, cursor + 1)))
            cursor += 1
        return parts


class SplitMarkerTokenizer(TinyContextTokenizer):
    def _scan(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        if text.startswith(OBJECT_REF_START_TOKEN):
            head = [
                ("<|object_ref_", (0, len("<|object_ref_"))),
                ("start|>", (len("<|object_ref_"), len(OBJECT_REF_START_TOKEN))),
            ]
            tail = super()._scan(text[len(OBJECT_REF_START_TOKEN) :])
            shifted = [
                (token, (start + len(OBJECT_REF_START_TOKEN), end + len(OBJECT_REF_START_TOKEN)))
                for token, (start, end) in tail
            ]
            return head + shifted
        return super()._scan(text)


class NoBosTokenizer:
    stop_token_id = TinyContextTokenizer.stop_token_id

    def __init__(self) -> None:
        self._inner = TinyContextTokenizer()

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._inner.convert_tokens_to_ids(token)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._inner.encode(text, add_special_tokens=add_special_tokens)

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        return self._inner(
            text,
            return_offsets_mapping=return_offsets_mapping,
            add_special_tokens=add_special_tokens,
        )

    def token_id(self, token: str) -> int:
        return self._inner.token_id(token)

    def role_vocab(self) -> RoleVocab:
        return self._inner.role_vocab()


def _sample(
    objects: tuple[NormalizedDetectionObject, ...],
    *,
    image_id: int = 42,
    file_name: str = "image.jpg",
) -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=(file_name,),
        objects=objects,
        width=640,
        height=480,
        image_id=image_id,
        file_name=file_name,
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized(
            tuple(obj.source_object_index for obj in objects)
        ),
    )


def _object(
    desc: str,
    bbox: tuple[int, int, int, int],
    *,
    index: int = 0,
    source_index: int | None = None,
) -> NormalizedDetectionObject:
    source = index if source_index is None else source_index
    return NormalizedDetectionObject(
        normalized_object_index=index,
        source_object_index=source,
        object_instance_id=f"img-42:ann-{100 + index}:src-{source}",
        desc=desc,
        bbox_2d=CoordinateTokenBox(*bbox),
        category_id=index + 1,
        category_name=desc,
        coco_ann_id=100 + index,
    )


def _build(
    sample: NormalizedDetectionSample | dict[str, Any],
    *,
    profile: str = "valid_set",
    policy_name: str = "random_permutation",
    epoch: int = 3,
    stable_sample_id: str = "sample-42",
    max_length: int | None = None,
    tokenizer: TinyContextTokenizer | None = None,
):
    tok = tokenizer or TinyContextTokenizer()
    result = build_teacher_forcing_target(
        sample,
        tokenizer=tok,
        profile=profile,
        epoch=epoch,
        stable_sample_id=stable_sample_id,
        max_length=max_length,
        policy_name=policy_name,
    )
    return result, tok


def _atoms_for_selected_text(result, tokenizer: TinyContextTokenizer, token: str):
    token_id = tokenizer.token_id(token)
    return [atom for atom in result.target_ir.atoms if atom.selected_token_id == token_id]


def _validate(result, tokenizer: TinyContextTokenizer) -> None:
    validate_target_ir(
        result.target_ir,
        input_ids=torch.tensor([result.input_ids]),
        role_vocab=tokenizer.role_vocab(),
    )


def test_hard_sft_profile_emits_singleton_valid_token_ids() -> None:
    result, tokenizer = _build(_sample((_object("cat", (1, 2, 10, 20)),)), profile="hard_sft")

    assert result.ok
    assert result.target_ir is not None
    assert result.target_ir.metadata["marginal_scope"] == "sampled_path_next_token"
    assert all(atom.valid_token_ids == frozenset({atom.selected_token_id}) for atom in result.target_ir.atoms)
    _validate(result, tokenizer)


def test_valid_set_profile_emits_all_legal_next_tokens_at_ambiguous_prefixes() -> None:
    sample = _sample(
        (
            _object("cat", (1, 2, 10, 20), index=0),
            _object("dog", (3, 4, 30, 40), index=1),
        )
    )
    result, tokenizer = _build(sample, profile="valid_set")

    object_ref_atom = _atoms_for_selected_text(result, tokenizer, OBJECT_REF_START_TOKEN)[0]

    assert object_ref_atom.valid_token_ids == frozenset({tokenizer.token_id(OBJECT_REF_START_TOKEN)})
    next_desc_atom = result.target_ir.atoms[result.target_ir.atoms.index(object_ref_atom) + 1]
    assert next_desc_atom.valid_token_ids == frozenset(
        {tokenizer.token_id("cat"), tokenizer.token_id("dog")}
    )
    _validate(result, tokenizer)


def test_sorted_rollin_uses_teacher_order_with_same_residual_set_atoms() -> None:
    sample = _sample(
        (
            _object("cat", (1, 2, 10, 20), index=0),
            _object("dog", (3, 4, 30, 40), index=1),
        )
    )
    result, tokenizer = _build(sample, profile="valid_set", policy_name="sorted")

    assert result.ok
    assert result.target_ir is not None
    assert result.target_ir.metadata["rollin_policy"] == "sorted"
    assert result.target_ir.metadata["selected_normalized_object_indices"] == (0, 1)
    assert result.rendered_text.startswith(f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}")

    first_desc_atom = next(
        atom
        for atom in result.target_ir.atoms
        if atom.selected_token_id == tokenizer.token_id("cat")
    )
    assert first_desc_atom.valid_token_ids == frozenset(
        {tokenizer.token_id("cat"), tokenizer.token_id("dog")}
    )
    _validate(result, tokenizer)


def test_valid_set_emits_text_schema_mixed_role_atom_for_car_and_carrot() -> None:
    sample = _sample(
        (
            _object("car", (1, 2, 10, 20), index=0),
            _object("carrot", (3, 4, 30, 40), index=1),
        )
    )
    result, tokenizer = _build(sample, profile="valid_set")

    mixed_atoms = [
        atom
        for atom in result.target_ir.atoms
        if atom.allowed_token_roles == frozenset({TokenRole.TEXT, TokenRole.SCHEMA})
    ]

    assert mixed_atoms
    assert mixed_atoms[0].valid_token_ids == frozenset(
        {tokenizer.token_id(BOX_START_TOKEN), tokenizer.token_id("rot")}
    )
    _validate(result, tokenizer)


def test_same_description_repeated_objects_stay_ambiguous_until_x1() -> None:
    sample = _sample(
        (
            _object("car", (1, 2, 10, 20), index=0),
            _object("car", (3, 4, 30, 40), index=1),
        )
    )
    result, tokenizer = _build(sample, profile="valid_set")

    ambiguous_x1_atoms = [
        atom
        for atom in result.target_ir.atoms
        if atom.coord_role == "x1"
        and atom.valid_token_ids
        == frozenset({tokenizer.token_id("<|coord_1|>"), tokenizer.token_id("<|coord_3|>")})
    ]

    assert ambiguous_x1_atoms
    _validate(result, tokenizer)


def test_selecting_x1_filters_candidate_objects_before_y1_x2_y2() -> None:
    sample = _sample(
        (
            _object("car", (1, 2, 10, 20), index=0),
            _object("car", (3, 4, 30, 40), index=1),
        )
    )
    result, tokenizer = _build(sample, profile="valid_set")

    ambiguous_x1_index = next(
        index
        for index, atom in enumerate(result.target_ir.atoms)
        if atom.coord_role == "x1"
        and atom.valid_token_ids
        == frozenset({tokenizer.token_id("<|coord_1|>"), tokenizer.token_id("<|coord_3|>")})
    )
    first_y1 = result.target_ir.atoms[ambiguous_x1_index + 1]
    first_x2 = result.target_ir.atoms[ambiguous_x1_index + 2]
    first_y2 = result.target_ir.atoms[ambiguous_x1_index + 3]

    assert first_y1.coord_role == "y1"
    assert len(first_y1.valid_token_ids) == 1
    assert len(first_x2.valid_token_ids) == 1
    assert len(first_y2.valid_token_ids) == 1
    _validate(result, tokenizer)


def test_selected_token_id_matches_rendered_input_ids_target_position() -> None:
    result, tokenizer = _build(_sample((_object("cat", (1, 2, 10, 20)),)))

    assert result.rendered_text.startswith(OBJECT_REF_START_TOKEN)
    for atom in result.target_ir.atoms:
        assert atom.selected_token_id == result.input_ids[atom.target_position]
    _validate(result, tokenizer)


def test_builder_accepts_detection_scene_with_sample_semantic_parity() -> None:
    sample = _sample(
        (
            _object("cat", (1, 2, 10, 20), index=0),
            _object("dog", (3, 4, 30, 40), index=1),
        )
    )
    scene = detection_scene_from_normalized_sample_bridge(
        sample,
        image_reference="/resolved/image.jpg",
    )
    tokenizer = TinyContextTokenizer()

    sample_result = build_teacher_forcing_target(
        sample,
        tokenizer=tokenizer,
        profile="valid_set",
        epoch=3,
        stable_sample_id="scene-parity",
    )
    scene_result = build_teacher_forcing_target(
        scene,
        tokenizer=tokenizer,
        profile="valid_set",
        epoch=3,
        stable_sample_id="scene-parity",
    )

    assert scene_result.ok
    assert scene_result.rendered_text == sample_result.rendered_text
    assert scene_result.input_ids == sample_result.input_ids
    assert scene_result.target_ir == sample_result.target_ir
    _validate(scene_result, tokenizer)


def test_tokenizer_bos_prefix_source_is_recorded_in_target_metadata() -> None:
    result, tokenizer = _build(_sample((_object("cat", (1, 2, 10, 20)),)))

    assert result.ok
    assert result.input_ids[0] == tokenizer.bos_token_id
    assert result.target_ir.metadata["input_prefix_token_source"] == "tokenizer_bos"


def test_configured_input_prefix_source_is_recorded_in_target_metadata() -> None:
    tokenizer = NoBosTokenizer()
    builder = TeacherForcingTargetBuilder(
        tokenizer=tokenizer,
        profile="valid_set",
        input_prefix_token_id=777,
    )

    result = builder.build(
        _sample((_object("cat", (1, 2, 10, 20)),)),
        epoch=0,
        stable_sample_id="configured-prefix",
    )

    assert result.ok
    assert result.input_ids[0] == 777
    assert result.target_ir.metadata["input_prefix_token_source"] == "configured"
    _validate(result, tokenizer)


def test_missing_input_prefix_token_drops_without_synthetic_fallback() -> None:
    result, _tokenizer = _build(
        _sample((_object("cat", (1, 2, 10, 20)),)),
        tokenizer=NoBosTokenizer(),
    )

    assert not result.ok
    assert result.drop_reason == "missing_input_prefix_token"
    assert result.target_ir is None
    assert result.input_ids == ()


def test_missing_detection_list_drops_sample() -> None:
    result, _tokenizer = _build({"image_id": 1, "file_name": "missing.jpg"})

    assert not result.ok
    assert result.drop_reason == "missing_objects"
    assert result.target_ir is None
    assert result.input_ids == ()


def test_explicit_empty_coco_object_list_drops_sample() -> None:
    result, _tokenizer = _build({"objects": [], "image_id": 1, "file_name": "empty.jpg"})

    assert not result.ok
    assert result.drop_reason == "empty_objects"
    assert result.target_ir is None
    assert result.input_ids == ()


@pytest.mark.parametrize(
    "sample",
    [
        {"objects": [{"desc": "cat"}], "image_id": 1, "file_name": "missing-bbox.jpg"},
        {
            "objects": [{"bbox_2d": (1, 2, 10, 20)}],
            "image_id": 1,
            "file_name": "missing-desc.jpg",
        },
        {
            "objects": [{"desc": "cat", "bbox_2d": (1, 2, 10, 20)}],
            "image_id": "not-an-int",
            "file_name": "bad-image-id.jpg",
        },
        {
            "objects": [{"desc": "cat", "bbox_2d": (1, 2, 10, 20)}],
            "image_id": 1,
            "width": "not-an-int",
            "file_name": "bad-width.jpg",
        },
        {
            "objects": [{"desc": None, "bbox_2d": (1, 2, 10, 20)}],
            "image_id": 1,
            "file_name": "bad-desc.jpg",
        },
        {
            "objects": [
                {
                    "desc": "cat",
                    "category_name": [],
                    "bbox_2d": (1, 2, 10, 20),
                }
            ],
            "image_id": 1,
            "file_name": "bad-category-name.jpg",
        },
        {
            "objects": [
                {
                    "desc": "cat",
                    "object_instance_id": {},
                    "bbox_2d": (1, 2, 10, 20),
                }
            ],
            "image_id": 1,
            "file_name": "bad-object-instance-id.jpg",
        },
        {
            "objects": [{"desc": "cat", "bbox_2d": (1, 2, 10, 20)}],
            "image_id": 1,
            "file_name": [],
        },
        {
            "objects": [{"desc": "cat", "bbox_2d": (1, 2, 10, 20)}],
            "image_id": 1,
            "file_name": "bad-metadata-source.jpg",
            "metadata": {"source": [], "split": "train"},
        },
        {
            "objects": [{"desc": "cat", "bbox_2d": (1, 2, 10, 20)}],
            "image_id": 1,
            "file_name": "bad-metadata-split.jpg",
            "metadata": {"source": "unit", "split": None},
        },
        {
            "objects": [{"desc": "cat", "bbox_2d": (1, 2, 10)}],
            "image_id": 1,
            "file_name": "bad-bbox.jpg",
        },
    ],
)
def test_malformed_mapping_samples_drop_as_invalid_sample(sample: dict[str, Any]) -> None:
    result, _tokenizer = _build(sample)

    assert not result.ok
    assert result.drop_reason == "invalid_sample"
    assert result.target_ir is None
    assert result.input_ids == ()


def test_overlength_sample_drops_before_partial_target_construction() -> None:
    sample = _sample((_object("cat", (1, 2, 10, 20)),))

    result, _tokenizer = _build(sample, max_length=2)

    assert not result.ok
    assert result.drop_reason == "overlength"
    assert result.target_ir is None
    assert result.input_ids == ()


def test_rollin_seed_derives_from_base_seed_epoch_sample_id_policy_name_and_version() -> None:
    sample = _sample((_object("cat", (1, 2, 10, 20)),))

    result, _tokenizer = _build(sample, epoch=5, stable_sample_id="stable-7")
    expected = derive_rollin_seed(
        base_seed=17,
        epoch=5,
        stable_sample_id="stable-7",
        policy_name="random_permutation",
        policy_version=1,
    )

    assert result.target_ir.metadata["rollin_policy"] == "random_permutation"
    assert result.target_ir.metadata["rollin_seed"] == expected
    assert result.target_ir.metadata["serialization_policy"] == "marker_delimited"

    changed = derive_rollin_seed(
        base_seed=17,
        epoch=6,
        stable_sample_id="stable-7",
        policy_name="random_permutation",
        policy_version=1,
    )
    assert changed != expected


def test_invalid_description_drops_sample_without_partial_ir() -> None:
    sample = _sample((_object("bad\tcar", (1, 2, 10, 20)),))

    result, _tokenizer = _build(sample)

    assert not result.ok
    assert result.drop_reason == "invalid_description"
    assert result.target_ir is None


def test_non_unique_marker_boundary_mapping_drops_sample_without_partial_ir() -> None:
    sample = _sample((_object("cat", (1, 2, 10, 20)),))

    result, _tokenizer = _build(sample, tokenizer=SplitMarkerTokenizer())

    assert not result.ok
    assert result.drop_reason == "non_unique_marker_boundary"
    assert result.target_ir is None


def test_builder_accepts_explicit_class_api() -> None:
    tokenizer = TinyContextTokenizer()
    builder = TeacherForcingTargetBuilder(tokenizer=tokenizer, profile="valid_set")
    sample = _sample((_object("cat", (1, 2, 10, 20)),))

    result = builder.build(sample, epoch=0, stable_sample_id="class-api")

    assert result.ok
    assert result.target_ir is not None
    _validate(result, tokenizer)
