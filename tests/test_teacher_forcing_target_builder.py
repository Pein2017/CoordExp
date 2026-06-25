from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import pytest
import torch

from src.common.detection_compact_rows import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IM_END_TOKEN,
    OBJECT_REF_END_TOKEN,
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
from src.detection.template import get_detection_template
from src.detection.template_contracts import COMPACT_TEMPLATE_IDS
from src.training.objectives.runner import ObjectiveRunner
from src.training.objectives.types import ObjectiveSpec
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import TeacherForcingTargetDistribution
from src.training.supervision.spans import SupervisionSpan
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
            OBJECT_REF_END_TOKEN: 12,
            BOX_END_TOKEN: 13,
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

    def token_text(self, token_id: int) -> str:
        for token, value in self._ids.items():
            if value == int(token_id):
                return token
        raise KeyError(token_id)

    def role_vocab(self) -> RoleVocab:
        return RoleVocab(
            schema_token_ids=frozenset(
                {
                    self.token_id(OBJECT_REF_START_TOKEN),
                    self.token_id(OBJECT_REF_END_TOKEN),
                    self.token_id(BOX_START_TOKEN),
                    self.token_id(BOX_END_TOKEN),
                    self.token_id("\n"),
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
            for special in (
                OBJECT_REF_START_TOKEN,
                OBJECT_REF_END_TOKEN,
                BOX_START_TOKEN,
                BOX_END_TOKEN,
                IM_END_TOKEN,
            ):
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


class NonmonotonicCoordTokenizer(TinyContextTokenizer):
    def _id_for(self, token: str) -> int:
        coord = re.fullmatch(r"<\|coord_(\d{1,3})\|>", token)
        if coord is not None:
            if token not in self._ids:
                self._ids[token] = 50000 - int(coord.group(1))
            return self._ids[token]
        return super()._id_for(token)


class CountingCoordTokenizer(TinyContextTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.coord_single_token_encode_calls = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if re.fullmatch(r"<\|coord_\d{1,3}\|>", text):
            self.coord_single_token_encode_calls += 1
        return super().encode(text, add_special_tokens=add_special_tokens)


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
    detection_template_id: str = "compact",
    profile: str = "valid_set",
    epoch: int = 3,
    stable_sample_id: str = "sample-42",
    max_length: int | None = None,
    policy_name: str = "random_permutation",
    tokenizer: TinyContextTokenizer | None = None,
):
    tok = tokenizer or TinyContextTokenizer()
    result = build_teacher_forcing_target(
        sample,
        tokenizer=tok,
        detection_template_id=detection_template_id,  # type: ignore[arg-type]
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


def _supervision_batch_for_result(result, *, sample_id: str) -> SupervisionBatch:
    return SupervisionBatch(
        spans=(
            SupervisionSpan(
                sample_id=sample_id,
                role="schema",
                label_positions=tuple(
                    atom.target_position for atom in result.target_ir.atoms
                ),
                distribution=TeacherForcingTargetDistribution(
                    target_ir=result.target_ir
                ),
            ),
        ),
        batch_id="target-builder-integration",
    )


@pytest.mark.parametrize("template_id", COMPACT_TEMPLATE_IDS)
def test_teacher_forcing_target_builder_uses_detection_template_contract(
    template_id: str,
) -> None:
    sample = _sample((_object("cat", (1, 2, 10, 20)),))
    result, tokenizer = _build(sample, detection_template_id=template_id)
    rendered = get_detection_template(template_id).render_assistant(sample)

    assert result.ok
    assert result.rendered_text == rendered.text
    assert result.target_ir.metadata["detection_template_id"] == template_id

    selected_texts = tuple(
        tokenizer.token_text(atom.selected_token_id)
        for atom in result.target_ir.atoms[:-1]
    )
    assert OBJECT_REF_START_TOKEN in selected_texts
    assert BOX_START_TOKEN in selected_texts
    assert tuple(text for text in selected_texts if text.startswith("<|coord_")) == (
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_10|>",
        "<|coord_20|>",
    )
    if template_id in {
        "compact_object_closed",
        "compact_object_box_closed",
        "compact_object_box_closed_lines",
    }:
        assert OBJECT_REF_END_TOKEN in selected_texts
    else:
        assert OBJECT_REF_END_TOKEN not in selected_texts
    if template_id in {"compact", "compact_object_closed"}:
        assert BOX_END_TOKEN not in selected_texts
    else:
        assert BOX_END_TOKEN in selected_texts
    assert ("\n" in selected_texts) is (
        template_id == "compact_object_box_closed_lines"
    )
    _validate(result, tokenizer)


def test_teacher_forcing_target_builder_normalizes_object_field_order() -> None:
    sample = _sample((_object("cat", (1, 2, 10, 20)),))
    tokenizer = TinyContextTokenizer()
    builder = TeacherForcingTargetBuilder(
        tokenizer=tokenizer,
        detection_template_id="compact_object_box_closed",
        object_field_order=" Geometry_First ",
    )

    result = builder.build(
        sample,
        epoch=3,
        stable_sample_id="normalized-field-order",
    )
    rendered = get_detection_template("compact_object_box_closed").render_assistant(
        sample,
        object_field_order="geometry_first",
    )

    assert builder.object_field_order == "geometry_first"
    assert result.ok
    assert result.rendered_text == rendered.text
    assert result.rendered_text.startswith(BOX_START_TOKEN)
    assert result.target_ir.metadata["object_field_order"] == "geometry_first"
    _validate(result, tokenizer)


def test_hard_sft_profile_emits_singleton_valid_token_ids() -> None:
    result, tokenizer = _build(_sample((_object("cat", (1, 2, 10, 20)),)), profile="hard_sft")

    assert result.ok
    assert result.target_ir is not None
    assert result.target_ir.metadata["marginal_scope"] == "sampled_path_next_token"
    assert all(atom.valid_token_ids == frozenset({atom.selected_token_id}) for atom in result.target_ir.atoms)
    _validate(result, tokenizer)


def test_hard_sft_marks_continue_and_stop_boundaries() -> None:
    result, tokenizer = _build(
        _sample(
            (
                _object("cat", (1, 2, 10, 20), index=0),
                _object("dog", (3, 4, 30, 40), index=1),
            )
        ),
        profile="hard_sft",
        policy_name="sorted",
    )

    assert result.ok
    boundary_atoms = [
        atom
        for atom in result.target_ir.atoms
        if atom.provenance.get("continuation_boundary") is True
    ]
    continue_atoms = [
        atom
        for atom in boundary_atoms
        if atom.provenance["continuation_target"] == "continue"
    ]
    stop_atom = next(
        atom
        for atom in boundary_atoms
        if atom.provenance["continuation_target"] == "stop"
    )

    assert len(boundary_atoms) == 3
    assert [atom.provenance["remaining_object_count"] for atom in continue_atoms] == [2, 1]
    assert all(
        atom.provenance["continuation_token_ids"]
        == (tokenizer.token_id(OBJECT_REF_START_TOKEN),)
        for atom in continue_atoms
    )
    assert all(
        atom.provenance["stop_token_id"] == tokenizer.token_id(IM_END_TOKEN)
        for atom in boundary_atoms
    )
    assert stop_atom.selected_token_role is TokenRole.STOP
    assert stop_atom.selected_token_id == tokenizer.token_id(IM_END_TOKEN)
    assert stop_atom.provenance["terminal"] == IM_END_TOKEN
    assert stop_atom.provenance["remaining_object_count"] == 0
    assert stop_atom.provenance["continuation_token_ids"] == (
        tokenizer.token_id(OBJECT_REF_START_TOKEN),
    )
    _validate(result, tokenizer)


def test_coord_tail_atoms_carry_selected_bbox_for_geometry() -> None:
    result, tokenizer = _build(_sample((_object("cat", (1, 2, 10, 20)),)))

    assert result.ok
    x2_atom = next(atom for atom in result.target_ir.atoms if atom.coord_role == "x2")
    y2_atom = next(atom for atom in result.target_ir.atoms if atom.coord_role == "y2")

    assert x2_atom.provenance["selected_bbox_xyxy"] == (1, 2, 10, 20)
    assert x2_atom.provenance["bbox_positive_area"] is True
    assert x2_atom.provenance["geometry_axis"] == "x"
    assert x2_atom.provenance["geometry_threshold_bin"] == 1
    assert y2_atom.provenance["selected_bbox_xyxy"] == (1, 2, 10, 20)
    assert y2_atom.provenance["bbox_positive_area"] is True
    assert y2_atom.provenance["geometry_axis"] == "y"
    assert y2_atom.provenance["geometry_threshold_bin"] == 2
    assert x2_atom.provenance["bbox_positive_area_valid_token_ids"]
    assert x2_atom.provenance["bbox_positive_area_invalid_token_ids"]
    assert y2_atom.provenance["bbox_positive_area_valid_token_ids"]
    assert y2_atom.provenance["bbox_positive_area_invalid_token_ids"]
    assert tokenizer.token_id("<|coord_2|>") in x2_atom.provenance[
        "bbox_positive_area_valid_token_ids"
    ]
    assert tokenizer.token_id("<|coord_1|>") in x2_atom.provenance[
        "bbox_positive_area_invalid_token_ids"
    ]
    assert tokenizer.token_id("<|coord_3|>") in y2_atom.provenance[
        "bbox_positive_area_valid_token_ids"
    ]
    assert tokenizer.token_id("<|coord_2|>") in y2_atom.provenance[
        "bbox_positive_area_invalid_token_ids"
    ]
    _validate(result, tokenizer)


def test_geometry_valid_invalid_coord_ids_use_coord_bin_sequence_not_token_id_order() -> None:
    tokenizer = NonmonotonicCoordTokenizer()
    result, _tokenizer = _build(
        _sample((_object("cat", (1, 2, 10, 20)),)),
        tokenizer=tokenizer,
    )

    assert result.ok
    x2_atom = next(atom for atom in result.target_ir.atoms if atom.coord_role == "x2")
    valid_ids = x2_atom.provenance["bbox_positive_area_valid_token_ids"]
    invalid_ids = x2_atom.provenance["bbox_positive_area_invalid_token_ids"]

    assert tokenizer.token_id("<|coord_999|>") < tokenizer.token_id("<|coord_0|>")
    assert tokenizer.token_id("<|coord_999|>") in valid_ids
    assert tokenizer.token_id("<|coord_2|>") in valid_ids
    assert tokenizer.token_id("<|coord_1|>") in invalid_ids
    assert tokenizer.token_id("<|coord_0|>") in invalid_ids
    _validate(result, tokenizer)


def test_real_builder_ir_runs_objective_with_bbox_positive_area_enabled() -> None:
    result, tokenizer = _build(_sample((_object("cat", (1, 2, 10, 20)),)))

    assert result.ok
    input_ids = torch.tensor([result.input_ids], dtype=torch.long)
    logits = torch.zeros(
        (1, len(result.input_ids), max(result.input_ids) + 1),
        dtype=torch.float32,
    )
    objective_result = ObjectiveRunner().run(
        logits=logits,
        supervision=_supervision_batch_for_result(
            result,
            sample_id="real-builder-positive-area",
        ),
        objectives=(
            ObjectiveSpec(
                "teacher_forcing",
                config={
                    "input_ids": input_ids,
                    "role_vocab": tokenizer.role_vocab(),
                    "bbox_positive_area_weight": 0.25,
                },
            ),
        ),
        sample_id_to_batch_index={"real-builder-positive-area": 0},
    )

    loss = objective_result.objectives["teacher_forcing"].loss
    assert torch.isfinite(loss)


def test_prepare_object_reuses_build_level_coord_bin_token_ids() -> None:
    tokenizer = CountingCoordTokenizer()
    sample = _sample(
        (
            _object("cat", (1, 2, 10, 20), index=0),
            _object("dog", (3, 4, 30, 40), index=1),
        )
    )

    result, _tokenizer = _build(
        sample,
        tokenizer=tokenizer,
        policy_name="sorted",
    )

    assert result.ok
    assert tokenizer.coord_single_token_encode_calls == 1008


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
        detection_template_id="compact",
        profile="valid_set",
        epoch=3,
        stable_sample_id="scene-parity",
    )
    scene_result = build_teacher_forcing_target(
        scene,
        tokenizer=tokenizer,
        detection_template_id="compact",
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
        detection_template_id="compact",
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


def test_sorted_rollin_policy_preserves_source_order() -> None:
    sample = _sample(
        (
            _object("cat", (1, 2, 10, 20), index=0, source_index=20),
            _object("dog", (3, 4, 30, 40), index=1, source_index=10),
        )
    )

    result, _tokenizer = _build(sample, policy_name="sorted")

    assert result.target_ir is not None
    assert result.target_ir.metadata["rollin_policy"] == "sorted"
    assert result.target_ir.metadata["selected_normalized_object_indices"] == (0, 1)
    assert result.target_ir.metadata["selected_source_object_indices"] == (20, 10)


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
    builder = TeacherForcingTargetBuilder(
        tokenizer=tokenizer,
        detection_template_id="compact",
        profile="valid_set",
    )
    sample = _sample((_object("cat", (1, 2, 10, 20)),))

    result = builder.build(sample, epoch=0, stable_sample_id="class-api")

    assert result.ok
    assert result.target_ir is not None
    _validate(result, tokenizer)
