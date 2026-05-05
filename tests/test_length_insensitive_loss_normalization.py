from __future__ import annotations

from dataclasses import replace
import re

import pytest

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.objective import (
    SemanticRole,
    normalize_recursive_detection_token_losses,
    prepare_detection_training_example,
)
from src.detection.template import CompactFullTemplate, Stage1JsonPrettyTemplate

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


class SpecialTokenAwareTokenizer:
    eos_token = "<|im_end|>"

    def __init__(self) -> None:
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
            self._id_to_token.setdefault(token_id, token_text)
            input_ids.append(token_id)
            offsets.append((cursor, token_end))
            cursor = token_end

        return {
            "input_ids": input_ids,
            "offset_mapping": offsets,
        }

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
        coco_ann_id=8000 + source_index,
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
        image_id=19,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.random_permutation(
            seed=31,
            seed_source="unit-test",
        ).with_realized(realized),
    )


BASE_ROLE_LOSSES = {
    SemanticRole.DESC_IDENTITY: 1.0,
    SemanticRole.BBOX_COORD: 3.0,
    SemanticRole.ENTRY_TRIE_DECISION: 2.0,
    SemanticRole.OBJECT_CONTROL: 2.0,
    SemanticRole.SEPARATOR_CONTINUE: 2.10,
    SemanticRole.TERMINAL_STOP: 2.10,
    SemanticRole.SCHEMA_CONTROL: 0.0,
    SemanticRole.CHAT_STOP: 2.10,
}


def _prepare(template: object, *, normalization: str) -> object:
    return prepare_detection_training_example(
        _sample(
            _object(
                normalized_index=0,
                source_index=7,
                instance_id="img-19:ann-901:src-7",
                desc="cat",
                coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
            ),
            _object(
                normalized_index=1,
                source_index=3,
                instance_id="img-19:ann-902:src-3",
                desc="dog",
                coords=("<|coord_110|>", "<|coord_120|>", "<|coord_130|>", "<|coord_140|>"),
            ),
        ),
        template=template,
        tokenizer=SpecialTokenAwareTokenizer(),
        mode="random_permutation_et_rmp_ce",
        state_weighting="uniform_permutation",
        normalization=normalization,
    )


def _semantic_losses(prepared: object) -> dict[int, float]:
    assert prepared.recursive_detection_targets is not None
    schema_loss = _schema_equivalent_image_loss(prepared.recursive_detection_targets)
    return {
        target.position: (
            schema_loss
            if target.semantic_role is SemanticRole.SCHEMA_CONTROL
            else BASE_ROLE_LOSSES[target.semantic_role]
        )
        for target in prepared.recursive_detection_targets.token_targets
    }


def _schema_equivalent_image_loss(recursive_targets: object) -> float:
    object_role_weights = {
        SemanticRole.DESC_IDENTITY: 0.35,
        SemanticRole.BBOX_COORD: 0.45,
        SemanticRole.ENTRY_TRIE_DECISION: 0.15,
        SemanticRole.OBJECT_CONTROL: 0.05,
    }
    object_losses: list[float] = []
    object_atoms: dict[str, list[object]] = {}
    for atom in recursive_targets.loss_atoms:
        if atom.object_instance_id is None:
            continue
        object_atoms.setdefault(atom.object_instance_id, []).append(atom)

    for atoms in object_atoms.values():
        numerator = 0.0
        denominator = 0.0
        for atom in atoms:
            weight = object_role_weights.get(atom.semantic_role)
            if weight is None:
                continue
            numerator += weight * BASE_ROLE_LOSSES[atom.semantic_role]
            denominator += weight
        object_losses.append(numerator / denominator)

    boundary_terms: list[tuple[float, float]] = []
    if any(
        atom.semantic_role is SemanticRole.SEPARATOR_CONTINUE
        for atom in recursive_targets.loss_atoms
    ):
        boundary_terms.append((0.50, BASE_ROLE_LOSSES[SemanticRole.SEPARATOR_CONTINUE]))
    if any(
        atom.semantic_role in {SemanticRole.TERMINAL_STOP, SemanticRole.CHAT_STOP}
        for atom in recursive_targets.loss_atoms
    ):
        boundary_terms.append((0.50, BASE_ROLE_LOSSES[SemanticRole.TERMINAL_STOP]))

    object_component = sum(object_losses) / len(object_losses)
    boundary_component = sum(weight * loss for weight, loss in boundary_terms) / sum(
        weight for weight, _ in boundary_terms
    )
    return (1.00 * object_component + 0.30 * boundary_component) / (1.00 + 0.30)


def test_profiles_are_distinct_and_report_different_normalization_diagnostics() -> None:
    legacy_prepared = _prepare(
        Stage1JsonPrettyTemplate(),
        normalization="legacy_row_mean_equivalence",
    )
    semantic_prepared = _prepare(
        Stage1JsonPrettyTemplate(),
        normalization="semantic_image_bucket_balanced",
    )

    legacy = normalize_recursive_detection_token_losses(
        legacy_prepared.recursive_detection_targets,
        _semantic_losses(legacy_prepared),
    )
    semantic = normalize_recursive_detection_token_losses(
        semantic_prepared.recursive_detection_targets,
        _semantic_losses(semantic_prepared),
    )

    assert legacy.profile_id == "legacy_row_mean_equivalence"
    assert semantic.profile_id == "semantic_image_bucket_balanced"
    assert legacy.diagnostics.profile_id != semantic.diagnostics.profile_id
    assert legacy.diagnostics.state_weight_sum > 0.0
    assert semantic.diagnostics.atom_count == len(
        semantic_prepared.recursive_detection_targets.loss_atoms
    )
    assert semantic.diagnostics.gt_count_bucket == "0-3"


def test_semantic_profile_equalizes_pretty_json_and_compact_examples() -> None:
    pretty_prepared = _prepare(
        Stage1JsonPrettyTemplate(),
        normalization="semantic_image_bucket_balanced",
    )
    compact_prepared = _prepare(
        CompactFullTemplate(),
        normalization="semantic_image_bucket_balanced",
    )

    pretty = normalize_recursive_detection_token_losses(
        pretty_prepared.recursive_detection_targets,
        _semantic_losses(pretty_prepared),
    )
    compact = normalize_recursive_detection_token_losses(
        compact_prepared.recursive_detection_targets,
        _semantic_losses(compact_prepared),
    )

    assert pretty_prepared.recursive_detection_targets is not None
    assert compact_prepared.recursive_detection_targets is not None
    assert len(pretty_prepared.recursive_detection_targets.token_targets) != len(
        compact_prepared.recursive_detection_targets.token_targets
    )
    assert pretty.profile_id == "semantic_image_bucket_balanced"
    assert compact.profile_id == "semantic_image_bucket_balanced"
    assert pretty.normalized_loss == pytest.approx(compact.normalized_loss)


def test_compact_multi_positive_roles_stay_semantic_not_token_surface_specific() -> None:
    prepared = _prepare(
        CompactFullTemplate(),
        normalization="semantic_image_bucket_balanced",
    )

    assert prepared.recursive_detection_targets is not None
    semantic_roles = {
        target.position: target.semantic_role
        for target in prepared.recursive_detection_targets.token_targets
    }

    assert any(role is SemanticRole.ENTRY_TRIE_DECISION for role in semantic_roles.values())
    assert any(role is SemanticRole.BBOX_COORD for role in semantic_roles.values())
    assert any(role is SemanticRole.OBJECT_CONTROL for role in semantic_roles.values())
