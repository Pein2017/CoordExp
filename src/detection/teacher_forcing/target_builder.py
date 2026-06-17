"""Build Stage-1 compact teacher-forcing target IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, NamedTuple, Sequence

from src.common.detection_compact_rows import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    END_OF_TEXT_TOKEN,
    IM_END_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from src.detection.data import (
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
    _parse_coordinate_box,
)
from src.detection.scene import DetectionScene, normalized_detection_sample_from_scene
from src.detection.teacher_forcing.description_tokens import (
    DescriptionTokenPath,
    DescriptionTokenizationError,
    tokenize_description_context,
)
from src.detection.teacher_forcing.rollin import (
    DEFAULT_ROLLIN_BASE_SEED,
    ROLLIN_POLICY_NAME,
    ROLLIN_POLICY_VERSION,
    derive_rollin_seed,
    random_permutation_rollin,
)
from src.detection.teacher_forcing.trie import (
    TokenBranch,
    filter_branches_by_selected_token,
    next_token_ids_for_prefix,
    token_roles_for_prefix,
)
from src.detection.template import DetectionSequenceTemplate, TemplateId, get_detection_template
from src.detection.template_contracts import resolve_detection_template_contract
from src.training.teacher_forcing.constants import (
    MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
    TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
)
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole

TeacherForcingBuilderProfile = Literal["hard_sft", "valid_set", "valid_set_marginal"]
_COORD_ROLES = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True)
class TeacherForcingBuildResult:
    rendered_text: str = ""
    input_ids: tuple[int, ...] = ()
    target_ir: TeacherForcingTargetIR | None = None
    drop_reason: str | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.drop_reason is None and self.target_ir is not None


@dataclass(frozen=True)
class _PreparedObject:
    obj: NormalizedDetectionObject
    branch: TokenBranch
    rendered_text: str


class _InputPrefixToken(NamedTuple):
    token_id: int
    source: Literal["configured", "tokenizer_bos"]


@dataclass(frozen=True)
class TeacherForcingTargetBuilder:
    tokenizer: Any
    detection_template_id: TemplateId
    profile: TeacherForcingBuilderProfile = "valid_set"
    base_seed: int = DEFAULT_ROLLIN_BASE_SEED
    policy_name: str = ROLLIN_POLICY_NAME
    policy_version: int = ROLLIN_POLICY_VERSION
    serialization_policy: str = "marker_delimited"
    input_prefix_token_id: int | None = None

    def __post_init__(self) -> None:
        contract = resolve_detection_template_contract(self.detection_template_id)
        if not contract.is_compact:
            raise ValueError(
                "teacher_forcing target IR requires a compact detection template; "
                f"got detection_template.id={self.detection_template_id!r}"
            )

    def build(
        self,
        sample: DetectionScene | NormalizedDetectionSample | Mapping[str, Any],
        *,
        epoch: int,
        stable_sample_id: str | None = None,
        max_length: int | None = None,
    ) -> TeacherForcingBuildResult:
        parsed = _coerce_sample(sample)
        if isinstance(parsed, TeacherForcingBuildResult):
            return parsed
        if not parsed.objects:
            return _drop("empty_objects")

        normalized_profile = _normalize_profile(self.profile)
        sample_id = stable_sample_id or _stable_sample_id(parsed)
        seed = derive_rollin_seed(
            base_seed=self.base_seed,
            epoch=epoch,
            stable_sample_id=sample_id,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
        )

        try:
            detection_template = get_detection_template(self.detection_template_id)
            prepared_by_index = {
                index: _prepare_object(
                    obj,
                    self.tokenizer,
                    detection_template=detection_template,
                )
                for index, obj in enumerate(parsed.objects)
            }
            rollin_indices = random_permutation_rollin(
                tuple(range(len(parsed.objects))),
                seed=seed,
            )
            rendered_text = "".join(
                prepared_by_index[index].rendered_text for index in rollin_indices
            )
            rendered_ids = _encode_rendered_text(self.tokenizer, rendered_text)
            stop_token_id = _single_token_id(self.tokenizer, IM_END_TOKEN)
            expected_rendered_ids = tuple(
                token_id
                for index in rollin_indices
                for token_id in prepared_by_index[index].branch.token_ids
            )
            if rendered_ids != expected_rendered_ids:
                return _drop("tokenization_mismatch")
        except DescriptionTokenizationError as exc:
            return _drop(exc.reason)
        except ValueError:
            return _drop("invalid_sample")

        try:
            prefix = _input_prefix_token_id(self.tokenizer, self.input_prefix_token_id)
        except ValueError as exc:
            return _drop(str(exc), rendered_text=rendered_text)

        input_ids = (prefix.token_id, *rendered_ids)
        input_ids = (*input_ids, stop_token_id)
        if max_length is not None and len(input_ids) > int(max_length):
            return _drop("overlength", rendered_text=rendered_text)

        branches_by_index = {
            index: prepared.branch
            for index, prepared in prepared_by_index.items()
        }
        atoms = _build_atoms(
            rollin_indices=rollin_indices,
            branches_by_index=branches_by_index,
            profile=normalized_profile,
            stop_token_id=stop_token_id,
        )
        target_ir = TeacherForcingTargetIR(
            schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
            atoms=atoms,
            metadata={
                "marginal_scope": MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
                "rollin_policy": self.policy_name,
                "rollin_seed": seed,
                "serialization_policy": self.serialization_policy,
                "detection_template_id": detection_template.template_id,
                "stop_token_text": IM_END_TOKEN,
                "pad_token_text": END_OF_TEXT_TOKEN,
                "parser_mode": "strict_expected",
                "compact_grammar_enabled": True,
                "rollin_policy_version": self.policy_version,
                "stable_sample_id": sample_id,
                "input_prefix_token_source": prefix.source,
                "selected_source_object_indices": tuple(
                    parsed.objects[index].source_object_index for index in rollin_indices
                ),
                "selected_normalized_object_indices": tuple(int(index) for index in rollin_indices),
            },
        )
        return TeacherForcingBuildResult(
            rendered_text=rendered_text,
            input_ids=input_ids,
            target_ir=target_ir,
            metadata={"rollin_seed": seed},
        )


def build_teacher_forcing_target(
    sample: DetectionScene | NormalizedDetectionSample | Mapping[str, Any],
    *,
    tokenizer: Any,
    detection_template_id: TemplateId,
    profile: TeacherForcingBuilderProfile = "valid_set",
    epoch: int,
    stable_sample_id: str | None = None,
    max_length: int | None = None,
    base_seed: int = DEFAULT_ROLLIN_BASE_SEED,
    input_prefix_token_id: int | None = None,
) -> TeacherForcingBuildResult:
    builder = TeacherForcingTargetBuilder(
        tokenizer=tokenizer,
        detection_template_id=detection_template_id,
        profile=profile,
        base_seed=base_seed,
        input_prefix_token_id=input_prefix_token_id,
    )
    return builder.build(
        sample,
        epoch=epoch,
        stable_sample_id=stable_sample_id,
        max_length=max_length,
    )


def _build_atoms(
    *,
    rollin_indices: tuple[int, ...],
    branches_by_index: Mapping[int, TokenBranch],
    profile: Literal["hard_sft", "valid_set"],
    stop_token_id: int,
) -> tuple[SupervisionAtom, ...]:
    atoms: list[SupervisionAtom] = []
    remaining = tuple(rollin_indices)
    rendered_token_position = 0

    for selected_index in rollin_indices:
        selected_branch = branches_by_index[selected_index]
        compatible = tuple(branches_by_index[index] for index in remaining)
        for branch_position, selected_token_id in enumerate(selected_branch.token_ids):
            valid_token_ids = next_token_ids_for_prefix(compatible, position=branch_position)
            allowed_roles = token_roles_for_prefix(compatible, position=branch_position)
            selected_role = selected_branch.token_role_at(branch_position)
            if profile == "hard_sft":
                valid_token_ids = frozenset({selected_token_id})
                allowed_roles = frozenset({selected_role})

            target_position = rendered_token_position + 1
            atoms.append(
                SupervisionAtom(
                    batch_index=0,
                    logit_position=target_position - 1,
                    target_position=target_position,
                    allowed_token_roles=allowed_roles,
                    selected_token_role=selected_role,
                    valid_token_ids=valid_token_ids,
                    selected_token_id=selected_token_id,
                    latent_valid_token_ids=valid_token_ids,
                    coverage_target_weights=None,
                    loss_tags=frozenset({profile}),
                    loss_weight=1.0,
                    coord_role=selected_branch.coord_role_at(branch_position),
                    provenance={
                        "object_index": selected_index,
                        "object_instance_id": selected_branch.object_instance_id,
                        "branch_position": branch_position,
                        "candidate_object_indices": tuple(
                            branch.object_index for branch in compatible
                        ),
                    },
                )
            )
            compatible = filter_branches_by_selected_token(
                compatible,
                position=branch_position,
                selected_token_id=selected_token_id,
            )
            rendered_token_position += 1
        remaining = tuple(index for index in remaining if index != selected_index)

    target_position = rendered_token_position + 1
    atoms.append(
        SupervisionAtom(
            batch_index=0,
            logit_position=target_position - 1,
            target_position=target_position,
            allowed_token_roles=frozenset({TokenRole.STOP}),
            selected_token_role=TokenRole.STOP,
            valid_token_ids=frozenset({int(stop_token_id)}),
            selected_token_id=int(stop_token_id),
            latent_valid_token_ids=frozenset({int(stop_token_id)}),
            coverage_target_weights=None,
            loss_tags=frozenset({profile}),
            loss_weight=1.0,
            coord_role=None,
            provenance={"terminal": IM_END_TOKEN},
        )
    )
    return tuple(atoms)


def _prepare_object(
    obj: NormalizedDetectionObject,
    tokenizer: Any,
    *,
    detection_template: DetectionSequenceTemplate,
) -> _PreparedObject:
    contract = resolve_detection_template_contract(detection_template.template_id)
    token_path = tokenize_description_context(
        obj.desc,
        tokenizer,
        detection_template_id=detection_template.template_id,
    )
    coord_tokens = obj.bbox_2d.tokens
    coord_token_ids = tuple(_single_token_id(tokenizer, token) for token in coord_tokens)
    token_ids: list[int] = [
        token_path.object_ref_start_token_id,
        *token_path.description_token_ids,
    ]
    token_roles: list[TokenRole] = [
        TokenRole.SCHEMA,
        *(TokenRole.TEXT for _ in token_path.description_token_ids),
    ]
    coord_roles: list[str | None] = [
        None,
        *(None for _ in token_path.description_token_ids),
    ]
    if token_path.object_ref_end_token_id is not None:
        token_ids.append(token_path.object_ref_end_token_id)
        token_roles.append(TokenRole.SCHEMA)
        coord_roles.append(None)
    token_ids.append(token_path.bbox_start_token_id)
    token_roles.append(TokenRole.SCHEMA)
    coord_roles.append(None)
    token_ids.extend(coord_token_ids)
    token_roles.extend(TokenRole.COORD for _ in coord_token_ids)
    coord_roles.extend(_COORD_ROLES)
    if contract.include_box_end:
        token_ids.append(_single_token_id(tokenizer, BOX_END_TOKEN))
        token_roles.append(TokenRole.SCHEMA)
        coord_roles.append(None)
    if contract.canonical_final_separator:
        separator_ids = _encode_rendered_text(
            tokenizer,
            contract.canonical_final_separator,
        )
        token_ids.extend(separator_ids)
        token_roles.extend(TokenRole.SCHEMA for _ in separator_ids)
        coord_roles.extend(None for _ in separator_ids)

    rendered_text = detection_template.render_entry(obj)
    _assert_context_matches_rendered_prefix(token_path, rendered_text, tokenizer)
    return _PreparedObject(
        obj=obj,
        branch=TokenBranch(
            object_index=obj.normalized_object_index,
            object_instance_id=obj.object_instance_id,
            token_ids=tuple(token_ids),
            token_roles=tuple(token_roles),
            coord_roles=tuple(coord_roles),
        ),
        rendered_text=rendered_text,
    )


def _assert_context_matches_rendered_prefix(
    token_path: DescriptionTokenPath,
    rendered_text: str,
    tokenizer: Any,
) -> None:
    context = rendered_text[: rendered_text.index(BOX_START_TOKEN) + len(BOX_START_TOKEN)]
    context_ids = _encode_rendered_text(tokenizer, context)
    if context_ids != token_path.context_token_ids:
        raise DescriptionTokenizationError(
            "tokenization_mismatch",
            "rendered compact context tokenization changed",
        )


def _single_token_id(tokenizer: Any, token: str) -> int:
    ids = tuple(int(token_id) for token_id in tokenizer.encode(token, add_special_tokens=False))
    if len(ids) != 1:
        raise ValueError(f"{token!r} must tokenize to a single token")
    return ids[0]


def _encode_rendered_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False))


def _input_prefix_token_id(tokenizer: Any, configured: int | None) -> _InputPrefixToken:
    if configured is not None:
        return _InputPrefixToken(int(configured), "configured")
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        return _InputPrefixToken(int(bos), "tokenizer_bos")
    raise ValueError("missing_input_prefix_token")


def _normalize_profile(profile: str) -> Literal["hard_sft", "valid_set"]:
    normalized = profile.strip().lower().replace("-", "_")
    if normalized in {"valid_set_marginal", "pure_valid_set_marginal"}:
        normalized = "valid_set"
    if normalized not in {"hard_sft", "valid_set"}:
        raise ValueError("teacher-forcing profile must be one of {'hard_sft', 'valid_set'}")
    return normalized  # type: ignore[return-value]


def _coerce_sample(
    sample: DetectionScene | NormalizedDetectionSample | Mapping[str, Any],
) -> NormalizedDetectionSample | TeacherForcingBuildResult:
    if isinstance(sample, DetectionScene):
        return normalized_detection_sample_from_scene(sample)
    if isinstance(sample, NormalizedDetectionSample):
        return sample
    if not isinstance(sample, Mapping):
        return _drop("invalid_sample")
    if "objects" not in sample:
        return _drop("missing_objects")
    try:
        objects_raw = sample["objects"]
        if not isinstance(objects_raw, Sequence) or isinstance(objects_raw, (str, bytes)):
            return _drop("invalid_sample")
        if not objects_raw:
            return _drop("empty_objects")
        objects = tuple(
            _object_from_mapping(obj, index=index)
            for index, obj in enumerate(objects_raw)
        )
        image_id = int(sample.get("image_id", 0))
        file_name = _optional_str(sample, "file_name", default="", path="file_name")
        width = int(sample.get("width", 0))
        height = int(sample.get("height", 0))
        source, split = _metadata_source_split(sample)
    except (KeyError, TypeError, ValueError, OverflowError):
        return _drop("invalid_sample")
    return NormalizedDetectionSample(
        images=(file_name,) if file_name else (),
        objects=objects,
        width=width,
        height=height,
        image_id=image_id,
        file_name=file_name,
        metadata=DetectionMetadata(
            source=source,
            split=split,
        ),
        object_ordering=ObjectOrderingPlan.sorted().with_realized(
            tuple(obj.source_object_index for obj in objects)
        ),
    )


def _object_from_mapping(obj: Any, *, index: int) -> NormalizedDetectionObject:
    if not isinstance(obj, Mapping):
        raise TypeError("object must be a mapping")
    desc = _require_str(obj["desc"], path=f"objects[{index}].desc")
    bbox = _parse_coordinate_box(obj["bbox_2d"], path=f"objects[{index}].bbox_2d")
    source_index = int(obj.get("source_object_index", index))
    return NormalizedDetectionObject(
        normalized_object_index=index,
        source_object_index=source_index,
        object_instance_id=_optional_str(
            obj,
            "object_instance_id",
            default=f"mapping:src-{source_index}",
            path=f"objects[{index}].object_instance_id",
        ),
        desc=desc,
        bbox_2d=bbox,
        category_id=int(obj.get("category_id", 0)),
        category_name=_optional_str(
            obj,
            "category_name",
            default=desc,
            path=f"objects[{index}].category_name",
        ),
        coco_ann_id=int(obj.get("coco_ann_id", index)),
        object_id=_optional_nullable_str(
            obj,
            "object_id",
            path=f"objects[{index}].object_id",
        ),
    )


def _metadata_source_split(sample: Mapping[str, Any]) -> tuple[str, str]:
    if "metadata" not in sample:
        return (
            _optional_str(sample, "source", default="mapping", path="source"),
            _optional_str(sample, "split", default="unknown", path="split"),
        )

    metadata = sample["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    return (
        _optional_str(metadata, "source", default="mapping", path="metadata.source"),
        _optional_str(metadata, "split", default="unknown", path="metadata.split"),
    )


def _optional_str(
    mapping: Mapping[str, Any],
    key: str,
    *,
    default: str,
    path: str,
) -> str:
    if key not in mapping:
        return default
    return _require_str(mapping[key], path=path)


def _optional_nullable_str(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> str | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _require_str(mapping[key], path=path)


def _require_str(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _stable_sample_id(sample: NormalizedDetectionSample) -> str:
    if sample.image_id:
        return str(sample.image_id)
    if sample.file_name:
        return sample.file_name
    return "|".join(obj.object_instance_id for obj in sample.objects)


def _drop(reason: str, *, rendered_text: str = "") -> TeacherForcingBuildResult:
    return TeacherForcingBuildResult(
        rendered_text=rendered_text,
        input_ids=(),
        target_ir=None,
        drop_reason=reason,
    )


__all__ = [
    "TeacherForcingBuildResult",
    "TeacherForcingBuilderProfile",
    "TeacherForcingTargetBuilder",
    "build_teacher_forcing_target",
]
