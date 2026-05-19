"""Build Stage-1 compact_full teacher-forcing target IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from src.common.detection_compact_rows import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
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
from src.detection.template import CompactFullTemplate
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


@dataclass(frozen=True)
class TeacherForcingTargetBuilder:
    tokenizer: Any
    profile: TeacherForcingBuilderProfile = "valid_set"
    base_seed: int = DEFAULT_ROLLIN_BASE_SEED
    policy_name: str = ROLLIN_POLICY_NAME
    policy_version: int = ROLLIN_POLICY_VERSION
    serialization_policy: str = "marker_delimited"
    input_prefix_token_id: int | None = None

    def build(
        self,
        sample: NormalizedDetectionSample | Mapping[str, Any],
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
            prepared_by_index = {
                index: _prepare_object(obj, self.tokenizer)
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

        prefix_id = _input_prefix_token_id(self.tokenizer, self.input_prefix_token_id)
        input_ids = (prefix_id, *rendered_ids)
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
        )
        target_ir = TeacherForcingTargetIR(
            schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
            atoms=atoms,
            metadata={
                "marginal_scope": MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
                "rollin_policy": self.policy_name,
                "rollin_seed": seed,
                "serialization_policy": self.serialization_policy,
                "rollin_policy_version": self.policy_version,
                "stable_sample_id": sample_id,
                "selected_source_object_indices": tuple(
                    parsed.objects[index].source_object_index for index in rollin_indices
                ),
            },
        )
        return TeacherForcingBuildResult(
            rendered_text=rendered_text,
            input_ids=input_ids,
            target_ir=target_ir,
            metadata={"rollin_seed": seed},
        )


def build_teacher_forcing_target(
    sample: NormalizedDetectionSample | Mapping[str, Any],
    *,
    tokenizer: Any,
    profile: TeacherForcingBuilderProfile = "valid_set",
    epoch: int,
    stable_sample_id: str | None = None,
    max_length: int | None = None,
    base_seed: int = DEFAULT_ROLLIN_BASE_SEED,
) -> TeacherForcingBuildResult:
    builder = TeacherForcingTargetBuilder(
        tokenizer=tokenizer,
        profile=profile,
        base_seed=base_seed,
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

    return tuple(atoms)


def _prepare_object(obj: NormalizedDetectionObject, tokenizer: Any) -> _PreparedObject:
    token_path = tokenize_description_context(obj.desc, tokenizer)
    coord_tokens = obj.bbox_2d.tokens
    coord_token_ids = tuple(_single_token_id(tokenizer, token) for token in coord_tokens)
    token_ids = (
        token_path.object_ref_start_token_id,
        *token_path.description_token_ids,
        token_path.bbox_start_token_id,
        *coord_token_ids,
    )
    token_roles = (
        TokenRole.SCHEMA,
        *(TokenRole.TEXT for _ in token_path.description_token_ids),
        TokenRole.SCHEMA,
        *(TokenRole.COORD for _ in coord_token_ids),
    )
    coord_roles = (
        None,
        *(None for _ in token_path.description_token_ids),
        None,
        *_COORD_ROLES,
    )
    rendered_text = CompactFullTemplate().render_entry(obj)
    _assert_context_matches_rendered_prefix(token_path, rendered_text, tokenizer)
    return _PreparedObject(
        obj=obj,
        branch=TokenBranch(
            object_index=obj.normalized_object_index,
            object_instance_id=obj.object_instance_id,
            token_ids=token_ids,
            token_roles=token_roles,
            coord_roles=coord_roles,
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
            "rendered compact_full context tokenization changed",
        )


def _single_token_id(tokenizer: Any, token: str) -> int:
    ids = tuple(int(token_id) for token_id in tokenizer.encode(token, add_special_tokens=False))
    if len(ids) != 1:
        raise ValueError(f"{token!r} must tokenize to a single token")
    return ids[0]


def _encode_rendered_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False))


def _input_prefix_token_id(tokenizer: Any, configured: int | None) -> int:
    if configured is not None:
        return int(configured)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        return int(bos)
    return 0


def _normalize_profile(profile: str) -> Literal["hard_sft", "valid_set"]:
    normalized = profile.strip().lower().replace("-", "_")
    if normalized == "valid_set_marginal":
        normalized = "valid_set"
    if normalized not in {"hard_sft", "valid_set"}:
        raise ValueError("teacher-forcing profile must be one of {'hard_sft', 'valid_set'}")
    return normalized  # type: ignore[return-value]


def _coerce_sample(
    sample: NormalizedDetectionSample | Mapping[str, Any],
) -> NormalizedDetectionSample | TeacherForcingBuildResult:
    if isinstance(sample, NormalizedDetectionSample):
        return sample
    if not isinstance(sample, Mapping):
        return _drop("invalid_sample")
    if "objects" not in sample:
        return _drop("missing_objects")
    objects_raw = sample["objects"]
    if not isinstance(objects_raw, Sequence) or isinstance(objects_raw, (str, bytes)):
        return _drop("invalid_sample")
    if not objects_raw:
        return _drop("empty_objects")
    try:
        objects = tuple(
            _object_from_mapping(obj, index=index)
            for index, obj in enumerate(objects_raw)
        )
    except (TypeError, ValueError):
        return _drop("invalid_sample")
    image_id = int(sample.get("image_id", 0))
    file_name = str(sample.get("file_name", ""))
    return NormalizedDetectionSample(
        images=(file_name,) if file_name else (),
        objects=objects,
        width=int(sample.get("width", 0)),
        height=int(sample.get("height", 0)),
        image_id=image_id,
        file_name=file_name,
        metadata=DetectionMetadata(
            source=str(sample.get("source", "mapping")),
            split=str(sample.get("split", "unknown")),
        ),
        object_ordering=ObjectOrderingPlan.sorted().with_realized(
            tuple(obj.source_object_index for obj in objects)
        ),
    )


def _object_from_mapping(obj: Any, *, index: int) -> NormalizedDetectionObject:
    if not isinstance(obj, Mapping):
        raise TypeError("object must be a mapping")
    bbox_raw = obj["bbox_2d"]
    if not isinstance(bbox_raw, Sequence) or isinstance(bbox_raw, (str, bytes)):
        raise TypeError("bbox_2d must be a sequence")
    bbox = CoordinateTokenBox(*tuple(bbox_raw))
    source_index = int(obj.get("source_object_index", index))
    return NormalizedDetectionObject(
        normalized_object_index=index,
        source_object_index=source_index,
        object_instance_id=str(obj.get("object_instance_id", f"mapping:src-{source_index}")),
        desc=str(obj["desc"]),
        bbox_2d=bbox,
        category_id=int(obj.get("category_id", 0)),
        category_name=str(obj.get("category_name", obj["desc"])),
        coco_ann_id=int(obj.get("coco_ann_id", index)),
        object_id=obj.get("object_id"),
    )


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
