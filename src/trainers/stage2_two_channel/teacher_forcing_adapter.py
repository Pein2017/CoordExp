from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence, cast

import torch

from src.training.teacher_forcing.constants import (
    MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
    TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
)
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.validation import validate_target_ir
from src.training.teacher_forcing.vocab import RoleVocab
from src.trainers.stage2_two_channel.residual_set import CorrectionEvent

_COORD_ROLES = ("x1", "y1", "x2", "y2")
_ROLLOUT_ZERO_POSITIVE_POLICIES = frozenset(
    {"duplicate_certified", "pseudo_positive", "shielded"}
)


@dataclass(frozen=True, slots=True)
class Stage2TeacherForcingObjectTarget:
    object_source: str
    positive_policy: str
    coord_positions: tuple[int, int, int, int]
    gt_bins: tuple[int, int, int, int]
    loss_weight: float = 1.0
    gt_index: int | None = None
    rollout_index: int | None = None
    bbox_group_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coord_positions",
            _tuple4(self.coord_positions, field_name="coord_positions"),
        )
        object.__setattr__(
            self,
            "gt_bins",
            _tuple4(self.gt_bins, field_name="gt_bins"),
        )


def build_stage2_teacher_forcing_target_ir(
    *,
    input_ids: torch.Tensor,
    batch_index: int,
    meta: Mapping[str, Any],
    coord_token_ids: Sequence[int],
    object_targets: Sequence[Stage2TeacherForcingObjectTarget] | None = None,
) -> TeacherForcingTargetIR:
    """Build Stage-2's subordinate bridge into the shared target IR.

    Stage-2 still owns rollout construction, duplicate filtering, and FN triage.
    This adapter only translates the already-decided positive supervision
    targets into shared next-token atoms.
    """

    coord_vocab = frozenset(int(token_id) for token_id in coord_token_ids)
    atoms: list[SupervisionAtom] = []
    channel = str(meta.get("stage2_channel", "") or "").upper() or "B"
    prompt_len = _int_meta(meta, "prompt_len", default=0)
    prefix_len = _int_meta(meta, "prefix_len", default=0)
    encoded_len = _int_meta(meta, "encoded_len", default=int(input_ids.shape[-1]))
    train_len = _int_meta(meta, "train_len", default=max(0, encoded_len - prompt_len))

    if object_targets is None:
        object_targets = _object_targets_from_legacy_meta(meta)

    for target in object_targets:
        if _is_zero_positive_target(target):
            continue
        source = str(target.object_source)
        tags = {"stage2", f"channel_{channel.lower()}", source}
        if source == "fn":
            tags.add("fn")
        if source == "recovered_fn":
            tags.update({"fn", "recovered_fn"})
        provenance = {
            "stage": "stage2",
            "channel": channel,
            "object_source": source,
            "positive_policy": target.positive_policy,
        }
        if target.gt_index is not None:
            provenance["gt_index"] = int(target.gt_index)
        if target.rollout_index is not None:
            provenance["rollout_index"] = int(target.rollout_index)
        if target.bbox_group_index is not None:
            provenance["bbox_group_index"] = int(target.bbox_group_index)
        atoms.extend(
            _coord_atoms_for_target(
                input_ids=input_ids,
                batch_index=batch_index,
                target=target,
                coord_vocab=coord_vocab,
                loss_tags=frozenset(tags),
                provenance=provenance,
            )
        )

    for desc_position in _absolute_tail_positions(
        meta.get("tail_desc_pos", ()),
        prompt_len=prompt_len,
        prefix_len=prefix_len,
    ):
        if not _position_is_trainable(
            desc_position,
            prompt_len=prompt_len,
            encoded_len=encoded_len,
            train_len=train_len,
        ):
            continue
        token_id = int(input_ids[batch_index, desc_position].item())
        atoms.append(
            SupervisionAtom(
                batch_index=batch_index,
                logit_position=desc_position - 1,
                target_position=desc_position,
                allowed_token_roles=frozenset({TokenRole.TEXT}),
                selected_token_role=TokenRole.TEXT,
                valid_token_ids=frozenset({token_id}),
                selected_token_id=token_id,
                latent_valid_token_ids=frozenset({token_id}),
                coverage_target_weights=None,
                loss_tags=frozenset(
                    {"stage2", f"channel_{channel.lower()}", "description"}
                ),
                loss_weight=1.0,
                coord_role=None,
                provenance={
                    "stage": "stage2",
                    "channel": channel,
                    "object_source": "description",
                    "context": "gt_context" if channel == "A" else "rollout_context",
                },
            )
        )

    metadata = {
        "stage": "stage2",
        "stage2_channel": channel,
        "stage2_context": (
            "gt_context"
            if channel == "A"
            else str(meta.get("rollout_context", "rollout_context"))
        ),
        "marginal_scope": MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
        "prompt_len": prompt_len,
        "prefix_len": prefix_len,
        "train_len": train_len,
        "encoded_len": encoded_len,
    }
    stop_token_id = meta.get("stop_token_id")
    if stop_token_id is not None:
        metadata["stop_token_id"] = int(stop_token_id)
    return TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=tuple(atoms),
        metadata=metadata,
    )


def build_residual_set_target_ir(
    *,
    input_ids: torch.Tensor,
    batch_index: int,
    events: Sequence[CorrectionEvent],
    role_vocab: RoleVocab,
    position_space: str = "batch_tensor",
) -> TeacherForcingTargetIR:
    """Translate residual-set correction events into shared target IR atoms."""

    position_space = str(position_space or "").strip()
    if position_space not in {"segment_local", "batch_tensor"}:
        raise ValueError("residual_set target IR position_space must be segment_local or batch_tensor")

    atoms: list[SupervisionAtom] = []
    for event in events:
        for draft_index, draft in enumerate(event.atom_drafts):
            if draft.target_position != draft.logit_position + 1:
                raise ValueError(
                    "residual_set correction draft requires "
                    "target_position = logit_position + 1"
                )
            if not draft.valid_actions:
                raise ValueError("residual_set correction draft valid_actions must be nonempty")

            live_token_id = _live_token_id(
                input_ids,
                batch_index=batch_index,
                target_position=draft.target_position,
            )
            valid_ids = frozenset(int(action.token_id) for action in draft.valid_actions)
            if live_token_id not in valid_ids:
                raise ValueError(
                    "residual_set correction live token must be inside valid actions"
                )

            roles = frozenset(action.token_role for action in draft.valid_actions)
            if len(roles) != 1:
                raise ValueError(
                    "residual_set correction valid actions must have the same token role"
                )
            selected_action = next(
                action for action in draft.valid_actions if int(action.token_id) == live_token_id
            )
            coord_roles = frozenset(
                action.coord_role
                for action in draft.valid_actions
                if action.coord_role is not None
            )
            if selected_action.token_role is TokenRole.COORD:
                if coord_roles != frozenset({selected_action.coord_role}):
                    raise ValueError(
                        "residual_set correction coord_role must be consistent within a draft"
                    )
            elif coord_roles:
                raise ValueError(
                    "residual_set correction non-coordinate actions must not carry coord_role"
                )

            atoms.append(
                SupervisionAtom(
                    batch_index=batch_index,
                    logit_position=int(draft.logit_position),
                    target_position=int(draft.target_position),
                    allowed_token_roles=frozenset({selected_action.token_role}),
                    selected_token_role=selected_action.token_role,
                    valid_token_ids=valid_ids,
                    selected_token_id=live_token_id,
                    latent_valid_token_ids=valid_ids,
                    coverage_target_weights=None,
                    loss_tags=frozenset({"stage2", "channel_b", "residual_set"}),
                    loss_weight=_action_loss_weight(selected_action),
                    coord_role=selected_action.coord_role,
                    provenance=_residual_atom_provenance(
                        event=event,
                        draft=draft,
                        draft_index=draft_index,
                        observed_token_id=draft.metadata.get(
                            "observed_token_id",
                            event.metadata.get("observed_token_id"),
                        ),
                        valid_ids=valid_ids,
                        valid_actions=draft.valid_actions,
                    ),
                )
            )

    target_ir = TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=tuple(atoms),
        metadata={
            "stage": "stage2",
            "stage2_channel": "B",
            "objective": "residual_set_correction",
            "marginal_scope": MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
            "position_space": position_space,
        },
    )
    validate_target_ir(target_ir, input_ids=input_ids, role_vocab=role_vocab)
    return target_ir


def _coord_atoms_for_target(
    *,
    input_ids: torch.Tensor,
    batch_index: int,
    target: Stage2TeacherForcingObjectTarget,
    coord_vocab: frozenset[int],
    loss_tags: frozenset[str],
    provenance: Mapping[str, Any],
) -> tuple[SupervisionAtom, ...]:
    atoms: list[SupervisionAtom] = []
    for slot_index, target_position in enumerate(target.coord_positions):
        token_id = int(input_ids[batch_index, target_position].item())
        # Stage-2 legacy metadata stores coordinate bins. The live teacher-forced
        # token is the supervision token; the bin remains diagnostic provenance.
        valid_ids = frozenset({token_id})
        if coord_vocab and token_id not in coord_vocab:
            valid_ids = frozenset({token_id})
        atom_provenance = dict(provenance)
        atom_provenance.update(
            {
                "coord_role": _COORD_ROLES[slot_index],
                "gt_bin": int(target.gt_bins[slot_index]),
            }
        )
        atoms.append(
            SupervisionAtom(
                batch_index=batch_index,
                logit_position=target_position - 1,
                target_position=target_position,
                allowed_token_roles=frozenset({TokenRole.COORD}),
                selected_token_role=TokenRole.COORD,
                valid_token_ids=valid_ids,
                selected_token_id=token_id,
                latent_valid_token_ids=valid_ids,
                coverage_target_weights=None,
                loss_tags=loss_tags,
                loss_weight=float(target.loss_weight),
                coord_role=_COORD_ROLES[slot_index],
                provenance=atom_provenance,
            )
        )
    return tuple(atoms)


def _live_token_id(
    input_ids: torch.Tensor,
    *,
    batch_index: int,
    target_position: int,
) -> int:
    try:
        return int(input_ids[batch_index, target_position].item())
    except Exception as exc:
        raise ValueError(
            "residual_set correction target_position is out of bounds for input_ids"
        ) from exc


def _action_loss_weight(action: Any) -> float:
    if hasattr(action, "loss_weight"):
        value = action.loss_weight
        source = "action.loss_weight"
    else:
        metadata = getattr(action, "metadata", {})
        if not isinstance(metadata, Mapping) or "loss_weight" not in metadata:
            return 1.0
        value = metadata["loss_weight"]
        source = "action.metadata['loss_weight']"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"residual_set correction {source} must be a finite nonnegative real number"
        )
    loss_weight = float(value)
    if not math.isfinite(loss_weight) or loss_weight < 0.0:
        raise ValueError(
            f"residual_set correction {source} must be a finite nonnegative real number"
        )
    return loss_weight


def _residual_atom_provenance(
    *,
    event: CorrectionEvent,
    draft: Any,
    draft_index: int,
    observed_token_id: Any,
    valid_ids: frozenset[int],
    valid_actions: Sequence[Any],
) -> Mapping[str, Any]:
    rollout_index = draft.metadata.get("rollout_index", event.metadata.get("rollout_index"))
    anchor_position = draft.metadata.get(
        "anchor_position", event.metadata.get("anchor_position")
    )
    provenance = {
        "stage": "stage2",
        "channel": "B",
        "correction_kind": draft.correction_kind,
        "draft_index": int(draft_index),
        "observed_token_id": _optional_int(observed_token_id),
        "rollout_index": _optional_int(rollout_index),
        "sample_id": event.sample_id,
        "anchor_position": _optional_int(anchor_position),
        "valid_token_ids": tuple(sorted(valid_ids)),
        "support_provenance": _support_provenance_for_actions(valid_actions),
    }
    if event.metadata.get("target_builder") is not None:
        provenance["target_builder"] = str(event.metadata["target_builder"])
    if event.correction_kind != draft.correction_kind:
        provenance["event_correction_kind"] = event.correction_kind
    return provenance


def _support_provenance_for_actions(valid_actions: Sequence[Any]) -> tuple[str, ...]:
    support: set[str] = set()
    for action in valid_actions:
        metadata = getattr(action, "metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        raw = metadata.get("support_provenance")
        if raw is None:
            continue
        if isinstance(raw, str):
            support.add(raw)
            continue
        try:
            support.update(str(item) for item in raw)
        except TypeError:
            support.add(str(raw))
    if not support:
        support.add("labeled")
    return tuple(sorted(support))


def _object_targets_from_legacy_meta(
    meta: Mapping[str, Any],
) -> tuple[Stage2TeacherForcingObjectTarget, ...]:
    targets: list[Stage2TeacherForcingObjectTarget] = []
    for bbox_group_index, group in enumerate(
        _mapping_sequence(meta.get("bbox_groups_prefix", ()))
    ):
        targets.append(
            Stage2TeacherForcingObjectTarget(
                object_source="rollout",
                positive_policy=str(
                    group.get("positive_policy", "rollout_positive")
                ),
                coord_positions=_tuple4(
                    group.get("pos", ()),
                    field_name="bbox_groups_prefix.pos",
                ),
                gt_bins=_tuple4(
                    group.get("gt_bins", ()),
                    field_name="bbox_groups_prefix.gt_bins",
                ),
                loss_weight=float(group.get("weight", 1.0)),
                gt_index=_optional_int(group.get("gt_index")),
                rollout_index=_optional_int(group.get("rollout_index")),
                bbox_group_index=bbox_group_index,
            )
        )

    fn_gt_indices = _int_sequence(meta.get("fn_gt_indices_final", ()))
    recovered_gt_indices = set(_int_sequence(meta.get("recovered_gt_indices", ())))
    fn_weights = tuple(
        float(v) for v in _number_sequence(meta.get("fn_object_weights", ()))
    )
    for bbox_group_index, group in enumerate(
        _mapping_sequence(meta.get("bbox_groups_fn", ()))
    ):
        gt_index = (
            fn_gt_indices[bbox_group_index]
            if bbox_group_index < len(fn_gt_indices)
            else _optional_int(group.get("gt_index"))
        )
        object_source = "recovered_fn" if gt_index in recovered_gt_indices else "fn"
        weight = (
            fn_weights[bbox_group_index]
            if bbox_group_index < len(fn_weights)
            else float(group.get("weight", 1.0))
        )
        targets.append(
            Stage2TeacherForcingObjectTarget(
                object_source=object_source,
                positive_policy=str(group.get("positive_policy", object_source)),
                coord_positions=_tuple4(
                    group.get("pos", ()), field_name="bbox_groups_fn.pos"
                ),
                gt_bins=_tuple4(
                    group.get("gt_bins", ()),
                    field_name="bbox_groups_fn.gt_bins",
                ),
                loss_weight=weight,
                gt_index=gt_index,
                bbox_group_index=bbox_group_index,
            )
        )
    return tuple(targets)


def _is_zero_positive_target(target: Stage2TeacherForcingObjectTarget) -> bool:
    return (
        target.object_source == "rollout"
        and target.positive_policy in _ROLLOUT_ZERO_POSITIVE_POLICIES
    )


def _position_is_trainable(
    position: int,
    *,
    prompt_len: int,
    encoded_len: int,
    train_len: int,
) -> bool:
    return prompt_len <= position < encoded_len and position < prompt_len + train_len


def _absolute_tail_positions(
    values: Any,
    *,
    prompt_len: int,
    prefix_len: int,
) -> tuple[int, ...]:
    return tuple(prompt_len + prefix_len + rel for rel in _int_sequence(values))


def _int_meta(meta: Mapping[str, Any], key: str, *, default: int) -> int:
    value = meta.get(key, default)
    if value is None:
        return default
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _tuple4(value: Sequence[Any], *, field_name: str) -> tuple[int, int, int, int]:
    values = tuple(int(v) for v in value)
    if len(values) != 4:
        raise ValueError(f"{field_name} must contain exactly four integers")
    return cast(tuple[int, int, int, int], values)


def _int_sequence(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(v) for v in value)


def _number_sequence(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    return tuple(float(v) for v in value)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


__all__ = [
    "Stage2TeacherForcingObjectTarget",
    "build_residual_set_target_ir",
    "build_stage2_teacher_forcing_target_ir",
]
