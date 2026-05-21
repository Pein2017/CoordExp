from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import torch

from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.probabilities import teacher_forcing_atom_loss
from src.training.teacher_forcing.validation import validate_target_ir
from src.training.teacher_forcing.vocab import RoleVocab

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from ..token_types import iter_segment_views


_METRIC_PREFIX = "stage2_ab/channel_b/residual_set"


@dataclass(frozen=True)
class ResidualSetCorrectionConfig:
    lambda_type: float = 1.0
    lambda_inner: float = 1.0


_RESIDUAL_SET_V1_CONFIG_KEYS = frozenset(
    {
        "prepared_rollout_jsonl",
        "expected_num_rollouts",
        "base_seed",
        "lambda_type",
        "lambda_inner",
        "fallback_loss_weight",
        "lambda_ul_promoted",
        "label_conflict_weight",
        "commit_iou_threshold",
        "duplicate_burst_iou_threshold",
        "ul_cluster_iou_threshold",
        "ul_gray_iou_low",
        "ul_consensus_ratio",
        "min_ul_valid_rollouts",
        "clean_gt_sft_mix",
        "strict_prepared_rollout_tokens",
        "legacy_reencode_fallback",
        "strict_builder_invariants",
    }
)


def _coerce_nonnegative_finite_float(value: Any, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"residual_set_correction config {key} must be numeric, not bool"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"residual_set_correction config {key} must be finite")
    if result < 0.0:
        raise ValueError(f"residual_set_correction config {key} must be nonnegative")
    return result


def build_residual_set_correction_config(raw_config: Any) -> ResidualSetCorrectionConfig:
    cfg = raw_config if isinstance(raw_config, Mapping) else {}
    unknown_keys = sorted(str(key) for key in cfg if str(key) not in _RESIDUAL_SET_V1_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(
            "residual_set_correction config contains unsupported key(s): "
            + ", ".join(unknown_keys)
        )
    default = ResidualSetCorrectionConfig()
    return ResidualSetCorrectionConfig(
        lambda_type=_coerce_nonnegative_finite_float(
            cfg.get("lambda_type", default.lambda_type),
            key="lambda_type",
        ),
        lambda_inner=_coerce_nonnegative_finite_float(
            cfg.get("lambda_inner", default.lambda_inner),
            key="lambda_inner",
        ),
    )


def _zero_result(context: TeacherForcingContext) -> ModuleResult:
    zero = context.logits.float().sum() * 0.0
    metrics = {
        f"{_METRIC_PREFIX}/atom_count": 0.0,
        f"{_METRIC_PREFIX}/loss": 0.0,
        f"{_METRIC_PREFIX}/component/type": 0.0,
        f"{_METRIC_PREFIX}/component/valid": 0.0,
        f"{_METRIC_PREFIX}/component/coverage": 0.0,
        f"{_METRIC_PREFIX}/valid_prob_mean": 0.0,
        f"{_METRIC_PREFIX}/allowed_prob_mean": 0.0,
        f"{_METRIC_PREFIX}/labeled_atom_count": 0.0,
        f"{_METRIC_PREFIX}/ul_atom_count": 0.0,
        f"{_METRIC_PREFIX}/mixed_atom_count": 0.0,
    }
    return ModuleResult(
        loss=zero,
        metrics=metrics,
        state={"residual_set_correction_contrib": zero},
    )


def _role_vocab_from_context(context: TeacherForcingContext) -> RoleVocab:
    role_vocab = context.extra.get("role_vocab")
    if not isinstance(role_vocab, RoleVocab):
        raise ValueError(
            "residual_set_correction requires context.extra['role_vocab'] to be a RoleVocab"
        )
    return role_vocab


def _validate_segment_atom(
    *,
    atom_index: int,
    atom_batch_index: int,
    atom_logit_position: int,
    atom_target_position: int,
    batch_index: int,
    segment_start: int,
    segment_end: int,
    sequence_length: int,
) -> None:
    prefix = f"residual_set_target_ir.atoms[{atom_index}]"
    if int(atom_batch_index) != int(batch_index):
        raise ValueError(f"{prefix}: batch_index must match current segment")
    if int(atom_logit_position) < int(segment_start):
        raise ValueError(f"{prefix}: logit_position precedes current segment")
    if int(atom_target_position) >= int(segment_end):
        raise ValueError(f"{prefix}: target_position exceeds current segment")
    if not (
        int(segment_start)
        <= int(atom_logit_position)
        < int(atom_target_position)
        < int(segment_end)
    ):
        raise ValueError(
            f"{prefix}: expected segment_start <= logit_position < target_position < segment_end"
        )
    if int(atom_logit_position) >= int(sequence_length):
        raise ValueError(f"{prefix}: logit_position is out of bounds for logits")


def _support_provenance_flags(raw: Any) -> tuple[bool, bool]:
    if raw is None:
        return False, False
    if isinstance(raw, str):
        values = {raw.lower()}
    else:
        try:
            values = {str(item).lower() for item in raw}
        except TypeError:
            values = {str(raw).lower()}
    has_labeled = any(
        value in {"labeled", "labelled", "gt", "supervised"} for value in values
    )
    has_ul = any(
        value in {"ul", "unlabeled", "unlabelled", "pseudo"} for value in values
    )
    return has_labeled, has_ul


def _copy_atom_for_segment(
    atom: SupervisionAtom,
    *,
    batch_index: int,
    position_offset: int,
) -> SupervisionAtom:
    return SupervisionAtom(
        batch_index=int(batch_index),
        logit_position=int(atom.logit_position) + int(position_offset),
        target_position=int(atom.target_position) + int(position_offset),
        allowed_token_roles=atom.allowed_token_roles,
        selected_token_role=atom.selected_token_role,
        valid_token_ids=atom.valid_token_ids,
        selected_token_id=int(atom.selected_token_id),
        latent_valid_token_ids=atom.latent_valid_token_ids,
        coverage_target_weights=atom.coverage_target_weights,
        loss_tags=atom.loss_tags,
        loss_weight=float(atom.loss_weight),
        coord_role=atom.coord_role,
        provenance=atom.provenance,
    )


def _project_target_ir_for_segment(
    target_ir: TeacherForcingTargetIR,
    *,
    batch_index: int,
    segment_start: int,
) -> TeacherForcingTargetIR:
    """Project declared residual IR position space to batch tensor coordinates."""

    if not target_ir.atoms:
        return target_ir

    position_space_raw = target_ir.metadata.get("position_space")
    position_space = str(position_space_raw or "")
    if not position_space:
        raise ValueError(
            "residual_set_target_ir.metadata.position_space is required "
            "for non-empty residual_set_target_ir"
        )
    if position_space not in {"segment_local", "batch_tensor"}:
        raise ValueError(
            "residual_set_target_ir.metadata.position_space must be "
            "'segment_local' or 'batch_tensor'"
        )

    position_offset = int(segment_start) if position_space == "segment_local" else 0
    atoms = [
        _copy_atom_for_segment(
            atom,
            batch_index=int(batch_index),
            position_offset=int(position_offset),
        )
        for atom in target_ir.atoms
    ]
    metadata = dict(target_ir.metadata)
    metadata["positions_rebased"] = True
    metadata["position_space"] = "batch_tensor"
    metadata["source_position_space"] = position_space
    metadata["segment_start"] = int(segment_start)
    metadata["batch_index"] = int(batch_index)
    return TeacherForcingTargetIR(
        schema_version=target_ir.schema_version,
        atoms=tuple(atoms),
        metadata=metadata,
    )


def run_residual_set_correction_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> ModuleResult:
    if not isinstance(context, TeacherForcingContext):
        raise TypeError(
            "residual_set_correction requires context to be a TeacherForcingContext"
        )
    if not isinstance(spec, PipelineModuleSpec):
        raise TypeError(
            "residual_set_correction requires spec to be a PipelineModuleSpec"
        )

    config = build_residual_set_correction_config(spec.config)
    if str(context.channel or "").strip().upper() != "B":
        return _zero_result(context)

    role_vocab: RoleVocab | None = None

    loss_terms: list[torch.Tensor] = []
    type_terms: list[torch.Tensor] = []
    valid_terms: list[torch.Tensor] = []
    coverage_terms: list[torch.Tensor] = []
    valid_probs: list[torch.Tensor] = []
    allowed_probs: list[torch.Tensor] = []
    labeled_atom_count = 0
    ul_atom_count = 0
    mixed_atom_count = 0

    for batch_index, segment_start, segment_end, segment_meta in iter_segment_views(
        input_ids=context.input_ids,
        meta=context.meta,
    ):
        target_ir = segment_meta.get("residual_set_target_ir")
        if target_ir is None:
            raise ValueError(
                "residual_set_correction requires residual_set_target_ir "
                "for every B-channel segment"
            )
        if not isinstance(target_ir, TeacherForcingTargetIR):
            raise TypeError("residual_set_target_ir must be a TeacherForcingTargetIR")
        if role_vocab is None:
            role_vocab = _role_vocab_from_context(context)

        # Stage-2 target_builder stores residual sidecars as segment-local IR.
        # Other producers may hand us batch-tensor IR, so project according to
        # the explicit position_space contract before validation/use.
        target_ir = _project_target_ir_for_segment(
            target_ir,
            batch_index=int(batch_index),
            segment_start=int(segment_start),
        )
        validate_target_ir(
            target_ir,
            input_ids=context.input_ids,
            role_vocab=role_vocab,
        )

        for atom_index, atom in enumerate(target_ir.atoms):
            _validate_segment_atom(
                atom_index=atom_index,
                atom_batch_index=atom.batch_index,
                atom_logit_position=atom.logit_position,
                atom_target_position=atom.target_position,
                batch_index=batch_index,
                segment_start=segment_start,
                segment_end=segment_end,
                sequence_length=int(context.logits.shape[1]),
            )

            row_logits = context.logits[atom.batch_index, atom.logit_position]
            atom_loss = teacher_forcing_atom_loss(
                row_logits,
                atom=atom,
                role_vocab=role_vocab,
                coverage_strength=0.0,
            )
            weighted_atom_loss = (
                atom_loss.type * float(config.lambda_type)
                + atom_loss.valid * float(config.lambda_inner)
            )
            atom_weight = row_logits.new_tensor(
                float(atom.loss_weight),
                dtype=torch.float32,
            )
            loss_terms.append(weighted_atom_loss * atom_weight)
            type_terms.append(atom_loss.type * float(config.lambda_type) * atom_weight)
            valid_terms.append(atom_loss.valid * float(config.lambda_inner) * atom_weight)
            coverage_terms.append(atom_loss.coverage * 0.0 * atom_weight)
            valid_probs.append(atom_loss.valid_probability)
            allowed_probs.append(atom_loss.allowed_probability)

            has_labeled, has_ul = _support_provenance_flags(
                atom.provenance.get("support_provenance")
            )
            labeled_atom_count += int(has_labeled)
            ul_atom_count += int(has_ul)
            mixed_atom_count += int(has_labeled and has_ul)

    if not loss_terms:
        return _zero_result(context)

    loss = torch.stack(loss_terms).mean()
    atom_count = float(len(loss_terms))
    metrics = {
        f"{_METRIC_PREFIX}/atom_count": atom_count,
        f"{_METRIC_PREFIX}/loss": float(loss.detach().cpu().item()),
        f"{_METRIC_PREFIX}/component/type": float(
            torch.stack(type_terms).mean().detach().cpu().item()
        ),
        f"{_METRIC_PREFIX}/component/valid": float(
            torch.stack(valid_terms).mean().detach().cpu().item()
        ),
        f"{_METRIC_PREFIX}/component/coverage": float(
            torch.stack(coverage_terms).mean().detach().cpu().item()
        ),
        f"{_METRIC_PREFIX}/valid_prob_mean": float(
            torch.stack(valid_probs).mean().detach().cpu().item()
        ),
        f"{_METRIC_PREFIX}/allowed_prob_mean": float(
            torch.stack(allowed_probs).mean().detach().cpu().item()
        ),
        f"{_METRIC_PREFIX}/labeled_atom_count": float(labeled_atom_count),
        f"{_METRIC_PREFIX}/ul_atom_count": float(ul_atom_count),
        f"{_METRIC_PREFIX}/mixed_atom_count": float(mixed_atom_count),
    }

    return ModuleResult(
        loss=loss,
        metrics=metrics,
        state={"residual_set_correction_contrib": loss},
    )
