from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import torch

from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.probabilities import teacher_forcing_atom_loss
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.validation import validate_target_ir
from src.training.teacher_forcing.vocab import RoleVocab

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from ..token_types import iter_segment_views


_METRIC_PREFIX = "stage2_rollout_correction/residual_set"


@dataclass(frozen=True)
class ResidualSetCorrectionConfig:
    lambda_type: float = 1.0
    lambda_inner: float = 1.0


_RESIDUAL_SET_V1_CONFIG_KEYS = frozenset(
    {
        "expected_num_rollouts",
        "base_seed",
        "lambda_type",
        "lambda_inner",
        "fallback_loss_weight",
        "lambda_ul_promoted",
        "label_conflict_weight",
        "commit_iou_threshold",
        "duplicate_burst_iou_threshold",
        "duplicate_burst_prefix_rollback",
        "ul_cluster_iou_threshold",
        "ul_gray_iou_low",
        "ul_consensus_ratio",
        "min_ul_valid_rollouts",
        "clean_gt_sft_mix",
        "strict_builder_invariants",
    }
)

_COMPACT_METRIC_KEYS = (
    "sequence_count",
    "atom_count",
    "atom_weight_sum",
    "raw_atom_loss_sum",
    "sequence_loss",
    "type_loss",
    "inner_loss",
    "wrong_type_mass",
    "valid_set_mass",
    "dirty_prefix_sequence_count",
    "committed_gt_rows",
    "committed_ul_rows",
    "pending_ul_candidates",
    "promoted_ul_clusters",
    "uncommitted_invalid_geometry",
    "uncommitted_malformed",
    "uncommitted_duplicate",
    "duplicate_prefix_rollback",
    "uncommitted_fp_or_unpromoted",
    "spatial_wrong_desc_conflict",
    "label_conflict_atoms",
    "label_conflict_no_atom",
    "eos_targets",
    "continue_targets",
    "dirty_prefix_reencoded",
    "clean_success_skipped",
    "ambiguous_token_targets",
    "strict_token_targets",
    "target_token_mismatch",
    "coord_ambiguous_token_targets",
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
    unknown_keys = sorted(
        str(key) for key in cfg if str(key) not in _RESIDUAL_SET_V1_CONFIG_KEYS
    )
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
    metrics = _compact_metrics()
    return ModuleResult(
        loss=zero,
        metrics=metrics,
        state={"residual_set_correction_contrib": zero},
    )


def _compact_metrics(initial: Mapping[str, float] | None = None) -> dict[str, float]:
    metrics = {f"{_METRIC_PREFIX}/{key}": 0.0 for key in _COMPACT_METRIC_KEYS}
    if initial:
        for key, value in initial.items():
            metrics[f"{_METRIC_PREFIX}/{key}"] = float(value)
    return metrics


def _metric_value(raw: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool):
            return float(1.0 if value else 0.0)
        if isinstance(value, Real):
            result = float(value)
            return result if math.isfinite(result) else 0.0
    return 0.0


def _sum_metric_values(raw: Mapping[str, Any], *keys: str) -> float:
    return sum(_metric_value(raw, key) for key in keys)


def _add_sequence_source_metrics(
    totals: Counter[str],
    raw: Mapping[str, Any],
    *,
    atom_count: int,
) -> None:
    totals["committed_gt_rows"] += _metric_value(
        raw, "committed_gt_rows", "scanner_row_decision/committed"
    )
    totals["committed_ul_rows"] += _metric_value(raw, "committed_ul_rows")
    totals["pending_ul_candidates"] += _metric_value(raw, "pending_ul_candidates")
    totals["promoted_ul_clusters"] += _metric_value(
        raw, "promoted_ul_clusters", "ul_promoted_object_count"
    )
    totals["uncommitted_invalid_geometry"] += _metric_value(
        raw, "uncommitted_invalid_geometry", "scanner_row_decision/invalid_geometry"
    )
    totals["uncommitted_malformed"] += _metric_value(
        raw, "uncommitted_malformed"
    ) + _sum_metric_values(
        raw,
        "scanner_row_decision/malformed_span",
        "scanner_row_decision/trailing_incomplete",
    )
    totals["uncommitted_duplicate"] += _metric_value(
        raw, "uncommitted_duplicate", "scanner_row_decision/duplicate_burst"
    )
    totals["duplicate_prefix_rollback"] += _metric_value(
        raw, "scanner_duplicate_prefix_rollback"
    )
    totals["uncommitted_fp_or_unpromoted"] += _metric_value(
        raw,
        "uncommitted_fp_or_unpromoted",
        "scanner_row_decision/unmatched_dirty_context",
    )
    totals["spatial_wrong_desc_conflict"] += _metric_value(
        raw,
        "spatial_wrong_desc_conflict",
        "scanner_row_decision/spatial_wrong_desc_conflict",
    )
    totals["label_conflict_no_atom"] += _metric_value(raw, "label_conflict_no_atom")
    totals["dirty_prefix_reencoded"] += _metric_value(raw, "dirty_prefix_reencoded")
    totals["clean_success_skipped"] += _metric_value(raw, "clean_success_skipped")
    if _metric_value(raw, "scanner_dirty_context_span_count") > 0.0 or _metric_value(
        raw, "no_event_dropped_dirty_prefix"
    ) > 0.0:
        totals["dirty_prefix_sequence_count"] += 1.0
    elif atom_count > 0 and _metric_value(raw, "dirty_prefix_sequence_count") > 0.0:
        totals["dirty_prefix_sequence_count"] += _metric_value(
            raw, "dirty_prefix_sequence_count"
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
    if str(context.channel or "").strip() != "rollout_correction":
        return _zero_result(context)

    role_vocab: RoleVocab | None = None
    sequence_losses: list[torch.Tensor] = []
    sequence_type_losses: list[torch.Tensor] = []
    sequence_inner_losses: list[torch.Tensor] = []
    sequence_wrong_type_mass: list[torch.Tensor] = []
    sequence_valid_set_mass: list[torch.Tensor] = []
    raw_atom_loss_terms: list[torch.Tensor] = []
    atom_weight_sum = 0.0
    atom_count = 0
    source_metric_totals: Counter[str] = Counter()
    decode_mode_counts: Counter[str] = Counter()
    decode_mode_atom_counts: Counter[str] = Counter()

    for batch_index, segment_start, segment_end, segment_meta in iter_segment_views(
        input_ids=context.input_ids,
        meta=context.meta,
    ):
        target_ir = segment_meta.get("residual_set_target_ir")
        if target_ir is None:
            raise ValueError(
                "residual_set_correction requires residual_set_target_ir "
                "for every rollout-correction segment"
            )
        if not isinstance(target_ir, TeacherForcingTargetIR):
            raise TypeError("residual_set_target_ir must be a TeacherForcingTargetIR")
        if role_vocab is None:
            role_vocab = _role_vocab_from_context(context)

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

        sequence_weighted_losses: list[torch.Tensor] = []
        sequence_weighted_type_losses: list[torch.Tensor] = []
        sequence_weighted_inner_losses: list[torch.Tensor] = []
        sequence_weighted_wrong_type_mass: list[torch.Tensor] = []
        sequence_weighted_valid_set_mass: list[torch.Tensor] = []
        sequence_weights: list[torch.Tensor] = []
        sequence_atom_count = 0

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

            atom_weight_value = float(atom.loss_weight)
            if atom_weight_value <= 0.0:
                continue

            row_logits = context.logits[atom.batch_index, atom.logit_position]
            atom_loss = teacher_forcing_atom_loss(
                row_logits,
                atom=atom,
                role_vocab=role_vocab,
                coverage_strength=0.0,
            )
            type_loss = atom_loss.type * float(config.lambda_type)
            inner_loss = atom_loss.valid * float(config.lambda_inner)
            weighted_atom_loss = type_loss + inner_loss
            atom_weight = row_logits.new_tensor(
                atom_weight_value,
                dtype=torch.float32,
            )

            sequence_weighted_losses.append(weighted_atom_loss * atom_weight)
            sequence_weighted_type_losses.append(type_loss * atom_weight)
            sequence_weighted_inner_losses.append(inner_loss * atom_weight)
            sequence_weighted_wrong_type_mass.append(
                (1.0 - atom_loss.allowed_probability) * atom_weight
            )
            sequence_weighted_valid_set_mass.append(
                atom_loss.valid_probability * atom_weight
            )
            sequence_weights.append(atom_weight)
            raw_atom_loss_terms.append(weighted_atom_loss)
            atom_weight_sum += atom_weight_value
            atom_count += 1
            sequence_atom_count += 1

            if atom.selected_token_role is TokenRole.STOP:
                source_metric_totals["eos_targets"] += 1.0
            else:
                source_metric_totals["continue_targets"] += 1.0
            if len(atom.valid_token_ids) > 1:
                source_metric_totals["ambiguous_token_targets"] += 1.0
                if atom.selected_token_role is TokenRole.COORD:
                    source_metric_totals["coord_ambiguous_token_targets"] += 1.0
            else:
                source_metric_totals["strict_token_targets"] += 1.0
            live_token_id = int(
                context.input_ids[atom.batch_index, atom.target_position].item()
            )
            if int(atom.selected_token_id) != live_token_id:
                source_metric_totals["target_token_mismatch"] += 1.0
            correction_kind = str(atom.provenance.get("correction_kind", ""))
            if correction_kind == "spatial_wrong_desc_conflict":
                source_metric_totals["label_conflict_atoms"] += 1.0

        raw_metrics = segment_meta.get("residual_set_metrics")
        if isinstance(raw_metrics, Mapping):
            _add_sequence_source_metrics(
                source_metric_totals,
                raw_metrics,
                atom_count=sequence_atom_count,
            )

        if not sequence_weighted_losses:
            continue

        sequence_weight = torch.stack(sequence_weights).sum()
        sequence_losses.append(
            torch.stack(sequence_weighted_losses).sum() / sequence_weight
        )
        sequence_type_losses.append(
            torch.stack(sequence_weighted_type_losses).sum() / sequence_weight
        )
        sequence_inner_losses.append(
            torch.stack(sequence_weighted_inner_losses).sum() / sequence_weight
        )
        sequence_wrong_type_mass.append(
            torch.stack(sequence_weighted_wrong_type_mass).sum() / sequence_weight
        )
        sequence_valid_set_mass.append(
            torch.stack(sequence_weighted_valid_set_mass).sum() / sequence_weight
        )

        decode_mode = str(
            segment_meta.get("decode_mode", target_ir.metadata.get("decode_mode", ""))
            or "unknown"
        ).strip().lower()
        if not decode_mode:
            decode_mode = "unknown"
        decode_mode_counts[decode_mode] += 1.0
        decode_mode_atom_counts[decode_mode] += float(sequence_atom_count)

    if not sequence_losses:
        zero = context.logits.float().sum() * 0.0
        return ModuleResult(
            loss=zero,
            metrics=_compact_metrics(
                {key: float(value) for key, value in source_metric_totals.items()}
            ),
            state={"residual_set_correction_contrib": zero},
        )

    loss = torch.stack(sequence_losses).mean()
    type_loss = torch.stack(sequence_type_losses).mean()
    inner_loss = torch.stack(sequence_inner_losses).mean()
    wrong_type_mass = torch.stack(sequence_wrong_type_mass).mean()
    valid_set_mass = torch.stack(sequence_valid_set_mass).mean()
    raw_atom_loss_sum = torch.stack(raw_atom_loss_terms).sum()

    metrics = _compact_metrics(
        {
            "sequence_count": float(len(sequence_losses)),
            "atom_count": float(atom_count),
            "atom_weight_sum": float(atom_weight_sum),
            "raw_atom_loss_sum": float(raw_atom_loss_sum.detach().cpu().item()),
            "sequence_loss": float(loss.detach().cpu().item()),
            "type_loss": float(type_loss.detach().cpu().item()),
            "inner_loss": float(inner_loss.detach().cpu().item()),
            "wrong_type_mass": float(wrong_type_mass.detach().cpu().item()),
            "valid_set_mass": float(valid_set_mass.detach().cpu().item()),
            **{key: float(value) for key, value in source_metric_totals.items()},
        }
    )
    for decode_mode, sequence_count in sorted(decode_mode_counts.items()):
        metrics[f"{_METRIC_PREFIX}/decode_mode/{decode_mode}/sequence_count"] = float(
            sequence_count
        )
        metrics[f"{_METRIC_PREFIX}/decode_mode/{decode_mode}/atom_count"] = float(
            decode_mode_atom_counts[decode_mode]
        )

    return ModuleResult(
        loss=loss,
        metrics=metrics,
        state={"residual_set_correction_contrib": loss},
    )
