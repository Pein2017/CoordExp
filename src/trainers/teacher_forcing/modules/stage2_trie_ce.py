from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import torch

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from ..token_types import iter_segment_views


_RESERVED_WEIGHT_KEYS: tuple[str, ...] = (
    "support_weight",
    "balance_weight",
    "struct_weight",
    "desc_weight",
    "coord_hard_ce_weight",
    "eos_weight",
)
_SEMANTIC_ROLE_METRIC_KEYS: tuple[str, ...] = (
    "text",
    "struct",
    "desc",
    "coord",
    "eos",
)


@dataclass(frozen=True)
class Stage2TrieCEConfig:
    support_weight: float = 1.0
    balance_weight: float = 1.0
    struct_weight: float = 1.0
    desc_weight: float = 1.0
    coord_hard_ce_weight: float = 1.0
    eos_weight: float = 1.0
    normalization: str = "token_mean"


def _read_config_value(raw_config: Any, key: str, default: Any) -> Any:
    if isinstance(raw_config, Mapping):
        return raw_config.get(key, default)

    return getattr(raw_config, key, default)


def _coerce_float(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"stage2_trie_ce config {key} must be numeric, not bool")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"stage2_trie_ce config {key} must be numeric") from exc

    if not isfinite(result):
        raise ValueError(f"stage2_trie_ce config {key} must be finite")

    return result


def build_stage2_trie_ce_config(raw_config: Any) -> Stage2TrieCEConfig:
    default = Stage2TrieCEConfig()

    weight_values = {
        key: _coerce_float(
            _read_config_value(raw_config, key, getattr(default, key)),
            key,
        )
        for key in _RESERVED_WEIGHT_KEYS
    }

    normalization = str(
        _read_config_value(raw_config, "normalization", default.normalization)
        or default.normalization
    ).strip().lower()
    if normalization != "token_mean":
        raise ValueError(
            "Stage-2 trie CE pure hard CE v0 only supports "
            "normalization='token_mean'; semantic bucket balancing is reserved "
            "for a future objective refinement."
        )

    return Stage2TrieCEConfig(
        support_weight=weight_values["support_weight"],
        balance_weight=weight_values["balance_weight"],
        struct_weight=weight_values["struct_weight"],
        desc_weight=weight_values["desc_weight"],
        coord_hard_ce_weight=weight_values["coord_hard_ce_weight"],
        eos_weight=weight_values["eos_weight"],
        normalization=normalization,
    )


def _is_active_stage2_trie_segment(
    *,
    context: TeacherForcingContext,
    segment_meta: Mapping[str, Any],
) -> bool:
    """Return whether a segment should contribute Stage-2 trie CE terms."""

    if str(context.channel or "").strip().upper() != "B":
        return False

    segment_channel = segment_meta.get("stage2_channel")
    if segment_channel is not None and str(segment_channel).strip().upper() != "B":
        return False

    if bool(segment_meta.get("stage2_trie_skip_loss")):
        return False

    if "stage2_trie_targets" not in segment_meta:
        return False

    _validate_stage2_trie_targets_shape(segment_meta.get("stage2_trie_targets"))
    return True


def _is_non_string_sequence(value: Any) -> bool:
    """Return whether a value is an iterable structural sequence."""

    return (
        hasattr(value, "__iter__")
        and hasattr(value, "__len__")
        and not isinstance(value, (str, bytes))
    )


def _require_attributes(*, value: Any, attrs: tuple[str, ...], context: str) -> None:
    """Raise a clear type error when a structural sidecar object is incomplete."""

    missing = tuple(attr for attr in attrs if not hasattr(value, attr))
    if missing:
        raise TypeError(
            f"Malformed Stage-2 trie sidecar at {context}: "
            f"missing required attribute(s) {missing!r}"
        )


def _require_string(*, value: Any, context: str) -> None:
    """Raise a clear type error when a structural string field is malformed."""

    if not isinstance(value, str):
        raise TypeError(
            f"Malformed Stage-2 trie sidecar at {context}: "
            f"expected str, got {type(value).__name__}"
        )


def _require_int(*, value: Any, context: str) -> int:
    """Return an integer field while rejecting bool and lossy coercions."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"Malformed Stage-2 trie sidecar at {context}: "
            f"expected int, got {type(value).__name__}"
        )

    return int(value)


def _require_positive_int(*, value: Any, context: str) -> None:
    """Raise a clear error when a structural positive integer is malformed."""

    int_value = _require_int(value=value, context=context)
    if int_value <= 0:
        raise ValueError(
            f"Malformed Stage-2 trie sidecar at {context}: "
            f"expected int > 0, got {value!r}"
        )


def _require_non_negative_int(*, value: Any, context: str) -> None:
    """Raise a clear error when a structural non-negative integer is malformed."""

    int_value = _require_int(value=value, context=context)
    if int_value < 0:
        raise ValueError(
            f"Malformed Stage-2 trie sidecar at {context}: "
            f"expected non-negative int, got {value!r}"
        )


def _require_finite_non_negative_number(*, value: Any, context: str) -> None:
    """Raise a clear error when a structural source weight is malformed."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Malformed Stage-2 trie sidecar at {context}: "
            f"expected numeric, got {type(value).__name__}"
        )
    if not isfinite(float(value)):
        raise ValueError(
            f"Malformed Stage-2 trie sidecar at {context}: "
            f"expected finite numeric value, got {value!r}"
        )
    if float(value) < 0.0:
        raise ValueError(
            f"Malformed Stage-2 trie sidecar at {context}: "
            f"expected non-negative numeric value, got {value!r}"
        )


def _validate_stage2_trie_targets_shape(value: Any) -> None:
    """Validate the import-free structural sidecar contract.

    The teacher-forcing loss intentionally does not import Stage-2 trie
    dataclasses, but malformed sidecars should fail loudly instead of taking
    the no-loss compatibility path.
    """

    _require_attributes(
        value=value,
        attrs=("token_targets", "summary"),
        context="stage2_trie_targets",
    )

    token_targets = getattr(value, "token_targets")
    if not _is_non_string_sequence(token_targets):
        raise TypeError(
            "Malformed Stage-2 trie sidecar at stage2_trie_targets.token_targets: "
            "expected a non-string sequence"
        )

    summary = getattr(value, "summary")
    _require_attributes(
        value=summary,
        attrs=(
            "candidate_count",
            "fallback_candidate_count",
            "weak_positive_fp_count",
            "target_positions",
            "branch_points",
            "max_branching_factor",
        ),
        context="stage2_trie_targets.summary",
    )
    for attr in (
        "candidate_count",
        "fallback_candidate_count",
        "weak_positive_fp_count",
        "target_positions",
        "branch_points",
        "max_branching_factor",
    ):
        _require_non_negative_int(
            value=getattr(summary, attr),
            context=f"stage2_trie_targets.summary.{attr}",
        )

    for target_index, target in enumerate(token_targets):
        target_context = f"stage2_trie_targets.token_targets[{target_index}]"
        _require_attributes(
            value=target,
            attrs=(
                "position",
                "positive_token_ids",
                "source_weights",
                "semantic_role",
            ),
            context=target_context,
        )
        _require_positive_int(
            value=getattr(target, "position"),
            context=f"{target_context}.position",
        )
        _require_string(
            value=getattr(target, "semantic_role"),
            context=f"{target_context}.semantic_role",
        )

        positive_token_ids = getattr(target, "positive_token_ids")
        source_weights = getattr(target, "source_weights")
        if not _is_non_string_sequence(positive_token_ids):
            raise TypeError(
                f"Malformed Stage-2 trie sidecar at {target_context}.positive_token_ids: "
                "expected a non-string sequence"
            )
        if not _is_non_string_sequence(source_weights):
            raise TypeError(
                f"Malformed Stage-2 trie sidecar at {target_context}.source_weights: "
                "expected a non-string sequence"
            )

        positive_token_ids_tuple = tuple(positive_token_ids)
        source_weights_tuple = tuple(source_weights)
        if not positive_token_ids_tuple:
            raise TypeError(
                f"Malformed Stage-2 trie sidecar at {target_context}: "
                "positive_token_ids must be non-empty"
            )
        if len(positive_token_ids_tuple) != len(source_weights_tuple):
            raise TypeError(
                f"Malformed Stage-2 trie sidecar at {target_context}: "
                "positive_token_ids and source_weights must have the same length"
            )
        for token_index, token_id in enumerate(positive_token_ids_tuple):
            _require_non_negative_int(
                value=token_id,
                context=f"{target_context}.positive_token_ids[{token_index}]",
            )
        for weight_index, source_weight in enumerate(source_weights_tuple):
            _require_finite_non_negative_number(
                value=source_weight,
                context=f"{target_context}.source_weights[{weight_index}]",
            )


def _semantic_weight(target: Any, config: Stage2TrieCEConfig) -> float:
    """Map a target semantic role to its configured token weight."""

    role = _semantic_role_name(target)
    if role == "struct":
        return float(config.struct_weight)
    if role == "desc":
        return float(config.desc_weight)
    if role == "coord":
        return float(config.coord_hard_ce_weight)
    if role == "eos":
        return float(config.eos_weight)

    return 1.0


def _semantic_role_name(target: Any) -> str:
    """Normalize a target role into the public metric and weighting buckets."""

    role = str(getattr(target, "semantic_role", "text") or "text").strip().lower()
    if role in _SEMANTIC_ROLE_METRIC_KEYS:
        return role

    return "text"


def _fallback_candidate_share(summary: Any) -> float:
    """Estimate fallback exposure from summary candidate counts.

    ``Stage2TrieSummary`` does not currently carry the total candidate loss
    weight, so this ratio is explicitly a candidate-count proxy. The legacy
    fallback-loss-share metric is set to this value until a true total loss
    weight denominator exists.
    """

    if int(summary.candidate_count) <= 0:
        return 0.0

    return float(summary.fallback_candidate_count) / float(summary.candidate_count)


def _metrics_for_no_active_segments(loss: torch.Tensor) -> dict[str, float]:
    """Build zero-valued metrics for the no-sidecar compatibility path."""

    return {
        "stage2_trie/target_positions": 0.0,
        "stage2_trie/branch_points": 0.0,
        "stage2_trie/max_branching_factor": 0.0,
        "stage2_trie/candidate_count_mean": 0.0,
        "stage2_trie/fallback_candidate_share": 0.0,
        "stage2_trie/fallback_loss_share": 0.0,
        "stage2_trie/fallback_dominance_warning": 0.0,
        "stage2_trie/fp_policy_weak_positive_count": 0.0,
        "stage2_trie/role_text_targets": 0.0,
        "stage2_trie/role_struct_targets": 0.0,
        "stage2_trie/role_desc_targets": 0.0,
        "stage2_trie/role_coord_targets": 0.0,
        "stage2_trie/role_eos_targets": 0.0,
        "loss/B/stage2_trie_ce": float(loss.detach().cpu().item()),
    }


def _validate_target_positions(
    *,
    target: Any,
    batch_index: int,
    segment_start: int,
    segment_end: int,
    sequence_length: int,
) -> int:
    """Validate segment-local label position and return the row logit position."""

    local_label_position = int(target.position)
    if local_label_position <= 0:
        raise ValueError(
            "Stage-2 trie target local position must have an in-segment predicting "
            "logit row: "
            f"batch={batch_index} local_target_position={local_label_position} "
            f"segment=[{segment_start}, {segment_end})"
        )

    local_logit_position = int(local_label_position - 1)
    label_position = int(segment_start + local_label_position)
    logit_position = int(segment_start + local_logit_position)
    if (
        label_position < 0
        or label_position >= int(sequence_length)
        or logit_position < 0
        or logit_position >= int(sequence_length)
    ):
        raise ValueError(
            "Stage-2 trie target position is outside logits sequence length: "
            f"batch={batch_index} local_target_position={local_label_position} "
            f"target_position={label_position} "
            f"logit_position={logit_position} logits_seq_len={sequence_length}"
        )

    if (
        local_label_position < 0
        or label_position >= int(segment_end)
        or local_logit_position < 0
        or logit_position >= int(segment_end)
    ):
        raise ValueError(
            "Stage-2 trie target/logit positions must be inside the same segment: "
            f"batch={batch_index} local_target_position={local_label_position} "
            f"target_position={label_position} "
            f"logit_position={logit_position} "
            f"segment=[{segment_start}, {segment_end})"
        )

    return logit_position


def _target_loss_term(
    *,
    logits_f32: torch.Tensor,
    target: Any,
    batch_index: int,
    logit_position: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Compute source-weighted hard-union CE for one trie token target.

    Positives with the same source weight remain a hard union scaled by that
    source weight: predicting any one same-weight child can satisfy that branch.
    Mixed source weights are split into separate hard-union branches and summed
    with their source coefficients, so a weak fallback child cannot satisfy a
    strong child at full strength.
    """

    token_ids = torch.tensor(
        list(target.positive_token_ids),
        dtype=torch.long,
        device=logits_f32.device,
    )
    if bool(torch.any(token_ids < 0).item()) or bool(
        torch.any(token_ids >= int(logits_f32.shape[-1])).item()
    ):
        raise ValueError(
            "Stage-2 trie positive token id is outside logits vocabulary: "
            f"batch={batch_index} target_position={target.position} "
            f"vocab_size={int(logits_f32.shape[-1])} "
            f"positive_token_ids={tuple(int(i) for i in target.positive_token_ids)}"
        )

    weights = torch.tensor(
        list(target.source_weights),
        dtype=torch.float32,
        device=logits_f32.device,
    )
    active_mask = weights > 0
    if not bool(active_mask.any().item()):
        return None

    active_token_ids = token_ids[active_mask]
    active_weights = weights[active_mask]
    log_probs = torch.log_softmax(logits_f32[batch_index, logit_position], dim=-1)
    unique_weights = torch.unique(active_weights, sorted=True)

    branch_terms: list[torch.Tensor] = []
    for source_weight in unique_weights:
        branch_mask = active_weights == source_weight
        branch_token_ids = active_token_ids[branch_mask]
        branch_positive_log_prob = torch.logsumexp(
            log_probs.index_select(0, branch_token_ids),
            dim=0,
        )
        branch_terms.append(-branch_positive_log_prob * source_weight)

    loss = torch.stack(branch_terms).sum()

    return loss, logits_f32.new_tensor(1.0, dtype=torch.float32)


def run_stage2_trie_ce_module(
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> ModuleResult:
    config = build_stage2_trie_ce_config(spec.config)

    logits_f32 = context.logits.float()
    zero = logits_f32.sum() * 0.0

    loss_terms: list[torch.Tensor] = []
    token_weights: list[torch.Tensor] = []
    active_summaries: list[Any] = []
    role_counts = {role: 0.0 for role in _SEMANTIC_ROLE_METRIC_KEYS}

    for batch_index, segment_start, segment_end, segment_meta in iter_segment_views(
        input_ids=context.input_ids,
        meta=context.meta,
    ):
        if not _is_active_stage2_trie_segment(
            context=context,
            segment_meta=segment_meta,
        ):
            continue

        targets = segment_meta["stage2_trie_targets"]
        active_summaries.append(targets.summary)

        for target in targets.token_targets:
            role_counts[_semantic_role_name(target)] += 1.0
            logit_position = _validate_target_positions(
                target=target,
                batch_index=batch_index,
                segment_start=segment_start,
                segment_end=segment_end,
                sequence_length=int(logits_f32.shape[1]),
            )
            loss_term_with_source = _target_loss_term(
                logits_f32=logits_f32,
                target=target,
                batch_index=batch_index,
                logit_position=logit_position,
            )
            if loss_term_with_source is None:
                continue

            token_weight = max(0.0, _semantic_weight(target, config))
            if token_weight <= 0.0:
                continue

            loss_term, source_multiplier = loss_term_with_source
            weight_tensor = logits_f32.new_tensor(token_weight, dtype=torch.float32)
            loss_terms.append(loss_term * weight_tensor * source_multiplier)
            token_weights.append(weight_tensor)

    if loss_terms:
        numerator = torch.stack(loss_terms).sum()
        denominator = torch.stack(token_weights).sum().clamp_min(1.0e-12)
        loss = numerator / denominator
    else:
        loss = zero

    if active_summaries:
        target_positions = sum(
            float(summary.target_positions) for summary in active_summaries
        )
        branch_points = sum(float(summary.branch_points) for summary in active_summaries)
        max_branching_factor = max(
            float(summary.max_branching_factor) for summary in active_summaries
        )
        candidate_count_mean = sum(
            float(summary.candidate_count) for summary in active_summaries
        ) / float(len(active_summaries))
        fallback_candidate_share = sum(
            _fallback_candidate_share(summary) for summary in active_summaries
        ) / float(len(active_summaries))
        weak_positive_fp_count = sum(
            float(summary.weak_positive_fp_count) for summary in active_summaries
        )
        metrics = {
            "stage2_trie/target_positions": float(target_positions),
            "stage2_trie/branch_points": float(branch_points),
            "stage2_trie/max_branching_factor": float(max_branching_factor),
            "stage2_trie/candidate_count_mean": float(candidate_count_mean),
            "stage2_trie/fallback_candidate_share": float(fallback_candidate_share),
            "stage2_trie/fallback_loss_share": float(fallback_candidate_share),
            "stage2_trie/fallback_dominance_warning": float(
                1.0 if fallback_candidate_share > 0.35 else 0.0
            ),
            "stage2_trie/fp_policy_weak_positive_count": float(
                weak_positive_fp_count
            ),
            "stage2_trie/role_text_targets": float(role_counts["text"]),
            "stage2_trie/role_struct_targets": float(role_counts["struct"]),
            "stage2_trie/role_desc_targets": float(role_counts["desc"]),
            "stage2_trie/role_coord_targets": float(role_counts["coord"]),
            "stage2_trie/role_eos_targets": float(role_counts["eos"]),
            "loss/B/stage2_trie_ce": float(loss.detach().cpu().item()),
        }
    else:
        metrics = _metrics_for_no_active_segments(loss)

    state = {
        "stage2_trie_ce": loss,
        "stage2_trie_ce_contrib": loss,
    }

    return ModuleResult(loss=loss, metrics=metrics, state=state)
