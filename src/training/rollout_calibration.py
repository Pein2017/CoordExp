"""Streaming event-balanced rollout-calibration training seam."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.common.errors import LossContractError, RuntimeContractError
from src.losses import (
    CandidatePath,
    field_balanced_duplicate_rejection_loss,
    GateSiteIdentity,
    LossBundle,
    RolloutGateSite,
    first_wrong_coordinate_preference,
    grouped_entity_transition_preference,
    positive_path_imitation_loss,
    rollout_site_token_type_gate,
)
from src.losses.normalizers import SegmentBalancedDenominator
from src.losses.runner import LossTermResult
from src.rollout_calibration import CalibrationEventMetadata
from src.training.schedule import ResolvedStepSchedule
from src.training.supervised_trainer import SupervisedMicroStep


ENTITY_TERM = "rollout_entity_transition"
POSITIVE_TERM = "rollout_positive_path_imitation"
DUPLICATE_TERM = "rollout_duplicate_rejection"
COORDINATE_TERM = "rollout_coordinate_boundary"
GATE_TERM = "rollout_site_token_type_gate"

_DUPLICATE_TREATMENT_PROFILES = frozenset(
    {
        "recovery_positive_only",
        "local_duplicate_rejection_and_recovery",
        "duplicate_cleaned_imitation_only",
        "combined_duplicate_rejection_and_cleaned_imitation",
    }
)
_COMBINED_DUPLICATE_WINDOW_FAMILIES = (
    "source_route_imitation",
    "local_duplicate_rejection",
    "duplicate_cleaned_imitation",
)
_LOCAL_DUPLICATE_WINDOW_FAMILIES = (
    "source_route_imitation",
    "local_duplicate_rejection",
)


@dataclass(frozen=True)
class CalibrationTokenSequence:
    """Empty ordinary-supervision sidecar for a rollout-only packed row."""

    pack_index: int
    atoms: tuple[Any, ...] = ()


@dataclass(frozen=True)
class RolloutCalibrationLossContext:
    logits: torch.Tensor
    logits_position_ids: tuple[int, ...] | None
    pack_input_ids: tuple[int, ...]
    metadata: CalibrationEventMetadata
    vocab_groups: Any
    coordinate_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class RolloutCalibrationLossPlan:
    denominators: Mapping[str, SegmentBalancedDenominator]
    enabled_terms: tuple[str, ...]
    denominator_scope: str
    world_size: int
    rank: int
    backend_gradient_scale: float


@dataclass(frozen=True)
class RolloutCalibrationLossRunner:
    """Adapt the pure research losses to the existing streaming trainer."""

    profile: str
    entity_weight: float
    entity_margin: float
    entity_smooth_max_temperature: float
    coordinate_weight: float
    coordinate_margin: float
    gate_weight: float
    duplicate_weight: float = 0.0
    duplicate_margin: float = 0.0
    rejection_count: int = 0

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        rejection_count: int = 0,
    ) -> "RolloutCalibrationLossRunner":
        calibration = config.rollout_calibration
        gate = config.losses.protected.rollout_site_token_type_gate
        if calibration is None or gate is None:
            raise LossContractError(
                "rollout calibration requires its objective and gate configuration",
                code="loss.rollout_calibration_config_missing",
            )
        return cls(
            profile=calibration.profile,
            entity_weight=float(calibration.entity_transition.weight),
            entity_margin=float(calibration.entity_transition.margin),
            entity_smooth_max_temperature=float(
                calibration.entity_transition.smooth_max_temperature
            ),
            coordinate_weight=float(calibration.coordinate_boundary.weight),
            coordinate_margin=float(calibration.coordinate_boundary.margin),
            gate_weight=float(gate.weight),
            duplicate_weight=float(calibration.duplicate_rejection.weight),
            duplicate_margin=float(calibration.duplicate_rejection.margin),
            rejection_count=int(rejection_count),
        )

    @property
    def enabled_terms(self) -> tuple[str, ...]:
        terms: list[str] = []
        if _complete_row_profile(self.profile) and self.entity_weight > 0.0:
            terms.append(POSITIVE_TERM)
        elif self.entity_weight > 0.0:
            terms.append(ENTITY_TERM)
        if _duplicate_rejection_profile(self.profile) and self.duplicate_weight > 0.0:
            terms.append(DUPLICATE_TERM)
        if self.coordinate_weight > 0.0:
            terms.append(COORDINATE_TERM)
        terms.append(GATE_TERM)
        return tuple(terms)

    def prepare_planned_step(
        self,
        micro_steps: Sequence[SupervisedMicroStep],
        *,
        denominator_gatherer: Callable[
            [Mapping[str, Mapping[str, Any]]],
            Sequence[Mapping[str, Mapping[str, Any]]],
        ]
        | None = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> RolloutCalibrationLossPlan:
        if not micro_steps:
            raise LossContractError(
                "rollout calibration requires a nonempty planned-step window",
                code="loss.rollout_calibration_empty_window",
            )
        metadata = tuple(_micro_step_metadata(item) for item in micro_steps)
        local = {
            term: _local_denominator(term, metadata, profile=self.profile)
            for term in self.enabled_terms
        }
        checked_world_size = int(world_size)
        checked_rank = int(rank)
        if (
            checked_world_size <= 0
            or checked_rank < 0
            or checked_rank >= checked_world_size
        ):
            raise LossContractError(
                "invalid rollout-calibration rank topology",
                code="loss.rollout_calibration_rank",
                context={"world_size": world_size, "rank": rank},
            )
        if checked_world_size == 1:
            merged = local
            scope = "planned_step"
            backend_scale = 1.0
        else:
            if denominator_gatherer is None:
                raise LossContractError(
                    "multi-rank rollout calibration requires global denominator gathering",
                    code="loss.rollout_calibration_gather_missing",
                    context={"world_size": checked_world_size, "rank": checked_rank},
                )
            gathered = tuple(
                denominator_gatherer(
                    {name: value.to_artifact_dict() for name, value in local.items()}
                )
            )
            merged = _merge_denominators(
                local,
                gathered,
                expected_world_size=checked_world_size,
            )
            scope = "planned_step_global"
            backend_scale = float(checked_world_size)
        for term, denominator in merged.items():
            if denominator.eligible_segment_count <= 0:
                raise LossContractError(
                    "enabled rollout-calibration term has no complete eligible event",
                    code="loss.rollout_calibration_zero_eligible",
                    context={"term": term, "profile": self.profile},
                )
        return RolloutCalibrationLossPlan(
            denominators=merged,
            enabled_terms=self.enabled_terms,
            denominator_scope=scope,
            world_size=checked_world_size,
            rank=checked_rank,
            backend_gradient_scale=backend_scale,
        )

    def compute_micro_step(
        self,
        context: RolloutCalibrationLossContext,
        plan: RolloutCalibrationLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossBundle:
        if not isinstance(context, RolloutCalibrationLossContext):
            raise LossContractError(
                "rollout calibration received the wrong loss context",
                code="loss.rollout_calibration_context_type",
                context={"value_type": type(context).__name__},
            )
        terms: list[LossTermResult] = []
        if ENTITY_TERM in plan.enabled_terms:
            terms.append(
                self._entity_term(
                    context,
                    plan,
                    local_micro_step_index=local_micro_step_index,
                )
            )
        if DUPLICATE_TERM in plan.enabled_terms:
            terms.append(
                self._duplicate_rejection_term(
                    context,
                    plan,
                    local_micro_step_index=local_micro_step_index,
                )
            )
        if POSITIVE_TERM in plan.enabled_terms:
            terms.append(
                self._positive_path_term(
                    context,
                    plan,
                    local_micro_step_index=local_micro_step_index,
                )
            )
        if COORDINATE_TERM in plan.enabled_terms:
            terms.append(
                self._coordinate_term(
                    context,
                    plan,
                    local_micro_step_index=local_micro_step_index,
                )
            )
        terms.append(
            self._gate_term(
                context,
                plan,
                local_micro_step_index=local_micro_step_index,
            )
        )
        total = sum(
            (term.weighted_loss for term in terms),
            context.logits.sum() * 0.0,
        )
        finite = bool(torch.isfinite(total.detach()).all().item()) and all(
            bool(torch.isfinite(term.weighted_loss.detach()).all().item())
            for term in terms
        )
        unknown_entity = sum(
            candidate.entity_review_status != "trusted"
            for candidate in context.metadata.candidates
        )
        unknown_geometry = sum(
            candidate.geometry_review_status != "trusted"
            for candidate in context.metadata.candidates
        )
        count_scale = plan.backend_gradient_scale
        complete_row_family = _complete_row_family(context.metadata, self.profile)
        return LossBundle(
            total_loss=total,
            terms=tuple(terms),
            metrics={
                "loss/total": float(total.detach().item()),
                "calibration/image_balanced_event_weight": float(
                    context.metadata.image_balanced_event_weight
                ),
                "calibration/unknown_entity_count_contribution": float(
                    unknown_entity * count_scale
                ),
                "calibration/unknown_geometry_count_contribution": float(
                    unknown_geometry * count_scale
                ),
            },
            counts={
                "event_count": 1,
                "unknown_entity_count": int(unknown_entity),
                "unknown_geometry_count": int(unknown_geometry),
                "positive_path_imitation_event_count": int(
                    (complete_row_family == "positive_path_imitation") * count_scale
                ),
                "source_route_imitation_event_count": int(
                    (complete_row_family == "source_route_imitation") * count_scale
                ),
                "recovery_positive_imitation_event_count": int(
                    (complete_row_family == "recovery_positive_imitation")
                    * count_scale
                ),
                "duplicate_cleaned_imitation_event_count": int(
                    (complete_row_family == "duplicate_cleaned_imitation")
                    * count_scale
                ),
                "local_duplicate_rejection_event_count": int(
                    context.metadata.local_duplicate_rejection_eligible * count_scale
                ),
            },
            diagnostics={
                "normalizer": "event_balanced",
                "normalizer_scope": "planned_step_streaming",
                "denominator_scope": plan.denominator_scope,
                "event_id": context.metadata.event_id,
                "profile": self.profile,
                "complete_row_imitation_family": complete_row_family,
                "local_micro_step_index": int(local_micro_step_index),
            },
            finite_status={
                "all_finite": finite,
                "total_loss_finite": bool(torch.isfinite(total.detach()).all().item()),
                "terms": {
                    term.name: bool(
                        torch.isfinite(term.weighted_loss.detach()).all().item()
                    )
                    for term in terms
                },
            },
        )

    def finalize_planned_step(
        self,
        micro_loss_artifacts: Sequence[dict[str, Any]],
        plan: RolloutCalibrationLossPlan,
    ) -> dict[str, Any]:
        artifacts = tuple(dict(item) for item in micro_loss_artifacts)
        if not artifacts:
            raise LossContractError(
                "rollout calibration cannot finalize an empty planned step",
                code="loss.rollout_calibration_empty_artifacts",
            )
        term_rows: list[dict[str, Any]] = []
        metrics: dict[str, float] = {}
        total = 0.0
        for term_name in plan.enabled_terms:
            matching = [
                term
                for artifact in artifacts
                for term in artifact.get("terms", ())
                if term.get("name") == term_name
            ]
            if len(matching) != len(artifacts):
                raise LossContractError(
                    "rollout-calibration micro artifacts are missing an enabled term",
                    code="loss.rollout_calibration_term_artifact_missing",
                    context={"term": term_name},
                )
            raw = sum(float(item["raw_loss"]) for item in matching)
            weighted = sum(float(item["weighted_loss"]) for item in matching)
            total += weighted
            denominator = plan.denominators[term_name]
            term_rows.append(
                {
                    "name": term_name,
                    "raw_loss": raw,
                    "weighted_loss": weighted,
                    "weight": float(matching[0]["weight"]),
                    "segment_mean_numerator": sum(
                        float(item["segment_mean_numerator"]) for item in matching
                    ),
                    "denominator": denominator.to_artifact_dict(),
                    "reducer_name": "event_balanced",
                    "selected_count": sum(
                        int(item["selected_count"]) for item in matching
                    ),
                    "skipped_count": sum(
                        int(item["skipped_count"]) for item in matching
                    ),
                    "math_dtype": "float32",
                    "token_weighted_diagnostic": raw,
                    "diagnostics": {
                        "events": [
                            dict(item.get("diagnostics", {})) for item in matching
                        ]
                    },
                }
            )
            metrics[f"loss/{term_name}/raw"] = raw
            metrics[f"loss/{term_name}"] = weighted
            metrics[f"calibration/{term_name}/eligible_event_count"] = float(
                denominator.eligible_segment_count
            )
            metrics[f"calibration/{term_name}/ignored_event_count"] = float(
                denominator.skipped_segment_count
            )
            diagnostics = [dict(item.get("diagnostics", {})) for item in matching]
            if term_name in {ENTITY_TERM, DUPLICATE_TERM, COORDINATE_TERM}:
                metrics[f"calibration/{term_name}/target_margin"] = sum(
                    float(item.get("target_margin_contribution", 0.0))
                    for item in diagnostics
                )
            if term_name == ENTITY_TERM:
                for metric_name in (
                    "continuation_loss",
                    "schema_description_continuation_loss",
                    "coordinate_continuation_loss",
                ):
                    metrics[f"calibration/{term_name}/{metric_name}"] = sum(
                        float(item.get(metric_name, 0.0)) for item in diagnostics
                    )
            if term_name == POSITIVE_TERM:
                for metric_name in (
                    "schema_description_loss",
                    "coordinate_loss",
                    "selected_token_count",
                ):
                    metrics[f"calibration/{term_name}/{metric_name}"] = sum(
                        float(item.get(metric_name, 0.0)) for item in diagnostics
                    )
            if term_name == DUPLICATE_TERM:
                for metric_name in (
                    "positive_row_score",
                    "duplicate_row_score",
                    "positive_description_mean_log_probability",
                    "positive_coordinate_mean_log_probability",
                    "duplicate_description_mean_log_probability",
                    "duplicate_coordinate_mean_log_probability",
                    "selected_token_count",
                ):
                    metrics[f"calibration/{term_name}/{metric_name}"] = sum(
                        float(item.get(metric_name, 0.0)) for item in diagnostics
                    )
            if term_name == GATE_TERM:
                metrics["calibration/rollout_site_token_type_gate/legal_mass"] = sum(
                    float(item.get("legal_mass_contribution", 0.0))
                    for item in diagnostics
                )
        metrics["loss/total"] = total
        metrics["calibration/unknown_entity_count"] = sum(
            float(
                artifact.get("metrics", {}).get(
                    "calibration/unknown_entity_count_contribution", 0.0
                )
            )
            for artifact in artifacts
        )
        metrics["calibration/unknown_geometry_count"] = sum(
            float(
                artifact.get("metrics", {}).get(
                    "calibration/unknown_geometry_count_contribution", 0.0
                )
            )
            for artifact in artifacts
        )
        metrics["calibration/image_balanced_event_weight"] = sum(
            float(
                artifact.get("metrics", {}).get(
                    "calibration/image_balanced_event_weight", 1.0
                )
            )
            for artifact in artifacts
        ) / len(artifacts)
        metrics["calibration/rejected_record_count"] = float(self.rejection_count)
        family_counts = {
            family: sum(
                int(
                    artifact.get("counts", {}).get(
                        f"{family}_event_count", 0
                    )
                )
                for artifact in artifacts
            )
            for family in _profile_family_names(self.profile)
        }
        for family, count in family_counts.items():
            metrics[f"calibration/{family}/admitted_event_count"] = float(count)
        all_finite = all(
            bool(artifact.get("finite_status", {}).get("all_finite"))
            for artifact in artifacts
        )
        metrics["finite/total_loss"] = 1.0 if all_finite else 0.0
        return {
            "total_loss": total,
            "terms": term_rows,
            "metrics": metrics,
            "counts": {
                "event_count": len(artifacts),
                "complete_row_imitation_family_counts": family_counts,
                "global_eligible_event_counts": {
                    name: plan.denominators[name].eligible_segment_count
                    for name in plan.enabled_terms
                },
            },
            "diagnostics": {
                "normalizer": "event_balanced",
                "denominator_scope": plan.denominator_scope,
                "profile": self.profile,
                "world_size": plan.world_size,
                "rank": plan.rank,
            },
            "finite_status": {
                "all_finite": all_finite,
                "total_loss_finite": all_finite,
            },
        }

    def _positive_path_term(
        self,
        context: RolloutCalibrationLossContext,
        plan: RolloutCalibrationLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossTermResult:
        family = _complete_row_family(context.metadata, self.profile)
        eligible = family is not None
        if eligible:
            if context.metadata.entity_transition_eligible or context.metadata.coordinate_boundary_eligible:
                raise LossContractError(
                    "positive-path imitation metadata conflicts with objective-family eligibility",
                    code="loss.rollout_positive_path_event_flags",
                    context={"event_id": context.metadata.event_id},
                )
            candidates = tuple(
                item
                for item in context.metadata.candidates
                if item.role == "positive" and item.entity_eligible
            )
            if len(context.metadata.candidates) != 1 or len(candidates) != 1 or any(
                item.role == "harmful" for item in context.metadata.candidates
            ):
                raise LossContractError(
                    "positive-path imitation event must contain exactly one positive and no harmful candidate",
                    code="loss.rollout_positive_path_event_incomplete",
                    context={
                        "event_id": context.metadata.event_id,
                        "candidate_count": len(context.metadata.candidates),
                        "positive_count": len(candidates),
                    },
                )
            candidate = candidates[0]
            path = _candidate_path(context, candidate)
            result = positive_path_imitation_loss(path)
            raw_event = result.raw_loss
            selected_count = result.selected_token_count
            diagnostics = {
                "event_id": context.metadata.event_id,
                "eligible": True,
                "complete_row_imitation_family": family,
                "schema_description_loss": float(
                    result.schema_description_loss.detach().item()
                ),
                "coordinate_loss": float(result.coordinate_loss.detach().item()),
                "selected_token_count": result.selected_token_count,
                "schema_description_token_count": result.schema_description_token_count,
                "coordinate_token_count": result.coordinate_token_count,
                "image_balanced_event_weight": context.metadata.image_balanced_event_weight,
            }
        else:
            raw_event = context.logits.sum() * 0.0
            selected_count = 0
            diagnostics = {
                "event_id": context.metadata.event_id,
                "eligible": False,
                "complete_row_imitation_family": None,
                "image_balanced_event_weight": context.metadata.image_balanced_event_weight,
            }
        return _term_result(
            name=POSITIVE_TERM,
            raw_event=raw_event,
            weight=self.entity_weight,
            denominator=plan.denominators[POSITIVE_TERM],
            backend_scale=plan.backend_gradient_scale,
            eligible=eligible,
            selected_count=selected_count,
            local_micro_step_index=local_micro_step_index,
            diagnostics=diagnostics,
            metric_name="selected_token_count",
            metric_value=selected_count if eligible else None,
            event_weight=(
                context.metadata.image_balanced_event_weight if eligible else 1.0
            ),
            preserve_declared_event_credit=_uses_declared_event_credit(
                self.profile
            ),
        )

    def _entity_term(
        self,
        context: RolloutCalibrationLossContext,
        plan: RolloutCalibrationLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossTermResult:
        eligible = context.metadata.entity_transition_eligible
        if eligible:
            positives: list[CandidatePath] = []
            harmful: CandidatePath | None = None
            for candidate in context.metadata.candidates:
                if not candidate.entity_eligible:
                    continue
                path = _candidate_path(context, candidate)
                if candidate.role == "positive":
                    positives.append(path)
                elif candidate.role == "harmful":
                    if harmful is not None:
                        raise LossContractError(
                            "entity-transition event declares multiple harmful paths",
                            code="loss.rollout_entity_harmful_count",
                            context={"event_id": context.metadata.event_id},
                        )
                    harmful = path
            if not positives or harmful is None:
                raise LossContractError(
                    "eligible entity-transition event is incomplete",
                    code="loss.rollout_entity_event_incomplete",
                    context={"event_id": context.metadata.event_id},
                )
            result = grouped_entity_transition_preference(
                tuple(positives),
                harmful,
                margin=self.entity_margin,
                smooth_max_temperature=self.entity_smooth_max_temperature,
            )
            raw_event = result.raw_loss
            selected_count = sum(
                len(path.target_token_ids) for path in positives
            ) + len(harmful.target_token_ids)
            diagnostics = {
                "event_id": context.metadata.event_id,
                "target_margin": float(result.target_margin.detach().item()),
                "branch_margin": float(result.target_margin.detach().item()),
                "continuation_loss": float(
                    result.positive_continuation_loss.detach().item()
                ),
                "schema_description_continuation_loss": float(
                    result.positive_schema_description_loss.detach().item()
                ),
                "coordinate_continuation_loss": float(
                    result.positive_coordinate_loss.detach().item()
                ),
                "first_divergence_offsets": {
                    candidate_id: int(offset)
                    for candidate_id, offset in result.first_divergence_offsets
                },
                "positive_path_count": result.positive_path_count,
                "distinct_owner_count": result.distinct_owner_count,
                "duplicate_alias_count": result.duplicate_alias_count,
                "positive_mean_log_probability": float(
                    torch.stack(
                        tuple(
                            item.mean_log_probability for item in result.positive_paths
                        )
                    )
                    .mean()
                    .detach()
                    .item()
                ),
                "harmful_mean_log_probability": float(
                    result.harmful_path.mean_log_probability.detach().item()
                ),
            }
        else:
            raw_event = context.logits.sum() * 0.0
            selected_count = 0
            diagnostics = {"event_id": context.metadata.event_id, "eligible": False}
        return _term_result(
            name=ENTITY_TERM,
            raw_event=raw_event,
            weight=self.entity_weight,
            denominator=plan.denominators[ENTITY_TERM],
            backend_scale=plan.backend_gradient_scale,
            eligible=eligible,
            selected_count=selected_count,
            local_micro_step_index=local_micro_step_index,
            diagnostics=diagnostics,
            metric_name="target_margin_contribution",
            metric_value=diagnostics.get("target_margin"),
        )

    def _coordinate_term(
        self,
        context: RolloutCalibrationLossContext,
        plan: RolloutCalibrationLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossTermResult:
        eligible = context.metadata.coordinate_boundary_eligible
        if eligible:
            candidates = [
                item
                for item in context.metadata.candidates
                if item.geometry_eligible and item.coordinate_decision is not None
            ]
            if len(candidates) != 1:
                raise LossContractError(
                    "eligible coordinate-boundary event must declare one wrong decision",
                    code="loss.rollout_coordinate_event_incomplete",
                    context={
                        "event_id": context.metadata.event_id,
                        "candidate_count": len(candidates),
                    },
                )
            candidate = candidates[0]
            decision = candidate.coordinate_decision
            assert decision is not None
            logits_position = candidate.coordinate_physical_logits_position
            target_position = candidate.coordinate_physical_target_position
            if logits_position is None or target_position is None:
                raise LossContractError(
                    "coordinate-boundary metadata is missing physical positions",
                    code="loss.rollout_coordinate_position_missing",
                    context={"event_id": context.metadata.event_id},
                )
            wrong_id = _coordinate_token_id(
                context.coordinate_token_ids,
                decision.actual_wrong_coordinate_value,
            )
            observed_id = context.pack_input_ids[target_position]
            if observed_id != wrong_id:
                raise LossContractError(
                    "stored wrong coordinate value does not match its exact candidate token",
                    code="loss.rollout_coordinate_token_mismatch",
                    context={
                        "event_id": context.metadata.event_id,
                        "candidate_id": candidate.candidate_id,
                        "expected_token_id": wrong_id,
                        "observed_token_id": observed_id,
                    },
                )
            result = first_wrong_coordinate_preference(
                _logits_row(context, logits_position),
                acceptable_token_ids=tuple(
                    _coordinate_token_id(context.coordinate_token_ids, value)
                    for value in decision.acceptable_coordinate_values
                ),
                wrong_token_id=wrong_id,
                margin=self.coordinate_margin,
            )
            raw_event = result.raw_loss
            selected_count = 1
            diagnostics = {
                "event_id": context.metadata.event_id,
                "target_margin": float(result.target_margin.detach().item()),
                "acceptable_token_count": result.acceptable_token_count,
            }
        else:
            raw_event = context.logits.sum() * 0.0
            selected_count = 0
            diagnostics = {"event_id": context.metadata.event_id, "eligible": False}
        return _term_result(
            name=COORDINATE_TERM,
            raw_event=raw_event,
            weight=self.coordinate_weight,
            denominator=plan.denominators[COORDINATE_TERM],
            backend_scale=plan.backend_gradient_scale,
            eligible=eligible,
            selected_count=selected_count,
            local_micro_step_index=local_micro_step_index,
            diagnostics=diagnostics,
            metric_name="target_margin_contribution",
            metric_value=diagnostics.get("target_margin"),
        )

    def _duplicate_rejection_term(
        self,
        context: RolloutCalibrationLossContext,
        plan: RolloutCalibrationLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossTermResult:
        eligible = context.metadata.local_duplicate_rejection_eligible
        if eligible:
            positives = tuple(
                candidate
                for candidate in context.metadata.candidates
                if candidate.role == "positive" and candidate.entity_eligible
            )
            duplicates = tuple(
                candidate
                for candidate in context.metadata.candidates
                if (
                    candidate.role == "harmful"
                    and candidate.harmful_kind == "duplicate"
                    and candidate.entity_eligible
                )
            )
            if len(positives) != 1 or len(duplicates) != 1:
                raise LossContractError(
                    "local duplicate-rejection event must declare one recovery and one duplicate row",
                    code="loss.rollout_duplicate_event_incomplete",
                    context={
                        "event_id": context.metadata.event_id,
                        "positive_count": len(positives),
                        "duplicate_count": len(duplicates),
                    },
                )
            positive_path = _candidate_path(
                context,
                positives[0],
                complete_row_token_types=True,
            )
            duplicate_path = _candidate_path(
                context,
                duplicates[0],
                complete_row_token_types=True,
            )
            result = field_balanced_duplicate_rejection_loss(
                positive_path,
                duplicate_path,
                margin=self.duplicate_margin,
            )
            raw_event = result.raw_loss
            selected_count = result.selected_token_count
            diagnostics = {
                "event_id": context.metadata.event_id,
                "eligible": True,
                "target_margin": float(result.target_margin.detach().item()),
                "positive_row_score": float(result.positive_row_score.detach().item()),
                "duplicate_row_score": float(result.duplicate_row_score.detach().item()),
                "positive_description_mean_log_probability": float(
                    result.positive_description_mean_log_probability.detach().item()
                ),
                "positive_coordinate_mean_log_probability": float(
                    result.positive_coordinate_mean_log_probability.detach().item()
                ),
                "duplicate_description_mean_log_probability": float(
                    result.duplicate_description_mean_log_probability.detach().item()
                ),
                "duplicate_coordinate_mean_log_probability": float(
                    result.duplicate_coordinate_mean_log_probability.detach().item()
                ),
                "positive_description_token_count": result.positive_description_token_count,
                "positive_coordinate_token_count": result.positive_coordinate_token_count,
                "duplicate_description_token_count": result.duplicate_description_token_count,
                "duplicate_coordinate_token_count": result.duplicate_coordinate_token_count,
                "selected_token_count": selected_count,
                "burst_credit": (
                    None
                    if context.metadata.duplicate_trajectory_evidence is None
                    else context.metadata.duplicate_trajectory_evidence.burst_credit
                ),
                "image_balanced_event_weight": context.metadata.image_balanced_event_weight,
            }
        else:
            raw_event = context.logits.sum() * 0.0
            selected_count = 0
            diagnostics = {"event_id": context.metadata.event_id, "eligible": False}
        return _term_result(
            name=DUPLICATE_TERM,
            raw_event=raw_event,
            weight=self.duplicate_weight,
            denominator=plan.denominators[DUPLICATE_TERM],
            backend_scale=plan.backend_gradient_scale,
            eligible=eligible,
            selected_count=selected_count,
            local_micro_step_index=local_micro_step_index,
            diagnostics=diagnostics,
            metric_name="target_margin_contribution",
            metric_value=diagnostics.get("target_margin"),
            event_weight=(
                context.metadata.image_balanced_event_weight if eligible else 1.0
            ),
            preserve_declared_event_credit=_uses_declared_event_credit(
                self.profile
            ),
        )

    def _gate_term(
        self,
        context: RolloutCalibrationLossContext,
        plan: RolloutCalibrationLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossTermResult:
        declarations = tuple(
            RolloutGateSite(
                event_id=context.metadata.event_id,
                identity=GateSiteIdentity(
                    segment_id=f"{context.metadata.event_id}:{candidate.candidate_id}",
                    logits_position=site.physical_logits_position,
                ),
                logits=_logits_row(context, site.physical_logits_position),
                intended_token_type=site.intended_token_type,
                allowed_token_ids=tuple(
                    context.vocab_groups.allowed_ids(site.intended_token_type)
                ),
            )
            for candidate in context.metadata.candidates
            for site in _active_selected_sites(candidate, self.profile)
        )
        result = rollout_site_token_type_gate(declarations)
        diagnostics = {
            "event_id": context.metadata.event_id,
            "legal_mass": float(result.legal_mass.detach().item()),
            "distinct_site_count": result.distinct_site_count,
            "declaration_count": result.declaration_count,
            "duplicate_declaration_count": result.duplicate_declaration_count,
        }
        return _term_result(
            name=GATE_TERM,
            raw_event=result.raw_loss,
            weight=self.gate_weight,
            denominator=plan.denominators[GATE_TERM],
            backend_scale=plan.backend_gradient_scale,
            eligible=True,
            selected_count=result.distinct_site_count,
            local_micro_step_index=local_micro_step_index,
            diagnostics=diagnostics,
            metric_name="legal_mass_contribution",
            metric_value=diagnostics["legal_mass"],
            event_weight=(
                context.metadata.image_balanced_event_weight
                if _weighted_duplicate_or_complete_row_event(
                    context.metadata, self.profile
                )
                else 1.0
            ),
            preserve_declared_event_credit=(
                _uses_declared_event_credit(self.profile)
                and _weighted_duplicate_or_complete_row_event(
                    context.metadata, self.profile
                )
            ),
        )


def rollout_calibration_loss_context(
    micro_step: SupervisedMicroStep,
    forward_result: Any,
) -> RolloutCalibrationLossContext:
    metadata = _micro_step_metadata(micro_step)
    coordinate_token_ids = (
        micro_step.metadata.get("coordinate_token_ids") if micro_step.metadata else None
    )
    if not isinstance(coordinate_token_ids, tuple):
        raise RuntimeContractError(
            "rollout-calibration micro-step is missing semantic coordinate-token order",
            code="training.rollout_coordinate_identity_missing",
        )
    return RolloutCalibrationLossContext(
        logits=forward_result.logits,
        logits_position_ids=forward_result.logits_position_ids,
        pack_input_ids=tuple(int(value) for value in micro_step.pack.input_ids),
        metadata=metadata,
        vocab_groups=micro_step.vocab_groups,
        coordinate_token_ids=coordinate_token_ids,
    )


def build_calibration_micro_step_stream(
    base_micro_steps: Sequence[SupervisedMicroStep],
    schedule: ResolvedStepSchedule,
    *,
    profile: str,
    rank: int,
    world_size: int,
) -> Iterator[SupervisedMicroStep]:
    """Expose each profile-admitted frozen-bank event exactly once."""

    if not base_micro_steps:
        raise RuntimeContractError(
            "rollout calibration has no admitted train events",
            code="training.rollout_calibration_empty_train",
        )
    admitted = tuple(
        item
        for item in base_micro_steps
        if _profile_admits(_micro_step_metadata(item), profile)
    )
    if not admitted:
        raise RuntimeContractError(
            "rollout calibration profile admits no train events",
            code="training.rollout_calibration_profile_empty",
            context={"profile": profile},
        )
    if profile in _DUPLICATE_TREATMENT_PROFILES and not any(
        _micro_step_metadata(item).source_route_imitation_eligible
        for item in admitted
    ):
        raise RuntimeContractError(
            "duplicate-treatment calibration requires its fixed Source-preservation family",
            code="training.rollout_calibration_source_family_missing",
            context={"profile": profile},
        )
    if (
        world_size != schedule.runtime_batch.world_size
        or rank < 0
        or rank >= world_size
    ):
        raise RuntimeContractError(
            "rollout-calibration stream rank topology differs from the schedule",
            code="training.rollout_calibration_rank",
            context={"rank": rank, "world_size": world_size},
        )
    slots = schedule.runtime_batch.effective_batch_size
    scheduled_exposure = schedule.resolved_max_steps * slots
    if scheduled_exposure != len(admitted):
        raise RuntimeContractError(
            "rollout-calibration schedule must expose every admitted frozen-bank event exactly once",
            code="training.rollout_calibration_frozen_exposure",
            context={
                "profile": profile,
                "admitted_event_count": len(admitted),
                "scheduled_event_count": scheduled_exposure,
                "resolved_max_steps": schedule.resolved_max_steps,
                "effective_batch_size": slots,
            },
        )
    if profile == "combined_duplicate_rejection_and_cleaned_imitation":
        admitted = _family_stratified_combined_events(
            admitted,
            slots=slots,
            planned_step_count=schedule.resolved_max_steps,
        )
    elif profile == "local_duplicate_rejection_and_recovery":
        admitted = _family_stratified_local_duplicate_events(
            admitted,
            slots=slots,
            planned_step_count=schedule.resolved_max_steps,
        )
    if profile == "joint":
        complete_families = {
            "entity": any(
                _micro_step_metadata(item).entity_transition_eligible
                for item in admitted
            ),
            "coordinate": any(
                _micro_step_metadata(item).coordinate_boundary_eligible
                for item in admitted
            ),
        }
        if not all(complete_families.values()):
            raise RuntimeContractError(
                "complete frozen joint schedule must exercise both objective families",
                code="training.rollout_calibration_joint_family_missing",
                context=complete_families,
            )
    for planned_index in range(schedule.resolved_max_steps):
        window = admitted[planned_index * slots : (planned_index + 1) * slots]
        if profile == "combined_duplicate_rejection_and_cleaned_imitation":
            _validate_combined_window(
                window,
                planned_step_id=planned_index + 1,
            )
        elif profile == "local_duplicate_rejection_and_recovery":
            _validate_local_duplicate_window(
                window,
                planned_step_id=planned_index + 1,
            )
        for local_accum_index in range(
            schedule.runtime_batch.resolved_grad_accum_steps
        ):
            yield window[local_accum_index * world_size + rank]


def _family_stratified_combined_events(
    admitted: tuple[SupervisedMicroStep, ...],
    *,
    slots: int,
    planned_step_count: int,
) -> tuple[SupervisedMicroStep, ...]:
    """Deterministically place both combined families in every global window."""

    return _family_stratified_duplicate_events(
        admitted,
        slots=slots,
        planned_step_count=planned_step_count,
        family_order=_COMBINED_DUPLICATE_WINDOW_FAMILIES,
        profile_label="combined duplicate",
        error_code_prefix="combined",
    )


def _family_stratified_local_duplicate_events(
    admitted: tuple[SupervisedMicroStep, ...],
    *,
    slots: int,
    planned_step_count: int,
) -> tuple[SupervisedMicroStep, ...]:
    """Place Source and local-duplicate events in every global window."""

    return _family_stratified_duplicate_events(
        admitted,
        slots=slots,
        planned_step_count=planned_step_count,
        family_order=_LOCAL_DUPLICATE_WINDOW_FAMILIES,
        profile_label="local duplicate",
        error_code_prefix="local_duplicate",
    )


def _family_stratified_duplicate_events(
    admitted: tuple[SupervisedMicroStep, ...],
    *,
    slots: int,
    planned_step_count: int,
    family_order: tuple[str, ...],
    profile_label: str,
    error_code_prefix: str,
) -> tuple[SupervisedMicroStep, ...]:
    """Deterministically cover every required duplicate family per window."""

    if slots < len(family_order):
        raise RuntimeContractError(
            f"{profile_label} calibration requires one global slot for every enabled family per planned step",
            code=f"training.rollout_calibration_{error_code_prefix}_window_capacity",
            context={
                "effective_batch_size": slots,
                "required_family_count": len(family_order),
            },
        )
    queues = {
        family: sorted(
            (
                item
                for item in admitted
                if _combined_window_family(_micro_step_metadata(item)) == family
            ),
            key=lambda item: _micro_step_metadata(item).event_id,
        )
        for family in family_order
    }
    if any(len(queues[family]) < planned_step_count for family in family_order):
        raise RuntimeContractError(
            f"{profile_label} calibration lacks one enabled family for a planned global window",
            code=f"training.rollout_calibration_{error_code_prefix}_family_missing",
            context={
                "planned_step_id": None,
                "required_windows": planned_step_count,
                **{
                    f"{family}_event_count": len(queues[family])
                    for family in family_order
                },
                "missing_families": [
                    family
                    for family in family_order
                    if len(queues[family]) < planned_step_count
                ],
            },
        )
    if sum(len(queues[family]) for family in family_order) != len(admitted):
        raise RuntimeContractError(
            f"{profile_label} calibration admits an unsupported event family",
            code=f"training.rollout_calibration_{error_code_prefix}_family_unknown",
            context={
                "admitted_event_count": len(admitted),
                **{
                    f"{family}_event_count": len(queues[family])
                    for family in family_order
                },
            },
        )

    queues = {family: list(items) for family, items in queues.items()}
    windows: list[SupervisedMicroStep] = []
    for planned_index in range(planned_step_count):
        window: list[SupervisedMicroStep] = []
        for family in family_order:
            window.append(queues[family].pop(0))
        remaining_windows = planned_step_count - planned_index - 1
        while len(window) < slots:
            eligible_families = [
                family
                for family in family_order
                if len(queues[family]) > remaining_windows
            ]
            if not eligible_families:
                raise RuntimeContractError(
                    f"{profile_label} calibration cannot fill a family-stratified planned window",
                    code=f"training.rollout_calibration_{error_code_prefix}_window_fill",
                    context={
                        "planned_step_id": planned_index + 1,
                        "effective_batch_size": slots,
                        "remaining_windows": remaining_windows,
                        "remaining_by_family": {
                            family: len(queues[family]) for family in family_order
                        },
                    },
                )
            family = max(
                eligible_families,
                key=lambda item: (len(queues[item]), -family_order.index(item)),
            )
            window.append(queues[family].pop(0))
        windows.extend(window)
    if any(queues[family] for family in family_order):
        raise RuntimeContractError(
            f"{profile_label} calibration left an unplanned admitted event",
            code=f"training.rollout_calibration_{error_code_prefix}_window_residue",
            context={
                "remaining_by_family": {
                    family: len(queues[family]) for family in family_order
                }
            },
        )
    return tuple(windows)


def _validate_combined_window(
    window: Sequence[SupervisedMicroStep],
    *,
    planned_step_id: int,
) -> None:
    present = {
        family: any(
            _combined_window_family(_micro_step_metadata(item)) == family
            for item in window
        )
        for family in _COMBINED_DUPLICATE_WINDOW_FAMILIES
    }
    missing = [family for family, available in present.items() if not available]
    if missing:
        raise RuntimeContractError(
            "combined duplicate calibration window lacks an enabled global family before forward",
            code="training.rollout_calibration_combined_family_missing",
            context={
                "planned_step_id": planned_step_id,
                "missing_families": missing,
                "event_ids": [
                    _micro_step_metadata(item).event_id for item in window
                ],
            },
        )


def _validate_local_duplicate_window(
    window: Sequence[SupervisedMicroStep],
    *,
    planned_step_id: int,
) -> None:
    present = {
        family: any(
            _combined_window_family(_micro_step_metadata(item)) == family
            for item in window
        )
        for family in _LOCAL_DUPLICATE_WINDOW_FAMILIES
    }
    missing = [family for family, available in present.items() if not available]
    if missing:
        raise RuntimeContractError(
            "local duplicate calibration window lacks an enabled global family before forward",
            code="training.rollout_calibration_local_duplicate_family_missing",
            context={
                "planned_step_id": planned_step_id,
                "missing_families": missing,
                "event_ids": [
                    _micro_step_metadata(item).event_id for item in window
                ],
            },
        )


def _combined_window_family(metadata: CalibrationEventMetadata) -> str | None:
    if metadata.source_route_imitation_eligible:
        return "source_route_imitation"
    if metadata.local_duplicate_rejection_eligible:
        return "local_duplicate_rejection"
    if metadata.duplicate_cleaned_imitation_eligible:
        return "duplicate_cleaned_imitation"
    return None


def _uses_declared_event_credit(profile: str) -> bool:
    """Keep duplicate-bank credits additive after planned-step normalization.

    Historical profiles continue to use event-balanced means.  Duplicate-bank
    event weights instead encode declared burst/image credit; multiplying by
    the planned global eligible count cancels the reducer's event-count mean
    so a burst's declared credits add to its intended total.
    """

    return profile in _DUPLICATE_TREATMENT_PROFILES


def _profile_admits(metadata: CalibrationEventMetadata, profile: str) -> bool:
    if profile == "positive_path_imitation_only":
        return metadata.positive_path_imitation_eligible
    if profile == "sampled_path_and_source_route_imitation_only":
        return bool(
            metadata.positive_path_imitation_eligible
            or metadata.source_route_imitation_eligible
        )
    if profile == "recovery_positive_only":
        return bool(
            metadata.source_route_imitation_eligible
            or metadata.recovery_positive_imitation_eligible
        )
    if profile == "local_duplicate_rejection_and_recovery":
        return bool(
            metadata.source_route_imitation_eligible
            or metadata.local_duplicate_rejection_eligible
        )
    if profile == "duplicate_cleaned_imitation_only":
        return bool(
            metadata.source_route_imitation_eligible
            or metadata.duplicate_cleaned_imitation_eligible
        )
    if profile == "combined_duplicate_rejection_and_cleaned_imitation":
        return bool(
            metadata.source_route_imitation_eligible
            or metadata.local_duplicate_rejection_eligible
            or metadata.duplicate_cleaned_imitation_eligible
        )
    if profile == "transition_only":
        return metadata.entity_transition_eligible
    if profile in {"coordinate_boundary_only", "coordinate_boundary_gate_only"}:
        return metadata.coordinate_boundary_eligible
    if profile == "joint":
        return (
            metadata.entity_transition_eligible or metadata.coordinate_boundary_eligible
        )
    raise RuntimeContractError(
        "unknown rollout-calibration profile",
        code="training.rollout_calibration_profile_unknown",
        context={"profile": profile},
    )


def _micro_step_metadata(micro_step: SupervisedMicroStep) -> CalibrationEventMetadata:
    metadata = micro_step.calibration_metadata
    if not isinstance(metadata, CalibrationEventMetadata):
        raise LossContractError(
            "rollout calibration requires typed event metadata",
            code="loss.rollout_calibration_metadata_type",
            context={"value_type": type(metadata).__name__},
        )
    return metadata


def _local_denominator(
    term: str,
    metadata: tuple[CalibrationEventMetadata, ...],
    *,
    profile: str,
) -> SegmentBalancedDenominator:
    eligible = sum(_event_eligible(item, term, profile=profile) for item in metadata)
    selected = sum(_selected_count(item, term, profile=profile) for item in metadata)
    return SegmentBalancedDenominator(
        term_name=term,
        denominator_scope="planned_step",
        eligible_segment_count=int(eligible),
        selected_atom_count=int(selected),
        skipped_segment_count=len(metadata) - int(eligible),
        context_count=len(metadata),
    )


def _merge_denominators(
    local: Mapping[str, SegmentBalancedDenominator],
    gathered: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    expected_world_size: int,
) -> dict[str, SegmentBalancedDenominator]:
    if len(gathered) != expected_world_size:
        raise LossContractError(
            "rollout-calibration denominator gather returned the wrong rank count",
            code="loss.rollout_calibration_gather_count",
            context={"expected": expected_world_size, "actual": len(gathered)},
        )
    merged: dict[str, SegmentBalancedDenominator] = {}
    for term in local:
        try:
            rows = tuple(payload[term] for payload in gathered)
        except (KeyError, TypeError) as exc:
            raise LossContractError(
                "rollout-calibration denominator gather omitted an enabled term",
                code="loss.rollout_calibration_gather_payload",
                context={"term": term},
                cause=exc,
            ) from exc
        merged[term] = SegmentBalancedDenominator(
            term_name=term,
            denominator_scope="planned_step_global",
            eligible_segment_count=sum(
                int(row["eligible_segment_count"]) for row in rows
            ),
            selected_atom_count=sum(int(row["selected_atom_count"]) for row in rows),
            skipped_segment_count=sum(
                int(row["skipped_segment_count"]) for row in rows
            ),
            context_count=sum(int(row["context_count"]) for row in rows),
        )
    return merged


def _event_eligible(
    metadata: CalibrationEventMetadata, term: str, *, profile: str
) -> bool:
    if term == POSITIVE_TERM:
        return _complete_row_family(metadata, profile) is not None
    if term == DUPLICATE_TERM:
        return metadata.local_duplicate_rejection_eligible
    if term == ENTITY_TERM:
        return metadata.entity_transition_eligible
    if term == COORDINATE_TERM:
        return metadata.coordinate_boundary_eligible
    if term == GATE_TERM:
        return any(
            _active_selected_sites(candidate, profile)
            for candidate in metadata.candidates
        )
    raise KeyError(term)


def _selected_count(
    metadata: CalibrationEventMetadata, term: str, *, profile: str
) -> int:
    if term == POSITIVE_TERM:
        if not _event_eligible(metadata, term, profile=profile):
            return 0
        return sum(
            1
            for candidate in metadata.candidates
            if candidate.role == "positive" and candidate.entity_eligible
            for _site in _active_selected_sites(candidate, profile)
        )
    if term == DUPLICATE_TERM:
        if not _event_eligible(metadata, term, profile=profile):
            return 0
        return sum(
            1
            for candidate in metadata.candidates
            if candidate.role in {"positive", "harmful"}
            for site in candidate.selected_sites
            if site.intended_token_type in {"desc_text", "coordinate"}
        )
    if term == GATE_TERM:
        return len(
            {
                (candidate.candidate_id, site.physical_logits_position)
                for candidate in metadata.candidates
                for site in _active_selected_sites(candidate, profile)
            }
        )
    if not _event_eligible(metadata, term, profile=profile):
        return 0
    if term == COORDINATE_TERM:
        return 1
    count = 0
    for candidate in metadata.candidates:
        if not candidate.entity_eligible:
            continue
        if candidate.owner_resolution_physical_target_interval is None:
            count += 1
        else:
            start, end = candidate.owner_resolution_physical_target_interval
            count += end - start
    return count


def _active_selected_sites(candidate: Any, profile: str) -> tuple[Any, ...]:
    if profile in {
        "positive_path_imitation_only",
        "sampled_path_and_source_route_imitation_only",
        "recovery_positive_only",
        "duplicate_cleaned_imitation_only",
    }:
        if not candidate.entity_eligible or candidate.role != "positive":
            return ()
        interval = candidate.owner_resolution_candidate_interval
        if interval is None:
            return ()
        start, end = interval
        return tuple(
            site
            for site in candidate.selected_sites
            if start <= site.candidate_token_offset < end
            and not (
                site.intended_token_type == "coordinate"
                and candidate.geometry_review_status != "trusted"
            )
        )
    if profile in {
        "local_duplicate_rejection_and_recovery",
        "combined_duplicate_rejection_and_cleaned_imitation",
    }:
        if not candidate.entity_eligible:
            return ()
        interval = candidate.owner_resolution_candidate_interval
        if interval is None:
            return ()
        start, end = interval
        return tuple(
            site
            for site in candidate.selected_sites
            if start <= site.candidate_token_offset < end
            and not (
                site.intended_token_type == "coordinate"
                and candidate.geometry_review_status != "trusted"
            )
        )
    active_offsets: set[int] = set()
    if profile in {"transition_only", "joint"} and candidate.entity_eligible:
        interval = candidate.owner_resolution_candidate_interval
        if interval is None:
            if candidate.harmful_kind == "premature_terminal":
                active_offsets.add(0)
        else:
            start, end = interval
            active_offsets.update(range(start, end))
    if (
        profile
        in {"coordinate_boundary_only", "coordinate_boundary_gate_only", "joint"}
        and candidate.geometry_eligible
        and candidate.coordinate_decision is not None
    ):
        active_offsets.add(candidate.coordinate_decision.candidate_token_offset)
    return tuple(
        site
        for site in candidate.selected_sites
        if site.candidate_token_offset in active_offsets
    )


def _complete_row_profile(profile: str) -> bool:
    return profile in {
        "positive_path_imitation_only",
        "sampled_path_and_source_route_imitation_only",
        "recovery_positive_only",
        "local_duplicate_rejection_and_recovery",
        "duplicate_cleaned_imitation_only",
        "combined_duplicate_rejection_and_cleaned_imitation",
    }


def _duplicate_rejection_profile(profile: str) -> bool:
    return profile in {
        "local_duplicate_rejection_and_recovery",
        "combined_duplicate_rejection_and_cleaned_imitation",
    }


def _complete_row_family(
    metadata: CalibrationEventMetadata,
    profile: str,
) -> str | None:
    enabled_families = tuple(
        family
        for family, enabled in (
            ("positive_path_imitation", metadata.positive_path_imitation_eligible),
            ("source_route_imitation", metadata.source_route_imitation_eligible),
            (
                "recovery_positive_imitation",
                metadata.recovery_positive_imitation_eligible,
            ),
            (
                "duplicate_cleaned_imitation",
                metadata.duplicate_cleaned_imitation_eligible,
            ),
        )
        if enabled
    )
    if len(enabled_families) > 1:
        raise LossContractError(
            "one event cannot enable both complete-row imitation families",
            code="loss.rollout_complete_row_family_conflict",
            context={"event_id": metadata.event_id, "families": enabled_families},
        )
    positive_path = metadata.positive_path_imitation_eligible
    source_route = metadata.source_route_imitation_eligible
    if profile == "positive_path_imitation_only":
        return "positive_path_imitation" if positive_path else None
    if profile == "sampled_path_and_source_route_imitation_only":
        if positive_path:
            return "positive_path_imitation"
        if source_route:
            return "source_route_imitation"
    if profile == "recovery_positive_only":
        if source_route:
            return "source_route_imitation"
        return (
            "recovery_positive_imitation"
            if metadata.recovery_positive_imitation_eligible
            else None
        )
    if profile == "duplicate_cleaned_imitation_only":
        if source_route:
            return "source_route_imitation"
        return (
            "duplicate_cleaned_imitation"
            if metadata.duplicate_cleaned_imitation_eligible
            else None
        )
    if profile == "combined_duplicate_rejection_and_cleaned_imitation":
        if source_route:
            return "source_route_imitation"
        return (
            "duplicate_cleaned_imitation"
            if metadata.duplicate_cleaned_imitation_eligible
            else None
        )
    if profile == "local_duplicate_rejection_and_recovery":
        return "source_route_imitation" if source_route else None
    return None


def _weighted_duplicate_or_complete_row_event(
    metadata: CalibrationEventMetadata,
    profile: str,
) -> bool:
    return bool(
        metadata.local_duplicate_rejection_eligible
        or _complete_row_family(metadata, profile) is not None
    )


def _profile_family_names(profile: str) -> tuple[str, ...]:
    if profile == "sampled_path_and_source_route_imitation_only":
        return ("positive_path_imitation", "source_route_imitation")
    if profile == "recovery_positive_only":
        return ("source_route_imitation", "recovery_positive_imitation")
    if profile == "local_duplicate_rejection_and_recovery":
        return ("source_route_imitation", "local_duplicate_rejection")
    if profile == "duplicate_cleaned_imitation_only":
        return ("source_route_imitation", "duplicate_cleaned_imitation")
    if profile == "combined_duplicate_rejection_and_cleaned_imitation":
        return (
            "source_route_imitation",
            "local_duplicate_rejection",
            "duplicate_cleaned_imitation",
        )
    if profile == "positive_path_imitation_only":
        return ("positive_path_imitation",)
    return ()


def _candidate_path(
    context: RolloutCalibrationLossContext,
    candidate: Any,
    *,
    complete_row_token_types: bool = False,
) -> CandidatePath:
    interval = candidate.owner_resolution_physical_target_interval
    premature = candidate.harmful_kind == "premature_terminal"
    if interval is None:
        if not premature or len(candidate.selected_sites) != 1:
            raise LossContractError(
                "entity candidate lacks one unambiguous score interval",
                code="loss.rollout_entity_interval_missing",
                context={"candidate_id": candidate.candidate_id},
            )
        start = candidate.selected_sites[0].physical_target_position
        end = start + 1
    else:
        start, end = interval
    logits = torch.stack(
        tuple(_logits_row(context, position) for position in range(start - 1, end - 1))
    )
    token_types: tuple[str, ...] | None = None
    if candidate.role == "positive" or complete_row_token_types:
        interval = candidate.owner_resolution_candidate_interval
        if interval is None:
            raise LossContractError(
                "complete-row candidate lacks a token-type interval",
                code="loss.rollout_continuation_interval_missing",
                context={"candidate_id": candidate.candidate_id},
            )
        interval_start, interval_end = interval
        token_type_by_offset = {
            site.candidate_token_offset: site.intended_token_type
            for site in candidate.selected_sites
        }
        missing_offsets = [
            offset
            for offset in range(interval_start, interval_end)
            if offset not in token_type_by_offset
        ]
        if missing_offsets:
            raise LossContractError(
                "positive continuation is missing intended token-type sites",
                code="loss.rollout_continuation_token_type_missing",
                context={
                    "candidate_id": candidate.candidate_id,
                    "missing_offsets": missing_offsets,
                },
            )
        token_types_list: list[str] = []
        for offset in range(interval_start, interval_end):
            intended = token_type_by_offset[offset]
            if (
                intended == "coordinate"
                and candidate.geometry_review_status != "trusted"
            ):
                # Keep entity/schema supervision while excluding exact-token
                # coordinate supervision whose geometry was not trusted.
                intended = "untrusted_coordinate"
            token_types_list.append(intended)
        token_types = tuple(token_types_list)
    return CandidatePath(
        candidate_id=candidate.candidate_id,
        physical_owner_id=candidate.physical_owner_id,
        logits=logits,
        target_token_ids=tuple(context.pack_input_ids[start:end]),
        premature_terminal=premature,
        token_types=token_types,
    )


def _logits_row(
    context: RolloutCalibrationLossContext, physical_position: int
) -> torch.Tensor:
    logits = context.logits
    if logits.ndim == 3:
        if int(logits.shape[0]) != 1:
            raise LossContractError(
                "rollout calibration expects one no-padding packed row",
                code="loss.rollout_logits_batch",
                context={"shape": list(logits.shape)},
            )
        logits = logits[0]
    if logits.ndim != 2:
        raise LossContractError(
            "rollout calibration logits must be rank two or singleton-batch rank three",
            code="loss.rollout_logits_shape",
            context={"shape": list(context.logits.shape)},
        )
    if context.logits_position_ids is None:
        row_index = physical_position
    else:
        try:
            row_index = context.logits_position_ids.index(physical_position)
        except ValueError as exc:
            raise LossContractError(
                "compact Qwen logits omitted a selected calibration site",
                code="loss.rollout_logits_position_missing",
                context={"physical_position": physical_position},
                cause=exc,
            ) from exc
    if row_index < 0 or row_index >= int(logits.shape[0]):
        raise LossContractError(
            "calibration logits row index is outside the forward result",
            code="loss.rollout_logits_position_bounds",
            context={"row_index": row_index, "row_count": int(logits.shape[0])},
        )
    return logits[row_index]


def _coordinate_token_id(coordinate_token_ids: tuple[int, ...], value: int) -> int:
    if value < 0 or value >= len(coordinate_token_ids):
        raise LossContractError(
            "coordinate value is outside the semantic coordinate-token vocabulary",
            code="loss.rollout_coordinate_value_bounds",
            context={"value": value, "coordinate_count": len(coordinate_token_ids)},
        )
    return int(coordinate_token_ids[value])


def _term_result(
    *,
    name: str,
    raw_event: torch.Tensor,
    weight: float,
    denominator: SegmentBalancedDenominator,
    backend_scale: float,
    eligible: bool,
    selected_count: int,
    local_micro_step_index: int,
    diagnostics: dict[str, Any],
    metric_name: str,
    metric_value: Any,
    event_weight: float = 1.0,
    preserve_declared_event_credit: bool = False,
) -> LossTermResult:
    if eligible:
        raw = raw_event.float() * (
            float(backend_scale) / float(denominator.eligible_segment_count)
        )
    else:
        raw = raw_event.float()
    checked_event_weight = float(event_weight)
    if not torch.isfinite(raw.new_tensor(checked_event_weight)).item() or checked_event_weight <= 0.0:
        raise LossContractError(
            "rollout-calibration event weight must be finite and positive",
            code="loss.rollout_event_weight",
            context={"event_weight": checked_event_weight, "term": name},
        )
    denominator_compensation = (
        float(denominator.eligible_segment_count)
        if eligible and preserve_declared_event_credit
        else 1.0
    )
    event_coefficient = checked_event_weight * denominator_compensation
    weighted = raw * float(weight) * event_coefficient
    result_diagnostics = {
        **diagnostics,
        "local_micro_step_index": int(local_micro_step_index),
        "backend_gradient_scale": float(backend_scale),
        "image_balanced_event_weight": checked_event_weight,
        "event_weight_convention": (
            "declared_credit"
            if preserve_declared_event_credit
            else "event_balanced"
        ),
        "event_weight_denominator_compensation": denominator_compensation,
        "effective_event_coefficient": event_coefficient,
    }
    if metric_value is not None and eligible:
        result_diagnostics[metric_name] = float(metric_value) * (
            float(backend_scale) / float(denominator.eligible_segment_count)
        )
    return LossTermResult(
        name=name,
        raw_loss=raw,
        weighted_loss=weighted,
        weight=float(weight),
        segment_mean_numerator=raw.detach(),
        denominator=denominator,
        reducer_name="event_balanced",
        selected_count=int(selected_count),
        skipped_count=0 if eligible else 1,
        math_dtype="float32",
        token_weighted_diagnostic=raw.detach(),
        diagnostics=result_diagnostics,
    )


__all__ = [
    "CalibrationTokenSequence",
    "DUPLICATE_TERM",
    "RolloutCalibrationLossContext",
    "RolloutCalibrationLossPlan",
    "RolloutCalibrationLossRunner",
    "build_calibration_micro_step_stream",
    "rollout_calibration_loss_context",
]
