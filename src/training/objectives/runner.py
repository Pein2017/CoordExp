"""Typed semantic objective runner."""

from __future__ import annotations

from types import MappingProxyType

import torch

from src.training.objectives.box_regression import BoxRegressionObjective
from src.training.objectives.coord_soft_ce import CoordinateSoftCEObjective
from src.training.objectives.token_ce import TokenCEObjective
from src.training.objectives.trie_ce import TrieCEObjective
from src.training.objectives.types import (
    LabelLogitRowMap,
    ObjectiveResult,
    ObjectiveRunResult,
    ObjectiveSpec,
    ResolvedObjectiveSpan,
)
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import (
    BoxRegressionDistribution,
    CoordinateSoftTokenDistribution,
    HardTokenDistribution,
    MultiPositiveTokenDistribution,
    SUPPORTED_OBJECTIVES_BY_DISTRIBUTION_KIND,
    TargetDistribution,
    TargetDistributionRegistry,
)


class ObjectiveRunner:
    """Run typed objectives over semantic supervision spans."""

    def __init__(self) -> None:
        """Initialize the closed objective module registry."""

        self._objectives = {
            "token_ce": TokenCEObjective(),
            "trie_ce": TrieCEObjective(),
            "coord_soft_ce": CoordinateSoftCEObjective(),
            "box_regression": BoxRegressionObjective(),
        }

    def run(
        self,
        *,
        logits: torch.Tensor,
        supervision: SupervisionBatch,
        objectives: tuple[ObjectiveSpec, ...] | list[ObjectiveSpec],
        label_rows: LabelLogitRowMap | None = None,
        sample_id_to_batch_index: dict[str, int] | None = None,
    ) -> ObjectiveRunResult:
        """Run requested objectives and sum objective-local weighted losses."""

        # validate caller-provided execution contracts.
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if type(supervision) is not SupervisionBatch:
            raise TypeError("supervision must be a SupervisionBatch")
        specs = tuple(objectives)
        for spec in specs:
            if type(spec) is not ObjectiveSpec:
                raise TypeError("objectives must be ObjectiveSpec entries")
            if spec.objective_id not in self._objectives:
                raise ValueError(f"unsupported objective: {spec.objective_id!r}")
        objective_ids = [spec.objective_id for spec in specs]
        if len(set(objective_ids)) != len(objective_ids):
            raise ValueError("objective ids must be unique")

        # guard deferred type-gate semantics before any objective math runs.
        self._reject_deferred_metadata(supervision)
        self._validate_distribution_objective_coverage(supervision, objective_ids)
        if len(supervision.spans) == 0:
            return self._run_empty_objectives(
                logits=logits,
                specs=specs,
                label_rows=label_rows,
            )

        # resolve the causal row map once for all objective modules.
        row_map = label_rows or LabelLogitRowMap.from_logits(
            logits,
            sample_id_to_batch_index=sample_id_to_batch_index,
        )
        if type(row_map) is not LabelLogitRowMap:
            raise TypeError("label_rows must be a LabelLogitRowMap")
        row_map.validate_logits(logits)

        # run each requested objective over compatible resolved spans only.
        results: dict[str, ObjectiveResult] = {}
        metric_events = []
        for spec in specs:
            resolved_spans = self._resolve_compatible_spans(
                logits=logits,
                supervision=supervision,
                spec=spec,
                row_map=row_map,
            )
            result = self._objectives[spec.objective_id].run(
                spec=spec,
                spans=resolved_spans,
                logits=logits,
            )
            results[spec.objective_id] = result
            metric_events.extend(result.metric_events)

        return self._make_run_result(
            logits=logits,
            results=results,
            metric_events=metric_events,
            state={"label_row_map": row_map},
        )

    def _run_empty_objectives(
        self,
        *,
        logits: torch.Tensor,
        specs: tuple[ObjectiveSpec, ...],
        label_rows: LabelLogitRowMap | None,
    ) -> ObjectiveRunResult:
        """Run requested objective preflights without resolving unused rows."""

        # validate caller-provided row maps only when the caller supplied one.
        state: dict[str, object] = {}
        if label_rows is not None:
            if type(label_rows) is not LabelLogitRowMap:
                raise TypeError("label_rows must be a LabelLogitRowMap")
            label_rows.validate_logits(logits)
            state["label_row_map"] = label_rows

        # preserve per-objective zero/config-validation behavior for empty batches.
        results: dict[str, ObjectiveResult] = {}
        metric_events = []
        for spec in specs:
            result = self._objectives[spec.objective_id].run(
                spec=spec,
                spans=(),
                logits=logits,
            )
            results[spec.objective_id] = result
            metric_events.extend(result.metric_events)

        return self._make_run_result(
            logits=logits,
            results=results,
            metric_events=metric_events,
            state=state,
        )

    def _make_run_result(
        self,
        *,
        logits: torch.Tensor,
        results: dict[str, ObjectiveResult],
        metric_events: list[object],
        state: dict[str, object],
    ) -> ObjectiveRunResult:
        """Return the aggregated runner result."""

        # sum objective-local weighted losses without a mixed global denominator.
        if results:
            total_loss = torch.stack(
                [result.weighted_loss for result in results.values()]
            ).sum()
        else:
            total_loss = logits.float().sum() * 0.0
        total_loss = total_loss.to(dtype=torch.float32)

        return ObjectiveRunResult(
            loss=total_loss,
            objectives=MappingProxyType(results),
            metric_events=tuple(metric_events),
            state=MappingProxyType(state),
        )

    def _resolve_compatible_spans(
        self,
        *,
        logits: torch.Tensor,
        supervision: SupervisionBatch,
        spec: ObjectiveSpec,
        row_map: LabelLogitRowMap,
    ) -> tuple[ResolvedObjectiveSpan, ...]:
        """Return spans compatible with the requested objective."""

        # validate distribution/objective pairs through the semantic registry.
        resolved: list[ResolvedObjectiveSpan] = []
        for span in supervision.spans:
            try:
                TargetDistributionRegistry.validate_objective_support(
                    span.distribution,
                    spec.objective_id,
                )
            except ValueError:
                continue

            self._validate_resolved_span_arity(span.distribution, span.label_positions)
            rows = row_map.resolve_span(span)
            resolved.append(
                ResolvedObjectiveSpan(
                    span=span,
                    rows=rows,
                    logits=row_map.gather(logits, rows),
                )
            )

        return tuple(resolved)

    def _reject_deferred_metadata(self, supervision: SupervisionBatch) -> None:
        """Reject semantic metadata surfaces not represented in distributions."""

        # fail fast rather than silently ignoring legacy type-gate metadata.
        for span in supervision.spans:
            for key in span.metadata:
                if key == "type_gate" or key.startswith("type_gate_"):
                    raise NotImplementedError(
                        "type_gate metadata is unsupported in the semantic "
                        "distribution contract and is deferred for a future "
                        "typed objective"
                    )

    def _validate_distribution_objective_coverage(
        self,
        supervision: SupervisionBatch,
        objective_ids: list[str],
    ) -> None:
        """Require requested objectives to cover every span distribution kind."""

        # allow empty batches to return graph-anchored objective zeros.
        if len(supervision.spans) == 0:
            return

        # collect missing semantic objectives before any objective module runs.
        requested = frozenset(objective_ids)
        missing: list[str] = []
        seen: set[tuple[str, str]] = set()
        for span in supervision.spans:
            supported = SUPPORTED_OBJECTIVES_BY_DISTRIBUTION_KIND[
                span.distribution.kind
            ]
            for objective_id in sorted(supported - requested):
                key = (span.distribution.kind, objective_id)
                if key in seen:
                    continue
                seen.add(key)
                missing.append(
                    f"{objective_id} for distribution kind {span.distribution.kind}"
                )

        if missing:
            raise ValueError(
                "missing objective coverage for supervision distributions: "
                + "; ".join(missing)
            )

    def _validate_resolved_span_arity(
        self,
        distribution: TargetDistribution,
        label_positions: tuple[int, ...],
    ) -> None:
        """Require objective-compatible row arity before gathering logits."""

        # keep one token target distribution from being copied across many rows.
        if type(distribution) in (
            HardTokenDistribution,
            MultiPositiveTokenDistribution,
            CoordinateSoftTokenDistribution,
        ):
            if len(label_positions) != 1:
                raise ValueError(
                    "token target distributions must resolve exactly one "
                    "label position; use one span per target token"
                )
            return

        # preserve box regression as the explicit grouped four-coordinate case.
        if type(distribution) is BoxRegressionDistribution:
            if len(label_positions) != 4:
                raise ValueError(
                    "box_regression target distributions must resolve exactly "
                    "four label positions"
                )
            return

        raise TypeError(f"unsupported target distribution: {type(distribution)!r}")
