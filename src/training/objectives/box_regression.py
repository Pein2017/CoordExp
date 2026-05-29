"""Box-regression objective over resolved coordinate slots."""

from __future__ import annotations

from types import MappingProxyType

import torch

from src.training.objectives.types import (
    CoordinateVocabulary,
    DEFAULT_PRECISION_POLICY,
    ObjectiveResult,
    ObjectiveSpec,
    ResolvedObjectiveSpan,
    config_float,
    loss_float,
    make_sum_event,
    make_weighted_mean_event,
    metadata_float,
    precision_context,
)
from src.training.supervision.distributions import BoxRegressionDistribution
from src.trainers.teacher_forcing.geometry import (
    bbox_smoothl1_ciou_loss,
    canonicalize_bbox_xyxy,
    compute_bbox_regression_loss,
    expectation_decode_coords,
)


class BoxRegressionObjective:
    """BBox geometry objective over four resolved coordinate rows."""

    objective_id = "box_regression"
    _ALLOWED_CONFIG_KEYS = frozenset(
        (
            "coord_token_ids",
            "temperature",
            "smoothl1_weight",
            "ciou_weight",
            "parameterization",
            "center_weight",
            "size_weight",
        )
    )
    _DEFERRED_CONFIG_KEYS = frozenset(
        (
            "log_wh_weight",
            "oversize_penalty_weight",
            "oversize_area_frac_threshold",
            "oversize_log_w_threshold",
            "oversize_log_h_threshold",
            "bbox_size_aux",
            "bbox_size_aux_weight",
            "bbox_size_aux_style",
        )
    )

    def run(
        self,
        *,
        spec: ObjectiveSpec,
        spans: tuple[ResolvedObjectiveSpan, ...],
        logits: torch.Tensor,
    ) -> ObjectiveResult:
        """Compute objective-local normalized bbox regression loss."""

        # validate config before any spans can silently no-op unsupported knobs.
        self._validate_config(spec.config)

        # return a graph-anchored zero for batches with no compatible spans.
        if len(spans) == 0:
            return ObjectiveResult.zero(
                objective_id=spec.objective_id,
                weight=spec.weight,
                logits=logits,
            )

        # parse geometry-local configuration once.
        coord_vocab = CoordinateVocabulary.from_config(spec.config)
        coord_ids = coord_vocab.as_tensor(device=logits.device)
        if int(coord_ids.max().detach().cpu().item()) >= int(logits.shape[-1]):
            raise ValueError("coord_token_ids exceed logits vocab size")
        temperature = config_float(spec.config, "temperature", default=1.0, minimum=0.0)
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        smoothl1_weight = config_float(
            spec.config,
            "smoothl1_weight",
            default=1.0,
            minimum=0.0,
        )
        ciou_weight = config_float(
            spec.config,
            "ciou_weight",
            default=1.0,
            minimum=0.0,
        )
        parameterization = str(spec.config.get("parameterization", "xyxy") or "xyxy")
        center_weight = config_float(
            spec.config,
            "center_weight",
            default=1.0,
            minimum=0.0,
        )
        size_weight = config_float(
            spec.config,
            "size_weight",
            default=1.0,
            minimum=0.0,
        )

        # decode every four-slot span and accumulate group-normalized losses.
        group_losses: list[torch.Tensor] = []
        group_weights: list[torch.Tensor] = []
        decoded_boxes: list[torch.Tensor] = []
        target_boxes: list[torch.Tensor] = []
        smoothl1_terms: list[torch.Tensor] = []
        ciou_terms: list[torch.Tensor] = []
        for resolved in spans:
            distribution = resolved.span.distribution
            if type(distribution) is not BoxRegressionDistribution:
                raise TypeError(
                    "box_regression spans must carry BoxRegressionDistribution"
                )
            if len(resolved.rows) != 4:
                raise ValueError("box_regression spans must carry exactly four slots")

            group_weight = metadata_float(
                resolved.span.metadata,
                "group_weight",
                default=1.0,
            )
            pred_box, target_box, smoothl1, ciou = _decode_box_group(
                resolved=resolved,
                distribution=distribution,
                coord_ids=coord_ids,
                temperature=temperature,
                parameterization=parameterization,
                center_weight=center_weight,
                size_weight=size_weight,
            )
            group_loss = float(smoothl1_weight) * smoothl1 + float(ciou_weight) * ciou

            group_losses.append(group_loss)
            group_weights.append(group_loss.new_tensor(group_weight))
            decoded_boxes.append(pred_box)
            target_boxes.append(target_box)
            smoothl1_terms.append(smoothl1)
            ciou_terms.append(ciou)

        losses = torch.stack(group_losses).to(dtype=torch.float32)
        weights = torch.stack(group_weights).to(dtype=torch.float32)
        numerator = (losses * weights).sum().to(dtype=torch.float32)
        denominator = weights.sum().to(dtype=torch.float32)
        loss = _safe_normalize(numerator, denominator)
        weighted_loss = loss * float(spec.weight)

        decoded = torch.cat(decoded_boxes, dim=0).to(dtype=torch.float32)
        targets = torch.cat(target_boxes, dim=0).to(dtype=torch.float32)
        smoothl1_all = torch.stack(smoothl1_terms).to(dtype=torch.float32)
        ciou_all = torch.stack(ciou_terms).to(dtype=torch.float32)

        # publish canonical metric events and parity state.
        metric_events = (
            make_weighted_mean_event(
                key="training/objectives/box_regression/loss",
                value=loss,
                weight=denominator,
                objective_id=self.objective_id,
            ),
            make_weighted_mean_event(
                key="training/objectives/box_regression/smoothl1",
                value=_safe_normalize((smoothl1_all * weights).sum(), denominator),
                weight=denominator,
                objective_id=self.objective_id,
                diagnostic_only=True,
            ),
            make_weighted_mean_event(
                key="training/objectives/box_regression/ciou",
                value=_safe_normalize((ciou_all * weights).sum(), denominator),
                weight=denominator,
                objective_id=self.objective_id,
                diagnostic_only=True,
            ),
            make_sum_event(
                key="training/objectives/box_regression/span_count",
                value=len(spans),
                objective_id=self.objective_id,
                diagnostic_only=True,
            ),
        )
        state = {
            "decoded_boxes_xyxy": decoded,
            "target_boxes_xyxy": targets,
            "smoothl1": smoothl1_all,
            "ciou": ciou_all,
            "group_weights": weights,
        }

        return ObjectiveResult(
            objective_id=spec.objective_id,
            loss=loss,
            weighted_loss=weighted_loss,
            numerator=numerator,
            denominator=denominator,
            span_count=len(spans),
            weight=spec.weight,
            precision_policy=DEFAULT_PRECISION_POLICY,
            metric_events=metric_events,
            state=MappingProxyType(state),
        )

    def _validate_config(self, config: object) -> None:
        """Reject unsupported box-regression config keys."""

        # keep the skeleton strict so legacy bbox-size aux knobs cannot no-op.
        keys = set(config.keys())
        deferred = {
            key
            for key in keys
            if key in self._DEFERRED_CONFIG_KEYS or key.startswith("bbox_size_aux")
        }
        if deferred:
            raise NotImplementedError(
                "deferred/unsupported box_regression config keys: "
                f"{', '.join(sorted(deferred))}"
            )
        unsupported = keys - self._ALLOWED_CONFIG_KEYS
        if unsupported:
            raise ValueError(
                "deferred/unsupported box_regression config keys: "
                f"{', '.join(sorted(unsupported))}"
            )


def _decode_box_group(
    *,
    resolved: ResolvedObjectiveSpan,
    distribution: BoxRegressionDistribution,
    coord_ids: torch.Tensor,
    temperature: float,
    parameterization: str,
    center_weight: float,
    size_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one four-slot bbox group and compute geometry terms."""

    # restrict full-vocab rows to the typed coordinate vocabulary.
    with precision_context(resolved.logits):
        row_logits = loss_float(resolved.logits)
        coord_logits = row_logits.index_select(dim=-1, index=coord_ids)
        pred_coords = expectation_decode_coords(
            coord_logits=coord_logits,
            temperature=temperature,
            mode="exp",
        )
        pred_box = canonicalize_bbox_xyxy(pred_coords.reshape(1, 4))
        target_box = _target_bbox_tensor(
            distribution,
            device=resolved.logits.device,
        )

        regression = compute_bbox_regression_loss(
            pred_boxes_xyxy=pred_box,
            target_boxes_xyxy=target_box,
            parameterization=parameterization,
            center_weight=center_weight,
            size_weight=size_weight,
        )
        ciou = bbox_smoothl1_ciou_loss(pred_xyxy=pred_box, gt_xyxy=target_box)

    return (
        pred_box.to(dtype=torch.float32),
        target_box.to(dtype=torch.float32),
        regression.per_box.reshape(-1)[0].to(dtype=torch.float32),
        ciou.ciou.to(dtype=torch.float32),
    )


def _target_bbox_tensor(
    distribution: BoxRegressionDistribution,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return a normalized target xyxy tensor."""

    target = torch.tensor(
        [distribution.target_bbox],
        device=device,
        dtype=torch.float32,
    )
    if bool((target > 1.0).any().item()):
        target = target / 999.0

    return canonicalize_bbox_xyxy(target)


def _safe_normalize(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Return numerator divided by denominator, or zero when empty."""

    if float(denominator.detach().cpu().item()) <= 0.0:
        return numerator * 0.0

    return (numerator / denominator.clamp(min=1e-6)).to(dtype=torch.float32)
