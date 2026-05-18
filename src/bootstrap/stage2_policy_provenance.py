from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


STAGE2_POLICY_PROVENANCE_SCHEMA_VERSION = 1
STAGE2_TWO_CHANNEL_TRAINER_VARIANT = "stage2_two_channel"
LEGACY_CHANNEL_B_DUPLICATE_FILTER_POLICY_ID = "legacy_channel_b_duplicate_control"


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None

    out = float(value)
    if not math.isfinite(out):
        raise ValueError("Stage-2 policy provenance contains non-finite float")
    return out


def _string_field(
    payload: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    raw = payload.get(key, default)
    text = str(raw).strip()
    return text if text else default


def _object_ordering_strategy_id(insertion_order: str) -> str:
    normalized = str(insertion_order).strip().lower()
    if normalized == "sorted":
        return "top_left_spatial"
    return "legacy_tail_append"


def build_stage2_policy_provenance(
    training_config: Any,
    *,
    trainer_variant: str | None = None,
) -> dict[str, Any] | None:
    """Return first-class Stage-2 policy provenance for rank-0 manifests.

    :param training_config: Resolved typed training config or equivalent mapping.
    :param trainer_variant: Resolved trainer variant. When omitted, the helper
        falls back to ``custom.trainer_variant`` from ``training_config``.
    :returns: A JSON-serializable policy block for ``stage2_two_channel`` runs,
        or ``None`` when the active run is not a Stage-2 two-channel surface.
    """

    # normalize the resolved config into the policy-owning sections
    root = _as_mapping(training_config)
    custom = _as_mapping(root.get("custom"))
    variant = str(trainer_variant or custom.get("trainer_variant") or "").strip()
    if variant != STAGE2_TWO_CHANNEL_TRAINER_VARIANT:
        return None

    stage2_ab = _as_mapping(root.get("stage2_ab"))
    if not stage2_ab:
        return None

    channel_b = _as_mapping(stage2_ab.get("channel_b"))
    assignment = _as_mapping(channel_b.get("assignment"))
    duplicate_control = _as_mapping(channel_b.get("duplicate_control"))
    rollout_matching = _as_mapping(root.get("rollout_matching"))

    # resolve assignment strategy and effective IoU threshold
    assignment_strategy = _string_field(
        assignment,
        "strategy",
        "greedy_iou",
    )
    configured_assignment_iou = _optional_finite_float(
        assignment.get("iou_threshold")
    )
    rollout_maskiou_gate = _optional_finite_float(
        rollout_matching.get("maskiou_gate", 0.3)
    )
    if configured_assignment_iou is not None:
        effective_assignment_iou = configured_assignment_iou
        assignment_iou_threshold_source = (
            "stage2_ab.channel_b.assignment.iou_threshold"
        )
    else:
        effective_assignment_iou = rollout_maskiou_gate
        assignment_iou_threshold_source = "rollout_matching.maskiou_gate"

    # resolve duplicate-control and ordering policies
    duplicate_iou_threshold = _optional_finite_float(
        duplicate_control.get("iou_threshold", 0.90)
    )
    duplicate_center_radius_scale = _optional_finite_float(
        duplicate_control.get("center_radius_scale", 0.80)
    )
    insertion_order = _string_field(channel_b, "insertion_order", "tail_append")
    fallback_loss_weight = _optional_finite_float(
        channel_b.get("fallback_loss_weight", 1.0)
    )

    return {
        "schema_version": STAGE2_POLICY_PROVENANCE_SCHEMA_VERSION,
        "trainer_variant": variant,
        "assignment_strategy": assignment_strategy,
        "assignment_iou_threshold": configured_assignment_iou,
        "assignment_iou_threshold_effective": effective_assignment_iou,
        "assignment_iou_threshold_source": assignment_iou_threshold_source,
        "duplicate_filter_strategy": LEGACY_CHANNEL_B_DUPLICATE_FILTER_POLICY_ID,
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "duplicate_center_radius_scale": duplicate_center_radius_scale,
        "object_ordering_policy": insertion_order,
        "object_ordering_strategy_id": _object_ordering_strategy_id(insertion_order),
        "sample_object_ordering": _string_field(custom, "object_ordering", "sorted"),
        "rollout_template_family": _string_field(
            channel_b,
            "rollout_template_family",
            "coordjson",
        ),
        "rollout_decode_policy": _string_field(
            channel_b,
            "rollout_decode_policy",
            "legacy_coordjson",
        ),
        "invalid_rollout_policy": _string_field(
            channel_b,
            "invalid_rollout_policy",
            "abort",
        ),
        "fallback_loss_weight": fallback_loss_weight,
    }
