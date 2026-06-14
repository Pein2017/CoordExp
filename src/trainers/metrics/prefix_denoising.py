from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch

from src.detection.dataset import strip_non_model_detection_sidecars
from src.detection.prefix_denoising.loss import (
    PrefixDenoisingSegmentSpan,
    compute_branch_balanced_hard_ce,
    topk_accuracy_from_logits,
)
from src.detection.prefix_denoising.metrics import prefix_denoising_ce_events
from src.metrics.events import flatten_metric_events
from src.metrics.reporter import SwiftMetricReporter
from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras
from src.trainers.teacher_forcing.forwards import (
    assert_unsliced_logits,
    prepare_forward_inputs,
    run_no_cache_forward,
)


class PrefixDenoisingObjectiveMixin:
    """Trainer mixin for V1 prefix-denoising hard-CE supervision."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        if not isinstance(inputs, MutableMapping):
            raise TypeError("prefix_denoising objective requires dict-like inputs")

        extras = maybe_pop_and_stash_batch_extras(self, inputs)
        _require_prefix_hybrids(extras.prefix_denoising_hybrid)
        labels = inputs.get("labels")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("prefix_denoising objective requires a labels tensor")
        if "logits_to_keep" in inputs:
            raise ValueError(
                "prefix_denoising objective requires full sequence logits; "
                "logits_to_keep is unsupported"
            )

        packing_enabled = _resolve_prefix_denoising_packing_enabled(self)
        if packing_enabled and extras.packed_hybrid_boundary_map is None:
            raise ValueError(
                "prefix_denoising packed training requires "
                "packed_hybrid_boundary_map sidecar"
            )
        segment_spans = _segment_spans_from_meta(
            extras.prefix_denoising_segment_meta,
            packing_enabled=packing_enabled,
        )

        strip_non_model_detection_sidecars(inputs)
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("prefix_denoising objective requires input_ids tensor")

        _, inputs_for_model, _ = prepare_forward_inputs(
            model=model,
            inputs=inputs,
            ignored_keys=("labels",),
            packing_enabled=packing_enabled,
            where="prefix_denoising",
        )
        outputs = run_no_cache_forward(model=model, inputs_for_model=inputs_for_model)
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "prefix_denoising objective requires model outputs with logits"
            )
        assert_unsliced_logits(
            logits=logits,
            input_ids=input_ids,
            where="prefix_denoising",
        )

        ce = compute_branch_balanced_hard_ce(
            logits=logits,
            labels=labels,
            segment_spans=segment_spans,
        )
        token_acc = topk_accuracy_from_logits(
            logits=logits,
            labels=labels,
            segment_spans=segment_spans,
            topk=(1, 5),
        )
        metric_events = prefix_denoising_ce_events(
            ce_balanced=_scalar(ce.loss),
            ce_clean=_scalar(ce.clean_ce),
            ce_noisy=_scalar(ce.noisy_ce),
            ce_token_pooled=_scalar(ce.token_pooled_ce),
            token_top1=float(token_acc[1]),
            token_top5=float(token_acc[5]),
            clean_denominator=ce.clean_denominator,
            noisy_denominator=ce.noisy_denominator,
        )
        SwiftMetricReporter(self).update_many(
            {
                str(key): float(value)
                for key, value in flatten_metric_events(metric_events).items()
                if isinstance(value, (int, float))
            }
        )
        return (ce.loss, outputs) if return_outputs else ce.loss


def _require_prefix_hybrids(payload: Any) -> None:
    if payload is None:
        raise ValueError("prefix_denoising objective requires prefix_denoising_hybrid sidecar")
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if len(payload) == 0:
            raise ValueError("prefix_denoising_hybrid sidecar batch is empty")


def _resolve_prefix_denoising_packing_enabled(trainer: Any) -> bool:
    value = getattr(trainer, "prefix_denoising_packing_enabled", None)
    if value is None:
        raise ValueError(
            "prefix_denoising trainer requires explicit "
            "prefix_denoising_packing_enabled runtime state"
        )
    return bool(value)


def _segment_spans_from_meta(
    payload: Any,
    *,
    packing_enabled: bool,
) -> tuple[PrefixDenoisingSegmentSpan, ...]:
    if payload is None:
        raise ValueError(
            "prefix_denoising objective requires prefix_denoising_segment_meta sidecar"
        )

    records = list(_iter_segment_meta_records(payload))
    if not records:
        raise ValueError("prefix_denoising_segment_meta sidecar batch is empty")

    spans = tuple(
        _segment_span_from_record(
            record,
            default_batch_index=batch_index,
            packing_enabled=packing_enabled,
        )
        for batch_index, record in records
    )
    branch_ids = {span.branch_id for span in spans}
    missing = {"clean_full", "noisy_full"} - branch_ids
    if missing:
        raise ValueError(
            "prefix_denoising_segment_meta must include clean_full and noisy_full "
            f"segments; missing={sorted(missing)}"
        )
    return spans


def _iter_segment_meta_records(payload: Any) -> list[tuple[int, Mapping[str, Any]]]:
    out: list[tuple[int, Mapping[str, Any]]] = []

    def visit(value: Any, *, batch_index: int) -> None:
        if isinstance(value, Mapping):
            out.append((batch_index, value))
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item, batch_index=batch_index)
            return
        raise TypeError(
            "prefix_denoising_segment_meta entries must be mappings or sequences "
            f"of mappings, got {type(value).__name__}"
        )

    if isinstance(payload, Mapping):
        visit(payload, batch_index=0)
        return out
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if all(isinstance(item, Mapping) for item in payload):
            for item in payload:
                visit(item, batch_index=0)
            return out
        for batch_index, item in enumerate(payload):
            visit(item, batch_index=batch_index)
        return out
    raise TypeError(
        "prefix_denoising_segment_meta must be a mapping or sequence of mappings"
    )


def _segment_span_from_record(
    record: Mapping[str, Any],
    *,
    default_batch_index: int,
    packing_enabled: bool,
) -> PrefixDenoisingSegmentSpan:
    branch_id = str(_require_record_value(record, "branch_id"))
    if branch_id not in {"clean_full", "noisy_full"}:
        raise ValueError(f"unsupported prefix denoising branch_id: {branch_id!r}")

    batch_index = int(record.get("batch_index", default_batch_index))
    if "token_start" in record and "token_end" in record:
        token_start = int(record["token_start"])
        token_end = int(record["token_end"])
    elif not packing_enabled and "local_token_start" in record and "local_token_end" in record:
        token_start = int(record["local_token_start"])
        token_end = int(record["local_token_end"])
    else:
        raise ValueError(
            "prefix_denoising_segment_meta requires token_start/token_end "
            "physical spans"
        )

    return PrefixDenoisingSegmentSpan(
        batch_index=batch_index,
        token_start=token_start,
        token_end=token_end,
        branch_id=branch_id,  # type: ignore[arg-type]
        segment_id=str(record.get("segment_id", f"{batch_index}:{branch_id}")),
    )


def _require_record_value(record: Mapping[str, Any], key: str) -> Any:
    if key not in record:
        raise ValueError(f"prefix_denoising_segment_meta missing required {key!r}")
    return record[key]


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


__all__ = ["PrefixDenoisingObjectiveMixin"]
