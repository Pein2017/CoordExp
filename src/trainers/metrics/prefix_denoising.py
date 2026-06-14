from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch

from src.detection.dataset import strip_non_model_detection_sidecars
from src.detection.prefix_denoising.loss import (
    PrefixDenoisingSegmentSpan,
    compute_branch_balanced_hard_ce,
    compute_local_coord_kl,
    topk_accuracy_from_logits,
)
from src.detection.prefix_denoising.metrics import (
    prefix_denoising_ce_events,
    prefix_denoising_kl_events,
)
from src.detection.prefix_denoising.types import (
    HybridPrefixDenoisingSample,
    PrefixDenoisingKLSite,
    ResolvedPrefixDenoisingKLSite,
)
from src.metrics.events import flatten_metric_events
from src.metrics.reporter import SwiftMetricReporter
from src.tokens.coord.codec import get_coord_token_ids
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
        kl_weight = _resolve_prefix_denoising_kl_weight(self)

        packing_enabled = _resolve_prefix_denoising_packing_enabled(self)
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
        total_loss = ce.loss
        kl_events = ()
        if kl_weight > 0.0:
            resolved_sites = _resolve_kl_sites_from_extras(
                extras=extras,
                packing_enabled=packing_enabled,
            )
            if not resolved_sites:
                raise ValueError(
                    "positive prefix_denoising KL requires at least one "
                    "resolved KL site"
                )
            coord_token_ids = _coord_token_id_tensor(
                trainer=self,
                logits=logits,
            )
            kl = compute_local_coord_kl(
                clean_logits=logits,
                noisy_logits=logits,
                sites=resolved_sites,
                coord_token_ids=coord_token_ids,
            )
            weighted_kl = kl.raw_loss * float(kl_weight)
            total_loss = ce.loss + weighted_kl
            kl_events = prefix_denoising_kl_events(
                kl=kl,
                raw_loss=_scalar(kl.raw_loss),
                weighted_loss=_scalar(weighted_kl),
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
            llm_loss=_scalar(total_loss),
        )
        SwiftMetricReporter(self).update_many(
            {
                str(key): float(value)
                for key, value in flatten_metric_events(
                    tuple(metric_events) + tuple(kl_events)
                ).items()
                if isinstance(value, (int, float))
            }
        )
        return (total_loss, outputs) if return_outputs else total_loss


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


def _resolve_prefix_denoising_kl_weight(trainer: Any) -> float:
    raw_value = getattr(trainer, "prefix_denoising_kl_weight", 0.0)
    try:
        value = float(raw_value or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "prefix_denoising current_object_kl.weight must be numeric"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "prefix_denoising current_object_kl.weight must be finite and >= 0"
        )
    return value


def _coord_token_id_tensor(
    *,
    trainer: Any,
    logits: torch.Tensor,
) -> torch.Tensor:
    tokenizer = _resolve_prefix_denoising_tokenizer(trainer)
    coord_ids = get_coord_token_ids(tokenizer, validate=True)
    values = [int(value) for value in coord_ids]
    if len(values) != 1000:
        raise ValueError(
            f"coord token id lookup must return exactly 1000 ids; got {len(values)}"
        )
    return torch.tensor(values, device=logits.device, dtype=torch.long)


def _resolve_prefix_denoising_tokenizer(trainer: Any) -> Any:
    direct = getattr(trainer, "tokenizer", None)
    if direct is not None:
        return direct
    processing_class = getattr(trainer, "processing_class", None)
    tokenizer = getattr(processing_class, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    template = getattr(trainer, "template", None)
    tokenizer = getattr(template, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    raise RuntimeError(
        "prefix_denoising KL weight > 0 requires a tokenizer on trainer.tokenizer, "
        "trainer.processing_class.tokenizer, or trainer.template.tokenizer"
    )


def _resolve_kl_sites_from_extras(
    *,
    extras: Any,
    packing_enabled: bool,
) -> tuple[ResolvedPrefixDenoisingKLSite, ...]:
    resolved_payload = getattr(extras, "prefix_denoising_resolved_kl_sites", None)
    if resolved_payload is not None:
        return _flatten_resolved_kl_sites(resolved_payload)
    if packing_enabled:
        raise ValueError(
            "packed prefix_denoising KL requires prefix_denoising_resolved_kl_sites"
        )

    hybrids = _unpacked_hybrids(getattr(extras, "prefix_denoising_hybrid", None))
    if not hybrids:
        return ()
    meta_by_segment_id = _segment_meta_by_segment_id(
        getattr(extras, "prefix_denoising_segment_meta", None),
        packing_enabled=packing_enabled,
    )
    resolved: list[ResolvedPrefixDenoisingKLSite] = []
    for hybrid in hybrids:
        for site in hybrid.kl_sites:
            if type(site) is not PrefixDenoisingKLSite:
                raise TypeError(
                    "prefix_denoising_hybrid kl_sites must contain "
                    "PrefixDenoisingKLSite entries"
                )
            try:
                clean_meta = meta_by_segment_id[site.clean_segment_id]
                noisy_meta = meta_by_segment_id[site.noisy_segment_id]
            except KeyError as exc:
                raise ValueError(
                    "prefix_denoising_segment_meta is missing a segment referenced "
                    "by a KL site"
                ) from exc
            resolved.append(
                ResolvedPrefixDenoisingKLSite(
                    clean_batch_index=int(clean_meta["batch_index"]),
                    noisy_batch_index=int(noisy_meta["batch_index"]),
                    clean_label_position=int(clean_meta["token_start"])
                    + int(site.clean_label_position),
                    noisy_label_position=int(noisy_meta["token_start"])
                    + int(site.noisy_label_position),
                    clean_gt_bin=int(site.clean_gt_bin),
                    support_bins=tuple(int(value) for value in site.support_bins),
                    coord_slot=site.coord_slot,
                    object_index=int(site.object_index),
                    identical_prefix=bool(site.identical_prefix),
                )
            )
    return tuple(resolved)


def _flatten_resolved_kl_sites(payload: Any) -> tuple[ResolvedPrefixDenoisingKLSite, ...]:
    resolved: list[ResolvedPrefixDenoisingKLSite] = []

    def visit(value: Any) -> None:
        if type(value) is ResolvedPrefixDenoisingKLSite:
            resolved.append(value)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)
            return
        raise TypeError(
            "prefix_denoising_resolved_kl_sites entries must be "
            "ResolvedPrefixDenoisingKLSite"
        )

    visit(payload)
    return tuple(resolved)


def _unpacked_hybrids(payload: Any) -> tuple[HybridPrefixDenoisingSample, ...]:
    if payload is None:
        return ()
    if type(payload) is HybridPrefixDenoisingSample:
        return (payload,)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        hybrids = tuple(payload)
        for hybrid in hybrids:
            if type(hybrid) is not HybridPrefixDenoisingSample:
                raise TypeError(
                    "prefix_denoising_hybrid unpacked sidecar must contain "
                    "HybridPrefixDenoisingSample entries"
                )
        return hybrids
    raise TypeError(
        "prefix_denoising_hybrid unpacked sidecar must be a "
        "HybridPrefixDenoisingSample sequence"
    )


def _segment_meta_by_segment_id(
    payload: Any,
    *,
    packing_enabled: bool,
) -> dict[str, dict[str, int]]:
    records = list(_iter_segment_meta_records(payload))
    out: dict[str, dict[str, int]] = {}
    for default_batch_index, record in records:
        segment_id = str(_require_record_value(record, "segment_id"))
        out[segment_id] = {
            "batch_index": int(record.get("batch_index", default_batch_index)),
            "token_start": _segment_token_start(
                record,
                packing_enabled=packing_enabled,
            ),
        }
    return out


def _segment_token_start(
    record: Mapping[str, Any],
    *,
    packing_enabled: bool,
) -> int:
    if "token_start" in record:
        return int(record["token_start"])
    if not packing_enabled and "local_token_start" in record:
        return int(record["local_token_start"])
    raise ValueError(
        "prefix_denoising_segment_meta requires token_start or unpacked "
        "local_token_start for KL site resolution"
    )


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
