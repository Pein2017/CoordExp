from typing import Any, Callable, Dict, List, Mapping, Optional

from src.config.schema import TokenTypeMetricsConfig
from src.data_collators.enrichers import (
    DatasetMetaEnricher,
    InstabilityMetaEnricher,
    TokenTypesEnricher,
)


def build_dataset_metrics_collator(
    template: Any,
    base_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None,
    token_type_cfg: Optional[TokenTypeMetricsConfig] = None,
    instability_monitor_cfg: Optional[Mapping[str, Any]] = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Wrap the template collator to attach debug/diagnostics batch extras.

    Supports padded batches and packed batches (pack emitted as a list of samples).

    NOTE: These extras are consumed by trainer-side mixins and MUST NOT be forwarded
    into model(**inputs).
    """

    collate_fn = base_collator or template.data_collator

    instab_enabled = False
    max_meta_samples = 16
    if isinstance(instability_monitor_cfg, Mapping):
        instab_enabled = bool(instability_monitor_cfg.get("enabled", False))
        try:
            max_meta_samples = int(instability_monitor_cfg.get("max_meta_samples", 16))
        except Exception:
            max_meta_samples = 16
        max_meta_samples = max(0, max_meta_samples)

    meta_enricher = DatasetMetaEnricher()

    token_type_enricher = None
    if token_type_cfg is not None and bool(getattr(token_type_cfg, "enabled", False)):
        token_type_enricher = TokenTypesEnricher(template=template, cfg=token_type_cfg)

    instab_enricher = None
    if instab_enabled:
        instab_enricher = InstabilityMetaEnricher(max_meta_samples=max_meta_samples)

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = collate_fn(batch)

        meta = meta_enricher(batch=batch, collated=collated)

        if instab_enricher is not None:
            try:
                instab_enricher(batch=batch, collated=collated, packed=meta.packed)
            except Exception:
                # Best-effort only; never block training.
                pass

        if token_type_enricher is not None:
            try:
                token_type_enricher(
                    collated=collated,
                    raw_batch=batch,
                    dataset_labels=meta.dataset_labels,
                    packed=meta.packed,
                )
            except Exception:
                # Best-effort only; never block training.
                pass

        return collated

    return _collate
