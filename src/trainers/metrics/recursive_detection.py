import math

import torch

from src.detection.loss import (
    RecursiveDetectionLossWeights,
    compute_recursive_detection_ce_batch_loss,
)


class RecursiveDetectionCEMixin:
    """Trainer mixin for teacher-forced recursive detection CE.

    The recursive objective owns the token loss for the batch because trie branch
    positions use sparse multi-positive targets instead of one-hot hard CE.
    """

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        from src.metrics.events import flatten_metric_events
        from src.metrics.reporter import SwiftMetricReporter, best_effort
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        if not isinstance(inputs, dict):
            raise TypeError("recursive_detection_ce enabled but inputs is not a dict")

        from src.detection.dataset import strip_non_model_detection_sidecars
        extras = maybe_pop_and_stash_batch_extras(self, inputs)
        strip_non_model_detection_sidecars(inputs)
        recursive_targets = getattr(extras, "recursive_detection_targets", None)
        if recursive_targets is None:
            raise ValueError(
                "recursive_detection_ce requires recursive_detection_targets sidecars "
                "from the collator"
            )

        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "recursive_detection_ce is enabled, but model outputs do not contain logits."
            )

        cfg = getattr(self, "recursive_detection_ce_cfg", None)
        if cfg is None:
            raise ValueError(
                "recursive_detection_ce requires trainer.recursive_detection_ce_cfg"
            )
        trie_support_weight = getattr(cfg, "trie_support_weight", None)
        trie_balance_weight = getattr(cfg, "trie_balance_weight", None)
        for field_name, raw_value in (
            ("trie_support_weight", trie_support_weight),
            ("trie_balance_weight", trie_balance_weight),
        ):
            if raw_value is None:
                raise ValueError(f"recursive_detection_ce_cfg.{field_name} is required")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"recursive_detection_ce_cfg.{field_name} must be finite and >= 0"
                )
        if float(trie_support_weight) + float(trie_balance_weight) <= 0.0:
            raise ValueError(
                "recursive_detection_ce_cfg trie weights must sum to > 0"
            )
        weights = RecursiveDetectionLossWeights(
            support_weight=float(trie_support_weight),
            balance_weight=float(trie_balance_weight),
        )
        loss_result = compute_recursive_detection_ce_batch_loss(
            logits=logits,
            targets=tuple(recursive_targets),
            weights=weights,
        )
        loss = loss_result.loss
        metric_event_logs = flatten_metric_events(loss_result.metric_events)

        reporter = SwiftMetricReporter(self)

        def _log_recursive_detection_metrics() -> None:
            reporter.update_many(
                {
                    "loss/recursive_detection_ce": float(
                        loss_result.loss.detach().cpu().item()
                    ),
                    "recursive_detection_ce/batch_size": float(
                        loss_result.metrics.get("batch_size", 0.0)
                    ),
                    "recursive_detection_ce/trie_support_weight": float(
                        weights.support_weight
                    ),
                    "recursive_detection_ce/trie_balance_weight": float(
                        weights.balance_weight
                    ),
                }
            )
            reporter.update_many(metric_event_logs)

        best_effort(
            self,
            name="recursive_detection_ce_metrics",
            fn=_log_recursive_detection_metrics,
        )

        compute_acc = getattr(self, "_compute_acc", None)
        if callable(compute_acc):
            labels = inputs.get("labels")
            best_effort(
                self,
                name="recursive_detection_ce_acc",
                fn=lambda: compute_acc(outputs, labels),
            )

        return (loss, outputs) if return_outputs else loss


__all__ = [
    "RecursiveDetectionCEMixin",
]
