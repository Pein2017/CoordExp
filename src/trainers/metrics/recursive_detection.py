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
        if "logits_to_keep" in inputs:
            raise ValueError(
                "recursive_detection_ce requires full sequence logits; "
                "logits_to_keep is unsupported because recursive sidecar target "
                "positions are absolute encoded-token positions"
            )
        recursive_targets = getattr(extras, "recursive_detection_targets", None)
        if recursive_targets is None:
            raise ValueError(
                "recursive_detection_ce requires recursive_detection_targets sidecars "
                "from the collator"
            )

        labels_for_acc = inputs.get("labels")
        _validate_recursive_detection_sidecar_alignment(
            inputs=inputs,
            recursive_targets=tuple(recursive_targets),
        )
        model_inputs = dict(inputs)
        # Recursive CE owns token supervision; do not let the base trainer/model
        # compute an additional shifted one-hot CE from labels.
        model_inputs.pop("labels", None)

        outputs = model(**model_inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "recursive_detection_ce is enabled, but model outputs do not contain logits."
            )
        input_ids = model_inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and logits.shape[:2] != input_ids.shape[:2]:
            raise RuntimeError(
                "recursive_detection_ce requires full unsliced logits with the same "
                f"[batch, time] shape as input_ids; got logits={tuple(logits.shape[:2])} "
                f"input_ids={tuple(input_ids.shape[:2])}"
            )

        cfg = getattr(self, "recursive_detection_ce_cfg", None)
        if cfg is None:
            raise ValueError(
                "recursive_detection_ce requires trainer.recursive_detection_ce_cfg"
            )
        trie_support_weight = getattr(cfg, "trie_support_weight", None)
        trie_balance_weight = getattr(cfg, "trie_balance_weight", None)
        separator_continue_weight = getattr(cfg, "separator_continue_weight", 0.50)
        eos_stop_weight = getattr(cfg, "eos_stop_weight", 0.50)
        boundary_component_weight = getattr(cfg, "boundary_component_weight", 0.30)
        for field_name, raw_value in (
            ("trie_support_weight", trie_support_weight),
            ("trie_balance_weight", trie_balance_weight),
            ("separator_continue_weight", separator_continue_weight),
            ("eos_stop_weight", eos_stop_weight),
            ("boundary_component_weight", boundary_component_weight),
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
            separator_continue_weight=float(separator_continue_weight),
            eos_stop_weight=float(eos_stop_weight),
            boundary_component_weight=float(boundary_component_weight),
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
                    "recursive_detection_ce/boundary/separator_continue_weight": float(
                        weights.separator_continue_weight
                    ),
                    "recursive_detection_ce/boundary/eos_stop_weight": float(
                        weights.eos_stop_weight
                    ),
                    "recursive_detection_ce/boundary/component_weight": float(
                        weights.boundary_component_weight
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
            best_effort(
                self,
                name="recursive_detection_ce_acc",
                fn=lambda: compute_acc(outputs, labels_for_acc),
            )

        return (loss, outputs) if return_outputs else loss


def _validate_recursive_detection_sidecar_alignment(
    *,
    inputs: dict[str, object],
    recursive_targets: tuple[object, ...],
) -> None:
    input_ids = inputs.get("input_ids")
    labels = inputs.get("labels")
    attention_mask = inputs.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("recursive_detection_ce requires 2D tensor input_ids")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("recursive_detection_ce requires 2D tensor labels")
    if input_ids.shape != labels.shape:
        raise ValueError(
            "recursive_detection_ce requires input_ids and labels to have the same shape; "
            f"got input_ids={tuple(input_ids.shape)} labels={tuple(labels.shape)}"
        )
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise ValueError("recursive_detection_ce requires 2D tensor attention_mask")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "recursive_detection_ce requires attention_mask to match input_ids shape; "
            f"got attention_mask={tuple(attention_mask.shape)} "
            f"input_ids={tuple(input_ids.shape)}"
        )
    batch_size, seq_len = input_ids.shape
    if len(recursive_targets) != batch_size:
        raise ValueError(
            "recursive_detection_ce recursive_detection_targets length must match batch "
            f"size; got targets={len(recursive_targets)} batch={batch_size}"
        )

    for batch_index, sample_targets in enumerate(recursive_targets):
        for target in getattr(sample_targets, "token_targets", ()):
            position = int(getattr(target, "position"))
            teacher_token_id = int(getattr(target, "teacher_token_id"))
            if position <= 0 or position >= seq_len:
                raise ValueError(
                    "recursive_detection_ce TokenTarget.position must satisfy "
                    f"0 < position < seq_len; got position={position} seq_len={seq_len} "
                    f"batch_index={batch_index}"
                )
            input_token = int(input_ids[batch_index, position].detach().cpu().item())
            if input_token != teacher_token_id:
                raise ValueError(
                    "recursive_detection_ce input_ids at TokenTarget.position must "
                    "match teacher_token_id; "
                    f"batch_index={batch_index} position={position} "
                    f"input_ids={input_token} teacher_token_id={teacher_token_id}"
                )
            label_token = int(labels[batch_index, position].detach().cpu().item())
            if label_token != teacher_token_id:
                raise ValueError(
                    "recursive_detection_ce labels at TokenTarget.position must "
                    "match teacher_token_id; "
                    f"batch_index={batch_index} position={position} "
                    f"labels={label_token} teacher_token_id={teacher_token_id}"
                )
            current_mask = int(attention_mask[batch_index, position].detach().cpu().item())
            previous_mask = int(attention_mask[batch_index, position - 1].detach().cpu().item())
            if current_mask <= 0 or previous_mask <= 0:
                raise ValueError(
                    "recursive_detection_ce TokenTarget.position and previous logits "
                    "context must be inside attention_mask; "
                    f"batch_index={batch_index} position={position} "
                    f"attention_mask[position]={current_mask} "
                    f"attention_mask[position-1]={previous_mask}"
                )


__all__ = [
    "RecursiveDetectionCEMixin",
]
