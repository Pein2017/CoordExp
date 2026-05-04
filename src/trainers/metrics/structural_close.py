import torch
import torch.nn.functional as F


class SFTStructuralCloseLossMixin:
    """Weighted base CE for ordinary Stage-1 global CoordJSON close tokens."""

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        extras = maybe_pop_and_stash_batch_extras(self, inputs)
        cfg = getattr(self, "sft_structural_close_cfg", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return super().compute_loss(  # type: ignore[misc]
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if not isinstance(inputs, dict):
            raise TypeError("sft_structural_close enabled but inputs is not a dict")
        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "sft_structural_close enabled but inputs['labels'] is missing or not a torch.Tensor"
            )
        token_weights = getattr(extras, "sft_structural_close_token_weights", None)
        if token_weights is None or not isinstance(token_weights, torch.Tensor):
            raise ValueError(
                "custom.sft_structural_close requires sft_structural_close_token_weights from the collator"
            )
        if tuple(token_weights.shape) != tuple(labels.shape):
            raise ValueError(
                "sft_structural_close_token_weights must match labels shape"
            )

        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "sft_structural_close is enabled, but model outputs do not contain logits."
            )

        seq_len = min(int(logits.shape[1]), max(int(labels.shape[1]) - 1, 0))
        if seq_len <= 0:
            loss = logits.new_zeros(())
            return (loss, outputs) if return_outputs else loss

        logits_next = logits[:, :seq_len, :]
        labels_next = labels[:, 1 : seq_len + 1]
        weights_next = token_weights[:, 1 : seq_len + 1].to(
            device=logits.device,
            dtype=torch.float32,
        )
        labels_next = labels_next.to(device=logits.device)
        supervised = labels_next != -100
        weights_next = torch.where(
            supervised,
            weights_next,
            torch.zeros_like(weights_next),
        )

        flat_loss = F.cross_entropy(
            logits_next.float().reshape(-1, int(logits_next.shape[-1])),
            labels_next.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        flat_weights = weights_next.reshape(-1).to(dtype=flat_loss.dtype)
        denom = flat_weights.sum().clamp(min=1e-6)
        loss = torch.nan_to_num(
            (flat_loss * flat_weights).sum() / denom,
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )

        from src.metrics.reporter import SwiftMetricReporter, best_effort

        reporter = SwiftMetricReporter(self)

        def _log_structural_close_metrics() -> None:
            close_mask = supervised & weights_next.ne(1.0)
            close_count = int(close_mask.sum().detach().cpu().item())
            reporter.update(
                "sft_structural_close/final_close_weight",
                float(getattr(cfg, "final_close_weight", 1.0)),
            )
            reporter.update(
                "sft_structural_close/final_close_tokens",
                float(close_count),
            )
            reporter.update(
                "sft_structural_close/weighted_token_sum",
                float(flat_weights.sum().detach().cpu().item()),
            )
            reporter.update("loss/sft_structural_close_base_ce", float(loss.detach().cpu().item()))
            if close_count > 0:
                close_losses = flat_loss.reshape_as(labels_next)[close_mask]
                reporter.update("loss/eod", float(close_losses.mean().detach().cpu().item()))
            else:
                reporter.update("loss/eod", 0.0)

        best_effort(self, name="sft_structural_close_metrics", fn=_log_structural_close_metrics)

        compute_acc = getattr(self, "_compute_acc", None)
        if callable(compute_acc):
            best_effort(
                self,
                name="sft_structural_close_acc",
                fn=lambda: compute_acc(outputs, labels),
            )

        return (loss, outputs) if return_outputs else loss


__all__ = [
    "SFTStructuralCloseLossMixin",
]
