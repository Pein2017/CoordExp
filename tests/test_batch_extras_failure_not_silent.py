import types

import torch

from src.trainers.metrics.mixins import AggregateTokenTypeMetricsMixin


class _BaseTrainer:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = torch.tensor(0.0)
        outputs = types.SimpleNamespace(logits=torch.zeros((1, 1, 8), dtype=torch.float32))
        return (loss, outputs) if return_outputs else loss


class _Trainer(AggregateTokenTypeMetricsMixin, _BaseTrainer):
    def _log_aggregate_metrics(self, outputs, labels, token_types):
        return None

    def _sync_dataset_metrics(self):
        return None


def test_batch_extras_failure_is_warned_once(monkeypatch) -> None:
    t = _Trainer()

    def _boom(_trainer, _inputs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.trainers.batch_extras.maybe_pop_and_stash_batch_extras",
        _boom,
    )

    # Should not raise (best-effort diagnostics), but should record a warn-once marker.
    _ = t.compute_loss(model=None, inputs={"labels": torch.tensor([[-100]])})

    warned = getattr(t, "_coordexp_warned_once", set())
    assert "batch_extras_failed" in warned
