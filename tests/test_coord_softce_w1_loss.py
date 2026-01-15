from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.config.schema import CoordSoftCEW1Config
from src.metrics.dataset_metrics import CoordSoftCEW1LossMixin


class DummyTokenizer:
    """Minimal tokenizer that maps `<|coord_k|>` -> 100+k and everything else -> 1."""

    def convert_tokens_to_ids(self, tokens):
        out = []
        for tok in tokens:
            if tok.startswith("<|coord_") and tok.endswith("|>"):
                out.append(100 + int(tok[len("<|coord_") : -len("|>")]))
            else:
                out.append(1)
        return out


class DummyTemplate:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


class DummyBaseTrainer:
    def __init__(self):
        self.seen_labels = None
        self.seen_num_items_in_batch = None

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        logits = inputs["fake_logits"]
        labels = inputs["labels"]
        # Record labels after any masking performed by mixins.
        self.seen_labels = labels.detach().clone()
        self.seen_num_items_in_batch = num_items_in_batch

        seq_len = min(logits.shape[1], max(labels.shape[1] - 1, 0))
        logits_next = logits[:, :seq_len, :]
        labels_next = labels[:, 1 : seq_len + 1]
        vocab = int(logits_next.shape[-1])
        loss = F.cross_entropy(
            logits_next.reshape(-1, vocab),
            labels_next.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        outputs = SimpleNamespace(logits=logits)
        return (loss, outputs) if return_outputs else loss


class DummyTrainer(CoordSoftCEW1LossMixin, DummyBaseTrainer):
    def __init__(self, cfg):
        super().__init__()
        self.coord_soft_ce_w1_cfg = cfg
        self.template = DummyTemplate()
        self.model = None
        self.args = SimpleNamespace(average_tokens_across_devices=False)
        self.model_accepts_loss_kwargs = False


def test_stage1_softce_w1_masks_coord_tokens_from_base_ce_and_applies_gate_penalty():
    vocab = 1200
    # Labels: [BOS, text(5), coord(bin=3)->103, text(6), coord(bin=7)->107]
    labels = torch.tensor([[0, 5, 103, 6, 107]], dtype=torch.long)

    # Build logits for 4 prediction positions (seq_len = labels_len-1 = 4).
    logits = torch.full((1, 4, vocab), -20.0)

    # Text positions predicted correctly (base CE should be ~0).
    logits[0, 0, 5] = 20.0  # predicts label at index 1
    logits[0, 2, 6] = 20.0  # predicts label at index 3

    # Coord positions: within coord bins, correct coord id has high logit.
    logits[0, 1, 103] = 20.0
    logits[0, 3, 107] = 20.0

    cfg = CoordSoftCEW1Config.from_mapping(
        {
            "enabled": True,
            "soft_ce_weight": 1.0,
            "w1_weight": 1.0,
            "gate_weight": 1.0,
            "temperature": 1.0,
            "target_sigma": 1.0,
            "target_truncate": 0,  # one-hot target
        }
    )
    trainer = DummyTrainer(cfg)

    loss_no_leak = trainer.compute_loss(
        model=None,
        inputs={"labels": labels, "fake_logits": logits},
        return_outputs=False,
        num_items_in_batch=1,
    )

    # Coord labels are masked to ignore_index for the base CE path.
    assert trainer.seen_labels is not None
    assert int(trainer.seen_labels[0, 2].item()) == -100
    assert int(trainer.seen_labels[0, 4].item()) == -100
    # Non-coord labels are kept.
    assert int(trainer.seen_labels[0, 1].item()) == 5
    assert int(trainer.seen_labels[0, 3].item()) == 6

    # With coord targets masked, only 2 next-token labels are supervised here.
    # The mixin should override upstream `num_items_in_batch` (often batch size)
    # to ensure mean-normalized loss under packing.
    assert trainer.seen_num_items_in_batch == 2

    assert float(loss_no_leak.detach().item()) < 1e-3

    # If a non-coord token dominates the full-vocab logits at coord positions, the
    # coord-vocab gate MUST penalize probability mass leaking out of the coord vocab.
    logits_leak = logits.clone()
    noncoord_id = 42
    logits_leak[0, 1, noncoord_id] = 80.0
    logits_leak[0, 3, noncoord_id] = 80.0
    loss_leak = trainer.compute_loss(
        model=None,
        inputs={"labels": labels, "fake_logits": logits_leak},
        return_outputs=False,
        num_items_in_batch=1,
    )
    assert float(loss_leak.detach().item()) > float(loss_no_leak.detach().item()) + 1.0
