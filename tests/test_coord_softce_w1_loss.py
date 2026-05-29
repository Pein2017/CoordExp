from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.config.schema import CoordSoftCEW1Config
from src.metrics.dataset_metrics import CoordSoftCEW1LossMixin
from src.trainers.losses.coord_soft_ce_w1 import compute_coord_soft_ce_w1_loss


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

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        mapping = {
            0: "",
            5: '{"objects":[{"desc":"x","bbox_2d":[',
            6: ",",
            103: "<|coord_3|>,",
            104: "<|coord_4|>,",
            107: "<|coord_7|>,",
            108: "<|coord_8|>]}]}",
        }
        return "".join(mapping.get(int(t), f"<tok_{int(t)}>") for t in token_ids)


class DummyTemplate:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


class PieceTokenizer:
    def __init__(self, pieces_by_id: dict[int, str]):
        self._pieces_by_id = {int(k): str(v) for k, v in pieces_by_id.items()}

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        return "".join(
            self._pieces_by_id.get(
                int(t),
                (
                    f"<|coord_{int(t) - 100}|>"
                    if int(t) >= 100
                    else f"<tok_{int(t)}>"
                ),
            )
            for t in token_ids
        )


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


def _build_coord_id_map(vocab: int, coord_token_ids: list[int]) -> torch.Tensor:
    coord_id_map = torch.full((vocab,), -1, dtype=torch.long)
    for coord_bin, token_id in enumerate(coord_token_ids):
        coord_id_map[int(token_id)] = int(coord_bin)
    return coord_id_map


def _perfect_next_token_logits(labels: torch.Tensor, *, vocab: int) -> torch.Tensor:
    seq_len = max(int(labels.shape[1]) - 1, 0)
    logits = torch.full((int(labels.shape[0]), seq_len, vocab), -20.0)
    for batch_idx in range(int(labels.shape[0])):
        for pos in range(seq_len):
            target_id = int(labels[batch_idx, pos + 1].item())
            if target_id >= 0:
                logits[batch_idx, pos, target_id] = 20.0
    return logits


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


def test_stage1_softce_w1_manual_grad_accum_scaling_matches_transformers_path():
    """Regression: when num_items_in_batch is provided, transformers>=4.57 does not
    apply its usual gradient-accumulation loss scaling.

    CoordSoftCEW1LossMixin overrides num_items_in_batch (to be packing-safe), so
    it must manually divide by grad-accum steps to keep loss/gradients on the
    same scale as the num_items_in_batch=None path.
    """

    class _ScalingBaseTrainer:
        def compute_loss(
            self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
        ):
            logits = inputs["fake_logits"]
            labels = inputs["labels"]

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

            # Simulate transformers>=4.57: grad-accum scaling happens only when
            # num_items_in_batch is None.
            if (
                bool(getattr(getattr(self, "model", None), "training", False))
                and bool(getattr(self, "model_accepts_loss_kwargs", False))
                and num_items_in_batch is None
            ):
                gas = int(
                    getattr(getattr(self, "args", None), "gradient_accumulation_steps", 1)
                    or 1
                )
                if gas > 1:
                    loss = loss / float(gas)

            outputs = SimpleNamespace(logits=logits)
            return (loss, outputs) if return_outputs else loss

    class _Trainer(CoordSoftCEW1LossMixin, _ScalingBaseTrainer):
        def __init__(self, cfg):
            super().__init__()
            self.coord_soft_ce_w1_cfg = cfg
            self.template = DummyTemplate()
            self.args = SimpleNamespace(
                average_tokens_across_devices=False,
                gradient_accumulation_steps=4,
            )
            self.current_gradient_accumulation_steps = 4
            self.model_accepts_loss_kwargs = True
            self.model = SimpleNamespace(training=True)

    vocab = 1200
    labels = torch.tensor([[0, 5, 6, 7]], dtype=torch.long)
    logits = torch.zeros((1, labels.shape[1], vocab), dtype=torch.float32)

    cfg = CoordSoftCEW1Config.from_mapping({"enabled": True})
    trainer = _Trainer(cfg)

    # Baseline: the "transformers scaled" path (num_items_in_batch=None).
    loss_scaled = _ScalingBaseTrainer.compute_loss(
        trainer,
        model=None,
        inputs={"labels": labels, "fake_logits": logits},
        return_outputs=False,
        num_items_in_batch=None,
    )

    # Mixin path: overrides num_items_in_batch (packing-safe), so base does NOT
    # scale by grad-accum; the mixin must divide by grad-accum steps.
    loss_mixin = trainer.compute_loss(
        model=None,
        inputs={"labels": labels, "fake_logits": logits},
        return_outputs=False,
        num_items_in_batch=1,
    )

    assert torch.allclose(loss_mixin, loss_scaled, atol=1e-6)


def test_stage1_text_gate_penalizes_coord_vocab_mass_on_non_coord_positions() -> None:
    vocab = 256
    coord_token_ids = [100, 101, 102, 103]
    coord_id_map = _build_coord_id_map(vocab, coord_token_ids)
    labels = torch.tensor([[0, 5, 100, 6, 101]], dtype=torch.long)
    masked_labels = labels.clone()
    masked_labels[0, 2] = -100
    masked_labels[0, 4] = -100
    token_types = torch.tensor(
        [[-1, 1, 2, 3, 2]],
        dtype=torch.long,
    )

    logits = torch.full((1, 4, vocab), -20.0)
    logits[0, 0, 100] = 20.0
    logits[0, 1, 100] = 20.0
    logits[0, 2, 101] = 20.0
    logits[0, 3, 101] = 20.0

    cfg = CoordSoftCEW1Config.from_mapping(
        {
            "enabled": True,
            "ce_weight": 1.0,
            "soft_ce_weight": 0.0,
            "w1_weight": 0.0,
            "gate_weight": 0.0,
            "text_gate_weight": 1.0,
            "temperature": 1.0,
            "target_sigma": 2.0,
        }
    )

    result = compute_coord_soft_ce_w1_loss(
        logits=logits,
        labels=labels,
        masked_labels=masked_labels,
        coord_token_weights=None,
        coord_token_ids=coord_token_ids,
        coord_id_map=coord_id_map,
        tokenizer=DummyTokenizer(),
        token_types=token_types,
        cfg=cfg,
        average_tokens_across_devices=False,
        model_accepts_loss_kwargs=False,
        accelerator_num_processes=None,
    )

    assert result is not None
    assert float(result.text_gate_contrib.detach().item()) > 0.5
    assert result.text_gate_coord_mass_mean is not None
    assert float(result.text_gate_coord_mass_mean.detach().item()) > 0.9
