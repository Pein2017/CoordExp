from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.config.schema import CoordSoftCEW1Config
from src.data_collators.token_types import TokenType
from src.metrics.dataset_metrics import (
    AggregateTokenTypeMetricsMixin,
    CoordSoftCEW1LossMixin,
    GradAccumLossScaleMixin,
)


class _DummyMetric:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, v: float) -> None:
        self.values.append(float(v))


def _load_doc_keys() -> set[str]:
    # tests/ -> repo root
    doc_path = Path(__file__).resolve().parents[1] / "docs/training/METRICS_LOSSES.md"
    text = doc_path.read_text(encoding="utf-8")
    keys: set[str] = set()

    # Extract inline-code spans, then keep only strings that *look like* metric keys.
    for span in re.findall(r"`([^`]+)`", text):
        if any(ch in span for ch in ("/", "_")):
            keys.add(span)
    return keys


class _DummyTokenizer:
    """Maps `<|coord_k|>` -> 100+k (full 1000-bin coord vocab in-range)."""

    def convert_tokens_to_ids(self, tokens):
        out = []
        for tok in tokens:
            if tok.startswith("<|coord_") and tok.endswith("|>"):
                out.append(100 + int(tok[len("<|coord_") : -len("|>")]))
            else:
                out.append(1)
        return out


class _DummyTemplate:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()


class _DummyBaseTrainer:
    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        logits = inputs["fake_logits"]
        labels = inputs["labels"]
        seq_len = min(int(logits.shape[1]), max(int(labels.shape[1]) - 1, 0))

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


class _DummyTrainer(
    GradAccumLossScaleMixin,
    AggregateTokenTypeMetricsMixin,
    CoordSoftCEW1LossMixin,
    _DummyBaseTrainer,
):
    def __init__(self, cfg: CoordSoftCEW1Config):
        self.coord_soft_ce_w1_cfg = cfg
        self.template = _DummyTemplate()
        self.args = SimpleNamespace(
            average_tokens_across_devices=False,
            gradient_accumulation_steps=1,
        )
        self.current_gradient_accumulation_steps = 1
        self.model_accepts_loss_kwargs = False
        self.model = SimpleNamespace(training=True)

        # ms-swift-compatible container: mode -> key -> metric object with .update(float)
        self.custom_metrics = {
            "train": defaultdict(_DummyMetric),
            "eval": defaultdict(_DummyMetric),
        }


def _make_toy_batch(*, bsz: int, with_token_types: bool, pack_num_samples: torch.Tensor):
    # Labels: [BOS, text(5), coord(bin=3)->103, format(6), coord(bin=7)->107]
    labels = torch.tensor([[0, 5, 103, 6, 107]] * bsz, dtype=torch.long)
    vocab = 1200
    # Trainer mixins expect logits to be sequence-length-aligned with labels (before shifting).
    logits = torch.full((bsz, labels.shape[1], vocab), -20.0)
    logits[..., 5] = 20.0
    logits[..., 103] = 20.0
    logits[..., 6] = 20.0
    logits[..., 107] = 20.0

    inputs = {
        "labels": labels,
        "fake_logits": logits,
        # Batch-extras (must be stripped before model forward, but stashed for diagnostics).
        "pack_num_samples": pack_num_samples,
        "dataset_labels": ["lvis"] * bsz,
        "dataset_segments": [[0]] * bsz,
    }
    if with_token_types:
        types = torch.tensor(
            [[TokenType.IGNORE, TokenType.DESC, TokenType.COORD, TokenType.FORMAT, TokenType.COORD]]
            * bsz,
            dtype=torch.long,
        )
        inputs["token_types"] = types
    return inputs


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("with_token_types", [False, True])
@pytest.mark.parametrize("coord_enabled", [False, True])
def test_stage1_metric_keys_are_documented_and_aggregate_only(
    *, packed: bool, with_token_types: bool, coord_enabled: bool
) -> None:
    doc_keys = _load_doc_keys()

    cfg = CoordSoftCEW1Config.from_mapping({"enabled": bool(coord_enabled)})
    trainer = _DummyTrainer(cfg)

    if packed:
        # One "packed unit" that represents multiple original samples.
        inputs = _make_toy_batch(
            bsz=1, with_token_types=with_token_types, pack_num_samples=torch.tensor([3])
        )
    else:
        inputs = _make_toy_batch(
            bsz=2, with_token_types=with_token_types, pack_num_samples=torch.tensor([1, 1])
        )

    _ = trainer.compute_loss(
        model=None, inputs=inputs, return_outputs=False, num_items_in_batch=1
    )

    emitted_train = set(trainer.custom_metrics["train"].keys())

    # No per-dataset buckets in Stage-1 metrics.
    assert all("lvis" not in k and "coco" not in k for k in emitted_train)

    # Parity check: emitted keys should be documented (feature-conditional keys are fine).
    missing = sorted(k for k in emitted_train if k not in doc_keys)
    assert not missing, f"Found undocumented metric keys: {missing}"

    # Eval bucket should use the same keys; ms-swift is responsible for the `eval_` prefix
    # during logging.
    trainer.model.training = False

    # compute_loss mutates/pops batch-extras from `inputs`; use a fresh payload for eval.
    if packed:
        inputs_eval = _make_toy_batch(
            bsz=1, with_token_types=with_token_types, pack_num_samples=torch.tensor([3])
        )
    else:
        inputs_eval = _make_toy_batch(
            bsz=2,
            with_token_types=with_token_types,
            pack_num_samples=torch.tensor([1, 1]),
        )

    _ = trainer.compute_loss(
        model=None, inputs=inputs_eval, return_outputs=False, num_items_in_batch=1
    )
    emitted_eval = set(trainer.custom_metrics["eval"].keys())
    assert emitted_eval == emitted_train
