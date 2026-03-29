from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.config.schema import BBoxGeoConfig, BBoxSizeAuxConfig, CoordSoftCEW1Config
from src.data_collators.token_types import TokenType
from src.metrics.dataset_metrics import (
    AggregateTokenTypeMetricsMixin,
    BBoxGeoLossMixin,
    BBoxSizeAuxLossMixin,
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
    doc_path = Path(__file__).resolve().parents[1] / "docs/training/METRICS.md"
    text = doc_path.read_text(encoding="utf-8")
    # Ignore fenced code blocks so inline-code extraction is stable even when the doc
    # contains YAML/CLI examples that also use backticks.
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
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

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        mapping = {
            0: '{"objects":[{"desc":"x","bbox_2d":[',
            5: "text",
            6: "fmt",
            103: "<|coord_3|>",
            104: "<|coord_4|>",
            107: "<|coord_7|>",
            108: "<|coord_8|>",
        }
        return "".join(mapping.get(int(t), f"<tok_{int(t)}>") for t in token_ids)


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
    BBoxSizeAuxLossMixin,
    BBoxGeoLossMixin,
    CoordSoftCEW1LossMixin,
    _DummyBaseTrainer,
):
    def __init__(
        self,
        cfg: CoordSoftCEW1Config,
        bbox_geo_cfg: BBoxGeoConfig,
        bbox_cfg: BBoxSizeAuxConfig,
    ):
        self.coord_soft_ce_w1_cfg = cfg
        self.bbox_geo_cfg = bbox_geo_cfg
        self.bbox_size_aux_cfg = bbox_cfg
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
    # Labels: [BOS, text(5), bbox_2d=[3,4,7,8], format(6)]
    labels = torch.tensor([[0, 5, 103, 104, 107, 108, 6]] * bsz, dtype=torch.long)
    vocab = 1200
    # Trainer mixins expect logits to be sequence-length-aligned with labels (before shifting).
    logits = torch.full((bsz, labels.shape[1], vocab), -20.0)
    logits[:, 0, 5] = 20.0
    logits[:, 1, 103] = 20.0
    logits[:, 2, 104] = 20.0
    logits[:, 3, 107] = 20.0
    logits[:, 4, 108] = 20.0
    logits[:, 5, 6] = 20.0

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
            [[
                TokenType.IGNORE,
                TokenType.DESC,
                TokenType.COORD,
                TokenType.COORD,
                TokenType.COORD,
                TokenType.COORD,
                TokenType.FORMAT,
            ]]
            * bsz,
            dtype=torch.long,
        )
        inputs["token_types"] = types
    return inputs


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("with_token_types", [False, True])
@pytest.mark.parametrize("coord_enabled", [False, True])
@pytest.mark.parametrize("bbox_geo_enabled", [False, True])
@pytest.mark.parametrize("bbox_enabled", [False, True])
def test_stage1_metric_keys_are_documented_and_aggregate_only(
    *,
    packed: bool,
    with_token_types: bool,
    coord_enabled: bool,
    bbox_geo_enabled: bool,
    bbox_enabled: bool,
) -> None:
    doc_keys = _load_doc_keys()

    cfg = CoordSoftCEW1Config.from_mapping({"enabled": bool(coord_enabled)})
    bbox_geo_cfg = BBoxGeoConfig.from_mapping(
        {
            "enabled": bool(bbox_geo_enabled),
            "smoothl1_weight": 0.0,
            "ciou_weight": 1.0,
        }
    )
    bbox_cfg = BBoxSizeAuxConfig.from_mapping(
        {
            "enabled": bool(bbox_enabled),
            "log_wh_weight": 0.05,
            "oversize_penalty_weight": 0.0,
            "oversize_area_frac_threshold": None,
            "oversize_log_w_threshold": None,
            "oversize_log_h_threshold": None,
            "eps": 1e-6,
        }
    )
    trainer = _DummyTrainer(cfg, bbox_geo_cfg, bbox_cfg)

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

    train_metrics = trainer.custom_metrics["train"]
    total_metric = train_metrics.get("stage1/total_loss_per_sample_est")
    base_metric = train_metrics.get("base_ce/loss_per_sample")
    coord_metric = train_metrics.get("coord_diag/loss_per_sample")
    bbox_geo_metric = train_metrics.get("bbox_geo/loss_per_sample")
    bbox_metric = train_metrics.get("bbox_size_aux/loss_per_sample")
    if total_metric is not None and base_metric is not None and base_metric.values:
        expected_total = float(base_metric.values[-1])
        if coord_metric is not None and coord_metric.values:
            expected_total += float(coord_metric.values[-1])
        if bbox_geo_metric is not None and bbox_geo_metric.values:
            expected_total += float(bbox_geo_metric.values[-1])
        if bbox_metric is not None and bbox_metric.values:
            expected_total += float(bbox_metric.values[-1])
        assert total_metric.values[-1] == pytest.approx(expected_total)

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
