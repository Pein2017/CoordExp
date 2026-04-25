from __future__ import annotations

import math
import re
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from src.config.schema import CoordSoftCEW1Config, Stage1SetContinuationConfig
from src.data_collators.stage1_set_continuation_collator import (
    build_stage1_set_continuation_collator,
)
from src.trainers.stage1_set_continuation import Stage1SetContinuationTrainer
from src.trainers.stage1_set_continuation.sampling import Stage1SetContinuationSample


OBJECT_A = {
    "desc": "cat",
    "bbox_2d": [
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    ],
}
OBJECT_B = {
    "desc": "dog",
    "bbox_2d": [
        "<|coord_11|>",
        "<|coord_12|>",
        "<|coord_13|>",
        "<|coord_14|>",
    ],
}
OBJECT_C = {
    "desc": "bus",
    "bbox_2d": [
        "<|coord_21|>",
        "<|coord_22|>",
        "<|coord_23|>",
        "<|coord_24|>",
    ],
}


class _Metric:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))


class _FakeTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_text_id = 1

    def _text_id(self, token: str) -> int:
        if token not in self._token_to_id:
            while 100 <= self._next_text_id < 1100:
                self._next_text_id += 1
            token_id = self._next_text_id
            self._next_text_id += 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]

    @staticmethod
    def _coord_id(token: str) -> int:
        return 100 + int(token[len("<|coord_") : -len("|>")])

    def encode_text(self, text: str) -> list[int]:
        tokens = re.findall(r"<\|coord_\d+\|>|[A-Za-z0-9_]+|[^\s]", text)
        ids: list[int] = []
        for token in tokens:
            if token.startswith("<|coord_") and token.endswith("|>"):
                token_id = self._coord_id(token)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            else:
                token_id = self._text_id(token)
            ids.append(token_id)
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token.get(int(idx), f"<tok_{int(idx)}>") for idx in ids]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            if tokens.startswith("<|coord_") and tokens.endswith("|>"):
                return self._coord_id(tokens)
            return self._text_id(tokens)
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(self.convert_ids_to_tokens(list(token_ids)))


class _FakeTemplate:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.system = "template-system"

    def encode(
        self, rendered: Mapping[str, Any], return_length: bool = True
    ) -> dict[str, Any]:
        assistant = next(
            message
            for message in rendered["messages"]
            if message.get("role") == "assistant"
        )
        content = assistant["content"]
        if isinstance(content, str):
            text = content
        else:
            text = next(
                item["text"]
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            )
        ids = self.tokenizer.encode_text(text)
        return {
            "input_ids": ids,
            "labels": list(ids),
            "attention_mask": [1] * len(ids),
            "length": len(ids),
        }


class _FakeModel:
    def __init__(self, vocab_size: int = 1300) -> None:
        self.training = True
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=int(vocab_size))
        )
        self._emb = SimpleNamespace(weight=torch.zeros((vocab_size, 4)))

    def get_input_embeddings(self):
        return self._emb

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        vocab_size = int(self.config.text_config.vocab_size)
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], vocab_size),
            -12.0,
            dtype=torch.float32,
        )
        if input_ids.shape[1] > 1:
            next_ids = input_ids[:, 1:]
            for batch_idx in range(input_ids.shape[0]):
                for pos in range(input_ids.shape[1] - 1):
                    logits[batch_idx, pos, int(next_ids[batch_idx, pos])] = 12.0
        logits[:, -1, 0] = 12.0
        return SimpleNamespace(logits=logits)


def _raw_sample(objects: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "assistant_payload": {"objects": list(objects or [OBJECT_A, OBJECT_B])},
        "objects": [{"metadata": "not-the-serialized-objects"}],
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/image.jpg"},
                    {"type": "text", "text": "detect all objects"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"objects": []}'}],
            },
        ],
        "metadata": {"dataset": "toy", "image_id": 7},
        "sample_id": "toy-7",
        "base_idx": 7,
    }


def _cfg(
    *,
    empty: float = 1.0,
    random_subset: float = 0.0,
    leave_one_out: float = 0.0,
    full_prefix: float = 0.0,
    anti_close_weight: float = 0.0,
    final_close_weight: float = 0.0,
    pem: Mapping[str, Any] | None = None,
) -> Stage1SetContinuationConfig:
    return Stage1SetContinuationConfig.from_mapping(
        {
            "subset_sampling": {
                "empty_prefix_ratio": empty,
                "random_subset_ratio": random_subset,
                "leave_one_out_ratio": leave_one_out,
                "full_prefix_ratio": full_prefix,
                "prefix_order": "dataset",
            },
            "candidates": {"mode": "exact"},
            "structural_close": {
                "anti_close_weight": anti_close_weight,
                "final_close_weight": final_close_weight,
            },
            "positive_evidence_margin": dict(pem or {"mode": "disabled"}),
        }
    )


def _trainer(cfg: Stage1SetContinuationConfig) -> Stage1SetContinuationTrainer:
    trainer = object.__new__(Stage1SetContinuationTrainer)
    trainer.stage1_set_continuation_cfg = cfg
    trainer.object_field_order = "desc_first"
    trainer.args = SimpleNamespace(seed=123, process_index=0, local_rank=-1)
    trainer.state = SimpleNamespace(global_step=0, epoch=0)
    trainer.template = _FakeTemplate()
    trainer.custom_metrics = {
        "train": defaultdict(_Metric),
        "eval": defaultdict(_Metric),
    }
    return trainer


def _batch(objects: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return build_stage1_set_continuation_collator()([_raw_sample(objects)])


def _batch_many(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return build_stage1_set_continuation_collator()(rows)


def _latest_metric(trainer: Stage1SetContinuationTrainer, key: str) -> float:
    values = trainer.custom_metrics["train"][key].values
    assert values, f"metric {key!r} was not emitted"
    return values[-1]


def _forced_sample(
    *,
    mode: str,
    prefix: tuple[int, ...],
    remaining: tuple[int, ...],
    candidates: tuple[int, ...],
) -> Stage1SetContinuationSample:
    return Stage1SetContinuationSample(
        selected_mode=mode,
        configured_mixture={mode: 1.0},
        resolved_valid_mixture={mode: 1.0},
        prefix_indices=prefix,
        remaining_indices=remaining,
        candidate_indices=candidates,
        candidate_scoring_mode="exact",
        scored_candidate_fraction=1.0,
    )


def test_trainer_exact_mp_scores_two_independent_branches_and_logs_metrics() -> None:
    trainer = _trainer(_cfg())
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert (
        len(model.calls) == 3
    )  # two candidate branches plus close-start metric branch
    decoded_calls = [
        trainer.template.tokenizer.decode(call["input_ids"].reshape(-1).tolist())
        for call in model.calls
    ]
    candidate_calls = [text for text in decoded_calls if "desc" in text]
    assert sum("cat" in text for text in candidate_calls) == 1
    assert sum("dog" in text for text in candidate_calls) == 1
    assert all(not ("cat" in text and "dog" in text) for text in candidate_calls)
    assert _latest_metric(trainer, "loss/mp") < 0.1
    assert _latest_metric(trainer, "mp/num_prefix_objects") == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/num_remaining_objects") == pytest.approx(2.0)
    assert _latest_metric(trainer, "mp/num_candidates_scored") == pytest.approx(2.0)
    assert _latest_metric(trainer, "mp/configured_ratio_empty_prefix") == pytest.approx(
        1.0
    )
    assert _latest_metric(
        trainer, "mp/resolved_valid_ratio_empty_prefix"
    ) == pytest.approx(1.0)
    assert _latest_metric(trainer, "stop/p_close_start_when_remaining_exists") > 0.99
    assert _latest_metric(trainer, "stop/p_continue_start_when_remaining_exists") < 0.01


def test_trainer_pem_replacement_logs_pem_and_mp_diagnostic() -> None:
    trainer = _trainer(
        _cfg(
            pem={
                "mode": "replace_mp",
                "log_rho": math.log(0.9),
                "threshold_calibration": "authored_fixed_ablation",
            }
        )
    )
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert _latest_metric(trainer, "loss/pem") == pytest.approx(0.0, abs=1e-6)
    assert _latest_metric(trainer, "loss/mp_diagnostic") < 0.1


def test_trainer_anti_close_weight_contributes_when_remaining_exists() -> None:
    trainer = _trainer(_cfg(anti_close_weight=0.5))
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert _latest_metric(trainer, "loss/anti_close_start") > 1.0
    assert _latest_metric(trainer, "stop/p_close_start_when_remaining_exists") > 0.99


def test_trainer_adds_branch_local_coord_aux_when_enabled() -> None:
    trainer = _trainer(_cfg())
    trainer.coord_soft_ce_w1_cfg = CoordSoftCEW1Config.from_mapping(
        {
            "enabled": True,
            "ce_weight": 0.1,
            "soft_ce_weight": 0.0,
            "w1_weight": 0.0,
            "gate_weight": 0.0,
        }
    )
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert _latest_metric(trainer, "loss/aux_coord_soft_ce_w1") >= 0.0
    assert _latest_metric(
        trainer, "aux/coord_soft_ce_w1/candidate_count"
    ) == pytest.approx(2.0)
    assert _latest_metric(trainer, "aux/coord_soft_ce_w1/position_count") > 0.0
    assert _latest_metric(
        trainer, "aux/coord_soft_ce_w1/skipped_candidates"
    ) == pytest.approx(0.0)


def test_trainer_weak_schema_close_runs_for_full_prefix_samples() -> None:
    trainer = _trainer(_cfg(empty=0.0, full_prefix=1.0, final_close_weight=0.25))
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert len(model.calls) == 1
    assert _latest_metric(trainer, "loss/weak_schema_close") < 0.1
    assert _latest_metric(trainer, "mp/loss_mp_denominator_samples") == pytest.approx(
        0.0
    )
    assert _latest_metric(trainer, "stop/p_close_start_when_remaining_empty") > 0.99


def test_full_prefix_metric_only_sample_does_not_dilute_batch_objective() -> None:
    cfg = _cfg(empty=1.0, final_close_weight=0.0)
    single_trainer = _trainer(cfg)
    single_model = _FakeModel()
    single_trainer.model = single_model

    mixed_trainer = _trainer(cfg)
    mixed_model = _FakeModel()
    mixed_trainer.model = mixed_model

    def _single_state(*, meta: Mapping[str, Any], sample_offset: int):
        del meta, sample_offset
        return _forced_sample(
            mode="empty_prefix",
            prefix=(),
            remaining=(0, 1),
            candidates=(0, 1),
        )

    def _mixed_state(*, meta: Mapping[str, Any], sample_offset: int):
        del meta
        if sample_offset == 0:
            return _forced_sample(
                mode="empty_prefix",
                prefix=(),
                remaining=(0, 1),
                candidates=(0, 1),
            )
        return _forced_sample(
            mode="full_prefix",
            prefix=(0, 1),
            remaining=(),
            candidates=(),
        )

    single_trainer._sample_state = _single_state  # type: ignore[method-assign]
    mixed_trainer._sample_state = _mixed_state  # type: ignore[method-assign]

    single_loss = single_trainer.compute_loss(
        single_model,
        _batch_many([_raw_sample()]),
        return_outputs=False,
    )
    mixed_loss = mixed_trainer.compute_loss(
        mixed_model,
        _batch_many([_raw_sample(), _raw_sample()]),
        return_outputs=False,
    )

    assert mixed_loss.detach().item() == pytest.approx(single_loss.detach().item())
    assert _latest_metric(
        mixed_trainer, "mp/loss_mp_denominator_samples"
    ) == pytest.approx(0.5)


def test_repeated_forward_budget_ratio_uses_token_counts() -> None:
    trainer = _trainer(_cfg(empty=0.0, leave_one_out=1.0))
    trainer._sample_state = lambda **_: _forced_sample(
        mode="leave_one_out",
        prefix=(0,),
        remaining=(1, 2),
        candidates=(1, 2),
    )
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(
        model,
        _batch([OBJECT_A, OBJECT_B, OBJECT_C]),
        return_outputs=False,
    )

    assert torch.isfinite(loss)
    prefix_tokens = _latest_metric(trainer, "mp/prefix_tokens_mean")
    candidate_tokens = _latest_metric(trainer, "mp/total_candidate_tokens_scored")
    ratio = _latest_metric(trainer, "mp/repeated_forward_token_ratio_vs_baseline")
    expected = (prefix_tokens * 2.0 + candidate_tokens) / (
        prefix_tokens + candidate_tokens
    )
    assert ratio == pytest.approx(expected)
    assert ratio > 1.0


def test_return_outputs_uses_explicit_batch_aligned_aggregate_output() -> None:
    trainer = _trainer(_cfg())
    model = _FakeModel()
    trainer.model = model

    loss, outputs = trainer.compute_loss(
        model,
        _batch_many([_raw_sample(), _raw_sample()]),
        return_outputs=True,
    )

    assert torch.isfinite(loss)
    assert isinstance(outputs, Mapping)
    assert outputs["loss"] is loss
    assert outputs["logits"].shape == (2, 0, 0)
    assert outputs["set_continuation_outputs"] == "aggregate_loss_only"


def test_numeric_estimator_and_branch_semantics_metrics_are_emitted() -> None:
    trainer = _trainer(_cfg())
    model = _FakeModel()
    trainer.model = model

    trainer.compute_loss(model, _batch(), return_outputs=False)

    assert _latest_metric(trainer, "mp/logZ_estimator") == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/candidate_scoring_mode") == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/prefix_attach_mode") == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/branch_isolation") == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/prefix_gradient") == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/prefix_tokens_mean") > 0.0
    assert _latest_metric(trainer, "mp/branch_forwards_per_sample") == pytest.approx(
        3.0
    )


def test_set_continuation_requires_raw_metadata_batch() -> None:
    trainer = _trainer(_cfg())
    model = _FakeModel()
    trainer.model = model

    with pytest.raises(ValueError, match="set_continuation_meta"):
        trainer.compute_loss(model, {"input_ids": torch.tensor([[1]])})
