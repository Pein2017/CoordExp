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
        logits_to_keep = int(kwargs.get("logits_to_keep") or 0)
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
        if logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:, :]
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
    close_start_suppression_weight: float = 0.0,
    final_schema_close_weight: float = 0.0,
    pem: Mapping[str, Any] | None = None,
    train_forward: Mapping[str, Any] | None = None,
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
                "close_start_suppression_weight": close_start_suppression_weight,
                "final_schema_close_weight": final_schema_close_weight,
            },
            "positive_evidence_margin": dict(pem or {"objective": "disabled"}),
            "train_forward": dict(train_forward or {}),
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


def test_retained_graph_runtime_preserves_current_exact_branch_behavior() -> None:
    cfg = _cfg()
    trainer = _trainer(cfg)
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert len(model.calls) == 3
    assert _latest_metric(trainer, "mp/branch_runtime_mode") == pytest.approx(0.0)
    assert _latest_metric(
        trainer, "mp/retained_graph_branch_forwards"
    ) == pytest.approx(2.0)
    assert _latest_metric(trainer, "mp/checkpointed_branch_forwards") == pytest.approx(
        0.0
    )
    assert _latest_metric(
        trainer, "mp/objective_fidelity_exact_samples"
    ) == pytest.approx(1.0)
    assert _latest_metric(
        trainer, "mp/objective_fidelity_approx_samples"
    ) == pytest.approx(0.0)


def test_trainer_evaluate_uses_callback_without_prediction_loop() -> None:
    trainer = _trainer(_cfg())
    model = _FakeModel()
    trainer.model = model
    trainer.control = SimpleNamespace(should_evaluate=True)
    logged: list[dict[str, float]] = []

    class _CallbackHandler:
        def __init__(self, model):
            self.model = model

        def call_event(self, event, args, state, control, **kwargs):
            assert event == "on_evaluate"
            kwargs["model"] = self.model
            return self.on_evaluate(args, state, control, **kwargs)

        def on_evaluate(self, args, state, control, **kwargs):
            assert kwargs["model"] is self.model
            metrics = kwargs["metrics"]
            metrics["eval_det_f1ish/f1@0.5"] = 0.25
            return control

    def _fail_prediction_step(*args, **kwargs):
        raise AssertionError("generic prediction loop should not run")

    trainer.callback_handler = _CallbackHandler(model)
    trainer.prediction_step = _fail_prediction_step
    trainer.log = lambda metrics: logged.append(dict(metrics))

    metrics = trainer.evaluate()

    assert metrics["eval_det_f1ish/f1@0.5"] == pytest.approx(0.25)
    assert "eval/runtime" in metrics
    assert trainer.control.should_evaluate is False
    assert logged and logged[-1]["eval_det_f1ish/f1@0.5"] == pytest.approx(0.25)


def test_trainer_pem_threshold_loss_logs_pem_and_mp_diagnostic() -> None:
    trainer = _trainer(
        _cfg(
            pem={
                "objective": "threshold_loss",
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


def test_trainer_close_start_suppression_contributes_when_remaining_exists() -> None:
    trainer = _trainer(_cfg(close_start_suppression_weight=0.5))
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
    trainer = _trainer(_cfg(empty=0.0, full_prefix=1.0, final_schema_close_weight=0.25))
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
    cfg = _cfg(empty=1.0, final_schema_close_weight=0.0)
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


def test_trainer_pads_candidate_forwards_to_distributed_max_count() -> None:
    trainer = _trainer(_cfg())
    trainer._ddp_max_int = lambda value, model: 4
    model = _FakeModel()
    trainer.model = model

    trainer.compute_loss(model, _batch(), return_outputs=False)

    assert len(model.calls) == 5
    assert _latest_metric(trainer, "mp/num_candidates_scored") == pytest.approx(2.0)


def test_checkpointed_exact_runtime_pads_candidate_forwards_through_selected_runtime() -> (
    None
):
    trainer = _trainer(
        _cfg(train_forward={"branch_runtime": {"mode": "checkpointed_exact"}})
    )
    trainer._ddp_max_int = lambda value, model: 4
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert len(model.calls) == 5
    assert _latest_metric(trainer, "mp/branch_runtime_mode") == pytest.approx(1.0)
    assert _latest_metric(
        trainer, "mp/retained_graph_branch_forwards"
    ) == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/checkpointed_branch_forwards") == pytest.approx(
        4.0
    )
    assert _latest_metric(
        trainer, "mp/ddp_candidate_padding_forwards"
    ) == pytest.approx(2.0)


def test_smart_batched_exact_runtime_batches_candidate_forwards() -> None:
    trainer = _trainer(
        _cfg(
            train_forward={
                "branch_runtime": {"mode": "smart_batched_exact"},
                "branch_batching": {"enabled": True, "max_branch_rows": 8},
                "logits": {"mode": "supervised_suffix"},
                "ddp_sync": {"candidate_padding": "none"},
            }
        )
    )
    trainer._ddp_max_int = lambda value, model: 4
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert len(model.calls) == 2  # one candidate batch plus close-start metric branch
    assert model.calls[0]["input_ids"].shape[0] == 2
    assert int(model.calls[0].get("logits_to_keep") or 0) > 0
    assert _latest_metric(trainer, "mp/branch_runtime_mode") == pytest.approx(2.0)
    assert _latest_metric(
        trainer, "mp/retained_graph_branch_forwards"
    ) == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/checkpointed_branch_forwards") == pytest.approx(
        0.0
    )
    assert _latest_metric(trainer, "mp/smart_batched_branch_forwards") == pytest.approx(
        1.0
    )
    assert _latest_metric(trainer, "mp/branch_batch_count") == pytest.approx(1.0)
    assert _latest_metric(trainer, "mp/branch_batch_rows_max") == pytest.approx(2.0)
    assert _latest_metric(
        trainer, "mp/ddp_candidate_padding_forwards"
    ) == pytest.approx(0.0)


def test_smart_batched_exact_matches_retained_graph_mp_and_logz_metrics() -> None:
    retained_trainer = _trainer(
        _cfg(
            train_forward={
                "logits": {"mode": "supervised_suffix"},
                "ddp_sync": {"candidate_padding": "none"},
            }
        )
    )
    smart_trainer = _trainer(
        _cfg(
            train_forward={
                "branch_runtime": {"mode": "smart_batched_exact"},
                "branch_batching": {"enabled": True, "max_branch_rows": 8},
                "logits": {"mode": "supervised_suffix"},
                "ddp_sync": {"candidate_padding": "none"},
            }
        )
    )
    retained_model = _FakeModel()
    smart_model = _FakeModel()
    retained_trainer.model = retained_model
    smart_trainer.model = smart_model

    retained_loss = retained_trainer.compute_loss(
        retained_model, _batch(), return_outputs=False
    )
    smart_loss = smart_trainer.compute_loss(smart_model, _batch(), return_outputs=False)

    assert torch.allclose(retained_loss.detach(), smart_loss.detach(), atol=1e-6)
    for key in (
        "loss/mp",
        "loss/mp_diagnostic",
        "mp/logZ_scored_raw",
        "mp/logZ_remaining_est",
        "mp/responsibility_entropy_scored",
    ):
        assert _latest_metric(smart_trainer, key) == pytest.approx(
            _latest_metric(retained_trainer, key),
            abs=1e-6,
        )
    assert len(smart_model.calls) < len(retained_model.calls)


def test_smart_batched_exact_runtime_rejects_branch_local_aux_without_schema() -> None:
    trainer = _trainer(
        _cfg(
            train_forward={
                "branch_runtime": {"mode": "smart_batched_exact"},
                "branch_batching": {"enabled": True},
            }
        )
    )
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

    with pytest.raises(ValueError, match="smart_batched_exact.*branch-local aux"):
        trainer.compute_loss(model, _batch(), return_outputs=False)


def test_supervised_suffix_runtime_scores_without_full_prefix_logits() -> None:
    trainer = _trainer(
        _cfg(
            train_forward={
                "logits": {"mode": "supervised_suffix"},
                "ddp_sync": {"candidate_padding": "none"},
            }
        )
    )
    trainer._ddp_max_int = lambda value, model: 4
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert len(model.calls) == 3
    assert all(int(call.get("logits_to_keep") or 0) > 0 for call in model.calls)
    assert _latest_metric(
        trainer, "mp/ddp_candidate_padding_forwards"
    ) == pytest.approx(0.0)
    assert _latest_metric(
        trainer, "mp/ddp_candidate_forward_local_count"
    ) == pytest.approx(2.0)
    assert _latest_metric(
        trainer, "mp/ddp_candidate_forward_max_count"
    ) == pytest.approx(4.0)
    assert _latest_metric(trainer, "mp/ddp_candidate_padding_policy") == pytest.approx(
        1.0
    )


def test_budget_fallback_scores_subset_and_reports_approximate_fidelity() -> None:
    trainer = _trainer(
        _cfg(
            train_forward={
                "budget_policy": {
                    "enabled": True,
                    "exact_until": {"max_candidates": 2},
                    "fallback": {
                        "mode": "approximate_uniform_subsample",
                        "max_candidates": 1,
                        "estimator": "uniform_importance",
                    },
                }
            }
        )
    )
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(
        model,
        _batch([OBJECT_A, OBJECT_B, OBJECT_C]),
        return_outputs=False,
    )

    assert torch.isfinite(loss)
    assert _latest_metric(trainer, "mp/num_remaining_objects") == pytest.approx(3.0)
    assert _latest_metric(trainer, "mp/num_candidates_scored") == pytest.approx(1.0)
    assert _latest_metric(
        trainer, "mp/objective_fidelity_exact_samples"
    ) == pytest.approx(0.0)
    assert _latest_metric(
        trainer, "mp/objective_fidelity_approx_samples"
    ) == pytest.approx(1.0)
    assert _latest_metric(trainer, "mp/fallback_applied_samples") == pytest.approx(1.0)
    assert _latest_metric(
        trainer, "mp/fallback_reason_candidate_budget"
    ) == pytest.approx(1.0)


def test_set_continuation_requires_raw_metadata_batch() -> None:
    trainer = _trainer(_cfg())
    model = _FakeModel()
    trainer.model = model

    with pytest.raises(ValueError, match="set_continuation_meta"):
        trainer.compute_loss(model, {"input_ids": torch.tensor([[1]])})
