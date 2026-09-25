"""Characterize retained native operations without loading a model or dataset."""
from types import SimpleNamespace

import pytest
import torch

from src.qwen import native_row_scores as scoring
from src.qwen.saved_prefix import EOS, full_prefix, prefix_tokens
from src.eval.numerical_recurrence import longest_run, rows, release_metrics
from src.losses.token_scores import masked_active_mean_ce


def test_companion_conditioning_policies_stay_explicitly_distinct():
    raw = [{"token_ids": [7]}, {"token_ids": [8, 9]}]
    batch = SimpleNamespace(prompt_token_ids=[[1], [2]])
    assert prefix_tokens(raw, 3, 0) == [[7, EOS, 0], [8, 9, EOS]]
    assert scoring.candidate_histories(batch, raw, 1, [3, 4, 5], 0) == [
        [1, 7, 0, 0], [2, 3, 4, 5]]
    assert raw == [{"token_ids": [7]}, {"token_ids": [8, 9]}]


def test_prefix_keeps_images_and_rejects_wrong_literal_mutation():
    pixels = torch.tensor([[1.0, 2.0]])
    batch = SimpleNamespace(inputs={
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "pixel_values": pixels, "position_ids": torch.zeros(3, 2, 2),
        "use_cache": True, "logits_to_keep": 1,
    })
    raw = [{"token_ids": [7]}, {"token_ids": [8, 9]}]
    actual = full_prefix(batch, raw, 3, 0, "cpu", (1, 0, 8, 6))
    assert actual["input_ids"].tolist() == [[1, 2, 7, EOS, 0], [3, 4, 6, 9, EOS]]
    assert actual["pixel_values"] is pixels
    assert actual["attention_mask"].tolist() == [[1] * 5, [1] * 5]
    assert not {"position_ids", "use_cache", "logits_to_keep"} & actual.keys()
    with pytest.raises(ValueError, match="original prefix"):
        full_prefix(batch, raw, 3, 0, "cpu", (1, 0, 99, 6))


@pytest.mark.parametrize("paired", [True, False])
def test_trace_comparison_preserves_both_saved_formats_and_winner_checks(paired):
    logits = torch.tensor([2.0, 1.0, -1.0])
    step = {"chosen": [0], "chosen_raw_logits": [2.0],
            "logsumexp": [float(logits.logsumexp(-1))],
            "raw_top2": [[[0, 2.0], [1, 1.0]]] if paired else [[2.0, 1.0]],
            "raw_winners": [0], "raw_runnerups": [1]}
    args = dict(logits=logits, trace={"steps": [step]}, batch_index=0,
                absolute_offset=0, token_id=0, role="x1")
    assert scoring.compare_saved_trace(**args)["passed"]
    step["chosen"] = [1]
    assert not scoring.compare_saved_trace(**args)["passed"]


def test_compact_row_scoring_aligns_causal_rows_and_keeps_inference_only(monkeypatch):
    seen = {}

    def materialize(model, inputs, histories, **kwargs):
        seen.update(histories=histories, **kwargs)
        return {"input_ids": torch.tensor(histories),
                "position_ids": torch.arange(4).expand(3, 1, 4)}

    class TinyModel:
        def __call__(self, **inputs):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(logits=torch.tensor([
                [[0., 1., 2.], [2., 1., 0.], [7., 0., 0.]]]))

    monkeypatch.setattr(scoring, "exact_history_inputs", materialize)
    actual = scoring.score_candidate(
        model=TinyModel(), batch=SimpleNamespace(inputs={}, prompt_token_ids=[[1]]),
        raw=[{"token_ids": [0]}], target=0, prefix=[0], tokens=[2, 1],
        pad=0, device=torch.device("cpu"))
    assert seen == {"histories": [[1, 0, 2, 1]], "pad_token_id": 0, "logits_to_keep": 3}
    expected = torch.tensor([[0., 1., 2.], [2., 1., 0.]]).log_softmax(-1)
    assert actual["token_logprobs"] == pytest.approx([float(expected[0, 2]), float(expected[1, 1])])
    assert actual["positions"] == [[1, 1, 1], [2, 2, 2]]
    assert actual["action_logits"].shape == (2, 3)


def test_near_recurrence_does_not_use_transitive_neighbor_chaining():
    def row(x):
        return [151646, 100, 151647, 151648, 151670+x, 151671, 151700, 151720, 151649]

    tokens = row(0) + row(7) + row(14)
    assert longest_run(rows(tokens), 8) == 2
    assert release_metrics(tokens, rows(row(0))[0])["physical_recovery"] == "not_established_by_numerical_metrics"


def test_masked_ce_keeps_active_token_mean_and_direct_gradient_mask():
    logits = torch.tensor([[1., 0.], [0., 2.], [0., 1.]], requires_grad=True)
    target = torch.tensor([0, 0, 1])
    loss, receipt = masked_active_mean_ce(logits, target, [1, 0, 1])
    expected = -torch.log_softmax(logits, -1)[[0, 2], [0, 1]].mean()
    assert torch.equal(loss, expected)
    assert receipt["active_tokens"] == 2
    loss.backward()
    assert torch.equal(logits.grad[1], torch.zeros(2))
    with pytest.raises(ValueError, match="no active positions"):
        masked_active_mean_ce(logits, target, [0, 0, 0])
