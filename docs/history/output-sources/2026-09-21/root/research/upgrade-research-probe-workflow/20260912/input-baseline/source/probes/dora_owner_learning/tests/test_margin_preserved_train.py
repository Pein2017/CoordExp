from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from probes.dora_owner_learning import margin_preserved_train as margin_train
from probes.dora_owner_learning import repeat_recovery_train as old_train


def _margin_case(*, positions=(0, 1), source=(2.0, 0.4), targets=(0, 1)) -> dict:
    return {
        "key": "normal-1",
        "eligible_positions": list(positions),
        "original_kl_positions": [0, 1],
        "target_ids": list(targets),
        "source_margins": list(source),
        "floors": [min(0.1, 0.5 * value) for value in source],
    }


def test_current_best_other_is_recomputed_over_full_vocabulary() -> None:
    logits = torch.tensor([[3.0, 2.0, 9.0, 8.0], [1.0, 7.0, 6.0, 5.0]])
    targets = torch.tensor([2, 1])
    margins, competitors = margin_train.current_target_margins(logits, targets)
    assert competitors.tolist() == [3, 2]
    assert margins.tolist() == [1.0, 1.0]


def test_source_floor_has_exact_zero_value_and_gradient() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.4, -1.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    penalty, stats = margin_train.worst_margin_penalty(
        logits, targets, _margin_case(),
    )
    assert float(penalty.detach()) == 0.0
    assert stats["active_count"] == 0
    penalty.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_violated_worst_margin_has_nonzero_target_and_competitor_gradient() -> None:
    logits = torch.tensor([[0.02, 0.0, -1.0], [0.0, 0.4, -1.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    penalty, stats = margin_train.worst_margin_penalty(
        logits, targets, _margin_case(),
    )
    assert float(penalty.detach()) == pytest.approx(0.08)
    assert stats["active_count"] == 1 and stats["worst_best_other_id"] == 1
    penalty.backward()
    assert float(logits.grad[0, 0]) == -1.0
    assert float(logits.grad[0, 1]) == 1.0
    assert float(logits.grad[1].abs().sum()) == 0.0


def test_margin_ddp_scale_is_exact_global_mean56() -> None:
    penalties = [torch.tensor(float(index + 1)) for index in range(56)]
    local = [sum(
        margin_train.local_margin_contribution(value, weight=10.0)
        for value in penalties[rank::8]
    ) for rank in range(8)]
    assert float(sum(local) / 8) == pytest.approx(10 * sum(range(1, 57)) / 56)


def test_margin_input_keeps_literal_targets_original_mask_and_frozen_floor() -> None:
    case = _margin_case()
    checked = margin_train.validate_margin_case(
        case, action_ids=[0, 1], kl_positions=[0, 1], expected_key="normal-1",
    )
    assert checked["target_ids"] == [0, 1]

    outside = {**case, "eligible_positions": [0, 2], "target_ids": [0, 2]}
    with pytest.raises(ValueError):
        margin_train.validate_margin_case(
            outside, action_ids=[0, 1, 2], kl_positions=[0, 1], expected_key="normal-1",
        )
    retargeted = {**case, "target_ids": [1, 0]}
    with pytest.raises(ValueError):
        margin_train.validate_margin_case(
            retargeted, action_ids=[0, 1], kl_positions=[0, 1], expected_key="normal-1",
        )
    refloored = {**case, "floors": [0.1, 0.19]}
    with pytest.raises(ValueError):
        margin_train.validate_margin_case(
            refloored, action_ids=[0, 1], kl_positions=[0, 1], expected_key="normal-1",
        )


def test_final_reference_summary_is_post_update_mean56_with_explicit_burden() -> None:
    rows = []
    for index in range(56):
        eligible = 145 if index == 0 else 107
        rows.append({
            "key": f"normal-{index}",
            "raw_normal_kl": index / 1000,
            "raw_margin_R_i": 0.2 if index < 2 else 0.0,
            "margin": {
                "eligible_count": eligible,
                "active_count": 3 if index < 2 else 0,
                "literal_argmax_flips": 4 if index == 0 else 0,
            },
        })
    summary = margin_train.summarize_final_reference(rows, margin_weight=10.0)
    assert summary["eligible_positions"] == 6030
    assert summary["raw_normal_kl_mean"] == pytest.approx(sum(range(56)) / 56_000)
    assert summary["scaled_normal_kl"] == pytest.approx(100 * sum(range(56)) / 56_000)
    assert summary["raw_margin_R_mean"] == pytest.approx(0.4 / 56)
    assert summary["scaled_margin"] == pytest.approx(4.0 / 56)
    assert summary["active_images"] == 2
    assert summary["active_floors"] == 6
    assert summary["eligible_literal_argmax_flips"] == 4
    assert len(summary["active_worst_positions"]) == 2


class _Replay:
    def __init__(self, targets: list[int]) -> None:
        self.inputs = {}
        self.target_ids = torch.tensor(targets, dtype=torch.long)

    def aligned_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


class _Model(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())
        self.calls = 0

    def forward(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(logits=self.logits)


def test_normal_kl_and_margin_share_one_forward_and_weight0_matches_old_a(monkeypatch) -> None:
    targets = [0, 1]
    replay = _Replay(targets)
    monkeypatch.setattr(margin_train, "prepare_replay", lambda *args, **kwargs: replay)
    monkeypatch.setattr(old_train, "prepare_replay", lambda *args, **kwargs: replay)
    logits = torch.tensor([[0.02, 0.0, -1.0], [0.0, 0.4, -1.0]])
    reference = torch.log_softmax(torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.4, -1.0]]), -1)

    model = _Model(logits)
    scorer = margin_train.MarginPreservedScorer(model, margin_weight=10.0)
    loss, stats = scorer(
        {}, [], targets, kind="normal", reference_logp=reference,
        positions=[0, 1], margin=_margin_case(),
    )
    assert model.calls == 1
    expected_kl = old_train.reference_kl(logits, reference, [0, 1])
    assert float(loss) == pytest.approx(
        float(expected_kl) * 100 * 8 / 56 + 0.08 * 10 * 8 / 56,
    )
    assert stats["normal_kl"] == pytest.approx(float(expected_kl))
    assert stats["margin_penalty"] == pytest.approx(0.08)

    old_model, zero_model = _Model(logits), _Model(logits)
    old_loss, _ = old_train.RepeatRecoveryScorer(old_model)(
        {}, [], targets, kind="normal", reference_logp=reference, positions=[0, 1],
    )
    zero_loss, zero_stats = margin_train.MarginPreservedScorer(
        zero_model, margin_weight=0.0,
    )({}, [], targets, kind="normal", reference_logp=reference,
      positions=[0, 1], margin=_margin_case())
    assert torch.equal(zero_loss, old_loss)
    assert zero_stats["margin_scaled_loss"] == 0.0
    zero_loss.backward()
    old_loss.backward()
    assert torch.equal(zero_model.logits.grad, old_model.logits.grad)
