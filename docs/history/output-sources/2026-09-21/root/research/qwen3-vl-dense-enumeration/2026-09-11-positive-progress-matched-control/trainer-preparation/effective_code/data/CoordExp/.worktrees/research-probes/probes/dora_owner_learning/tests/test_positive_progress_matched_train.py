from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from probes.dora_owner_learning import positive_progress_matched_train as control
from probes.dora_owner_learning import repeat_recovery_train as old_train


def _margin_case() -> dict:
    return {
        "key": "normal-1",
        "eligible_positions": [0, 1],
        "original_kl_positions": [0, 1],
        "target_ids": [0, 1],
        "source_margins": [2.0, 0.4],
        "floors": [0.1, 0.1],
    }


class _Replay:
    def __init__(self) -> None:
        self.inputs = {}
        self.target_ids = torch.tensor([0, 1], dtype=torch.long)

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


def test_lead_accepted_selection_binds_all_17_archived_states() -> None:
    selection, oracles = control.validate_selection(control.SELECTION)

    assert [row["step"] for row in oracles] == list(range(1, 18))
    assert selection["selected_scalar_step"] == 18
    assert oracles[-1]["artifact"]["sha256"] == (
        "b9c24b7c3e8f82ef46a702273094967c8e4d76e5769d72a2fc504d4e116f12dc"
    )
    assert oracles[-1]["adapter_sha256"] == control.EXPECTED_FINAL_ADAPTER_SHA256
    assert oracles[-1]["optimizer_sha256"] == control.EXPECTED_FINAL_OPTIMIZER_SHA256
    assert oracles[-1]["gradient_sha256"] == control.EXPECTED_FINAL_GRADIENT_SHA256
    assert selection["selected"]["sum_positive_nll"] == control.EXPECTED_SUM_POSITIVE_NLL


def test_step_parity_fails_on_scalar_or_any_state_hash() -> None:
    _, oracles = control.validate_selection(control.SELECTION)
    oracle = oracles[0]
    observed = dict(
        step=1,
        observed_positive=oracle["positive_objective_before_update"],
        gradient_sha256=oracle["gradient_sha256"],
        adapter_sha256=oracle["adapter_sha256"],
        optimizer_sha256=oracle["optimizer_sha256"],
        oracle=oracle,
    )
    control.assert_step_parity(**observed)

    for key, value in (
        ("observed_positive", observed["observed_positive"] + 2e-6),
        ("gradient_sha256", "0" * 64),
        ("adapter_sha256", "0" * 64),
        ("optimizer_sha256", "0" * 64),
    ):
        changed = {**observed, key: value}
        with pytest.raises(ValueError, match="diverged from archived A at step 1"):
            control.assert_step_parity(**changed)


def test_protocol_is_one_fixed_no_sampling_d17_contract() -> None:
    protocol = control._protocol()

    assert protocol["arms"] == ["D"]
    assert protocol["updates"] == {"matched": 17}
    assert protocol["margin"]["training_weight"] == 0.0
    assert protocol["margin"]["diagnostic_only"] is True
    assert protocol["sampling"] is None
    assert protocol["fixed_bounds"] == {
        "model_loads": 8,
        "reference_forwards": 80,
        "source_reference_forwards": 80,
        "model_forwards": 1961,
        "image_forwards": 1961,
        "backwards": 1768,
        "synchronized_backwards": 136,
        "post_update_normal_readback_forwards": 56,
        "sampling_calls": 0,
    }


def test_d_weight0_normal_replay_is_exact_old_a_value_and_gradient(monkeypatch) -> None:
    replay = _Replay()
    monkeypatch.setattr(control, "prepare_replay", lambda *args, **kwargs: replay)
    monkeypatch.setattr(old_train, "prepare_replay", lambda *args, **kwargs: replay)
    logits = torch.tensor([[0.02, 0.0, -1.0], [0.0, 0.4, -1.0]])
    reference = torch.log_softmax(
        torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.4, -1.0]]), -1,
    )

    old_model, d_model = _Model(logits), _Model(logits)
    old_loss, _ = old_train.RepeatRecoveryScorer(old_model)(
        {}, [], [0, 1], kind="normal", reference_logp=reference, positions=[0, 1],
    )
    d_loss, stats = control.MarginPreservedScorer(
        d_model, margin_weight=0.0,
    )({}, [], [0, 1], kind="normal", reference_logp=reference,
      positions=[0, 1], margin=_margin_case())

    assert old_model.calls == d_model.calls == 1
    assert torch.equal(d_loss, old_loss)
    assert stats["margin_penalty"] == pytest.approx(0.08)
    assert stats["margin_scaled_loss"] == 0.0
    d_loss.backward()
    old_loss.backward()
    assert torch.equal(d_model.logits.grad, old_model.logits.grad)
    with pytest.raises(ValueError, match="must be zero"):
        control.MarginPreservedScorer(_Model(logits), margin_weight=10.0)


def test_d17_resource_envelope_is_fixed_and_selection_bound(tmp_path) -> None:
    limits = {
        "max_rank_seconds": 1500,
        "max_cuda_allocated_bytes": 24 * 1024**3,
        "max_cuda_reserved_bytes": 24 * 1024**3,
        "max_rss_bytes": 24 * 1024**3,
        "max_model_forwards_per_rank": 295,
        "max_image_forwards_per_rank": 295,
    }
    envelope = tmp_path / "envelope.json"
    envelope.write_text(json.dumps({
        "schema": control.RESOURCE_SCHEMA,
        "status": "lead_accepted",
        "arm": "D",
        "mode": "matched",
        "optimizer_updates": 17,
        "selection": {"path": str(control.SELECTION), "sha256": control.SELECTION_SHA256},
        "limits": limits,
    }), encoding="utf-8")
    packet = {"selection": {"path": str(control.SELECTION), "sha256": control.SELECTION_SHA256}}

    observed, reference = control._resource_limits(
        packet, arm="D", mode="matched", envelope_path=envelope,
    )
    assert observed == limits
    assert reference is not None and reference["path"] == str(envelope.resolve())
    with pytest.raises(ValueError, match="fixed D17 arm/mode"):
        control._resource_limits(packet, arm="D", mode="full", envelope_path=envelope)


def test_selected_positive_vector_sums_to_frozen_q() -> None:
    selection, _ = control.validate_selection(control.SELECTION)
    scores = selection["selected"]["positive_scores"]

    assert tuple(scores) == control.SELECTED
    assert sum(-row["sum_logprob"] for row in scores.values()) == control.EXPECTED_SUM_POSITIVE_NLL
    assert [row["argmax_target_tokens"] for row in scores.values()] == [11, 9, 9]
