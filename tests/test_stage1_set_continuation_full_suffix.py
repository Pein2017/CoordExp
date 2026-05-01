from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.trainers.stage1_set_continuation.entry_trie import EntryTrieCandidate
from src.trainers.stage1_set_continuation.full_suffix import (
    FullSuffixTargetStep,
    build_recursive_entry_trie_steps,
    compute_full_suffix_loss,
    score_full_suffix_batch_padding_free_packed,
    score_full_suffix_batch_retained,
    score_full_suffix_retained,
)


class _TinySuffixModel(nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 5)
        self.proj = nn.Linear(5, vocab_size)
        self.calls: list[dict[str, int | tuple[int, ...]]] = []

    def forward(self, input_ids: torch.Tensor, **kwargs):
        logits_to_keep = int(kwargs.pop("logits_to_keep", 0) or 0)
        if kwargs:
            raise AssertionError(f"unexpected kwargs: {sorted(kwargs)}")
        hidden = self.embed(input_ids)
        if logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        self.calls.append(
            {
                "shape": tuple(int(dim) for dim in input_ids.shape),
                "logits_to_keep": logits_to_keep,
            }
        )
        return type("Output", (), {"logits": self.proj(hidden)})()


class _FixedSuffixLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor, *, crop_mode: str) -> None:
        super().__init__()
        self.logits = logits
        self.crop_mode = crop_mode
        self.calls: list[dict[str, int]] = []

    def forward(self, input_ids: torch.Tensor, **kwargs):
        logits_to_keep = int(kwargs.pop("logits_to_keep", 0) or 0)
        if kwargs:
            raise AssertionError(f"unexpected kwargs: {sorted(kwargs)}")
        if self.crop_mode == "respect":
            logits = self.logits[:, -logits_to_keep:, :]
        elif self.crop_mode == "ignore":
            logits = self.logits
        elif self.crop_mode == "short":
            logits = self.logits[:, -max(1, logits_to_keep - 1) :, :]
        elif self.crop_mode == "long":
            logits = self.logits[:, -(logits_to_keep + 1) :, :]
        else:
            raise AssertionError(f"unexpected crop_mode: {self.crop_mode}")
        self.calls.append({"logits_to_keep": logits_to_keep})
        return type("Output", (), {"logits": logits.to(device=input_ids.device)})()


class _PackedSuffixModel(nn.Module):
    def __init__(self, source: _TinySuffixModel) -> None:
        super().__init__()
        self.embed = copy.deepcopy(source.embed)
        self.proj = copy.deepcopy(source.proj)
        self.calls: list[dict[str, object]] = []

    def forward(self, input_ids: torch.Tensor, **kwargs):
        if "logits_to_keep" in kwargs:
            raise AssertionError("padding-free packed scorer must not crop logits")
        self.calls.append(
            {
                "shape": tuple(int(dim) for dim in input_ids.shape),
                "keys": tuple(sorted(kwargs)),
                "cu_seq_lens_q": kwargs["cu_seq_lens_q"].detach().cpu().tolist(),
                "cu_seq_lens_k": kwargs["cu_seq_lens_k"].detach().cpu().tolist(),
                "max_length_q": int(kwargs["max_length_q"]),
                "max_length_k": int(kwargs["max_length_k"]),
                "position_ids_shape": tuple(
                    int(dim) for dim in kwargs["position_ids"].shape
                ),
                "text_position_ids": kwargs["text_position_ids"]
                .detach()
                .cpu()
                .tolist(),
                "pack_num_samples": kwargs["pack_num_samples"]
                .detach()
                .cpu()
                .tolist(),
            }
        )
        hidden = self.embed(input_ids)
        return type("Output", (), {"logits": self.proj(hidden)})()


def _two_object_suffix_case() -> tuple[torch.Tensor, tuple[FullSuffixTargetStep, ...]]:
    labels = torch.tensor([[1, 2, 10, 30, 3]], dtype=torch.long)
    steps = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=2,
            teacher_token_id=10,
            token_type="text",
            phase="entry",
            targets=((10, 0.5), (11, 0.5)),
            active_object_count=2,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=3,
            teacher_token_id=30,
            token_type="coord",
            phase="entry",
            targets=((30, 1.0),),
            active_object_count=1,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=4,
            teacher_token_id=3,
            token_type="structural",
            phase="close",
            targets=((3, 1.0),),
            active_object_count=0,
        ),
    )
    return labels, steps


def test_recursive_entry_steps_update_remaining_after_each_emitted_object() -> None:
    entries = {
        0: EntryTrieCandidate(
            object_index=0,
            tokens=(10, 30),
            token_types=("text", "coord"),
        ),
        1: EntryTrieCandidate(
            object_index=1,
            tokens=(10, 31),
            token_types=("text", "coord"),
        ),
        2: EntryTrieCandidate(
            object_index=2,
            tokens=(11, 32),
            token_types=("text", "coord"),
        ),
    }

    steps = build_recursive_entry_trie_steps(
        entry_candidates=entries,
        suffix_order=(0, 1, 2),
        start_label_position=3,
    )

    assert [step.label_position for step in steps] == [3, 4, 5, 6, 7, 8]
    assert steps[0].is_branch
    assert steps[0].active_object_count == 3
    assert {target.token_id: target.probability for target in steps[0].targets} == {
        10: pytest.approx(2 / 3),
        11: pytest.approx(1 / 3),
    }
    assert steps[2].is_branch
    assert steps[2].active_object_count == 2
    assert {target.token_id: target.probability for target in steps[2].targets} == {
        10: pytest.approx(0.5),
        11: pytest.approx(0.5),
    }
    assert not steps[4].is_branch
    assert steps[4].active_object_count == 1


def test_full_suffix_loss_separates_branch_unique_boundary_close_and_eos_ce() -> None:
    logits = torch.full((1, 7, 40), -5.0)
    labels = torch.tensor([[1, 2, 10, 30, 3, 4, 5]], dtype=torch.long)
    steps = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=2,
            teacher_token_id=10,
            token_type="text",
            phase="entry",
            targets=((10, 0.75), (11, 0.25)),
            active_object_count=4,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=3,
            teacher_token_id=30,
            token_type="coord",
            phase="entry",
            targets=((30, 1.0),),
            active_object_count=3,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=4,
            teacher_token_id=3,
            token_type="structural",
            phase="boundary",
            targets=((3, 1.0),),
            active_object_count=0,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=5,
            teacher_token_id=4,
            token_type="structural",
            phase="close",
            targets=((4, 1.0),),
            active_object_count=0,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=6,
            teacher_token_id=5,
            token_type="other",
            phase="eos",
            targets=((5, 1.0),),
            active_object_count=0,
        ),
    )
    logits[0, 1, 10] = 4.0
    logits[0, 1, 11] = 3.0
    logits[0, 2, 30] = 5.0
    logits[0, 3, 3] = 5.0
    logits[0, 4, 4] = 5.0
    logits[0, 5, 5] = 5.0

    result = compute_full_suffix_loss(logits=logits, labels=labels, steps=steps)

    assert result.branch_tokens == 1
    assert result.unique_tokens == 1
    assert result.boundary_tokens == 1
    assert result.close_tokens == 1
    assert result.eos_tokens == 1
    assert result.loss > 0
    assert result.metrics["rmp/branch_nodes"] == pytest.approx(1.0)
    branch_log_probs = torch.log_softmax(logits[0, 1].float(), dim=-1)
    branch_valid_probs = torch.exp(branch_log_probs[[10, 11]])
    expected_valid_mass = float(branch_valid_probs.sum())
    expected_support = float(-torch.logsumexp(branch_log_probs[[10, 11]], dim=-1))
    expected_ce = float(
        -(torch.tensor([0.75, 0.25]) * branch_log_probs[[10, 11]]).sum()
    )
    expected_balance = expected_ce - expected_support
    assert result.branch_support_loss.item() == pytest.approx(expected_support)
    assert result.branch_balance_loss.item() == pytest.approx(expected_balance)
    assert result.metrics["loss/rmp_branch_support"] == pytest.approx(expected_support)
    assert result.metrics["loss/rmp_branch_balance"] == pytest.approx(expected_balance)
    assert result.metrics["loss/rmp_branch_total"] == pytest.approx(
        float(result.branch_loss.detach())
    )
    assert result.metrics["loss/rmp_branch_ce"] == pytest.approx(expected_ce)
    assert result.metrics["rmp/valid_child_mass_mean"] == pytest.approx(
        expected_valid_mass
    )
    assert result.metrics["rmp/valid_child_mass_min"] == pytest.approx(
        expected_valid_mass
    )
    assert result.metrics["rmp/valid_child_mass_p10"] == pytest.approx(
        expected_valid_mass
    )
    assert result.metrics["rmp/valid_child_mass_p50"] == pytest.approx(
        expected_valid_mass
    )
    assert result.metrics["rmp/valid_child_mass_p90"] == pytest.approx(
        expected_valid_mass
    )
    assert result.metrics["rmp/valid_child_mass_desc_text"] == pytest.approx(
        expected_valid_mass
    )
    assert result.metrics["rmp/valid_child_mass_coord"] == pytest.approx(0.0)
    assert result.metrics["rmp/valid_child_mass_structural"] == pytest.approx(0.0)
    assert result.metrics["rmp/valid_child_mass_other"] == pytest.approx(0.0)
    assert result.metrics["loss/rmp_unique_ce"] == pytest.approx(
        float(result.unique_loss.detach())
    )
    assert result.metrics["loss/rmp_boundary_ce"] == pytest.approx(
        float(result.boundary_loss.detach())
    )
    assert result.metrics["loss/rmp_close_ce"] == pytest.approx(
        float(result.close_loss.detach())
    )
    assert result.metrics["loss/rmp_eos_ce"] == pytest.approx(
        float(result.eos_loss.detach())
    )


def test_full_suffix_ce_mode_uses_hard_ce_at_entry_branch_positions() -> None:
    logits = torch.zeros((1, 4, 16))
    labels = torch.tensor([[1, 2, 10, 30]], dtype=torch.long)
    steps = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=2,
            teacher_token_id=10,
            token_type="text",
            phase="entry",
            targets=((10, 0.5), (11, 0.5)),
            active_object_count=2,
        ),
    )
    logits[0, 1, 10] = 2.0
    logits[0, 1, 11] = 5.0

    rmp = compute_full_suffix_loss(
        logits=logits,
        labels=labels,
        steps=steps,
        entry_trie_mp=True,
    )
    hard = compute_full_suffix_loss(
        logits=logits,
        labels=labels,
        steps=steps,
        entry_trie_mp=False,
    )

    assert rmp.branch_tokens == 1
    assert hard.branch_tokens == 0
    assert hard.unique_tokens == 1
    assert hard.loss > rmp.loss


def test_entry_trie_rmp_branch_support_weight_increases_total_without_changing_ce() -> (
    None
):
    logits = torch.zeros((1, 4, 16))
    labels = torch.tensor([[1, 2, 10, 30]], dtype=torch.long)
    steps = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=2,
            teacher_token_id=10,
            token_type="text",
            phase="entry",
            targets=((10, 0.5), (11, 0.5)),
            active_object_count=2,
        ),
    )
    logits[0, 1, 10] = 2.0
    logits[0, 1, 11] = 5.0

    baseline = compute_full_suffix_loss(
        logits=logits,
        labels=labels,
        steps=steps,
        entry_trie_mp=True,
    )
    support_weighted = compute_full_suffix_loss(
        logits=logits,
        labels=labels,
        steps=steps,
        entry_trie_mp=True,
        branch_support_weight=2.0,
        branch_balance_weight=1.0,
    )

    assert support_weighted.metrics["loss/rmp_branch_ce"] == pytest.approx(
        baseline.metrics["loss/rmp_branch_ce"]
    )
    assert support_weighted.branch_support_loss.item() == pytest.approx(
        baseline.branch_support_loss.item()
    )
    assert support_weighted.branch_balance_loss.item() == pytest.approx(
        baseline.branch_balance_loss.item()
    )
    expected_total = 2.0 * float(baseline.branch_support_loss.detach()) + float(
        baseline.branch_balance_loss.detach()
    )
    assert support_weighted.metrics["loss/rmp_branch_total"] == pytest.approx(
        expected_total
    )
    assert support_weighted.loss.item() == pytest.approx(expected_total)
    assert support_weighted.loss > baseline.loss


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_entry_trie_rmp_default_matches_fp32_soft_ce_for_low_precision_logits(
    dtype: torch.dtype,
) -> None:
    logits = torch.full((1, 4, 16), -3.75, dtype=dtype)
    labels = torch.tensor([[1, 2, 10, 30]], dtype=torch.long)
    steps = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=2,
            teacher_token_id=10,
            token_type="text",
            phase="entry",
            targets=((10, 0.2), (11, 0.3), (12, 0.5)),
            active_object_count=3,
        ),
    )
    logits[0, 1, 0] = 6.25
    logits[0, 1, 10] = -1.125
    logits[0, 1, 11] = 2.375
    logits[0, 1, 12] = 0.625

    result = compute_full_suffix_loss(
        logits=logits,
        labels=labels,
        steps=steps,
        entry_trie_mp=True,
        branch_support_weight=1.0,
        branch_balance_weight=1.0,
    )

    log_probs = torch.log_softmax(logits[0, 1].float(), dim=-1)
    valid_log_probs = log_probs[torch.tensor([10, 11, 12])]
    probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    expected_support = -torch.logsumexp(valid_log_probs, dim=-1)
    expected_ce = -(probabilities * valid_log_probs).sum()
    expected_balance = expected_ce - expected_support
    expected_valid_mass = torch.exp(torch.logsumexp(valid_log_probs, dim=-1))

    assert result.branch_loss.dtype is torch.float32
    assert result.branch_ce_loss.dtype is torch.float32
    assert result.branch_support_loss.dtype is torch.float32
    assert result.branch_balance_loss.dtype is torch.float32
    assert result.branch_support_loss.item() == pytest.approx(
        float(expected_support), abs=1e-6
    )
    assert result.branch_balance_loss.item() == pytest.approx(
        float(expected_balance), abs=1e-6
    )
    assert result.branch_ce_loss.item() == pytest.approx(float(expected_ce), abs=1e-6)
    assert result.branch_loss.item() == pytest.approx(float(expected_ce), abs=1e-6)
    assert result.metrics["loss/rmp_branch_total"] == pytest.approx(
        result.metrics["loss/rmp_branch_ce"], abs=1e-6
    )
    assert result.metrics["rmp/valid_child_mass_mean"] == pytest.approx(
        float(expected_valid_mass), abs=1e-6
    )


def test_entry_trie_rmp_default_gradient_matches_fp32_soft_ce() -> None:
    logits = torch.full((1, 4, 16), -3.0, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        logits[0, 1, 0] = 4.5
        logits[0, 1, 10] = -1.0
        logits[0, 1, 11] = 2.0
        logits[0, 1, 12] = 0.25
    labels = torch.tensor([[1, 2, 10, 30]], dtype=torch.long)
    steps = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=2,
            teacher_token_id=10,
            token_type="text",
            phase="entry",
            targets=((10, 0.2), (11, 0.3), (12, 0.5)),
            active_object_count=3,
        ),
    )

    result = compute_full_suffix_loss(
        logits=logits,
        labels=labels,
        steps=steps,
        entry_trie_mp=True,
        branch_support_weight=1.0,
        branch_balance_weight=1.0,
    )
    result.loss.backward()
    actual_grad = logits.grad.detach().clone()

    manual_logits = logits.detach().clone().requires_grad_(True)
    log_probs = torch.log_softmax(manual_logits[0, 1].float(), dim=-1)
    probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    expected_loss = -(probabilities * log_probs[torch.tensor([10, 11, 12])]).sum()
    expected_loss.backward()

    assert torch.allclose(actual_grad, manual_logits.grad, atol=1e-7, rtol=1e-7)


def test_full_suffix_smart_batch_matches_serial_and_excludes_prefix_logits() -> None:
    torch.manual_seed(29)
    model_serial = _TinySuffixModel()
    model_batch = copy.deepcopy(model_serial)
    labels_a = torch.tensor([[1, 2, 10, 30, 3]], dtype=torch.long)
    labels_b = torch.tensor([[4, 5, 6, 11, 31, 3]], dtype=torch.long)
    steps_a = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=2,
            teacher_token_id=10,
            token_type="text",
            phase="entry",
            targets=((10, 0.5), (11, 0.5)),
            active_object_count=2,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=3,
            teacher_token_id=30,
            token_type="coord",
            phase="entry",
            targets=((30, 1.0),),
            active_object_count=1,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=4,
            teacher_token_id=3,
            token_type="structural",
            phase="close",
            targets=((3, 1.0),),
            active_object_count=0,
        ),
    )
    steps_b = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=3,
            teacher_token_id=11,
            token_type="text",
            phase="entry",
            targets=((11, 1.0),),
            active_object_count=1,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=4,
            teacher_token_id=31,
            token_type="coord",
            phase="entry",
            targets=((31, 1.0),),
            active_object_count=1,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=5,
            teacher_token_id=3,
            token_type="structural",
            phase="close",
            targets=((3, 1.0),),
            active_object_count=0,
        ),
    )

    serial = [
        score_full_suffix_retained(
            model=model_serial,
            model_inputs={"input_ids": labels_a.clone()},
            labels=labels_a,
            steps=steps_a,
            logits_mode="supervised_suffix",
        ),
        score_full_suffix_retained(
            model=model_serial,
            model_inputs={"input_ids": labels_b.clone()},
            labels=labels_b,
            steps=steps_b,
            logits_mode="supervised_suffix",
        ),
    ]
    batched = score_full_suffix_batch_retained(
        model=model_batch,
        items=[
            {
                "model_inputs": {"input_ids": labels_a.clone()},
                "labels": labels_a,
                "steps": steps_a,
            },
            {
                "model_inputs": {"input_ids": labels_b.clone()},
                "labels": labels_b,
                "steps": steps_b,
            },
        ],
        logits_mode="supervised_suffix",
    )

    assert len(model_serial.calls) == 2
    assert len(model_batch.calls) == 1
    assert model_batch.calls[0]["logits_to_keep"] == 5
    for expected, actual in zip(serial, batched, strict=True):
        assert torch.allclose(expected.loss.detach(), actual.loss.detach(), atol=1e-6)
        assert torch.allclose(
            expected.branch_loss.detach(),
            actual.branch_loss.detach(),
            atol=1e-6,
        )
        assert torch.allclose(
            expected.unique_loss.detach(),
            actual.unique_loss.detach(),
            atol=1e-6,
        )


def test_padding_free_packed_full_suffix_scores_rows_in_one_forward_without_padding() -> (
    None
):
    torch.manual_seed(29)
    model_serial = _TinySuffixModel()
    model_packed = _PackedSuffixModel(model_serial)
    labels_a = torch.tensor([[1, 2, 10, 30, 3]], dtype=torch.long)
    labels_b = torch.tensor([[4, 5, 6, 11, 31, 3]], dtype=torch.long)
    steps_a = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=2,
            teacher_token_id=10,
            token_type="text",
            phase="entry",
            targets=((10, 0.5), (11, 0.5)),
            active_object_count=2,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=3,
            teacher_token_id=30,
            token_type="coord",
            phase="entry",
            targets=((30, 1.0),),
            active_object_count=1,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=4,
            teacher_token_id=3,
            token_type="structural",
            phase="close",
            targets=((3, 1.0),),
            active_object_count=0,
        ),
    )
    steps_b = (
        FullSuffixTargetStep(
            row_index=0,
            label_position=3,
            teacher_token_id=11,
            token_type="text",
            phase="entry",
            targets=((11, 1.0),),
            active_object_count=1,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=4,
            teacher_token_id=31,
            token_type="coord",
            phase="entry",
            targets=((31, 1.0),),
            active_object_count=1,
        ),
        FullSuffixTargetStep(
            row_index=0,
            label_position=5,
            teacher_token_id=3,
            token_type="structural",
            phase="close",
            targets=((3, 1.0),),
            active_object_count=0,
        ),
    )

    serial = [
        score_full_suffix_retained(
            model=model_serial,
            model_inputs={"input_ids": labels_a.clone()},
            labels=labels_a,
            steps=steps_a,
            logits_mode="full",
        ),
        score_full_suffix_retained(
            model=model_serial,
            model_inputs={"input_ids": labels_b.clone()},
            labels=labels_b,
            steps=steps_b,
            logits_mode="full",
        ),
    ]
    packed = score_full_suffix_batch_padding_free_packed(
        model=model_packed,
        items=[
            {
                "model_inputs": {
                    "input_ids": labels_a.clone(),
                    "attention_mask": torch.ones_like(labels_a),
                },
                "labels": labels_a,
                "steps": steps_a,
            },
            {
                "model_inputs": {
                    "input_ids": labels_b.clone(),
                    "attention_mask": torch.ones_like(labels_b),
                },
                "labels": labels_b,
                "steps": steps_b,
            },
        ],
        logits_mode="full",
    )

    assert len(model_serial.calls) == 2
    assert len(model_packed.calls) == 1
    assert model_packed.calls[0]["shape"] == (1, 11)
    assert model_packed.calls[0]["cu_seq_lens_q"] == [0, 5, 11]
    assert model_packed.calls[0]["cu_seq_lens_k"] == [0, 5, 11]
    assert model_packed.calls[0]["max_length_q"] == 6
    assert model_packed.calls[0]["max_length_k"] == 6
    assert model_packed.calls[0]["position_ids_shape"] == (3, 1, 11)
    assert model_packed.calls[0]["text_position_ids"] == [
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]
    ]
    assert model_packed.calls[0]["pack_num_samples"] == [2]
    assert "attention_mask" not in model_packed.calls[0]["keys"]
    for expected, actual in zip(serial, packed, strict=True):
        assert torch.allclose(expected.loss.detach(), actual.loss.detach(), atol=1e-6)
        assert torch.allclose(
            expected.branch_loss.detach(),
            actual.branch_loss.detach(),
            atol=1e-6,
        )
        assert torch.allclose(
            expected.unique_loss.detach(),
            actual.unique_loss.detach(),
            atol=1e-6,
        )


@pytest.mark.parametrize("crop_mode", ("ignore", "short", "long"))
def test_retained_supervised_suffix_rejects_misaligned_cropped_logits(
    crop_mode: str,
) -> None:
    labels, steps = _two_object_suffix_case()
    logits = torch.zeros((1, labels.shape[-1], 32))
    model = _FixedSuffixLogitsModel(logits, crop_mode=crop_mode)

    with pytest.raises(
        ValueError,
        match="supervised_suffix.*logits_to_keep.*expected.*actual",
    ):
        score_full_suffix_retained(
            model=model,
            model_inputs={"input_ids": labels.clone()},
            labels=labels,
            steps=steps,
            logits_mode="supervised_suffix",
        )


@pytest.mark.parametrize("crop_mode", ("ignore", "short", "long"))
def test_batched_retained_supervised_suffix_rejects_misaligned_cropped_logits(
    crop_mode: str,
) -> None:
    labels, steps = _two_object_suffix_case()
    logits = torch.zeros((2, labels.shape[-1], 32))
    model = _FixedSuffixLogitsModel(logits, crop_mode=crop_mode)

    with pytest.raises(
        ValueError,
        match="supervised_suffix.*logits_to_keep.*expected.*actual",
    ):
        score_full_suffix_batch_retained(
            model=model,
            items=[
                {
                    "model_inputs": {"input_ids": labels.clone()},
                    "labels": labels,
                    "steps": steps,
                },
                {
                    "model_inputs": {"input_ids": labels.clone()},
                    "labels": labels,
                    "steps": steps,
                },
            ],
            logits_mode="supervised_suffix",
        )


def test_retained_supervised_suffix_uses_shifted_suffix_logits_not_prefix_logits() -> (
    None
):
    labels, steps = _two_object_suffix_case()
    logits = torch.full((1, labels.shape[-1], 32), -30.0)
    # Original prefix position 0 is adversarially confident in invalid tokens. If
    # shifted scoring ever uses it, the branch and unique CE losses explode.
    logits[0, 0, 0] = 30.0
    logits[0, 0, 1] = 29.0
    # Cropped suffix position 0 corresponds to original position 1 and predicts
    # label position 2.
    logits[0, 1, 10] = 30.0
    logits[0, 1, 11] = 30.0
    logits[0, 2, 30] = 30.0
    logits[0, 3, 3] = 30.0
    model = _FixedSuffixLogitsModel(logits, crop_mode="respect")

    result = score_full_suffix_retained(
        model=model,
        model_inputs={"input_ids": labels.clone()},
        labels=labels,
        steps=steps,
        logits_mode="supervised_suffix",
    )

    assert model.calls == [{"logits_to_keep": 4}]
    assert result.branch_support_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert result.branch_balance_loss.item() == pytest.approx(
        float(torch.log(torch.tensor(2.0))), abs=1e-6
    )
    assert result.unique_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert result.close_loss.item() == pytest.approx(0.0, abs=1e-6)
