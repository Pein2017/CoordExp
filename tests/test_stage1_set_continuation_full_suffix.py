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
    assert result.metrics["loss/rmp_branch_ce"] == pytest.approx(
        float(result.branch_loss.detach())
    )
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
