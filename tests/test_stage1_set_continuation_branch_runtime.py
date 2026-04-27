from __future__ import annotations

import copy

import pytest
import torch
from torch import nn


class _TinyBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.proj = nn.Linear(4, 16)
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, **kwargs):
        self.calls += 1
        hidden = self.embed(input_ids)
        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        if kwargs:
            raise AssertionError(f"unexpected kwargs: {sorted(kwargs)}")
        if isinstance(logits_to_keep, int) and int(logits_to_keep) > 0:
            hidden = hidden[:, -int(logits_to_keep) :, :]
        logits = self.proj(hidden)
        return type("Output", (), {"logits": logits})()


def test_checkpointed_exact_matches_retained_graph_gradient_on_tiny_branch() -> None:
    from src.trainers.stage1_set_continuation.branch_scorer import (
        score_tensor_checkpointed,
        score_tensor_retained,
    )

    torch.manual_seed(11)
    model_a = _TinyBranchModel()
    model_b = copy.deepcopy(model_a)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = input_ids.clone()
    candidate_mask = torch.tensor([[False, True, True, True]])
    coord_mask = torch.tensor([[False, False, False, False]])
    coord_token_ids = torch.tensor([10, 11, 12], dtype=torch.long)

    retained = score_tensor_retained(
        model=model_a,
        model_inputs={"input_ids": input_ids},
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )
    checkpointed = score_tensor_checkpointed(
        model=model_b,
        model_inputs={"input_ids": input_ids},
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
        use_reentrant=False,
        preserve_rng_state=True,
    )

    (-retained.score).backward()
    (-checkpointed.score).backward()

    assert torch.allclose(
        retained.score.detach(), checkpointed.score.detach(), atol=1e-6
    )
    assert torch.allclose(
        retained.coord_score.detach(), checkpointed.coord_score.detach(), atol=1e-6
    )
    assert torch.allclose(
        retained.non_coord_score.detach(),
        checkpointed.non_coord_score.detach(),
        atol=1e-6,
    )
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert param_a.grad is not None
        assert param_b.grad is not None
        assert torch.allclose(param_a.grad, param_b.grad, atol=1e-6)
    assert model_b.calls >= 2


def test_supervised_suffix_candidate_score_matches_full_logits() -> None:
    from src.trainers.stage1_set_continuation.branch_scorer import (
        score_tensor_retained,
    )

    torch.manual_seed(17)
    model_a = _TinyBranchModel()
    model_b = copy.deepcopy(model_a)
    input_ids = torch.tensor([[1, 2, 3, 10, 4, 11]], dtype=torch.long)
    labels = input_ids.clone()
    candidate_mask = torch.tensor([[False, False, False, True, True, True]])
    coord_mask = torch.tensor([[False, False, False, True, False, True]])
    coord_token_ids = torch.tensor([10, 11, 12], dtype=torch.long)

    full = score_tensor_retained(
        model=model_a,
        model_inputs={"input_ids": input_ids},
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )
    suffix = score_tensor_retained(
        model=model_b,
        model_inputs={"input_ids": input_ids},
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
        logits_mode="supervised_suffix",
    )

    assert torch.allclose(full.score.detach(), suffix.score.detach(), atol=1e-6)
    assert torch.allclose(
        full.coord_score.detach(), suffix.coord_score.detach(), atol=1e-6
    )
    assert torch.allclose(
        full.non_coord_score.detach(),
        suffix.non_coord_score.detach(),
        atol=1e-6,
    )
    assert suffix.tokens == full.tokens
    assert suffix.coord_tokens == full.coord_tokens
    assert model_b.calls == 1


def test_supervised_suffix_start_accepts_cpu_mask_after_prepared_labels_move() -> None:
    from src.trainers.stage1_set_continuation.branch_scorer import (
        supervised_suffix_start,
    )

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to reproduce prepared-label/mask device split")

    labels = torch.tensor([[1, 2, 3, 10, 4, 11]], device="cuda")
    candidate_mask = torch.tensor([[False, False, False, True, True, True]])

    assert (
        supervised_suffix_start(
            labels=labels,
            supervised_label_mask=candidate_mask,
        )
        == 2
    )


def test_smart_batched_suffix_candidate_scores_match_serial_retained_graph() -> None:
    from src.trainers.stage1_set_continuation.branch_scorer import (
        TensorBranchScoreInput,
        score_tensor_batch_retained,
        score_tensor_retained,
    )

    torch.manual_seed(19)
    model_serial = _TinyBranchModel()
    model_batch = copy.deepcopy(model_serial)
    coord_token_ids = torch.tensor([10, 11, 12], dtype=torch.long)

    branch_a = TensorBranchScoreInput(
        model_inputs={"input_ids": torch.tensor([[1, 2, 3, 10, 4, 11]])},
        labels=torch.tensor([[1, 2, 3, 10, 4, 11]]),
        candidate_entry_label_mask=torch.tensor(
            [[False, False, False, True, True, True]]
        ),
        coord_label_mask=torch.tensor([[False, False, False, True, False, True]]),
    )
    branch_b = TensorBranchScoreInput(
        model_inputs={"input_ids": torch.tensor([[5, 6, 7, 12, 8, 10]])},
        labels=torch.tensor([[5, 6, 7, 12, 8, 10]]),
        candidate_entry_label_mask=torch.tensor(
            [[False, False, False, False, True, True]]
        ),
        coord_label_mask=torch.tensor([[False, False, False, False, False, True]]),
    )

    serial = [
        score_tensor_retained(
            model=model_serial,
            model_inputs=item.model_inputs,
            labels=item.labels,
            candidate_entry_label_mask=item.candidate_entry_label_mask,
            coord_label_mask=item.coord_label_mask,
            coord_token_ids=coord_token_ids,
            logits_mode="supervised_suffix",
        )
        for item in (branch_a, branch_b)
    ]
    batched = score_tensor_batch_retained(
        model=model_batch,
        items=[branch_a, branch_b],
        coord_token_ids=coord_token_ids,
        logits_mode="supervised_suffix",
    )

    assert model_serial.calls == 2
    assert model_batch.calls == 1
    for expected, actual in zip(serial, batched, strict=True):
        assert torch.allclose(expected.score.detach(), actual.score.detach(), atol=1e-6)
        assert torch.allclose(
            expected.coord_score.detach(), actual.coord_score.detach(), atol=1e-6
        )
        assert torch.allclose(
            expected.non_coord_score.detach(),
            actual.non_coord_score.detach(),
            atol=1e-6,
        )
        assert actual.tokens == expected.tokens
        assert actual.coord_tokens == expected.coord_tokens


def test_supervised_suffix_close_losses_match_full_logits() -> None:
    from src.trainers.stage1_set_continuation.branch_scorer import (
        crop_tensors_for_logits,
        supervised_suffix_start,
    )
    from src.trainers.stage1_set_continuation.losses import (
        compute_close_sequence_nll,
        compute_close_start_nll,
    )

    torch.manual_seed(23)
    model_full = _TinyBranchModel()
    model_suffix = copy.deepcopy(model_full)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    labels = input_ids.clone()
    close_start_mask = torch.tensor([[False, False, False, False, True, False]])
    close_sequence_mask = torch.tensor([[False, False, False, False, True, True]])

    full_logits = model_full(input_ids=input_ids).logits
    full_start = compute_close_start_nll(
        logits=full_logits,
        labels=labels,
        structural_close_start_mask=close_start_mask,
    )
    full_sequence = compute_close_sequence_nll(
        logits=full_logits,
        labels=labels,
        structural_close_sequence_mask=close_sequence_mask,
    )

    suffix_start = supervised_suffix_start(
        labels=labels,
        supervised_label_mask=close_sequence_mask,
    )
    logits_to_keep = input_ids.shape[-1] - suffix_start
    suffix_logits = model_suffix(
        input_ids=input_ids, logits_to_keep=logits_to_keep
    ).logits
    (
        cropped_labels,
        cropped_start_mask,
        cropped_sequence_mask,
        _schema_open_mask,
        _json_structural_mask,
    ) = crop_tensors_for_logits(
        suffix_start=suffix_start,
        labels=labels,
        candidate_entry_label_mask=close_start_mask,
        coord_label_mask=close_sequence_mask,
    )

    suffix_start_loss = compute_close_start_nll(
        logits=suffix_logits,
        labels=cropped_labels,
        structural_close_start_mask=cropped_start_mask,
    )
    suffix_sequence_loss = compute_close_sequence_nll(
        logits=suffix_logits,
        labels=cropped_labels,
        structural_close_sequence_mask=cropped_sequence_mask,
    )

    assert torch.allclose(full_start.detach(), suffix_start_loss.detach(), atol=1e-6)
    assert torch.allclose(
        full_sequence.detach(), suffix_sequence_loss.detach(), atol=1e-6
    )
