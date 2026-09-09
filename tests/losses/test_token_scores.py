import pytest
import torch
import torch.nn.functional as F

from src.common.errors import LossContractError
from src.losses.token_scores import aligned_token_logprobs


@pytest.mark.parametrize("objective", ["ce", "rloo", "coordinate", "full_action"])
def test_aligned_scores_preserve_caller_objectives_and_gradients(objective):
    logits = torch.randn(6, 11, generator=torch.Generator().manual_seed(29), dtype=torch.float64, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    targets = torch.tensor([3, 1, 8, 4, 2, 9])
    chosen = aligned_token_logprobs(logits, targets)
    expected = reference.float().log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    # Preserve the existing eight-rank global formulas, including sum vs mean.
    if objective == "ce":
        loss, old_loss = -chosen.mean() * (8 / 256), -expected.mean() * (8 / 256)
    elif objective == "rloo":
        loss, old_loss = -1.25 * chosen.sum() * (8 / (256 * 4)), -1.25 * expected.sum() * (8 / (256 * 4))
    elif objective == "coordinate":
        loss = -chosen[[1, 2, 3, 4]].sum() * -0.75 * (8 / (16 * 4))
        old_loss = -expected[[1, 2, 3, 4]].sum() * -0.75 * (8 / (16 * 4))
    else:
        loss, old_loss = -chosen.sum() * -0.75 * (8 / (16 * 4)), -expected.sum() * -0.75 * (8 / (16 * 4))
    loss.backward()
    old_loss.backward()
    assert chosen.shape == (6,) and chosen.dtype == torch.float32
    assert chosen.device == logits.device
    torch.testing.assert_close(chosen, expected)
    torch.testing.assert_close(-chosen, F.cross_entropy(logits.float(), targets, reduction="none"))
    torch.testing.assert_close(loss, old_loss)
    torch.testing.assert_close(logits.grad, reference.grad)


def test_empty_selection_preserves_an_empty_gradient_path():
    logits = torch.empty(0, 7, requires_grad=True)
    result = aligned_token_logprobs(logits, torch.empty(0, dtype=torch.long))
    assert result.shape == (0,)
    result.sum().backward()
    assert logits.grad is not None and logits.grad.shape == logits.shape


@pytest.mark.parametrize("logits,targets", [
    (torch.zeros(1, 2, 3), torch.tensor([1, 2])),
    (torch.zeros(2, 3), torch.tensor([[1, 2]])),
    (torch.zeros(2, 3), torch.tensor([1])),
    (torch.zeros(2, 3), torch.tensor([1.0, 2.0])),
    (torch.zeros(2, 3, dtype=torch.long), torch.tensor([1, 2])),
    (torch.zeros(2, 0), torch.tensor([0, 0])),
])
def test_aligned_scores_reject_invalid_tensor_contract(logits, targets):
    with pytest.raises(LossContractError):
        aligned_token_logprobs(logits, targets)


@pytest.mark.parametrize("targets", [torch.tensor([-1, 0]), torch.tensor([0, 3])])
def test_aligned_scores_reject_out_of_vocabulary_targets(targets):
    with pytest.raises((RuntimeError, IndexError)):
        aligned_token_logprobs(torch.zeros(2, 3), targets)
