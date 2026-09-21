from types import SimpleNamespace

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.inference.coordinate_output_norm import CoordinateOutputNorm
from src.qwen.special_token_embeddings import SelectedDeltaOutputHead, SpecialTokenSelection


def test_output_norm_matches_effective_rows_and_preserves_other_logits():
    torch.manual_seed(17)
    base = torch.nn.Linear(4, 1002, bias=False)
    selection = SpecialTokenSelection(token_strings=[str(i) for i in range(1000)], token_ids=range(1, 1001))
    delta = torch.nn.Parameter(torch.randn(1000, 4))
    head = SelectedDeltaOutputHead(base, selection, delta)
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda s: int(s[8:-2]) + 1)
    processor = CoordinateOutputNorm(SimpleNamespace(get_output_embeddings=lambda: head), tokenizer)
    hidden = torch.randn(2, 4)
    scores = head(hidden).detach()
    original = scores.clone()
    actual = processor(None, scores)
    rows = base.weight[1:1001].detach() + delta.detach()
    norms = rows.double().norm(dim=1)
    expected = hidden.double() @ (rows.double() * (norms.median() / norms)[:, None]).T
    torch.testing.assert_close(actual[:, 1:1001].double(), expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(actual[:, [0, 1001]], original[:, [0, 1001]])
    assert torch.equal(scores, original)
    assert not torch.equal(actual, original)
    processor.factors.fill_(1)
    assert torch.equal(processor(None, scores), original)
    with torch.no_grad():
        delta[0].copy_(-base.weight[1])
    with pytest.raises(RuntimeContractError, match="finite and positive"):
        CoordinateOutputNorm(SimpleNamespace(get_output_embeddings=lambda: head), tokenizer)
