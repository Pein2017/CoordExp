import torch

from src.coord_tokens.loss import topk_expectation_decode


def test_topk_expectation_decode_handles_non_finite():
    vocab = 1000
    logits = torch.full((1, 3, vocab), -1e4)
    logits[0, 0, 0] = float("nan")
    logits[0, 1, 1] = float("inf")
    logits[0, 2, 2] = -float("inf")
    coord_ids = list(range(vocab))

    out = topk_expectation_decode(logits, coord_ids, top_k=0.1, temperature=1.0)
    assert torch.isfinite(out).all().item()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
