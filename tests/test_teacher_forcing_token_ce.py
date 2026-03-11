from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.trainers.teacher_forcing.contracts import (
    PipelineModuleSpec,
    TeacherForcingContext,
)
from src.trainers.teacher_forcing.modules.token_ce import run_token_ce_module


def test_token_ce_chunked_matches_dense_reference() -> None:
    torch.manual_seed(7)

    bsz = 1
    seq_len = 9000
    vocab = 64

    input_ids = torch.randint(5, vocab, (bsz, seq_len), dtype=torch.long)
    logits = torch.randn(bsz, seq_len, vocab, dtype=torch.float32, requires_grad=True)

    meta = [
        {
            "prompt_len": 16,
            "prefix_len": 0,
            "train_len": seq_len - 16,
            "tail_ignore_pos": [],
            "tail_desc_pos": [],
            "tail_closure_pos": [],
            "prefix_struct_pos": [],
            "drop_invalid_total": 0,
        }
    ]

    context = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits,
        meta=meta,
        coord_token_ids=[],
        temperature=1.0,
        decode_mode="greedy",
    )
    spec = PipelineModuleSpec(
        name="token_ce",
        enabled=True,
        weight=1.0,
        channels=("A", "B"),
        config={},
    )

    out = run_token_ce_module(context=context, spec=spec)
    assert out.loss.requires_grad

    labels_masked = out.state["labels_masked"]
    weights_masked = out.state["weights_masked"]

    logits_next = logits[:, :-1, :]
    labels_next = labels_masked[:, 1:]
    weights_next = weights_masked[:, 1:]
    per_tok = F.cross_entropy(
        logits_next.reshape(-1, vocab),
        labels_next.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(bsz, -1)
    denom = weights_next.sum().clamp(min=1e-6)
    dense_ref = (per_tok * weights_next).sum() / denom

    assert torch.allclose(out.loss, dense_ref, atol=1e-5, rtol=1e-5)
    assert out.metrics["loss/token_ce"] == pytest.approx(
        float(dense_ref.detach().cpu().item()),
        abs=1e-5,
    )

    out.loss.backward()
    assert logits.grad is not None


def test_token_ce_tail_desc_weights_scale_desc_tokens() -> None:
    input_ids = torch.tensor([[9, 1, 2, 3, 4]], dtype=torch.long)
    logits = torch.zeros((1, 5, 32), dtype=torch.float32, requires_grad=True)

    meta = [
        {
            "prompt_len": 1,
            "prefix_len": 0,
            "train_len": 4,
            "tail_ignore_pos": [],
            "tail_desc_pos": [0],
            "tail_desc_weights": [2.0, 1.0, 1.0, 1.0],
            "tail_closure_pos": [],
            "prefix_struct_pos": [],
            "drop_invalid_total": 0,
        }
    ]

    context = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits,
        meta=meta,
        coord_token_ids=[],
        temperature=1.0,
        decode_mode="greedy",
    )
    spec = PipelineModuleSpec(
        name="token_ce",
        enabled=True,
        weight=1.0,
        channels=("B",),
        config={"rollout_fn_desc_weight": 1.0},
    )

    out = run_token_ce_module(context=context, spec=spec)
    weights = out.state["weights_masked"]
    # shifted by CE next-token convention: desc at tail rel=0 corresponds to position prompt+0
    assert float(weights[0, 1].item()) == pytest.approx(2.0)
