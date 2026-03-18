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


def _run_simple_stop_signal_case(
    *,
    stop_cfg: dict | None,
    meta_override: dict | None = None,
):
    vocab = 32
    stop_id = 7
    cont_id = 8
    input_ids = torch.tensor([[1, 2, stop_id, 3, 4]], dtype=torch.long)
    logits = torch.full((1, input_ids.shape[1], vocab), -6.0, dtype=torch.float32)

    # Predict the next token correctly at all supervised positions.
    logits[0, 0, 2] = 4.0
    logits[0, 1, stop_id] = 2.0
    logits[0, 1, cont_id] = 0.0
    logits[0, 2, 3] = 4.0
    logits[0, 3, 4] = 4.0
    logits.requires_grad_(True)

    meta = [
        {
            "prompt_len": 1,
            "prefix_len": 0,
            "train_len": 4,
            "encoded_len": 5,
            "tail_ignore_pos": [],
            "tail_desc_pos": [],
            "tail_closure_pos": [2, 3],
            "prefix_struct_pos": [],
            "drop_invalid_total": 0,
            "stop_rel_pos": 1,
            "stop_token_id": stop_id,
            "continue_token_id": cont_id,
        }
    ]
    if meta_override is not None:
        meta = [dict(meta[0], **meta_override)]

    context = TeacherForcingContext(
        channel="A",
        registry_context="gt",
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
        config={"stop_signal_damping": dict(stop_cfg or {})},
    )
    out = run_token_ce_module(context=context, spec=spec)
    return out, logits, input_ids


def test_token_ce_stop_signal_metrics_follow_pair_local_branch_math() -> None:
    out, _logits, _input_ids = _run_simple_stop_signal_case(
        stop_cfg={
            "enabled": True,
            "min_weight": 0.2,
            "max_weight": 1.0,
            "branch_temperature": 2.0,
            "curve_gamma": 2.0,
            "detach_gate": True,
        }
    )

    p_stop = float(torch.sigmoid(torch.tensor(1.0)).item())
    p_cont = 1.0 - p_stop
    weight = 0.2 + 0.8 * (p_stop**2)

    assert out.metrics["stop_signal/p_stop_mean"] == pytest.approx(p_stop)
    assert out.metrics["stop_signal/p_cont_mean"] == pytest.approx(p_cont)
    assert out.metrics["stop_signal/margin_mean"] == pytest.approx(1.0)
    assert out.metrics["stop_signal/weight_mean"] == pytest.approx(weight)
    assert out.metrics["stop_signal/eligible_seq_count"] == pytest.approx(1.0)
    assert out.metrics["stop_signal/branch_count"] == pytest.approx(1.0)
    assert out.state["weights_masked"][0, 2].item() == pytest.approx(weight)
    assert "loss/token_ce_stop_signal" in out.metrics
    assert out.metrics["loss/stop_signal_ce"] > 0.0


def test_token_ce_stop_signal_detach_gate_changes_branch_gradients() -> None:
    out_detach, logits_detach, _ = _run_simple_stop_signal_case(
        stop_cfg={
            "enabled": True,
            "min_weight": 0.2,
            "max_weight": 1.0,
            "branch_temperature": 1.0,
            "curve_gamma": 2.0,
            "detach_gate": True,
        }
    )
    out_detach.loss.backward()
    grad_detach = logits_detach.grad.detach().clone()

    out_live, logits_live, _ = _run_simple_stop_signal_case(
        stop_cfg={
            "enabled": True,
            "min_weight": 0.2,
            "max_weight": 1.0,
            "branch_temperature": 1.0,
            "curve_gamma": 2.0,
            "detach_gate": False,
        }
    )
    out_live.loss.backward()
    grad_live = logits_live.grad.detach().clone()

    branch_row = 1
    assert not torch.allclose(
        grad_detach[0, branch_row, 7:9],
        grad_live[0, branch_row, 7:9],
        atol=1e-6,
        rtol=1e-6,
    )


def test_token_ce_stop_signal_disabled_falls_back_to_struct_ce() -> None:
    out, _logits, _input_ids = _run_simple_stop_signal_case(stop_cfg={"enabled": False})

    assert out.state["weights_masked"][0, 2].item() == pytest.approx(1.0)
    assert "loss/token_ce_stop_signal" not in out.metrics
    assert "stop_signal/p_stop_mean" not in out.metrics


def test_token_ce_stop_signal_missing_metadata_fails_fast_when_enabled() -> None:
    with pytest.raises(ValueError, match=r"missing stop-rel/token metadata"):
        _run_simple_stop_signal_case(
            stop_cfg={"enabled": True},
            meta_override={
                "stop_rel_pos": None,
                "stop_token_id": None,
                "continue_token_id": None,
            },
        )


def test_token_ce_stop_signal_weights_stay_segment_local_under_packing() -> None:
    stop1 = 7
    cont1 = 8
    stop2 = 9
    cont2 = 10
    input_ids = torch.tensor([[1, 2, stop1, 3, 4, 5, 6, stop2, 11, 12]], dtype=torch.long)
    vocab = 32
    logits = torch.full((1, input_ids.shape[1], vocab), -6.0, dtype=torch.float32)

    logits[0, 0, 2] = 4.0
    logits[0, 1, stop1] = 3.0
    logits[0, 2, 3] = 4.0
    logits[0, 3, 4] = 4.0

    logits[0, 5, 6] = 4.0
    logits[0, 6, stop2] = 3.0
    logits[0, 7, 11] = 4.0
    logits[0, 8, 12] = 4.0
    logits.requires_grad_(True)

    meta = [
        {
            "prompt_len": 1,
            "prefix_len": 0,
            "train_len": 4,
            "encoded_len": 5,
            "tail_ignore_pos": [],
            "tail_desc_pos": [],
            "tail_closure_pos": [2, 3],
            "prefix_struct_pos": [],
            "drop_invalid_total": 0,
            "stop_rel_pos": 1,
            "stop_token_id": stop1,
            "continue_token_id": cont1,
        },
        {
            "prompt_len": 1,
            "prefix_len": 0,
            "train_len": 4,
            "encoded_len": 5,
            "tail_ignore_pos": [],
            "tail_desc_pos": [],
            "tail_closure_pos": [2, 3],
            "prefix_struct_pos": [],
            "drop_invalid_total": 0,
            "stop_rel_pos": 1,
            "stop_token_id": stop2,
            "continue_token_id": cont2,
        },
    ]

    context = TeacherForcingContext(
        channel="A",
        registry_context="gt",
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
        config={
            "stop_signal_damping": {
                "enabled": True,
                "min_weight": 0.5,
                "max_weight": 0.5,
                "branch_temperature": 1.0,
                "curve_gamma": 1.0,
                "detach_gate": True,
            }
        },
    )

    out = run_token_ce_module(context=context, spec=spec)

    assert out.state["weights_masked"][0, 2].item() == pytest.approx(0.5)
    assert out.state["weights_masked"][0, 7].item() == pytest.approx(0.5)
    assert out.state["weights_masked"][0, 3].item() == pytest.approx(1.0)
    assert out.state["weights_masked"][0, 8].item() == pytest.approx(1.0)
    assert out.metrics["stop_signal/eligible_seq_count"] == pytest.approx(2.0)
    assert out.metrics["stop_signal/branch_count"] == pytest.approx(2.0)
