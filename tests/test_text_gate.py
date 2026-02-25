from __future__ import annotations

import pytest
import torch

from src.trainers.teacher_forcing.contracts import PipelineModuleSpec, TeacherForcingContext
from src.trainers.teacher_forcing.modules.coord_reg import run_coord_reg_module
from src.trainers.teacher_forcing.token_types import build_token_type_masks


def _coord_reg_state(*, vocab: int, coord_vocab: int) -> dict[str, torch.Tensor]:
    return {
        # coord_reg requires bbox_geo-produced state to be present (even when only text_gate is enabled).
        "coord_logits": torch.zeros((1, coord_vocab), dtype=torch.float32),
        "coord_logits_full": torch.zeros((1, vocab), dtype=torch.float32),
        "coord_target_bins": torch.zeros((1,), dtype=torch.long),
    }


def _set_coord_mass(
    logits: torch.Tensor,
    *,
    pos: int,
    coord_ids: list[int],
    coord_high: bool,
    other_id: int,
    hi: float = 10.0,
    lo: float = -10.0,
) -> None:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("expected logits shape [1, T, V]")
    if pos < 0 or pos >= int(logits.shape[1]):
        raise ValueError("pos out of range")

    logits[0, pos, :] = 0.0
    if coord_high:
        logits[0, pos, coord_ids] = float(hi)
        logits[0, pos, other_id] = float(lo)
    else:
        logits[0, pos, coord_ids] = float(lo)
        logits[0, pos, other_id] = float(hi)


def test_text_gate_math_high_coord_mass_increases_loss() -> None:
    # Build a minimal context with exactly one supervised text token at position p=2.
    #
    # text_gate is computed on logits predicting that token => logits position (p-1)=1.
    seq_len = 4
    vocab = 8
    coord_ids = [1, 2]

    token_type_masks = {
        "struct": torch.tensor([[False, False, True, False]], dtype=torch.bool),
        "desc": torch.zeros((1, seq_len), dtype=torch.bool),
        "coord": torch.zeros((1, seq_len), dtype=torch.bool),
        "eos": torch.zeros((1, seq_len), dtype=torch.bool),
    }

    meta = [{"prompt_len": 0, "prefix_len": 0, "train_len": 0}]
    input_ids = torch.zeros((1, seq_len), dtype=torch.long)

    logits_low = torch.zeros((1, seq_len, vocab), dtype=torch.float32)
    logits_high = torch.zeros((1, seq_len, vocab), dtype=torch.float32)

    _set_coord_mass(
        logits_low,
        pos=1,
        coord_ids=coord_ids,
        coord_high=False,
        other_id=3,
    )
    _set_coord_mass(
        logits_high,
        pos=1,
        coord_ids=coord_ids,
        coord_high=True,
        other_id=3,
    )

    spec = PipelineModuleSpec(
        name="coord_reg",
        enabled=True,
        weight=1.0,
        channels=("A", "B"),
        config={"text_gate_weight": 1.0},
    )
    state = _coord_reg_state(vocab=vocab, coord_vocab=len(coord_ids))

    ctx_low = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits_low,
        logits_ce=logits_low,
        meta=meta,
        coord_token_ids=coord_ids,
        temperature=1.0,
        decode_mode="exp",
        token_type_masks=token_type_masks,
        rollout_subset_masks={},
        extra={},
    )
    ctx_high = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits_high,
        logits_ce=logits_high,
        meta=meta,
        coord_token_ids=coord_ids,
        temperature=1.0,
        decode_mode="exp",
        token_type_masks=token_type_masks,
        rollout_subset_masks={},
        extra={},
    )

    out_low = run_coord_reg_module(context=ctx_low, spec=spec, state=state)
    out_high = run_coord_reg_module(context=ctx_high, spec=spec, state=state)

    gate_low = float(out_low.metrics.get("loss/text_gate", 0.0) or 0.0)
    gate_high = float(out_high.metrics.get("loss/text_gate", 0.0) or 0.0)
    assert gate_high > gate_low + 0.1


def test_rollout_text_gate_ignores_fp_prefix_tokens() -> None:
    # Build a rollout-style segment with a matched-prefix struct position and a couple
    # of unsupervised prefix tokens (FP-like). text_gate must ignore those unsupervised
    # prefix tokens (FP-neutral).
    vocab = 12
    coord_ids = [1, 2]
    seq_len = 8

    # Keep all tokens as non-coord ids so coord masking doesn't interfere.
    input_ids = torch.full((1, seq_len), 10, dtype=torch.long)
    meta = [
        {
            "prompt_len": 1,
            "prefix_len": 3,
            "train_len": 6,
            # Only position (prefix_start + 0) is supervised struct in the prefix.
            # Other prefix tokens are unsupervised (FP-like).
            "prefix_struct_pos": [0],
            "prefix_coord_pos": [],
            # Tail: position (tail_start + 0) is a desc span; rest is struct.
            "tail_desc_pos": [0],
            "tail_ignore_pos": [],
            "tail_closure_pos": [],
        }
    ]

    token_type_masks = build_token_type_masks(
        input_ids=input_ids,
        meta=meta,
        coord_id_set=set(coord_ids),
        channel="B",
    )

    logits_base = torch.zeros((1, seq_len, vocab), dtype=torch.float32)
    logits_fp_spike = torch.zeros((1, seq_len, vocab), dtype=torch.float32)

    # text_gate uses logits positions that predict supervised text tokens:
    # - prefix matched struct at p=1 => logits pos=0
    # - tail tokens at p=4,5,6 => logits pos=3,4,5
    for pos in (0, 3, 4, 5):
        _set_coord_mass(
            logits_base,
            pos=pos,
            coord_ids=coord_ids,
            coord_high=False,
            other_id=3,
        )
        _set_coord_mass(
            logits_fp_spike,
            pos=pos,
            coord_ids=coord_ids,
            coord_high=False,
            other_id=3,
        )

    # FP-like prefix tokens live at p=2 and p=3 => logits pos=1 and 2.
    for pos in (1, 2):
        _set_coord_mass(
            logits_fp_spike,
            pos=pos,
            coord_ids=coord_ids,
            coord_high=True,
            other_id=3,
        )

    spec = PipelineModuleSpec(
        name="coord_reg",
        enabled=True,
        weight=1.0,
        channels=("A", "B"),
        config={"text_gate_weight": 1.0},
    )
    state = _coord_reg_state(vocab=vocab, coord_vocab=len(coord_ids))

    ctx_base = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits_base,
        logits_ce=logits_base,
        meta=meta,
        coord_token_ids=coord_ids,
        temperature=1.0,
        decode_mode="exp",
        token_type_masks=token_type_masks,
        rollout_subset_masks={},
        extra={},
    )
    ctx_fp = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits_fp_spike,
        logits_ce=logits_fp_spike,
        meta=meta,
        coord_token_ids=coord_ids,
        temperature=1.0,
        decode_mode="exp",
        token_type_masks=token_type_masks,
        rollout_subset_masks={},
        extra={},
    )

    out_base = run_coord_reg_module(context=ctx_base, spec=spec, state=state)
    out_fp = run_coord_reg_module(context=ctx_fp, spec=spec, state=state)

    gate_base = float(out_base.metrics.get("loss/text_gate", 0.0) or 0.0)
    gate_fp = float(out_fp.metrics.get("loss/text_gate", 0.0) or 0.0)
    assert gate_fp == pytest.approx(gate_base, abs=1e-6)


def test_text_gate_ignores_unsupervised_desc_tokens_when_desc_weight_is_zero() -> None:
    vocab = 12
    coord_ids = [1, 2]
    seq_len = 6

    # Two text tokens:
    # - struct supervised at p=2 => logits pos=1
    # - desc UNSUPERVISED (weight=0) at p=4 => logits pos=3
    weights_masked = torch.zeros((1, seq_len), dtype=torch.float32)
    weights_masked[0, 2] = 1.0
    weights_masked[0, 4] = 0.0

    token_type_masks = {
        "struct": torch.tensor([[False, False, True, False, False, False]], dtype=torch.bool),
        "desc": torch.tensor([[False, False, False, False, True, False]], dtype=torch.bool),
        "coord": torch.zeros((1, seq_len), dtype=torch.bool),
        "eos": torch.zeros((1, seq_len), dtype=torch.bool),
    }

    meta = [{"prompt_len": 0, "prefix_len": 0, "train_len": 0}]
    input_ids = torch.zeros((1, seq_len), dtype=torch.long)

    logits_base = torch.zeros((1, seq_len, vocab), dtype=torch.float32)
    logits_desc_spike = torch.zeros((1, seq_len, vocab), dtype=torch.float32)

    # Supervised struct token (always included in the gate).
    _set_coord_mass(
        logits_base,
        pos=1,
        coord_ids=coord_ids,
        coord_high=False,
        other_id=3,
    )
    _set_coord_mass(
        logits_desc_spike,
        pos=1,
        coord_ids=coord_ids,
        coord_high=False,
        other_id=3,
    )

    # UNSUPERVISED desc token: make coord mass high here only in logits_desc_spike.
    _set_coord_mass(
        logits_base,
        pos=3,
        coord_ids=coord_ids,
        coord_high=False,
        other_id=3,
    )
    _set_coord_mass(
        logits_desc_spike,
        pos=3,
        coord_ids=coord_ids,
        coord_high=True,
        other_id=3,
    )

    spec = PipelineModuleSpec(
        name="coord_reg",
        enabled=True,
        weight=1.0,
        channels=("A", "B"),
        config={"text_gate_weight": 1.0},
    )
    state = _coord_reg_state(vocab=vocab, coord_vocab=len(coord_ids))
    state["weights_masked"] = weights_masked

    ctx_base = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits_base,
        logits_ce=logits_base,
        meta=meta,
        coord_token_ids=coord_ids,
        temperature=1.0,
        decode_mode="exp",
        token_type_masks=token_type_masks,
        rollout_subset_masks={},
        extra={},
    )
    ctx_spike = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits_desc_spike,
        logits_ce=logits_desc_spike,
        meta=meta,
        coord_token_ids=coord_ids,
        temperature=1.0,
        decode_mode="exp",
        token_type_masks=token_type_masks,
        rollout_subset_masks={},
        extra={},
    )

    out_base = run_coord_reg_module(context=ctx_base, spec=spec, state=state)
    out_spike = run_coord_reg_module(context=ctx_spike, spec=spec, state=state)

    gate_base = float(out_base.metrics.get("loss/text_gate", 0.0) or 0.0)
    gate_spike = float(out_spike.metrics.get("loss/text_gate", 0.0) or 0.0)
    assert gate_spike == pytest.approx(gate_base, abs=1e-6)
