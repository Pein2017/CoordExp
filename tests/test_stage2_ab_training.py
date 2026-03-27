import math
from dataclasses import asdict
import types
from contextlib import nullcontext
from typing import List, Sequence

import pytest
import torch
import torch.nn as nn

from src.config.loader import ConfigLoader
from src.trainers.stage2_rollout_aligned import (
    GTObject,
    _serialize_append_fragment,
    parse_rollout_for_matching,
)
from src.trainers.stage2_two_channel import (
    Stage2ABTrainingTrainer,
    _PendingStage2Log,
    _bbox_groups_from_token_ids,
    _build_canonical_prefix_text_data,
    _build_channel_b_supervision_targets,
    _build_channel_b_triage,
    _bbox_smoothl1_ciou_loss,
    _build_canonical_prefix_data,
    _build_duplicate_burst_unlikelihood_targets,
    _build_teacher_forced_payload,
    _compute_duplicate_diagnostics,
    _expectation_decode_coords,
    _extract_gt_bboxonly,
    _matched_prefix_structure_positions,
    _sequential_dedup_bbox_objects,
    _stage2_ab_tail_closure_positions,
)


class _DummyOut:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.past_key_values = None


class _DummyModel(nn.Module):
    def __init__(
        self, *, vocab: int = 1200, hidden: int = 8, model_type: str = "qwen3_vl"
    ):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "has_input_ids": input_ids is not None,
                "has_inputs_embeds": inputs_embeds is not None,
                "position_ids": position_ids,
                "use_cache": use_cache,
                "past_key_values": past_key_values,
            }
        )
        assert (input_ids is None) ^ (inputs_embeds is None)
        assert use_cache is False
        assert past_key_values is None
        if position_ids is not None:
            # Qwen packing contract: 4-row position_ids ([text_position_ids; mRoPE]).
            assert position_ids.shape[0] == 4

        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        return _DummyOut(self.lm_head(x))

class _DummySlicedModel(_DummyModel):
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        # Simulate logits_to_keep-style slicing (seq_len shrinks), which stage2-ab forbids.
        out.logits = out.logits[:, :-1, :]
        return out


class _DummyAlwaysTokenModel(nn.Module):
    def __init__(
        self,
        *,
        pred_id: int = 1100,
        vocab: int = 1200,
        model_type: str = "qwen3_vl",
    ):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.pred_id = int(pred_id)
        self.vocab = int(vocab)

    def forward(
        self,
        *,
        input_ids=None,
        position_ids=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        assert input_ids is not None
        assert use_cache is False
        assert past_key_values is None
        if position_ids is not None:
            assert position_ids.shape[0] == 4

        bsz, seqlen = input_ids.shape
        logits = torch.full(
            (bsz, seqlen, self.vocab),
            -100.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., self.pred_id] = 0.0
        return _DummyOut(logits)


class _DummyCallIndexedTokenModel(nn.Module):
    """Dummy model that returns different constant logits per forward call."""

    def __init__(
        self,
        *,
        pred_ids: list[int],
        vocab: int = 1200,
        hidden: int = 8,
        model_type: str = "qwen3_vl",
    ):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.embed = nn.Embedding(int(vocab), int(hidden))
        self.vocab = int(vocab)
        self.pred_ids = [int(x) for x in list(pred_ids)]
        if not self.pred_ids:
            raise ValueError("pred_ids must be non-empty")
        self.calls = 0

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        assert (input_ids is None) ^ (inputs_embeds is None)
        assert use_cache is False
        assert past_key_values is None
        if position_ids is not None:
            assert position_ids.shape[0] == 4

        idx = min(int(self.calls), int(len(self.pred_ids) - 1))
        pred_id = int(self.pred_ids[idx])
        self.calls += 1

        if input_ids is not None:
            bsz, seqlen = input_ids.shape
            device = input_ids.device
        else:
            bsz, seqlen = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        logits = torch.full(
            (int(bsz), int(seqlen), int(self.vocab)),
            -100.0,
            dtype=torch.float32,
            device=device,
        )
        logits[..., int(pred_id)] = 0.0
        return _DummyOut(logits)


class _DummyConstantCoord999Model(nn.Module):
    """Dummy model that makes softctx updates deterministic.

    It always assigns the highest coord-bin logit to bin 999, so the expected
    coord embedding is embedding(999) for every coord slot on iteration >=1.
    """

    def __init__(
        self, *, vocab: int = 1200, hidden: int = 8, model_type: str = "qwen3_vl"
    ):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.embed = nn.Embedding(vocab, hidden)
        # Make embeddings deterministic/unique so bitwise equality checks are meaningful.
        with torch.no_grad():
            w = torch.arange(vocab, dtype=torch.float32).unsqueeze(1).repeat(1, hidden)
            self.embed.weight.copy_(w)
        self.vocab = int(vocab)
        self.calls = []
        self.inputs_embeds_calls = []

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "has_input_ids": input_ids is not None,
                "has_inputs_embeds": inputs_embeds is not None,
                "position_ids": position_ids,
                "use_cache": use_cache,
                "past_key_values": past_key_values,
            }
        )
        assert (input_ids is None) ^ (inputs_embeds is None)
        assert use_cache is False
        assert past_key_values is None
        if position_ids is not None:
            assert position_ids.shape[0] == 4

        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        self.inputs_embeds_calls.append(x.detach().cpu().clone())

        bsz, seqlen = x.shape[:2]
        logits = torch.full(
            (bsz, seqlen, self.vocab),
            -100.0,
            dtype=torch.float32,
            device=x.device,
        )
        logits[..., 999] = 0.0
        return _DummyOut(logits)


class _DummyTokenizer:
    def __init__(self):
        # Reserve [0,999] for coord tokens.
        self._next_id = 1000
        self._tok_to_id: dict[str, int] = {}
        self._id_to_tok: dict[int, str] = {}
        # Make '{' stable.
        self._id_for("{")

    def _id_for(self, tok: str) -> int:
        if tok not in self._tok_to_id:
            idx = int(self._next_id)
            self._next_id += 1
            self._tok_to_id[tok] = idx
            self._id_to_tok[idx] = tok
        return int(self._tok_to_id[tok])

    def encode(self, text: str, add_special_tokens: bool = False):
        # Keep coord tokens as single ids when present; otherwise char-level.
        if (
            isinstance(text, str)
            and text.startswith("<|coord_")
            and text.endswith("|>")
        ):
            try:
                n = int(text[len("<|coord_") : -len("|>")])
                return [int(n)]
            except Exception:
                return [self._id_for(text)]
        return [self._id_for(ch) for ch in str(text)]

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        out = []
        for t in token_ids:
            tid = int(t)
            if 0 <= tid <= 999:
                out.append(f"<|coord_{tid}|>")
            else:
                out.append(self._id_to_tok.get(tid, "?"))
        return "".join(out)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            toks = [tokens]
            scalar = True
        else:
            toks = list(tokens)
            scalar = False

        ids: list[int] = []
        for tok in toks:
            s = str(tok)
            if s.startswith("<|coord_") and s.endswith("|>"):
                try:
                    n = int(s[len("<|coord_") : -len("|>")])
                except Exception:
                    n = -1
                ids.append(int(n))
            else:
                ids.append(self._id_for(s))
        return ids[0] if scalar else ids


class _PieceFrameMismatchTokenizer(_DummyTokenizer):
    """Tokenizer stub where per-token decode and full decode have different lengths."""

    def __init__(self):
        super().__init__()
        self._mismatch_id = self._id_for("~")

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        ids = [int(t) for t in token_ids]
        if len(ids) == 1 and int(ids[0]) == int(self._mismatch_id):
            # Per-token decode expands this token to two chars.
            return "~~"

        out = []
        for tid in ids:
            if 0 <= tid <= 999:
                out.append(f"<|coord_{tid}|>")
            elif int(tid) == int(self._mismatch_id):
                # Full decode contracts the same token to one char.
                out.append("~")
            else:
                out.append(self._id_to_tok.get(int(tid), "?"))
        return "".join(out)


class _BoundaryMergingTokenizer(_DummyTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False):
        if (
            isinstance(text, str)
            and text.startswith("<|coord_")
            and text.endswith("|>")
        ):
            return super().encode(text, add_special_tokens=add_special_tokens)

        s = str(text)
        merged_pieces = (
            '[{"desc": "book", "bbox_2d": [',
            ', {"desc": "book", "bbox_2d": [',
            "]}",
        )
        out: list[int] = []
        i = 0
        while i < len(s):
            match = None
            for piece in merged_pieces:
                if s.startswith(piece, i) and (
                    match is None or len(piece) > len(match)
                ):
                    match = piece
            if match is not None:
                out.append(self._id_for(match))
                i += len(match)
                continue
            out.append(self._id_for(s[i]))
            i += 1
        return out


def test_b_ratio_schedule_is_deterministic():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {"schedule": {"b_ratio": 0.5}}
    t._stage2_channel_override = None
    got = [t._stage2_channel_for_step(i) for i in range(6)]
    assert got == ["A", "B", "A", "B", "A", "B"]

    t.stage2_ab_cfg = {"schedule": {"b_ratio": 0.05}}
    got2 = [t._stage2_channel_for_step(i) for i in range(20)]
    assert got2.count("B") == 1
    assert got2[-1] == "B"


def test_legacy_stop_neutral_key_is_rejected() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "channel_b": {
            "stop_neutral": {"enabled": True},
        },
    }
    with pytest.raises(ValueError, match="stop_neutral"):
        _ = t._ab_channel_b_cfg()


def test_expectation_decode_is_mean_not_argmax():
    logits = torch.full((1, 1000), -100.0)
    logits[0, 0] = 0.0
    logits[0, 999] = 0.0
    out = _expectation_decode_coords(coord_logits=logits, temperature=1.0)
    assert float(out.item()) == pytest.approx(0.5, abs=1e-6)


def test_expectation_decode_st_has_hard_forward_and_soft_grad() -> None:
    logits = torch.full((1, 1000), -100.0)
    logits[0, 0] = 0.0
    logits[0, 999] = 0.0

    out_exp = _expectation_decode_coords(
        coord_logits=logits,
        temperature=1.0,
        mode="exp",
    )
    out_st = _expectation_decode_coords(
        coord_logits=logits,
        temperature=1.0,
        mode="st",
    )

    # Forward values differ: exp is mean-like, ST uses hard argmax forward.
    assert float(out_exp.item()) == pytest.approx(0.5, abs=1e-6)
    assert float(out_st.item()) == pytest.approx(0.0, abs=1e-6)


def test_expectation_decode_st_propagates_gradients() -> None:
    logits = torch.randn(2, 1000, requires_grad=True)
    out_st = _expectation_decode_coords(
        coord_logits=logits,
        temperature=1.0,
        mode="st",
    )

    loss = out_st.sum()
    loss.backward()

    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_bbox_losses_stable_on_noncanonical_pred():
    pred = torch.tensor([[1.2, -0.1, -0.2, 0.5]], dtype=torch.float32)
    gt = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    smoothl1, ciou = _bbox_smoothl1_ciou_loss(pred_xyxy=pred, gt_xyxy=gt)
    assert torch.isfinite(smoothl1).item()
    assert torch.isfinite(ciou).item()


def _make_stage2_pipeline_manifest(
    *,
    token_ce_enabled: bool = True,
    token_ce_weight: float = 1.0,
    duplicate_burst_unlikelihood_enabled: bool = False,
    duplicate_burst_unlikelihood_weight: float = 1.0,
    desc_ce_weight: float = 1.0,
    rollout_fn_desc_weight: float | None = None,
    rollout_global_prefix_struct_ce_weight: float = 1.0,
    bbox_geo_enabled: bool = True,
    bbox_geo_weight: float = 1.0,
    bbox_smoothl1_weight: float = 1.0,
    bbox_ciou_weight: float = 1.0,
    bbox_size_aux_enabled: bool = True,
    bbox_size_aux_weight: float = 1.0,
    bbox_log_wh_weight: float = 0.0,
    bbox_oversize_weight: float = 0.0,
    coord_reg_enabled: bool = True,
    coord_reg_weight: float = 1.0,
    coord_ce_weight: float = 0.0,
    coord_soft_ce_weight: float = 0.0,
    coord_w1_weight: float = 0.0,
    coord_gate_weight: float = 0.0,
    text_gate_weight: float = 0.0,
) -> dict:
    token_cfg: dict[str, object] = {
        "desc_ce_weight": float(desc_ce_weight),
        "rollout_global_prefix_struct_ce_weight": float(
            rollout_global_prefix_struct_ce_weight
        ),
    }
    if rollout_fn_desc_weight is not None:
        token_cfg["rollout_fn_desc_weight"] = float(rollout_fn_desc_weight)

    return {
        "objective": [
            {
                "name": "token_ce",
                "enabled": bool(token_ce_enabled),
                "weight": float(token_ce_weight),
                "channels": ["A", "B"],
                "application": {"preset": "anchor_text_only"},
                "config": token_cfg,
            },
            {
                "name": "loss_duplicate_burst_unlikelihood",
                "enabled": bool(duplicate_burst_unlikelihood_enabled),
                "weight": float(duplicate_burst_unlikelihood_weight),
                "channels": ["B"],
                "application": {"preset": "rollout_only"},
                "config": {},
            },
            {
                "name": "bbox_geo",
                "enabled": bool(bbox_geo_enabled),
                "weight": float(bbox_geo_weight),
                "channels": ["A", "B"],
                "application": {"preset": "anchor_only"},
                "config": {
                    "smoothl1_weight": float(bbox_smoothl1_weight),
                    "ciou_weight": float(bbox_ciou_weight),
                },
            },
            {
                "name": "bbox_size_aux",
                "enabled": bool(bbox_size_aux_enabled),
                "weight": float(bbox_size_aux_weight),
                "channels": ["A", "B"],
                "application": {"preset": "anchor_only"},
                "config": {
                    "log_wh_weight": float(bbox_log_wh_weight),
                    "oversize_penalty_weight": float(bbox_oversize_weight),
                    "oversize_area_frac_threshold": None,
                    "oversize_log_w_threshold": None,
                    "oversize_log_h_threshold": None,
                    "eps": 1e-6,
                },
            },
            {
                "name": "coord_reg",
                "enabled": bool(coord_reg_enabled),
                "weight": float(coord_reg_weight),
                "channels": ["A", "B"],
                "application": {"preset": "anchor_only"},
                "config": {
                    "coord_ce_weight": float(coord_ce_weight),
                    "soft_ce_weight": float(coord_soft_ce_weight),
                    "w1_weight": float(coord_w1_weight),
                    "coord_gate_weight": float(coord_gate_weight),
                    "text_gate_weight": float(text_gate_weight),
                },
            },
        ],
        "diagnostics": [],
    }


def _make_min_trainer():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 0.0},
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "desc_ce_weight": 1.0,
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=float(t.stage2_ab_cfg["desc_ce_weight"]),
        bbox_smoothl1_weight=float(t.stage2_ab_cfg["bbox_smoothl1_weight"]),
        bbox_ciou_weight=float(t.stage2_ab_cfg["bbox_ciou_weight"]),
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)
    return t


def test_channel_a_runs_single_forward_and_enforces_qwen_posids():
    trainer = _make_min_trainer()
    model = _DummyModel()

    # Prompt (2 tokens) + assistant (5 tokens, includes 4 coord slots)
    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    position_ids = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
    text_position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    loss = trainer.compute_loss(
        model,
        {
            "_stage2_ab_channel": "A",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "text_position_ids": text_position_ids,
        },
    )
    assert isinstance(loss, torch.Tensor)
    assert len(model.calls) == 1


def test_channel_a_does_not_use_inputs_embeds_iteration_path():
    trainer = _make_min_trainer()
    model = _DummyModel()

    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    position_ids = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
    text_position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    _ = trainer.compute_loss(
        model,
        {
            "_stage2_ab_channel": "A",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "text_position_ids": text_position_ids,
        },
    )
    assert len(model.calls) == 1


def test_channel_a_multimodal_path_uses_single_inputs_embeds_forward():
    trainer = _make_min_trainer()
    model = _DummyConstantCoord999Model()

    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    position_ids = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
    text_position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    _ = trainer.compute_loss(
        model,
        {
            "_stage2_ab_channel": "A",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "text_position_ids": text_position_ids,
            # Multimodal batches carry pixel_values; trainer must not perturb non-coord slots.
            "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        },
    )

    assert len(model.inputs_embeds_calls) == 1


def test_channel_a_ce_uses_single_forward_logits():
    trainer = _make_min_trainer()
    # Isolate anchor CE and non-text objectives.
    trainer.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )

    # Prompt (2 tokens) + assistant (5 tokens, includes 4 coord slots + 1 non-coord token).
    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    position_ids = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
    text_position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
        }
    ]

    # First forward predicts token 1102 (correct), second predicts 1103 (wrong).
    model_good_a1 = _DummyCallIndexedTokenModel(pred_ids=[1102, 1103])
    loss_good = trainer.compute_loss(
        model_good_a1,
        {
            "_stage2_ab_channel": "A",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "text_position_ids": text_position_ids,
        },
    )

    # If CE incorrectly ignored the first forward logits, this comparison would collapse.
    model_bad_a1 = _DummyCallIndexedTokenModel(pred_ids=[1103, 1102])
    loss_bad = trainer.compute_loss(
        model_bad_a1,
        {
            "_stage2_ab_channel": "A",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "text_position_ids": text_position_ids,
        },
    )

    assert float(loss_good.detach().cpu().item()) < float(loss_bad.detach().cpu().item())


def test_parse_rollout_fallback_prefix_brace_is_deterministic():
    tok = _DummyTokenizer()
    resp_ids = tok.encode("hello", add_special_tokens=False)

    p1 = parse_rollout_for_matching(tokenizer=tok, response_token_ids=list(resp_ids))
    p2 = parse_rollout_for_matching(tokenizer=tok, response_token_ids=list(resp_ids))

    assert p1.prefix_token_ids == tok.encode('{"objects": [', add_special_tokens=False)
    assert p1.prefix_token_ids == p2.prefix_token_ids
    assert p1.prefix_text == p2.prefix_text == '{"objects": ['
    assert p1.valid_objects == []
    assert p1.invalid_rollout is True


def test_channel_b_matching_uses_candidate_top_k_not_decode_top_k(monkeypatch):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {}
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t.state = types.SimpleNamespace(global_step=0)

    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 7,
        "maskiou_resolution": 256,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 16
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        response_token_ids=[],
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )

    captured = {}

    class _StopAfterMatch(RuntimeError):
        pass

    def _fake_match(*, preds, gts, top_k, **kwargs):
        captured["top_k"] = int(top_k)
        raise _StopAfterMatch("stop once matcher receives top_k")

    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [0, 0, 0, 0], "desc": "x"}],
        },
    }
    with pytest.raises(_StopAfterMatch):
        t._prepare_batch_inputs_b([sample], _segments_only=True)

    assert captured["top_k"] == 7


def test_channel_b_invalid_rollout_keeps_sample_via_empty_prefix_fallback(monkeypatch):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: default

    tok = _DummyTokenizer()
    t.template = types.SimpleNamespace(tokenizer=tok)
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=list(tok.encode('{"objects": [', add_special_tokens=False)),
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=True,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )

    captured = {}

    class _StopAfterMatch(RuntimeError):
        pass

    def _fake_match(*, preds, gts, **kwargs):
        captured["n_pred"] = int(len(preds))
        captured["n_gt"] = int(len(gts))
        raise _StopAfterMatch("stop after invalid-rollout fallback reaches matcher")

    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "x"}],
        },
    }
    with pytest.raises(_StopAfterMatch):
        t._prepare_batch_inputs_b([sample], _segments_only=True)

    assert captured == {"n_pred": 0, "n_gt": 1}


def test_channel_b_enabled_pseudo_positive_drops_invalid_anchor_sample(
    monkeypatch,
):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    ab_cfg = {
        "pseudo_positive.enabled": True,
        "triage_posterior.num_rollouts": 4,
        "invalid_rollout_policy": "dump_and_continue",
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: ab_cfg.get(key, default)

    tok = _DummyTokenizer()
    t.template = types.SimpleNamespace(tokenizer=tok)
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = (
        lambda chunk, decode_override=None, request_index_offset=0: [
            ([], "", "sampling", []) for _ in chunk
        ]
    )
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=list(tok.encode('{"objects": [', add_special_tokens=False)),
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=True,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "x"}],
        },
    }
    segments, batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)
    assert segments == []
    assert batch_metrics["stage2/invalid_rollout"] == pytest.approx(4.0)
    assert batch_metrics[
        "stage2_ab/channel_b/invalid_rollout_sample_dropped"
    ] == pytest.approx(1.0)


def test_channel_b_closure_resolution_failure_falls_back_without_dropping_sample(monkeypatch):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel._stage2_ab_tail_closure_positions",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("synthetic closure ambiguity")),
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [9, 9, 10, 10], "desc": "fn"}],
        },
    }

    segments, batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)

    assert len(segments) == 1
    meta = segments[0][1]
    assert meta["tail_closure_pos"] == []
    assert batch_metrics["stage2_ab/channel_b/closure_supervision/N_drop"] == pytest.approx(1.0)
    assert batch_metrics["stage2_ab/channel_b/invalid_rollout"] == pytest.approx(0.0)


def test_channel_b_duplicate_iou_threshold_zero_propagates_to_dedup(monkeypatch):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = (
        lambda key, default=None: 0.0 if key == "duplicate_iou_threshold" else default
    )

    tok = _DummyTokenizer()
    t.template = types.SimpleNamespace(tokenizer=tok)
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )

    captured = {}

    class _StopAfterDedup(RuntimeError):
        pass

    def _fake_dedup(*, parsed_bbox_objects_raw, duplicate_iou_threshold):
        captured["threshold"] = float(duplicate_iou_threshold)
        raise _StopAfterDedup("stop after dedup threshold capture")

    monkeypatch.setattr(
        "src.trainers.stage2_two_channel._sequential_dedup_bbox_objects",
        _fake_dedup,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "x"}],
        },
    }
    with pytest.raises(_StopAfterDedup):
        t._prepare_batch_inputs_b([sample], _segments_only=True)

    assert captured["threshold"] == pytest.approx(0.0)


def test_channel_b_suspicious_monitor_dump_buffers_full_eval_style_payload(
    monkeypatch,
):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "train_monitor_dump": {
            "enabled": True,
            "max_samples": 1,
            "max_text_chars": 32,
            "write_markdown": True,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t.state = types.SimpleNamespace(global_step=7, epoch=1.5)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    full_rollout_text = "rollout:" + (" very-duplicated" * 128)
    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=list(range(12)),
        response_text=full_rollout_text,
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="person",
            ),
            types.SimpleNamespace(
                index=1,
                geom_type="bbox_2d",
                coord_token_indices=[4, 5, 6, 7],
                desc="person",
            ),
            types.SimpleNamespace(
                index=2,
                geom_type="bbox_2d",
                coord_token_indices=[8, 9, 10, 11],
                desc="book",
            ),
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )

    coord_lookup = {
        (0, 1, 2, 3): [10, 10, 20, 20],
        (4, 5, 6, 7): [10, 10, 20, 20],
        (8, 9, 10, 11): [40, 40, 60, 60],
    }
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.points_from_coord_tokens",
        lambda **kwargs: list(coord_lookup[tuple(kwargs["coord_token_indices"])]),
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[(0, 0)],
            fn_gt_indices=[],
            fp_pred_indices=[1],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
    )

    sample = {
        "sample_id": "sample-314",
        "image_id": 314,
        "images": ["scene_314.jpg"],
        "width": 1000,
        "height": 1000,
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [10, 10, 20, 20], "desc": "person"}],
        },
    }

    segments, batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)
    assert len(segments) == 1
    assert batch_metrics["stage2_ab/channel_b/dup/N_duplicates"] == pytest.approx(1.0)
    assert len(t._stage2_train_monitor_candidates) == 1

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )
    t._stage2_flush_train_monitor_dump(global_step=8)

    dumped = captured["payload"]["samples"][0]
    assert captured["global_step"] == 8
    assert captured["payload"]["kind"] == "train_monitor_dump"
    assert dumped["image_id"] == 314
    assert dumped["gt"] == [{"desc": "person", "bbox_2d": [10, 10, 20, 20]}]
    assert dumped["pred"] == [
        {"desc": "person", "bbox_2d": [10, 10, 20, 20]},
        {"desc": "person", "bbox_2d": [10, 10, 20, 20]},
        {"desc": "book", "bbox_2d": [40, 40, 60, 60]},
    ]
    assert dumped["duplication"]["duplicates"] == 1
    assert dumped["duplication"]["clean_accepted"] == 2
    assert dumped["match"]["fp_pred_indices"] == [1]
    assert dumped["rollout_text"] == full_rollout_text
    assert "...<truncated>" not in dumped["rollout_text"]


def test_stage2_train_monitor_dump_prefers_most_duplicate_candidate():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "max_samples": 1,
            "write_markdown": False,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_backend = lambda: "hf"
    t.state = types.SimpleNamespace(global_step=11, epoch=0.0)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )

    t._stage2_reset_train_monitor_dump(global_step=11)
    t._stage2_note_train_monitor_candidate(
        global_step=11,
        sample={
            "sample_id": "low",
            "duplication": {"duplicates": 1, "duplicate_bursts": 1},
            "stats": {"fp_count": 0, "raw_valid_pred_objects": 2},
        },
    )
    t._stage2_note_train_monitor_candidate(
        global_step=11,
        sample={
            "sample_id": "high",
            "duplication": {"duplicates": 3, "duplicate_bursts": 2},
            "stats": {"fp_count": 1, "raw_valid_pred_objects": 5},
        },
    )

    t._stage2_flush_train_monitor_dump(global_step=11)

    assert captured["global_step"] == 11
    assert captured["payload"]["samples"][0]["sample_id"] == "high"


def test_stage2_train_monitor_dump_uses_logged_step_not_preincrement_step() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "every_steps": 40,
            "max_samples": 1,
            "write_markdown": False,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_backend = lambda: "hf"
    t.args = types.SimpleNamespace(logging_steps=10, logging_first_step=True)
    t.state = types.SimpleNamespace(global_step=39, epoch=0.0)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []
    t._stage2_train_monitor_dump_count = 0
    t._stage2_train_monitor_dump_last_step = None

    assert t._stage2_train_monitor_step_allowed(global_step=40) is True
    assert t._stage2_train_monitor_step_allowed(global_step=39) is False

    t._stage2_reset_train_monitor_dump(global_step=40)
    t._stage2_note_train_monitor_candidate(
        global_step=40,
        sample={
            "sample_id": "dup-heavy",
            "duplication": {"duplicates": 2, "duplicate_bursts": 1},
            "stats": {"fp_count": 1, "raw_valid_pred_objects": 3},
        },
    )

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )
    t._stage2_flush_train_monitor_dump(global_step=40)

    assert captured["global_step"] == 40
    assert captured["payload"]["samples"][0]["sample_id"] == "dup-heavy"


def test_stage2_train_monitor_dump_every_channel_b_steps_ignores_global_step_aliasing() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "every_steps": 40,
            "every_channel_b_steps": 3,
            "max_samples": 1,
            "write_markdown": False,
        },
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_backend = lambda: "hf"
    t.args = types.SimpleNamespace(logging_steps=10, logging_first_step=False)
    t.state = types.SimpleNamespace(global_step=11, epoch=0.0)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []
    t._stage2_train_monitor_b_step_count = 0
    t._stage2_train_monitor_dump_count = 0
    t._stage2_train_monitor_dump_last_step = None

    t._stage2_reset_train_monitor_dump(global_step=4)
    assert t._stage2_train_monitor_b_step_count == 1
    assert t._stage2_train_monitor_step_allowed(global_step=4) is False

    t._stage2_reset_train_monitor_dump(global_step=8)
    assert t._stage2_train_monitor_b_step_count == 2
    assert t._stage2_train_monitor_step_allowed(global_step=8) is False

    t._stage2_reset_train_monitor_dump(global_step=12)
    assert t._stage2_train_monitor_b_step_count == 3
    assert t._stage2_train_monitor_step_allowed(global_step=12) is True

    t._stage2_note_train_monitor_candidate(
        global_step=12,
        sample={
            "sample_id": "third-b-step",
            "duplication": {"duplicates": 2, "duplicate_bursts": 1},
            "stats": {"fp_count": 1, "raw_valid_pred_objects": 3},
        },
    )

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )
    t._stage2_flush_train_monitor_dump(global_step=12)

    assert captured["global_step"] == 12
    assert captured["payload"]["samples"][0]["sample_id"] == "third-b-step"


def test_stage2_train_monitor_dump_keeps_eval_budget_and_same_step_eligibility():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "decode_mode": "greedy",
        "train_monitor_dump": {
            "enabled": True,
            "max_events": 1,
            "max_samples": 1,
            "write_markdown": False,
        },
        "eval_monitor_dump": {"enabled": True, "every_evals": 1},
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._rollout_backend = lambda: "hf"
    t.args = types.SimpleNamespace(logging_steps=1, logging_first_step=True)
    t.state = types.SimpleNamespace(global_step=11, epoch=0.0)
    t.is_world_process_zero = True
    t._monitor_dump_count = 0
    t._monitor_dump_last_step = None
    t._stage2_train_monitor_pending_gs = None
    t._stage2_train_monitor_candidates = []
    t._stage2_train_monitor_dump_count = 0
    t._stage2_train_monitor_dump_last_step = None

    captured = {}
    t._write_monitor_dump = lambda *, global_step, payload: captured.update(
        {"global_step": int(global_step), "payload": payload}
    )

    t._stage2_reset_train_monitor_dump(global_step=11)
    t._stage2_note_train_monitor_candidate(
        global_step=11,
        sample={
            "sample_id": "dup-heavy",
            "duplication": {"duplicates": 4, "duplicate_bursts": 2},
            "stats": {"fp_count": 1, "raw_valid_pred_objects": 5},
        },
    )
    t._stage2_flush_train_monitor_dump(global_step=11)

    assert captured["global_step"] == 11
    assert t._stage2_train_monitor_dump_count == 1
    assert t._stage2_train_monitor_dump_last_step == 11
    assert t._monitor_dump_count == 0
    assert t._monitor_dump_last_step is None
    assert t._should_eval_monitor_dump(global_step=11, eval_index=1) is True


def test_channel_b_fn_bbox_groups_anchor_to_clean_prefix_not_raw_prefix(monkeypatch):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._decoding_params = lambda: (0.7, 0.95, -1)
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._rollout_many = lambda chunk: [([], "", "sampling", []) for _ in chunk]
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[2000 + i for i in range(64)],
        prefix_text="unused-raw-prefix",
        response_token_ids=[10, 10, 20, 20, 10, 10, 20, 20],
        response_text="",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="cat",
            ),
            types.SimpleNamespace(
                index=1,
                geom_type="bbox_2d",
                coord_token_indices=[4, 5, 6, 7],
                desc="cat",
            ),
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: fake_parse,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[0],
            fp_pred_indices=[0],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [9, 9, 10, 10], "desc": "fn"}],
        },
    }

    segments, _batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)
    meta = segments[0][1]

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tok,
        objects=[
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 10, 20, 20],
                desc="cat",
            )
        ],
        object_field_order="desc_first",
    )
    fn_obj = GTObject(
        index=0,
        geom_type="bbox_2d",
        points_norm1000=[9, 9, 10, 10],
        desc="fn",
    )
    append_text = _serialize_append_fragment(
        fn_objects=[fn_obj],
        prefix_text=clean_prefix.prefix_text,
    )
    rel_groups = _bbox_groups_from_token_ids(
        token_ids=list(tok.encode(append_text)),
        coord_id_set=set(range(1000)),
        gt_objs=[fn_obj],
    )
    expected_pos = [int(meta["prompt_len"] + meta["prefix_len"] + p) for p in rel_groups[0]]

    assert len(fake_parse.prefix_token_ids) > int(meta["prefix_len"])
    assert meta["bbox_groups_fn"][0]["pos"] == expected_pos


def test_channel_b_dual_rollout_triage_emits_recovered_ground_truth_weight_multipliers(
    monkeypatch,
) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.explorer_top_p": 0.9,
        "triage_posterior.explorer_top_k": 12,
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
        "triage_posterior.recovered_ground_truth_weight_multiplier": 2.5,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls: list[dict[str, object]] = []

    def _fake_rollout_many(chunk, decode_override=None):
        rollout_calls.append(dict(decode_override or {}))
        temp = float((decode_override or {}).get("temperature", 0.0) or 0.0)
        marker = 101 if temp <= 0.0 else 202
        mode = "greedy" if temp <= 0.0 else "sampling"
        return [([marker], "", mode, []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    anchor_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="anchor",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="bad",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    explorer_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[202],
        response_text="explorer",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[4, 5, 6, 7],
                desc="explore",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )

    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: (
            anchor_parse
            if int(kwargs["response_token_ids"][0]) == 101
            else explorer_parse
        ),
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )

    def _fake_match(*, preds, **kwargs):
        descs = [str(obj.desc) for obj in preds]
        if descs == ["explore"]:
            return types.SimpleNamespace(
                matched_pairs=[(0, 0)],
                fn_gt_indices=[],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=1.0,
                matched_maskiou_count=1,
            )
        return types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[0],
            fp_pred_indices=list(range(len(preds))),
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [10, 10, 20, 20], "desc": "gt"}],
        },
    }

    segments, batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)

    assert len(rollout_calls) == 2
    assert rollout_calls[0]["temperature"] == pytest.approx(0.0)
    assert rollout_calls[1]["temperature"] == pytest.approx(0.7)
    assert rollout_calls[1]["top_p"] == pytest.approx(0.9)
    assert rollout_calls[1]["top_k"] == 12

    meta = segments[0][1]
    assert meta["anchor_gt_backed_indices"] == []
    assert meta["shielded_anchor_indices"] == []
    assert meta["dead_anchor_indices"] == [0]
    assert meta["recovered_gt_indices"] == [0]
    assert meta["fn_object_weights"] == [pytest.approx(2.5)]
    assert meta["bbox_groups_fn"][0]["weight"] == pytest.approx(2.5)
    assert meta["tail_desc_weights"]
    assert set(float(x) for x in meta["tail_desc_weights"]) == {2.5}
    assert meta["duplicate_burst_unlikelihood_targets"] == []

    assert batch_metrics["train/triage/dead_anchor_count"] == pytest.approx(
        1.0
    )
    assert batch_metrics["train/triage/recovered_ground_truth_count"] == pytest.approx(
        1.0
    )
    assert batch_metrics["train/triage/recovered_ground_truth_rate_den"] == pytest.approx(
        1.0
    )
    assert batch_metrics["train/triage/recovered_ground_truth_rate"] == pytest.approx(
        1.0
    )
    assert batch_metrics["train/triage/dead_anchor_rate"] == pytest.approx(1.0)
    assert batch_metrics["train/triage/explorer_only_dead_rate"] == pytest.approx(
        0.0
    )
    assert batch_metrics["rollout/anchor/pred_objects"] == pytest.approx(1.0)
    assert batch_metrics["rollout/anchor/valid_pred_objects"] == pytest.approx(1.0)
    assert batch_metrics["rollout/anchor/gen_new_tokens_mean"] == pytest.approx(1.0)
    assert batch_metrics["rollout/anchor/gen_new_tokens_p90"] == pytest.approx(1.0)
    assert batch_metrics["rollout/explorer/pred_objects"] == pytest.approx(1.0)
    assert batch_metrics["rollout/explorer/valid_pred_objects"] == pytest.approx(1.0)
    assert batch_metrics["rollout/explorer/gen_new_tokens_mean"] == pytest.approx(1.0)
    assert batch_metrics["rollout/explorer/gen_new_tokens_p90"] == pytest.approx(1.0)
    assert batch_metrics["rollout/explorer/temperature"] == pytest.approx(0.7)
    assert batch_metrics["rollout/explorer/do_sample"] == pytest.approx(1.0)
    assert batch_metrics["rollout/explorer/top_p"] == pytest.approx(0.9)
    assert batch_metrics["rollout/explorer/top_k"] == pytest.approx(12.0)
    assert batch_metrics[
        "rollout/matched_for_supervision_over_valid_pred"
    ] == pytest.approx(0.0)


def test_channel_b_dual_rollout_chunking_is_policy_symmetric(monkeypatch) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: default

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 2
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls: list[tuple[int, float]] = []

    def _fake_rollout_many(chunk, decode_override=None):
        rollout_calls.append(
            (
                int(len(chunk)),
                float((decode_override or {}).get("temperature", 0.0) or 0.0),
            )
        )
        marker = 101 if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0 else 202
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    with monkeypatch.context() as mp:
        mp.setattr(
            "src.trainers.stage2_two_channel.parse_rollout_for_matching",
            lambda **kwargs: fake_parse,
        )
        mp.setattr(
            "src.trainers.stage2_two_channel._extract_gt_bboxonly",
            lambda _sample: [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[0, 0, 1, 1],
                    desc="gt",
                )
            ],
        )
        mp.setattr(
            "src.trainers.stage2_two_channel.hungarian_match_maskiou",
            lambda **kwargs: types.SimpleNamespace(
                matched_pairs=[],
                fn_gt_indices=[],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=0.0,
                matched_maskiou_count=0,
            ),
        )
        mp.setattr(
            "src.trainers.stage2_two_channel._bbox_groups_from_token_ids",
            lambda **kwargs: [[0, 1, 2, 3] for _ in kwargs["gt_objs"]],
        )

        samples = [
            {
                "messages": [],
                "assistant_payload": {
                    "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "gt"}]
                },
            }
            for _ in range(3)
        ]
        _segments, _batch_metrics = t._prepare_batch_inputs_b(
            samples,
            _segments_only=True,
        )

    assert rollout_calls == [
        (2, 0.0),
        (2, 0.7),
        (1, 0.0),
        (1, 0.7),
    ]


def test_channel_b_enabled_pseudo_positive_uses_k4_rollouts_and_keeps_zero_object_explorer(
    monkeypatch,
) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    ab_cfg = {
        "pseudo_positive.enabled": True,
        "triage_posterior.num_rollouts": 4,
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.explorer_top_p": 0.95,
        "triage_posterior.explorer_top_k": -1,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: ab_cfg.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls: list[dict[str, float]] = []

    def _fake_rollout_many(
        chunk, decode_override=None, request_index_offset=0
    ):
        rollout_calls.append(
            {
                "temperature": float(
                    (decode_override or {}).get("temperature", 0.0) or 0.0
                ),
                "request_index_offset": float(request_index_offset),
            }
        )
        marker = 101 + len(rollout_calls)
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    def _parse_with_optional_empty_view(**kwargs):
        marker = int(kwargs["response_token_ids"][0])
        valid_objects = []
        if marker in {102, 103, 105}:
            valid_objects = [
                types.SimpleNamespace(
                    index=0,
                    geom_type="bbox_2d",
                    coord_token_indices=[0, 1, 2, 3],
                    desc="obj",
                )
            ]
        return types.SimpleNamespace(
            prefix_token_ids=[],
            prefix_text='{"objects": [',
            response_token_ids=list(kwargs["response_token_ids"]),
            response_text="",
            valid_objects=valid_objects,
            dropped_invalid_by_reason={},
            dropped_invalid=0,
            dropped_ambiguous=0,
            truncated=False,
            invalid_rollout=False,
        )

    with monkeypatch.context() as mp:
        mp.setattr(
            "src.trainers.stage2_two_channel.parse_rollout_for_matching",
            _parse_with_optional_empty_view,
        )
        mp.setattr(
            "src.trainers.stage2_two_channel.points_from_coord_tokens",
            lambda **kwargs: [10, 10, 20, 20],
        )
        mp.setattr(
            "src.trainers.stage2_two_channel._extract_gt_bboxonly",
            lambda _sample: [],
        )
        mp.setattr(
            "src.trainers.stage2_two_channel.hungarian_match_maskiou",
            lambda **kwargs: types.SimpleNamespace(
                matched_pairs=[],
                fn_gt_indices=[],
                fp_pred_indices=list(range(len(kwargs["preds"]))),
                gating_rejections=0,
                matched_maskiou_sum=0.0,
                matched_maskiou_count=0,
            ),
        )

        sample = {"messages": [], "assistant_payload": {"objects": []}}
        segments, batch_metrics = t._prepare_batch_inputs_b(
            [sample],
            _segments_only=True,
        )

    assert [call["temperature"] for call in rollout_calls] == [
        pytest.approx(0.0),
        pytest.approx(0.7),
        pytest.approx(0.7),
        pytest.approx(0.7),
    ]
    assert [call["request_index_offset"] for call in rollout_calls] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(1.0),
        pytest.approx(2.0),
    ]

    meta = segments[0][1]
    assert meta["shielded_anchor_indices"] == []
    assert meta["dead_anchor_indices"] == []
    assert meta["pseudo_positive_anchor_indices"] == [0]
    assert meta["valid_explorer_count"] == 3
    assert meta["anchor_support_counts"] == [2]
    assert meta["anchor_support_rates"] == pytest.approx([2.0 / 3.0])
    assert batch_metrics["stage2/raw_rollouts"] == pytest.approx(4.0)
    assert batch_metrics["rollout/explorer/pred_objects"] == pytest.approx(2.0 / 3.0)
    assert batch_metrics["rollout/explorer/valid_pred_objects"] == pytest.approx(
        2.0 / 3.0
    )
    assert batch_metrics["rollout/explorer/parse_truncated_rate"] == pytest.approx(0.0)


def test_channel_b_enabled_pseudo_positive_aborts_on_invalid_explorer(
    monkeypatch,
) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    ab_cfg = {
        "pseudo_positive.enabled": True,
        "triage_posterior.num_rollouts": 4,
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.explorer_top_p": 0.95,
        "triage_posterior.explorer_top_k": -1,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: ab_cfg.get(key, default)

    tok = _DummyTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls = 0

    def _fake_rollout_many(
        chunk, decode_override=None, request_index_offset=0
    ):
        nonlocal rollout_calls
        rollout_calls += 1
        marker = 101 + rollout_calls
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    def _parse_with_invalid_middle_explorer(**kwargs):
        marker = int(kwargs["response_token_ids"][0])
        valid_objects = [
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="obj",
            )
        ]
        invalid_rollout = marker == 104
        return types.SimpleNamespace(
            prefix_token_ids=[],
            prefix_text='{"objects": [',
            response_token_ids=list(kwargs["response_token_ids"]),
            response_text="",
            valid_objects=[] if invalid_rollout else valid_objects,
            dropped_invalid_by_reason={},
            dropped_invalid=0,
            dropped_ambiguous=0,
            truncated=False,
            invalid_rollout=invalid_rollout,
        )

    with monkeypatch.context() as mp:
        mp.setattr(
            "src.trainers.stage2_two_channel.parse_rollout_for_matching",
            _parse_with_invalid_middle_explorer,
        )
        mp.setattr(
            "src.trainers.stage2_two_channel.points_from_coord_tokens",
            lambda **kwargs: [10, 10, 20, 20],
        )
        mp.setattr(
            "src.trainers.stage2_two_channel._extract_gt_bboxonly",
            lambda _sample: [],
        )
        mp.setattr(
            "src.trainers.stage2_two_channel.hungarian_match_maskiou",
            lambda **kwargs: types.SimpleNamespace(
                matched_pairs=[],
                fn_gt_indices=[],
                fp_pred_indices=list(range(len(kwargs["preds"]))),
                gating_rejections=0,
                matched_maskiou_sum=0.0,
                matched_maskiou_count=0,
            ),
        )

        sample = {
            "messages": [],
            "assistant_payload": {"objects": []},
            "sample_id": "sample-0",
            "image_id": "image-0",
        }
        with pytest.raises(
            ValueError,
            match=(
                r"invalid_labels=\['explorer_1'\].*"
                r"sample_id=sample-0.*image_id=image-0.*manual_analysis_required=true"
            ),
        ):
            t._prepare_batch_inputs_b([sample], _segments_only=True)


def test_channel_b_triage_enabled_k2_remains_no_promotion_control() -> None:
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 10, 20, 20],
            desc="obj",
        )
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 10, 20, 20],
                desc="obj",
            )
        ]
    ]

    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_channel_b_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    assert triage.valid_explorer_count == 1
    assert triage.anchor_support_counts == [1]
    assert triage.anchor_support_rates == pytest.approx([1.0])
    assert triage.pseudo_positive_candidate_indices == []
    assert triage.pseudo_positive_anchor_indices == []
    assert triage.shielded_anchor_indices == [0]
    assert triage.dead_anchor_indices == []


def test_channel_b_triage_clusters_pseudo_positive_candidates_by_support_rate() -> None:
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[0, 0, 100, 100],
            desc="obj-a",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[0, 0, 100, 90],
            desc="obj-b",
        ),
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="obj-a",
            ),
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 90],
                desc="obj-b",
            ),
        ],
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="obj-a",
            ),
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 90],
                desc="obj-b",
            ),
        ],
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="obj-a",
            )
        ],
    ]

    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_channel_b_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    assert triage.valid_explorer_count == 3
    assert triage.anchor_support_counts == [3, 2]
    assert triage.anchor_support_rates == pytest.approx([1.0, 2.0 / 3.0])
    assert triage.pseudo_positive_candidate_indices == [0, 1]
    assert triage.pseudo_positive_anchor_indices == [0]
    assert triage.pseudo_positive_cluster_demoted_indices == [1]
    assert triage.shielded_anchor_indices == [1]
    assert triage.dead_anchor_indices == []


def test_channel_b_supervision_targets_make_pseudo_positive_coord_only_and_anchor_owned() -> None:
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="obj",
        )
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="obj",
            )
        ],
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="obj",
            )
        ],
        [],
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_channel_b_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_channel_b_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert triage.pseudo_positive_anchor_indices == [0]
    assert targets.prefix_struct_pos == []
    assert targets.tail_desc_pos == []
    assert targets.fn_bbox_groups == []
    assert len(targets.prefix_bbox_groups) == 1
    assert targets.prefix_bbox_groups[0]["gt_bins"] == [10, 20, 30, 40]
    assert targets.prefix_bbox_groups[0]["weight"] == pytest.approx(0.4)
    assert targets.prefix_bins == [10, 20, 30, 40]


def test_channel_b_supervision_targets_allow_partial_pseudo_positive_coord_for_shielded_anchor() -> None:
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="obj",
        )
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="obj",
            )
        ],
        [],
        [],
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_channel_b_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_channel_b_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.6,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert triage.pseudo_positive_anchor_indices == []
    assert triage.shielded_anchor_indices == [0]
    assert targets.prefix_struct_pos == []
    assert targets.tail_desc_pos == []
    assert targets.fn_bbox_groups == []
    assert len(targets.prefix_bbox_groups) == 1
    assert targets.prefix_bbox_groups[0]["gt_bins"] == [10, 20, 30, 40]
    assert targets.prefix_bbox_groups[0]["weight"] == pytest.approx(0.6 * (1.0 / 3.0))
    assert targets.prefix_bins == [10, 20, 30, 40]


def test_channel_b_supervision_targets_skip_duplicate_burst_unlikelihood_for_non_duplicate_dead_anchor() -> None:
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="kept",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[100, 200, 300, 400],
            desc="far-away",
        ),
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="kept",
            )
        ],
        [],
        [],
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_channel_b_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_channel_b_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert [obj.index for obj in accepted_clean] == [0, 1]
    assert triage.dead_anchor_indices == [1]
    assert targets.duplicate_burst_unlikelihood_targets == []
    assert targets.duplicate_burst_unlikelihood_boundary_count == 0


def test_channel_b_supervision_targets_keep_duplicate_burst_unlikelihood_when_duplicate_survivor_is_kept() -> None:
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="dup",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="dup",
        ),
    ]
    explorer_objects_by_view = [
        [
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[10, 20, 30, 40],
                desc="dup",
            )
        ],
        [],
        [],
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_channel_b_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=explorer_objects_by_view,
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_channel_b_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert [obj.index for obj in accepted_clean] == [0]
    assert triage.dead_anchor_indices == []
    assert targets.duplicate_burst_unlikelihood_targets
    assert targets.duplicate_burst_unlikelihood_boundary_count == 1


def test_channel_b_supervision_targets_keep_duplicate_burst_unlikelihood_when_all_cluster_members_die() -> None:
    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()
    raw_anchor_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="dup",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="dup",
        ),
    ]
    accepted_clean, duplicate_bursts_by_boundary = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=raw_anchor_objects,
        duplicate_iou_threshold=0.9,
    )
    triage = _build_channel_b_triage(
        accepted_objects_clean=accepted_clean,
        duplicate_bursts_by_boundary=duplicate_bursts_by_boundary,
        explorer_accepted_objects_clean_by_view=[[], [], []],
        anchor_match_by_pred={},
        explorer_match_by_pred_by_view=[{}, {}, {}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )

    targets = _build_channel_b_supervision_targets(
        tokenizer=tok,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=[],
        match=types.SimpleNamespace(matched_pairs=[]),
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        duplicate_iou_threshold=0.9,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )

    assert [obj.index for obj in accepted_clean] == [0]
    assert triage.dead_anchor_indices == [0]
    assert triage.kept_anchor_objects == []
    assert sorted(triage.duplicate_bursts_by_boundary.keys()) == [0]
    assert targets.duplicate_burst_unlikelihood_targets
    assert targets.duplicate_burst_unlikelihood_boundary_count == 1

def test_channel_b_triage_posterior_nested_config_reaches_live_accessor_and_vllm_offsets(
    monkeypatch,
) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "sampling",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)

    training_cfg = ConfigLoader.load_materialized_training_config(
        "configs/stage2_two_channel/base.yaml"
    )
    t.stage2_ab_cfg = asdict(training_cfg.stage2_ab)
    t.stage2_ab_cfg["channel_b"]["triage_posterior"]["explorer_temperature"] = 0.55
    t.stage2_ab_cfg["channel_b"]["triage_posterior"]["explorer_top_p"] = 0.91
    t.stage2_ab_cfg["channel_b"]["triage_posterior"]["explorer_top_k"] = 7
    t.stage2_ab_cfg["channel_b"]["triage_posterior"]["unlabeled_consistent_iou_threshold"] = 0.82
    t.stage2_ab_cfg["channel_b"]["triage_posterior"]["recovered_ground_truth_weight_multiplier"] = 3.0

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 64
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "vllm"
    t._rollout_decode_batch_size_per_rank = lambda: 2
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()

    rollout_calls: list[tuple[int, float, int]] = []

    def _fake_rollout_many(chunk, decode_override=None, request_index_offset=0):
        rollout_calls.append(
            (
                int(len(chunk)),
                float((decode_override or {}).get("temperature", 0.0) or 0.0),
                int(request_index_offset),
            )
        )
        marker = (
            101
            if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0
            else 202
        )
        return [([marker], "", "sampling", []) for _ in chunk]

    t._rollout_many = _fake_rollout_many

    fake_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[],
        response_text="",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    with monkeypatch.context() as mp:
        mp.setattr(
            "src.trainers.stage2_two_channel.parse_rollout_for_matching",
            lambda **kwargs: fake_parse,
        )
        mp.setattr(
            "src.trainers.stage2_two_channel._extract_gt_bboxonly",
            lambda _sample: [
                GTObject(
                    index=0,
                    geom_type="bbox_2d",
                    points_norm1000=[0, 0, 1, 1],
                    desc="gt",
                )
            ],
        )
        mp.setattr(
            "src.trainers.stage2_two_channel.hungarian_match_maskiou",
            lambda **kwargs: types.SimpleNamespace(
                matched_pairs=[],
                fn_gt_indices=[],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=0.0,
                matched_maskiou_count=0,
            ),
        )
        mp.setattr(
            "src.trainers.stage2_two_channel._bbox_groups_from_token_ids",
            lambda **kwargs: [[0, 1, 2, 3] for _ in kwargs["gt_objs"]],
        )

        samples = [
            {
                "messages": [],
                "assistant_payload": {
                    "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": "gt"}]
                },
            }
            for _ in range(5)
        ]
        _segments, _batch_metrics = t._prepare_batch_inputs_b(
            samples,
            _segments_only=True,
        )

    assert t._ab_channel_b_get("triage_posterior.explorer_temperature", None) == pytest.approx(
        0.55
    )
    assert t._ab_channel_b_get("triage_posterior.recovered_ground_truth_weight_multiplier", None) == pytest.approx(
        3.0
    )
    assert rollout_calls == [
        (2, 0.0, 0),
        (2, 0.55, 0),
        (2, 0.0, 2),
        (2, 0.55, 2),
        (1, 0.0, 4),
        (1, 0.55, 4),
    ]


def test_channel_b_anchor_only_gt_hit_projects_anchor_gt_backed(
    monkeypatch,
) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()
    t._rollout_many = lambda chunk, decode_override=None: [
        ([101 if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0 else 202], "", "sampling", [])
        for _ in chunk
    ]

    shared_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="anchor",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="anchor",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: shared_parse,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )

    call_idx = {"n": 0}

    def _fake_match(*, preds, **kwargs):
        idx = int(call_idx["n"])
        call_idx["n"] += 1
        if idx == 0:
            return types.SimpleNamespace(
                matched_pairs=[(0, 0)],
                fn_gt_indices=[],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=1.0,
                matched_maskiou_count=1,
            )
        return types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[0],
            fp_pred_indices=[0],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [{"bbox_2d": [10, 10, 20, 20], "desc": "gt"}],
        },
    }

    segments, batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)
    meta = segments[0][1]

    assert meta["anchor_gt_backed_indices"] == [0]
    assert meta["recovered_gt_indices"] == []
    assert meta["dead_anchor_indices"] == []
    assert meta["fn_object_weights"] == []
    assert meta["bbox_groups_prefix"]
    assert batch_metrics["train/triage/gt_backed_count"] == pytest.approx(
        1.0
    )


def test_channel_b_shielded_anchor_stays_neutral_context(monkeypatch) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()
    t._rollout_many = lambda chunk, decode_override=None: [
        ([101 if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0 else 202], "", "sampling", [])
        for _ in chunk
    ]

    shared_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="shield",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="shield",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: shared_parse,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel._extract_gt_bboxonly",
        lambda _sample: [],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=[0],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    sample = {"messages": [], "assistant_payload": {"objects": []}}
    segments, batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)
    meta = segments[0][1]

    assert meta["shielded_anchor_indices"] == [0]
    assert meta["anchor_gt_backed_indices"] == []
    assert meta["dead_anchor_indices"] == []
    assert meta["prefix_struct_pos"] == []
    assert meta["bbox_groups_prefix"] == []
    assert meta["bbox_groups_fn"] == []
    assert batch_metrics["train/triage/unlabeled_consistent_count"] == pytest.approx(
        1.0
    )


def test_channel_b_explorer_only_dead_emits_no_explore_branch(monkeypatch) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()
    t._rollout_many = lambda chunk, decode_override=None: [
        ([101 if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0 else 202], "", "sampling", [])
        for _ in chunk
    ]

    anchor_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="anchor",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    explorer_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[202],
        response_text="explorer",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="explorer-only",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: (
            anchor_parse
            if int(kwargs["response_token_ids"][0]) == 101
            else explorer_parse
        ),
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel._extract_gt_bboxonly",
        lambda _sample: [],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        lambda **kwargs: types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=list(range(len(kwargs["preds"]))),
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        ),
    )

    sample = {"messages": [], "assistant_payload": {"objects": []}}
    segments, batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)
    meta = segments[0][1]

    assert meta["dead_explorer_indices_by_view"] == [[0]]
    assert meta["anchor_gt_backed_indices"] == []
    assert meta["shielded_anchor_indices"] == []
    assert meta["dead_anchor_indices"] == []
    assert meta["bbox_groups_prefix"] == []
    assert meta["bbox_groups_fn"] == []
    assert meta["duplicate_burst_unlikelihood_targets"] == []
    assert batch_metrics["train/triage/explorer_only_dead_count"] == pytest.approx(
        1.0
    )


def test_channel_b_recovered_ground_truth_weight_multipliers_only_apply_to_recovered_tail_objects(
    monkeypatch,
) -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    cfg = {
        "maskiou_gate": 0.3,
        "candidate_top_k": 5,
        "maskiou_resolution": 64,
        "fp_cost": 1.0,
        "fn_cost": 1.0,
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    t._cfg = lambda key, default=None: cfg.get(key, default)
    t._ab_channel_b_get = lambda key, default=None: {
        "triage_posterior.explorer_temperature": 0.7,
        "triage_posterior.recovered_ground_truth_weight_multiplier": 2.5,
        "triage_posterior.unlabeled_consistent_iou_threshold": 0.8,
    }.get(key, default)

    class _CoordLiteralTokenizer(_DummyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False):
            s = str(text)
            out: list[int] = []
            i = 0
            while i < len(s):
                if s.startswith("<|coord_", i):
                    j = s.find("|>", i)
                    if j >= 0:
                        out.extend(super().encode(s[i : j + 2], add_special_tokens=False))
                        i = j + 2
                        continue
                out.append(self._id_for(s[i]))
                i += 1
            return out

    tok = _CoordLiteralTokenizer()

    class _FakeTemplate:
        tokenizer = tok

        def encode(self, data, return_length=True):
            assistant_ids = [int(x) for x in data["messages"][-1]["content"]]
            return {
                "input_ids": list(assistant_ids),
                "labels": list(assistant_ids),
                "length": len(assistant_ids),
            }

    t.template = _FakeTemplate()
    t._template_train_mode = lambda: nullcontext()
    t._extract_encoded_len = lambda encoded: int(len(encoded["input_ids"]))
    t._get_coord_token_ids = lambda: list(range(1000))
    t._coord_id_map = lambda: {i: i for i in range(1000)}
    t._packing_enabled = lambda: False
    t._packing_drop_last = lambda: True
    t._packing_buffer_cap = lambda: 1
    t._packing_length = lambda: 256
    t._derive_rollout_seed_base = lambda *, global_step: 0
    t._rollout_backend = lambda: "hf"
    t._rollout_decode_batch_size_per_rank = lambda: 1
    t._dist_info = lambda: (0, 1, None)
    t._object_field_order = lambda: "desc_first"
    t._stage2_train_monitor_step_allowed = lambda global_step: False
    t.state = types.SimpleNamespace(global_step=0)

    class _NoSeedCtx:
        def __enter__(self):
            return False

        def __exit__(self, exc_type, exc, tb):
            return False

    t._hf_sampling_seed_context = lambda **kwargs: _NoSeedCtx()
    t._rollout_many = lambda chunk, decode_override=None: [
        ([101 if float((decode_override or {}).get("temperature", 0.0) or 0.0) <= 0.0 else 202], "", "sampling", [])
        for _ in chunk
    ]

    anchor_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[101],
        response_text="anchor",
        valid_objects=[],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    explorer_parse = types.SimpleNamespace(
        prefix_token_ids=[],
        prefix_text='{"objects": [',
        response_token_ids=[202],
        response_text="explorer",
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="recovered",
            )
        ],
        dropped_invalid_by_reason={},
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
        invalid_rollout=False,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.parse_rollout_for_matching",
        lambda **kwargs: (
            anchor_parse
            if int(kwargs["response_token_ids"][0]) == 101
            else explorer_parse
        ),
    )
    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.points_from_coord_tokens",
        lambda **kwargs: [10, 10, 20, 20],
    )

    def _fake_match(*, preds, gts, **kwargs):
        if preds and str(preds[0].desc) == "recovered":
            return types.SimpleNamespace(
                matched_pairs=[(0, 0)],
                fn_gt_indices=[1],
                fp_pred_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=1.0,
                matched_maskiou_count=1,
            )
        return types.SimpleNamespace(
            matched_pairs=[],
            fn_gt_indices=[0, 1],
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    monkeypatch.setattr(
        "src.trainers.stage2_two_channel.hungarian_match_maskiou",
        _fake_match,
    )

    sample = {
        "messages": [],
        "assistant_payload": {
            "objects": [
                {"bbox_2d": [10, 10, 20, 20], "desc": "recovered-gt"},
                {"bbox_2d": [30, 30, 40, 40], "desc": "ordinary-fn"},
            ]
        },
    }

    segments, _batch_metrics = t._prepare_batch_inputs_b([sample], _segments_only=True)
    meta = segments[0][1]

    assert meta["recovered_gt_indices"] == [0]
    assert meta["fn_object_weights"] == [pytest.approx(2.5), pytest.approx(1.0)]
    assert [group["weight"] for group in meta["bbox_groups_fn"]] == [
        pytest.approx(2.5),
        pytest.approx(1.0),
    ]
    assert {round(float(weight), 2) for weight in meta["tail_desc_weights"]} == {
        1.0,
        2.5,
    }


def test_channel_b_tail_desc_weights_scale_desc_ce() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        rollout_fn_desc_weight=1.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[1100, 1100, 1101, 1101, 1100, 1100]], dtype=torch.long)

    meta_base = {
        "prompt_len": 0,
        "prefix_len": 2,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [0],
        "tail_desc_pos": [0, 1],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    loss_default = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta_base)],
            "input_ids": input_ids,
        },
    )

    meta_weighted = dict(meta_base)
    meta_weighted["tail_desc_weights"] = [2.0, 2.0]
    loss_weighted = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_weighted],
            "input_ids": input_ids,
        },
    )

    assert float(loss_weighted.detach().cpu().item()) > float(
        loss_default.detach().cpu().item()
    )


def test_channel_b_bbox_group_weights_scale_geo_loss() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 0.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=1.0,
        bbox_smoothl1_weight=1.0,
        bbox_ciou_weight=1.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=0)
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    meta = {
        "prompt_len": 0,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [],
        "tail_desc_pos": [],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [
            {"pos": [1, 2, 3, 4], "gt_bins": [0, 0, 0, 0], "weight": 1.0},
            {"pos": [5, 6, 7, 8], "gt_bins": [999, 999, 999, 999], "weight": 1.0},
        ],
    }

    loss_default = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    meta_weighted = dict(meta)
    meta_weighted["bbox_groups_fn"] = [
        {"pos": [1, 2, 3, 4], "gt_bins": [0, 0, 0, 0], "weight": 1.0},
        {"pos": [5, 6, 7, 8], "gt_bins": [999, 999, 999, 999], "weight": 4.0},
    ]
    loss_weighted = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_weighted],
            "input_ids": input_ids,
        },
    )

    assert float(loss_weighted.detach().cpu().item()) > float(
        loss_default.detach().cpu().item()
    )


def test_channel_b_coord_slot_weights_scale_coord_reg_loss() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 0.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=0.0,
        bbox_smoothl1_weight=1.0,
        bbox_ciou_weight=1.0,
        coord_reg_enabled=True,
        coord_reg_weight=1.0,
        coord_ce_weight=1.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=0)
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    meta = {
        "prompt_len": 0,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [],
        "tail_desc_pos": [],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [
            {"pos": [1, 2, 3, 4], "gt_bins": [0, 0, 0, 0], "weight": 1.0},
            {"pos": [5, 6, 7, 8], "gt_bins": [999, 999, 999, 999], "weight": 1.0},
        ],
    }

    loss_default = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    meta_weighted = dict(meta)
    meta_weighted["bbox_groups_fn"] = [
        {"pos": [1, 2, 3, 4], "gt_bins": [0, 0, 0, 0], "weight": 1.0},
        {"pos": [5, 6, 7, 8], "gt_bins": [999, 999, 999, 999], "weight": 4.0},
    ]
    loss_weighted = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_weighted],
            "input_ids": input_ids,
        },
    )

    assert float(loss_weighted.detach().cpu().item()) > float(
        loss_default.detach().cpu().item()
    )


def test_derive_rollout_seed_base_is_deterministic_and_matches_formula():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.args = types.SimpleNamespace(seed=123)

    out1 = t._derive_rollout_seed_base(global_step=7)
    out2 = t._derive_rollout_seed_base(global_step=7)

    expected = int((123 + 7 * 1000003) & 0x7FFFFFFF)
    assert out1 == expected
    assert out2 == expected
    assert 0 <= out1 <= 0x7FFFFFFF


def test_hf_sampling_seeding_calls_seed_everything(monkeypatch):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)

    # Verify we call transformers.trainer_utils.set_seed(...) during HF sampling seeding.
    called = {}

    def _fake_set_seed(seed: int):
        called["seed"] = int(seed)

    tu = pytest.importorskip("transformers.trainer_utils")
    monkeypatch.setattr(tu, "set_seed", _fake_set_seed, raising=True)

    with t._hf_sampling_seed_context(
        seed_base=123, backend="hf", do_sample=True
    ) as seeded:
        assert seeded is True
    assert called["seed"] == 123


def test_hf_sampling_seeding_restores_python_rng_state():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)

    import random

    orig = random.getstate()
    try:
        random.seed(0)
        saved = random.getstate()

        with t._hf_sampling_seed_context(
            seed_base=123, backend="hf", do_sample=True
        ) as seeded:
            assert seeded is True
            _ = random.random()  # mutate RNG under the seeded context

        # After exit, RNG state should be restored to `saved`.
        assert random.getstate() == saved
    finally:
        random.setstate(orig)


def test_packing_enabled_requires_qwen_packing_metadata():
    trainer = _make_min_trainer()
    trainer.rollout_matching_cfg = {"packing_enabled": True}
    model = _DummyModel()

    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    with pytest.raises(ValueError, match="packing enabled"):
        trainer.compute_loss(
            model,
            {
                "_stage2_ab_channel": "A",
                "_rollout_matching_meta": meta,
                "input_ids": input_ids,
            },
        )


def test_post_rollout_packing_selector_is_remainder_aware():
    pytest.importorskip("binpacking")

    from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer

    packing_length = 10
    min_fill_ratio = 0.5
    target_len = int(math.ceil(float(min_fill_ratio) * float(packing_length)))

    # Construct a small pool where FIFO-greedy produces a tiny remainder pack,
    # but a stage-1-like "smart" plan can avoid underfill by leaving a small
    # segment to pair with the leftover medium segment.
    lens = [9, 6, 4, 4, 1]

    def _legacy_fifo(buf_lens: Sequence[int]) -> List[int]:
        used = int(buf_lens[0])
        sel = [0]
        for i in range(1, len(buf_lens)):
            sl = int(buf_lens[i])
            if sl <= 0:
                continue
            if used + sl <= int(packing_length):
                sel.append(int(i))
                used += sl
        return sel

    def _simulate(buf_lens: Sequence[int], *, selector) -> int:
        buf = [int(x) for x in buf_lens]
        underfilled = 0
        while buf:
            idx = selector(buf)
            assert idx
            assert idx[0] == 0

            total = int(sum(int(buf[i]) for i in idx))
            if total < int(target_len):
                underfilled += 1

            for i in reversed(idx):
                buf.pop(int(i))
        return int(underfilled)

    legacy_underfilled = _simulate(lens, selector=_legacy_fifo)

    def _smart(buf_lens: Sequence[int]) -> List[int]:
        return RolloutMatchingSFTTrainer._select_post_rollout_segment_indices(
            buf_lens,
            packing_length,
            min_fill_ratio=min_fill_ratio,
        )

    smart_underfilled = _simulate(lens, selector=_smart)

    assert legacy_underfilled == 1
    assert smart_underfilled == 0



def test_extract_gt_bboxonly_rejects_poly_geometry():
    sample = {
        "assistant_payload": {"objects": [{"desc": "x", "poly": [0, 1, 2, 3]}]}
    }
    with pytest.raises(ValueError, match="bbox-only v1"):
        _extract_gt_bboxonly(sample)


def test_extract_gt_bboxonly_rejects_other_geometry_key_even_with_bbox():
    sample = {
        "assistant_payload": {
            "objects": [
                {
                    "desc": "x",
                    "bbox_2d": [0, 0, 10, 10],
                    "mask_rle": {"counts": "abc", "size": [1, 1]},
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="bbox-only v1"):
        _extract_gt_bboxonly(sample)


def test_extract_gt_bboxonly_rejects_invalid_bbox_order():
    sample = {
        "assistant_payload": {
            "objects": [{"desc": "x", "bbox_2d": [10, 10, 0, 0]}]
        },
    }
    with pytest.raises(ValueError, match="invalid bbox_2d"):
        _extract_gt_bboxonly(sample)


def test_build_teacher_forced_payload_honors_object_field_order():
    gt_objects = [
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[1, 2, 3, 4],
            desc="cat",
        )
    ]

    desc_first = _build_teacher_forced_payload(
        gt_objects=gt_objects, object_field_order="desc_first"
    )
    geometry_first = _build_teacher_forced_payload(
        gt_objects=gt_objects, object_field_order="geometry_first"
    )

    assert list(desc_first["objects"][0].keys()) == ["desc", "bbox_2d"]
    assert list(geometry_first["objects"][0].keys()) == ["bbox_2d", "desc"]


def test_extract_gt_bboxonly_preserves_assistant_payload_order() -> None:
    sample = {
        "assistant_payload": {
            "objects": [
                {"desc": "later", "bbox_2d": [10, 10, 20, 20]},
                {"desc": "earlier", "bbox_2d": [0, 0, 5, 5]},
            ]
        }
    }

    gt_objects = _extract_gt_bboxonly(sample)

    assert [obj.desc for obj in gt_objects] == ["later", "earlier"]
    assert [obj.index for obj in gt_objects] == [0, 1]


def test_build_canonical_prefix_text_data_preserves_object_sequence() -> None:
    gt_objects = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[10, 10, 20, 20],
            desc="later",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[0, 0, 5, 5],
            desc="earlier",
        ),
    ]

    prefix_text, boundary_prefix_texts, _spans = _build_canonical_prefix_text_data(
        objects=gt_objects,
        object_field_order="desc_first",
    )

    assert prefix_text.index("later") < prefix_text.index("earlier")
    assert "later" in boundary_prefix_texts[1]
    assert "earlier" not in boundary_prefix_texts[1]


def test_stage2_channel_b_fragment_supports_geometry_first_order():
    frag = _serialize_append_fragment(
        fn_objects=[
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[5, 6, 7, 8],
                desc="fn",
            )
        ],
        prefix_text='{"objects": [',
        object_field_order="geometry_first",
    )
    assert frag.index('"bbox_2d"') < frag.index('"desc"')


def test_compute_loss_raises_on_sliced_logits():
    trainer = _make_min_trainer()
    model = _DummySlicedModel()

    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    position_ids = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
    text_position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    with pytest.raises(ValueError, match="sliced logits"):
        trainer.compute_loss(
            model,
            {
                "_stage2_ab_channel": "A",
                "_rollout_matching_meta": meta,
                "input_ids": input_ids,
                "position_ids": position_ids,
                "text_position_ids": text_position_ids,
            },
        )



def test_channel_b_includes_fn_geometry_loss():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        # Turn off CE so we isolate geometry contribution.
        "desc_ce_weight": 0.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=1.0,
        bbox_smoothl1_weight=1.0,
        bbox_ciou_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyConstantCoord999Model()
    input_ids = torch.tensor([[1100, 1100, 0, 1, 2, 3]], dtype=torch.long)

    # Only FN groups are present in Channel-B metadata; this should still contribute
    # geometry loss under the unified one-pass contract.
    meta = [
        {
            "prompt_len": 0,
            "prefix_len": 0,
            "train_len": int(input_ids.shape[1]),
            "encoded_len": int(input_ids.shape[1]),
            "tail_ignore_pos": [],
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [{"pos": [2, 3, 4, 5], "gt_bins": [0, 0, 0, 0]}],
        }
    ]

    loss = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
        },
    )
    assert float(loss.detach().cpu().item()) > 0.0


def test_stage2_coord_soft_ce_w1_adds_coord_distribution_loss() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 0.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    soft_ce_weight = 0.25
    w1_weight = 0.25

    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        # Must run bbox_geo to populate coord_reg state, but don't let it contribute to total loss.\n        bbox_geo_weight=0.0,
        bbox_smoothl1_weight=0.0,
        bbox_ciou_weight=0.0,
        coord_reg_enabled=True,
        coord_reg_weight=1.0,
        coord_soft_ce_weight=float(soft_ce_weight),
        coord_w1_weight=float(w1_weight),
    )

    model = _DummyConstantCoord999Model()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    meta = [
        {
            "prompt_len": 0,
            "prefix_len": 0,
            "train_len": int(input_ids.shape[1]),
            "encoded_len": int(input_ids.shape[1]),
            "tail_ignore_pos": [],
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [{"pos": [1, 2, 3, 4], "gt_bins": [0, 0, 0, 0]}],
        }
    ]

    loss = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
        },
    )

    assert float(loss.detach().cpu().item()) > 0.0

    pending = t._stage2_pending_train_logs.get(1)
    assert pending is not None
    finalized = pending.finalize()
    assert float(finalized["loss/B_coord/coord_soft_ce"]) > 0.0
    assert float(finalized["loss/B_coord/coord_w1"]) > 0.0
    assert "loss/B_coord/coord_reg" not in finalized
    assert "loss/coord_soft_ce" not in finalized
    assert "loss/coord_w1" not in finalized
    assert "loss/coord_reg" not in finalized


def test_stage2_coord_soft_ce_w1_disabled_contributes_zero() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 0.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=0.0,
        bbox_smoothl1_weight=0.0,
        bbox_ciou_weight=0.0,
        coord_reg_enabled=True,
        coord_reg_weight=1.0,
        coord_soft_ce_weight=0.0,
        coord_w1_weight=0.0,
    )

    model = _DummyAlwaysTokenModel(pred_id=999)
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    meta = [
        {
            "prompt_len": 0,
            "prefix_len": 0,
            "train_len": int(input_ids.shape[1]),
            "encoded_len": int(input_ids.shape[1]),
            "tail_ignore_pos": [],
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [{"pos": [1, 2, 3, 4], "gt_bins": [0, 0, 0, 0]}],
        }
    ]

    loss = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
        },
    )

    assert float(loss.detach().cpu().item()) == pytest.approx(0.0)

    pending = t._stage2_pending_train_logs.get(1)
    assert pending is not None
    finalized = pending.finalize()

    assert "loss/B_coord/coord_soft_ce" not in finalized
    assert "loss/B_coord/coord_w1" not in finalized
    assert "loss/B_coord/coord_reg" not in finalized
    assert "loss/coord_soft_ce" not in finalized
    assert "loss/coord_w1" not in finalized
    assert "loss/coord_reg" not in finalized


def test_channel_a_bbox_size_aux_logs_bbox_log_wh_immediately() -> None:
    t = _make_min_trainer()
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=0.0,
        bbox_smoothl1_weight=0.0,
        bbox_ciou_weight=0.0,
        bbox_size_aux_enabled=True,
        bbox_size_aux_weight=1.0,
        bbox_log_wh_weight=0.05,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )

    model = _DummyConstantCoord999Model()
    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    position_ids = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
    text_position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    loss = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "A",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "text_position_ids": text_position_ids,
        },
    )

    assert float(loss.detach().cpu().item()) > 0.0

    pending = t._stage2_pending_train_logs.get(1)
    assert pending is not None
    finalized = pending.finalize()
    assert float(finalized["loss/coord/bbox_log_wh"]) > 0.0


def test_channel_b_bbox_group_weights_scale_bbox_size_aux_loss() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 0.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=0.0,
        bbox_smoothl1_weight=0.0,
        bbox_ciou_weight=0.0,
        bbox_size_aux_enabled=True,
        bbox_size_aux_weight=1.0,
        bbox_log_wh_weight=0.05,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyConstantCoord999Model()
    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    meta = {
        "prompt_len": 2,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]) - 2,
        "encoded_len": int(input_ids.shape[1]),
        "tail_desc_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [
            {"pos": [2, 3, 4, 5], "gt_bins": [999, 999, 999, 999], "weight": 1.0},
            {"pos": [6, 7, 8, 9], "gt_bins": [0, 1, 2, 3], "weight": 1.0},
        ],
    }

    loss_default = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    meta_weighted = dict(meta)
    meta_weighted["bbox_groups_fn"] = [
        {"pos": [2, 3, 4, 5], "gt_bins": [999, 999, 999, 999], "weight": 1.0},
        {"pos": [6, 7, 8, 9], "gt_bins": [0, 1, 2, 3], "weight": 4.0},
    ]
    loss_weighted = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_weighted],
            "input_ids": input_ids,
        },
    )

    assert float(loss_weighted.detach().cpu().item()) > float(
        loss_default.detach().cpu().item()
    )


def test_channel_a_teacher_forcing_logits_drive_coord_losses_under_single_pass_names() -> None:
    t = _make_min_trainer()
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=1.0,
        bbox_smoothl1_weight=1.0,
        bbox_ciou_weight=1.0,
        bbox_size_aux_enabled=True,
        bbox_size_aux_weight=1.0,
        bbox_log_wh_weight=0.05,
        coord_reg_enabled=True,
        coord_reg_weight=1.0,
        coord_ce_weight=0.25,
        coord_soft_ce_weight=0.25,
        coord_w1_weight=0.25,
    )

    model = _DummyConstantCoord999Model()
    input_ids = torch.tensor([[1100, 1101, 0, 1, 2, 3, 1102]], dtype=torch.long)
    position_ids = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
    text_position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    meta = [
        {
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": int(input_ids.shape[1]),
            "tail_desc_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [
                {"pos": [2, 3, 4, 5], "gt_bins": [0, 1, 2, 3]},
            ],
        }
    ]

    loss = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "A",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "text_position_ids": text_position_ids,
        },
    )

    assert float(loss.detach().cpu().item()) > 0.0
    assert len(model.calls) == 1

    pending = t._stage2_pending_train_logs.get(1)
    assert pending is not None
    finalized = pending.finalize()

    assert float(finalized["loss/coord/bbox_smoothl1"]) > 0.0
    assert float(finalized["loss/coord/bbox_ciou"]) > 0.0
    assert float(finalized["loss/coord/bbox_log_wh"]) > 0.0
    assert float(finalized["loss/coord/coord_token_ce"]) > 0.0
    assert float(finalized["loss/coord/coord_soft_ce"]) > 0.0
    assert float(finalized["loss/coord/coord_w1"]) > 0.0
    assert not any(key.startswith("loss/A1_") for key in finalized)


def test_channel_b_unused_meta_flag_does_not_change_supervision_semantics() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        bbox_smoothl1_weight=1.0,
        bbox_ciou_weight=1.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[1100, 1100, 1101, 1101, 1100, 1100]], dtype=torch.long)

    base_meta = {
        "prompt_len": 0,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "tail_ignore_pos": [],
        "tail_desc_pos": [2, 3],
        "bbox_groups_prefix": [{"pos": [1, 2, 3, 4], "gt_bins": [1, 1, 2, 2]}],
        "bbox_groups_fn": [{"pos": [1, 2, 3, 4], "gt_bins": [1, 1, 2, 2]}],
    }

    loss_no_repeat = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(base_meta, legacy_unused_flag=0)],
            "input_ids": input_ids,
        },
    )
    loss_with_repeat = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(base_meta, legacy_unused_flag=1)],
            "input_ids": input_ids,
        },
    )

    assert float(loss_no_repeat.detach().cpu().item()) == pytest.approx(
        float(loss_with_repeat.detach().cpu().item())
    )


def test_channel_b_tail_ignore_pos_masks_ce_tokens():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[1100, 1100, 1101, 1101, 1100, 1100]], dtype=torch.long)

    meta_base = {
        "prompt_len": 0,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "tail_ignore_pos": [],
        "tail_desc_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    loss_full = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta_base)],
            "input_ids": input_ids,
        },
    )

    # Mask the two wrong tokens from CE.
    meta_mask = dict(meta_base)
    meta_mask["tail_ignore_pos"] = [2, 3]
    loss_masked = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_mask],
            "input_ids": input_ids,
        },
    )

    assert float(loss_masked.detach().cpu().item()) < float(
        loss_full.detach().cpu().item()
    )


def test_matched_prefix_structure_positions_excludes_desc_and_fp_tokens():
    tok = _DummyTokenizer()

    prefix_text = (
        '{"objects":[{"desc":"matched","bbox_2d":[0,0,1,1]},'
        '{"desc":"fp","bbox_2d":[2,2,3,3]}'
    )
    prefix_token_ids = list(tok.encode(prefix_text))

    first_obj_anchor = int(prefix_text.find('{"desc":"matched"'))
    value_start = int(first_obj_anchor)
    value_end = int(prefix_text.find('},{"desc":"fp"')) + 1
    matched_obj = types.SimpleNamespace(
        value_span=(value_start, value_end),
    )

    rel = _matched_prefix_structure_positions(
        tokenizer=tok,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        matched_pred_objects=[matched_obj],
    )

    struct_pos_matched = int(prefix_text.find('"bbox_2d"'))
    struct_pos_fp = int(prefix_text.find('{"desc":"fp"'))
    desc_pos_matched = int(prefix_text.find("matched"))

    assert struct_pos_matched >= 0 and struct_pos_matched in rel
    assert struct_pos_fp >= 0 and struct_pos_fp not in rel
    assert desc_pos_matched >= 0 and desc_pos_matched not in rel


def test_matched_prefix_structure_positions_uses_parser_char_frame_for_span_checks():
    tok = _PieceFrameMismatchTokenizer()

    source_text = '{"objects":[{"desc":"~","bbox_2d":[0,0,1,1]}'
    prefix_token_ids = list(tok.encode(source_text))

    prefix_text = tok.decode(
        prefix_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    piece_text = "".join(
        tok.decode(
            [int(t)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in prefix_token_ids
    )
    assert int(len(piece_text)) > int(len(prefix_text))

    value_start = int(piece_text.find('{"desc":'))
    value_end = int(piece_text.rfind("}")) + 1
    matched_obj = types.SimpleNamespace(
        value_span=(value_start, value_end),
    )

    rel = _matched_prefix_structure_positions(
        tokenizer=tok,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        matched_pred_objects=[matched_obj],
    )

    key_pos = int(piece_text.find('"bbox_2d"'))
    desc_tok_idx = next(
        i
        for i, t in enumerate(prefix_token_ids)
        if int(t) == int(tok._mismatch_id)
    )

    assert key_pos >= 0 and key_pos in rel
    assert desc_tok_idx not in rel


def test_channel_b_prefix_structure_supervision_uses_global_prefix_knob():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)

    # With the global prefix knob enabled, all non-coordinate prefix tokens
    # contribute structure CE regardless of prefix_struct_pos sparsity.
    input_ids = torch.tensor(
        [[1100, 1101, 1101, 1101, 1100, 1101, 1100, 1101]], dtype=torch.long
    )

    base_meta = {
        "prompt_len": 0,
        "prefix_len": 4,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "tail_ignore_pos": [],
        "tail_desc_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    meta_matched_only = dict(base_meta)
    meta_matched_only["prefix_struct_pos"] = [1]

    meta_oversupervised = dict(base_meta)
    meta_oversupervised["prefix_struct_pos"] = [1, 2, 3]

    loss_matched_only = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_matched_only],
            "input_ids": input_ids,
        },
    )

    loss_oversupervised = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_oversupervised],
            "input_ids": input_ids,
        },
    )

    assert float(loss_matched_only.detach().cpu().item()) == pytest.approx(
        float(loss_oversupervised.detach().cpu().item()),
        rel=1e-6,
        abs=1e-6,
    )


def test_channel_b_fn_desc_default_on_and_can_be_disabled_via_pipeline() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)

    # Prefix len=4 (only rel=1 is supervised struct); Tail len=4 includes
    # one FN desc slot (rel=0) and two closure/EOS-like tail tokens (rel=2,3).
    input_ids = torch.tensor(
        [[1100, 1100, 1100, 1100, 1101, 1100, 1101, 1101]],
        dtype=torch.long,
    )

    meta_base = {
        "prompt_len": 0,
        "prefix_len": 4,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [1],
        "tail_desc_pos": [0],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    # Default: FN desc is supervised.
    loss_default = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta_base)],
            "input_ids": input_ids,
        },
    )

    # Disable FN desc supervision via token_ce module config.
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        rollout_fn_desc_weight=0.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    loss_fn_desc_off = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta_base)],
            "input_ids": input_ids,
        },
    )

    # Mask closure/EOS-like tail positions explicitly; this should also reduce loss,
    # confirming those positions are supervised when not masked.
    meta_mask_tail = dict(meta_base)
    meta_mask_tail["tail_ignore_pos"] = [2, 3]
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    loss_tail_masked = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_mask_tail],
            "input_ids": input_ids,
        },
    )

    assert float(loss_fn_desc_off.detach().cpu().item()) < float(
        loss_default.detach().cpu().item()
    )
    assert float(loss_tail_masked.detach().cpu().item()) < float(
        loss_default.detach().cpu().item()
    )


def test_stage2_pipeline_canonical_bbox_geo_weights_control_precomputed_geo_loss() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": 0.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "coord_ce_weight": 0.0,
        "coord_gate_weight": 0.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=1.0,
        bbox_smoothl1_weight=1.0,
        bbox_ciou_weight=1.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    meta = {
        "prompt_len": 0,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [],
        "tail_desc_pos": [],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [{"pos": [2, 3, 4, 5], "gt_bins": [10, 20, 30, 40]}],
        "bbox_groups_fn": [],
    }

    loss_default = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        bbox_geo_enabled=True,
        bbox_geo_weight=1.0,
        bbox_smoothl1_weight=0.0,
        bbox_ciou_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    loss_geo_off = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    assert float(loss_geo_off.detach().cpu().item()) < float(
        loss_default.detach().cpu().item()
    )


def test_stage2_pipeline_default_parity_channel_b_desc_weighting_unpacked() -> None:
    desc_w = 0.35

    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": float(desc_w),
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "coord_ce_weight": 0.0,
        "coord_gate_weight": 0.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=float(desc_w),
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[1100, 1101, 1101, 1101, 1101, 1101, 1101, 1101]], dtype=torch.long)
    meta = {
        "prompt_len": 0,
        "prefix_len": 4,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "prefix_struct_pos": [1],
        "tail_desc_pos": [0, 1],
        "tail_ignore_pos": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    loss_from_desc_ce = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        rollout_fn_desc_weight=float(desc_w),
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    loss_from_rollout_fn_desc = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta)],
            "input_ids": input_ids,
        },
    )

    assert float(loss_from_desc_ce.detach().cpu().item()) == pytest.approx(
        float(loss_from_rollout_fn_desc.detach().cpu().item()), rel=1e-6, abs=1e-6
    )


def test_stage2_pipeline_default_parity_channel_b_desc_weighting_packed() -> None:
    desc_w = 0.4

    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "desc_ce_weight": float(desc_w),
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "coord_ce_weight": 0.0,
        "coord_gate_weight": 0.0,
        "channel_b": {},
    }
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=float(desc_w),
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)

    seg_len = 8
    input_ids = torch.tensor(
        [[1100, 1101, 1101, 1101, 1101, 1101, 1101, 1101, 1100, 1101, 1101, 1101, 1101, 1101, 1101, 1101]],
        dtype=torch.long,
    )
    meta = [
        {
            "prompt_len": 0,
            "prefix_len": 4,
            "train_len": seg_len,
            "encoded_len": seg_len,
            "prefix_struct_pos": [1],
            "tail_desc_pos": [0],
            "tail_ignore_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
        },
        {
            "prompt_len": 0,
            "prefix_len": 4,
            "train_len": seg_len,
            "encoded_len": seg_len,
            "prefix_struct_pos": [0, 2],
            "tail_desc_pos": [1],
            "tail_ignore_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
        },
    ]

    loss_from_desc_ce = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta[0]), dict(meta[1])],
            "input_ids": input_ids,
        },
    )

    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        desc_ce_weight=1.0,
        rollout_fn_desc_weight=float(desc_w),
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    loss_from_rollout_fn_desc = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta[0]), dict(meta[1])],
            "input_ids": input_ids,
        },
    )

    assert float(loss_from_desc_ce.detach().cpu().item()) == pytest.approx(
        float(loss_from_rollout_fn_desc.detach().cpu().item()), rel=1e-6, abs=1e-6
    )


def test_tail_closure_positions_match_same_brace_used_for_fn_injection():
    tok = _DummyTokenizer()

    rollout_text = '{"objects":[{"desc":"p","bbox_2d":[<|coord_0|>,<|coord_0|>,<|coord_1|>,<|coord_1|>]}]}'
    parsed = parse_rollout_for_matching(
        tokenizer=tok,
        response_token_ids=list(tok.encode(rollout_text)),
    )

    fn_fragment = _serialize_append_fragment(
        fn_objects=[
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[5, 5, 6, 6],
                desc="fn",
            )
        ],
        prefix_text=parsed.prefix_text,
    )

    assistant_text = parsed.prefix_text + fn_fragment
    assistant_ids = list(tok.encode(assistant_text))
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    ignore_rel = _stage2_ab_tail_closure_positions(
        tokenizer=tok,
        assistant_span_ids=assistant_ids + [im_end_id],
        prefix_len=int(len(parsed.prefix_token_ids)),
    )

    tail_text = assistant_text[int(len(parsed.prefix_token_ids)) :]
    close_rel = int(tail_text.rfind("}"))

    assert close_rel >= 0
    assert ignore_rel == [close_rel, len(tail_text)]


def test_tail_closure_positions_ignore_braces_inside_quoted_desc():
    tok = _DummyTokenizer()

    # Include a literal '}' inside a quoted desc string; closure-marker parsing must
    # ignore it and select the brace that closes the *outermost* JSON object.
    json_text = '{"objects":[{"bbox_2d":[0,0,1,1],"desc":"a } b"}]}'
    ids = list(tok.encode(json_text))
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    assistant_span_ids = ids + [im_end_id]
    ignore_rel = _stage2_ab_tail_closure_positions(
        tokenizer=tok,
        assistant_span_ids=assistant_span_ids,
        prefix_len=0,
    )

    assert ignore_rel == [len(json_text) - 1, len(json_text)]


def test_tail_closure_positions_prefer_turn_end_after_json_close():
    tok = _DummyTokenizer()

    # Include a literal '<|im_end|>' substring inside a quoted desc string; closure-marker
    # parsing must select the *turn-end* token that occurs after the outermost JSON close brace.
    json_text = '{"objects":[{"bbox_2d":[0,0,1,1],"desc":"a <|im_end|> b"}]}'
    ids = list(tok.encode(json_text))
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    assistant_span_ids = ids + [im_end_id]
    ignore_rel = _stage2_ab_tail_closure_positions(
        tokenizer=tok,
        assistant_span_ids=assistant_span_ids,
        prefix_len=0,
    )

    assert ignore_rel == [len(json_text) - 1, len(json_text)]


def test_channel_b_sequential_dedup_attaches_duplicates_to_clean_boundaries() -> None:
    raw = [
        GTObject(index=0, geom_type="bbox_2d", points_norm1000=[10, 10, 20, 20], desc="cat"),
        GTObject(index=1, geom_type="bbox_2d", points_norm1000=[10, 10, 20, 20], desc="cat"),
        GTObject(index=2, geom_type="bbox_2d", points_norm1000=[100, 100, 150, 150], desc="dog"),
        GTObject(index=3, geom_type="bbox_2d", points_norm1000=[10, 10, 20, 20], desc="cat"),
    ]

    accepted, bursts = _sequential_dedup_bbox_objects(
        parsed_bbox_objects_raw=raw,
        duplicate_iou_threshold=0.90,
    )

    assert [obj.desc for obj in accepted] == ["cat", "dog"]
    assert sorted(bursts.keys()) == [1, 2]
    assert [obj.index for obj in bursts[1]] == [1]
    assert [obj.index for obj in bursts[2]] == [3]

    diag = _compute_duplicate_diagnostics(raw)
    assert diag["dup/max_desc_count"] == pytest.approx(3.0)
    assert diag["dup/near_iou90_pairs_same_desc_count"] == pytest.approx(3.0)
    assert diag["dup/near_iou90_pairs_any_desc_count"] == pytest.approx(3.0)
    assert diag["dup/saturation_rate"] == pytest.approx(0.0)


def test_duplicate_burst_unlikelihood_targets_use_lcp_divergence_and_collapse_same_boundary_token() -> None:
    tok = _DummyTokenizer()
    accepted_clean = [
        GTObject(index=0, geom_type="bbox_2d", points_norm1000=[1, 1, 2, 2], desc="cat"),
        GTObject(index=1, geom_type="bbox_2d", points_norm1000=[5, 5, 6, 6], desc="book"),
    ]
    duplicate_bursts = {
        1: [
            GTObject(index=2, geom_type="bbox_2d", points_norm1000=[1, 1, 2, 2], desc="book"),
            GTObject(index=3, geom_type="bbox_2d", points_norm1000=[1, 1, 2, 2], desc="book"),
        ]
    }

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tok,
        objects=accepted_clean,
        object_field_order="desc_first",
    )
    y_train_ids = list(clean_prefix.prefix_token_ids) + list(tok.encode("]}"))
    targets, ul_boundaries, skipped = _build_duplicate_burst_unlikelihood_targets(
        tokenizer=tok,
        y_train_ids=y_train_ids,
        clean_target_text=clean_prefix.prefix_text + "]}",
        accepted_objects_clean=accepted_clean,
        fn_objects=[],
        duplicate_bursts_by_boundary=duplicate_bursts,
        boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
        object_field_order="desc_first",
    )

    assert len(targets) == 1
    assert ul_boundaries == 1
    assert skipped == 0

    target = targets[0]
    assert target["boundary"] == 1
    assert tok.decode([target["token_id"]]) == "1"
    assert tok.decode([y_train_ids[target["rel_pos"]]]) == "5"


def test_duplicate_burst_unlikelihood_targets_skip_when_no_safe_divergence_exists() -> None:
    tok = _DummyTokenizer()
    accepted_clean = [
        GTObject(index=0, geom_type="bbox_2d", points_norm1000=[5, 5, 6, 6], desc="book"),
    ]
    duplicate_bursts = {
        0: [
            GTObject(index=1, geom_type="bbox_2d", points_norm1000=[1, 1, 2, 2], desc="book"),
        ]
    }

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tok,
        objects=accepted_clean,
        object_field_order="desc_first",
    )
    y_train_ids = []
    targets, ul_boundaries, skipped = _build_duplicate_burst_unlikelihood_targets(
        tokenizer=tok,
        y_train_ids=y_train_ids,
        clean_target_text=clean_prefix.prefix_text + "]}",
        accepted_objects_clean=accepted_clean,
        fn_objects=[],
        duplicate_bursts_by_boundary=duplicate_bursts,
        boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
        object_field_order="desc_first",
    )

    assert targets == []
    assert ul_boundaries == 0
    assert skipped == 1


def test_duplicate_burst_unlikelihood_targets_resolve_boundary_crossing_tokenization() -> None:
    tok = _BoundaryMergingTokenizer()
    accepted_clean = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[5, 5, 6, 6],
            desc="book",
        ),
    ]
    duplicate_bursts = {
        0: [
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[1, 1, 2, 2],
                desc="book",
            ),
        ]
    }

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tok,
        objects=accepted_clean,
        object_field_order="desc_first",
    )
    y_train_ids = list(clean_prefix.prefix_token_ids) + list(tok.encode("]}"))
    targets, ul_boundaries, skipped = _build_duplicate_burst_unlikelihood_targets(
        tokenizer=tok,
        y_train_ids=y_train_ids,
        clean_target_text=clean_prefix.prefix_text + "]}",
        accepted_objects_clean=accepted_clean,
        fn_objects=[],
        duplicate_bursts_by_boundary=duplicate_bursts,
        boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
        object_field_order="desc_first",
    )

    assert len(targets) == 1
    assert ul_boundaries == 1
    assert skipped == 0

    target = targets[0]
    assert target["boundary"] == 0
    assert tok.decode([target["token_id"]]) == "1"
    assert tok.decode([y_train_ids[target["rel_pos"]]]) == "5"


def test_stage2_channel_b_duplicate_burst_unlikelihood_logs_weighted_objective_atom() -> None:
    t = _make_min_trainer()
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        duplicate_burst_unlikelihood_enabled=True,
        duplicate_burst_unlikelihood_weight=2.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    model = _DummyCallIndexedTokenModel(pred_ids=[7], vocab=1200)
    input_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    meta = [
        {
            "stage2_channel": "B",
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 2,
            "encoded_len": 4,
            "prefix_struct_pos": [],
            "prefix_coord_pos": [],
            "tail_desc_pos": [],
            "tail_ignore_pos": [],
            "tail_closure_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
            "duplicate_burst_unlikelihood_targets": [
                {"boundary": 0, "rel_pos": 0, "token_id": 7},
            ],
        }
    ]

    loss = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
        },
    )

    assert float(loss.detach().cpu().item()) > 0.0
    pending = t._stage2_pending_train_logs[1].finalize()
    assert "train/optimization/loss_duplicate_burst_unlikelihood" in pending
    assert pending["train/optimization/loss_duplicate_burst_unlikelihood"] > 0.0


def test_stage2_channel_a_rejects_deprecated_stop_signal_damping_config() -> None:
    t = _make_min_trainer()
    manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=True,
        token_ce_weight=2.0,
        duplicate_burst_unlikelihood_enabled=False,
        duplicate_burst_unlikelihood_weight=0.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        bbox_size_aux_enabled=False,
        bbox_size_aux_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    manifest["objective"][0]["config"]["stop_signal_damping"] = {"enabled": True}
    t.stage2_pipeline_manifest = manifest
    model = _DummyCallIndexedTokenModel(pred_ids=[1107], vocab=1200)
    input_ids = torch.tensor([[10, 1002, 1107, 1003, 1004]], dtype=torch.long)
    meta = [
        {
            "stage2_channel": "A",
            "prompt_len": 1,
            "prefix_len": 0,
            "train_len": 4,
            "encoded_len": 5,
            "prefix_struct_pos": [],
            "prefix_coord_pos": [],
            "prefix_coord_target_bins": [],
            "tail_desc_pos": [],
            "tail_ignore_pos": [],
            "tail_closure_pos": [2, 3],
            "stop_rel_pos": 1,
            "stop_token_id": 1107,
            "continue_token_id": 1108,
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"token_ce\.config\.stop_signal_damping is deprecated and unsupported",
    ):
        t.compute_loss(
            model,
            {
                "_stage2_ab_channel": "A",
                "_rollout_matching_meta": meta,
                "input_ids": input_ids,
            },
        )


def test_stage2_channel_b_compute_loss_copies_triage_and_split_rollout_telemetry() -> None:
    t = _make_min_trainer()
    t.stage2_pipeline_manifest = _make_stage2_pipeline_manifest(
        token_ce_enabled=False,
        token_ce_weight=0.0,
        duplicate_burst_unlikelihood_enabled=True,
        duplicate_burst_unlikelihood_weight=1.0,
        bbox_geo_enabled=False,
        bbox_geo_weight=0.0,
        coord_reg_enabled=False,
        coord_reg_weight=0.0,
    )
    model = _DummyAlwaysTokenModel(pred_id=7)
    input_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    meta = [
        {
            "stage2_channel": "B",
            "prompt_len": 2,
            "prefix_len": 0,
            "train_len": 2,
            "encoded_len": 4,
            "prefix_struct_pos": [],
            "prefix_coord_pos": [],
            "tail_desc_pos": [],
            "tail_ignore_pos": [],
            "tail_closure_pos": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
            "duplicate_burst_unlikelihood_targets": [
                {"boundary": 0, "rel_pos": 0, "token_id": 7},
            ],
        }
    ]

    t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": meta,
            "_rollout_matching_batch_metrics": {
                "train/triage/gt_backed_count": 2.0,
                "train/triage/recovered_ground_truth_count": 1.0,
                "train/triage/recovered_ground_truth_rate_num": 1.0,
                "train/triage/recovered_ground_truth_rate_den": 4.0,
                "train/triage/recovered_ground_truth_rate": 0.25,
                "rollout/anchor/pred_objects": 5.0,
                "rollout/explorer/pred_objects": 7.0,
                "rollout/explorer/temperature": 0.7,
                "rollout/explorer/do_sample": 1.0,
                "rollout/matched_for_supervision_over_valid_pred": 0.5,
                "rollout/matched_for_supervision_count": 3.0,
                "rollout/valid_pred_objects_total": 6.0,
            },
            "input_ids": input_ids,
        },
    )

    pending = t._stage2_pending_train_logs[1].finalize()
    assert pending["train/triage/gt_backed_count"] == pytest.approx(2.0)
    assert pending["train/triage/recovered_ground_truth_rate"] == pytest.approx(0.25)
    assert pending["rollout/anchor/pred_objects"] == pytest.approx(5.0)
    assert pending["rollout/explorer/pred_objects"] == pytest.approx(7.0)
    assert pending["rollout/explorer/temperature"] == pytest.approx(0.7)
    assert pending["rollout/explorer/do_sample"] == pytest.approx(1.0)
    assert pending["rollout/matched_for_supervision_over_valid_pred"] == pytest.approx(
        0.5
    )
    assert pending["loss/B_rollout_text/duplicate_burst_unlikelihood"] > 0.0
    assert pending["diag/duplicate_burst/num_terms"] == pytest.approx(1.0)
    assert pending["diag/duplicate_burst/num_ul_boundaries"] == pytest.approx(1.0)
    assert pending["diag/duplicate_burst/loss_per_term"] > 0.0


def test_pending_stage2_log_aggregates_closure_and_invalid_rollout_metrics() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2/raw_rollouts": 3.0,
            "stage2_ab/channel_b/invalid_rollout": 1.0,
            "stage2_ab/channel_b/closure_supervision/N_drop": 1.0,
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 3.0,
        }
    )
    pending.add(
        {
            "stage2/raw_rollouts": 7.0,
            "stage2_ab/channel_b/invalid_rollout": 2.0,
            "stage2_ab/channel_b/closure_supervision/N_drop": 4.0,
            "rollout/_parse_truncated_num": 4.0,
            "rollout/_parse_truncated_den": 7.0,
        }
    )

    out = pending.finalize()

    assert out["stage2/raw_rollouts"] == pytest.approx(10.0)
    assert out["stage2_ab/channel_b/invalid_rollout"] == pytest.approx(3.0)
    assert out["stage2_ab/channel_b/closure_supervision/N_drop"] == pytest.approx(5.0)
    assert out["rollout/parse_truncated_rate"] == pytest.approx(0.5)
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out


def test_pending_stage2_log_aggregates_strict_drop_metrics_and_reasons() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2_ab/channel_b/strict_drop/N_valid_pred": 3.0,
            "stage2_ab/channel_b/strict_drop/N_drop_invalid": 2.0,
            "stage2_ab/channel_b/strict_drop/reason/order_violation": 1.0,
            "stage2_ab/channel_b/strict_drop/reason/wrong_arity": 1.0,
        }
    )
    pending.add(
        {
            "stage2_ab/channel_b/strict_drop/N_valid_pred": 4.0,
            "stage2_ab/channel_b/strict_drop/N_drop_invalid": 3.0,
            "stage2_ab/channel_b/strict_drop/reason/order_violation": 2.0,
            "stage2_ab/channel_b/strict_drop/reason/missing_desc": 1.0,
        }
    )

    out = pending.finalize()

    assert out["stage2_ab/channel_b/strict_drop/N_valid_pred"] == pytest.approx(7.0)
    assert out["stage2_ab/channel_b/strict_drop/N_drop_invalid"] == pytest.approx(5.0)
    assert out["stage2_ab/channel_b/strict_drop/reason/order_violation"] == pytest.approx(3.0)
    assert out["stage2_ab/channel_b/strict_drop/reason/wrong_arity"] == pytest.approx(1.0)
    assert out["stage2_ab/channel_b/strict_drop/reason/missing_desc"] == pytest.approx(1.0)


def test_pending_stage2_log_omits_channel_b_keys_when_not_provided() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2/channel_a": 1.0,
            "loss/coord/bbox_smoothl1": 0.25,
        }
    )

    out = pending.finalize()

    assert out["stage2/channel_a"] == pytest.approx(1.0)
    assert out["loss/coord/bbox_smoothl1"] == pytest.approx(0.25)
    assert "stage2/channel_b" not in out
    assert "stage2_ab/channel_b/invalid_rollout" not in out
    assert "stage2_ab/channel_b/strict_drop/N_valid_pred" not in out
    assert "stage2_ab/channel_b/strict_drop/N_drop_invalid" not in out


def test_reduce_stage2_pending_metrics_global_recomputes_ratio_and_sums_invalid_rollout() -> None:
    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    class _FakeDist:
        ReduceOp = _FakeReduceOp

        def all_gather_object(self, gathered: list[object], obj: object) -> None:
            for i in range(len(gathered)):
                gathered[i] = list(obj) if isinstance(obj, list) else obj

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            if op == self.ReduceOp.SUM:
                tensor.add_(
                    torch.tensor(
                        [3.0, 6.0, 6.0, 1.0, 2.0],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )

    trainer = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    trainer._dist_info = lambda: (0, 2, _FakeDist())

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 4.0,
            "stage2/raw_rollouts": 4.0,
            "stage2_ab/channel_b/invalid_rollout": 1.0,
            "rollout/parse_truncated": 1.0,
        }
    )

    assert out["rollout/parse_truncated_rate"] == pytest.approx(0.4)
    assert out["stage2_ab/channel_b/invalid_rollout"] == pytest.approx(2.0)
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out


def test_reduce_stage2_pending_metrics_global_uses_weight_total_for_means() -> None:
    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    class _FakeDist:
        ReduceOp = _FakeReduceOp

        def all_gather_object(self, gathered: list[object], obj: object) -> None:
            # Mirror the local keys across all ranks (key union should be stable).
            for i in range(len(gathered)):
                gathered[i] = list(obj) if isinstance(obj, list) else obj

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            if op == self.ReduceOp.SUM:
                # Simulate rank1 having: weight_total=3, loss_mean=20 => numerator=60.
                tensor.add_(
                    torch.tensor(
                        [3.0, 60.0],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )

    trainer = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    trainer._dist_info = lambda: (0, 2, _FakeDist())

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "stage2/_log_weight_total": 1.0,
            "loss/coord/bbox_smoothl1": 10.0,
        }
    )

    assert out["loss/coord/bbox_smoothl1"] == pytest.approx((10.0 * 1.0 + 20.0 * 3.0) / 4.0)
    assert "stage2/_log_weight_total" not in out


def test_reduce_stage2_pending_metrics_global_treats_train_optimization_losses_as_weighted_means() -> None:
    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    class _FakeDist:
        ReduceOp = _FakeReduceOp

        def all_gather_object(self, gathered: list[object], obj: object) -> None:
            for i in range(len(gathered)):
                gathered[i] = list(obj) if isinstance(obj, list) else obj

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            if op == self.ReduceOp.SUM:
                tensor.add_(
                    torch.tensor(
                        [3.0, 60.0],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )

    trainer = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    trainer._dist_info = lambda: (0, 2, _FakeDist())

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "stage2/_log_weight_total": 1.0,
            "train/optimization/loss_duplicate_burst_unlikelihood": 10.0,
        }
    )

    assert out["train/optimization/loss_duplicate_burst_unlikelihood"] == pytest.approx(
        (10.0 * 1.0 + 20.0 * 3.0) / 4.0
    )
    assert "stage2/_log_weight_total" not in out


def test_reduce_stage2_pending_metrics_global_strips_internal_underscore_keys() -> None:
    trainer = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    trainer._dist_info = lambda: (0, 1, None)

    out = trainer._reduce_stage2_pending_metrics_global(
        {
            "stage2/_log_weight_total": 2.0,
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 4.0,
            "rollout/parse_truncated": 1.0,
            "stage2/raw_rollouts": 4.0,
            "loss/coord/bbox_smoothl1": 1.0,
        }
    )

    assert "stage2/_log_weight_total" not in out
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out
    assert all(not str(k).startswith("rollout/_") for k in out)



def test_channel_b_step_budgeted_path_is_supported_under_ddp_mock(monkeypatch):
    # Stage2-AB standardizes Channel-B to a single step-budgeted pathway.
    # This should not be rejected just because torch.distributed is initialized.
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2, raising=False)

    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    # Even if a legacy key is present in a hand-built dict, the trainer should not consult it.
    t.stage2_ab_cfg = {"schedule": {"b_ratio": 1.0}, "channel_b": {"mode": "async"}}

    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._stage2_channel_override = None

    t.args = types.SimpleNamespace(seed=123)
    t.state = types.SimpleNamespace(global_step=0)

    # Minimal executor shim state.
    t._stage2_b_step_gs = None
    t._stage2_b_step_micro = 0
    t._stage2_b_step_raw = []

    # Avoid heavy rollout/packing work: just confirm the call path is allowed.
    t._stage2_training_step_b_step_mode = (
        lambda model, inputs, global_step: torch.tensor(1.0)
    )

    class _M:
        training = True

    out = t.training_step(_M(), [{"messages": []}])
    assert isinstance(out, torch.Tensor)


def test_b_ratio_realized_tracks_optimizer_steps_once():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t._stage2_ab_realized_last_gs = None

    t._stage2_record_realized_step(global_step=0, executed_b=False)
    t._stage2_record_realized_step(global_step=0, executed_b=True)  # same step, ignored
    t._stage2_record_realized_step(global_step=1, executed_b=True)

    assert pytest.approx(t._stage2_b_ratio_realized(), rel=1e-6) == 0.5


def test_merge_rollout_matching_batch_metrics_preserves_existing_keys():
    t = _make_min_trainer()
    batch = {"_rollout_matching_batch_metrics": {"rollout/backend_vllm": 1.0}}
    t._merge_rollout_matching_batch_metrics(
        batch,
        {
            "stage2_ab/b_ratio_realized": 3.0,
            "rollout/backend_vllm": 2.0,
        },
    )
    bm = batch.get("_rollout_matching_batch_metrics")
    assert isinstance(bm, dict)
    assert bm["stage2_ab/b_ratio_realized"] == 3.0
    assert bm["rollout/backend_vllm"] == 2.0


def _make_eval_ready_stage2_ab_trainer() -> Stage2ABTrainingTrainer:
    trainer = object.__new__(Stage2ABTrainingTrainer)

    class _EvalModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

    trainer.model = _EvalModel()
    trainer.args = types.SimpleNamespace()
    trainer.state = types.SimpleNamespace(global_step=11)
    trainer.control = types.SimpleNamespace(tag="ctrl")
    trainer.template = types.SimpleNamespace(tokenizer=object())
    trainer._cfg = lambda _k, default=None: default
    trainer._desc_monitor_cfg = lambda: {"enabled": False}
    trainer._coord_id_map = lambda: {i: i for i in range(1000)}
    trainer.get_eval_dataloader = lambda _eval_dataset=None: [[{"sample_id": 0}]]
    trainer._rollout_many = lambda batch, prompt_variant_override=None, **_kwargs: [
        ([101], "{}", "greedy", []) for _ in batch
    ]
    trainer._maybe_eval_vllm_colocate_window = lambda **_kwargs: nullcontext()
    trainer.callback_handler = types.SimpleNamespace(
        on_evaluate=lambda args, state, control, metrics: control
    )
    trainer.log = lambda _metrics: None
    return trainer


def test_stage2_two_channel_eval_emits_rollout_map_and_coco_contract(monkeypatch) -> None:
    trainer = _make_eval_ready_stage2_ab_trainer()

    parse_obj = types.SimpleNamespace(
        response_token_ids=[101],
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="cat",
            )
        ],
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
    )

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned.parse_rollout_for_matching",
        lambda **_kwargs: parse_obj,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._points_from_coord_tokens",
        lambda **_kwargs: [0, 0, 10, 10],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._extract_gt_objects",
        lambda _sample: [
            GTObject(
                index=0,
                geom_type="bbox",
                points_norm1000=[0, 0, 10, 10],
                desc="cat",
            )
        ],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned.hungarian_match_maskiou",
        lambda **_kwargs: types.SimpleNamespace(
            matched_pairs=[(0, 0)],
            fp_pred_indices=[],
            fn_gt_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._compute_eval_detection_coco_metrics",
        lambda **_kwargs: (
            {"bbox_AP": 0.25, "bbox_AP50": 0.5, "segm_AP": 0.75},
            {"empty_pred": 0},
        ),
    )

    logged_metrics: dict[str, float] = {}
    trainer.log = lambda metrics: logged_metrics.update(dict(metrics))

    metrics = trainer.evaluate(metric_key_prefix="eval")

    assert trainer.model.training is True
    assert logged_metrics == metrics
    assert metrics["eval/detection/mAP"] == pytest.approx(0.25)
    assert metrics["eval/runtime/coco_eval_ok"] == pytest.approx(1.0)
    assert all(not k.startswith("eval/detection/bbox_") for k in metrics)
    assert all(not k.startswith("eval/detection/segm_") for k in metrics)


def test_stage2_two_channel_eval_raises_when_coco_eval_fails(monkeypatch) -> None:
    trainer = _make_eval_ready_stage2_ab_trainer()

    parse_obj = types.SimpleNamespace(
        response_token_ids=[101],
        valid_objects=[
            types.SimpleNamespace(
                index=0,
                geom_type="bbox_2d",
                coord_token_indices=[0, 1, 2, 3],
                desc="cat",
            )
        ],
        dropped_invalid=0,
        dropped_ambiguous=0,
        truncated=False,
    )

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned.parse_rollout_for_matching",
        lambda **_kwargs: parse_obj,
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._points_from_coord_tokens",
        lambda **_kwargs: [0, 0, 10, 10],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._extract_gt_objects",
        lambda _sample: [
            GTObject(
                index=0,
                geom_type="bbox",
                points_norm1000=[0, 0, 10, 10],
                desc="cat",
            )
        ],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned.hungarian_match_maskiou",
        lambda **_kwargs: types.SimpleNamespace(
            matched_pairs=[(0, 0)],
            fp_pred_indices=[],
            fn_gt_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=1.0,
            matched_maskiou_count=1,
        ),
    )

    def _raise_coco_eval(**_kwargs):
        raise ValueError("synthetic coco eval failure")

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._compute_eval_detection_coco_metrics",
        _raise_coco_eval,
    )

    with pytest.raises(RuntimeError, match=r"Eval-step COCO/mAP failed"):
        trainer.evaluate(metric_key_prefix="eval")
