import types
import threading

import pytest
import torch
import torch.nn as nn

from src.trainers.rollout_matching_sft import (
    GTObject,
    _serialize_append_fragment,
    parse_rollout_for_matching,
)
from src.trainers.stage2_ab_training import (
    Stage2ABTrainingTrainer,
    _PendingStage2Log,
    _bbox_smoothl1_ciou_loss,
    _expectation_decode_coords,
    _extract_gt_bboxonly,
    _matched_prefix_structure_positions,
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


def test_bbox_losses_stable_on_noncanonical_pred():
    pred = torch.tensor([[1.2, -0.1, -0.2, 0.5]], dtype=torch.float32)
    gt = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    smoothl1, ciou = _bbox_smoothl1_ciou_loss(pred_xyxy=pred, gt_xyxy=gt)
    assert torch.isfinite(smoothl1).item()
    assert torch.isfinite(ciou).item()


def _make_min_trainer(*, n_softctx_iter: int):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 0.0},
        "n_softctx_iter": int(n_softctx_iter),
        "softctx_temperature": 1.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "desc_ce_weight": 1.0,
    }
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)
    return t


def test_channel_a_softctx_runs_n_forwards_and_enforces_cache_and_qwen_posids():
    trainer = _make_min_trainer(n_softctx_iter=2)
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
    assert len(model.calls) == 2


def test_channel_a_softctx_n1_runs_single_forward():
    trainer = _make_min_trainer(n_softctx_iter=1)
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


def test_channel_a_placeholder_embedding_invariance_bitwise():
    trainer = _make_min_trainer(n_softctx_iter=2)
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

    assert len(model.inputs_embeds_calls) == 2
    e0, e1 = model.inputs_embeds_calls

    # Prompt tokens should be bitwise unchanged across iterations.
    assert torch.equal(e0[0, 0], e1[0, 0])
    assert torch.equal(e0[0, 1], e1[0, 1])
    # Non-coord tail tokens should also be unchanged.
    assert torch.equal(e0[0, 6], e1[0, 6])
    # Coord-slot rows may differ (softctx scatter-update).
    assert not torch.equal(e0[0, 2], e1[0, 2])


def test_channel_a_ce_uses_a1_logits_not_final_logits():
    trainer = _make_min_trainer(n_softctx_iter=2)
    # Isolate CE by disabling geometry losses and groups.
    trainer.stage2_ab_cfg["bbox_smoothl1_weight"] = 0.0
    trainer.stage2_ab_cfg["bbox_ciou_weight"] = 0.0

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

    # If CE incorrectly used the final logits, this would be low; with CE@A1 it should be high.
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

    assert p1.prefix_token_ids == tok.encode("{", add_special_tokens=False)
    assert p1.prefix_token_ids == p2.prefix_token_ids
    assert p1.prefix_text == p2.prefix_text == "{"
    assert p1.valid_objects == []
    assert p1.truncated is True


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
    trainer = _make_min_trainer(n_softctx_iter=1)
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


def test_extract_gt_bboxonly_rejects_poly_geometry():
    sample = {
        "assistant_payload": {
            "object_1": {
                "desc": "x",
                "poly": [0, 1, 2, 3],
            }
        }
    }
    with pytest.raises(ValueError, match="bbox-only v1"):
        _extract_gt_bboxonly(sample)


def test_extract_gt_bboxonly_rejects_other_geometry_key_even_with_bbox():
    sample = {
        "assistant_payload": {
            "object_1": {
                "desc": "x",
                "bbox_2d": [0, 0, 10, 10],
                "mask_rle": {"counts": "abc", "size": [1, 1]},
            }
        }
    }
    with pytest.raises(ValueError, match="bbox-only v1"):
        _extract_gt_bboxonly(sample)


def test_extract_gt_bboxonly_rejects_invalid_bbox_order():
    sample = {
        "assistant_payload": {
            "object_1": {
                "desc": "x",
                "bbox_2d": [10, 10, 0, 0],
            }
        }
    }
    with pytest.raises(ValueError, match="invalid bbox_2d"):
        _extract_gt_bboxonly(sample)


def test_compute_loss_raises_on_sliced_logits():
    trainer = _make_min_trainer(n_softctx_iter=1)
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


def test_channel_b_desc_ce_split_allows_downweight_matched_desc_only():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "n_softctx_iter": 1,
        "softctx_temperature": 1.0,
        "desc_ce_weight": 1.0,
        # Turn off bbox losses so CE effect is isolated.
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {"desc_ce_weight_matched": 0.0},
    }
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)

    # Targets at positions 2,3 are 'hard' (1101) under pred_id=1100.
    input_ids = torch.tensor([[1100, 1100, 1101, 1101, 1100, 1100]], dtype=torch.long)

    meta = [
        {
            "prompt_len": 0,
            "prefix_len": 0,
            "train_len": int(input_ids.shape[1]),
            "encoded_len": int(input_ids.shape[1]),
            # Provide both lists so compute_loss prefers the split behavior.
            "tail_desc_pos": [2, 3],
            "tail_desc_pos_matched": [2, 3],
            "tail_desc_pos_missing": [],
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [],
        }
    ]

    loss_down = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
        },
    )

    t.stage2_ab_cfg["channel_b"]["desc_ce_weight_matched"] = 1.0
    loss_full = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": meta,
            "input_ids": input_ids,
        },
    )

    assert float(loss_down.detach().cpu().item()) < float(
        loss_full.detach().cpu().item()
    )


def test_channel_b_includes_fn_geometry_loss():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "n_softctx_iter": 1,
        "softctx_temperature": 1.0,
        # Turn off CE so we isolate geometry contribution.
        "desc_ce_weight": 0.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=999)
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


def test_channel_b_repeat_trigger_flag_does_not_change_supervision_semantics() -> None:
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "n_softctx_iter": 1,
        "softctx_temperature": 1.0,
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 1.0,
        "bbox_ciou_weight": 1.0,
        "channel_b": {},
    }
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
        "tail_desc_pos_matched": [2, 3],
        "tail_desc_pos_missing": [],
        "bbox_groups_prefix": [{"pos": [1, 2, 3, 4], "gt_bins": [1, 1, 2, 2]}],
        "bbox_groups_fn": [{"pos": [1, 2, 3, 4], "gt_bins": [1, 1, 2, 2]}],
    }

    loss_no_repeat = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(base_meta, repeat_terminate_triggered=0)],
            "input_ids": input_ids,
        },
    )
    loss_with_repeat = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(base_meta, repeat_terminate_triggered=1)],
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
        "n_softctx_iter": 1,
        "softctx_temperature": 1.0,
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
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
        '{"object_1":{"desc":"matched","bbox_2d":[0,0,1,1]},'
        '"object_2":{"desc":"fp","bbox_2d":[2,2,3,3]}'
    )
    prefix_token_ids = list(tok.encode(prefix_text))

    key_anchor = int(prefix_text.find('"object_1"'))
    value_start = int(prefix_text.find("{", key_anchor))
    value_end = int(prefix_text.find('},"object_2"')) + 1
    matched_obj = types.SimpleNamespace(
        key="object_1",
        value_span=(value_start, value_end),
    )

    rel = _matched_prefix_structure_positions(
        tokenizer=tok,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        matched_pred_objects=[matched_obj],
    )

    key_pos_matched = int(prefix_text.find('"object_1"'))
    key_pos_fp = int(prefix_text.find('"object_2"'))
    desc_pos_matched = int(prefix_text.find("matched"))

    assert key_pos_matched >= 0 and key_pos_matched in rel
    assert key_pos_fp >= 0 and key_pos_fp not in rel
    assert desc_pos_matched >= 0 and desc_pos_matched not in rel


def test_channel_b_prefix_structure_supervision_is_matched_only():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "n_softctx_iter": 1,
        "softctx_temperature": 1.0,
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {},
    }
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)

    # Prefix (len=4):
    # - rel=1 -> matched structure token (should be supervised)
    # - rel=2 -> matched desc token (should stay masked)
    # - rel=3 -> FP token (should stay masked)
    # Tail (len=4): FN structure/desc tokens are supervised by default.
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

    assert float(loss_matched_only.detach().cpu().item()) < float(
        loss_oversupervised.detach().cpu().item()
    )


def test_tail_closure_positions_match_same_brace_used_for_fn_injection():
    tok = _DummyTokenizer()

    rollout_text = '{"object_7":{"desc":"p","bbox_2d":[0,0,1,1]}}'
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
        start_index=8,
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
    json_text = '{"object_1":{"bbox_2d":[0,0,1,1],"desc":"a } b"}}'
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
    json_text = '{"object_1":{"bbox_2d":[0,0,1,1],"desc":"a <|im_end|> b"}}'
    ids = list(tok.encode(json_text))
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))

    assistant_span_ids = ids + [im_end_id]
    ignore_rel = _stage2_ab_tail_closure_positions(
        tokenizer=tok,
        assistant_span_ids=assistant_span_ids,
        prefix_len=0,
    )

    assert ignore_rel == [len(json_text) - 1, len(json_text)]


def test_pending_stage2_log_aggregates_closure_and_repeat_metrics() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "stage2/raw_rollouts": 3.0,
            "stage2_ab/channel_b/closure_supervision/N_drop": 1.0,
            "rollout/repeat_terminate_active": 0.0,
            "rollout/repeat_terminate_triggered_sequences": 2.0,
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 3.0,
        }
    )
    pending.add(
        {
            "stage2/raw_rollouts": 7.0,
            "stage2_ab/channel_b/closure_supervision/N_drop": 4.0,
            "rollout/repeat_terminate_active": 1.0,
            "rollout/repeat_terminate_triggered_sequences": 5.0,
            "rollout/_parse_truncated_num": 4.0,
            "rollout/_parse_truncated_den": 7.0,
        }
    )

    out = pending.finalize()

    assert out["stage2/raw_rollouts"] == pytest.approx(10.0)
    assert out["stage2_ab/channel_b/closure_supervision/N_drop"] == pytest.approx(5.0)
    assert out["rollout/repeat_terminate_active"] == pytest.approx(1.0)
    assert out["rollout/repeat_terminate_triggered_sequences"] == pytest.approx(7.0)
    assert out["rollout/parse_truncated_rate"] == pytest.approx(0.5)
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out


def test_channel_b_semantic_mask_list_ignores_matched_desc_tokens():
    # This exercises the compute_loss masking hook used by the semantic-desc gate:
    # meta["tail_desc_pos_matched_masked"] contributes to the ignore set.
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "n_softctx_iter": 1,
        "softctx_temperature": 1.0,
        "desc_ce_weight": 1.0,
        "bbox_smoothl1_weight": 0.0,
        "bbox_ciou_weight": 0.0,
        "channel_b": {"desc_ce_weight_matched": 1.0},
    }
    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._get_coord_token_ids = lambda: list(range(1000))
    t.state = types.SimpleNamespace(global_step=0)

    model = _DummyAlwaysTokenModel(pred_id=1100)
    input_ids = torch.tensor([[1100, 1100, 1101, 1101, 1100, 1100]], dtype=torch.long)

    meta_no_mask = {
        "prompt_len": 0,
        "prefix_len": 0,
        "train_len": int(input_ids.shape[1]),
        "encoded_len": int(input_ids.shape[1]),
        "tail_ignore_pos": [],
        "tail_desc_pos": [2, 3],
        "tail_desc_pos_matched": [2, 3],
        "tail_desc_pos_missing": [],
        "tail_desc_pos_matched_masked": [],
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [],
    }

    loss_no_mask = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [dict(meta_no_mask)],
            "input_ids": input_ids,
        },
    )

    meta_masked = dict(meta_no_mask)
    meta_masked["tail_desc_pos_matched_masked"] = [2, 3]
    loss_masked = t.compute_loss(
        model,
        {
            "_stage2_ab_channel": "B",
            "_rollout_matching_meta": [meta_masked],
            "input_ids": input_ids,
        },
    )

    assert float(loss_masked.detach().cpu().item()) < float(
        loss_no_mask.detach().cpu().item()
    )


def test_channel_b_step_mode_rejected_for_ddp_safety(monkeypatch):
    # Step-budgeted Channel-B mode is intentionally disallowed under multi-GPU DDP.
    monkeypatch.setattr(
        torch.distributed, "is_available", lambda: True, raising=False
    )
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: True, raising=False
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda: 2, raising=False
    )

    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {"schedule": {"b_ratio": 1.0}, "channel_b": {"mode": "step"}}

    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._stage2_channel_override = None

    t.args = types.SimpleNamespace(seed=123)
    t.state = types.SimpleNamespace(global_step=0)

    class _M:
        training = True

    with pytest.raises(ValueError, match=r"multi-GPU learner \(DDP\)"):
        t.training_step(_M(), [{"messages": []}])


def test_channel_b_step_mode_error_message_mentions_micro_mode(monkeypatch):
    # Error message includes actionable mitigation.
    monkeypatch.setattr(
        torch.distributed, "is_available", lambda: True, raising=False
    )
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: True, raising=False
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda: 2, raising=False
    )

    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {"schedule": {"b_ratio": 1.0}, "channel_b": {"mode": "step"}}

    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._stage2_channel_override = None

    t.args = types.SimpleNamespace(seed=123)
    t.state = types.SimpleNamespace(global_step=7)

    class _M:
        training = True

    with pytest.raises(ValueError, match="channel_b\\.mode='micro'"):
        t.training_step(_M(), [{"messages": []}])


def test_async_step_kind_falls_back_to_a_when_queue_empty(monkeypatch):
    # Queue-gated async mode should fall back to A if scheduled B is infeasible.
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "channel_b": {"mode": "async", "async": {"queue_limit": 8}},
    }

    t.args = types.SimpleNamespace(gradient_accumulation_steps=4, seed=123)
    t.model = types.SimpleNamespace(device=torch.device("cpu"))

    # Minimal async state.
    from collections import deque

    t._stage2_async_ready = deque()
    t._stage2_async_ready_lock = threading.Lock()
    t._stage2_async_ver = 0
    t._stage2_async_drop_stale_total = 0
    t._stage2_async_drop_oldest_total = 0

    # No dist.
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False, raising=False)

    assert (
        t._stage2_async_decide_step_kind(global_step=0, policy_wants_b=True) == "A"
    )


def test_async_step_kind_selects_b_when_queue_has_gas(monkeypatch):
    # If each rank has >= GAS ready packs, scheduled B is feasible.
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"b_ratio": 1.0},
        "channel_b": {"mode": "async", "async": {"queue_limit": 8}},
    }

    t.args = types.SimpleNamespace(gradient_accumulation_steps=4, seed=123)
    t.model = types.SimpleNamespace(device=torch.device("cpu"))

    from collections import deque

    t._stage2_async_ready = deque()
    t._stage2_async_ready_lock = threading.Lock()
    t._stage2_async_ver = 0
    t._stage2_async_drop_stale_total = 0
    t._stage2_async_drop_oldest_total = 0

    from src.trainers.stage2_ab import Stage2AsyncReadyPack

    for _ in range(4):
        t._stage2_async_ready.append(Stage2AsyncReadyPack(ver=0, batch={}))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: False, raising=False)

    assert (
        t._stage2_async_decide_step_kind(global_step=0, policy_wants_b=True) == "B"
    )


def test_b_ratio_realized_tracks_optimizer_steps_once():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t._stage2_ab_realized_last_gs = None

    t._stage2_record_realized_step(global_step=0, executed_b=False)
    t._stage2_record_realized_step(global_step=0, executed_b=True)  # same step, ignored
    t._stage2_record_realized_step(global_step=1, executed_b=True)

    assert pytest.approx(t._stage2_b_ratio_realized(), rel=1e-6) == 0.5


def test_merge_rollout_matching_batch_metrics_preserves_existing_keys():
    t = _make_min_trainer(n_softctx_iter=1)
    batch = {"_rollout_matching_batch_metrics": {"rollout/backend_vllm": 1.0}}
    t._merge_rollout_matching_batch_metrics(
        batch,
        {
            "stage2_ab/async/ver": 3.0,
            "rollout/backend_vllm": 2.0,
        },
    )
    bm = batch.get("_rollout_matching_batch_metrics")
    assert isinstance(bm, dict)
    assert bm["stage2_ab/async/ver"] == 3.0
    assert bm["rollout/backend_vllm"] == 2.0
