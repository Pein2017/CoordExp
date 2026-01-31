import types

import pytest
import torch
import torch.nn as nn

from src.trainers.rollout_matching_sft import parse_rollout_for_matching
from src.trainers.stage2_ab_training import (
    Stage2ABTrainingTrainer,
    _bbox_l1_giou_loss,
    _expectation_decode_coords,
    _extract_gt_bboxonly,
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


def test_pattern_schedule_repeats_deterministically():
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {"schedule": {"pattern": ["A", "A", "B"]}}
    t._stage2_channel_override = None
    got = [t._stage2_channel_for_step(i) for i in range(5)]
    assert got == ["A", "A", "B", "A", "A"]


def test_expectation_decode_is_mean_not_argmax():
    logits = torch.full((1, 1000), -100.0)
    logits[0, 0] = 0.0
    logits[0, 999] = 0.0
    out = _expectation_decode_coords(coord_logits=logits, temperature=1.0)
    assert float(out.item()) == pytest.approx(0.5, abs=1e-6)


def test_bbox_losses_stable_on_noncanonical_pred():
    pred = torch.tensor([[1.2, -0.1, -0.2, 0.5]], dtype=torch.float32)
    gt = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    l1, giou = _bbox_l1_giou_loss(pred_xyxy=pred, gt_xyxy=gt)
    assert torch.isfinite(l1).item()
    assert torch.isfinite(giou).item()


def _make_min_trainer(*, n_softctx_iter: int):
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {
        "schedule": {"pattern": ["A"]},
        "n_softctx_iter": int(n_softctx_iter),
        "softctx_temperature": 1.0,
        "bbox_l1_weight": 1.0,
        "bbox_giou_weight": 1.0,
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
    with pytest.raises(ValueError, match="polygons"):
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
        "schedule": {"pattern": ["B"]},
        "n_softctx_iter": 1,
        "softctx_temperature": 1.0,
        "desc_ce_weight": 1.0,
        # Turn off bbox losses so CE effect is isolated.
        "bbox_l1_weight": 0.0,
        "bbox_giou_weight": 0.0,
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


def test_channel_b_step_mode_runs_only_on_final_microstep(monkeypatch):
    # Minimal trainer stub: verify step-mode buffers raw samples and executes only
    # on the final micro-step of the accumulation window.
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {"schedule": {"pattern": ["B"]}, "channel_b": {"mode": "step"}}

    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._stage2_channel_override = None

    # Required by helper methods.
    t._stage2_post_rollout_segments = {"A": [], "B": []}
    t._stage2_b_step_gs = None
    t._stage2_b_step_micro = 0
    t._stage2_b_step_raw = []

    t.args = types.SimpleNamespace(
        gradient_accumulation_steps=4,
        train_batch_size=1,
        per_device_train_batch_size=1,
        seed=123,
    )
    t.state = types.SimpleNamespace(global_step=0)

    # training_step uses self.model.device for the dummy 0-loss tensor.
    t.model = types.SimpleNamespace(device=torch.device("cpu"))

    # Disable rollout-buffer path.
    t._maybe_init_rollout_buffer = lambda: None

    called = []

    def _fake_budgeted_train(model, *, raw_samples, global_step):
        called.append({"n": len(raw_samples), "gs": int(global_step)})
        return torch.tensor(3.0)

    t._stage2_b_step_budgeted_train = _fake_budgeted_train

    class _M:
        training = True

    m = _M()
    sample = {"messages": []}

    for _ in range(3):
        loss = t.training_step(m, [sample])
        assert float(loss.detach().cpu().item()) == pytest.approx(0.0)
        assert called == []

    loss = t.training_step(m, [sample])
    assert float(loss.detach().cpu().item()) == pytest.approx(3.0)
    assert called == [{"n": 4, "gs": 0}]


def test_channel_b_step_mode_stable_order_and_seed_base_deterministic(monkeypatch):
    # Regression: in step-budgeted mode, raw sample ordering across micro-steps is stable
    # and the step-level seed base is deterministic.
    t = Stage2ABTrainingTrainer.__new__(Stage2ABTrainingTrainer)
    t.stage2_ab_cfg = {"schedule": {"pattern": ["B"]}, "channel_b": {"mode": "step"}}

    t._stage2_pending_train_logs = {}
    t._rm_pending_train_logs = {}
    t._stage2_channel_override = None

    # Required by helper methods.
    t._stage2_post_rollout_segments = {"A": [], "B": []}
    t._stage2_b_step_gs = None
    t._stage2_b_step_micro = 0
    t._stage2_b_step_raw = []

    t.args = types.SimpleNamespace(
        gradient_accumulation_steps=4,
        train_batch_size=1,
        per_device_train_batch_size=1,
        seed=123,
    )
    t.state = types.SimpleNamespace(global_step=7)

    # training_step uses self.model.device for the dummy 0-loss tensor.
    t.model = types.SimpleNamespace(device=torch.device("cpu"))

    # Disable rollout-buffer path.
    t._maybe_init_rollout_buffer = lambda: None

    called = []

    def _fake_budgeted_train(model, *, raw_samples, global_step):
        ids = [int(s.get("id", -1)) for s in raw_samples]
        gs = int(global_step)
        seed_base = int(t._derive_rollout_seed_base(global_step=gs))
        called.append({"ids": ids, "gs": gs, "seed_base": seed_base})
        return torch.tensor(3.0)

    t._stage2_b_step_budgeted_train = _fake_budgeted_train

    class _M:
        training = True

    m = _M()

    def _run_cycle() -> None:
        for i in range(3):
            loss = t.training_step(m, [{"id": i, "messages": []}])
            assert float(loss.detach().cpu().item()) == pytest.approx(0.0)

        loss = t.training_step(m, [{"id": 3, "messages": []}])
        assert float(loss.detach().cpu().item()) == pytest.approx(3.0)

    _run_cycle()
    _run_cycle()

    expected_seed_base = int((123 + 7 * 1000003) & 0x7FFFFFFF)
    assert called == [
        {"ids": [0, 1, 2, 3], "gs": 7, "seed_base": expected_seed_base},
        {"ids": [0, 1, 2, 3], "gs": 7, "seed_base": expected_seed_base},
    ]
