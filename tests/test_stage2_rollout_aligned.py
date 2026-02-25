from __future__ import annotations

import re
import sys
import threading
import types
from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch

from src.config.prompts import build_dense_user_prompt
from src.trainers.stage2_rollout_aligned import (
    GTObject,
    RolloutMatchingSFTTrainer,
    _PendingTrainRolloutLog,
    _build_labels_and_coord_targets_for_batch,
    _build_labels_and_coord_targets_for_sample,
    _per_server_rank_request_caps,
    _serialize_append_fragment,
    _sinkhorn_barycentric_targets,
    hungarian_match_maskiou,
    parse_rollout_for_matching,
)
from src.trainers.stage2_two_channel import Stage2ABTrainingTrainer


def test_stage2_two_channel_reuses_rollout_aligned_eval_contract() -> None:
    assert Stage2ABTrainingTrainer.evaluate is RolloutMatchingSFTTrainer.evaluate
    assert Stage2ABTrainingTrainer.prediction_step is RolloutMatchingSFTTrainer.prediction_step


class _DummyTokenizerRM:
    """Minimal tokenizer stub for token-aligned rollout parsing tests.

    - coord tokens <|coord_k|> map to ids 0..999
    - all other characters map to ids >= 1000 (1 char = 1 token)
    - supports a special fused token id that decodes to ']}'
    """

    _coord_re = re.compile(r"<\|coord_(\d{1,4})\|>")

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self._next = 1000
        # Fused ']}' token
        self.fused_brace_id = 2000
        self._id_to_piece[self.fused_brace_id] = "]}"

    def convert_tokens_to_ids(self, tokens):
        out = []
        for tok in tokens:
            m = self._coord_re.fullmatch(tok)
            if m:
                out.append(int(m.group(1)))
            else:
                out.append(-1)
        return out

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        pieces = []
        for tid in ids:
            tid = int(tid)
            if 0 <= tid <= 999:
                pieces.append(f"<|coord_{tid}|>")
            else:
                pieces.append(self._id_to_piece.get(tid, ""))
        return "".join(pieces)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        # Special-case token-internal cuts: allow encoding of single braces.
        if text == "]}":
            return [self.fused_brace_id]
        if text and text in self._char_to_id:
            return [self._char_to_id[text]]

        ids = []
        i = 0
        while i < len(text):
            if text.startswith("<|coord_", i):
                j = text.find("|>", i)
                if j < 0:
                    raise ValueError("unterminated coord token in dummy encode")
                tok = text[i : j + 2]
                m = self._coord_re.fullmatch(tok)
                if not m:
                    raise ValueError(f"bad coord token: {tok}")
                ids.append(int(m.group(1)))
                i = j + 2
                continue
            ch = text[i]
            tid = self._char_to_id.get(ch)
            if tid is None:
                tid = self._next
                self._next += 1
                self._char_to_id[ch] = tid
                self._id_to_piece[tid] = ch
            ids.append(tid)
            i += 1
        return ids


class _DummySession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DummyVLLMClient:
    def __init__(self) -> None:
        self.sessions = [_DummySession(), _DummySession()]
        self.close_calls = 0

    def close_communicator(self) -> None:
        self.close_calls += 1


class _FakeHTTPResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = int(status_code)
        self.text = ""

    def json(self):
        return self._payload


class _FakeHTTPSession:
    def __init__(self, payload, payload_log):
        self._payload = payload
        self._payload_log = payload_log

    def post(self, url, json, timeout):
        self._payload_log.append({"url": url, "json": json, "timeout": timeout})
        return _FakeHTTPResponse(self._payload, status_code=200)


@dataclass
class _FakeRequestConfig:
    n: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    stop: list[str]
    return_details: bool
    seed: int | None = None


def _make_rollout_server_trainer():
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {
        "decode_mode": "greedy",
        "max_new_tokens": 8,
        "repetition_penalty": 1.0,
    }
    trainer.state = types.SimpleNamespace(global_step=3)
    trainer._vllm_server_last_logged_step = -1
    # Avoid real network calls in unit tests; treat the fake server as 1 DP replica.
    trainer._vllm_server_cached_world_sizes = [1]

    trainer._decoding_params = lambda: (0.0, 1.0, -1)
    trainer._derive_rollout_seed_base = lambda global_step: 17 + int(global_step)
    trainer._sync_vllm_server_rollout_model_if_needed = lambda: None
    trainer._vllm_server_specs = lambda: [
        {"base_url": "http://127.0.0.1:9000", "group_port": 19000}
    ]
    trainer._vllm_server_timeouts = lambda: (30.0, 30.0)
    trainer._vllm_server_infer_guard = lambda: nullcontext()
    trainer._effective_vllm_server_sync_mode = lambda: "full"
    trainer._maybe_debug_dump_vllm_server_rollouts = lambda **_kwargs: None
    return trainer


def test_shutdown_vllm_server_client_closes_resources():
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer._vllm_server_client_lock = threading.Lock()
    client = _DummyVLLMClient()
    trainer._vllm_server_client = client
    trainer._vllm_server_comm_inited = True

    trainer._shutdown_vllm_server_client(
        close_communicator=True,
        close_sessions=True,
    )

    assert client.close_calls == 1
    assert all(s.closed for s in client.sessions)
    assert trainer._vllm_server_client is None
    assert trainer._vllm_server_comm_inited is False


def test_vllm_server_timeouts_allow_null_or_non_positive_infer_timeout() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    trainer._vllm_server_cfg = lambda: {"timeout_s": 60.0, "infer_timeout_s": None}
    timeout_s, infer_timeout_s = trainer._vllm_server_timeouts()
    assert timeout_s == pytest.approx(60.0)
    assert infer_timeout_s is None

    trainer._vllm_server_cfg = lambda: {"timeout_s": 60.0, "infer_timeout_s": 0}
    timeout_s, infer_timeout_s = trainer._vllm_server_timeouts()
    assert timeout_s == pytest.approx(60.0)
    assert infer_timeout_s is None


def test_rollout_decode_batch_size_per_rank_fails_fast_on_infeasible_topology(
    monkeypatch,
) -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer._decode_batch_size = lambda: 1
    trainer._rollout_backend = lambda: "vllm"
    trainer._vllm_mode = lambda: "server"
    trainer._vllm_server_world_sizes = lambda: [1]

    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)

    with pytest.raises(ValueError, match="decode_batch_size cap is infeasible"):
        trainer._rollout_decode_batch_size_per_rank()


def test_per_server_rank_caps_preserve_per_rank_chunk_on_multi_server_topology() -> None:
    # Regression for the [1, 1] two-server topology where each rank must still
    # receive one request when per-rank chunk is 1.
    caps_rank0 = _per_server_rank_request_caps(
        per_rank_chunk_size=1,
        server_world_sizes=[1, 1],
        learner_world_size=2,
        learner_rank=0,
    )
    caps_rank1 = _per_server_rank_request_caps(
        per_rank_chunk_size=1,
        server_world_sizes=[1, 1],
        learner_world_size=2,
        learner_rank=1,
    )

    assert sum(caps_rank0) == 1
    assert sum(caps_rank1) == 1
    assert [a + b for a, b in zip(caps_rank0, caps_rank1)] == [1, 1]


def test_rollout_many_enforces_server_chunk_cap_for_all_callers() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {}
    trainer.template = types.SimpleNamespace(system=None)

    call_samples: list[list[dict[str, object]]] = []
    call_debug_samples: list[list[dict[str, object]]] = []
    call_offsets: list[int] = []

    def _capture_rollout_many_vllm(
        samples,
        *,
        debug_samples=None,
        request_index_offset=0,
    ):
        call_samples.append(list(samples))
        call_debug_samples.append(
            list(debug_samples) if debug_samples is not None else []
        )
        call_offsets.append(int(request_index_offset))
        return [([1], "{}", "greedy", [2]) for _ in samples]

    trainer._rollout_backend = lambda: "vllm"
    trainer._vllm_mode = lambda: "server"
    trainer._rollout_decode_batch_size_per_rank = lambda: 2
    trainer._rollout_many_vllm = _capture_rollout_many_vllm

    original_samples = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"prompt-{i}"},
                {"role": "assistant", "content": '{"objects": [{"desc": "cat", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'},
            ]
        }
        for i in range(5)
    ]

    out = trainer._rollout_many(original_samples)

    assert len(out) == 5
    assert [len(x) for x in call_samples] == [2, 2, 1]
    assert call_offsets == [0, 2, 4]

    for chunk in call_samples:
        assert all(sample["messages"][-1]["role"] == "user" for sample in chunk)

    for chunk in call_debug_samples:
        assert all(sample["messages"][-1]["role"] == "assistant" for sample in chunk)


def test_vllm_server_rollout_uses_no_http_timeout_when_infer_timeout_disabled(
    monkeypatch,
):
    trainer = _make_rollout_server_trainer()

    captured_payloads: list[dict] = []
    monkeypatch.setitem(
        sys.modules,
        "swift.llm",
        types.SimpleNamespace(RequestConfig=_FakeRequestConfig),
    )

    response_payload = [
        {
            "response": {
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "message": {"content": "{}"},
                        "token_ids": [3, 4],
                        "finish_reason": "stop",
                    }
                ],
            }
        }
    ]
    fake_client = types.SimpleNamespace(
        sessions=[_FakeHTTPSession(response_payload, captured_payloads)]
    )
    trainer._ensure_vllm_server_client = lambda: fake_client
    trainer._vllm_server_timeouts = lambda: (30.0, None)

    sample = {"messages": [{"role": "user", "content": "ping"}]}
    trainer._rollout_many_vllm_server([sample])

    assert captured_payloads
    assert captured_payloads[0]["timeout"] is None


def test_rollout_many_passes_untrimmed_samples_for_server_debug_dump():
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {}
    trainer.template = types.SimpleNamespace(system=None)

    captured: dict[str, object] = {}

    def _capture_rollout_many_vllm(
        samples,
        *,
        debug_samples=None,
        request_index_offset=0,
    ):
        captured["samples"] = samples
        captured["debug_samples"] = debug_samples
        captured["request_index_offset"] = int(request_index_offset)
        return [([1], "{}", "greedy", [2])]

    trainer._rollout_backend = lambda: "vllm"
    trainer._vllm_mode = lambda: "server"
    trainer._rollout_decode_batch_size_per_rank = lambda: 4
    trainer._rollout_many_vllm = _capture_rollout_many_vllm

    original_samples = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": '{"objects": [{"desc": "cat", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'},
            ]
        }
    ]

    out = trainer._rollout_many(original_samples)

    assert out == [([1], "{}", "greedy", [2])]

    rollout_samples = captured["samples"]
    debug_samples = captured["debug_samples"]

    assert isinstance(rollout_samples, list) and rollout_samples
    assert rollout_samples[0]["messages"][-1]["role"] == "user"

    assert isinstance(debug_samples, list) and debug_samples
    assert debug_samples[0]["messages"][-1]["role"] == "assistant"
    assert captured["request_index_offset"] == 0


def test_reduce_train_rollout_log_payload_global_uses_ddp_max_for_p99(monkeypatch):
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    monkeypatch.setattr(dist, "ReduceOp", _FakeReduceOp, raising=False)

    def _fake_all_gather_object(out, obj):
        for i in range(len(out)):
            out[i] = list(obj)

    monkeypatch.setattr(dist, "all_gather_object", _fake_all_gather_object)

    def _fake_all_reduce(tensor: torch.Tensor, op: str) -> None:
        if op == _FakeReduceOp.SUM:
            # Simulate two identical learner ranks for SUM-reduced metrics.
            tensor.mul_(2.0)
        elif op == _FakeReduceOp.MAX:
            # Max-reduced rollout percentiles and timer use peer maxima.
            if int(tensor.numel()) >= 1:
                tensor[0] = torch.maximum(
                    tensor[0],
                    torch.tensor(20.0, dtype=tensor.dtype, device=tensor.device),
                )
            if int(tensor.numel()) >= 2:
                tensor[1] = torch.maximum(
                    tensor[1],
                    torch.tensor(5.0, dtype=tensor.dtype, device=tensor.device),
                )

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)

    out = trainer._reduce_train_rollout_log_payload_global(
        {
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 4.0,
            "train/samples_total": 4.0,
            "rollout/gen_new_tokens_total": 32.0,
            "rollout/gen_new_tokens_mean": 8.0,
            "rollout/gen_new_tokens_p99": 12.0,
            "time/rollout_generate_s": 2.0,
            "rollout/gen_tokens_per_s": 16.0,
        }
    )

    assert out["rollout/parse_truncated_rate"] == pytest.approx(0.25)
    assert out["rollout/gen_new_tokens_p99"] == pytest.approx(20.0)
    assert out["rollout/gen_new_tokens_mean"] == pytest.approx(8.0)
    assert out["rollout/gen_tokens_per_s"] == pytest.approx(12.8)
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out


def test_build_rollout_metrics_from_meta_skips_inactive_rollout_steps() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    out = trainer._build_rollout_metrics_from_meta(
        [
            {
                "decode_mode": "none",
                "rollout_len": 0,
                "gt_objects": 2,
                "valid_pred_objects": 2,
            }
        ]
    )

    assert out == {}


def test_build_rollout_metrics_from_meta_uses_counter_suffixes() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer._cfg = lambda _k, default=None: default
    trainer._decoding_params = lambda: (0.0, 1.0, -1)

    metrics = trainer._build_rollout_metrics_from_meta(
        [
            {
                "rollout_len": 8,
                "decode_mode": "greedy",
                "gt_objects": 2,
                "matched_for_supervision": 1,
                "valid_pred_objects": 3,
                "excluded_from_supervision": 0,
                "parse_dropped_invalid": 1,
                "parse_dropped_ambiguous": 0,
                "parse_truncated": False,
                "gating_rejections": 0,
                "matched_maskiou_sum": 0.8,
                "matched_maskiou_count": 1,
                "fn_count": 1,
                "prefix_coord_target_bins": [1, 2],
                "tail_ignore_pos": [7],
                "prompt_len": 10,
                "prefix_len": 12,
                "train_len": 16,
                "encoded_len": 20,
            }
        ]
    )

    assert metrics["rollout/gt_objects_total"] == pytest.approx(2.0)
    assert metrics["rollout/valid_pred_objects_total"] == pytest.approx(3.0)
    assert metrics["rollout/fp_total"] == pytest.approx(2.0)
    assert metrics["rollout/fn_total"] == pytest.approx(1.0)
    assert metrics["rollout/fn_appended_total"] == pytest.approx(1.0)
    assert "rollout/gt_objects" not in metrics
    assert "rollout/valid_pred_objects" not in metrics
    assert "rollout/fp" not in metrics
    assert "rollout/fn" not in metrics
    assert "rollout/fn_appended" not in metrics


def test_reduce_train_rollout_log_payload_global_omits_parse_rate_without_parse_inputs() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    out = trainer._reduce_train_rollout_log_payload_global(
        {
            "train/samples_total": 4.0,
            "loss/ce": 1.5,
        }
    )

    assert out["train/samples_total"] == pytest.approx(4.0)
    assert out["loss/ce"] == pytest.approx(1.5)
    assert "rollout/parse_truncated_rate" not in out


def test_reduce_train_rollout_log_payload_global_strips_internal_underscore_keys() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer._cfg = lambda _k, default=None: default

    out = trainer._reduce_train_rollout_log_payload_global(
        {
            "train/samples_total": 2.0,
            "loss/ce": 1.0,
            "loss/coord": 0.5,
            "rollout/_matched_maskiou_sum": 1.2,
            "rollout/matched_maskiou_count": 2.0,
            "rollout/_sample_valid_pred_num": 1.0,
            "rollout/sample_valid_pred_rate": 0.5,
            "rollout/_desc_exact_ok": 1.0,
            "rollout/desc_pairs_total": 2.0,
            "rollout/desc_exact_acc_on_matched": 0.5,
        }
    )

    assert all(not str(k).startswith("rollout/_") for k in out.keys())
    assert out["loss/ce"] == pytest.approx(1.0)
    assert out["loss/coord"] == pytest.approx(0.5)
    assert out["rollout/matched_maskiou_mean"] == pytest.approx(0.6)
    assert out["rollout/sample_valid_pred_rate"] == pytest.approx(0.5)
    assert out["rollout/desc_exact_acc_on_matched"] == pytest.approx(0.5)


def test_build_train_rollout_log_payload_uses_segment_weighted_losses() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    pending = _PendingTrainRolloutLog()
    pending.add_micro(
        meta=[{"decode_mode": "none", "rollout_len": 0}],
        ce_loss=10.0,
        coord_loss=0.0,
        coord_prefix=0.0,
        coord_tail=0.0,
        time_forward_s=0.0,
        time_mask_build_s=0.0,
        batch_metrics=None,
    )
    pending.add_micro(
        meta=[{"decode_mode": "none", "rollout_len": 0} for _ in range(3)],
        ce_loss=20.0,
        coord_loss=0.0,
        coord_prefix=0.0,
        coord_tail=0.0,
        time_forward_s=0.0,
        time_mask_build_s=0.0,
        batch_metrics=None,
    )

    out = trainer._build_train_rollout_log_payload(pending)

    assert out["train/samples_total"] == pytest.approx(4.0)
    assert out["train/micro_steps"] == pytest.approx(2.0)
    assert out["loss/ce"] == pytest.approx((10.0 * 1.0 + 20.0 * 3.0) / 4.0)


def test_reduce_train_rollout_log_payload_global_weights_losses_by_sample_total(
    monkeypatch,
) -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0, raising=False)

    class _FakeReduceOp:
        SUM = "sum"
        MAX = "max"

    monkeypatch.setattr(dist, "ReduceOp", _FakeReduceOp, raising=False)

    def _fake_all_gather_object(out, obj):
        for i in range(len(out)):
            out[i] = list(obj)

    monkeypatch.setattr(dist, "all_gather_object", _fake_all_gather_object)

    def _fake_all_reduce(tensor: torch.Tensor, op: str) -> None:
        if op == _FakeReduceOp.SUM:
            assert int(tensor.numel()) == 2
            a = float(tensor[0].item())
            b = float(tensor[1].item())

            # Local rank0 payload represents 1 segment with mean loss 10.
            # Peer rank represents 3 segments with mean loss 20.
            # Expected global mean = (10*1 + 20*3) / 4 = 17.5.
            if abs(a - 10.0) < 1e-6 and abs(b - 1.0) < 1e-6:
                tensor[0] = torch.tensor(70.0, dtype=tensor.dtype, device=tensor.device)
                tensor[1] = torch.tensor(4.0, dtype=tensor.dtype, device=tensor.device)
            elif abs(a - 1.0) < 1e-6 and abs(b - 10.0) < 1e-6:
                tensor[0] = torch.tensor(4.0, dtype=tensor.dtype, device=tensor.device)
                tensor[1] = torch.tensor(70.0, dtype=tensor.dtype, device=tensor.device)
            else:
                raise AssertionError(f"unexpected SUM tensor values: {a}, {b}")
        elif op == _FakeReduceOp.MAX:
            return

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)

    out = trainer._reduce_train_rollout_log_payload_global(
        {
            "train/samples_total": 1.0,
            "loss/ce": 10.0,
        }
    )

    assert out["train/samples_total"] == pytest.approx(4.0)
    assert out["loss/ce"] == pytest.approx(17.5)
    assert "loss/ce_total" not in out


def test_evaluate_emits_rollout_metrics_and_runs_callback(monkeypatch) -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    class _DummyEvalModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

    trainer.model = _DummyEvalModel()
    trainer.args = types.SimpleNamespace()
    trainer.state = types.SimpleNamespace(global_step=7)
    trainer.control = types.SimpleNamespace(tag="ctrl")
    trainer.template = types.SimpleNamespace(tokenizer=_DummyTokenizerRM())
    trainer._cfg = lambda _k, default=None: default
    trainer._desc_monitor_cfg = lambda: {"enabled": False}
    trainer._coord_id_map = lambda: {i: i for i in range(1000)}

    samples = [{"sample_id": 0}, {"sample_id": 1}]
    trainer.get_eval_dataloader = lambda _eval_dataset=None: [
        [samples[0]],
        [samples[1]],
    ]
    trainer._rollout_many = lambda batch: [
        ([100 + int(item["sample_id"])], "{}", "greedy", []) for item in batch
    ]

    parse_map = {
        100: types.SimpleNamespace(
            response_token_ids=[100],
            valid_objects=[
                    types.SimpleNamespace(
                        index=0,
                        geom_type="bbox_2d",
                        coord_token_indices=[0, 1, 2, 3],
                        desc="",
                    )
            ],
            dropped_invalid=0,
            dropped_ambiguous=0,
            truncated=False,
        ),
        101: types.SimpleNamespace(
            response_token_ids=[101],
            valid_objects=[
                    types.SimpleNamespace(
                        index=0,
                        geom_type="bbox_2d",
                        coord_token_indices=[0, 1, 2, 3],
                        desc="",
                    )
            ],
            dropped_invalid=1,
            dropped_ambiguous=1,
            truncated=True,
        ),
    }

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned.parse_rollout_for_matching",
        lambda **kwargs: parse_map[int(kwargs["response_token_ids"][0])],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._points_from_coord_tokens",
        lambda **_kwargs: [0, 0, 10, 10],
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._extract_gt_objects",
        lambda _sample: [
            GTObject(index=0, geom_type="bbox", points_norm1000=[0, 0, 10, 10], desc="")
        ],
    )

    match_calls = {"idx": 0}

    def _fake_hungarian(**_kwargs):
        idx = int(match_calls["idx"])
        match_calls["idx"] = idx + 1
        if idx == 0:
            return types.SimpleNamespace(
                matched_pairs=[(0, 0)],
                fp_pred_indices=[],
                fn_gt_indices=[],
                gating_rejections=0,
                matched_maskiou_sum=0.8,
                matched_maskiou_count=1,
            )
        return types.SimpleNamespace(
            matched_pairs=[],
            fp_pred_indices=[0],
            fn_gt_indices=[0],
            gating_rejections=1,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned.hungarian_match_maskiou",
        _fake_hungarian,
    )

    logged_metrics: dict[str, float] = {}
    trainer.log = lambda metrics: logged_metrics.update(dict(metrics))

    callback_metrics: list[dict[str, float]] = []

    def _on_evaluate(args, state, control, metrics):
        callback_metrics.append(dict(metrics))
        return control

    trainer.callback_handler = types.SimpleNamespace(on_evaluate=_on_evaluate)

    metrics = trainer.evaluate()

    assert trainer.model.training is True
    assert callback_metrics and callback_metrics[0] == metrics
    assert logged_metrics == metrics

    assert metrics["eval_rollout/precision"] == pytest.approx(0.5)
    assert metrics["eval_rollout/recall"] == pytest.approx(0.5)
    assert metrics["eval_rollout/f1"] == pytest.approx(0.5)
    assert metrics["eval_rollout/fp"] == pytest.approx(1.0)
    assert metrics["eval_rollout/fn"] == pytest.approx(1.0)
    assert metrics["eval_rollout/gating_rejections"] == pytest.approx(1.0)
    assert metrics["eval_rollout/parse_truncated_rate"] == pytest.approx(0.5)
    assert metrics["eval_rollout/sample_valid_pred_rate"] == pytest.approx(1.0)
    assert metrics["eval_rollout/sample_any_match_rate"] == pytest.approx(0.5)
    assert metrics["eval_rollout/matched_maskiou_mean"] == pytest.approx(0.8)
    assert "eval_loss" not in metrics
    assert "eval_time/runtime_s" in metrics


def test_rollout_many_overrides_last_user_prompt_for_eval_variant() -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)
    trainer.rollout_matching_cfg = {
        "rollout_backend": "hf",
        "object_ordering": "sorted",
    }
    trainer.object_field_order = "geometry_first"

    captured: dict[str, object] = {}

    def _fake_rollout_many_hf(samples):
        captured["samples"] = samples
        return [([], "{}", "greedy", []) for _ in samples]

    trainer._rollout_many_hf = _fake_rollout_many_hf

    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "old prompt"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"objects": []}'}],
            },
        ]
    }

    _ = trainer._rollout_many([sample], prompt_variant_override="coco_80")

    used_samples = captured.get("samples")
    assert isinstance(used_samples, list) and len(used_samples) == 1
    used_messages = used_samples[0]["messages"]
    assert isinstance(used_messages, list)
    # Rollout prompt must end at user turn (assistant completion stripped).
    assert len(used_messages) == 1
    user_content = used_messages[0]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "image"
    expected_prompt = build_dense_user_prompt(
        ordering="sorted",
        coord_mode="coord_tokens",
        prompt_variant="coco_80",
        object_field_order="geometry_first",
    )
    assert user_content[-1]["text"] == expected_prompt


def test_evaluate_emits_coco_map_metrics_when_eval_detection_enabled(
    monkeypatch,
) -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    class _DummyEvalModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

    trainer.model = _DummyEvalModel()
    trainer.args = types.SimpleNamespace()
    trainer.state = types.SimpleNamespace(global_step=11)
    trainer.control = types.SimpleNamespace(tag="ctrl")
    trainer.template = types.SimpleNamespace(tokenizer=_DummyTokenizerRM())
    trainer.rollout_matching_cfg = {
        "eval_prompt_variant": "coco_80",
        "object_ordering": "sorted",
        "eval_detection": {
            "enabled": True,
            "metrics": "coco",
            "score_mode": "constant",
            "constant_score": 1.0,
            "pred_score_source": "eval_rollout_constant",
            "pred_score_version": 2,
        },
    }
    trainer._desc_monitor_cfg = lambda: {"enabled": False}
    trainer._coord_id_map = lambda: {i: i for i in range(1000)}

    sample = {
        "sample_id": 0,
        "width": 640,
        "height": 480,
        "images": ["img.jpg"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "old prompt"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"objects": []}'}],
            },
        ],
    }
    trainer.get_eval_dataloader = lambda _eval_dataset=None: [[sample]]
    trainer._rollout_many = lambda batch, **_kwargs: [([100], "{}", "greedy", []) for _ in batch]

    parse_obj = types.SimpleNamespace(
        response_token_ids=[100],
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
                geom_type="bbox_2d",
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
            matched_maskiou_sum=0.9,
            matched_maskiou_count=1,
        ),
    )
    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._compute_eval_detection_coco_metrics",
        lambda **_kwargs: (
            {"bbox_AP": 0.123, "bbox_AP50": 0.456},
            {"empty_pred": 0},
        ),
    )

    logged_metrics: dict[str, float] = {}
    trainer.log = lambda metrics: logged_metrics.update(dict(metrics))
    trainer.callback_handler = types.SimpleNamespace(
        on_evaluate=lambda args, state, control, metrics: control
    )

    metrics = trainer.evaluate()

    assert logged_metrics == metrics
    assert metrics["eval_rollout/mAP"] == pytest.approx(0.123)
    assert metrics["eval_rollout/coco_eval_ok"] == pytest.approx(1.0)
    assert metrics["eval_rollout/prompt_variant_is_coco_80"] == pytest.approx(1.0)
    assert all(not k.startswith("eval_rollout/bbox_") for k in metrics)
    assert all(not k.startswith("eval_rollout/segm_") for k in metrics)


def test_evaluate_emits_coco_map_metrics_with_confidence_postop(monkeypatch) -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    class _DummyEvalModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

    trainer.model = _DummyEvalModel()
    trainer.args = types.SimpleNamespace()
    trainer.state = types.SimpleNamespace(global_step=11)
    trainer.control = types.SimpleNamespace(tag="ctrl")
    trainer.template = types.SimpleNamespace(tokenizer=_DummyTokenizerRM())
    trainer.rollout_matching_cfg = {
        "rollout_backend": "hf",
        "eval_prompt_variant": "coco_80",
        "object_ordering": "sorted",
        "eval_detection": {
            "enabled": True,
            "metrics": "coco",
            "score_mode": "confidence_postop",
            # These should be overridden by confidence_postop provenance.
            "pred_score_source": "eval_rollout_constant",
            "pred_score_version": 2,
            "confidence": {},
        },
    }
    trainer._desc_monitor_cfg = lambda: {"enabled": False}
    trainer._coord_id_map = lambda: {i: i for i in range(1000)}
    trainer._prepare_samples_for_rollout = lambda batch, **_kwargs: batch

    sample = {
        "sample_id": 0,
        "width": 640,
        "height": 480,
        "images": ["img.jpg"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "old prompt"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"objects": []}'}],
            },
        ],
    }
    trainer.get_eval_dataloader = lambda _eval_dataset=None: [[sample]]

    trainer._rollout_many_hf_traced = lambda batch: [
        ([100], "{}", "greedy", [], [0.0], ["tok"]) for _ in batch
    ]

    parse_obj = types.SimpleNamespace(
        response_token_ids=[100],
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
                geom_type="bbox_2d",
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
            matched_maskiou_sum=0.9,
            matched_maskiou_count=1,
        ),
    )

    monkeypatch.setattr(
        "src.eval.confidence_postop._compute_sample_confidence_objects",
        lambda **_kwargs: [
            {
                "object_idx": 0,
                "kept": True,
                "score": 0.9,
                "confidence": 0.9,
                "confidence_details": {},
            }
        ],
    )

    def _fake_coco(*, pred_records, eval_cfg):
        assert isinstance(eval_cfg, dict)
        assert len(pred_records) == 1
        rec = pred_records[0]
        assert rec["pred_score_source"] == "confidence_postop"
        assert rec["pred_score_version"] == 2
        assert rec["pred"][0]["score"] == pytest.approx(0.9)
        return {"bbox_AP": 0.123, "bbox_AP50": 0.456}, {"empty_pred": 0}

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._compute_eval_detection_coco_metrics",
        _fake_coco,
    )

    logged_metrics: dict[str, float] = {}
    trainer.log = lambda metrics: logged_metrics.update(dict(metrics))
    trainer.callback_handler = types.SimpleNamespace(
        on_evaluate=lambda args, state, control, metrics: control
    )

    metrics = trainer.evaluate()

    assert logged_metrics == metrics
    assert metrics["eval_rollout/mAP"] == pytest.approx(0.123)
    assert metrics["eval_rollout/coco_eval_ok"] == pytest.approx(1.0)
    assert metrics["eval_rollout/prompt_variant_is_coco_80"] == pytest.approx(1.0)


def test_evaluate_emits_coco_map_metrics_with_confidence_postop_vllm(
    monkeypatch,
) -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    class _DummyEvalModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

    trainer.model = _DummyEvalModel()
    trainer.args = types.SimpleNamespace()
    trainer.state = types.SimpleNamespace(global_step=11)
    trainer.control = types.SimpleNamespace(tag="ctrl")
    trainer.template = types.SimpleNamespace(tokenizer=_DummyTokenizerRM())
    trainer.rollout_matching_cfg = {
        "rollout_backend": "vllm",
        "eval_prompt_variant": "coco_80",
        "object_ordering": "sorted",
        "eval_detection": {
            "enabled": True,
            "metrics": "coco",
            "score_mode": "confidence_postop",
            "pred_score_source": "eval_rollout_constant",
            "pred_score_version": 2,
            "confidence": {},
        },
    }
    trainer._desc_monitor_cfg = lambda: {"enabled": False}
    trainer._coord_id_map = lambda: {i: i for i in range(1000)}
    trainer._prepare_samples_for_rollout = lambda batch, **_kwargs: batch

    sample = {
        "sample_id": 0,
        "width": 640,
        "height": 480,
        "images": ["img.jpg"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "old prompt"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"objects": []}'}],
            },
        ],
    }
    trainer.get_eval_dataloader = lambda _eval_dataset=None: [[sample]]

    called = {"vllm": 0}

    def _fake_vllm_traced(batch):
        called["vllm"] += 1
        return [([100], "{}", "greedy", [], [0.0], ["tok"]) for _ in batch]

    trainer._rollout_many_vllm_traced = _fake_vllm_traced

    parse_obj = types.SimpleNamespace(
        response_token_ids=[100],
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
                geom_type="bbox_2d",
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
            matched_maskiou_sum=0.9,
            matched_maskiou_count=1,
        ),
    )

    monkeypatch.setattr(
        "src.eval.confidence_postop._compute_sample_confidence_objects",
        lambda **_kwargs: [
            {
                "object_idx": 0,
                "kept": True,
                "score": 0.9,
                "confidence": 0.9,
                "confidence_details": {},
            }
        ],
    )

    def _fake_coco(*, pred_records, eval_cfg):
        assert isinstance(eval_cfg, dict)
        assert len(pred_records) == 1
        rec = pred_records[0]
        assert rec["pred_score_source"] == "confidence_postop"
        assert rec["pred_score_version"] == 2
        assert rec["pred"][0]["score"] == pytest.approx(0.9)
        return {"bbox_AP": 0.123, "bbox_AP50": 0.456}, {"empty_pred": 0}

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._compute_eval_detection_coco_metrics",
        _fake_coco,
    )

    logged_metrics: dict[str, float] = {}
    trainer.log = lambda metrics: logged_metrics.update(dict(metrics))
    trainer.callback_handler = types.SimpleNamespace(
        on_evaluate=lambda args, state, control, metrics: control
    )

    metrics = trainer.evaluate()

    assert called["vllm"] == 1
    assert logged_metrics == metrics
    assert metrics["eval_rollout/mAP"] == pytest.approx(0.123)
    assert metrics["eval_rollout/coco_eval_ok"] == pytest.approx(1.0)
    assert metrics["eval_rollout/prompt_variant_is_coco_80"] == pytest.approx(1.0)


def test_evaluate_emits_zero_map_when_coco_eval_fails(monkeypatch) -> None:
    trainer = object.__new__(RolloutMatchingSFTTrainer)

    class _DummyEvalModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

    trainer.model = _DummyEvalModel()
    trainer.args = types.SimpleNamespace()
    trainer.state = types.SimpleNamespace(global_step=11)
    trainer.control = types.SimpleNamespace(tag="ctrl")
    trainer.template = types.SimpleNamespace(tokenizer=_DummyTokenizerRM())
    trainer.rollout_matching_cfg = {
        "eval_prompt_variant": "coco_80",
        "object_ordering": "sorted",
        "eval_detection": {
            "enabled": True,
            "metrics": "coco",
            "score_mode": "constant",
            "constant_score": 1.0,
            "pred_score_source": "eval_rollout_constant",
            "pred_score_version": 2,
        },
    }
    trainer._desc_monitor_cfg = lambda: {"enabled": False}
    trainer._coord_id_map = lambda: {i: i for i in range(1000)}

    sample = {
        "sample_id": 0,
        "width": 640,
        "height": 480,
        "images": ["img.jpg"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "old prompt"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"objects": []}'}],
            },
        ],
    }
    trainer.get_eval_dataloader = lambda _eval_dataset=None: [[sample]]
    trainer._rollout_many = lambda batch, **_kwargs: [([100], "{}", "greedy", []) for _ in batch]

    parse_obj = types.SimpleNamespace(
        response_token_ids=[100],
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
                geom_type="bbox_2d",
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
            matched_maskiou_sum=0.9,
            matched_maskiou_count=1,
        ),
    )

    def _raise_coco(**_kwargs):
        raise RuntimeError("forced coco failure")

    monkeypatch.setattr(
        "src.trainers.stage2_rollout_aligned._compute_eval_detection_coco_metrics",
        _raise_coco,
    )

    logged_metrics: dict[str, float] = {}
    trainer.log = lambda metrics: logged_metrics.update(dict(metrics))
    trainer.callback_handler = types.SimpleNamespace(
        on_evaluate=lambda args, state, control, metrics: control
    )

    metrics = trainer.evaluate()

    assert logged_metrics == metrics
    assert metrics["eval_rollout/coco_eval_ok"] == pytest.approx(0.0)
    assert metrics["eval_rollout/mAP"] == pytest.approx(0.0)
    assert all(not k.startswith("eval_rollout/bbox_") for k in metrics)
    assert all(not k.startswith("eval_rollout/segm_") for k in metrics)


def test_vllm_server_rollout_enforces_strict_per_server_rank_caps(monkeypatch):
    trainer = _make_rollout_server_trainer()
    trainer.rollout_matching_cfg["decode_batch_size"] = 1

    captured_payloads: list[dict] = []
    monkeypatch.setitem(
        sys.modules,
        "swift.llm",
        types.SimpleNamespace(RequestConfig=_FakeRequestConfig),
    )

    class _SequencedSession:
        def __init__(self, payload_log):
            self._payload_log = payload_log
            self._next = 0

        def post(self, url, json, timeout):
            self._payload_log.append({"url": url, "json": json, "timeout": timeout})
            n = len(json.get("infer_requests") or [])
            out = []
            for _ in range(n):
                idx = self._next
                self._next += 1
                out.append(
                    {
                        "response": {
                            "prompt_token_ids": [100 + idx],
                            "choices": [
                                {
                                    "message": {"content": f"r{idx}"},
                                    "token_ids": [200 + idx],
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                        "coordexp": {},
                    }
                )
            return _FakeHTTPResponse(out, status_code=200)

    fake_client = types.SimpleNamespace(sessions=[_SequencedSession(captured_payloads)])
    trainer._ensure_vllm_server_client = lambda: fake_client

    samples = [
        {"messages": [{"role": "user", "content": "q1"}]},
        {"messages": [{"role": "user", "content": "q2"}]},
        {"messages": [{"role": "user", "content": "q3"}]},
    ]

    outputs = trainer._rollout_many_vllm_server(samples)

    assert len(outputs) == 3
    assert [len(p["json"]["infer_requests"]) for p in captured_payloads] == [1, 1, 1]
    assert [int(p["json"]["request_config"]["seed"]) for p in captured_payloads] == [
        20,
        21,
        22,
    ]


def test_rollout_parse_preserves_appearance_order_and_prefix_cut():
    tok = _DummyTokenizerRM()
    text = (
        '{"objects": [{"desc": "a", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, '
        '{"desc": "b", "bbox_2d": [<|coord_5|>, <|coord_6|>, <|coord_7|>, <|coord_8|>]}]}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)

    assert [o.index for o in parsed.valid_objects] == [0, 1]
    assert parsed.invalid_rollout is False
    # Prefix ends after last object dict; top-level array/object are left open for append.
    assert parsed.prefix_text.endswith("}")
    assert parsed.prefix_text == text[:-2]


def test_rollout_parse_drops_invalid_objects_without_repair():
    tok = _DummyTokenizerRM()
    # objects[1] has only 3 bbox coords -> invalid and dropped.
    text = (
        '{"objects": ['
        '{"desc": "a", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, '
        '{"desc": "b", "bbox_2d": [<|coord_5|>, <|coord_6|>, <|coord_7|>]}, '
        '{"desc": "c", "bbox_2d": [<|coord_8|>, <|coord_9|>, <|coord_10|>, <|coord_11|>]}'
        ']}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert [o.index for o in parsed.valid_objects] == [0, 2]
    assert parsed.dropped_invalid >= 1


def test_rollout_parse_drops_order_violation_for_geometry_first():
    tok = _DummyTokenizerRM()
    text = (
        '{"objects": ['
        '{"desc": "a", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, '
        '{"bbox_2d": [<|coord_5|>, <|coord_6|>, <|coord_7|>, <|coord_8|>], "desc": "b"}'
        ']}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(
        tokenizer=tok,
        response_token_ids=ids,
        object_field_order="geometry_first",
    )

    assert [o.index for o in parsed.valid_objects] == [1]
    assert parsed.dropped_invalid >= 1
    assert parsed.dropped_invalid_by_reason.get("order_violation", 0) >= 1


def test_serialize_append_fragment_rejects_non_append_ready_prefix():
    tok = _DummyTokenizerRM()
    fn_obj = GTObject(
        index=0,
        geom_type="bbox_2d",
        points_norm1000=[5, 6, 7, 8],
        desc="fn",
    )
    with pytest.raises(ValueError, match="append-ready"):
        _serialize_append_fragment(fn_objects=[fn_obj], prefix_text="{}")
    _ = tok  # keep tokenizer fixture parity in this test module


def test_serialize_append_fragment_comma_policy_is_prefix_entry_aware():
    fn_objs = [
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[1, 2, 3, 4],
            desc="fn",
        )
    ]

    frag_empty = _serialize_append_fragment(
        fn_objects=fn_objs,
        prefix_text='{"objects": [',
    )
    assert frag_empty.startswith('{"')
    assert frag_empty.endswith("]}")

    prefix_with_entry = (
        '{"objects": [{"desc": "p", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}'
    )
    frag_non_empty = _serialize_append_fragment(
        fn_objects=fn_objs,
        prefix_text=prefix_with_entry,
    )
    assert frag_non_empty.startswith(", ")
    assert '"bbox_2d"' in frag_non_empty


def test_serialize_append_fragment_supports_geometry_first_field_order():
    frag_bbox = _serialize_append_fragment(
        fn_objects=[
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[1, 2, 3, 4],
                desc="bbox",
            )
        ],
        prefix_text='{"objects": [',
        object_field_order="geometry_first",
    )
    assert frag_bbox.index('"bbox_2d"') < frag_bbox.index('"desc"')

    frag_poly = _serialize_append_fragment(
        fn_objects=[
            GTObject(
                index=2,
                geom_type="poly",
                points_norm1000=[10, 20, 30, 20, 30, 40, 10, 40],
                desc="poly",
            )
        ],
        prefix_text='{"objects": [',
        object_field_order="geometry_first",
    )
    assert frag_poly.index('"poly"') < frag_poly.index('"desc"')


def test_rollout_parse_poly_captures_coord_indices_for_flat_arrays():
    tok = _DummyTokenizerRM()
    text = (
        '{"objects": [{"desc": "p", "poly": [<|coord_1|>, <|coord_2|>, <|coord_3|>, '
        '<|coord_4|>, <|coord_5|>, <|coord_6|>, <|coord_7|>, <|coord_8|>]}]}'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert len(parsed.valid_objects) == 1
    obj = parsed.valid_objects[0]
    assert obj.geom_type == "poly"
    assert len(obj.coord_token_indices) == 8


def test_rollout_parse_truncated_tail_is_suffix_trimmed():
    tok = _DummyTokenizerRM()
    # Truncated mid-objects[1]: only objects[0] should remain.
    text = (
        '{"objects": ['
        '{"desc": "a", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, '
        '{"desc": "b", "bbox_2d": [<|coord_5|>'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert parsed.truncated is True
    assert [o.index for o in parsed.valid_objects] == [0]
    assert parsed.prefix_text.endswith("}")


def test_rollout_parse_truncated_tail_keeps_all_previous_complete_objects():
    tok = _DummyTokenizerRM()
    # Truncated mid-objects[2]: objects[0] and objects[1] should remain.
    text = (
        '{"objects": ['
        '{"desc": "a", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, '
        '{"desc": "b", "bbox_2d": [<|coord_5|>, <|coord_6|>, <|coord_7|>, <|coord_8|>]}, '
        '{"desc": "c", "bbox_2d": [<|coord_9|>'
    )
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert parsed.truncated is True
    assert [o.index for o in parsed.valid_objects] == [0, 1]
    assert parsed.prefix_text.endswith("}")


def test_rollout_parse_handles_fused_closing_braces_token_internal_cut():
    tok = _DummyTokenizerRM()
    base = (
        '{"objects": [{"desc": "a", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    )
    ids = tok.encode(base, add_special_tokens=False)
    # Replace the final top-level ']}' with a single fused token.
    assert tok.decode(ids[-2:], clean_up_tokenization_spaces=False) == "]}"
    ids_fused = ids[:-2] + [tok.fused_brace_id]

    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids_fused)
    assert [o.index for o in parsed.valid_objects] == [0]
    # Prefix should exclude the top-level closing ']}' while preserving record closure.
    assert parsed.prefix_text.endswith("}")
    assert parsed.prefix_text == base[:-2]


def test_rollout_parse_strips_im_end_hard_stop():
    tok = _DummyTokenizerRM()
    base = (
        '{"objects": [{"desc": "a", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
        "<|im_end|>GARBAGE"
    )
    ids = tok.encode(base, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert "<|im_end|>" not in parsed.response_text
    assert "<|im_end|>" not in parsed.prefix_text
    # Prefix ends after last object value dict; top-level JSON is left open for append.
    assert parsed.prefix_text.endswith("}")


def test_rollout_parse_fallback_when_no_open_brace():
    tok = _DummyTokenizerRM()
    text = "NOT_JSON<|im_end|>"
    ids = tok.encode(text, add_special_tokens=False)
    parsed = parse_rollout_for_matching(tokenizer=tok, response_token_ids=ids)
    assert parsed.prefix_text == '{"objects": ['
    assert parsed.valid_objects == []
    assert parsed.invalid_rollout is True


def test_hungarian_matching_with_gating_and_dummy_augmentation():
    preds = [
        GTObject(
            index=1, geom_type="bbox_2d", points_norm1000=[100, 100, 200, 200], desc=""
        ),
        GTObject(
            index=2, geom_type="bbox_2d", points_norm1000=[500, 500, 600, 600], desc=""
        ),
    ]
    gts = [
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[100, 100, 200, 200],
            desc="gt1",
        ),
        GTObject(
            index=2,
            geom_type="bbox_2d",
            points_norm1000=[500, 500, 600, 600],
            desc="gt2",
        ),
    ]
    match = hungarian_match_maskiou(
        preds=preds,
        gts=gts,
        top_k=2,
        gate_threshold=0.5,
        mask_resolution=64,
        fp_cost=1.0,
        fn_cost=1.0,
    )
    assert sorted(match.matched_pairs) == [(0, 0), (1, 1)]
    assert match.fn_gt_indices == []
    assert match.fp_pred_indices == []

    # With disjoint boxes and a non-trivial gate, nothing matches -> all FN and FP.
    preds_far = [
        GTObject(index=1, geom_type="bbox_2d", points_norm1000=[0, 0, 50, 50], desc=""),
        GTObject(
            index=2, geom_type="bbox_2d", points_norm1000=[900, 900, 950, 950], desc=""
        ),
    ]
    match2 = hungarian_match_maskiou(
        preds=preds_far,
        gts=gts,
        top_k=2,
        gate_threshold=0.5,
        mask_resolution=64,
        fp_cost=1.0,
        fn_cost=1.0,
    )
    assert match2.matched_pairs == []
    assert match2.fn_gt_indices == [0, 1]
    assert match2.fp_pred_indices == [0, 1]


def test_sinkhorn_barycentric_targets_sanity_poly_involved():
    # Pred and GT are the same square -> barycentric targets should be close.
    pred = torch.tensor(
        [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
    )
    gt = pred.clone()
    out = _sinkhorn_barycentric_targets(
        pred_points=pred.numpy(),
        gt_points=gt.numpy(),
        epsilon=1.0,
        iters=50,
        cost="l2",
    )
    assert out.shape == (4, 2)
    assert abs(out - gt.numpy()).max() < 2.0


def test_loss_masking_invariants_prefix_and_tail():
    # Build a synthetic sequence with:
    # - prompt_len=5
    # - prefix_len=3 (positions 5..7)
    # - train_len=6  (assistant content positions 5..10)
    # tail is positions [8..10)
    prompt_len = 5
    prefix_len = 3
    train_len = 6
    # coord tokens ids for this test:
    coord_id_set = {10, 20}
    coord_id_to_bin = {10: 10, 20: 20}

    # Sequence length 12, with a coord token in prefix and one in tail.
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 10, 7, 8, 20, 9, 0], dtype=torch.long)
    labels, coord_pos, coord_bins, coord_is_prefix = (
        _build_labels_and_coord_targets_for_sample(
            input_ids_1d=input_ids,
            prompt_len=prompt_len,
            prefix_len=prefix_len,
            train_len=train_len,
            coord_id_set=coord_id_set,
            coord_id_to_bin=coord_id_to_bin,
            prefix_coord_pos=[1],  # assistant-local index -> full position 6
            prefix_coord_target_bins=[123],
        )
    )

    # Prefix tokens contribute no CE.
    assert labels[5].item() == -100
    assert labels[6].item() == -100  # coord token
    assert labels[7].item() == -100

    # Tail: non-coord tokens contribute to CE, coord tokens do not.
    assert labels[8].item() == 8
    assert labels[9].item() == -100  # coord token
    assert labels[10].item() == 9

    # Coord supervision includes both prefix slot and tail coord token.
    assert 6 in coord_pos
    assert 9 in coord_pos
    # Prefix uses provided target bin; tail uses bin derived from token id.
    prefix_idx = coord_pos.index(6)
    tail_idx = coord_pos.index(9)
    assert coord_bins[prefix_idx] == 123
    assert coord_bins[tail_idx] == 20
    assert coord_is_prefix[prefix_idx] is True
    assert coord_is_prefix[tail_idx] is False


def test_loss_masking_can_ignore_desc_like_tail_tokens():
    # Same setup as the invariant test, but ignore tail position 0 and 2
    # (simulating masking desc value tokens in the appended tail).
    prompt_len = 5
    prefix_len = 3
    train_len = 6
    coord_id_set = {10, 20}
    coord_id_to_bin = {10: 10, 20: 20}
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 10, 7, 8, 20, 9, 0], dtype=torch.long)
    labels, coord_pos, coord_bins, coord_is_prefix = (
        _build_labels_and_coord_targets_for_sample(
            input_ids_1d=input_ids,
            prompt_len=prompt_len,
            prefix_len=prefix_len,
            train_len=train_len,
            coord_id_set=coord_id_set,
            coord_id_to_bin=coord_id_to_bin,
            prefix_coord_pos=[1],
            prefix_coord_target_bins=[123],
            tail_ignore_pos=[0, 2],
        )
    )
    # Tail spans positions [8..10). Ignore relative 0 -> p=8, and relative 2 -> p=10.
    assert labels[8].item() == -100
    assert labels[9].item() == -100  # coord token
    assert labels[10].item() == -100

    # Coord supervision still includes both prefix and tail coord token.
    assert 6 in coord_pos
    assert 9 in coord_pos


def test_rollout_context_masking_full_idea_semantics_prefix_fn_fp_and_closure():
    prompt_len = 2
    prefix_len = 4
    train_len = 8

    coord_id_set = {10}
    coord_id_to_bin = {10: 10}

    # assistant span: [2, 10)
    # prefix: [2, 6), tail: [6, 10)
    input_ids = torch.tensor([1, 2, 30, 31, 32, 33, 40, 41, 10, 42, 0, 0], dtype=torch.long)

    labels, coord_pos, coord_bins, coord_is_prefix = _build_labels_and_coord_targets_for_sample(
        input_ids_1d=input_ids,
        prompt_len=prompt_len,
        prefix_len=prefix_len,
        train_len=train_len,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
        prefix_coord_pos=[],
        prefix_coord_target_bins=[],
        # ignore tail rel=0 and rel=3, but rel=3 is closure and must stay supervised
        tail_ignore_pos=[0, 3],
        # matched-prefix struct-only CE
        prefix_struct_pos=[0, 2],
        # FN desc token rel=1 is supervised by default
        tail_desc_pos=[1],
        # closure/EOS supervision (tail rel=3)
        tail_closure_pos=[3],
    )

    # Prefix matched-struct CE on rel=0,2 only (FP prefix spans masked).
    assert labels[2].item() == 30
    assert labels[3].item() == -100
    assert labels[4].item() == 32
    assert labels[5].item() == -100

    # Tail FN semantics: desc token is supervised, coord token masked, closure supervised.
    assert labels[6].item() == -100  # ignored non-closure
    assert labels[7].item() == 41    # desc supervised by default
    assert labels[8].item() == -100  # coord
    assert labels[9].item() == 42    # closure supervised even if listed in tail_ignore_pos

    # Tail coord token still contributes to coord supervision targets.
    assert coord_pos == [8]
    assert coord_bins == [10]
    assert coord_is_prefix == [False]


def test_packed_label_mask_equivalence_two_segments():
    # Two synthetic segments; packed-mode labels should equal concatenated per-segment labels.
    coord_id_set = {10, 20}
    coord_id_to_bin = {10: 10, 20: 20}

    # Segment 1: reuse the invariant pattern (with one coord in prefix and one in tail).
    prompt_len1 = 5
    prefix_len1 = 3
    train_len1 = 6
    input_ids1 = torch.tensor([1, 2, 3, 4, 5, 6, 10, 7, 8, 20, 9, 0], dtype=torch.long)
    labels1, cpos1, cbins1, cis1 = _build_labels_and_coord_targets_for_sample(
        input_ids_1d=input_ids1,
        prompt_len=prompt_len1,
        prefix_len=prefix_len1,
        train_len=train_len1,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
        prefix_coord_pos=[1],
        prefix_coord_target_bins=[123],
    )

    # Segment 2: different sizes to validate offset logic.
    prompt_len2 = 4
    prefix_len2 = 2
    train_len2 = 5
    # Place a coord token in the tail (assistant-local) and one prefix-supervised slot.
    input_ids2 = torch.tensor([11, 12, 13, 14, 15, 10, 20, 16, 17], dtype=torch.long)
    labels2, cpos2, cbins2, cis2 = _build_labels_and_coord_targets_for_sample(
        input_ids_1d=input_ids2,
        prompt_len=prompt_len2,
        prefix_len=prefix_len2,
        train_len=train_len2,
        coord_id_set=coord_id_set,
        coord_id_to_bin=coord_id_to_bin,
        prefix_coord_pos=[1],
        prefix_coord_target_bins=[999],
    )

    packed_ids = torch.cat([input_ids1, input_ids2], dim=0).unsqueeze(0)
    meta = [
        {
            "prompt_len": prompt_len1,
            "prompt_ids": input_ids1[:prompt_len1].tolist(),
            "prefix_len": prefix_len1,
            "train_len": train_len1,
            "encoded_len": int(input_ids1.numel()),
            "prefix_coord_pos": [1],
            "prefix_coord_target_bins": [123],
        },
        {
            "prompt_len": prompt_len2,
            "prompt_ids": input_ids2[:prompt_len2].tolist(),
            "prefix_len": prefix_len2,
            "train_len": train_len2,
            "encoded_len": int(input_ids2.numel()),
            "prefix_coord_pos": [1],
            "prefix_coord_target_bins": [999],
        },
    ]
    labels_packed, sb, spos, sbin, sis_prefix = (
        _build_labels_and_coord_targets_for_batch(
            input_ids=packed_ids,
            meta=meta,
            coord_id_set=coord_id_set,
            coord_id_to_bin=coord_id_to_bin,
        )
    )

    expected = torch.cat([labels1, labels2], dim=0)
    assert labels_packed.shape == (1, expected.numel())
    assert torch.equal(labels_packed[0], expected)

    # Coord supervision positions for segment 2 must be offset by len(segment 1).
    off = int(input_ids1.numel())
    expected_pos = set(cpos1) | {off + int(p) for p in cpos2}
    got_pos = set(int(p) for p in spos)
    assert got_pos == expected_pos

    # Bins should match (order not guaranteed).
    expected_bins = [int(x) for x in cbins1] + [int(x) for x in cbins2]
    assert sorted(int(x) for x in sbin) == sorted(expected_bins)

    # Packed mode should report batch index 0 for all supervision targets.
    assert set(int(b) for b in sb) == {0}

    # Prefix flags should be preserved (order not guaranteed).
    expected_is_prefix = [bool(x) for x in cis1] + [bool(x) for x in cis2]
    assert sorted(bool(x) for x in sis_prefix) == sorted(expected_is_prefix)


def test_packed_prompt_prefix_sanity_check_rejects_mismatch():
    coord_id_set = {10}
    coord_id_to_bin = {10: 10}

    input_ids1 = torch.tensor([1, 2, 3, 4, 5, 10, 6], dtype=torch.long)
    input_ids2 = torch.tensor([7, 8, 9, 10, 11, 0], dtype=torch.long)
    packed_ids = torch.cat([input_ids1, input_ids2], dim=0).unsqueeze(0)

    # Segment 2 prompt_ids intentionally wrong to trigger the strict sanity check.
    meta = [
        {
            "prompt_len": 3,
            "prompt_ids": input_ids1[:3].tolist(),
            "prefix_len": 1,
            "train_len": 3,
            "encoded_len": int(input_ids1.numel()),
            "prefix_coord_pos": [],
            "prefix_coord_target_bins": [],
        },
        {
            "prompt_len": 2,
            "prompt_ids": [999, 999],  # mismatch
            "prefix_len": 0,
            "train_len": 2,
            "encoded_len": int(input_ids2.numel()),
            "prefix_coord_pos": [],
            "prefix_coord_target_bins": [],
        },
    ]

    with pytest.raises(ValueError, match="prompt tokenization mismatch"):
        _build_labels_and_coord_targets_for_batch(
            input_ids=packed_ids,
            meta=meta,
            coord_id_set=coord_id_set,
            coord_id_to_bin=coord_id_to_bin,
        )


def test_adaptive_raw_microbatch_stacker_bumps_on_underfill() -> None:
    from src.trainers.stage2_rollout_aligned import _AdaptiveRawMicroBatchStacker

    class _DummyTrainer:
        def __init__(self) -> None:
            # Overestimated avg length would otherwise pick too few raw samples.
            self._rm_avg_segment_len = 1100.0
            # Last pack stats indicate underfill with 8 segments and an empty buffer.
            self._rm_last_pack_fill = 0.579
            self._rm_last_pack_segments = 8
            self._rm_last_pack_buffer_after = 0

        def _packing_length(self) -> int:
            return 12000

        def _packing_min_fill_ratio(self) -> float:
            return 0.7

        def _packing_buffer_cap(self) -> int:
            return 256

    trainer = _DummyTrainer()
    stacker = _AdaptiveRawMicroBatchStacker(dataloader=[], trainer=trainer)

    # Need >= ceil(8 * 0.7 / 0.579) == 10 to avoid repeating the same underfill.
    assert stacker._target_microbatch_size() >= 10
