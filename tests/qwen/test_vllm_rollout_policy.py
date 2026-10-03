"""CPU caller seams for resident acquisition; no engine/model allocation."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from src.qwen.native import NativeRequest
from src.qwen.generation import NativeGenerationPolicy
from src.qwen.vllm_rollout import VllmDoraRollout, _generate, _refresh_engine


def request(request_id="image", prompt=(10, 11)):
    return NativeRequest(request_id, "prompt", Image.new("RGB", (32, 32)),
                         expected_token_ids=prompt)


def paired_engine(requests, emitted, *, raw, policy):
    calls = []

    def generate(prompts, params, **kwargs):
        calls.append(("generate", prompts, params))
        return [SimpleNamespace(request_id=f"native-{i}", prompt_token_ids=r.expected_token_ids,
                                outputs=[SimpleNamespace(token_ids=ids,
                                    finish_reason="stop" if 99 in ids else "length",
                                    logprobs=[{t: SimpleNamespace(logprob=p)} for t, p in zip(ids, scores)])])
                for i, (r, ids, scores) in enumerate(zip(requests, emitted, policy, strict=True))]

    def collective_rpc(method, args=()):
        calls.append((method, args))
        if method == "coordexp_trace_finalize":
            return [{r.request_id: {"token_ids": ids, "raw_logprobs": a,
                                   "policy_logprobs": b, "excluded_async_suffix": 0}
                     for r, ids, a, b in zip(requests, emitted, raw, policy, strict=True)}]
        return [None]

    return SimpleNamespace(generate=generate, collective_rpc=collective_rpc, calls=calls)


def test_pair_trace_has_genuine_raw_noncoordinate_denominator():
    # Token 2 itself is unchanged; scaling coordinate token 0 still changes its
    # probability through the full vocabulary denominator.
    raw_logits = torch.tensor([4.0, 1.0, 2.0])
    policy_logits = torch.tensor([1.0, 1.0, 2.0])
    raw = float(raw_logits.log_softmax(0)[2])
    normalized = float(policy_logits.log_softmax(0)[2])
    r = request()
    engine = paired_engine([r], [[2]], raw=[[raw]], policy=[[normalized]])
    result = _generate(engine, [r], [1], 99, 0, True, identity="snapshot")[0]
    assert result.raw_logprobs == (raw,)
    assert result.policy_logprobs == (normalized,)
    assert result.raw_logprobs != result.policy_logprobs


def sampled_policy(**kwargs):
    return NativeGenerationPolicy(temperature=1, use_model_defaults=False, **kwargs)


def test_default_greedy_keeps_caller_contract_and_no_trace_rpc():
    r = request()
    engine = paired_engine([r], [[2]], raw=[[-2.0]], policy=[[-1.0]])
    result = _generate(engine, [r], [1], 99, 0, False, identity="s1")[0]
    assert result.token_ids == (2,) and result.trace is None
    assert [c[0] for c in engine.calls] == ["generate"]
    params = engine.calls[0][2][0]
    assert params.temperature == 0 and params.top_p == 1 and params.top_k == 0
    assert params.seed is None and params.logprobs is None
    assert params.allowed_token_ids is None and params.logit_bias is None
    assert params.extra_args == {"coordexp_paired_trace": {
        "request_id": "image", "snapshot_id": "s1", "max_new_tokens": 1}}


def test_seeded_fullsupport_mixed_budgets_preserves_pad_and_eos_pairs():
    requests = [request("early"), request("pad", (20, 21))]
    engine = paired_engine(requests, [[3, 99], [0, 4, 99]],
        raw=[[-3.0, -2.0], [-4.0, -3.0, -2.0]],
        policy=[[-1.0, -0.2], [-1.5, -1.0, -0.1]])
    receipt = {}
    results = _generate(engine, requests, [8, 3], 99, 0, True,
        sampled_policy(), [13, 29], identity="s1", receipt=receipt)
    assert [(r.request_id, r.token_ids) for r in results] == [
        ("early", (3, 99)), ("pad", (0, 4, 99))]
    assert results[1].trace.token_ids == (0, 4, 99)
    assert results[1].raw_logprobs == (-4.0, -3.0, -2.0)
    assert results[1].policy_logprobs == (-1.5, -1.0, -0.1)
    params = next(c[2] for c in engine.calls if c[0] == "generate")
    assert [p.seed for p in params] == [13, 29]
    assert [p.max_tokens for p in params] == [8, 3]
    assert all(p.temperature == 1 and p.logprobs == 0 and p.top_k == -1
               and p.top_p == 1 and p.min_p == 0 and p.repetition_penalty == 1
               and p.presence_penalty == p.frequency_penalty == 0
               and p.allowed_token_ids is None and p.logit_bias is None for p in params)
    assert engine.calls[0] == ("coordexp_trace_begin", ("s1", [
        {"request_id": "early", "snapshot_id": "s1", "max_new_tokens": 8},
        {"request_id": "pad", "snapshot_id": "s1", "max_new_tokens": 3}]))
    assert receipt["snapshot_id"] == "s1"
    assert [r["emitted_actions"] for r in receipt["requests"]] == [2, 3]
    assert all(r["excluded_async_suffix"] == 0 for r in receipt["requests"])


@pytest.mark.parametrize("policy,seeds", [
    (NativeGenerationPolicy(), None),
    (NativeGenerationPolicy(temperature=1), [1]),
    (sampled_policy(top_p=0.9), [1]),
    (sampled_policy(top_k=1), [1]),
    (sampled_policy(repetition_penalty=1.1), [1]),
    (NativeGenerationPolicy(temperature=0.5, use_model_defaults=False), [1]),
    (sampled_policy(), None), (sampled_policy(), []),
    (sampled_policy(), [True]), (sampled_policy(), [-1]),
    (sampled_policy(), [2**32]),
    (NativeGenerationPolicy(use_model_defaults=False), [1]),
    ({"temperature": 1}, [1]),
])
def test_unsupported_policy_and_unaligned_seeds_rejected_before_generation(policy, seeds):
    r = request()
    engine = Mock()
    with pytest.raises(ValueError):
        _generate(engine, [r], [1], 99, 0, False, policy, seeds, identity="s1")
    engine.generate.assert_not_called()
    engine.collective_rpc.assert_not_called()


def test_public_seeds_are_aligned_and_snapshot_validated_before_transport():
    runtime = object.__new__(VllmDoraRollout)
    runtime.identity = "s1"
    runtime._call = Mock(return_value="generated")
    r = request()
    assert runtime.generate([r], budgets=[1], eos_token_id=99, pad_token_id=0,
        identity="s1", policy=sampled_policy(), seeds=iter([7]), trace=True) == "generated"
    payload = runtime._call.call_args.args[1]
    assert payload[0] == "s1" and payload[7] == (7,) and payload[8] is False
    for kwargs in ({"identity": "old"}, {"identity": "s1", "policy": sampled_policy()},
                   {"identity": "s1", "policy": NativeGenerationPolicy()}):
        with pytest.raises(ValueError):
            runtime.generate([r], budgets=[1], eos_token_id=99, pad_token_id=0, **kwargs)
    assert runtime._call.call_count == 1


def test_generation_prompt_failure_aborts_active_trace_and_closes_images(monkeypatch):
    from src.qwen import vllm_rollout as v
    r = request()
    engine = paired_engine([r], [[2]], raw=[[-2.0]], policy=[[-1.0]])
    generate = engine.generate
    def bad_prompt(*args, **kwargs):
        outputs = generate(*args, **kwargs)
        outputs[0].prompt_token_ids = [12, 11]
        return outputs
    engine.generate = bad_prompt
    image = Mock()
    monkeypatch.setattr(v, "_open_image", lambda _: image)
    with pytest.raises(RuntimeError, match="prompt tokens"):
        _generate(engine, [r], [1], 99, 0, True, identity="s1")
    assert engine.calls[-1][0] == "coordexp_trace_abort"
    image.close.assert_called_once()


@pytest.mark.parametrize("change,message", [
    (lambda rows: rows.clear(), "identities"),
    (lambda rows: rows["image"].update(token_ids=[3]), "tokens differ"),
    (lambda rows: rows["image"].update(raw_logprobs=[]), "action count"),
    (lambda rows: rows["image"].update(policy_logprobs=[float("nan")]), "nonfinite"),
])
def test_paired_trace_result_fail_closed(change, message):
    r = request()
    engine = paired_engine([r], [[2]], raw=[[-2.0]], policy=[[-1.0]])
    rpc = engine.collective_rpc
    def corrupt(method, args=()):
        result = rpc(method, args=args)
        if method == "coordexp_trace_finalize":
            change(result[0])
        return result
    engine.collective_rpc = corrupt
    with pytest.raises(RuntimeError, match=message):
        _generate(engine, [r], [1], 99, 0, True, identity="s1")


def test_refresh_ack_requires_idle_weights_and_cache_before_return():
    events = []
    engine = SimpleNamespace(
        collective_rpc=lambda method, args: events.append(method) or [True],
        apply_model=lambda operation: events.append("weights") or ["s2"],
        reset_prefix_cache=lambda: events.append("cache") or True)
    assert _refresh_engine(engine, {}, {}, "s2") == "s2"
    assert events == ["coordexp_trace_assert_idle", "weights", "cache"]
    for fault, message in [("active", "active trace"), ("ack", "not acknowledged"),
                           ("cache", "prefix cache")]:
        events.clear()
        if fault == "active":
            engine.collective_rpc = Mock(side_effect=RuntimeError("active trace"))
        else:
            engine.collective_rpc = lambda method, args: events.append(method) or [True]
        engine.apply_model = lambda operation: events.append("weights") or (["old"] if fault == "ack" else ["s2"])
        engine.reset_prefix_cache = lambda: events.append("cache") or fault != "cache"
        with pytest.raises(RuntimeError, match=message):
            _refresh_engine(engine, {}, {}, "s2")
        if fault == "active":
            assert events == []
        elif fault == "ack":
            assert "cache" not in events


def test_failed_refresh_transport_never_updates_public_identity():
    runtime = object.__new__(VllmDoraRollout)
    runtime.identity = "s1"
    runtime._call = Mock(side_effect=RuntimeError("cache reset failed"))
    with patch("peft.get_peft_model_state_dict", return_value={"a": torch.ones(1)}):
        with pytest.raises(RuntimeError, match="cache reset"):
            runtime.refresh(None, SimpleNamespace(delta_tensors=lambda: {"e": torch.ones(1)}), identity="s2")
    assert runtime.identity == "s1"
