"""CPU counterexamples for same-action, bounded resident likelihood traces."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import src.qwen.vllm_trace as trace


def descriptor(request_id, budget=4, snapshot="snapshot-1"):
    return dict(request_id=request_id, snapshot_id=snapshot, max_new_tokens=budget)


def sampling_params(metadata):
    return SimpleNamespace(
        extra_args={"coordexp_paired_trace": metadata.copy()},
        n=1, temperature=1.0, seed=37, top_p=1.0, top_k=-1, min_p=0.0,
        frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0,
        min_tokens=0, max_tokens=metadata["max_new_tokens"], logprobs=0,
        stop=[], ignore_eos=False, bad_words=[], _bad_words_token_ids=[],
        allowed_token_ids=None, logit_bias=None, structured_outputs=None,
        logits_processors=[], thinking_token_budget=None,
        repetition_detection=None, logprob_token_ids=None, prompt_logprobs=None,
    )


class FakeSampler(nn.Module):
    logprobs_mode = "raw_logprobs"

    def forward(self, logits, sampling_metadata):
        if getattr(self, "raise_error", False):
            raise RuntimeError("native sampler failed")
        selected = self.tokens.to(dtype=torch.int32).view(-1, 1)
        logprobs = logits.log_softmax(-1, dtype=torch.float32).gather(
            1, selected.long()
        )
        if getattr(self, "nonfinite", False):
            logprobs[0, 0] = float("nan")
        policy_ids = selected.clone()
        if getattr(self, "mismatch", False):
            policy_ids[0, 0] += 1
        return SimpleNamespace(
            sampled_token_ids=selected,
            logprobs_tensors=SimpleNamespace(
                logprob_token_ids=policy_ids, logprobs=logprobs,
            ),
        )


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.coordexp_dora_identity = "snapshot-1"

    def compute_logits(self, logits):
        trace.capture_raw_logits(self, logits)
        logits[:, 1] *= 2
        return logits


class FakeWorker(trace.CoordExpTraceWorkerExtension):
    def __init__(self, descriptors, *, asynchronous=True):
        self.model = FakeModel()
        self.model_runner = SimpleNamespace(
            sampler=FakeSampler(), requests={}, use_async_scheduling=asynchronous,
            speculative_config=None,
        )
        for index, metadata in enumerate(descriptors):
            self.model_runner.requests[f"internal-{index}"] = SimpleNamespace(
                prompt_token_ids=[8, 9, 10], sampling_params=sampling_params(metadata),
            )
        self.descriptors = descriptors

    def get_model(self):
        return self.model

    def begin(self):
        return self.coordexp_trace_begin("snapshot-1", self.descriptors)

    def finalize(self, ids):
        return self.coordexp_trace_finalize(
            "snapshot-1",
            [dict(request_id=key, token_ids=tokens) for key, tokens in ids.items()],
            [2],
        )


def step(worker, order, positions, tokens, discarded=None):
    runner = worker.model_runner
    runner.input_batch = SimpleNamespace(
        req_ids=[f"internal-{index}" for index in order], num_reqs=len(order),
        num_computed_tokens_cpu=np.array(positions) + 2,
    )
    runner.num_scheduled_tokens = SimpleNamespace(np=np.ones(len(order), dtype=int))
    runner.discard_request_mask = SimpleNamespace(
        np=np.array(discarded or [False] * len(order)),
    )
    runner.sampler.tokens = torch.tensor(tokens)
    raw = torch.tensor([[0.5, 1.5, -0.5, 2.5, 0.1]] * len(order))
    policy = worker.model.compute_logits(raw.clone())
    output = runner.sampler(logits=policy, sampling_metadata=SimpleNamespace(max_num_logprobs=0))
    return raw, policy, output


def test_reordering_compaction_mixed_budgets_pad_eos_and_verified_async_suffix():
    worker = FakeWorker([descriptor("a", 3), descriptor("b", 3), descriptor("c", 1)])
    assert worker.begin()["snapshot_id"] == "snapshot-1"
    raw, policy, _ = step(worker, [1, 0, 2], [0, 0, 0], [3, 0, 2])
    step(worker, [0, 1], [1, 1], [2, 4])
    step(worker, [1, 0, 2], [2, 2, 1], [2, 1, 4])
    result = worker.finalize({"c": [2], "a": [0, 2], "b": [3, 4, 2]})
    assert result["a"]["token_ids"] == [0, 2]  # Literal PAD is an action.
    assert result["a"]["excluded_async_suffix"] == 1
    assert result["b"]["excluded_async_suffix"] == 0
    assert result["c"]["dropped_budget_actions"] == 1
    assert result["a"]["raw_logprobs"][0] == pytest.approx(raw.log_softmax(-1)[0, 0].item())
    assert result["a"]["policy_logprobs"][0] == pytest.approx(policy.log_softmax(-1)[0, 0].item())
    assert result["a"]["raw_logprobs"] != result["a"]["policy_logprobs"]
    assert worker.coordexp_trace_assert_idle()
    assert worker.model._coordexp_paired_trace_state is None


def test_incomplete_prefill_is_discarded_and_step_does_not_synchronize(monkeypatch):
    worker = FakeWorker([descriptor("a", 1)])
    worker.begin()
    def forbidden(*args, **kwargs):
        raise AssertionError("per-step CPU transfer or scalar synchronization")
    original_tensor = torch.tensor
    def forbid_host_to_device(*args, **kwargs):
        if "device" in kwargs:
            raise AssertionError("per-step host-built device tensor")
        return original_tensor(*args, **kwargs)
    with monkeypatch.context() as scoped:
        for name in ("cpu", "item", "tolist"):
            scoped.setattr(torch.Tensor, name, forbidden)
        # CPU fixtures still construct their inputs. Any explicit device-targeted
        # torch.tensor call reproduces the same host construction used on CUDA.
        scoped.setattr(torch, "tensor", forbid_host_to_device)
        step(worker, [0], [-1], [4], [True])
        step(worker, [0], [0], [2])
    result = worker.finalize({"a": [2]})
    assert result["a"]["discarded_prefill_actions"] == 1


@pytest.mark.parametrize("positions,tokens,expected,match", [
    ([1], [2], [2], "positions"),
    ([0], [3], [2], "token"),
    ([0], [2], [3, 4], "positions"),
])
def test_incomplete_or_mismatched_evidence_fails_and_releases(positions, tokens, expected, match):
    worker = FakeWorker([descriptor("a")])
    worker.begin()
    step(worker, [0], positions, tokens)
    with pytest.raises(RuntimeError, match=match):
        worker.finalize({"a": expected})
    assert worker.coordexp_trace_assert_idle()


def test_duplicate_position_fails_before_trace_storage_can_grow():
    worker = FakeWorker([descriptor("a", 1)])
    worker.begin()
    step(worker, [0], [0], [2])
    with pytest.raises(RuntimeError, match="duplicate"):
        step(worker, [0], [0], [2])
    assert worker.model._coordexp_paired_trace_state.pending_raw_logits is None
    worker.coordexp_trace_abort()
    assert worker.coordexp_trace_assert_idle()


@pytest.mark.parametrize("asynchronous,emitted", [(False, [2]), (True, [3])])
def test_only_verified_post_eos_async_suffix_is_excluded(asynchronous, emitted):
    worker = FakeWorker([descriptor("a")], asynchronous=asynchronous)
    worker.begin()
    step(worker, [0], [0], emitted)
    step(worker, [0], [1], [4])
    with pytest.raises(RuntimeError, match="async.*EOS"):
        worker.finalize({"a": emitted})
    assert worker.coordexp_trace_assert_idle()


@pytest.mark.parametrize("mutation", ["model", "request", "snapshot", "budget"])
def test_stale_identity_and_changed_request_metadata_are_rejected(mutation):
    worker = FakeWorker([descriptor("a")])
    worker.begin()
    if mutation == "model":
        worker.model.coordexp_dora_identity = "snapshot-2"
    else:
        metadata = worker.model_runner.requests["internal-0"].sampling_params.extra_args["coordexp_paired_trace"]
        metadata[{"request": "request_id", "snapshot": "snapshot_id", "budget": "max_new_tokens"}[mutation]] = {"request": "alien", "snapshot": "snapshot-2", "budget": 9}[mutation]
    with pytest.raises(RuntimeError, match="identity|metadata"):
        step(worker, [0], [0], [2])
    worker.coordexp_trace_abort()
    assert worker.coordexp_trace_assert_idle()


def test_trace_off_warmup_retains_no_logits_or_trace_buffers():
    worker = FakeWorker([descriptor("a")])
    for _ in range(3):
        step(worker, [0], [0], [4])
    assert not hasattr(worker.model, "_coordexp_paired_trace_state")
    assert not worker.model_runner.sampler._forward_hooks
    worker.begin()
    step(worker, [0], [0], [2])
    worker.finalize({"a": [2]})
    for _ in range(3):
        step(worker, [0], [0], [4])
    assert worker.coordexp_trace_assert_idle()
    assert worker._coordexp_trace_state is None


def test_raw_overwrite_and_refresh_are_rejected_then_abort_releases():
    worker = FakeWorker([descriptor("a")])
    worker.begin()
    trace.capture_raw_logits(worker.model, torch.ones(1, 5))
    with pytest.raises(RuntimeError, match="pending"):
        trace.capture_raw_logits(worker.model, torch.ones(1, 5))
    with pytest.raises(RuntimeError, match="active"):
        trace.assert_trace_idle(worker.model)
    with pytest.raises(RuntimeError, match="active"):
        worker.coordexp_trace_assert_idle()
    worker.coordexp_trace_abort()
    trace.assert_trace_idle(worker.model)
    assert worker.coordexp_trace_assert_idle()


@pytest.mark.parametrize("failure", ["native", "scoring", "nonfinite", "policy_ids"])
def test_sampler_exceptions_and_invalid_pairs_release_pending_and_fail_closed(monkeypatch, failure):
    worker = FakeWorker([descriptor("a")])
    worker.begin()
    if failure == "native":
        worker.model_runner.sampler.raise_error = True
    elif failure == "scoring":
        def broken(*args, **kwargs):
            raise RuntimeError("selected scoring failed")
        monkeypatch.setattr(trace, "chosen_token_logprobs", broken)
    elif failure == "nonfinite":
        worker.model_runner.sampler.nonfinite = True
    else:
        worker.model_runner.sampler.mismatch = True
    if failure in ("native", "scoring"):
        with pytest.raises(RuntimeError, match="failed"):
            step(worker, [0], [0], [2])
        assert worker.model._coordexp_paired_trace_state.pending_raw_logits is None
        with pytest.raises(RuntimeError, match="failed"):
            worker.finalize({"a": [2]})
    else:
        step(worker, [0], [0], [2])
        with pytest.raises(RuntimeError, match="finite|selected"):
            worker.finalize({"a": [2]})
    assert worker.coordexp_trace_assert_idle()


@pytest.mark.parametrize("field,value", [
    ("n", 2), ("temperature", 0.5), ("top_p", 0.9), ("top_k", 4),
    ("min_tokens", 1), ("frequency_penalty", 0.2), ("seed", None),
    ("allowed_token_ids", [2]), ("logprobs", -1),
    ("prompt_logprobs", 0), ("thinking_token_budget", 0),
])
def test_trace_rejects_unsupported_policy_controls(field, value):
    worker = FakeWorker([descriptor("a")])
    worker.begin()
    setattr(worker.model_runner.requests["internal-0"].sampling_params, field, value)
    with pytest.raises(RuntimeError, match="sampling"):
        step(worker, [0], [0], [2])
    worker.coordexp_trace_abort()


def test_greedy_vllm_normalized_parameters_and_pending_finalization():
    worker = FakeWorker([descriptor("a")])
    params = worker.model_runner.requests["internal-0"].sampling_params
    params.temperature, params.top_k, params.seed = 0.0, 0, None
    worker.begin()
    step(worker, [0], [0], [2])
    assert worker.finalize({"a": [2]})["a"]["token_ids"] == [2]
    worker.begin()
    trace.capture_raw_logits(worker.model, torch.ones(1, 5))
    with pytest.raises(RuntimeError, match="pending"):
        worker.finalize({"a": [2]})
    assert worker.coordexp_trace_assert_idle()


def test_speculative_decode_and_mismatched_final_identity_are_rejected():
    worker = FakeWorker([descriptor("a")])
    worker.model_runner.speculative_config = object()
    with pytest.raises(RuntimeError, match="speculative"):
        worker.begin()
    worker.model_runner.speculative_config = None
    worker.begin()
    step(worker, [0], [0], [2])
    with pytest.raises(RuntimeError, match="identity"):
        worker.coordexp_trace_finalize("snapshot-2", [{"request_id": "a", "token_ids": [2]}], [2])
    assert worker.coordexp_trace_assert_idle()
