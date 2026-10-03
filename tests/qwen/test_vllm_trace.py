"""CPU counterexamples at the installed plain-callable V2 sampler contract."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
import src.qwen.vllm_trace as trace


def descriptor(request_id, budget=4, snapshot="snapshot-1"):
    return dict(request_id=request_id, snapshot_id=snapshot, max_new_tokens=budget)


def sampling_params(metadata):
    return SimpleNamespace(extra_args={"coordexp_paired_trace": metadata.copy()},
        n=1, temperature=1.0, seed=37, top_p=1.0, top_k=-1, min_p=0.0,
        frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0,
        min_tokens=0, max_tokens=metadata["max_new_tokens"], logprobs=0,
        stop=[], ignore_eos=False, bad_words=[], _bad_words_token_ids=[],
        allowed_token_ids=None, logit_bias=None, structured_outputs=None,
        logits_processors=[], thinking_token_budget=None,
        repetition_detection=None, logprob_token_ids=None, prompt_logprobs=None)


class FakeSampler:
    """Intentionally a plain callable, as installed V2 Sampler actually is."""
    logprobs_mode = "raw_logprobs"
    num_speculative_tokens = 1  # V2 names its total decode width this way.

    def __init__(self, req_states):
        self.req_states = req_states
        self.calls = self.add_calls = self.staged_calls = 0
        self.tokens = torch.tensor([2])
        self.num_sampled = torch.tensor([1])

    def add_request(self, req_idx, prompt_len, sampling_params):
        self.add_calls += 1
        if getattr(self, "add_error", None) is not None:
            raise self.add_error
        return (req_idx, prompt_len)

    def apply_staged_writes(self):
        self.staged_calls += 1
        return "staged"

    def __call__(self, logits, input_batch):
        self.calls += 1
        if getattr(self, "error", None) is not None:
            raise self.error
        selected = self.tokens.to(dtype=torch.int32).view(-1, 1)
        if getattr(self, "extra_sample", False):
            selected = selected.repeat(1, 2)
        scores = logits.log_softmax(-1, dtype=torch.float32).gather(1, selected.clamp(0, 4).long())
        self.last_output = SimpleNamespace(sampled_token_ids=selected,
            logprobs_tensors=SimpleNamespace(logprob_token_ids=selected.clone(), logprobs=scores),
            num_sampled=self.num_sampled)
        if getattr(self, "nonfinite", False):
            self.last_output.logprobs_tensors.logprobs[0, 0] = float("nan")
        if getattr(self, "mismatch", False):
            self.last_output.logprobs_tensors.logprob_token_ids[0, 0] += 1
        return self.last_output


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
        req_states = SimpleNamespace(index_to_req_id={}, prompt_len=SimpleNamespace(gpu=torch.full((8,), 3)))
        self.original_sampler = FakeSampler(req_states)
        self.model_runner = SimpleNamespace(sampler=self.original_sampler, req_states=req_states,
            scheduler_config=SimpleNamespace(async_scheduling=asynchronous),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            speculative_config=None, batch_sharder=None)
        self.descriptors = descriptors
        self.params = [sampling_params(metadata) for metadata in descriptors]

    def get_model(self):
        return self.model

    def add_request(self, index, *, slot=None, internal_id=None, params=None):
        slot = index if slot is None else slot
        internal_id = f"internal-{index}" if internal_id is None else internal_id
        self.model_runner.req_states.index_to_req_id[slot] = internal_id
        return self.model_runner.sampler.add_request(slot, 3, params or self.params[index])

    def begin(self, *, bind=True):
        ack = self.coordexp_trace_begin("snapshot-1", self.descriptors)
        if bind:
            for index in range(len(self.descriptors)):
                self.add_request(index)
        return ack

    def finalize(self, ids):
        return self.coordexp_trace_finalize("snapshot-1",
            [dict(request_id=key, token_ids=tokens) for key, tokens in ids.items()], [2])


def inputs(worker, order, positions, tokens, num_sampled=None, *, slots=None):
    slots = order if slots is None else slots
    worker.original_sampler.tokens = torch.tensor(tokens)
    worker.original_sampler.num_sampled = torch.tensor([1] * len(order) if num_sampled is None else num_sampled)
    count = len(order)
    return SimpleNamespace(req_ids=[f"internal-{index}" for index in order], num_reqs=count,
        idx_mapping_np=np.array(slots), idx_mapping=torch.tensor(slots),
        positions=torch.tensor(positions) + 2, logits_indices=torch.arange(count),
        cu_num_logits_np=np.arange(count + 1), num_draft_tokens=0,
        num_computed_tokens_np=np.full(count, 9000))  # Wrong optimistic CPU counters.


def step(worker, order, positions, tokens, num_sampled=None, *, slots=None):
    batch = inputs(worker, order, positions, tokens, num_sampled, slots=slots)
    raw = torch.tensor([[0.5, 1.5, -0.5, 2.5, 0.1]] * len(order))
    policy = worker.model.compute_logits(raw.clone())
    output = worker.model_runner.sampler(policy, batch)
    return raw, policy, output, batch


def test_plain_v2_sampler_begin_receipt_and_scope_restore():
    worker = FakeWorker([descriptor("a")])
    ack = worker.begin()
    assert ack == dict(snapshot_id="snapshot-1", request_ids=["a"],
        runner_class="types.SimpleNamespace", sampler_class=f"{FakeSampler.__module__}.{FakeSampler.__qualname__}")
    assert worker.model_runner.sampler is not worker.original_sampler
    _, _, output, _ = step(worker, [0], [0], [2])
    assert output is worker.original_sampler.last_output
    assert worker.original_sampler.calls == worker.original_sampler.add_calls == 1
    assert worker.model_runner.sampler.apply_staged_writes() == "staged"
    assert worker.original_sampler.staged_calls == 1
    worker.model_runner.sampler.use_flashinfer = False
    assert worker.original_sampler.use_flashinfer is False
    worker.finalize({"a": [2]})
    assert worker.model_runner.sampler is worker.original_sampler
    assert worker.coordexp_trace_assert_idle()


def test_reordering_compaction_mixed_budgets_pad_eos_and_verified_async_suffix():
    worker = FakeWorker([descriptor("a", 3), descriptor("b", 3), descriptor("c", 1)])
    worker.begin()
    raw, policy, _, _ = step(worker, [1, 0, 2], [0, 0, 0], [3, 0, 2])
    step(worker, [0, 1], [1, 1], [2, 4])
    step(worker, [1, 0, 2], [2, 2, 1], [2, 1, 4])
    result = worker.finalize({"c": [2], "a": [0, 2], "b": [3, 4, 2]})
    assert result["a"]["token_ids"] == [0, 2]
    assert result["a"]["excluded_async_suffix"] == 1
    assert result["b"]["excluded_async_suffix"] == 0
    assert result["c"]["dropped_budget_actions"] == 1
    assert result["a"]["raw_logprobs"][0] == pytest.approx(raw.log_softmax(-1)[0, 0].item())
    assert result["a"]["policy_logprobs"][0] == pytest.approx(policy.log_softmax(-1)[0, 0].item())
    assert result["a"]["raw_logprobs"] != result["a"]["policy_logprobs"]
    assert sum(value["captured_rows"] for value in result.values()) == 8
    assert all(value["captured_rows"] == len(value["token_ids"]) + value["excluded_async_suffix"]
        + value["dropped_budget_actions"] + value["discarded_prefill_actions"] for value in result.values())


def test_reused_gpu_input_and_output_buffers_are_snapshotted():
    worker = FakeWorker([descriptor("a", 2)])
    worker.begin()
    _, _, output, batch = step(worker, [0], [0], [3])
    batch.positions.fill_(99)
    worker.model_runner.req_states.prompt_len.gpu.fill_(99)
    output.num_sampled.zero_()
    output.sampled_token_ids.fill_(4)
    output.logprobs_tensors.logprobs.fill_(float("nan"))
    worker.model_runner.req_states.prompt_len.gpu.fill_(3)
    step(worker, [0], [1], [2])
    assert worker.finalize({"a": [3, 2]})["a"]["token_ids"] == [3, 2]


def test_gpu_logit_indices_and_request_prompt_lengths_define_action_positions():
    worker = FakeWorker([descriptor("a", 1), descriptor("b", 1)])
    worker.begin()
    worker.model_runner.req_states.prompt_len.gpu[1] = 5
    batch = inputs(worker, [1, 0], [0, 0], [2, 2])
    # Multiple query tokens per request, reordered slots and distinct prompts.
    batch.positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    batch.logits_indices = torch.tensor([4, 7])
    logits = worker.model.compute_logits(torch.ones(2, 5))
    worker.model_runner.sampler(logits, batch)
    result = worker.finalize({"a": [2], "b": [2]})
    assert result["a"]["token_ids"] == result["b"]["token_ids"] == [2]


def test_prefill_zero_precedes_negative_position_token_and_probability_checks():
    worker = FakeWorker([descriptor("a", 1)])
    worker.begin()
    worker.original_sampler.nonfinite = worker.original_sampler.mismatch = True
    step(worker, [0], [-2], [-19], [0])
    worker.original_sampler.nonfinite = worker.original_sampler.mismatch = False
    step(worker, [0], [0], [2])
    result = worker.finalize({"a": [2]})["a"]
    assert result["discarded_prefill_actions"] == 1 and result["captured_rows"] == 2


def test_budget_overshoot_is_classified_before_irrelevant_ids_and_scores():
    worker = FakeWorker([descriptor("a", 1)])
    worker.begin()
    step(worker, [0], [0], [2])
    worker.original_sampler.nonfinite = worker.original_sampler.mismatch = True
    step(worker, [0], [1], [-19])
    result = worker.finalize({"a": [2]})["a"]
    assert result["dropped_budget_actions"] == 1 and result["captured_rows"] == 2


def test_sampler_step_has_no_synchronization_or_host_tensor_construction(monkeypatch):
    worker = FakeWorker([descriptor("a", 1)])
    worker.begin()
    batch = inputs(worker, [0], [0], [2])
    logits = worker.model.compute_logits(torch.ones(1, 5))
    def forbidden(*args, **kwargs):
        raise AssertionError("per-step synchronization or host tensor construction")
    with monkeypatch.context() as scoped:
        for name in ("cpu", "item", "tolist", "__bool__"):
            scoped.setattr(torch.Tensor, name, forbidden)
        for name in ("tensor", "as_tensor", "from_numpy"):
            scoped.setattr(torch, name, forbidden)
        output = worker.model_runner.sampler(logits, batch)
    assert output is worker.original_sampler.last_output
    worker.finalize({"a": [2]})


@pytest.mark.parametrize("positions,tokens,expected,match", [
    ([-1], [2], [2], "negative"), ([1], [2], [2], "positions"),
    ([0], [3], [2], "token"), ([0], [2], [3, 4], "positions")])
def test_missing_or_mismatched_evidence_restores(positions, tokens, expected, match):
    worker = FakeWorker([descriptor("a")]); worker.begin()
    step(worker, [0], positions, tokens)
    with pytest.raises(RuntimeError, match=match):
        worker.finalize({"a": expected})
    assert worker.model_runner.sampler is worker.original_sampler
    assert worker.coordexp_trace_assert_idle()


def test_duplicate_positions_are_rejected_at_finalization():
    worker = FakeWorker([descriptor("a", 1)]); worker.begin()
    step(worker, [0], [0], [2]); step(worker, [0], [0], [2])
    with pytest.raises(RuntimeError, match="duplicate"):
        worker.finalize({"a": [2]})
    assert worker.model_runner.sampler is worker.original_sampler


@pytest.mark.parametrize("asynchronous,emitted", [(False, [2]), (True, [3])])
def test_only_verified_async_post_eos_suffix_is_excluded(asynchronous, emitted):
    worker = FakeWorker([descriptor("a")], asynchronous=asynchronous); worker.begin()
    step(worker, [0], [0], emitted); step(worker, [0], [1], [4])
    with pytest.raises(RuntimeError, match="async.*EOS"):
        worker.finalize({"a": emitted})
    assert worker.coordexp_trace_assert_idle()


@pytest.mark.parametrize("mutation", ["model", "request", "snapshot", "budget"])
def test_stale_identity_and_changed_admission_metadata(mutation):
    worker = FakeWorker([descriptor("a")]); worker.begin(bind=False)
    if mutation == "model":
        worker.add_request(0); worker.model.coordexp_dora_identity = "snapshot-2"
        operation = lambda: step(worker, [0], [0], [2])
    else:
        worker.params[0].extra_args["coordexp_paired_trace"][
            {"request": "request_id", "snapshot": "snapshot_id", "budget": "max_new_tokens"}[mutation]] = {
            "request": "alien", "snapshot": "snapshot-2", "budget": 9}[mutation]
        operation = lambda: worker.add_request(0)
    with pytest.raises(RuntimeError, match="identity|metadata"):
        operation()
    worker.coordexp_trace_abort(); assert worker.coordexp_trace_assert_idle()


def test_slot_reuse_requires_current_admission():
    worker = FakeWorker([descriptor("a"), descriptor("b")]); worker.begin()
    worker.model_runner.req_states.index_to_req_id[0] = "internal-1"
    with pytest.raises(RuntimeError, match="slot|identity"):
        step(worker, [1], [0], [2], slots=[0])
    worker.coordexp_trace_abort()
    worker.begin(); step(worker, [0], [0], [2])
    worker.add_request(1, slot=0); step(worker, [1], [0], [2], slots=[0])
    result = worker.finalize({"b": [2], "a": [2]})
    assert result["a"]["captured_rows"] == result["b"]["captured_rows"] == 1


def test_trace_off_original_is_untouched_and_abort_clears_metadata():
    worker = FakeWorker([descriptor("a")])
    for _ in range(3):
        step(worker, [0], [0], [4])
    assert worker.model_runner.sampler is worker.original_sampler
    assert not hasattr(worker.model, "_coordexp_paired_trace_state")
    worker.begin(); state = worker.model._coordexp_paired_trace_state
    step(worker, [0], [0], [2]); worker.finalize({"a": [2]})
    assert not state.batches and not state.bindings and not state.requests
    for _ in range(3):
        step(worker, [0], [0], [4])
    assert worker.model_runner.sampler is worker.original_sampler
    assert worker._coordexp_trace_state is None


@pytest.mark.parametrize("failure", ["native", "scoring", "nonfinite", "policy_ids", "admission"])
def test_exceptions_and_invalid_pairs_fail_closed_and_restore(monkeypatch, failure):
    worker = FakeWorker([descriptor("a")]); worker.begin(bind=failure != "admission")
    state = worker.model._coordexp_paired_trace_state
    error = ValueError("native sampler failed")
    if failure == "native":
        worker.original_sampler.error = error
    elif failure == "scoring":
        def broken(*args, **kwargs):
            raise RuntimeError("selected scoring failed")
        monkeypatch.setattr(trace, "chosen_token_logprobs", broken)
    elif failure == "nonfinite":
        worker.original_sampler.nonfinite = True
    elif failure == "policy_ids":
        worker.original_sampler.mismatch = True
    else:
        worker.original_sampler.add_error = error
    if failure in ("native", "scoring", "admission"):
        with pytest.raises((RuntimeError, ValueError), match="failed") as caught:
            worker.add_request(0) if failure == "admission" else step(worker, [0], [0], [2])
        if failure != "scoring":
            assert caught.value is error
        assert state.pending_raw_logits is None
        with pytest.raises(RuntimeError, match="failed"):
            worker.finalize({"a": [2]})
    else:
        step(worker, [0], [0], [2])
        with pytest.raises(RuntimeError, match="finite|selected"):
            worker.finalize({"a": [2]})
    assert worker.model_runner.sampler is worker.original_sampler
    assert not state.batches and not state.bindings


@pytest.mark.parametrize("field,value", [("n", 2), ("temperature", 0.5), ("top_p", 0.9),
    ("top_k", 4), ("min_tokens", 1), ("frequency_penalty", 0.2), ("seed", None),
    ("allowed_token_ids", [2]), ("logprobs", -1), ("prompt_logprobs", 0), ("thinking_token_budget", 0)])
def test_unsupported_policy_is_rejected_before_native_admission(field, value):
    worker = FakeWorker([descriptor("a")]); worker.begin(bind=False)
    setattr(worker.params[0], field, value)
    with pytest.raises(RuntimeError, match="sampling"):
        worker.add_request(0)
    assert worker.original_sampler.add_calls == 0
    worker.coordexp_trace_abort()


@pytest.mark.parametrize("layout", ["speculative", "shard", "tp", "pp", "width"])
def test_unsupported_runner_layout_is_rejected_before_wrapping(layout):
    worker = FakeWorker([descriptor("a")]); runner = worker.model_runner
    if layout == "width":
        worker.original_sampler.num_speculative_tokens = 2
    elif layout in ("tp", "pp"):
        setattr(runner.parallel_config, "tensor_parallel_size" if layout == "tp" else "pipeline_parallel_size", 2)
    else:
        setattr(runner, "speculative_config" if layout == "speculative" else "batch_sharder", object())
    with pytest.raises(RuntimeError, match="speculative|shard|TP|PP"):
        worker.begin()
    assert runner.sampler is worker.original_sampler


@pytest.mark.parametrize("layout", ["draft", "logits", "counts", "samples"])
def test_speculative_batch_or_output_layout_is_rejected(layout):
    worker = FakeWorker([descriptor("a")]); worker.begin()
    batch = inputs(worker, [0], [0], [2])
    if layout == "draft":
        batch.num_draft_tokens = 1
    elif layout == "logits":
        batch.logits_indices = torch.tensor([0, 0]); batch.cu_num_logits_np = np.array([0, 2])
    elif layout == "counts":
        worker.original_sampler.num_sampled.fill_(2)
    else:
        worker.original_sampler.extra_sample = True
    logits = worker.model.compute_logits(torch.ones(1, 5))
    if layout == "counts":
        worker.model_runner.sampler(logits, batch)
        operation = lambda: worker.finalize({"a": [2]})
    else:
        operation = lambda: worker.model_runner.sampler(logits, batch)
    with pytest.raises(RuntimeError, match="speculative|layout|sampled"):
        operation()
    worker.coordexp_trace_abort()
    assert worker.model_runner.sampler is worker.original_sampler


def test_greedy_pending_overwrite_refresh_and_mismatched_final_identity():
    worker = FakeWorker([descriptor("a")]); params = worker.params[0]
    params.temperature, params.top_k, params.seed = 0.0, 0, None
    worker.begin(); step(worker, [0], [0], [2]); worker.finalize({"a": [2]})
    worker.begin(); trace.capture_raw_logits(worker.model, torch.ones(1, 5))
    with pytest.raises(RuntimeError, match="active"):
        trace.assert_trace_idle(worker.model)
    with pytest.raises(RuntimeError, match="pending"):
        worker.finalize({"a": [2]})
    worker.begin(); trace.capture_raw_logits(worker.model, torch.ones(1, 5))
    with pytest.raises(RuntimeError, match="pending"):
        trace.capture_raw_logits(worker.model, torch.ones(1, 5))
    worker.coordexp_trace_abort()
    worker.begin(); step(worker, [0], [0], [2])
    with pytest.raises(RuntimeError, match="identity"):
        worker.coordexp_trace_finalize("snapshot-2", [{"request_id": "a", "token_ids": [2]}], [2])
    assert worker.coordexp_trace_assert_idle()
