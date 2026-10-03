from types import SimpleNamespace

import pytest
import torch

from probes.rule_stability.policy import MedianPolicy, replay_difference
from src.losses.token_scores import aligned_token_logprobs
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.native import NativeBatch
from src.qwen.untied_embeddings import SelectedDeltaOutputHead, SpecialTokenSelection


def model_fixture():
    base = torch.nn.Linear(2, 1003, bias=False, dtype=torch.bfloat16)
    base.requires_grad_(False)
    with torch.no_grad():
        base.weight.fill_(0)
        base.weight[:, 0] = 1
    # Distinct FP32 norms make the lower median's derivative unambiguous.
    delta = torch.nn.Parameter(torch.zeros(1001, 2))
    with torch.no_grad():
        delta[:1000, 0] = torch.arange(1000) / 1000
    selection = SpecialTokenSelection(
        token_strings=[f"coord-{i}" for i in range(1000)] + ["wrapper"],
        token_ids=list(range(1, 1001)) + [1002],
    )
    head = SelectedDeltaOutputHead(base, selection, delta)
    model = SimpleNamespace(
        get_output_embeddings=lambda: head,
        get_input_embeddings=lambda: SimpleNamespace(
            shared_embed_delta=torch.nn.Parameter(torch.zeros_like(delta))
        ),
    )
    return model, head, delta


def test_exact_median_precision_and_cast_and_unmodified_raw_channels():
    model, head, delta = model_fixture()
    policy = MedianPolicy(model, list(range(1, 1001)))
    effective = head.weight[1:1001].float() + delta[:1000]
    norms = torch.linalg.vector_norm(effective.double(), dim=1)
    reference = torch.median(norms) / norms
    assert policy.factors().dtype == torch.float64
    assert torch.equal(policy.factors(), reference)
    assert reference[499] == 1  # torch.median is the lower median for 1,000 rows.
    for dtype in (torch.bfloat16, torch.float32):
        raw = torch.linspace(-2, 2, 1003, dtype=dtype)[None]
        saved = raw.clone()
        actual = policy.transform_logits(raw)
        expected = raw.clone()
        expected[:, 1:1001] = (raw[:, 1:1001].double() * reference).to(dtype)
        assert actual.dtype == dtype
        assert torch.equal(actual, expected)
        assert torch.equal(raw, saved)
        assert torch.equal(actual[:, [0, 1001, 1002]], raw[:, [0, 1001, 1002]])
    bf16 = torch.linspace(-2, 2, 1003, dtype=torch.bfloat16)[None]
    assert torch.equal(policy.transform_replay(bf16), policy.transform_logits(bf16.float()))


def test_replay_factor_derivative_survives_and_detached_factors_fail():
    model, _, delta = model_fixture()
    policy = MedianPolicy(model, list(range(1, 1001)))
    raw = torch.linspace(-2, 2, 1003)[None]
    target = torch.tensor([800])
    connected = aligned_token_logprobs(policy.transform_replay(raw), target).sum()
    gradient = torch.autograd.grad(connected, delta)[0]
    assert gradient[799, 0].abs() > 0.1
    assert gradient[499, 0].abs() > 0.1  # The median numerator also retains its derivative.
    # A transform built for inference retains identical values but loses this gradient.
    detached = policy.generation_transform()(torch.tensor([[0]]), raw)
    assert torch.equal(detached, policy.transform_replay(raw))
    assert not detached.requires_grad
    step = 0.001
    values = []
    with torch.no_grad():
        for change in (step, -2 * step, step):
            delta[799, 0] += change
            values.append(float(aligned_token_logprobs(policy.transform_replay(raw), target).sum()))
    finite_difference = (values[0] - values[1]) / (2 * step)
    assert float(gradient[799, 0]) == pytest.approx(finite_difference, rel=0.003, abs=0.001)


def test_generation_seam_keeps_full_support_raw_policy_and_actual_eos():
    model, _, delta = model_fixture()
    norm = MedianPolicy(model, list(range(1, 1001)))
    raw = torch.zeros(1, 1003)
    raw[0, 1] = 1.5
    raw[0, 999] = 2
    captured = []

    class FakeModel:
        def generate(self, **kwargs):
            captured.append(kwargs)
            # Substitutes only model computation; actual continuation/readback runs.
            processors = kwargs["logits_processor"]
            score = processors(kwargs["input_ids"], raw)
            assert torch.isfinite(score).all()  # Full vocabulary support survives.
            assert score.argmax(-1).item() != raw.argmax(-1).item()
            first = int(score.argmax(-1))
            return SimpleNamespace(
                sequences=torch.cat([kwargs["input_ids"], torch.tensor([[first, 1001]])], 1),
                scores=(score, score), logits=(raw, raw),
            )

    batch = NativeBatch({"input_ids": torch.tensor([[0]])}, ("image-01",))
    transform = norm.generation_transform()
    frozen_factor = transform.factors.clone()
    result = generate_continuations(
        FakeModel(), batch, extensions=[[]], budgets=[3], eos_token_id=1001,
        pad_token_id=1001, policy=NativeGenerationPolicy(
            temperature=1, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False,
        ), seed=92711, trace="raw_and_policy", logits_processor=[transform],
    )[0]
    assert result.request_id == "image-01"
    assert result.token_ids == (1, 1001)
    assert result.stop_reason == "im_end"
    targets = torch.tensor(result.token_ids)
    expected_raw = aligned_token_logprobs(raw.expand(2, -1), targets)
    expected_policy = aligned_token_logprobs(norm.transform_replay(raw).expand(2, -1), targets)
    assert result.raw_logprobs == pytest.approx(expected_raw.tolist())
    assert result.policy_logprobs == pytest.approx(expected_policy.tolist())
    assert result.raw_logprobs != result.policy_logprobs
    config = captured[0]["generation_config"]
    assert (config.temperature, config.top_p, config.top_k, config.repetition_penalty) == (1, 1, 0, 1)
    assert config.suppress_tokens is None and config.min_new_tokens is None
    with torch.no_grad():
        delta[0, 0] += 0.5
    assert torch.equal(transform.factors, frozen_factor)
    assert not torch.equal(norm.generation_transform().factors, frozen_factor)


def test_numeric_gap_report_preserves_behavior_and_reports_per_action():
    raw_behavior = (-2.0, -1.0)
    policy_behavior = (-1.0, -0.5)
    report = replay_difference(
        request_id="image-01", token_ids=(3, 9),
        behavior_raw_logprobs=raw_behavior, behavior_policy_logprobs=policy_behavior,
        replay_raw_logprobs=torch.tensor([-2.125, -1.0], requires_grad=True),
        replay_policy_logprobs=torch.tensor([-0.75, -0.625], requires_grad=True),
    )
    assert report["actions"] == 2 and report["token_ids"] == [3, 9]
    assert report["raw"]["delta_replay_minus_behavior"] == [-0.125, 0]
    assert report["policy"]["delta_replay_minus_behavior"] == [0.25, -0.125]
    assert report["policy"]["max_abs_delta"] == 0.25
    assert report["policy"]["mean_abs_delta"] == 0.1875
    assert raw_behavior == (-2.0, -1.0) and policy_behavior == (-1.0, -0.5)
    with pytest.raises(ValueError, match="align"):
        replay_difference(request_id="bad", token_ids=(3,),
                          behavior_raw_logprobs=raw_behavior, behavior_policy_logprobs=policy_behavior,
                          replay_raw_logprobs=torch.zeros(1), replay_policy_logprobs=torch.zeros(1))


@pytest.mark.parametrize("fault", ["fp32-base", "bf16-delta", "tied", "aliased", "missing-id", "duplicate-id", "bias", "zero-norm"])
def test_policy_fails_closed_on_wrong_effective_weight_schema(fault):
    model, head, delta = model_fixture()
    ids = list(range(1, 1001))
    if fault == "fp32-base":
        head.base.float()
    elif fault == "bf16-delta":
        head.shared_embed_delta = torch.nn.Parameter(delta.bfloat16())
    elif fault == "tied":
        model.get_input_embeddings = lambda: SimpleNamespace(shared_embed_delta=delta)
    elif fault == "aliased":
        model.get_input_embeddings = lambda: SimpleNamespace(
            shared_embed_delta=torch.nn.Parameter(delta.detach())
        )
    elif fault == "missing-id":
        ids[0] = 0
    elif fault == "duplicate-id":
        ids[0] = ids[1]
    elif fault == "bias":
        head.base.bias = torch.nn.Parameter(torch.zeros(1003))
    else:
        with torch.no_grad():
            delta[0, 0] = -1
    with pytest.raises(ValueError):
        MedianPolicy(model, ids).factors()
