from contextlib import nullcontext
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


def test_native_caller_preserves_sampled_pad_and_eos_actions(monkeypatch):
    from probes.rule_stability.runner import NativeEngine
    from src.qwen.generation import trim_suffix
    from src.qwen.native import exact_history_inputs

    pad, eos = 151643, 151645
    raw = torch.zeros(1, 152670)
    raw[0, pad], raw[0, eos] = .5, 1.

    def transform(_ids, scores):
        result = scores.clone()
        result[..., 1] += 3.
        return result

    captured = []

    class FakeComputation:
        def eval(self):
            return self

        def generate(self, **kwargs):
            captured.append(kwargs)
            scores = kwargs['logits_processor'](kwargs['input_ids'], raw)
            # Only the first two are actions; the tail is post-EOS batch padding.
            suffix = torch.tensor([[pad, eos, pad, pad]])
            return SimpleNamespace(sequences=torch.cat([kwargs['input_ids'], suffix], 1),
                logits=(raw,) * 4, scores=(scores,) * 4)

        def get_rope_index(self, input_ids, mm_token_type_ids, **kwargs):
            assert torch.equal(kwargs['attention_mask'], torch.ones_like(input_ids))
            return torch.arange(input_ids.shape[1]).reshape(1, 1, -1).expand(3, 1, -1), None

    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    computation = FakeComputation()
    batch = NativeBatch({'input_ids': torch.tensor([[10]]),
        'image_grid_thw': torch.tensor([[1, 1, 1]])}, ('pad-eos',))
    engine = object.__new__(NativeEngine)
    engine.q = SimpleNamespace(model=computation, tokenizer=SimpleNamespace(pad_token_id=pad,
        convert_tokens_to_ids=lambda token: eos,
        decode=lambda ids, **kwargs: '|'.join(map(str, ids))))
    engine.batches = {11: batch}
    engine.norm = SimpleNamespace(generation_transform=lambda: transform)
    engine.calls = {'sample': 0}
    result = engine.generate({'image_id': 11}, 0, 'sample')
    assert result['token_ids'] == [pad, eos]
    assert result['stop_reason'] == 'im_end' and engine.calls['sample'] == 1
    targets = torch.tensor(result['token_ids'])
    replay_raw = raw.expand(2, -1).clone().requires_grad_()
    replay_policy = transform(None, replay_raw)
    assert result['raw_logprobs'] == pytest.approx(aligned_token_logprobs(replay_raw, targets).tolist())
    assert result['policy_logprobs'] == pytest.approx(aligned_token_logprobs(replay_policy, targets).tolist())
    assert result['raw_logprobs'] != result['policy_logprobs']
    assert len(result['raw_logprobs']) == len(result['policy_logprobs']) == 2
    assert captured[0]['generation_config'].eos_token_id == eos
    history = list(batch.prompt_token_ids[0]) + result['token_ids']
    replay = exact_history_inputs(computation, batch.inputs, [history], pad_token_id=pad)
    assert replay['input_ids'].tolist() == [[10, pad, eos]]
    assert replay['attention_mask'].tolist() == [[1, 1, 1]]
    assert replay['position_ids'][0, 0].tolist() == [0, 1, 2]
    likelihood = aligned_token_logprobs(replay_policy, targets)
    likelihood.sum().backward()
    assert replay_raw.grad[0, pad] != 0 and replay_raw.grad[1, eos] != 0
    gap = replay_difference(request_id='pad-eos', token_ids=targets.tolist(),
        behavior_raw_logprobs=result['raw_logprobs'], behavior_policy_logprobs=result['policy_logprobs'],
        replay_raw_logprobs=aligned_token_logprobs(replay_raw, targets), replay_policy_logprobs=likelihood)
    assert gap['actions'] == 2 and gap['token_ids'] == [pad, eos]
    # Budget padding is likewise removed without treating an in-budget PAD as EOS.
    assert trim_suffix([pad, 42, pad], budget=2, eos_token_id=eos,
                       pad_token_id=pad, allow_pad_tokens=True) == ((pad, 42), 'length')
    with pytest.raises(ValueError):
        trim_suffix([pad, eos, 42], budget=3, eos_token_id=eos,
                    pad_token_id=pad, allow_pad_tokens=True)


def test_qualification_six_action_diagnostic_uses_unforced_scores_without_training(monkeypatch):
    from probes.rule_stability.policy import TechnicalSuffixSelection
    from probes.rule_stability.runner import NativeEngine

    pad, image, video, start, end, eos = actions = (151643, 151655, 151656, 151652, 151653, 151645)
    prompt = (start, image, end)
    raw = torch.linspace(-1., 1., 152670)[None]

    def median_substitute(_ids, scores):
        result = scores.clone()
        result[..., 1] += 3.
        return result

    class FakeComputation:
        config = SimpleNamespace(image_token_id=image, video_token_id=video,
            vision_start_token_id=start, vision_end_token_id=end,
            vision_config=SimpleNamespace(spatial_merge_size=2))

        def __init__(self):
            self.generations = self.replays = 0

        def eval(self):
            return self

        def generate(self, **kwargs):
            self.generations += 1
            processors = kwargs['logits_processor']
            assert processors[0] is median_substitute
            assert isinstance(processors[1], TechnicalSuffixSelection)
            history, forced = kwargs['input_ids'], []
            for expected in actions:
                score = processors(history, raw)
                chosen = score.argmax(-1).reshape(1, 1)
                assert int(chosen[0, 0]) == expected
                history = torch.cat((history, chosen), dim=1)
                forced.append(score)
            return SimpleNamespace(sequences=history, scores=tuple(forced), logits=(raw,) * 6)

        def get_rope_index(self, input_ids, mm_token_type_ids, **kwargs):
            assert mm_token_type_ids.tolist() == [[0, 1, 0] + [0] * 6]
            assert kwargs['image_grid_thw'].tolist() == [[1, 2, 2]]
            return torch.arange(input_ids.shape[1]).reshape(1, 1, -1).expand(3, 1, -1), None

        def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
            assert input_ids.tolist() == [list(prompt)]
            image_mask = (input_ids == image).unsqueeze(-1)
            assert int(image_mask.sum()) == image_features.shape[0] == 1
            return image_mask, (input_ids == video).unsqueeze(-1)

        def __call__(self, **kwargs):
            # Fixed-score substitution; the actual replay/mask seams still run.
            self.replays += 1
            assert kwargs['input_ids'].tolist() == [[*prompt, *actions]]
            assert kwargs['attention_mask'].tolist() == [[1] * 9]
            embeds = torch.ones(1, 9, 2)
            mask, _ = self.get_placeholder_mask(kwargs['input_ids'], embeds, torch.ones(1, 2))
            assert mask[0, :, 0].tolist() == [False, True, False] + [False] * 6
            embeds = embeds.masked_scatter(mask, torch.ones(1, 2))
            assert embeds.shape == (1, 9, 2)
            assert kwargs['logits_to_keep'].tolist() == list(range(2, 8))
            return SimpleNamespace(logits=raw[:, None, :].expand(1, 6, -1).clone())

    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    tensor = torch.tensor
    def cpu_tensor(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(*args, **kwargs)
    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    computation = FakeComputation()
    names = ('<|endoftext|>', '<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>', '<|im_end|>')
    tokenizer = SimpleNamespace(pad_token_id=pad, convert_tokens_to_ids=dict(zip(names, actions)).__getitem__)
    engine = object.__new__(NativeEngine)
    engine.q = SimpleNamespace(model=computation, tokenizer=tokenizer)
    engine.batches = {1584: NativeBatch(dict(input_ids=torch.tensor([prompt]),
        attention_mask=torch.ones(1, 3, dtype=torch.long), image_grid_thw=torch.tensor([[1, 2, 2]]),
        pixel_values=torch.ones(4, 3)), ('technical-1584',))}
    engine.norm = SimpleNamespace(generation_transform=lambda: median_substitute,
                                 transform_replay=lambda scores: median_substitute(None, scores))
    engine.calls = dict(greedy=0, sample=0, positive_replay=0, geometry_replay=0, duplicate_replay=0)
    result = engine.technical_suffix_diagnostic()
    assert computation.generations == computation.replays == 1
    assert 'get_placeholder_mask' not in computation.__dict__
    assert result['token_ids'] == list(actions) and result['actions'] == 6
    assert result['forced_selection_logprobs'] == [0.] * 6
    assert result['comparison']['policy']['behavior_logprobs'] != [0.] * 6
    assert result['comparison']['raw']['max_abs_delta'] == 0
    assert result['comparison']['policy']['max_abs_delta'] == 0
    assert result['training_contribution'] is result['scientific_metric'] is False
    assert result['backward_calls'] == 0 and set(engine.calls.values()) == {0}
    assert result['replay_media']['mm_token_type_ids'] == [[0, 1, 0] + [0] * 6]
    assert result['replay_media']['calls'][0]['prompt_image_positions'] == 1
    assert result['replay_media']['calls'][0]['suffix_image_true'] == 0


def test_technical_processor_order_and_causal_prefix_have_teeth():
    from probes.rule_stability.policy import TechnicalSuffixSelection
    raw = torch.zeros(1, 8)
    normalized = raw.clone()
    normalized[0, 7] = 2.
    history = torch.tensor([[7]])
    before = TechnicalSuffixSelection(range(6), 1)
    after = TechnicalSuffixSelection(range(6), 1)
    before(history, raw)
    forced = after(history, normalized)
    assert before.unforced_policy_logprobs != after.unforced_policy_logprobs
    assert after.unforced_policy_logprobs[0] == pytest.approx(float(torch.log_softmax(normalized, -1)[0, 0]))
    assert float(torch.log_softmax(forced, -1)[0, 0]) == 0.
    with pytest.raises(ValueError, match='cached action/score alignment'):
        after(torch.tensor([[7, 5]]), normalized)


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
