from types import SimpleNamespace

import pytest
import torch

from src.qwen.generation import (
    NativeGenerationPolicy,
    generate_continuations,
    trim_suffix,
)
from src.qwen.native import NativeBatch, padded_histories


def batch(rows):
    ids, mask = padded_histories(rows, pad_token_id=23)
    return NativeBatch(
        dict(input_ids=ids, attention_mask=mask),
        tuple(f"r{i}" for i in range(len(rows))),
    )


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(13)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=24,
            n_positions=32,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=21,
            eos_token_id=22,
            pad_token_id=23,
        )
    )
    # Predict token 0 deterministically, avoiding accidental EOS in this budget
    # oracle; actual GenerationMixin executes its normal masking/stopping loop.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    return model


def test_real_generation_mixin_mixed_budgets_exact_histories_and_disabled_traces():
    model = tiny_model()
    calls = []
    original = model.generate

    def observed(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    model.generate = observed
    prepared = batch([[1, 2], [3, 4, 5], [6], [7]])
    results = generate_continuations(
        model,
        prepared,
        extensions=[[8], [9, 10], [22], []],
        budgets=[2, 4, 0, 0],
        eos_token_id=22,
        pad_token_id=23,
    )
    assert [r.request_id for r in results] == ["r0", "r1", "r2", "r3"]
    assert [r.token_ids for r in results] == [(0, 0), (0, 0, 0, 0), (), ()]
    assert [r.stop_reason for r in results] == [
        "length",
        "length",
        "forced_eos",
        "length",
    ]
    assert len(calls) == 1
    assert calls[0]["input_ids"].tolist() == [[23, 23, 1, 2, 8], [3, 4, 5, 9, 10]]
    for name in (
        "output_scores",
        "output_logits",
        "output_hidden_states",
        "output_attentions",
        "return_dict_in_generate",
    ):
        assert calls[0][name] is False
    assert all(r.raw_logprobs is None and r.policy_logprobs is None for r in results)
    # Separate real calls yield identical token suffixes for this fixed tiny model.
    for i in (0, 1):
        one = generate_continuations(
            model,
            batch([prepared.prompt_token_ids[i]]),
            extensions=[[8] if i == 0 else [9, 10]],
            budgets=[2 if i == 0 else 4],
            eos_token_id=22,
            pad_token_id=23,
        )
        assert one[0].token_ids == results[i].token_ids


def test_all_terminal_requests_never_touch_model():
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f"terminal request touched model.{name}")

    result = generate_continuations(
        Forbidden(),
        batch([[1], [2]]),
        extensions=[[22], []],
        budgets=[3, 0],
        eos_token_id=22,
        pad_token_id=23,
    )
    assert [r.stop_reason for r in result] == ["forced_eos", "length"]


def test_seeded_policy_uses_fresh_config_and_exact_fixed_batch_replay():
    model = tiny_model()
    calls = []
    original = model.generate

    def observed(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    model.generate = observed
    policy = NativeGenerationPolicy(
        temperature=1, top_p=1, repetition_penalty=1, top_k=0, use_model_defaults=False
    )
    args = dict(
        extensions=[[]],
        budgets=[5],
        eos_token_id=22,
        pad_token_id=23,
        policy=policy,
        seed=42,
        trace="raw_and_policy",
    )
    a = generate_continuations(model, batch([[1, 2]]), **args)
    b = generate_continuations(model, batch([[1, 2]]), **args)
    assert a == b
    assert calls[0]["generation_config"].top_k == 0
    assert calls[0]["use_model_defaults"] is False
    assert a[0].policy_logprobs == pytest.approx(a[0].raw_logprobs)
    with pytest.raises(ValueError, match="seed"):
        generate_continuations(model, batch([[1]]), **(args | {"seed": None}))


def test_raw_and_processed_channels_differ_and_have_selected_token_length():
    class Model:
        def generate(self, **kwargs):
            ids = kwargs["input_ids"]
            raw = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            policy = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
            return SimpleNamespace(
                sequences=torch.cat([ids, torch.tensor([[2, 3]])], 1),
                logits=(raw, raw),
                scores=(policy, policy),
            )

    result = generate_continuations(
        Model(),
        batch([[1]]),
        extensions=[[]],
        budgets=[3],
        eos_token_id=3,
        pad_token_id=0,
        trace="raw_and_policy",
    )[0]
    assert result.token_ids == (2, 3)
    assert result.policy_logprobs == pytest.approx(
        torch.log_softmax(torch.tensor([4.0, 3.0, 2.0, 1.0]), 0)[[2, 3]].tolist()
    )
    assert result.raw_logprobs == pytest.approx(
        torch.log_softmax(torch.tensor([1.0, 2.0, 3.0, 4.0]), 0)[[2, 3]].tolist()
    )


def test_repetition_penalty_groups_histories_and_keeps_image_patch_alignment():
    seen = []

    class Model:
        def generate(self, **kwargs):
            seen.append(kwargs)
            ids = kwargs["input_ids"]
            return torch.cat([ids, torch.full((ids.shape[0], 1), 5)], 1)

    prepared = batch([[1], [2, 3], [4]])
    prepared = NativeBatch(
        dict(
            prepared.inputs,
            image_grid_thw=torch.tensor([[1, 1, 1], [1, 1, 2], [1, 1, 1]]),
            pixel_values=torch.tensor([[10.0], [20.0], [21.0], [30.0]]),
        ),
        prepared.request_ids,
    )
    results = generate_continuations(
        Model(),
        prepared,
        extensions=[[], [], [22]],
        budgets=[1, 1, 0],
        eos_token_id=22,
        pad_token_id=23,
        policy=NativeGenerationPolicy(repetition_penalty=1.2),
    )
    assert [x["pixel_values"].tolist() for x in seen] == [[[10.0]], [[20.0], [21.0]]]
    assert [r.token_ids for r in results] == [(5,), (5,), ()]


@pytest.mark.parametrize(
    "ids,budget", [([7], 4), ([7, 22, 8], 3), ([23, 22], 2), ([7, 8, 9], 2)]
)
def test_suffix_rejects_truncation_post_stop_tokens_and_padding(ids, budget):
    with pytest.raises(ValueError):
        trim_suffix(ids, budget=budget, eos_token_id=22, pad_token_id=23)


def test_literal_pad_policy_and_pad_equal_eos_budget_boundary():
    assert trim_suffix(
        [23, 22], budget=3, eos_token_id=22, pad_token_id=23, allow_pad_tokens=True
    ) == ((23, 22), "im_end")
    assert trim_suffix([1, 2, 22, 22], budget=2, eos_token_id=22, pad_token_id=22) == (
        (1, 2),
        "length",
    )


def test_traced_generation_rejects_nonfinite_selected_likelihoods():
    from src.common.errors import RuntimeContractError

    class Model:
        def generate(self, **kwargs):
            ids = kwargs["input_ids"]
            return SimpleNamespace(
                sequences=torch.cat([ids, torch.tensor([[2]])], 1),
                scores=(torch.full((1, 4), float("nan")),),
            )

    with pytest.raises(RuntimeContractError, match="nonfinite"):
        generate_continuations(
            Model(),
            batch([[1]]),
            extensions=[[]],
            budgets=[1],
            eos_token_id=3,
            pad_token_id=0,
            trace="policy",
        )
