from __future__ import annotations


import pytest
import torch
from types import SimpleNamespace


from probes.logit_lens import natural as probe


class _Layer(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 1.0


class _Past:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class _GenerationModel(torch.nn.Module):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: _Past | None = None,
        pixel_values: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        del past_key_values, pixel_values, use_cache
        return input_ids


def test_direction_only_state_preserves_source_radius_and_uses_donor_unit() -> None:
    source = torch.tensor([3.0, 4.0])
    donor = torch.tensor([-12.0, 5.0])
    replacement, receipt = probe.direction_only_state(source, donor)
    assert torch.allclose(replacement.norm(), source.norm(), atol=probe.ATOL, rtol=probe.RTOL)
    assert torch.allclose(
        replacement / replacement.norm(),
        donor / donor.norm(),
        atol=probe.ATOL,
        rtol=probe.RTOL,
    )
    assert receipt["radius_passed"] is True
    assert receipt["direction_passed"] is True


@pytest.mark.parametrize(
    ("source", "donor", "message"),
    [
        (torch.zeros(4), torch.ones(4), "source radius"),
        (torch.ones(4), torch.zeros(4), "donor radius"),
    ],
)
def test_direction_only_state_rejects_zero_radius(
    source: torch.Tensor, donor: torch.Tensor, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        probe.direction_only_state(source, donor)


def test_one_shot_patch_changes_only_first_prefill_call() -> None:
    layer = _Layer()
    prefill = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    cached_step = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
    replacement = torch.full((6,), -3.0)
    patch = probe.one_shot_patch(
        layer,
        position=2,
        replacement=replacement,
        expected_before=prefill[0, 2],
    )
    with patch:
        patched_prefill = layer(prefill)
        untouched_cached_step = layer(cached_step)
    assert torch.equal(patched_prefill[0, 2], replacement)
    assert torch.equal(patched_prefill[0, :2], prefill[0, :2])
    assert torch.equal(patched_prefill[0, 3:], prefill[0, 3:])
    assert torch.equal(untouched_cached_step, cached_step)
    assert patch.receipt()["hook_calls"] == 1


def test_one_shot_patch_rejects_shifted_state() -> None:
    layer = _Layer()
    prefill = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    patch = probe.one_shot_patch(
        layer,
        position=2,
        replacement=torch.zeros(6),
        expected_before=prefill[0, 1],
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        with patch:
            layer(prefill)


def test_native_source_branch_extends_exact_prefix_and_preserves_cached_execution() -> None:
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.visual = torch.nn.Identity()
            self.generation_config = SimpleNamespace(use_cache=True)
            self.seen = None

        def forward(self, input_ids, pixel_values=None, **kwargs):
            if pixel_values is not None:
                self.visual(pixel_values)
            return input_ids

        def generate(self, **kwargs):
            self.seen = kwargs
            ids = kwargs['input_ids']
            self(input_ids=ids, pixel_values=kwargs['pixel_values'], use_cache=True)
            self(input_ids=torch.tensor([[31]]), past_key_values=_Past(ids.shape[1]), use_cache=True)
            return torch.cat((ids, torch.tensor([[31, probe.parent.IM_END]])), dim=1)

    model = Generator()
    tokenizer = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda _: probe.parent.IM_END,
                                decode=lambda ids, **_: str(list(ids)))
    native = {
        'input_ids': torch.tensor([[7, 8]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long),
        'image_grid_thw': torch.tensor([[1, 1, 1]]), 'pixel_values': torch.ones(1, 3),
        'position_ids': torch.tensor([[0, 1]]), 'rope_deltas': torch.tensor([[5]]),
        'past_key_values': 'stale',
    }
    branch, patch = probe._generate_source_branch(
        components=SimpleNamespace(model=model), native_inputs=native, generated_prefix_ids=[19, 23, 29],
        suffix_cap=3, tokenizer=tokenizer, visual_module=model.visual, patch=None,
    )
    assert model.seen['input_ids'].tolist() == [[7, 8, 19, 23, 29]]
    assert model.seen['attention_mask'].tolist() == [[1, 1, 1, 1, 1]]
    assert not any(key in model.seen for key in ('position_ids', 'rope_deltas', 'past_key_values'))
    assert model.seen['output_scores'] is False and model.seen['return_dict_in_generate'] is False
    assert branch['full_generated_token_ids'] == [19, 23, 29, 31, probe.parent.IM_END]
    assert branch['finish_reason'] == 'im_end'
    assert branch['vision_encoder_forward_calls'] == 1
    assert branch['generation_cache']['cached_followup_count'] == 1
    assert patch is None and not model._forward_pre_hooks



def test_generation_trace_requires_full_prefill_then_cached_single_token_steps() -> None:
    model = _GenerationModel()
    with probe.GenerationCallTrace(model) as trace:
        model(input_ids=torch.ones(1, 9), pixel_values=torch.ones(1, 3), use_cache=True)
        model(input_ids=torch.ones(1, 1), past_key_values=_Past(9), use_cache=True)
        model(input_ids=torch.ones(1, 1), past_key_values=_Past(10), use_cache=True)
    receipt = trace.validate(full_input_width=9, generated_count=3)
    assert receipt["prefill_full_width_and_image_passed"] is True
    assert receipt["cached_followups_passed"] is True
    assert receipt["cache_followup_observed"] is True


def test_generation_trace_rejects_non_cached_followup() -> None:
    model = _GenerationModel()
    with probe.GenerationCallTrace(model) as trace:
        model(input_ids=torch.ones(1, 9), pixel_values=torch.ones(1, 3), use_cache=True)
        model(input_ids=torch.ones(1, 1), use_cache=True)
    with pytest.raises(RuntimeError, match="KV-cache"):
        trace.validate(full_input_width=9, generated_count=2)
