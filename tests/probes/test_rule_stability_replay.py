"""CPU media-boundary counterexamples; no model constructors or forwards."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.losses.token_scores import aligned_token_logprobs
from src.qwen.native import derive_position_ids, exact_history_inputs


class FakeQwenOwner:
    # Exercise installed boundary methods on a plain owner, never a model instance.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    get_rope_index = Qwen3VLModel.get_rope_index
    get_placeholder_mask = Qwen3VLModel.get_placeholder_mask

    def __init__(self):
        self.config = SimpleNamespace(image_token_id=7, video_token_id=8,
                                      vision_config=SimpleNamespace(spatial_merge_size=2))

    def get_vision_position_ids(self, current, grid, temporal_merge, spatial_merge, *, device):
        count = int(grid.prod()) // spatial_merge**2
        return torch.arange(count, device=device).expand(3, -1) + current


def fixture(suffix=(7, 2), supplied_types=None):
    owner = FakeQwenOwner()
    model = SimpleNamespace(model=SimpleNamespace(model=owner))
    prompt = [3, 5, 7, 6, 4]
    native = dict(input_ids=torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                  image_grid_thw=torch.tensor([[1, 2, 2]]), pixel_values=torch.ones(4, 3))
    if supplied_types is not None:
        native['mm_token_type_ids'] = torch.tensor([supplied_types])
    return owner, model, prompt, native, prompt + list(suffix)


def test_legacy_generated_image_demands_extra_grid_and_placeholder():
    owner, model, prompt, native, history = fixture()
    with pytest.raises(StopIteration) as rope_error:
        exact_history_inputs(model, native, [history], pad_token_id=0)
    embeds = torch.ones(1, len(history), 3)
    with pytest.raises(ValueError, match='Image features and image tokens do not match') as mask_error:
        owner.get_placeholder_mask(torch.tensor([history]), embeds, image_features=torch.ones(1, 3))
    print('BASELINE second-grid:', type(rope_error.value).__name__,
          '; placeholder:', str(mask_error.value))


@pytest.mark.parametrize('suffix', [(7, 2), (8, 5, 6, 7, 2)])
def test_prompt_types_and_masks_keep_generated_media_and_eos_literal(suffix):
    from probes.rule_stability.replay import prompt_only_placeholder_masks
    owner, model, prompt, native, history = fixture(suffix)
    replay = exact_history_inputs(model, native, [history], pad_token_id=0, prompt_only_media=True)
    assert replay['input_ids'].tolist() == [history]
    assert replay['mm_token_type_ids'].tolist() == [[0, 0, 1, 0, 0] + [0] * len(suffix)]
    assert replay['image_grid_thw'] is native['image_grid_thw']
    assert replay['pixel_values'] is native['pixel_values']
    assert replay['use_cache'] is False
    assert 'get_placeholder_mask' not in owner.__dict__
    embeds = torch.ones(1, len(history), 3)
    image_features = torch.ones(1, 3)
    with prompt_only_placeholder_masks(model, prompt, record_masks=True) as receipt:
        image_mask, video_mask = owner.get_placeholder_mask(replay['input_ids'], embeds,
                                                            image_features=image_features)
    assert 'get_placeholder_mask' not in owner.__dict__
    assert image_mask.shape == video_mask.shape == (1, len(history), 1)
    assert image_mask[0, :, 0].tolist() == [False, False, True, False, False] + [False] * len(suffix)
    assert not video_mask.any()
    entry = receipt['calls'][0]
    assert receipt['prompt_length'] == len(prompt)
    assert entry['history_width'] == len(history)
    assert entry['original_image_mask_shape'] == [1, len(prompt), 1]
    assert entry['padded_image_mask_shape'] == [1, len(history), 1]
    assert entry['prompt_image_positions'] == 1 and entry['prompt_video_positions'] == 0
    assert entry['suffix_image_true'] == entry['suffix_video_true'] == 0


def test_explicit_prompt_types_are_reused_and_suffix_is_zero():
    owner, model, prompt, native, history = fixture(supplied_types=[0, 0, 1, 0, 0])
    class SuppliedTypeConfig:
        vision_config = SimpleNamespace(spatial_merge_size=2)

        @property
        def image_token_id(self):
            raise AssertionError('supplied modality types must not be recomputed')
    owner.config = SuppliedTypeConfig()
    native['mm_token_type_ids'] = native['mm_token_type_ids'].to(torch.int32)
    supplied = native['mm_token_type_ids'].clone()
    replay = exact_history_inputs(model, native, [history], pad_token_id=0, prompt_only_media=True)
    assert torch.equal(replay['mm_token_type_ids'][:, :len(prompt)], supplied)
    assert torch.equal(native['mm_token_type_ids'], supplied)
    assert replay['mm_token_type_ids'].dtype == supplied.dtype == torch.int32
    # The explicit derive seam must avoid reclassifying the emitted image action.
    positions = derive_position_ids(model=model, input_ids=replay['input_ids'],
        attention_mask=replay['attention_mask'], image_grid_thw=replay['image_grid_thw'],
        video_grid_thw=None, mm_token_type_ids=replay['mm_token_type_ids'])
    assert torch.equal(replay['position_ids'], positions)
    assert positions.shape == (3, 1, len(history))
    assert positions[0, 0].tolist() == list(range(len(history)))


def test_absent_video_mask_receipt_and_feature_arguments_are_preserved():
    from probes.rule_stability.replay import prompt_only_placeholder_masks
    owner, model, prompt, native, history = fixture()
    original = owner.get_placeholder_mask
    features, video_features = torch.ones(1, 3), torch.empty(0, 3)
    calls = []
    def observed(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        calls.append((image_features, video_features))
        image_mask, _ = original(input_ids, inputs_embeds, image_features, video_features)
        return image_mask, None
    owner.get_placeholder_mask = MethodType(observed, owner)
    saved = owner.get_placeholder_mask
    with prompt_only_placeholder_masks(model, prompt, record_masks=True) as receipt:
        image_mask, video_mask = owner.get_placeholder_mask(torch.tensor([history]),
            torch.ones(1, len(history), 3), image_features=features, video_features=video_features)
    assert calls[0][0] is features and calls[0][1] is video_features
    assert owner.get_placeholder_mask is saved and video_mask is None
    assert image_mask.shape == (1, len(history), 1)
    assert receipt['calls'][0]['original_video_mask_shape'] is None
    assert receipt['calls'][0]['padded_video_mask_shape'] is None
    assert receipt['calls'][0]['prompt_video_positions'] == receipt['calls'][0]['suffix_video_true'] == 0


def test_left_padded_original_prompt_unpads_types_with_the_same_attention_mask():
    owner, model, prompt, native, history = fixture(supplied_types=[0, 0, 1, 0, 0])
    native['input_ids'] = torch.tensor([[0, 0] + prompt])
    native['attention_mask'] = torch.tensor([[0, 0] + [1] * len(prompt)])
    native['mm_token_type_ids'] = torch.tensor([[0, 0, 0, 0, 1, 0, 0]])
    replay = exact_history_inputs(model, native, [history], pad_token_id=0, prompt_only_media=True)
    assert replay['input_ids'].tolist() == [history]
    assert replay['mm_token_type_ids'].tolist() == [[0, 0, 1, 0, 0, 0, 0]]


@pytest.mark.parametrize('fault', ['prefix', 'short-history', 'batch', 'grid', 'video', 'types-shape', 'types-value'])
def test_opt_in_fails_closed_before_media_replay(fault):
    owner, model, prompt, native, history = fixture()
    if fault == 'prefix':
        history[0] = 9
    elif fault == 'short-history':
        history = history[:2]
    elif fault == 'batch':
        native['input_ids'] = native['input_ids'].expand(2, -1)
        native['attention_mask'] = native['attention_mask'].expand(2, -1)
    elif fault == 'grid':
        native['image_grid_thw'] = torch.tensor([[1, 2, 2], [1, 2, 2]])
    elif fault == 'video':
        native['video_grid_thw'] = torch.tensor([[1, 2, 2]])
    elif fault == 'types-shape':
        native['mm_token_type_ids'] = torch.tensor([[0]])
    else:
        native['mm_token_type_ids'] = torch.tensor([[0, 0, 3, 0, 0]])
    with pytest.raises((ValueError, RuntimeContractError)):
        exact_history_inputs(model, native, [history], pad_token_id=0, prompt_only_media=True)


@pytest.mark.parametrize('instance_attribute', [False, True])
@pytest.mark.parametrize('failure', ['none', 'body', 'features', 'prefix'])
def test_scoped_original_arguments_and_restoration(instance_attribute, failure):
    from probes.rule_stability.replay import prompt_only_placeholder_masks
    owner, model, prompt, native, history = fixture()
    original = owner.get_placeholder_mask
    calls = []
    if instance_attribute:
        def observed(self, input_ids, inputs_embeds, image_features=None, video_features=None):
            calls.append((input_ids, inputs_embeds, image_features, video_features))
            return original(input_ids, inputs_embeds, image_features, video_features)
        owner.get_placeholder_mask = MethodType(observed, owner)
    saved = owner.__dict__.get('get_placeholder_mask')
    embeds = torch.ones(1, len(history), 3, requires_grad=True)
    features = torch.ones(2 if failure == 'features' else 1, 3, requires_grad=True)
    ids = torch.tensor([history])
    if failure == 'prefix':
        ids[0, 0] = 9

    def exercise():
        with prompt_only_placeholder_masks(model, prompt) as receipt:
            owner.get_placeholder_mask(ids, embeds, image_features=features)
            assert 'prompt_image_positions' not in receipt['calls'][0]
            if failure == 'body':
                raise RuntimeError('body failed')
    if failure == 'none':
        exercise()
    else:
        with pytest.raises((ValueError, RuntimeError)):
            exercise()
    assert ('get_placeholder_mask' in owner.__dict__) is instance_attribute
    if instance_attribute:
        assert owner.__dict__['get_placeholder_mask'] is saved
        if failure != 'prefix':
            seen_ids, seen_embeds, seen_features, seen_video = calls[0]
            assert seen_ids.tolist() == [prompt] and seen_embeds.shape == (1, len(prompt), 3)
            assert seen_embeds.requires_grad and seen_embeds.untyped_storage().data_ptr() == embeds.untyped_storage().data_ptr()
            assert seen_features is features and seen_video is None
    else:
        assert owner.get_placeholder_mask.__func__ is original.__func__


def test_scatter_deepstack_positions_and_causal_action_gradients_are_preserved():
    from probes.rule_stability.replay import prompt_only_placeholder_masks
    owner, model, prompt, native, history = fixture((8, 5, 6, 7, 2))
    replay = exact_history_inputs(model, native, [history], pad_token_id=0, prompt_only_media=True)
    width, hidden = len(history), 3
    embeds = torch.nn.Parameter(torch.arange(width * hidden, dtype=torch.float32).reshape(1, width, hidden) / 100)
    features = torch.nn.Parameter(torch.tensor([[.1, .2, .3]]))
    deepstack = torch.nn.Parameter(torch.tensor([[.01, .02, .03]]))
    with prompt_only_placeholder_masks(model, prompt):
        image_mask, _ = owner.get_placeholder_mask(replay['input_ids'], embeds, image_features=features)
        scattered = embeds.masked_scatter(image_mask, features)
        visual_pos_masks = image_mask.squeeze(-1)
        assert visual_pos_masks.shape == (1, width)
        assert visual_pos_masks.nonzero().tolist() == [[0, 2]]
        assert int(visual_pos_masks.sum()) == deepstack.shape[0] == 1
        fused = scattered.clone()
        fused[visual_pos_masks] = fused[visual_pos_masks] + deepstack
    # Synthetic tensor math stands in for decoder computation; no model forward.
    state = fused.cumsum(1)
    head = torch.arange(hidden * 10, dtype=torch.float32).reshape(hidden, 10) / 30
    logits = state @ head
    suffix = history[len(prompt):]
    causal = tuple(range(len(prompt) - 1, len(history) - 1))
    assert causal[-1] == len(history) - 2 and suffix[-1] == 2
    scores = aligned_token_logprobs(logits[0, list(causal)], torch.tensor(suffix))
    scores.sum().backward()
    assert features.grad.abs().sum() > 0 and deepstack.grad.abs().sum() > 0
    assert embeds.grad[0, 2].abs().sum() == 0  # Original image placeholder was replaced.
    assert embeds.grad[0, 0].abs().sum() > 0 and embeds.grad[0, -2].abs().sum() > 0
    assert torch.equal(replay['input_ids'], torch.tensor([history]))
