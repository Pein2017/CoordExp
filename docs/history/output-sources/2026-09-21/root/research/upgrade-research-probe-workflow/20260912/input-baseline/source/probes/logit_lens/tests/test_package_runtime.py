from types import SimpleNamespace

import torch

from probes.logit_lens import causal


class Qwen3VLTextDecoderLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 0.01


class TextStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3VLTextDecoderLayer() for _ in range(28)])
        self.norm = torch.nn.Identity()

    def forward(self, *, inputs_embeds, **kwargs):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = TextStack()
        self.weight = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(8, 8) / 63)
        self.head = torch.nn.Linear(8, 8, bias=False)

    def get_rope_index(self, ids, grid, video, *, attention_mask):
        return torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1), None

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, position_ids, attention_mask, logits_to_keep, **kwargs):
        hidden = self.weight[input_ids] + self.weight[0] * position_ids[0, :, :, None]
        output = self.model.language_model(inputs_embeds=hidden, position_ids=position_ids,
            attention_mask=attention_mask, cache_position=None, visual_pos_masks=None,
            deepstack_visual_embeds=None)
        return SimpleNamespace(logits=self.head(output.last_hidden_state[:, logits_to_keep]))


def test_native_history_and_shared_captures_reach_actual_causal_consumer():
    model = Model()
    native = {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones(1, 2),
              'image_grid_thw': torch.tensor([[1, 1, 1]]), 'position_ids': 'stale',
              'past_key_values': 'stale'}
    bundle = causal._capture_existing_session(
        name='source', opened=SimpleNamespace(receipt=SimpleNamespace(to_artifact_dict=lambda: {})),
        components=SimpleNamespace(model=model, tokenizer=object()), native_inputs=native,
        prompt_ids=[1, 2], trajectory={'token_ids': [3, 4, 5]},
        sites=[{'position': 1}, {'position': 3}], input_receipt={}, blocks=(24, 27, 28),
    )
    initial = model.weight[torch.tensor([[1, 2, 3, 4, 5]])] + model.weight[0] * torch.arange(5).view(1, 5, 1)
    torch.testing.assert_close(bundle.states[24], initial + .24)
    expected = model.head((initial + .28)[:, [1, 3]])
    torch.testing.assert_close(bundle.baseline_logits, expected)
    assert not torch.allclose(bundle.baseline_logits, model.head((initial + .28)[:, [2, 4]]))
    assert bundle.baseline_checks['native_hooks_off_vs_direct_capture_passed']
    assert not model.model.language_model._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in model.model.language_model.layers)
