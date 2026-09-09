from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from probes.human13 import output_qp
from probes.human13.panel import FrozenPanelRow, OwnerInput
from probes.human13.runtime import prepare_decision_history
from src.qwen.inspection import CaptureInputs


class TinyNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(8, 6) / 47)
        self.head = torch.nn.Linear(6, 8, bias=False)

    def get_rope_index(self, ids, grid, video, *, attention_mask):
        return torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1), None

    def forward(self, input_ids, position_ids, logits_to_keep, **kwargs):
        states = self.weight[input_ids] + self.weight[0] * position_ids[0, :, :, None]
        return self.head(states[:, logits_to_keep])


class TinyProcessor:
    tokenizer = SimpleNamespace(padding_side='right')

    def __call__(self, **kwargs):
        return {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.ones(1, 2, dtype=torch.long),
            'image_grid_thw': torch.tensor([[1, 1, 1]]),
            'pixel_values': torch.ones(1, 4),
            'position_ids': 'stale',
            'past_key_values': 'stale',
        }


def test_actual_native_decision_history_keeps_causal_rows_and_backward(tmp_path):
    path = tmp_path / 'image.png'
    Image.new('RGB', (2, 2)).save(path)
    request = SimpleNamespace(request_id='one', chat_text='prompt', image_path=path,
        expected_executed_prompt_token_ids=(1, 2), expected_image_grid_thw=(1, 1, 1),
        decoded_image_width=2, decoded_image_height=2,
        image_sha256=output_qp.sha256_file(path), logical_transform_id='identity')
    model = TinyNative()
    inputs, prompt, positions = prepare_decision_history(
        SimpleNamespace(processor=TinyProcessor(), model=model), request, (3, 4, 5))
    assert prompt == (1, 2)
    assert inputs['input_ids'].tolist() == [[1, 2, 3, 4, 5]]
    assert positions.tolist() == [1, 2, 3]
    assert 'past_key_values' not in inputs
    with CaptureInputs(model.head) as captured:
        logits = model(**inputs)
    assert captured.args[0].shape == (1, 3, 6)
    expected = model.weight[torch.tensor([[2, 3, 4]])] + model.weight[0] * torch.tensor([1, 2, 3]).view(1, 3, 1)
    torch.testing.assert_close(captured.args[0], expected)
    target = torch.tensor([3, 4, 5])
    loss = torch.nn.functional.cross_entropy(logits[0], target)
    loss.backward()
    assert model.weight.grad is not None and model.weight.grad.abs().sum() > 0
    wrong = model(**{**inputs, 'logits_to_keep': positions + 1})
    assert not torch.allclose(logits, wrong)
    assert not model.head._forward_pre_hooks


def _decoded_rows():
    pieces = []
    for category in ('person', 'car'):
        pieces += ['<|object_ref_start|>', category, '<|object_ref_end|>', '<|box_start|>',
                   '<|coord_0|>', '<|coord_0|>', '<|coord_100|>', '<|coord_100|>', '<|box_end|>']
    ids = tuple(range(1, len(pieces) + 1)) + (151645,)
    trace = [dict(step_index=i, token_id=ids[i], token_text=piece, is_stop=False, is_pad=False)
             for i, piece in enumerate(pieces)]
    trace.append(dict(step_index=len(pieces), token_id=151645, token_text='<|im_end|>', is_stop=True, is_pad=False))
    return SimpleNamespace(generated_token_ids=ids, parser_text=''.join(pieces), token_trace=trace, stop_reason='im_end')


def test_public_evidence_reader_keeps_category_independent_dedup_and_token_spans(monkeypatch):
    import probes.human13.panel as panel
    result = _decoded_rows()
    frozen = FrozenPanelRow(1, 'panel', 'image', (OwnerInput('gt:1:0', 'person', (0., 0., 100., 100.), 0),))
    monkeypatch.setattr(panel, 'load_frozen_panel', lambda: (frozen,))
    monkeypatch.setattr(output_qp, 'load_panel', lambda: ({'image_id': 1, 'width': 1000, 'height': 1000},))
    def evaluate():
        return output_qp._evaluate_result(result=result, image_id=1,
            canonical_ids=result.generated_token_ids, backend_version='fixture', repetition_penalty=1.)
    report = evaluate()
    assert report['matched_owner_count'] == {'50': 1, '60': 1, '80': 1}
    assert report['duplicate_count'] == 1  # different category still removed by geometry
    assert report['unmatched_prediction_count'] == 0
    assert report['malformed_count'] == 0 and report['exact_route']
    result.token_trace[1]['token_text'] = 'wrong'
    with pytest.raises(ValueError, match='reconstruct parser text'):
        evaluate()
