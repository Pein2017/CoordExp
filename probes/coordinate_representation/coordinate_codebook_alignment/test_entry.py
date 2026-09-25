"""Real CPU request-boundary regression for this frozen pilot admission."""
import json
from pathlib import Path

import pytest

from src.config.loader import load_train_config
from src.inference.bound_requests import build_bound_native_requests
from src.qwen import load_qwen_components
from probes.coordinate_representation.coordinate_codebook_alignment.parity import _infer_config, _read_rows, _plan_case, _target_ids
from probes.coordinate_representation.coordinate_codebook_alignment.evaluation import _normalize_case, _config, _target_ids as evaluation_targets


def test_real_prepared_requests_and_targets_reject_corrupt_prompt_binding():
    config = load_train_config('configs/research/coordinate_codebook_alignment/qualification-single.yaml').config
    qwen = load_qwen_components(config, load_model=False)
    dataset = Path(config.data.train.path)
    infer = _infer_config(config, dataset)
    for row in _read_rows(dataset):
        case = _plan_case(qwen, row, infer, dataset)
        build_bound_native_requests(qwen, infer.model_dump(mode='json'), [case])
        assert _target_ids(qwen, {**case['input_record'], '_parity_line_number': 1}, infer, dataset)
    case['image_plan']['backend_prompt_token_count'] += 1
    with pytest.raises(ValueError, match='prompt width'):
        build_bound_native_requests(qwen, infer.model_dump(mode='json'), [case])
    root = dataset.parent.parent
    admission = json.loads((root / 'selection-v4/admission.json').read_text())
    dataset = root / 'selection-v4/fit.coord.jsonl'
    config = _config(admission, dataset)
    for row in map(json.loads, dataset.read_text().splitlines()):
        case = _normalize_case(qwen, row, config, dataset)
        build_bound_native_requests(qwen, config, [case])
        assert evaluation_targets(qwen, case, config, dataset)


def test_counters_include_peft_generation_base_forward():
    import torch
    from torch import nn
    from peft import LoraConfig, get_peft_model
    from probes.coordinate_representation.coordinate_codebook_alignment.evaluation import _hook_counters
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4,4)
            self.model = nn.Module()
            self.model.visual = nn.Linear(4,4)
        def forward(self, x):
            return self.linear(x) + self.model.visual(x)
    wrapped = get_peft_model(Model(), LoraConfig(r=2, target_modules=['linear']))
    counts, handles = _hook_counters(wrapped)
    wrapped.get_base_model()(torch.ones(1,4))
    for handle in handles: handle.remove()
    assert counts == {'model_forwards':1, 'vision_forwards':1}
