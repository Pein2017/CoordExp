"""Compare actual equal-dose single/replicated training checkpoint states."""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.artifacts.utf8_json import binding


def compare(single: Path, distributed: Path, output: Path):
    first = torch.load(single / 'training_state.pt', map_location='cpu', weights_only=False)
    second = torch.load(distributed / 'training_state.pt', map_location='cpu', weights_only=False)
    assert first['step'] == second['step'] == 2
    assert first['optimizer']['param_groups'] == second['optimizer']['param_groups']
    squared_error = squared_reference = 0.0
    mutation_squared_error = 0.0
    maximum = mutation_maximum = 0.0
    tensor_count = 0
    for key, state in first['optimizer']['state'].items():
        a, b = state['exp_avg'].double(), second['optimizer']['state'][key]['exp_avg'].double()
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
        squared_error += float((a - b).square().sum())
        mutation_squared_error += float((a - (0.5 * b)).square().sum())
        squared_reference += float(a.square().sum())
        maximum = max(maximum, float((a - b).abs().max()))
        mutation_maximum = max(mutation_maximum, float((a - (0.5 * b)).abs().max()))
        tensor_count += 1
    relative = (squared_error / max(squared_reference, 1e-30)) ** .5
    mutation_relative = (mutation_squared_error / max(squared_reference, 1e-30)) ** .5
    payload = {}
    for folder, filename in [('adapter', 'adapter_model.safetensors'),
                             ('special_token_embeddings', 'special_token_embeddings.safetensors'),
                             ('coordinate_codebook', 'coordinate_codebook.safetensors')]:
        a = load_file(str(single / folder / filename))
        b = load_file(str(distributed / folder / filename))
        assert a.keys() == b.keys()
        payload[folder] = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    result = {'status': 'passed' if relative <= .02 and maximum <= 2e-4 else 'failed',
              'moment_relative_l2': relative, 'moment_max_abs': maximum,
              'bounds': {'relative_l2': .02, 'max_abs': 2e-4},
              'moment_tensor_count': tensor_count, 'model_payload_maxima': payload,
              'half_gradient_mutation_relative_l2': mutation_relative,
              'half_gradient_mutation_max_abs': mutation_maximum,
              'half_gradient_mutation_rejected': mutation_relative > .02 or mutation_maximum > 2e-4,
              'bindings': [binding(single / 'training_state.pt'), binding(distributed / 'training_state.pt'),
                           binding(__file__)]}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    assert result['status'] == 'passed', result
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', type=Path, required=True)
    parser.add_argument('--distributed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.single, args.distributed, args.output)
