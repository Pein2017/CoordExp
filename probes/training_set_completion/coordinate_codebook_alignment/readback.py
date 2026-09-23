"""Check fresh evaluation processes against their exact saved model payloads."""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file

from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment.qualify import _tensor_hash
from src.adapters.dora import normalize_dora_state_key


def saved_hashes(root):
    values = {
        normalize_dora_state_key(k, adapter_name='default'): _tensor_hash(v)
        for k, v in load_file(str(root / 'adapter/adapter_model.safetensors')).items()
    }
    values = {k + '.weight' if k.endswith('.lora_magnitude_vector') else k: v
              for k, v in values.items()}
    rows = load_file(str(root / 'special_token_embeddings/special_token_embeddings.safetensors'))
    for key, name in [('input_embed_delta', 'model.language_model.embed_tokens.shared_embed_delta'),
                      ('output_embed_delta', 'lm_head.shared_embed_delta')]:
        values[name] = _tensor_hash(rows[key])
    gain = root / 'coordinate_codebook/coordinate_codebook.safetensors'
    if gain.exists():
        codebook = load_file(str(gain))
        values['coordinate_codebook.raw_gain'] = _tensor_hash(codebook['raw_gain'])
        if 'projection.weight' in codebook:
            values['coordinate_codebook.projection.weight'] = _tensor_hash(codebook['projection.weight'])
    return values


def check(input_root, output):
    expected, counts, cells = {}, {}, []
    for path in sorted(input_root.glob('*/cells/*.json')):
        cell = json.loads(path.read_text())
        if cell.get('status') != 'complete':
            continue
        identity = cell['checkpoint']
        root = Path(identity['checkpoint_root'])
        if root not in expected:
            expected[root] = saved_hashes(root)
            counts[str(root)] = {'parameter_count': len(expected[root]), 'cells': 0}
        if identity['parameter_hashes'] != expected[root]:
            raise ValueError(f'saved-to-loaded parameter mismatch: {path}')
        for item in identity['payload_bindings']:
            if binding(item['path']) != item:
                raise ValueError(f'payload binding changed: {item["path"]}')
        counts[str(root)]['cells'] += 1
        cells.append(binding(path))
    if not cells:
        raise ValueError('no complete saved evaluation cells')
    # The equality gate must reject a changed learned tensor, not merely keys.
    original = next(iter(expected.values()))
    corrupted = dict(original)
    corrupted[next(iter(corrupted))] = '0' * 64
    assert corrupted != original
    result = {'status': 'passed', 'checkpoints': counts, 'cells': cells,
              'changed_tensor_mutation_rejected': True, 'producer': binding(__file__),
              'scope': 'Exact saved tensors versus hashes captured after fresh HF load; no new model calls.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(binding(output)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.input_root, args.output)
