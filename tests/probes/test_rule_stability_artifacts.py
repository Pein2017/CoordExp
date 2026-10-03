"""Falsification at exclusive publication, source and checkpoint consumer boundaries."""
import json
from pathlib import Path

import pytest

from probes.rule_stability import artifacts as a


def test_publication_collision_and_nonfinite_leave_original(tmp_path):
    path = tmp_path / 'record.json'
    a.write(path, {'owner': 'first'})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        a.write(path, {'owner': 'second'})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        a.write(tmp_path / 'invalid.json', {'loss': float('nan')})
    assert not (tmp_path / 'invalid.json').exists()
    assert not list(tmp_path.glob('*.partial-*'))


def test_inventory_corruption_extra_file_and_traversal(tmp_path):
    a.write(tmp_path / 'data.json', {'data': [1, 2]})
    manifest = a.file_manifest(tmp_path)
    a.verify_manifest(tmp_path, manifest)
    (tmp_path / 'extra').write_text('unexpected')
    with pytest.raises(ValueError, match='inventory'):
        a.verify_manifest(tmp_path, manifest)
    (tmp_path / 'extra').unlink()
    (tmp_path / 'data.json').write_text('corrupt')
    with pytest.raises(ValueError, match='changed'):
        a.verify_manifest(tmp_path, manifest)


def test_source_self_inclusion_and_staleness(monkeypatch, tmp_path):
    source = tmp_path / 'probes/rule_stability/artifacts.py'
    source.parent.mkdir(parents=True)
    source.write_text('producer_v1')
    monkeypatch.setattr(a, 'ROOT', tmp_path)
    paths = a.source_paths()
    assert 'probes/rule_stability/artifacts.py' in paths
    identity = {'files': {path: a.digest(tmp_path / path) for path in paths}}
    a.verify_candidate_source(identity)
    with pytest.raises(ValueError, match='cover'):
        a.verify_candidate_source({'files': {}})
    source.write_text('producer_v2')
    with pytest.raises(ValueError, match='stale'):
        a.verify_candidate_source(identity)
    source.write_text('producer_v1')
    (source.parent / '__main__.py').write_text('new_runtime_owner')
    with pytest.raises(ValueError, match='cover'):
        a.verify_candidate_source(identity)


def test_unreleased_packet_rejected_before_source_or_model(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(a, 'source_paths', lambda: calls.append('source'))
    with pytest.raises(ValueError, match='release absent'):
        a.validate_packet({'released': False}, tmp_path / 'job', mode='qualification')
    assert not calls
    assert not (tmp_path / 'job').exists()


def checkpoint_fixture(tmp_path):
    import torch
    from safetensors.torch import save_file
    directory = tmp_path / 'checkpoint'
    (directory / 'adapter').mkdir(parents=True)
    (directory / 'special_token_embeddings').mkdir()
    parameters = [torch.nn.Parameter(torch.tensor([0.])),
                  torch.nn.Parameter(torch.tensor([[1.]])), torch.nn.Parameter(torch.tensor([[2.]]))]
    optimizer = torch.optim.AdamW([{'params': [p], 'lr': lr} for p, lr in zip(parameters, [1e-5, 5e-6, 5e-6])])
    sum(p.square().sum() for p in parameters).backward()
    optimizer.step()
    save_file({'fixture_language': parameters[0].detach()}, str(directory / 'adapter/adapter_model.safetensors'))
    save_file({'input_embed_delta': parameters[1].detach().reshape(1, 1),
               'output_embed_delta': parameters[2].detach().reshape(1, 1)},
              str(directory / 'special_token_embeddings/special_token_embeddings.safetensors'))
    a.write(directory / 'special_token_embeddings/special_token_embeddings.json',
            dict(tie_word_embeddings=False, tensor_shape=[1, 1], tensor_dtype='float32'))
    torch.save(dict(completed_updates=1, arm='B', state_dict=optimizer.state_dict()), directory / 'optimizer.pt')
    schema = [dict(name=name, role=role, shape=list(p.shape), dtype=str(p.dtype))
              for name, role, p in zip(('language', 'input_embed_delta', 'output_embed_delta'),
                                       ('language', 'input_delta', 'output_delta'), parameters)]
    a.seal_checkpoint(directory, arm='B', version=1, engine='cpu_fixture', parameter_schema=schema)
    return directory


def test_checkpoint_schema_optimizer_and_independent_deltas(tmp_path):
    directory = checkpoint_fixture(tmp_path)
    assert a.checkpoint_readback(directory, arm='B', version=1, engine='cpu_fixture')['status'] == 'complete'
    with pytest.raises(ValueError, match='schema'):
        a.checkpoint_readback(directory, arm='A', version=1, engine='cpu_fixture')
    metadata = a.load(directory / 'checkpoint.json')
    metadata['parameter_schema'][1]['shape'] = [2]
    (directory / 'checkpoint.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='tensor shape'):
        a.checkpoint_readback(directory, arm='B', version=1, engine='cpu_fixture')


@pytest.mark.parametrize('corruption', ['metadata', 'payload'])
def test_delta_metadata_payload_parameter_consistency(tmp_path, corruption):
    import torch
    from safetensors.torch import save_file
    directory = checkpoint_fixture(tmp_path)
    if corruption == 'metadata':
        (directory / 'special_token_embeddings/special_token_embeddings.json').write_text(
            json.dumps(dict(tie_word_embeddings=False, tensor_shape=[2, 2], tensor_dtype='bfloat16')))
    else:
        save_file({'input_embed_delta': torch.zeros(2, 1), 'output_embed_delta': torch.zeros(2, 1)},
            str(directory / 'special_token_embeddings/special_token_embeddings.safetensors'))
        (directory / 'special_token_embeddings/special_token_embeddings.json').write_text(
            json.dumps(dict(tie_word_embeddings=False, tensor_shape=[2, 1], tensor_dtype='float32')))
    metadata = a.load(directory / 'checkpoint.json')
    metadata['files'] = a.file_manifest(directory, excluded=('checkpoint.json',))
    (directory / 'checkpoint.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='delta.*(metadata|parameter)'):
        a.checkpoint_readback(directory, arm='B', version=1, engine='cpu_fixture')


def test_gradient_sum_is_image_mean_with_unequal_shards(monkeypatch):
    import torch
    from probes.rule_stability.runner import synchronize_gradients
    import torch.distributed as dist
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    local_derivative = sum([2., 4., 9.]) / 18
    remote_derivative = sum(range(15)) / 18
    parameter.grad = torch.tensor([local_derivative])
    monkeypatch.setattr(dist, 'all_reduce', lambda tensor, op: tensor.add_(remote_derivative))
    synchronize_gradients([parameter])
    assert float(parameter.grad) == pytest.approx((2 + 4 + 9 + sum(range(15))) / 18)
    wrong_rank_means = ((2 + 4 + 9) / 3 + sum(range(15)) / 15) / 2
    assert float(parameter.grad) != pytest.approx(wrong_rank_means)
